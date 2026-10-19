"""TRAIN-only exponent bases optimized for the actual exact tiered wire cost.

The existing producer models remain unchanged. These extra frozen controls fit
bases for final K, pre-RoPE K, and projected K separately; V always uses actual
final V. Every channel considers every exponent base in 0..255, minimizing
2 + 3*[zigzag(error)>=3] + 9*[zigzag(error)>=10] bits per value. This is a
stronger source-specific TRAIN control, not evidence of held-out savings.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import producer_models as producer

_SOURCES = {"cost_final": "k", "cost_pre_rope": "k_pre_rope", "cost_kproj": "k_proj"}


def _histogram(exponents):
    exponents = np.asarray(exponents)
    if (exponents.ndim != 2 or min(exponents.shape) < 1 or not np.issubdtype(exponents.dtype, np.integer)
            or np.any(exponents < 0) or np.any(exponents > 255)):
        raise ValueError("frontier exponent fit requires nonempty integer[T,C] in 0..255")
    columns = exponents.shape[1]
    offset = np.arange(columns, dtype=np.int64)[None, :] * 256
    return np.bincount((exponents.astype(np.int64) + offset).reshape(-1),
        minlength=256 * columns).reshape(columns, 256).T


def _fit_bases(exponents):
    """Return exhaustive tier-cost optimum, its TRAIN median, and actual bit costs.

    Prefix histograms provide all 256 candidates at once. Zigzag's asymmetric
    rare threshold makes the secondary window [base-5,base+4], not +/-5.
    Equal-cost candidates choose the one nearest floor(median), then lower.
    """
    histogram = _histogram(exponents)
    count, channels = exponents.shape
    prefix = np.concatenate((np.zeros((1, channels), dtype=np.int64), histogram.cumsum(axis=0)))
    candidates = np.arange(256, dtype=np.int64)
    close3 = prefix[np.minimum(candidates + 2, 256)] - prefix[np.maximum(candidates - 1, 0)]
    close10 = prefix[np.minimum(candidates + 5, 256)] - prefix[np.maximum(candidates - 5, 0)]
    costs = 14 * count - 3 * close3 - 9 * close10
    cumulative = prefix[1:]
    lower = (cumulative > ((count - 1) // 2)).argmax(axis=0)
    upper = (cumulative > (count // 2)).argmax(axis=0)
    median = (lower + upper) // 2
    # The secondary score is at most 65535, so it cannot outweigh one bit.
    score = costs * 65536 + np.abs(candidates[:, None] - median[None, :]) * 256 + candidates[:, None]
    chosen = score.argmin(axis=0)
    columns = np.arange(channels)
    return chosen.astype(np.uint8), median.astype(np.uint8), costs[chosen, columns], costs[median, columns]


def _source_words(record, field):
    if field in ("k", "v"):
        return producer._word_array(record, field)
    if field not in record:
        raise ValueError("frontier fitting requires actual producer source: " + field)
    value = np.asarray(record[field])
    expected = producer._word_array(record, "k")
    if value.dtype != np.uint16 or value.ndim != 2 or value.shape != expected.shape:
        raise ValueError("frontier producer source geometry mismatch: " + field)
    return value


def _exponents(records, field):
    return np.concatenate([((_source_words(record, field).astype(np.uint32) >> 7) & 255)
                           for record in records]).astype(np.int16)


def load(path, **expected):
    """Canonical producer-model loader, including complete CRC/SHA verification."""
    return producer.load(path, **expected)


def fit(training_record_sets, output_dir, rank=8, *, training_metadata, model_identity, seed=71):
    """Return existing producer models plus three source-specific tier-cost fits.

    Inputs are exactly the three explicit TRAIN producer captures. Source hashes
    bind every original field (including unused arrays) before and after BOTH
    fitting phases. No DEV/TEST data or observed TEST winner changes a model.
    """
    if (not isinstance(training_record_sets, Sequence) or isinstance(training_record_sets, (str, bytes))
            or len(training_record_sets) != 3 or any(not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records
                                                    for records in training_record_sets)):
        raise ValueError("frontier fitting requires three explicit nonempty TRAIN record sets")
    if training_metadata is None or len(training_metadata) != 3:
        raise ValueError("frontier fitting requires metadata for three TRAIN captures")
    normalized_metadata = producer._json(training_metadata)
    if any(not isinstance(row, dict) or row.get("split") not in ("train", "training") for row in normalized_metadata):
        raise ValueError("frontier fitting accepts explicitly labeled TRAIN captures only")
    identity = producer._json(model_identity)
    if not isinstance(identity, (dict, str)) or not identity:
        raise ValueError("explicit frontier source model identity is required")
    before = [producer._source_hash(records) for records in training_record_sets]
    metadata_before = producer._source_hash(dict(training_metadata=normalized_metadata, model_identity=identity))
    grouped = {}
    for records in training_record_sets:
        for record in records:
            layer, _ = producer._validate_record(record)
            for field in _SOURCES.values():
                _source_words(record, field)
            grouped.setdefault(layer, []).append(record)
    output = Path(output_dir).resolve()
    paths = {name: output / f"producer_{name}.bin" for name in _SOURCES}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("frontier model artifacts exist; use a fresh output directory")
    models = producer.fit(training_record_sets, output, rank=rank,
        training_metadata=training_metadata, model_identity=model_identity, seed=seed)
    frozen_control = models["frozen"]
    parameters, statistics = {}, {}
    # V has one actual source in every control; fit it once and copy each full
    # parameter set into its independent self-contained artifact.
    common_v = {}
    for layer in sorted(grouped):
        target = _exponents(grouped[layer], "v")
        base, median, cost, median_cost = _fit_bases(target)
        common_v[layer] = (base, median, cost, median_cost, target.shape[0])
    for name, source in _SOURCES.items():
        per_layer, stats = {}, {}
        for layer in sorted(grouped):
            target = _exponents(grouped[layer], source)
            base, median, cost, median_cost = _fit_bases(target)
            vbase, vmedian, vcost, vmedian_cost, vrows = common_v[layer]
            per_layer[layer] = dict(base_K=base.copy(), base_V=vbase.copy())
            stats[layer] = {}
            for kind, selected, reference, bits, reference_bits, rows in (
                    ("K", base, median, cost, median_cost, target.shape[0]),
                    ("V", vbase, vmedian, vcost, vmedian_cost, vrows)):
                if np.any(bits > reference_bits):
                    raise AssertionError("exhaustive TRAIN cost optimum exceeds its median control")
                stats[layer][kind] = dict(training_rows=rows, channels=len(selected),
                    actual_tier_bits=int(bits.sum()), median_control_tier_bits=int(reference_bits.sum()),
                    training_bits_saved_over_median=int((reference_bits - bits).sum()),
                    training_bits_per_value=float(bits.sum() / (rows * len(selected))),
                    changed_channels_from_source_median=int(np.count_nonzero(selected != reference)))
        parameters[name], statistics[name] = per_layer, stats
    after = [producer._source_hash(records) for records in training_record_sets]
    metadata_after = producer._source_hash(dict(training_metadata=producer._json(training_metadata),
        model_identity=producer._json(model_identity)))
    if before != after or metadata_before != metadata_after:
        raise AssertionError("frontier TRAIN source fields changed during fitting")
    if before != frozen_control.metadata["training_hashes"]:
        raise AssertionError("frontier and producer controls used different TRAIN captures")
    for name, source in _SOURCES.items():
        metadata = copy.deepcopy(frozen_control.metadata)
        metadata.update(model_variant=name, model_family="producer_source_cost_base",
            source_k=source, source_v="v", training_stats=statistics[name],
            scope="Actual producer-stage TRAIN sources only. Source-specific frozen exponent bases minimize the exact 2/3/9 tier payload cost. Final corrections remain mandatory; serialized state, offsets, framing and replay corrections are charged separately by the codec.")
        metadata["config"].update(exponent_base="global exhaustive minimum actual tier bits on three TRAIN captures",
            source_k=source, source_v="v", candidate_bases=256,
            cost_objective="2+3*(zigzag(error)>=3)+9*(zigzag(error)>=10)",
            cost_tie_break="nearest floor-median on TRAIN, then lower exponent base",
            objective_scope="exponent tier payload bits; no TEST tuning; other fields are unaffected by base choice")
        params = parameters[name]
        for layer in params.values():
            for array in layer.values():
                array.flags.writeable = False
        model = producer.Model("frozen", params, metadata, str(paths[name]))
        paths[name].write_bytes(producer.model_bytes(model))
        models[name] = producer.load(paths[name], expected_model_identity=identity,
            expected_training_hashes=before, expected_model_id=model.model_id, expected_kind="frozen")
    return models
