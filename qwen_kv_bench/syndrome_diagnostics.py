"""Fixed DEV probes for the prefill-coset training study.

This module observes frozen-in-time model tensors. It never optimizes, selects
checkpoints, constructs packets, or sees TEST. Group byte counts are diagnostic
pre-header projections for the current single-error code, not encoded savings.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from . import syndrome_models as models, producer_models as producer
from .side_probability import joint

FORMAT = "prefill_coset_fixed_dev_diagnostics_v1"
CALIBRATION_BINS = 10


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return int(value)


def _validate_records(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("Fixed DEV probe needs nonempty records")
    seen = {}
    for r in records:
        if r.get("split", "dev") != "dev":
            raise ValueError("Fixed diagnostics require DEV records only")
        layer, start, tokens = int(r["layer"]), int(r["start"]), _positive(r["tokens"], "tokens")
        words = joint(r)
        if layer < 0 or start < 0 or len(words) != tokens:
            raise ValueError("Invalid fixed DEV record geometry")
        if any(start < end and begin < start + tokens for begin, end in seen.get(layer, [])):
            raise ValueError("Fixed DEV records overlap")
        seen.setdefault(layer, []).append((start, start + tokens))
    return sorted(records, key=lambda r: (int(r["layer"]), int(r["start"])))


def _slice(r, begin, end):
    result = dict(r)
    for key, value in r.items():
        if isinstance(value, np.ndarray) and value.ndim and len(value) == r["tokens"]:
            result[key] = value[begin:end].copy()
    result.update(start=int(r["start"]) + begin, tokens=end - begin, split="dev")
    return result


def _probe_manifest(records):
    addresses = [dict(layer=int(r["layer"]), start=int(r["start"]), tokens=int(r["tokens"]),
                      channels=int(joint(r).shape[1])) for r in records]
    return dict(split="dev", sha256=producer._source_hash(records), records=addresses,
                record_count=len(records), word_values=sum(r["tokens"] * joint(r).shape[1] for r in records),
                hash_contract="producer_models._source_hash over selected complete DEV record metadata and array dtype/shape/bytes")


def make_probe(records, *, tokens_per_layer=64, max_layers=None):
    """Select all channels at fixed head/tail token spans in each selected layer.

    A 64-token budget selects 32 head and 32 tail tokens when the layer has
    more data. Groups never span a source-record boundary. All layers are used
    by default; a layer cap uses uniformly spaced layer IDs, including ends.
    """
    records = _validate_records(records)
    budget = _positive(tokens_per_layer, "tokens_per_layer")
    grouped = {}
    for r in records:
        grouped.setdefault(int(r["layer"]), []).append(r)
    layers = sorted(grouped)
    if max_layers is not None:
        count = min(_positive(max_layers, "max_layers"), len(layers))
        layers = [layers[i] for i in np.linspace(0, len(layers)-1, count, dtype=int)]
    selected = []
    for layer in layers:
        rows = grouped[layer]
        available = sum(int(r["tokens"]) for r in rows)
        if available <= budget:
            selected.extend(_slice(r, 0, int(r["tokens"])) for r in rows)
            continue
        head, tail = (budget + 1) // 2, budget // 2
        spans = {}
        left = head
        for i, r in enumerate(rows):
            n = min(left, int(r["tokens"]))
            if n:
                spans.setdefault(i, []).append((0, n)); left -= n
            if not left:
                break
        left = tail
        for i in range(len(rows)-1, -1, -1):
            n = min(left, int(rows[i]["tokens"]))
            if n:
                spans.setdefault(i, []).append((int(rows[i]["tokens"])-n, int(rows[i]["tokens"]))); left -= n
            if not left:
                break
        for i, parts in sorted(spans.items()):
            for begin, end in sorted(parts):
                selected.append(_slice(rows[i], begin, end))
    selected = _validate_records(selected)
    manifest = _probe_manifest(selected)
    manifest.update(tokens_per_layer=budget, selected_layers=layers,
        selection="Fixed head/tail contiguous token spans per selected layer, all KV channels; no content or target-driven selection",
        source_records_sha256=producer._source_hash(records))
    return selected, manifest


def _empty(calibration=True):
    value = dict(values=0, bit_errors=0, original_ones=0, predicted_ones=0,
                 BCE_sum_bits=0., Brier_sum=0., groups=0, errors_per_group=[0]*32,
                 recoverable_groups=0, tail_bits=0, candidate_mode_bytes=0,
                 candidate_short_bytes=0, candidate_raw_group_bytes=0,
                 candidate_tail_bytes=0, candidate_payload_bytes=0, raw_plane_bytes=0)
    if calibration:
        value["calibration"] = dict(bin_edges=np.linspace(0., 1., CALIBRATION_BINS+1).tolist(),
            counts=[0]*CALIBRATION_BINS, probability_sums=[0.]*CALIBRATION_BINS,
            truth_sums=[0.]*CALIBRATION_BINS)
    return value


def _add(destination, item):
    for key, value in item.items():
        if key in ("BCE_bits", "Brier_mean", "bit_error_rate", "recoverable_fraction"):
            continue
        if key == "calibration":
            if key in destination:
                for name in ("counts", "probability_sums", "truth_sums"):
                    destination[key][name] = [x+y for x,y in zip(destination[key][name], value[name])]
        elif isinstance(value, list):
            destination[key] = [x+y for x,y in zip(destination[key], value)]
        elif isinstance(value, (int, float)):
            destination[key] += value


def _finish(value, *, probability=True):
    value["bit_error_rate"] = value["bit_errors"] / value["values"] if value["values"] else None
    value["recoverable_fraction"] = value["recoverable_groups"] / value["groups"] if value["groups"] else None
    if probability:
        value["BCE_bits"] = value["BCE_sum_bits"] / value["values"] if value["values"] else None
        value["Brier_mean"] = value["Brier_sum"] / value["values"] if value["values"] else None
    else:
        for key in ("BCE_sum_bits", "Brier_sum", "calibration"):
            value.pop(key, None)
    return value


def _metric(logits, actual, *, hard_zero=False, calibration=True):
    actual = np.asarray(actual, dtype=np.uint8)
    if actual.ndim != 2 or not len(actual):
        raise ValueError("Metric truth needs nonempty token/channel plane")
    out = _empty(calibration and not hard_zero)
    guess = np.zeros_like(actual) if hard_zero else (logits >= 0).astype(np.uint8)
    error = guess ^ actual
    out.update(values=int(actual.size), bit_errors=int(np.sum(error)), original_ones=int(np.sum(actual)),
               predicted_ones=int(np.sum(guess)))
    groups, tail = divmod(len(actual), 31)
    counts = error.T[:, :groups*31].reshape(-1, 31).sum(axis=1, dtype=np.int64)
    eligible = int(np.sum(counts <= 1))
    out.update(groups=len(counts), errors_per_group=np.bincount(counts, minlength=32).tolist(),
               recoverable_groups=eligible, tail_bits=actual.shape[1]*tail,
               candidate_mode_bytes=(len(counts)+7)//8, candidate_short_bytes=(eligible*5+7)//8,
               candidate_raw_group_bytes=((len(counts)-eligible)*31+7)//8,
               candidate_tail_bytes=(actual.shape[1]*tail+7)//8, raw_plane_bytes=(actual.size+7)//8)
    out["candidate_payload_bytes"] = sum(out[k] for k in
        ("candidate_mode_bytes", "candidate_short_bytes", "candidate_raw_group_bytes", "candidate_tail_bytes"))
    if not hard_zero:
        logits = np.broadcast_to(np.asarray(logits, dtype=np.float64), actual.shape)
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite fixed DEV logits")
        probabilities = np.exp(-np.logaddexp(0., -logits))
        out["BCE_sum_bits"] = float(np.sum(np.logaddexp(0., logits)-logits*actual)/math.log(2.))
        out["Brier_sum"] = float(np.sum((probabilities-actual)**2))
        if calibration:
            bins = np.minimum((probabilities*CALIBRATION_BINS).astype(np.int64), CALIBRATION_BINS-1)
            out["calibration"].update(counts=np.bincount(bins.ravel(), minlength=CALIBRATION_BINS).tolist(),
                probability_sums=np.bincount(bins.ravel(), weights=probabilities.ravel(), minlength=CALIBRATION_BINS).tolist(),
                truth_sums=np.bincount(bins.ravel(), weights=actual.ravel(), minlength=CALIBRATION_BINS).tolist())
    return out


def _code_summary(features):
    rank = features[0].shape[1]
    value = np.concatenate(features)
    return dict(tokens=len(value), rank=rank, retained_BF16_bytes=int(value.size*2),
        zero_values=int(np.sum(value == 0)), at_clamp_values=int(np.sum(np.abs(value) >= 32.)),
        mean_by_feature=np.mean(value, axis=0, dtype=np.float64).tolist(),
        std_by_feature=np.std(value.astype(np.float64), axis=0).tolist(),
        min_by_feature=np.min(value, axis=0).tolist(), max_by_feature=np.max(value, axis=0).tolist(),
        distinct_values_by_feature=[int(np.unique(value[:, i]).size) for i in range(rank)],
        BF16_words_sha256=hashlib.sha256((np.ascontiguousarray(value).view(np.uint32) >> 16).astype("<u2").tobytes()).hexdigest(),
        code_hash_contract="SHA256 over little-endian uint16 BF16 retained words in fixed record order")


def _predict(params, geometry, record, features, b):
    device = next(iter(params.values())).device
    words = joint(record)
    # Supply only the decoder-known words. _forward also masks internally.
    known = words & np.uint16(0xff80 | ((1 << b)-1))
    rows = max(1, models.PREDICT_WORD_CHUNK // words.shape[1])
    output = []
    with torch.autocast(device_type=device.type, enabled=False):
        for begin in range(0, len(words), rows):
            logits = models._forward(params, geometry, int(record["layer"]),
                torch.tensor(features[begin:begin+rows].copy(), device=device), int(record["start"])+begin,
                torch.tensor(known[begin:begin+rows].copy(), device=device), b)
            output.append(logits.cpu().numpy().copy())
    return np.concatenate(output), known


@torch.inference_mode()
def evaluate(params, geometry, mode, records, *, residue_bits=(0,2,4,6), seed=71,
             ablations=True, prior_logits=None, rope_scaling=1.):
    """Evaluate a fixed DEV probe without fitting or checkpoint selection.

    ``prior_logits`` must be the immutable TRAIN initialization, not the learned
    current ``params['prior']``. Constant-zero is a hard-decision diagnostic;
    its infinite-confidence probability loss is deliberately not reported.
    """
    records = _validate_records(records)
    budgets = tuple(residue_bits)
    if not budgets or len(set(budgets)) != len(budgets) or any(isinstance(b, bool) or b not in range(7) for b in budgets):
        raise ValueError("Fixed DEV residue widths must be unique integers in 0..6")
    if mode not in models.MODES or seed < 0:
        raise ValueError("Invalid fixed DEV mode or seed")
    prior = None if prior_logits is None else np.asarray(prior_logits, dtype=np.float32)
    if prior is not None and (prior.shape != (geometry["layers"], 2*geometry["channels"], 7) or not np.isfinite(prior).all()):
        raise ValueError("Fixed TRAIN prior shape or finiteness mismatch")
    device = next(iter(params.values())).device
    inverse = params["rope_inv_freq"].detach().cpu().numpy()
    scaling = float(rope_scaling)
    if not math.isfinite(scaling) or scaling <= 0:
        raise ValueError("Fixed DEV needs positive finite RoPE scaling")
    features = []
    for r in records:
        if joint(r).shape != (r["tokens"], 2*geometry["channels"]) or not 0 <= int(r["layer"]) < geometry["layers"]:
            raise ValueError("Fixed DEV/model geometry mismatch")
        if mode == "none":
            value = np.empty((r["tokens"], 0), np.float32)
        else:
            source = models.source_values(r, mode, inverse, scaling, geometry["head_dim"])
            with torch.autocast(device_type=device.type, enabled=False):
                value = models._encode(params, torch.tensor(source.copy(), device=device), int(r["layer"]),
                    straight_through=False).cpu().numpy().copy()
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite fixed DEV retained code")
        features.append(value)
    altered = {}
    if ablations and mode != "none":
        altered["zero"] = [np.zeros_like(f) for f in features]
        shuffled = [None]*len(records)
        layers = sorted(set(int(r["layer"]) for r in records))
        for layer in layers:
            indices = [i for i,r in enumerate(records) if int(r["layer"]) == layer]
            value = np.concatenate([features[i] for i in indices])
            permutation = np.random.default_rng(seed+layer).permutation(len(value))
            value = value[permutation]; begin = 0
            for i in indices:
                shuffled[i] = value[begin:begin+len(features[i])].copy(); begin += len(features[i])
        altered["shuffled"] = shuffled
    result = dict(format=FORMAT, split="dev", probe=_probe_manifest(records), residue_bits=list(budgets),
        mode=mode, ablations_evaluated=bool(altered), code_summary=_code_summary(features),
        prior_logits_sha256=None if prior is None else hashlib.sha256(prior.astype("<f4").tobytes()).hexdigest(),
        scope="Fixed DEV observation only; no optimization/checkpoint selection/TEST, raw captures are not decoded, and group candidate byte counts exclude headers/features/state and are not encoded packet savings.",
        by_residue={})
    for b in budgets:
        learned = dict(omitted=_empty(), by_bitplane={str(j):_empty() for j in range(b,7)},
                       by_kind={k:_empty() for k in ("K","V")}, by_layer={})
        controls = {"constant_zero":dict(omitted=_empty(False), by_bitplane={str(j):_empty(False) for j in range(b,7)})}
        if prior is not None:
            controls["train_prior"] = dict(omitted=_empty(), by_bitplane={str(j):_empty() for j in range(b,7)})
        ablation_rows = {name:dict(prediction_changed_bits=0, omitted_values=0, BCE_sum_bits_delta=0.,
            bit_errors_delta=0, by_bitplane={str(j):dict(prediction_changed_bits=0, values=0,
                BCE_sum_bits_delta=0.,bit_errors_delta=0) for j in range(b,7)}) for name in altered}
        known_hash = hashlib.sha256()
        for index, r in enumerate(records):
            logits, known = _predict(params, geometry, r, features[index], b)
            known_hash.update(np.ascontiguousarray(known, dtype="<u2").tobytes())
            truth = joint(r)
            layer = str(int(r["layer"])); by_layer = learned["by_layer"].setdefault(layer, _empty(False))
            alternative_logits = {name:_predict(params, geometry, r, values[index], b)[0] for name, values in altered.items()}
            for j in range(b, 7):
                actual = ((truth >> j)&1).astype(np.uint8)
                metric = _metric(logits[...,j], actual)
                _add(learned["omitted"], metric); _add(learned["by_bitplane"][str(j)], metric); _add(by_layer, metric)
                for kind, span in (("K",slice(0,geometry["channels"])), ("V",slice(geometry["channels"],None))):
                    _add(learned["by_kind"][kind], _metric(logits[:,span,j],actual[:,span]))
                zero = _metric(None, actual, hard_zero=True)
                _add(controls["constant_zero"]["omitted"],zero); _add(controls["constant_zero"]["by_bitplane"][str(j)],zero)
                if prior is not None:
                    pm = _metric(np.broadcast_to(prior[int(r["layer"]),:,j],actual.shape), actual)
                    _add(controls["train_prior"]["omitted"],pm); _add(controls["train_prior"]["by_bitplane"][str(j)],pm)
                for name, alternative in alternative_logits.items():
                    am = _metric(alternative[...,j],actual,calibration=False)
                    changed = int(np.sum((alternative[...,j]>=0) != (logits[...,j]>=0)))
                    row = ablation_rows[name]; small = row["by_bitplane"][str(j)]
                    row["prediction_changed_bits"] += changed; row["omitted_values"] += actual.size
                    row["BCE_sum_bits_delta"] += am["BCE_sum_bits"]-metric["BCE_sum_bits"]
                    row["bit_errors_delta"] += am["bit_errors"]-metric["bit_errors"]
                    small["prediction_changed_bits"] += changed; small["values"] += actual.size
                    small["BCE_sum_bits_delta"] += am["BCE_sum_bits"]-metric["BCE_sum_bits"]
                    small["bit_errors_delta"] += am["bit_errors"]-metric["bit_errors"]
        _finish(learned["omitted"])
        for field in ("by_bitplane", "by_kind", "by_layer"):
            for metric in learned[field].values(): _finish(metric)
        for name, control in controls.items():
            _finish(control["omitted"], probability=name!="constant_zero")
            for metric in control["by_bitplane"].values(): _finish(metric, probability=name!="constant_zero")
        for row in ablation_rows.values():
            row["BCE_bits_delta"] = row["BCE_sum_bits_delta"]/row["omitted_values"]
            for small in row["by_bitplane"].values(): small["BCE_bits_delta"] = small["BCE_sum_bits_delta"]/small["values"]
        result["by_residue"][str(b)] = dict(known_fields_sha256=known_hash.hexdigest(),learned=learned,
            controls=controls, ablations=ablation_rows)
    return result
