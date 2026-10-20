"""Matched compact-prefill hints and page-local progressive exact KV coding.

Small learned networks may run on the selected device. Entropy packets are a
CPU reference. All candidates are reported; a winner is frozen on DEV complete
packet and full-state estimates before fresh TEST captures. This is neither a
GPU entropy kernel nor a concurrent-serving capacity experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from . import model_runtime as runtime, neural_study as previous
from . import side_study as base, syndrome_study as field_study
from .fixtures import TRAIN, DEV
from .coset_training_study import FRESH_TEST
from .neural_accounting import account
from .reporting import write_json


def _note(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=runtime.DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="Random local tiny Qwen; functionality only")
    parser.add_argument("--variants", nargs="+", help="Prespecified subset; default all fourteen variants")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--batch-tokens", type=int)
    parser.add_argument("--channels-per-step", type=int)
    parser.add_argument("--train-tokens", type=int)
    parser.add_argument("--dev-tokens", type=int)
    parser.add_argument("--lengths", nargs="+", type=int)
    parser.add_argument("--cases", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--tile-tokens", type=int)
    parser.add_argument("--max-pairs", type=int, default=72)
    parser.add_argument("--ablation-max-pairs", type=int, default=12,
        help="Prespecified spread of DEV pages for fixed-model hint CDF diagnostics")
    parser.add_argument("--generation-steps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path)
    return parser


def _defaults(args):
    values = dict(steps=2 if args.tiny else 4096, width=16 if args.tiny else 32,
        batch_tokens=8 if args.tiny else 32, channels_per_step=8 if args.tiny else 64,
        train_tokens=32 if args.tiny else 512, dev_tokens=32 if args.tiny else 128,
        lengths=[64] if args.tiny else [2048, 8192],
        chunk_tokens=32 if args.tiny else 512, tile_tokens=16 if args.tiny else 64)
    for key, value in values.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.tiny and args.device == "auto":
        args.device = "cpu"
    if args.revision is None and args.model == runtime.DEFAULT_MODEL:
        args.revision = runtime.DEFAULT_REVISION


def _validate(args, parser, specs):
    if any(n < 1 for n in (args.steps, args.width, args.batch_tokens, args.channels_per_step,
            args.train_tokens, args.dev_tokens, args.chunk_tokens, args.tile_tokens,
            args.max_pairs, args.ablation_max_pairs, args.repeats, args.threads, *args.lengths)):
        parser.error("All budgets and lengths must be positive")
    if args.seed < 0 or not 1 <= args.generation_steps <= 64:
        parser.error("Require nonnegative seed and generation steps in 1..64")
    available = {row["name"] for row in specs}
    requested = args.variants or [row["name"] for row in specs]
    if not requested or len(requested) != len(set(requested)) or set(requested) - available:
        parser.error("--variants must be unique known names: " + ", ".join(sorted(available)))
    args.variants = requested
    # The same-decoder no-hint control is required for every tested family.
    needs = {"msb_pool_none" if row["layout"] == "msb" else
             "lsb_kfirst_none" if row["key_first"] else "lsb_pool_none"
             for row in specs if row["name"] in requested and row["source"] != "none"}
    if needs - set(requested):
        parser.error("Requested hints need matched no-hint controls: " + ", ".join(sorted(needs - set(requested))))
    return [row for row in specs if row["name"] in requested]


def _hint_summary(codes, spec):
    arrays = [np.asarray(value, dtype=np.uint8).reshape(-1) for _, value in sorted(codes.items())]
    values = np.concatenate(arrays) if arrays else np.empty(0, np.uint8)
    count = (1 << spec["side_bits"]) if spec["side_bits"] else 0
    hist = np.bincount(values, minlength=count).tolist() if count else []
    positive = np.asarray(hist, dtype=np.float64)
    positive = positive[positive > 0]
    entropy = float(-np.sum((positive / positive.sum()) * np.log2(positive / positive.sum()))) if positive.size else 0.
    transitions = sum(int(np.count_nonzero(value[1:] != value[:-1])) for value in arrays)
    possible = sum(max(0, len(value) - 1) for value in arrays)
    return dict(symbols=int(len(values)), side_bits=int(spec["side_bits"]), histogram=hist,
        empirical_marginal_entropy_bits=entropy, unique_symbols=int(len(np.unique(values))),
        within_page_transitions=transitions, within_page_transition_opportunities=possible,
        within_page_transition_fraction=transitions / possible if possible else None,
        unpacked_uint8_bytes=int(values.nbytes), sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        scope="Raw categorical hint observations; entropy is diagnostic, not the charged wire length")


class InlineHints:
    """Emit actual compact hint IDs inside the existing actual-prefill callback."""
    def __init__(self, models, device):
        self.prepared, self.codes, self.rows = {}, {}, {}
        self.specs = {}
        for name, model in models.items():
            prepared, measured = runtime.timed_call(lambda model=model: model.prepare(device), device)
            self.prepared[name] = prepared
            self.specs[name] = dict(model.spec)
            self.codes[name] = {}
            self.rows[name] = dict(calls=0, unpacked_hint_bytes=0, wall_seconds=0., preparation=measured)

    def __call__(self, record, context):
        for name, prepared in self.prepared.items():
            address = (int(record["layer"]), int(record["start"]))
            if address in self.codes[name]:
                raise AssertionError("Compact prefill hint emitted more than once")
            hints, elapsed = runtime.timed_call(lambda: prepared.encode_hints(record), prepared.device)
            spec = self.specs[name]
            hints = np.asarray(hints)
            expected = 0 if spec["source"] == "none" else (1 if spec["page_shared"] else record["tokens"])
            if (hints.dtype != np.uint8 or hints.shape != (expected,)
                    or (hints.size and np.any(hints >= 1 << spec["side_bits"]))):
                raise AssertionError("Inline compact hint IDs have invalid shape/range")
            hints = hints.copy(); hints.flags.writeable = False
            self.codes[name][address] = hints
            row = self.rows[name]
            row["calls"] += 1; row["unpacked_hint_bytes"] += hints.nbytes
            row["wall_seconds"] += elapsed["wall_seconds"]

    def report(self):
        return dict(methods={name: dict(row, hint_summary=_hint_summary(self.codes[name], self.specs[name]))
            for name, row in self.rows.items()}, observed_inside_attention_forward=True,
            scope="Actual synchronous attention on_tile callbacks; includes diagnostic staging and all requested hint encoders. IDs are later bitpacked. Full raw KV remains resident; no isolated prefill-overhead claim.")

    def release_prepared(self):
        self.prepared.clear(); gc.collect()


def _hint_file(output, name, codes, spec, *, placement):
    path = output / "hints" / (name + ".npz")
    path.parent.mkdir(exist_ok=True)
    arrays = {f"layer_{layer:03d}_start_{start:08d}": np.asarray(value, dtype=np.uint8)
              for (layer, start), value in sorted(codes.items())}
    np.savez_compressed(path, **arrays)
    return dict(file=str(path.relative_to(output)), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), placement=placement,
        summary=_hint_summary(codes, spec), contains="Categorical hint IDs and page addresses only; no raw KV/hidden states")


def _aggregate(records, field):
    result = {}
    for row in records:
        group = result.setdefault(str(row[field]), Counter())
        group["pages"] += 1
        group["raw_bytes"] += row["raw_bytes"]
        group["packet_bytes"] += row["packet_bytes"]
        group.update(row["component_bytes"])
    return {key: dict(value) for key, value in result.items()}


def _encode(records, prepared, geometry, args, codes=None):
    from . import compact_codec as codec
    pages, traces, components = [], [], Counter()
    before = time.perf_counter()
    for record in sorted(records, key=lambda row: (row["layer"], row["start"])):
        address = (int(record["layer"]), int(record["start"]))
        supplied = None if codes is None else codes[address]
        wire, stats = codec.encode_page(record, prepared, hint_ids=supplied)
        if int(stats["packet_bytes"]) != len(wire) or sum(stats["component_bytes"].values()) != len(wire):
            raise AssertionError("Compact page byte ledger differs from actual packet")
        row = dict(stats, layer=address[0], start=address[1], tokens=int(record["tokens"]),
            hint_ids_supplied=supplied is not None)
        traces.append(row); pages.append(wire); components.update(row["component_bytes"])
    wire = base.pack_pages(pages, layers=int(geometry["layers"]), tokens=int(geometry["tokens"]),
        channels=int(geometry["channels"]), tile_tokens=args.tile_tokens)
    components["container_framing_bytes"] = len(wire) - sum(map(len, pages))
    if sum(components.values()) != len(wire):
        raise AssertionError("Compact indexed-container byte ledger differs")
    raw = sum(int(row["raw_bytes"]) for row in traces)
    return wire, dict(raw_sample_bytes=raw, packet_bytes=len(wire), component_bytes=dict(components),
        records_measured=len(traces), records=traces, by_layer=_aggregate(traces, "layer"),
        by_token_start=_aggregate(traces, "start"), complete_packet_bits_per_KV_value=16 * len(wire) / raw,
        encode_wall_seconds=time.perf_counter() - before, inline_hints_used=codes is not None,
        maximum_dependency_depth=0, raw_prerequisite_pages=0,
        scope="Actual independent pages and complete indexed envelope. CPU entropy coding; selected-device probability inference. Inter-page dependency depth is zero; refinement/K-to-V stages remain internal page dependencies.")


def restore_pages(wire, prepared, *, require_full=False, seed=71):
    from . import compact_codec as codec
    geometry, pages = base.unpack_pages(wire)
    layers, tokens, channels, tile = (int(geometry[key]) for key in ("layers", "tokens", "channels", "tile_tokens"))
    addresses, seen = [], set()
    for page in pages:
        meta = codec.page_metadata(page)
        layer, start, count = (meta[key] for key in ("layer", "start", "tokens"))
        if (any(isinstance(n, bool) or not isinstance(n, int) for n in (layer, start, count))
                or not 0 <= layer < layers or not 0 <= start < tokens or start % tile
                or count != min(tile, tokens - start) or meta.get("channels") != channels
                or (layer, start) in seen):
            raise ValueError("Compact page address differs from indexed geometry")
        addresses.append((layer, start, count)); seen.add((layer, start))
    if require_full and seen != {(layer, start) for layer in range(layers) for start in range(0, tokens, tile)}:
        raise ValueError("Full compact restore requires all layer/token pages")
    order = np.random.default_rng(seed).permutation(len(pages)).tolist()
    if len(order) > 1 and order == list(range(len(pages))):
        order.reverse()
    result = [(np.empty((tokens, channels), np.uint16), np.empty((tokens, channels), np.uint16))
              for _ in range(layers)] if require_full else []
    totals, traces = Counter(), []
    began = time.perf_counter()
    for ordinal in order:
        layer, start, count = addresses[ordinal]
        words, measured = codec.decode_page(pages[ordinal], prepared, return_stats=True)
        if words.dtype != np.uint16 or words.shape != (count, 2 * channels):
            raise AssertionError("Compact decoder returned invalid joint KV shape")
        if require_full:
            result[layer][0][start:start + count] = words[:, :channels]
            result[layer][1][start:start + count] = words[:, channels:]
        else:
            result.append(dict(layer=layer, start=start, tokens=count,
                k=words[:, :channels].copy(), v=words[:, channels:].copy()))
        for name, value in measured.items():
            if name.endswith("seconds") and isinstance(value, (int, float)):
                totals["page_wall_seconds" if name == "wall_seconds" else name] += value
        traces.append(dict(layer=layer, start=start, tokens=count, **measured))
    return result, dict(wall_seconds=time.perf_counter() - began, **dict(totals),
        records_decoded=len(pages), full_prefix=require_full, records=traces,
        decode_order=[[addresses[i][0], addresses[i][1]] for i in order],
        order_is_shuffled=order != list(range(len(pages))) if len(pages) > 1 else None,
        returned_output_bytes=4 * layers * tokens * channels if require_full else
            sum(row["k"].nbytes + row["v"].nbytes for row in result),
        maximum_dependency_depth=0, raw_prerequisite_pages=0, source_arrays_used=False,
        predictor_device=str(prepared.device), packet_backend="CPU small-alphabet arithmetic reference",
        scope="Own-page plus frozen-state restore; internal refinement stages remain serial. No capacity-bound or transport-speed claim.")


def _repeats(wire, prepared, matrices, starts, args):
    rows = []
    for index in range(args.repeats):
        restored, measured = restore_pages(wire, prepared, seed=args.seed + index)
        base._check_sample(restored, matrices, starts, args.tile_tokens)
        rows.append(measured)
    samples = [row["wall_seconds"] for row in rows]
    return dict(all_repeats_exact=True, roundtrip_exact=True, repeats=args.repeats,
        wall_seconds_samples=samples, median_wall_seconds=statistics.median(samples),
        independent_packet_decode=rows[0])


def _artifact(model, output):
    files = []
    for filename in model.artifact_paths:
        path = Path(filename)
        files.append(dict(file=str(path.relative_to(output)), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if sum(row["bytes"] for row in files) != model.state_bytes:
        raise AssertionError("Compact complete-state artifact ledger differs")
    return dict(model_id=model.model_id, serialized_state_bytes=model.state_bytes,
        files=files, metadata=model.metadata, spec=dict(model.spec),
        scope="Full encoder/decoder/calibration artifact charged, even if some state is producer-only")


def _select(development_rows):
    """One prespecified storage-first selection; never inspect TEST results."""
    eligible = [row for row in development_rows if row.get("status") == "ok"]
    if not eligible:
        raise ValueError("No exact DEV candidate available")
    return min(eligible, key=lambda row: (row["accounting"]["full_cache_estimate"]["estimated_total_bytes"],
        row["name"]))["name"]


def _packet_partition(measured):
    """Separate coding benefit from hint cost and descriptive header changes."""
    parts = measured.get("component_bytes")
    if not isinstance(parts, dict) or sum(parts.values()) != measured.get("packet_bytes"):
        return None
    hints = sum(int(value) for name, value in parts.items() if name in ("hint_bytes", "feature_bytes"))
    framing = sum(int(value) for name, value in parts.items() if "framing" in name or "envelope" in name)
    stages = sum(int(value) for name, value in parts.items() if name.startswith("stage") and name.endswith("_bytes"))
    return dict(main_payload_bytes=int(measured["packet_bytes"]) - hints - framing,
        hint_payload_bytes=hints, framing_bytes=framing, learned_stage_payload_bytes=stages)


def _comparison(row, other, spec, control_spec):
    a, b = row["measurement"], other["measurement"]
    pa, pb = _packet_partition(a), _packet_partition(b)
    partition_delta = None if pa is None or pb is None else {name: pa[name] - pb[name] for name in pa}
    ca, cb = a.get("component_bytes"), b.get("component_bytes")
    component_delta = None if ca is None or cb is None else {
        name: int(ca.get(name, 0)) - int(cb.get(name, 0)) for name in sorted(set(ca) | set(cb))}
    matching_context = control_spec is not None and all(spec[key] == control_spec[key]
        for key in ("layout", "key_first", "pooled", "base_bits"))
    matching_hint_control = matching_context and (control_spec["source"] == "none" or (
        control_spec["source"] == "kv" and spec["side_bits"] == control_spec["side_bits"]
        and spec["page_shared"] == control_spec["page_shared"]))
    payload_reduced = None if partition_delta is None else partition_delta["learned_stage_payload_bytes"] < 0
    packet_delta = a["packet_bytes"] - b["packet_bytes"]
    return dict(method=row["name"], control=other["name"], packet_bytes_delta=packet_delta,
        component_bytes_delta=component_delta, packet_partition_delta=partition_delta,
        charged_state_bytes_delta=row["accounting"].get("state", {}).get("total_charged_bytes", 0) -
            other["accounting"].get("state", {}).get("total_charged_bytes", 0),
        producer_gain_requires_coded_payload_reduction=True,
        learned_stage_payload_reduced=payload_reduced,
        same_layout_and_conditioning_control=bool(matching_hint_control),
        producer_specific_payload_gain_observed=bool(spec["source"] == "hidden" and
            matching_hint_control and payload_reduced),
        packet_gain_without_main_payload_reduction=bool(packet_delta < 0 and partition_delta is not None
            and partition_delta["main_payload_bytes"] >= 0),
        estimated_net_savings_pp_delta=row["accounting"]["full_cache_estimate"]["net_reduction_pct"] -
            other["accounting"]["full_cache_estimate"]["net_reduction_pct"],
        decoder_median_seconds_delta=a.get("median_wall_seconds", 0.) -
            b.get("median_wall_seconds", b.get("decode_wall_seconds", 0.)),
        interpretation="Hint/model provenance needs reduced learned-stage payload against the matched same-layout control; shorter names/JSON headers and raw/zlib layout changes alone are not producer information gains.")


def _diagnostic_records(records, maximum):
    ordered = sorted(records, key=lambda row: (row["layer"], row["start"]))
    indices = np.linspace(0, len(ordered) - 1, min(maximum, len(ordered)), dtype=np.int64)
    return [ordered[int(index)] for index in indices]


def _hint_ablations(codes, spec, seed):
    original = {address: np.asarray(value, dtype=np.uint8).copy() for address, value in codes.items()}
    if spec["source"] == "none":
        return [dict(kind="original", status="applied", codes=original)]
    rows = [dict(kind="original", status="applied", codes=original),
        dict(kind="zero_id", status="applied", codes={key: np.zeros_like(value) for key, value in original.items()})]
    rng = np.random.default_rng(seed)
    shuffled = {key: value.copy() for key, value in original.items()}
    if spec["page_shared"]:
        grouped = {}
        for address in sorted(original):
            grouped.setdefault(address[0], []).append(address)
        eligible = []
        for addresses in grouped.values():
            if len(addresses) < 2:
                continue
            order = rng.permutation(len(addresses)).tolist()
            if order == list(range(len(addresses))):
                order = order[1:] + order[:1]
            for index, address in enumerate(addresses):
                shuffled[address] = original[addresses[order[index]]].copy()
            eligible.extend(addresses)
        rows.append(dict(kind="shuffled_pages_within_layer", status="applied" if eligible else "not_applicable",
            eligible_pages=len(eligible), codes=shuffled,
            explanation="A page-shared ID is constant within its page. Shuffle whole IDs only among pages of the same layer; layers with one page are unchanged."))
    else:
        for address, value in original.items():
            shuffled[address] = value[rng.permutation(len(value))].copy()
        rows.append(dict(kind="shuffled_tokens_within_page", status="applied", codes=shuffled))
    return rows


def _cdf_diagnostic(records, prepared, codes):
    """Score fixed integer CDFs with only earlier exact fields as context.

    Full targets are used only for scoring and teacher forcing after each
    stage. No entropy stream is built and no TEST data is inspected here.
    """
    from . import compact_codec as codec
    spec = prepared.spec
    stages, total_values = {}, 0
    began = time.perf_counter()
    for record in records:
        target = np.concatenate((record["k"], record["v"]), axis=1)
        channels = target.shape[1] // 2
        selected = slice(channels, None) if spec["key_first"] else slice(None)
        known = target & codec.initial_mask(spec)
        if spec["key_first"]:
            known[:, :channels] = target[:, :channels]
        hints = codes[(record["layer"], record["start"])]
        total_values += target.size
        for index, (shift, width) in enumerate(codec.stage_plan(spec)):
            counts = np.asarray(prepared.predict_counts(record["layer"], record["start"], known.copy(), hints, index))
            if (counts.shape != target.shape + (1 << width,) or not np.issubdtype(counts.dtype, np.integer)
                    or np.any(counts <= 0) or np.any(counts.sum(axis=-1) != codec.TOTAL)):
                raise ValueError("Invalid compact diagnostic integer counts")
            actual = ((target[:, selected] >> shift) & ((1 << width) - 1)).astype(np.int64)
            probabilities = counts[:, selected, :] / float(codec.TOTAL)
            mass = np.take_along_axis(probabilities, actual[..., None], axis=-1)[..., 0]
            loss = -np.log2(mass)
            errors = np.argmax(probabilities, axis=-1) != actual
            row = stages.setdefault(str(index), dict(shift=shift, width=width,
                symbols=0, integer_cdf_bits=0., argmax_errors=0, by_kind={}))
            row["symbols"] += actual.size; row["integer_cdf_bits"] += float(loss.sum())
            row["argmax_errors"] += int(errors.sum())
            for kind, section in (("K", slice(0, channels)), ("V", slice(channels, None))):
                if spec["key_first"]:
                    if kind == "K":
                        continue
                    section = slice(None)
                item = row["by_kind"].setdefault(kind, dict(symbols=0, integer_cdf_bits=0., argmax_errors=0))
                item["symbols"] += loss[:, section].size
                item["integer_cdf_bits"] += float(loss[:, section].sum())
                item["argmax_errors"] += int(errors[:, section].sum())
            # Update only after predicting the current tuple. Future bits are absent.
            known[:, selected] |= (actual.astype(np.uint16) << shift)
    by_kind = {}
    for row in stages.values():
        row["integer_cdf_bits_per_symbol"] = row["integer_cdf_bits"] / row["symbols"]
        for kind, item in row["by_kind"].items():
            total = by_kind.setdefault(kind, dict(symbols=0, integer_cdf_bits=0., argmax_errors=0))
            for key in total:
                total[key] += item[key]
            item["integer_cdf_bits_per_symbol"] = item["integer_cdf_bits"] / item["symbols"]
    bits = sum(row["integer_cdf_bits"] for row in stages.values())
    for item in by_kind.values():
        item["integer_cdf_bits_per_symbol"] = item["integer_cdf_bits"] / item["symbols"]
    return dict(integer_cdf_bits=bits, original_KV_values=total_values,
        integer_cdf_bits_per_original_KV_value=bits / total_values, stages=stages, by_kind=by_kind,
        wall_seconds=time.perf_counter() - began, teacher_forced_known_fields=True,
        entropy_streams_constructed=False,
        scope="DEV-only fixed-model integer-CDF NLL diagnostic. This excludes hint/header/state costs and does not measure a new packet or achievable coding gain.")


def _dev_hint_ablations(records, prepared, codes, *, maximum, seed):
    selected = _diagnostic_records(records, maximum)
    addresses = [(record["layer"], record["start"]) for record in selected]
    selected_original = {address: codes[address] for address in addresses}
    rows = []
    original_rate = None
    for ablation in _hint_ablations(selected_original, prepared.spec, seed):
        altered = ablation.pop("codes")
        selected_codes = {address: altered[address] for address in addresses}
        row = dict(ablation, hint_summary=_hint_summary(selected_codes, prepared.spec),
            changed_hint_ids=sum(int(np.count_nonzero(altered[address] != codes[address])) for address in addresses))
        if row["status"] == "applied":
            row["rate"] = _cdf_diagnostic(selected, prepared, altered)
            if original_rate is None:
                original_rate = row["rate"]["integer_cdf_bits_per_original_KV_value"]
            row["NLL_delta_vs_original_bits_per_KV_value"] = row["rate"]["integer_cdf_bits_per_original_KV_value"] - original_rate
        rows.append(row)
    return dict(model_id=prepared.model_id, spec=dict(prepared.spec),
        sampled_addresses=[list(address) for address in addresses], maximum_pages=maximum,
        coverage=dict(selected_pages=len(addresses), total_dev_pages=len(records),
            selected_layers=sorted({address[0] for address in addresses}),
            selected_page_fraction=len(addresses) / len(records)),
        prespecified_sampling="Evenly spaced indices in sorted DEV(layer,start) records",
        rows=rows, selection_effect="Diagnostic only; does not tune model, code, stage layout or winner")


def _decisions(report):
    chosen = report.get("selection", {}).get("winner")
    cases = []
    specs = {row["name"]: row for row in report.get("specs", [])}
    for case in report.get("cases", []):
        rows = {row["name"]: row for row in case["results"] if row.get("status") == "ok"}
        pairs = []
        for name, row in rows.items():
            spec = specs.get(name)
            if spec is None:
                continue
            controls = ["field_baseline", "frozen_exponent_control", "lsb_local_none"]
            if spec["layout"] == "msb":
                controls += ["msb_pool_none", "msb_pool_kv2"]
            elif spec["key_first"]:
                controls += ["lsb_kfirst_none", "lsb_kfirst_kv2"]
            else:
                controls += ["lsb_pool_none", "lsb_page_kv2" if spec["page_shared"] else "lsb_pool_kv2"]
            for control in dict.fromkeys(controls):
                if control not in rows or control == name:
                    continue
                other = rows[control]
                pairs.append(_comparison(row, other, spec, specs.get(control)))
        if chosen in rows:
            chosen_net = rows[chosen]["accounting"]["full_cache_estimate"]["net_reduction_pct"]
            confirmed = all(chosen_net > rows[name]["accounting"]["full_cache_estimate"]["net_reduction_pct"]
                for name in ("field_baseline", "frozen_exponent_control") if name in rows and name != chosen)
        else:
            confirmed = None
        cases.append(dict(name=case["name"], tokens=case["tokens"], frozen_dev_winner=chosen,
            frozen_winner_beats_storage_controls=confirmed, comparisons=pairs))
    return dict(format="compact_prefill_decisions_v1", status=report["status"],
        selection=report.get("selection"), cases=cases,
        scope="Winner chosen on DEV state-inclusive estimate only. TEST reports confirmation or failure; it never replaces the frozen winner. Storage-first selection is not a throughput or novelty verdict.")


def _save(output, report):
    write_json(output / "results.json", report)
    write_json(output / "decision_summary.json", _decisions(report))
    write_json(output / "producer_summary.json", dict(status=report["status"],
        training_captures=report.get("training_captures", []), development=report.get("development", {}),
        test_captures=[dict(name=case["name"], tokens=case["tokens"], capture=case["capture"]) for case in report.get("cases", [])]))
    fields = ["case", "tokens", "method", "status", "packet_bytes", "sample_packet_savings_pct",
        "estimated_net_savings_pct", "charged_state_bytes", "decode_seconds"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in report.get("cases", []):
            for row in case["results"]:
                measured, acc = row.get("measurement", {}), row.get("accounting", {})
                writer.writerow(dict(case=case["name"], tokens=case["tokens"], method=row["name"], status=row["status"],
                    packet_bytes=measured.get("packet_bytes"), sample_packet_savings_pct=acc.get("sample", {}).get("payload_reduction_pct"),
                    estimated_net_savings_pct=acc.get("full_cache_estimate", {}).get("net_reduction_pct"),
                    charged_state_bytes=acc.get("state", {}).get("total_charged_bytes"),
                    decode_seconds=measured.get("median_wall_seconds", measured.get("decode_wall_seconds"))))
    lines = ["# Compact prefill metadata and progressive exact KV study", "",
        f"Status: **{report['status']}**", "", report["scope"], "",
        "Frozen DEV winner: **" + str(report.get("selection", {}).get("winner", "pending")) + "**.", "",
        "| Case | Tokens | Method | Packet reduction | Net full-cache estimate | Decode seconds |",
        "|---|---:|---|---:|---:|---:|"]
    for case in report.get("cases", []):
        for row in case["results"]:
            acc, measured = row.get("accounting", {}), row.get("measurement", {})
            packet = acc.get("sample", {}).get("payload_reduction_pct")
            net = acc.get("full_cache_estimate", {}).get("net_reduction_pct")
            decode = measured.get("median_wall_seconds", measured.get("decode_wall_seconds"))
            fmt = lambda value: "—" if value is None else f"{value:.3f}"
            lines.append(f"| {case['name']} | {case['tokens']} | {row['name']} | {fmt(packet)}% | {fmt(net)}% | {fmt(decode)} |")
    lines += ["", "All reported storage numbers use real complete packets. Models, lookup parameters, calibration, hint padding, raw fallbacks and all framing are charged. TEST cannot change DEV selection.", "",
        "A producer-information gain requires reduced learned-stage payload against the same-layout no-hint/KV control. Descriptive header/name length, hint cost and full state deltas are reported separately.", "",
        "The current reference performs entropy coding on CPU and probability inference on the selected device. All raw KV remains resident during capture. Timings include diagnostic staging; no fused GPU decoder, SSD transfer, concurrent admission or serving speedup is established.", "",
        "Full DEV continuation checks: " + ", ".join(row["name"] + ": " + row["status"] for row in report.get("generation_checks", [])), "",
        "Raw categorical hints are retained under hints/ for future TRAIN-only fitting and DEV/TEST sensitivity analysis. Raw KV and hidden captures are not exported."]
    if report.get("error"):
        lines += ["", "Error: " + report["error"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _centers(captures):
    rows = []
    for layer in range(len(captures[0])):
        words = np.concatenate([np.concatenate(pair[layer], axis=1) for pair in captures])
        rows.append(np.floor(np.median((words.astype(np.uint32) >> 7) & 255, axis=0)).astype(np.uint8))
    return np.stack(rows)


def _capture_metadata(captured, fixture, tokens):
    return dict(split=fixture["split"], name=fixture["name"], tokens=tokens,
        sha256=captured["producer_records_sha256"],
        producer_records_sha256=captured["producer_records_sha256"],
        producer_inputs_sha256=captured["producer_inputs_sha256"],
        input_ids_sha256=captured["input_ids_sha256"], kv_words_sha256=captured["kv_words_sha256"],
        geometry=captured["geometry"], producer_audit=captured["producer_audit"], timings=captured["timings"])


def main(argv=None):
    from . import compact_models as learned
    from .runner import _source_manifest
    parser = _parser(); args = parser.parse_args(argv); _defaults(args)
    specs = _validate(args, parser, learned.default_specs())
    output = args.output or Path("runs") / ("compact_prefill_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True); output = output.resolve()
    report = dict(format="qwen_compact_prefill_study_v1", status="running", specs=specs,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source_manifest=_source_manifest(), models={}, cases=[], generation_checks=[],
        scope="Exact original-BF16 page-local progressive coding with zero-payload known-field summaries, 1/2/4-bit actual-prefill hints, page sharing and exact same-page K-to-V controls. Real complete CPU-reference packets and full state count. Full-cache/concurrency numbers are estimates, not measured serving or GPU codec gains.")
    write_json(output / "source_manifest.json", report["source_manifest"]); _save(output, report)
    began = time.perf_counter()
    try:
        torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
        device = runtime.select_device(args.device)
        report["environment"] = runtime.doctor(device, probe=False)
        _note("Loading model for compact prefill study on " + str(device))
        if args.tiny:
            from .producer_study import _tiny
            report["input_preflight"] = previous._preflight_inputs(None, args, test_fixtures=FRESH_TEST)
            model, tokenizer, identity = _tiny(device, args)
        else:
            def validate_inputs(tokenizer):
                report["input_preflight"] = previous._preflight_inputs(tokenizer, args, test_fixtures=FRESH_TEST)
                _note("Prompt-budget preflight passed before weights")
            model, tokenizer, identity = runtime.load_model(args.model, args.revision, device,
                args.local_files_only, tokenizer_validator=validate_inputs)
        report["model_identity"] = identity
        layers, head_dim = int(model.config.num_hidden_layers), int(model.config.head_dim)
        if args.max_pairs < layers:
            raise ValueError("--max-pairs must cover every layer")
        inverse, scaling = previous._rope(model); signature = (inverse.tobytes(), scaling)
        train_records, train_matrices, metadata = [], [], []
        for fixture in TRAIN:
            _note("TRAIN producer capture: " + fixture["name"])
            captured, records, context = field_study._capture(model, tokenizer, fixture,
                args.train_tokens, device, args)
            if not context.get("rope_stable"):
                raise ValueError("Compact study requires stable RoPE")
            train_records.append(records); train_matrices.append(captured["matrices"])
            metadata.append(_capture_metadata(captured, fixture, args.train_tokens))
        report["training_captures"] = metadata
        hashes = [field_study._producer_inputs_hash(rows) for rows in train_records]
        (output / "calibration").mkdir(exist_ok=True)
        anchor = previous.fit_anchor(train_matrices, metadata, identity, device)
        anchor.dump(output / "calibration" / "frozen_exponent_state.bin")
        report["anchor_state"] = dict(serialized_bytes=anchor.serialized_bytes,
            resident_bytes=anchor.physical_nbytes, file="calibration/frozen_exponent_state.bin",
            sha256=hashlib.sha256(anchor.wire_bytes).hexdigest())
        field, field_info = field_study._field_state(_centers(train_matrices), output, head_dim)
        report["field_state"] = field_info
        _note("Fitting TRAIN-only compact models: " + ", ".join(args.variants))
        started = time.perf_counter()
        models = learned.fit(train_records, output / "calibration", training_metadata=metadata,
            model_identity=identity, head_dim=head_dim, steps=args.steps, width=args.width,
            batch_tokens=args.batch_tokens, channels_per_step=args.channels_per_step,
            device=device, seed=args.seed, specs=specs)
        report["training_wall_seconds"] = time.perf_counter() - started
        if hashes != [field_study._producer_inputs_hash(rows) for rows in train_records]:
            raise AssertionError("TRAIN producer records changed while fitting")
        report["training_hint_arrays"] = {}
        for name, fitted in models.items():
            reloaded = learned.load(fitted.path, expected_model_id=fitted.model_id,
                expected_model_identity=identity,
                expected_training_hashes=fitted.metadata["training_hashes"])
            if reloaded.model_id != fitted.model_id:
                raise AssertionError("Reloaded compact predictor identity changed")
            models[name] = reloaded; report["models"][name] = _artifact(reloaded, output)
            prepared = reloaded.prepare(device)
            for ordinal, rows in enumerate(train_records):
                codes = {(row["layer"], row["start"]): prepared.encode_hints(row) for row in rows}
                report["training_hint_arrays"].setdefault(name, []).append(_hint_file(output,
                    f"train_{ordinal}_{name}", codes, reloaded.spec,
                    placement="Post-fit replay of actual TRAIN hook records; no DEV/TEST fitting"))
            del prepared
        freeze = dict(format="compact_prefill_freeze_v1", model_identity=identity,
            source_sha256=report["source_manifest"]["sources_sha256"], training_input_hashes=hashes,
            models=report["models"], field_state=field_info, anchor_state=report["anchor_state"],
            protocol="TRAIN fitting only. DEV freezes one storage-first winner. TEST reports all candidates without tuning.")
        write_json(output / "frozen_manifest.json", freeze); report["freeze"] = freeze
        del train_records, train_matrices; gc.collect()

        _note("DEV actual producer capture and exact packet comparison")
        inline = InlineHints(models, device)
        development, records, context = field_study._capture(model, tokenizer, DEV[0],
            args.dev_tokens, device, args, inline=inline)
        inline.release_prepared()
        if (context["rope_inv_freq"].tobytes(), float(context["rope_attention_scaling"])) != signature:
            raise ValueError("DEV RoPE differs from TRAIN")
        report["development"] = {key: value for key, value in development.items() if key != "matrices"}
        report["development_results"] = []; report["development_hint_ablations"] = {}; dev_packets = {}
        # Prespecified selection cache length avoids charging state against the short DEV prefix.
        selection_tokens = max(args.lengths)
        selection_raw = development["raw_payload_bytes"] * selection_tokens // args.dev_tokens
        for name, fitted in models.items():
            _note("DEV complete compact packets: " + name)
            prepared = fitted.prepare(device)
            if fitted.spec["source"] != "none":
                report["development_hint_ablations"][name] = _dev_hint_ablations(
                    records, prepared, inline.codes[name], maximum=args.ablation_max_pairs, seed=args.seed)
            wire, measured = _encode(records, prepared, development["geometry"], args, inline.codes[name])
            restored, decoded = restore_pages(wire, prepared, require_full=True, seed=args.seed)
            if previous._matrices_hash(restored) != development["kv_words_sha256"]:
                raise AssertionError("DEV compact packet did not restore complete original cache")
            wire, artifact = base._packet_file(output, "dev_" + name, wire)
            measured.update(roundtrip_exact=True, independent_packet_decode=decoded, packet_artifact=artifact,
                hint_array=_hint_file(output, "dev_" + name, inline.codes[name], fitted.spec,
                    placement="Inside actual DEV attention callback"))
            measured["median_wall_seconds"] = decoded["wall_seconds"]
            acc = account(measured["raw_sample_bytes"], len(wire), selection_raw,
                fitted.state_bytes, prepared.physical_nbytes, 0, 0)
            report["development_results"].append(dict(name=name, status="ok", measurement=measured, accounting=acc))
            dev_packets[name] = wire
            del restored, prepared; _save(output, report)
        field_wire, field_measure = field_study._encode(records, field, development["geometry"], args, None, 7, "syndrome")
        field_restored, _ = field_study.restore_pages(field_wire, field, require_full=True, seed=args.seed)
        if previous._matrices_hash(field_restored) != development["kv_words_sha256"]:
            raise AssertionError("DEV field control did not restore complete cache")
        field_wire, field_artifact = base._packet_file(output, "dev_field_baseline", field_wire)
        field_measure.update(roundtrip_exact=True, packet_artifact=field_artifact)
        report["development_results"].append(dict(name="field_baseline", status="ok", measurement=field_measure,
            accounting=account(field_measure["raw_sample_bytes"], len(field_wire), selection_raw,
                field_info["serialized_bytes"], field_info["resident_bytes"], 0, 0)))
        control = previous._baseline(development["matrices"], anchor,
            list(range(0, args.dev_tokens, args.tile_tokens)), args.tile_tokens, head_dim)
        report["development_results"].append(dict(name="frozen_exponent_control", status="ok", measurement=control,
            accounting=account(control["raw_sample_bytes"], control["packet_bytes"], selection_raw,
                0, 0, anchor.serialized_bytes, anchor.physical_nbytes)))
        chosen = _select(report["development_results"])
        best_learned = _select([row for row in report["development_results"] if row["name"] in models])
        report["selection"] = dict(winner=chosen, best_learned_candidate=best_learned,
            split="dev", name=DEV[0]["name"], tokens=args.dev_tokens,
            selection_reference_cache_tokens=selection_tokens,
            rule="Minimum complete DEV packet/raw ratio scaled to prespecified max requested TEST length, plus max(serialized,resident) full state; lexicographic tie break. TEST never replaces this selection.",
            input_ids_sha256=development["input_ids_sha256"], producer_inputs_sha256=development["producer_inputs_sha256"])
        freeze = dict(freeze, selection=dict(report["selection"]))
        write_json(output / "frozen_manifest.json", freeze); report["freeze"] = freeze
        check_names = list(dict.fromkeys([best_learned, *[name for name in (
            "lsb_pool_none", "lsb_pool_hidden2", "msb_pool_hidden2", "lsb_kfirst_hidden2")
            if name in models], "field_baseline"]))
        report["generation_check_scope"] = dict(methods=check_names,
            frozen_storage_winner_has_continuation_check=chosen in check_names,
            frozen_exponent_control_scope="Every measured anchor page is independently serialized/restored and checked by the existing exponent baseline; no separate full-prefix continuation is run for this control.")
        for name in check_names:
            if name == "field_baseline":
                prepared, wire = field, field_wire
                def restore(wire=wire, prepared=prepared):
                    return field_study.restore_pages(wire, prepared, require_full=True, seed=args.seed)[0]
            else:
                prepared, wire = models[name].prepare(device), dev_packets[name]
                def restore(wire=wire, prepared=prepared):
                    return restore_pages(wire, prepared, require_full=True, seed=args.seed)[0]
            _note("DEV packet-only continuation: " + name)
            check = runtime.generation_check(model, development["matrices"], restore, device,
                first_token=development["first_token"], steps=args.generation_steps)
            if not previous._generation_passed(check):
                raise AssertionError("DEV compact finite/bitwise continuation failed for " + name)
            report["generation_checks"].append(dict(name=name, status="ok", result=check,
                complete_prefix_tokens=args.dev_tokens, decode_inputs="Persisted own-page container and immutable state only"))
            del restore, prepared
        del development, records, context, inline, dev_packets, field_restored; gc.collect(); _save(output, report)

        for length in args.lengths:
            starts = previous._starts(length, args.tile_tokens, layers, args.max_pairs)
            for fixture in FRESH_TEST[:args.cases]:
                _note(f"TEST producer capture: {fixture['name']}, {length} tokens, starts {starts}")
                inline = InlineHints(models, device)
                captured, records, context = field_study._capture(model, tokenizer, fixture,
                    length, device, args, starts, inline=inline)
                inline.release_prepared()
                if (context["rope_inv_freq"].tobytes(), float(context["rope_attention_scaling"])) != signature:
                    raise ValueError("TEST RoPE differs from frozen TRAIN/DEV")
                matrices = captured.pop("matrices")
                expected = {(layer, start) for layer in range(layers) for start in starts}
                if {(row["layer"], row["start"]) for row in records} != expected:
                    raise AssertionError("TEST addresses differ from matched controls")
                raw_sample = sum(row["k"].nbytes + row["v"].nbytes for row in records)
                total_pages = layers * ((length + args.tile_tokens - 1) // args.tile_tokens)
                case = dict(name=fixture["name"], split="test", tokens=length, capture=captured,
                    selected_starts=starts, matched_record_addresses=sorted(map(list, expected)), results=[],
                    coverage=dict(measured_pages=len(records), total_pages=total_pages,
                        measured_page_fraction=len(records) / total_pages, measured_raw_bytes=raw_sample,
                        total_raw_cache_bytes=captured["raw_payload_bytes"],
                        measured_raw_fraction=raw_sample / captured["raw_payload_bytes"]))
                report["cases"].append(case)
                control = previous._baseline(matrices, anchor, starts, args.tile_tokens, head_dim)
                control["maximum_dependency_depth"] = 0
                case["results"].append(dict(name="frozen_exponent_control", status="ok", measurement=control,
                    accounting=account(control["raw_sample_bytes"], control["packet_bytes"], captured["raw_payload_bytes"],
                        0, 0, anchor.serialized_bytes, anchor.physical_nbytes)))
                field_wire, measured = field_study._encode(records, field, captured["geometry"], args, None, 7, "syndrome")
                field_wire, artifact = base._packet_file(output, f"test_{fixture['name']}_{length}_field", field_wire)
                repeated = field_study._decoder_repeats(field_wire, field, matrices, starts, args)
                measured.update(repeated, roundtrip_exact=True, packet_artifact=artifact)
                case["results"].append(dict(name="field_baseline", status="ok", measurement=measured,
                    accounting=account(measured["raw_sample_bytes"], len(field_wire), captured["raw_payload_bytes"],
                        field_info["serialized_bytes"], field_info["resident_bytes"], 0, 0)))
                for name, fitted in models.items():
                    _note("TEST exact compact pages: " + name)
                    initial = runtime.reset_peak_memory(device)
                    prepared, preparation = runtime.timed_call(lambda fitted=fitted: fitted.prepare(device), device)
                    wire, measured = _encode(records, prepared, captured["geometry"], args, inline.codes[name])
                    wire, artifact = base._packet_file(output, f"test_{fixture['name']}_{length}_{name}", wire)
                    measured.update(_repeats(wire, prepared, matrices, starts, args), packet_artifact=artifact,
                        state_preparation=preparation, initial_device_memory=initial,
                        after_device_memory=runtime.memory_snapshot(device),
                        hint_array=_hint_file(output, f"test_{fixture['name']}_{length}_{name}", inline.codes[name], fitted.spec,
                            placement="Inside actual TEST attention callback"))
                    case["results"].append(dict(name=name, status="ok", measurement=measured,
                        accounting=account(measured["raw_sample_bytes"], len(wire), captured["raw_payload_bytes"],
                            fitted.state_bytes, prepared.physical_nbytes, 0, 0)))
                    del prepared, wire; _save(output, report)
                del captured, records, context, inline, matrices; gc.collect()
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _note(report["error"])
    report["total_wall_seconds"] = time.perf_counter() - began
    _save(output, report)
    _note("Report: " + str(output / "summary.md")); _note("Status: " + report["status"])
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
