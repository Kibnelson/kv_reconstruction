"""Actual-prefill metadata with exact, page-independent bitplane coset coding.

Neural code extraction/prediction may run on the selected accelerator. Packet
coding is a CPU reference. Reported rates count real serialized packets, modes,
escapes, framing, side codes and full state; no concurrent-serving gain is
inferred. Every candidate retains its own metadata even when groups escape.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import gc
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from . import model_runtime as runtime, neural_study as previous, side_study as base
from .fixtures import TRAIN, DEV, TEST
from .reporting import write_json


def _note(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--model", default=runtime.DEFAULT_MODEL)
    p.add_argument("--revision")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--tiny", action="store_true", help="No-download random tiny Qwen; functionality only")
    p.add_argument("--rank", type=int)
    p.add_argument("--width", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-tokens", type=int)
    p.add_argument("--channels-per-step", type=int)
    p.add_argument("--channel-sampling", choices=("random", "balanced"), default="random")
    p.add_argument("--lr-horizon", type=int, help="Shared cosine horizon for matched short/long training")
    p.add_argument("--diagnostic-updates", nargs="+", type=int,
                   help="Fixed DEV probe at initialization and these updates; diagnostic only, no early stopping")
    p.add_argument("--diagnostic-tokens", type=int, default=64,
                   help="Fixed DEV tokens per layer, contiguous head/tail segments; no TEST probes")
    p.add_argument("--fresh-test-fixtures", action="store_true",
                   help="Use prespecified training-study holdouts instead of previously inspected TEST fixtures")
    p.add_argument("--residue-bits", nargs="+", type=int, default=[0, 2, 4, 6],
                   help="DEV grid; freeze one global retained-mantissa width per model before TEST")
    p.add_argument("--modes", nargs="+", choices=("none", "kv", "kv_canonical", "hidden", "hidden_norm"),
                   default=["none", "kv", "kv_canonical", "hidden", "hidden_norm"])
    p.add_argument("--train-tokens", type=int)
    p.add_argument("--dev-tokens", type=int)
    p.add_argument("--lengths", nargs="+", type=int)
    p.add_argument("--cases", type=int, choices=(1, 2, 3), default=1)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--tile-tokens", type=int)
    p.add_argument("--max-pairs", type=int, default=72)
    p.add_argument("--generation-steps", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--output", type=Path)
    return p


def _defaults(args):
    for key, value in dict(rank=2 if args.tiny else 8, width=16 if args.tiny else 32,
        steps=2 if args.tiny else 1000, batch_tokens=8 if args.tiny else 32,
        channels_per_step=8 if args.tiny else 64,
        train_tokens=64 if args.tiny else 512, dev_tokens=32 if args.tiny else 128,
        lengths=[64] if args.tiny else [2048], chunk_tokens=32 if args.tiny else 512,
        tile_tokens=32 if args.tiny else 64).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.tiny and args.device == "auto":
        args.device = "cpu"
    if args.revision is None and args.model == runtime.DEFAULT_MODEL:
        args.revision = runtime.DEFAULT_REVISION


def _validate(args, parser):
    positive = [args.rank, args.width, args.steps, args.batch_tokens, args.channels_per_step,
        args.train_tokens, args.dev_tokens, args.chunk_tokens, args.tile_tokens,
        args.max_pairs, args.threads, args.repeats, *args.lengths]
    if any(n < 1 for n in positive) or not 1 <= args.generation_steps <= 64 or args.seed < 0:
        parser.error("Budgets must be positive, generation steps in 1..64 and seed nonnegative")
    if not args.residue_bits or any(not 0 <= n <= 6 for n in args.residue_bits):
        parser.error("--residue-bits must contain values 0..6; residue7 is a separate exact field control")
    if len(set(args.residue_bits)) != len(args.residue_bits) or len(set(args.modes)) != len(args.modes):
        parser.error("Residue widths and modes must be unique")
    if "none" not in args.modes:
        parser.error("--modes must include none for the matched no-code and field controls")
    if args.lr_horizon is not None and args.lr_horizon < args.steps:
        parser.error("--lr-horizon must be at least --steps")
    if args.diagnostic_tokens < 2:
        parser.error("--diagnostic-tokens must be at least 2")
    if args.diagnostic_updates is not None and (not args.diagnostic_updates
            or any(n < 0 or n > args.steps for n in args.diagnostic_updates)):
        parser.error("--diagnostic-updates must be in 0..steps")


def _feature_summary(codes):
    arrays = [np.asarray(value, dtype=np.uint16) for _, value in sorted(codes.items())]
    rank = arrays[0].shape[1] if arrays else 0
    words = np.concatenate(arrays) if arrays else np.empty((0, rank), np.uint16)
    numeric = (words.astype(np.uint32) << 16).view(np.float32)
    finite = np.isfinite(numeric)
    return dict(tokens=int(len(words)), rank=int(rank), bytes=int(words.nbytes),
        bitwise_zero_words=int(np.sum(words == 0)), finite_words=int(np.sum(finite)),
        nonfinite_words=int(np.sum(~finite)),
        distinct_words_by_feature=[int(len(np.unique(words[:, i]))) for i in range(rank)],
        mean_by_feature=[float(np.mean(numeric[:, i])) for i in range(rank)],
        std_by_feature=[float(np.std(numeric[:, i])) for i in range(rank)],
        min_by_feature=[float(np.min(numeric[:, i])) for i in range(rank)],
        max_by_feature=[float(np.max(numeric[:, i])) for i in range(rank)],
        at_latent_clamp_words=int(np.sum(np.abs(numeric) >= 32.)),
        code_sha256=hashlib.sha256(np.ascontiguousarray(words).tobytes()).hexdigest())


def _producer_inputs_hash(records):
    """Cover inputs absent from the older side-probability record hash too."""
    digest = hashlib.sha256(base._records_hash(records).encode())
    for record in sorted(records, key=lambda row: (row["layer"], row["start"])):
        projection = np.ascontiguousarray(record["k_proj"])
        digest.update(base._json(dict(layer=int(record["layer"]), start=int(record["start"]),
            k_proj_dtype=str(projection.dtype), k_proj_shape=list(projection.shape),
            k_norm_eps=float(record["k_norm_eps"]))))
        digest.update(projection.tobytes())
    return digest.hexdigest()


def _capture(*args, **kwargs):
    captured, records, context = base._capture(*args, **kwargs)
    captured["producer_inputs_sha256"] = _producer_inputs_hash(records)
    captured["timings"]["prefill"]["scope"] = (
        "Instrumented forward includes actual producer hooks, diagnostic CPU copies and selected-device "
        "code emission when supplied. CPU syndrome/position packets are constructed afterward. Full raw KV remains resident.")
    return captured, records, context


class InlineCodes(base.InlineCodes):
    """Keep bounded encoder diagnostics alongside the exact emitted words."""
    def __call__(self, record, context):
        super().__call__(record, context)
        for name, prepared in self.prepared.items():
            diagnostic = getattr(prepared, "last_feature_diagnostics", None)
            if diagnostic is not None:
                self.rows[name]["records"][-1]["encoder_diagnostics"] = dict(diagnostic)


def _coding_summary(traces):
    totals = Counter()
    grouped = {"by_bitplane": {}, "by_kind_bitplane": {}, "by_head_bitplane": {}}
    numeric = ("values", "bit_errors", "original_ones", "predicted_ones", "groups", "zero_error_groups",
        "one_error_groups", "multi_error_groups", "recoverable_groups", "retained_short_groups", "retained_raw_groups",
        "mode_bytes", "short_bytes", "raw_group_bytes", "tail_bytes", "tail_bits", "raw_plane_bytes",
        "retained_payload_bytes", "candidate_payload_bytes", "candidate_payload_saving_bytes", "candidate_complete_saving_bytes")
    def add(destination, item):
        for key in numeric:
            if key in item:
                destination[key] = destination.get(key, 0) + int(item[key])
        if "errors_per_group" in item:
            hist = destination.setdefault("errors_per_group", [0] * 32)
            for index, count in enumerate(item["errors_per_group"]):
                hist[index] += int(count)
    for trace in traces:
        totals["pages"] += 1
        totals["pages_using_short_correction"] += int(trace.get("retained_short_groups", 0) > 0)
        totals["feature_bytes"] += int(trace["component_bytes"].get("feature_bytes", 0))
        for plane in trace.get("bitplanes", []):
            bit = str(plane["bit"])
            add(totals, plane)
            add(grouped["by_bitplane"].setdefault(bit, {}), plane)
            for group, label in (("by_kind", "by_kind_bitplane"), ("by_head", "by_head_bitplane")):
                for key, values in plane.get(group, {}).items():
                    add(grouped[label].setdefault(key + "/" + bit, {}), values)
    def rates(item):
        values = item.get("values", 0)
        item["bit_error_rate"] = item.get("bit_errors", 0) / values if values else None
        groups = sum(item.get("errors_per_group", []))
        item["zero_or_one_error_fraction"] = sum(item.get("errors_per_group", [0, 0])[:2]) / groups if groups else None
        return item
    total = rates(dict(totals))
    total["retained_plane_saving_after_feature_bytes"] = total.get("raw_plane_bytes", 0) - total.get("retained_payload_bytes", 0) - total["feature_bytes"]
    return dict(totals=total, **{name: {key: rates(row) for key, row in rows.items()} for name, rows in grouped.items()},
        interpretation="Actual retained short-group counts distinguish exercised syndrome recovery from exact raw escapes. Error histograms include rejected candidates. Code-use ablations, not short-group presence alone, test dependence on the code.")


def _aggregate_traces(traces, field):
    result = {}
    for trace in traces:
        key = str(trace[field])
        row = result.setdefault(key, Counter())
        row["records"] += 1
        row["raw_bytes"] += int(trace["raw_bytes"])
        row["packet_bytes"] += int(trace["packet_bytes"])
        for component, count in trace["component_bytes"].items():
            row[component] += int(count)
    return {key: dict(row, packet_reduction_pct=100 * (1 - row["packet_bytes"] / row["raw_bytes"]))
            for key, row in result.items()}


def _encode(records, prepared, geometry, args, codes, residue_bits, correction):
    from . import syndrome_codec as codec
    began = time.perf_counter()
    pages, traces, components = [], [], Counter()
    for record in sorted(records, key=lambda row: (row["layer"], row["start"])):
        address = (int(record["layer"]), int(record["start"]))
        code = None if codes is None else codes[address]
        wire, trace = codec.encode_page(record, prepared, residue_bits=residue_bits,
            correction=correction, feature_words=code)
        if int(trace["packet_bytes"]) != len(wire) or sum(trace["component_bytes"].values()) != len(wire):
            raise AssertionError("Coset page byte ledger differs from actual wire")
        trace = dict(trace, layer=address[0], start=address[1], tokens=int(record["tokens"]),
                     feature_words_supplied=code is not None)
        pages.append(wire); traces.append(trace)
        components.update(trace["component_bytes"])
    wire = base.pack_pages(pages, layers=geometry["layers"], tokens=geometry["tokens"],
                           channels=geometry["channels"], tile_tokens=args.tile_tokens)
    components["container_framing_bytes"] = len(wire) - sum(map(len, pages))
    raw = sum(int(trace["raw_bytes"]) for trace in traces)
    if sum(components.values()) != len(wire):
        raise AssertionError("Coset container byte ledger differs from actual wire")
    return wire, dict(raw_sample_bytes=raw, packet_bytes=len(wire), records_measured=len(records),
        records_total=geometry["layers"] * ((geometry["tokens"] + args.tile_tokens - 1) // args.tile_tokens),
        component_bytes=dict(components), records=traces, residue_bits=residue_bits, correction=correction,
        coding_summary=_coding_summary(traces),
        complete_packet_bits_per_KV_value=16 * len(wire) / raw,
        encode_wall_seconds=time.perf_counter() - began,
        by_layer=_aggregate_traces(traces, "layer"), by_token_start=_aggregate_traces(traces, "start"),
        maximum_dependency_depth=0, raw_prerequisite_pages=0, inline_codes_used=codes is not None,
        scope="Actual independent pages and reopened indexed container; CPU packet coder and selected-device bit prediction. All raw group escapes and side codes retained; no whole-page fallback.")


def restore_pages(wire, prepared, *, require_full=False, seed=71):
    """Decode in shuffled order from own packet plus frozen state only."""
    from . import syndrome_codec as codec
    geometry, pages = base.unpack_pages(wire)
    layers, tokens, channels, tile = (int(geometry[k]) for k in ("layers", "tokens", "channels", "tile_tokens"))
    addresses, seen = [], set()
    for page in pages:
        metadata = codec.page_metadata(page)
        layer, start, count = (metadata[k] for k in ("layer", "start", "tokens"))
        if (any(isinstance(n, bool) or not isinstance(n, int) for n in (layer, start, count))
            or not 0 <= layer < layers or not 0 <= start < tokens or start % tile
            or count != min(tile, tokens - start) or metadata.get("channels") != channels
            or (layer, start) in seen):
            raise ValueError("Coset page address differs from indexed cache geometry")
        addresses.append((layer, start, count)); seen.add((layer, start))
    if require_full and seen != {(layer, start) for layer in range(layers) for start in range(0, tokens, tile)}:
        raise ValueError("Full coset restore requires every layer/token tile")
    order = np.random.default_rng(seed).permutation(len(pages)).tolist()
    if len(order) > 1 and order == list(range(len(pages))):
        order.reverse()
    restored = [(np.empty((tokens, channels), np.uint16), np.empty((tokens, channels), np.uint16))
                for _ in range(layers)] if require_full else []
    stats, diagnostics = Counter(), []
    began = time.perf_counter()
    for ordinal in order:
        layer, start, count = addresses[ordinal]
        words, stages = codec.decode_page(pages[ordinal], prepared, return_stats=True)
        if words.dtype != np.uint16 or words.shape != (count, 2 * channels):
            raise AssertionError("Coset decoder returned unexpected joint KV geometry")
        if require_full:
            restored[layer][0][start:start + count] = words[:, :channels]
            restored[layer][1][start:start + count] = words[:, channels:]
        else:
            restored.append(dict(layer=layer, start=start, tokens=count,
                k=words[:, :channels].copy(), v=words[:, channels:].copy()))
        for name, value in stages.items():
            if name.endswith("seconds") and isinstance(value, (int, float)):
                stats[name] += value
        diagnostics.append(dict(layer=layer, start=start, tokens=count, **stages))
    returned = 4 * layers * tokens * channels if require_full else sum(row["k"].nbytes + row["v"].nbytes for row in restored)
    return restored, dict(wall_seconds=time.perf_counter() - began, **dict(stats), records=diagnostics,
        records_decoded=len(pages), full_prefix=require_full, order_is_shuffled=order != list(range(len(pages))) if len(pages) > 1 else None,
        decode_order=[[addresses[i][0], addresses[i][1]] for i in order], returned_output_bytes=returned,
        maximum_dependency_depth=0, raw_prerequisite_pages=0, source_arrays_used=False,
        predictor_device=str(prepared.device), packet_backend="CPU NumPy/stdlib syndrome reference",
        scope="Functional own-page restore; output buffers, neural/coder workspace and shared state remain necessary. No serving-capacity bound.")


def _field_state(centers, output, head_dim):
    centers = np.ascontiguousarray(centers, dtype=np.uint8)
    metadata = base._json(dict(format="coset_field_centers_v1", shape=list(centers.shape), dtype="uint8",
        head_dim=int(head_dim), source="Frozen TRAIN-only exponent centers", predictor="none"))
    wire = b"COSFCTR1" + struct.pack("<I", len(metadata)) + metadata + centers.tobytes()
    path = output / "calibration" / "coset_field_centers.bin"
    path.parent.mkdir(exist_ok=True); path.write_bytes(wire)
    return load_field_state(path, expected_state_id=hashlib.sha256(wire).hexdigest())


def load_field_state(path, *, expected_state_id=None):
    """Reopen the complete field-only state; no neural artifact is required."""
    path = Path(path); wire = path.read_bytes()
    identity = hashlib.sha256(wire).hexdigest()
    if expected_state_id is not None and identity != expected_state_id:
        raise ValueError("Field-only state SHA256 differs")
    if len(wire) < 12 or wire[:8] != b"COSFCTR1":
        raise ValueError("Invalid field-only state header")
    count = struct.unpack_from("<I", wire, 8)[0]
    if not 2 <= count <= 65536 or 12 + count > len(wire):
        raise ValueError("Invalid field-only state metadata length")
    try:
        metadata = json.loads(wire[12:12 + count])
        shape, head_dim = metadata["shape"], metadata["head_dim"]
        if (metadata["format"] != "coset_field_centers_v1" or metadata["dtype"] != "uint8"
            or metadata["predictor"] != "none" or base._json(metadata) != wire[12:12 + count]
            or not isinstance(shape, list) or len(shape) != 2
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in shape)
            or isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim < 2
            or head_dim % 2 or shape[1] % (2 * head_dim)
            or len(wire) - 12 - count != shape[0] * shape[1]):
            raise ValueError("Invalid field-only state geometry")
    except (ValueError, TypeError, KeyError, UnicodeDecodeError) as error:
        raise ValueError("Invalid field-only state metadata") from error
    centers = np.frombuffer(wire[12 + count:], dtype=np.uint8).copy().reshape(shape)
    centers.flags.writeable = False
    def no_predict(*args):
        raise AssertionError("The residue7 field control must not call a learned predictor")
    prepared = SimpleNamespace(mode="none", model_id=identity, state_id=identity, device=torch.device("cpu"),
        exponent_centers=centers, state_bytes=len(wire), physical_nbytes=centers.nbytes,
        metadata={"geometry": dict(layers=len(centers), channels=centers.shape[1] // 2, rank=0, head_dim=head_dim)},
        encode_features=lambda record: np.empty((record["tokens"], 0), dtype=np.uint16), predict_bits=no_predict)
    info = dict(model_id=identity, serialized_bytes=len(wire), resident_bytes=centers.nbytes,
        charged_bytes=max(len(wire), centers.nbytes), file=str(path.relative_to(path.parent.parent)),
        sha256=identity, decoder_inputs="Frozen exponent centers only; no learned weights or retained code")
    return prepared, info


def _artifact_info(model, output):
    files = [dict(file=str(Path(path).relative_to(output)), bytes=Path(path).stat().st_size,
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for path in model.artifact_paths]
    if sum(row["bytes"] for row in files) != model.state_bytes:
        raise AssertionError("Complete learned state size differs from artifact bytes")
    return dict(model_id=model.model_id, serialized_state_bytes=model.state_bytes,
        parameter_count=model.parameter_count, metadata=model.metadata, files=files,
        scope="Complete encoder, decoder and normalization state charged. Small codec calibration .bin/.json files are shared; original model weights, raw captures and packet files are omitted by the reports packager.")


def _decoder_repeats(wire, prepared, matrices, starts, args):
    rows = []
    for repeat in range(args.repeats):
        restored, measured = restore_pages(wire, prepared, seed=args.seed + repeat)
        base._check_sample(restored, matrices, starts, args.tile_tokens)
        measured["roundtrip_exact"] = True
        rows.append(measured)
        del restored
    return dict(repeats=args.repeats, wall_seconds_samples=[row["wall_seconds"] for row in rows],
        median_wall_seconds=statistics.median(row["wall_seconds"] for row in rows),
        independent_packet_decode=rows[0], all_repeats_exact=True,
        scope="Repeated full sampled packet decode with resident prepared state; no raw-transfer or SSD comparison")


def _ablation_codes(codes, seed):
    rng = np.random.default_rng(seed)
    return dict(zero={address: np.zeros_like(value) for address, value in codes.items()},
        shuffled={address: value[rng.permutation(len(value))].copy() for address, value in codes.items()})


def _prediction_delta(records, prepared, original, altered, residue_bits):
    mask = np.uint16(0xff80 | ((1 << residue_bits) - 1))
    changed = total = 0
    for record in records:
        address = (record["layer"], record["start"])
        known = np.concatenate((record["k"], record["v"]), axis=1) & mask
        a = prepared.predict_bits(record["layer"], original[address], record["start"], known, residue_bits)
        b = prepared.predict_bits(record["layer"], altered[address], record["start"], known, residue_bits)
        changed += int(np.count_nonzero(a[..., residue_bits:] != b[..., residue_bits:]))
        total += int(a[..., residue_bits:].size)
    return dict(changed_omitted_prediction_bits=changed, omitted_prediction_bits=total,
        changed_fraction=changed / total if total else 0.)


def _decision_summary(report):
    decisions = dict(format="prefill_coset_decisions_v1", status=report["status"],
        selected_residue_bits=report.get("selection", {}), cases=[],
        scope="Matched measured sample deltas and state-inclusive full-cache ESTIMATES. TEST never tunes selection. No novelty, 40%, latency or capacity claim.")
    for case in report["cases"]:
        rows = {row["name"]: row for row in case["results"] if row["status"] == "ok"}
        comparisons = []
        for mode in report["config"]["modes"]:
            name = "coset_" + mode + "_syndrome"
            if name not in rows:
                continue
            current = rows[name]
            for control in ("frozen_exponent_control", "field_baseline", "coset_none_syndrome", "coset_kv_syndrome", "coset_kv_canonical_syndrome"):
                if control not in rows or control == name:
                    continue
                other = rows[control]
                comparisons.append(dict(method=name, control=control,
                    sample_packet_bytes_delta=current["measurement"]["packet_bytes"] - other["measurement"]["packet_bytes"],
                    estimated_net_savings_pp_delta=current["accounting"]["full_cache_estimate"]["net_reduction_pct"] - other["accounting"]["full_cache_estimate"]["net_reduction_pct"]))
        common = case.get("common_residue_methods", {})
        common_comparisons = []
        for mode, name in common.items():
            if name not in rows:
                continue
            for control_mode in ("none", "kv", "kv_canonical"):
                control = common.get(control_mode)
                if control is None or control == name or control not in rows:
                    continue
                current, other = rows[name], rows[control]
                common_comparisons.append(dict(method=name, control=control, residue_bits=case["common_residue_bits"],
                    sample_packet_bytes_delta=current["measurement"]["packet_bytes"] - other["measurement"]["packet_bytes"],
                    estimated_net_savings_pp_delta=current["accounting"]["full_cache_estimate"]["net_reduction_pct"] - other["accounting"]["full_cache_estimate"]["net_reduction_pct"]))
        decisions["cases"].append(dict(name=case["name"], tokens=case["tokens"], comparisons=comparisons,
            comparisons_scope="Per-model DEV-selected residue width; end-to-end result rather than a pure conditioner ablation",
            common_residue_bits=case.get("common_residue_bits"), common_residue_comparisons=common_comparisons,
            syndrome_positions=case.get("syndrome_positions", {})))
    return decisions


def _save(output, report):
    import csv
    write_json(output / "results.json", report)
    fields = ["case", "tokens", "method", "status", "residue_bits", "sample_packet_savings_pct", "estimated_net_savings_pct", "charged_state_bytes", "decode_seconds"]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in report["cases"]:
            for row in case["results"]:
                acc, measured = row.get("accounting", {}), row.get("measurement", {})
                writer.writerow(dict(case=case["name"], tokens=case["tokens"], method=row["name"], status=row["status"],
                    residue_bits=measured.get("residue_bits"), sample_packet_savings_pct=acc.get("sample", {}).get("payload_reduction_pct"),
                    estimated_net_savings_pct=acc.get("full_cache_estimate", {}).get("net_reduction_pct"),
                    charged_state_bytes=acc.get("state", {}).get("total_charged_bytes"), decode_seconds=measured.get("median_wall_seconds", measured.get("decode_wall_seconds"))))
    write_json(output / "decision_summary.json", _decision_summary(report))
    write_json(output / "producer_summary.json", dict(status=report["status"],
        training_captures=report.get("training_captures", []), development=report.get("development", {}),
        test_captures=[dict(name=case["name"], tokens=case["tokens"], capture=case["capture"]) for case in report["cases"]]))
    lines = ["# Prefill metadata and exact vector-coset study", "", f"Status: **{report['status']}**", "", report["scope"], "",
        "| Case | Tokens | Method | Residue bits | Sample packet savings | Net full-cache estimate |",
        "|---|---:|---|---:|---:|---:|"]
    for case in report["cases"]:
        for row in case["results"]:
            acc = row.get("accounting", {})
            def fmt(value):
                return "—" if value is None else f"{value:.2f}%"
            lines.append(f"| {case['name']} | {case['tokens']} | {row['name']} ({row['status']}) | {row.get('measurement', {}).get('residue_bits', '—')} | {fmt(acc.get('sample', {}).get('payload_reduction_pct'))} | {fmt(acc.get('full_cache_estimate', {}).get('net_reduction_pct'))} |")
    lines += ["", "Residue widths are selected on complete DEV packet bytes and frozen before TEST. Candidates keep their side code even if every correction group escapes; no learned whole-page fallback hides the tested path.", "",
        "The syndrome and sparse-position representations are matched controls, not stackable savings. Actual packet components, bitplane/K/V error-group histograms, code-use ablations, layer/position breakdowns and model/packet identities are in results.json and decision_summary.json.", "",
        "Packet coding runs on CPU. Code extraction and bit prediction run on the selected device, with diagnostic staging. Instrumented prefill includes all encoders and retains full raw KV; no isolated prefill overhead, fused GPU decoder, SSD transfer or concurrent-serving benefit is measured.", "",
        "Full DEV packet-only continuation checks:", ""]
    lines += [f"- {check['name']}: {check['status']}" for check in report["generation_checks"]]
    if report.get("error"):
        lines += ["", "Error: " + report["error"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    from . import syndrome_models as learned
    from .neural_accounting import account
    from .runner import _source_manifest
    parser = _parser(); args = parser.parse_args(argv); _defaults(args); _validate(args, parser)
    if args.fresh_test_fixtures:
        from .coset_training_study import FRESH_TEST
        test_fixtures = FRESH_TEST
    else:
        test_fixtures = TEST
    output = args.output or Path("runs") / ("prefill_coset_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must be fresh or empty")
    output.mkdir(parents=True, exist_ok=True); output = output.resolve()
    report = dict(format="qwen_prefill_coset_study_v1", status="running",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        source_manifest=_source_manifest(), model_identity=None, models={}, cases=[], generation_checks=[],
        scope="Exact original-BF16 independent pages using TRAIN-fitted metadata-conditioned bit predictors, selected mantissa residues and verified 31-bit vector-coset/sparse-position corrections. All measured packet and full state bytes count. CPU reference coding, selected-device neural work, sampled full-cache estimates; no compact attention, speedup, serving-capacity or novelty claim.")
    write_json(output / "source_manifest.json", report["source_manifest"]); _save(output, report)
    began = time.perf_counter()
    try:
        device = runtime.select_device(args.device)
        torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
        report["environment"] = runtime.doctor(device, probe=False)
        _note("Loading model for prefill coset study on " + str(device))
        if args.tiny:
            from .producer_study import _tiny
            report["input_preflight"] = previous._preflight_inputs(None, args, test_fixtures=test_fixtures)
            model, tokenizer, identity = _tiny(device, args)
        else:
            def validate_inputs(tokenizer):
                report["input_preflight"] = previous._preflight_inputs(tokenizer, args, test_fixtures=test_fixtures)
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
            captured, records, context = _capture(model, tokenizer, fixture, args.train_tokens, device, args)
            if not context.get("rope_stable"):
                raise ValueError("The prefill coset reference requires stable RoPE")
            train_records.append(records); train_matrices.append(captured["matrices"])
            metadata.append(dict(split="train", name=fixture["name"], tokens=args.train_tokens,
                sha256=captured["producer_records_sha256"], producer_records_sha256=captured["producer_records_sha256"],
                producer_inputs_sha256=captured["producer_inputs_sha256"],
                input_ids_sha256=captured["input_ids_sha256"], kv_words_sha256=captured["kv_words_sha256"],
                geometry=captured["geometry"], producer_audit=captured["producer_audit"], timings=captured["timings"]))
        report["training_captures"] = metadata
        diagnostics_out = {}
        diagnostic_callback = None
        if args.diagnostic_updates is not None:
            from . import syndrome_diagnostics as diagnostics
            _note("Capturing fixed DEV probe before TRAIN fitting; diagnostic-only, no checkpoint selection")
            probe_capture, probe_records, probe_context = _capture(
                model, tokenizer, DEV[0], args.dev_tokens, device, args)
            if (probe_context["rope_inv_freq"].tobytes(), float(probe_context["rope_attention_scaling"])) != signature:
                raise ValueError("Fixed DEV probe RoPE differs from TRAIN")
            probe_records, probe_manifest = diagnostics.make_probe(
                probe_records, tokens_per_layer=args.diagnostic_tokens)
            probe_capture.pop("matrices")
            probe_identity = _producer_inputs_hash(probe_records)
            diagnostic_rows = []
            def diagnostic_callback(*, mode, update, params, geometry, training_reference):
                _note(f"Fixed DEV probe: {mode}, update {update}")
                def memory():
                    return (dict(allocated_bytes=torch.cuda.memory_allocated(device),
                        reserved_bytes=torch.cuda.memory_reserved(device),
                        peak_allocated_since_external_reset_bytes=torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda" else {})
                memory_before = memory()
                probe_started = time.perf_counter()
                metrics = diagnostics.evaluate(params, geometry, mode, probe_records,
                    residue_bits=tuple(sorted(args.residue_bits)), seed=args.seed,
                    rope_scaling=scaling,
                    prior_logits=training_reference["prior_logits"],
                    ablations=update in (0, args.steps))
                metrics["execution"] = dict(device=str(device), evaluation_wall_seconds=time.perf_counter()-probe_started,
                    parameter_resident_bytes=sum(t.numel()*t.element_size() for t in params.values()),
                    memory_before=memory_before, memory_after=memory(),
                    scope="Diagnostic evaluation including CPU statistics/staging. Allocator peaks are cumulative without resetting counters; no isolated inference, kernel or serving measurement.")
                if _producer_inputs_hash(probe_records) != probe_identity:
                    raise AssertionError("Fixed DEV probe inputs changed")
                diagnostic_rows.append(dict(mode=mode, update=update, metrics=metrics))
                write_json(output / "training_probe_progress.json", dict(
                    format="prefill_coset_probe_progress_v1", probe_manifest=probe_manifest,
                    checkpoints=diagnostic_rows, scope="Diagnostic progress only; no early stopping or TEST input"))
                return metrics
        hashes = [_producer_inputs_hash(rows) for rows in train_records]
        anchor = previous.fit_anchor(train_matrices, metadata, identity, device)
        (output / "calibration").mkdir(exist_ok=True); anchor.dump(output / "calibration" / "frozen_exponent_state.bin")
        report["anchor_state"] = dict(serialized_bytes=anchor.serialized_bytes, resident_bytes=anchor.physical_nbytes,
            file="calibration/frozen_exponent_state.bin", sha256=hashlib.sha256(anchor.wire_bytes).hexdigest())
        report["training_budget"] = dict(modes=args.modes, updates_each=args.steps, rank=args.rank, width=args.width,
            requested_batch_tokens=args.batch_tokens, channels_per_step=args.channels_per_step,
            objective="BCE over omitted mantissa bitplanes is a training surrogate; actual complete packets select residue width on DEV",
            matching="Same TRAIN data, decoder width, update schedule and retained code rank where applicable; encoder inputs/state sizes differ and are charged")
        _note("Fitting TRAIN-only bit predictors: " + ", ".join(args.modes))
        started = time.perf_counter()
        models = learned.fit(train_records, output / "calibration", training_metadata=metadata, model_identity=identity,
            inverse=inverse, rope_scaling=scaling, head_dim=head_dim, rank=args.rank, width=args.width,
            steps=args.steps, batch_tokens=args.batch_tokens, channels_per_step=args.channels_per_step,
            device=device, seed=args.seed, modes=tuple(args.modes), sampling=args.channel_sampling,
            lr_horizon=args.lr_horizon, diagnostic_callback=diagnostic_callback,
            diagnostic_updates=args.diagnostic_updates, diagnostics_out=diagnostics_out)
        report["training_wall_seconds"] = time.perf_counter() - started
        report["training_budget"].update(channel_sampling=args.channel_sampling,
            learning_rate_horizon=args.lr_horizon or args.steps,
            timing_scope="Fit wall time includes fixed DEV diagnostic work when enabled; per-checkpoint diagnostic time is logged separately")
        if diagnostic_callback is not None:
            diagnostic_file = output / "training_diagnostics.json"
            write_json(diagnostic_file, dict(format="qwen_prefill_coset_training_diagnostics_v1",
                probe_manifest=probe_manifest, capture=probe_capture,
                evaluation=dict(residue_bits=sorted(args.residue_bits),
                    updates=sorted(set([0, *args.diagnostic_updates, args.steps])),
                    token_budget=args.diagnostic_tokens, ablations_at_updates=sorted(set([0, args.steps])),
                    protocol="Fixed DEV diagnostics never optimize or select checkpoints. Final requested TRAIN update is always frozen; TEST never tunes.") ,
                modes=diagnostics_out))
            raw = diagnostic_file.read_bytes()
            report["training_diagnostics"] = dict(file=diagnostic_file.name, bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())
            del probe_capture, probe_records, probe_context, diagnostic_callback, diagnostic_rows, diagnostics_out
        for name, fitted in models.items():
            reloaded = learned.load(fitted.path, expected_model_identity=identity,
                expected_training_hashes=fitted.metadata["training_hashes"],
                expected_model_id=fitted.model_id, expected_mode=name)
            if reloaded.model_id != fitted.model_id:
                raise AssertionError("Reloaded syndrome model differs from fitted artifact")
            models[name] = reloaded
            report["models"][name] = dict(_artifact_info(reloaded, output), reloaded_from_calibration=True)
        if hashes != [_producer_inputs_hash(rows) for rows in train_records]:
            raise AssertionError("TRAIN producer records changed while fitting")
        none_prepared = models["none"].prepare("cpu")
        field_state, field_info = _field_state(none_prepared.exponent_centers, output, head_dim)
        report["field_state"] = field_info
        freeze = dict(format="prefill_coset_training_freeze_v1", model_identity=identity,
            source_sha256=report["source_manifest"]["sources_sha256"], training_hashes=hashes,
            training_hash_contract="syndrome_study._producer_inputs_hash_v1: side_study._records_hash plus actual k_proj dtype/shape/bytes and k_norm_eps, sorted by layer/start",
            training_capture_hashes=[row["producer_records_sha256"] for row in metadata],
            training_producer_input_hashes=hashes,
            models=report["models"], field_state=field_info, anchor_state=report["anchor_state"],
            protocol="Learned weights frozen before DEV. DEV chooses one residue width per mode; TEST chooses nothing.")
        write_json(output / "frozen_manifest.json", freeze); report["freeze"] = freeze
        del train_records, train_matrices, none_prepared; gc.collect()

        _note("DEV actual producer capture, residue selection and packet-only checks")
        inline = InlineCodes(models, device)
        development, records, context = _capture(model, tokenizer, DEV[0], args.dev_tokens, device, args, inline=inline)
        inline.release_prepared()
        if (context["rope_inv_freq"].tobytes(), float(context["rope_attention_scaling"])) != signature:
            raise ValueError("RoPE context changed after TRAIN")
        report["development"] = {key: value for key, value in development.items() if key != "matrices"}
        report["selection"] = {}; report["development_grid"] = {}; report["development_code_ablations"] = {}
        for name, fitted in models.items():
            _note("DEV residue grid: " + name)
            prepared = fitted.prepare(device); codes = inline.codes[name]
            candidates = []
            for bits in sorted(args.residue_bits):
                wire, measured = _encode(records, prepared, development["geometry"], args, codes, bits, "syndrome")
                restored, decoded = restore_pages(wire, prepared, require_full=True, seed=args.seed)
                if previous._matrices_hash(restored) != development["kv_words_sha256"]:
                    raise AssertionError("A DEV residue candidate did not reconstruct the complete cache")
                candidates.append(dict(residue_bits=bits, packet_bytes=len(wire), roundtrip_exact=True,
                    measurement=measured, independent_packet_decode=decoded))
                del restored
            chosen = min(candidates, key=lambda row: (row["packet_bytes"], row["residue_bits"]))
            bits = chosen["residue_bits"]; report["selection"][name] = bits
            report["development_grid"][name] = candidates
            report.setdefault("code_summaries", {}).setdefault("development", {})[name] = _feature_summary(codes)
            if name != "none":
                ablations = []
                for kind, altered in _ablation_codes(codes, args.seed).items():
                    wire, measured = _encode(records, prepared, development["geometry"], args, altered, bits, "syndrome")
                    restored, _ = restore_pages(wire, prepared, require_full=True, seed=args.seed)
                    if previous._matrices_hash(restored) != development["kv_words_sha256"]:
                        raise AssertionError("DEV code-use ablation did not reconstruct exactly")
                    ablations.append(dict(kind=kind, selection_effect="diagnostic only; never changes selected residue width",
                        packet_bytes=len(wire), packet_delta_vs_supplied_code=len(wire) - chosen["packet_bytes"],
                        roundtrip_exact=True, prediction_delta=_prediction_delta(records, prepared, codes, altered, bits),
                        measurement=measured))
                    del restored
                report["development_code_ablations"][name] = ablations
            wire, measured = _encode(records, prepared, development["geometry"], args, codes, bits, "syndrome")
            wire, artifact = base._packet_file(output, "dev_" + name, wire)
            def restore(wire=wire, prepared=prepared):
                return restore_pages(wire, prepared, require_full=True, seed=args.seed)[0]
            check = runtime.generation_check(model, development["matrices"], restore, device,
                first_token=development["first_token"], steps=args.generation_steps)
            if not previous._generation_passed(check):
                raise AssertionError("Full DEV coset finite/bitwise continuation failed")
            report["generation_checks"].append(dict(name="coset_" + name + "_syndrome", status="ok", result=check,
                complete_prefix_tokens=args.dev_tokens, measurement=measured, packet_artifact=artifact,
                syndrome_recovery_exercised=measured["coding_summary"]["totals"]["pages_using_short_correction"] > 0,
                decode_inputs="Persisted own-page container and immutable prepared state only; no producer captures"))
            del restore, prepared, wire; gc.collect(); _save(output, report)
        freeze = dict(freeze, residue_selection=dict(report["selection"]),
            selection_provenance=dict(split="dev", name=DEV[0]["name"], tokens=args.dev_tokens,
                producer_records_sha256=development["producer_records_sha256"],
                producer_inputs_sha256=development["producer_inputs_sha256"],
                rule="Minimum actual complete DEV packet bytes; tie chooses smaller residue width; no TEST selection"))
        write_json(output / "frozen_manifest.json", freeze); report["freeze"] = freeze
        del development, records, context, inline; gc.collect()

        for length in args.lengths:
            starts = previous._starts(length, args.tile_tokens, layers, args.max_pairs)
            for fixture in test_fixtures[:args.cases]:
                _note(f"TEST producer capture: {fixture['name']}, {length} tokens, starts {starts}")
                inline = InlineCodes(models, device)
                captured, records, context = _capture(model, tokenizer, fixture, length, device, args, starts, inline=inline)
                inline.release_prepared()
                if (context["rope_inv_freq"].tobytes(), float(context["rope_attention_scaling"])) != signature:
                    raise ValueError("RoPE context changed after frozen DEV selection")
                matrices = captured.pop("matrices")
                expected = {(layer, start) for layer in range(layers) for start in starts}
                if {(row["layer"], row["start"]) for row in records} != expected:
                    raise AssertionError("TEST candidate addresses differ from matched controls")
                case = dict(name=fixture["name"], split="test", tokens=length, capture=captured,
                    selected_starts=starts, matched_record_addresses=sorted(map(list, expected)),
                    common_residue_bits=report["selection"]["none"], common_residue_methods={},
                    code_summaries={name: _feature_summary(codes) for name, codes in inline.codes.items()},
                    results=[], syndrome_positions={})
                measured_raw = sum(row["k"].nbytes + row["v"].nbytes for row in records)
                total_pages = layers * ((length + args.tile_tokens - 1) // args.tile_tokens)
                case["coverage"] = dict(measured_pages=len(records), total_pages=total_pages,
                    measured_page_fraction=len(records) / total_pages, measured_raw_bytes=measured_raw,
                    total_raw_cache_bytes=captured["raw_payload_bytes"], measured_raw_fraction=measured_raw / captured["raw_payload_bytes"],
                    scope="Exact sampled pages; full-cache sizes extrapolate measured packet/raw ratio")
                report["cases"].append(case)
                control = previous._baseline(matrices, anchor, starts, args.tile_tokens, head_dim)
                control["maximum_dependency_depth"] = 0
                case["results"].append(dict(name="frozen_exponent_control", status="ok", measurement=control,
                    accounting=account(control["raw_sample_bytes"], control["packet_bytes"], captured["raw_payload_bytes"],
                                       0, 0, anchor.serialized_bytes, anchor.physical_nbytes)))
                baseline_wire, baseline = _encode(records, field_state, captured["geometry"], args, None, 7, "syndrome")
                baseline_wire, artifact = base._packet_file(output, f"test_{fixture['name']}_{length}_field", baseline_wire)
                baseline.update(_decoder_repeats(baseline_wire, field_state, matrices, starts, args),
                    roundtrip_exact=True, container_roundtrip_verified=True, packet_artifact=artifact)
                case["results"].append(dict(name="field_baseline", status="ok", measurement=baseline,
                    accounting=account(baseline["raw_sample_bytes"], len(baseline_wire), captured["raw_payload_bytes"],
                        field_info["serialized_bytes"], field_info["resident_bytes"], 0, 0)))
                for name, fitted in models.items():
                    _note("TEST exact coset/position pages: " + name)
                    initial = runtime.reset_peak_memory(device)
                    prepared, preparation = runtime.timed_call(lambda: fitted.prepare(device), device)
                    widths = report["selection"][name]
                    sizes = {}
                    for correction in ("syndrome", "positions"):
                        wire, measured = _encode(records, prepared, captured["geometry"], args, inline.codes[name], widths, correction)
                        wire, artifact = base._packet_file(output, f"test_{fixture['name']}_{length}_{name}_{correction}", wire)
                        measured.update(_decoder_repeats(wire, prepared, matrices, starts, args),
                            roundtrip_exact=True, container_roundtrip_verified=True, packet_artifact=artifact,
                            state_preparation=preparation, initial_device_memory=initial,
                            after_device_memory=runtime.memory_snapshot(device))
                        sizes[correction] = len(wire)
                        case["results"].append(dict(name="coset_" + name + "_" + correction, status="ok", measurement=measured,
                            accounting=account(measured["raw_sample_bytes"], len(wire), captured["raw_payload_bytes"],
                                fitted.state_bytes, prepared.physical_nbytes, 0, 0),
                            execution_contract=getattr(prepared, "execution_contract", None)))
                        del wire; _save(output, report)
                    case["syndrome_positions"][name] = dict(sizes,
                        packet_byte_difference=sizes["syndrome"] - sizes["positions"],
                        interpretation="Matched correction information; any framing difference must be separated from coding gain")
                    common_bits = case["common_residue_bits"]
                    if widths == common_bits:
                        case["common_residue_methods"][name] = "coset_" + name + "_syndrome"
                    else:
                        _note("TEST common-residue metadata comparison: " + name)
                        wire, measured = _encode(records, prepared, captured["geometry"], args, inline.codes[name], common_bits, "syndrome")
                        wire, artifact = base._packet_file(output, f"test_{fixture['name']}_{length}_{name}_common", wire)
                        measured.update(_decoder_repeats(wire, prepared, matrices, starts, args),
                            roundtrip_exact=True, container_roundtrip_verified=True, packet_artifact=artifact,
                            state_preparation=preparation, initial_device_memory=initial,
                            after_device_memory=runtime.memory_snapshot(device))
                        common_name = "coset_" + name + "_common_syndrome"
                        case["results"].append(dict(name=common_name, status="ok", measurement=measured,
                            accounting=account(measured["raw_sample_bytes"], len(wire), captured["raw_payload_bytes"],
                                fitted.state_bytes, prepared.physical_nbytes, 0, 0),
                            execution_contract=getattr(prepared, "execution_contract", None)))
                        case["common_residue_methods"][name] = common_name
                        del wire
                    del prepared; gc.collect()
                del matrices, records, captured, context, inline; gc.collect(); _save(output, report)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _note(report["error"])
    report["elapsed_wall_seconds"] = time.perf_counter() - began
    _save(output, report); _note("Report: " + str(output / "summary.md")); _note("Status: " + report["status"])
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
