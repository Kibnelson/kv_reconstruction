#!/usr/bin/env python3
"""Audit a learned-probability results package without importing model code.

The standard library is sufficient. Sources and the portable launcher's ZIP
are read and hashed, never executed or extracted. This checks report integrity
and consistency; omitted neural artifacts, original KV and a new hardware
replay cannot be independently verified from a reports-only package.
"""
from __future__ import annotations

import argparse
import ast
import base64
from collections import defaultdict
import datetime as dt
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _safe(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes root or is a symlink: {relative}")
    return path


def _load(path):
    return json.loads(path.read_bytes(), parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))


def _close(observed, expected):
    return isinstance(observed, (int, float)) and math.isfinite(observed) and math.isclose(
        observed, float(expected), rel_tol=1e-12, abs_tol=1e-10)


def _rational(value):
    value = Fraction(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def _fraction_field(observed, expected):
    return observed == _rational(expected)


def _stages(traces, timing_key):
    """Sum disjoint named stage walls; retain event sums as separate columns."""
    totals = defaultdict(float)
    for trace in traces:
        timing = trace.get(timing_key, {})
        for name in ("cpu_entropy_encode_seconds", "cpu_entropy_decode_seconds",
                     "anchor_packet_to_cpu_seconds", "anchor_restore_wall_seconds"):
            totals[name] += float(timing.get(name, 0.))
        # anchor_restore includes anchor_decode, so do not add the latter to
        # disjoint wall totals. Its device event is still useful separately.
        anchor = timing.get("anchor_encode", {})
        totals["anchor_encode_wall_seconds"] += anchor.get("wall_seconds", 0.)
        totals["anchor_encode_device_event_seconds"] += anchor.get("device_event_seconds") or 0.
        decoded_anchor = timing.get("anchor_decode", {})
        totals["anchor_decode_device_event_seconds"] += decoded_anchor.get("device_event_seconds") or 0.
        for name in ("input_staging", "neural_prediction", "prediction_output_to_cpu"):
            value = timing.get("prediction", {}).get(name, {})
            totals[name + "_wall_seconds"] += value.get("wall_seconds", 0.)
            totals[name + "_device_event_seconds"] += value.get("device_event_seconds") or 0.
    wall_keys = ("cpu_entropy_encode_seconds", "cpu_entropy_decode_seconds", "anchor_packet_to_cpu_seconds",
                 "anchor_restore_wall_seconds", "anchor_encode_wall_seconds", "input_staging_wall_seconds",
                 "neural_prediction_wall_seconds", "prediction_output_to_cpu_wall_seconds")
    totals["named_disjoint_stage_wall_seconds"] = sum(totals[key] for key in wall_keys)
    return dict(totals, scope="One-pass diagnostic stage sums. Metadata, hashing, indexing and other Python overhead are additional; GPU event and wall columns overlap and must not be summed together.")


def _ledger(measurement):
    traces = measurement.get("record_diagnostics", [])
    neural = [row for row in traces if row["layer"] > 0]
    raw = sum(row["raw_bytes"] for row in neural)
    ideal = sum(row["ideal_continuous_bits"] for row in neural)
    cdf = sum(row["integer_cdf_bits"] for row in neural)
    entropy_bytes = sum(row["component_bytes"].get("cpu_entropy_payload", 0) for row in neural)
    parts = measurement["component_bytes"]
    anchor = sum(value for key, value in parts.items() if key.startswith("anchor_"))
    framing = sum(value for key, value in parts.items() if key in
                  ("page_framing", "page_descriptor", "cache_header", "page_offset_table"))
    return dict(neural_target_raw_bytes=raw, ideal_interval_bits=ideal, integer_cdf_bits=cdf,
        actual_neural_entropy_bytes=entropy_bytes, cdf_rounding_delta_bits=cdf - ideal,
        arithmetic_termination_guard_and_byte_padding_bits=8 * entropy_bytes - cdf,
        anchor_packet_bytes=anchor, all_page_and_container_framing_bytes=framing,
        actual_complete_container_bytes=measurement["serialized_bytes"],
        ideal_interval_reduction_on_neural_targets_pct=100 * (1 - ideal / (8 * raw)) if raw else None,
        scope="Ideal and integer-CDF bit sums apply only to neural target layers. Actual complete container bytes additionally include the independent layer-zero anchor, descriptors, CRC framing and index. Complete decoder state is charged separately.")


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append(dict(name=name, passed=bool(passed), **details))

    def accounting(self, row, raw_cache, label):
        a = row["accounting"]
        raw, packet = a["sample"]["raw_bytes"], a["sample"]["packet_bytes"]
        neural, anchor = a["state"]["neural"], a["state"]["anchor"]
        state = max(neural["serialized_bytes"], neural["resident_bytes"]) + max(anchor["serialized_bytes"], anchor["resident_bytes"])
        ratio = Fraction(packet, raw)
        estimated = raw_cache * ratio
        full = a["full_cache_estimate"]
        ok = (a["state"]["total_charged_bytes"] == state
              and neural["charged_bytes"] == max(neural["serialized_bytes"], neural["resident_bytes"])
              and anchor["charged_bytes"] == max(anchor["serialized_bytes"], anchor["resident_bytes"])
              and a["sample"]["state_amortized_over_sample"] is False
              and full["raw_bytes"] == raw_cache and full["charged_state_bytes"] == state
              and _close(a["sample"]["payload_reduction_pct"], 100 * (1 - ratio))
              and _fraction_field(a["sample"]["payload_reduction_exact"], 1 - ratio)
              and _fraction_field(full["estimated_packet_bytes_exact"], estimated)
              and _fraction_field(full["estimated_total_bytes_exact"], estimated + state)
              and _close(full["net_reduction_pct"], 100 * (1 - (estimated + state) / raw_cache)))
        for concurrent in a["concurrent_shared_state_estimates"]:
            n = concurrent["contexts"]
            ok &= (n > 0 and concurrent["raw_bytes"] == n * raw_cache
                   and concurrent["charged_state_bytes"] == state
                   and _fraction_field(concurrent["estimated_total_bytes_exact"], n * estimated + state)
                   and _close(concurrent["net_reduction_pct"], 100 * (1 - (n * estimated + state) / (n * raw_cache))))
        saving = raw_cache - estimated
        positive = int(state // saving + 1) if saving > 0 else None
        margin = Fraction(3, 5) * raw_cache - estimated
        target = (max(1, -(-(Fraction(state) / margin).numerator // (Fraction(state) / margin).denominator))
                  if margin > 0 else 1 if margin == 0 and state == 0 else None)
        ok &= (a["break_even"]["positive_reduction_contexts"] == positive
               and a["break_even"]["target_reduction_contexts"] == target)
        self.check(label + ": independent byte/state/accounting arithmetic", ok)
        return dict(method=row["name"], status=row["status"], raw_sample_bytes=raw,
            complete_sample_packet_bytes=packet, measured_sample_payload_reduction_pct=float(100 * (1 - ratio)),
            charged_neural_state_bytes=neural["charged_bytes"], charged_anchor_state_bytes=anchor["charged_bytes"],
            charged_total_state_bytes=state, full_raw_cache_bytes=raw_cache,
            estimated_full_packet_bytes=float(estimated), estimated_net_single_cache_reduction_pct=float(100 * (1 - (estimated + state) / raw_cache)),
            concurrency_estimates=a["concurrent_shared_state_estimates"], break_even_estimates=a["break_even"],
            estimate_scope="ESTIMATE: complete sampled-container/raw density scaled to full cache; full decoder state charged once. No concurrent serving or GPU capacity result.")

    def chain(self, measurement, label, *, layers, tokens, channels, tile, full):
        traces = measurement["record_diagnostics"]
        starts = measurement["selected_starts"]
        expected = {(layer, start, min(tile, tokens - start)) for start in starts for layer in range(layers)}
        observed = {(row["layer"], row["start"], row["tokens"]) for row in traces}
        raw = sum(row["raw_bytes"] for row in traces)
        expected_raw = sum(4 * channels * count for _, _, count in expected)
        pair_count = layers * ((tokens + tile - 1) // tile)
        self.check(label + ": complete selected dependency chains and page checks",
            len(traces) == len(expected) == measurement["records_measured"] and observed == expected
            and raw == expected_raw == measurement["raw_payload_bytes"]
            and measurement["records_total"] == pair_count
            and all(row["roundtrip_exact"] is True and row["dependency_depth"] == row["layer"]
                    and sum(row["component_bytes"].values()) == row["serialized_bytes"] for row in traces)
            and all(measurement.get(key) is True for key in ("exact_verified", "roundtrip_exact", "container_roundtrip_verified")))
        self.check(label + ": actual container byte ledger",
            sum(measurement["component_bytes"].values()) == measurement["serialized_bytes"]
            and sum(row["serialized_bytes"] for row in traces) == measurement["actual_record_payload_bytes"])
        self.check(label + ": declared coverage",
            (measurement["coverage"] == "full") == (len(traces) == pair_count)
            and measurement["size_is_estimated"] == (len(traces) != pair_count)
            and (not full or set(starts) == set(range(0, tokens, tile))))
        independent = measurement.get("independent_packet_decode")
        if independent is not None:
            decoded = independent["record_diagnostics"]
            self.check(label + ": independent packet decode records",
                independent["records_decoded"] == len(traces) == len(decoded)
                and {(row["layer"], row["start"], row["tokens"]) for row in decoded} == expected)
            if "source_arrays_used" in independent:
                self.check(label + ": decoder does not use original arrays", independent["source_arrays_used"] is False)
        return dict(encode=_stages(traces, "encode_timing"),
            immediate_verification_decode=_stages(traces, "verification_decode_timing"),
            independent_decode=_stages(independent["record_diagnostics"], "timing") if independent else None,
            ledger=_ledger(measurement))


def review(package, source_root, audit):
    manifest = _load(package / "package_manifest.json")
    listed = manifest["files"]
    bad = []
    for row in listed:
        path = _safe(package, row["file"])
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha(path.read_bytes()) != row["sha256"]:
            bad.append(row["file"])
    audit.check("All package-manifest file identities", not bad, files_checked=len(listed), mismatches=bad)
    if bad:
        raise ValueError("Package integrity failed; refusing to interpret inconsistent reports")
    raw_manifest = manifest["uncompressed_results"]
    with gzip.open(_safe(package, raw_manifest["compressed_as"]), "rb") as stream:
        raw = stream.read(raw_manifest["bytes"] + 1)
        tail = stream.read(1)
    raw_results_sha256 = _sha(raw)
    audit.check("Gzip decompression, CRC, raw result size and SHA256",
                not tail and len(raw) == raw_manifest["bytes"] and _sha(raw) == raw_manifest["sha256"])
    if (package / "results.json").exists():
        audit.check("Locally unpacked results.json matches authoritative gzip", (package / "results.json").read_bytes() == raw)
    report = json.loads(raw, parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text)))
    audit.check("Benchmark completed", report["status"] == "completed")
    source = report["source_manifest"]
    source_file = _load(package / "source_manifest.json")
    audit.check("Source manifest report/file and canonical aggregate", source_file == source
                and _sha(_canonical(source["files"])) == source["sources_sha256"])
    mismatches = [name for name, digest in source["files"].items()
                  if not _safe(source_root, name).is_file() or _sha(_safe(source_root, name).read_bytes()) != digest]
    audit.check("All declared local benchmark source SHA256 identities", not mismatches,
                files_checked=len(source["files"]), mismatches=mismatches)

    # Parse only literal constants; never import/execute the copied launcher.
    tree = ast.parse((source_root / "qwen_kv_gpu_bench.py").read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("_PAYLOAD_B64", "_PAYLOAD_SHA256", "_MEMBER_COUNT")}
    payload = base64.b64decode(constants["_PAYLOAD_B64"], validate=True)
    audit.check("Portable launcher embedded payload SHA256", _sha(payload) == constants["_PAYLOAD_SHA256"] == source["embedded_payload_sha256"])
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        names = zipped.namelist()
        safe = (len(names) == len(set(names)) == constants["_MEMBER_COUNT"]
                and all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts for name in names)
                and sum(item.file_size for item in zipped.infolist()) <= 32 * 1024 * 1024)
        audit.check("Portable payload archive framing/CRC and safe names", safe and zipped.testzip() is None,
                    members_checked=len(names))
        inner = json.loads(zipped.read("_qwen_source_manifest.json"))["files"]
        embedded_bad = [name for name, descriptor in inner.items() if len(zipped.read(name)) != descriptor["bytes"]
                        or _sha(zipped.read(name)) != descriptor["sha256"]]
        embedded_bad += [name for name, digest in source["files"].items() if name not in inner or inner[name]["sha256"] != digest]
        audit.check("Portable source members match inner manifest and benchmark sources", not embedded_bad,
                    manifest_members_checked=len(inner), mismatches=embedded_bad)

    freeze = report["freeze"]
    audit.check("Frozen manifest persisted report/file identity", _load(package / "frozen_manifest.json") == freeze)
    audit.check("Frozen source, base model, neural models and anchor state identities",
        freeze["source_sha256"] == source["sources_sha256"] and freeze["model_identity"] == report["model_identity"]
        and freeze["models"] == report["models"] and freeze["anchor_state"] == report["anchor_state"])
    training = report["training_captures"]
    audit.check("Explicit TRAIN captures and unchanged frozen KV identities",
        all(row["split"] == "train" for row in training)
        and freeze["training_kv_sha256"] == [row["kv_words_sha256"] for row in training])
    prompts = report["input_preflight"]["prompts"]
    audit.check("Successful tokenizer preflight and TRAIN prompt identities",
        report["input_preflight"]["status"] == "passed"
        and all(any(prompt["split"] == "train" and prompt["name"] == row["name"]
                    and prompt["target_tokens"] == row["tokens"] and prompt["input_ids_sha256"] == row["input_ids_sha256"]
                    for prompt in prompts) for row in training))
    anchor = report["anchor_state"]
    anchor_wire = _safe(package, anchor["file"]).read_bytes()
    audit.check("Actual serialized anchor SHA256 and charged state bytes",
        _sha(anchor_wire) == anchor["state_id"] and len(anchor_wire) == anchor["serialized_bytes"]
        and anchor["charged_bytes"] == max(anchor["serialized_bytes"], anchor["physical_tensor_bytes"]))
    geometry = training[0]["geometry"]
    layers, channels = geometry["layers"], geometry["columns"]
    models = []
    for name, model in report["models"].items():
        meta, files = model["metadata"], model["files"]
        network_updates = sum(row["optimizer_updates"] for row in meta["training"]["networks"].values())
        per_layer_updates = sum(sum(row["updates_by_layer"].values()) for row in meta["training"]["networks"].values())
        audit.check(name + ": declared complete artifact bytes and TRAIN provenance",
            sum(row["bytes"] for row in files) == model["state_bytes"] and len({row["file"] for row in files}) == len(files)
            and files[0]["sha256"] == model["model_id"] and meta["training_only"] is True
            and meta["model_identity"] == report["model_identity"]
            and len(meta["training_captures"]) == len(training)
            and meta["training_hashes"] == [row["sha256"] for row in meta["training_captures"]]
            and all(row["split"] == "train" and row["metadata"] == original for row, original in zip(meta["training_captures"], training))
            and meta["training"]["total_optimizer_updates"] == network_updates == per_layer_updates
            and meta["compute"]["trainable_parameter_count"] == model["parameter_count"])
        present = [row for row in files if _safe(package, row["file"]).is_file()]
        if present:
            audit.check(name + ": independently available model artifact identities",
                all(_safe(package, row["file"]).stat().st_size == row["bytes"]
                    and _sha(_safe(package, row["file"]).read_bytes()) == row["sha256"] for row in present))
        models.append(dict(name=name, declared_artifact_files=len(files), declared_serialized_bytes=model["state_bytes"],
            trainable_parameters=model["parameter_count"], complete_float32_state_bytes=4 * meta["compute"]["complete_float32_state_values"],
            total_optimizer_updates=network_updates, independently_available_artifact_files=len(present),
            independently_verified_complete_neural_weights=len(present) == len(files),
            scope="TRAIN/provenance/declared sums verified from reports. Omitted neural artifact bytes cannot be independently rehashed or replayed."))

    tile = report["config"]["tile_tokens"]
    dev_checks, test_rows = [], []
    logged_pairs = generation_steps = independent_pairs = 0
    for check in report["generation_checks"]:
        name, result, measurement = check["name"], check["result"], check["chain_measurement"]
        prefix = check["complete_prefix_tokens"]
        audit.check(name + ": full DEV continuation and finite exact generation",
            check["status"] == "ok" and prefix == report["config"]["dev_tokens"]
            and all(result[key] is True for key in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"))
            and result["max_abs_logit_difference"] == 0 and result["steps"] == len(result["per_step"]) == report["config"]["generation_steps"]
            and all(step["logits_finite"] is True and step["logits_bitwise_equal"] is True
                    and step["tokens_equal"] is True and step["appended_full_cache_bitwise_equal"] is True
                    and step["reference_next_token"] == step["restored_next_token"] for step in result["per_step"]))
        stages = audit.chain(measurement, "DEV " + name, layers=layers, tokens=prefix, channels=channels, tile=tile, full=True)
        audit.check(name + ": DEV actual container byte count", check["actual_container_bytes"] == measurement["serialized_bytes"])
        decoder_state = measurement["decoder_state_info"]
        declared_model = report["models"][name]
        audit.check(name + ": DEV decoder uses the frozen neural and complete anchor state",
            decoder_state["neural_model_id"] == declared_model["model_id"]
            and decoder_state["neural_serialized_bytes"] == declared_model["state_bytes"]
            and decoder_state["neural_resident_bytes"] == 4 * declared_model["metadata"]["compute"]["complete_float32_state_values"]
            and decoder_state["anchor_state_id"] == anchor["state_id"]
            and decoder_state["anchor_serialized_bytes"] == anchor["serialized_bytes"]
            and decoder_state["anchor_resident_bytes"] == anchor["physical_tensor_bytes"])
        raw = 4 * layers * prefix * channels
        dev_checks.append(dict(mode=name, complete_prefix_tokens=prefix, complete_raw_cache_bytes=raw,
            actual_container_bytes=check["actual_container_bytes"], actual_payload_reduction_pct=100 * (1 - check["actual_container_bytes"] / raw),
            generation_steps=result["steps"],
            all_logits_and_tokens_and_appended_KV_exact=all(result[key] is True for key in
                ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"))
                and all(step["logits_finite"] is True for step in result["per_step"]),
            max_abs_logit_difference=result["max_abs_logit_difference"],
            records=measurement["records_measured"], coverage=measurement["coverage"], stages=stages,
            continuation_cpu_decode_and_rebuild_seconds=sum(step["cpu_decode_and_rebuild_seconds"] for step in result["per_step"]),
            scope="Measured complete DEV correctness check with fully restored raw model KV; not encoded-resident attention or a TEST compression score."))
        logged_pairs += len(measurement["record_diagnostics"])
        generation_steps += len(result["per_step"])
    for case in report["cases"]:
        tokens = case["tokens"]
        raw_cache = case["capture"]["raw_payload_bytes"]
        audit.check(case["name"] + ": actual raw cache geometry and held-out prompt identity",
            raw_cache == 4 * layers * channels * tokens and case["split"] == "test"
            and any(row["split"] == "test" and row["name"] == case["name"] and row["target_tokens"] == tokens
                    and row["input_ids_sha256"] == case["capture"]["input_ids_sha256"] for row in prompts))
        for row in case["results"]:
            label = case["name"] + " " + row["name"]
            audit.check(label + ": method completed", row["status"] == "ok")
            summary = audit.accounting(row, raw_cache, label)
            summary.update(case=case["name"], full_context_tokens=tokens)
            measurement = row["measurement"]
            if row["name"] == "frozen_exponent_control":
                audit.check(label + ": frozen control actual packet/envelope bytes",
                    measurement["container_roundtrip_verified"] is True
                    and measurement["packet_bytes"] == measurement["inner_packet_bytes"] + measurement["indexed_envelope_bytes"]
                    and measurement["raw_sample_bytes"] == row["accounting"]["sample"]["raw_bytes"]
                    and row["accounting"]["state"]["neural"]["charged_bytes"] == 0
                    and row["accounting"]["state"]["anchor"]["charged_bytes"] == anchor["charged_bytes"])
                summary["frozen_control_wall_seconds"] = dict(encode=measurement["encode_wall_seconds"], decode=measurement["decode_wall_seconds"])
            else:
                stages = audit.chain(measurement, label, layers=layers, tokens=tokens, channels=channels, tile=tile, full=False)
                declared_model = report["models"][row["name"].removeprefix("neural_")]
                charged = row["accounting"]["state"]
                audit.check(label + ": complete frozen neural and anchor state charged",
                    charged["neural"]["serialized_bytes"] == declared_model["state_bytes"]
                    and charged["neural"]["resident_bytes"] == 4 * declared_model["metadata"]["compute"]["complete_float32_state_values"]
                    and charged["anchor"]["serialized_bytes"] == anchor["serialized_bytes"]
                    and charged["anchor"]["resident_bytes"] == anchor["physical_tensor_bytes"]
                    and measurement["decoder_state_info"]["neural_model_id"] == declared_model["model_id"]
                    and measurement["decoder_state_info"]["anchor_state_id"] == anchor["state_id"])
                audit.check(label + ": measurement/accounting byte identities",
                    measurement["serialized_bytes"] == measurement["actual_container_bytes"] == row["accounting"]["sample"]["packet_bytes"]
                    and measurement["raw_payload_bytes"] == measurement["raw_sample_bytes"] == row["accounting"]["sample"]["raw_bytes"]
                    and measurement["decoder_state_bytes"] == row["accounting"]["state"]["total_charged_bytes"])
                summary.update(selected_tile_starts=measurement["selected_starts"], selected_tokens_per_layer=sum(min(tile, tokens - start) for start in measurement["selected_starts"]),
                    sampled_fraction_of_full_raw_cache=row["accounting"]["sample"]["raw_bytes"] / raw_cache,
                    records_measured=measurement["records_measured"], records_total=measurement["records_total"], stages=stages,
                    independent_packet_decode_wall_seconds=measurement["independent_packet_decode"]["wall_seconds"],
                    encode_verify_and_independent_decode_wall_seconds=measurement["encode_verify_and_independent_decode_wall_seconds"])
                logged_pairs += len(measurement["record_diagnostics"])
                independent_pairs += measurement["independent_packet_decode"]["records_decoded"]
            test_rows.append(summary)
    audit.check("Requested model modes have DEV checks and every TEST comparison",
        {row["name"] for row in report["generation_checks"]} == set(report["config"]["modes"])
        and all({row["name"] for row in case["results"]} == {"frozen_exponent_control"} | {"neural_" + name for name in report["config"]["modes"]} for case in report["cases"]))

    return dict(format="neural_results_independent_review_v1", source_package=str(package),
        benchmark_status=report["status"], model_identity=report["model_identity"], environment=report["environment"], config=report["config"],
        identities=dict(sources_sha256=source["sources_sha256"], embedded_payload_sha256=source["embedded_payload_sha256"],
            uncompressed_results_sha256=raw_results_sha256, package_manifest_files=len(listed), source_files=len(source["files"])),
        counts=dict(test_cases=len(report["cases"]), test_rows=len(test_rows), dev_checks=len(dev_checks),
            neural_logged_roundtrip_pair_checks=logged_pairs, test_independent_decode_pair_checks=independent_pairs,
            generation_steps=generation_steps),
        models=models, actual_anchor_state_bytes=len(anchor_wire), training_budget=report["training_budget"],
        training_wall_seconds=report["training_wall_seconds"], elapsed_wall_seconds=report["elapsed_wall_seconds"],
        test_rows=test_rows, dev_checks=dev_checks,
        limitations=[
            "Source, package and report consistency checks do not independently reproduce omitted raw KV or neural weights.",
            "The reported TEST selection may contain only the first tile of a common system/chat prefix; it need not represent held-out document body or later context positions.",
            "Full-cache and concurrent shared-state reductions are arithmetic estimates, not full serialized-cache, memory-capacity or concurrent-serving measurements.",
            "KV containers are serialized and reopened in memory by the benchmark; no durable packet files or cross-process replay artifacts are exported in the normal reports package.",
            "Neural inference is device-specific FP32 with prediction hashes; cross-device/software restart determinism is not established.",
            "GPU neural prediction and CPU entropy stages do not constitute a GPU entropy codec, compact attention implementation or serving latency result.",
        ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    analysis = dict(format="neural_results_independent_review_v1", status="invalid")
    error = None
    try:
        analysis = review(args.package.resolve(), args.source_root.resolve(), audit)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"
        audit.check("Review completed without an interpretation error", False, error=error)
    passed = bool(audit.checks) and all(row["passed"] for row in audit.checks)
    validation = dict(format="neural_results_validation_v1", status="passed" if passed else "failed",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), checks=audit.checks,
        checks_total=len(audit.checks), checks_passed=sum(row["passed"] for row in audit.checks),
        error=error, scope="Read-only independent standard-library integrity/consistency audit; benchmark or launcher code was never executed.")
    analysis["validation_status"] = validation["status"]
    for filename, value in (("validation.json", validation), ("analysis.json", analysis)):
        (args.output / filename).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=validation["status"], checks_passed=validation["checks_passed"],
        checks_total=validation["checks_total"], output=str(args.output), error=error), sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
