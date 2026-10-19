#!/usr/bin/env python3
"""Audit conditional-study packages, recorded exactness and complete byte costs.

The default audit uses only the standard library and never reruns a GPU kernel.
--verify-models additionally reloads canonical calibration artifacts using the
matching source tree and its NumPy/Torch dependencies. Historical runs require
their historical --source-root; changed source files are failures, not warnings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

NAMES = (
    "producer_frozen_exponent", "frontier_cost_final_raw8", "conditional_fast_pre_rope",
    "conditional_huffman_exponent", "conditional_huffman_exponent_sign", "conditional_tier_sign",
    "conditional_raw_pair", "conditional_raw_fullword", "conditional_mean_predictor",
    "conditional_anchor_r8", "conditional_anchor_r32", "conditional_latent_r8", "conditional_latent_r32",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def stream_info(stream):
    checksum, size = hashlib.sha256(), 0
    while chunk := stream.read(1 << 20):
        checksum.update(chunk)
        size += len(chunk)
    return dict(bytes=size, sha256=checksum.hexdigest())


def file_info(path):
    with path.open("rb") as stream:
        return stream_info(stream)


def inside(root, name):
    relative = Path(name)
    path = root / relative
    if relative.is_absolute() or ".." in relative.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Unsafe artifact/source path: " + str(name))
    return path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def outer(count):
    return 32 + 8 * (count + 1)


def build(package, source_root, verify_models=False):
    errors, counts = [], Counter()

    def check(condition, label):
        if not condition:
            errors.append(label)

    def close(actual, expected, label):
        check(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label)

    manifest = read_json(package / "package_manifest.json")
    files = manifest["files"]
    check(len({entry["file"] for entry in files}) == len(files), "Duplicate package manifest paths")
    for entry in files:
        path = inside(package, entry["file"])
        check(file_info(path) == {key: entry[key] for key in ("bytes", "sha256")}, "Package file identity: " + entry["file"])
    compressed = package / "results.json.gz"
    with gzip.open(compressed, "rb") as stream:
        raw_identity = stream_info(stream)  # EOF also verifies gzip CRC and length.
    expected_raw = manifest["uncompressed_results"]
    check(raw_identity == {key: expected_raw[key] for key in ("bytes", "sha256")}, "Decompressed report identity")
    raw = package / "results.json"
    if raw.exists():
        check(file_info(raw) == raw_identity, "Optional raw report equals gzip report")
    with gzip.open(compressed, "rt", encoding="utf-8") as stream:
        report = json.load(stream)
    source = read_json(package / "source_manifest.json")
    frozen = read_json(package / "frozen_manifest.json")
    check(report["source_manifest"] == source, "Report/source manifest agreement")
    check(digest(source["files"]) == source["sources_sha256"], "Source manifest canonical SHA256")
    source_mismatches = []
    for name, expected in source["files"].items():
        path = inside(source_root, name)
        actual = file_info(path)["sha256"] if path.is_file() else None
        if actual != expected:
            source_mismatches.append(dict(file=name, expected_sha256=expected, actual_sha256=actual))
    check(not source_mismatches, "Matching source tree; use the historical --source-root for historical runs")
    check(report["freeze"] == frozen, "Report/frozen manifest agreement")
    freeze = frozen["manifest"]
    check(digest(freeze) == frozen["sha256"], "Frozen manifest canonical SHA256")
    for key, report_key in (("model_identity", "model_identity"), ("calibration", "calibration"),
                            ("candidates", "candidates"), ("configuration", "config"),
                            ("experiment_protocol", "experiment_protocol")):
        check(freeze[key] == report[report_key], "Frozen " + key)
    check(freeze["source_sha256"] == source["sources_sha256"], "Frozen source identity")
    artifacts = {}
    for name, expected in freeze["decoder_artifacts"].items():
        actual = file_info(inside(package, name))
        artifacts[name] = actual
        check(actual == expected, "Frozen artifact identity: " + name)
    check(report["status"] == "completed" and not report.get("fatal_error") and not report.get("warnings"), "Completed without recorded errors/warnings")
    config = report["config"]
    check(config["producer_preset"] == "conditional" and not config["skip_inline"], "Conditional preset with inside-forward encoding enabled")
    names = [row["name"] for row in report["candidates"]]
    check(names == list(NAMES), "All 13 fixed candidates, in declared order")
    calibration = report["calibration"]
    train_hashes = calibration["frozen"]["metadata"]["training_hashes"]
    check(len(train_hashes) == 3 and len(set(train_hashes)) == 3, "Three distinct TRAIN captures")
    train_names, train_records = set(), set()
    for name, entry in calibration.items():
        meta = entry["metadata"]
        check(meta["training_only"] is True and meta["training_hashes"] == train_hashes, "TRAIN provenance: " + name)
        check(meta["model_identity"] == report["model_identity"], "Calibration model identity: " + name)
        sets = meta["training_record_sets"]
        check(len(sets) == 3 and all(row["split"] == "train" for row in sets), "Explicit TRAIN split: " + name)
        train_names.update(row["name"] for row in sets)
        train_records.update(len(row["records"]) for row in sets)
        check(file_info(inside(package, entry["file"])) == dict(bytes=entry["serialized_bytes"], sha256=entry["model_id"]), "Calibration file identity: " + name)
    check(len(calibration) == 12, "All 12 calibration models")
    check(train_names.isdisjoint({case["fixture"]["name"] for case in report["cases"]}), "TRAIN names disjoint from DEV/TEST")
    models_status = dict(requested=verify_models, canonical_models_verified=0,
                         scope="File SHA256/size and recorded provenance verified; binary CRC/schema not reloaded unless --verify-models is requested")
    if verify_models:
        sys.path.insert(0, str(source_root))
        from qwen_kv_bench import conditional_models, producer_models
        for name, entry in calibration.items():
            api = conditional_models if name.startswith("conditional_") else producer_models
            model = api.load(inside(package, entry["file"]), expected_model_identity=report["model_identity"],
                expected_training_hashes=train_hashes, expected_model_id=entry["model_id"], expected_kind=entry["metadata"]["kind"])
            check(model.metadata == entry["metadata"] and model.state_bytes == entry["serialized_bytes"], "Canonical model reload: " + name)
            models_status["canonical_models_verified"] += 1
        models_status["scope"] = "Canonical binary CRC/schema/SHA256, complete byte size, model identity and TRAIN hashes verified by matching-source model loaders"
    cases, timing_contracts = [], set()
    for case in report["cases"]:
        label = case["name"]
        audit = case["capture"]["producer_audit"]
        layers, channels, tokens = audit["layers"], audit["channels"], case["tokens"]
        tile = config["tile_tokens"]
        total = layers * ((tokens + tile - 1) // tile)
        measured = audit["selected_tile_pairs"]
        check(all(audit[key] is True for key in ("all_stage_targets_exact", "v_hook_equals_cache", "k_rope_replay_equals_cache", "exported_cache_equals_hook_cache", "rope_stable", "hooks_removed", "on_tile_enabled")), label + ": producer audit exactness")
        check(audit["on_tile_callback_count"] == audit["selected_tiles_ready_inside_attention"] == measured, label + ": inside-forward callback counts")
        check(audit["total_tokens"] == tokens and audit["total_tile_pairs"] == total and audit["channels"] == audit["kv_heads"] * audit["head_dim"], label + ": capture geometry")
        check(len(audit["selected_records"]) == measured and all(row["all_stage_targets_exact"] for row in audit["selected_records"]), label + ": per-record producer audit")
        inline = {row["name"]: row for row in case["capture"]["inline_recipe_metrics"]["methods"]}
        check(set(inline) == set(names), label + ": inline candidate coverage")
        check([row["name"] for row in case["results"]] == names, label + ": ordered candidate coverage")
        reference, same_records = None, True
        for row in case["results"]:
            key = label + "/" + row["name"]
            check(row["status"] == "ok", key + ": successful measurement")
            m = row["measurement"]
            traces, parts, info = m["record_diagnostics"], m["component_bytes"], m["decoder_state_info"]
            check(all(m[flag] is True for flag in ("roundtrip_exact", "exact_verified", "inline_record_payload_matches_replay")), key + ": recorded exactness")
            check(m["records_total"] == total and m["records_measured"] == m["records_checked"] == len(traces) == measured, key + ": record counts")
            addresses = [(t["layer"], t["start"], t["tokens"], t["raw_bytes"]) for t in traces]
            if reference is None:
                reference = addresses
            same_records = same_records and addresses == reference
            check(addresses == reference and len(set(addresses)) == measured, key + ": same unique ordered addresses")
            check(set(t["layer"] for t in traces) == set(range(layers)), key + ": all layers sampled")
            accumulated = Counter()
            for trace in traces:
                check(sum(trace["component_bytes"].values()) == trace["serialized_bytes"], key + ": inner component sum")
                check(trace["raw_bytes"] == trace["tokens"] * channels * 4 and 0 <= trace["start"] <= tokens - trace["tokens"], key + ": record geometry")
                accumulated.update(trace["component_bytes"])
            check(sum(t["serialized_bytes"] for t in traces) == m["actual_record_payload_bytes"], key + ": inner payload total")
            check(sum(t["raw_bytes"] for t in traces) == m["raw_payload_bytes"], key + ": raw byte total")
            inner_parts = dict(accumulated)
            accumulated.update(cache_header=32, pair_offset_table=8 * (measured + 1))
            check(dict(accumulated) == parts and sum(parts.values()) == m["serialized_bytes"], key + ": complete outer component sum")
            check(m["calculated_outer_framing_bytes"] == outer(measured), key + ": outer framing")
            full_raw = 4 * layers * tokens * channels
            expected = m["actual_record_payload_bytes"] * total / measured + outer(total)
            check(m["full_raw_payload_bytes"] == full_raw, key + ": full raw denominator")
            sampled = measured != total
            check(m["size_is_estimated"] == sampled and m["coverage"] == ("sampled" if sampled else "full"), key + ": coverage labels")
            if sampled:
                close(m["estimated_full_cache_serialized_bytes"], expected, key + ": full-cache estimate")
            else:
                close(m["serialized_bytes"], expected, key + ": complete measured cache")
            actual_state = file_info(inside(package, info["file"]))
            check(actual_state == dict(bytes=info["serialized_bytes"], sha256=info["sha256"]), key + ": complete serialized state")
            check(m["native_state_physical_bytes"] == info["physical_tensor_bytes"], key + ": resident state")
            check(m["decoder_state_bytes"] == info["charged_bytes"] == max(info["serialized_bytes"], info["physical_tensor_bytes"]), key + ": conservative state charge")
            close(m["savings_including_state_percent"], 100 * (1 - (m["serialized_bytes"] + m["decoder_state_bytes"]) / m["raw_payload_bytes"]), key + ": measured-sample savings")
            bits = 16 * (expected + m["decoder_state_bytes"]) / full_raw
            close(m["effective_bits_per_BF16_word_including_state"], bits, key + ": complete bits/value")
            if sampled:
                close(m["estimated_full_savings_including_state_percent"], 100 * (1 - bits / 16), key + ": estimated savings")
            check(m["target_40_percent"]["met"] == (bits <= 9.6), key + ": 40% target flag")
            observed = inline[row["name"]]
            check(observed["status"] == "ok" and observed["records_checked"] == measured and observed["state_id"] == m["state_id"] and observed["decoder_state_bytes"] == m["decoder_state_bytes"], key + ": inline state/count")
            address = lambda t: (t["layer"], t["start"], t["tokens"])
            check({address(t): (t["serialized_bytes"], t["component_bytes"]) for t in observed["records"]} == {address(t): (t["serialized_bytes"], t["component_bytes"]) for t in traces}, key + ": inline/replay actual fields")
            check(observed["serialized_record_bytes"] == m["actual_record_payload_bytes"] and observed["raw_bytes"] == m["raw_payload_bytes"] and observed["component_bytes"] == inner_parts, key + ": inline byte totals")
            sample = m["sample"]
            count = min(measured, config["sample_pairs"])
            ordinals = [0] if count == 1 else [int(i * ((measured - 1) / (count - 1))) for i in range(count)]
            sample_raw = sum(traces[i]["raw_bytes"] for i in ordinals)
            sample_wire = sum(traces[i]["serialized_bytes"] for i in ordinals)
            check(sample["records"] == count and sample["raw_bytes"] == sample_raw and sample["encoded_bytes"] == sample_wire, key + ": identical timing sample ordinals/bytes")
            for field in ("sample_decode", "sample_packet_transfer_then_decode"):
                timing = m[field]
                check(timing["repeats"] == len(timing["seconds"]) == config["repeats"] and all(value > 0 for value in timing["seconds"]), key + ": timing repeats")
                close(timing["median_seconds"], statistics.median(timing["seconds"]), key + ": timing median")
                check(timing["payload_bytes"] == sample_raw, key + ": timing raw denominator")
            timing_contracts.add((count, sample_raw, config["repeats"]))
            counts.update(method_rows=1, primary_pair_checks=m["records_checked"], inline_pair_checks=observed["records_checked"])
        cases.append(dict(name=label, split=case["split"], tokens=tokens, methods=len(case["results"]),
            records_measured=measured, records_total=total, coverage_fraction=measured / total,
            inline_callbacks=audit["on_tile_callback_count"], all_methods_same_ordered_records=same_records))
    check(len(cases) == 1 + config["cases"] * len(config["lengths"]), "Complete declared DEV/TEST case grid")
    generation = report["generation_checks"]
    check([row["name"] for row in generation] == names, "All 13 generation checks")
    for row in generation:
        key = row["name"]
        check(row["status"] == "ok", key + ": generation successful, not skipped")
        m = row["measurement"]
        check(row["context_tokens"] == config["dev_tokens"] and m["steps"] == len(m["per_step"]) == config["generation_steps"] and m["steps"] > 0, key + ": full DEV continuation geometry")
        check(all(m[field] is True for field in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal")), key + ": generation aggregate exactness")
        check(m["max_abs_logit_difference"] == 0, key + ": zero logit difference")
        for step in m["per_step"]:
            exact = all(step[field] is True for field in ("logits_finite", "logits_bitwise_equal", "tokens_equal", "appended_full_cache_bitwise_equal"))
            check(exact, key + ": finite/exact generation step")
            counts["finite_exact_generation_steps"] += int(exact)
    return dict(format="conditional_review_validation_v1", all_checks_passed=not errors,
        errors=errors, counts=dict(counts), source_files=len(source["files"]), source_root=str(source_root),
        source_mismatches=source_mismatches, source_sha256=source["sources_sha256"],
        embedded_payload_sha256=source.get("embedded_payload_sha256"), frozen_sha256=frozen["sha256"],
        package_files_verified=len(files), frozen_artifacts_verified=len(artifacts),
        canonical_model_verification=models_status, training_names=sorted(train_names),
        training_hashes=train_hashes, selected_TRAIN_record_counts=sorted(train_records),
        cases=cases, generation_methods=len(generation), timing_samples=[dict(records=n, raw_bytes=b, repeats=k) for n,b,k in sorted(timing_contracts)],
        compressed_report=file_info(compressed), uncompressed_report=dict(**raw_identity, gzip_crc_verified=True),
        environment=report["environment"], model_identity=report["model_identity"],
        scope="Recorded GPU exactness, artifact/source identities, matched samples and byte arithmetic audited; no GPU rerun or model-weight attestation. TEST sizes are sampled estimates, outer containers are calculated charges. Runtime changes prevent attributing cross-run differences solely to algorithms.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-models", action="store_true", help="Reload all canonical models; requires matching-source NumPy/Torch dependencies")
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be fresh or empty")
    try:
        validation = build(args.package.resolve(), args.source_root.resolve(), args.verify_models)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        validation = dict(format="conditional_review_validation_v1", all_checks_passed=False,
            error=f"{type(error).__name__}: {error}", scope="Audit could not complete; no success claimed")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(all_checks_passed=validation["all_checks_passed"], counts=validation.get("counts"),
                         errors=validation.get("errors", validation.get("error")), output=str(args.output))))
    return 0 if validation["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
