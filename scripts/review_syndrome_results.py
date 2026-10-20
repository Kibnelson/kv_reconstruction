#!/usr/bin/env python3
"""Audit prefill-coset run/package evidence with the Python standard library.

Checks recorded exactness, provenance, actual file hashes and byte arithmetic.
This does not execute the model or establish GPU speed, capacity or priority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import zlib


FORMAT = "qwen_prefill_coset_study_v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe artifact path: " + str(relative))
    result = root / path
    if result.is_symlink() or not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes input: " + str(relative))
    return result


def read_json(path):
    data = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        data = gzip.decompress(data)  # Entire stream: verifies CRC and length.
    value = json.loads(data, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    return value, data


def _close(a, b):
    return abs(float(a) - float(b)) <= 1e-8


def review(directory, source_root=None):
    directory = Path(directory)
    paths = [directory / "results.json", directory / "results.json.gz"]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise ValueError("Input needs results.json or results.json.gz")
    report, raw = read_json(path)
    checks = []
    verified_files = []
    omitted_packets = []

    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))

    def verify_file(item, *, optional=False):
        filename = item["file"]
        target = safe_path(directory, filename)
        if not target.is_file() and optional:
            omitted_packets.append(filename)
            return
        if not target.is_file():
            check("artifact present: " + filename, False)
            return
        data = target.read_bytes()
        check("artifact bytes/hash: " + filename,
              len(data) == item["bytes"] and sha(data) == item["sha256"])
        verified_files.append(filename)

    check("recognized report format", report.get("format") == FORMAT)
    check("completed report", report.get("status") == "completed")
    manifest_path = directory / "package_manifest.json"
    if manifest_path.is_file():
        manifest, _ = read_json(manifest_path)
        for item in manifest["files"]:
            verify_file(item)
        expected = manifest["uncompressed_results"]
        check("complete gzip JSON identity", len(raw) == expected["bytes"] and sha(raw) == expected["sha256"])

    source, _ = read_json(directory / "source_manifest.json")
    frozen, _ = read_json(directory / "frozen_manifest.json")
    check("source manifest matches report", source == report["source_manifest"])
    check("frozen manifest matches report", frozen == report["freeze"])
    check("source aggregate identity",
          sha(canonical(source["files"])) == source["sources_sha256"] == frozen["source_sha256"])
    if source_root is not None:
        for name, digest in source["files"].items():
            filename = safe_path(Path(source_root), name)
            check("reviewed source hash: " + name, filename.is_file() and sha(filename.read_bytes()) == digest)
    check("frozen target-model identity", frozen["model_identity"] == report["model_identity"])
    check("frozen model declarations", frozen["models"] == report["models"])
    modes = report["config"]["modes"]
    check("model/control set", set(modes) == set(report["models"]) == set(report["selection"]) and "none" in modes)
    captures = report["training_captures"]
    check("TRAIN-only captures", bool(captures) and all(row["split"] == "train" for row in captures))
    check("frozen TRAIN producer-input identities",
          frozen["training_hashes"] == [row["producer_inputs_sha256"] for row in captures]
          and frozen["training_producer_input_hashes"] == frozen["training_hashes"])
    check("frozen TRAIN capture identities",
          frozen["training_capture_hashes"] == [row["producer_records_sha256"] for row in captures])
    check("frozen DEV residue selection", frozen["residue_selection"] == report["selection"])
    provenance = frozen["selection_provenance"]
    check("selection uses DEV producer identity", provenance["split"] == "dev"
          and provenance["producer_records_sha256"] == report["development"]["producer_records_sha256"]
          and provenance["tokens"] == report["config"]["dev_tokens"])

    for mode, model in report["models"].items():
        files = model["files"]
        check("declared full fitted state: " + mode,
              sum(item["bytes"] for item in files) == model["serialized_state_bytes"])
        for item in files:
            verify_file(item)
        check("fitted model manifest identity: " + mode,
              any(item["sha256"] == model["model_id"] for item in files))
        metadata = model["metadata"]
        manifest_item = next((item for item in files if item["sha256"] == model["model_id"]), None)
        if manifest_item is not None:
            manifest_file = safe_path(directory, manifest_item["file"])
            if manifest_file.is_file():
                descriptor, descriptor_raw = read_json(manifest_file)
                check("actual model metadata matches report: " + mode,
                      descriptor.get("format") == "syndrome_model_artifact_v1"
                      and descriptor.get("metadata") == metadata
                      and canonical(descriptor) == descriptor_raw)
                shard = descriptor["shard"]
                shard_file = safe_path(directory, str(Path(manifest_item["file"]).parent / shard["file"]))
                shard_relative = str(shard_file.relative_to(directory))
                item = next((item for item in files if item["file"] == shard_relative), None)
                check("model shard descriptor matches delivered state: " + mode,
                      item is not None and item["bytes"] == shard["bytes"] and item["sha256"] == shard["sha256"])
                check("model manifest CRC and complete size: " + mode,
                      descriptor["metadata_and_shard_crc32"] == zlib.crc32(canonical(dict(metadata=descriptor["metadata"], shard=shard)))
                      and descriptor["manifest_bytes"] == len(descriptor_raw)
                      and descriptor["total_serialized_bytes"] == len(descriptor_raw) + shard["bytes"] == model["serialized_state_bytes"])
        check("TRAIN-only fitted model: " + mode, metadata.get("training_only") is True)
        if "model_identity" in metadata:
            check("fitted target identity: " + mode, metadata["model_identity"] == report["model_identity"])
        train = metadata.get("training", {})
        if "total_optimizer_updates" in train:
            check("declared optimizer budget: " + mode,
                  train["total_optimizer_updates"] == report["config"]["steps"])

    for family in ("field_state", "anchor_state"):
        state = report[family]
        verify_file(dict(file=state["file"], bytes=state["serialized_bytes"], sha256=state["sha256"]))
        check("frozen " + family, frozen[family] == state)

    for mode in modes:
        candidates = report["development_grid"][mode]
        best = min(candidates, key=lambda row: (row["packet_bytes"], row["residue_bits"]))
        check("DEV minimum complete packet selection: " + mode,
              best["residue_bits"] == report["selection"][mode]
              and {row["residue_bits"] for row in candidates} == set(report["config"]["residue_bits"]))
        for row in candidates:
            check("DEV exact packet candidate: " + mode + "/" + str(row["residue_bits"]),
                  row["roundtrip_exact"] is True and row["packet_bytes"] == row["measurement"]["packet_bytes"])
        for row in report.get("development_code_ablations", {}).get(mode, []):
            check("DEV code-use ablation exact: " + mode + "/" + row["kind"],
                  row["roundtrip_exact"] is True and row["packet_delta_vs_supplied_code"] == row["packet_bytes"] - best["packet_bytes"])

    generation = {row["name"]: row for row in report["generation_checks"]}
    check("all fitted modes have DEV continuation", set(generation) == {"coset_" + mode + "_syndrome" for mode in modes})
    for name, row in generation.items():
        result = row["result"]
        check("full DEV finite exact continuation: " + name, row["status"] == "ok"
              and row["complete_prefix_tokens"] == report["config"]["dev_tokens"]
              and all(result.get(key) is True for key in ("all_tokens_equal", "all_logits_bitwise_equal", "all_appended_KV_bitwise_equal"))
              and len(result["per_step"]) == report["config"]["generation_steps"]
              and all(step["logits_finite"] is True for step in result["per_step"]))
        verify_file(row["packet_artifact"], optional=True)

    summaries = []
    for case in report["cases"]:
        label = case["name"] + "/" + str(case["tokens"])
        capture = case["capture"]
        geometry = capture["geometry"]
        addresses = {tuple(pair) for pair in case["matched_record_addresses"]}
        check("matched address plan: " + label,
              addresses == {(layer, start) for layer in range(geometry["layers"]) for start in case["selected_starts"]})
        inline = capture["inline_side_codes"]
        audit = capture["producer_audit"]
        check("actual producer exact and hooks removed: " + label,
              audit["all_stage_targets_exact"] is True and audit["hooks_removed"] is True
              and inline["observed_inside_attention_forward"] is True)
        methods, sample_sizes = {}, set()
        for row in case["results"]:
            name, measured, accounting = row["name"], row["measurement"], row["accounting"]
            tag = label + "/" + name
            raw_bytes, packet_bytes = measured["raw_sample_bytes"], measured["packet_bytes"]
            sample_sizes.add(raw_bytes)
            sample, state, estimate = accounting["sample"], accounting["state"], accounting["full_cache_estimate"]
            check("successful TEST method: " + tag, row["status"] == "ok")
            check("measured sample ledger: " + tag,
                  raw_bytes == sample["raw_bytes"] and packet_bytes == sample["packet_bytes"])
            charged = sum(max(state[part]["serialized_bytes"], state[part]["resident_bytes"])
                          for part in ("neural", "anchor"))
            expected = Fraction(packet_bytes, raw_bytes) * capture["raw_payload_bytes"]
            net = 100 * (1 - (expected + charged) / capture["raw_payload_bytes"])
            check("complete state charge: " + tag, charged == state["total_charged_bytes"] == estimate["charged_state_bytes"])
            check("full-cache sampled estimate: " + tag,
                  _close(estimate["estimated_packet_bytes"], expected)
                  and _close(estimate["net_reduction_pct"], net)
                  and estimate["raw_bytes"] == capture["raw_payload_bytes"])
            check("sample reduction arithmetic: " + tag,
                  _close(sample["payload_reduction_pct"], 100 * (1 - Fraction(packet_bytes, raw_bytes))))
            summary = dict(method=name, packet_bytes=packet_bytes, raw_sample_bytes=raw_bytes,
                sample_reduction_pct=sample["payload_reduction_pct"], estimated_net_reduction_pct=float(net),
                charged_state_bytes=charged, sample_fraction_of_full_cache=raw_bytes / capture["raw_payload_bytes"])
            if name == "frozen_exponent_control":
                methods[name] = summary
                continue
            traces = measured["records"]
            parts = measured["component_bytes"]
            check("complete container component ledger: " + tag,
                  sum(parts.values()) == packet_bytes
                  and sum(trace["packet_bytes"] for trace in traces) + parts["container_framing_bytes"] == packet_bytes)
            check("all selected pages and raw bytes: " + tag,
                  {(trace["layer"], trace["start"]) for trace in traces} == addresses
                  and len(traces) == len(addresses) == measured["records_measured"]
                  and sum(trace["raw_bytes"] for trace in traces) == raw_bytes)
            check("record component totals: " + tag,
                  all(sum(trace["component_bytes"].values()) == trace["packet_bytes"] for trace in traces)
                  and all(parts[key] == sum(trace["component_bytes"][key] for trace in traces)
                          for key in parts if key != "container_framing_bytes"))
            decoded = measured["independent_packet_decode"]
            check("logged exact independent restore: " + tag,
                  measured["roundtrip_exact"] is True and measured["container_roundtrip_verified"] is True
                  and measured["all_repeats_exact"] is True
                  and measured["maximum_dependency_depth"] == decoded["maximum_dependency_depth"] == 0
                  and measured["raw_prerequisite_pages"] == decoded["raw_prerequisite_pages"] == 0
                  and decoded["source_arrays_used"] is False
                  and {(trace["layer"], trace["start"]) for trace in decoded["records"]} == addresses)
            check("repeat timing count: " + tag,
                  measured["repeats"] == len(measured["wall_seconds_samples"]) == report["config"]["repeats"])
            verify_file(measured["packet_artifact"], optional=True)
            if name.startswith("coset_"):
                mode, correction = name[len("coset_"):].rsplit("_", 1)
                common_width = mode.endswith("_common")
                if common_width:
                    mode = mode[:-len("_common")]
                bits = case["common_residue_bits"] if common_width else report["selection"][mode]
                check("TEST uses frozen residue width: " + tag, measured["residue_bits"] == bits)
                rank = 0 if mode == "none" else report["config"]["rank"]
                feature_bytes = sum(trace["tokens"] * rank * 2 for trace in traces)
                check("actual inline feature bytes retained: " + tag,
                      measured["inline_codes_used"] is True
                      and all(trace["feature_words_supplied"] is True for trace in traces)
                      and parts["feature_bytes"] == feature_bytes == inline["methods"][mode]["feature_bytes"]
                      and inline["methods"][mode]["calls"] == len(traces))
                for trace in traces:
                    ttag = tag + "/" + str(trace["layer"]) + ":" + str(trace["start"])
                    planes = trace["bitplanes"]
                    check("omitted plane coverage: " + ttag, [plane["bit"] for plane in planes] == list(range(bits, 7)))
                    for plane in planes:
                        ptag = ttag + "/bit" + str(plane["bit"])
                        hist = plane["errors_per_group"]
                        weighted = sum(i * value for i, value in enumerate(hist))
                        check("error group histogram: " + ptag,
                              len(hist) == 32 and all(isinstance(value, int) and value >= 0 for value in hist)
                              and sum(hist) == plane["groups"]
                              and hist[0] == plane["zero_error_groups"] and hist[1] == plane["one_error_groups"]
                              and sum(hist[2:]) == plane["multi_error_groups"]
                              and hist[0] + hist[1] == plane["recoverable_groups"]
                              and weighted <= plane["bit_errors"] <= weighted + plane["tail_bits"])
                        check("retained plane payload ledger: " + ptag,
                              sum(plane[key] for key in ("mode_bytes", "short_bytes", "raw_group_bytes", "tail_bytes")) == plane["retained_payload_bytes"]
                              and plane["candidate_payload_saving_bytes"] == plane["raw_plane_bytes"] - plane["candidate_payload_bytes"]
                              and plane["retained_short_groups"] + plane["retained_raw_groups"] == plane["groups"])
                        if plane["retained_mode"] == "raw":
                            check("raw plane is explicit: " + ptag,
                                  plane["retained_short_groups"] == 0 and plane["retained_payload_bytes"] == plane["raw_plane_bytes"])
                    check("retained correction plane total: " + ttag,
                          sum(plane["retained_payload_bytes"] for plane in planes) == trace["plane_bytes"]
                          and sum(plane["retained_short_groups"] for plane in planes) == trace["retained_short_groups"])
                summary.update(component_bytes=parts, feature_bytes=feature_bytes,
                    retained_short_groups=sum(trace["retained_short_groups"] for trace in traces),
                    candidate_groups=sum(trace["candidate_groups"] for trace in traces),
                    planes_using_short_code=sum(trace["planes_used_syndrome"] + trace["planes_used_positions"] for trace in traces),
                    per_bitplane_errors=dict(Counter({str(bit): sum(plane["bit_errors"] for trace in traces for plane in trace["bitplanes"] if plane["bit"] == bit)
                                                      for bit in range(bits, 7)})))
            else:
                check("field control has no feature/neural dependency: " + tag,
                      measured["residue_bits"] == 7 and parts["feature_bytes"] == 0
                      and charged == max(report["field_state"]["serialized_bytes"], report["field_state"]["resident_bytes"]))
            summary["median_decode_seconds"] = measured["median_wall_seconds"]
            methods[name] = summary
        check("same measured raw pages across methods: " + label, len(sample_sizes) == 1)
        for mode in modes:
            syndrome = methods["coset_" + mode + "_syndrome"]
            positions = methods["coset_" + mode + "_positions"]
            declared = case["syndrome_positions"][mode]
            check("matched syndrome/position payload size: " + label + "/" + mode,
                  syndrome["packet_bytes"] == positions["packet_bytes"] == declared["syndrome"] == declared["positions"]
                  and declared["packet_byte_difference"] == 0)
        if "common_residue_methods" in case:
            rows_by_name = {row["name"]: row for row in case["results"]}
            check("common-residue comparison shares frozen no-code width: " + label,
                  case["common_residue_bits"] == report["selection"]["none"]
                  and set(case["common_residue_methods"]) == set(modes)
                  and all(rows_by_name[name]["measurement"]["residue_bits"] == case["common_residue_bits"]
                          for name in case["common_residue_methods"].values()))
        summaries.append(dict(name=case["name"], tokens=case["tokens"], selected_starts=case["selected_starts"], methods=list(methods.values())))

    check("TEST cases present", bool(summaries))
    analysis = dict(format="prefill_coset_review_v1", results_json_sha256=sha(raw), source_sha256=source["sources_sha256"],
        config=report["config"], selected_residue_bits=report["selection"], cases=summaries,
        verified_artifact_files=sorted(set(verified_files)), omitted_packet_files=sorted(set(omitted_packets)),
        source_files_verified=source_root is not None,
        limitations=["Recorded model/packet exactness is not a new GPU rerun.",
            "Only delivered files are independently hashed; omitted packet streams cannot be decoded by this audit.",
            "Full-cache estimates extrapolate sampled pages. No serving speed, capacity or novelty is established.",
            "Checks of source files require --source-root pointing to the recorded source snapshot."])
    validation = dict(format="prefill_coset_validation_v1", checks=checks, all_passed=all(row["passed"] for row in checks),
        passed=sum(row["passed"] for row in checks), total=len(checks))
    return analysis, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original run or safely packaged results directory")
    parser.add_argument("--source-root", type=Path, help="Optional recorded source snapshot for file hash verification")
    parser.add_argument("--output", type=Path, required=True, help="Fresh or empty audit output directory")
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be fresh or empty")
    try:
        analysis, validation = review(args.input, args.source_root)
    except (OSError, EOFError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        parser.error(type(error).__name__ + ": " + str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("analysis.json", analysis), ("validation.json", validation)):
        (args.output / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest = dict(format="prefill_coset_review_manifest_v1", input_results_sha256=analysis["results_json_sha256"],
        files=[dict(file=name, bytes=(args.output / name).stat().st_size, sha256=sha((args.output / name).read_bytes()))
               for name in ("analysis.json", "validation.json")])
    (args.output / "review_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(all_passed=validation["all_passed"], passed=validation["passed"], total=validation["total"], output=str(args.output))))
    return 0 if validation["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
