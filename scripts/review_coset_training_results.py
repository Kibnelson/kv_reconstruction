#!/usr/bin/env python3
"""Audit the prespecified coverage-by-duration prefill-coset experiment.

Uses recorded evidence and delivered artifacts. No GPU/model execution is done.
Each child also passes the original exact-packet and full-state audit. Cross-arm
checks bind the fixed DEV probe, TRAIN/TEST identities, optimizer horizon,
initialization, and short/long training prefixes; they never require a gain.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

_spec = importlib.util.spec_from_file_location("coset_packet_audit", Path(__file__).with_name("review_syndrome_results.py"))
packet_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(packet_audit)
canonical, sha, safe_path, read_json = (getattr(packet_audit, k) for k in ("canonical", "sha", "safe_path", "read_json"))
FORMAT = "qwen_prefill_coset_training_study_v1"


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-8, abs_tol=1e-7)


def _report(directory):
    path = next((directory / s for s in ("results.json", "results.json.gz") if (directory / s).is_file()), None)
    if path is None:
        raise ValueError("Missing results.json or results.json.gz in " + str(directory))
    return read_json(path)



def _parameter_hash(data):
    """Independently hash the delivered canonical float32 parameter shard."""
    header = struct.Struct("<4sBIII")
    if len(data) < header.size:
        raise ValueError("Truncated parameter shard")
    magic, version, metadata_size, payload_size, checksum = header.unpack_from(data)
    body = data[header.size:]
    if magic != b"NPKV" or version != 1 or len(body) != metadata_size + payload_size or zlib.crc32(body) != checksum:
        raise ValueError("Invalid parameter shard header/CRC")
    meta = json.loads(body[:metadata_size])
    if canonical(meta) != body[:metadata_size] or meta.get("format") != "neural_parameter_shard_v1" or meta.get("group") != "syndrome_parameters":
        raise ValueError("Invalid parameter shard metadata")
    digest, offset, last = hashlib.sha256(), 0, ""
    for entry in meta["array_manifest"]:
        shape, name = entry["shape"], entry["name"]
        if (entry["dtype"] != "<f4" or not isinstance(name, str) or name <= last or not shape
                or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in shape)
                or entry["offset"] != offset or entry["nbytes"] != 4*math.prod(shape)
                or offset + entry["nbytes"] > payload_size):
            raise ValueError("Invalid parameter array bounds/order")
        payload = body[metadata_size + offset:metadata_size + offset + entry["nbytes"]]
        if any(not math.isfinite(value) for (value,) in struct.iter_unpack("<f", payload)):
            raise ValueError("Nonfinite parameter array")
        digest.update(canonical(dict(name=name, shape=shape, dtype="<f4")))
        digest.update(payload)
        offset += entry["nbytes"]; last = name
    if offset != payload_size:
        raise ValueError("Unreferenced parameter bytes")
    return digest.hexdigest()



def _package_checks(directory, raw, check):
    """Verify the optional root delivery manifest and all declared bytes."""
    path = directory / "package_manifest.json"
    if not path.is_file():
        return None
    manifest, _ = read_json(path)
    check("root package format", manifest.get("format") == "qwen_results_package_v1")
    files = manifest["files"]
    check("root package unique file declarations", len({row["file"] for row in files}) == len(files))
    for item in files:
        target = safe_path(directory, item["file"])
        data = target.read_bytes()
        check("root delivered file identity: " + item["file"], len(data) == item["bytes"] and sha(data) == item["sha256"])
    expected = manifest["uncompressed_results"]
    check("root complete gzip JSON identity", len(raw) == expected["bytes"] and sha(raw) == expected["sha256"]
          and expected["name"] == "results.json" and expected["compressed_as"] == "results.json.gz"
          and expected["gzip_roundtrip_verified"] is True)
    check("root reports delivered", {"results.json.gz", "results.csv", "summary.md", "source_manifest.json", "frozen_manifest.json"} <= {row["file"] for row in files})
    return manifest


def _training_package_checks(directory, arms, root_package, check):
    path = directory / "training_package_manifest.json"
    if root_package is None and not path.is_file():
        return
    check("complete training package manifest present", path.is_file() and root_package is not None)
    if not path.is_file() or root_package is None:
        return
    manifest, _ = read_json(path)
    check("training package format", manifest.get("format") == "qwen_coset_training_package_v1")
    root_bytes = (directory / "package_manifest.json").read_bytes()
    check("training package root manifest identity", manifest["root_package_manifest"] == dict(bytes=len(root_bytes), sha256=sha(root_bytes)))
    children = manifest["arms"]
    check("training package complete unique arm set", len(children) == len(arms)
          and {row["directory"] for row in children} == {row["directory"] for row in arms})
    for child in children:
        root = safe_path(directory, child["directory"])
        child_manifest, data = read_json(root / "package_manifest.json")
        check("training package child manifest identity: " + child["directory"],
              child["package_manifest"] == dict(bytes=len(data), sha256=sha(data)))
        check("training package child raw identity: " + child["directory"],
              child["uncompressed_results"] == child_manifest["uncompressed_results"])


def _metric_checks(row, label, check):
    """Check count arithmetic independently of the neural predictions."""
    n, errors, ones, predicted = (int(row[k]) for k in ("values", "bit_errors", "original_ones", "predicted_ones"))
    check(label + " bit counts", n >= 0 and 0 <= errors <= n and 0 <= ones <= n and 0 <= predicted <= n
          and abs(ones - predicted) <= errors <= min(ones + predicted, 2*n - ones - predicted)
          and (ones + predicted - errors) % 2 == 0)
    if "BCE_sum_bits" in row:
        check(label + " BCE arithmetic", math.isfinite(row["BCE_sum_bits"]) and row["BCE_sum_bits"] >= 0
              and _close(row["BCE_bits"], row["BCE_sum_bits"] / n if n else 0))
    if "Brier_sum" in row:
        check(label + " Brier arithmetic", 0 <= row["Brier_sum"] <= n + 1e-6
              and _close(row["Brier_mean"], row["Brier_sum"] / n if n else 0))
    if "errors_per_group" in row:
        histogram = row["errors_per_group"]
        groups, tails = int(row["groups"]), int(row["tail_bits"])
        group_errors = sum(i*int(v) for i, v in enumerate(histogram))
        check(label + " group histogram", len(histogram) == 32 and all(int(v) == v and v >= 0 for v in histogram)
              and sum(histogram) == groups and 31*groups + tails == n
              and group_errors <= errors <= group_errors + tails
              and row["recoverable_groups"] == sum(histogram[:2]))
    if "candidate_payload_bytes" in row:
        check(label + " diagnostic candidate byte ledger", row["candidate_payload_bytes"] == sum(row[k] for k in (
            "candidate_mode_bytes", "candidate_short_bytes", "candidate_raw_group_bytes", "candidate_tail_bytes"))
              and all(isinstance(row[k], int) and row[k] >= 0 for k in (
                  "candidate_mode_bytes", "candidate_short_bytes", "candidate_raw_group_bytes", "candidate_tail_bytes", "raw_plane_bytes")))
    calibration = row.get("calibration")
    if calibration:
        counts = calibration["counts"]
        probs, truth = calibration["probability_sums"], calibration["truth_sums"]
        check(label + " calibration counts", len(counts) == len(probs) == len(truth) == 10
              and sum(counts) == n and _close(sum(truth), ones)
              and all(int(c) == c and c >= 0 and 0 <= p <= c + 1e-6 and 0 <= t <= c
                      for c, p, t in zip(counts, probs, truth)))


def _diagnostic_checks(diag, mode, checkpoint, check):
    metrics = checkpoint["metrics"]
    label = mode + " update " + str(checkpoint["update"])
    check(label + " recognized fixed DEV metrics", metrics.get("format") == "prefill_coset_fixed_dev_diagnostics_v1")
    check(label + " fixed DEV probe", metrics["probe"]["split"] == "dev"
          and metrics["probe"]["sha256"] == diag["probe_manifest"]["sha256"])
    truth = {}
    for b, row in metrics["by_residue"].items():
        learned = row["learned"]
        _metric_checks(learned["omitted"], label + "/b" + b, check)
        by_plane = learned["by_bitplane"]
        check(label + "/b" + b + " omitted planes", set(map(int, by_plane)) == set(range(int(b), 7)))
        for plane, value in by_plane.items():
            _metric_checks(value, label + "/b" + b + "/bit" + plane, check)
        for key in ("values", "bit_errors", "original_ones", "predicted_ones"):
            check(label + "/b" + b + " plane sum " + key,
                  sum(value[key] for value in by_plane.values()) == learned["omitted"][key])
        for kind, value in learned.get("by_kind", {}).items():
            _metric_checks(value, label + "/b" + b + "/" + kind, check)
        if "by_kind" in learned:
            for key in ("values", "bit_errors", "original_ones", "predicted_ones"):
                check(label + "/b" + b + " KV sum " + key,
                      sum(value[key] for value in learned["by_kind"].values()) == learned["omitted"][key])
        check(label + "/b" + b + " required controls", set(row["controls"]) == {"constant_zero", "train_prior"})
        for control, block in row["controls"].items():
            value = block["omitted"]
            _metric_checks(value, label + "/b" + b + "/" + control, check)
            check(label + "/b" + b + "/" + control + " same truth",
                  value["values"] == learned["omitted"]["values"] and value["original_ones"] == learned["omitted"]["original_ones"])
            if control == "constant_zero":
                check(label + "/b" + b + " constant zero definition", value["predicted_ones"] == 0 and value["bit_errors"] == value["original_ones"])
        for kind, value in row.get("ablations", {}).items():
            check(label + "/b" + b + "/" + kind + " ablation population",
                  value["omitted_values"] == learned["omitted"]["values"]
                  and 0 <= value["prediction_changed_bits"] <= value["omitted_values"])
        truth[b] = dict(known_fields_sha256=row["known_fields_sha256"], controls=row["controls"])
    return dict(probe=metrics["probe"], prior_logits_sha256=metrics["prior_logits_sha256"], truth=truth)


def review(directory, source_root=None):
    directory = Path(directory).resolve()
    report, raw = _report(directory)
    checks, child_reviews, signatures, mode_records = [], {}, {}, {}
    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))
    root_package = _package_checks(directory, raw, check)
    check("recognized training study", report.get("format") == FORMAT)
    check("completed four-arm study", report.get("status") == "completed")
    plan, _ = read_json(directory / "frozen_manifest.json")
    source, _ = read_json(directory / "source_manifest.json")
    cfg = report["config"]
    short, long = int(cfg["short_steps"]), int(cfg["long_steps"])
    check("duration budgets ordered", 0 < short < long)
    check("frozen prespecified plan", plan.get("format") == "prefill_coset_training_plan_v1"
          and plan["config"] == cfg and plan["lr_horizon"] == long)
    check("source aggregate identity", sha(canonical(source["files"])) == source["sources_sha256"])
    if "plan" in report:
        check("reported plan matches frozen file", report["plan"] == plan)
    arms = report["arms"]
    _training_package_checks(directory, arms, root_package, check)
    expected = {(sampling, steps) for sampling in ("random", "balanced") for steps in (short, long)}
    check("complete unique factorial", len(arms) == 4 and {(row["sampling"], row["steps"]) for row in arms} == expected
          and len({row["name"] for row in arms}) == 4 and len({row["directory"] for row in arms}) == 4)
    check("arm plan unchanged", [{k: row[k] for k in ("name", "sampling", "steps")} for row in arms] == plan["arms"])
    for arm in arms:
        name = arm["name"]
        root = safe_path(directory, arm["directory"])
        check(name + " completed", arm["status"] == "completed")
        child_analysis, child_validation = packet_audit.review(root, source_root)
        child_reviews[name] = dict(all_passed=child_validation["all_passed"], passed=child_validation["passed"], total=child_validation["total"],
            results_json_sha256=child_analysis["results_json_sha256"])
        check(name + " exact packet and artifact audit", child_validation["all_passed"])
        child, child_raw = _report(root)
        check(name + " child raw report identity", sha(child_raw) == arm["report_sha256"] == child_analysis["results_json_sha256"])
        descriptor = child["training_diagnostics"]
        diag, diag_raw = read_json(safe_path(root, descriptor["file"]))
        check(name + " external diagnostics identity", sha(diag_raw) == descriptor["sha256"] == arm["diagnostics_sha256"] and len(diag_raw) == descriptor["bytes"])
        check(name + " diagnostics format", diag["format"] == "qwen_prefill_coset_training_diagnostics_v1")
        check(name + " child source identity", child["source_manifest"] == source)
        check(name + " matching prescribed optimizer budget", child["config"]["steps"] == arm["steps"])
        identities = dict(model_identity=child["model_identity"], training_hashes=[row["producer_inputs_sha256"] for row in child["training_captures"]],
            development_inputs=child["development"]["producer_inputs_sha256"], probe_manifest=diag["probe_manifest"],
            diagnostic_capture={k:diag["capture"][k] for k in ("producer_inputs_sha256", "input_ids_sha256", "kv_words_sha256")},
            test=[{k: case[k] for k in ("name", "tokens", "selected_starts", "matched_record_addresses")} | dict(
                input_ids_sha256=case["capture"]["input_ids_sha256"], kv_words_sha256=case["capture"]["kv_words_sha256"],
                producer_inputs_sha256=case["capture"]["producer_inputs_sha256"]) for case in child["cases"]],
            config={k: child["config"][k] for k in ("rank", "width", "batch_tokens", "channels_per_step", "seed", "train_tokens", "dev_tokens", "lengths", "cases", "tile_tokens", "chunk_tokens", "max_pairs", "modes", "residue_bits")})
        signatures[name] = identities
        if "measurements" in arm:
            derived = [dict(case=case["name"], tokens=case["tokens"], method=row["name"],
                packet_savings_pct=row["accounting"]["sample"]["payload_reduction_pct"],
                net_savings_pct=row["accounting"]["full_cache_estimate"]["net_reduction_pct"],
                state_bytes=row["accounting"]["state"]["total_charged_bytes"],
                short_groups=row["measurement"].get("coding_summary", {}).get("totals", {}).get("retained_short_groups", 0))
                for case in child["cases"] for row in case["results"] if row["status"] == "ok"]
            check(name + " wrapper measurements match actual child ledger", arm["measurements"] == derived)
        check(name + " delivered mode diagnostics", set(diag["modes"]) == set(child["models"]))
        check(name + " identical mode sampling schedule", len({m["metadata"]["training"]["sampling_sha256"] for m in child["models"].values()}) == 1)
        if "fresh_test_fixtures" in plan:
            check(name + " prespecified fresh TEST names", {case["name"] for case in child["cases"]} == {f["name"] for f in plan["fresh_test_fixtures"]})
        for mode, model in child["models"].items():
            label = name + "/" + mode
            train = model["metadata"]["training"]
            diagnostics = diag["modes"][mode]
            points = {int(row["update"]): row for row in diagnostics["checkpoints"]}
            check(label + " fixed diagnostic schedule", len(points) == len(diagnostics["checkpoints"])
                  and 0 in points and arm["steps"] in points and set(points) == set(diag["evaluation"]["updates"]))
            check(label + " sampler and LR horizon", train["channel_sampling"] == arm["sampling"] and train["learning_rate_horizon"] == long)
            check(label + " coverage identity", sha(canonical(diagnostics["coverage"])) == train["coverage_sha256"])
            cover = diagnostics["coverage"]
            update_rows, word_rows = cover["optimizer_updates_by_layer_channel"], cover["word_examples_by_layer_channel"]
            g = model["metadata"]["geometry"]
            check(label + " coverage dimensions", len(update_rows) == len(word_rows) == g["layers"]
                  and all(len(row) == 2*g["channels"] for row in update_rows + word_rows))
            check(label + " coverage counts nonnegative", all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for row in update_rows + word_rows for v in row))
            if arm["sampling"] == "balanced":
                check(label + " balanced per-layer channel exposure", all(max(row) - min(row) <= 1 for row in update_rows))
            check(label + " coverage total words", sum(sum(row) for row in word_rows) == train["total_word_examples"])
            check(label + " coverage total channel updates", sum(sum(row) for row in update_rows) == arm["steps"] * min(train["channels_per_step"], 2*g["channels"]))
            check(label + " initial/final numerical identity", points[0]["parameter_sha256"] == train["initial_parameter_sha256"]
                  and points[arm["steps"]]["parameter_sha256"] == train["final_parameter_sha256"])
            artifact = next((item for item in model["files"] if item["file"].endswith(".bin")), None)
            check(label + " final numerical state matches delivered shard", artifact is not None
                  and _parameter_hash(safe_path(root, artifact["file"]).read_bytes()) == train["final_parameter_sha256"])
            logged = {int(row["update"]): row for row in train["checkpoints"]}
            reference = None
            for update, point in sorted(points.items()):
                if update:
                    logged_point = logged.get(update)
                    check(label + " diagnostic binds frozen TRAIN checkpoint " + str(update), logged_point is not None
                          and point["parameter_sha256"] == logged_point["parameter_sha256"]
                          and point["sampling_prefix_sha256"] == logged_point["sampling_prefix_sha256"])
                fixed = _diagnostic_checks(diag, label, point, check)
                if reference is None:
                    reference = fixed
                else:
                    check(label + " fixed probe/truth/prior at " + str(update), fixed == reference)
            mode_records[(arm["sampling"], arm["steps"], mode)] = dict(initial=train["initial_parameter_sha256"], final=train["final_parameter_sha256"],
                sampling_sha256=train["sampling_sha256"], checkpoints=points, geometry=g, fixed_reference=reference, train_checkpoints=logged)
        del child, child_raw, diag, diag_raw
    if signatures:
        first_name = next(iter(signatures)); reference = signatures[first_name]
        for name, value in signatures.items():
            check("matched sources, probes, TEST addresses and architecture: " + name, value == reference)
        modes = reference["config"]["modes"]
        for mode in modes:
            initial = {row["initial"] for key, row in mode_records.items() if key[2] == mode}
            check(mode + " same initialization across all arms", len(initial) == 1)
            fixed_truth = [row["fixed_reference"] for key, row in mode_records.items() if key[2] == mode]
            check(mode + " same probe truth and frozen priors across arms", all(row == fixed_truth[0] for row in fixed_truth))
            for budget in (short, long):
                random = mode_records[("random", budget, mode)]["train_checkpoints"]
                balanced = mode_records[("balanced", budget, mode)]["train_checkpoints"]
                fields = ("update", "layer", "capture", "record", "offset", "start", "tokens", "residue_bits", "learning_rate")
                check(mode + "/" + str(budget) + " matching logged source/window/residue/LR draws", set(random) == set(balanced)
                      and all({k:random[u].get(k) for k in fields} == {k:balanced[u].get(k) for k in fields} for u in random))
            for sampling in ("random", "balanced"):
                a, b = mode_records[(sampling, short, mode)], mode_records[(sampling, long, mode)]
                point = b["checkpoints"].get(short)
                check(mode + "/" + sampling + " identical short numerical prefix", point is not None and point["parameter_sha256"] == a["final"])
                check(mode + "/" + sampling + " identical short sampling prefix", point is not None and point["sampling_prefix_sha256"] == a["sampling_sha256"])
    validation = dict(format="prefill_coset_training_validation_v1", checks=checks, all_passed=all(row["passed"] for row in checks), passed=sum(row["passed"] for row in checks), total=len(checks))
    analysis = dict(format="prefill_coset_training_review_v1", results_json_sha256=sha(raw), child_reviews=child_reviews,
        matched_arms=report["arms"], limitations=["Recorded evidence audit; no GPU rerun or omitted-packet replay.", "Fixed DEV diagnostic curves are not packet-rate measurements. Final packets and complete state remain separately charged.", "The audit does not require or establish savings, speed, novelty, convergence or serving-capacity gains."])
    return analysis, validation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
    print(json.dumps(dict(all_passed=validation["all_passed"], passed=validation["passed"], total=validation["total"], output=str(args.output))))
    return 0 if validation["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
