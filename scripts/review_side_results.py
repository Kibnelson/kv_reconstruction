#!/usr/bin/env python3
"""Audit and summarize submitted prefill-side reports using only stdlib.

This checks supplied evidence and byte arithmetic. Model shards, original KV,
and compressed pages omitted by package_results.py cannot be replayed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path


MIB = 1 << 20
ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))), raw


def grouped(records, field):
    rows = defaultdict(lambda: dict(raw_bytes=0, packet_bytes=0, feature_bytes=0, ideal_bits=0., integer_cdf_bits=0.))
    for record in records:
        row = rows[str(record[field])]
        for key in row:
            row[key] += record[key]
    for row in rows.values():
        row.update(packet_reduction_pct=100 * (1 - row["packet_bytes"] / row["raw_bytes"]),
                   ideal_bits_per_value=2 * row["ideal_bits"] / row["raw_bytes"])
    return dict(rows)


def historical_comparison(package, previous):
    current, _ = read_json(package / "results.json.gz")
    prior, _ = read_json(previous / "results.json.gz")
    comparisons = []
    for case in current["cases"]:
        matches = [c for c in prior["cases"] if (c["name"], c["tokens"]) == (case["name"], case["tokens"])]
        if not matches:
            continue
        old = matches[0]
        if any(case["capture"][key] != old["capture"][key] for key in ("input_ids_sha256", "kv_words_sha256")):
            raise ValueError("Historical prompt or complete KV differs")
        shared = next(r for r in old["results"] if r["name"] == "neural_shared")
        addresses = {tuple(a) for a in case["matched_record_addresses"]}
        traces = [r for r in shared["measurement"]["record_diagnostics"] if (r["layer"], r["start"]) in addresses]
        if {(r["layer"], r["start"]) for r in traces} != addresses:
            raise ValueError("Historical subset misses a requested page")
        raw = sum(r["raw_bytes"] for r in traces)
        rows = [dict(method="historical_neural_shared", page_bytes=sum(r["serialized_bytes"] for r in traces),
                     charged_state_bytes=shared["accounting"]["state"]["total_charged_bytes"],
                     width=prior["config"]["width"], total_updates=prior["models"]["shared"]["metadata"]["training"]["total_optimizer_updates"])]
        for r in case["results"]:
            if not r["name"].startswith("side_"):
                continue
            m = r["measurement"]
            rows.append(dict(method=r["name"], page_bytes=sum(t["packet_bytes"] for t in m["records"]),
                             charged_state_bytes=r["accounting"]["state"]["total_charged_bytes"],
                             width=current["config"]["width"], total_updates=current["config"]["steps"]))
        for row in rows:
            row["page_reduction_pct"] = 100 * (1 - row["page_bytes"] / raw)
        comparisons.append(dict(case=case["name"], tokens=case["tokens"], raw_sample_bytes=raw, pages=len(traces),
            input_ids_sha256=case["capture"]["input_ids_sha256"], kv_words_sha256=case["capture"]["kv_words_sha256"],
            outer_container_and_index_excluded=True, scope="Existing logged per-page bytes on identical KV addresses; no new serialization or model execution. Width and training budgets differ.", methods=rows))
    return comparisons


def review(package, source_root):
    report, raw = read_json(package / "results.json.gz")
    manifest, _ = read_json(package / "package_manifest.json")
    checks = []

    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))

    check("completed report", report["status"] == "completed")
    for row in manifest["files"]:
        relative = Path(row["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe package filename")
        data = (package / relative).read_bytes()
        check("package bytes/hash: " + row["file"], len(data) == row["bytes"] and sha(data) == row["sha256"])
    expected = manifest["uncompressed_results"]
    check("gzip raw JSON bytes/hash", len(raw) == expected["bytes"] and sha(raw) == expected["sha256"])
    source, _ = read_json(package / "source_manifest.json")
    frozen, _ = read_json(package / "frozen_manifest.json")
    check("source manifest matches report", source == report["source_manifest"])
    check("frozen manifest matches report", frozen == report["freeze"])
    digest = sha(json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode())
    check("source aggregate identity", source["sources_sha256"] == digest == frozen["source_sha256"])
    for path, expected_hash in source["files"].items():
        check("source hash: " + path, sha((source_root / path).read_bytes()) == expected_hash)
    check("frozen model identity", frozen["model_identity"] == report["model_identity"])
    check("frozen models and budget", frozen["models"] == report["models"] and frozen["training_budget"] == report["training_budget"])
    prompts = report["input_preflight"]["prompts"]
    check("distinct full prompt identities", len({p["input_ids_sha256"] for p in prompts}) == len(prompts))
    check("TRAIN-only captures", all(x["split"] == "train" for x in report["training_captures"]))
    for row in report["generation_checks"]:
        result = row["result"]
        check("logged full DEV exactness: " + row["name"], row["status"] == "ok"
              and row["complete_prefix_tokens"] == report["config"]["dev_tokens"]
              and all(result.get(k) is True for k in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"))
              and len(result["per_step"]) == report["config"]["generation_steps"]
              and all(step["logits_finite"] is True for step in result["per_step"]))
    local = {name: m for name, m in report["models"].items() if name != "previous_layer_shared"}
    schedule = None
    for name, model in report["models"].items():
        size = sum(f["bytes"] for f in model["files"])
        check("declared complete state: " + name, size == model.get("serialized_state_bytes", model.get("state_bytes")))
        check("TRAIN-only model: " + name, model["metadata"]["training_only"] is True)
        if name in local:
            t = model["metadata"]["training"]
            signature = (t["updates_by_layer"], t["token_examples_by_layer"], t["batch_tokens_histogram"])
            schedule = signature if schedule is None else schedule
            check("matched local sampling: " + name, signature == schedule and t["total_optimizer_updates"] == report["config"]["steps"])
    cases = []
    for case in report["cases"]:
        rows, methods = [], {}
        geometry = case["capture"]["geometry"]
        inline = case["capture"]["inline_side_codes"]
        raw_cache = case["capture"]["raw_payload_bytes"]
        addresses = {tuple(x) for x in case["matched_record_addresses"]}
        check("matched address plan: " + case["name"], addresses == {(l, s) for l in range(geometry["layers"]) for s in case["selected_starts"]})
        check("logged actual unmodified producer: " + case["name"], case["capture"]["producer_audit"]["all_stage_targets_exact"]
              and case["capture"]["producer_audit"]["hooks_removed"] and inline["observed_inside_attention_forward"])
        sample_sizes = set()
        for result in case["results"]:
            name, m, a = result["name"], result["measurement"], result["accounting"]
            sample, state = a["sample"], a["state"]
            sample_sizes.add(sample["raw_bytes"])
            check("successful row: " + name, result["status"] == "ok")
            wire = m.get("packet_bytes", m.get("serialized_bytes"))
            check("sample ledger: " + name, sample["packet_bytes"] == wire and sample["raw_bytes"] == m["raw_sample_bytes"])
            charge = sum(max(state[k]["serialized_bytes"], state[k]["resident_bytes"]) for k in ("neural", "anchor"))
            check("state charge: " + name, charge == state["total_charged_bytes"])
            net = 100 * (1 - (Fraction(wire, sample["raw_bytes"]) * raw_cache + charge) / raw_cache)
            check("full-cache estimate: " + name, abs(float(net) - a["full_cache_estimate"]["net_reduction_pct"]) < 1e-9)
            decoded = m.get("independent_packet_decode", {})
            seconds = decoded.get("wall_seconds", m.get("decode_wall_seconds"))
            row = dict(method=name, packet_bytes=wire, raw_sample_bytes=sample["raw_bytes"],
                packet_reduction_pct=sample["payload_reduction_pct"], net_estimated_reduction_pct=float(net),
                charged_state_bytes=charge, state_MiB=charge / MIB, dependency_depth=m["maximum_dependency_depth"],
                decode_seconds=seconds, component_bytes=m.get("component_bytes"),
                sample_fraction_of_full_cache=sample["raw_bytes"] / raw_cache)
            if name.startswith("side_"):
                mode = name[5:]; traces = m["records"]; values = sample["raw_bytes"] // 2
                check("page byte ledger: " + name, sum(m["component_bytes"].values()) == wire
                      and sum(x["packet_bytes"] for x in traces) + m["component_bytes"]["container_framing_bytes"] == wire)
                check("logged exact independent restore: " + name, m["roundtrip_exact"] and m["container_roundtrip_verified"]
                      and decoded["source_arrays_used"] is False and decoded["maximum_dependency_depth"] == 0
                      and decoded["raw_prerequisite_pages"] == 0 and decoded["order_is_shuffled"])
                check("all selected pages logged: " + name, {(x["layer"], x["start"]) for x in traces} == addresses
                      and {(x["layer"], x["start"]) for x in decoded["records"]} == addresses)
                check("inline code reused and charged: " + name, all(x["feature_words_supplied"] for x in traces)
                      and inline["methods"][mode]["calls"] == len(traces)
                      and inline["methods"][mode]["feature_bytes"] == m["component_bytes"]["feature_bytes"])
                row.update(ideal_bits_per_value=m["ideal_bits"] / values,
                    integer_CDF_bits_per_value=m["integer_cdf_bits"] / values,
                    retained_code_bits_per_value=m["side_bits_per_KV_value"],
                    complete_packet_bits_per_value=8 * wire / values,
                    CPU_entropy_decode_seconds=decoded["entropy_decode_seconds"],
                    CPU_entropy_decode_pct=100 * decoded["entropy_decode_seconds"] / seconds,
                    prediction_seconds=decoded["prediction_seconds"],
                    per_layer=grouped(traces, "layer"), per_position=grouped(traces, "start"),
                    inline_code_seconds=inline["methods"][mode]["wall_seconds"])
            rows.append(row); methods[name] = row
        check("same raw sample across methods: " + case["name"], len(sample_sizes) == 1)
        baseline = methods["side_none"]
        for row in rows:
            row["net_delta_vs_none_pp"] = row["net_estimated_reduction_pct"] - baseline["net_estimated_reduction_pct"]
            if row["method"].startswith("side_"):
                row["packet_layers_beating_none"] = [int(l) for l, v in row["per_layer"].items() if v["packet_bytes"] < baseline["per_layer"][l]["packet_bytes"]]
                row["ideal_rate_layers_beating_none"] = [int(l) for l, v in row["per_layer"].items() if v["ideal_bits"] < baseline["per_layer"][l]["ideal_bits"]]
                row["loss_components_vs_none_pp"] = {
                    k: 100 * (row["component_bytes"][k] - baseline["component_bytes"][k]) / row["raw_sample_bytes"]
                    for k in row["component_bytes"]}
                row["loss_components_vs_none_pp"]["state"] = 100 * (row["charged_state_bytes"] - baseline["charged_state_bytes"]) / raw_cache
        cases.append(dict(name=case["name"], tokens=case["tokens"], selected_starts=case["selected_starts"], methods=rows))
    training = report["models"]["none"]["metadata"]["training"]
    analysis = dict(format="prefill_side_review_v1", results_json_sha256=sha(raw), source_sha256=source["sources_sha256"],
        config=report["config"], elapsed_seconds=report["elapsed_wall_seconds"], training_seconds=report["training_wall_seconds"],
        updates_per_layer_min=min(training["updates_by_layer"].values()), updates_per_layer_max=max(training["updates_by_layer"].values()),
        available_layer_token_rows=sum(x["tokens"] * x["geometry"]["layers"] for x in report["training_captures"]),
        actual_training_token_examples=training["total_token_examples"], cases=cases,
        limitations=["Logged GPU exactness is not replayed by this stdlib audit.",
            "Learned model shards, compressed pages and original KV are omitted from the shared package; their hashes/byte counts are declarations.",
            "Full-cache/concurrency ratios are sampled estimates; no serving capacity or speedup is measured."])
    validation = dict(format="prefill_side_report_validation_v1", checks=checks,
                      all_passed=all(x["passed"] for x in checks), passed=sum(x["passed"] for x in checks), total=len(checks))
    return analysis, validation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--previous", type=Path, help="Optional previous shared position-study package for identical-KV page comparison")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    analysis, validation = review(args.package, args.source_root)
    if args.previous:
        analysis["historical_comparison"] = historical_comparison(args.package, args.previous)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("analysis.json", analysis), ("validation.json", validation)):
        (args.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=validation["passed"], total=validation["total"], all_passed=validation["all_passed"])))
    return 0 if validation["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
