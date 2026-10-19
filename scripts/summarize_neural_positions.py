#!/usr/bin/env python3
"""Summarize declared neural position-study measurements using only stdlib.

This is analysis of submitted reports, not another model run or artifact audit.
Numerical shard comparisons use declared hashes when model files are omitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean

MIB = 1 << 20
PREVIOUS = Path(__file__).resolve().parents[1] / "benchmark_results/qwen_neural_probability_smoke_v2_a4500"


def load(package):
    path = package / "results.json" if package.is_dir() else package
    if not path.exists() and package.is_dir():
        path = package / "results.json.gz"
    raw = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    report = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if report["status"] != "completed":
        raise ValueError("A completed report is required: " + str(path))
    return report, dict(package=path.parent.name, results_json_bytes=len(raw),
                        results_json_sha256=hashlib.sha256(raw).hexdigest())


def stages(records, field):
    totals = Counter()
    for record in records:
        timing = record.get(field, {})
        for key in ("cpu_entropy_encode_seconds", "cpu_entropy_decode_seconds",
                    "anchor_packet_to_cpu_seconds", "anchor_restore_wall_seconds"):
            totals[key] += timing.get(key, 0.)
        # Anchor restoration already contains anchor_decode; never add it twice.
        totals["anchor_encode_wall_seconds"] += timing.get("anchor_encode", {}).get("wall_seconds", 0.)
        prediction = timing.get("prediction", {})
        for key in ("input_staging", "neural_prediction", "prediction_output_to_cpu"):
            totals[key + "_wall_seconds"] += prediction.get(key, {}).get("wall_seconds", 0.)
        for key in ("input_bytes", "output_bytes"):
            totals[key] += prediction.get(key, 0)
    totals["named_disjoint_wall_seconds"] = sum(v for k, v in totals.items() if k.endswith("seconds"))
    return dict(totals)


def ledger(records, components, wire_bytes, raw_bytes):
    conditional = [r for r in records if r["layer"] > 0]
    words = sum(r["raw_bytes"] // 2 for r in conditional)
    ideal = sum(r["ideal_continuous_bits"] for r in conditional)
    integer = sum(r["integer_cdf_bits"] for r in conditional)
    entropy = components.get("cpu_entropy_payload", 0)
    anchor_fields = components.get("anchor_native_fields", 0)
    headers = wire_bytes - entropy - anchor_fields
    if sum(components.values()) != wire_bytes:
        raise ValueError("Component bytes do not reconcile")
    return dict(raw_bytes=raw_bytes, wire_bytes=wire_bytes,
        reduction_pct=100 * (1 - wire_bytes / raw_bytes), conditional_words=words,
        ideal_interval_bits=ideal, integer_cdf_bits=integer,
        ideal_bits_per_conditional_word=ideal / words,
        integer_bits_per_conditional_word=integer / words,
        integer_minus_ideal_bits_per_conditional_word=(integer - ideal) / words,
        arithmetic_bits_per_conditional_word=8 * entropy / words,
        arithmetic_minus_integer_bits_per_conditional_word=(8 * entropy - integer) / words,
        neural_entropy_payload_bytes=entropy, anchor_native_fields_bytes=anchor_fields,
        non_payload_header_bytes=headers, header_bits_per_all_word=16 * headers / raw_bytes,
        header_pct_of_raw=100 * headers / raw_bytes, component_bytes=dict(components))


def summarize_case(case):
    control = next(r for r in case["results"] if r["name"] == "frozen_exponent_control")
    cm, ca = control["measurement"], control["accounting"]
    rows = []
    for row in case["results"]:
        if not row["name"].startswith("neural_"):
            continue
        m, a = row["measurement"], row["accounting"]
        traces = m["record_diagnostics"]
        decoded = m["independent_packet_decode"]
        by_position, decode_by_position = defaultdict(list), defaultdict(list)
        for record in traces:
            by_position[record["start"]].append(record)
        for record in decoded["record_diagnostics"]:
            decode_by_position[record["start"]].append(record)
        positions = []
        for start, records in sorted(by_position.items()):
            parts = Counter()
            for record in records:
                parts.update(record["component_bytes"])
            item = ledger(records, parts, sum(r["serialized_bytes"] for r in records),
                          sum(r["raw_bytes"] for r in records))
            item.update(start=start, tokens=records[0]["tokens"], records=len(records),
                outer_container_excluded=True, matched_control_position_bytes=None,
                encode_stages=stages(records, "encode_timing"),
                verification_decode_stages=stages(records, "verification_decode_timing"),
                independent_decode_stages=stages(decode_by_position[start], "timing"))
            positions.append(item)
        sample, full = a["sample"], a["full_cache_estimate"]
        payload, net = sample["payload_reduction_pct"], full["net_reduction_pct"]
        control_net = ca["full_cache_estimate"]["net_reduction_pct"]
        decode_stages = stages(decoded["record_diagnostics"], "timing")
        metrics = dict(payload_reduction_pct=payload, net_estimated_reduction_pct=net,
            control_payload_reduction_pct=ca["sample"]["payload_reduction_pct"],
            control_net_estimated_reduction_pct=control_net,
            net_minus_control_pp=net - control_net, state_penalty_pp=payload - net,
            payload_gap_to_40_pp=40 - payload, net_gap_to_40_pp=40 - net,
            decode_ms_per_raw_MiB=1000 * decoded["wall_seconds"] / (sample["raw_bytes"] / MIB),
            cpu_entropy_decode_pct=100 * decode_stages.get("cpu_entropy_decode_seconds", 0.) / decoded["wall_seconds"],
            control_decode_ms_per_raw_MiB=1000 * cm["decode_wall_seconds"] / (sample["raw_bytes"] / MIB))
        rows.append(dict(method=row["name"], metrics=metrics, positions=positions,
            ledger=ledger(traces, m["component_bytes"], sample["packet_bytes"], sample["raw_bytes"]),
            records_measured=m["records_measured"], records_total=m["records_total"],
            coverage_fraction=m["records_measured"] / m["records_total"],
            state=a["state"], full_cache_estimate=full,
            control_sample_packet_bytes=ca["sample"]["packet_bytes"],
            encode_and_verify_wall_seconds=m["one_pass_wall_seconds"],
            independent_decode_wall_seconds=decoded["wall_seconds"],
            independent_decode_stages=decode_stages,
            independent_decode_unassigned_wall_seconds=decoded["wall_seconds"] - decode_stages["named_disjoint_wall_seconds"],
            encode_stages=stages(traces, "encode_timing"),
            verification_decode_stages=stages(traces, "verification_decode_timing"),
            maximum_dependency_depth=m["maximum_dependency_depth"],
            peak_raw_dependency_context_bytes=m["peak_raw_dependency_context_bytes"],
            gpu_peak_delta_bytes=m["after_device_memory"]["peak_allocated_bytes"] - m["initial_device_memory"]["allocated_bytes"]))
    return dict(name=case["name"], tokens=case["tokens"], methods=rows)


def declared_state(current, previous):
    result = {}
    for name in sorted(current["models"].keys() & previous["models"].keys()):
        new, old = current["models"][name], previous["models"][name]
        def shards(model):
            return {Path(f["file"]).name: (f["bytes"], f["sha256"]) for f in model["files"]
                    if Path(f["file"]).suffix == ".npkv"}
        n, o = shards(new), shards(old)
        result[name] = dict(current_model_id=new["model_id"], previous_model_id=old["model_id"],
            current_numerical_shards=len(n), previous_numerical_shards=len(o),
            matching_shards=sorted(k for k in n.keys() & o.keys() if n[k] == o[k]),
            mismatched_shards=sorted(k for k in n.keys() & o.keys() if n[k] != o[k]),
            numerical_shard_names_equal=n.keys() == o.keys(), all_declared_numerical_shards_equal=n == o,
            train_word_hashes_equal=new["metadata"]["training_hashes"] == old["metadata"]["training_hashes"],
            model_config_equal=new["metadata"]["config"] == old["metadata"]["config"],
            training_logs_equal=new["metadata"]["training"] == old["metadata"]["training"],
            serialized_state_bytes_delta=new["state_bytes"] - old["state_bytes"],
            manifest_bytes_delta=sum(f["bytes"] for f in new["files"] if Path(f["file"]).suffix == ".json") - sum(f["bytes"] for f in old["files"] if Path(f["file"]).suffix == ".json"),
            scope="Declared .npkv SHA256/size comparisons only; JSON manifest excluded; omitted tensors not independently rehashed.")
    return result


def compare_dev(current, previous):
    comparisons = []
    old = {r["name"]: r for r in previous["generation_checks"]}
    for new in current["generation_checks"]:
        if new["name"] not in old:
            continue
        prior = old[new["name"]]
        def get(row):
            m = row["chain_measurement"]
            return ledger(m["record_diagnostics"], m["component_bytes"], m["serialized_bytes"], m["raw_payload_bytes"])
        n, o = get(new), get(prior)
        delta = {k: n[k] - o[k] for k in ("wire_bytes", "neural_entropy_payload_bytes",
                 "anchor_native_fields_bytes", "non_payload_header_bytes", "reduction_pct")}
        comparisons.append(dict(method=new["name"], current=n, previous=o, new_minus_previous=delta,
            prefix_tokens_equal=new["complete_prefix_tokens"] == prior["complete_prefix_tokens"],
            current_prefix_tokens=new["complete_prefix_tokens"],
            full_KV_word_hash_equal=new["full_cache_word_hash"] == prior["full_cache_word_hash"],
            full_KV_word_hash=new["full_cache_word_hash"],
            header_share_of_saved_bytes=(o["non_payload_header_bytes"] - n["non_payload_header_bytes"]) / (o["wire_bytes"] - n["wire_bytes"]) if o["wire_bytes"] != n["wire_bytes"] else None))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--previous-package", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path, required=True, help="JSON filename or destination directory")
    args = parser.parse_args()
    current, provenance = load(args.package)
    previous, old_provenance = load(args.previous_package)
    cases = [summarize_case(case) for case in current["cases"]]
    groups, position_groups = defaultdict(list), defaultdict(list)
    for case in cases:
        for row in case["methods"]:
            groups[(case["tokens"], row["method"])].append(row)
            for position in row["positions"]:
                position_groups[(case["tokens"], row["method"], position["start"])].append(position)
    means = [dict(tokens=tokens, method=method, cases=len(rows),
        mean_metrics={key: mean(r["metrics"][key] for r in rows) for key in rows[0]["metrics"]})
        for (tokens, method), rows in sorted(groups.items())]
    position_means = [dict(tokens=tokens, method=method, start=start, cases=len(rows),
        means={key: mean(r[key] for r in rows) for key in ("reduction_pct", "ideal_bits_per_conditional_word",
               "integer_bits_per_conditional_word", "arithmetic_bits_per_conditional_word")},
        scope="Equal-case means of neural pages excluding outer index; no per-position control rates recorded.")
        for (tokens, method, start), rows in sorted(position_groups.items())]
    report = dict(format="neural_position_analysis_v1", current=provenance, previous=old_provenance,
        config=current["config"], elapsed_wall_seconds=current["elapsed_wall_seconds"],
        training_wall_seconds=current["training_wall_seconds"], cases=cases, per_length_means=means,
        per_length_position_means=position_means,
        declared_numerical_state_comparison=declared_state(current, previous),
        matched_DEV_comparison=compare_dev(current, previous),
        source_manifests_equal=current["source_manifest"] == previous["source_manifest"],
        limitations=["Report-only analysis; original KV, omitted weights and GPU execution are not independently reproduced.",
            "TEST totals are actual sampled bytes; full-cache/state rates are estimates from those addresses.",
            "Per-position page wires exclude the outer index. Control per-position bytes were not recorded; no per-position gain versus control is inferred.",
            "Timings are one synchronized functional pass, not repeated performance samples. Encode includes immediate verification decode; independent decode is separate.",
            "Two dependency tiles exclude returned outputs, staged inputs, probability arrays, coder workspace and state. GPU peaks include state preparation and functional work.",
            "Layer/position samples and page sizes differ from earlier conditional studies; percentages and latency are not paired across those studies."])
    output = args.output if args.output.suffix.lower() == ".json" else args.output / "position_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
