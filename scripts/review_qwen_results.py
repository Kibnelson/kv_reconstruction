#!/usr/bin/env python3
"""Reproduce a byte-accounted review of completed Qwen KV reports.

Uses Python's standard library only. Inputs and optional source files are
read-only; only analysis.json and validation.json are created in a fresh output
directory. No benchmark is rerun, no codec is modified, and no projection is
reported as measured compression or serving performance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


REPORTS = ("results.json", "results.csv", "summary.md", "frozen_manifest.json",
           "source_manifest.json", "recipe_analysis.json")
EXPONENT = "native_exponent_group_bitpack"
MEDIAN = "native_prefill_exponent_median_time_tiered"
FIXED = "native_prefill_exponent_fixed127_time_tiered"
MIB = 1 << 20


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_info(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def outer(count):
    return 32 + 8 * (count + 1)


def primary_savings(m):
    value = m.get("estimated_full_savings_including_state_percent") if m.get("size_is_estimated") else m.get("savings_including_state_percent")
    if value is None:
        raise ValueError("missing primary state-inclusive savings")
    return value


def full_bytes(m):
    value = m.get("estimated_full_cache_serialized_bytes") if m.get("size_is_estimated") else m.get("serialized_bytes")
    return value + m["decoder_state_bytes"]


def normalized(sample):
    sample = sample or {}
    seconds, raw = sample.get("median_seconds"), sample.get("payload_bytes")
    return 1000 * seconds * MIB / raw if seconds is not None and raw else None


def summary(values):
    finite = [value for value in values if isinstance(value, (float, int)) and math.isfinite(value)]
    return dict(count=len(finite), mean=statistics.mean(finite), minimum=min(finite), maximum=max(finite)) if finite else dict(count=0, mean=None, minimum=None, maximum=None)


def records(row):
    m = row["measurement"]
    traces = m.get("record_diagnostics")
    if not isinstance(traces, list) or len(traces) != m["records_measured"]:
        raise ValueError(f"{row['name']}: missing or inconsistent record diagnostics")
    raw = sum(t["raw_bytes"] for t in traces)
    inner = sum(t["serialized_bytes"] for t in traces)
    if raw != m["raw_payload_bytes"] or inner != m["actual_record_payload_bytes"]:
        raise ValueError(f"{row['name']}: record size sums differ from measurement")
    if m["serialized_bytes"] != inner + outer(len(traces)):
        raise ValueError(f"{row['name']}: outer framing disagrees")
    for t in traces:
        if sum(t["component_bytes"].values()) != t["serialized_bytes"]:
            raise ValueError(f"{row['name']}: record component sum disagrees")
    identity = [tuple(t[k] for k in ("layer", "kind", "start", "tokens", "raw_bytes")) for t in traces]
    if len(set(identity)) != len(identity):
        raise ValueError(f"{row['name']}: duplicate record identities")
    return traces, identity


def method_row(row):
    m = row["measurement"]
    raw = m["raw_payload_bytes"]
    result = dict(name=row["name"], track=row["track"], execution=m.get("execution"),
        savings_including_state_percent=primary_savings(m), full_cache_bytes_including_state=full_bytes(m),
        full_cache_size_is_estimated=bool(m.get("size_is_estimated")),
        records_measured=m["records_measured"], records_total=m["records_total"],
        measured_raw_bytes=raw, full_raw_bytes=m["full_raw_payload_bytes"],
        measured_serialized_bytes=m["serialized_bytes"], decoder_state_bytes=m["decoder_state_bytes"],
        native_state_physical_bytes=m.get("native_state_physical_bytes"), decoder_state_info=m.get("decoder_state_info"),
        sample_decode_ms_per_raw_MiB=normalized(m.get("sample_decode")),
        packet_restore_ms_per_raw_MiB=normalized(m.get("sample_packet_transfer_then_decode")),
        component_bits_per_BF16_word={k: 16 * value / raw for k, value in m.get("component_bytes", {}).items()},
        decoder_state_bits_per_full_cache_BF16_word=16 * m["decoder_state_bytes"] / m["full_raw_payload_bytes"],
        prefill_recipe=m.get("prefill_recipe"), exact_verified=m.get("exact_verified"),
        roundtrip_exact=m.get("roundtrip_exact"))
    return result


def compare(rows, baseline, candidate, label):
    if baseline not in rows or candidate not in rows:
        return dict(baseline=baseline, candidate=candidate, status="unavailable", reason="candidate missing")
    a, b = rows[baseline]["measurement"], rows[candidate]["measurement"]
    at, ai = records(rows[baseline]); bt, bi = records(rows[candidate])
    if ai != bi or (a["records_total"], a["full_raw_payload_bytes"]) != (b["records_total"], b["full_raw_payload_bytes"]):
        return dict(baseline=baseline, candidate=candidate, status="unavailable", reason="record identity or full-cache geometry mismatch")
    components_a, components_b = a["component_bytes"], b["component_bytes"]
    by_kind = {}
    for kind in ("K", "V"):
        selected = [(x, y) for x, y in zip(at, bt) if x["kind"] == kind]
        raw = sum(x["raw_bytes"] for x, _ in selected)
        gain = sum(x["serialized_bytes"] - y["serialized_bytes"] for x, y in selected)
        by_kind[kind] = dict(raw_bytes=raw, record_payload_bytes_saved=gain,
            record_payload_savings_gain_percentage_points=100 * gain / raw if raw else None,
            scope="Matched packet payloads; cache-wide decoder state is not assigned to K/V.")
    return dict(baseline=baseline, candidate=candidate, status="ok", comparison=label,
        records_measured=len(at), measured_raw_bytes=a["raw_payload_bytes"],
        full_cache_size_is_estimated=bool(a.get("size_is_estimated")),
        savings_gain_percentage_points=primary_savings(b) - primary_savings(a),
        full_cache_bytes_saved=full_bytes(a) - full_bytes(b),
        decoder_state_increment_bytes=b["decoder_state_bytes"] - a["decoder_state_bytes"],
        measured_component_bytes_saved={key: components_a.get(key, 0) - components_b.get(key, 0) for key in sorted(set(components_a) | set(components_b))},
        per_kind=by_kind,
        scope="Positive gain means fewer candidate bytes. Same ordered records; full decoder state once per cache. Family/capacity comparisons change multiple mechanisms.")


def stateless_oracle(rows, embedded):
    eligible = {name: row for name, row in rows.items() if row["measurement"]["decoder_state_bytes"] == 0}
    if not eligible:
        return dict(status="unavailable", reason="no stateless candidates")
    traced = {name: records(row) for name, row in eligible.items()}
    first = next(iter(eligible.values()))["measurement"]
    identity = next(iter(traced.values()))[1]
    if any(value[1] != identity for value in traced.values()):
        raise ValueError("stateless oracle record identity mismatch")
    names = sorted(eligible)
    if len(names) > 256:
        raise ValueError("one mode byte cannot identify all candidates")
    choices = [min(names, key=lambda name: (traced[name][0][index]["serialized_bytes"], name)) for index in range(len(identity))]
    inner = sum(traced[name][0][index]["serialized_bytes"] for index, name in enumerate(choices))
    count = len(identity)
    total = inner + count + outer(count)
    estimated = (inner + count) * first["records_total"] / count + outer(first["records_total"])
    best = min(names, key=lambda name: (eligible[name]["measurement"]["serialized_bytes"], name))
    best_m = eligible[best]["measurement"]
    keys = dict(measured_total_bytes=total, full_cache_total_bytes=estimated,
                mode_counts=dict(Counter(choices)), best_fixed_stateless_candidate=best)
    matches = {key: embedded.get(key) == value for key, value in keys.items()}
    return dict(status="ok", **keys, records_measured=count,
        mode_id_bytes=count, measured_raw_bytes=first["raw_payload_bytes"],
        full_cache_savings_percent=100 * (1 - estimated / first["full_raw_payload_bytes"]),
        full_cache_size_is_estimated=count != first["records_total"],
        savings_gain_over_best_fixed_percentage_points=100 * (full_bytes(best_m) - estimated) / first["full_raw_payload_bytes"],
        embedded_oracle_checks=matches, embedded_oracle=embedded,
        scope="Recomputed size-only oracle: actual stateless packets + one mode byte per record + outer framing. All candidates were already encoded. No online selection, state union, encoder cost or latency benefit is measured.")


def storage_projections(row):
    """Hypothetical starts-only index; no payload-value assumptions are used."""
    m, spec = row["measurement"], row.get("spec") or {}
    segment = (spec.get("options") or {}).get("segment_size", 256)
    traces, _ = records(row)
    old = proposed = headers = anchors = 0
    geometries = Counter()
    for t in traces:
        if t["raw_bytes"] % 2 or not t["tokens"]:
            raise ValueError("invalid BF16 record geometry")
        words = t["raw_bytes"] // 2
        if words % t["tokens"]:
            raise ValueError("record token/channel geometry is inconsistent")
        channels = words // t["tokens"]
        segments = (words + segment - 1) // segment
        maximum_start = max(0, (segments - 1) * segment)
        width = 2 if maximum_start <= 65535 else 4
        # The original header already carries compact-stream terminal counts.
        # Store segment STARTS only; a 65,536 terminal count need not fit u16.
        expected_old = 2 * (segments + 1) * 4
        actual_old = sum(t["component_bytes"].get(key, 0) for key in ("secondary_offsets", "rare_offsets"))
        if actual_old != expected_old:
            raise ValueError("existing exponent offset bytes differ from geometry")
        old += actual_old
        proposed += 2 * segments * width
        headers += max(0, t["component_bytes"].get("header", 0) - 32)
        anchors += t["component_bytes"].get("exponent_anchor", 0)
        geometries[(t["tokens"], channels, segments, width, maximum_start)] += 1
    gain = old - proposed
    ratio = m["records_total"] / m["records_measured"]
    def projection(saved):
        projected = full_bytes(m) - saved * ratio
        return dict(measured_record_bytes_removed=saved,
            savings_gain_percentage_points=100 * saved * ratio / m["full_raw_payload_bytes"],
            projected_full_cache_savings_percent=100 * (1 - projected / m["full_raw_payload_bytes"]),
            projected_full_cache_bytes_including_state=projected)
    return dict(name=row["name"], segment_size=segment,
        current_measured_offset_bytes=old, projected_measured_start_offset_bytes=proposed,
        geometries=[dict(tokens=k[0], channels=k[1], segments=k[2], start_offset_bytes=k[3],
            maximum_possible_segment_start=k[4], records=v) for k, v in sorted(geometries.items())],
        starts_only_index=projection(gain), binary_32_byte_header_only=projection(headers),
        starts_only_plus_binary_header=projection(gain + headers),
        optimistic_anchor_removal_upper_bound_percentage_points=100 * anchors / m["raw_payload_bytes"],
        scope="PROJECTIONS, not measured new codecs or GPU results. Start-only prefix arrays omit terminal counts already in the existing header. Width is proven for all payload patterns using segment geometry; u32 is retained when u16 cannot bound every start. The independent 32-byte binary-header substitution is hypothetical and requires a complete validated descriptor design. Anchor-removal bound assumes zero correction growth and is not an achievable savings claim. Original tier payloads and attention behavior are unchanged.")


def source_audit(manifest, root):
    if root is None:
        return dict(measured=False, reason="--source-root not supplied; manifest internal digest still checked")
    root = root.resolve()
    checks = []
    for relative, expected in sorted(manifest["files"].items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            checks.append(dict(file=relative, status="invalid_path", expected_sha256=expected))
        elif not path.is_file():
            checks.append(dict(file=relative, status="missing", expected_sha256=expected))
        else:
            actual = file_info(path)
            checks.append(dict(file=relative, status="match" if actual["sha256"] == expected else "mismatch",
                               expected_sha256=expected, actual_sha256=actual["sha256"], bytes=actual["bytes"]))
    return dict(measured=True, source_root=str(root), files_checked=len(checks),
                status_counts=dict(Counter(c["status"] for c in checks)), files=checks)


def build(input_dir, source_root=None):
    source = Path(input_dir).resolve()
    inputs = {name: file_info(source / name) for name in REPORTS}
    report = json.loads((source / "results.json").read_text())
    frozen = json.loads((source / "frozen_manifest.json").read_text())
    manifests = json.loads((source / "source_manifest.json").read_text())
    external_analysis = json.loads((source / "recipe_analysis.json").read_text())
    tests = [case for case in report["cases"] if case.get("split") == "test"]
    calibration_bytes = (json.dumps(report["calibration"], indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    checks = dict(report_completed=report.get("status") == "completed",
        report_equals_external_source_manifest=report.get("source_manifest") == manifests,
        report_equals_external_frozen_manifest=report.get("freeze") == frozen,
        source_manifest_canonical_digest_valid=digest_json(manifests["files"]) == manifests["sources_sha256"],
        frozen_manifest_canonical_digest_valid=digest_json(frozen["manifest"]) == frozen["sha256"],
        frozen_source_matches_manifest=frozen["manifest"]["source_sha256"] == manifests["sources_sha256"],
        frozen_calibration_matches_reconstructed_metadata_file=hashlib.sha256(calibration_bytes).hexdigest() == frozen["manifest"]["calibration_sha256"],
        expected_seven_cases=len(report["cases"]) == 7,
        expected_six_test_cases=len(tests) == 6)
    case_validation = []
    for case in report["cases"]:
        row_counts = Counter((row.get("track"), row.get("status")) for row in case.get("results", []))
        success = [row for row in case.get("results", []) if row.get("status") == "ok"]
        native = [row for row in success if row.get("track") == "native_packet"]
        exact = all(row["measurement"].get("exact_verified") is True and row["measurement"].get("roundtrip_exact") is True for row in success)
        coverages = {}
        for track in sorted({row.get("track") for row in success}):
            measurements = [row["measurement"] for row in success if row["track"] == track]
            coverages[track] = dict(methods=len(measurements),
                records_checked=sum(m.get("records_checked", 0) for m in measurements),
                record_geometries=sorted({(m["records_measured"], m["records_total"]) for m in measurements}),
                all_sampled=all(m.get("size_is_estimated") is True for m in measurements))
        for row in native:
            records(row)
        same_analysis = case.get("recipe_analysis") == external_analysis.get(case["name"])
        checks["external_analysis_matches_" + case["name"]] = same_analysis
        case_validation.append(dict(name=case["name"], split=case["split"], tokens=case["tokens"],
            status=case.get("status", "not separately labelled"), exact_for_all_successful_rows=exact,
            row_status_counts=[dict(track=key[0], status=key[1], count=value) for key, value in sorted(row_counts.items())],
            coverage_by_track=coverages, native_record_diagnostics_validated=len(native)))
    generation = []
    for g in report.get("generation_checks", []):
        m = g.get("measurement") or {}
        steps = m.get("per_step") or []
        passed = (g.get("status") == "ok" and m.get("all_logits_bitwise_equal") is True
            and m.get("all_tokens_equal") is True and m.get("all_appended_KV_bitwise_equal") is True
            and len(steps) == m.get("steps") and all(step.get("logits_finite") is True for step in steps))
        generation.append(dict(name=g.get("name"), track=g.get("track"), status=g.get("status"),
            context_tokens=g.get("context_tokens"), steps=m.get("steps"), exact_and_finite=passed,
            max_abs_logit_difference=m.get("max_abs_logit_difference"), scope=m.get("scope")))
    audit = source_audit(manifests, Path(source_root) if source_root else None)
    if audit["measured"]:
        checks["all_local_source_files_match"] = audit["status_counts"] == {"match": len(manifests["files"])}
    checks["all_successful_rows_exact"] = all(c["exact_for_all_successful_rows"] for c in case_validation)
    checks["all_generation_checks_exact_and_finite"] = bool(generation) and all(g["exact_and_finite"] for g in generation)
    states = {}
    for case in report["cases"]:
        for row in case.get("results", []):
            if row.get("status") == "ok" and row.get("track") == "native_packet":
                m = row["measurement"]
                states.setdefault(row["name"], dict(charged_bytes=m["decoder_state_bytes"],
                    physical_tensor_bytes=m.get("native_state_physical_bytes"), state_id=m.get("state_id"),
                    decoder_state_info=m.get("decoder_state_info")))
    validation = dict(format="qwen_results_review_validation_v1", input_directory=str(source), inputs=inputs,
        checks=checks, all_checks_passed=all(checks.values()), source_files=audit,
        report_status=report.get("status"), model_identity=report.get("model_identity"),
        environment=report.get("environment"), config=report.get("config"),
        source_file_count=len(manifests["files"]), source_manifest_digest=manifests["sources_sha256"],
        frozen_manifest_digest=frozen["sha256"], calibration_metadata_reconstructed_sha256=hashlib.sha256(calibration_bytes).hexdigest(),
        calibration_model_binaries_verified=False, calibration_scope="Metadata serialization matches its frozen hash; missing model binary artifacts are not verified by this report-only review.",
        cases=case_validation, generation_checks=generation, state_contracts=states,
        source_warning="Source verification, when requested, compares local files to the recorded manifest. It does not certify the original runtime or model weight contents.")
    analyzed = []
    method_observations = defaultdict(list)
    for case in tests:
        successful = [row for row in case["results"] if row.get("status") == "ok" and row.get("track") in ("native_packet", "cpu_reference")]
        native = {row["name"]: row for row in successful if row["track"] == "native_packet"}
        methods = [method_row(row) for row in successful]
        pairs = [(EXPONENT, MEDIAN, "representation_family_comparison"), (FIXED, MEDIAN, "matched_predictor_prefill_metadata"),
                 ("native_prefill_affine_residual", "native_prefill_affine_time_g8_residual", "matched_layout"),
                 ("native_prefill_midpoint_residual", "native_prefill_midpoint_time_g8_residual", "matched_layout"),
                 ("native_prefill_midpoint_time_g8_residual", "native_prefill_midpoint_time_g64_residual", "matched_group_size")]
        for layout in ("channel", "time"):
            affine = "native_prefill_affine_residual" if layout == "channel" else "native_prefill_affine_time_g8_residual"
            for size in (16, 64):
                pairs.append((affine, f"native_prefill_affine_dictionary{size}_{layout}_g8_residual", "add_dictionary_ids_corrections_and_full_state"))
            pairs.append((f"native_prefill_affine_dictionary16_{layout}_g8_residual", f"native_prefill_affine_dictionary64_{layout}_g8_residual", "coupled_dictionary_capacity_comparison"))
        comparisons = [compare(native, a, b, label) for a, b, label in pairs]
        embedded = case["recipe_analysis"]["stateless_size_oracle"]
        oracle = stateless_oracle(native, embedded)
        checks["recomputed_oracle_matches_" + case["name"]] = all(oracle.get("embedded_oracle_checks", {}).values())
        analyzed.append(dict(name=case["name"], split="test", tokens=case["tokens"], methods=methods,
            matched_comparisons=comparisons, stateless_size_oracle=oracle,
            storage_projections=storage_projections(native[MEDIAN]),
            scope="Held-out TEST case only. Method sizes are estimates where flagged; timing uses bounded samples, not complete cache or attention execution."))
        for method in methods:
            method_observations[(method["track"], method["name"])].append(dict(case=case["name"], tokens=case["tokens"], **method))
    aggregates = []
    for (track, name), observations in sorted(method_observations.items()):
        aggregates.append(dict(name=name, track=track, test_case_count=len(observations),
            savings_percent=summary([x["savings_including_state_percent"] for x in observations]),
            sample_decode_ms_per_raw_MiB=summary([x["sample_decode_ms_per_raw_MiB"] for x in observations]),
            packet_restore_ms_per_raw_MiB=summary([x["packet_restore_ms_per_raw_MiB"] for x in observations]),
            by_context_length={str(length): dict(savings_percent=summary([x["savings_including_state_percent"] for x in observations if x["tokens"] == length]),
                sample_decode_ms_per_raw_MiB=summary([x["sample_decode_ms_per_raw_MiB"] for x in observations if x["tokens"] == length]),
                packet_restore_ms_per_raw_MiB=summary([x["packet_restore_ms_per_raw_MiB"] for x in observations if x["tokens"] == length])) for length in sorted({x["tokens"] for x in observations})}))
    validation["all_checks_passed"] = all(checks.values())
    analysis = dict(format="qwen_results_review_analysis_v1", inputs=inputs,
        test_cases=analyzed, method_summary=aggregates,
        aggregation_scope="Unweighted descriptive mean/min/max across the six TEST cases only; no DEV selection rows are pooled. Lengths of a fixture are correlated and synthetic fixtures are not representative workload evidence.",
        scope="Extraction of measured report data and separately labelled arithmetic projections. Full-cache estimates remain estimates. No cold SSD, compact attention, concurrency, admission, TTFT/TPOT or faster-restoration claim follows. CPU reference and native GPU wire formats remain distinct.")
    # Guard against a report being replaced while it is read.
    if inputs != {name: file_info(source / name) for name in REPORTS}:
        raise ValueError("input reports changed during review")
    return analysis, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory; never overwrites")
    parser.add_argument("--source-root", type=Path, help="Optional local checkout to compare with all recorded source hashes")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a fresh, nonexistent directory")
    analysis, validation = build(args.input_dir, args.source_root)
    texts = {"analysis.json": json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n",
             "validation.json": json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n"}
    args.output.mkdir(parents=True)
    for name, text in texts.items():
        (args.output / name).write_text(text, encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), test_cases=len(analysis["test_cases"]),
        methods=len(analysis["method_summary"]), all_checks_passed=validation["all_checks_passed"],
        source_files=validation["source_files"].get("status_counts")), sort_keys=True))
    return 0 if validation["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
