"""Matched, byte-accounted analyses of one benchmark case at a time.

The portfolio is a SIZE-ONLY oracle over already encoded candidate packets.
It does not implement a prefill selector, repackage payloads, or measure an
online selection/latency benefit. Development and test cases are never pooled.
Only the standard ``native_packet`` rows are used; observer measurements and
CPU reference formats have different execution/accounting contracts.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import json
import math


_HEADER_BYTES = 32
_OFFSET_BYTES = 8
_MODE_BYTES = 1
_MIB = 1 << 20


def _outer(count):
    return _HEADER_BYTES + _OFFSET_BYTES * (count + 1)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _saving(stored, raw):
    return 100.0 * (1.0 - stored / raw) if raw else None


def _per_mib(seconds, raw):
    return 1000.0 * seconds / (raw / _MIB) if _number(seconds) and raw else None


def _timing(measurement):
    sample = measurement.get("sample_decode") or {}
    transfer = measurement.get("sample_packet_transfer_then_decode") or {}
    raw = measurement.get("raw_payload_bytes", 0)
    return dict(
        encode_ms_per_MiB=_per_mib(measurement.get("encode_seconds_total"), raw),
        one_pass_decode_ms_per_MiB=_per_mib(measurement.get("decode_seconds_total"), raw),
        repeated_decode_ms_per_MiB=_per_mib(sample.get("median_seconds"), sample.get("payload_bytes", 0)),
        packet_transfer_then_decode_ms_per_MiB=_per_mib(transfer.get("median_seconds"), transfer.get("payload_bytes", 0)),
        repeated_decode_raw_bytes=sample.get("payload_bytes"),
        packet_transfer_raw_bytes=transfer.get("payload_bytes"),
        execution=measurement.get("execution"), device=measurement.get("device"),
        scope="Normalized by original raw BF16 bytes in each timing sample. Repeated timing may use a smaller sample than byte-accounting records. CPU tensor runs are functionality checks, not GPU timings.")


def _view(row):
    """Validate record identity/accounting before deriving matched conclusions."""
    m = row.get("measurement") or {}
    if m.get("roundtrip_exact") is not True or m.get("exact_verified") is not True:
        return None, "exact roundtrip verification was not reported"
    traces = m.get("record_diagnostics")
    if not isinstance(traces, list) or not traces:
        return None, "record_diagnostics missing or empty; aggregate sizes cannot establish matched record identity"
    if not _integer(m.get("records_measured"), 1) or len(traces) != m["records_measured"]:
        return None, "record_diagnostics count differs from records_measured"
    if not _integer(m.get("records_total"), len(traces)):
        return None, "invalid records_total"
    if not _integer(m.get("full_raw_payload_bytes"), 1):
        return None, "missing or invalid full_raw_payload_bytes"
    if not _integer(m.get("decoder_state_bytes")):
        return None, "missing or invalid decoder_state_bytes; statelessness cannot be assumed"
    identity = []
    raw = inner = 0
    components = Counter()
    by_kind = {}
    for trace in traces:
        if not isinstance(trace, dict):
            return None, "invalid record diagnostic entry"
        fields = tuple(trace.get(key) for key in ("layer", "kind", "start", "tokens"))
        if (not _integer(fields[0]) or fields[1] not in ("K", "V") or
                not _integer(fields[2]) or not _integer(fields[3], 1)):
            return None, "missing or invalid (layer, kind, start, tokens) record identity"
        if not _integer(trace.get("raw_bytes"), 1) or not _integer(trace.get("serialized_bytes")):
            return None, "invalid record byte counts"
        if trace["raw_bytes"] % 2:
            return None, "raw BF16 record byte count is not divisible by two"
        parts = trace.get("component_bytes")
        if (not isinstance(parts, dict) or any(not isinstance(k, str) or not _integer(v) for k, v in parts.items())
                or sum(parts.values()) != trace["serialized_bytes"]):
            return None, "record component bytes do not sum to serialized_bytes"
        identity.append((*fields, trace["raw_bytes"]))
        raw += trace["raw_bytes"]
        inner += trace["serialized_bytes"]
        components.update(parts)
        kind = by_kind.setdefault(fields[1], dict(records=0, raw_bytes=0, record_payload_bytes=0, component_bytes=Counter()))
        kind["records"] += 1
        kind["raw_bytes"] += trace["raw_bytes"]
        kind["record_payload_bytes"] += trace["serialized_bytes"]
        kind["component_bytes"].update(parts)
    if len(set(identity)) != len(identity):
        return None, "duplicate record identities"
    if raw != m.get("raw_payload_bytes"):
        return None, "record raw-byte sum differs from raw_payload_bytes"
    if "actual_record_payload_bytes" in m and inner != m["actual_record_payload_bytes"]:
        return None, "record encoded-byte sum differs from actual_record_payload_bytes"
    if m.get("serialized_bytes") != inner + _outer(len(traces)):
        return None, "serialized_bytes disagrees with record bytes plus declared outer framing"
    if m["records_total"] == len(traces) and m["full_raw_payload_bytes"] != raw:
        return None, "full coverage raw bytes disagree with full_raw_payload_bytes"
    if m["full_raw_payload_bytes"] < raw:
        return None, "full_raw_payload_bytes is smaller than measured raw bytes"
    for kind in by_kind.values():
        kind["component_bytes"] = dict(kind["component_bytes"])
    return dict(identity=identity, records=traces, raw=raw, inner=inner,
                components=dict(components), by_kind=by_kind, measured=len(traces),
                total=m["records_total"], full_raw=m["full_raw_payload_bytes"],
                state=m["decoder_state_bytes"]), None


def _full_size(view):
    return view["inner"] * view["total"] / view["measured"] + _outer(view["total"]) + view["state"]


def _candidate(row, view, error):
    m = row.get("measurement") or {}
    result = dict(name=row.get("name"), spec=row.get("spec") or {},
                  exact_verified=m.get("exact_verified"), roundtrip_exact=m.get("roundtrip_exact"),
                  reported_savings_including_state_percent=m.get("savings_including_state_percent"),
                  reported_estimated_full_savings_including_state_percent=m.get("estimated_full_savings_including_state_percent"),
                  reported_component_bytes=m.get("component_bytes") or {},
                  recipe_diagnostics=m.get("prefill_recipe") or {},
                  state_contract=dict(decoder_state_bytes=m.get("decoder_state_bytes"),
                      native_state_physical_bytes=m.get("native_state_physical_bytes"), state_id=m.get("state_id"),
                      serialized_model_bytes=(m.get("decoder_state_info") or {}).get("serialized_bytes"),
                      accounting=(m.get("decoder_state_info") or {}).get("accounting"),
                      scope="Full decoder_state_bytes follows the reported state contract and is charged once per candidate cache, never once per record or per K/V subtotal. Physical tensor and serialized model bytes are reported separately; they are not assumed identical."),
                  timing=_timing(m), matched_records_valid=view is not None)
    if error:
        result["record_validation_error"] = error
        return result
    measured_size = view["inner"] + _outer(view["measured"]) + view["state"]
    result.update(records_measured=view["measured"], records_total=view["total"],
                  measured_raw_bytes=view["raw"], measured_record_payload_bytes=view["inner"],
                  measured_outer_framing_bytes=_outer(view["measured"]),
                  measured_total_including_state_bytes=measured_size,
                  measured_savings_including_state_percent=_saving(measured_size, view["raw"]),
                  full_cache_total_including_state_bytes=_full_size(view),
                  full_cache_savings_including_state_percent=_saving(_full_size(view), view["full_raw"]),
                  full_cache_size_is_estimated=view["measured"] != view["total"],
                  record_component_bytes=view["components"],
                  record_component_bits_per_BF16_value={k: 16.0 * v / view["raw"] for k, v in view["components"].items()},
                  per_kind=view["by_kind"])
    return result


def _spec(row):
    spec = row.get("spec") or {}
    method = spec.get("method")
    options = dict(spec.get("options") or {})
    # Preserve original option defaults without importing accelerator modules.
    if method == "prefill":
        for key, value in dict(predictor="midpoint", correction="coset", layout="channel", group_size=8, step_log2=6).items():
            options.setdefault(key, value)
    elif method == "prefill_dictionary":
        for key, value in dict(predictor="affine", correction="residual", layout="channel", group_size=8, assignment="bitcost").items():
            options.setdefault(key, value)
    elif method == "prefill_exponent":
        for key, value in dict(predictor="median", layout="time", segment_size=256).items():
            options.setdefault(key, value)
    return method, options, spec.get("recipe_model")


def _ordered_pair(first, second):
    """Describe isolated option changes; mixed-family baselines are labelled."""
    ma, a, model_a = _spec(first)
    mb, b, model_b = _spec(second)
    if ma == mb and ma in ("prefill", "prefill_exponent", "prefill_dictionary"):
        # Different frozen dictionaries are not a one-axis coding ablation.
        if model_a != model_b:
            if ma == "prefill_dictionary" and a == b:
                count_a = (first.get("spec") or {}).get("dictionary_size")
                count_b = (second.get("spec") or {}).get("dictionary_size")
                if _integer(count_a, 1) and _integer(count_b, 1) and count_a != count_b:
                    baseline, candidate = (first, second) if count_a < count_b else (second, first)
                    return baseline, candidate, "dictionary_capacity_comparison", ["dictionary_capacity_and_ids_and_frozen_state"]
            return None
        changed = [key for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]
        if len(changed) != 1:
            return None
        axis = changed[0]
        order = {"layout": ("channel", "time"), "group_size": (8, 64),
                 "predictor": ("zero", "fixed127", "midpoint", "median", "affine", "e8"),
                 "correction": ("residual", "coset"), "assignment": ("l1", "bitcost")}.get(axis)
        if order is None or a.get(axis) not in order or b.get(axis) not in order:
            return None
        baseline, candidate = (first, second) if order.index(a[axis]) < order.index(b[axis]) else (second, first)
        return baseline, candidate, "matched_" + axis, [axis]
    if {ma, mb} == {"prefill", "prefill_dictionary"}:
        baseline, candidate = (first, second) if ma == "prefill" else (second, first)
        _, plain, _ = _spec(baseline)
        _, learned, _ = _spec(candidate)
        if (plain.get("predictor") == "affine" and plain.get("correction") == "residual"
                and plain.get("layout") == learned.get("layout")
                and plain.get("group_size") == learned.get("group_size")):
            return baseline, candidate, "add_frozen_residual_dictionary", ["residual_dictionary_and_ids_and_state"]
    if ma == "exponent" and mb in ("prefill", "prefill_dictionary", "prefill_exponent"):
        return first, second, "exponent_baseline_comparison", ["representation_family"]
    if mb == "exponent" and ma in ("prefill", "prefill_dictionary", "prefill_exponent"):
        return second, first, "exponent_baseline_comparison", ["representation_family"]
    return None


def _mismatch(a, b):
    if a["identity"] != b["identity"]:
        return "ordered (layer, kind, start, tokens, raw_bytes) record identities differ"
    if (a["total"], a["full_raw"]) != (b["total"], b["full_raw"]):
        return "full-cache geometry or raw-byte denominator differs"
    return None


def _compare(baseline, candidate, a, b, kind, changed):
    measured_a = a["inner"] + _outer(a["measured"]) + a["state"]
    measured_b = b["inner"] + _outer(b["measured"]) + b["state"]
    full_a, full_b = _full_size(a), _full_size(b)
    components = {key: a["components"].get(key, 0) - b["components"].get(key, 0)
                  for key in sorted(set(a["components"]) | set(b["components"]))}
    per_kind = {}
    for name in sorted(a["by_kind"]):
        ak, bk = a["by_kind"][name], b["by_kind"][name]
        gain = ak["record_payload_bytes"] - bk["record_payload_bytes"]
        per_kind[name] = dict(records=ak["records"], raw_bytes=ak["raw_bytes"],
            baseline_record_payload_bytes=ak["record_payload_bytes"], candidate_record_payload_bytes=bk["record_payload_bytes"],
            record_payload_bytes_saved=gain, record_payload_savings_gain_percentage_points=100.0 * gain / ak["raw_bytes"],
            scope="Actual matched record payloads only; cache-wide framing and decoder state are not assigned to K or V.")
    ar = baseline["measurement"].get("record_diagnostics", [])
    br = candidate["measurement"].get("record_diagnostics", [])
    record_gains = [x["serialized_bytes"] - y["serialized_bytes"] for x, y in zip(ar, br)]
    notes = []
    if kind == "matched_layout" and _spec(baseline)[0] == "prefill_exponent":
        notes.append("Exponent channel/time layout permutes the same correction multiset. Fixed tier counts, stream bit totals and segment counts are invariant; different header spelling can change total bytes. A byte delta is not evidence of a histogram or entropy improvement.")
    elif kind == "matched_layout":
        notes.append("Layout permutes corrections before fixed-width grouping; predictor and original sign information are preserved. Actual width, correction, padding and header costs are included.")
    if kind == "matched_group_size":
        notes.append("Group size trades width metadata against the maximum correction width paid by each group. Both costs are included; fewer width bytes alone do not establish a net gain.")
    if kind == "add_frozen_residual_dictionary":
        notes.append("Adds a frozen TRAIN residual dictionary, IDs, and final corrections after the same affine predictor. Full dictionary state is charged once; this is a combined mechanism rather than a single metadata-free option change.")
    if kind == "dictionary_capacity_comparison":
        notes.append("Changing dictionary capacity changes fitted centroids, ID widths, final corrections and full state cost together. This is a coupled capacity comparison, not an isolated option or free additional predictor.")
    return dict(baseline=baseline["name"], candidate=candidate["name"], comparison=kind, changed_axes=changed,
        isolated_option_change=kind.startswith("matched_"), records_measured=a["measured"], records_total=a["total"],
        measured_raw_bytes=a["raw"], measured_bytes_saved=measured_a - measured_b,
        measured_savings_gain_percentage_points=100.0 * (measured_a - measured_b) / a["raw"],
        baseline_measured_total_including_state_bytes=measured_a, candidate_measured_total_including_state_bytes=measured_b,
        decoder_state_increment_bytes=b["state"] - a["state"], record_component_bytes_saved=components,
        full_cache_bytes_saved=full_a - full_b,
        full_cache_savings_gain_percentage_points=100.0 * (full_a - full_b) / a["full_raw"],
        full_cache_size_is_estimated=a["measured"] != a["total"],
        record_wins=sum(x > 0 for x in record_gains), record_ties=sum(x == 0 for x in record_gains),
        record_losses=sum(x < 0 for x in record_gains), per_kind=per_kind, interpretation_notes=notes,
        baseline_timing=_timing(baseline["measurement"]), candidate_timing=_timing(candidate["measurement"]),
        scope="Positive gains mean fewer bytes for the candidate. Same ordered records and raw-byte denominator; specified framing plus full decoder state charged once per cache. Family comparisons change multiple mechanisms and do not isolate causality. Full-cache extrapolation assumes measured bytes per record.")


def _portfolio(rows, views, errors):
    eligible, excluded = [], []
    for row in rows:
        state = (row.get("measurement") or {}).get("decoder_state_bytes")
        if not _integer(state):
            excluded.append(dict(name=row["name"], reason="decoder_state_bytes missing or invalid"))
        elif state:
            excluded.append(dict(name=row["name"], reason="stateful candidate", decoder_state_bytes=state))
        else:
            eligible.append(row)
    result = dict(status="unavailable", mode_id_bytes_per_record=_MODE_BYTES,
        excluded_candidates=excluded, eligible_candidates=[row["name"] for row in eligible],
        scope="SIZE-ONLY oracle over packets that were all separately encoded. One byte per record for the selected mode plus specified cache header/index costs. No packet repacking or online selector exists here; selection compute, mode dispatch latency, encoder work, and attention benefit are not measured. This is not an implemented prefill policy or a latency result.")
    if not eligible:
        return dict(result, reason="no known stateless native candidates")
    if len(eligible) > 256:
        return dict(result, reason="one-byte mode identifier cannot address more than 256 candidates")
    for row in eligible:
        if errors.get(row["name"]):
            return dict(result, reason=f"eligible candidate {row['name']}: {errors[row['name']]}")
    reference = views[eligible[0]["name"]]
    for row in eligible[1:]:
        mismatch = _mismatch(reference, views[row["name"]])
        if mismatch:
            return dict(result, reason=f"eligible candidate {row['name']}: {mismatch}")
    # A stable name tie-break prevents row ordering from changing mode counts.
    eligible = sorted(eligible, key=lambda row: row["name"])
    names = [row["name"] for row in eligible]
    chosen = []
    for index in range(reference["measured"]):
        name = min(names, key=lambda name: (views[name]["records"][index]["serialized_bytes"], name))
        chosen.append((name, views[name]["records"][index]))
    count = reference["measured"]
    inner = sum(record["serialized_bytes"] for _, record in chosen)
    selector = count * _MODE_BYTES
    measured = inner + selector + _outer(count)
    full = (inner + selector) * reference["total"] / count + _outer(reference["total"])
    best_name = min(names, key=lambda name: (views[name]["inner"], name))
    best = views[best_name]
    best_measured = best["inner"] + _outer(count)
    best_full = _full_size(best)
    modes_by_kind = {}
    for name, record in chosen:
        modes_by_kind.setdefault(record["kind"], Counter()).update([name])
    result.update(status="ok", records_measured=count, records_total=reference["total"],
        measured_raw_bytes=reference["raw"], chosen_record_payload_bytes=inner,
        mode_id_bytes=selector, measured_outer_framing_bytes=_outer(count),
        measured_total_bytes=measured, measured_savings_percent=_saving(measured, reference["raw"]),
        full_cache_total_bytes=full, full_cache_savings_percent=_saving(full, reference["full_raw"]),
        full_cache_size_is_estimated=count != reference["total"],
        mode_counts=dict(Counter(name for name, _ in chosen)),
        mode_counts_by_kind={key: dict(value) for key, value in modes_by_kind.items()},
        best_fixed_stateless_candidate=best_name, best_fixed_measured_total_bytes=best_measured,
        measured_bytes_saved_over_best_fixed=best_measured - measured,
        measured_savings_gain_over_best_fixed_percentage_points=100.0 * (best_measured - measured) / reference["raw"],
        best_fixed_full_cache_total_bytes=best_full,
        full_cache_bytes_saved_over_best_fixed=best_full - full,
        full_cache_savings_gain_over_best_fixed_percentage_points=100.0 * (best_full - full) / reference["full_raw"],
        accounting="Selected actual inner record bytes + 1 mode byte per record + 32-byte cache header + 8*(record_count+1) offset bytes; no external decoder state. Full-cache estimates scale selected inner+mode bytes by total/measured before adding full framing. A mode table/dispatcher implementation is not serialized or benchmarked.")
    return result


def analyze_case(case):
    """Return JSON-serializable incremental evidence for this case only.

    Unmatched or missing traces produce explicit reasons, never pooled size
    comparisons. Stateful candidates remain in paired comparisons, but cannot
    enter the stateless portfolio. The input is not modified.
    """
    rows = [row for row in case.get("results", [])
            if row.get("track") == "native_packet" and row.get("status") == "ok"]
    names = [row.get("name") for row in rows]
    output = dict(format="qwen_recipe_analysis_v1", case=case.get("name"), split=case.get("split"),
        tokens=case.get("tokens"), candidates=[], comparisons=[], skipped_comparisons=[],
        scope="One standard benchmark case; DEV and TEST are never combined. Actual native packet records only, separate from direct-prefill observer measurements and CPU reference codecs. Savings include all measured packet components, calculated outer framing and full decoder state once. An estimate is not a full-cache measurement.")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        output["stateless_size_oracle"] = dict(status="unavailable", reason="missing or duplicate native candidate names")
        output["analysis_error"] = "missing or duplicate native candidate names"
        return output
    views, errors = {}, {}
    for row in rows:
        view, error = _view(row)
        views[row["name"]], errors[row["name"]] = view, error
        output["candidates"].append(_candidate(row, view, error))
    for first, second in combinations(rows, 2):
        pairing = _ordered_pair(first, second)
        if not pairing:
            continue
        baseline, candidate, kind, changed = pairing
        a, b = views[baseline["name"]], views[candidate["name"]]
        reason = errors[baseline["name"]] or errors[candidate["name"]]
        reason = reason or _mismatch(a, b)
        if reason:
            output["skipped_comparisons"].append(dict(baseline=baseline["name"], candidate=candidate["name"], comparison=kind, reason=reason))
        else:
            output["comparisons"].append(_compare(baseline, candidate, a, b, kind, changed))
    output["stateless_size_oracle"] = _portfolio(rows, views, errors)
    # Check that all supported primitives remain strict JSON before the caller
    # writes the artifact. This catches accidental tensor/NumPy values early.
    json.dumps(output, allow_nan=False)
    return output
