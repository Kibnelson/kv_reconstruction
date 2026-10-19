"""Decision evidence for the four conditional-reconstruction experiments.

No fit, winner selection or new codec runs here. Every candidate, including a
loss or failure, stays visible. Full-cache state and framing count once rather
than being multiplied by the number of sampled records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import statistics


def protocol(specs):
    return dict(
        format="qwen_conditional_protocol_v1",
        ideas={
            "1": "Frozen TRAIN Huffman coding of exponent prediction symbols",
            "2": "Separate exact sign prediction/coding; retain seven mantissa bits",
            "3": "Joint K/V prediction from independently stored KV anchor coordinates",
            "4": "Joint K/V prediction from small actual-prefill hidden features",
        },
        candidates=[dict(spec) for spec in specs],
        fit_source="Three explicit TRAIN captures only; no DEV or TEST refitting",
        evaluation="Same ordered layer/start/token records and timing sample for every candidate",
        state_accounting="Complete activated serialized state or physical device tensors, whichever is larger, once per full cache",
        rate_objective="Compare actual complete packet bytes and decoder state; fit objectives are separately recorded by each model",
        exactness="Original BF16 words; verified upstream replay plus exact correction fallback; complete DEV continuation representatives",
        logging="Per-record components, model/probability identities, prediction/correction diagnostics, fallback decisions, encode/decode/transfer times and isolated allocation peaks",
        decoder_scope="Selected-device eager tensor implementations; no implicit CPU entropy decode",
        decision_scope="All fixed outcomes retained. TEST summaries are descriptive, not a TEST-trained online selector",
        raw_capture_policy="Reports/state only; original hidden tensors and raw KV are not exported",
        serving_scope="Raw KV remains resident. No compact-attention, concurrency, capacity or serving-speedup result is inferred")


def _range(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, prefix + "." + str(key) if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{prefix}[{index}]")
    elif isinstance(value, (bool, str, int, float)) or value is None:
        yield prefix, value


def _aggregate_diagnostics(traces):
    numeric, categorical = defaultdict(list), defaultdict(Counter)
    for trace in traces:
        for path, value in _leaves(trace.get("recipe_diagnostics", {})):
            if isinstance(value, bool) or isinstance(value, str):
                categorical[path][str(value)] += 1
            elif isinstance(value, (int, float)) and math.isfinite(value):
                numeric[path].append(value)
    return dict(
        numeric={key: dict(**_range(values), sum=sum(values)) for key, values in sorted(numeric.items())},
        categorical={key: dict(counts) for key, counts in sorted(categorical.items())},
        scope="Scalar diagnostic paths, including histogram bins, aggregated across recorded pairs. Missing paths are not treated as zeros; count gives observations. Sums of non-count metrics have no coding interpretation.")


def _metric(row):
    m = row["measurement"]
    estimated = bool(m["size_is_estimated"])
    total = m["estimated_full_cache_serialized_bytes"] if estimated else m["serialized_bytes"]
    raw = m["full_raw_payload_bytes"]
    timing = m.get("sample_decode", {})
    restore = m.get("sample_packet_transfer_then_decode", {})
    encode_ms = 1000 * m["encode_seconds_total"] * (1 << 20) / m["raw_payload_bytes"]
    convert = lambda sample: (1000 * sample["median_seconds"] * (1 << 20) / sample["payload_bytes"]
                              if sample.get("payload_bytes") else None)
    # Inner fields scale with measured raw words. Outer cache framing is
    # charged once using full geometry, so the ledger sums to the reported rate.
    scale = m["records_total"] / m["records_measured"] if estimated else 1
    parts = {key: 16 * value * scale / raw
             for key, value in m["component_bytes"].items()
             if key not in ("cache_header", "pair_offset_table")}
    parts["outer_cache_framing"] = 16 * (32 + 8 * (m["records_total"] + 1)) / raw
    parts["complete_decoder_state"] = 16 * m["decoder_state_bytes"] / raw
    bits = 16 * (total + m["decoder_state_bytes"]) / raw
    if not math.isclose(sum(parts.values()), bits, rel_tol=0, abs_tol=1e-9):
        raise AssertionError("Conditional decision ledger does not sum to actual complete rate")
    addresses = [(t["layer"], t["start"], t["tokens"], t["raw_bytes"]) for t in m["record_diagnostics"]]
    layers = defaultdict(list)
    for trace in m["record_diagnostics"]:
        layers[str(trace["layer"])].append(16 * trace["serialized_bytes"] / trace["raw_bytes"])
    return dict(name=row["name"], spec=row.get("spec"), status=row["status"],
                size_is_estimated=estimated, effective_bits_per_word=bits,
                savings_including_state_percent=100 * (1 - bits / 16),
                gap_bits_to_40_percent=bits - 9.6,
                target_40_percent_met=bits <= 9.6,
                state_id=m["state_id"], decoder_state_bytes=m["decoder_state_bytes"],
                records_measured=m["records_measured"], records_total=m["records_total"],
                component_bits_per_word=parts,
                encode_ms_per_raw_MiB=encode_ms,
                state_preparation=m.get("state_preparation"),
                decode_ms_per_raw_MiB=convert(timing),
                packet_transfer_then_decode_ms_per_raw_MiB=convert(restore),
                isolated_workspace=m.get("isolated_workspace"),
                per_layer_record_bits_per_word={key: _range(values) for key, values in sorted(layers.items())},
                diagnostics=_aggregate_diagnostics(m["record_diagnostics"])), addresses


def analyze(report):
    cases, test_values = [], defaultdict(list)
    candidates = {row["name"]: row for row in report.get("candidates", [])}
    for case in report.get("cases", []):
        methods, addresses = [], {}
        for row in case.get("results", []):
            if row.get("status") != "ok":
                methods.append(dict(name=row["name"], status=row.get("status"), error=row.get("error")))
                continue
            value, ids = _metric(row)
            methods.append(value); addresses[row["name"]] = ids
            if case["split"] == "test":
                test_values[row["name"]].append(value)
        controls = {m["name"]: m for m in methods if m["status"] == "ok" and (
            m["name"] == "producer_frozen_exponent" or candidates.get(m["name"], {}).get("family") == "control")}
        for m in methods:
            if m["status"] != "ok":
                continue
            m["matched_control_deltas"] = {}
            for name, baseline in controls.items():
                if addresses[name] == addresses[m["name"]]:
                    m["matched_control_deltas"][name] = dict(
                        savings_percentage_points=m["savings_including_state_percent"] - baseline["savings_including_state_percent"],
                        decode_ms_per_raw_MiB=(m["decode_ms_per_raw_MiB"] - baseline["decode_ms_per_raw_MiB"]
                                               if m["decode_ms_per_raw_MiB"] is not None and baseline["decode_ms_per_raw_MiB"] is not None else None))
        same = len({tuple(x) for x in addresses.values()}) <= 1
        cases.append(dict(name=case["name"], split=case["split"], tokens=case["tokens"],
                          all_successful_methods_use_same_ordered_records=same, methods=methods,
                          source_head_diagnostics=case.get("conditional_diagnostics"),
                          source_head_diagnostics_cpu_seconds=case.get("conditional_diagnostics_cpu_seconds")))
    averages = []
    for name, values in test_values.items():
        fields = ("effective_bits_per_word", "savings_including_state_percent", "gap_bits_to_40_percent",
                  "encode_ms_per_raw_MiB", "decode_ms_per_raw_MiB", "packet_transfer_then_decode_ms_per_raw_MiB")
        averages.append(dict(name=name, spec=candidates.get(name), cases=len(values),
                             **{key: _range([v[key] for v in values]) for key in fields}))
    return dict(format="qwen_conditional_decision_summary_v1", status=report["status"],
                experiment_protocol=report.get("experiment_protocol"),
                source_sha256=(report.get("source_manifest") or {}).get("sources_sha256"),
                frozen_sha256=(report.get("freeze") or {}).get("sha256"),
                calibration=report.get("calibration"), cases=cases,
                training_fit_wall_seconds=report.get("training_fit_wall_seconds"),
                test_case_averages=averages,
                generation_checks=[dict(name=row["name"], status=row["status"], error=row.get("error"))
                                   for row in report.get("generation_checks", [])],
                scope="Every candidate retained, including losses/errors. Averages are descriptive unweighted TEST case means; length variants are correlated. No TEST-derived fitted policy, new codec, entropy bound or serving result is produced here.")
