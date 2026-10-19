#!/usr/bin/env python3
"""Audit and analyze a forward-producer study without rerunning a model.

Standard library only. Candidate comparisons use identical paired records.
Projections are arithmetic bounds, not implemented codecs or serving results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import statistics

MIB = 1 << 20
MEDIAN = "producer_kv_median"
FROZEN = "producer_frozen_exponent"
HIDDEN = "producer_hidden_exponent"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def info(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def outer(count):
    return 32 + 8 * (count + 1)


def savings(m):
    return m["estimated_full_savings_including_state_percent"] if m["size_is_estimated"] else m["savings_including_state_percent"]


def full_bytes(m):
    return (m["estimated_full_cache_serialized_bytes"] if m["size_is_estimated"] else m["serialized_bytes"]) + m["decoder_state_bytes"]


def ms(sample):
    return 1000 * sample["median_seconds"] * MIB / sample["payload_bytes"]


def category(name):
    if name in ("cache_header", "pair_offset_table"):
        return "outer_framing"
    if name == "header":
        return name
    if name == "shared_features":
        return "features"
    if name.endswith(("__lower", "__sign_mantissa")):
        return "lower_bits"
    if name.endswith("offsets"):
        return "stream_indices"
    if name.endswith("exponent_anchor"):
        return "anchors"
    if name.endswith("correction_widths"):
        return "correction_widths"
    if name.endswith("__correction"):
        return "corrections"
    if name.endswith(("primary", "secondary", "rare")):
        return "exponent_payload"
    raise ValueError("Unclassified byte component: " + name)


def grouped(parts):
    result = Counter()
    for key, value in parts.items():
        result[category(key)] += value
    return result


def compare(a, b):
    """Positive component gain means candidate b removes bytes."""
    ac, bc = grouped(a["component_bytes"]), grouped(b["component_bytes"])
    raw = a["raw_payload_bytes"]
    gains = {k: 100 * (ac[k] - bc[k]) / raw for k in sorted(ac.keys() | bc.keys())}
    gains["state"] = 100 * (a["decoder_state_bytes"] - b["decoder_state_bytes"]) / a["full_raw_payload_bytes"]
    net = savings(b) - savings(a)
    assert abs(sum(gains.values()) - net) < 1e-9
    layer = defaultdict(lambda: [0, 0])
    for x, y in zip(a["record_diagnostics"], b["record_diagnostics"]):
        assert tuple(x[k] for k in ("layer", "start", "tokens", "raw_bytes")) == tuple(y[k] for k in ("layer", "start", "tokens", "raw_bytes"))
        layer[x["layer"]][0] += x["serialized_bytes"] - y["serialized_bytes"]
        layer[x["layer"]][1] += x["raw_bytes"]
    return dict(net_savings_gain_percentage_points=net, component_gains_percentage_points=gains,
        record_payload_gain_by_layer={str(k): 100 * v[0] / v[1] for k, v in sorted(layer.items())},
        scope="Actual paired packet components plus full state once per cache; positive means fewer candidate bytes. Changes in header format are included and are not predictor-quality evidence.")


def oracle(rows, allowed):
    """Offline size envelope with mode bytes and a conservative complete state union."""
    chosen = [rows[name] for name in allowed]
    base = chosen[0]
    traces = [m["record_diagnostics"] for m in chosen]
    winners = [min(range(len(chosen)), key=lambda j: (traces[j][i]["serialized_bytes"], allowed[j])) for i in range(base["records_measured"])]
    states = {}
    for m in chosen:
        states[m["state_id"]] = m["decoder_state_bytes"]
    count = base["records_measured"]
    mode_bytes = int(len(allowed)>1)
    payload = sum(traces[j][i]["serialized_bytes"] for i, j in enumerate(winners)) + mode_bytes*count
    estimated = payload * base["records_total"] / count + outer(base["records_total"]) + sum(states.values())
    fixed = max(savings(m) for m in chosen)
    result = 100 * (1 - estimated / base["full_raw_payload_bytes"])
    return dict(candidates=allowed, savings_percent=result, gain_over_best_fixed_percentage_points=result-fixed,
        mode_bytes_per_pair=mode_bytes, all_candidate_states_charged_bytes=sum(states.values()),
        selections=dict(Counter(allowed[j] for j in winners)),
        scope="TEST-informed offline size calculation, not a deployment policy or tested adaptive codec. All allowed state artifacts are charged, shared identical state IDs once. Existing packet headers remain; one selector byte per pair is added for multiple candidates. No selection/encoding/restore speed is measured.")


def best_subset_oracle(rows):
    names=list(rows)
    candidates=(oracle(rows,list(subset)) for n in range(1,len(names)+1) for subset in itertools.combinations(names,n))
    best=max(candidates,key=lambda x:x["savings_percent"])
    best["gain_over_best_existing_fixed_percentage_points"] = best["savings_percent"]-max(savings(m) for m in rows.values())
    best["subsets_examined"] = 2**len(names)-1
    best["scope"] += " Exhaustive candidate-subset search includes fixed single methods with no selector byte; all choices use TEST results and provide no held-out adaptive-policy evidence."
    return best


def build(source, source_root):
    r, source_manifest, frozen = (read(source/name) for name in ("results.json", "source_manifest.json", "frozen_manifest.json"))
    checks = dict(completed=r["status"] == "completed", source_manifest_matches=r["source_manifest"] == source_manifest,
        source_digest=digest(source_manifest["files"]) == source_manifest["sources_sha256"],
        frozen_manifest_matches=r["freeze"] == frozen, frozen_digest=digest(frozen["manifest"]) == frozen["sha256"],
        frozen_source=frozen["manifest"]["source_sha256"] == source_manifest["sources_sha256"],
        frozen_calibration=frozen["manifest"]["calibration"] == r["calibration"],
        frozen_candidates=frozen["manifest"]["candidates"] == r["candidates"],
        frozen_configuration=frozen["manifest"]["configuration"] == r["config"])
    source_files = {name: info(source_root/name)["sha256"] == expected for name, expected in source_manifest["files"].items()}
    checks["all_local_source_hashes_match"] = all(source_files.values())
    artifact_paths = {v["file"] for v in r["calibration"].values()}
    artifacts = {}
    cases, observations, case_checks = [], defaultdict(list), []
    total_pairs = 0
    for case in r["cases"]:
        rows = {row["name"]: row["measurement"] for row in case["results"]}
        assert set(rows) == {s["name"] for s in r["candidates"]} and len(rows) == len(case["results"])
        audit = case["capture"]["producer_audit"]
        inline = {row["name"]: row for row in case["capture"]["inline_recipe_metrics"]["methods"]}
        expected = None
        for row in case["results"]:
            assert row["status"] == "ok", row
            name, m = row["name"], row["measurement"]
            assert m["roundtrip_exact"] and m["exact_verified"] and m["inline_record_payload_matches_replay"]
            traces = m["record_diagnostics"]
            identities = [tuple(t[k] for k in ("layer", "start", "tokens", "raw_bytes")) for t in traces]
            assert len(identities) == len(set(identities)) == m["records_measured"]
            if expected is None:
                expected = identities
            assert expected == identities
            assert sum(t["raw_bytes"] for t in traces) == m["raw_payload_bytes"]
            assert sum(t["serialized_bytes"] for t in traces) == m["actual_record_payload_bytes"]
            assert m["serialized_bytes"] == m["actual_record_payload_bytes"] + outer(len(traces))
            assert sum(m["component_bytes"].values()) == m["serialized_bytes"]
            assert all(sum(t["component_bytes"].values()) == t["serialized_bytes"] for t in traces)
            assert abs(savings(m) - 100 * (1-full_bytes(m)/m["full_raw_payload_bytes"])) < 1e-9
            shape = case["capture"]["geometry"]
            assert m["records_total"] == shape["layers"] * ((case["tokens"]+r["config"]["tile_tokens"]-1)//r["config"]["tile_tokens"])
            if m["size_is_estimated"]:
                estimate = m["actual_record_payload_bytes"]*m["records_total"]/m["records_measured"]+outer(m["records_total"])
                assert estimate == m["estimated_full_cache_serialized_bytes"]
            si = m["decoder_state_info"]
            artifact_paths.add(si["file"])
            actual = info(source/si["file"])
            assert actual == dict(bytes=si["serialized_bytes"], sha256=si["sha256"])
            assert m["decoder_state_bytes"] == max(actual["bytes"], m["native_state_physical_bytes"]) == si["charged_bytes"]
            ir = inline[name]
            assert ir["status"] == "ok" and ir["records_checked"] == len(traces) and ir["state_id"] == m["state_id"]
            keyed = lambda seq: {(t["layer"],t["start"],t["tokens"]):(t["serialized_bytes"],t["component_bytes"]) for t in seq}
            assert keyed(ir["records"]) == keyed(traces)
            total_pairs += len(traces)
        assert all(audit[k] for k in ("all_stage_targets_exact", "v_hook_equals_cache", "k_rope_replay_equals_cache", "exported_cache_equals_hook_cache", "rope_stable", "hooks_removed"))
        assert audit["on_tile_callback_count"] == len(expected) and audit["on_tile_stage"].startswith("attention_forward_post_hook;")
        case_checks.append(dict(name=case["name"], split=case["split"], methods=len(rows), pair_records=len(expected),
            total_pair_records=next(iter(rows.values()))["records_total"], exact=True, inline_matches_replay=True))
        if case["split"] != "test":
            continue
        result = dict(name=case["name"], tokens=case["tokens"], methods=[], comparisons={}, projections={}, oracles={})
        for name, m in rows.items():
            value = dict(name=name, savings_percent=savings(m), state_bytes=m["decoder_state_bytes"],
                decode_ms_per_raw_MiB=ms(m["sample_decode"]), transfer_decode_ms_per_raw_MiB=ms(m["sample_packet_transfer_then_decode"]),
                staging_ms_per_raw_MiB=1000*m["input_staging_seconds_total"]*MIB/m["raw_payload_bytes"],
                component_bits_per_BF16_word={k:16*v/m["raw_payload_bytes"] for k,v in grouped(m["component_bytes"]).items()})
            result["methods"].append(value); observations[name].append(value)
        for baseline, candidate in ((MEDIAN,FROZEN),(FROZEN,HIDDEN),(FROZEN,"producer_hidden_abs_exponent"),(MEDIAN,"producer_pre_rope_coset"),(MEDIAN,"producer_kproj_norm_rope_coset")):
            result["comparisons"][baseline+"__to__"+candidate] = compare(rows[baseline],rows[candidate])
        for name in ("producer_pre_rope_coset","producer_kproj_norm_rope_coset"):
            m = rows[name]; parts = m["component_bytes"]
            assert parts["K__correction"] == 0 and all(t["component_bytes"]["K__correction"] == 0 for t in m["record_diagnostics"])
            removable = parts["K__correction_widths"]-m["records_measured"]
            result["projections"][name] = dict(zero_correction_payload_verified=True,
                width_bytes=parts["K__correction_widths"], proposed_flag_bytes=m["records_measured"],
                projected_savings_gain_percentage_points=100*removable/m["raw_payload_bytes"],
                projected_savings_percent=savings(m)+100*removable/m["raw_payload_bytes"],
                scope="Unimplemented zero-correction mode projection: remove observed all-zero width tables, charge one mode byte per pair, keep all other bytes/state. Encoder must verify exact replay and retain a correction fallback; this is not measured new-format savings or speed.")
        for src in ("pre_rope","kproj_norm_rope"):
            a,b=rows[f"producer_{src}_residual"],rows[f"producer_{src}_coset"]
            assert {k:v for k,v in a["component_bytes"].items() if k!="header"} == {k:v for k,v in b["component_bytes"].items() if k!="header"}
        result["oracles"]["frozen_median_prerope"] = oracle(rows,[FROZEN,MEDIAN,"producer_pre_rope_coset"])
        result["oracles"]["all_nine"] = oracle(rows,list(rows))
        result["oracles"]["best_subset_including_state"] = best_subset_oracle(rows)
        cases.append(result)
    for path in sorted(artifact_paths):
        artifacts[path] = info(source/path)
    for c in r["calibration"].values():
        assert artifacts[c["file"]] == dict(bytes=c["serialized_bytes"],sha256=c["model_id"])
        assert c["metadata"]["training_only"] and len(c["metadata"]["training_record_sets"]) == 3
        assert all(t["split"]=="train" for t in c["metadata"]["training_record_sets"])
    generation=[]
    for row in r["generation_checks"]:
        m=row["measurement"]
        assert row["status"]=="ok" and all(m[k] for k in ("all_logits_bitwise_equal","all_tokens_equal","all_appended_KV_bitwise_equal"))
        assert all(s["logits_finite"] for s in m["per_step"])
        generation.append(dict(name=row["name"],context_tokens=row["context_tokens"],steps=m["steps"],exact_and_finite=True))
    aggregate=[]
    for name,values in observations.items():
        item=dict(name=name,cases=len(values))
        for key in ("savings_percent","decode_ms_per_raw_MiB","transfer_decode_ms_per_raw_MiB","staging_ms_per_raw_MiB"):
            vals=[v[key] for v in values]
            item[key]=dict(mean=statistics.mean(vals),minimum=min(vals),maximum=max(vals))
        aggregate.append(item)
    validation=dict(format="producer_review_validation_v1",all_checks_passed=all(checks.values()),checks=checks,
        source_files_checked=len(source_files),artifact_files_verified=artifacts,cases=case_checks,generation_checks=generation,
        exact_pair_packet_checks_per_inline_or_replay_track=total_pairs,
        scope="Recorded checks and artifact/source identities verified; model weights and original runtime are not independently attested. The separate model-loader audit checks binary canonicalization/CRC.")
    analysis=dict(format="producer_review_analysis_v1",environment=r["environment"],config=r["config"],elapsed_seconds=r["elapsed_seconds"],
        test_case_count=len(cases),test_case_method_means=aggregate,cases=cases,
        scope="Six correlated synthetic TEST cases; full-cache sizes are sample extrapolations. Same-record comparisons within this run. Timing describes eager, synchronized, prepared-state microbenchmarks. No matched raw transfer, cold SSD, compact attention, concurrency or serving gain is demonstrated.")
    return validation,analysis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        p.error("--output must be fresh or empty")
    validation,analysis=build(a.input.resolve(),a.source_root.resolve())
    a.output.mkdir(parents=True,exist_ok=True)
    for name,value in (("validation.json",validation),("analysis.json",analysis)):
        (a.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(all_checks_passed=validation["all_checks_passed"],source_files=validation["source_files_checked"],
        artifacts=len(validation["artifact_files_verified"]),test_cases=len(analysis["cases"]),output=str(a.output))))
    return 0 if validation["all_checks_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
