#!/usr/bin/env python3
"""Reproduce the gzip-aware A4500 frontier audit using the standard library.

Reads reports/state only. It does not rerun GPU kernels, infer a universal
entropy bound, or report hypothetical formats as measured implementations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import statistics

MIB=1<<20
FROZEN="producer_frozen_exponent"
BEST="frontier_shift_pre_rope_raw8"


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def file_info(path):
    data=path.read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def outer(count):return 32+8*(count+1)
def saving(m):return m["estimated_full_savings_including_state_percent"] if m["size_is_estimated"] else m["savings_including_state_percent"]
def full(m):return (m["estimated_full_cache_serialized_bytes"] if m["size_is_estimated"] else m["serialized_bytes"])+m["decoder_state_bytes"]
def ms(sample):return 1000*sample["median_seconds"]*MIB/sample["payload_bytes"]


def categories(parts):
    output=Counter()
    for key,value in parts.items():
        if key=="header":label="header"
        elif key in ("cache_header","pair_offset_table"):label="outer_framing"
        elif "offsets" in key:label="stream_indices"
        elif "head_shift" in key:label="head_shifts"
        elif "correction_width" in key:label="correction_widths"
        elif "correction" in key:label="correction_payload"
        elif "__primary__" in key:label="primary"
        elif "__lower" in key or "sign_mantissa" in key:label="lower8"
        elif key.endswith("primary"):label="primary"
        elif key.endswith("secondary"):label="secondary"
        elif key.endswith("rare"):label="rare"
        elif key.endswith("anchor"):label="anchors"
        else:raise ValueError("Unclassified component "+key)
        output[label]+=value
    return output


def compare(a,b):
    ac,bc=categories(a["component_bytes"]),categories(b["component_bytes"])
    gains={k:100*(ac[k]-bc[k])/a["raw_payload_bytes"] for k in sorted(ac.keys()|bc.keys())}
    gains["state"]=100*(a["decoder_state_bytes"]-b["decoder_state_bytes"])/a["full_raw_payload_bytes"]
    net=saving(b)-saving(a)
    assert abs(sum(gains.values())-net)<1e-9
    return dict(net_gain_percentage_points=net,component_gains_percentage_points=gains,
        scope="Same ordered pairs; positive removes candidate bytes. Headers and complete state count, so the total is not a pure predictor ablation.")


def subset_oracle(rows):
    names=[n for n in rows if n.startswith("producer_") or n.endswith("raw8")]
    # Every omitted plane packet has identical nonheader bytes and a longer
    # header than its raw8 counterpart; that dominance is checked in build().
    best=None
    for count in range(1,len(names)+1):
        for allowed in itertools.combinations(names,count):
            m=rows[allowed[0]];traces=[rows[n]["record_diagnostics"] for n in allowed]
            winners=[min(range(count),key=lambda j:(traces[j][i]["serialized_bytes"],allowed[j])) for i in range(m["records_measured"])]
            states={rows[n]["state_id"]:rows[n]["decoder_state_bytes"] for n in allowed}
            mode_bytes=int(count>1)*m["records_measured"]
            inner=sum(traces[j][i]["serialized_bytes"] for i,j in enumerate(winners))+mode_bytes
            size=inner*m["records_total"]/m["records_measured"]+outer(m["records_total"])+sum(states.values())
            savings=100*(1-size/m["full_raw_payload_bytes"])
            value=dict(candidates=list(allowed),savings_percent=savings,
                gain_over_best_fixed_percentage_points=savings-max(saving(x) for x in rows.values()),
                complete_state_union_bytes=sum(states.values()),selections=dict(Counter(allowed[j] for j in winners)))
            if best is None or savings>best["savings_percent"]:best=value
    best["subsets_examined"]=2**len(names)-1
    best["scope"]="TEST-informed offline size envelope over seven nondominated/raw candidates. Existing headers, one mode byte per pair when mixing, and complete activated artifact state union count. No learned selector, encoding cost, or adaptive latency improvement is demonstrated. Shared subarrays in distinct serialized states are conservatively charged separately."
    return best


def build(source,source_root):
    compressed=source/"results.json.gz"
    with gzip.open(compressed,"rb") as f:
        checksum=hashlib.sha256();nraw=0
        while chunk:=f.read(1<<20):checksum.update(chunk);nraw+=len(chunk)
    with gzip.open(compressed,"rt",encoding="utf-8") as f:r=json.load(f)
    read=lambda n:json.loads((source/n).read_text())
    sm,frozen=read("source_manifest.json"),read("frozen_manifest.json")
    checks=dict(completed=r["status"]=="completed",source_manifest_matches=r["source_manifest"]==sm,
        source_digest=digest(sm["files"])==sm["sources_sha256"],frozen_manifest_matches=r["freeze"]==frozen,
        frozen_digest=digest(frozen["manifest"])==frozen["sha256"],frozen_source=frozen["manifest"]["source_sha256"]==sm["sources_sha256"],
        frozen_candidates=frozen["manifest"]["candidates"]==r["candidates"],frozen_configuration=frozen["manifest"]["configuration"]==r["config"],
        frozen_calibration=frozen["manifest"]["calibration"]==r["calibration"])
    checks["source_files_match"]=all(file_info(source_root/p)["sha256"]==h for p,h in sm["files"].items())
    artifacts={};case_validation=[];cases=[];observations=defaultdict(list);pair_checks=0
    candidate_names={s["name"] for s in r["candidates"]}
    for case in r["cases"]:
        assert len(case["results"])==len(candidate_names)
        rows={x["name"]:x["measurement"] for x in case["results"]};assert set(rows)==candidate_names
        inline={x["name"]:x for x in case["capture"]["inline_recipe_metrics"]["methods"]}
        audit=case["capture"]["producer_audit"];identity=None
        for row in case["results"]:
            assert row["status"]=="ok"
            name,m=row["name"],row["measurement"];traces=m["record_diagnostics"]
            ids=[tuple(t[k] for k in ("layer","start","tokens","raw_bytes")) for t in traces]
            assert len(set(ids))==len(ids)==m["records_measured"]==m["records_checked"]
            if identity is None:identity=ids
            assert ids==identity and m["exact_verified"] and m["roundtrip_exact"] and m["inline_record_payload_matches_replay"]
            assert sum(t["raw_bytes"] for t in traces)==m["raw_payload_bytes"]
            assert sum(t["serialized_bytes"] for t in traces)==m["actual_record_payload_bytes"]
            assert m["serialized_bytes"]==m["actual_record_payload_bytes"]+outer(len(traces))==sum(m["component_bytes"].values())
            assert all(sum(t["component_bytes"].values())==t["serialized_bytes"] for t in traces)
            if m["size_is_estimated"]:assert m["estimated_full_cache_serialized_bytes"]==m["actual_record_payload_bytes"]*m["records_total"]/len(traces)+outer(m["records_total"])
            bits=16*full(m)/m["full_raw_payload_bytes"]
            assert abs(bits-m["effective_bits_per_BF16_word_including_state"])<1e-10
            assert abs(saving(m)-100*(1-full(m)/m["full_raw_payload_bytes"]))<1e-9
            assert m["target_40_percent"]["met"]==(bits<=9.6)
            si=m["decoder_state_info"];actual=file_info(source/si["file"]);artifacts[si["file"]]=actual
            assert actual==dict(bytes=si["serialized_bytes"],sha256=si["sha256"])
            assert m["decoder_state_bytes"]==max(actual["bytes"],m["native_state_physical_bytes"])==si["charged_bytes"]
            ir=inline[name];keyed=lambda seq:{(t["layer"],t["start"],t["tokens"]):(t["serialized_bytes"],t["component_bytes"]) for t in seq}
            assert ir["status"]=="ok" and ir["state_id"]==m["state_id"] and ir["records_checked"]==len(traces)
            assert keyed(ir["records"])==keyed(traces)
            assert m["sample"]["records"]==16 and m["sample"]["raw_bytes"]==4*MIB and m["sample_decode"]["repeats"]==3
            pair_checks+=len(traces)
        assert all(audit[k] for k in ("all_stage_targets_exact","v_hook_equals_cache","k_rope_replay_equals_cache","exported_cache_equals_hook_cache","rope_stable","hooks_removed"))
        assert audit["on_tile_callback_count"]==len(identity)
        case_validation.append(dict(name=case["name"],split=case["split"],methods=len(rows),pairs=len(identity),total_pairs=audit["total_tile_pairs"],exact=True,inline_matches=True))
        if case["split"]!="test":continue
        result=dict(name=case["name"],tokens=case["tokens"],methods=[],comparisons={})
        for name,m in rows.items():
            counts=Counter();side=Counter()
            for t in m["record_diagnostics"]:
                d=t["recipe_diagnostics"]
                if "replay_correction_mode" in d:counts["replay_"+d["replay_correction_mode"]]+=1
                for kind,sd in d.get("side_diagnostics",{}).items():
                    side[kind+"_shift_vectors_applied"]+=int(sd["head_shift_applied"])
                    side[kind+"_primary_"+sd["primary_storage"]]+=1
                    side[kind+"_lower_"+sd["lower_storage"]]+=1
            initial,after=m["memory"]["initial"],m["memory"]["after_measurement"]
            component_bits={k:16*v/m["raw_payload_bytes"] for k,v in categories(m["component_bytes"]).items()}
            # Inner record components extrapolate with the selected pairs;
            # the cache header/offset table is instead charged once at full size.
            component_bits["outer_framing"]=16*outer(m["records_total"])/m["full_raw_payload_bytes"]
            state_bits=16*m["decoder_state_bytes"]/m["full_raw_payload_bytes"]
            assert abs(sum(component_bits.values())+state_bits-16*full(m)/m["full_raw_payload_bytes"])<1e-10
            value=dict(name=name,savings_percent=saving(m),effective_bits=16*full(m)/m["full_raw_payload_bytes"],state_bytes=m["decoder_state_bytes"],
                decode_ms_per_raw_MiB=ms(m["sample_decode"]),restore_ms_per_raw_MiB=ms(m["sample_packet_transfer_then_decode"]),
                encode_ms_per_raw_MiB=1000*m["encode_seconds_total"]*MIB/m["raw_payload_bytes"],
                encode_decode_peak_extra_MiB=(after["peak_allocated_bytes"]-initial["allocated_bytes"])/MIB,
                component_bits_per_word=component_bits,
                state_bits_per_word=state_bits,choices=dict(counts),side_choices=dict(side))
            result["methods"].append(value);observations[name].append(value)
        chain=[FROZEN,"frontier_cost_final_raw8","frontier_shift_final_raw8",BEST,"frontier_shift_kproj_raw8"]
        for a,b in zip(chain,chain[1:]):result["comparisons"][a+"__to__"+b]=compare(rows[a],rows[b])
        for src in ("final","pre_rope","kproj"):
            base=rows[f"frontier_shift_{src}_raw8"]
            for mode in ("plane_raw","plane_first","plane_prev"):
                candidate=rows[f"frontier_shift_{src}_{mode}"]
                for a,b in zip(base["record_diagnostics"],candidate["record_diagnostics"]):
                    assert {k:v for k,v in a["component_bytes"].items() if k!="header"}=={k:v for k,v in b["component_bytes"].items() if k!="header"}
                    assert b["serialized_bytes"]>=a["serialized_bytes"]
        result["offline_subset_oracle"]=subset_oracle(rows)
        result["diagnostics"]={}
        for name,d in case["frontier_diagnostics"]["domains"].items():
            condition=d["conditional_lower8_given_exponent"]
            result["diagnostics"][name]=dict(lower_marginal=d["entropy"]["lower8"],exponent_marginal=d["entropy"]["exponent8"],
                lower_given_exponent=condition["global_pooled"],lower_given_layer_exponent=condition["pooled_by_layer"],
                lower_given_layer_channel_exponent=condition["pooled_by_layer_channel"],temporal=d["temporal_lower8"],cpu_lower_controls=d["cpu_lower8_controls"])
        cases.append(result)
    for cal in r["calibration"].values():
        actual=file_info(source/cal["file"]);artifacts[cal["file"]]=actual
        assert actual==dict(bytes=cal["serialized_bytes"],sha256=cal["model_id"])
        assert cal["metadata"]["training_only"] and len(cal["metadata"]["training_record_sets"])==3
    gen=[]
    for row in r["generation_checks"]:
        m=row["measurement"]
        assert row["status"]=="ok" and all(m[k] for k in ("all_logits_bitwise_equal","all_tokens_equal","all_appended_KV_bitwise_equal"))
        assert all(s["logits_finite"] for s in m["per_step"])
        gen.append(dict(name=row["name"],tokens=row["context_tokens"],steps=m["steps"],exact_and_finite=True))
    aggregate=[]
    for name,values in observations.items():
        item=dict(name=name)
        for k in ("savings_percent","effective_bits","decode_ms_per_raw_MiB","restore_ms_per_raw_MiB","encode_ms_per_raw_MiB","encode_decode_peak_extra_MiB"):
            v=[x[k] for x in values];item[k]=dict(mean=statistics.mean(v),minimum=min(v),maximum=max(v))
        item["component_bits_per_word"]={k:statistics.mean(x["component_bits_per_word"].get(k,0) for x in values) for k in sorted({k for x in values for k in x["component_bits_per_word"]})}
        item["state_bits_per_word"]=statistics.mean(x["state_bits_per_word"] for x in values)
        aggregate.append(item)
    validation=dict(format="frontier_review_validation_v1",all_checks_passed=all(checks.values()),checks=checks,source_files=len(sm["files"]),
        artifact_files_verified=artifacts,cases=case_validation,generation_checks=gen,pair_checks_per_track=pair_checks,
        compressed_report=file_info(compressed),uncompressed_report=dict(bytes=nraw,sha256=checksum.hexdigest(),gzip_crc_verified=True),
        scope="Recorded exactness, source/artifact identities and arithmetic audited. No independent GPU rerun or model-weight attestation. Model binary CRC/canonical/source verification is recorded by the separate loader audit.")
    analysis=dict(format="frontier_review_analysis_v1",environment=r["environment"],config=r["config"],elapsed_seconds=r["elapsed_seconds"],
        test_case_means=aggregate,cases=cases,target_bits_per_word=9.6,
        scope="Six correlated synthetic TEST cases, sampled full-cache sizes. Prepared-state eager codec timings; no matched raw-transfer, cold SSD, compact attention or concurrency evidence. Entropies are finite-sample estimates, never proven lower bounds or achieved encodings.")
    return validation,analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("input",type=Path);p.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists() and any(a.output.iterdir()):p.error("Output must be fresh or empty")
    v,r=build(a.input.resolve(),a.source_root.resolve());a.output.mkdir(parents=True,exist_ok=True)
    for name,value in (("analysis.json",r),("validation.json",v)):(a.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(all_checks_passed=v["all_checks_passed"],method_rows=sum(c["methods"] for c in v["cases"]),pair_checks_per_track=v["pair_checks_per_track"],output=str(a.output))))
    return 0 if v["all_checks_passed"] else 1


if __name__=="__main__":raise SystemExit(main())
