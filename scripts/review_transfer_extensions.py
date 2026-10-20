#!/usr/bin/env python3
"""Review a transfer-extension run and screen paid metadata opportunities.

Layer scores are ideal-bit diagnostics on held-out TEST captures. They are not
measured per-layer packets and cannot define a deployment gate; fit gates on DEV.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

import numpy as np

FORMAT = "transfer_extension_review_v1"
PREFIX = struct.Struct("<8sI")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def key(row):
    return row["case"], int(row["start"]), int(row["tokens"])


def close(a, b, tolerance=1e-7):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tolerance*max(1.,abs(float(a)),abs(float(b)))


def resolve_artifact(root, reported, folder):
    # Copied run bundles retain standard states/packet_checks subdirectories.
    name = Path(reported).name
    candidate = root/folder/name
    if candidate.is_file():
        return candidate
    reported = Path(reported)
    if reported.is_file() and reported.resolve().is_relative_to(root.resolve()):
        return reported
    raise ValueError(f"Missing run artifact: {folder}/{name}")


def inspect_state(path):
    with np.load(path, allow_pickle=False) as data:
        description = json.loads(data["description"].tobytes())
        if description.get("format") != "team_transfer_reference_v1":
            raise ValueError("Unsupported frozen-state format")
        expected = description.pop("state_sha256")
        digest = hashlib.sha256(canonical(description))
        arrays = {name:data[name] for name in data.files if name != "description"}
        for name, array in sorted(arrays.items()):
            digest.update(name.encode()); digest.update(array.dtype.str.encode())
            digest.update(canonical(list(array.shape))); digest.update(np.ascontiguousarray(array).tobytes())
        return dict(verified=digest.hexdigest()==expected, state_sha256=expected,
                    resident_array_bytes=sum(a.nbytes for a in arrays.values()),
                    serialized_bytes=path.stat().st_size, geometry=description["geometry"])


def inspect_packet(path):
    raw = path.read_bytes()
    if len(raw) < PREFIX.size:
        raise ValueError("Truncated packet")
    magic, size = PREFIX.unpack_from(raw)
    if magic != b"KVTRF001" or size > 1 << 20 or len(raw) < PREFIX.size+size:
        raise ValueError("Invalid packet prefix")
    header = json.loads(raw[PREFIX.size:PREFIX.size+size])
    if header.get("format") != "team_transfer_reference_v1":
        raise ValueError("Unsupported packet format")
    payload = raw[PREFIX.size+size:]
    if len(payload) != header["metadata_bytes"]+header["stream_bytes"] or sha(payload) != header["payload_sha256"]:
        raise ValueError("Packet payload identity mismatch")
    return header, len(raw), PREFIX.size+size


def selector_candidates(label, selection, dev, selected, block_name):
    if label.endswith("_matched_hidden_rank"):
        family = label.removesuffix("_matched_hidden_rank")
        return [r for r in dev if re.match(rf"^block_{family}_r\d+$", r["variant"]) and
                r.get("metadata_rank") == selected["hidden"]["rank"]]
    if label in ("kv", "hidden", "hidden_shuffled"):
        expression = rf"^block_{label}_r\d+$"
    elif label == "coset_plain":
        expression = rf"^{re.escape(block_name)}_coset_uniform_r\d+$"
    elif label == "coset_hidden":
        expression = rf"^{re.escape(selected['hidden']['variant'])}_coset_uniform_r\d+$"
    else:
        # Explicit candidate lists allow future selection families without
        # inferring a policy from TEST measurements.
        names = selection.get("candidate_variants")
        if names is None:
            raise ValueError(f"Unknown DEV selection family: {label}")
        return [r for r in dev if r["variant"] in names]
    return [r for r in dev if re.match(expression, r["variant"])]


def review(input_path, source_root=None):
    input_path = Path(input_path)
    if input_path.is_dir():
        input_path = input_path/"results.json"
    root = input_path.resolve().parent
    raw_report = input_path.read_bytes()
    report = json.loads(raw_report)
    checks = []
    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
    check("completed known study", report.get("format")=="transfer_extension_study_v1" and report.get("status")=="completed")
    repository = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[1]
    source_checks = []
    for source in report.get("provenance",{}).get("sources",[]):
        source_path = (repository/source["path"]).resolve()
        valid = source_path.is_relative_to(repository) and source_path.is_file() and sha(source_path.read_bytes())==source["sha256"]
        check(f"executed source hash matches source-root file: {source['path']}",valid)
        source_checks.append(dict(path=source["path"],sha256=source["sha256"],verified_against_current_file=valid))
    test, dev = report["test_rows"], report["dev_rows"]
    check("nonempty DEV and TEST", bool(dev) and bool(test))
    if not dev or not test:
        raise ValueError("Run has no DEV or TEST rows")
    block_name = f"block{report['arguments']['block_size']}"
    by_address = defaultdict(dict)
    for row in test:
        address = key(row)
        check(f"unique TEST row: {address}:{row['variant']}", row["variant"] not in by_address[address])
        by_address[address][row["variant"]] = row
    expected_variants = set(report["aggregate"])
    for address, variants in by_address.items():
        check(f"matched TEST grid: {address}", set(variants)==expected_variants and block_name in variants)
        base = variants.get(block_name, next(iter(variants.values())))
        check(f"same source geometry: {address}", all(r["values"]==base["values"] and r["raw_bytes"]==base["raw_bytes"] for r in variants.values()))
    check("DEV and TEST phases labelled", all(r.get("phase")=="DEV" for r in dev) and all(r.get("phase")=="TEST" for r in test))
    split = report.get("development_split")
    if split:
        stop = split["scale_stop"]
        selection_start = split["selection_start"]
        check("disjoint DEV scales and selection", split["scale_start"]==0 and stop<=selection_start and
              selection_start+split["selection_tokens"]<=split["dev_capture_tokens"] and
              all(r["start"]==selection_start and r["tokens"]==split["selection_tokens"] for r in dev))
    else:
        # HF runner specifies the DEV capture length. ACT imports need explicit
        # split metadata to establish this boundary; do not silently assume it.
        count = report["arguments"].get("capture_tokens")
        expected_start = count//2 if count else None
        check("DEV scale boundary matches selection pages", not report["arguments"].get("act_dir") and
              expected_start is not None and all(r["start"]==expected_start for r in dev),
              evidence="derived from HF capture_tokens; explicit development_split preferred")
    for row in dev:
        stats = row["codec_stats"]
        check(f"DEV complete-packet accounting: {row['variant']}",
              row["packet_bytes"]==stats["packet_bytes"]==stats["framing_bytes"]+stats["metadata_bytes"]+stats["entropy_stream_bytes"]+stats.get("raw_fallback_bytes",0) and
              close(row["bits_per_value"],8*row["packet_bytes"]/row["values"]))
    selections = []
    selected = report["selected_on_dev"]
    for label, selection in selected.items():
        candidates = selector_candidates(label, selection, dev, report["selected_on_dev"], block_name)
        group_addresses = {key(r) for r in candidates}
        check(f"matched DEV candidates: {label}", bool(candidates) and len(group_addresses)==1)
        if not candidates:
            continue
        field = "residue_bits" if label.startswith("coset_") else "metadata_rank"
        winner = min(candidates, key=lambda r:(r["packet_bytes"],r.get(field,0)))
        expected_name = winner["variant"]
        matched = label.endswith("_matched_hidden_rank")
        if matched and selected["kv" if label.startswith("kv_") else "hidden_shuffled"]["variant"] != expected_name:
            expected_name += "_matched_hidden_rank"
        frozen = selection["variant"]==expected_name and selection.get("rank",selection.get("residue_bits"))==winner.get(field)
        check(f"DEV {'matched-budget' if matched else 'minimum'} frozen: {label}", frozen,
              expected_variant=expected_name, recorded_variant=selection["variant"])
        check(f"frozen DEV choice reported on TEST: {label}", all(selection["variant"] in v for v in by_address.values()))
        for row in test:
            if row["variant"]==selection["variant"]:
                check(f"TEST uses frozen {field}: {label}:{key(row)}", row.get(field)==winner.get(field))
        selections.append(dict(family=label, winner=selection["variant"], dev_packet_bytes=winner["packet_bytes"],
                               candidate_variants=[r["variant"] for r in candidates], selection_split="DEV"))
    state_cache = {}
    evidence = []
    for row in test:
        tag = f"{row['variant']}:{row['case']}:{row['start']}"
        stats = row["codec_stats"]
        check(f"row arithmetic: {tag}", row["raw_bytes"]==2*row["values"] and
              close(row["bits_per_value"],8*row["packet_bytes"]/row["values"]) and
              close(row["saving_fraction"],1-row["packet_bytes"]/row["raw_bytes"]))
        check(f"paid packet decomposition: {tag}", row["packet_bytes"]==stats["packet_bytes"] and
              row["packet_bytes"]==stats["framing_bytes"]+stats["metadata_bytes"]+stats["entropy_stream_bytes"]+stats.get("raw_fallback_bytes",0))
        check(f"separate packet-only exact roundtrip reported: {tag}", row.get("bit_exact") is True and row.get("packet_only_subprocess") is True)
        inputs = row.get("decoder_inputs")
        check(f"only state and packet decoder inputs: {tag}", isinstance(inputs,list) and len(inputs)==2 and
              Path(inputs[0]).suffix==".npz" and "_encoder" not in Path(inputs[0]).name and Path(inputs[1]).suffix==".bin")
        if not isinstance(inputs,list) or len(inputs)!=2:
            continue
        try:
            state_path = resolve_artifact(root,inputs[0],"states")
            packet_path = resolve_artifact(root,inputs[1],"packet_checks")
            if state_path not in state_cache:
                state_cache[state_path] = inspect_state(state_path)
            state = state_cache[state_path]
            header, actual_bytes, header_bytes = inspect_packet(packet_path)
            child_path = packet_path.with_suffix(".json")
            child = json.loads(child_path.read_text())
            decoded_path = packet_path.with_suffix(".npy")
            decoded = np.load(decoded_path,allow_pickle=False)
            shape = header["shape"]
            decoded_hash = sha(decoded.astype("<u2",copy=False).tobytes())
            child_array_hash = hashlib.sha256(str(decoded.dtype).encode()+canonical(decoded.shape)+decoded.tobytes()).hexdigest()
            child_inputs = child.get("inputs",[])
            check(f"artifact evidence valid: {tag}", state["verified"] and
                  header["state_sha256"]==state["state_sha256"]==stats["state_sha256"] and
                  actual_bytes==row["packet_bytes"] and header_bytes==stats["framing_bytes"] and
                  header["start"]==row["start"] and math.prod(shape)==row["values"] and shape[1]==row["tokens"] and
                  tuple(shape)==decoded.shape and decoded.dtype==np.uint16 and decoded_hash==header["decoded_sha256"] and
                  child["stats"].get("exact_checksum_verified") is True and child["decoded_sha256"]==child_array_hash and
                  len(child_inputs)==2 and [Path(p).name for p in child_inputs]==[Path(p).name for p in inputs] and
                  header["correction"]==row["correction"] and header["residue_bits"]==row["residue_bits"] and
                  header["metadata_bytes"]==stats["metadata_bytes"])
            if not stats.get("fallback"):
                check(f"metadata bytes fully charged: {tag}",stats["metadata_bytes"]==shape[0]*shape[1]*row["metadata_rank"])
            evidence.append(dict(variant=row["variant"],case=row["case"],start=row["start"],tokens=row["tokens"],
                                 state_file=state_path.name,packet_file=packet_path.name,
                                 packet_sha256=sha(packet_path.read_bytes()),decoded_sha256=decoded_hash,
                                 geometry=state["geometry"],decoder_resident_array_bytes=state["resident_array_bytes"],
                                 decoder_serialized_bytes=state["serialized_bytes"]))
        except (OSError,ValueError,KeyError,TypeError) as error:
            check(f"artifact evidence valid: {tag}",False,error=str(error))
    for address, variants in by_address.items():
        matched=[e for e in evidence if (e["case"],e["start"],e["tokens"])==address]
        check(f"paired variants recover identical original words: {address}",
              len(matched)==len(variants) and len({e["decoded_sha256"] for e in matched})==1)
    # Bind every compared page to the actual stored source capture, not just
    # self-consistent packet checksums or agreement among all variants.
    capture_records = {r["name"]:r for r in report.get("provenance",{}).get("model_identity",{}).get("captures",[]) if r.get("split")=="test"}
    capture_cache = {}
    capture_evidence = []
    for address, variants in by_address.items():
        case,start,count = address
        try:
            if case not in capture_cache:
                record = capture_records.get(case)
                if record:
                    path = resolve_artifact(root,record["capture_file"],"captures")
                else:
                    act_dir = report["arguments"].get("act_dir")
                    if not act_dir or Path(case).name != case:
                        raise ValueError("Missing actual TEST capture identity")
                    path = Path(act_dir).resolve()/(case+".npz")
                with np.load(path,allow_pickle=False) as archive:
                    keys,values = archive["K_bits"],archive["V_bits"]
                    if keys.dtype!=np.uint16 or values.dtype!=np.uint16 or keys.shape!=values.shape or keys.ndim!=3:
                        raise ValueError("Invalid original KV capture geometry")
                    original = np.concatenate((keys,values),axis=2)
                    full_hash = hashlib.sha256(str(original.dtype).encode()+canonical(original.shape)+original.tobytes()).hexdigest()
                    if record and full_hash != record["kv_sha256"]:
                        raise ValueError("Actual capture does not match recorded full-KV hash")
                    if record and "ids_sha256" in record:
                        ids=np.ascontiguousarray(archive["ids"])
                        ids_hash=hashlib.sha256(str(ids.dtype).encode()+canonical(ids.shape)+ids.tobytes()).hexdigest()
                        if ids_hash!=record["ids_sha256"]:
                            raise ValueError("Actual capture does not match recorded token IDs")
                capture_cache[case]=(path,original,full_hash)
            path,original,full_hash=capture_cache[case]
            if start<0 or start+count>original.shape[1]:
                raise ValueError("TEST page is outside its actual capture")
            page=original[:,start:start+count]
            source_hash=sha(page.astype("<u2",copy=False).tobytes())
            matched=[e for e in evidence if (e["case"],e["start"],e["tokens"])==address]
            check(f"all variants match actual captured source words: {address}",
                  len(matched)==len(variants) and all(e["decoded_sha256"]==source_hash for e in matched) and
                  all(r["values"]==page.size for r in variants.values()))
            capture_evidence.append(dict(case=case,start=start,tokens=count,capture_file=path.name,
                                         full_kv_sha256=full_hash,page_raw_sha256=source_hash,values=int(page.size)))
        except (OSError,ValueError,KeyError,TypeError) as error:
            check(f"all variants match actual captured source words: {address}",False,error=str(error))
    paired = []
    layers = defaultdict(lambda:defaultdict(lambda:dict(pages=0,values=0,baseline_ideal_bits=0.,candidate_ideal_bits=0.,metadata_bits=0)))
    for address, variants in sorted(by_address.items()):
        if block_name not in variants:
            continue
        base = variants[block_name]
        for name,row in sorted(variants.items()):
            stats,b = row["codec_stats"],base["codec_stats"]
            paired.append(dict(case=row["case"],start=row["start"],tokens=row["tokens"],variant=name,
                               values=row["values"],baseline_packet_bytes=base["packet_bytes"],packet_bytes=row["packet_bytes"],
                               packet_bytes_saved=base["packet_bytes"]-row["packet_bytes"],
                               packet_bpv_saved=8*(base["packet_bytes"]-row["packet_bytes"])/row["values"],
                               framing_bytes_delta=stats["framing_bytes"]-b["framing_bytes"],
                               metadata_bytes=stats["metadata_bytes"],metadata_bytes_delta=stats["metadata_bytes"]-b["metadata_bytes"],
                               entropy_bytes_saved=b["entropy_stream_bytes"]-stats["entropy_stream_bytes"],
                               correction=row["correction"],metadata_rank=row["metadata_rank"]))
            if row["correction"]=="coset_model" and row["metadata_rank"]==0:
                check(f"same-model coset ideal chain identity: {address}:{name}",
                      close(stats["ideal_bits"],b["ideal_bits"],1e-9),
                      direct_ideal_bits=b["ideal_bits"],coset_ideal_bits=stats["ideal_bits"])
            original_layers = b.get("layer_ideal_bits",[])
            candidate_layers = stats.get("layer_ideal_bits",[])
            if row["correction"]!="direct" or not row["metadata_rank"] or not original_layers or not candidate_layers:
                continue
            check(f"layer diagnostic geometry: {address}:{name}", len(original_layers)==len(candidate_layers) and
                  row["values"]%len(original_layers)==0 and
                  close(sum(candidate_layers),stats["ideal_bits"]) and
                  close(sum(original_layers),b["ideal_bits"]))
            if len(original_layers)!=len(candidate_layers):
                continue
            for layer,(before,after) in enumerate(zip(original_layers,candidate_layers)):
                totals=layers[name][layer]
                totals["pages"]+=1;totals["values"]+=row["values"]//len(original_layers)
                totals["baseline_ideal_bits"]+=before;totals["candidate_ideal_bits"]+=after
                totals["metadata_bits"]+=8*row["metadata_rank"]*row["tokens"]
    layer_screening = {}
    for name, table in layers.items():
        rows=[]
        for layer,totals in sorted(table.items()):
            net=totals["baseline_ideal_bits"]-totals["candidate_ideal_bits"]-totals["metadata_bits"]
            rows.append(dict(layer=layer,**totals,net_ideal_bits_saved_after_metadata=net,
                             net_ideal_bpv_saved_after_metadata=net/totals["values"]))
        layer_screening[name]=dict(scope="TEST ideal-bit opportunity screening only; not measured layer packets or a deployable gate",
                                   rows=rows,positive_layers=[r["layer"] for r in rows if r["net_ideal_bits_saved_after_metadata"]>0],
                                   negative_or_zero_layers=[r["layer"] for r in rows if r["net_ideal_bits_saved_after_metadata"]<=0],
                                   required_next_step="Learn any layer/rank gate on DEV; encode its decisions/state and evaluate once on fresh TEST.")
    aggregate=[]
    for name,original in report["aggregate"].items():
        rows=[r for r in test if r["variant"]==name]
        values=sum(r["values"] for r in rows);packet_bytes=sum(r["packet_bytes"] for r in rows)
        check(f"aggregate arithmetic: {name}", original["values"]==values and original["packet_bytes"]==packet_bytes and
              close(original["bits_per_value"],8*packet_bytes/values))
        project=original.get("projector_statistics",{})
        decoder=original["model_statistics"]["frozen_array_bytes"]
        encoder=project.get("array_bytes",0)
        serialized=original["shared_state_wire_bytes"]
        matching=[e for e in evidence if e["variant"]==name]
        if matching:
            check(f"decoder state bytes match arrays: {name}",all(e["decoder_resident_array_bytes"]==decoder for e in matching))
        check(f"encoder and decoder state separately account total: {name}",original["shared_state_bytes"]==decoder+encoder and
              original.get("resident_decoder_state_bytes",decoder)==decoder and
              original.get("resident_encoder_projector_bytes",encoder)==encoder)
        check(f"serialized endpoint state bytes: {name}",serialized==original.get("serialized_decoder_state_bytes", matching[0]["decoder_serialized_bytes"] if matching else 0)+
              original.get("serialized_encoder_projector_bytes",project.get("serialized_bytes",0)))
        bpv=8*packet_bytes/values
        per_token=values/sum(r["tokens"] for r in rows)
        amortized={}
        for tokens in (1024,8192,32768,131072):
            denominator=tokens*per_token
            amortized[str(tokens)]={
                "decoder_resident_state_plus_packet_bpv":bpv+8*decoder/denominator,
                "encoder_resident_projector_bpv_equivalent":8*encoder/denominator,
                "serialized_all_state_plus_packet_bpv":bpv+8*serialized/denominator,
                "assumption":"Observed packet rate held constant; one shared state per compatible model. Decoder and encoder bytes reside on different machines."}
        paired_rows=[p for p in paired if p["variant"]==name]
        aggregate.append(dict(variant=name,values=values,packet_bytes=packet_bytes,packet_bpv=bpv,
                              saving_fraction=1-bpv/16,packet_bytes_saved_vs_block=sum(p["packet_bytes_saved"] for p in paired_rows),
                              packet_bpv_saved_vs_block=8*sum(p["packet_bytes_saved"] for p in paired_rows)/values,
                              metadata_bytes=sum(r["codec_stats"]["metadata_bytes"] for r in rows),
                              metadata_rank=rows[0]["metadata_rank"],correction=rows[0]["correction"],
                              decoder_resident_array_bytes=decoder,encoder_resident_projector_bytes=encoder,
                              combined_resident_bytes=decoder+encoder,serialized_all_state_bytes=serialized,
                              decoder_serialized_bytes=matching[0]["decoder_serialized_bytes"] if matching else None,
                              encoder_serialized_bytes=project.get("serialized_bytes",0),
                              amortization_by_tokens=amortized,
                              encode_seconds=sum(r["encode_wall_seconds"] for r in rows),
                              decode_seconds=sum(r["decode_wall_seconds"] for r in rows)))
    # Report whether the frozen selected hidden and KV controls actually have
    # equal byte budgets; unequal DEV-selected ranks remain valid alternatives.
    control={}
    selected=report["selected_on_dev"]
    if "hidden" in selected and "kv" in selected:
        control=dict(hidden_rank=selected["hidden"]["rank"],kv_rank=selected["kv"]["rank"],
                     equal_rank=selected["hidden"]["rank"]==selected["kv"]["rank"],
                     interpretation="Equal-rank DEV candidates provide a matched control. Unequal frozen TEST ranks do not establish an equal-budget hidden-vs-KV comparison.")
    rank_controls=[]
    for hidden in dev:
        if not re.match(r"^block_hidden_r\d+$",hidden["variant"]):
            continue
        controls=[r for r in dev if r["variant"]==f"block_kv_r{hidden['metadata_rank']}" and key(r)==key(hidden)]
        if controls:
            kv=controls[0]
            rank_controls.append(dict(rank=hidden["metadata_rank"],split="DEV",hidden_packet_bytes=hidden["packet_bytes"],
                                      kv_packet_bytes=kv["packet_bytes"],hidden_bytes_saved_vs_kv=kv["packet_bytes"]-hidden["packet_bytes"]))
    matched_control={}
    if "hidden" in selected and "kv_matched_hidden_rank" in selected:
        matched_control=dict(hidden_variant=selected["hidden"]["variant"],kv_variant=selected["kv_matched_hidden_rank"]["variant"],
                             hidden_rank=selected["hidden"]["rank"],kv_rank=selected["kv_matched_hidden_rank"]["rank"],
                             equal_rank=selected["hidden"]["rank"]==selected["kv_matched_hidden_rank"]["rank"])
    return dict(format=FORMAT,all_passed=all(c["passed"] for c in checks),passed=sum(c["passed"] for c in checks),total=len(checks),
                source_results_sha256=sha(raw_report),checks=checks,source_scope=report.get("scope"),source_provenance_checks=source_checks,source_provenance_root=str(repository),
                exact_original_reproduction=report.get("exact_original_reproduction",False),
                frozen_dev_selections=selections,paired_packet_deltas=paired,aggregate=aggregate,
                packet_only_artifact_evidence=evidence,source_capture_evidence=capture_evidence,layer_ideal_screening=layer_screening,
                selected_hidden_kv_control=control,matched_hidden_kv_control=matched_control,matched_rank_dev_controls=rank_controls,
                inference_limits=["Historical team rates use a different checkpoint/corpus; this study cannot establish their exact numerical reproduction.",
                                  "Layer screening uses TEST ideal costs and pays raw metadata bytes; it excludes layer framing/state differences and cannot be presented as a deployed gated codec.",
                                  "Coset model is a same-distribution factorization control; complete packet bytes decide benefit.",
                                  "Codec timing is CPU reference timing; encoder and decoder state occupy different machines."])


def markdown(insights):
    lines=["# Transfer extension review","",f"**{insights['passed']}/{insights['total']} checks passed.**",
           "", "All results below refer to independent controlled captures. They do not reproduce the team's original GGUF/document rates.","",
           "## Complete warm packet comparison","",
           "| Variant | Packet bpv | Reduction | Bpv saved vs block | Metadata bytes | Decoder MiB | Encoder projector MiB |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for row in insights["aggregate"]:
        lines.append(f"| {row['variant']} | {row['packet_bpv']:.4f} | {100*row['saving_fraction']:.2f}% | {row['packet_bpv_saved_vs_block']:+.4f} | {row['metadata_bytes']:,} | {row['decoder_resident_array_bytes']/2**20:.2f} | {row['encoder_resident_projector_bytes']/2**20:.2f} |")
    lines += ["", "Packet deltas include framing, metadata and coded corrections. Serialized state and separate encoder/decoder resident amortization are retained in insights.json.","",
              "## Layer opportunities: diagnostic only","",
              "These are TEST ideal-bit screens after paying 8 × metadata_rank × token_count bits per layer. They are not measured layer packets or a deployed gate. Any gate must be fitted on DEV, serialized and tested on fresh held-out data.",""]
    for name,screen in insights["layer_ideal_screening"].items():
        lines += [f"**{name}**: positive layers {screen['positive_layers']}; nonpositive layers {screen['negative_or_zero_layers']}.","",
                  "| Layer | Net ideal bpv saved after metadata |","|---|---:|"]
        for row in screen["rows"]:
            lines.append(f"| {row['layer']} | {row['net_ideal_bpv_saved_after_metadata']:+.4f} |")
        lines.append("")
    lines += ["## Interpretation limits","",*[f"- {line}" for line in insights["inference_limits"]],""]
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1],
                        help="Repository or preserved executed-source snapshot for provenance hash checks")
    args=parser.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be fresh or empty")
    result=review(args.input,source_root=args.source_root)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"insights.json").write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    (args.output/"summary.md").write_text(markdown(result))
    print(json.dumps(dict(all_passed=result["all_passed"],passed=result["passed"],total=result["total"],
                          failed_checks=[c["check"] for c in result["checks"] if not c["passed"]],output=str(args.output))))
    return 0 if result["all_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
