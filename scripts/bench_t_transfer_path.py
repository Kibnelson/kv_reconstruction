#!/usr/bin/env python3
"""GPU -> checked compressed host packet -> GPU KV; network is not measured."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_head_spread import check_base,file_sha,load_words,Window,write_json
from scripts import bench_t_baselines as base

PATHS=("legacy_checked","pinned_checked")


def view(packet):
    from kv_compare.pinned_transfer import HostPacket
    return packet.view() if isinstance(packet,HostPacket) else memoryview(packet)


class Legacy:
    def __init__(self,adapter,job):self.adapter,self.job=adapter,job
    def pack(self,encoded,digest):return base.packet_from(self.adapter,self.job["method"],encoded,self.job,digest)[0]
    def receive(self,packet):return base.packet_to(self.adapter,self.job["method"],packet,self.job,self.adapter.device)
    def memory(self):return dict(pinned_host_bytes=0,device_transfer_bytes=0,allocation="per-call original checked serialization")
    def poison_receiver(self):pass


def invoke(flag,job,result,logfile):
    with Path(logfile).open("w") as log:
        p=subprocess.run([sys.executable,"-u",str(Path(__file__).resolve()),flag,str(job)],stdout=log,stderr=subprocess.STDOUT)
    if p.returncode or not Path(result).is_file():
        raise RuntimeError(f"worker exited {p.returncode}: {Path(logfile).read_text(errors='replace')[-3000:]}")
    return json.loads(Path(result).read_text())


def decode_worker(job):
    """New receiver has only packet and frozen shared-state inputs."""
    import torch
    from kv_compare import pinned_transfer as transfer
    device=base.gpu()
    adapter=base.checked_codec(job,device)
    try:
        data=Path(job["packet"]).read_bytes()
        path=transfer.make(adapter,job["method"],job["shape"],job["state_sha256"],len(data))
        path.poison_receiver()
        packet=transfer.HostPacket.copy_from(data,pin=True)
        encoded,expected=path.receive(packet)
        actual=base.digest_words(adapter.decode(encoded))
        if actual!=expected or actual!=job["expected_sha256"]:
            raise AssertionError("fresh pinned receiver changed BF16 words")
        write_json(job["result"],dict(bit_exact=True,packet_sha256=hashlib.sha256(data).hexdigest(),
            source_words_sha256=actual,capture_path_provided=False,receiver="pinned_checked",gpu_executed=True))
    finally:adapter.close()


def run_fresh(adapter,path,words,digest,device):
    import torch
    begin=time.perf_counter()
    encoded=adapter.encode(words)
    torch.cuda.synchronize(device);t1=time.perf_counter()
    packet=path.pack(encoded,digest)
    del encoded
    torch.cuda.synchronize(device);t2=time.perf_counter()
    received,expected=path.receive(packet)
    torch.cuda.synchronize(device);t3=time.perf_counter()
    if expected!=digest:raise AssertionError("packet source identity changed")
    output=adapter.decode(received)
    torch.cuda.synchronize(device);end=time.perf_counter()
    return output,packet,dict(encode_ms=(t1-begin)*1000,pack_D2H_ms=(t2-t1)*1000,
        restore_H2D_ms=(t3-t2)*1000,decode_ms=(end-t3)*1000,total_ms=(end-begin)*1000)


def run_retrieval(adapter,path,packet,digest,device):
    import torch
    begin=time.perf_counter()
    received,expected=path.receive(packet)
    torch.cuda.synchronize(device);middle=time.perf_counter()
    if expected!=digest:raise AssertionError("packet source identity changed")
    output=adapter.decode(received)
    torch.cuda.synchronize(device);end=time.perf_counter()
    return output,dict(restore_H2D_ms=(middle-begin)*1000,decode_ms=(end-middle)*1000,total_ms=(end-begin)*1000)


def timing_worker(job):
    import torch,triton
    from kv_compare.predict import Doc
    from kv_compare import pinned_transfer as transfer
    device=base.gpu();check_base()
    adapter=base.checked_codec(job,device)
    rows,pending=[],[]
    try:
        for ci,case in enumerate(job["cases"]):
            begin=time.perf_counter()
            words=load_words(Window(Doc(job["captures"],case["name"]),0,job["shape"][1]),job["shape"][0],device)
            torch.cuda.synchronize(device)
            staging_ms=(time.perf_counter()-begin)*1000
            if list(words.shape)!=job["shape"] or base.digest_words(words)!=case["source_words_sha256"]:
                raise ValueError("capture words differ from the frozen reference")
            paths=list(PATHS)
            if (ci+job["round"])%2:paths.reverse()
            for name in paths:
                print(job["method"],name,case["name"],"round",job["round"]+1,flush=True)
                if job["method"] in base.FROZEN:adapter.arena.zero_()
                if job["method"]=="exans":adapter.records.zero_()
                torch.cuda.synchronize(device)
                before=torch.cuda.mem_get_info(device)[0];begin=time.perf_counter()
                path=(Legacy(adapter,job) if name=="legacy_checked" else
                    transfer.make(adapter,job["method"],job["shape"],job["state_sha256"],job["capacity"]))
                path.poison_receiver()
                torch.cuda.synchronize(device)
                setup=dict(ms=(time.perf_counter()-begin)*1000,
                    device_free_delta_bytes=before-torch.cuda.mem_get_info(device)[0],**path.memory())
                # Canonical packet equality is checked for this exact encoded output.
                encoded=adapter.encode(words)
                canonical=base.packet_from(adapter,job["method"],encoded,job,case["source_words_sha256"])[0]
                packet=path.pack(encoded,case["source_words_sha256"])
                sha=hashlib.sha256(view(packet)).hexdigest()
                if sha!=hashlib.sha256(canonical).hexdigest() or len(view(packet))!=len(canonical):
                    raise AssertionError("optimized serialization changed packet bytes")
                if len(canonical)!=case["packet_bytes"]:
                    raise AssertionError("packet size differs from frozen GPU comparison")
                if job["method"] in base.FROZEN and sha!=case["packet_sha256"]:
                    raise AssertionError("T packet differs from frozen GPU reference")
                received,expected=path.receive(packet)
                if expected!=case["source_words_sha256"] or not torch.equal(adapter.decode(received),words):
                    raise AssertionError("poisoned receiver failed full reconstruction")
                del canonical,encoded,received,packet
                for _ in range(job["warmup"]):
                    output,packet,_=run_fresh(adapter,path,words,case["source_words_sha256"],device)
                    if not torch.equal(output,words):raise AssertionError("fresh warmup changed words")
                    del output,packet
                fresh=[]
                for sample in range(job["repeats"]):
                    # Release the previous immutable packet before timing the next.
                    if sample:del packet
                    baseline=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                    output,packet,timing=run_fresh(adapter,path,words,case["source_words_sha256"],device)
                    timing["torch_peak_incremental_bytes"]=max(0,torch.cuda.max_memory_allocated(device)-baseline)
                    if not torch.equal(output,words) or len(view(packet))!=case["packet_bytes"]:
                        raise AssertionError("fresh timed path changed words or paid size")
                    timing["bit_exact"]=True;fresh.append(timing);del output
                # Freeze the last prepared host packet for the retrieval scenario.
                # Only restore+decode are timed; no saved decoded output is used.
                sha=hashlib.sha256(view(packet)).hexdigest()
                if job["method"] in base.FROZEN and sha!=case["packet_sha256"]:
                    raise AssertionError("timed T packet changed frozen bytes")
                for _ in range(job["warmup"]):
                    output,_=run_retrieval(adapter,path,packet,case["source_words_sha256"],device)
                    if not torch.equal(output,words):raise AssertionError("retrieval warmup changed words")
                    del output
                retrieval=[]
                for _ in range(job["repeats"]):
                    baseline=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                    output,timing=run_retrieval(adapter,path,packet,case["source_words_sha256"],device)
                    timing["torch_peak_incremental_bytes"]=max(0,torch.cuda.max_memory_allocated(device)-baseline)
                    if not torch.equal(output,words):raise AssertionError("retrieval timed path changed words")
                    timing["bit_exact"]=True;retrieval.append(timing);del output
                row=dict(method=job["method"],path=name,case=case["name"],round=job["round"],
                    packet_bytes=len(view(packet)),packet_sha256=sha,source_words_sha256=case["source_words_sha256"],
                    raw_bytes=words.numel()*2,values=words.numel(),canonical_packet_identical=True,
                    frozen_T_packet_identical=True if job["method"] in base.FROZEN else None,
                    fresh=fresh,retrieval=retrieval,setup=setup,input_staging_ms=staging_ms,
                    bit_exact=True,packet_only_proofs=None)
                if name=="pinned_checked" and job["round"]==0:
                    prefix=Path(job["output"])/(job["method"]+"_"+case["name"])
                    packet_path=prefix.with_suffix(".packet")
                    with packet_path.open("wb") as f:f.write(view(packet))
                    proof=dict(method=job["method"],state=job["state"],state_sha256=job["state_sha256"],
                        shape=job["shape"],packet=str(packet_path),expected_sha256=case["source_words_sha256"])
                    pending.append((len(rows),prefix,proof,packet_path))
                rows.append(row)
                del packet,path
                gc.collect();torch.cuda.empty_cache()
            del words
    finally:adapter.close()
    del adapter
    gc.collect();torch.cuda.empty_cache()
    # Separate receiver processes cannot inherit original KV tensors or arena data.
    for index,prefix,proof,packet_path in pending:
        proofs={}
        for receiver in ("legacy","pinned"):
            pj=dict(proof,result=str(prefix)+"."+receiver+".proof.json")
            jp=Path(str(prefix)+"."+receiver+".receiver.json");write_json(jp,pj)
            logfile=Path(str(prefix)+"."+receiver+".receiver.log")
            proofs[receiver]=(base.invoke("--decode-job",jp,pj["result"],logfile) if receiver=="legacy"
                else invoke("--decode-job",jp,pj["result"],logfile))
        rows[index]["packet_only_proofs"]=proofs
        packet_path.unlink()
    write_json(job["result"],dict(rows=rows,device_name=torch.cuda.get_device_name(device),
        torch=torch.__version__,triton=triton.__version__))


def aggregate(rows,methods,cases,rounds,repeats):
    table=[]
    for method in methods:
        for path in PATHS:
            rr=[r for r in rows if r["method"]==method and r["path"]==path]
            complete={(r["case"],r["round"]) for r in rr}=={(c,i) for c in cases for i in range(rounds)}
            if not rr:
                table.append(dict(method=method,path=path,complete=False));continue
            first={r["case"]:r for r in rr};values=sum(r["values"] for r in first.values())
            packet=sum(r["packet_bytes"] for r in first.values())
            result=dict(method=method,path=path,complete=complete and len(rr)==len(cases)*rounds,
                packet_bytes=packet,values=values,bpv=packet*8/values,saved_percent=100*(1-packet/(2*values)),
                all_exact=all(r["bit_exact"] and r["canonical_packet_identical"] and
                    all(s["bit_exact"] for sc in ("fresh","retrieval") for s in r[sc]) for r in rr),
                sample_counts_valid=all(len(r[sc])==repeats for r in rr for sc in ("fresh","retrieval")),
                packet_sizes_stable=all(len({r["packet_bytes"] for r in rr if r["case"]==c})==1 for c in first))
            proof_rows=[r for r in rr if r["round"]==0]
            result["fresh_receiver_proofs_valid"]=(path=="legacy_checked" or
                (len(proof_rows)==len(first) and all(r.get("packet_only_proofs") and
                    set(r["packet_only_proofs"])=={"legacy","pinned"} and
                    all(p["bit_exact"] and p["packet_sha256"]==r["packet_sha256"] and
                        p["source_words_sha256"]==r["source_words_sha256"] for p in r["packet_only_proofs"].values())
                    for r in proof_rows)))
            for scenario in ("fresh","retrieval"):
                fields=("total_ms","encode_ms","pack_D2H_ms","restore_H2D_ms","decode_ms") if scenario=="fresh" else ("total_ms","restore_H2D_ms","decode_ms")
                for field in fields:
                    result[scenario+"_"+field]=statistics.mean(statistics.median(
                        statistics.median(s[field] for s in r[scenario]) for r in rr if r["case"]==c) for c in first)
                result[scenario+"_ideal_serial_link_ms"]={str(rate):result[scenario+"_total_ms"]+packet/len(first)*8/(rate*1e6)
                    for rate in (3,10,25,50,100,200,400)}
            table.append(result)
    return table


def decision(report):
    table={(r["method"],r["path"]):r for r in report.get("aggregate",[])}
    full=(report.get("gpu_executed",False) and report.get("full_reference_coverage",False)
          and not report.get("errors") and report["arguments"]["rounds"]>=3)
    result={"automatic_merge":False,"network_measured":False}
    for method in report["arguments"]["methods"]:
        old=table.get((method,"legacy_checked"),{});new=table.get((method,"pinned_checked"),{})
        valid=full and all(r.get("complete") and r.get("all_exact") and r.get("sample_counts_valid")
            and r.get("packet_sizes_stable") and r.get("fresh_receiver_proofs_valid") for r in (old,new))
        gain=1-new["fresh_total_ms"]/old["fresh_total_ms"] if old.get("fresh_total_ms",0)>0 and new.get("fresh_total_ms") is not None else None
        result[method]=dict(candidate_for_review=bool(valid and gain is not None and gain>=.05),
            fresh_gain_fraction=gain,criterion="GPU, all reference windows, >=3 rounds, unchanged bytes, exact decode, >=5% complete local-path gain")
    return result


def summary(report):
    lines=["# Checked local transfer path: original versus pinned buffers","",
        f"Status: **{report['status']}**. GPU executed: **{report.get('gpu_executed',False)}**. Full reference coverage: **{report.get('full_reference_coverage',False)}**.","",
        "| Method / path | BF16 saved | Fresh local path ms | Stored-packet retrieval ms | Complete |",
        "|---|---:|---:|---:|---|"]
    for r in report.get("aggregate",[]):
        if "fresh_total_ms" in r:lines.append(f"| {r['method']} / {r['path']} | {r['saved_percent']:.4f}% | {r['fresh_total_ms']:.3f} | {r['retrieval_total_ms']:.3f} | {r['complete']} |")
        else:lines.append(f"| {r['method']} / {r['path']} | unavailable | | | False |")
    lines += ["", "Fresh: resident GPU KV -> encode -> compact checked host packet -> restore on GPU -> decode. Stored-packet retrieval starts with that compressed host packet already prepared and times restore+decode. Both include full packet SHA-256 verification. CUDA is synchronized at serial stage boundaries. All byte/word parity checks are outside timers; the protocol checksum stays inside.","",
        "This is a one-GPU host-buffer handoff. It measures no NIC/socket/network or SSD I/O, no stage overlap, no shared-state loading and no P1 conversion. Network time is only modeled separately in results.json. Capture-file input staging and buffer setup are separate fields. Pinned input must remain owned until receive completes.","",
        "Every method uses the same new serializer treatment; native encoder/decoder allocations and algorithm remain unchanged. Canonical packets must retain the earlier byte format; T packets additionally match the frozen reference SHA-256. Reused receive buffers are poisoned before first validation, and independent original/new receivers decode the pinned packets with no capture input.","",
        json.dumps(report.get("decision",{}),sort_keys=True),"","No automatic merge, calibration or refit."]
    if report.get("errors"):lines += ["", "Errors:"]+["- "+str(x) for x in report["errors"]]
    if report.get("fatal_error"):lines += ["", "```",report["fatal_error"],"```"]
    return "\n".join(lines)+"\n"


def benchmark(args,output,report):
    reference_path=Path(args.reference_run).resolve()/"results.json"
    ref=json.loads(reference_path.read_text())
    if not ref.get("full_comparison_validated") or not ref.get("gpu_executed") or ref.get("errors"):
        raise ValueError("use the completed full native GPU comparison as reference")
    check_base()
    for name,digest in ref["sources_sha256"].items():
        if file_sha(ROOT/name)!=digest:raise ValueError("reference implementation changed: "+name)
    cases=args.cases.split(",") if args.cases else ref["cases"]
    if len(set(cases))!=len(cases) or not set(cases)<=set(ref["cases"]):raise ValueError("distinct frozen reference cases required")
    captures=Path(args.captures or Path(ref["inputs"][0]["path"]).parent).resolve()
    expected={Path(x["path"]).name:x["sha256"] for x in ref["inputs"]}
    inputs=[]
    for name in ["meta.json"]+[c+"_"+s for c in cases for s in ("K_bits.npy","V_bits.npy","ids.npy")]:
        p=captures/name;h=file_sha(p)
        if h!=expected[name]:raise ValueError("frozen capture changed: "+name)
        inputs.append(dict(path=str(p),sha256=h))
    states={}
    for method in args.methods:
        info=ref["states"][method]
        if file_sha(info["path"])!=info["sha256"]:raise ValueError("shared state changed: "+method)
        states[method]=info
    old={(x["method"],x["case"]):x for x in ref["rows"] if x["round"]==0}
    report.update(reference_results_sha256=file_sha(reference_path),cases=cases,shape=ref["shape"],inputs=inputs,
        model=ref["model"],states=states,full_reference_coverage=set(cases)==set(ref["cases"]),
        gpu_executed=False,actual_network_measured=False,automatic_merge=False,refit=False,errors=[],rows=[],workers=[],
        source_sha256={name:file_sha(ROOT/name) for name in ("kv_compare/pinned_transfer.py","scripts/bench_t_transfer_path.py","scripts/run_t_transfer_path_nuc.sh")})
    for ri in range(args.rounds):
        order=args.methods[ri%len(args.methods):]+args.methods[:ri%len(args.methods)]
        for method in order:
            prefix=output/(method+"_round"+str(ri));state=states[method]
            cc=[dict(name=c,source_words_sha256=old[method,c]["source_words_sha256"],
                packet_bytes=old[method,c]["packet_bytes"],packet_sha256=old[method,c]["packet_sha256"]) for c in cases]
            cc=cc[ri%len(cc):]+cc[:ri%len(cc)]
            job=dict(method=method,state=state["path"],state_sha256=state["sha256"],shape=ref["shape"],
                capacity=max(old[method,c]["packet_bytes"] for c in ref["cases"]),cases=cc,captures=str(captures),
                round=ri,warmup=args.warmup,repeats=args.repeats,output=str(output),result=str(prefix)+".json")
            jp=Path(str(prefix)+".job.json");write_json(jp,job)
            print(f"GPU local path {ri+1}/{args.rounds}: {method}",flush=True)
            try:
                value=invoke("--timing-job",jp,job["result"],Path(str(prefix)+".log"))
                report["rows"].extend(value.pop("rows"));report["workers"].append(dict(method=method,round=ri,**value))
                report["gpu_executed"]=True
            except Exception as e:
                report["errors"].append(dict(method=method,round=ri,error=str(e)));print(str(e),flush=True)
            write_json(output/"results.json",report)
    report["aggregate"]=aggregate(report["rows"],args.methods,cases,args.rounds,args.repeats)
    report["status"]="completed" if not report["errors"] else "incomplete"
    report["decision"]=decision(report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timing-job");p.add_argument("--decode-job")
    p.add_argument("--reference-run");p.add_argument("--output");p.add_argument("--captures");p.add_argument("--cases")
    p.add_argument("--methods",default=",".join(base.METHODS))
    p.add_argument("--rounds",type=int,default=3);p.add_argument("--warmup",type=int,default=1);p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args()
    if args.decode_job:decode_worker(json.loads(Path(args.decode_job).read_text()));return
    if args.timing_job:timing_worker(json.loads(Path(args.timing_job).read_text()));return
    args.methods=args.methods.split(",")
    if (len(set(args.methods))!=len(args.methods) or not set(args.methods)<=set(base.METHODS)
            or args.rounds<1 or args.repeats<1 or args.warmup<0):p.error("invalid methods or repetitions")
    if not args.reference_run or not args.output:p.error("--reference-run and --output required")
    device=base.gpu()  # reject CPU/interpreter before making a GPU result directory
    import torch
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",arguments=vars(args),device_name=torch.cuda.get_device_name(device),automatic_merge=False)
    try:benchmark(args,output,report)
    except Exception:
        report["status"]="failed";report["fatal_error"]=traceback.format_exc();print(report["fatal_error"],file=sys.stderr)
    finally:
        write_json(output/"results.json",report);(output/"summary.md").write_text(summary(report))
        with zipfile.ZipFile(output/"share.zip","w",zipfile.ZIP_DEFLATED) as z:
            for file in sorted(output.iterdir()):
                if file.is_file() and file.suffix in (".json",".log",".md"):z.write(file,file.name)
        print("Reports:",output/"share.zip",flush=True)
    if report["status"]!="completed":raise SystemExit(1)


if __name__=="__main__":main()
