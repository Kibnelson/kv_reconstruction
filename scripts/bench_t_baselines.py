#!/usr/bin/env python3
"""Matched native GPU codecs on frozen T captures; never fits T or edits old runs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_head_spread import check_base, file_sha, load_words, measure, raw_digest, select_device, Window, write_json

METHODS = ("raw_bf16", "t_fast", "t_hbh", "splitzip", "exans")
FROZEN = {"t_fast": ("dense", "legacy_global"), "t_hbh": ("hbh", "learned_selected")}


def invoke(flag, job, result, logfile):
    """A native codec crash must not erase the rest of the comparison."""
    with Path(logfile).open("w") as log:
        p = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), flag, str(job)],
                           stdout=log, stderr=subprocess.STDOUT)
    if p.returncode or not Path(result).is_file():
        tail = Path(logfile).read_text(errors="replace")[-3500:]
        raise RuntimeError(f"worker exited {p.returncode}; {logfile}\n{tail}")
    return json.loads(Path(result).read_text())


def gpu():
    import torch
    device = select_device("cuda", False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    return device


def digest_words(words):
    return hashlib.sha256(words.detach().contiguous().cpu().numpy().astype("<i2", copy=False).tobytes()).hexdigest()


def checked_codec(job, device):
    from kv_compare.native_compare import make
    if file_sha(job["state"]) != job["state_sha256"]:
        raise ValueError("shared state changed")
    return make(job["method"], job["state"], job["shape"], device)


def packet_from(codec, method, encoded, job, expected):
    from kv_compare.native_compare import pack_parts
    if method in FROZEN:
        return codec.pack(encoded, expected)
    parts, aux = codec.parts(encoded)
    return pack_parts(method, job["shape"], job["state_sha256"], expected, parts, aux)


def packet_to(codec, method, packet, job, device):
    from kv_compare.native_compare import unpack_parts
    if method in FROZEN:
        encoded, meta = codec.receive_packet(packet)
        return encoded, meta["words_sha256"]
    parts, meta = unpack_parts(packet, method, job["shape"], job["state_sha256"], device)
    return codec.receive(parts, meta), meta["words_sha256"]


def decode_worker(job):
    device = gpu()
    codec = checked_codec(job, device)
    try:
        packet = Path(job["packet"]).read_bytes()
        begin = time.perf_counter()
        encoded, expected = packet_to(codec, job["method"], packet, job, device)
        result = codec.decode(encoded)
        actual = digest_words(result)
        if actual != expected or actual != job["expected_sha256"]:
            raise AssertionError("fresh packet-only receiver changed BF16 words")
        write_json(job["result"], dict(bit_exact=True, source_words_sha256=actual,
            packet_sha256=hashlib.sha256(packet).hexdigest(),
            restore_decode_and_hash_ms=1000*(time.perf_counter()-begin), capture_path_provided=False))
    finally:
        codec.close()


def native_preflight(method, state, state_sha, device):
    """Exercise native fallback and full-word patterns before trusting timed data."""
    import torch
    from kv_compare.native_compare import make
    if method == "raw_bf16":
        return {"raw_copy": True}
    if method in FROZEN:
        from kv_compare.fast_dense_checks import check
        return check(device, all_patterns=True)
    if method == "splitzip":
        samples = [("all_bf16_words", torch.arange(65536, dtype=torch.int64).to(torch.int16))]
    else:
        from kv_compare.exans_official import BLOCK_WORDS
        gen = torch.Generator().manual_seed(831)
        samples = [("compressible_block_and_tail", torch.zeros(BLOCK_WORDS+137, dtype=torch.int16)),
                   ("random_blocks_and_tail", torch.randint(-32768,32768,(2*BLOCK_WORDS+137,),dtype=torch.int16,generator=gen))]
    checks = []
    for name, cpu in samples:
        words = cpu.to(device)
        job = dict(method=method, state=state, state_sha256=state_sha, shape=list(words.shape))
        codec = make(method, state, words.shape, device)
        try:
            encoded = codec.encode(words)
            if not torch.equal(codec.decode(encoded), words):
                raise AssertionError("native preflight reconstruction failed: " + name)
            packet, cost = packet_from(codec, method, encoded, job, digest_words(words))
            restored, _ = packet_to(codec, method, packet, job, device)
            if not torch.equal(codec.decode(restored), words):
                raise AssertionError("native preflight packet reconstruction failed: " + name)
            record = dict(case=name, bit_exact=True, packet_bytes=len(packet), cost=cost)
            if method == "exans":
                _, aux = codec.parts(encoded)
                record.update(aux)
                if name.startswith("compressible") and not aux["fit_blocks"]:
                    raise AssertionError("compressible ExANS test did not exercise fitted decode")
                if name.startswith("random") and not aux["raw_blocks"]:
                    raise AssertionError("random ExANS test did not exercise raw fallback")
            checks.append(record)
        finally:
            codec.close()
        del codec, words, encoded, restored, packet
        gc.collect()
        torch.cuda.empty_cache()
    return checks


def prepare_worker(job):
    import torch
    from kv_compare.native_compare import load_splitzip, pinned_checkout
    device = gpu()
    if job["method"] == "splitzip":
        from kv_compare.predict import Doc
        from kv_compare.mixed import SPLITZIP_COMMIT
        module = load_splitzip(job["source"])
        doc = Window(Doc(job["captures"], job["train_doc"]), 0, job["train_tokens"])
        words = load_words(doc, job["layers"], device)
        source_hash = digest_words(words)
        if source_hash != job["calibration_sha256"]:
            raise ValueError("SplitZip calibration capture changed")
        codec = module.ChunkLocalSplitZipGPU(device=str(device), chunk_size=1024)
        coverage = codec.calibrate(words.view(torch.bfloat16))
        torch.save(dict(method="splitzip", source=job["source"], commit=SPLITZIP_COMMIT,
            calibration=dict(document=job["train_doc"], source_words_sha256=source_hash, top16_coverage=coverage,
                             test_used_for_calibration=False),
            tables={k:getattr(codec,k).cpu() for k in ("enc_lut","enc_lut_marked","dec_lut","common_lut")}), job["state"])
        info = dict(commit=SPLITZIP_COMMIT, source=job["source"], calibration_top16_coverage=coverage,
                    calibration_document=job["train_doc"], test_used_for_calibration=False)
    else:
        from kv_compare import exans_official as ex
        pinned_checkout(job["source"], ex.EXANS_COMMIT, "crates/openlake_kv_client/cuda")
        if not Path(job["cuda_home"], "bin", "nvcc").is_file():
            raise ValueError("existing CUDA toolkit not found; set CUDA_HOME without changing the T environment")
        info = ex.build(Path(job["work"]), job["source"], job["cuda_home"])
        write_json(job["state"], dict(method="exans", build=info, library_sha256=file_sha(info["library"])))
        shutil.copyfile(Path(job["work"])/"official_test.log",Path(job["state"]).parent/"exans_authors_test.log")
        shutil.copyfile(Path(job["work"])/"build_fix.diff",Path(job["state"]).parent/"exans_build_fix.txt")
    write_json(job["result"], info)


def timing_worker(job):
    import torch
    import triton
    from kv_compare.predict import Doc
    from kv_compare.head_spread_wire import tensors_bytes
    device = gpu()
    check_base()
    if job["round"] == 0:
        preflight = native_preflight(job["method"], job["state"], job["state_sha256"], device)
    else:
        preflight = "passed in round 0"
    torch.set_num_threads(2)
    free_before = torch.cuda.mem_get_info(device)[0]
    begin = time.perf_counter()
    codec = checked_codec(job, device)
    torch.cuda.synchronize(device)
    setup = dict(ms=1000*(time.perf_counter()-begin),
        device_free_delta_bytes=free_before-torch.cuda.mem_get_info(device)[0],
        shared_tensor_bytes=codec.shared_bytes(), state_file_bytes=Path(job["state"]).stat().st_size,
        allocation=codec.allocation, opaque_native_workspace_in_torch_stats=False if job["method"]=="exans" else None)
    setup["preallocated_record_arena_bytes"]=tensors_bytes([getattr(codec,"arena",None),getattr(codec,"records",None)])
    setup["preallocated_output_bytes"]=tensors_bytes([getattr(codec,"out",None)])
    setup["preallocated_T_scratch_bytes"]=tensors_bytes(getattr(codec,"bufs",{}).values())
    rows, pending_proofs = [], []
    try:
        for case in job["cases"]:
            print(job["method"], "round", job["round"]+1, case["name"], flush=True)
            doc = Window(Doc(job["captures"], case["name"]), 0, job["shape"][1])
            begin = time.perf_counter()
            words = load_words(doc, job["shape"][0], device)
            torch.cuda.synchronize(device)
            staging_ms = 1000*(time.perf_counter()-begin)
            if list(words.shape)!=job["shape"] or digest_words(words)!=case["source_words_sha256"]:
                raise ValueError("TEST words differ from frozen reference: " + case["name"])
            # Old frozen frames use zero-initialized arenas. Alignment bytes from
            # a previous window must never leak into this window's packet.
            if job["method"] in FROZEN:codec.arena.zero_()
            if job["method"] == "exans":codec.records.zero_()
            encoded = codec.encode(words)
            if not torch.equal(codec.decode(encoded), words):
                raise AssertionError("native warm reconstruction changed BF16 words")
            encoded, enc = measure(lambda:codec.encode(words), device, job["warmup"], job["repeats"])
            decoded, dec = measure(lambda:codec.decode(encoded), device, job["warmup"], job["repeats"])
            del decoded
            def pair():
                return codec.decode(codec.encode(words))
            result, both = measure(pair, device, job["warmup"], job["repeats"])
            if not torch.equal(result, words):
                raise AssertionError("timed pair changed BF16 words")
            # Re-encode: an in-place native arena may have been overwritten by pair().
            encoded = codec.encode(words)
            torch.cuda.synchronize(device)
            begin = time.perf_counter()
            packet, cost = packet_from(codec, job["method"], encoded, job, case["source_words_sha256"])
            packing_ms = 1000*(time.perf_counter()-begin)
            packet_sha = hashlib.sha256(packet).hexdigest()
            if job["method"] in FROZEN and packet_sha!=case["packet_sha256"]:
                raise AssertionError("T packet differs from frozen reference")
            begin = time.perf_counter()
            received, expected = packet_to(codec, job["method"], packet, job, device)
            torch.cuda.synchronize(device)
            restore_ms = 1000*(time.perf_counter()-begin)
            if expected!=case["source_words_sha256"] or not torch.equal(codec.decode(received), words):
                raise AssertionError("complete transfer packet changed BF16 words")
            proof = None
            if job["round"] == 0:
                # Only these inputs reach the new receiver process; no capture or training data path.
                prefix = Path(job["output"]) / (job["method"]+"_"+case["name"])
                packet_path = prefix.with_suffix(".packet")
                packet_path.write_bytes(packet)
                proof_job = dict(method=job["method"], state=job["state"], state_sha256=job["state_sha256"],
                    shape=job["shape"], packet=str(packet_path), expected_sha256=case["source_words_sha256"],
                    result=str(prefix.with_suffix(".proof.json")))
                proof_path = prefix.with_suffix(".receiver.json")
                write_json(proof_path, proof_job)
                pending_proofs.append((len(rows),proof_path,proof_job["result"],prefix.with_suffix(".receiver.log"),packet_path))
            row = dict(method=job["method"], round=job["round"], case=case["name"], shape=job["shape"],
                values=words.numel(), raw_bytes=words.numel()*2, packet_bytes=len(packet), packet_sha256=packet_sha,
                source_words_sha256=expected, bit_exact=True, frozen_T_packet_identical=True if job["method"] in FROZEN else None,
                cost=cost, encode=enc, decode=dec, pair=both, input_staging_ms=staging_ms,
                packet_packing_and_D2H_ms=packing_ms, packet_expansion_and_H2D_ms=restore_ms, packet_only_proof=proof)
            rows.append(row)
            del words, result, encoded, received, packet
            gc.collect()
    finally:
        codec.close()
    # Close sender workspaces before independent receivers allocate theirs.
    # Their correctness must not depend on two full native contexts fitting VRAM.
    del codec
    gc.collect()
    torch.cuda.empty_cache()
    for index,path,result,logfile,packet_path in pending_proofs:
        rows[index]["packet_only_proof"]=invoke("--decode-job",path,result,logfile)
        packet_path.unlink()
    write_json(job["result"], dict(rows=rows, preflight=preflight, setup=setup,
        device_name=torch.cuda.get_device_name(device), torch=torch.__version__, triton=triton.__version__))


def aggregate(rows, methods, cases, rounds):
    result = []
    for method in methods:
        rr = [r for r in rows if r["method"]==method]
        complete = {(r["case"],r["round"]) for r in rr}=={(c,i) for c in cases for i in range(rounds)}
        if not rr:
            result.append(dict(method=method, complete=False))
            continue
        first = {r["case"]:r for r in rr}
        packet_bytes = sum(r["packet_bytes"] for r in first.values())
        values = sum(r["values"] for r in first.values())
        out = dict(method=method, complete=complete, bit_exact=all(r["bit_exact"] for r in rr),
            case_count=len(first), rounds=rounds, packet_bytes=packet_bytes, values=values,
            bpv=8*packet_bytes/values, saved_percent=100*(1-packet_bytes/(2*values)),
            packet_size_stable=all(len({r["packet_bytes"] for r in rr if r["case"]==c})==1 for c in first))
        for part in ("encode","decode","pair"):
            for clock in ("cuda","wall"):
                out[part+"_"+clock+"_ms"] = statistics.mean(statistics.median(r[part][clock+"_median_ms"]
                    for r in rr if r["case"]==case) for case in first)
        out["ideal_serial_link_ms"]={str(rate):out["pair_wall_ms"]+packet_bytes/len(first)*8/(rate*1e6)
                                      for rate in (3,10,25,50,100,200,400)}
        result.append(out)
    return result


def summary(report):
    lines = ["# Frozen T versus native exact GPU baselines", "",
        f"Status: **{report['status']}**. Full reference coverage: **{report.get('full_reference_coverage',False)}**. No automatic merge.", "",
        "| Method | BF16 saved | bpv | Encode CUDA ms | Decode CUDA ms | Pair CUDA ms | Pair wall ms | Complete |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in report.get("aggregate",[]):
        if "bpv" not in r:
            lines.append(f"| {r['method']} | unavailable | | | | | | False |")
        else:
            lines.append(f"| {r['method']} | {r['saved_percent']:.4f}% | {r['bpv']:.6f} | {r['encode_cuda_ms']:.3f} | {r['decode_cuda_ms']:.3f} | {r['pair_cuda_ms']:.3f} | {r['pair_wall_ms']:.3f} | {r['complete']} |")
    lines += ["", "CUDA and wall times use warm native API calls with resident inputs; medians across samples and rounds per window, then mean across windows. Pair is a direct sequential encode+decode measurement. Authors' host synchronizations remain included. Each method retains its native allocation behavior, recorded in results.json.", "",
        "Complete packet bytes include headers, indexes, every raw fallback block and the partial-block tail. Original T packets must match the earlier experiment byte for byte. Decode is checked against every BF16 word, including in a fresh process with only packet and shared state. Shared state, native workspace, input staging, packet packing/D2H and expansion/H2D are recorded separately. Setup and host paths are not included in warm codec times.", "",
        "Raw BF16 is a resident reference copy; network transport is not measured. Ideal link times in results.json use wall codec pair plus packet bytes / bandwidth, with shared state resident and no overlap. SplitZip and ExANS are authors' pinned GPU implementations. ExANS adds paid raw handling for rejected and partial blocks; its existing one-line build fix is recorded.", "",
        "AQUA-KV is a separate lossy quality stage requiring its authors' environment and actual packed buffers; this exact-BF16 table makes no AQUA-KV performance or quality claim."]
    if report.get("errors"):
        lines += ["", "Errors (available rows are partial results):"] + ["- "+str(e) for e in report["errors"]]
    if report.get("fatal_error"):
        lines += ["", "Run failed:", "```", report["fatal_error"], "```"]
    return "\n".join(lines)+"\n"


def benchmark(args, output, report):
    from kv_compare.predict import Doc
    reference_dir = Path(args.reference_run).resolve()
    reference = json.loads((reference_dir/"results.json").read_text())
    if reference.get("status")!="completed" or not reference.get("gpu_executed") or reference.get("test_used_for_selection"):
        raise ValueError("a completed frozen GPU reference is required")
    if json.loads((reference_dir/"selection_frozen.json").read_text())!=reference["selected_on_dev"]:
        raise ValueError("reference selection changed")
    report["pinned_T_sources"] = check_base()
    for r in reference["experiment_sources"]:
        if file_sha(ROOT/r["path"])!=r["sha256"]:
            raise ValueError("frozen experiment source changed: "+r["path"])
    captures = Path(args.captures or reference["arguments"]["captures"]).resolve()
    cases = args.cases.split(",") if args.cases else reference["split"]["test_docs"]
    if len(set(cases))!=len(cases) or not set(cases)<=set(reference["split"]["test_docs"]):
        raise ValueError("use distinct frozen reference windows")
    by_key = {(r["mode"],r["variant"],r["case"]):r for r in reference["test_rows"]}
    canonical = [by_key[("dense","legacy_global",case)] for case in cases]
    shape = [reference["layers"],reference["split"]["test_tokens_each"],2*reference["model_identity"]["kv_heads"]*reference["model_identity"]["head_dim"]]
    train = reference["split"]["train_docs"][0]
    train_tokens = reference["split"]["train_tokens_each"][0]
    expected = {Path(r["path"]).name:r["sha256"] for r in reference["inputs"]}
    names = ["meta.json"]+[c+"_"+s for c in cases+([train] if "splitzip" in args.methods else []) for s in ("K_bits.npy","V_bits.npy","ids.npy")]
    inputs = []
    for name in names:
        path = captures/name
        h = file_sha(path)
        if h!=expected[name]:raise ValueError("capture changed: "+name)
        inputs.append(dict(path=str(path),sha256=h,bytes=path.stat().st_size))
    report.update(reference_results_sha256=file_sha(reference_dir/"results.json"),
        model=reference["model_identity"], cases=cases, shape=shape, inputs=inputs,
        full_reference_coverage=set(cases)==set(reference["split"]["test_docs"]) and reference["full_model"],
        GPU_requested=True, gpu_executed=False, test_used_for_selection=False,
        sources_sha256={str(p.relative_to(ROOT)):file_sha(p) for p in (Path(__file__).resolve(),
            ROOT/"kv_compare/native_compare.py",ROOT/"kv_compare/fast_dense_encode.py",
            ROOT/"kv_compare/exans_official.py",ROOT/"kv_compare/mixed.py")},
        states={}, preparation={}, workers=[], errors=[], rows=[])
    for method in args.methods:
        state = reference_dir/"states"/("_".join(FROZEN[method])+".pt") if method in FROZEN else output/(method+"_state."+("pt" if method=="splitzip" else "json"))
        try:
            if method=="raw_bf16":write_json(state, dict(method=method))
            elif method not in FROZEN:
                prep = dict(method=method,state=str(state),result=str(output/(method+"_prepare.json")),
                    source=str(Path(args.splitzip_source if method=="splitzip" else args.exans_source).resolve()),
                    cuda_home=args.cuda_home,work=str(output/"exans_build"), captures=str(captures),layers=shape[0],
                    train_doc=train,train_tokens=train_tokens)
                if method=="splitzip":prep["calibration_sha256"]=raw_digest(Window(Doc(captures,train),0,train_tokens),shape[0])
                path=output/(method+"_prepare_job.json");write_json(path,prep)
                print("Preparing pinned native",method,flush=True)
                report["preparation"][method]=invoke("--prepare-job",path,prep["result"],output/(method+"_prepare.log"))
            report["states"][method]=dict(path=str(state),sha256=file_sha(state),bytes=state.stat().st_size)
        except Exception as e:
            report["errors"].append(dict(method=method,stage="prepare",error=str(e)))
            print("Unavailable:",method,str(e),flush=True)
    for round_index in range(args.rounds):
        order = args.methods[round_index%len(args.methods):]+args.methods[:round_index%len(args.methods)]
        for method in order:
            if method not in report["states"]:continue
            info=report["states"][method]
            cc=[]
            for case,r in zip(cases,canonical):
                item=dict(name=case,source_words_sha256=r["source_words_sha256"])
                if method in FROZEN:item["packet_sha256"]=by_key[(*FROZEN[method],case)]["packet_sha256"]
                cc.append(item)
            cc=cc[round_index%len(cc):]+cc[:round_index%len(cc)]
            prefix=output/(method+"_round"+str(round_index))
            job=dict(method=method,state=info["path"],state_sha256=info["sha256"],shape=shape,cases=cc,
                captures=str(captures),round=round_index,warmup=args.warmup,repeats=args.repeats,
                output=str(output),result=str(prefix.with_suffix(".json")))
            job_path=prefix.with_suffix(".job.json");write_json(job_path,job)
            print(f"GPU round {round_index+1}/{args.rounds}: {method}",flush=True)
            try:
                value=invoke("--timing-job",job_path,job["result"],prefix.with_suffix(".log"))
                report["rows"].extend(value.pop("rows"))
                report["workers"].append(dict(method=method,round=round_index,**value))
                report["gpu_executed"]=True
            except Exception as e:
                report["errors"].append(dict(method=method,round=round_index,error=str(e)))
                print("Failed:",method,str(e),flush=True)
            write_json(output/"results.json",report)
    report["aggregate"]=aggregate(report["rows"],args.methods,cases,args.rounds)
    report["status"]="completed" if not report["errors"] else "incomplete"
    report["full_comparison_validated"]=(report["status"]=="completed" and report["full_reference_coverage"]
        and args.rounds>=3 and set(args.methods)==set(METHODS)
        and all(r["complete"] and r["bit_exact"] and r["packet_size_stable"] for r in report["aggregate"]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("prepare-job","timing-job","decode-job"):parser.add_argument("--"+name)
    parser.add_argument("--reference-run")
    parser.add_argument("--output")
    parser.add_argument("--captures")
    parser.add_argument("--splitzip-source")
    parser.add_argument("--exans-source")
    parser.add_argument("--cuda-home",default=os.environ.get("CUDA_HOME",""))
    parser.add_argument("--methods",default=",".join(METHODS))
    parser.add_argument("--cases")
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--repeats",type=int,default=10)
    args=parser.parse_args()
    for attr,fn in (("prepare_job",prepare_worker),("timing_job",timing_worker),("decode_job",decode_worker)):
        if getattr(args,attr):
            fn(json.loads(Path(getattr(args,attr)).read_text()))
            return
    args.methods=args.methods.split(",")
    if (len(set(args.methods))!=len(args.methods) or not set(args.methods)<=set(METHODS)
            or args.rounds<1 or args.repeats<1 or args.warmup<0):parser.error("invalid methods or repetitions")
    if not args.reference_run or not args.output:parser.error("--reference-run and --output required")
    # Check CUDA before creating output; never turn CPU checks into GPU results.
    device=gpu()
    import torch
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status="running",automatic_merge=False,python=platform.python_version(),
        arguments=vars(args),device_name=torch.cuda.get_device_name(device))
    # Release the parent's context allocations before workers use the device.
    try:
        benchmark(args,output,report)
    except Exception:
        report["status"]="failed";report["fatal_error"]=traceback.format_exc()
        print(report["fatal_error"],file=sys.stderr)
    finally:
        write_json(output/"results.json",report)
        (output/"summary.md").write_text(summary(report))
        # No packets, models, state .pt or native .so files in the review archive.
        with zipfile.ZipFile(output/"share.zip","w",zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob("*")):
                if path.is_file() and "exans_build" not in path.parts and path.suffix in (".json",".md",".log",".txt"):
                    z.write(path,path.relative_to(output))
        print("Reports:",output/"share.zip",flush=True)
    if report["status"]!="completed":raise SystemExit(1)


if __name__=="__main__":main()
