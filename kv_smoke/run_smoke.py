"""CPU smoke test on a pinned, pretrained BF16 causal language model.

No GPU kernel, production cache integration, E8 codec, or learned predictor is
implemented here. See README.md for the exact claims and accounting.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, pathlib, platform, time, zlib
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from codec import bf16_words_to_float32
from cache_container import encode_cache, decode_cache, decode_cache_tile, cache_shape
from prompts import HELDOUT_PROMPTS, build_input_ids, fixture_manifest
from download_model import MODEL_ID, REVISION

METHODS=["raw","zero_coset","lattice2_coset","lattice4_coset",
         "lattice2_sign_coset","exponent"]

def clock():
    return time.perf_counter()

def cache_matrices(cache):
    result=[]
    for k,v in cache.to_legacy_cache():
        pair=[]
        for t in (k,v):
            if t.dtype != torch.bfloat16:
                raise AssertionError("Native BF16 cache required")
            matrix=t[0].permute(1,0,2).contiguous().reshape(t.shape[2],-1)
            pair.append(matrix.view(torch.uint16).cpu().numpy().copy())
        result.append(tuple(pair))
    return result

def tensor_cache(matrices, config):
    cache=DynamicCache(config=config)
    h=config.num_key_value_heads
    d=config.hidden_size//config.num_attention_heads
    for layer,(k,v) in enumerate(matrices):
        pair=[]
        for w in (k,v):
            t=torch.from_numpy(np.ascontiguousarray(w)).view(torch.bfloat16)
            pair.append(t.reshape(w.shape[0],h,d).permute(1,0,2).unsqueeze(0).contiguous())
        cache.update(pair[0],pair[1],layer)
    return cache

def raw_bytes(matrices):
    return b"".join(a.astype("<u2",copy=False).tobytes() for pair in matrices for a in pair)

def equal_matrices(a,b):
    return len(a)==len(b) and all(np.array_equal(x,y)
        for pa,pb in zip(a,b) for x,y in zip(pa,pb))

@torch.inference_mode()
def step(model, token, matrices):
    n=matrices[0][0].shape[0]
    return model(input_ids=torch.tensor([[token]],dtype=torch.long),
                 attention_mask=torch.ones((1,n+1),dtype=torch.long),
                 past_key_values=tensor_cache(matrices,model.config),
                 use_cache=True,logits_to_keep=1)

def tiled_attention_check(blob, original, config):
    """Seeded synthetic queries, real K/V. Online softmax; CPU NumPy only."""
    shape=cache_shape(blob)
    h=config.num_key_value_heads
    qh=config.num_attention_heads
    d=config.hidden_size//qh
    head_map=np.arange(qh)//(qh//h)
    rng=np.random.default_rng(81)
    checks=[]
    for layer in sorted({0,len(original)//2,len(original)-1}):
        q=rng.standard_normal((qh,d),dtype=np.float32)
        k=bf16_words_to_float32(original[layer][0]).reshape(-1,h,d)[:,head_map]
        v=bf16_words_to_float32(original[layer][1]).reshape(-1,h,d)[:,head_map]
        scores=np.einsum("hd,thd->ht",q,k)/np.float32(math.sqrt(d))
        p=np.exp(scores-scores.max(axis=1,keepdims=True))
        p/=p.sum(axis=1,keepdims=True)
        ref=np.einsum("ht,thd->hd",p,v)
        m=np.full(qh,-np.inf,dtype=np.float32)
        denom=np.zeros(qh,dtype=np.float32)
        accum=np.zeros((qh,d),dtype=np.float32)
        max_words=0
        for tile in range(shape["ntiles"]):
            kw,vw,start=decode_cache_tile(blob,layer,tile)
            max_words=max(max_words,kw.nbytes+vw.nbytes)
            kt=bf16_words_to_float32(kw).reshape(-1,h,d)[:,head_map]
            vt=bf16_words_to_float32(vw).reshape(-1,h,d)[:,head_map]
            st=np.einsum("hd,thd->ht",q,kt)/np.float32(math.sqrt(d))
            next_m=np.maximum(m,st.max(axis=1))
            alpha=np.exp(m-next_m)
            local=np.exp(st-next_m[:,None])
            denom=denom*alpha+local.sum(axis=1)
            accum=accum*alpha[:,None]+np.einsum("ht,thd->hd",local,vt)
            m=next_m
        out=accum/denom[:,None]
        error=float(np.max(np.abs(ref-out)))
        if not np.allclose(ref,out,rtol=2e-5,atol=2e-5):
            raise AssertionError(f"Tiled attention mismatch {error}")
        checks.append({"layer":layer,"max_abs_error":error,
                       "largest_decoded_KV_word_tile_bytes":max_words,
                       "note":"Excludes encoded resident state, FP32 conversions, score buffers, query/output accumulators, Python overhead, and model state."})
    return checks

@torch.inference_mode()
def generation_check(model,tokenizer,original,blob,first_token,nsteps):
    """Re-decode prefix every step, append at most nsteps raw recent KV rows."""
    reference=original
    recent=[(np.empty((0,k.shape[1]),dtype=np.uint16),
             np.empty((0,v.shape[1]),dtype=np.uint16)) for k,v in original]
    token=first_token
    tokens=[]
    errors=[]
    rebuild_seconds=0.0
    for _ in range(nsteps):
        reference_out=step(model,token,reference)
        start=clock()
        reconstructed=decode_cache(blob)
        rebuilt=[(np.concatenate((k,rk)),np.concatenate((v,rv)))
                 for (k,v),(rk,rv) in zip(reconstructed,recent)]
        rebuild_seconds+=clock()-start
        test_out=step(model,token,rebuilt)
        error=float(torch.max(torch.abs(reference_out.logits.float()-test_out.logits.float())))
        errors.append(error)
        if not torch.equal(reference_out.logits,test_out.logits):
            raise AssertionError(f"Generation logits differ by {error}")
        next_ref=int(reference_out.logits[0,-1].argmax())
        next_test=int(test_out.logits[0,-1].argmax())
        if next_ref!=next_test:
            raise AssertionError("Generation tokens differ")
        tokens.append(next_ref)
        reference=cache_matrices(reference_out.past_key_values)
        test_state=cache_matrices(test_out.past_key_values)
        if not equal_matrices(reference,test_state):
            raise AssertionError("Appended KV differs")
        recent=[(np.concatenate((rk,k[-1:])),np.concatenate((rv,v[-1:])))
                for (rk,rv),(k,v) in zip(recent,test_state)]
        token=next_ref
    return {"steps":nsteps,"all_step_logits_bitwise_equal":True,
            "max_abs_logit_difference":max(errors,default=0.0),
            "token_ids":tokens,"decoded_text":tokenizer.decode(tokens),
            "prefix_reconstruction_seconds_total":rebuild_seconds,
            "recent_raw_KV_bytes_at_end":sum(x.nbytes for pair in recent for x in pair),
            "implementation":"CPU full-prefix restore each step plus bounded recent raw buffer; not a fused GPU or production serving path."}

def aggregate(cases):
    totals={}
    for method in METHODS+["zlib_level1"]:
        rows=[case["codecs"][method] for case in cases]
        total_raw=sum(r["raw_payload_bytes"] for r in rows)
        total_encoded=sum(r["serialized_bytes"] for r in rows)
        totals[method]={"raw_payload_bytes":total_raw,"serialized_bytes":total_encoded,
            "ratio_vs_raw_payload":total_raw/total_encoded,
            "space_saving_percent":100*(1-total_encoded/total_raw),
            "all_roundtrips_exact":all(r["roundtrip_exact"] for r in rows),
            "encode_seconds_total":sum(r["encode_seconds"] for r in rows),
            "decode_seconds_total":sum(r["decode_seconds"] for r in rows)}
        for key in ["mode_counts","candidate_correction_width_histogram",
                    "accepted_correction_width_histogram","fallback_counts","component_bytes"]:
            counts=collections.Counter()
            for r in rows: counts.update(r.get(key,{}))
            totals[method][key]=dict(counts)
    return totals

def report_text(result):
    lines=["# KV reconstruction smoke test",
      "",
      "This is a CPU mechanism test on a real pretrained small language model. It is not a GPU serving benchmark.",
      "",
      f"Model: **{MODEL_ID}**, revision **{REVISION}**.",
      f"Runtime: Python {platform.python_version()}, PyTorch {torch.__version__}, Transformers {transformers.__version__}; native BF16 model and KV, CPU, {result['runtime']['threads']} threads.",
      "",
      "## What ran",
      "",
      f"- {len(result['cases'])} synthetic held-out prompt/length cases covering prose, Python code, and structured retrieval.",
      f"- Context lengths: {result['configuration']['lengths']} tokens.",
      "- Exact local records with real bit packing, raw fallbacks, matrix/group headers, and a 64-bit temporal-tile offset index.",
      "- A scalar integer-grid lattice control, a sign-preserving variant, an exact exponent-packing baseline, and a trivial-predictor control.",
      "- No E8 lattice, neural learned predictor, GPU kernel, or high-concurrency service was tested.",
      "- No predictor/codebook calibration was performed: these controls compute small per-group state from the prefill KV and store it in each record.",
      "",
      "## Complete serialized sizes",
      "",
      "Ratio is raw BF16 payload bytes divided by all encoded bytes. A ratio below 1 means expansion.",
      "",
      "| Method | Raw MiB | Encoded MiB | Ratio | Space saving | Exact |",
      "|---|---:|---:|---:|---:|---|"]
    for method,row in result["aggregate"].items():
        lines.append(f"| {method} | {row['raw_payload_bytes']/2**20:.3f} | {row['serialized_bytes']/2**20:.3f} | {row['ratio_vs_raw_payload']:.4f}x | {row['space_saving_percent']:.2f}% | {row['all_roundtrips_exact']} |")
    lines+=["","The raw codec row includes container overhead; the raw-payload denominator has none. Zlib is a whole-cache CPU storage control without the local temporal index and is not an equivalent random-access execution format.",
      "",
      "## Correctness and execution checks",""]
    for case in result["cases"]:
        if "next_step_checks" in case:
            lines.append(f"- {case['name']} at {case['tokens']} tokens: "+", ".join(f"{method}: max logit error {v['max_abs_logit_difference']:.3g}" for method,v in case["next_step_checks"].items())+".")
        if "generation" in case:
            g=case["generation"]
            lines.append(f"- Repeated reconstruction generation ({case['name']}, {case['tokens']} tokens, {case['generation_method']}): {g['steps']} steps; exact logits and matching tokens; maximum logit difference {g['max_abs_logit_difference']:.3g}.")
        if "tiled_attention" in case:
            err=max(x["max_abs_error"] for x in case["tiled_attention"])
            words=max(x["largest_decoded_KV_word_tile_bytes"] for x in case["tiled_attention"])
            lines.append(f"- CPU online tiled attention ({case['name']}, {case['tokens']}): three layers, seeded synthetic queries and real KV; max absolute difference {err:.3g}; largest reconstructed K+V word tile {words:,} bytes. This word-tile size is not total workspace or a GPU memory measurement.")
    lines+=["","## Timing interpretation","",
      "Single CPU invocations use perf_counter. Python/Numpy serialization is a reference implementation. The measurements include predictor generation, bit packing, and record parsing but do not establish GPU throughput, TPOT, admission gains, or a performance advantage. Data capture, model loading, and source validation are separate from codec timings.",
      "",
      "| Method | Total encode seconds | Total decode seconds |",
      "|---|---:|---:|"]
    for method,row in result["aggregate"].items():
        lines.append(f"| {method} | {row['encode_seconds_total']:.4f} | {row['decode_seconds_total']:.4f} |")
    lines+=["","## What this does and does not establish","",
      "The results establish exact recovery for tested KV words, actual serialized-size accounting, independent temporal-tile reconstruction, and limited model-integration correctness. They show whether these specific simple predictors save space on these particular fixtures.",
      "",
      "They do not establish the best possible reconstruction record, representative workload compression, Qwen applicability, a learned or E8 codec result, bounded GPU execution, production cache growth/promotion, long-context serving SLOs, or high-concurrency admission benefit.",
      "",
      "Inspect results.json for per-case metrics, header/predictor/correction bytes, candidate versus accepted correction widths, fallback counts, checkpoint hashes, and runtime provenance. Code and fixtures are included for reproduction.",
      "",
      "## Sources","",
      "- [Official model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)",
      "- [Pinned checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/tree/"+REVISION+")",
      "- [Transformers cache API](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation)",
      "",
      "The reference design is kv_reconstruction_records_team_review.md. This report is a separate experiment, not a revision of the design's unproven performance claims.",
      ""]
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--output-dir",default="results")
    parser.add_argument("--lengths",default="128,512,2048")
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--group-size",type=int,default=64)
    parser.add_argument("--tile-tokens",type=int,default=16)
    parser.add_argument("--generation-steps",type=int,default=8)
    args=parser.parse_args()
    if args.generation_steps>args.tile_tokens:
        raise ValueError("This smoke test only implements a bounded recent buffer of one tile")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(7)
    np.random.seed(7)
    output=pathlib.Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    lengths=[int(x) for x in args.lengths.split(",")]
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    load_start=clock()
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation="sdpa").eval()
    result={"model":{"id":MODEL_ID,"revision":REVISION,"parameter_count":sum(p.numel() for p in model.parameters()),
             "config":model.config.to_dict(),"load_seconds":clock()-load_start},
       "runtime":{"python":platform.python_version(),"torch":torch.__version__,
          "transformers":transformers.__version__,"numpy":np.__version__,"cuda_available":torch.cuda.is_available(),
          "device":"cpu","dtype":"bfloat16","threads":args.threads,"machine":platform.machine(),
          "cpu":next((line.split(":",1)[1].strip() for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),"unknown")},
       "configuration":{"lengths":lengths,"group_size":args.group_size,"tile_tokens":args.tile_tokens,
             "generation_steps":args.generation_steps,"timings":"single CPU invocations; no GPU speed extrapolation"},
       "fixture_provenance":fixture_manifest(),"cases":[]}
    manifest=pathlib.Path(args.model_dir)/"download_manifest.json"
    if manifest.exists():
        result["checkpoint_files"]=json.loads(manifest.read_text())
    for length in lengths:
        for fixture in HELDOUT_PROMPTS:
            name=fixture["name"]
            ids=build_input_ids(tokenizer,fixture,length)
            assert len(ids)==length
            print(f"prefill {name} length={length}",flush=True)
            with torch.inference_mode():
                start=clock()
                prefill=model(torch.tensor([ids],dtype=torch.long),use_cache=True,logits_to_keep=1)
                prefill_seconds=clock()-start
                first=int(prefill.logits[0,-1].argmax())
                matrices=cache_matrices(prefill.past_key_values)
            payload=raw_bytes(matrices)
            case={"name":name,"kind":fixture["kind"],"tokens":length,
                  "input_token_ids":ids,"prefill_seconds":prefill_seconds,
                  "raw_payload_bytes":len(payload),"raw_sha256":hashlib.sha256(payload).hexdigest(),"codecs":{}}
            encoded_by_method={}
            # Check all six codec/model paths on the smallest prose case.
            check_all=length==lengths[0] and fixture is HELDOUT_PROMPTS[0]
            reference_logits=step(model,first,matrices).logits.detach().clone() if check_all else None
            if check_all: case["next_step_checks"]={}
            for method in METHODS:
                start=clock()
                encoded=encode_cache(matrices,method,args.group_size,args.tile_tokens)
                encode_seconds=clock()-start
                start=clock()
                recovered=decode_cache(encoded.data)
                decode_seconds=clock()-start
                exact=equal_matrices(matrices,recovered)
                if not exact: raise AssertionError("KV mismatch "+method)
                stats=dict(encoded.stats)
                stats.update(encode_seconds=encode_seconds,decode_seconds=decode_seconds,roundtrip_exact=exact,
                             decoded_word_count=len(payload)//2,
                             encoded_sha256=hashlib.sha256(encoded.data).hexdigest())
                case["codecs"][method]=stats
                encoded_by_method[method]=encoded
                if check_all:
                    observed=step(model,first,recovered).logits
                    error=float(torch.max(torch.abs(observed.float()-reference_logits.float())))
                    same=bool(torch.equal(observed,reference_logits))
                    if not same: raise AssertionError("Next-step model mismatch "+method)
                    case["next_step_checks"][method]={"logits_bitwise_equal":same,"max_abs_logit_difference":error}
                print(f"  {method}: {len(encoded.data)/2**20:.3f}MiB ratio={len(payload)/len(encoded.data):.4f}, exact={exact}",flush=True)
            start=clock()
            compressed=zlib.compress(payload,level=1)
            encoded_time=clock()-start
            start=clock()
            restored=zlib.decompress(compressed)
            decoded_time=clock()-start
            assert restored==payload
            case["codecs"]["zlib_level1"]={"raw_payload_bytes":len(payload),"serialized_bytes":len(compressed),
                "ratio_vs_raw_payload":len(payload)/len(compressed),"encode_seconds":encoded_time,
                "decode_seconds":decoded_time,"roundtrip_exact":True}
            if length==lengths[0] and fixture is HELDOUT_PROMPTS[0]:
                sample=encoded_by_method["lattice2_sign_coset"]
                (output/"sample_encoded_cache.bin").write_bytes(sample.data)
                np.savez_compressed(output/"sample_raw_cache.npz",
                    **{f"layer_{i:02d}_{kind}":matrix for i,pair in enumerate(matrices) for kind,matrix in zip(("K","V"),pair)})
            if length==512 and fixture is HELDOUT_PROMPTS[0]:
                method="lattice2_sign_coset"
                blob=encoded_by_method[method].data
                case["generation_method"]=method
                case["generation"]=generation_check(model,tokenizer,matrices,blob,first,args.generation_steps)
                case["tiled_attention"]=tiled_attention_check(blob,matrices,model.config)
                print("  generation and tiled-attention checks passed",flush=True)
            result["cases"].append(case)
            result["aggregate"]=aggregate(result["cases"])
            (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
            (output/"smoke_test_report.md").write_text(report_text(result))
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in pathlib.Path(__file__).parent.glob("*.py")}
    result["source_sha256"]=sources
    result["aggregate"]=aggregate(result["cases"])
    (output/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    (output/"smoke_test_report.md").write_text(report_text(result))
    print(json.dumps({"aggregate":result["aggregate"],"cases":len(result["cases"])},indent=2),flush=True)

if __name__=="__main__":
    main()

