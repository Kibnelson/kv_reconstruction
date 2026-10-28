#!/usr/bin/env python3
"""Lossless and lossy KV transfer codecs on one model's KV: ours against SplitZip, ExANS, AQUA-KV and KIVI.

Default model: unsloth/Llama-3.2-3B at revision d4446454d87d51aa42e1fb174f25acc5f8762331, the ungated copy of the
original BF16 Llama-3.2-3B weights that AQUA-KV's README points to. It is the same pinned checkpoint as
scripts/run_llama_official_a4500.py (branch experiment/llama32_official_gpu), so no Hugging Face login is needed and
an existing cache is reused.

    python scripts/bench_llama_compare.py --device cuda --output runs/llama32_3b_compare

Stages (all by default; --stages picks a subset, reusing earlier outputs in the same directory):
  capture    prefill nine documents (8,192 tokens each) and save BF16 KV words; calibration documents also
             save q_proj outputs for the attention input
  lossless   exact per-word costs: SplitZip-style, ExANS-style and our configurations (prev, block, nohint,
             fast) on the two test documents (whole documents and the four 256-token reference chunks)
  lossy      the first --prompt-tokens tokens of each test document coded by every lossy method; the model then
             processes the next --continuation-tokens tokens with that KV; perplexity change vs the captured exact
             KV (restoring it twice must agree exactly; the model's own prefill of the prompt is reported beside it)
  frontier   (optional) bits against compute: ~37 predictor variants (inputs, steps, reduced rank, 8-bit integer
             formats) fitted from one covariance per layer and scored like the lossless stage -> frontier.json
  gpu_speed  (optional, needs the GPU to itself) measured throughput: the predictor of five frontier variants in
             fp32, BF16 and exact 8-bit integer formats, and the Triton value-slice coder encoding and decoding
             real KV (bit-exact round trip) -> gpu_speed.json
  pool       (optional, needs the GPU to itself) GPU-resident KV pool: bringing a cached prefix back by
             recomputing it, copying raw KV from CPU memory, or decoding our compressed KV; the cost of adding it;
             tokens that fit -> pool.json
  format     (optional) bits of our one-step prediction inside SplitZip's kind of format: fixed-length codes,
             code book per value chosen by the prediction, no entropy coder (dense and --format-ranks products)
             -> format.json
  exans      (optional, needs nvcc, CMake and the nvCOMP SDK) the official ExANS codec built from the authors'
             source (one build fix, see kv_compare/exans_official.py) on the test documents: bits, bit-exact round
             trip, encode / decode speed -> exans.json
  report     summary.md

Document roles (same protocol as the SmolLM2 study): calibration = PEP 484 and six CPython modules, scales =
dataclasses.py, tests = PEP 8 (text) and argparse.py (code). Costs are computed -log2 P (the reference range
coder matched computed costs within 0.011 bits per value on SmolLM2); rates of the fixed-rate baselines include
their scale/offset overhead. GPU timings are not part of these stages.
"""
from __future__ import annotations

import argparse
import os
import json
import math
import signal
import traceback
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEXTS = ROOT / "benchmark_fixtures" / "llama_compare_texts"
DEFAULT_MODEL = "unsloth/Llama-3.2-3B"                          # ungated copy of the original BF16 weights
DEFAULT_REVISION = "d4446454d87d51aa42e1fb174f25acc5f8762331"  # same pin as scripts/run_llama_official_a4500.py
ROLES = {
    "train0": "pep-0484.rst", "train1": "code_typing.py.txt", "train2": "code_inspect.py.txt",
    "train3": "code_subprocess.py.txt", "train4": "code_tarfile.py.txt", "train5": "code_logging.py.txt",
    "tables": "code_dataclasses.py.txt", "test_text": "pep-0008.rst", "test_code": "code_argparse.py.txt",
}
TRAIN = [f"train{i}" for i in range(6)]
TESTS = ["test_text", "test_code"]


LOG_PATH = None


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def memory_note(device=None):
    """Peak host memory of this process and current GPU memory, for the run log."""
    import resource
    note = f"host peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20:.1f} GB"
    try:
        import torch
        if device is not None and torch.device(device).type == "cuda" and torch.cuda.is_available():
            note += (f"; GPU allocated {torch.cuda.memory_allocated(device) / 2 ** 30:.1f} GB, "
                     f"reserved {torch.cuda.memory_reserved(device) / 2 ** 30:.1f} GB")
    except ImportError:
        pass
    return note


STAGES = ("capture", "evalcap", "lossless", "lossy", "frontier", "gpu_speed", "pool", "format", "szcodec", "exans",
          "report")
BRANCH = "llama32-3b-lossy-lossless-compare"


def environment(args):
    """Versions, device and source commit, recorded with every run."""
    import platform
    import subprocess
    import torch
    import transformers
    info = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "transformers": transformers.__version__, "numpy": np.__version__,
            "device": args.device}
    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
    try:
        info["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        info["git_branch"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                                capture_output=True, text=True).stdout.strip())
    except OSError:
        info["git_commit"] = None
    return info


def stage_capture(args, out):
    import torch
    from kv_compare.capture import capture_documents, load_model, model_meta
    device = torch.device(args.device)
    tokenizer, model = load_model(args.model, args.revision, device)
    documents = {name: Path(args.texts) / ROLES[name] for name in ROLES}
    meta = model_meta(model, args.model, args.revision)
    queries = set(TRAIN) if not args.no_queries else set()
    return capture_documents(model, tokenizer, documents, out / "captures", device, args.tokens, queries, meta, log)


def stage_evalcap(args, out):
    """Capture evaluation documents for a larger lossy-quality test: --eval-docs windows of --eval-tokens tokens from
    --eval-text (WikiText-2 raw test, joined with blank lines as AQUA-KV's perplexity evaluation does), each window
    starting with the BOS token like a real request, then consecutive text. Saved as eval0, eval1, ..."""
    import torch
    from kv_compare.capture import capture_ids, load_model
    if args.eval_text is None or not Path(args.eval_text).is_file():
        raise SystemExit("evalcap needs --eval-text (a text file, e.g. WikiText-2 raw test)")
    device = torch.device(args.device)
    tokenizer, model = load_model(args.model, args.revision, device)
    ids = tokenizer(Path(args.eval_text).read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
    bos = tokenizer.bos_token_id
    step = args.eval_tokens - (1 if bos is not None else 0)
    if len(ids) < step * args.eval_docs:
        raise SystemExit(f"{args.eval_text} has {len(ids)} tokens; {step * args.eval_docs} needed")
    reports = []
    for i in range(args.eval_docs):
        window = ([bos] if bos is not None else []) + ids[i * step:(i + 1) * step]
        log(f"evalcap eval{i}: tokens {i * step}-{(i + 1) * step - 1} of {Path(args.eval_text).name}")
        info = capture_ids(model, window, out / "captures", f"eval{i}", device)
        info["source"] = f"{args.eval_text} tokens {i * step}-{(i + 1) * step - 1}"
        reports.append(info)
    (out / "captures" / "eval_meta.json").write_text(json.dumps({"text": str(args.eval_text), "tokens": args.eval_tokens,
                                                                  "captures": reports}, indent=1))
    return reports


def chunk_spans(tokens):
    spans = [(0, 256)]
    if tokens >= 4352:
        spans.append((4096, 4352))
    return spans


def stage_lossless(args, out):
    import torch
    from kv_compare.baselines import exans_bits, splitzip_codebook, splitzip_token_bits
    from kv_compare.predict import CONFIGS, LABELS, Study, macs_per_value
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, args.max_layers, log=log)
    L = study.L
    spans = chunk_spans(study.tests[0].tokens)
    result = {"layers": L, "spans": spans, "methods": {}}

    def summarize(per_layer):            # {doc: [L, N]} -> whole, chunks, per-layer
        whole = {d: float(np.mean(v)) for d, v in per_layer.items()}
        chunks = {f"{d} {a}-{b - 1}": float(np.mean(v[:, a:b])) for d, v in per_layer.items() for a, b in spans}
        layers = {d: [float(x) for x in np.mean(v, 1)] for d, v in per_layer.items()}
        return {"whole": whole, "chunks": chunks, "chunk_average": float(np.mean(list(chunks.values()))), "per_layer": layers}

    inside = splitzip_codebook(study.train, L)
    sz = {d.name: splitzip_token_bits(d, L, inside) for d in study.tests}
    result["methods"]["SplitZip-style"] = dict(summarize(sz), macs_per_value=0, lossless=True)
    ex = {"whole": {d.name: exans_bits(d, range(L), 0, d.tokens) for d in study.tests},
          "chunks": {f"{d.name} {a}-{b - 1}": exans_bits(d, range(L), a, b) for d in study.tests for a, b in spans}}
    ex["chunk_average"] = float(np.mean(list(ex["chunks"].values())))
    result["methods"]["ExANS-style"] = dict(ex, macs_per_value=0, lossless=True)
    log(f"SplitZip-style {result['methods']['SplitZip-style']['chunk_average']:.3f}, ExANS-style {ex['chunk_average']:.3f}")
    for name in args.configs:
        cfg = CONFIGS[name]
        if "ATT" in cfg.kinds and any(d.Q is None for d in study.train):
            log(f"skip {name}: no calibration queries captured")
            continue
        begin = time.time()
        per_layer, _ = study.lossless(cfg)
        entry = summarize({d: v.numpy() for d, v in per_layer.items()})
        entry.update(label=LABELS[name], macs_per_value=macs_per_value(name, study.Hk, study.hd, study.Hq),
                     lossless=True, seconds=time.time() - begin)
        result["methods"][LABELS[name]] = entry
        log(f"{LABELS[name]}: {entry['chunk_average']:.3f} bits/value (chunk average)")
        study._f.clear(); study._att.clear(); study._qmap.clear()
        torch.cuda.empty_cache() if study.device.type == "cuda" else None
    (out / "lossless.json").write_text(json.dumps(result, indent=1))
    return result


def stage_lossy(args, out):
    import torch
    from kv_compare import baselines
    from kv_compare.capture import load_model
    from kv_compare.core import words_to_float
    from kv_compare.predict import CONFIGS, Study
    from kv_compare.quality import continuation_nll, matrices_from_doc, matrices_from_values
    if args.lossy_docs == "eval":                 # the larger quality test (stage evalcap): its own result file
        docs = [f"eval{i}" for i in range(args.eval_docs)]
        P, M, result_file = args.eval_prompt_tokens, args.eval_continuation_tokens, "lossy_eval.json"
    else:
        docs, P, M, result_file = TESTS, args.prompt_tokens, args.continuation_tokens, "lossy.json"
    study = Study(out / "captures", args.device, TRAIN, "tables", docs, None, log=log)
    cfg = CONFIGS[args.lossy_config]
    if "ATT" in cfg.kinds and any(d.Q is None for d in study.train):
        log(f"{args.lossy_config} needs calibration queries; using fast_noatt")
        cfg = CONFIGS["fast_noatt"]
    import gc
    device = torch.device(args.device)

    # References first (fails in seconds, before the fits). The reference is the captured exact KV restored into
    # the cache: what a lossless transfer delivers. The captures come from prefilling the whole 8,192-token
    # document; on a GPU a prefill of the prompt alone can round differently (kernels depend on the sequence
    # length), so the model's own prefill is reported beside it and only has to agree closely (a wrong cache
    # layout would be far off). Restoring the same KV twice must agree exactly.
    log(f"loading {args.model} for the reference perplexities ({memory_note(args.device)})")
    _, model = load_model(args.model, args.revision, device)
    refs = {}
    for doc in study.tests:
        own = continuation_nll(model, doc.ids, P, M, device)
        base = continuation_nll(model, doc.ids, P, M, device, matrices_from_doc(doc, P))
        again = continuation_nll(model, doc.ids, P, M, device, matrices_from_doc(doc, P))
        own_pct = float(100 * (np.exp(own - base) - 1))
        log(f"{doc.name}: perplexity {np.exp(base):.4f} with the captured exact KV; the model's own prefill of the "
            f"prompt gives {np.exp(own):.4f} ({own_pct:+.3f}%)")
        if again != base:
            raise RuntimeError(f"restoring the same KV twice gave different results ({again} vs {base})")
        if abs(own - base) > 0.01 * abs(own) + 2e-3:
            raise RuntimeError(f"captured KV is far from the model's own prefill ({base} vs {own}): check the cache layout")
        refs[doc.name] = (base, own, own_pct)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    chosen = set(args.lossy_methods)
    fits, aqua = {}, {}
    gconsts = None
    if "gpu" in chosen:                  # our GPU codec: the per-layer one-step predictor's integer constants
        from kv_compare import fused as _fused
        log("preparing the GPU lossy codec's one-step predictor")
        gconsts = _fused.prepare(study, log=log)

    VARIANTS = {"base": ("", 0, False), "sinks": (", sinks", None, False), "token": (", token step", 0, True),
                "both": (", sinks + token step", None, True)}
    variants = [VARIANTS[v] for v in args.lossy_variants]

    def gpu_lossy(doc, kappa, prefix=0, token_step=False):
        """Our closed-loop lossy codec on the GPU (kv_compare.lossy_gpu): a real stream, decoded bit for bit as
        the encoder reconstructed it; its bits are the stream's."""
        from kv_compare.lossy_gpu import LossyLayerwise
        w = torch.stack([doc.words(l, device, P) for l in range(study.L)])
        w16 = (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
        TB, ch = (64, 2) if P % 128 == 0 else (16, 1)
        codec = LossyLayerwise(gconsts, study.C, study.hd, kappa, tokens_per_program=TB, chunks=ch, prefix=prefix,
                               token_step=token_step)
        rec, xall, prior = codec.quantize(w16)
        buf, starts, _ = codec.encode(w16, rec, xall, prior)
        dec = codec.decode(buf, starts, P)
        if not torch.equal(dec, rec):
            raise RuntimeError("the GPU lossy decode differs from the encoder's reconstruction")
        vals = torch.stack([words_to_float(dec[l].to(torch.int32) & 0xFFFF).to(torch.float32).cpu()
                            for l in range(study.L)])
        return vals, codec.bits(starts, P) / (study.L * P * study.W)
    if "ours" in chosen:
        log(f"fitting our predictor ({cfg.name}) for closed-loop coding")
        study.keep_qmaps = True          # the closed loop needs every layer's query map; keep them (~0.7 GB)
        all_tests, study.tests = study.tests, study.tests[:1]      # the fit needs no test costs; one is enough
        study.lossless(cfg, fits)
        study.tests = all_tests
        study._f.clear(); study._att.clear()
    if "aqua" in chosen:
        for b in (2, 3, 4):              # one fit per bit width, each on its own closed-loop reconstructions
            log(f"fitting AQUA-KV {b}-bit predictors as {baselines.AQUA_SOURCE} does ({memory_note(args.device)})")
            aqua[b] = baselines.aqua_official_fit(study, study.L, b, log)
            study._f.clear(); study._att.clear()
    gc.collect()
    if study.device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"loading {args.model} for the quality check ({memory_note(args.device)})")
    _, model = load_model(args.model, args.revision, device)
    log(f"model loaded ({memory_note(args.device)})")
    result = {"prompt_tokens": P, "continuation_tokens": M, "lossy_config": cfg.name,
              "reference": "captured exact KV restored into the cache", "documents": {}}
    previous = None
    if chosen != {"ours", "aqua", "kivi", "fp8", "gpu"} and (out / result_file).is_file():
        previous = json.loads((out / result_file).read_text())
        if (previous.get("prompt_tokens"), previous.get("continuation_tokens")) != (P, M):
            raise SystemExit("--lossy-methods merges into lossy.json, which used other prompt/continuation lengths")
        log(f"rerunning {sorted(chosen)}; other rows are kept from the existing lossy.json")
        result["documents"] = dict(previous["documents"])      # a crash part-way keeps the other documents
    for doc in study.tests:
        base, own, own_pct = refs[doc.name]
        check = continuation_nll(model, doc.ids, P, M, device, matrices_from_doc(doc, P))
        if check != base:                     # every row below is measured with this loaded model
            log(f"{doc.name}: after reloading the model the exact-KV NLL is {check} (was {base}); using {check}")
            base = check
        exact = torch.stack([words_to_float(doc.words(l, study.device, P)).to(torch.float32).cpu() for l in range(study.L)])
        rows = {"exact KV": {"bits": 16.0, "nll": base, "ppl": float(np.exp(base)), "ppl_change_pct": 0.0, "rel_error": 0.0,
                             "model_prefill_nll": own, "model_prefill_ppl_change_pct": own_pct}}
        methods = []
        if "ours" in chosen:
            methods += [(f"Ours, step {k}{tag}", lambda k=k, pf=pf, ts=ts: study.lossy(
                cfg, fits, doc, P, k, prefix=args.sink_tokens if pf is None else pf, token_step=ts))
                for tag, pf, ts in variants for k in args.kappas]
        if "aqua" in chosen:
            methods += [(f"AQUA-KV, {b}-bit", lambda b=b: baselines.aqua_official(study, aqua[b], doc, P, b)) for b in (2, 3, 4)]
        if "kivi" in chosen:
            methods += [(f"KIVI, {b}-bit", lambda b=b: baselines.kivi_official(study, doc, P, b)) for b in (2, 4)]
        if "fp8" in chosen:
            methods += [("FP8 cast", lambda: baselines.fp8_cast(study, doc, P))]
        if "gpu" in chosen:
            methods += [(f"Ours, GPU codec (one-step, measured stream), step {k}{tag}", lambda k=k, pf=pf, ts=ts:
                         gpu_lossy(doc, k, prefix=args.sink_tokens if pf is None else pf, token_step=ts))
                        for tag, pf, ts in variants for k in args.kappas]
        if previous and doc.name in previous["documents"]:          # keep the rows of methods not rerun
            old = previous["documents"][doc.name]
            stale = tuple(p for m, p in (("aqua", "AQUA-KV-style"), ("kivi", "KIVI-style")) if m in chosen)
            for label, row in old.items():           # (the earlier stand-ins are dropped when their method reruns)
                if label != "exact KV" and not any(label == m[0] for m in methods) and \
                        not (stale and label.startswith(stale)):
                    rows[label] = row
            if old["exact KV"]["nll"] != base:
                log(f"{doc.name}: note - the earlier run's exact-KV NLL was {old['exact KV']['nll']} (now {base}); "
                    "kept rows are measured against the earlier value")
        for label, run in methods:
            rec, rate = run()
            num = sum(float(((rec[l].double() - exact[l].double()) ** 2).sum()) for l in range(study.L))
            den = sum(float((exact[l].double() ** 2).sum()) for l in range(study.L))
            rel = math.sqrt(num / den)
            nll = continuation_nll(model, doc.ids, P, M, device, matrices_from_values(rec, study.C))
            rows[label] = {"bits": float(rate), "nll": nll, "ppl": float(np.exp(nll)),
                           "ppl_change_pct": float(100 * (np.exp(nll - base) - 1)), "rel_error": rel}
            log(f"{doc.name} {label}: {rate:.3f} bits/value, KV error {100 * rel:.2f}%, perplexity "
                f"{rows[label]['ppl_change_pct']:+.2f}% ({memory_note(args.device)})")
            study._f.clear()
            del rec
        if previous and doc.name in previous["documents"]:          # keep the earlier row order
            order = [k for k in previous["documents"][doc.name] if k in rows]
            rows = {k: rows[k] for k in order + [k for k in rows if k not in order]}
        result["documents"][doc.name] = rows
        (out / result_file).write_text(json.dumps(result, indent=1))           # partial results survive a crash
    log(f"lossy stage done ({memory_note(args.device)})")
    del model, study, fits, aqua
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def stage_frontier(args, out):
    from kv_compare.frontier import Frontier, default_variants, summarize
    from kv_compare.predict import Study
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, args.max_layers, log=log)
    with_att = all(d.Q is not None for d in study.train)
    variants = default_variants(with_att=with_att, quick=args.frontier_quick)
    log(f"frontier: {len(variants)} variants over {study.L} layers ({memory_note(args.device)})")
    begin = time.time()
    vs, bits, macs = Frontier(study, variants, log).run()
    rows = summarize(vs, bits, macs, chunk_spans(study.tests[0].tokens))
    res = {"layers": study.L, "seconds": time.time() - begin, "with_attention_input": with_att, "variants": rows}
    (out / "frontier.json").write_text(json.dumps(res, indent=1))
    log(f"frontier done in {res['seconds']:.0f} s ({memory_note(args.device)})")
    return res


SPLITZIP_A4500 = ("SplitZip's native codec on the same RTX A4500 (official run, branch experiment/llama32_official_gpu, "
                  "benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517): 12.04 bits per "
                  "value, encode 137 GB/s, decode 311 GB/s")


def stage_exans(args, out):
    """The official ExANS codec (built from the authors' source with a one-line build fix) on the test documents:
    real bits per value, bit-exact round trip and its encode / decode speed on this GPU."""
    import gc
    import torch
    from kv_compare import exans_official as ex
    from kv_compare.predict import Study
    device = torch.device(args.device)
    log(f"exans: building {ex.EXANS_REPO} at {ex.EXANS_COMMIT[:7]} (CUDA {args.cuda_home})")
    try:
        info = ex.build(out / "exans", args.exans_source, args.cuda_home, log)
    except Exception as exc:                         # the other stages still run; the report says why
        log(f"exans: build failed: {exc}")
        (out / "exans.json").write_text(json.dumps({"error": str(exc)[-3000:], "documents": {}}, indent=1))
        return None
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, None, log=log)
    res = {"build": info, "documents": {}}
    for d in study.tests:
        w = torch.stack([d.words(l, device) for l in range(study.L)])
        w16 = (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
        del w
        log(f"exans: {d.name} ({w16.numel() / 1e6:.0f}M values)")
        try:
            res["documents"][d.name] = ex.measure(info, w16, log=log)
        except Exception as exc:
            log(f"exans: {d.name} failed: {exc}")
            res.setdefault("errors", {})[d.name] = str(exc)[-2000:]
        (out / "exans.json").write_text(json.dumps(res, indent=1))
        del w16
        gc.collect()
        torch.cuda.empty_cache()
    return res


def stage_format(args, out):
    """Bits of our one-step prediction inside SplitZip's kind of format (fixed-length codes chosen per value by the
    prediction; no entropy coder), for the dense product and the reduced-rank fast-link products."""
    import gc
    import torch
    from kv_compare import fused, layerwise, szformat
    from kv_compare.predict import Study
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, args.max_layers, log=log)
    consts = fused.prepare(study, log=log)
    res = {"products": {}}
    log("format: dense one-step product")
    res["products"]["dense"] = szformat.probe(study, consts, None, log=log)
    (out / "format.json").write_text(json.dumps(res, indent=1))
    for r in args.format_ranks:
        rc = layerwise.prepare_rank(study, consts, r, log)
        log(f"format: rank {r} product")
        res["products"][f"rank {r}"] = szformat.probe(study, consts, rc, log=log)
        (out / "format.json").write_text(json.dumps(res, indent=1))
        del rc
        gc.collect()
    for name, pr in res["products"].items():
        log(f"format ({name}): " + "; ".join(f"{v}: {x['mean']:.3f}" for v, x in pr["variants"].items()))
    del study, consts
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res


def stage_szcodec(args, out):
    """Our prediction in SplitZip's format as a real GPU codec (kv_compare.szcodec): fixed-length codes chosen by
    the one-step prediction, decoded bit-exact and timed on the first test document; dense and reduced-rank
    products, tile sizes, a CUDA graph, and mixes with the authors' SplitZip per layer."""
    import gc
    import torch
    from kv_compare import fused
    from kv_compare.gpu_speed import szcodec_rows
    from kv_compare.predict import Study
    device = torch.device(args.device)
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, args.max_layers, log=log)
    consts = fused.prepare(study, log=log)

    def w16(d):
        T = d.tokens - d.tokens % 128
        w = torch.stack([d.words(l, device, rows=T) for l in range(study.L)])
        return (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
    words = w16(study.tests[0])
    cal = [w16(d) for d in study.train[:2]]
    log(f"szcodec: {study.tests[0].name}, {words.numel() / 1e6:.0f}M values; code books from "
        f"{', '.join(d.name for d in study.train[:2])}; products {args.sz_ranks or ['dense']}, tiles {args.sz_configs}")
    rows = szcodec_rows(consts, study, words, cal, device, log, ranks=[r or None for r in args.sz_ranks],
                        configs=args.sz_configs, row_configs=args.sz_row_configs, chunk_counts=args.sz_chunks,
                        splitzip_source=(args.splitzip_source or (out / "splitzip")) if device.type == "cuda" else None,
                        counts=args.mix_counts)
    res = {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "doc": study.tests[0].name,
           "calibration": [d.name for d in study.train[:2]], "rows": rows}
    (out / "szcodec.json").write_text(json.dumps(res, indent=1))
    del study, consts, words, cal
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return res


def szcodec_lines(out):
    """The SplitZip-format codec: the measured modes, then time and bytes per link next to SplitZip, our rANS codec
    and the rANS mix (from gpu_speed.json when present)."""
    res = json.loads((out / "szcodec.json").read_text())
    rows = [r for r in res["rows"] if r["part"] == "szcodec"]
    ok = [r for r in rows if r.get("bit_exact")]
    lines = ["## Our prediction in SplitZip's format: the real codec (kv_compare/szcodec.py)", "",
             "Fixed-length codes chosen per value by our one-step prediction (sign agreement + magnitude residual, "
             "per-context b raw bits and a k-bit code, escapes carry the word), so every value of a tile is placed by "
             "prefix sums, as in SplitZip; no entropy coder. Code books from "
             f"{', '.join(res['calibration'])}; measured on {res['doc']} on the {res['gpu']}, bit-exact round trip "
             "checked. Bits include the 32-bit segment starts and padding.", "",
             "Layouts: tile = a program decodes tokens x 128 lanes at once (prefix sums over the tile); row = token by "
             "token over the head's 128 lanes, like our rANS kernel (prefix sums over 128 lanes, a running offset), a "
             "token's codes then its payloads (two reads per value); row1 = the same with each value's code and raw "
             "bits in one field and the escaped words after the token (one read per value; an escape keeps its raw "
             "bits' place, so code books are chosen for that cost). Reduced rank \"inside\": the second product runs "
             "in the decoding kernel on tensor cores (row layouts: for the program's whole tile, its prediction kept "
             "in L2 for the token loop). Chunks: token chunks decoded on separate CUDA streams.", "",
             "| Product | Layout | Tokens x head per program, warps | Bits per value | Decode | G values/s | Exact |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        prod = ("dense (cuBLAS, through memory)" if not r["rank"] else
                f"rank {r['rank']}, " + ("inside the kernel" if r["product"] == "fused" else "through memory"))
        tile = (f"{r['tokens_per_program']}, {r['warps']}" + (f", {r['chunks']} chunks" if r.get("chunks", 1) > 1 else "")
                + (", CUDA graph" if r.get("graph") else ""))
        lay = r.get("layout", "tile")
        if "error" in r:
            lines.append(f"| {prod} | {lay} | {tile} | not measured: {r['error'][:120]} | | | |")
        else:
            lines.append(f"| {prod} | {lay} | {tile} | {r['stream_bits_per_value']:.3f} | {r['decode_ms']:.2f} ms | "
                         f"{r['decode_gvalues_per_s']:.1f} | {r['bit_exact']} |")
    lines += sz_budget_lines(res)
    parts = [r for r in res["rows"] if r["part"] == "szcodec_parts"]
    if parts:
        lines += ["", "Where the decode time goes (the fastest mode of each layout, one chunk; the ablations decode "
                  "wrong words and only time the kernel):", "",
                  "| Product | Layout | Whole decode | Products alone | Coding alone | Coding without stream reads | "
                  "Coding without prefix sums |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in parts:
            prod = ("dense" if not r["rank"] else f"rank {r['rank']}") + (", second product inside" if r["product"] ==
                                                                           "fused" else "")
            lines.append(f"| {prod} | {r['layout']} {r['tokens_per_program']} x {r['warps']} | {r['whole_ms']:.2f} ms | "
                         f"{r['products_only_ms']:.2f} ms | {r['coding_only_ms']:.2f} ms | "
                         f"{r['coding_no_reads_ms']:.2f} ms | {r['coding_no_prefix_sums_ms']:.2f} ms |")
    if not ok:
        return lines + [""]
    n = ok[0]["values"]
    szr = next((r for r in res["rows"] if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    g = json.loads((out / "gpu_speed.json").read_text())["rows"] if (out / "gpu_speed.json").is_file() else []
    if szr is None:
        szr = next((r for r in g if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    if szr is None:
        return lines + ["", "(SplitZip not measured in this run: no per-link comparison.)", ""]
    rans = [r for r in g if r["part"] in ("fused", "layerwise") and r.get("bit_exact") and r.get("values") == n]
    rmix = [r for r in g if r["part"] == "mixed" and r.get("bit_exact") and r.get("values") == n]
    smix = [r for r in res["rows"] if r["part"] == "szcodec_mixed" and r.get("bit_exact")]
    t = lambda r, R: max(r["stream_bits_per_value"] / R, 1 / r["decode_gvalues_per_s"]) * n / 1e9
    ms = lambda sec: f"{1e3 * sec:.1f} ms"
    mib = lambda r: f"{r['stream_bits_per_value'] * n / 8 / 2 ** 20:.0f} MiB"

    def best(pool, R):
        return min(pool, key=lambda r: t(r, R)) if pool else None

    def desc(r):
        if r is None:
            return "-"
        kind = f"rank {r['rank']}" if r.get("rank") else "dense"
        if r["part"] == "szcodec_mixed":
            return f"{kind}, {r['ours_layers']} layers ours"
        if r["part"] == "mixed":
            return f"{r['mode']}, {r['ours_layers']} layers ours"
        return kind

    def cell(r, R):
        return f"{ms(t(r, R))}, {mib(r)}" if r is not None else "not measured"
    lines += ["", f"Per link ({n / 1e6:.0f}M values; time = the longer of the wire time and the decode, decoding "
              f"overlapped with arrival; SplitZip: the authors' codec run here, {szr['stream_bits_per_value']:.3f} bits/value, "
              f"{szr['decode_ms']:.2f} ms decode). Each column is the fastest measured mode of that kind for the link; "
              "the mixes code the layers where ours saves the most with ours and the rest with SplitZip.", "",
              "| Link | SplitZip | Ours, rANS | Ours, rANS + SplitZip | Ours, SplitZip format | SplitZip format + SplitZip | "
              "Best of ours vs SplitZip |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for R, _band in LINK_BANDS:
        ts = t(szr, R)
        a, b, c, d = best(rans, R), best(rmix, R), best(ok, R), best(smix, R)
        top = min((x for x in (a, b, c, d) if x is not None), key=lambda x: t(x, R))
        lines.append(f"| {R} Gb/s | {ms(ts)}, {mib(szr)} | {cell(a, R)} | {cell(b, R)} ({desc(b)}) | "
                     f"{cell(c, R)} ({desc(c)}) | {cell(d, R)} ({desc(d)}) | {ts / t(top, R):.2f}x ({desc(top)}"
                     f"{', SplitZip format' if top['part'].startswith('szcodec') else ', rANS'}) |")
    return lines + [""]


def sz_budget_lines(res):
    """The SplitZip-format codec's bit budget (kv_compare.szcodec.bit_budget) per product and code-book kind."""
    bud = [r for r in res["rows"] if r["part"] == "szcodec_budget"]
    if not bud:
        return []
    lines = ["", "Where the format's bits go on the test document, and the room left (bits per value; without the "
             "segment starts and padding): codes + raw bits + escapes with the code books fitted on the calibration "
             "documents, the share of values escaped, the same with up to 255 code words per context (8-bit codes), "
             "with code books fitted on this document itself (the best a fixed code book could do), and the entropy "
             "of the symbols given the context (an ideal adaptive coder with the same contexts; our rANS codec is a "
             "different symbolization).", "",
             "| Product | Code books | Codes | Raw bits | Escapes | Total | Escaped | 8-bit codes | Fitted on the "
             "document (6-bit / 8-bit) | Entropy given the context |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in bud:
        lines.append(f"| {'rank ' + str(r['rank']) if r['rank'] else 'dense'} | "
                     f"{'one read (escapes keep the raw bits)' if r['escape_keeps_raw'] else 'two reads'} | "
                     f"{r['code']:.3f} | {r['raw']:.3f} | {r['escape']:.3f} | {r['books_bits']:.3f} | "
                     f"{100 * r['escape_rate']:.2f}% | {r['k8_bits']:.3f} | {r['oracle_bits']:.3f} / "
                     f"{r['oracle_k8_bits']:.3f} | {r['entropy_bits']:.3f} |")
    return lines


def bytes_time_lines(out):
    """Every measured lossless mode as a trade of bytes against time: the modes no other mode beats on both bits
    and decode time, then per link the fastest mode and the fewest bytes within +10%, +25% and +50% of its time."""
    modes = {}

    def add(label, bits, rate, n):
        if label not in modes or rate > modes[label][1]:
            modes[label] = (bits, rate, n)
    g = json.loads((out / "gpu_speed.json").read_text())["rows"] if (out / "gpu_speed.json").is_file() else []
    z = json.loads((out / "szcodec.json").read_text())["rows"] if (out / "szcodec.json").is_file() else []
    parts = ("splitzip_official", "fused", "layerwise", "mixed", "szcodec", "szcodec_mixed")
    n0 = next((r["values"] for r in z + g if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    for r in g + z:
        if r["part"] not in parts or not r.get("bit_exact") or not r.get("decode_gvalues_per_s"):
            continue
        if r.get("values") != n0:
            continue
        n = n0
        kind = f"rank {r['rank']}" if r.get("rank") else "dense"
        if r["part"] == "splitzip_official":
            add("SplitZip (authors' codec)", r["stream_bits_per_value"], r["decode_gvalues_per_s"], n)
        elif r["part"] == "fused":
            add("ours, rANS, fused one-kernel codec", r["stream_bits_per_value"], r["decode_gvalues_per_s"], n)
        elif r["part"] == "layerwise":
            add(f"ours, rANS, {kind}", r["stream_bits_per_value"], r["decode_gvalues_per_s"], n)
        elif r["part"] == "mixed" and r["ours_layers"]:      # (all layers ours: our codec alone)
            whole = r["ours_layers"] == len(r["ours_mask"])
            add(f"ours, rANS, {r['mode']}" if whole else
                f"ours, rANS + SplitZip, {r['mode']}, {r['ours_layers']} layers ours", r["stream_bits_per_value"],
                r["decode_gvalues_per_s"], n)
        elif r["part"] == "szcodec":
            add(f"ours, SplitZip format, {kind}", r["stream_bits_per_value"], r["decode_gvalues_per_s"], n)
        elif r["part"] == "szcodec_mixed" and r["ours_layers"]:
            whole = r["ours_layers"] == len(r["ours_mask"])
            add(f"ours, SplitZip format, {kind}" if whole else
                f"ours, SplitZip format + SplitZip, {kind}, {r['ours_layers']} layers ours",
                r["stream_bits_per_value"], r["decode_gvalues_per_s"], n)
    if (out / "exans.json").is_file() and n0:
        for d, v in json.loads((out / "exans.json").read_text()).get("documents", {}).items():
            if d == "test_text":
                add("ExANS (authors' codec)", v["bits_per_value"], v["decode_gbps_wall"] / 2, n0)
    if not modes or "SplitZip (authors' codec)" not in modes:
        return []
    n = n0
    mib = lambda bits: bits * n / 8 / 2 ** 20
    t = lambda m, R: max(m[0] / R, 1 / m[1]) * n / 1e9
    sz = modes["SplitZip (authors' codec)"]
    front, best_ms = [], float("inf")
    for label, m in sorted(modes.items(), key=lambda kv: (kv[1][0], -kv[1][1])):
        d = n / m[1] / 1e6
        if d < best_ms:
            front.append((label, m))
            best_ms = d
    shown = []                                       # from the fastest: a mode per 0.15 bits saved at least
    for label, m in reversed(front):
        if not shown or shown[-1][1][0] - m[0] >= 0.15 or label == front[0][0]:
            shown.append((label, m))
    front = list(reversed(shown))
    lines = ["", "## Bytes against time: every measured lossless mode", "",
             "Fewer bytes cost decode time. The modes below are the ones no other measured mode beats on both bits "
             "and decode time (one per 0.15 bits per value or more), from fewest bytes to fastest decode (one "
             f"document, {n / 1e6:.0f}M values; every one a real stream decoded bit-exact):", "",
             "| Mode | Bits per value | Sent | Bytes vs SplitZip | Decode |", "| --- | --- | --- | --- | --- |"]
    for label, m in front:
        lines.append(f"| {label} | {m[0]:.3f} | {mib(m[0]):.0f} MiB | {100 * (1 - m[0] / sz[0]):.0f}% fewer | "
                     f"{n / m[1] / 1e6:.1f} ms |")
    lines += ["", "Per link: the fastest mode, and the fewest bytes one can send while spending at most 10%, 25% or 50% "
              "more time than the fastest (time = the longer of the wire time and the decode; each cell: time, "
              "bytes sent, mode):", "",
              "| Link | Fastest | Fewest bytes within +10% | within +25% | within +50% | Fewest bytes at all |",
              "| --- | --- | --- | --- | --- | --- |"]
    cell = lambda lm, R: f"{1e3 * t(lm[1], R):.1f} ms, {mib(lm[1][0]):.0f} MiB ({lm[0]})"
    for R, _band in LINK_BANDS:
        ranked = sorted(modes.items(), key=lambda kv: t(kv[1], R))
        fast = ranked[0]
        tf = t(fast[1], R)
        cells = []
        for x in (1.10, 1.25, 1.50):
            ok = [kv for kv in modes.items() if t(kv[1], R) <= x * tf]
            cells.append(cell(min(ok, key=lambda kv: kv[1][0]), R))
        least = min(modes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
        lines.append(f"| {R} Gb/s | {cell(fast, R)} | " + " | ".join(cells) + f" | {cell(least, R)} |")
    return lines + [""]


def format_lines(out):
    res = json.loads((out / "format.json").read_text())
    prods = list(res["products"])
    if not prods:
        return []
    first = res["products"][prods[0]]
    lines = ["## Our prediction inside SplitZip's format (bits only; scripts: kv_compare/szformat.py)", "",
             "Fixed-length codes as in SplitZip (no entropy coder: every value can be placed independently), with the "
             "code book of each value chosen by our one-step prediction; a value's code length depends only on its "
             "context, which the decoder knows from the prediction. Code books from the calibration documents "
             f"({', '.join(first['calibration'])}), scored on the test documents. The decoder still computes the "
             "prediction (the products, timed above) but no rANS. See kv_compare/szformat.py for the variants.", "",
             "| Variant | " + " | ".join(f"{p} (text / code)" for p in prods) + " | Mean vs SplitZip-style |",
             "| --- | " + " | ".join("---" for _ in prods) + " | --- |"]
    base = {p: res["products"][p]["variants"]["SplitZip-style"]["mean"] for p in prods}
    for v in first["variants"]:
        cells = []
        for p in prods:
            x = res["products"][p]["variants"][v]
            cells.append(f"{x['mean']:.3f} ({x['bits']['test_text']:.3f} / {x['bits']['test_code']:.3f})")
        best = min(res["products"][p]["variants"][v]["mean"] for p in prods)
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {best - base[prods[0]]:+.3f} |")
    return lines + [""]


def stage_gpu_speed(args, out):
    import gc
    import torch
    from kv_compare.gpu_coder import SliceCoder
    from kv_compare.gpu_speed import coder_inputs, coder_rows, fused_rows, overlap_row, predictor_layer, predictor_rows
    from kv_compare.predict import CONFIGS, Study
    device = torch.device(args.device)
    meta = json.loads((out / "captures" / "meta.json").read_text())
    Hk, hd, Hq = int(meta["kv_heads"]), int(meta["head_dim"]), int(meta["query_heads"])
    W, Qw = 2 * Hk * hd, Hq * hd
    note = gpu_sharing_note(args.device)
    if note and "WARNING" in note:
        log("gpu_speed: " + note)
    variants = [
        {"label": "P, 1 pivot step", "ne": W, "steps": 1},
        {"label": "P, 16 pivot steps, rank 128", "ne": W, "rank": 128, "steps": 16},
        {"label": "P, 4 pivot steps", "ne": W, "steps": 4},
        {"label": "P, 8 pivot steps", "ne": W, "steps": 8},
        {"label": "P, 16 pivot steps", "ne": W, "steps": 16},
        {"label": "L0+P2+P, 16 pivot steps", "ne": 3 * W, "steps": 16},
        {"label": "L0+P2+P+ATT, 16 pivot steps", "ne": 3 * W + Qw, "steps": 16, "att": True},
    ]
    log(f"gpu_speed: predictor, one layer of {args.speed_tokens} tokens, {len(variants)} variants")
    rows = predictor_rows(Hk, hd, Hq, args.speed_tokens, device, variants, args.speed_formats, log)
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, None, log=log)
    cfg = CONFIGS[args.speed_config]
    if "ATT" in cfg.kinds and any(d.Q is None for d in study.train):
        cfg = CONFIGS["lite"]
    layers = [l for l in args.speed_layers if l < study.L]
    log(f"gpu_speed: coder inputs from {cfg.name} on {study.tests[0].name}, layers {layers}")
    data = coder_inputs(study, cfg, study.tests[0], layers)
    del study
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    if device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1":
        crows = coder_rows(SliceCoder, device, data, args.speed_settings, log,
                           max_values=None if device.type == "cuda" else 4096, modes=args.speed_modes)
        rows += crows
        ok = [r for r in crows if r["bit_exact"]]
        fmt = "int8x8" if "int8x8" in args.speed_formats else args.speed_formats[0]
        for best in ([max([r for r in ok if r["mode"] == m], key=lambda r: r["decode_gvalues_per_s"])
                      for m in args.speed_modes if any(r["mode"] == m for r in ok)] if device.type == "cuda" else []):
            for label in ("P, 1 pivot step", "P, 8 pivot steps", "P, 16 pivot steps"):
                v = next(x for x in variants if x["label"] == label)
                gen = torch.Generator(device=device).manual_seed(1)
                layer_fn, _ = predictor_layer(Hk, hd, Hq, args.speed_tokens, device, v, fmt, gen)
                try:
                    row = overlap_row(SliceCoder, device, data, (best["values_per_stream"], best["block"], best["warps"]),
                                      layer_fn, log, mode=best["mode"], search=best.get("search", False))
                    row.update(predictor=f"{label} ({fmt})", variant=label)
                    rows.append(row)
                except Exception as exc:
                    log(f"  overlap {label}: not measured ({exc!r})")
                del layer_fn
    else:
        log("gpu_speed: the coder needs a CUDA GPU (or TRITON_INTERPRET=1 for a slow CPU check); skipped")
    if device.type == "cuda" and (args.fused_configs or args.layer_configs):
        del data
        gc.collect()
        torch.cuda.empty_cache()
        study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, None, log=log)
        log(f"gpu_speed: fused one-step codec, whole {study.tests[0].name} ({study.L} layers), configs {args.fused_configs}")
        rows += fused_rows(study, device, args.fused_configs, log, modes=args.fused_modes,
                           layer_configs=args.layer_configs, ranks=args.layer_ranks, lossy_kappas=args.lossy_speed_kappas,
                           splitzip_source=args.splitzip_source or (out / "splitzip"), mix_counts=args.mix_counts)
        del study
    bits = {}
    if (out / "frontier.json").is_file():
        for r in json.loads((out / "frontier.json").read_text())["variants"]:
            bits[r["label"]] = r["chunk_average"]
    res = {"gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
           "gpu_memory_at_start": note, "coder_config": cfg.name, "coder_layers": layers,
           "tokens_per_layer": args.speed_tokens, "values_per_layer": args.speed_tokens * W,
           "bits_per_value": bits, "splitzip_reference": SPLITZIP_A4500, "rows": rows}
    (out / "gpu_speed.json").write_text(json.dumps(res, indent=1))
    log(f"gpu_speed done ({memory_note(args.device)})")
    return res


def stage_pool(args, out):
    """GPU-resident KV pool, measured parts: the model's prefill (what a pool miss costs), host-to-GPU copies (pinned
    and pageable) and the GPU memory left after the model. Every method's decode and encode times come from the
    other stages (gpu_speed, exans, the authors'-code timings), its bits and quality from the lossy stage."""
    import gc
    import torch
    from kv_compare.capture import load_model
    from kv_compare.pool import copy_seconds, prefill_seconds
    from kv_compare.predict import Study
    device = torch.device(args.device)
    if device.type != "cuda":
        log("pool: needs a CUDA GPU; skipped")
        return None
    note = gpu_sharing_note(args.device)
    if note and "WARNING" in note:
        log("pool: " + note)
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, None, log=log)
    doc = study.tests[0]
    L, W, T = study.L, study.W, doc.tokens
    ids = [int(v) for v in doc.ids[: doc.tokens]]
    del study
    gc.collect()
    log("pool: loading the model for the recompute baseline")
    _tok, model = load_model(args.model, args.revision, device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    prefill = {}
    for t in args.pool_tokens:
        if t <= len(ids):
            prefill[t] = prefill_seconds(model, ids[:t], device)
            log(f"  prefill of {t} tokens: {1e3 * prefill[t]:.1f} ms")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    n = L * T * W
    copies = {"raw BF16": copy_seconds(2 * n, device)}
    c = copies["raw BF16"]
    log(f"  copy of the raw KV ({2 * n / 2 ** 20:.0f} MiB): host to GPU {1e3 * c['h2d_pinned']:.1f} ms pinned, "
        f"{1e3 * c['h2d_pageable']:.1f} ms pageable")
    res = {"gpu": torch.cuda.get_device_name(device), "gpu_memory_at_start": note, "doc": doc.name, "tokens": T,
           "layers": L, "values": n, "free_after_model_bytes": free, "total_bytes": total,
           "prefill_seconds": {str(k): v for k, v in prefill.items()}, "copies": copies, "version": 2}
    (out / "pool.json").write_text(json.dumps(res, indent=1))
    log(f"pool done ({memory_note(args.device)})")
    return res


def pool_lines(out):
    """Every method as a GPU-resident KV pool for one cached prefix: memory held, prefixes that fit, time to bring
    it back from GPU or CPU memory, the cost of adding it, and (lossy methods) quality."""
    res = json.loads((out / "pool.json").read_text())
    T, n, L = res["tokens"], res["values"], res["layers"]
    raw = 2 * n
    free = res["free_after_model_bytes"]
    pre = res["prefill_seconds"].get(str(T))
    rate_h2d = raw / res["copies"]["raw BF16"]["h2d_pinned"]           # bytes per second, pinned
    mib = lambda bits: bits * n / 8 / 2 ** 20
    gs = json.loads((out / "gpu_speed.json").read_text())["rows"] if (out / "gpu_speed.json").is_file() else []
    gs += res.get("codec_rows", [])
    lres = json.loads((out / "lossy.json").read_text()) if (out / "lossy.json").is_file() else None
    docs = list(lres["documents"]) if lres else []
    entries = []                                     # (method, quality, bits, decode s or None, encode s or None)

    def quality(label):
        if not lres:
            return None
        rs = [lres["documents"][d].get(label) for d in docs]
        if any(r is None for r in rs):
            return None
        return float(np.mean([r["bits"] for r in rs])), " / ".join(f"{r['ppl_change_pct']:+.2f}%" for r in rs)

    szr = next((r for r in gs if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    if szr:
        entries.append(("SplitZip (authors' codec)", "lossless", szr["stream_bits_per_value"], szr["decode_ms"] / 1e3,
                        szr["encode_ms"] / 1e3))
    elif (out / "lossless.json").is_file():
        b = float(np.mean(json.loads((out / "lossless.json").read_text())["methods"]["SplitZip-style"]["per_layer"][res["doc"]]))
        entries.append(("SplitZip (official A4500 speed, earlier run)", "lossless", b, raw / 311e9, raw / 137e9))
    if (out / "exans.json").is_file():
        ex = json.loads((out / "exans.json").read_text())["documents"].get(res["doc"])
        if ex:
            entries.append(("ExANS (authors' codec)", "lossless", ex["bits_per_value"], ex["decode_ms_wall"] / 1e3,
                            ex["encode_ms_wall"] / 1e3))
    best = {}
    for r in gs:                                     # ours lossless: the fewest bits and the fastest decode
        if r["part"] in ("fused", "layerwise") and r.get("bit_exact"):
            for key, better in (("fewest bits", lambda a, b: a["stream_bits_per_value"] < b["stream_bits_per_value"]),
                                ("fastest decode", lambda a, b: a["decode_ms"] < b["decode_ms"])):
                if key not in best or better(r, best[key]):
                    best[key] = r
    for key, r in best.items():
        enc = (r.get("model_ms") or 35.0) / 1e3 + r["encode_ms"] / 1e3
        kind = "fused" if r["part"] == "fused" else f"per layer{', rank ' + str(r['rank']) if r.get('rank') else ''}"
        entries.append((f"Ours lossless, {key} ({kind}, {r['mode']})", "lossless", r["stream_bits_per_value"],
                        r["decode_ms"] / 1e3, enc))
    lspeed = {}
    for r in gs:
        if r["part"] == "lossy_layerwise" and r.get("bit_exact") and r.get("rank") is None:
            lspeed[float(r["kappa"])] = (r["decode_ms"] / 1e3, (r["quantize_ms"] + r["encode_ms"]) / 1e3)
    if lres:
        for label in lres["documents"][docs[0]]:
            if label == "exact KV":
                continue
            q = quality(label)
            if q is None:
                continue
            dec = enc = None
            if label.startswith("Ours, GPU codec"):
                k = float(label.split("step ", 1)[1].split(",")[0])
                dec, enc = lspeed.get(k, (None, None))
            for name, path in (("AQUA-KV", "time_aqua.json"), ("KIVI", "time_kivi.json")):
                if label.startswith(name + ",") and (out / path).is_file():
                    tb = json.loads((out / path).read_text())["bits"].get(label.split(", ")[1].split("-")[0])
                    if tb:
                        dec, enc = tb["decode_ms"] / 1e3, tb["encode_ms"] / 1e3
            entries.append((label, q[1], q[0], dec, enc))
    ms = lambda sec: "not timed" if sec is None else f"{1e3 * sec:,.0f} ms"
    lines = [f"## GPU-resident KV pool ({res['gpu']})", "",
             f"A pool keeps finished requests' KV so that a later request with the same prefix skips its prefill. One "
             f"cached prefix of {T} tokens ({res['doc']}, {L} layers; {raw / 2 ** 20:.0f} MiB as raw BF16). Recomputing "
             f"it (the model's prefill, what a miss costs) takes {ms(pre)}. From GPU memory a method pays its decode; "
             f"from CPU memory, the host-to-GPU copy of its bytes ({rate_h2d / 1e9:.1f} GB/s pinned, measured with the "
             "raw KV) and then the decode. Adding a prefix costs its encode. Prefixes that fit: the "
             f"{free / 2 ** 30:.1f} GiB left after the model's weights (no room kept for activations, so an upper "
             "bound). Decode and encode times are measured on this GPU: ours and SplitZip in gpu_speed, ExANS in its "
             "stage, AQUA-KV and KIVI with the authors' own code (scripts/time_baseline_decoders.py); our lossy "
             "variants (sinks, token step) decode at the base variant's speed. Lossy quality = perplexity change "
             f"({' / '.join(docs)}).", "",
             "| Method | Lossless / perplexity change | Bits per value | MiB per prefix | Prefixes in free GPU memory | "
             "Back from GPU memory | Back from CPU memory | Adding it |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |",
             f"| Recompute (the model's prefill) | lossless | - | 0 | - | {ms(pre)} | {ms(pre)} | 0 |",
             f"| Raw BF16 | lossless | 16.00 | {mib(16):.0f} | {free / (raw):,.1f} | 0 ms | "
             f"{ms(res['copies']['raw BF16']['h2d_pinned'])} | 0 |"]
    for name, qual, bits, dec, enc in entries:
        nbytes = bits * n / 8
        cpu = None if dec is None else nbytes / rate_h2d + dec
        lines.append(f"| {name} | {qual} | {bits:.2f} | {mib(bits):.0f} | {free / nbytes:,.1f} | {ms(dec)} | "
                     f"{ms(cpu)} | {ms(enc)} |")
    return lines + [""]



def speed_lines(out):
    res = json.loads((out / "gpu_speed.json").read_text())
    pred = [r for r in res["rows"] if r["part"] == "predictor"]
    coder = [r for r in res["rows"] if r["part"] == "coder"]
    over = [r for r in res["rows"] if r["part"] == "overlap"]
    lines = [f"## GPU speed ({res['gpu']})", "",
             f"Kernel time only (CUDA-event medians after warm-up), like the published SplitZip and ExANS GB/s; GB/s "
             f"counts raw BF16 bytes (2 per value). Predictor: one layer of {res['tokens_per_layer']} tokens, random "
             "data of the real shapes (the inputs product, then one product per group in sequence; the attention "
             "input adds the query map and the windowed attention, in BF16 for the integer formats). Not included: "
             "RoPE rotation and the spread arithmetic (element-wise).", "",
             "| Predictor | Format | ms per layer | G values/s | GB/s | As a CUDA graph: ms, G values/s |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in pred:
        if "error" in r:
            lines.append(f"| {r['variant']} | {r['format']} | not measured | | | |")
        else:
            gr = (f"{r['ms_per_layer_graph']:.2f}, {r['gvalues_per_s_graph']:.1f}" if r.get("ms_per_layer_graph") else "")
            lines.append(f"| {r['variant']} | {r['format']} | {r['ms_per_layer']:.2f} | {r['gvalues_per_s']:.1f} | {r['gb_per_s']:.1f} | {gr} |")
    if coder:
        lines += ["", f"Value-slice coder (Triton) on {coder[0]['values']:,} real values ({res['coder_config']} "
                  f"predictions, layers {res['coder_layers']}); the round trip must be bit-exact.", "",
                  "Stage 1: how the decoder finds the slice symbol (direct: a table and one comparison; search: a "
                  "coarse table and a binary search); the streams are the same.", "",
                  "| Mode | Stage 1 | Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in coder:
            lines.append(f"| {r.get('mode', 'exact')} | {'search' if r.get('search', True) else 'direct'} | {r['values_per_stream']} x {r['block']} ({r.get('warps', max(1, r['block'] // 32))}) | {r['bit_exact']} | {r['stream_bits_per_value']:.4f} | "
                         f"{r['exact_model_bits_per_value']:.4f} | {r['encode_gb_per_s']:.1f} | {r['decode_gb_per_s']:.1f} |")
    ok = [r for r in coder if r["bit_exact"]]
    if ok and pred:
        best = max(ok, key=lambda r: r["decode_gvalues_per_s"])
        lines += ["", f"Decode estimate per predictor (fastest exact integer format, else BF16, CUDA graph if faster; then "
                  f"the fastest coder, {best.get('mode', 'exact')} mode; one after the other, not overlapped). Bits include "
                  "that coder's measured overhead. Link it keeps up with = decoded values/s x bits per value.", "",
                  "| Predictor | Bits per value | Decode G values/s | Decode GB/s | Keeps up with a link of |",
                  "| --- | --- | --- | --- | --- |"]
        for v in dict.fromkeys(r["variant"] for r in pred):
            cand = [r for r in pred if r["variant"] == v and "error" not in r and r["format"] in ("int8x8", "int8x16")]   # dense only
            cand = cand or [r for r in pred if r["variant"] == v and "error" not in r and r["format"] == "bf16"]
            if not cand:
                continue
            p_ = max(cand, key=lambda r: max(r["gvalues_per_s"], r.get("gvalues_per_s_graph") or 0))
            pg = max(p_["gvalues_per_s"], p_.get("gvalues_per_s_graph") or 0)
            g = 1 / (1 / pg + 1 / best["decode_gvalues_per_s"])
            b = res["bits_per_value"].get(v)
            b = None if b is None else b + best["stream_bits_per_value"] - best["exact_model_bits_per_value"]
            link = f"{g * b:.0f} Gb/s" if b else "n/a"
            lines.append(f"| {v} ({p_['format']}) | {'n/a' if b is None else f'{b:.3f}'} | {g:.1f} | {2 * g:.1f} | {link} |")
    for r in over:
        b = res["bits_per_value"].get(r["predictor"].split(" (")[0])
        lines += ["", f"Overlap, {r.get('mode', 'exact')} coder ({r['predictor']} on one CUDA stream, the coder decoding on another, as a pipelined "
                  f"decoder would on different token blocks): predictor {r['predictor_ms']:.2f} ms + decode "
                  f"{r['decode_ms']:.2f} ms one after the other, {r['both_ms']:.2f} ms together, so "
                  f"{r['overlapped_gvalues_per_s']:.1f} G values/s ({2 * r['overlapped_gvalues_per_s']:.1f} GB/s)"
                  + (f", keeping up with a {r['overlapped_gvalues_per_s'] * b:.0f} Gb/s link." if b else ".")]
    fused = [r for r in res["rows"] if r["part"] == "fused"]
    if fused:
        f0 = next((r for r in fused if "error" not in r), fused[0])
        lines += ["", f"Whole-document decode, fused one-step codec (Triton, measured): {f0['doc']}, {f0['layers']} layers x "
                  f"{f0['tokens']} tokens = {f0['values'] / 1e6:.0f}M values, encoded and decoded layer by layer as a receiver "
                  "must: one kernel predicts each layer from the decoded previous layer (int8 tensor-core product, RoPE, "
                  "spread), decodes it and makes the next layer's features. Times are the kernels end to end (CUDA "
                  "events); the decode must be bit-exact. Bits are the stream's own (states included); the predictor's "
                  "exact-model bits show the cost of its int8 coefficients and features.", "",
                  "| Tokens per program x rows (warps) | Mode | Stage 1 | Bit-exact | Stream bits/value | Predictor exact-model bits/value | Encoder forward + encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in fused:
            head = f"| {r['tokens_per_program']} x {r['rows']} ({r['warps']}) | {r['mode']} | {'search' if r['search'] else 'direct'} |"
            if "error" in r:
                lines.append(head + " not measured | | | | | | | |")
                continue
            g = r["decode_gvalues_per_s"]
            lines.append(head + f" {r['bit_exact']} | {r['stream_bits_per_value']:.4f} | {r['exact_model_bits_per_value']:.4f} | "
                         f"{r['model_ms']:.1f} + {r['encode_ms']:.1f} | {r['decode_ms']:.1f} | {g:.1f} | {2 * g:.1f} | "
                         f"{g * r['stream_bits_per_value']:.0f} Gb/s |")
    lw = [r for r in res["rows"] if r["part"] == "layerwise"]
    if lw:
        lines += ["", "Whole-document decode, per-layer one-step codec (measured; same document, predictor and checks): "
                  "for each layer a cuBLAS int8 product (the previous layer's 8-bit features times the 8-bit "
                  "coefficients, exact int32), then one Triton kernel of one-warp programs (a token block x one head) "
                  "that scales, rotates, codes and makes the next layer's features. Streams stay open across layers. "
                  "Chunks: the tokens are split into that many parts, each decoded on its own CUDA stream, so one "
                  "part's product runs while another part decodes. Encode time excludes the encoder's forward pass "
                  "(as in the fused table).", "",
                  "| Tokens per program | Chunks | Mode | Stage 1 | Variant | Bit-exact | Stream bits/value | Encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in lw:
            head = (f"| {r['tokens_per_program']} | {r['chunks']} | {r['mode']} | {'search' if r['search'] else 'direct'} | "
                    f"{r.get('variant') or '-'} |")
            if "error" in r:
                lines.append(head + " not measured | | | | | | |")
                continue
            g = r["decode_gvalues_per_s"]
            lines.append(head + f" {r['bit_exact']} | {r['stream_bits_per_value']:.4f} | {r['encode_ms']:.1f} | "
                         f"{r['decode_ms']:.1f} | {g:.1f} | {2 * g:.1f} | {g * r['stream_bits_per_value']:.0f} Gb/s |")
        for r in lw:
            if "coding_only_ms" in r:
                lines += ["", f"Where the decode time goes ({r['tokens_per_program']} tokens per program, {r['chunks']} "
                          f"chunk(s){', ' + r['variant'] if r.get('variant') else ''}): the 27 products alone "
                          f"{r['products_only_ms']:.1f} ms, the 28 coding kernels alone "
                          f"(products precomputed) {r['coding_only_ms']:.1f} ms, both as decoded {r['decode_ms']:.1f} ms; "
                          f"overlap hides {r['products_only_ms'] + r['coding_only_ms'] - r['decode_ms']:.1f} ms."]
    lines += ["", "Reference: " + res["splitzip_reference"] + ".", ""]
    return lines


LINK_BANDS = [(1, "ordinary cloud, storage"), (5, "ordinary cloud"), (10, "consumer, cloud Ethernet"),
              (25, "cloud Ethernet"), (50, "workstation"), (100, "workstation, cross-cluster"),
              (200, "datacenter RDMA, per GPU"), (400, "datacenter RDMA, per GPU"), (800, "next-generation RDMA")]


def mix_lines(out, res):
    """Transfer time of one document's KV over links of several speeds: raw BF16 and SplitZip as references, and
    our codec alone (no SplitZip coding anywhere): its best single predictor for that link, and a per-layer
    choice among our predictors (one step, 8 groups, 16 groups) that minimises max(time on the wire, decode
    time). Speeds are the measured overlapped (predictor and coder on separate CUDA streams) decode rates, bits the
    frontier costs plus the coder's measured stream overhead. Ideal link: time = bits / link speed, decoding
    overlapped with arrival; no network was measured."""
    if not ((out / "lossless.json").is_file() and (out / "frontier.json").is_file()):
        return []
    ll = json.loads((out / "lossless.json").read_text())
    fr = {r["label"]: r for r in json.loads((out / "frontier.json").read_text())["variants"]}
    meta = json.loads((out / "captures" / "meta.json").read_text())
    docs = [d for d in TESTS if d in ll["methods"]["SplitZip-style"]["per_layer"]]
    if not docs:
        return []
    per_layer = lambda label: np.mean([fr[label]["per_layer"][d] for d in docs], 0)
    sz_bits = float(np.mean([ll["methods"]["SplitZip-style"]["per_layer"][d] for d in docs]))
    Ts = 311.0 / 2                                                     # SplitZip native decode on the A4500
    lines = []
    for mode in dict.fromkeys(r.get("mode", "exact") for r in res["rows"] if r["part"] == "overlap"):
        crow = [r for r in res["rows"] if r["part"] == "coder" and r.get("mode", "exact") == mode and r["bit_exact"]]
        if not crow:
            continue
        cbest = max(crow, key=lambda r: r["decode_gvalues_per_s"])
        extra = cbest["stream_bits_per_value"] - cbest["exact_model_bits_per_value"]
        opts = []
        for r in res["rows"]:
            v = r.get("variant") or r.get("predictor", "").split(" (")[0]
            if r["part"] == "overlap" and r.get("mode", "exact") == mode and v in fr:
                opts.append((v, per_layer(v) + extra, r["overlapped_gvalues_per_s"]))
        if not opts:
            continue
        L = len(opts[0][1])
        n = int(meta["tokens"]) * 2 * int(meta["kv_heads"]) * int(meta["head_dim"]) * L
        B = np.stack([o[1] for o in opts])
        C = np.array([1 / o[2] for o in opts])
        K = len(opts)
        dp = {tuple([0] * K): 0.0}                                   # option counts -> fewest total bits
        for l in range(L):
            nxt = {}
            for cnt, bits in dp.items():
                for o in range(K):
                    c2 = list(cnt)
                    c2[o] += 1
                    c2 = tuple(c2)
                    val = bits + B[o, l]
                    if val < nxt.get(c2, float("inf")):
                        nxt[c2] = val
            dp = nxt
        lines += ["", f"Our codec alone ({mode} coder; no SplitZip coding anywhere): transfer time of one document's KV "
                  f"({n / 1e6:.0f}M values), from measured overlapped decode speeds and an ideal link (decoding "
                  "overlapped with arrival; no network measured). Our predictors: " +
                  "; ".join(f"{o[0]} ({float(np.mean(o[1])):.2f} bits/value, {o[2]:.1f} G values/s)" for o in opts) +
                  ". SplitZip and raw BF16 are references only.", "",
                  "| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for R, _band in LINK_BANDS:
            single = min(((max(float(np.mean(o[1])) / R, 1 / o[2]), o[0]) for o in opts))
            best = min(((max(bits / L / R, float(np.dot(cnt, C)) / L), cnt) for cnt, bits in dp.items()))
            t_sz = max(sz_bits / R, 1 / Ts)
            ms = lambda t: f"{t * n / 1e6:.0f} ms"
            counts = ", ".join(f"{c} {opts[i][0].replace('P, ', '')}" for i, c in enumerate(best[1]) if c)
            lines.append(f"| {R} Gb/s | {ms(16 / R)} | {ms(t_sz)} | {ms(single[0])} ({single[1].replace('P, ', '')}) | "
                         f"{ms(best[0])} ({counts}) | {t_sz / best[0]:.2f}x |")
    lines += measured_transfer_lines(out, res, ll, sz_bits)
    return lines + ([""] if lines else [])


def measured_transfer_lines(out, res, ll, sz_bits):
    """Two versions of the measured transfer comparison, each with time and bytes for every link: (1) our codec
    alone; (2) our codec mixed with SplitZip per layer (the measured mixes). SplitZip: the authors' codec as
    measured in this run when available, else its earlier official A4500 run."""
    rows = res["rows"]
    whole = [r for r in rows if r["part"] in ("fused", "layerwise") and r.get("bit_exact")]
    if not whole:
        return []

    def name(r):
        if r["part"] == "fused":
            return f"fused {r['tokens_per_program']}x{r['rows']}, {r['mode']}"
        v = r.get("variant") or ""
        kind = (f"rank {r['rank']}" if r.get("rank") else "reduced rank") if "reduced-rank" in v else "dense"
        return f"per layer, {kind} product, {r['mode']}"

    best = {}
    for r in whole:
        k = name(r)
        if k not in best or r["decode_gvalues_per_s"] > best[k]["decode_gvalues_per_s"]:
            best[k] = r
    cands = list(best.values())
    f0 = cands[0]
    n = f0["values"]
    szr = next((r for r in rows if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    if szr:
        sz, sz_rate, sz_src = szr["stream_bits_per_value"], szr["decode_gvalues_per_s"], \
            f"the authors' codec ({szr['commit'][:7]}) run here: {szr['stream_bits_per_value']:.3f} bits/value, " \
            f"{szr['decode_gbps']:.0f} GB/s decode"
    else:
        sz = (float(np.mean(ll["methods"]["SplitZip-style"]["per_layer"][f0["doc"]]))
              if f0["doc"] in ll["methods"]["SplitZip-style"]["per_layer"] else sz_bits)
        sz_rate, sz_src = 311.0 / 2, f"{sz:.3f} bits/value (SplitZip-style cost), 311 GB/s (official A4500 run)"
    exr = None
    if (out / "exans.json").is_file():
        exr = json.loads((out / "exans.json").read_text())["documents"].get(f0["doc"])
    mib = lambda bits: bits * n / 8 / 2 ** 20
    t = lambda bits, rate, R: max(bits / R, 1 / rate) * n / 1e9          # seconds
    ms = lambda sec: f"{1e3 * sec:.0f} ms"

    def ours_best(R, pool):
        return min(((t(r["stream_bits_per_value"], r["decode_gvalues_per_s"], R), r) for r in pool),
                   key=lambda x: x[0])

    lines = ["", f"### Measured transfer, version 1: our codec alone ({f0['doc']}: {n / 1e6:.0f}M values)", "",
             "Each cell: time to move the document over an ideal link (decoding overlapped with arrival: the longer of "
             "the time on the wire and the decode) and the bytes sent. Our row is the measured mode with the shortest "
             "time for that link, among: " + "; ".join(
                 f"{name(r)} ({r['stream_bits_per_value']:.3f} bits/value, {r['decode_gvalues_per_s']:.1f} G values/s)"
                 for r in cands) + f". SplitZip: {sz_src}." +
             (f" ExANS: the authors' codec run here, {exr['bits_per_value']:.3f} bits/value, "
              f"{exr['decode_gbps_wall']:.0f} GB/s decode." if exr else ""), "",
             "| Link | Typical setting | SplitZip | ExANS | Ours | Mode | Ours: time vs SplitZip | Ours: bytes vs SplitZip | Ours: time vs ExANS |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for R, band in LINK_BANDS:
        to, r = ours_best(R, cands)
        tsz = t(sz, sz_rate, R)
        ob = r["stream_bits_per_value"]
        exc = f"{ms(t(exr['bits_per_value'], exr['decode_gbps_wall'] / 2, R))}, {mib(exr['bits_per_value']):.0f} MiB" if exr else "not run"
        exv = f"{t(exr['bits_per_value'], exr['decode_gbps_wall'] / 2, R) / to:.2f}x" if exr else "-"
        lines.append(f"| {R} Gb/s | {band} | {ms(tsz)}, {mib(sz):.0f} MiB | {exc} | {ms(to)}, {mib(ob):.0f} MiB | "
                     f"{name(r)} | {tsz / to:.2f}x | {100 * (1 - ob / sz):.0f}% fewer | {exv} |")
    worst = [R for R, _ in LINK_BANDS if ours_best(R, cands)[0] > t(sz, sz_rate, R)]
    if worst:
        R = worst[0]
        to, r = ours_best(R, cands)
        lines += ["", f"Where our codec alone is slower (from {R} Gb/s), it is our decode that sets the time, not the "
                  f"link: at {R} Gb/s it takes {ms(to)} against SplitZip's {ms(t(sz, sz_rate, R))}, but the link is busy "
                  f"for only {ms(r['stream_bits_per_value'] * n / (R * 1e9))} and carries "
                  f"{100 * (1 - r['stream_bits_per_value'] / sz):.0f}% fewer bytes ({mib(r['stream_bits_per_value']):.0f} "
                  f"against {mib(sz):.0f} MiB): a shared or metered link carries more of everything else."]
    mixed = [r for r in rows if r["part"] == "mixed" and r.get("bit_exact")]
    if mixed:
        lines += ["", f"### Measured transfer, version 2: our codec + SplitZip per layer", "",
                  "The layers where ours saves the most are coded by ours, the rest by the authors' SplitZip codec; "
                  "every mix is a real stream, decoded bit-exact and timed. Measured mixes: " + ", ".join(
                      f"{r['mode']} {r['ours_layers']}/{len(r['ours_mask'])} ({r['stream_bits_per_value']:.2f} bits, "
                      f"{r['decode_ms']:.1f} ms)" for r in sorted(mixed, key=lambda r: (r["mode"], r["ours_layers"]))) +
                  ". With 0 layers of ours the mix is SplitZip itself, so the best mix is never slower than SplitZip.",
                  "", "| Link | SplitZip | Ours alone | Ours + SplitZip mix | Layers ours (mode) | Mix: time vs SplitZip | Mix: bytes vs SplitZip |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for R, _band in LINK_BANDS:
            tsz = t(sz, sz_rate, R)
            to, r1 = ours_best(R, cands)
            tm, rm = ours_best(R, mixed + cands)
            lay = f"{rm['ours_layers']} of {len(rm['ours_mask'])} ({rm['mode']})" if rm["part"] == "mixed" else \
                f"all ({name(rm)})"
            lines.append(f"| {R} Gb/s | {ms(tsz)}, {mib(sz):.0f} MiB | {ms(to)}, {mib(r1['stream_bits_per_value']):.0f} MiB | "
                         f"{ms(tm)}, {mib(rm['stream_bits_per_value']):.0f} MiB | {lay} | {tsz / tm:.2f}x | "
                         f"{100 * (1 - rm['stream_bits_per_value'] / sz):.0f}% fewer |")
    return lines


def splitzip_mix_lines(out, res):
    """A separate, clearly labelled hybrid (not our codec alone): every layer is coded either by our measured
    per-layer codec or by SplitZip's own format, chosen per link speed. Estimated from measured parts: our mode's
    decode time per layer (whole-document decode / layers), its measured bits per layer, SplitZip's bits per layer
    on the same document and its measured A4500 decode speed, plus a measured features pass for every
    SplitZip-coded layer (our next layer predicts from features). Ideal link, decoding overlapped with arrival."""
    if not (out / "lossless.json").is_file():
        return []
    ll = json.loads((out / "lossless.json").read_text())
    sz_layers = ll["methods"].get("SplitZip-style", {}).get("per_layer", {})
    rows = [r for r in res["rows"] if r["part"] == "layerwise" and r.get("bit_exact") and r.get("layer_bits_per_value")
            and r["doc"] in sz_layers]
    if not rows:
        return []
    fp = [r for r in res["rows"] if r["part"] == "feature_pass"]
    t_feat = fp[0]["ms_per_layer"] / 1e3 if fp else 0.0
    Ts = 311.0 / 2 * 1e9                                                # SplitZip native decode, values per second
    best = {}
    for r in rows:                                   # per product kind and mode, the fastest measured row
        k = (r.get("rank"), r["mode"])
        if k not in best or r["decode_gvalues_per_s"] > best[k]["decode_gvalues_per_s"]:
            best[k] = r
    doc = rows[0]["doc"]
    z = np.array(sz_layers[doc], dtype=float)
    L = len(z)
    nl = rows[0]["values"] / L
    t_z = nl / Ts                                    # a SplitZip layer; a features pass only before one of ours

    def kind(r):
        return f"{'rank ' + str(r['rank']) if r.get('rank') else 'dense'}, {r['mode']}"

    def plan(r, R):
        b = np.array(r["layer_bits_per_value"], dtype=float)
        b += r["stream_bits_per_value"] - b.mean()                      # starts and alignment, spread evenly
        t_m = r["decode_ms"] / 1e3 / L
        order = np.argsort(-(z - b))                                    # layers where ours saves the most first
        out_ = []
        for n in range(L + 1):
            mine = np.zeros(L, bool)
            mine[order[:n]] = True
            bits = float(np.where(mine, b, z).sum()) * nl
            ours_after_sz = int(sum(1 for l in range(1, L) if mine[l] and not mine[l - 1]))
            dec = n * t_m + (L - n) * t_z + ours_after_sz * t_feat
            out_.append((max(bits / (R * 1e9), dec), n, bits / (nl * L), dec))
        return min(out_)

    lines = ["", f"Hybrid with SplitZip's format (not our codec alone; estimated from measured parts): each of the {L} "
             f"layers of {doc} coded either by our per-layer codec or by SplitZip's format, chosen per link to minimise "
             "the transfer. Our modes: " + "; ".join(f"{kind(r)} ({r['stream_bits_per_value']:.3f} bits/value, "
                                                      f"{r['decode_ms']:.1f} ms)" for r in best.values()) +
             f". SplitZip: {float(z.mean()):.3f} bits/value, 311 GB/s decode, plus a features pass of "
             f"{1e3 * t_feat:.3f} ms per SplitZip layer. Not modelled: a layer after a SplitZip layer has no "
             "surprise prior (slightly more bits), and SplitZip layers could also decode beside ours.", "",
             "| Link | SplitZip | Ours alone (measured) | Hybrid | Layers ours (mode) | Hybrid bits/value | Hybrid vs SplitZip |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    n_all = rows[0]["values"]
    for R, _band in LINK_BANDS:
        t_sz = max(float(z.mean()) * n_all / (R * 1e9), n_all / Ts)
        alone = min(max(r["stream_bits_per_value"] / (R * 1e9), r["decode_ms"] / 1e3 / n_all) * n_all for r in best.values())
        t, n, bits, _dec, r = min((plan(r, R) + (r,) for r in best.values()), key=lambda x: x[0])
        lines.append(f"| {R} Gb/s | {1e3 * t_sz:.1f} ms | {1e3 * alone:.1f} ms | {1e3 * t:.1f} ms | {n} ({kind(r)}) | "
                     f"{bits:.3f} | {t_sz / t:.2f}x |")
    return lines + [""]


def lossy_link_lines(out):
    """Transfer of one whole document's KV per link speed, lossless and lossy methods side by side, with quality:
    time = max(bits / link, decode time) where the decode was measured on this GPU (ours, SplitZip), else the
    time on the wire alone (a lower bound for that method). Bits and quality from the lossy stage."""
    if not ((out / "lossy.json").is_file() and (out / "gpu_speed.json").is_file()):
        return []
    lres = json.loads((out / "lossy.json").read_text())
    rows = json.loads((out / "gpu_speed.json").read_text())["rows"]
    docs = list(lres["documents"])
    whole = [r for r in rows if r["part"] in ("fused", "layerwise") and r.get("bit_exact")]
    if not whole:
        return []
    n = whole[0]["values"]
    lossy_speed = {}
    for r in rows:                                   # GPU lossy codec: dense product, the lossy stage's shape
        if r["part"] == "lossy_layerwise" and r.get("bit_exact") and r.get("rank") is None:
            k = float(r["kappa"])
            if k not in lossy_speed or r["decode_gvalues_per_s"] > lossy_speed[k]:
                lossy_speed[k] = r["decode_gvalues_per_s"]
    links = (1, 10, 25, 100, 200, 400)
    ms = lambda t: f"{1e3 * t:,.0f}"
    lines = ["", f"## Lossless and lossy side by side ({whole[0]['doc']}: {n / 1e6:.0f}M values per link)", "",
             "Time to move one document's KV over an ideal link, decoding overlapped with arrival: max(bits / link, "
             "decode time) with the decode measured on this GPU (AQUA-KV and KIVI with the authors' code); a * marks "
             "a method whose decode was not measured (its time on the wire only, so a lower bound). MiB = bytes sent "
             "for the document. Bits and quality of the lossy rows from the lossy stage "
             f"(first {lres['prompt_tokens']} tokens of each test document; quality = perplexity change of the next "
             f"{lres['continuation_tokens']} tokens, {' / '.join(docs)}).", "",
             "| Method | Bits per value | MiB | Perplexity change | " + " | ".join(f"{R} Gb/s (ms)" for R in links) + " |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in links) + " |"]
    mibs = lambda bits: f"{bits * n / 8 / 2 ** 20:.0f}"

    def times(bits, rate):
        cells = []
        for R in links:
            t = bits * n / (R * 1e9)
            if rate:
                t = max(t, n / (rate * 1e9))
            cells.append(ms(t) + ("" if rate else "*"))
        return cells

    sz = 12.021
    if (out / "lossless.json").is_file():
        ll = json.loads((out / "lossless.json").read_text())["methods"].get("SplitZip-style")
        if ll and whole[0]["doc"] in ll.get("per_layer", {}):
            sz = float(np.mean(ll["per_layer"][whole[0]["doc"]]))
    szr = next((r for r in rows if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    sz_rate = szr["decode_gvalues_per_s"] if szr else 311.0 / 2
    sz = szr["stream_bits_per_value"] if szr else sz
    lines.append(f"| SplitZip (lossless) | {sz:.2f} | {mibs(sz)} | 0 | " + " | ".join(times(sz, sz_rate)) + " |")
    best = []
    for R in links:                                  # ours lossless: the best measured mode for each link
        best.append(min(max(r["stream_bits_per_value"] * n / (R * 1e9), n / (r["decode_gvalues_per_s"] * 1e9))
                        for r in whole))
    lo, hi = min(r["stream_bits_per_value"] for r in whole), max(r["stream_bits_per_value"] for r in whole)
    lines.append(f"| Ours, lossless (best measured mode per link) | {lo:.2f}-{hi:.2f} | {mibs(lo)}-{mibs(hi)} | 0 | " +
                 " | ".join(ms(t) for t in best) + " |")
    for label in lres["documents"][docs[0]]:
        if label == "exact KV":
            continue
        rs = [lres["documents"][d][label] for d in docs if label in lres["documents"][d]]
        bits = float(np.mean([x["bits"] for x in rs]))
        ppl = " / ".join(f"{x['ppl_change_pct']:+.2f}%" for x in rs)
        rate = None
        for name, path in (("AQUA-KV", "time_aqua.json"), ("KIVI", "time_kivi.json")):
            if label.startswith(name + ",") and (out / path).is_file():
                tb = json.loads((out / path).read_text())["bits"].get(label.split(", ")[1].split("-")[0])
                if tb:
                    rate = tb["decode_gvalues_per_s"]
        if label.startswith("Ours, GPU codec"):
            rate = lossy_speed.get(float(label.split("step ", 1)[1].split(",")[0]))
        lines.append(f"| {label} | {bits:.2f} | {mibs(bits)} | {ppl} | " + " | ".join(times(bits, rate)) + " |")
    return lines + [""]


def frontier_lines(out):
    res = json.loads((out / "frontier.json").read_text())
    lines = [f"## Bits against compute ({len(res['variants'])} predictor variants, {res['layers']} layers)", "",
             "Lossless, exact costs like the table above. Multiply-adds per value include the shared projection "
             "of reduced-rank variants and, for the attention input, the query map and the attention itself. "
             "Integer formats: int8 x int16 needs 2 and int8 x int8 1 eight-bit tensor-core product per "
             "multiply-add, and the result is exact on any GPU. P = Pareto point among the floating-point "
             "variants (no cheaper variant has fewer bits).", "",
             "| Variant | Multiply-adds per value | 8-bit products per value | Four-chunk average | PEP 8 / argparse.py whole | |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in res["variants"]:
        prod = f"{r['macs_per_value'] * r['products_per_mac']:,.0f}" if r["products_per_mac"] else "float"
        w = r["whole"]
        lines.append(f"| {r['label']} | {r['macs_per_value']:,.0f} | {prod} | {r['chunk_average']:.3f} | "
                     f"{w.get('test_text', float('nan')):.3f} / {w.get('test_code', float('nan')):.3f} | "
                     f"{'P' if r['pareto'] else ''} |")
    return lines + [""]


def eval_lines(out):
    """The larger quality test (lossy --lossy-docs eval): every method on the WikiText-2 windows, averaged."""
    res = json.loads((out / "lossy_eval.json").read_text())
    docs = list(res["documents"])
    if not docs:
        return []
    em = out / "captures" / "eval_meta.json"
    src = json.loads(em.read_text()) if em.is_file() else {}
    n = len(docs)
    lines = [f"## Lossy quality on WikiText-2 ({n} windows; prompt {res['prompt_tokens']} tokens, perplexity of the "
             f"next {res['continuation_tokens']} tokens)", "",
             f"Windows of {src.get('tokens', '?')} consecutive tokens of {Path(src.get('text', 'WikiText-2')).name} "
             "(the raw test split joined with blank lines, as AQUA-KV's evaluation reads it), each starting with the BOS "
             f"token; {n * res['continuation_tokens']:,} scored tokens in all. Pooled change = perplexity over all scored "
             "tokens against the exact KV; per-window = mean ± standard error of the windows' own changes (the spread "
             "says how far a difference between two rows can be trusted).", "",
             "| Method | Bits per value | KV relative error | Perplexity change, pooled | Per-window mean ± SE | "
             "Windows worse |", "| --- | --- | --- | --- | --- | --- |"]
    labels = [k for k in res["documents"][docs[0]] if k != "exact KV" and all(k in res["documents"][d] for d in docs)]
    for label in labels:
        r = [res["documents"][d][label] for d in docs]
        base = [res["documents"][d]["exact KV"]["nll"] for d in docs]
        pooled = 100 * (math.exp(float(np.mean([x["nll"] for x in r])) - float(np.mean(base))) - 1)
        ch = np.array([x["ppl_change_pct"] for x in r])
        se = float(ch.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        lines.append(f"| {label} | {np.mean([x['bits'] for x in r]):.3f} | {100 * np.mean([x['rel_error'] for x in r]):.2f}% | "
                     f"{pooled:+.2f}% | {ch.mean():+.2f}% ± {se:.2f} | {int((ch > 0).sum())} of {n} |")
    lines += ["", "Same methods and settings as the two-document table above (AQUA-KV and KIVI as the authors' code "
              "has them; our variants: sinks = the first tokens exact, token step = one step per token)."]
    return lines + [""]


def stage_report(args, out):
    lines = [f"# KV transfer codecs on {args.model}", ""]
    meta = json.loads((out / "captures" / "meta.json").read_text())
    lines.append(f"Captures: {len(meta['captures'])} documents x {meta['tokens']} tokens; {meta['layers']} layers, "
                 f"{meta['kv_heads']} KV heads x {meta['head_dim']}; torch {meta['torch']}, transformers {meta['transformers']}.")
    lines.append("")
    if (out / "lossless.json").is_file():
        res = json.loads((out / "lossless.json").read_text())
        lines += ["## Lossless (bits per value; BF16 = 16)", "",
                  "| Method | Four-chunk average | PEP 8 whole | argparse.py whole | Predictor multiply-adds per value |",
                  "| --- | --- | --- | --- | --- |"]
        for name, m in res["methods"].items():
            w = m["whole"]
            lines.append(f"| {name} | {m['chunk_average']:.3f} | {w.get('test_text', float('nan')):.3f} | "
                         f"{w.get('test_code', float('nan')):.3f} | {m['macs_per_value']:,.0f} |")
            if name == "ExANS-style" and (out / "exans.json").is_file():
                ex = json.loads((out / "exans.json").read_text())["documents"]
                g = lambda d: f"{ex[d]['bits_per_value']:.3f}" if d in ex else "-"
                lines.append(f"| ExANS, official codec (measured stream) | - | {g('test_text')} | {g('test_code')} | 0 |")
        lines.append("")
        ex = json.loads((out / "exans.json").read_text()) if (out / "exans.json").is_file() else None
        if ex and not ex["documents"]:
            lines += ["ExANS official: not measured (" + (ex.get("error") or str(ex.get("errors")))[-300:] + ")", ""]
        elif ex:
            r0 = next(iter(ex["documents"].values()))
            lines += [f"ExANS official: {ex['build']['commit'][:7]} built with one fix ({ex['build']['build_fix']}); "
                      f"the authors' test passed; {r0['blocks']} blocks of 499,712 values, all bit-exact: "
                      + ", ".join(f"{d} {v['bits_per_value']:.3f} bits/value, decode {v['decode_gbps_wall']:.0f} GB/s "
                                  f"(encode {v['encode_gbps_wall']:.0f} GB/s)" for d, v in ex["documents"].items())
                      + " (wall clock of the codec calls, raw BF16 bytes).", ""]
    if (out / "frontier.json").is_file():
        lines += frontier_lines(out)
    if (out / "gpu_speed.json").is_file():
        lines += speed_lines(out)
        lines += mix_lines(out, json.loads((out / "gpu_speed.json").read_text()))
        _g = json.loads((out / "gpu_speed.json").read_text())
        if not any(r["part"] == "mixed" for r in _g["rows"]):
            lines += splitzip_mix_lines(out, _g)
    if (out / "pool.json").is_file():
        lines += pool_lines(out)
    if (out / "format.json").is_file():
        lines += format_lines(out)
    if (out / "szcodec.json").is_file():
        lines += szcodec_lines(out)
    if (out / "gpu_speed.json").is_file() or (out / "szcodec.json").is_file():
        lines += bytes_time_lines(out)
    lines += lossy_link_lines(out)
    if (out / "cheap_products.json").is_file():
        cp = json.loads((out / "cheap_products.json").read_text())
        lines += ["## Cheaper one-step products (scripts/probe_cheap_products.py)", "",
                  f"Lossless bits with a cheaper product in the one-step predictor (layers 1-{cp['layers'] - 1}; layer 0 "
                  "is predicted by its mean in every variant). Halving the product is for fast links, where decode "
                  "speed matters more than bits.", "",
                  "| Product | Multiply-adds per token | Bits per value (text / code) | vs dense int8 |",
                  "| --- | --- | --- | --- |"]
        for name, v in cp["variants"].items():
            lines.append(f"| {name} | {v['macs_per_token']:,} | {v['bits']['test_text']:.3f} / {v['bits']['test_code']:.3f} | "
                         f"{v['delta']:+.3f} |")
        lines.append("")
    if (out / "lossy.json").is_file():
        res = json.loads((out / "lossy.json").read_text())
        docs = list(res["documents"])
        lines += [f"## Lossy (prompt {res['prompt_tokens']} tokens; quality = perplexity of the next "
                  f"{res['continuation_tokens']} tokens with the transferred KV)", "",
                  "| Method | Bits per value | KV relative error | Perplexity change |", "| --- | --- | --- | --- |"]
        for label in res["documents"][docs[0]]:
            r = [res["documents"][d][label] for d in docs]
            lines.append(f"| {label} | " + " / ".join(f"{x['bits']:.2f}" for x in r) + " | " +
                         " / ".join(f"{100 * x['rel_error']:.2f}%" for x in r) + " | " +
                         " / ".join(f"{x['ppl_change_pct']:+.2f}%" for x in r) + " |")
        lines += ["", f"Values are {' / '.join(docs)}. Reference: the captured exact KV restored into the cache."]
        from kv_compare import baselines as _bl
        lines.append(f"AQUA-KV follows the authors' code ({_bl.AQUA_SOURCE}): HIGGS residuals on 2-D EDEN grids, "
                     "predictors fitted on their own closed-loop reconstructions, layer 0 at 4 bits, the first "
                     f"{_bl.AQUA_PREFIX} tokens in BF16; calibrated on this study's calibration documents (the authors "
                     "use 256 x 8192 RedPajama tokens). KIVI follows the authors' code "
                     f"({_bl.KIVI_SOURCE}) with the paper's group size {_bl.KIVI_GROUP} and residual length "
                     f"{_bl.KIVI_RESIDUAL} (recent tokens in BF16); its scales and minima are counted at 16 bits.")
        for name, path in (("AQUA-KV", "check_aqua.json"), ("KIVI", "check_kivi.json")):
            if (out / path).is_file():
                c = json.loads((out / path).read_text())
                if name == "AQUA-KV":
                    h = c["higgs"]
                    lines.append("Check against the AQUA-KV code (" + c["source_commit"][:7] + ", torch " + c["torch"] +
                                 "): HIGGS values identical for " + ", ".join(
                                     f"{100 * v['values_equal_fraction']:.2f}% ({k})" for k, v in h.items()) +
                                 " (relative error ours / theirs " + ", ".join(
                                     f"{v['relative_error_ours']:.4f} / {v['relative_error_theirs']:.4f}" for v in h.values()) +
                                 f"); regression weights within {c['regression_max_abs_diff_weight']:.2g}.")
                else:
                    lines.append("Check against the KIVI code (" + c["source_commit"][:7] + "): largest difference " +
                                 ", ".join(f"{k}: {max(v['keys_max_abs_diff_over_rms'], v['values_max_abs_diff_over_rms']):.2g} RMS"
                                           for k, v in c["cases"].items()) + ".")
        own = [res["documents"][d]["exact KV"].get("model_prefill_ppl_change_pct") for d in docs]
        if all(x is not None for x in own):
            lines.append("The model's own prefill of the prompt alone differs from it by " +
                         " / ".join(f"{x:+.3f}%" for x in own) + " (GPU kernels round differently at another sequence length).")
    if (out / "lossy_eval.json").is_file():
        lines += [""] + eval_lines(out)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face revision; 'none' for the latest")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--texts", type=Path, default=TEXTS)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--stages", default="capture,lossless,lossy,report")
    p.add_argument("--configs", default="prev,block,nohint,fast")
    p.add_argument("--lossy-config", default="fast")
    p.add_argument("--kappas", default="0.25,0.5,1,2,4")
    p.add_argument("--prompt-tokens", type=int, default=3072)
    p.add_argument("--continuation-tokens", type=int, default=512)
    p.add_argument("--max-layers", type=int, default=None, help="lossless and frontier stages: first N layers (smoke tests)")
    p.add_argument("--no-queries", action="store_true", help="skip q_proj capture (drops the attention input)")
    p.add_argument("--eval-text", type=Path, default=None, help="evalcap: the evaluation text (WikiText-2 raw test)")
    p.add_argument("--eval-docs", type=int, default=8, help="evalcap / lossy --lossy-docs eval: evaluation windows")
    p.add_argument("--eval-tokens", type=int, default=4096, help="evalcap: tokens per evaluation window")
    p.add_argument("--eval-prompt-tokens", type=int, default=3072, help="lossy --lossy-docs eval: transferred prompt")
    p.add_argument("--eval-continuation-tokens", type=int, default=1024,
                   help="lossy --lossy-docs eval: continuation tokens scored")
    p.add_argument("--lossy-docs", default="test", choices=["test", "eval"],
                   help="lossy: the two test documents (lossy.json) or the evalcap windows (lossy_eval.json)")
    p.add_argument("--lossy-variants", default="base,sinks,token,both",
                   help="lossy stage, ours and gpu: base; sinks (the first --sink-tokens tokens exact); token (the "
                        "step follows the token's adaptive spread); both")
    p.add_argument("--sink-tokens", type=int, default=4, help="lossy variants: tokens kept exact (AQUA-KV's 4)")
    p.add_argument("--lossy-methods", default="ours,aqua,kivi,fp8,gpu",
                   help="lossy stage: methods to (re)run; a subset is merged into the existing lossy.json")
    p.add_argument("--frontier-quick", action="store_true", help="frontier stage: a smaller grid of variants")
    p.add_argument("--speed-tokens", type=int, default=8192, help="gpu_speed: tokens per timed layer")
    p.add_argument("--speed-formats", default="fp32,bf16,int8x16,int8x8,int8x8s")
    p.add_argument("--speed-config", default="lite", help="gpu_speed: configuration whose predictions feed the coder")
    p.add_argument("--speed-layers", default="4,12,20,27", help="gpu_speed: test-document layers coded")
    p.add_argument("--speed-modes", default="exact,uniform",
                   help="gpu_speed: coder modes; uniform gives every word of a slice an equal share (faster, a few "
                        "hundredths of a bit more)")
    p.add_argument("--speed-settings", default="1024x128x1,1024x256x1,768x128x1,768x256x1,512x128x1",
                   help="gpu_speed: coder settings, values per stream x lanes per program [x warps per program; "
                        "default one thread per lane, fewer warps interleave several lanes per thread]")
    p.add_argument("--fused-configs", default="32x8x8,64x8x8,32x4x4,16x4x4,64x16x16",
                   help="gpu_speed: fused one-step codec on the whole test document, tokens per program x rows x "
                        "warps (empty: skip)")
    p.add_argument("--layer-configs", default="128x1,128x2,128x4,64x2,256x2,32x2",
                   help="gpu_speed: per-layer one-step codec (cuBLAS product + one coding kernel per layer), tokens "
                        "per program x token chunks decoded on separate CUDA streams (empty: skip)")
    p.add_argument("--pool-fused", default="32x4x4", help="pool: fused codec shape (TBxRxW)")
    p.add_argument("--pool-layer", default="128x2", help="pool: per-layer codec shape (TBxC)")
    p.add_argument("--pool-tokens", default="2048,8192", help="pool: prefix lengths for the recompute baseline")
    p.add_argument("--exans-source", type=Path,
                   default=Path.home() / "kv_reconstruction/runs/_llama_official/sources/exans",
                   help="exans: an openlake checkout (cloned into the run folder if missing)")
    p.add_argument("--cuda-home", type=Path, default=None,
                   help="exans: CUDA toolkit with nvcc (default: $CUDA_HOME, the earlier official run's toolkit, "
                        "or the nvcc on PATH)")
    p.add_argument("--splitzip-source", type=Path,
                   default=Path.home() / "kv_reconstruction/runs/_llama_official/sources/splitzip",
                   help="gpu_speed: a checkout of the authors' SplitZip (cloned into the run folder if missing)")
    p.add_argument("--mix-counts", default="0,2,4,6,8,10,12,14,16,20,24,28",
                   help="gpu_speed: numbers of layers coded by ours in the measured SplitZip mixes (empty: skip)")
    p.add_argument("--lossy-speed-kappas", default="0.25,0.5,1,2,4",
                   help="gpu_speed: steps of the GPU lossy codec timed on the whole test document (empty: skip)")
    p.add_argument("--sz-ranks", default="0,256,512",
                   help="szcodec: products of the SplitZip-format codec (0 = dense, else reduced rank)")
    p.add_argument("--sz-configs", default="32x4",
                   help="szcodec, tile layout: tokens per program x warps (a program is that many tokens x one head)")
    p.add_argument("--sz-row-configs", default="16x1,32x1",
                   help="szcodec, row layouts (two reads per value, and one): tokens per program x warps")
    p.add_argument("--sz-chunks", default="2,4",
                   help="szcodec: token chunks on CUDA streams, tried for the fastest of each layout")
    p.add_argument("--format-ranks", default="256",
                   help="format: reduced-rank products to probe besides the dense one (empty: dense only)")
    p.add_argument("--layer-ranks", default="512,384,256",
                   help="gpu_speed: reduced-rank products to try in the per-layer codec (the fast-link mode): each "
                        "with its requantization in a kernel and in the first product's epilogue (3 tile shapes)")
    p.add_argument("--fused-modes", default="uniform,exact",
                   help="gpu_speed: fused codec modes (every config in the first; the fastest also in the others)")
    return p


def gpu_sharing_note(device):
    """Free and total memory of the GPU; other processes on it make memory tight and timings meaningless."""
    try:
        import torch
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(torch.device(device))
        free += torch.cuda.memory_reserved(torch.device(device))   # cached by this process (an earlier stage)
        note = f"GPU memory free {free / 2 ** 30:.1f} of {total / 2 ** 30:.1f} GB at start"
        if free < 0.8 * total:
            note += (" - WARNING: other processes are using this GPU; memory may run out and any timing is "
                     "not representative (check nvidia-smi)")
        return note
    except Exception:  # pragma: no cover - informational only
        return None


def main(argv=None):
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parser().parse_args(argv)
    if Path(args.model).is_dir():          # a local checkout has no Hugging Face revision
        args.revision = None
    args.configs = [c for c in args.configs.split(",") if c]
    args.kappas = [float(k) for k in args.kappas.split(",") if k]
    args.lossy_methods = [m for m in args.lossy_methods.split(",") if m]
    args.lossy_variants = [v for v in args.lossy_variants.split(",") if v]
    if not set(args.lossy_variants) <= {"base", "sinks", "token", "both"}:
        raise SystemExit("--lossy-variants takes a comma list of base, sinks, token, both")
    if not set(args.lossy_methods) <= {"ours", "aqua", "kivi", "fp8", "gpu"}:
        raise SystemExit("--lossy-methods takes a comma list of ours, aqua, kivi, fp8, gpu")
    args.speed_formats = [f for f in args.speed_formats.split(",") if f]
    args.speed_modes = [m for m in args.speed_modes.split(",") if m]
    args.speed_layers = [int(l) for l in args.speed_layers.split(",") if l]
    args.speed_settings = [tuple(int(x) for x in t.split("x")) for t in args.speed_settings.split(",") if t]
    if any(len(t) not in (2, 3) for t in args.speed_settings):
        raise SystemExit("--speed-settings takes VxB or VxBxW items (values per stream x lanes x warps)")
    args.fused_configs = [tuple(int(x) for x in t.split("x")) for t in args.fused_configs.split(",") if t]
    if any(len(t) != 3 for t in args.fused_configs):
        raise SystemExit("--fused-configs takes TBxRxW items (tokens per program x rows x warps)")
    args.fused_modes = [m for m in args.fused_modes.split(",") if m]
    args.layer_configs = [tuple(int(x) for x in t.split("x")) for t in args.layer_configs.split(",") if t]
    args.layer_ranks = [int(r) for r in args.layer_ranks.split(",") if r]
    args.format_ranks = [int(r) for r in args.format_ranks.split(",") if r]
    args.sz_ranks = [int(r) for r in args.sz_ranks.split(",") if r]
    args.sz_configs = [tuple(int(x) for x in t.split("x")) for t in args.sz_configs.split(",") if t]
    args.sz_row_configs = [tuple(int(x) for x in t.split("x")) for t in args.sz_row_configs.split(",") if t]
    args.sz_chunks = [int(c) for c in args.sz_chunks.split(",") if c]
    args.lossy_speed_kappas = [float(k) for k in args.lossy_speed_kappas.split(",") if k]
    args.mix_counts = [int(c) for c in args.mix_counts.split(",") if c]
    if args.splitzip_source is not None and not (args.splitzip_source / "codec_gpu.py").is_file():
        args.splitzip_source = None
    if args.cuda_home is None:
        import shutil as _sh
        cand = [os.environ.get("CUDA_HOME"), str(Path.home() / "kv_reconstruction/runs/_matched_baselines/cuda12_8/toolkit")]
        nvcc = _sh.which("nvcc")
        cand.append(str(Path(nvcc).resolve().parents[1]) if nvcc else None)
        args.cuda_home = next((Path(c) for c in cand if c and (Path(c) / "bin/nvcc").exists()), None)
    if args.exans_source is not None and not (args.exans_source / ".git").exists():
        args.exans_source = None
    args.pool_fused = tuple(int(x) for x in args.pool_fused.split("x"))
    args.pool_layer = tuple(int(x) for x in args.pool_layer.split("x"))
    args.pool_tokens = [int(t) for t in args.pool_tokens.split(",") if t]
    if any(len(t) != 2 for t in args.layer_configs):
        raise SystemExit("--layer-configs takes TBxC items (tokens per program x chunks)")
    stages = [s for s in args.stages.split(",") if s]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; this script has {', '.join(STAGES)}. "
                         f"Check the branch: git rev-parse --abbrev-ref HEAD (expected {BRANCH})")
    out = args.output
    if "capture" in stages:
        if (out / "captures").exists() and any((out / "captures").iterdir()):
            raise SystemExit(f"{out}/captures is not empty: the capture stage already ran. Run the remaining stages "
                             f"with e.g. --stages lossy,report (or lossless,lossy,report), or use a fresh --output")
        out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    global LOG_PATH
    LOG_PATH = out / "run.log"
    started = time.time()
    (out / "run_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    env = environment(args)
    env["stages"] = stages
    (out / "environment.json").write_text(json.dumps(env, indent=1))
    log(f"run started: stages {stages}; {env.get('gpu', env['device'])}; branch {env.get('git_branch')}, "
        f"commit {env.get('git_commit')}; torch {env['torch']}, triton {env.get('triton')}")
    if env.get("git_branch") not in (BRANCH, None, "HEAD"):
        log(f"WARNING: running from branch {env.get('git_branch')}, not {BRANCH}")
    sharing = gpu_sharing_note(args.device)
    if sharing:
        log(sharing)
        env["gpu_memory_at_start"] = sharing
        (out / "environment.json").write_text(json.dumps(env, indent=1))

    def stop(signum, frame):
        raise SystemExit(f"stopped by signal {signal.Signals(signum).name}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        if "capture" in stages:
            stage_capture(args, out)
        if "evalcap" in stages:
            stage_evalcap(args, out)
        if "lossless" in stages:
            stage_lossless(args, out)
        if "lossy" in stages:
            stage_lossy(args, out)
        if "frontier" in stages:
            stage_frontier(args, out)
        if "gpu_speed" in stages:
            stage_gpu_speed(args, out)
        if "pool" in stages:
            stage_pool(args, out)
        if "format" in stages:
            stage_format(args, out)
        if "szcodec" in stages:
            stage_szcodec(args, out)
        if "exans" in stages:
            stage_exans(args, out)
        if "report" in stages:
            stage_report(args, out)
    except BaseException as exc:
        log("FAILED: " + "".join(traceback.format_exception(exc)).rstrip())
        log(f"memory at failure: {memory_note(args.device)}")
        log("finished stages keep their files; rerun only the rest with --stages, e.g. --stages lossy,report")
        raise
    log(f"run finished in {time.time() - started:.0f} s")
    log(f"share it: python scripts/package_llama_compare.py {out} --destination benchmark_results/<name>")


if __name__ == "__main__":
    main()
