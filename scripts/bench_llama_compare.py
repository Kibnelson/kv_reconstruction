#!/usr/bin/env python3
"""Lossless and lossy KV transfer codecs on one model's KV: ours against SplitZip, ExANS, AQUA-KV and KIVI styles.

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


def environment(args):
    """Versions, device and source commit, recorded with every run."""
    import platform
    import subprocess
    import torch
    import transformers
    info = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "transformers": transformers.__version__, "numpy": np.__version__,
            "device": args.device}
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
    try:
        info["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
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
    study = Study(out / "captures", args.device, TRAIN, "tables", TESTS, None, log=log)
    cfg = CONFIGS[args.lossy_config]
    if "ATT" in cfg.kinds and any(d.Q is None for d in study.train):
        log(f"{args.lossy_config} needs calibration queries; using fast_noatt")
        cfg = CONFIGS["fast_noatt"]
    import gc
    device = torch.device(args.device)
    P, M = args.prompt_tokens, args.continuation_tokens

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

    log(f"fitting our predictor ({cfg.name}) for closed-loop coding")
    fits = {}
    study.keep_qmaps = True              # the closed loop needs every layer's query map; keep them (~0.7 GB)
    study.lossless(cfg, fits)
    study._f.clear(); study._att.clear()
    log(f"fitting AQUA-KV-style predictors ({memory_note(args.device)})")
    aqua = baselines.aqua_fit(study, study.L, log)
    study._f.clear(); study._att.clear()
    gc.collect()
    if study.device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"loading {args.model} for the quality check ({memory_note(args.device)})")
    _, model = load_model(args.model, args.revision, device)
    log(f"model loaded ({memory_note(args.device)})")
    result = {"prompt_tokens": P, "continuation_tokens": M, "lossy_config": cfg.name,
              "reference": "captured exact KV restored into the cache", "documents": {}}
    for doc in study.tests:
        base, own, own_pct = refs[doc.name]
        check = continuation_nll(model, doc.ids, P, M, device, matrices_from_doc(doc, P))
        if check != base:                     # every row below is measured with this loaded model
            log(f"{doc.name}: after reloading the model the exact-KV NLL is {check} (was {base}); using {check}")
            base = check
        exact = torch.stack([words_to_float(doc.words(l, study.device, P)).to(torch.float32).cpu() for l in range(study.L)])
        rows = {"exact KV": {"bits": 16.0, "nll": base, "ppl": float(np.exp(base)), "ppl_change_pct": 0.0, "rel_error": 0.0,
                             "model_prefill_nll": own, "model_prefill_ppl_change_pct": own_pct}}
        methods = [(f"Ours, step {k}", lambda k=k: study.lossy(cfg, fits, doc, P, k)) for k in args.kappas]
        methods += [(f"AQUA-KV-style, {b}-bit", lambda b=b: baselines.aqua_style(study, aqua, doc, P, b)) for b in (2, 3, 4)]
        methods += [(f"KIVI-style, {b}-bit", lambda b=b: baselines.kivi_style(study, doc, P, b)) for b in (2, 3, 4)]
        methods += [("FP8 cast", lambda: baselines.fp8_cast(study, doc, P))]
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
        result["documents"][doc.name] = rows
        (out / "lossy.json").write_text(json.dumps(result, indent=1))          # partial results survive a crash
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


def stage_gpu_speed(args, out):
    import gc
    import torch
    from kv_compare.gpu_coder import SliceCoder
    from kv_compare.gpu_speed import coder_inputs, coder_rows, overlap_row, predictor_layer, predictor_rows
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
                                      layer_fn, log, mode=best["mode"])
                    row.update(predictor=f"{label} ({fmt})", variant=label)
                    rows.append(row)
                except Exception as exc:
                    log(f"  overlap {label}: not measured ({exc!r})")
                del layer_fn
    else:
        log("gpu_speed: the coder needs a CUDA GPU (or TRITON_INTERPRET=1 for a slow CPU check); skipped")
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
                  "| Mode | Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in coder:
            lines.append(f"| {r.get('mode', 'exact')} | {r['values_per_stream']} x {r['block']} ({r.get('warps', max(1, r['block'] // 32))}) | {r['bit_exact']} | {r['stream_bits_per_value']:.4f} | "
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
            cand = [r for r in pred if r["variant"] == v and "error" not in r and r["format"] in ("int8x8", "int8x16")]
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
    lines += ["", "Reference: " + res["splitzip_reference"] + ".", ""]
    return lines


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
        for R in (10, 25, 50, 100, 200, 400, 700):
            single = min(((max(float(np.mean(o[1])) / R, 1 / o[2]), o[0]) for o in opts))
            best = min(((max(bits / L / R, float(np.dot(cnt, C)) / L), cnt) for cnt, bits in dp.items()))
            t_sz = max(sz_bits / R, 1 / Ts)
            ms = lambda t: f"{t * n / 1e6:.0f} ms"
            counts = ", ".join(f"{c} {opts[i][0].replace('P, ', '')}" for i, c in enumerate(best[1]) if c)
            lines.append(f"| {R} Gb/s | {ms(16 / R)} | {ms(t_sz)} | {ms(single[0])} ({single[1].replace('P, ', '')}) | "
                         f"{ms(best[0])} ({counts}) | {t_sz / best[0]:.2f}x |")
    return lines + ([""] if lines else [])


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
        lines.append("")
    if (out / "frontier.json").is_file():
        lines += frontier_lines(out)
    if (out / "gpu_speed.json").is_file():
        lines += speed_lines(out)
        lines += mix_lines(out, json.loads((out / "gpu_speed.json").read_text()))
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
        own = [res["documents"][d]["exact KV"].get("model_prefill_ppl_change_pct") for d in docs]
        if all(x is not None for x in own):
            lines.append("The model's own prefill of the prompt alone differs from it by " +
                         " / ".join(f"{x:+.3f}%" for x in own) + " (GPU kernels round differently at another sequence length).")
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
    p.add_argument("--frontier-quick", action="store_true", help="frontier stage: a smaller grid of variants")
    p.add_argument("--speed-tokens", type=int, default=8192, help="gpu_speed: tokens per timed layer")
    p.add_argument("--speed-formats", default="fp32,bf16,int8x16,int8x8")
    p.add_argument("--speed-config", default="lite", help="gpu_speed: configuration whose predictions feed the coder")
    p.add_argument("--speed-layers", default="4,12,20,27", help="gpu_speed: test-document layers coded")
    p.add_argument("--speed-modes", default="exact,uniform",
                   help="gpu_speed: coder modes; uniform gives every word of a slice an equal share (faster, a few "
                        "hundredths of a bit more)")
    p.add_argument("--speed-settings", default="1024x64x1,1024x128x1,1024x256x2,512x128x1,2048x128x1",
                   help="gpu_speed: coder settings, values per stream x lanes per program [x warps per program; "
                        "default one thread per lane, fewer warps interleave several lanes per thread]")
    return p


def gpu_sharing_note(device):
    """Free and total memory of the GPU; other processes on it make memory tight and timings meaningless."""
    try:
        import torch
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info(torch.device(device))
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
    args.speed_formats = [f for f in args.speed_formats.split(",") if f]
    args.speed_modes = [m for m in args.speed_modes.split(",") if m]
    args.speed_layers = [int(l) for l in args.speed_layers.split(",") if l]
    args.speed_settings = [tuple(int(x) for x in t.split("x")) for t in args.speed_settings.split(",") if t]
    if any(len(t) not in (2, 3) for t in args.speed_settings):
        raise SystemExit("--speed-settings takes VxB or VxBxW items (values per stream x lanes x warps)")
    stages = [s for s in args.stages.split(",") if s]
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
    log(f"run started: stages {stages}; {env.get('gpu', env['device'])}; commit {env.get('git_commit')}")
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
        if "lossless" in stages:
            stage_lossless(args, out)
        if "lossy" in stages:
            stage_lossy(args, out)
        if "frontier" in stages:
            stage_frontier(args, out)
        if "gpu_speed" in stages:
            stage_gpu_speed(args, out)
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
