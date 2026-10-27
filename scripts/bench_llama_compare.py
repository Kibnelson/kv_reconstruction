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
             processes the next --continuation-tokens tokens with that KV; perplexity change vs exact KV
  frontier   (optional) bits against compute: ~37 predictor variants (inputs, steps, reduced rank, 8-bit integer
             formats) fitted from one covariance per layer and scored like the lossless stage -> frontier.json
  report     summary.md

Document roles (same protocol as the SmolLM2 study): calibration = PEP 484 and six CPython modules, scales =
dataclasses.py, tests = PEP 8 (text) and argparse.py (code). Costs are computed -log2 P (the reference range
coder matched computed costs within 0.011 bits per value on SmolLM2); rates of the fixed-rate baselines include
their scale/offset overhead. GPU timings are not part of these stages.
"""
from __future__ import annotations

import argparse
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
    log(f"fitting our predictor ({cfg.name}) for closed-loop coding")
    fits = {}
    study.lossless(cfg, fits)
    study._f.clear(); study._att.clear()
    log(f"fitting AQUA-KV-style predictors ({memory_note(args.device)})")
    aqua = baselines.aqua_fit(study, study.L, log)
    study._f.clear(); study._att.clear(); study._qmap.clear()
    import gc
    gc.collect()
    if study.device.type == "cuda":
        torch.cuda.empty_cache()
    device = torch.device(args.device)
    log(f"loading {args.model} for the quality check ({memory_note(args.device)})")
    _, model = load_model(args.model, args.revision, device)
    log(f"model loaded ({memory_note(args.device)})")
    P, M = args.prompt_tokens, args.continuation_tokens
    result = {"prompt_tokens": P, "continuation_tokens": M, "lossy_config": cfg.name, "documents": {}}
    for doc in study.tests:
        base = continuation_nll(model, doc.ids, P, M, device)
        check = continuation_nll(model, doc.ids, P, M, device, matrices_from_doc(doc, P))
        if check != base:
            raise RuntimeError(f"exact prompt KV did not reproduce the baseline ({check} vs {base})")
        log(f"{doc.name}: exact-KV perplexity {np.exp(base):.4f}, reproduced by the restored cache")
        exact = torch.stack([words_to_float(doc.words(l, study.device, P)).to(torch.float32).cpu() for l in range(study.L)])
        rows = {"exact KV": {"bits": 16.0, "nll": base, "ppl": float(np.exp(base)), "ppl_change_pct": 0.0, "rel_error": 0.0}}
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
            log(f"{doc.name} {label}: {rate:.3f} bits/value, KV error {100 * rel:.2f}%, perplexity {rows[label]['ppl_change_pct']:+.2f}%")
            study._f.clear()
            del rec
        result["documents"][doc.name] = rows
        (out / "lossy.json").write_text(json.dumps(result, indent=1))          # partial results survive a crash
    log(f"lossy stage done ({memory_note(args.device)})")
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
        lines += ["", f"Values are {' / '.join(docs)}."]
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
