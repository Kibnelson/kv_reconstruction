#!/usr/bin/env python3
"""The P gate on one GPU: is a compressed, GPU-resident KV pool worth building the transfer lifecycle around?

P is exact KV kept compressed in decode-GPU memory and decoded in registers by attention. exactkv
(github.com/zyfyz666/exactkv, kv_compare.more_baselines.EXACTKV_COMMIT; Apache-2.0) is the published P:
4-bit exponent offsets per 16-token page and channel, sign and mantissa raw, about 12.7 bits per value, and a
split-K decode-attention kernel whose dense twin performs the same arithmetic (bit-identical outputs). It is the
P backbone here; what is ours is measured on top of it.

  attn     decode attention over identical Llama KV (the run's captures): dense BF16 (exactkv's reference kernel,
           the same arithmetic; PyTorch SDPA as the production dense path) against exactkv's pool. Per config
           (batch x context): bit-identity, resident bits per value (allocated and used), time per layer and per
           decode step (all layers).
  e2e      greedy generation with exactkv's engine (the model's own weights and modules; only the cache and decode
           attention differ): BF16 cache vs pool at the same batch, and each at the largest batch that fits the
           same KV budget; bit-identical logits on a short check.
  install  T -> P: our per-layer stream decoded layer by layer straight into the pool (LayerwiseOneStep.decode
           with a sink: a two-layer ring, never the whole BF16 cache), against decoding the whole cache first
           (ours, SplitZip) and against raw BF16 received whole; time, peak extra memory, the pool checked bit for
           bit against the original KV; then the time until the pool is ready, per link speed.
  report   summary.md

    python scripts/bench_pool_gate.py --run ~/kv_a5000/runs/llama32_3b --eval-text ~/kv_a5000/runs/dataset_texts/wikitext2.txt

Needs a run folder of scripts/run_a5000.sh (calcap and datasets/context captures; the model and revision are read
from its run_args.json). Results: <run>/pool_gate/{results.json, summary.md}.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LINKS = (1, 10, 25, 50, 100, 200, 400)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_exactkv(source):
    from kv_compare.more_baselines import EXACTKV_COMMIT, EXACTKV_REPO, checkout
    src = checkout(EXACTKV_REPO, EXACTKV_COMMIT, source, log)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import exactkv
    from exactkv import kernels
    return exactkv, kernels


def timer(fn, warmup=5, reps=20):
    from kv_compare.gpu_speed import timer as t
    return t(fn, torch.device("cuda"), warmup=warmup, reps=reps)


def peak(fn):
    """(result, seconds, peak bytes above the memory allocated before) of fn() on the GPU."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    return r, sec, torch.cuda.max_memory_allocated() - base


def captures(run):
    """{name: tokens} of the run's captured documents (int16 words via kv_compare.predict.Doc)."""
    out = {}
    for f in sorted((run / "captures").glob("*_ids.npy")):
        name = f.name[:-len("_ids.npy")]
        if (run / "captures" / f"{name}_K_bits.npy").is_file():
            out[name] = int(np.load(f, mmap_mode="r").shape[0])
    return out


def layer_kv(doc, layer, T, H, D, device):
    """Keys and values of one layer, first T tokens: two bf16 [T, H, D] tensors (the cache's own words)."""
    def get(a):
        w = np.ascontiguousarray(a[layer][:T]).view(np.int16)
        return torch.from_numpy(w).to(device).view(torch.bfloat16).view(T, H, D)
    return get(doc.K), get(doc.V)


def sdpa_attn(q, kc, vc):
    """Dense decode attention with PyTorch SDPA (its fused flash / memory-efficient kernels) over the engine's BF16
    cache, read in place: [B, TS, H, D] viewed as [B, H, T, D] without a copy. q [B, Hq, D] -> [B, Hq, D]."""
    T = kc.T
    k = kc.buf[:, :T].transpose(1, 2)
    v = vc.buf[:, :T].transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(q.unsqueeze(2), k, v, enable_gqa=True).squeeze(2)


def flashinfer_baseline(a, log_errors=True):
    """exactkv's own speed baseline (bench/baselines.FlashInferBF16: FlashInfer's BF16 paged decode, tensor cores),
    or None when flashinfer is not installed (pip install flashinfer-python in the environment)."""
    try:
        bench = Path(a.exactkv_source) / "bench"
        if str(bench) not in sys.path:
            sys.path.insert(0, str(bench))
        from baselines import FlashInferBF16
        return FlashInferBF16()
    except Exception as exc:
        if log_errors:
            log(f"FlashInfer not available ({repr(exc)[:160]}); its rows are left out")
        return None


# ------------------------------------------------------------------------------------------------------------ attn
def stage_attn(a, run, meta, res):
    from kv_compare.predict import Doc
    exactkv, K = load_exactkv(a.exactkv_source)
    H, D, Hq, L = meta["kv_heads"], meta["head_dim"], meta["query_heads"], meta["layers"]
    have = captures(run)
    layers = sorted({round(i * (L - 1) / max(1, a.attn_layers - 1)) for i in range(a.attn_layers)})
    rows = res.setdefault("attn", [])
    done = {(r["batch"], r["tokens"], r["layer"]) for r in rows}
    dev = torch.device("cuda")
    fi = flashinfer_baseline(a)
    for B, T in a.attn_configs:
        # B distinct documents of at least T tokens (the context sweep's first, then any capture), repeated if fewer
        # evaluation documents first (the context sweep's own length, then the datasets), calibration ones last
        pool = sorted([n for n, t in have.items() if t >= T],
                      key=lambda n: (not n.startswith(f"ctx{T}_"), not n.startswith(("ctx", "ds_")), n))
        if not pool:
            log(f"attn {B}x{T}: no capture of {T} tokens; skipped")
            continue
        names = [pool[i % len(pool)] for i in range(B)]
        docs = {n: Doc(run / "captures", n) for n in set(names)}
        for l in layers:
            if (B, T, l) in done:
                continue
            ks, vs = zip(*[layer_kv(docs[n], l, T, H, D, dev) for n in names])
            k, v = torch.stack(ks).contiguous(), torch.stack(vs).contiguous()          # [B, T, H, D]
            del ks, vs
            g = torch.Generator(device=dev).manual_seed(l)
            q = (torch.randn(B, Hq, D, device=dev, generator=g) * k.float().std()).to(torch.bfloat16)
            kc, vc = exactkv.CompressedKV(B, T, H, D), exactkv.CompressedKV(B, T, H, D)
            for b in range(B):
                kc.prefill_slot(b, k[b:b + 1])
                vc.prefill_slot(b, v[b:b + 1])
            t_c, nw = min((timer(lambda: K.attention_compressed(q, kc, vc, T, num_warps=w)), w) for w in (4, 8))
            oc = K.attention_compressed(q, kc, vc, T, num_warps=nw)
            orf = K.attention_reference(q, k, v, T, num_warps=nw)
            same = bool(torch.equal(oc.view(torch.int16), orf.view(torch.int16)))
            t_r = timer(lambda: K.attention_reference(q, k, v, T, num_warps=nw))
            kh, vh = k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()     # [B, H, T, D]
            qh = q.unsqueeze(2)
            sdpa = lambda: torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, enable_gqa=True)
            try:
                t_s = timer(sdpa)
                d_s = float((sdpa().squeeze(2).float() - orf.float()).abs().max())
            except Exception as exc:                 # an SDPA without GQA support
                t_s, d_s = None, repr(exc)[:200]
            # SDPA as the engine runs it: the cache [B, T, H, D] read in place through a transposed view
            kv_view = type("C", (), {"buf": k, "T": T})(), type("C", (), {"buf": v, "T": T})()
            try:
                t_sv = timer(lambda: sdpa_attn(q, *kv_view))
            except Exception as exc:
                t_sv = None
                log(f"attn: SDPA on the cache view failed ({repr(exc)[:160]})")
            t_fi = None
            if fi is not None:
                try:
                    fi.plan(B, T, T, Hq, H, D)
                    t_fi = timer(lambda: fi.w.run(q, (fi.paged(k), fi.paged(v))))
                except Exception as exc:
                    log(f"attn: FlashInfer failed ({repr(exc)[:160]})")
            used = sum(sum(t.numel() * t.element_size() for t in (c.sm, c.code, c.base, c.rowmask, c.rowoff))
                       + int(c.cnt.item()) * D * 2 for c in (kc, vc))
            row = {"batch": B, "tokens": T, "layer": l, "documents": sorted(set(names)), "num_warps": nw,
                   "bit_identical": same, "pool_us": 1e6 * t_c, "dense_same_arithmetic_us": 1e6 * t_r,
                   "sdpa_us": None if t_s is None else 1e6 * t_s, "sdpa_max_abs_diff": d_s,
                   "sdpa_cache_view_us": None if t_sv is None else 1e6 * t_sv,
                   "flashinfer_us": None if t_fi is None else 1e6 * t_fi,
                   "bits_allocated": (kc.nbytes() + vc.nbytes()) * 8 / (2 * k.numel()),
                   "bits_used": used * 8 / (2 * k.numel())}
            rows.append(row)
            f1 = lambda v: "-" if v is None else f"{v:.1f}"
            log(f"attn {B}x{T} layer {l}: pool {row['pool_us']:.1f} us, dense same arithmetic "
                f"{row['dense_same_arithmetic_us']:.1f} us, SDPA {f1(row['sdpa_us'])} us (on the cache view "
                f"{f1(row['sdpa_cache_view_us'])}), FlashInfer {f1(row['flashinfer_us'])} us, identical={same}, "
                f"{row['bits_allocated']:.2f} bits allocated")
            del k, v, kc, vc, q, kh, vh, qh, oc, orf
            torch.cuda.empty_cache()
            save(a, res)


# ------------------------------------------------------------------------------------------------------------- e2e
def stage_e2e(a, run, meta, res):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    exactkv, K = load_exactkv(a.exactkv_source)
    kw = {} if meta.get("revision") in (None, "none") else {"revision": meta["revision"]}
    tok = AutoTokenizer.from_pretrained(meta["model"], **kw)
    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.bfloat16, **kw).to("cuda").eval()
    text = Path(a.eval_text).read_text(encoding="utf-8")
    ids_all = tok(text, add_special_tokens=False)["input_ids"]
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    e2e = res.setdefault("e2e", {})

    def prompts(B, PL):
        step = PL - len(bos)
        if B * step > len(ids_all):
            raise SystemExit(f"{a.eval_text}: {len(ids_all)} tokens, {B * step} needed")
        return torch.tensor([bos + ids_all[i * step:(i + 1) * step] for i in range(B)], device="cuda")

    # bit-identity of the two engines on a short greedy run
    if "identity" not in e2e:
        out = {}
        for name in ("bf16", "exactkv"):
            eng = exactkv.Engine(model, 2, 1024 + 32, backend=name)
            out[name] = eng.generate(prompts(2, 1024), 32)
            del eng
            torch.cuda.empty_cache()
        e2e["identity"] = {"tokens_equal": bool(torch.equal(out["bf16"][0], out["exactkv"][0])),
                           "logits_bit_identical": bool(torch.equal(out["bf16"][1].view(torch.int16),
                                                                     out["exactkv"][1].view(torch.int16))),
                           "batch": 2, "prompt": 1024, "generated": 32}
        log(f"e2e identity: {e2e['identity']}")
        save(a, res)
    for PL in a.e2e_prompts:
        key = str(PL)
        if key in e2e:
            continue
        TM = PL + a.e2e_gen

        def per_seq(name):
            e1, e2 = exactkv.Engine(model, 1, TM, backend=name), exactkv.Engine(model, 2, TM, backend=name)
            d = e2.kv_bytes() - e1.kv_bytes()
            del e1, e2
            torch.cuda.empty_cache()
            return d
        torch.cuda.empty_cache()
        budget = torch.cuda.mem_get_info()[0] - a.headroom_gib * 2 ** 30
        ps = {n: per_seq(n) for n in ("bf16", "exactkv")}
        fi = flashinfer_baseline(a, log_errors=False)
        arms = {"bf16 + SDPA": ("bf16", lambda B: exactkv.Engine(model, B, TM, backend="bf16", attn=sdpa_attn)),
                "pool (exactkv)": ("exactkv", lambda B: exactkv.Engine(model, B, TM, backend="exactkv"))}
        if fi is not None:
            arms["bf16 + FlashInfer"] = ("bf16", lambda B: fi.engine(model, B, TM))
        if a.e2e_reference:
            arms["bf16, same-arithmetic reference kernel"] = ("bf16", lambda B: exactkv.Engine(model, B, TM,
                                                                                           backend="bf16"))
        bmax = {n: max(1, int(budget // ps[n])) for n in ps}

        def run_one(arm, B):
            kind, make = arms[arm]
            name = arm
            while B > 0:
                eng = None
                try:
                    ids = prompts(B, PL)
                    eng = make(B)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    t = eng.prefill(ids)
                    torch.cuda.synchronize()
                    t1 = time.time()
                    for _ in range(a.e2e_gen - 1):
                        t = eng.step(t).argmax(-1)
                    torch.cuda.synchronize()
                    t2 = time.time()
                    r = {"backend": name, "batch": B, "kv_mib_per_sequence": ps[kind] / 2 ** 20,
                         "ms_per_step": (t2 - t1) / (a.e2e_gen - 1) * 1e3,
                         "decode_tokens_per_s": B * (a.e2e_gen - 1) / (t2 - t1), "prefill_s": t1 - t0,
                         "pool_grows": getattr(eng, "grows", 0)}
                    log(f"e2e prompt {PL}: {json.dumps(r)}")
                    return r
                except torch.cuda.OutOfMemoryError:
                    log(f"e2e prompt {PL}: {name} out of memory at batch {B}")
                    B -= 1
                finally:
                    del eng
                    gc.collect()
                    torch.cuda.empty_cache()
            return None
        rows = []
        for arm, (kind, _) in arms.items():
            rows.append(run_one(arm, bmax[kind]))     # each at the largest batch the budget fits
        dense = next((r for r in rows if r and r["backend"] == "bf16 + SDPA"), None)
        if dense:                                     # the pool at the dense batch: the same work
            r = run_one("pool (exactkv)", dense["batch"])
            if r:
                r["note"] = "at the dense batch"
            rows.append(r)
        e2e[key] = {"prompt": PL, "generated": a.e2e_gen, "kv_budget_gib": budget / 2 ** 30,
                    "kv_bytes_per_sequence": ps, "rows": rows}
        save(a, res)
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------- install
def stage_install(a, run, meta, res):
    from kv_compare import fused
    from kv_compare.context_sweep import rope_consts
    from kv_compare.fused import FusedOneStep
    from kv_compare.layerwise import LayerwiseOneStep
    from kv_compare.predict import Doc, Study
    exactkv, K = load_exactkv(a.exactkv_source)
    dev = torch.device("cuda")
    H, D = meta["kv_heads"], meta["head_dim"]
    have = captures(run)
    name = a.install_doc
    if name not in have:                              # any evaluation document of 8,192 tokens (not calibration)
        cand = sorted(n for n, t in have.items() if t >= 8192 and (n.startswith("ctx") or n.startswith("ds_")))
        if not cand:
            raise SystemExit(f"install: no capture {name} (nor another evaluation document) in {run / 'captures'}")
        log(f"install: no {name}; using {cand[0]}")
        name = cand[0]
    train = [f"wtrain{i}" for i in range(6)]
    study = Study(run / "captures", "cuda", train, "wtables", [], None, log=log)
    consts = fused.prepare(study, log=log)
    study._f.clear(); study._att.clear(); study._qmap.clear()
    doc = Doc(run / "captures", name)
    L, T, W, C = study.L, doc.tokens, study.W, study.C
    words = torch.empty(L, T, W, dtype=torch.int16, device=dev)
    for l in range(L):
        x = doc.words(l, dev)
        words[l] = (x - 65536 * (x >= 32768).to(torch.int32)).to(torch.int16)
    kc_ = rope_consts(study, consts, torch.arange(T, device=dev))
    codec = LayerwiseOneStep(kc_, C, study.hd, tokens_per_program=32, mode="uniform", search=True, chunks=2)
    xall, prior = FusedOneStep(kc_, C, study.hd, tokens_per_program=32, rows=4, warps=4).model(words)
    buf, starts = codec.encode(words, xall, prior)
    bufs = codec.buffers(T)
    ours_bits = codec.bits(starts, T) / words.numel()
    del xall, prior
    gc.collect()
    torch.cuda.empty_cache()
    ins = res.setdefault("install", {"document": name, "tokens": T, "layers": L, "bf16_mib": words.numel() * 2 / 2 ** 20})

    def new_pool():
        return [(exactkv.CompressedKV(1, T, H, D), exactkv.CompressedKV(1, T, H, D)) for _ in range(L)]

    def pack_layer(pool, l, w):                       # w: int16 [T, W] of one layer
        kv = w.view(torch.bfloat16)
        pool[l][0].prefill_slot(0, kv[:, :C].view(1, T, H, D))
        pool[l][1].prefill_slot(0, kv[:, C:].view(1, T, H, D))

    def pool_exact(pool):
        for l in range(L):
            for j, part in enumerate((slice(0, C), slice(C, None))):
                got = K.decode_to_bf16(pool[l][j].buffers(), T).view(torch.int16).view(T, C)
                if not torch.equal(got, words[l, :, part]):
                    return False
        return True

    def pool_bytes(pool):
        return sum(c.nbytes() for p in pool for c in p)

    def arm(label, fn, wire_bits):
        pool, sec, pk = peak(fn)
        ex = pool_exact(pool)
        r = {"method": label, "seconds": sec, "peak_extra_mib": pk / 2 ** 20, "pool_mib": pool_bytes(pool) / 2 ** 20,
             "transient_mib": (pk - pool_bytes(pool)) / 2 ** 20, "pool_exact": ex, "wire_bits": wire_bits}
        log(f"install {label}: {sec * 1e3:.1f} ms, peak +{r['peak_extra_mib']:.0f} MiB (pool {r['pool_mib']:.0f}, "
            f"transient {r['transient_mib']:.0f}), exact={ex}")
        del pool
        gc.collect()
        torch.cuda.empty_cache()
        return r
    rows = []

    def ours_sink():
        pool = new_pool()
        codec.decode(buf, starts, T, bufs=bufs, sink=lambda l, w: pack_layer(pool, l, w))
        return pool

    def ours_whole():
        pool = new_pool()
        dec = codec.decode(buf, starts, T, bufs=bufs)
        for l in range(L):
            pack_layer(pool, l, dec[l])
        del dec
        return pool
    for _ in range(2):                                # warm both paths (Triton compiles, the pool kernels)
        ours_sink()
        ours_whole()
    rows.append(arm("ours, layer by layer into the pool (two-layer ring)", ours_sink, ours_bits))
    rows.append(arm("ours, whole cache decoded, then the pool", ours_whole, ours_bits))
    try:
        from kv_compare.mixed import load_splitzip
        sz, _ = load_splitzip(a.splitzip_source, dev, log)
        cal = Doc(run / "captures", "wtrain0")
        cw = torch.empty(L, cal.tokens, W, dtype=torch.int16, device=dev)
        for l in range(L):
            x = cal.words(l, dev)
            cw[l] = (x - 65536 * (x >= 32768).to(torch.int32)).to(torch.int16)
        sz.calibrate(cw.view(torch.bfloat16))
        del cw
        enc = sz.encode(words.view(torch.bfloat16))
        sz_bits = 8 * enc.compressed_bytes / words.numel()

        def sz_whole():
            pool = new_pool()
            dec = sz.decode(enc).reshape(-1).view(torch.int16).view(L, T, W)
            for l in range(L):
                pack_layer(pool, l, dec[l])
            del dec
            return pool
        sz_whole()
        rows.append(arm("SplitZip, whole cache decoded, then the pool", sz_whole, sz_bits))
        del enc
    except Exception as exc:
        log(f"install: SplitZip not measured ({exc!r})")
        rows.append({"method": "SplitZip, whole cache decoded, then the pool", "error": repr(exc)[:300]})

    def raw_whole():                                  # BF16 received whole (a copy stands for the receive buffer)
        pool = new_pool()
        dec = words.clone()
        for l in range(L):
            pack_layer(pool, l, dec[l])
        del dec
        return pool
    raw_whole()
    rows.append(arm("raw BF16 received whole, then the pool", raw_whole, 16.0))
    ins["rows"] = rows
    ins["links_gbps"] = list(LINKS)
    save(a, res)
    del words, buf, starts, bufs, codec
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------- report
def report(a, run, meta, res):
    lines = [f"# The P gate on {res.get('gpu')}: {meta['model']}", "",
             "P = exact KV kept compressed in decode-GPU memory, decoded in registers by attention. The pool is exactkv "
             "(the published P; its dense kernel performs the same arithmetic). See scripts/bench_pool_gate.py.", ""]
    rows = res.get("attn", [])
    if rows:
        L = meta["layers"]
        lines += ["## Decode attention: the pool against production dense kernels, on identical KV", "",
                  f"Per decode step = the mean per layer x {L} layers. The production dense kernels are PyTorch SDPA "
                  "(contiguous [B, H, T, D], and on the engine's cache view) and FlashInfer (exactkv's own baseline; "
                  "when installed). The pool's bit-identity is checked against exactkv's dense kernel with the same "
                  "arithmetic, which is a reference, not a production kernel (it assembles BF16 from bytes). Bits "
                  "per value of the pool: allocated (directory and raw-row reserve) / used.", "",
                  "| Batch x context | Pool | SDPA | SDPA, cache view | FlashInfer | Pool vs fastest dense | "
                  "Same-arithmetic reference | Bit-identical | Bits allocated / used |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for (B, T) in sorted({(r["batch"], r["tokens"]) for r in rows}, key=lambda x: (x[1], x[0])):
            rr = [r for r in rows if r["batch"] == B and r["tokens"] == T]
            m = lambda k: sum(r[k] for r in rr) / len(rr)

            def mo(k):
                v = [r.get(k) for r in rr if r.get(k) is not None]
                return sum(v) / len(v) * L / 1e3 if len(v) == len(rr) else None
            dense = [v for v in (mo("sdpa_us"), mo("sdpa_cache_view_us"), mo("flashinfer_us")) if v is not None]
            pool = m("pool_us") * L / 1e3
            ms = lambda v: "-" if v is None else f"{v:.2f} ms"
            lines.append(f"| {B} x {T:,} | {pool:.2f} ms | {ms(mo('sdpa_us'))} | {ms(mo('sdpa_cache_view_us'))} | "
                         f"{ms(mo('flashinfer_us'))} | "
                         + (f"{min(dense) / pool:.2f}x" if dense else "-")
                         + f" | {m('dense_same_arithmetic_us') * L / 1e3:.2f} ms | "
                         f"{'yes' if all(r['bit_identical'] for r in rr) else 'NO'} | "
                         f"{m('bits_allocated'):.2f} / {m('bits_used'):.2f} |")
        lines += ["", "Pool vs fastest dense above 1 means the pool is faster.", ""]
    e2e = res.get("e2e", {})
    if e2e:
        lines += ["## Generation: BF16 cache vs pool (same engine and weights)", ""]
        if "identity" in e2e:
            i = e2e["identity"]
            lines += [f"Greedy check, batch {i['batch']}, {i['prompt']}-token prompts, {i['generated']} tokens: tokens "
                      f"equal {i['tokens_equal']}, logits bit-identical {i['logits_bit_identical']}.", ""]
        lines += ["| Prompt | Backend | Batch | KV per sequence | ms per step | Decode tokens/s |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for k, v in e2e.items():
            if k == "identity":
                continue
            for r in v["rows"]:
                if r:
                    lines.append(f"| {v['prompt']:,} + {v['generated']} | {r['backend']}"
                                 + (f" ({r['note']})" if r.get("note") else "") + f" | {r['batch']} | "
                                 f"{r['kv_mib_per_sequence']:.0f} MiB | {r['ms_per_step']:.2f} | {r['decode_tokens_per_s']:.0f} |")
        lines += ["", "Each backend runs at the largest batch the same KV budget fits, then the pool at the dense batch "
                  "(the same work). Same engine for all (exactkv's plain Python loop, no CUDA graphs); only the "
                  "cache and the decode-attention kernel differ.", ""]
    ins = res.get("install")
    if ins and ins.get("rows"):
        lines += [f"## T -> P installation: {ins['document']} ({ins['tokens']:,} tokens, {ins['layers']} layers, "
                  f"{ins['bf16_mib']:.0f} MiB in BF16)", "",
                  "| Path | Wire bits/value | Install time | Peak extra memory | of which transient | Pool | Pool exact |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in ins["rows"]:
            if "error" in r:
                lines.append(f"| {r['method']} | - | not measured | | | | |")
                continue
            lines.append(f"| {r['method']} | {r['wire_bits']:.2f} | {r['seconds'] * 1e3:.0f} ms | {r['peak_extra_mib']:.0f} MiB | "
                         f"{r['transient_mib']:.0f} MiB | {r['pool_mib']:.0f} MiB | {r['pool_exact']} |")
        vals = ins["bf16_mib"] * 2 ** 20 / 2            # values in the cache
        lines += ["", "**Time until the pool is ready, per link** (the wire and the installation in sequence; a "
                  "chunked protocol would overlap them):", "", "| Link | " + " | ".join(
                      r["method"].split(",")[0] + (" (layer by layer)" if "layer by layer" in r["method"] else "")
                      for r in ins["rows"] if "error" not in r) + " |",
                  "| --- |" + " --- |" * sum(1 for r in ins["rows"] if "error" not in r)]
        for g in ins.get("links_gbps", LINKS):
            cells = [f"{(r['wire_bits'] * vals / (g * 1e9) + r['seconds']) * 1e3:.0f} ms"
                     for r in ins["rows"] if "error" not in r]
            lines.append(f"| {g} Gb/s | " + " | ".join(cells) + " |")
        lines.append("")
    (Path(a.output) / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def save(a, res):
    Path(a.output).mkdir(parents=True, exist_ok=True)
    (Path(a.output) / "results.json").write_text(json.dumps(res, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True, help="a run folder of scripts/run_a5000.sh")
    p.add_argument("--output", type=Path, default=None, help="default <run>/pool_gate")
    p.add_argument("--stages", default="attn,e2e,install,report")
    p.add_argument("--exactkv-source", type=Path, default=Path.home() / "kv_a5000/sources/exactkv")
    p.add_argument("--splitzip-source", type=Path, default=Path.home() / "kv_a5000/sources/splitzip")
    p.add_argument("--eval-text", type=Path, default=Path.home() / "kv_a5000/runs/dataset_texts/wikitext2.txt")
    p.add_argument("--attn-configs", default="1x2048,16x2048,32x2048,1x8192,4x8192,8x8192,1x32768")
    p.add_argument("--attn-layers", type=int, default=4, help="layers timed per config (evenly spaced)")
    p.add_argument("--e2e-prompts", default="2048,8192")
    p.add_argument("--e2e-gen", type=int, default=128)
    p.add_argument("--headroom-gib", type=float, default=1.5)
    p.add_argument("--e2e-reference", action="store_true",
                   help="e2e: also time the engine with exactkv's same-arithmetic dense kernel (slow; it is the "
                        "bit-identity reference, not a production kernel)")
    p.add_argument("--install-doc", default="ctx8192_r0")
    p.add_argument("--rerun", default="", help="stages to measure again (results are kept otherwise)")
    a = p.parse_args(argv)
    a.output = a.output or a.run / "pool_gate"
    a.attn_configs = [tuple(int(v) for v in c.split("x")) for c in a.attn_configs.split(",") if c]
    a.e2e_prompts = [int(v) for v in a.e2e_prompts.split(",") if v]
    stages = [s for s in a.stages.split(",") if s]
    meta = json.loads((a.run / "captures" / "meta.json").read_text())
    rargs = json.loads((a.run / "run_args.json").read_text()) if (a.run / "run_args.json").is_file() else {}
    meta = dict(meta, model=rargs.get("model", meta["model"]), revision=rargs.get("revision", meta.get("revision")))
    path = Path(a.output) / "results.json"
    res = json.loads(path.read_text()) if path.is_file() else {}
    for s in [s for s in a.rerun.split(",") if s]:
        res.pop(s, None)
    import platform
    import subprocess
    import transformers
    import triton
    res.update(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, triton=triton.__version__,
               transformers=transformers.__version__, python=platform.python_version(), model=meta["model"],
               revision=meta.get("revision"), run=str(a.run),
               git_commit=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip())
    torch.backends.cuda.matmul.allow_tf32 = False
    if "attn" in stages:
        stage_attn(a, a.run, meta, res)
    if "e2e" in stages:
        stage_e2e(a, a.run, meta, res)
    if "install" in stages:
        stage_install(a, a.run, meta, res)
    save(a, res)
    if "report" in stages:
        report(a, a.run, meta, res)


if __name__ == "__main__":
    main()
