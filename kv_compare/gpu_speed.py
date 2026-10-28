"""Measured GPU throughput of our codec's two parts: the predictor and the value-slice entropy coder.

Predictor. Each layer of T tokens is predicted as in the decoder: one matrix product for the inputs that are
already decoded (other layers, or their reduced-rank projection), then one product per channel group using the
groups decoded before it, in sequence; plus, for the attention input, the query map and the windowed attention
(last 64 tokens + token 0). Formats:
  fp32       float32 products, TF32 off: the reference implementation's arithmetic
  bf16       BF16 tensor cores with float32 accumulation: fast, but bit-identical only on the same GPU and kernel
  int8x16    exact integers on the 8-bit tensor cores: 8-bit coefficients x 16-bit features as two 8-bit halves
             (two torch._int_mm products per multiply-add, combined in 64-bit integers)
  int8x8     8-bit coefficients x 8-bit features (one product)
  int8x8s    as int8x8 with the inputs product 2:4 sparse (two of every four coefficients zero; Ampere's sparse
             tensor cores via cuSPARSELt or CUTLASS). Speed only: the bit cost of the sparsity is not measured here
Random data of the real shapes is used: the cost of a dense product does not depend on the values.

Entropy coder. The Triton value-slice coder (gpu_coder.py) encodes and decodes the real words of the test
document against the real predictions and spreads of a fitted configuration; the round trip must be bit-exact,
and the stream size is compared with the exact model cost of the same predictions.

Timings are CUDA-event medians after warm-up, kernel time only (no host transfers), like SplitZip's and ExANS's
published GB/s. GB/s counts raw BF16 bytes (2 per value), so it compares directly with those numbers.
"""
from __future__ import annotations

import math
import statistics
import time

import numpy as np
import torch

from .core import LN2X2, exact_bits, words_to_float


def timer(fn, device, warmup=3, reps=10):
    """Median seconds of fn() (CUDA events on GPU, wall clock otherwise)."""
    for _ in range(warmup):
        fn()
    times = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b) / 1e3)
    else:
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
    return statistics.median(times)


# ---------------------------------------------------------------------------------------------- predictor
class Product:
    """X [T, K] times a fixed coefficient matrix [K, N] in one format."""

    def __init__(self, fmt, T, K, N, device, gen):
        self.fmt, self.K, self.N = fmt, K, N
        if fmt == "fp32":
            self.B = torch.randn(K, N, device=device, generator=gen)
            self.X = torch.randn(T, K, device=device, generator=gen)
        elif fmt == "bf16":
            self.B = torch.randn(K, N, device=device, generator=gen).to(torch.bfloat16)
            self.X = torch.randn(T, K, device=device, generator=gen).to(torch.bfloat16)
        elif fmt == "int8x8s":                              # 2:4 sparse coefficients (Ampere sparse tensor cores)
            rnd = lambda *shape: torch.randint(-127, 128, shape, device=device, generator=gen, dtype=torch.int8)
            W = rnd(N, K)
            keep = torch.rand(N, K // 4, 4, device=device, generator=gen).argsort(-1)[..., :2]
            mask = torch.zeros(N, K // 4, 4, dtype=torch.bool, device=device).scatter_(-1, keep, True).reshape(N, K)
            W = torch.where(mask, W, torch.zeros_like(W))
            self.Xt = rnd(T, K).t()                         # [K, T] column-major (the sparse kernels need it):
                                                            # the product is W @ Xt = (X W^T)^T
            self.kind = None
            try:                                            # cuSPARSELt first, then CUTLASS
                self.packed = torch._cslt_compress(W)
                torch._cslt_sparse_mm(self.packed, self.Xt, out_dtype=torch.int32)
                self.kind = "cusparselt"
            except Exception:
                from torch.sparse._semi_structured_conversions import sparse_semi_structured_from_dense_cutlass
                self.packed, self.meta = sparse_semi_structured_from_dense_cutlass(W)
                torch._sparse_semi_structured_mm(self.packed, self.meta, self.Xt, out_dtype=torch.int32)
                self.kind = "cutlass"
        else:
            rnd = lambda *shape: torch.randint(-127, 128, shape, device=device, generator=gen, dtype=torch.int8)
            self.Bt = rnd(N, K)                              # [N, K] row-major = [K, N] column-major for _int_mm
            self.hi = rnd(T, K)
            self.lo = rnd(T, K) if fmt == "int8x16" else None
            self.corr = torch.randint(-1000, 1000, (N,), device=device, generator=gen, dtype=torch.int64)

    def __call__(self):
        if self.fmt in ("fp32", "bf16"):
            return self.X @ self.B
        if self.fmt == "int8x8s":
            if self.kind == "cusparselt":
                return torch._cslt_sparse_mm(self.packed, self.Xt, out_dtype=torch.int32)
            return torch._sparse_semi_structured_mm(self.packed, self.meta, self.Xt, out_dtype=torch.int32)
        y = torch._int_mm(self.hi, self.Bt.t())
        if self.lo is None:
            return y
        z = torch._int_mm(self.lo, self.Bt.t())
        return y.to(torch.int64) * 256 + z.to(torch.int64) + self.corr      # x = 256 hi + (lo - 128) + 128


def attention_input(T, W, Hk, hd, Hq, device, dtype, gen, chunk=1024):
    """Query map and windowed attention of layer l-1 (window: last 64 tokens + token 0), as in the study."""
    G = Hq // Hk
    Wq = torch.randn(W + 1, Hq * hd, device=device, generator=gen).to(dtype)
    f = torch.randn(T, W + 1, device=device, generator=gen).to(dtype)
    K = torch.randn(T, Hk, hd, device=device, generator=gen).to(dtype)
    V = torch.randn(T, Hk, hd, device=device, generator=gen).to(dtype)
    win = torch.arange(64, device=device)
    scale = 1.0 / math.sqrt(hd)

    def run():
        Q = (f @ Wq).reshape(T, Hk, G, hd)
        out = torch.empty(T, Hk, G, hd, device=device, dtype=dtype)
        for t0 in range(0, T, chunk):
            t = torch.arange(t0, min(T, t0 + chunk), device=device)
            idx = torch.cat([torch.zeros_like(t)[:, None], t[:, None] - win[None, :]], 1)
            valid = idx >= 0
            valid[:, 1:] &= idx[:, 1:] != 0
            ic = idx.clamp_min(0)
            sc = torch.einsum("nkgd,nwkd->nkgw", Q[t], K[ic]) * scale
            sc = sc.masked_fill(~valid[:, None, None, :], float("-inf"))
            out[t] = torch.einsum("nkgw,nwkd->nkgd", torch.softmax(sc.float(), -1).to(dtype), V[ic])
        return out
    return run


def predictor_layer(Hk, hd, Hq, T, device, v, fmt, gen):
    """Callables (whole layer, products only) predicting one layer of T tokens for variant v in format fmt."""
    W = 2 * Hk * hd
    gsz = W // v["steps"]
    parts = []
    dense = "int8x8" if fmt == "int8x8s" else fmt                  # 2:4 sparsity only for the inputs product
    if v["ne"]:
        if v.get("rank"):
            parts += [Product(dense, T, v["ne"], v["rank"], device, gen), Product(dense, T, v["rank"], W, device, gen)]
        else:
            parts.append(Product(fmt, T, v["ne"], W, device, gen))
    parts += [Product(dense, T, g * gsz, gsz, device, gen) for g in range(1, v["steps"])]
    att = None
    if v.get("att"):
        att = attention_input(T, W, Hk, hd, Hq, device, torch.float32 if fmt == "fp32" else torch.bfloat16, gen)

    def layer():
        for p in parts:
            p()
        if att is not None:
            att()
    return layer, (lambda: [p() for p in parts])


def predictor_rows(Hk, hd, Hq, T, device, variants, formats, log=print, seed=0):
    """Time one layer of each variant in each format. variants: dicts with keys label, ne, rank, steps, att."""
    W = 2 * Hk * hd
    gen = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for v in variants:
        for fmt in formats:
            if fmt == "int8x8s" and (v.get("rank") or v.get("att") or not v["ne"]):
                continue                                     # sparse timing only for a plain inputs product
            try:
                layer, products = predictor_layer(Hk, hd, Hq, T, device, v, fmt, gen)
                att = v.get("att")
                sec = timer(layer, device)
                parts_only = timer(products, device) if att else sec
                graph_sec = None
                if device.type == "cuda" and not att:      # the same products replayed from a CUDA graph
                    try:
                        graph = torch.cuda.CUDAGraph()
                        side = torch.cuda.Stream(device)
                        side.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(side):
                            for _ in range(2):
                                products()
                        torch.cuda.current_stream(device).wait_stream(side)
                        with torch.cuda.graph(graph):
                            products()
                        graph_sec = timer(graph.replay, device)
                        del graph
                    except Exception as exc:                 # capture unsupported: report the plain timing only
                        log(f"  predictor {v['label']} [{fmt}]: CUDA graph not measured ({repr(exc)[:240]})")
                rows.append({"part": "predictor", "variant": v["label"], "format": fmt, "tokens": T,
                             "ms_per_layer": 1e3 * sec, "gvalues_per_s": T * W / sec / 1e9,
                             "gb_per_s": 2 * T * W / sec / 1e9,
                             "ms_without_attention": 1e3 * parts_only if att else None,
                             "ms_per_layer_graph": 1e3 * graph_sec if graph_sec else None,
                             "gvalues_per_s_graph": T * W / graph_sec / 1e9 if graph_sec else None})
                log(f"  predictor {v['label']} [{fmt}]: {1e3 * sec:.2f} ms/layer, {T * W / sec / 1e9:.1f} G values/s"
                    + (f"; as a CUDA graph {1e3 * graph_sec:.2f} ms, {T * W / graph_sec / 1e9:.1f} G values/s" if graph_sec else ""))
                del layer, products
            except Exception as exc:                         # e.g. _int_mm unsupported on this device
                msg = repr(exc)[:240]
                rows.append({"part": "predictor", "variant": v["label"], "format": fmt, "error": msg})
                log(f"  predictor {v['label']} [{fmt}]: not measured ({msg})")
    return rows


# ---------------------------------------------------------------------------------------------- coder inputs
def coder_inputs(study, cfg, doc, layers):
    """Real (words, mu, s) of `doc` for the given layers under configuration cfg, as the lossless stage
    computes them (fits on the calibration documents, spread on the scales document, adaptive multiplier)."""
    out = {}
    prior = torch.ones(doc.tokens, dtype=torch.float64, device=study.device)
    for layer in range(max(layers) + 1):
        F = study.fit(layer, cfg)
        F.spread = study.static_spread(F, layer)
        w = doc.words(layer, study.device)
        x = words_to_float(w)
        m = study.predict_exact(F, doc, layer)
        q = (x - m).abs() / (LN2X2 * F.spread)
        s = (F.spread * (study.multiplier(F, q, prior) if cfg.adaptive else 1.0)).expand_as(x)
        if layer in layers:
            out[layer] = (w.to(torch.int16).reshape(-1), m.to(torch.float32).reshape(-1),
                          s.to(torch.float32).clamp_min(1e-12).reshape(-1))
        prior = q.mean(1)
        study.forget(layer - 1)
    return out


def coder_rows(coder_cls, device, data, settings, log=print, max_values=None, modes=("exact",)):
    """Encode and decode the concatenated layers with each (values_per_stream, block) setting."""
    w = torch.cat([d[0] for d in data.values()])
    mu = torch.cat([d[1] for d in data.values()])
    s = torch.cat([d[2] for d in data.values()])
    if max_values:
        w, mu, s = w[:max_values], mu[:max_values], s[:max_values]
    n = w.numel()
    exact = float(exact_bits(w.to(torch.int32) & 0xFFFF, mu.double(), s.double()).mean())
    rows = []
    for mode, setting in [(m, st) for m in modes for st in settings]:
        V, block = setting[0], setting[1]
        warps = setting[2] if len(setting) > 2 else None
        coder = coder_cls(device, values_per_stream=V, block=block, warps=warps, mode=mode)
        buf, starts, _ = coder.encode(w, mu, s)
        bits = coder.bits(starts, n) / n
        ideal = coder.ideal_bits(w, mu, s) / n
        reps = 10 if device.type == "cuda" else 1
        warm = 1 if device.type != "cuda" else 3
        te = timer(lambda: coder.encode(w, mu, s), device, warmup=warm, reps=reps)
        for search in (False, True):             # the same streams, decoded with and without the stage-1 search
            coder.search = search
            dec = coder.decode(buf, starts, mu, s, n)
            ok = bool(torch.equal(dec, w))
            td = timer(lambda: coder.decode(buf, starts, mu, s, n), device, warmup=warm, reps=reps)
            row = {"part": "coder", "mode": mode, "search": search, "values": n, "values_per_stream": V, "block": block,
                   "warps": coder.warps, "bit_exact": ok,
                   "stream_bits_per_value": bits, "symbol_bits_per_value": ideal, "exact_model_bits_per_value": exact,
                   "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
                   "encode_gvalues_per_s": n / te / 1e9, "decode_gvalues_per_s": n / td / 1e9,
                   "encode_gb_per_s": 2 * n / te / 1e9, "decode_gb_per_s": 2 * n / td / 1e9}
            rows.append(row)
            log(f"  coder [{mode}{', stage-1 search' if search else ''}] V={V} block={block} warps={coder.warps}: "
                f"exact={ok}, {bits:.4f} bits/value (model {exact:.4f}), encode {row['encode_gb_per_s']:.1f} GB/s, "
                f"decode {row['decode_gb_per_s']:.1f} GB/s")
            del dec
        del buf, starts
    return rows


def overlap_row(coder_cls, device, data, setting, layer_fn, log=print, mode="exact", search=False):
    """Decode on one CUDA stream while the predictor runs on another (as a pipelined decoder would, on different
    token blocks): the predictor runs once per coded layer. Reports each part alone and both together."""
    w = torch.cat([d[0] for d in data.values()])
    mu = torch.cat([d[1] for d in data.values()])
    s = torch.cat([d[2] for d in data.values()])
    n, layers = w.numel(), len(data)
    coder = coder_cls(device, values_per_stream=setting[0], block=setting[1],
                      warps=setting[2] if len(setting) > 2 else None, mode=mode)
    coder.search = search
    buf, starts, _ = coder.encode(w, mu, s)
    s1, s2 = torch.cuda.Stream(device), torch.cuda.Stream(device)

    def predict():
        for _ in range(layers):
            layer_fn()

    def both():
        cur = torch.cuda.current_stream(device)
        s1.wait_stream(cur)
        s2.wait_stream(cur)
        with torch.cuda.stream(s1):
            predict()
        with torch.cuda.stream(s2):
            coder.decode(buf, starts, mu, s, n)
        cur.wait_stream(s1)
        cur.wait_stream(s2)
    tp, td, tb = timer(predict, device), timer(lambda: coder.decode(buf, starts, mu, s, n), device), timer(both, device)
    row = {"part": "overlap", "mode": mode, "search": search, "values": n, "setting": list(setting), "predictor_ms": 1e3 * tp, "decode_ms": 1e3 * td,
           "both_ms": 1e3 * tb, "sequential_gvalues_per_s": n / (tp + td) / 1e9, "overlapped_gvalues_per_s": n / tb / 1e9}
    log(f"  overlap: predictor {1e3 * tp:.2f} ms + decode {1e3 * td:.2f} ms one after the other; together "
        f"{1e3 * tb:.2f} ms -> {row['overlapped_gvalues_per_s']:.1f} G values/s (sequential {row['sequential_gvalues_per_s']:.1f})")
    return row


# ---------------------------------------------------------------------------------------------- fused codec
def fused_rows(study, device, configs, log=print, modes=("uniform",), kchunk=64, layer_configs=(), ranks=(),
               lossy_kappas=(), splitzip_source=None, mix_counts=()):
    """The one-step codec on a whole test document (every layer, decoded layer by layer as a receiver must), in
    two implementations: fused (one kernel per direction, configs: tokens per program, rows, warps) and per layer
    (cuBLAS int8 product, then one coding kernel; layer_configs: tokens per program, token chunks on separate CUDA
    streams). Bit-exact check, coded bits, the predictor's exact-model bits, and the time of the encoder's forward
    pass, the encoder and the decoder. Every config runs in the first mode (fused: search-free stage 1; per layer:
    with the stage-1 search); the fastest also with the other stage 1 and in the other modes."""
    import gc
    from .fused import FusedOneStep, prepare, reference_mu
    consts = prepare(study, log)
    doc = study.tests[0]
    L, W = study.L, study.W
    T = doc.tokens - doc.tokens % 128
    words = torch.empty(L, T, W, dtype=torch.int16, device=device)
    for l in range(L):
        w = doc.words(l, device, rows=T)
        words[l] = (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
    n = L * T * W
    rows, model_bits = [], None

    def run(TB, R, nw, mode, search):
        nonlocal model_bits
        tag = f"{TB} tokens x {R} rows, {nw} warps, {mode}{', stage-1 search' if search else ''}"
        row = {"part": "fused", "tokens_per_program": TB, "rows": R, "warps": nw, "mode": mode, "search": search,
               "values": n, "layers": L, "tokens": T, "doc": doc.name}
        try:
            codec = FusedOneStep(consts, study.C, study.hd, tokens_per_program=TB, rows=R, kchunk=kchunk, warps=nw,
                                 mode=mode, search=search)
            xall, prior = codec.model(words)
            buf, starts = codec.encode(words, xall, prior)
            P = T // TB
            scratch = (torch.zeros(2, T, W, dtype=torch.int8, device=device),) + codec._scratch(P)
            dec = codec.decode(buf, starts, T, scratch=scratch)
            ok = bool(torch.equal(dec, words))
            bits = codec.bits(starts, T) / n
            if model_bits is None:               # the predictor's exact-model cost (same for every config)
                tot, pos = 0.0, torch.arange(T, device=device)
                for l in range(L):
                    x = words_to_float(words[l].to(torch.int32) & 0xFFFF)
                    m = reference_mu(consts, l, xall[l - 1].float() if l else None, pos, study.C, study.hd)
                    mult = ((prior[l] * 64.0 + 16.0) * 0.0125).clamp(0.25, 4.0)
                    sp = consts["sp"][l][None, :] * mult[:, None]
                    tot += float(exact_bits(words[l].to(torch.int32) & 0xFFFF, m.double(), sp.double()).sum())
                    del x, m, sp
                model_bits = tot / n
            tm = timer(lambda: codec.model(words), device, warmup=1, reps=3)
            te = timer(lambda: codec.encode(words, xall, prior, out=buf), device, warmup=1, reps=3)
            td = timer(lambda: codec.decode(buf, starts, T, out=dec, scratch=scratch), device, warmup=2, reps=5)
            row.update(bit_exact=ok, stream_bits_per_value=bits, exact_model_bits_per_value=model_bits,
                       model_ms=1e3 * tm, encode_ms=1e3 * te, decode_ms=1e3 * td,
                       encode_gvalues_per_s=n / (tm + te) / 1e9, decode_gvalues_per_s=n / td / 1e9)
            log(f"  fused [{tag}]: exact={ok}, {bits:.4f} bits/value (model {model_bits:.4f}); forward {1e3 * tm:.1f} ms "
                f"+ encode {1e3 * te:.1f} ms, decode {1e3 * td:.1f} ms -> {n / td / 1e9:.1f} G values/s")
            del codec, xall, prior, buf, starts, dec, scratch
        except Exception as exc:                 # e.g. out of registers / shared memory for a config
            import traceback; traceback.print_exc()
            row["error"] = repr(exc)[:240]
            log(f"  fused [{tag}]: not measured ({row['error']})")
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None
        rows.append(row)
        return row

    for TB, R, nw in configs:
        run(TB, R, nw, modes[0], False)
    ok = [r for r in rows if r.get("bit_exact")]
    if ok:
        best = max(ok, key=lambda r: r["decode_gvalues_per_s"])
        run(best["tokens_per_program"], best["rows"], best["warps"], modes[0], True)
        for mode in modes[1:]:
            run(best["tokens_per_program"], best["rows"], best["warps"], mode, False)
    if layer_configs:
        rows += layerwise_rows(consts, study, words, layer_configs, device, log, modes, model_bits, ranks)
        if splitzip_source is not None and mix_counts:
            try:
                rows += mixed_rows(consts, study, words, device, log, splitzip_source, rows, mix_counts)
            except Exception as exc:
                log(f"  mix with SplitZip: not measured ({exc!r})")
                rows.append({"part": "mixed_error", "error": repr(exc)[:400]})
    if lossy_kappas:
        rows += lossy_rows(consts, study, words, device, log, lossy_kappas)
    del words
    return rows


def lossy_rows(consts, study, words, device, log=print, kappas=(0.5, 1.0, 2.0), configs=((64, 2),), ranks=(256,)):
    """The closed-loop lossy per-layer codec (kv_compare.lossy_gpu) on the whole document: forward quantization,
    encode and decode times, the stream's bits, and the decode checked against the encoder's reconstruction."""
    import gc
    from .layerwise import prepare_rank
    from .lossy_gpu import LossyLayerwise
    L, T, W = words.shape
    n = L * T * W
    rows = []
    rcs = {None: None}
    for r in ranks:
        try:
            rcs[r] = prepare_rank(study, consts, r, log)
        except Exception as exc:
            log(f"  lossy: rank {r} not prepared ({exc!r})")
    for kappa in kappas:
        for TB, ch in configs:
            for r, rc in rcs.items():
                tag = f"lossy per layer, step {kappa}, {TB} tokens per program, {ch} chunks" + \
                      (f", rank {r} product (epilogue 128x128x64)" if r else ", dense product")
                row = {"part": "lossy_layerwise", "kappa": kappa, "tokens_per_program": TB, "chunks": ch,
                       "rank": r, "values": n, "layers": L, "tokens": T, "doc": study.tests[0].name}
                try:
                    codec = LossyLayerwise(consts, study.C, study.hd, kappa, tokens_per_program=TB, chunks=ch,
                                           rank=rc, requant="epilogue" if rc else "kernel",
                                           rank_tiles=(128, 128, 64, 4))
                    rec, xall, prior = codec.quantize(words)
                    tq = timer(lambda: codec.quantize(words), device, warmup=1, reps=3)
                    buf, starts, _ = codec.encode(words, rec, xall, prior)
                    te = timer(lambda: codec.encode(words, rec, xall, prior, out=buf), device, warmup=1, reps=3)
                    bufs = codec.buffers(T)
                    dec = codec.decode(buf, starts, T, bufs=bufs)
                    ok = bool(torch.equal(dec, rec))
                    td = timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs), device, warmup=2, reps=5)
                    bits = codec.bits(starts, T) / n
                    x = (words.to(torch.int32) & 0xFFFF)
                    xr = (rec.to(torch.int32) & 0xFFFF)
                    fin = [torch.isfinite(_wf(x[l])) for l in range(L)]
                    rel = float(sum(((_wf(xr[l]) - _wf(x[l]))[fin[l]] ** 2).sum() for l in range(L)) /
                                sum((_wf(x[l])[fin[l]] ** 2).sum() for l in range(L))) ** 0.5
                    row.update(bit_exact=ok, stream_bits_per_value=bits, quantize_ms=1e3 * tq, encode_ms=1e3 * te,
                               decode_ms=1e3 * td, decode_gvalues_per_s=n / td / 1e9, rel_error=rel)
                    log(f"  [{tag}]: exact={ok}, {bits:.4f} bits/value, KV error {100 * rel:.2f}%; quantize "
                        f"{1e3 * tq:.1f} ms + encode {1e3 * te:.1f} ms, decode {1e3 * td:.1f} ms -> "
                        f"{n / td / 1e9:.1f} G values/s")
                    del codec, rec, xall, prior, buf, starts, bufs, dec
                except Exception as exc:
                    row["error"] = repr(exc)[:240]
                    log(f"  [{tag}]: not measured ({row['error']})")
                gc.collect()
                torch.cuda.empty_cache() if device.type == "cuda" else None
                rows.append(row)
    return rows


def mixed_rows(consts, study, words, device, log, splitzip_source, layer_rows, counts=(0, 4, 8, 12, 16, 20, 24, 28)):
    """Our per-layer codec mixed with the authors' SplitZip codec (kv_compare.mixed), measured: for the fastest
    dense and reduced-rank modes, the layers where ours saves the most coded by ours (counts of them), the rest by
    SplitZip; one real stream each, decoded bit-exact and timed. Also SplitZip alone on the whole document."""
    import gc
    from .layerwise import prepare_rank
    from .mixed import MixedOneStep, load_splitzip, mix_orders
    L, T, W = words.shape
    n = L * T * W
    sz, commit = load_splitzip(splitzip_source, device, log)
    cal = torch.stack([study.train[0].words(l, device) for l in range(L)])
    cal16 = (cal - 65536 * (cal >= 32768).to(torch.int32)).to(torch.int16)
    cover = sz.calibrate(cal16.view(torch.bfloat16))
    del cal, cal16
    rows = []
    enc_all = sz.encode(words.view(torch.bfloat16))
    dec_all = sz.decode(enc_all)
    ok = bool(torch.equal(dec_all.view(torch.int16), words.reshape(-1)))
    te = timer(lambda: sz.encode(words.view(torch.bfloat16)), device, warmup=3, reps=10)
    td = timer(lambda: sz.decode(enc_all), device, warmup=3, reps=10)
    sz_bits = 8 * enc_all.compressed_bytes / n
    rows.append({"part": "splitzip_official", "commit": commit, "bit_exact": ok, "calibration": study.train[0].name,
                 "coverage": cover, "stream_bits_per_value": sz_bits, "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
                 "decode_gvalues_per_s": n / td / 1e9, "decode_gbps": 2 * n / td / 1e9, "values": n,
                 "doc": study.tests[0].name})
    log(f"  SplitZip (official {commit[:7]}, calibrated on {study.train[0].name}): exact={ok}, {sz_bits:.4f} bits/value; "
        f"encode {1e3 * te:.1f} ms, decode {1e3 * td:.1f} ms ({2 * n / td / 1e9:.0f} GB/s)")
    del enc_all, dec_all
    sz_layer = [8 * sz.encode(words[l].view(torch.bfloat16)).compressed_bytes / (T * W) for l in range(L)]
    cands = {}
    for r in layer_rows:                             # the fastest dense (cuBLAS) and reduced-rank modes with layer bits
        if r.get("part") != "layerwise" or not r.get("bit_exact") or not r.get("layer_bits_per_value"):
            continue
        if r.get("mode") != "uniform" or r.get("product", "cublas") != "cublas":
            continue
        key = r.get("rank")
        if key not in cands or r["decode_gvalues_per_s"] > cands[key]["decode_gvalues_per_s"]:
            cands[key] = r
    for key, base in sorted(cands.items(), key=lambda kv: (kv[0] is not None, kv[0] or 0)):
        rc = prepare_rank(study, consts, key, log) if key else None
        tiles = tuple(base["rank_tiles"]) if base.get("rank_tiles") else (64, 128, 64, 4)
        kw = dict(tokens_per_program=base["tokens_per_program"], chunks=base["chunks"], rank=rc,
                  requant=base.get("requant") or "kernel", rank_tiles=tiles)
        label = f"rank {key}" if key else "dense"
        order = mix_orders(base["layer_bits_per_value"], sz_layer)
        probe = MixedOneStep(consts, study.C, study.hd, [True] * L, sz, **kw)
        xall, prior = probe.model(words)
        bufs = probe.buffers(T)
        for m in counts:
            mask = [False] * L
            for l in order[:min(m, L)]:
                mask[l] = True
            row = {"part": "mixed", "mode": label, "ours_layers": int(sum(mask)), "ours_mask": mask,
                   "tokens_per_program": base["tokens_per_program"], "chunks": base["chunks"], "values": n,
                   "doc": study.tests[0].name}
            try:
                codec = MixedOneStep(consts, study.C, study.hd, mask, sz, **kw)
                enc = codec.encode(words, xall, prior)
                dec = codec.decode_mixed(enc, T, bufs=bufs)
                ok = bool(torch.equal(dec, words))
                td = timer(lambda: codec.decode_mixed(enc, T, out=dec, bufs=bufs), device, warmup=2, reps=5)
                bits = codec.total_bits(enc, T) / n
                row.update(bit_exact=ok, stream_bits_per_value=bits, decode_ms=1e3 * td,
                           decode_gvalues_per_s=n / td / 1e9)
                log(f"  [mix {label}, {row['ours_layers']} of {L} layers ours]: exact={ok}, {bits:.4f} bits/value, "
                    f"decode {1e3 * td:.1f} ms -> {n / td / 1e9:.1f} G values/s")
                del codec, enc
            except Exception as exc:
                row["error"] = repr(exc)[:240]
                log(f"  [mix {label}, {m} layers ours]: not measured ({row['error']})")
            gc.collect()
            rows.append(row)
        del probe, xall, prior, bufs, rc
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return rows


def _wf(w):
    return (w.to(torch.int32) << 16).view(torch.float32).to(torch.float64)


def layerwise_rows(consts, study, words, configs, device, log=print, modes=("uniform",), model_bits=None, ranks=()):
    """The per-layer one-step codec (kv_compare.layerwise) on the same document: configs (tokens per program,
    chunks)."""
    import gc
    from .fused import FusedOneStep
    from .layerwise import LayerwiseOneStep, prepare_rank
    L, T, W = words.shape
    n = L * T * W
    xall, prior = FusedOneStep(consts, study.C, study.hd, tokens_per_program=32, rows=4, warps=4).model(words)
    rows = []

    def run(TB, chunks, mode, search, features="inline", lean=False, split=False, parts=False, product="cublas",
            tiles=(128, 128, 64, 4), graph=False, rank=None, requant="kernel", rank_tiles=(64, 128, 64, 4)):
        rq = (f", requantized in a Triton product epilogue {rank_tiles[0]}x{rank_tiles[1]}x{rank_tiles[2]} "
              f"({rank_tiles[3]} warps)" if requant == "epilogue" else "")
        variant = ", ".join(x for x in ("features in a separate pass" if features == "pass" else "",
                                        "lean registers" if lean else "",
                                        "products on high-priority streams" if split else "",
                                        f"Triton product {tiles[0]}x{tiles[1]}x{tiles[2]} ({tiles[3]} warps)"
                                        if product == "triton" else "",
                                        "CUDA graph" if graph else "",
                                        f"reduced-rank product (rank {rank['rank']}{rq})" if rank else "") if x)
        tag = f"per layer, {TB} tokens per program, {chunks} chunk{'s' if chunks > 1 else ''}, {mode}" + \
              (", stage-1 search" if search else "") + (f", {variant}" if variant else "")
        row = {"part": "layerwise", "tokens_per_program": TB, "chunks": chunks, "mode": mode, "search": search,
               "variant": variant, "values": n, "layers": L, "tokens": T, "doc": study.tests[0].name,
               "rank": rank["rank"] if rank else None, "requant": requant if rank else None,
               "rank_tiles": list(rank_tiles) if rank and requant == "epilogue" else None, "product": product,
               "gemm_tiles": list(tiles) if product == "triton" else None}
        try:
            codec = LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=TB, mode=mode, search=search,
                                     chunks=chunks, features=features, lean=lean, split_streams=split,
                                     product=product, gemm_tiles=tiles, rank=rank, requant=requant,
                                     rank_tiles=rank_tiles)
            xa, pr = (xall, prior)
            if rank is not None:                     # the rank predictor's own forward pass (its priors differ)
                xa, pr = codec.model(words)
                row["model_ms"] = 1e3 * timer(lambda: codec.model(words), device, warmup=1, reps=3)
            lw = torch.zeros(L, dtype=torch.int64, device=words.device)
            buf, starts = codec.encode(words, xa, pr, layer_words=lw)
            row["layer_bits_per_value"] = [16 * int(v) / (T * W) for v in lw.tolist()]   # states, alignment: layer 0
            bufs = codec.buffers(T)
            dec = codec.decode(buf, starts, T, bufs=bufs)
            ok = bool(torch.equal(dec, words))
            bits = codec.bits(starts, T) / n
            te = timer(lambda: codec.encode(words, xa, pr, out=buf), device, warmup=1, reps=3)
            if graph:                                # the whole decode captured once, then replayed
                g = torch.cuda.CUDAGraph()
                side = torch.cuda.Stream(device)
                side.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(side):
                    codec.decode(buf, starts, T, out=dec, bufs=bufs)
                torch.cuda.current_stream(device).wait_stream(side)
                with torch.cuda.graph(g):
                    codec.decode(buf, starts, T, out=dec, bufs=bufs)
                dec.zero_()
                g.replay()
                torch.cuda.synchronize(device)
                ok = ok and bool(torch.equal(dec, words))
                td = timer(g.replay, device, warmup=2, reps=5)
            else:
                td = timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs), device, warmup=2, reps=5)
            row["bit_exact"] = ok
            row.update(bit_exact=ok, stream_bits_per_value=bits, exact_model_bits_per_value=model_bits,
                       encode_ms=1e3 * te, decode_ms=1e3 * td, decode_gvalues_per_s=n / td / 1e9)
            log(f"  [{tag}]: exact={ok}, {bits:.4f} bits/value; encode {1e3 * te:.1f} ms (after the forward pass), "
                f"decode {1e3 * td:.1f} ms -> {n / td / 1e9:.1f} G values/s")
            if parts:                                # where the decode time goes
                accs = codec.products(xa)
                tc = timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs, accs=accs), device, warmup=2, reps=5)
                tp = timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs, products_only=True), device,
                           warmup=2, reps=5)
                row.update(coding_only_ms=1e3 * tc, products_only_ms=1e3 * tp)
                log(f"    parts: products alone {1e3 * tp:.1f} ms, coding alone {1e3 * tc:.1f} ms, together "
                    f"{1e3 * td:.1f} ms (overlap hides {1e3 * (tp + tc - td):.1f} ms)")
                del accs
            del codec, buf, starts, dec, bufs
        except Exception as exc:
            row["error"] = repr(exc)[:240]
            log(f"  [{tag}]: not measured ({row['error']})")
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None
        rows.append(row)

    for TB, chunks in configs:
        run(TB, chunks, modes[0], True)
    try:                                             # a separate features pass (what a layer coded otherwise costs)
        from .layerwise import _lfeat_kernel
        import triton
        lc = LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=configs[0][0])
        xo = torch.empty(T, W, dtype=torch.int8, device=device)
        k = lc.k
        tf = timer(lambda: lc._launch(_lfeat_kernel, (triton.cdiv(T * W, 1024),), words[1], xo, k["isx"][1], k["cos"],
                                      k["sin"], 0, T * W, W, study.C, HD=study.hd, BLOCK=1024, warps=4),
                   device, warmup=2, reps=10)
        rows.append({"part": "feature_pass", "ms_per_layer": 1e3 * tf, "tokens": T, "values_per_layer": T * W})
        log(f"  features pass alone: {1e3 * tf:.3f} ms per layer")
        del lc, xo
    except Exception as exc:
        log(f"  features pass: not measured ({exc!r})")
    ok = [r for r in rows if r.get("bit_exact")]
    if ok:
        best = max(ok, key=lambda r: r["decode_gvalues_per_s"])
        TB, ch = best["tokens_per_program"], best["chunks"]
        run(TB, ch, modes[0], True, parts=True)              # the same again, timed in parts
        run(TB, ch, modes[0], False)
        run(TB, ch, modes[0], True, graph=True)
        for tiles in ((128, 128, 64, 4), (64, 128, 64, 4), (64, 64, 64, 4)):
            run(TB, ch, modes[0], True, product="triton", tiles=tiles, parts=tiles == (64, 128, 64, 4))
            run(TB, ch, modes[0], True, product="triton", tiles=tiles, graph=True)
        for c2 in (4, 8):
            run(TB, c2, modes[0], True, product="triton", tiles=(64, 128, 64, 4), graph=True)
            run(TB, c2, modes[0], True, graph=True)
        for mode in modes[1:]:
            run(TB, ch, mode, True)
        for r in ranks:                              # the fast-link mode: a reduced-rank product
            try:
                rc = prepare_rank(study, consts, r, log)
            except Exception as exc:
                log(f"  reduced rank {r}: not prepared ({exc!r})")
                continue
            first = len(rows)
            run(TB, ch, modes[0], True, rank=rc, parts=True)             # cuBLAS, then a requantization kernel
            for rt in ((64, 128, 64, 4), (128, 128, 64, 4), (64, 64, 64, 4)):
                run(TB, ch, modes[0], True, rank=rc, requant="epilogue", rank_tiles=rt, parts=rt == (64, 128, 64, 4))
            done = [x for x in rows[first:] if x.get("bit_exact")]
            if done:                                 # the fastest of these with other chunk counts, and as a graph
                b = max(done, key=lambda x: x["decode_gvalues_per_s"])
                rt = next((t for t in ((64, 128, 64, 4), (128, 128, 64, 4), (64, 64, 64, 4))
                           if b["variant"].endswith(f"{t[0]}x{t[1]}x{t[2]} ({t[3]} warps))")), (64, 128, 64, 4))
                for c2 in (1, 2, 4):
                    if c2 != ch:
                        run(TB, c2, modes[0], True, rank=rc, requant=b["requant"], rank_tiles=rt, parts=c2 == 1)
                run(TB, ch, modes[0], True, rank=rc, requant=b["requant"], rank_tiles=rt, graph=True)
            del rc
    del xall, prior
    return rows


# ---------------------------------------------------------------------------------------- SplitZip-format codec
def szcodec_rows(consts, study, words, cal, device, log=print, ranks=(None, 256), configs=((16, 4), (32, 4), (32, 8),
                 (64, 8)), row_configs=((16, 1), (32, 1), (64, 1), (32, 2), (32, 4)), chunk_counts=(2, 4),
                 splitzip_source=None, counts=(), rank_tiles=(128, 128, 64, 4)):
    """Our prediction with SplitZip-style fixed-length codes (kv_compare.szcodec) on the whole document, per
    product (dense through cuBLAS; reduced rank with the second product inside the decoding kernel or through
    memory) and per layout: "tile" (configs: tokens x warps, a program decodes a tokens x 128 tile at once) and
    "row" (row_configs: token by token over the head's 128 lanes, products through memory); the fastest of each
    layout again with token chunks on CUDA streams. Code books fitted on `cal` (the calibration documents' word
    tensors). Bit-exact check, the stream's bits (per layer too), encode (once per layout) and decode times. For
    the fastest of each layout, where the time goes (products alone, coding alone from precomputed products, and
    the coding without stream reads or without prefix sums); the fastest overall as a CUDA graph. With
    splitzip_source: the authors' SplitZip on the same document, and mixes of the two per layer."""
    import gc
    import time
    from .layerwise import prepare_rank
    from .szcodec import SplitZipFormatCodec
    L, T, W = words.shape
    n = L * T * W
    rows = []
    sz = commit = None
    if splitzip_source is not None:
        try:
            from .mixed import load_splitzip
            sz, commit = load_splitzip(splitzip_source, device, log)
            cover = sz.calibrate(cal[0].view(torch.bfloat16))
            enc = sz.encode(words.view(torch.bfloat16))
            dec = sz.decode(enc)
            ok = bool(torch.equal(dec.view(torch.int16), words.reshape(-1)))
            td = timer(lambda: sz.decode(enc), device, warmup=3, reps=10)
            te = timer(lambda: sz.encode(words.view(torch.bfloat16)), device, warmup=3, reps=10)
            b = 8 * enc.compressed_bytes / n
            rows.append({"part": "splitzip_official", "commit": commit, "bit_exact": ok, "coverage": cover,
                         "stream_bits_per_value": b, "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
                         "decode_gvalues_per_s": n / td / 1e9, "decode_gbps": 2 * n / td / 1e9, "values": n,
                         "doc": study.tests[0].name,
                         "layer_bits_per_value": [8 * sz.encode(words[l].view(torch.bfloat16)).compressed_bytes /
                                                  (T * W) for l in range(L)]})
            log(f"  SplitZip (official {commit[:7]}): exact={ok}, {b:.4f} bits/value, decode {1e3 * td:.2f} ms")
            del enc, dec
        except Exception as exc:
            log(f"  SplitZip official: not measured ({exc!r})")
            sz = None
    for r in ranks:
        try:
            rc = prepare_rank(study, consts, r, log) if r else None
        except Exception as exc:
            log(f"  SplitZip format: rank {r} not prepared ({exc!r})")
            continue
        tiles = rank_tiles if not r or (T % (2 * rank_tiles[0]) == 0 and r % rank_tiles[1] == 0 and
                                        W % rank_tiles[2] == 0) else (16, 16, 32, 4)     # (small test models)
        kind = f"rank {r}" if r else "dense"
        books, done, enc_ms = {}, [], {}             # code books per escape cost (the one-read layout's differ)

        def make(layout, product, BM, nw, chunks):
            codec = SplitZipFormatCodec(consts, study.C, study.hd, rank=rc, product=product, tokens_per_program=BM,
                                        warps=nw, rank_tiles=tiles, layout=layout, chunks=chunks)
            return codec

        def run(layout, product, BM, nw, chunks=1):
            tag = (f"SplitZip format, {kind} product"
                   f"{' inside the decoding kernel' if product == 'fused' else ' through memory'}, {layout} layout, "
                   f"{BM} tokens x one head, {nw} warps{f', {chunks} chunks' if chunks > 1 else ''}")
            row = {"part": "szcodec", "rank": r, "product": product, "layout": layout, "tokens_per_program": BM,
                   "warps": nw, "chunks": chunks, "values": n, "layers": L, "tokens": T, "doc": study.tests[0].name,
                   "graph": False}
            try:
                codec = make(layout, product, BM, nw, chunks)
                if codec.esc_raw not in books:
                    t0 = time.perf_counter()
                    books[codec.esc_raw] = codec.calibrate(cal)
                    row["calibrate_s"] = time.perf_counter() - t0
                else:
                    codec.share_books(books[codec.esc_raw])
                lb = []
                stream, starts = codec.encode(words, layer_bits=lb)
                if (layout, BM) not in enc_ms:           # the packer's time depends on the layout only
                    enc_ms[(layout, BM)] = 1e3 * timer(lambda: codec.encode(words), device, warmup=1, reps=2)
                bufs = codec.buffers(T)
                dec = codec.decode(stream, starts, T, bufs=bufs)
                ok = bool(torch.equal(dec, words))
                td = timer(lambda: codec.decode(stream, starts, T, out=dec, bufs=bufs), device, warmup=2, reps=5)
                bits = codec.bits(stream, starts) / n
                row.update(bit_exact=ok, stream_bits_per_value=bits, calibration_bits=codec.calibration_bits,
                           layer_bits_per_value=[v / (T * W) for v in lb], encode_ms=enc_ms[(layout, BM)],
                           decode_ms=1e3 * td, decode_gvalues_per_s=n / td / 1e9)
                log(f"  [{tag}]: exact={ok}, {bits:.4f} bits/value (calibration {codec.calibration_bits:.3f}); "
                    f"encode {enc_ms[(layout, BM)]:.0f} ms (PyTorch packer), decode {1e3 * td:.2f} ms -> "
                    f"{n / td / 1e9:.1f} G values/s")
                if ok:
                    done.append((td, layout, product, BM, nw, chunks))
                del codec, stream, starts, dec, bufs
            except Exception as exc:
                row["error"] = repr(exc)[:400]
                log(f"  [{tag}]: not measured ({row['error']})")
            gc.collect()
            torch.cuda.empty_cache() if device.type == "cuda" else None
            rows.append(row)

        def best(layout):
            c = [d for d in done if d[1] == layout]
            return min(c) if c else None

        prods = ["acc"] if rc is None else ["fused", "acc"]     # reduced rank: the second product inside or not
        for BM, nw in configs:                       # tile layout
            run("tile", prods[0], BM, nw)
        for layout in ("row", "row1"):               # row layouts: two reads per value, or one
            for BM, nw in row_configs:
                for product in prods:
                    run(layout, product, BM, nw)
        for esc_raw, bk in books.items():            # where the bits go, and the room left in the code books
            try:
                from .szcodec import bit_budget
                t0 = time.perf_counter()
                bud = bit_budget(bk, words, cal)
                rows.append(dict(bud, part="szcodec_budget", rank=r, escape_keeps_raw=esc_raw,
                                 doc=study.tests[0].name, seconds=time.perf_counter() - t0))
                log(f"  [SplitZip format, {kind}, {'one-read' if esc_raw else 'two-read'} code books, bit budget]: "
                    f"codes {bud['code']:.3f} + raw {bud['raw']:.3f} + escapes {bud['escape']:.3f} = "
                    f"{bud['books_bits']:.3f} bits/value ({100 * bud['escape_rate']:.2f}% escaped); 8-bit codes "
                    f"{bud['k8_bits']:.3f}; books fitted on this document {bud['oracle_bits']:.3f} "
                    f"(8-bit {bud['oracle_k8_bits']:.3f}); entropy given the context {bud['entropy_bits']:.3f}")
            except Exception as exc:
                log(f"  SplitZip format bit budget: not measured ({exc!r})")
            gc.collect()
            torch.cuda.empty_cache() if device.type == "cuda" else None
        for layout in ("tile", "row", "row1"):       # the fastest of each layout with token chunks on streams
            b = best(layout)
            if b and device.type == "cuda":
                for ch in chunk_counts:
                    if T % (b[3] * ch) == 0 and (b[2] != "fused" or (T // ch) % tiles[0] == 0):
                        run(layout, b[2], b[3], b[4], ch)
        if done:                                     # parts, a CUDA graph, then mixed with SplitZip
            try:
                for layout in ("tile", "row", "row1"):
                    b = best(layout)
                    if not b:
                        continue
                    _, lay, product, BM, nw, ch = b
                    codec = make(lay, product, BM, nw, 1)
                    codec.share_books(books[codec.esc_raw])
                    stream, starts = codec.encode(words)
                    bufs = codec.buffers(T)
                    dec = torch.empty_like(words)
                    accs = codec.products(words)
                    part = {"part": "szcodec_parts", "rank": r, "product": product, "layout": lay,
                            "tokens_per_program": BM, "warps": nw, "values": n, "doc": study.tests[0].name}
                    part["whole_ms"] = 1e3 * timer(lambda: codec.decode(stream, starts, T, out=dec, bufs=bufs),
                                                   device, warmup=2, reps=5)
                    part["products_only_ms"] = 1e3 * timer(lambda: codec.decode(stream, starts, T, out=dec, bufs=bufs,
                                                                                products_only=True),
                                                           device, warmup=2, reps=5)
                    part["coding_only_ms"] = 1e3 * timer(lambda: codec.decode(stream, starts, T, out=dec, bufs=bufs,
                                                                              accs=accs), device, warmup=2, reps=5)
                    ok = bool(torch.equal(codec.decode(stream, starts, T, out=dec, bufs=bufs, accs=accs), words))
                    part["coding_only_exact"] = ok
                    for prof, key in ((1, "coding_no_reads_ms"), (2, "coding_no_prefix_sums_ms")):
                        part[key] = 1e3 * timer(lambda: codec.decode(stream, starts, T, out=dec, bufs=bufs, accs=accs,
                                                                     prof=prof), device, warmup=2, reps=5)
                    rows.append(part)
                    log(f"  [SplitZip format, {kind}, {lay} {BM} x {nw}, where the time goes]: whole "
                        f"{part['whole_ms']:.2f} ms; products alone {part['products_only_ms']:.2f} ms"
                        f"{' (first product only; the second is inside the kernel)' if product == 'fused' else ''}; "
                        f"coding alone {part['coding_only_ms']:.2f} ms (exact={ok}), without stream reads "
                        f"{part['coding_no_reads_ms']:.2f} ms, without prefix sums {part['coding_no_prefix_sums_ms']:.2f} ms")
                    del codec, stream, starts, bufs, dec, accs
                    gc.collect()
                    torch.cuda.empty_cache()
                if device.type != "cuda":
                    raise RuntimeError("no CUDA: no graph or SplitZip mix")
                _, lay, product, BM, nw, ch = min(done)
                codec = make(lay, product, BM, nw, ch)
                codec.share_books(books[codec.esc_raw])
                lb = []
                stream, starts = codec.encode(words, layer_bits=lb)
                bufs = codec.buffers(T)
                dec = torch.empty_like(words)
                row = {"part": "szcodec", "rank": r, "product": product, "layout": lay, "tokens_per_program": BM,
                       "warps": nw, "chunks": ch, "values": n, "layers": L, "tokens": T, "doc": study.tests[0].name,
                       "graph": True}
                try:
                    g = torch.cuda.CUDAGraph()
                    side = torch.cuda.Stream(device)
                    side.wait_stream(torch.cuda.current_stream(device))
                    with torch.cuda.stream(side):
                        codec.decode(stream, starts, T, out=dec, bufs=bufs)
                    torch.cuda.current_stream(device).wait_stream(side)
                    with torch.cuda.graph(g):
                        codec.decode(stream, starts, T, out=dec, bufs=bufs)
                    dec.zero_()
                    g.replay()
                    torch.cuda.synchronize(device)
                    ok = bool(torch.equal(dec, words))
                    td = timer(g.replay, device, warmup=2, reps=5)
                    bits = codec.bits(stream, starts) / n
                    row.update(bit_exact=ok, stream_bits_per_value=bits, layer_bits_per_value=[v / (T * W) for v in lb],
                               decode_ms=1e3 * td, decode_gvalues_per_s=n / td / 1e9)
                    log(f"  [SplitZip format, {kind}, {lay} {BM} x {nw}, {ch} chunks, CUDA graph]: exact={ok}, "
                        f"decode {1e3 * td:.2f} ms -> {n / td / 1e9:.1f} G values/s")
                    del g
                except Exception as exc:
                    row["error"] = repr(exc)[:400]
                    log(f"  [SplitZip format, CUDA graph]: not measured ({row['error']})")
                rows.append(row)
                szr = next((x for x in rows if x["part"] == "splitzip_official" and x.get("bit_exact")), None)
                if sz is not None and szr is not None and counts:
                    sav = [z - b for b, z in zip([v / (T * W) for v in lb], szr["layer_bits_per_value"])]
                    order = sorted(range(L), key=lambda l: -sav[l])
                    for m in counts:
                        mask = [False] * L
                        for l in order[:min(m, L)]:
                            mask[l] = True
                        mrow = {"part": "szcodec_mixed", "rank": r, "product": product, "layout": lay,
                                "tokens_per_program": BM, "warps": nw, "chunks": ch, "ours_layers": int(sum(mask)),
                                "ours_mask": mask, "values": n, "doc": study.tests[0].name}
                        try:
                            stream_m, starts_m, enc = codec.encode(words, ours=mask, other=sz)
                            dec = codec.decode(stream_m, starts_m, T, bufs=bufs, ours=mask, sz=enc, other=sz)
                            ok = bool(torch.equal(dec, words))
                            td = timer(lambda: codec.decode(stream_m, starts_m, T, out=dec, bufs=bufs, ours=mask,
                                                            sz=enc, other=sz), device, warmup=2, reps=5)
                            bits = codec.bits(stream_m, starts_m, enc, mask) / n
                            mrow.update(bit_exact=ok, stream_bits_per_value=bits, decode_ms=1e3 * td,
                                        decode_gvalues_per_s=n / td / 1e9)
                            log(f"  [SplitZip format {kind} + SplitZip, {mrow['ours_layers']} of {L} layers ours]: "
                                f"exact={ok}, {bits:.4f} bits/value, decode {1e3 * td:.2f} ms")
                            del stream_m, starts_m, enc
                        except Exception as exc:
                            mrow["error"] = repr(exc)[:400]
                            log(f"  [SplitZip format mix, {m} layers]: not measured ({mrow['error']})")
                        gc.collect()
                        rows.append(mrow)
                del codec, stream, starts, bufs, dec
            except Exception as exc:                 # the sweep's rows are kept
                log(f"  SplitZip format: parts / graph / mix not measured ({exc!r})")
                rows.append({"part": "szcodec_error", "rank": r, "error": repr(exc)[:400]})
        del books, rc
        gc.collect()
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return rows
