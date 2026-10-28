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
def fused_rows(study, device, configs, log=print, modes=("uniform",), kchunk=64, layer_configs=(), ranks=()):
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
    del words
    return rows


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
            tiles=(128, 128, 64, 4), graph=False, rank=None):
        variant = ", ".join(x for x in ("features in a separate pass" if features == "pass" else "",
                                        "lean registers" if lean else "",
                                        "products on high-priority streams" if split else "",
                                        f"Triton product {tiles[0]}x{tiles[1]}x{tiles[2]} ({tiles[3]} warps)"
                                        if product == "triton" else "",
                                        "CUDA graph" if graph else "",
                                        f"reduced-rank product (rank {rank['rank']})" if rank else "") if x)
        tag = f"per layer, {TB} tokens per program, {chunks} chunk{'s' if chunks > 1 else ''}, {mode}" + \
              (", stage-1 search" if search else "") + (f", {variant}" if variant else "")
        row = {"part": "layerwise", "tokens_per_program": TB, "chunks": chunks, "mode": mode, "search": search,
               "variant": variant, "values": n, "layers": L, "tokens": T, "doc": study.tests[0].name}
        try:
            codec = LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=TB, mode=mode, search=search,
                                     chunks=chunks, features=features, lean=lean, split_streams=split,
                                     product=product, gemm_tiles=tiles, rank=rank)
            xa, pr = (xall, prior)
            if rank is not None:                     # the rank predictor's own forward pass (its priors differ)
                xa, pr = codec.model(words)
                row["model_ms"] = 1e3 * timer(lambda: codec.model(words), device, warmup=1, reps=3)
            buf, starts = codec.encode(words, xa, pr)
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
            run(TB, ch, modes[0], True, rank=rc, parts=True)
            run(TB, ch, modes[0], True, rank=rc, graph=True)
            del rc
    del xall, prior
    return rows
