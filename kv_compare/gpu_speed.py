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
        else:
            rnd = lambda *shape: torch.randint(-127, 128, shape, device=device, generator=gen, dtype=torch.int8)
            self.Bt = rnd(N, K)                              # [N, K] row-major = [K, N] column-major for _int_mm
            self.hi = rnd(T, K)
            self.lo = rnd(T, K) if fmt == "int8x16" else None
            self.corr = torch.randint(-1000, 1000, (N,), device=device, generator=gen, dtype=torch.int64)

    def __call__(self):
        if self.fmt in ("fp32", "bf16"):
            return self.X @ self.B
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
    if v["ne"]:
        if v.get("rank"):
            parts += [Product(fmt, T, v["ne"], v["rank"], device, gen), Product(fmt, T, v["rank"], W, device, gen)]
        else:
            parts.append(Product(fmt, T, v["ne"], W, device, gen))
    parts += [Product(fmt, T, g * gsz, gsz, device, gen) for g in range(1, v["steps"])]
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
            try:
                layer, products = predictor_layer(Hk, hd, Hq, T, device, v, fmt, gen)
                att = v.get("att")
                sec = timer(layer, device)
                parts_only = timer(products, device) if att else sec
                rows.append({"part": "predictor", "variant": v["label"], "format": fmt, "tokens": T,
                             "ms_per_layer": 1e3 * sec, "gvalues_per_s": T * W / sec / 1e9,
                             "gb_per_s": 2 * T * W / sec / 1e9,
                             "ms_without_attention": 1e3 * parts_only if att else None})
                log(f"  predictor {v['label']} [{fmt}]: {1e3 * sec:.2f} ms/layer, {T * W / sec / 1e9:.1f} G values/s")
                del layer, products
            except Exception as exc:                         # e.g. _int_mm unsupported on this device
                rows.append({"part": "predictor", "variant": v["label"], "format": fmt, "error": repr(exc)})
                log(f"  predictor {v['label']} [{fmt}]: not measured ({exc!r})")
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


def coder_rows(coder_cls, device, data, settings, log=print, max_values=None):
    """Encode and decode the concatenated layers with each (values_per_stream, block) setting."""
    w = torch.cat([d[0] for d in data.values()])
    mu = torch.cat([d[1] for d in data.values()])
    s = torch.cat([d[2] for d in data.values()])
    if max_values:
        w, mu, s = w[:max_values], mu[:max_values], s[:max_values]
    n = w.numel()
    exact = float(exact_bits(w.to(torch.int32) & 0xFFFF, mu.double(), s.double()).mean())
    rows = []
    for setting in settings:
        V, block = setting[0], setting[1]
        warps = setting[2] if len(setting) > 2 else None
        coder = coder_cls(device, values_per_stream=V, block=block, warps=warps)
        buf, starts, _ = coder.encode(w, mu, s)
        dec = coder.decode(buf, starts, mu, s, n)
        ok = bool(torch.equal(dec, w))
        bits = coder.bits(starts, n) / n
        ideal = coder.ideal_bits(w, mu, s) / n
        reps = 10 if device.type == "cuda" else 1
        te = timer(lambda: coder.encode(w, mu, s), device, warmup=1 if device.type != "cuda" else 3, reps=reps)
        td = timer(lambda: coder.decode(buf, starts, mu, s, n), device, warmup=1 if device.type != "cuda" else 3, reps=reps)
        row = {"part": "coder", "values": n, "values_per_stream": V, "block": block, "warps": coder.warps, "bit_exact": ok,
               "stream_bits_per_value": bits, "symbol_bits_per_value": ideal, "exact_model_bits_per_value": exact,
               "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
               "encode_gvalues_per_s": n / te / 1e9, "decode_gvalues_per_s": n / td / 1e9,
               "encode_gb_per_s": 2 * n / te / 1e9, "decode_gb_per_s": 2 * n / td / 1e9}
        rows.append(row)
        log(f"  coder V={V} block={block} warps={coder.warps}: exact={ok}, {bits:.4f} bits/value (model {exact:.4f}), "
            f"encode {row['encode_gb_per_s']:.1f} GB/s, decode {row['decode_gb_per_s']:.1f} GB/s")
        del buf, starts, dec
    return rows


def overlap_row(coder_cls, device, data, setting, layer_fn, log=print):
    """Decode on one CUDA stream while the predictor runs on another (as a pipelined decoder would, on different
    token blocks): the predictor runs once per coded layer. Reports each part alone and both together."""
    w = torch.cat([d[0] for d in data.values()])
    mu = torch.cat([d[1] for d in data.values()])
    s = torch.cat([d[2] for d in data.values()])
    n, layers = w.numel(), len(data)
    coder = coder_cls(device, values_per_stream=setting[0], block=setting[1],
                      warps=setting[2] if len(setting) > 2 else None)
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
    row = {"part": "overlap", "values": n, "setting": list(setting), "predictor_ms": 1e3 * tp, "decode_ms": 1e3 * td,
           "both_ms": 1e3 * tb, "sequential_gvalues_per_s": n / (tp + td) / 1e9, "overlapped_gvalues_per_s": n / tb / 1e9}
    log(f"  overlap: predictor {1e3 * tp:.2f} ms + decode {1e3 * td:.2f} ms one after the other; together "
        f"{1e3 * tb:.2f} ms -> {row['overlapped_gvalues_per_s']:.1f} G values/s (sequential {row['sequential_gvalues_per_s']:.1f})")
    return row
