"""Baselines re-implemented from their published descriptions, run on the same captured KV.

Lossless (exact BF16):
  SplitZip-style   one top-16 exponent codebook calibrated on TRAIN (4-bit code), sign+mantissa byte raw,
                   an exponent outside the codebook is an escape (16-bit position + 8-bit exponent). (The official
                   SplitZip codec was run separately: 12.04 bits per value, its measured speed is used.)
  ExANS-style      block = 256 tokens x all layers x K/V; each (layer, K/V) gets its most common exponent as the
                   centre; centred exponents coded with one block-local entropy model (ideal entropy + 24 bits per
                   symbol for the model + 128-byte centre table + 8 rANS states of 32 bits); sign+mantissa raw.
Lossy, following the authors' code (see each function for the source and every setting taken from it):
  AQUA-KV          aqua_official_fit / aqua_official: goodevening13/aquakv at e744183. Keys of layer l predicted
                   from the reconstructed keys of layer l-1 (RoPE removed), values from the reconstructed keys of
                   layer l and values of layer l-1 (linear + bias in BF16, fitted in float32 with pinverse and
                   1e-3 x mean-diagonal damping on the calibration documents' own closed-loop reconstructions);
                   residual quantized with HIGGS (Hadamard transform over 1024 channels, per-token L2 scale,
                   2-D EDEN grid of 16 / 64 / 256 points = 2 / 3 / 4 bits); layer 0 HIGGS 4-bit without a
                   predictor; the first 4 tokens (attention sinks) kept in BF16.
  KIVI             kivi_official: jy-yuan/KIVI at 876b4d2 (prefill path of models/llama_kivi.py, quant/new_pack.py),
                   paper settings group size 32, residual length 128: keys per channel over 32-token groups (the
                   last T mod 128 tokens kept in BF16), values per token over 32-channel groups (the last 128 tokens
                   kept in BF16); asymmetric min-max, scale and minimum in BF16 (16 bits each per group).
  FP8 cast         every value cast to FP8 e4m3, no scale: 8 bits per value.
Earlier stand-ins (aqua_fit / aqua_style with a scalar min-max quantizer, kivi_style with 128-token / 64-channel
groups and no BF16 window) are kept for reference and the tests; the benchmark uses the versions above.
"""
from __future__ import annotations

import numpy as np
import torch

from pathlib import Path

from .core import to_bf16, words_to_float


# ---------------------------------------------------------------------------------------- lossless
def exponents(doc, layer):
    return np.concatenate([(np.asarray(doc.K[layer]).astype(np.int64) >> 7) & 255,
                           (np.asarray(doc.V[layer]).astype(np.int64) >> 7) & 255], 1)


def splitzip_codebook(train, layers):
    hist = np.zeros(256, np.int64)
    for d in train:
        for layer in range(layers):
            hist += np.bincount(exponents(d, layer).ravel(), minlength=256)
    inside = np.zeros(256, bool)
    inside[np.argsort(hist)[::-1][:16]] = True
    return inside


def splitzip_token_bits(doc, layers, inside):
    """[layers, tokens] mean bits per value."""
    out = np.zeros((layers, doc.tokens))
    for layer in range(layers):
        e = exponents(doc, layer)
        out[layer] = 12 + 24 * (~inside[e]).mean(1)
    return out


def exans_bits(doc, layers, t0, t1, block=256):
    """Mean bits per value over tokens [t0, t1) and the given layers, blocks of `block` tokens."""
    total, values = 0.0, 0
    for b0 in range(t0, t1, block):
        b1 = min(t1, b0 + block)
        symbols = []
        for layer in layers:
            for array in (doc.K, doc.V):
                e = (np.asarray(array[layer][b0:b1]).astype(np.int64) >> 7) & 255
                centre = np.bincount(e.ravel(), minlength=256).argmax()
                symbols.append(((e + 127 - centre) % 256).ravel())
        s = np.concatenate(symbols)
        counts = np.bincount(s, minlength=256)
        p = counts[counts > 0] / counts.sum()
        total += float(-(counts[counts > 0] * np.log2(p)).sum()) + 24 * int((counts > 0).sum()) + 128 * 8 + 8 * 32 + 8 * len(s)
        values += len(s)
    return total / values


# ---------------------------------------------------------------------------------------- lossy helpers
def quant_token_groups(x, bits, group=64):
    """Asymmetric min-max b-bit quantization per row and group of columns (scale/offset stored in BF16)."""
    n, c = x.shape
    group = group if c % group == 0 else c
    g = x.reshape(n, c // group, group)
    lo = to_bf16(g.amin(-1, keepdim=True))
    hi = to_bf16(g.amax(-1, keepdim=True))
    levels = 2 ** bits - 1
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    q = torch.round((g - lo) / scale).clamp(0, levels)
    return (lo + q * scale).reshape(n, c)


def quant_channel_blocks(x, bits, tokens=128):
    """Asymmetric min-max b-bit quantization per column over blocks of rows."""
    out = torch.empty_like(x)
    levels = 2 ** bits - 1
    for t0 in range(0, x.shape[0], tokens):
        blk = x[t0:t0 + tokens]
        lo = to_bf16(blk.amin(0, keepdim=True))
        hi = to_bf16(blk.amax(0, keepdim=True))
        scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
        out[t0:t0 + tokens] = lo + torch.round((blk - lo) / scale).clamp(0, levels) * scale
    return out


def fp8_e4m3(x):
    a = x.abs()
    e = torch.floor(torch.log2(a.clamp_min(2.0 ** -30)))
    q = torch.exp2(e.clamp_min(-6) - 3)
    return torch.sign(x) * torch.minimum(torch.round(a / q) * q, torch.full_like(a, 448.0))


def _lsq_stream(pairs, ridge=1e-6):
    """Least squares with a bias column from streamed (X, Y) blocks: normal equations in float64."""
    G = R = None
    for X, Y in pairs:
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1).to(torch.float64)
        Y = Y.to(torch.float64)
        G = X1.T @ X1 if G is None else G + X1.T @ X1
        R = X1.T @ Y if R is None else R + X1.T @ Y
    G += ridge * torch.trace(G) / G.shape[0] * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return torch.linalg.solve(G, R)


def _apply(Wm, X):
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1) @ Wm


def aqua_fit(study, layers, log=None, bits=None):
    """AQUA-KV-style predictors per layer: keys of layer l from keys of layer l-1, values of layer l from values
    of layer l-1 and the current keys (RoPE removed), least squares on the calibration documents.

    bits=None fits on exact KV. bits=b fits each layer on the calibration documents' own closed-loop b-bit
    reconstructions (layer 0 at 4 bits, then the residuals quantized exactly as aqua_style does), so the
    predictors see the quantization noise they will be applied to. A predictor fitted on exact KV can amplify
    that noise from layer to layer at 2-3 bits."""
    C = study.C
    out = {}
    if bits is None:
        for layer in range(1, layers):
            WK = _lsq_stream((study.feats(d, layer - 1)[:, :C], study.feats(d, layer)[:, :C]) for d in study.train)
            WV = _lsq_stream((torch.cat([study.feats(d, layer - 1)[:, C:], study.feats(d, layer)[:, :C]], 1),
                              study.feats(d, layer)[:, C:]) for d in study.train)
            out[layer] = (WK, WV)
            study.forget(layer - 1)
            if log:
                log(f"  AQUA-KV-style: layer {layer} fitted")
        return out
    prev = {}
    for d in study.train:                             # layer 0: direct 4-bit, as aqua_style codes it
        pos = study.positions(d.tokens)
        xd = study.feats(d, 0, cache=False).double()
        prev[d.name] = study.rope.derotate_row(to_bf16(study.rope.rotate_row(quant_token_groups(xd, 4), pos)), pos)
    for layer in range(1, layers):
        cur = {d.name: study.feats(d, layer, cache=False).double() for d in study.train}
        WK = _lsq_stream((prev[d.name][:, :C], cur[d.name][:, :C]) for d in study.train)
        kr = {}
        for d in study.train:
            kp = _apply(WK, prev[d.name][:, :C])
            kr[d.name] = kp + quant_token_groups(cur[d.name][:, :C] - kp, bits)
        WV = _lsq_stream((torch.cat([prev[d.name][:, C:], kr[d.name]], 1), cur[d.name][:, C:]) for d in study.train)
        for d in study.train:
            pos = study.positions(d.tokens)
            vp = _apply(WV, torch.cat([prev[d.name][:, C:], kr[d.name]], 1))
            vr = vp + quant_token_groups(cur[d.name][:, C:] - vp, bits)
            rec = to_bf16(study.rope.rotate_row(torch.cat([kr[d.name], vr], 1), pos))
            prev[d.name] = study.rope.derotate_row(rec, pos)
        out[layer] = (WK, WV)
        del cur, kr
        if log:
            log(f"  AQUA-KV-style {bits}-bit: layer {layer} fitted on its own reconstructions")
    return out


def aqua_style(study, fits, doc, rows, bits):
    C = study.C
    pos = study.positions(rows)
    out = torch.empty(study.L, rows, study.W, dtype=torch.float32)
    prev = None
    for layer in range(study.L):
        x = words_to_float(doc.words(layer, study.device, rows))
        xd = study.rope.derotate_row(x, pos)
        if layer == 0:
            rd = quant_token_groups(xd, 4)
        else:
            WK, WV = fits[layer]
            kp = _apply(WK, prev[:, :C])
            kr = kp + quant_token_groups(xd[:, :C] - kp, bits)
            vp = _apply(WV, torch.cat([prev[:, C:], kr], 1))
            vr = vp + quant_token_groups(xd[:, C:] - vp, bits)
            rd = torch.cat([kr, vr], 1)
        rec = to_bf16(study.rope.rotate_row(rd, pos))
        prev = study.rope.derotate_row(rec, pos)
        out[layer] = rec.to(torch.float32).cpu()
    rate = ((4 + 0.5) + (study.L - 1) * (bits + 0.5)) / study.L
    return out, rate


def kivi_style(study, doc, rows, bits):
    C = study.C
    out = torch.empty(study.L, rows, study.W, dtype=torch.float32)
    for layer in range(study.L):
        x = words_to_float(doc.words(layer, study.device, rows))
        rec = torch.cat([quant_channel_blocks(x[:, :C], bits), quant_token_groups(x[:, C:], bits)], 1)
        out[layer] = to_bf16(rec).to(torch.float32).cpu()
    return out, bits + (0.25 + 0.5) / 2


def fp8_cast(study, doc, rows):
    out = torch.empty(study.L, rows, study.W, dtype=torch.float32)
    for layer in range(study.L):
        out[layer] = to_bf16(fp8_e4m3(words_to_float(doc.words(layer, study.device, rows)))).to(torch.float32).cpu()
    return out, 8.0


# ---------------------------------------------------------------------------------------- AQUA-KV (authors' code)
AQUA_SOURCE = "goodevening13/aquakv@e744183f2f3a4a90381f20a6975456871fd1ae29"
AQUA_GRID = {2: 16, 3: 64, 4: 256}                # HIGGS d = 2 grids: points per 2 values
AQUA_PREFIX = 4                                   # evaluate_perplexity.py --prefix_size default (attention sinks)
AQUA_GROUP = 1024                                 # --hadamard_groupsize
AQUA_DAMP = 1e-3                                  # train_predictors.py --percdamp
_GRIDS = {}


def higgs_grid(n, device):
    """The EDEN grid with n points in 2 dimensions (vendored from the AQUA-KV repository, kv_compare/data)."""
    if not _GRIDS:
        with np.load(Path(__file__).with_name("data") / "higgs_grids.npz") as z:
            _GRIDS.update({k: z[k] for k in z.files})
    return torch.as_tensor(_GRIDS[f"EDEN2_{n}"], device=device)


def hadamard(x):
    """Unnormalised Walsh-Hadamard transform (Sylvester order) over the last dimension, a power of two: what
    fast_hadamard_transform.hadamard_transform(x, scale=1) computes."""
    n = x.shape[-1]
    y = x.reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        y = torch.stack([y[:, :, 0] + y[:, :, 1], y[:, :, 0] - y[:, :, 1]], dim=2)
        h *= 2
    return y.reshape(x.shape)


def higgs_qdq(x, n, group=AQUA_GROUP, rows=512):
    """HiggsQuantizer.quantize then .dequantize (aquakv/quantizers.py) of x [B, C]: float32 Hadamard transform of
    each group of `group` channels, divided by the group's L2 norm (the scale, one per token and group), each pair
    of values replaced by the nearest of the n grid points (argmax of 2 x.g - |g|^2), then grid point x scale cast
    to float16 and transformed back (scale 1 / group). Returns float16 values [B, C] (as float64)."""
    B, C = x.shape
    x32 = x.to(torch.float32)
    pad = (-C) % group
    if pad:
        x32 = torch.nn.functional.pad(x32, (0, pad))
    mult = x32.shape[1] // group
    xg = x32.reshape(B, mult, group)
    scales = torch.linalg.norm(xg, dim=-1)
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    z = (hadamard(xg) / safe[:, :, None]).reshape(B, mult, group // 2, 2)
    grid = higgs_grid(n, x.device)
    gn = grid.pow(2).sum(1)
    idx = torch.empty(B, mult, group // 2, dtype=torch.int64, device=x.device)
    for i in range(0, B, rows):
        idx[i:i + rows] = torch.argmax(2 * z[i:i + rows] @ grid.T - gn, dim=-1)
    deq = (grid[idx].flatten(2) * scales[:, :, None]).half()
    out = (hadamard(deq.to(torch.float32)) * (1.0 / group)).half()
    return out.reshape(B, -1)[:, :C].to(torch.float64)


def _aqua_lsq(pairs, damp=AQUA_DAMP):
    """linear_utils.fit_linear_regression (fit_intercept, float32): X^T X with a ones column, damping
    damp x mean |diagonal| on the diagonal, pseudo-inverse; weights and bias then stored in BF16 (nn.Linear in the
    activations' dtype)."""
    CXX = CXY = None
    for X, Y in pairs:
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1).to(torch.float32)
        Y = Y.to(torch.float32)
        CXX = X1.T @ X1 if CXX is None else CXX + X1.T @ X1
        CXY = X1.T @ Y if CXY is None else CXY + X1.T @ Y
    ix = torch.arange(CXX.shape[0], device=CXX.device)
    CXX[ix, ix] += damp * CXX.diagonal().abs().mean()
    Wt = (torch.linalg.pinv(CXX) @ CXY).T
    return to_bf16(Wt[:, :-1].to(torch.float64)), to_bf16(Wt[:, -1].to(torch.float64))


def _aqua_apply(lin, X):
    """BF16 nn.Linear: BF16 inputs, float32 accumulation, BF16 output."""
    W, b = lin
    return to_bf16(to_bf16(X) @ W.T + b)


def _aqua_recon(pred, target, n):
    """get_dequant_values / the cache's write path: residual (BF16 subtraction) quantized and dequantized, cast to
    BF16, added to the prediction in BF16."""
    return to_bf16(pred + to_bf16(higgs_qdq(to_bf16(target - pred), n)))


def aqua_official_fit(study, layers, bits, log=None):
    """train_predictors.py, layer by layer on the calibration documents (all of them; no holdout): layer 0
    reconstructed by HIGGS 4-bit without a predictor; for layer l >= 1 the key predictor is fitted from the
    reconstructed keys of layer l-1 to the exact keys of layer l, the keys reconstructed with it, then the value
    predictor fitted from [reconstructed keys of layer l, reconstructed values of layer l-1] to the exact values,
    and the values reconstructed. Keys without RoPE throughout."""
    C = study.C
    n, n0 = AQUA_GRID[bits], AQUA_GRID[4]
    out = {}
    prev = {}
    for d in study.train:
        x = to_bf16(study.feats(d, 0, cache=False).to(torch.float64))
        prev[d.name] = (_aqua_recon(torch.zeros_like(x[:, :C]), x[:, :C], n0),
                        _aqua_recon(torch.zeros_like(x[:, C:]), x[:, C:], n0))
    for layer in range(1, layers):
        cur = {d.name: to_bf16(study.feats(d, layer, cache=False).to(torch.float64)) for d in study.train}
        lk = _aqua_lsq((prev[d.name][0], cur[d.name][:, :C]) for d in study.train)
        kr = {d.name: _aqua_recon(_aqua_apply(lk, prev[d.name][0]), cur[d.name][:, :C], n) for d in study.train}
        lv = _aqua_lsq((torch.cat([kr[d.name], prev[d.name][1]], 1), cur[d.name][:, C:]) for d in study.train)
        for d in study.train:
            vr = _aqua_recon(_aqua_apply(lv, torch.cat([kr[d.name], prev[d.name][1]], 1)), cur[d.name][:, C:], n)
            prev[d.name] = (kr[d.name], vr)
        out[layer] = (lk, lv)
        del cur, kr
        if log:
            log(f"  AQUA-KV {bits}-bit: layer {layer} predictors fitted on their own reconstructions")
    return out


def aqua_official(study, fits, doc, rows, bits):
    """AQUA-KV's cache (aquakv/cache_utils.py: TreatPrefixSeparately + SingleChunkQuantizedCacheWithPredictors)
    applied to the first `rows` tokens as one prefill: the first AQUA_PREFIX tokens in BF16; the rest coded layer
    by layer with the predictors and HIGGS (one prefill of more than the 128-token buffer is quantized whole).
    Bits: b per value (4 in layer 0) plus one 16-bit scale per token, K/V and 1024 channels; the prefix 16."""
    C = study.C
    n, n0 = AQUA_GRID[bits], AQUA_GRID[4]
    pos = study.positions(rows)
    p = min(AQUA_PREFIX, rows)
    out = torch.empty(study.L, rows, study.W, dtype=torch.float32)
    prev = None
    for layer in range(study.L):
        x = words_to_float(doc.words(layer, study.device, rows))
        xd = to_bf16(study.rope.derotate_row(x, pos))[p:]
        if layer == 0:
            kr = _aqua_recon(torch.zeros_like(xd[:, :C]), xd[:, :C], n0)
            vr = _aqua_recon(torch.zeros_like(xd[:, C:]), xd[:, C:], n0)
        else:
            lk, lv = fits[layer]
            kr = _aqua_recon(_aqua_apply(lk, prev[0]), xd[:, :C], n)
            vr = _aqua_recon(_aqua_apply(lv, torch.cat([kr, prev[1]], 1)), xd[:, C:], n)
        prev = (kr, vr)
        rec = x.clone()
        rec[p:] = to_bf16(study.rope.rotate_row(torch.cat([kr, vr], 1), pos[p:]))
        out[layer] = rec.to(torch.float32).cpu()
    groups = -(-C // AQUA_GROUP)
    per_token = lambda b: 2 * C * b + 2 * groups * 16
    total = p * study.W * 16 * study.L + (rows - p) * (per_token(4) + (study.L - 1) * per_token(bits))
    return out, total / (study.L * rows * study.W)


# ---------------------------------------------------------------------------------------- KIVI (authors' code)
KIVI_SOURCE = "jy-yuan/KIVI@876b4d2"
KIVI_GROUP = 32                                   # paper: group size 32
KIVI_RESIDUAL = 128                               # paper: residual length 128 (full-precision recent tokens)


def _kivi_groups(g, bits):
    """triton_quantize_and_pack_along_last_dim on groups g [..., group] (BF16 data): min and max, scale =
    (max - min) / (2^b - 1) in BF16, q = round(clamp((x - min) / scale)) with the subtraction and division in
    BF16; dequantized q * scale + min. A group with max = min reconstructs to min."""
    levels = 2 ** bits - 1
    mn = g.amin(-1, keepdim=True)
    mx = g.amax(-1, keepdim=True)
    scale = to_bf16(to_bf16(mx - mn) / levels)
    ok = scale > 0
    u = to_bf16(to_bf16(g - mn) / torch.where(ok, scale, torch.ones_like(scale)))
    q = torch.where(ok, torch.round(u.clamp(0, levels)), torch.zeros_like(u))
    return q * scale + mn


def kivi_official(study, doc, rows, bits, group=KIVI_GROUP, residual=KIVI_RESIDUAL):
    """KIVI's prefill cache (models/llama_kivi.py) for the first `rows` tokens: post-RoPE keys quantized per
    channel over groups of `group` tokens, all but the last rows mod `residual` tokens; values per token over
    groups of `group` channels (within a head), all but the last `residual` tokens; the rest stays BF16.
    Bits: b per quantized value plus 2 x 16 bits per group; 16 per BF16 value."""
    C = study.C
    nk = rows - rows % residual if rows >= residual else 0
    nv = rows - residual if rows > residual else 0
    if nk % group or (C % group):
        raise ValueError("KIVI needs the quantized token count and channels to be multiples of the group size")
    out = torch.empty(study.L, rows, study.W, dtype=torch.float32)
    for layer in range(study.L):
        x = words_to_float(doc.words(layer, study.device, rows))
        rec = x.clone()
        if nk:
            k = x[:nk, :C].T.reshape(C, nk // group, group)
            rec[:nk, :C] = _kivi_groups(k, bits).reshape(C, nk).T
        if nv:
            v = x[:nv, C:].reshape(nv, C // group, group)
            rec[:nv, C:] = _kivi_groups(v, bits).reshape(nv, C)
        out[layer] = to_bf16(rec).to(torch.float32).cpu()
    qbits = lambda n: n * C * bits + (n * C // group) * 32
    total = qbits(nk) + (rows - nk) * C * 16 + qbits(nv) + (rows - nv) * C * 16
    return out, total / (rows * study.W)
