"""Baselines re-implemented from their published descriptions, run on the same captured KV.

Lossless (exact BF16):
  SplitZip-style   one top-16 exponent codebook calibrated on TRAIN (4-bit code), sign+mantissa byte raw,
                   an exponent outside the codebook is an escape (16-bit position + 8-bit exponent).
  ExANS-style      block = 256 tokens x all layers x K/V; each (layer, K/V) gets its most common exponent as the
                   centre; centred exponents coded with one block-local entropy model (ideal entropy + 24 bits per
                   symbol for the model + 128-byte centre table + 8 rANS states of 32 bits); sign+mantissa raw.
Lossy:
  AQUA-KV-style    closed loop: keys of layer l from the reconstructed keys of layer l-1 (RoPE removed), values from
                   the reconstructed values of layer l-1 and keys of layer l; linear + bias (least squares on
                   TRAIN); residual quantized by a b-bit uniform min-max quantizer per token and 64-channel group
                   (BF16 scale and offset: +0.5 bit per value); layer 0 quantized directly at 4 bits. AQUA-KV uses
                   a vector quantizer (HIGGS), which is more accurate than this scalar stand-in at equal bits.
  KIVI-style       direct b-bit quantization: keys per channel over 128-token blocks, values per token over
                   64-channel groups (BF16 scale and offset: +0.25 / +0.5 bit per value).
  FP8 cast         every value cast to FP8 e4m3, no scale: 8 bits per value.
"""
from __future__ import annotations

import numpy as np
import torch

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


def aqua_fit(study, layers, log=None):
    """AQUA-KV-style predictors per layer, least squares on TRAIN (exact KV, RoPE removed), one document at a
    time so memory stays small."""
    C = study.C
    out = {}
    for layer in range(1, layers):
        WK = _lsq_stream((study.feats(d, layer - 1)[:, :C], study.feats(d, layer)[:, :C]) for d in study.train)
        WV = _lsq_stream((torch.cat([study.feats(d, layer - 1)[:, C:], study.feats(d, layer)[:, :C]], 1),
                          study.feats(d, layer)[:, C:]) for d in study.train)
        out[layer] = (WK, WV)
        study.forget(layer - 1)
        if log:
            log(f"  AQUA-KV-style: layer {layer} fitted")
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
