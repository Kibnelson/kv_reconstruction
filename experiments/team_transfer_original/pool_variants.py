"""GPU-pool format (#1) variants, all 30 layers, sizes on whole test documents.

  python3 -I pool_variants.py ACT_DIR

All variants store sign + 7 mantissa bits raw (8 bits) and code only the exponent. Widths/references are
fitted per layer on train0..train5. Pool block = 16 tokens x 1 layer x all K/V channels (6,144 values).
  A0      current GPU format: per 8 channels of a token, 8-bit min exponent + 4-bit width + fixed-width deltas
  P1      per-channel reference exponent + static per-channel width; the all-ones code marks an escape, and
          escapes are listed in order after the block (8-bit raw exponent each; their position is the rank of
          the marker, so no index is stored); 16-bit escape count per block
  P2      P1 + a 4-bit exponent offset per (token, head) added to every channel's reference (one add at decode;
          the offset is the cheapest piece of a per-token predictor)
Fixed-size blocks: a pool that needs fixed-size blocks reserves room for `cap` escapes per block (per layer, set
from the FIT data's p99.9 or max per-block escape count); a test block with more escapes is stored raw (16 bits/value).
Widths are chosen per channel to minimize width + P(escape) x penalty, for several penalties (8 = the true escape cost).
"""
import os, sys
import numpy as np
D = sys.argv[1]
load = lambda n: np.load(os.path.join(D, f"{n}.npz"))
FIT = [load(f"train{i}") for i in range(6)]; TESTS = {"text": load("test_text"), "code": load("test_code")}
L, N, C = FIT[0]["K_bits"].shape; hd = int(FIT[0]["hd"]); NV = 2 * C; HEADS = NV // hd; BLK = 16

def words(d, l): return np.concatenate([d["K_bits"][l], d["V_bits"][l]], 1)
def expo(w): return (w.astype(np.int64) >> 7) & 0xFF
def zz(d): return np.where(d >= 0, 2 * d, -2 * d - 1)
def bitlen(a):
    a = np.asarray(a, np.int64); return np.where(a > 0, np.floor(np.log2(np.maximum(a, 1))).astype(np.int64) + 1, 0)

def fit_widths(z, lam):
    cand = np.arange(1, 9)
    cost = np.stack([w + (z > (1 << w) - 2).mean(0) * lam for w in cand])
    return cand[cost.argmin(0)]

def head_offset(e, ref):
    d = (e - ref[None, :]).reshape(len(e), HEADS, hd)
    return np.clip(np.round(np.median(d, 2)), -8, 7).astype(np.int64)          # [n, HEADS]

def per_block_escapes(e, ref, w, use_off):
    off = np.repeat(head_offset(e, ref), hd, 1) if use_off else 0
    esc = zz(e - off - ref[None, :]) > (1 << w[None, :]) - 2
    return esc.reshape(len(e) // BLK, BLK * NV).sum(1)

LAMS = (8, 12, 16, 24)
VARS = [("P1", False, lam) for lam in LAMS] + [("P2", True, lam) for lam in LAMS]
tot = {tn: {v: np.zeros(4) for v in VARS} for tn in TESTS}     # variable, fixed(p99.9 cap), fixed(max cap), raw blocks
a0 = {tn: 0.0 for tn in TESTS}
for l in range(L):
    efs = [expo(words(d, l)) for d in FIT]; ef = np.concatenate(efs)
    ref1 = np.median(ef, 0).astype(np.int64); ref2 = ref1.copy()
    for _ in range(2):
        ref2 = np.median(ef - np.repeat(head_offset(ef, ref2), hd, 1), 0).astype(np.int64)
    for v in VARS:
        name, use_off, lam = v; ref = ref2 if use_off else ref1
        zf = zz(ef - (np.repeat(head_offset(ef, ref), hd, 1) if use_off else 0) - ref[None, :])
        w = fit_widths(zf, lam)
        epb_fit = np.concatenate([per_block_escapes(x, ref, w, use_off) for x in efs])
        cap999, capmax = int(np.ceil(np.percentile(epb_fit, 99.9))), int(epb_fit.max())   # capacity from FIT data only
        pay = BLK * (8 * NV + w.sum() + (4 * HEADS if use_off else 0)) + 16
        for tn, d in TESTS.items():
            epb = per_block_escapes(expo(words(d, l)), ref, w, use_off)
            raw = BLK * NV * 16
            f999 = np.where(epb > cap999, raw, pay + 8 * cap999); fmax = np.where(epb > capmax, raw, pay + 8 * capmax)
            tot[tn][v] += [(pay + 8 * epb).sum(), f999.sum(), fmax.sum(), (epb > cap999).sum()]
    for tn, d in TESTS.items():
        e = expo(words(d, l)); g = e.reshape(-1, 8)
        a0[tn] += (8 * e.size + (8 + 4 + 8 * bitlen((g - g.min(1, keepdims=True)).max(1))).sum())
    print(f"layer {l} done", file=sys.stderr, flush=True)

nvals = L * N * NV; nblocks = L * N // BLK
pct = lambda b: 100 * (1 - b / nvals / 16)
for tn in TESTS:
    print(f"\n=== {tn}: bits per value, all {L} layers (BF16 = 16); escape room per layer set from FIT data ===")
    print(f"A0 current GPU format (variable size): {a0[tn] / nvals:.3f} ({pct(a0[tn]):.1f}%)")
    print(f"{'variant':<36}{'variable':>16}{'fixed, p99.9 room':>22}{'fixed, max room':>20}{'raw blocks':>12}")
    for v in VARS:
        name, use_off, lam = v; t = tot[tn][v]
        lab = f"{name} escape penalty {lam}"
        print(f"{lab:<36}{t[0] / nvals:8.3f} ({pct(t[0]):4.1f}%){t[1] / nvals:12.3f} ({pct(t[1]):4.1f}%){t[2] / nvals:10.3f} ({pct(t[2]):4.1f}%){100 * t[3] / nblocks:10.2f}%")
