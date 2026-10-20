"""Deployment checks for the linear predictor (B3) on SmolLM2 KV.

  python3 -I deploy_checks.py ACT_DIR LAYERS(comma list or 'all')

1. Parallelism: predict current-layer entries in blocks of `bsz` channels (K blocks keep RoPE pairs together);
   each block sees the previous layer + all earlier blocks of the token. bsz=2 is the exact_ideas.py B3,
   bsz=192 means one block for K then one for V, bsz=384 is the previous layer only (B1-like, fully parallel).
   Sequential steps per layer = number of blocks.
2. Determinism: the same predictor in fixed point (inputs and cos/sin on integer grids, int16-range coefficients,
   int64 accumulation, scale on a 1/8-octave grid). Integer arithmetic gives the same prediction on any hardware;
   this measures what the quantization costs in size.
Cost = exact -log2 P of each BF16 word under a logistic(mu, s) with a 2^-16 escape (same as exact_ideas.py).
"""
import os, sys
import numpy as np
D = sys.argv[1]
load = lambda n: np.load(os.path.join(D, f"{n}.npz"))
FIT = [load(f"train{i}") for i in range(6)]; TAB = load("tables"); TESTS = {"text": load("test_text"), "code": load("test_code")}
L, N, C = TAB["K_bits"].shape; Hk, hd, base = int(TAB["Hk"]), int(TAB["hd"]), float(TAB["base"])
LAYERS = list(range(1, L)) if sys.argv[2] == "all" else [int(x) for x in sys.argv[2].split(",")]

def from_bits(b): return (np.asarray(b).astype(np.uint32) << 16).view(np.float32)
def ordered(w):
    w = np.asarray(w).astype(np.int64); return np.where(w & 0x8000, (~w) & 0xFFFF, w ^ 0x8000)
def words(d, l): return np.concatenate([d["K_bits"][l], d["V_bits"][l]], 1)
ang = np.arange(N, dtype=np.float64)[:, None] * (base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd))[None, :]
COS_F, SIN_F = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
COS_Q, SIN_Q = np.round(COS_F * 2 ** 15) / 2 ** 15, np.round(SIN_F * 2 ** 15) / 2 ** 15   # int16 tables
def rope(k, inverse=False, q=False):
    n = k.shape[0]; k = k.reshape(n, Hk, hd); c, s = (COS_Q, SIN_Q) if q else (COS_F, SIN_F)
    c, s = c[:n], (-s[:n] if inverse else s[:n])
    o = np.empty_like(k); o[..., 0::2] = k[..., 0::2] * c - k[..., 1::2] * s; o[..., 1::2] = k[..., 0::2] * s + k[..., 1::2] * c
    return o.reshape(n, C)
FX = 2 ** 16                                                 # fixed-point grid for inputs / predictions
def fx(a): return np.round(a * FX) / FX
def feats(d, l, q=False):
    w = words(d, l).astype(np.uint16); x = from_bits(w).astype(np.float64)
    xp = from_bits(words(d, l - 1)).astype(np.float64)
    cur = np.concatenate([rope(x[:, :C], True, q), x[:, C:]], 1)
    prev = np.concatenate([rope(xp[:, :C], True, q), xp[:, C:]], 1)
    if q: cur, prev = fx(cur), fx(prev)
    return w, x, cur, prev

def bf16_interval(w):
    o = ordered(w)
    def val(sym):
        ww = np.where(sym & 0x8000, sym ^ 0x8000, (~sym) & 0xFFFF).astype(np.uint16); return from_bits(ww).astype(np.float64)
    x = val(o); lo, hi = (x + val(np.clip(o - 1, 0, 65535))) / 2, (x + val(np.clip(o + 1, 0, 65535))) / 2
    return np.where(o == 0x8000, 0.0, lo), np.where(o == 0x7FFF, 0.0, hi)
def exact_bits(w, mu, s):
    lo, hi = bf16_interval(w)
    with np.errstate(over="ignore"):
        P = np.where(lo >= mu, 1 / (1 + np.exp((lo - mu) / s)) - 1 / (1 + np.exp((hi - mu) / s)),
                     1 / (1 + np.exp(-(hi - mu) / s)) - 1 / (1 + np.exp(-(lo - mu) / s)))
    P = (1 - 2.0 ** -16) * np.maximum(P, 0) + 2.0 ** -16 / 65536
    return float(-np.log2(P).sum())

def fit(l):
    F = [feats(d, l) for d in FIT]
    cur = np.concatenate([f[2] for f in F]); prev = np.concatenate([f[3] for f in F])
    mu = np.concatenate([prev.mean(0), cur.mean(0)])
    Z = np.concatenate([prev, cur], 1) - mu; S = Z.T @ Z / len(Z)
    S += 1e-3 * np.eye(len(S)) * np.trace(S) / len(S)
    Lc = np.linalg.cholesky(S); Linv = np.linalg.solve(Lc, np.eye(len(S)))
    return Lc, Linv, mu, prev.shape[1]

def coeffs(Lc, Linv, mu, NP, bsz):
    """Direct form: pred_cur = bias + A @ [prev, cur]; row j only uses entries before its block start."""
    NB = Lc.shape[0]; j = np.arange(NP, NB); rel = j - NP
    start = NP + (rel // bsz) * bsz if bsz < 2 * C else np.full_like(j, NP)
    mask = np.arange(NB)[None, :] < start[:, None]
    A = (Lc[j] * mask) @ Linv                               # conditional-mean coefficients on Z (zero at/after start)
    bias = mu[j] - A @ mu
    return A, bias

def predict(A, bias, cur, prev, q):
    X = np.concatenate([prev, cur], 1)
    if not q: return X @ A.T + bias
    Ai = np.round(A * 2 ** 16).astype(np.int64)             # coefficients on a 2^-16 grid
    assert np.abs(Ai).max() < 2 ** 31
    Xi = np.round(X * FX).astype(np.int64)
    acc = Xi @ Ai.T                                         # int64 accumulation (exact, hardware-independent)
    return fx(acc / (2 ** 16 * FX) + bias)

def post(m, q): return np.concatenate([rope(m[:, :C], False, q), m[:, C:]], 1)
def qscale(s): return 2.0 ** (np.round(8 * np.log2(s)) / 8)

CFG = [("B3 bsz=2 (pairs, as measured)", 2, False), ("B3 bsz=8", 8, False), ("B3 bsz=32", 32, False),
       ("B3 bsz=192 (K block, then V block)", 192, False), ("B1-like bsz=384 (previous layer only)", 384, False),
       ("B3 bsz=2, fixed point", 2, True), ("B3 bsz=32, fixed point", 32, True)]
res = {tn: {c[0]: [] for c in CFG} for tn in TESTS}
for l in LAYERS:
    Lc, Linv, mu, NP = fit(l)
    for name, bsz, q in CFG:
        A, bias = coeffs(Lc, Linv, mu, NP, bsz)
        wt, xt, ct, pt = feats(TAB, l, q)
        mt = post(predict(A, bias, ct, pt, q), q)
        s = np.abs(xt - mt).mean(0) / (2 * np.log(2)) + 1e-12
        if q: s = qscale(s)
        for tn, d in TESTS.items():
            w, x, c, p = feats(d, l, q)
            m = post(predict(A, bias, c, p, q), q)
            if q: m = fx(m)
            res[tn][name].append(exact_bits(w, m, np.broadcast_to(s, m.shape)) / w.size)
    print(f"layer {l} done", file=sys.stderr, flush=True)
print(f"layers {LAYERS[0]}..{LAYERS[-1]} ({len(LAYERS)} layers), bits per value (BF16 = 16); steps = sequential blocks per layer")
for tn in TESTS:
    print(f"\n=== {tn} ===")
    for name, bsz, q in CFG:
        v = np.mean(res[tn][name]); steps = 1 if bsz >= 2 * C else 2 * C // bsz
        print(f"{name:<42}{v:7.3f}{100 * (1 - v / 16):7.1f}%   steps/layer {steps}")
