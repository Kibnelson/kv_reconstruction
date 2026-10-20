"""Tests of the repo's ideas in exact (lossless) form on SmolLM2 KV.

  python3 -I exact_ideas.py ACT_DIR

Fit data: train0..train5 (6 x 8,192 tokens); scales/tables: tables.npz; tests: test_text, test_code. All 30 layers.

A. Decode-cheap fixed-width exponent formats (sign + 7 mantissa bits raw in all of them):
   A0 group-min    per 8 channels of a token: 8-bit min exponent, 4-bit width, fixed-width deltas (current GPU format)
   A1 ref/token    per-channel static reference exponent (fit data median); per 8 channels: 4-bit width + zigzag deltas
   A2 ref/channel  same reference; groups of 16 tokens x 1 channel (tile layout)
   A3 ref/static   per-channel static width chosen on fit data; out-of-range values escape (8-bit exponent + 8-bit index)
   E  entropy      per-channel static exponent tables (the entropy-coded reference)
B. Linear prediction + per-value logistic distribution (exact arithmetic-coding cost of each BF16 word):
   B0 none         per-channel mean
   B1 prev layer   ridge regression from the previous layer's KV (keys handled before RoPE, prediction rotated)
   B2 in-token     causal regression on earlier channels of the same token (Cholesky order: K then V)
   B3 both         previous layer + earlier channels of the same token
   Scale per channel from the mean absolute residual on tables.npz (logistic: s = E|r| / (2 ln 2)).
C. Layer 0 from token IDs: V0 and pre-RoPE K0 depend only on the token; check that equal tokens give equal V0 words.
"""
import os, sys
import numpy as np
D = sys.argv[1]
load = lambda n: np.load(os.path.join(D, f"{n}.npz"))
FIT = [load(f"train{i}") for i in range(6)]; TAB = load("tables"); TESTS = {"text": load("test_text"), "code": load("test_code")}
L, N, C = TAB["K_bits"].shape; Hk, hd, base = int(TAB["Hk"]), int(TAB["hd"]), float(TAB["base"])

def from_bits(b): return (np.asarray(b).astype(np.uint32) << 16).view(np.float32)
def ordered(w):
    w = np.asarray(w).astype(np.int64); return np.where(w & 0x8000, (~w) & 0xFFFF, w ^ 0x8000)
def bitlen(a):
    a = np.asarray(a, np.int64); return np.where(a > 0, np.floor(np.log2(np.maximum(a, 1))).astype(np.int64) + 1, 0)
def zz(d): return np.where(d >= 0, 2 * d, -2 * d - 1)
def words(d, l): return np.concatenate([d["K_bits"][l], d["V_bits"][l]], 1)
def expo(w): return (w.astype(np.int64) >> 7) & 0xFF

ang = np.arange(N, dtype=np.float64)[:, None] * (base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd))[None, :]
COS, SIN = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
def rope(k, inverse=False):
    n = k.shape[0]; k = k.reshape(n, Hk, hd); c, s = COS[:n], (-SIN[:n] if inverse else SIN[:n])
    o = np.empty_like(k); o[..., 0::2] = k[..., 0::2] * c - k[..., 1::2] * s; o[..., 1::2] = k[..., 0::2] * s + k[..., 1::2] * c
    return o.reshape(n, C)

# ---------------- A: fixed-width exponent formats ----------------
def fixed_width_costs(l):
    ef = np.concatenate([expo(words(d, l)) for d in FIT]); ref = np.median(ef, 0).astype(np.int64)
    # A3 static widths: per channel choose w minimizing w + P(escape)*(16) on fit data
    zf = zz(ef - ref[None, :])
    cand = np.arange(0, 9)
    cost_w = np.stack([w + (zf >= (1 << w)).mean(0) * 16 for w in cand])     # [9, 2C]
    wstat = cand[cost_w.argmin(0)]
    pt = np.bincount(ef.ravel(), minlength=256) + 0.5; pt /= pt.sum()
    cc = np.zeros((2 * C, 256)); np.add.at(cc, (np.broadcast_to(np.arange(2 * C), ef.shape).ravel(), ef.ravel()), 1)
    pc = (cc + 32 * pt[None, :]) / (cc.sum(1, keepdims=True) + 32)
    out = {}
    for name, d in TESTS.items():
        e = expo(words(d, l)); n = e.size
        g = e.reshape(-1, 8); a0 = (8 + 4 + 8 * bitlen((g - g.min(1, keepdims=True)).max(1))).sum()
        z = zz(e - ref[None, :])
        a1 = (4 + 8 * bitlen(z.reshape(-1, 8).max(1))).sum()
        zt = z.reshape(N // 16, 16, 2 * C).transpose(0, 2, 1).reshape(-1, 16)
        a2 = (4 + 16 * bitlen(zt.max(1))).sum()
        a3 = (wstat[None, :] + (z >= (1 << wstat[None, :])) * 16).sum()
        ent = -np.log2(pc[np.broadcast_to(np.arange(2 * C), e.shape).ravel(), e.ravel()]).sum()
        out[name] = {k: 8 + v / n for k, v in dict(A0=a0, A1=a1, A2=a2, A3=a3, E=ent).items()}
    return out

# ---------------- B: linear prediction + logistic coding ----------------
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

def feats(d, l):
    """Current layer values (keys de-rotated) and previous layer features (keys de-rotated), float64."""
    w = words(d, l).astype(np.uint16); x = from_bits(w).astype(np.float64)
    xk = rope(x[:, :C], inverse=True); cur = np.concatenate([xk, x[:, C:]], 1)
    if l == 0: prev = np.zeros((N, 0))
    else:
        wp = words(d, l - 1); xp = from_bits(wp).astype(np.float64)
        prev = np.concatenate([rope(xp[:, :C], inverse=True), xp[:, C:]], 1)
    return w, x, cur, prev

def predictors(l):
    F = [feats(d, l) for d in FIT]
    cur = np.concatenate([f[2] for f in F]); prev = np.concatenate([f[3] for f in F])
    mu_c, mu_p = cur.mean(0), (prev.mean(0) if prev.shape[1] else np.zeros(0))
    lam = 1e-3
    preds = {"B0": None}
    if prev.shape[1]:
        P1 = np.concatenate([prev - mu_p, np.ones((len(prev), 1))], 1)
        W1 = np.linalg.solve(P1.T @ P1 + lam * np.eye(P1.shape[1]) * len(P1), P1.T @ (cur - mu_c))
        preds["B1"] = W1
    # causal (lower-triangular) regression via Cholesky of the joint covariance [prev, cur]
    for name, usep in (("B2", False), ("B3", True)):
        if usep and not prev.shape[1]: continue
        Z = np.concatenate([prev - mu_p, cur - mu_c], 1) if usep else cur - mu_c
        S = Z.T @ Z / len(Z) + lam * np.eye(Z.shape[1]) * np.trace(Z.T @ Z / len(Z)) / Z.shape[1]
        Lc = np.linalg.cholesky(S)                     # Z = Lc e, e white; x_j | earlier = L[j, :j] e[:j]
        preds[name] = (Lc, prev.shape[1] if usep else 0)
    return preds, mu_c, mu_p

def apply(pred, name, cur, prev, mu_c, mu_p):
    """Predicted current values (pre-RoPE keys) given decoded earlier quantities (exact cur values)."""
    if name == "B0": return np.broadcast_to(mu_c, cur.shape).copy()
    if name == "B1":
        P1 = np.concatenate([prev - mu_p, np.ones((len(prev), 1))], 1); return P1 @ pred + mu_c
    Lc, npv = pred
    Z = np.concatenate([prev - mu_p, cur - mu_c], 1) if npv else cur - mu_c
    E = np.linalg.solve(Lc, Z.T).T                      # innovations e (forward substitution: causal)
    # Decodability: the two channels of a RoPE pair are only known together (de-rotation needs both), so each
    # key pair is predicted from entries BEFORE the pair; previous-layer features and V channels are single blocks.
    nb = Z.shape[1]; blk = np.arange(nb)
    kpos = np.arange(C); blk[npv:npv + C] = npv + (kpos // 2) * 2
    Zhat = Z.copy()
    for j in range(nb):
        same = np.where((blk == blk[j]) & (np.arange(nb) <= j))[0]
        Zhat[:, j] -= E[:, same] @ Lc[j, same]
    return Zhat[:, npv:] + mu_c

def linear_costs(l):
    preds, mu_c, mu_p = predictors(l)
    out = {n: {} for n in TESTS}
    wt, xt, ct, pt_ = feats(TAB, l)
    for name, pr in preds.items():
        # rotate key predictions back to post-RoPE space; scale from tables doc residuals
        def post(m): return np.concatenate([rope(m[:, :C]), m[:, C:]], 1)
        mt = post(apply(pr, name, ct, pt_, mu_c, mu_p))
        s = np.abs(xt - mt).mean(0) / (2 * np.log(2)) + 1e-12
        for tn, d in TESTS.items():
            w, x, c, p = feats(d, l)
            m = post(apply(pr, name, c, p, mu_c, mu_p))
            out[tn][name] = exact_bits(w, m, np.broadcast_to(s, m.shape)) / w.size
    return out

# ---------------- C: layer 0 from token IDs ----------------
def layer0_check(d):
    ids = d["ids"] if "ids" in d.files else None
    return ids

A, B = {}, {}
for l in range(L):
    A[l] = fixed_width_costs(l); B[l] = linear_costs(l)
    print(f"layer {l} done", file=sys.stderr, flush=True)
for tn in TESTS:
    print(f"\n=== {tn}: bits per value, all {L} layers (BF16 = 16) ===")
    for k, lab in (("A0", "A0 group-min fixed width (current GPU format)"), ("A1", "A1 per-channel reference, groups of 8 channels"),
                   ("A2", "A2 per-channel reference, groups of 16 tokens"), ("A3", "A3 per-channel static width + escapes"),
                   ("E", "E  per-channel entropy-coded exponent")):
        v = np.mean([A[l][tn][k] for l in range(L)]); print(f"{lab:<58}{v:7.2f}{100 * (1 - v / 16):8.1f}%")
    for k, lab in (("B0", "B0 per-channel mean + logistic"), ("B1", "B1 linear from previous layer"),
                   ("B2", "B2 linear from earlier channels of the token"), ("B3", "B3 previous layer + earlier channels")):
        vals = [B[l][tn].get(k, B[l][tn].get("B2" if k == "B3" else "B0")) for l in range(L)]
        v = np.mean(vals); print(f"{lab:<58}{v:7.2f}{100 * (1 - v / 16):8.1f}%")
    print("   per layer B3:", " ".join(f"{B[l][tn].get('B3', B[l][tn].get('B2')):.2f}" for l in range(L)))
