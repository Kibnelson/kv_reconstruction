"""Combined approach, end to end: transfer format = linear predictor + entropy coding (#4),
GPU-pool format = per-channel reference exponent + static widths (#1).

  python3 -I combined.py ACT_DIR DOC T0 NTOK [BSZ]      e.g.  combined.py act test_text 2048 256 32

Prefill side   all 30 layers of tokens T0..T0+NTOK-1 -> one transfer stream.
Decoder side   decodes the stream layer by layer; each decoded layer is packed straight into pool blocks
               (16 tokens each) and kept only as the input for predicting the next layer.
Check          unpack every pool block and compare every word with the original BF16 words.

Determinism. Everything that decides a probability uses integers or single IEEE-754 operations (+, -, x),
which give the same bits on any IEEE machine; there is no exp/log at coding time:
  - inputs: each BF16 value as an integer in units of 2^-16 (exact scaling, then round-half-even)
  - RoPE de-rotation / rotation: integer cos/sin tables (Q15), integer multiply, arithmetic shift
  - prediction: integer coefficients (units 2^-16), int64 accumulation
  - probability of each BF16 word: logistic CDF from an integer table (16,385 points, linear integer
    interpolation) at the word's interval edges; 2^-16 of the mass spread uniformly (so every word is codable);
    probabilities are integer ratios over 2^48 handed to the coder
  - coding: (sign, exponent) group of the word first (512 symbols), then its 7 mantissa bits (128 symbols);
    the product of the two probabilities equals the word's probability exactly
Codec constants that both sides must share (fitted offline, per layer): coefficients, biases, scale classes,
pool references and widths; plus the cos/sin and CDF tables.
"""
import os, sys, time
import numpy as np
import constriction

D, DOC, T0, NT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
BSZ = int(sys.argv[5]) if len(sys.argv) > 5 else 32
load = lambda n: np.load(os.path.join(D, f"{n}.npz"))
FIT = [load(f"train{i}") for i in range(6)]; TAB = load("tables"); TEST = load(DOC)
L, N, C = TAB["K_bits"].shape; Hk, hd, base = int(TAB["Hk"]), int(TAB["hd"]), float(TAB["base"])
NV = 2 * C; NBLK = NV // BSZ; PB = 16
assert C % BSZ == 0 and BSZ % 2 == 0 and NT % PB == 0
POS = np.arange(T0, T0 + NT)

def from_bits(b): return (np.asarray(b).astype(np.uint32) << 16).view(np.float32)
def words(d, l): return np.concatenate([d["K_bits"][l], d["V_bits"][l]], 1).astype(np.uint16)
def ordered(w):
    w = np.asarray(w).astype(np.int64); return np.where(w & 0x8000, (~w) & 0xFFFF, w ^ 0x8000)
def word_of(o):
    o = np.asarray(o).astype(np.int64); return np.where(o & 0x8000, o ^ 0x8000, (~o) & 0xFFFF).astype(np.uint16)

# ----------------------------------------------------------------- shared integer helpers
ang = np.arange(N, dtype=np.float64)[:, None] * (base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd))[None, :]
COSI, SINI = np.rint(np.cos(ang) * 2 ** 15).astype(np.int64), np.rint(np.sin(ang) * 2 ** 15).astype(np.int64)  # shared tables
H14 = 1 << 14

def fxv(w): return np.rint(from_bits(w).astype(np.float64) * 65536.0).astype(np.int64)   # value in units of 2^-16

def k_pairs(x, ch0, pos):
    """x [n, m] channels ch0..ch0+m-1 of K (m even, ch0 even) -> pair views and their cos/sin rows."""
    n, m = x.shape; i = ((ch0 + np.arange(0, m, 2)) % hd) // 2
    return x[:, 0::2], x[:, 1::2], COSI[pos][:, i], SINI[pos][:, i]

def derot_int(x, ch0, pos):           # post-RoPE -> pre-RoPE, integers
    a, b, c, s = k_pairs(x, ch0, pos); o = np.empty_like(x)
    o[:, 0::2] = (a * c + b * s + H14) >> 15; o[:, 1::2] = (-a * s + b * c + H14) >> 15
    return o

def rot_int(x, ch0, pos):             # pre-RoPE -> post-RoPE, integers
    a, b, c, s = k_pairs(x, ch0, pos); o = np.empty_like(x)
    o[:, 0::2] = (a * c - b * s + H14) >> 15; o[:, 1::2] = (a * s + b * c + H14) >> 15
    return o

def feat_cols(w, ch0, pos):
    """Integer predictor features of decoded words for channels ch0..: keys de-rotated, values as they are."""
    x = fxv(w)
    return derot_int(x, ch0, pos) if ch0 < C else x

# ----------------------------------------------------------------- offline fit (float), then quantize
def float_feats(d, l):
    def derot(x):
        n = len(x); k = x[:, :C].reshape(n, Hk, hd); c, s = np.cos(ang)[:n, None, :], np.sin(ang)[:n, None, :]
        o = np.empty_like(k); o[..., 0::2] = k[..., 0::2] * c + k[..., 1::2] * s; o[..., 1::2] = -k[..., 0::2] * s + k[..., 1::2] * c
        return np.concatenate([o.reshape(n, C), x[:, C:]], 1)
    cur = derot(from_bits(words(d, l)).astype(np.float64))
    prev = derot(from_bits(words(d, l - 1)).astype(np.float64)) if l > 0 else np.zeros((N, 0))
    return cur, prev

def fit_layer(l):
    F = [float_feats(d, l) for d in FIT]
    cur = np.concatenate([f[0] for f in F]); prev = np.concatenate([f[1] for f in F]); NP = prev.shape[1]
    mu = np.concatenate([prev.mean(0), cur.mean(0)])
    Z = np.concatenate([prev, cur], 1) - mu; S = Z.T @ Z / len(Z)
    S += 1e-3 * np.eye(len(S)) * np.trace(S) / len(S)
    Lc = np.linalg.cholesky(S); Linv = np.linalg.solve(Lc, np.eye(len(S)))
    j = np.arange(NP, NP + NV); start = NP + ((j - NP) // BSZ) * BSZ
    A = (Lc[j] * (np.arange(NP + NV)[None, :] < start[:, None])) @ Linv     # conditional-mean coefficients
    bias = mu[j] - A @ mu
    Ai = np.rint(A * 65536).astype(np.int64); bi = np.rint(bias * 65536).astype(np.int64)
    # scale class per channel from the tables document, using the integer predictor itself
    w = words(TAB, l); x = from_bits(w).astype(np.float64); pos = np.arange(N)
    X = np.concatenate([feat_int_all(words(TAB, l - 1), pos) if l > 0 else np.zeros((N, 0), np.int64), feat_int_all(w, pos)], 1)
    pred = np.zeros((N, NV))
    for b in range(NBLK):
        st = NP + b * BSZ; rows = slice(b * BSZ, (b + 1) * BSZ)
        pred[:, rows] = predict_int(X[:, :st], Ai[rows, :st], bi[rows], b * BSZ, pos) / 65536.0
    s = np.abs(x - pred).mean(0) / (2 * np.log(2)) + 1e-12
    kq = np.rint(8 * np.log2(s)).astype(np.int64)
    return dict(NP=NP, A=Ai, b=bi, inv_s=2.0 ** (-kq / 8.0))     # inv_s: one of a fixed set of constants (shared table)

def feat_int_all(w, pos): return np.concatenate([derot_int(fxv(w[:, :C]), 0, pos), fxv(w[:, C:])], 1)

def predict_int(Xpart, Ablk, bblk, ch0, pos):
    acc = Xpart @ Ablk.T if Xpart.shape[1] else np.zeros((len(Xpart), len(bblk)), np.int64)
    pre = ((acc + (1 << 15)) >> 16) + bblk                   # pre-RoPE prediction, units of 2^-16
    return rot_int(pre, ch0, pos) if ch0 < C else pre

# ----------------------------------------------------------------- deterministic probability model
with np.errstate(invalid="ignore"):                                  # NaN words (signalling NaNs) raise a flag on cast
    xv = from_bits(word_of(np.arange(65536))).astype(np.float64)
    BD = np.empty(65537); BD[1:65536] = (xv[:-1] + xv[1:]) / 2      # interval edge between ordered symbols o-1 and o
BD[0:129] = -np.inf; BD[65408:] = np.inf                            # inf/NaN words get no logistic mass
assert not np.isnan(BD).any() and np.all(BD[1:] >= BD[:-1])
GB = BD[0::128]                                                      # 513 group edges
TT, TR = 32.0, 256
NTAB = int(2 * TT * TR) + 1
F_TAB = np.rint(2.0 ** 32 / (1 + np.exp(-(-TT + np.arange(NTAB) / TR)))).astype(np.int64)  # shared integer table
F_TAB[0], F_TAB[-1] = 0, 2 ** 32
assert np.all(np.diff(F_TAB) >= 0)
ONE16 = 2 ** 16

def cdf_int(t):
    u = np.clip((t + TT) * TR, 0.0, NTAB - 1.0)
    i = np.minimum(np.floor(u).astype(np.int64), NTAB - 2)
    f = np.floor((u - i) * 65536.0).astype(np.int64)
    return F_TAB[i] + (((F_TAB[i + 1] - F_TAB[i]) * f) >> 16)

def group_num(mu, inv_s):              # [R, 512] integer numerators over 2^48
    Cg = cdf_int((GB[None, :] - mu[:, None]) * inv_s[:, None])
    return (ONE16 - 1) * np.diff(Cg, axis=1) + 128 * ONE16

def mant_num(mu, inv_s, g):            # [R, 128] numerators of the 128 words of group g
    edges = BD[(g * 128)[:, None] + np.arange(129)[None, :]]
    Cm = cdf_int((edges - mu[:, None]) * inv_s[:, None])
    return (ONE16 - 1) * np.diff(Cm, axis=1) + ONE16

CAT = constriction.stream.model.Categorical(perfect=False)

def code_block(coder, mu, inv_s, w=None):
    """Encode (w given) or decode one block of values. mu, inv_s, w are flat [R]. Returns words and ideal bits."""
    gn = group_num(mu, inv_s)
    if w is not None:
        o = ordered(w); g = o >> 7
        coder.encode(g.astype(np.int32), CAT, gn / 2.0 ** 48)
    else:
        g = coder.decode(CAT, gn / 2.0 ** 48).astype(np.int64)
    mn = mant_num(mu, inv_s, g); gsel = gn[np.arange(len(g)), g]
    if w is not None:
        m = o & 127; coder.encode(m.astype(np.int32), CAT, mn / gsel[:, None])
    else:
        m = coder.decode(CAT, mn / gsel[:, None]).astype(np.int64)
    num = mn[np.arange(len(m)), m]
    return word_of(g * 128 + m), float(-np.log2(num / 2.0 ** 48).sum())

# ----------------------------------------------------------------- GPU-pool format (#1)
def zz(d): return np.where(d >= 0, 2 * d, -2 * d - 1)
def unzz(z): return np.where(z & 1, -((z + 1) >> 1), z >> 1)

def fit_pool(l):
    ef = np.concatenate([(words(d, l).astype(np.int64) >> 7) & 0xFF for d in FIT])
    ref = np.median(ef, 0).astype(np.int64); z = zz(ef - ref)
    cand = np.arange(1, 9); cost = np.stack([w + (z > (1 << w) - 2).mean(0) * 8 for w in cand])
    return ref, cand[cost.argmin(0)]

def pack_block(wd, ref, wid):
    """16 tokens x NV words -> bytes: [escape count u16][sign+mantissa bytes][exponent codes, fixed width per
    channel, MSB first][raw exponents of escapes, in position order]. The all-ones code marks an escape."""
    w = wd.astype(np.int64); e = (w >> 7) & 0xFF; sm = ((w >> 8) & 0x80) | (w & 0x7F)
    z = zz(e - ref[None, :]); esc = z > (1 << wid[None, :]) - 2
    code = np.where(esc, (1 << wid[None, :]) - 1, z)
    k = np.arange(8)[None, None, :]; W3 = wid[None, :, None]
    bits = (code[..., None] >> np.maximum(W3 - 1 - k, 0)) & 1
    bitvec = bits[np.broadcast_to(k < W3, bits.shape)].astype(np.uint8)
    nesc = int(esc.sum())
    return np.concatenate([np.array([nesc & 0xFF, nesc >> 8], np.uint8), sm.astype(np.uint8).ravel(),
                           np.packbits(bitvec), e[esc].astype(np.uint8)])

def unpack_block(buf, ref, wid):
    nesc = int(buf[0]) | (int(buf[1]) << 8); p = 2
    sm = buf[p:p + PB * NV].astype(np.int64).reshape(PB, NV); p += PB * NV
    widf = np.tile(wid, PB); nbits = int(widf.sum()); nb = (nbits + 7) // 8
    bitvec = np.unpackbits(buf[p:p + nb])[:nbits].astype(np.int64); p += nb
    starts = np.cumsum(widf) - widf; k = np.arange(8)[None, :]; msk = k < widf[:, None]
    idx = np.minimum(starts[:, None] + k, nbits - 1)
    code = (bitvec[idx] * np.where(msk, 1 << np.maximum(widf[:, None] - 1 - k, 0), 0)).sum(1).reshape(PB, NV)
    esc = code == (1 << wid[None, :]) - 1
    e = ref[None, :] + unzz(code)
    e[esc] = buf[p:p + nesc].astype(np.int64); p += nesc
    assert p == len(buf) and esc.sum() == nesc
    return (((sm & 0x80) << 8) | (e << 7) | (sm & 0x7F)).astype(np.uint16)

# ----------------------------------------------------------------- the two sides
t0 = time.time()
PRED = [fit_layer(l) for l in range(L)]; POOL = [fit_pool(l) for l in range(L)]
t_fit = time.time() - t0
ORIG = np.stack([words(TEST, l)[T0:T0 + NT] for l in range(L)])     # [L, NT, NV]

def transfer(mode, stream=None):
    coder = constriction.stream.queue.RangeEncoder() if mode == "enc" else constriction.stream.queue.RangeDecoder(stream)
    ideal = 0.0; pool = []; prev = np.zeros((NT, 0), np.int64); out = np.zeros_like(ORIG)
    for l in range(L):
        P = PRED[l]; NP = P["NP"]; cur = np.zeros((NT, NV), np.int64); wl = np.zeros((NT, NV), np.uint16)
        for b in range(NBLK):
            ch0 = b * BSZ; rows = slice(ch0, ch0 + BSZ); st = NP + ch0
            X = np.concatenate([prev, cur[:, :ch0]], 1)
            assert X.shape[1] == st
            mu = predict_int(X, P["A"][rows, :st], P["b"][rows], ch0, POS) / 65536.0
            inv = np.broadcast_to(P["inv_s"][rows], mu.shape)
            wb, bits = code_block(coder, mu.ravel(), inv.ravel(), ORIG[l][:, rows].ravel() if mode == "enc" else None)
            wb = wb.reshape(NT, BSZ); ideal += bits
            wl[:, rows] = wb; cur[:, rows] = feat_cols(wb, ch0, POS)
        out[l] = wl
        if mode == "dec":                                       # straight into the pool, one 16-token block at a time
            ref, wid = POOL[l]; pool.append([pack_block(wl[i:i + PB], ref, wid) for i in range(0, NT, PB)])
        prev = cur                                              # only this layer is kept for the next prediction
    return coder, out, ideal, pool

t0 = time.time(); enc, out_e, ideal, _ = transfer("enc"); t_enc = time.time() - t0
stream = enc.get_compressed()
t0 = time.time(); _, out_d, _, pool = transfer("dec", stream); t_dec = time.time() - t0
t0 = time.time()
unpacked = np.stack([np.concatenate([unpack_block(bb, *POOL[l]) for bb in pool[l]]) for l in range(L)]); t_unp = time.time() - t0

nval = L * NT * NV
pool_bits = 8 * sum(len(bb) for pl in pool for bb in pl)
e = (ORIG.astype(np.int64) >> 7) & 0xFF; g = e.reshape(-1, 8)
a0_bits = 8 * nval + (8 + 4 + 8 * np.floor(np.log2(np.maximum((g - g.min(1, keepdims=True)).max(1), 1)) + 1) * ((g - g.min(1, keepdims=True)).max(1) > 0)).sum()
pct = lambda b: 100 * (1 - b / nval / 16)
print(f"{DOC}, tokens {T0}-{T0 + NT - 1}, all {L} layers, {nval:,} values, decoder block {BSZ} channels ({NBLK} steps/layer)")
print(f"  transfer stream (real bytes)      {32 * len(stream) / nval:7.3f} bits/value  {pct(32 * len(stream)):5.1f}% smaller   ideal {ideal / nval:.3f}")
print(f"  GPU pool, packed (real bytes)     {pool_bits / nval:7.3f} bits/value  {pct(pool_bits):5.1f}% smaller")
print(f"  current GPU format on same tokens {a0_bits / nval:7.3f} bits/value  {pct(a0_bits):5.1f}% smaller")
print(f"  decoded words == original: {np.array_equal(out_d, ORIG)}; encoder pass == original: {np.array_equal(out_e, ORIG)}; "
      f"pool unpacked == original: {np.array_equal(unpacked, ORIG)}")
print(f"  CPU seconds (numpy, indicative only): fit {t_fit:.0f}, encode {t_enc:.0f}, decode+pack {t_dec:.0f}, unpack {t_unp:.1f}")
