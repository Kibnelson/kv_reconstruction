"""Closed-loop lossy one-step codec on the GPU (Triton), layer by layer like kv_compare.layerwise.

The prediction is the per-layer codec's (int8 product of the previous layer's 8-bit features, scale, bias, RoPE),
but computed from the receiver's reconstruction, so encoder and decoder see the same predictions. Each value x is
quantized to the lattice mu + k * step with step = kappa x the channel's spread (k = round((x - mu) / step)); the
reconstruction is that lattice point rounded to BF16, so |x - x_hat| <= step / 2 + BF16 rounding. A value whose k
falls outside the coded range (24 spreads), or that is not finite, travels as its exact 16-bit word (escape).

k is coded with 16-bit rANS against a logistic model with spread se = step / r, where r = kappa / multiplier is
rounded to a quarter-octave grid (the multiplier is the adaptive spread multiplier of the lossless codec, from the
previous layer's surprise sums of the reconstruction). One static table per grid class r: for r >= 1/8 one symbol
per lattice cell (cells k on |k r| < 24 spreads, the cell's logistic mass); for smaller r, 2^lm cells per symbol
(lm = ceil(log2(1 / (8 r)))) and the cell inside the symbol as lm raw bits; the escape symbol last. Streams stay open
across layers (one rANS state per lane, as in the lossless per-layer codec).

Encoder: a forward pass quantizes layer by layer (reconstruction, its features and priors), then a reverse pass codes
the symbols (rANS is last-in first-out).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import fused as fu
from . import gpu_coder as gc
from . import layerwise as lw

triton, tl = gc.triton, gc.tl
M = 1 << 16
LCLASSES = 45                    # r = 2^((c - LC0) / 4): 2^-6 .. 2^5
LC0 = 24
RANGE = 24.0
LUT_BITS = 10
LUT_N = (1 << LUT_BITS) + 1
LUT_W = LUT_N + 2                # a row: the buckets' (first symbol | candidates << 16), then (offf << 10 | K), (G | lm << 16)
# mantissa thresholds of the quarter-octave rounding: round(4 log2 r) steps at 2^(1/8), 2^(3/8), 2^(5/8), 2^(7/8)
_T = [int(round((2.0 ** (q / 8.0) - 1.0) * (1 << 23))) for q in (1, 3, 5, 7)]


def class_r(c):
    return 2.0 ** ((c - LC0) / 4.0)


def class_of(r):
    """Python mirror of _lclass: quarter-octave class of r > 0 (float32 bit arithmetic)."""
    b = int(np.float32(r).view(np.int32))
    e = ((b >> 23) & 0xFF) - 127
    mant = b & 0x7FFFFF
    q = sum(mant >= t for t in _T)
    return min(max(4 * e + q + LC0, 0), LCLASSES - 1)


def lattice_tables(device):
    """Static tables: cumf (per class, symbols 0..K: (cum << 16) | frequency) and lut [LCLASSES, LUT_W]."""
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    cumfs, luts, info = [], [], []
    posf = 0
    for c in range(LCLASSES):
        r = class_r(c)
        lm = 0 if r >= 0.125 else math.ceil(math.log2(1.0 / (8.0 * r)) - 1e-12)
        m = 1 << lm
        G = min(int(math.floor(RANGE / (m * r))) + 1, 511)
        K = 2 * G + 1
        P = []
        for gi in range(K):
            g = gi - G
            lo = ((g * m - m // 2 - 0.5) if m > 1 else (g - 0.5)) * r
            P.append(sig(lo + m * r) - sig(lo))
        f = [1 + int(math.floor(p * (M - K - 1))) for p in P] + [1]
        f[int(np.argmax(P))] += M - sum(f)
        cum = np.concatenate([[0], np.cumsum(f)]).astype(np.int64)
        assert cum[-1] == M and min(f) >= 1 and K < 1024
        sym = [int(np.searchsorted(cum, b << (16 - LUT_BITS), side="right") - 1) for b in range(1 << LUT_BITS)] + [K]
        row = [sym[b] | ((sym[b + 1] + 1 - sym[b]) << 16) for b in range(1 << LUT_BITS)] + [K | (1 << 16)]
        row += [(posf << 10) | K, G | (lm << 16)]
        luts.append(row)
        cumfs.append([(int(cum[k]) << 16) | int(cum[k + 1] - cum[k]) for k in range(K + 1)])
        info.append({"r": r, "lm": lm, "G": G, "K": K})
        posf += K + 1
    as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64).astype(np.uint32).view(np.int32), device=device)
    return {"cumf": as_t(np.concatenate([np.asarray(x, dtype=np.int64) for x in cumfs])),
            "lut": as_t(np.asarray(luts, dtype=np.int64)), "info": info}


def symbol_bits(tables, k, c):
    """Ideal bits (-log2 of the table frequencies) of lattice indices k (int64 tensor) in classes c (Python mirror
    of the coder's symbol mapping; escapes cost 16 + the escape symbol)."""
    info = tables["info"]
    out = torch.zeros(k.shape, dtype=torch.float64, device=k.device)
    cumf = tables["cumf"].to(torch.int64) & 0xFFFFFFFF
    off = 0
    for cc in range(LCLASSES):
        G, K, lm = info[cc]["G"], info[cc]["K"], info[cc]["lm"]
        sel = c == cc
        if bool(sel.any()):
            kk = k[sel]
            g = torch.div(kk + ((1 << lm) >> 1), 1 << lm, rounding_mode="floor") if lm else kk
            esc = g.abs() > G
            s = torch.where(esc, torch.full_like(g, K), g + G)
            f = (cumf[off + s] & 0xFFFF).to(torch.float64)
            out[sel] = 16 - torch.log2(f) + torch.where(esc, 16.0, float(lm))
        off += K + 1
    return out


if triton is not None:
    _T1, _T3, _T5, _T7 = (tl.constexpr(t) for t in _T)

    @triton.jit
    def _lclass(r):
        """Quarter-octave class of r (float32 > 0): round(4 log2 r) + LC0, clamped, from the bits."""
        b = r.to(tl.int32, bitcast=True)
        e = ((b >> 23) & 0xFF) - 127
        mant = b & 0x7FFFFF
        q = (mant >= _T1).to(tl.int32) + (mant >= _T3).to(tl.int32) + (mant >= _T5).to(tl.int32) + \
            (mant >= _T7).to(tl.int32)
        return tl.minimum(tl.maximum(4 * e + q + 24, 0), 44)

    @triton.jit
    def _lattice(v, mu, istep, w, G, lm):
        """Lattice index k, symbol index (escape: 2G + 1) and the cell inside the symbol, for exact value v."""
        kf = tl.floor((v - mu) * istep + 0.5)
        bad = (kf != kf) | (tl.abs(kf) > 16777216.0) | (((w >> 7) & 0xFF) == 0xFF)
        k = tl.where(bad, 0.0, kf).to(tl.int32)
        h = (1 << lm) >> 1
        g = (k + h) >> lm                                      # arithmetic shift: floor division
        j = (k + h) & ((1 << lm) - 1)
        esc = bad | (g > G) | (g < -G)
        sym = tl.where(esc, 2 * G + 1, g + G)
        return k, sym, j, esc

    @triton.jit
    def _token_step(step0, istep0, mult, KAPPA, TOKSTEP: tl.constexpr):
        """The step, its reciprocal and the table class for one token: kappa x spread (class from kappa / the
        multiplier) or, with TOKSTEP, kappa x spread x multiplier (then the class is kappa's for every token)."""
        if TOKSTEP:
            step = step0 * mult
            istep = tl.math.div_rn(tl.full(step0.shape, 1.0, tl.float32), step)
            c = _lclass(KAPPA + mult * 0.0)
        else:
            step = step0
            istep = istep0
            c = _lclass(tl.math.div_rn(KAPPA + mult * 0.0, mult))
        return step, istep, c

    @triton.jit
    def _recon(mu, k, step):
        return gc._bf16_word(mu + k.to(tl.float32) * step)

    @triton.jit
    def _prelude(ssum_in_ptr, smult_ptr, pid, t0, T, G, PRIOR_SCALE, FIRST: tl.constexpr, TB: tl.constexpr):
        rb = tl.arange(0, TB)
        smult = smult_ptr + pid * TB
        if FIRST:
            tl.store(smult + rb, fu._mult(tl.full([TB], 1.0, tl.float32)))
        else:
            sacc = tl.zeros([TB], dtype=tl.int32)
            for gg in range(0, G):
                sacc += tl.load(ssum_in_ptr + gg * T + t0 + rb)
            tl.store(smult + rb, fu._mult(sacc.to(tl.float32) * PRIOR_SCALE))
        tl.debug_barrier()
        return smult

    @triton.jit
    def _lyq_kernel(words_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr, inv_ptr, isx_ptr, cos_ptr, sin_ptr,
                    ssum_in_ptr, ssum_out_ptr, smult_ptr, lut_ptr, KAPPA, T, W, G, HG, PRIOR_SCALE,
                    FIRST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr, LUT_W: tl.constexpr,
                    PREFIX: tl.constexpr, TOKSTEP: tl.constexpr):
        """Encoder's forward pass for one layer of one token block x head: quantize, reconstruct, surprise sums and
        the next layer's features from the reconstruction."""
        pid = tl.program_id(0)
        g = pid % G
        t0 = (pid // G) * TB
        HD: tl.constexpr = 2 * H2
        LUT_N: tl.constexpr = LUT_W - 2
        cc = tl.arange(0, HD)
        key = g < HG
        low = cc < H2
        own = g * HD + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        smult = _prelude(ssum_in_ptr, smult_ptr, pid, t0, T, G, PRIOR_SCALE, FIRST, TB)
        sao = tl.load(sa_ptr + own)
        bo = tl.load(b_ptr + own)
        sap = tl.load(sa_ptr + part)
        bp = tl.load(b_ptr + part)
        spl = tl.load(sp_ptr + own)
        invl = tl.load(inv_ptr + own)
        isxl = tl.load(isx_ptr + own)
        step0 = spl * KAPPA
        istep0 = tl.math.div_rn(tl.full([HD], 1.0, tl.float32), step0)
        for i in range(0, TB):
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = lw._lane_mu(acc_ptr + t * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            mult = tl.load(smult + i)
            step, istep, c = _token_step(step0, istep0, mult, KAPPA, TOKSTEP)
            gp = tl.load(lut_ptr + c * LUT_W + LUT_N + 1)
            Gs = gp & 0xFFFF
            lm = gp >> 16
            w = tl.load(words_ptr + t * W + own).to(tl.int32) & 0xFFFF
            k, sym, j, esc = _lattice(gc._word_val(w), mu, istep, w, Gs, lm)
            esc = esc | (t < PREFIX)
            wq = tl.where(esc, w, _recon(mu, k, step))
            tl.store(out_ptr + t * W + own, wq.to(tl.int16))
            v = gc._word_val(wq)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            tl.debug_barrier()
            vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
            tl.store(xout_ptr + t * W + own, lw._feature(v, vp, cs, sn, low, key, isxl))

    @triton.jit
    def _lyenc_kernel(words_ptr, prior_ptr, state_ptr, pos_ptr, out_ptr, start_ptr, acc_ptr, sa_ptr, b_ptr, sp_ptr,
                      cos_ptr, sin_ptr, smult_ptr, cumf_ptr, lut_ptr, KAPPA, T, W, G, HG,
                      FLUSH: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr, LUT_W: tl.constexpr,
                      PREFIX: tl.constexpr, TOKSTEP: tl.constexpr):
        """Encode one layer of one token block x head, tokens in reverse (layers are launched last first)."""
        pid = tl.program_id(0)
        g = pid % G
        t0 = (pid // G) * TB
        HD: tl.constexpr = 2 * H2
        LUT_N: tl.constexpr = LUT_W - 2
        cc = tl.arange(0, HD)
        key = g < HG
        low = cc < H2
        own = g * HD + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        seg = pid
        valid = cc >= 0
        wp = tl.load(pos_ptr + seg)
        x = tl.load(state_ptr + seg * HD + cc).to(tl.uint32, bitcast=True)
        rb = tl.arange(0, TB)
        smult = smult_ptr + pid * TB
        tl.store(smult + rb, fu._mult(tl.load(prior_ptr + t0 + rb)))
        tl.debug_barrier()
        sao = tl.load(sa_ptr + own)
        bo = tl.load(b_ptr + own)
        sap = tl.load(sa_ptr + part)
        bp = tl.load(b_ptr + part)
        spl = tl.load(sp_ptr + own)
        step0 = spl * KAPPA
        istep0 = tl.math.div_rn(tl.full([HD], 1.0, tl.float32), step0)
        for ii in range(0, TB):
            i = TB - 1 - ii
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = lw._lane_mu(acc_ptr + t * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            mult = tl.load(smult + i)
            step, istep, c = _token_step(step0, istep0, mult, KAPPA, TOKSTEP)
            row = lut_ptr + c * LUT_W
            kop = tl.load(row + LUT_N)
            gp = tl.load(row + LUT_N + 1)
            offf = kop >> 10
            Gs = gp & 0xFFFF
            lm = gp >> 16
            w = tl.load(words_ptr + t * W + own).to(tl.int32) & 0xFFFF
            k, sym, j, esc = _lattice(gc._word_val(w), mu, istep, w, Gs, lm)
            forced = t < PREFIX                          # the first tokens: their exact word, no symbol
            esc = esc | forced
            p = tl.load(cumf_ptr + offf + sym)
            cum1 = (p >> 16) & 0xFFFF
            f1 = p & 0xFFFF
            sh = 16 - lm
            cum2 = tl.where(esc, w, j << sh)
            f2 = tl.where(esc, 1, 1 << sh)
            two = valid & (esc | (lm > 0))
            x, wp = gc._put(x, cum2, f2, two, out_ptr, wp, 0)
            x, wp = gc._put(x, cum1, f1, valid & (~forced), out_ptr, wp, 0)
        if FLUSH:
            base = tl.multiple_of(((wp - 2 * HD) // 8) * 8, 8)
            tl.store(out_ptr.to(tl.pointer_type(tl.int32)) + base // 2 + cc, x.to(tl.int32, bitcast=True))
            tl.store(start_ptr + seg, wp)
        else:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, wp)

    @triton.jit
    def _lydec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                      inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                      KAPPA, t_base, T, W, G, HG, PRIOR_SCALE,
                      FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                      LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, PREFIX: tl.constexpr, TOKSTEP: tl.constexpr):
        """Decode one layer of one token block x head: symbol (coarse table + binary search), the cell inside it,
        the reconstruction, surprise sums and the next layer's features."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
        t0 = t_base + blk * TB
        HD: tl.constexpr = 2 * H2
        LUT_N: tl.constexpr = LUT_W - 2
        cc = tl.arange(0, HD)
        key = g < HG
        low = cc < H2
        own = g * HD + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        seg = (t0 // TB) * G + g
        valid = cc >= 0
        if FIRST:
            rp = tl.load(start_ptr + seg)
            base = tl.multiple_of(((rp - 2 * HD) // 8) * 8, 8)
            x = tl.load(buf_ptr.to(tl.pointer_type(tl.int32)) + base // 2 + cc).to(tl.uint32, bitcast=True)
        else:
            rp = tl.load(pos_ptr + seg)
            x = tl.load(state_ptr + seg * HD + cc).to(tl.uint32, bitcast=True)
        smult = _prelude(ssum_in_ptr, smult_ptr, pid, t0, T, G, PRIOR_SCALE, FIRST, TB)
        sao = tl.load(sa_ptr + own)
        bo = tl.load(b_ptr + own)
        sap = tl.load(sa_ptr + part)
        bp = tl.load(b_ptr + part)
        spl = tl.load(sp_ptr + own)
        invl = tl.load(inv_ptr + own)
        isxl = tl.load(isx_ptr + own)
        step0 = spl * KAPPA
        for i in range(0, TB):
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = lw._lane_mu(acc_ptr + (t - t_base) * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            mult = tl.load(smult + i)
            step, _is, c = _token_step(step0, step0, mult, KAPPA, TOKSTEP)
            row = lut_ptr + c * LUT_W
            kop = tl.load(row + LUT_N)
            gp = tl.load(row + LUT_N + 1)
            K = kop & 1023
            offf = kop >> 10
            Gs = gp & 0xFFFF
            lm = gp >> 16
            slot = (x & 0xFFFF).to(tl.int32)
            lp = tl.load(row + (slot >> LUT_SHIFT))
            klo = lp & 0xFFFF
            khi = klo + (lp >> 16)
            while tl.max(khi - klo, 0) > 1:
                mid = (klo + khi) // 2
                go = ((tl.load(cumf_ptr + offf + mid) >> 16) & 0xFFFF) <= slot
                klo = tl.where(go, mid, klo)
                khi = tl.where(go, khi, mid)
            forced = t < PREFIX
            sym = klo
            p = tl.load(cumf_ptr + offf + sym)
            x, rp = gc._take(x, (p >> 16) & 0xFFFF, p & 0xFFFF, valid & (~forced), buf_ptr, rp, 0)
            esc = (sym == K) | forced
            slot2 = (x & 0xFFFF).to(tl.int32)
            sh = 16 - lm
            j = slot2 >> sh
            cum2 = tl.where(esc, slot2, j << sh)
            f2 = tl.where(esc, 1, 1 << sh)
            two = valid & (esc | (lm > 0))
            x, rp = gc._take(x, cum2, f2, two, buf_ptr, rp, 0)
            gsym = sym - Gs
            k = tl.where(lm > 0, gsym * (1 << lm) - ((1 << lm) >> 1) + j, gsym)
            w = tl.where(esc, slot2, _recon(mu, k, step))
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            v = gc._word_val(w)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            tl.debug_barrier()
            vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
            tl.store(xout_ptr + t * W + own, lw._feature(v, vp, cs, sn, low, key, isxl))
        if not LAST:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, rp)


class LossyLayerwise(lw.LayerwiseOneStep):
    """Closed-loop lossy coding of a document's KV (all layers) with the per-layer one-step predictor."""

    def __init__(self, consts, C, hd, kappa, tokens_per_program=64, chunks=2, rank=None, requant="kernel",
                 rank_tiles=(64, 128, 64, 4), product="cublas", fuse=False, prefix=0, token_step=False):
        """prefix: the first tokens travel exact (the attention sinks); token_step: step = kappa x spread x the
        token's adaptive multiplier (a coarser step where the model expects a larger error)."""
        super().__init__(consts, C, hd, tokens_per_program=tokens_per_program, mode="uniform", search=True,
                         chunks=chunks, fuse=fuse, product=product, rank=rank, requant=requant, rank_tiles=rank_tiles)
        self.kappa = float(kappa)
        self.prefix, self.token_step = int(prefix), bool(token_step)
        self.lt = lattice_tables(self.device)

    def quantize(self, words):
        """Forward pass: reconstruction words [L, T, W] int16, the features of the reconstruction [L, T, W] int8
        and the priors [L, T] the coder uses."""
        L, T, W = words.shape
        self.layout(T)
        dev, k = self.device, self.k
        rec = torch.empty_like(words)
        xall = torch.zeros(L, T, W, dtype=torch.int8, device=dev)
        prior = torch.ones(L, T, dtype=torch.float32, device=dev)
        ssum = torch.zeros(2, self.G, T, dtype=torch.int32, device=dev)
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        smult = torch.empty((T // self.TB) * self.G * self.TB, dtype=torch.float32, device=dev)
        for layer in range(L):
            if layer == 0:
                acc.zero_()
            else:
                self._product(xall[layer - 1], layer, acc)
                self._launch(lw._lprior_kernel, (triton.cdiv(T, 256),), ssum[(layer + 1) % 2], prior[layer], T,
                             self.G, self.prior_scale, BLOCK=256, warps=4)
            self._launch(_lyq_kernel, ((T // self.TB) * self.G,), words[layer], rec[layer], acc, xall[layer],
                         k["sa"][layer], k["b"][layer], k["sp"][layer], k["inv"][layer], k["isx"][layer], k["cos"],
                         k["sin"], ssum[(layer + 1) % 2], ssum[layer % 2], smult, self.lt["lut"], self.kappa,
                         T, W, self.G, self.HG, self.prior_scale, FIRST=layer == 0, TB=self.TB, H2=self.hd // 2,
                         LUT_W=LUT_W, PREFIX=self.prefix, TOKSTEP=self.token_step)
        return rec, xall, prior

    def encode(self, words, rec=None, xall=None, prior=None, out=None, layer_words=None):
        """Returns (stream, starts, reconstruction words)."""
        L, T, W = words.shape
        S, cap = self.layout(T)
        if rec is None:
            rec, xall, prior = self.quantize(words)
        dev, k = self.device, self.k
        if out is None:
            out = torch.empty(S * cap, dtype=torch.int16, device=dev)
        starts = torch.empty(S, dtype=torch.int32, device=dev)
        state = torch.full((S * self.hd,), 65536, dtype=torch.int32, device=dev)
        pos = (torch.arange(S, dtype=torch.int32, device=dev) + 1) * cap
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        smult = torch.empty(S * self.TB, dtype=torch.float32, device=dev)
        for layer in reversed(range(L)):
            if layer == 0:
                acc.zero_()
            else:
                self._product(xall[layer - 1], layer, acc)
            before = pos.sum(dtype=torch.int64) if layer_words is not None and layer > 0 else None
            self._launch(_lyenc_kernel, (S,), words[layer], prior[layer], state, pos, out, starts, acc,
                         k["sa"][layer], k["b"][layer], k["sp"][layer], k["cos"], k["sin"], smult, self.lt["cumf"],
                         self.lt["lut"], self.kappa, T, W, self.G, self.HG, FLUSH=layer == 0, TB=self.TB,
                         H2=self.hd // 2, LUT_W=LUT_W, PREFIX=self.prefix, TOKSTEP=self.token_step)
            if before is not None:
                layer_words[layer] = before - pos.sum(dtype=torch.int64)
        if layer_words is not None:
            layer_words[0] = (self.bits(starts, T) - 32 * S) // 16 - layer_words[1:].sum()
        return out, starts, rec

    def decode(self, buf, starts, T, out=None, bufs=None):
        L, W = self.L, self.W
        words = torch.empty(L, T, W, dtype=torch.int16, device=self.device) if out is None else out
        bufs = self.buffers(T) if bufs is None else bufs
        k = self.k
        Tc = T // self.chunks
        P = (Tc // self.TB) * self.G
        cur = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        streams = bufs["streams"]
        for s in streams or []:
            s.wait_stream(cur)
        for layer in range(L):
            for c in range(self.chunks):
                rows = slice(c * Tc, (c + 1) * Tc)
                acc = bufs["acc"][c]
                ctx = torch.cuda.stream(streams[c]) if streams else lw._null()
                with ctx:
                    if layer == 0:
                        acc.zero_()
                    else:
                        self._product(bufs["xbuf"][(layer + 1) % 2][rows], layer, acc, bufs["scratch"][c])
                    self._launch(_lydec_kernel, (P,), buf, starts, bufs["state"], bufs["pos"], words[layer], acc,
                                 bufs["xbuf"][layer % 2], k["sa"][layer], k["b"][layer], k["sp"][layer],
                                 k["inv"][layer], k["isx"][layer], k["cos"], k["sin"], bufs["ssum"][(layer + 1) % 2],
                                 bufs["ssum"][layer % 2], bufs["smult"][c * P * self.TB:], self.lt["cumf"],
                                 self.lt["lut"], self.kappa, c * Tc, T, W, self.G, self.HG, self.prior_scale,
                                 FIRST=layer == 0, LAST=layer == L - 1, TB=self.TB, H2=self.hd // 2,
                                 LUT_SHIFT=16 - LUT_BITS, LUT_W=LUT_W, PREFIX=self.prefix,
                                 TOKSTEP=self.token_step)
        for s in streams or []:
            cur.wait_stream(s)
        return words
