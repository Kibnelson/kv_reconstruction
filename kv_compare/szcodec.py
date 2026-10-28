"""Our one-step prediction in SplitZip's kind of format, as a GPU codec: fixed-length codes, no entropy coder.

SplitZip is fast because nothing is variable-length: every value has a raw sign+mantissa byte and a 4-bit code for
its exponent, so a decoder places every value independently. This codec keeps that property but lets our one-step
prediction (the per-layer codec's int8 product, previous layer to this layer, with its adaptive spread) choose each
value's code book (kv_compare.szformat, variant "sign + magnitude", measured the bits; this is the real codec):

  symbol    the sign's agreement with the prediction mu, and the magnitude residual r = |word| - |bf16(mu)| (BF16
            magnitude bits are ordered), or r = |word| - |bf16(spread)| when the signs differ
  context   c = floor(2 log2(spread / ulp of bf16(mu))) clamped to [-8, 28] (37 contexts), from bit patterns only
  code      per layer and context: b raw low bits of r and a k-bit code for floor(r / 2^b) (2^k - 1 code words for
            the most common values on the calibration documents; the last code word is an escape followed by the
            whole 16-bit word in place of the b bits)

The decoder knows every value's context before reading it (the prediction comes from the previous layer), so a
value's code length is known from its context and its code. A segment is one block of BM tokens x one head: first
the k-bit codes of its values, then their payloads (b bits, or 16 after an escape); two prefix sums over the tile
give every value's bit offset. One 32-bit start per segment.

Decoding, per layer: (reduced rank) the first 8-bit product with its requantization (the per-layer codec's Triton
epilogue kernel), then one kernel per block x head that computes the second product for its tile with tensor cores
(the int32 products never go through memory), the prediction, contexts, reads and rebuilds the words, and writes the
next layer's 8-bit features and the surprise sums of the next layer's spread. With the dense product (or
product="acc"), the int32 products come from cuBLAS through memory instead.

Code books are fitted on calibration documents (like SplitZip's exponent code book) and are part of the codec, not of
each document's stream (37 x 64 entries per layer)."""
from __future__ import annotations

import os

import torch

from . import fused as fu
from . import gpu_coder as gc
from . import layerwise as lw

triton, tl = gc.triton, gc.tl

NC = 37                          # contexts: floor(2 log2(spread / ulp)) in [-8, 28]
CTX_LO = -8
B_MAX = 15                       # raw low bits of the residual
K_MAX = 6                        # code bits (2^k - 1 code words + escape)
ESC_BITS = 16
NSYMC = 64                       # code-word table width per context (2^K_MAX)

if triton is not None:
    _ASM = tl.constexpr(os.environ.get("TRITON_INTERPRET") != "1")      # PTX funnel shift on the GPU

    @triton.jit
    def _ctx_of(mu, s):
        """Context and the BF16 words of the prediction and the spread (integer operations on the bit patterns:
        floor(2 log2 s) = 2 (exponent - 127) + (mantissa >= sqrt 2); ulp of bf16(mu) = 2^(exponent field - 134))."""
        wm = gc._bf16_word(mu)
        ws = gc._bf16_word(s)
        em = tl.maximum((wm >> 7) & 255, 1)
        sb = s.to(tl.uint32, bitcast=True)
        es = ((sb >> 23) & 255).to(tl.int32)
        half = ((sb & 0x7FFFFF) >= 0x3504F3).to(tl.int32)
        c = 2 * (es - 127) + half - 2 * (em - 134)
        c = tl.minimum(tl.maximum(c, -8), 28) + 8
        return c, wm, ws

    @triton.jit
    def _acc_tile(acc_ptr, x_ptr, a_ptr, rl, ca, cb, W, K, BM: tl.constexpr, H2: tl.constexpr, BK: tl.constexpr,
                  MODE: tl.constexpr):
        """The int32 products of a tile's two RoPE halves: MODE 0 none (layer 0), 1 from memory (acc [rows, W]),
        2 computed here: x [rows, K] int8 times A [W, K] int8 transposed (tensor cores)."""
        if MODE == 0:
            acc_a = tl.zeros([BM, H2], dtype=tl.int32)
            acc_b = tl.zeros([BM, H2], dtype=tl.int32)
        elif MODE == 1:
            acc_a = tl.load(acc_ptr + rl[:, None] * W + ca[None, :])
            acc_b = tl.load(acc_ptr + rl[:, None] * W + cb[None, :])
        else:
            acc_a = tl.zeros([BM, H2], dtype=tl.int32)
            acc_b = tl.zeros([BM, H2], dtype=tl.int32)
            rk = tl.arange(0, BK)
            for k0 in range(0, K, BK):
                xt = tl.load(x_ptr + rl[:, None] * K + (k0 + rk)[None, :])
                at = tl.load(a_ptr + ca[:, None] * K + (k0 + rk)[None, :])
                bt = tl.load(a_ptr + cb[:, None] * K + (k0 + rk)[None, :])
                acc_a = tl.dot(xt, tl.trans(at), acc_a, out_dtype=tl.int32)
                acc_b = tl.dot(xt, tl.trans(bt), acc_b, out_dtype=tl.int32)
        return acc_a, acc_b

    @triton.jit
    def _pred_tile(acc_a, acc_b, sa_ptr, b_ptr, sp_ptr, cos_ptr, sin_ptr, ssum_in_ptr, rt, ca, cb, rh, key, T, G,
                   PRIOR_SCALE, BM: tl.constexpr, H2: tl.constexpr, ONES: tl.constexpr):
        """Prediction (scale, bias, RoPE for keys) and spread of a tile's two halves; the spread multiplier per
        token from the previous layer's surprise sums (all heads), or prior 1."""
        if ONES:
            mult = fu._mult(tl.full([BM], 1.0, tl.float32))
        else:
            sacc = tl.zeros([BM], dtype=tl.int32)
            for gg in range(0, G):
                sacc += tl.load(ssum_in_ptr + gg * T + rt)
            mult = fu._mult(sacc.to(tl.float32) * PRIOR_SCALE)
        cs = tl.load(cos_ptr + rt[:, None] * H2 + rh[None, :])
        sn = tl.load(sin_ptr + rt[:, None] * H2 + rh[None, :])
        ao = acc_a.to(tl.float32) * tl.load(sa_ptr + ca)[None, :] + tl.load(b_ptr + ca)[None, :]
        ap = acc_b.to(tl.float32) * tl.load(sa_ptr + cb)[None, :] + tl.load(b_ptr + cb)[None, :]
        mua = tl.where(key, ao * cs - ap * sn, ao)
        mub = tl.where(key, ap * cs + ao * sn, ap)
        sa = tl.load(sp_ptr + ca)[None, :] * mult[:, None]
        sb = tl.load(sp_ptr + cb)[None, :] * mult[:, None]
        return mua, mub, sa, sb, cs, sn

    @triton.jit
    def _after_tile(wa, wb, mua, mub, cs, sn, key, inv_ptr, isx_ptr, ssum_out_ptr, xout_ptr, rt, ca, cb, g, T, W,
                    FEAT: tl.constexpr):
        """The tile's surprise sums (fixed point, per token, this head) and the next layer's 8-bit features
        (keys de-rotated, the fused codec's arithmetic)."""
        va = gc._word_val(wa)
        vb = gc._word_val(wb)
        sq = tl.sum(fu._sq(va, mua, tl.load(inv_ptr + ca)[None, :]), 1) + \
            tl.sum(fu._sq(vb, mub, tl.load(inv_ptr + cb)[None, :]), 1)
        tl.store(ssum_out_ptr + g * T + rt, sq)
        if FEAT:
            da = tl.where(key, va * cs + vb * sn, va)
            db = tl.where(key, vb * cs - va * sn, vb)
            qa = tl.floor(da * tl.load(isx_ptr + ca)[None, :] + 0.5)
            qb = tl.floor(db * tl.load(isx_ptr + cb)[None, :] + 0.5)
            qa = tl.where(qa == qa, qa, 0.0)
            qb = tl.where(qb == qb, qb, 0.0)
            qa = tl.minimum(tl.maximum(qa, -127.0), 127.0)
            qb = tl.minimum(tl.maximum(qb, -127.0), 127.0)
            tl.store(xout_ptr + rt[:, None] * W + ca[None, :], qa.to(tl.int8))
            tl.store(xout_ptr + rt[:, None] * W + cb[None, :], qb.to(tl.int8))

    @triton.jit
    def _symbol(w, mu, s):
        """Encoder: context << 17 | sign agreement << 16 | (r + 32768) of each word."""
        c, wm, ws = _ctx_of(mu, s)
        agree = ((w >> 15) & 1) == ((wm >> 15) & 1)
        r = (w & 0x7FFF) - tl.where(agree, wm & 0x7FFF, ws & 0x7FFF)
        return (c << 17) | (agree.to(tl.int32) << 16) | (r + 32768)

    @triton.jit
    def _szsym_kernel(words_ptr, acc_ptr, sym_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr, inv_ptr, isx_ptr, cos_ptr,
                      sin_ptr, ssum_in_ptr, ssum_out_ptr, T, W, G, HG, PRIOR_SCALE,
                      BM: tl.constexpr, H2: tl.constexpr, MODE: tl.constexpr, ONES: tl.constexpr, FEAT: tl.constexpr):
        """Encoder's pass over one layer (exact words): every value's symbol and context, the surprise sums and the
        next layer's features, with the decoder's arithmetic."""
        pid = tl.program_id(0)
        g = pid % G
        t0 = (pid // G) * BM
        rt = t0 + tl.arange(0, BM)
        rh = tl.arange(0, H2)
        HD: tl.constexpr = 2 * H2
        ca = g * HD + rh
        cb = ca + H2
        key = g < HG
        acc_a, acc_b = _acc_tile(acc_ptr, acc_ptr, acc_ptr, rt, ca, cb, W, 0, BM, H2, 16, MODE)
        mua, mub, sa, sb, cs, sn = _pred_tile(acc_a, acc_b, sa_ptr, b_ptr, sp_ptr, cos_ptr, sin_ptr, ssum_in_ptr, rt,
                                              ca, cb, rh, key, T, G, PRIOR_SCALE, BM, H2, ONES)
        wa = tl.load(words_ptr + rt[:, None] * W + ca[None, :]).to(tl.int32) & 0xFFFF
        wb = tl.load(words_ptr + rt[:, None] * W + cb[None, :]).to(tl.int32) & 0xFFFF
        tl.store(sym_ptr + rt[:, None] * W + ca[None, :], _symbol(wa, mua, sa))
        tl.store(sym_ptr + rt[:, None] * W + cb[None, :], _symbol(wb, mub, sb))
        _after_tile(wa, wb, mua, mub, cs, sn, key, inv_ptr, isx_ptr, ssum_out_ptr, xout_ptr, rt, ca, cb, g, T, W,
                    FEAT)

    @triton.jit
    def _excl(x, BM: tl.constexpr, H2: tl.constexpr):
        """Exclusive prefix sum of a [BM, H2] tile in row-major order, and its total."""
        rs = tl.sum(x, 1)
        ro = tl.cumsum(rs, 0) - rs
        return ro[:, None] + tl.cumsum(x, 1) - x, tl.sum(rs, 0)

    @triton.jit
    def _bits(p, off, mask):
        """Bits at bit offset off (LSB first in 32-bit words) of the segment at p, masked (at most 16 bits: two
        words; on the GPU one funnel shift, in the interpreter 64-bit arithmetic)."""
        i = off >> 5
        lo = tl.load(p + i)
        hi = tl.load(p + i + 1)
        if _ASM:
            v = tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;", "=r,r,r,r", [lo, hi, off & 31],
                                          dtype=tl.int32, is_pure=True, pack=1)
            return v & mask
        else:
            v = ((hi.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) |
                 lo.to(tl.uint32, bitcast=True).to(tl.uint64)) >> (off & 31).to(tl.uint64)
            return (v & mask.to(tl.uint64)).to(tl.int32)

    @triton.jit
    def _word_of(q, p, esc, sym, bmask, wm, ws):
        """The word from its code word's symbol and its b raw bits p (bmask = 2^b - 1), or p itself after an
        escape."""
        agree = (sym >> 16) & 1
        rhi = (sym & 0xFFFF) - 32768
        mag = tl.where(agree == 1, wm & 0x7FFF, ws & 0x7FFF) + rhi * (bmask + 1) + p
        sign = tl.where(agree == 1, wm & 0x8000, (wm & 0x8000) ^ 0x8000)
        return tl.where(esc, p, sign | mag)

    @triton.jit
    def _szdec_kernel(buf_ptr, start_ptr, out_ptr, x_ptr, a_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr, inv_ptr,
                      isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, tab_ptr, csym_ptr, t_base, T, W, G, HG, K,
                      PRIOR_SCALE, BM: tl.constexpr, H2: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr,
                      ONES: tl.constexpr, FEAT: tl.constexpr, PROF: tl.constexpr):
        """Decode one layer of one block of BM tokens x one head (pointers are the layer's; x_ptr and acc_ptr start
        at the chunk's first token). PROF (timing only, wrong words): 1 no stream reads, 2 no prefix sums."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
        t0 = t_base + blk * BM
        rt = t0 + tl.arange(0, BM)
        rh = tl.arange(0, H2)
        HD: tl.constexpr = 2 * H2
        ca = g * HD + rh
        cb = ca + H2
        key = g < HG
        acc_a, acc_b = _acc_tile(acc_ptr, x_ptr, a_ptr, rt - t_base, ca, cb, W, K, BM, H2, BK, MODE)
        mua, mub, sa, sb, cs, sn = _pred_tile(acc_a, acc_b, sa_ptr, b_ptr, sp_ptr, cos_ptr, sin_ptr, ssum_in_ptr, rt,
                                              ca, cb, rh, key, T, G, PRIOR_SCALE, BM, H2, ONES)
        xa, wma, wsa = _ctx_of(mua, sa)
        xb, wmb, wsb = _ctx_of(mub, sb)
        ea = tl.load(tab_ptr + xa)                   # b | k << 4 | escape code << 7 | (2^b - 1) << 13
        eb = tl.load(tab_ptr + xb)
        p = buf_ptr + tl.load(start_ptr + (t0 // BM) * G + g)
        if PROF == 2:                                # timing: fixed offsets instead of the prefix sums
            lane = (tl.arange(0, BM)[:, None] * H2 + rh[None, :]) * 4
            oa, na, ob, nb = lane, BM * H2 * 4, lane, BM * H2 * 4
        else:
            oa, na = _excl((ea >> 4) & 7, BM, H2)
            ob, nb = _excl((eb >> 4) & 7, BM, H2)
        if PROF == 1:                                # timing: no stream reads
            qa = (ea >> 4) & 1
            qb = (eb >> 4) & 1
        else:
            qa = _bits(p, oa, (ea >> 7) & 63)
            qb = _bits(p, na + ob, (eb >> 7) & 63)
        sa_esc = qa == ((ea >> 7) & 63)
        sb_esc = qb == ((eb >> 7) & 63)
        la = tl.where(sa_esc, 16, ea & 15)
        lb = tl.where(sb_esc, 16, eb & 15)
        if PROF == 2:
            pa_off, pa_n, pb_off = oa * 2, BM * H2 * 8, ob * 2
        else:
            pa_off, pa_n = _excl(la, BM, H2)
            pb_off, _ = _excl(lb, BM, H2)
        base = na + nb
        if PROF == 1:
            pa = la
            pb = lb
        else:
            pa = _bits(p, base + pa_off, tl.where(sa_esc, 0xFFFF, (ea >> 13) & 0x7FFF))
            pb = _bits(p, base + pa_n + pb_off, tl.where(sb_esc, 0xFFFF, (eb >> 13) & 0x7FFF))
        wa = _word_of(qa, pa, sa_esc, tl.load(csym_ptr + xa * 64 + qa), (ea >> 13) & 0x7FFF, wma, wsa)
        wb = _word_of(qb, pb, sb_esc, tl.load(csym_ptr + xb * 64 + qb), (eb >> 13) & 0x7FFF, wmb, wsb)
        tl.store(out_ptr + rt[:, None] * W + ca[None, :], wa.to(tl.int16))
        tl.store(out_ptr + rt[:, None] * W + cb[None, :], wb.to(tl.int16))
        _after_tile(wa, wb, mua, mub, cs, sn, key, inv_ptr, isx_ptr, ssum_out_ptr, xout_ptr, rt, ca, cb, g, T, W,
                    FEAT)

    @triton.jit
    def _szrow_kernel(buf_ptr, start_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr, inv_ptr, isx_ptr,
                      cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, tab_ptr, csym_ptr, t_base, T, W, G, HG,
                      PRIOR_SCALE, TB: tl.constexpr, H2: tl.constexpr, MODE: tl.constexpr, ONES: tl.constexpr,
                      FEAT: tl.constexpr, PROF: tl.constexpr):
        """Row layout: one program = TB tokens x one head, token by token like the rANS per-layer kernel: the
        head's 128 lanes form one vector, a token's codes then its payloads follow the previous token's, and a
        running offset carries the position (prefix sums over 128 lanes only). Products from memory (acc_ptr at
        the chunk's first token) or none (MODE 0). PROF as in _szdec_kernel."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
        t0 = t_base + blk * TB
        HD: tl.constexpr = 2 * H2
        cc = tl.arange(0, HD)
        key = g < HG
        low = cc < H2
        own = g * HD + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        rb = tl.arange(0, TB)
        smult = smult_ptr + pid * TB
        if ONES:
            tl.store(smult + rb, fu._mult(tl.full([TB], 1.0, tl.float32)))
        else:
            sacc = tl.zeros([TB], dtype=tl.int32)
            for gg in range(0, G):
                sacc += tl.load(ssum_in_ptr + gg * T + t0 + rb)
            tl.store(smult + rb, fu._mult(sacc.to(tl.float32) * PRIOR_SCALE))
        tl.debug_barrier()
        sao = tl.load(sa_ptr + own)
        bo = tl.load(b_ptr + own)
        sap = tl.load(sa_ptr + part)
        bp = tl.load(b_ptr + part)
        spl = tl.load(sp_ptr + own)
        invl = tl.load(inv_ptr + own)
        isxl = tl.load(isx_ptr + own)
        p = buf_ptr + tl.load(start_ptr + (t0 // TB) * G + g)
        off = 0
        for i in range(0, TB):
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            if MODE == 0:
                ao = tl.zeros([HD], dtype=tl.float32) * sao + bo
                ap = tl.zeros([HD], dtype=tl.float32) * sap + bp
            else:
                ao = tl.load(acc_ptr + (t - t_base) * W + own).to(tl.float32) * sao + bo
                ap = tl.load(acc_ptr + (t - t_base) * W + part).to(tl.float32) * sap + bp
            mu = tl.where(key, tl.where(low, ao * cs - ap * sn, ao * cs + ap * sn), ao)
            c, wm, ws = _ctx_of(mu, spl * tl.load(smult + i))
            e = tl.load(tab_ptr + c)                 # b | k << 4 | escape code << 7 | (2^b - 1) << 13
            k = (e >> 4) & 7
            km = (e >> 7) & 63
            if PROF == 2:
                ko, kt = cc * 4, 512
            else:
                ko, kt = tl.cumsum(k, 0) - k, tl.sum(k, 0)
            if PROF == 1:
                q = k & 1
            else:
                q = _bits(p, off + ko, km)
            esc = q == km
            ln = tl.where(esc, 16, e & 15)
            if PROF == 2:
                lo_, lt = cc * 8, 1024
            else:
                lo_, lt = tl.cumsum(ln, 0) - ln, tl.sum(ln, 0)
            if PROF == 1:
                pv = ln
            else:
                pv = _bits(p, off + kt + lo_, tl.where(esc, 0xFFFF, (e >> 13) & 0x7FFF))
            w = _word_of(q, pv, esc, tl.load(csym_ptr + c * 64 + q), (e >> 13) & 0x7FFF, wm, ws)
            off += kt + lt
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            v = gc._word_val(w)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            if FEAT:
                tl.debug_barrier()                   # the partner lane's word, for the de-rotated features
                vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
                d = tl.where(key, tl.where(low, v * cs + vp * sn, v * cs - vp * sn), v)
                qf = tl.floor(d * isxl + 0.5)
                qf = tl.where(qf == qf, qf, 0.0)
                qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)
                tl.store(xout_ptr + t * W + own, qf.to(tl.int8))


def ctx_ref(mu, s):
    """PyTorch mirror of _ctx_of (tests)."""
    wm = gc._bf16_word_t(mu)
    ws = gc._bf16_word_t(s)
    em = torch.clamp((wm >> 7) & 255, min=1)
    sb = s.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    es = (sb >> 23) & 255
    half = ((sb & 0x7FFFFF) >= 0x3504F3).to(torch.int64)
    c = 2 * (es - 127) + half - 2 * (em - 134)
    return torch.clamp(c, -8, 28) + 8, wm, ws


def choose_books(hist):
    """Per context (hist [NC, 2, 65536] counts of (sign agreement, r + 32768)): the raw bits b, code bits k and
    the 2^k - 1 code words (symbols agree * 65536 + floor(r / 2^b) + 32768) of least total size, with escapes
    costing k + 16 bits. Returns (b [NC], k [NC], code words [NC, 64] (-1 unused), bits on the counts)."""
    nc = hist.shape[0]
    h = hist.to(torch.int64)
    n = h.sum((1, 2)).to(torch.float64)
    best = torch.full((nc,), float("inf"), dtype=torch.float64, device=h.device)
    bsel = torch.zeros(nc, dtype=torch.int64, device=h.device)
    ksel = torch.ones(nc, dtype=torch.int64, device=h.device)
    words = torch.full((nc, NSYMC), -1, dtype=torch.int64, device=h.device)
    for b in range(B_MAX + 1):
        span = 65536 >> b
        hb = h.reshape(nc, 2, span, 1 << b).sum(-1).reshape(nc, 2 * span)
        top_v, top_i = torch.topk(hb, min((1 << K_MAX) - 1, 2 * span), dim=1)
        cov = torch.cumsum(top_v, 1).to(torch.float64)
        sym = (top_i // span) * 65536 + (top_i % span) - (32768 >> b) + 32768
        for k in range(1, K_MAX + 1):
            m = (1 << k) - 1
            if m > 2 * span:                         # fewer symbols than code words
                break
            inside = cov[:, m - 1]
            cost = n * k + inside * b + (n - inside) * ESC_BITS
            better = cost < best
            if bool(better.any()):
                best = torch.where(better, cost, best)
                bsel[better] = b
                ksel[better] = k
                rowsym = torch.full((nc, NSYMC), -1, dtype=torch.int64, device=h.device)
                rowsym[:, :m] = sym[:, :m]
                words[better] = rowsym[better]
    return bsel, ksel, words, float(best.sum())


class SplitZipFormatCodec:
    """Our one-step prediction with fixed-length codes (see the module docstring). consts: fused.prepare; rank: a
    prepare_rank result (the reduced-rank product) or None (dense); product: "fused" (reduced rank: the second
    product inside the decoding kernel) or "acc" (int32 products through memory from cuBLAS); tokens_per_program
    (BM) tokens x one head per program; layout: "tile" (a program decodes its BM x 128 tile at once, prefix sums
    over the tile) or "row" (token by token, prefix sums over the head's 128 lanes; products through memory);
    chunks: token chunks decoded on separate CUDA streams (one chunk's product overlaps another's coding)."""

    def __init__(self, consts, C, hd, rank=None, product="fused", tokens_per_program=32, warps=4, bk=64,
                 rank_tiles=(64, 128, 64, 4), fuse=False, layout="tile", chunks=1):
        if triton is None:
            raise RuntimeError("Triton is required")
        if product not in ("fused", "acc"):
            raise ValueError("product is 'fused' or 'acc'")
        if layout not in ("tile", "row"):
            raise ValueError("layout is 'tile' or 'row'")
        self.layout, self.chunks = layout, int(chunks)
        if layout == "row":
            product = "acc"
        self.lw = lw.LayerwiseOneStep(consts, C, hd, tokens_per_program=tokens_per_program, rank=rank,
                                      requant="epilogue" if rank is not None else "kernel", rank_tiles=rank_tiles,
                                      fuse=fuse)
        self.k, self.rank = self.lw.k, rank
        self.L, self.W, self.C, self.hd = self.lw.L, self.lw.W, C, hd
        self.G, self.HG = self.W // hd, C // hd
        self.BM, self.warps, self.bk = int(tokens_per_program), int(warps), int(bk)
        self.product = product if rank is not None else "acc"
        self.device = self.lw.device
        self.opts = self.lw.opts
        self.prior_scale = self.lw.prior_scale
        self.tab = self.csym = self.emap = None

    def _launch(self, kernel, grid, *args, **kw):
        try:
            return kernel[grid](*args, **kw, num_warps=self.warps, **self.opts)
        except TypeError:                            # interpreter: no launch options
            return kernel[grid](*args, **kw)

    def segments(self, T):
        if T % (self.BM * self.chunks):
            raise ValueError("tokens must be a multiple of tokens_per_program x chunks")
        if T > self.k["cos"].shape[0]:
            raise ValueError("more tokens than the RoPE tables cover")
        return (T // self.BM) * self.G

    # -------------------------------------------------------------------- encoder side
    def symbols(self, words, ones=None):
        """Encoder's forward pass (exact words [L, T, W] int16): yields (layer, symbols [T, W] int32). ones: per
        layer, start the spread from prior 1 (layer 0, and in a mix every layer after a SplitZip layer)."""
        L, T, W = words.shape
        self.segments(T)
        ones = [l == 0 for l in range(L)] if ones is None else ones
        dev, k = self.device, self.k
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        x = torch.zeros(2, T, W, dtype=torch.int8, device=dev)
        ssum = torch.zeros(2, self.G, T, dtype=torch.int32, device=dev)
        sym = torch.empty(T, W, dtype=torch.int32, device=dev)
        for l in range(L):
            if l:
                self.lw._product(x[(l + 1) % 2], l, acc)
            self._launch(_szsym_kernel, ((T // self.BM) * self.G,), words[l], acc, sym, x[l % 2], k["sa"][l],
                         k["b"][l], k["sp"][l], k["inv"][l], k["isx"][l], k["cos"], k["sin"], ssum[(l + 1) % 2],
                         ssum[l % 2], T, W, self.G, self.HG, self.prior_scale, BM=self.BM, H2=self.hd // 2,
                         MODE=0 if l == 0 else 1, ONES=bool(ones[l]), FEAT=l < L - 1)
            yield l, sym

    def calibrate(self, docs, log=None):
        """Code books from calibration documents (a list of [L, T, W] int16 word tensors), layer by layer."""
        gens = [self.symbols(w) for w in docs]
        tab = torch.zeros(self.L, NC, dtype=torch.int32, device=self.device)
        csym = torch.zeros(self.L, NC, NSYMC, dtype=torch.int32, device=self.device)
        self.books = []
        cal_bits, cal_n = 0.0, 0
        for l in range(self.L):
            hist = torch.zeros(NC * 2 * 65536, dtype=torch.int64, device=self.device)
            for gen in gens:
                _, s = next(gen)
                hist += torch.bincount((s >> 16).flatten().to(torch.int64) * 65536 + (s & 0xFFFF).flatten(),
                                       minlength=NC * 2 * 65536)
                cal_n += s.numel()
            b, kk, cw, bits = choose_books(hist.view(NC, 2, 65536))
            cal_bits += bits
            esc = (1 << kk) - 1
            tab[l] = (b | (kk << 4) | (esc << 7) | (((1 << b) - 1) << 13)).to(torch.int32)
            csym[l] = torch.where(cw >= 0, cw, torch.zeros_like(cw)).to(torch.int32)
            self.books.append((b, kk, cw))
            if log:
                log(f"  SplitZip-format codec: layer {l} code books ({bits / (hist.sum().item() or 1):.3f} bits/value "
                    "on the calibration documents)")
        for gen in gens:
            gen.close()
        self.tab, self.csym = tab, csym
        self.calibration_bits = cal_bits / max(cal_n, 1)
        return self

    def share_books(self, other):
        """Use another codec's code books (same predictor; a different tile size or warp count)."""
        self.tab, self.csym, self.books = other.tab, other.csym, other.books
        self.calibration_bits = other.calibration_bits
        return self

    def _emap(self, l):
        """Encoder's map (context, symbol) -> code word, -1 = escape."""
        b, kk, cw = self.books[l]
        m = torch.full((NC * 2 * 65536,), -1, dtype=torch.int8, device=self.device)
        c = torch.arange(NC, device=self.device)[:, None].expand(NC, NSYMC)
        q = torch.arange(NSYMC, device=self.device)[None, :].expand(NC, NSYMC)
        sel = cw >= 0
        m[(c[sel] * 131072 + cw[sel])] = q[sel].to(torch.int8)
        return m

    def _pack(self, l, s, words_l):
        """One layer's symbols -> (its 32-bit words, segment starts relative to the layer's first word)."""
        T, W = words_l.shape
        S = self.segments(T)
        BM, H2, G = self.BM, self.hd // 2, self.G
        b_t, k_t, _ = self.books[l]
        emap = self._emap(l)
        s = s.to(torch.int64)
        ctx, agree, u = s >> 17, (s >> 16) & 1, s & 0xFFFF
        b, kk = b_t[ctx], k_t[ctx]
        sym = agree * 65536 + (u >> b) - (32768 >> b) + 32768
        code = emap[ctx * 131072 + sym].to(torch.int64)
        del emap, sym, agree
        esc = code < 0
        cval = torch.where(esc, (1 << kk) - 1, code)
        pval = torch.where(esc, words_l.to(torch.int64) & 0xFFFF, u & ((1 << b) - 1))
        plen = torch.where(esc, torch.full_like(b, ESC_BITS), b)
        del code, esc, u, s, ctx

        if self.layout == "tile":                    # per segment: all codes (half a, then b), then all payloads
            def seg(t):
                return t.view(T // BM, BM, G, 2, H2).permute(0, 2, 3, 1, 4).reshape(S, 2 * BM * H2)
            kl, pl, cv, pv = seg(kk), seg(plen), seg(cval), seg(pval)
            del kk, plen, cval, pval, b
            ko = torch.cumsum(kl, 1) - kl
            ktot = kl.sum(1)
            po = torch.cumsum(pl, 1) - pl + ktot[:, None]
            nbits = ktot + pl.sum(1)
        else:                                        # per segment, token by token: its codes, then its payloads
            def seg(t):
                return t.view(T // BM, BM, G, 2 * H2).permute(0, 2, 1, 3).reshape(S, BM, 2 * H2)
            kl, pl, cv, pv = seg(kk), seg(plen), seg(cval), seg(pval)
            del kk, plen, cval, pval, b
            krow, prow = kl.sum(2), pl.sum(2)
            rs = torch.cumsum(krow + prow, 1) - (krow + prow)
            ko = (rs[..., None] + torch.cumsum(kl, 2) - kl).reshape(S, -1)
            po = (rs[..., None] + krow[..., None] + torch.cumsum(pl, 2) - pl).reshape(S, -1)
            nbits = (krow + prow).sum(1)
            cv, pv = cv.reshape(S, -1), pv.reshape(S, -1)
        nwords = (nbits + 31) // 32
        st = torch.cumsum(nwords, 0) - nwords
        n = int(nwords.sum())
        acc = torch.zeros(n + 2, dtype=torch.int64, device=self.device)
        for off, val in ((ko, cv), (po, pv)):
            o = (st[:, None] * 32 + off).flatten()
            x = val.flatten() << (o & 31)
            i = o >> 5
            acc.index_add_(0, i, x & 0xFFFFFFFF)
            acc.index_add_(0, i + 1, x >> 32)
            del o, x, i
        acc = acc[:n]
        return (acc - ((acc >> 31) & 1) * (1 << 32)).to(torch.int32), st

    def encode(self, words, layer_bits=None, ours=None, other=None):
        """Words [L, T, W] int16 -> (stream int32, starts [L, segments] int32). layer_bits: a list to receive each
        layer's coded bits (its segments, padding to 32 bits and starts). ours / other (a mix): per layer True (this
        codec) or False (the calibrated SplitZip codec `other`); then returns (stream, starts, SplitZip encodings)."""
        if self.tab is None:
            raise RuntimeError("calibrate first")
        L, T, W = words.shape
        S = self.segments(T)
        ours = [True] * L if ours is None else [bool(v) for v in ours]
        ones = [l == 0 or not ours[l - 1] for l in range(L)]
        sz = {l: other.encode(words[l].view(torch.bfloat16)) for l in range(L) if not ours[l]}
        parts, starts = [], torch.zeros(L, S, dtype=torch.int32, device=self.device)
        total = 0
        for l, s in self.symbols(words, ones):
            if not ours[l]:
                if layer_bits is not None:
                    layer_bits.append(8 * sz[l].compressed_bytes)
                continue
            data, st = self._pack(l, s, words[l])
            starts[l] = (st + total).to(torch.int32)
            parts.append(data)
            if layer_bits is not None:
                layer_bits.append(32 * data.numel() + 32 * S)
            total += data.numel()
        parts.append(torch.zeros(2, dtype=torch.int32, device=self.device))      # the decoder reads one word ahead
        stream = torch.cat(parts)
        return (stream, starts) if other is None else (stream, starts, sz)

    def bits(self, stream, starts, sz=None, ours=None):
        """Coded size: the stream's 32-bit words (without the two words of read-ahead), one start per segment of
        our layers, and the SplitZip layers' bytes in a mix."""
        n_ours = starts.shape[0] if ours is None else sum(bool(v) for v in ours)
        return 32 * (stream.numel() - 2) + 32 * starts.shape[1] * n_ours + \
            sum(8 * e.compressed_bytes for e in (sz or {}).values())

    # -------------------------------------------------------------------- decoder
    def buffers(self, T):
        dev, r = self.device, (self.rank["rank"] if self.rank is not None else 0)
        Tc = T // self.chunks
        cuda = dev.type == "cuda"
        return {"x": torch.zeros(2, T, self.W, dtype=torch.int8, device=dev),
                "ssum": torch.zeros(2, self.G, T, dtype=torch.int32, device=dev),
                "smult": torch.empty(T * self.G, dtype=torch.float32, device=dev),
                "acc": [torch.empty(Tc, self.W, dtype=torch.int32, device=dev) if self.product == "acc" else None
                        for _ in range(self.chunks)],
                "scratch": [self.lw._scratch(Tc) if self.rank is not None else None for _ in range(self.chunks)],
                "zq": [torch.empty(Tc, r, dtype=torch.int8, device=dev) if self.product == "fused" else None
                       for _ in range(self.chunks)],
                "streams": [torch.cuda.Stream(dev) for _ in range(self.chunks)] if cuda and self.chunks > 1 else None}

    def products(self, words):
        """Every layer's int32 products from the exact words' features (timing the coding alone)."""
        L, T, W = words.shape
        out = [torch.zeros(T, W, dtype=torch.int32, device=self.device)]
        x = torch.zeros(2, T, W, dtype=torch.int8, device=self.device)
        k = self.k
        for l in range(L - 1):
            self.lw._launch(lw._lfeat_kernel, (triton.cdiv(T * W, 1024),), words[l], x[l % 2], k["isx"][l],
                            k["cos"], k["sin"], 0, T * W, W, self.C, HD=self.hd, BLOCK=1024, warps=4)
            out.append(self.lw._product(x[l % 2], l + 1, torch.empty(T, W, dtype=torch.int32, device=self.device)))
        return out

    def decode(self, stream, starts, T, out=None, bufs=None, ours=None, sz=None, other=None, accs=None,
               products_only=False, prof=0):
        """ours / sz / other: a mix (the SplitZip layers are decoded first; a layer of ours after a SplitZip layer
        takes that layer's features from a features pass and starts its spread from prior 1). Timing only: accs
        (every layer's products, from products(): the coding alone), products_only (the products alone), prof
        (kernel ablations, see _szdec_kernel)."""
        L, W = self.L, self.W
        words = torch.empty(L, T, W, dtype=torch.int16, device=self.device) if out is None else out
        bufs = self.buffers(T) if bufs is None else bufs
        k, x, ssum = self.k, bufs["x"], bufs["ssum"]
        self.segments(T)
        Tc = T // self.chunks
        P = (Tc // self.BM) * self.G                 # programs per chunk and layer
        ours = [True] * L if ours is None else [bool(v) for v in ours]
        for l, e in (sz or {}).items():
            words[l].copy_(other.decode(e).view(torch.int16).view(T, W))
        cur = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        streams = bufs["streams"]
        for st in streams or []:
            st.wait_stream(cur)
        for l in range(L):
            if not ours[l]:
                continue
            for c in range(self.chunks):
                rows = slice(c * Tc, (c + 1) * Tc)
                ctx = torch.cuda.stream(streams[c]) if streams else lw._null()
                with ctx:
                    self._decode_layer(l, c, rows, Tc, P, stream, starts, words, bufs, ours, accs, products_only,
                                       prof)
        for st in streams or []:
            cur.wait_stream(st)
        return words

    def _decode_layer(self, l, c, rows, Tc, P, stream, starts, words, bufs, ours, accs, products_only, prof):
        L, W, T = self.L, self.W, words.shape[1]
        k, x, ssum = self.k, bufs["x"], bufs["ssum"]
        xin = x[(l + 1) % 2]
        if l and not ours[l - 1]:                    # features of the SplitZip layer before (this chunk's rows)
            self.lw._launch(lw._lfeat_kernel, (triton.cdiv(Tc * W, 1024),), words[l - 1], xin, k["isx"][l - 1],
                            k["cos"], k["sin"], c * Tc, Tc * W, W, self.C, HD=self.hd, BLOCK=1024, warps=4)
        mode, xp, ap, accp, K = 0, xin, xin, xin, 0
        if l and accs is not None:
            mode, accp = 1, accs[l][rows]
        elif l and self.product == "acc":
            self.lw._product(xin[rows], l, bufs["acc"][c], bufs["scratch"][c])
            mode, accp = 1, bufs["acc"][c]
        elif l:                                      # reduced rank: the first product (requantized), the second inside
            rc = self.rank
            BMq, BNq, BKq, nw = self.lw.rank_tiles
            zq, r = bufs["zq"][c], rc["rank"]
            if Tc % BMq or r % BNq or W % BKq:
                raise ValueError(f"epilogue tiles {BMq}x{BNq}x{BKq} do not divide {Tc} x {r} x {W}")
            self.lw._launch(lw._igemm_rq_kernel, ((Tc // BMq) * (r // BNq),), xin[rows], rc["Bq"][l], zq,
                            rc["m1"][l], rc["m0"][l], Tc, r, W, BM=BMq, BN=BNq, BK=BKq, GROUP=8, warps=nw,
                            num_stages=3)
            mode, xp, ap, K = 2, zq, rc["Uq"][l], r
        if products_only:
            return
        ones = l == 0 or not ours[l - 1]
        feat = l < L - 1 and ours[l + 1]
        if self.layout == "row":
            self._launch(_szrow_kernel, (P,), stream, starts[l], words[l], accp, x[l % 2], k["sa"][l], k["b"][l],
                         k["sp"][l], k["inv"][l], k["isx"][l], k["cos"], k["sin"], ssum[(l + 1) % 2], ssum[l % 2],
                         bufs["smult"][c * P * self.BM:], self.tab[l], self.csym[l], c * Tc, T, W, self.G, self.HG,
                         self.prior_scale, TB=self.BM, H2=self.hd // 2, MODE=mode, ONES=ones, FEAT=feat, PROF=prof)
        else:
            self._launch(_szdec_kernel, (P,), stream, starts[l], words[l], xp, ap, accp, x[l % 2], k["sa"][l],
                         k["b"][l], k["sp"][l], k["inv"][l], k["isx"][l], k["cos"], k["sin"], ssum[(l + 1) % 2],
                         ssum[l % 2], self.tab[l], self.csym[l], c * Tc, T, W, self.G, self.HG, K, self.prior_scale,
                         BM=self.BM, H2=self.hd // 2, BK=min(self.bk, K) if K else 16, MODE=mode, ONES=ones,
                         FEAT=feat, PROF=prof)
