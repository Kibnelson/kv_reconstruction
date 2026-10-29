"""One-step codec decoded layer by layer with the product in cuBLAS: per layer, an exact int8 matrix product
(torch._int_mm: the previous layer's 8-bit features times the 8-bit coefficients, int32 result), then one Triton
kernel that does everything else for that layer: the prediction's scale, bias and RoPE, the spread, the
value-slice coding, the surprise sums for the next layer's spread, and the next layer's 8-bit features.

Same predictor, spread and features as the fused codec (fused.prepare builds the constants, and the per-lane
arithmetic below repeats the fused tile arithmetic operation for operation, so the encoder can use the fused
forward pass for its features and priors).

A program is one warp: 128 lanes = the channels of one head, for a block of TB tokens; lane c carries one rANS
stream for channel c of that head and block through every layer (the state and position are kept in memory between
the per-layer kernels), so a layer has (T / TB) x heads programs while the stream states cost one 32-bit word per lane
for the whole document. Tokens can be split into chunks decoded on separate CUDA streams: a chunk's layer only
depends on the same chunk's previous layer, so one chunk's product runs while another chunk decodes."""
from __future__ import annotations

import torch

from . import fused as fu
from . import gpu_coder as gc

triton, tl = gc.triton, gc.tl


def int_mm(x, A, out=None):
    """Exact x [n, K] int8 times A [N, K] int8 transposed -> int32 [n, N] (cuBLAS on the GPU)."""
    if x.is_cuda:
        if out is not None:
            try:
                return torch._int_mm(x, A.t(), out=out)
            except TypeError:                    # no out= variant
                pass
        y = torch._int_mm(x, A.t())
        return y if out is None else out.copy_(y)
    y = (x.double() @ A.double().t()).to(torch.int32)    # exact: |sum| < 2^31 and float64 holds it
    return y if out is None else out.copy_(y)


if triton is not None:

    @triton.jit
    def _imm_tile(x_ptr, a_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
        """This program's [BM, BN] int32 tile of X [M, K] times A [N, K] transposed, with its row and column
        indices (grouped order: consecutive programs share A tiles in L2)."""
        pid = tl.program_id(0)
        nm = tl.cdiv(M, BM)
        nn = tl.cdiv(N, BN)
        gw = GROUP * nn
        gid = pid // gw
        fm = gid * GROUP
        gs = tl.minimum(nm - fm, GROUP)
        pm = fm + (pid % gw) % gs
        pn = (pid % gw) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.int32)
        for k0 in range(0, K, BK):
            xt = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :])
            at = tl.load(a_ptr + rn[:, None] * K + (k0 + rk)[None, :])
            acc = tl.dot(xt, tl.trans(at), acc, out_dtype=tl.int32)
        return acc, rm, rn

    @triton.jit
    def _igemm_rq_kernel(x_ptr, a_ptr, q_ptr, m1_ptr, m0_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr, GROUP: tl.constexpr):
        """Reduced-rank product, first half, with the requantization in the epilogue: q [M, N] int8 =
        clamp(floor(z * m1 + m0)) of z = X [M, K] times A [N, K] transposed. The same values as int_mm followed by
        _requant_kernel (same float32 operations, no fused multiply-add), but z never goes through memory."""
        acc, rm, rn = _imm_tile(x_ptr, a_ptr, M, N, K, BM, BN, BK, GROUP)
        q = tl.floor(acc.to(tl.float32) * tl.load(m1_ptr + rn)[None, :] + tl.load(m0_ptr + rn)[None, :])
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + rm[:, None] * N + rn[None, :], q.to(tl.int8))

    @triton.jit
    def _igemm_kernel(x_ptr, a_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      GROUP: tl.constexpr):
        """Exact int8 product C [M, N] int32 = X [M, K] times A [N, K] transposed (a small-footprint kernel that
        can share SMs with the coding kernel; the result equals cuBLAS's bit for bit, being integer)."""
        pid = tl.program_id(0)
        nm = tl.cdiv(M, BM)
        nn = tl.cdiv(N, BN)
        gw = GROUP * nn                              # grouped order: consecutive programs share A tiles in L2
        gid = pid // gw
        fm = gid * GROUP
        gs = tl.minimum(nm - fm, GROUP)
        pm = fm + (pid % gw) % gs
        pn = (pid % gw) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros([BM, BN], dtype=tl.int32)
        for k0 in range(0, K, BK):
            xt = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :])
            at = tl.load(a_ptr + rn[:, None] * K + (k0 + rk)[None, :])
            acc = tl.dot(xt, tl.trans(at), acc, out_dtype=tl.int32)
        tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)

    @triton.jit
    def _lane_mu(acc_row, own, part, sao, bo, sap, bp, cs, sn, low, key):
        """The fused tile's prediction for one token, per lane: scale and bias, then RoPE for keys (lane in the
        first half: own * cos - partner * sin; second half: own * cos + partner * sin)."""
        ao = tl.load(acc_row + own).to(tl.float32) * sao + bo
        ap = tl.load(acc_row + part).to(tl.float32) * sap + bp
        rot = tl.where(low, ao * cs - ap * sn, ao * cs + ap * sn)
        return tl.where(key, rot, ao)

    @triton.jit
    def _feature(v, vp, cs, sn, low, key, isx):
        """Next-layer 8-bit feature from a decoded value and its RoPE partner (the fused tile's arithmetic)."""
        d = tl.where(key, tl.where(low, v * cs + vp * sn, v * cs - vp * sn), v)
        q = tl.floor(d * isx + 0.5)
        q = tl.where(q == q, q, 0.0)
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        return q.to(tl.int8)

    @triton.jit
    def _lfeat_kernel(words_ptr, xout_ptr, isx_ptr, cos_ptr, sin_ptr, t_base, NV, W, C, HD: tl.constexpr,
                      BLOCK: tl.constexpr):
        """Features of a chunk of one decoded layer (a separate pass), element-wise over token x channel."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < NV
        idx = tl.where(m, idx, 0)
        t = t_base + idx // W
        c = idx % W
        cc = c % HD
        H2: tl.constexpr = HD // 2
        low = cc < H2
        part = tl.where(low, c + H2, c - H2)
        hh = tl.where(low, cc, cc - H2)
        v = gc._word_val(tl.load(words_ptr + t * W + c).to(tl.int32) & 0xFFFF)
        vp = gc._word_val(tl.load(words_ptr + t * W + part).to(tl.int32) & 0xFFFF)
        cs = tl.load(cos_ptr + t * H2 + hh)
        sn = tl.load(sin_ptr + t * H2 + hh)
        tl.store(xout_ptr + t * W + c, _feature(v, vp, cs, sn, low, c < C, tl.load(isx_ptr + c)), mask=m)

    @triton.jit
    def _ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, t_base, T, W, G, HG, G0, GN, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, FEAT: tl.constexpr, LEAN: tl.constexpr, ONES: tl.constexpr):
        """Decode one layer of one token block x head (pointers are the layer's: out_ptr, sa_ptr, ...).
        FEAT: make the next layer's features here (else a separate pass does); LEAN: reload the per-channel
        constants every token instead of keeping them in registers; FIRST: the stream states come from the stream
        (the first layer this codec decodes); ONES: prior 1 (no previous layer of ours to take surprise from).
        G0, GN: the heads this launch decodes, G0 .. G0 + GN - 1 (all G at once, or one head group)."""
        pid = tl.program_id(0)
        g = G0 + pid % GN
        blk = pid // GN
        t0 = t_base + blk * TB
        HD: tl.constexpr = 2 * H2
        cc = tl.arange(0, HD)
        c0 = g * HD
        key = g < HG
        low = cc < H2
        own = c0 + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        seg = (t0 // TB) * G + g
        valid = cc >= 0
        if FIRST:                                    # states from the stream: 8-aligned, just below the data
            rp = tl.load(start_ptr + seg)
            base = tl.multiple_of(((rp - 2 * HD) // 8) * 8, 8)
            x = tl.load(buf_ptr.to(tl.pointer_type(tl.int32)) + base // 2 + cc).to(tl.uint32, bitcast=True)
        else:
            rp = tl.load(pos_ptr + seg)
            x = tl.load(state_ptr + seg * HD + cc).to(tl.uint32, bitcast=True)
        # spread multipliers of the block's tokens, from the previous layer's surprise sums (all heads)
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
        if not LEAN:
            sao = tl.load(sa_ptr + own)
            bo = tl.load(b_ptr + own)
            sap = tl.load(sa_ptr + part)
            bp = tl.load(b_ptr + part)
            spl = tl.load(sp_ptr + own)
            invl = tl.load(inv_ptr + own)
            isxl = tl.load(isx_ptr + own)
        for i in range(0, TB):
            t = t0 + i
            if LEAN:
                sao = tl.load(sa_ptr + own)
                bo = tl.load(b_ptr + own)
                sap = tl.load(sa_ptr + part)
                bp = tl.load(b_ptr + part)
                spl = tl.load(sp_ptr + own)
                invl = tl.load(inv_ptr + own)
                isxl = tl.load(isx_ptr + own)
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = _lane_mu(acc_ptr + (t - t_base) * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            s = spl * tl.load(smult + i)
            x, rp, w = gc._dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, LUT_SHIFT, LUT_W,
                                     REFINE, UNIFORM, 0, SEARCH)
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            v = gc._word_val(w)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            if FEAT:
                tl.debug_barrier()                   # the partner lane's word, for the de-rotated features
                vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
                tl.store(xout_ptr + t * W + own, _feature(v, vp, cs, sn, low, key, isxl))
        if not LAST:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, rp)

    @triton.jit
    def _lenc_kernel(words_ptr, prior_ptr, state_ptr, pos_ptr, out_ptr, start_ptr, acc_ptr, sa_ptr, b_ptr, sp_ptr,
                     cos_ptr, sin_ptr, smult_ptr, cumf_ptr, ko_ptr, T, W, G, HG,
                     FLUSH: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr, UNIFORM: tl.constexpr):
        """Encode one layer of one token block x head, tokens in reverse (layers are launched last first)."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
        t0 = blk * TB
        HD: tl.constexpr = 2 * H2
        cc = tl.arange(0, HD)
        c0 = g * HD
        key = g < HG
        low = cc < H2
        own = c0 + cc
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
        for ii in range(0, TB):
            i = TB - 1 - ii
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = _lane_mu(acc_ptr + t * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            s = spl * tl.load(smult + i)
            w = tl.load(words_ptr + t * W + own).to(tl.int32) & 0xFFFF
            x, wp = gc._enc_value(x, wp, w, mu, s, valid, out_ptr, cumf_ptr, ko_ptr, UNIFORM, 0)
        if FLUSH:                                    # states 8-aligned just below the data; start = data start
            base = tl.multiple_of(((wp - 2 * HD) // 8) * 8, 8)
            tl.store(out_ptr.to(tl.pointer_type(tl.int32)) + base // 2 + cc, x.to(tl.int32, bitcast=True))
            tl.store(start_ptr + seg, wp)
        else:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, wp)


    @triton.jit
    def _requant_kernel(z_ptr, q_ptr, m1_ptr, m0_ptr, NV, R, BLOCK: tl.constexpr):
        """Reduced-rank product: the int32 intermediate requantized to 8 bits, q = clamp(floor(z * m1 + m0))."""
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = idx < NV
        j = idx % R
        z = tl.load(z_ptr + idx, mask=m, other=0).to(tl.float32)
        q = tl.floor(z * tl.load(m1_ptr + j) + tl.load(m0_ptr + j))
        q = tl.minimum(tl.maximum(q, -127.0), 127.0)
        tl.store(q_ptr + idx, q.to(tl.int8), mask=m)

    @triton.jit
    def _lsurp_kernel(words_ptr, acc_ptr, sa_ptr, b_ptr, inv_ptr, cos_ptr, sin_ptr, ssum_ptr, T, W, G, HG,
                      TB: tl.constexpr, H2: tl.constexpr):
        """Encoder's forward pass: one layer's surprise sums per head and token from the exact words, with the
        decoder's per-lane prediction arithmetic (so the priors agree bit for bit)."""
        pid = tl.program_id(0)
        g = pid % G
        t0 = (pid // G) * TB
        HD: tl.constexpr = 2 * H2
        cc = tl.arange(0, HD)
        key = g < HG
        low = cc < H2
        own = g * HD + cc
        part = tl.where(low, own + H2, own - H2)
        hh = tl.where(low, cc, cc - H2)
        sao = tl.load(sa_ptr + own)
        bo = tl.load(b_ptr + own)
        sap = tl.load(sa_ptr + part)
        bp = tl.load(b_ptr + part)
        invl = tl.load(inv_ptr + own)
        for i in range(0, TB):
            t = t0 + i
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = _lane_mu(acc_ptr + t * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            v = gc._word_val(tl.load(words_ptr + t * W + own).to(tl.int32) & 0xFFFF)
            tl.store(ssum_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))

    @triton.jit
    def _lprior_kernel(ssum_ptr, prior_ptr, T, G, PRIOR_SCALE, BLOCK: tl.constexpr):
        """Next layer's prior per token: the decoder's sum over heads, times the fixed-point scale."""
        t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = t < T
        sacc = tl.zeros([BLOCK], dtype=tl.int32)
        for gg in range(0, G):
            sacc += tl.load(ssum_ptr + gg * T + t, mask=m, other=0)
        tl.store(prior_ptr + t, sacc.to(tl.float32) * PRIOR_SCALE, mask=m)


def _requant_ref(z, m1, m0):
    """PyTorch mirror of _requant_kernel (calibration statistics only)."""
    q = torch.floor(z.to(torch.float32) * m1[None, :] + m0[None, :])
    return torch.clamp(q, -127, 127)


def _mu_from_acc(acc, sa, b, cos, sin, C, hd):
    """Prediction from int32 products (scale, bias, RoPE for keys), as reference_mu in fused."""
    n = acc.shape[0]
    mud = acc.to(torch.float32) * sa[None, :] + b[None, :]
    h2 = hd // 2
    k = mud[:, :C].reshape(n, -1, hd)
    a, bb = k[..., :h2], k[..., h2:]
    cs, sn = cos[:n][:, None, :], sin[:n][:, None, :]
    rot = torch.cat([a * cs - bb * sn, bb * cs + a * sn], -1).reshape(n, C)
    return torch.cat([rot, mud[:, C:]], 1)


def prepare_rank(study, consts, rank, log=print):
    """Constants of the reduced-rank one-step predictor (a faster product for fast links): per layer the dense
    one-step regression A, reduced to rank r (A ~ U U^T A, U the top r directions of the prediction's covariance),
    as two 8-bit products: z = Bq x_q (int32, r values), requantized to 8 bits per component (centred: z_q = clamp(
    floor(z * m1 + m0))), then acc = Uq z_q (int32) with per-channel scale sa and bias b (the calibration mean of
    the prediction matches the layer's mean). Features, RoPE tables and layer 0 are the dense constants'."""
    from .core import LN2X2, words_to_float
    L, W, hd, dev = study.L, study.W, study.hd, study.device
    f32, f64 = torch.float32, torch.float64
    r = int(rank)
    isx, sx = consts["isx"], consts["sx"]
    Bq = torch.zeros(L, r, W, dtype=torch.int8, device=dev)
    Uq = torch.zeros(L, W, r, dtype=torch.int8, device=dev)
    m1 = torch.ones(L, r, dtype=f32, device=dev)
    m0 = torch.zeros(L, r, dtype=f32, device=dev)
    sa = consts["sa"].clone()
    b = consts["b"].clone()

    def xq_of(d, l):
        f = study.feats(d, l, cache=False)
        return torch.clamp(torch.floor(f * isx[l][None, :] + 0.5), -127, 127).to(f64)

    for l in range(1, L):
        mu, S = study.covariance(l, ["P"])
        S.diagonal().add_(study.ridge * float(torch.trace(S)) / S.shape[0])
        Sxx, Syx = S[:W, :W], S[W:, :W]
        A = torch.linalg.solve(Sxx, Syx.T).T
        my = mu[W:]
        _, U = torch.linalg.eigh(A @ Sxx @ A.T)
        U = U[:, -r:]
        B = U.T @ A
        del S, Sxx, Syx, A
        Bf = B * sx[l - 1].to(f64)[None, :]
        sb = (Bf.abs().amax(1) / 127).clamp_min(1e-30)
        bq = torch.clamp(torch.round(Bf / sb[:, None]), -127, 127)
        zs = [(xq_of(d, l - 1) @ bq.T) for d in study.train]        # exact int32 values (in float64)
        n = sum(z.shape[0] for z in zs)
        cz = sum((z * sb[None, :]).sum(0) for z in zs) / n
        az = torch.stack([(z * sb[None, :] - cz).abs().amax(0) for z in zs]).amax(0)
        sz = (az / 127).clamp_min(1e-30)
        m1[l] = (sb / sz).to(f32)
        m0[l] = (-cz / sz + 0.5).to(f32)
        Uf = U * sz[None, :]
        su = (Uf.abs().amax(1) / 127).clamp_min(1e-30)
        uq = torch.clamp(torch.round(Uf / su[:, None]), -127, 127)
        macc = sum((_requant_ref(z, m1[l], m0[l]).to(f64) @ uq.T).sum(0) for z in zs)
        b[l] = (my - su * (macc / n)).to(f32)
        sa[l] = su.to(f32)
        Bq[l], Uq[l] = bq.to(torch.int8), uq.to(torch.int8)
        del zs
        study.forget(l - 1)
    rc = {"Bq": Bq, "Uq": Uq, "m1": m1, "m0": m0, "sa": sa, "b": b, "rank": r}
    sp = consts["sp"].clone()
    cos, sin = consts["cos"], consts["sin"]
    for l in range(1, L):
        x = words_to_float(study.tables.words(l, dev))
        z = xq_of(study.tables, l - 1) @ rc["Bq"][l].to(f64).T
        acc = (_requant_ref(z, m1[l], m0[l]).to(f64) @ rc["Uq"][l].to(f64).T)
        m = _mu_from_acc(acc, sa[l], b[l], cos, sin, study.C, hd)
        sp[l] = ((x - m.double()).abs().mean(0) / LN2X2 + 1e-12).to(f32)
    rc["sp"] = sp
    rc["inv"] = (fu.QS / (LN2X2 * sp.double())).to(f32)
    log(f"  reduced-rank one-step predictor (rank {r}) prepared for {L} layers")
    return rc


class LayerwiseOneStep:
    """Encode / decode a document's KV (all layers) with cuBLAS products and one coding kernel per layer."""

    def __init__(self, consts, C, hd, tokens_per_program=128, mode="uniform", search=True, chunks=1, fuse=False,
                 features="inline", lean=False, split_streams=False, product="cublas", gemm_tiles=(128, 128, 64, 4),
                 rank=None, requant="kernel", rank_tiles=(64, 128, 64, 4)):
        """features: "inline" (the coding kernel makes the next layer's features) or "pass" (a separate kernel);
        lean: reload per-channel constants every token (fewer registers); split_streams: per chunk, the products on
        a high-priority CUDA stream and the coding on another, so pending products are scheduled first; rank: the
        constants of prepare_rank (the reduced-rank product, for fast links) instead of the dense product;
        requant (reduced rank): "kernel" (cuBLAS first product, int32 intermediate, a requantization kernel) or
        "epilogue" (a Triton first product, tiles rank_tiles = (BM, BN, BK, warps), that requantizes in its
        epilogue: the same 8-bit intermediate, bit for bit); the second product follows `product`."""
        if triton is None:
            raise RuntimeError("Triton is required")
        if mode not in ("exact", "uniform"):
            raise ValueError("mode is 'exact' or 'uniform'")
        self.k = dict(consts)
        self.rank = rank
        if rank is not None:                         # the coding kernels read sa, b, sp, inv: the rank predictor's
            self.k.update(sa=rank["sa"], b=rank["b"], sp=rank["sp"], inv=rank["inv"])
        self.L, self.W = consts["Aq"].shape[0], consts["Aq"].shape[1]
        self.C, self.hd, self.TB = C, hd, int(tokens_per_program)
        self.mode, self.search, self.chunks = mode, bool(search), int(chunks)
        if features not in ("inline", "pass"):
            raise ValueError("features is 'inline' or 'pass'")
        self.features, self.lean, self.split = features, bool(lean), bool(split_streams)
        if product not in ("cublas", "triton"):
            raise ValueError("product is 'cublas' or 'triton'")
        self.product, self.gemm_tiles = product, tuple(int(v) for v in gemm_tiles)
        if requant not in ("kernel", "epilogue"):
            raise ValueError("requant is 'kernel' or 'epilogue'")
        self.requant, self.rank_tiles = requant, tuple(int(v) for v in rank_tiles)
        self.device = consts["Aq"].device
        self.tables = gc.slice_tables(self.device)
        self.opts = {} if fuse else {"enable_fp_fusion": False}
        self.G, self.HG = self.W // hd, C // hd
        self.prior_scale = 1.0 / (self.W * fu.QS)
        self.ng = int(consts.get("groups", 1))      # head groups of kv_compare.grouped (1: the one-step predictor)
        if self.ng > 1:
            if rank is not None or features != "inline" or split_streams:
                raise ValueError("head groups need the dense product, inline features and no split streams")
            if self.G % self.ng:
                raise ValueError("the head groups must divide the heads")
            self.GN = self.G // self.ng
            gw = self.GN * hd                        # lanes per group
            self.Ag = [None] + [consts["Acq"][:, j * gw:(j + 1) * gw, :j * gw].contiguous()
                                for j in range(1, self.ng)]   # group j's rows on the groups before it

    def _launch(self, kernel, grid, *args, warps=1, **kw):
        try:
            return kernel[grid](*args, **kw, num_warps=warps, **self.opts)
        except TypeError:                            # interpreter: no launch options
            return kernel[grid](*args, **kw)

    def layout(self, T):
        """(segments, capacity of a segment in 16-bit words); one segment per token block x head."""
        if T % (self.TB * self.chunks):
            raise ValueError("tokens must be a multiple of tokens_per_program x chunks")
        if T > self.k["cos"].shape[0]:
            raise ValueError("more tokens than the RoPE tables cover")
        S, cap = (T // self.TB) * self.G, self.hd * (2 * self.L * self.TB + 2) + 8
        if (S + 1) * cap >= 2 ** 31:                 # stream positions are 32-bit word offsets
            raise ValueError(f"{T} tokens need a {(S + 1) * cap / 2 ** 30:.1f} G-word stream buffer, over 2^31 words: "
                             f"code at most {(2 ** 31 // cap - 1) // self.G * self.TB} tokens per stream")
        return S, cap

    def model(self, words):
        """Features and priors of the exact words. Dense product: the fused forward pass (the same arithmetic);
        reduced rank and head groups: the features pass, then per layer the products and the decoder's surprise
        sums."""
        if self.ng > 1:
            return self._model_grouped(words)
        if self.rank is None:
            return fu.FusedOneStep(self.k, self.C, self.hd, tokens_per_program=32, rows=4, warps=4).model(words)
        L, T, W = words.shape
        dev, k = self.device, self.k
        xall = torch.zeros(L, T, W, dtype=torch.int8, device=dev)
        prior = torch.ones(L, T, dtype=torch.float32, device=dev)
        for l in range(L - 1):
            self._launch(_lfeat_kernel, (triton.cdiv(T * W, 1024),), words[l], xall[l], k["isx"][l], k["cos"],
                         k["sin"], 0, T * W, W, self.C, HD=self.hd, BLOCK=1024, warps=4)
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        ssum = torch.zeros(self.G, T, dtype=torch.int32, device=dev)
        for l in range(L - 1):
            if l == 0:
                acc.zero_()
            else:
                self._product(xall[l - 1], l, acc)
            self._launch(_lsurp_kernel, ((T // self.TB) * self.G,), words[l], acc, k["sa"][l], k["b"][l], k["inv"][l],
                         k["cos"], k["sin"], ssum, T, W, self.G, self.HG, TB=self.TB, H2=self.hd // 2)
            self._launch(_lprior_kernel, (triton.cdiv(T, 256),), ssum, prior[l + 1], T, self.G, self.prior_scale,
                         BLOCK=256, warps=4)
        return xall, prior

    def _model_grouped(self, words):
        L, T, W = words.shape
        dev, k = self.device, self.k
        xall = torch.zeros(L, T, W, dtype=torch.int8, device=dev)
        prior = torch.ones(L, T, dtype=torch.float32, device=dev)
        for l in range(L):                           # every layer's features: the same layer's later groups read them
            self._launch(_lfeat_kernel, (triton.cdiv(T * W, 1024),), words[l], xall[l], k["isx"][l], k["cos"],
                         k["sin"], 0, T * W, W, self.C, HD=self.hd, BLOCK=1024, warps=4)
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        ssum = torch.zeros(self.G, T, dtype=torch.int32, device=dev)
        for l in range(L - 1):
            self._acc_full(xall, l, acc)
            self._launch(_lsurp_kernel, ((T // self.TB) * self.G,), words[l], acc, k["sa"][l], k["b"][l], k["inv"][l],
                         k["cos"], k["sin"], ssum, T, W, self.G, self.HG, TB=self.TB, H2=self.hd // 2)
            self._launch(_lprior_kernel, (triton.cdiv(T, 256),), ssum, prior[l + 1], T, self.G, self.prior_scale,
                         BLOCK=256, warps=4)
        return xall, prior

    def _acc_full(self, xall, layer, acc):
        """The encoder's int32 sums of a layer: the previous layer's product, plus (head groups) the whole
        block-triangular product of the same layer's exact features."""
        if layer == 0:
            acc.zero_()
        else:
            self._product(xall[layer - 1], layer, acc)
        if self.ng > 1:
            acc += int_mm(xall[layer], self.k["Acq"][layer])

    def encode(self, words, xall=None, prior=None, out=None, layer_words=None):
        """layer_words: an int64 tensor [L] to receive the 16-bit words each layer adds to the streams (layer 0
        also gets the final states; starts and alignment are left out, see bits)."""
        L, T, W = words.shape
        S, cap = self.layout(T)
        if xall is None:
            xall, prior = self.model(words)
        dev, k, t = self.device, self.k, self.tables
        if out is None:
            out = torch.empty(S * cap, dtype=torch.int16, device=dev)
        starts = torch.empty(S, dtype=torch.int32, device=dev)
        state = torch.full((S * self.hd,), 65536, dtype=torch.int32, device=dev)
        pos = (torch.arange(S, dtype=torch.int32, device=dev) + 1) * cap
        acc = torch.zeros(T, W, dtype=torch.int32, device=dev)
        smult = torch.empty(S * self.TB, dtype=torch.float32, device=dev)
        for layer in reversed(range(L)):
            self._acc_full(xall, layer, acc)
            before = pos.sum(dtype=torch.int64) if layer_words is not None and layer > 0 else None
            self._launch(_lenc_kernel, (S,), words[layer], prior[layer], state, pos, out, starts, acc,
                         k["sa"][layer], k["b"][layer], k["sp"][layer], k["cos"], k["sin"], smult, t["cumf"], t["ko"],
                         T, W, self.G, self.HG, FLUSH=layer == 0, TB=self.TB, H2=self.hd // 2,
                         UNIFORM=self.mode == "uniform")
            if before is not None:
                layer_words[layer] = before - pos.sum(dtype=torch.int64)
        if layer_words is not None:                  # layer 0: the rest of the streams (its words, states, alignment)
            layer_words[0] = (self.bits(starts, T) - 32 * S) // 16 - layer_words[1:].sum()
        return out, starts

    def buffers(self, T):
        """Decoder working memory, reusable across calls."""
        S, _ = self.layout(T)
        dev = self.device
        Tc = T // self.chunks
        return {"xbuf": torch.zeros(2, T, self.W, dtype=torch.int8, device=dev),
                "ssum": torch.zeros(2, self.G, T, dtype=torch.int32, device=dev),
                "state": torch.empty(S * self.hd, dtype=torch.int32, device=dev),
                "pos": torch.empty(S, dtype=torch.int32, device=dev),
                "smult": torch.empty(S * self.TB, dtype=torch.float32, device=dev),
                "acc": [torch.zeros(Tc, self.W, dtype=torch.int32, device=dev) for _ in range(self.chunks)],
                "scratch": ([self._scratch(Tc) for _ in range(self.chunks)] if self.rank is not None
                            else [None] * self.chunks),
                "streams": ([torch.cuda.Stream(dev) for _ in range(self.chunks)]
                            if dev.type == "cuda" and (self.chunks > 1 or self.split) else None),
                "gemm_streams": ([torch.cuda.Stream(dev, priority=-1) for _ in range(self.chunks)]
                                 if dev.type == "cuda" and self.split else None)}

    def decode(self, buf, starts, T, out=None, bufs=None, accs=None, products_only=False):
        """accs: precomputed int32 products per layer (list of [T, W]; layer 0 ignored) - times the coding
        alone; products_only: run only the products (on features already in bufs["xbuf"]) - times them alone."""
        L, W = self.L, self.W
        S, _ = self.layout(T)
        words = torch.empty(L, T, W, dtype=torch.int16, device=self.device) if out is None else out
        bufs = self.buffers(T) if bufs is None else bufs
        k, t = self.k, self.tables
        Tc = T // self.chunks
        P = (Tc // self.TB) * self.G                  # programs per chunk and layer
        cur = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        streams, gstreams = bufs["streams"], bufs.get("gemm_streams")
        for s in (streams or []) + (gstreams or []):
            s.wait_stream(cur)
        for layer in range(L):                       # issued layer by layer, so the chunks' streams interleave
            for c in range(self.chunks):
                rows = slice(c * Tc, (c + 1) * Tc)
                acc = bufs["acc"][c] if accs is None else accs[layer][rows]
                if accs is None:
                    gctx = torch.cuda.stream(gstreams[c]) if gstreams else (
                        torch.cuda.stream(streams[c]) if streams else _null())
                    with gctx:
                        if gstreams:
                            gstreams[c].wait_stream(streams[c])      # features of the previous layer, acc free
                        if layer == 0:
                            acc.zero_()
                        else:
                            self._product(bufs["xbuf"][(layer + 1) % 2][rows], layer, acc, bufs["scratch"][c])
                    if gstreams:
                        streams[c].wait_stream(gstreams[c])
                if products_only:
                    continue
                ctx = torch.cuda.stream(streams[c]) if streams else _null()
                with ctx:
                    gw = self.GN * self.hd if self.ng > 1 else W
                    for j in range(self.ng):             # head groups in order (one launch for all heads if 1)
                        if j > 0:                        # the group's sums from the groups decoded before it
                            xin = bufs["xbuf"][layer % 2][rows, :j * gw].contiguous()
                            acc[:, j * gw:(j + 1) * gw] += int_mm(xin, self.Ag[j][layer])
                        GN = self.GN if self.ng > 1 else self.G
                        self._launch(_ldec_kernel, ((Tc // self.TB) * GN,), buf, starts, bufs["state"], bufs["pos"],
                                     words[layer], acc, bufs["xbuf"][layer % 2], k["sa"][layer], k["b"][layer],
                                     k["sp"][layer], k["inv"][layer], k["isx"][layer], k["cos"], k["sin"],
                                     bufs["ssum"][(layer + 1) % 2], bufs["ssum"][layer % 2],
                                     bufs["smult"][c * P * self.TB:], t["cumf"], t["lut"], t["tail"], c * Tc, T, W,
                                     self.G, self.HG, j * GN if self.ng > 1 else 0, GN, self.prior_scale,
                                     FIRST=layer == 0, LAST=layer == L - 1, TB=self.TB, H2=self.hd // 2,
                                     LUT_SHIFT=16 - gc.LUT_BITS, LUT_W=gc.LUT_W, REFINE=gc.REFINE,
                                     UNIFORM=self.mode == "uniform", SEARCH=self.search,
                                     FEAT=self.features == "inline", LEAN=self.lean, ONES=layer == 0)
                    if self.features == "pass" and layer < L - 1:
                        nv = Tc * W
                        self._launch(_lfeat_kernel, (triton.cdiv(nv, 1024),), words[layer], bufs["xbuf"][layer % 2],
                                     k["isx"][layer], k["cos"], k["sin"], c * Tc, nv, W, self.C, HD=self.hd, BLOCK=1024,
                                     warps=4)
        for s in (streams or []) + (gstreams or []):
            cur.wait_stream(s)
        return words

    def _scratch(self, n):
        """Working memory of the reduced-rank product for n tokens: the int32 and 8-bit intermediates."""
        r = self.rank["rank"]
        return {"z": torch.empty(n, r, dtype=torch.int32, device=self.device) if self.requant == "kernel" else None,
                "zq": torch.empty(n, r, dtype=torch.int8, device=self.device)}

    def _product(self, x, layer, out, scratch=None):
        """The layer's int32 products from 8-bit features x [n, W] into out [n, W]."""
        if self.rank is not None:                    # two 8-bit products with an 8-bit intermediate
            rc = self.rank
            n, r = x.shape[0], rc["rank"]
            s = self._scratch(n) if scratch is None else scratch
            zq = s["zq"]
            BM, BN, BK, nw = self.rank_tiles
            if self.requant == "epilogue":
                if n % BM or r % BN or self.W % BK:
                    raise ValueError(f"epilogue tiles {BM}x{BN}x{BK} do not divide {n} x {r} x {self.W}")
                self._launch(_igemm_rq_kernel, ((n // BM) * (r // BN),), x, rc["Bq"][layer], zq, rc["m1"][layer],
                             rc["m0"][layer], n, r, self.W, BM=BM, BN=BN, BK=BK, GROUP=8, warps=nw, num_stages=3)
            else:
                z = s["z"] if s["z"] is not None else torch.empty(n, r, dtype=torch.int32, device=x.device)
                int_mm(x, rc["Bq"][layer], out=z)
                self._launch(_requant_kernel, (triton.cdiv(z.numel(), 1024),), z, zq, rc["m1"][layer],
                             rc["m0"][layer], z.numel(), r, BLOCK=1024, warps=4)
            return self._dense(zq, rc["Uq"][layer], out)
        return self._dense(x, self.k["Aq"][layer], out)

    def _dense(self, x, A, out):
        """out [n, N] int32 = x [n, K] int8 times A [N, K] int8 transposed: cuBLAS, or the Triton kernel."""
        if self.product == "cublas" or not x.is_cuda:
            return int_mm(x, A, out=out)
        BM, BN, BK, nw = self.gemm_tiles
        M, K = x.shape
        N = A.shape[0]
        if M % BM or N % BN or K % BK:
            return int_mm(x, A, out=out)
        grid = ((M // BM) * (N // BN),)
        _igemm_kernel[grid](x, A, out, M, N, K, BM=BM, BN=BN, BK=BK, GROUP=8, num_warps=nw, num_stages=3)
        return out

    def products(self, xall):
        """The exact int32 products of every layer from features xall (for timing the coding alone)."""
        out = [None]
        for layer in range(1, self.L):
            out.append(self._product(xall[layer - 1], layer, torch.empty(xall.shape[1], self.W, dtype=torch.int32,
                                                                       device=xall.device)))
        out[0] = torch.zeros_like(out[1])
        return out

    def bits(self, starts, T):
        """Coded size: every segment's used words (data, states, alignment) plus a 32-bit start per segment."""
        S, cap = self.layout(T)
        ends = (torch.arange(S, device=starts.device, dtype=torch.int64) + 1) * cap
        base = torch.div(starts.to(torch.int64) - 2 * self.hd, 8, rounding_mode="floor") * 8
        return 16 * int((ends - base).sum()) + 32 * S


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
