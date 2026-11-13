"""Experiment-only coding kernels; baseline kernels are left untouched.

The decoder/encoder bodies are copied from layerwise.py at d95fd10. Only the
source of the probability-width prior changes. Both directions use the same
_head_prior arithmetic. All mean prediction, feature, RoPE and rANS operations
are inherited unchanged. FP fusion stays disabled, as in the pinned baseline.
"""
from . import fused as fu
from . import gpu_coder as gc
from .layerwise import _lane_mu, _feature

triton, tl = gc.triton, gc.tl

if triton is not None:
    @triton.jit
    def _head_prior(ssum, mix, bias, g, t, T, G, PRIOR_SCALE,
                    KIND: tl.constexpr, ALPHA: tl.constexpr, SHIFT: tl.constexpr):
        total = tl.full(t.shape, 0, tl.int32)
        for gg in range(0, G):
            total += tl.load(ssum + gg * T + t)
        global_prior = total.to(tl.float32) * PRIOR_SCALE
        if KIND == 0 or ALPHA == 0.0:
            return global_prior
        else:
            head_scale = PRIOR_SCALE * G
            if KIND == 1:
                local = tl.load(ssum + ((g + SHIFT) % G) * T + t).to(tl.float32) * head_scale
            else:
                local = tl.full(t.shape, 0.0, tl.float32) + tl.load(bias + g)
                for gg in range(0, G):
                    value = tl.load(ssum + ((gg + SHIFT) % G) * T + t).to(tl.float32) * head_scale
                    local = local + tl.load(mix + g * G + gg) * value
            return global_prior + ALPHA * (local - global_prior)

    @triton.jit
    def _head_prior_kernel(ssum, prior, mix, bias, T, G, PRIOR_SCALE,
                           KIND: tl.constexpr, ALPHA: tl.constexpr, SHIFT: tl.constexpr,
                           BLOCK: tl.constexpr):
        g = tl.program_id(0)
        t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        # The experiment's layout requires a complete multiple of BLOCK.
        value = _head_prior(ssum, mix, bias, g, t, T, G, PRIOR_SCALE, KIND, ALPHA, SHIFT)
        tl.store(prior + g * T + t, value)

    @triton.jit
    def _head_ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, t_base, T, W, G, HG, G0, GN, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, FEAT: tl.constexpr, LEAN: tl.constexpr, ONES: tl.constexpr, mix_ptr, bias_ptr,
                          POLICY_KIND: tl.constexpr, ALPHA: tl.constexpr, HEAD_SHIFT: tl.constexpr):
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
            prior = _head_prior(ssum_in_ptr, mix_ptr, bias_ptr, g, t0 + rb, T, G, PRIOR_SCALE,
                                POLICY_KIND, ALPHA, HEAD_SHIFT)
            tl.store(smult + rb, fu._mult(prior))
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
    def _head_lenc_kernel(words_ptr, prior_ptr, state_ptr, pos_ptr, out_ptr, start_ptr, acc_ptr, sa_ptr, b_ptr, sp_ptr,
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
        tl.store(smult + rb, fu._mult(tl.load(prior_ptr + g * T + t0 + rb)))
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
