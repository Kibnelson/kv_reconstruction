"""Skip unused decoder context without changing the lossless rANS stream.

The local kernel is derived from the unchanged ``team_onestep.layerwise``
``_ldec_kernel`` and keeps its decoded-word, probability, state and position
arithmetic. Only outputs that no subsequent coded layer reads are omitted.
The controller follows ``team_extensions.mixed.MixedOneStep.decode_mixed``.
Neither source snapshot is edited.

Reviewed base: ad305c56bf6446b39cc5698f1b7f94d3d58ebbf0.
Original layerwise.py SHA256:
b21b808673f7e956fa9caefd1ff14443fdf7722c81e349b67e07ee45639a5f61.
Original mixed.py SHA256:
3193e510df697087ea300e233217b04303f1370b543ce10bf8f87a2f577442e7.
"""
from __future__ import annotations

import torch

from .team_onestep import layerwise as lw

fu, gc = lw.fu, lw.gc
triton, tl = lw.triton, lw.tl


def skipplan(mask):
    """Layers whose decoded context is consumed by an adjacent rANS layer.

    rANS state and byte position still carry through every coded layer except
    the final coded layer, including across intervening SplitZip layers.
    """
    if not isinstance(mask, (list, tuple)) or not mask or any(type(x) is not bool for x in mask):
        raise ValueError('one explicit boolean per layer is required')
    coded = [i for i, value in enumerate(mask) if value]
    live = [i for i in coded if i + 1 < len(mask) and mask[i + 1]]
    dead = [i for i in coded if i not in live]
    return dict(rans_layers=coded, live_context_layers=live, dead_context_layers=dead,
        feature_output_layers=live, surprise_output_layers=live,
        feature_output_passes=len(live), surprise_output_passes=len(live),
        skipped_feature_output_passes=len(dead), skipped_surprise_output_passes=len(dead),
        reference_feature_output_passes=len(coded), reference_surprise_output_passes=len(coded),
        state_carry_layers=coded[:-1],
        prior_reset_layers=[i for i in coded if i == 0 or not mask[i-1]],
        scope='Layer counts; each layer is launched once per token chunk')


if triton is not None:
    @triton.jit
    def _ldec_context_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, t_base, T, W, G, HG, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, CONTEXT: tl.constexpr, LEAN: tl.constexpr, ONES: tl.constexpr):
        """Same rANS arithmetic; produce surprise/features only when consumed next."""
        pid = tl.program_id(0)
        g = pid % G
        blk = pid // G
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
            if CONTEXT:
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
                if CONTEXT:
                    invl = tl.load(inv_ptr + own)
                    isxl = tl.load(isx_ptr + own)
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = lw._lane_mu(acc_ptr + (t - t_base) * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            s = spl * tl.load(smult + i)
            x, rp, w = gc._dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, LUT_SHIFT, LUT_W,
                                     REFINE, UNIFORM, 0, SEARCH)
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            if CONTEXT:
                v = gc._word_val(w)
                tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
                tl.debug_barrier()                   # the partner lane's word, for the de-rotated features
                vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
                tl.store(xout_ptr + t * W + own, lw._feature(v, vp, cs, sn, low, key, isxl))
        if not LAST:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, rp)


def decode_selective(native, enc, T, out=None, bufs=None):
    """Decode an already-validated native mixed encoding with selective context.

    ``enc`` has the same trust boundary as the original native decoder. The
    checked transfer adapter must validate compact fields before this call.
    An empty ``enc['sz']`` with fallback layers requires a caller-provided output
    whose fallback rows have already been filled. Supplied buffers are caller
    owned and must be ordered across calls, as in the original decoder.
    """
    if type(T) is not int or T <= 0:
        raise ValueError('a positive token count is required')
    native.layout(T)
    plan = skipplan(native.ours)
    if len(native.ours) != native.L or plan['rans_layers'] != native.mine:
        raise ValueError('native mixed layer schedule differs')
    if not isinstance(enc, dict) or not isinstance(enc.get('sz'), dict):
        raise ValueError('validated native mixed encoding required')
    fallback = {i for i, ours in enumerate(native.ours) if not ours}
    provided = set(enc['sz'])
    if provided and provided != fallback:
        raise ValueError('native fallback layer set differs')
    if out is None and fallback and provided != fallback:
        raise ValueError('predecoded fallback rows require an explicit output')
    shape = (native.L, T, native.W)
    if out is None:
        words = torch.empty(shape, dtype=torch.int16, device=native.device)
    else:
        if (out.dtype != torch.int16 or out.device != native.device
            or tuple(out.shape) != shape or not out.is_contiguous()):
            raise ValueError('output geometry/device must match the original words')
        words = out
    for layer, encoded in enc['sz'].items():
        words[layer].copy_(native.other.decode(encoded).view(torch.int16).view(T, native.W))
    if not native.mine:
        return words
    bufs = native.buffers(T) if bufs is None else bufs
    k, tables = native.k, native.tables
    buf, starts = enc['stream'], enc['starts']
    Tc = T // native.chunks
    P = (Tc // native.TB) * native.G
    current = torch.cuda.current_stream(native.device) if native.device.type == 'cuda' else None
    streams = bufs['streams']
    for stream in streams or []:
        stream.wait_stream(current)
    first, last = native.mine[0], native.mine[-1]
    for layer in native.mine:
        produce_context = layer + 1 < native.L and native.ours[layer + 1]
        for chunk in range(native.chunks):
            rows = slice(chunk * Tc, (chunk + 1) * Tc)
            acc = bufs['acc'][chunk]
            context = torch.cuda.stream(streams[chunk]) if streams else lw._null()
            with context:
                if layer == 0:
                    acc.zero_()
                else:
                    if not native.ours[layer - 1]:
                        nv = Tc * native.W
                        native._launch(lw._lfeat_kernel, (triton.cdiv(nv, 1024),), words[layer - 1],
                            bufs['xbuf'][(layer + 1) % 2], k['isx'][layer - 1], k['cos'], k['sin'],
                            chunk * Tc, nv, native.W, native.C, HD=native.hd, BLOCK=1024, warps=4)
                    native._product(bufs['xbuf'][(layer + 1) % 2][rows], layer, acc, bufs['scratch'][chunk])
                native._launch(_ldec_context_kernel, (P,), buf, starts, bufs['state'], bufs['pos'],
                    words[layer], acc, bufs['xbuf'][layer % 2], k['sa'][layer], k['b'][layer], k['sp'][layer],
                    k['inv'][layer], k['isx'][layer], k['cos'], k['sin'], bufs['ssum'][(layer + 1) % 2],
                    bufs['ssum'][layer % 2], bufs['smult'][chunk * P * native.TB:], tables['cumf'], tables['lut'],
                    tables['tail'], chunk * Tc, T, native.W, native.G, native.HG, native.prior_scale,
                    FIRST=layer == first, LAST=layer == last, TB=native.TB, H2=native.hd // 2,
                    LUT_SHIFT=16 - gc.LUT_BITS, LUT_W=gc.LUT_W, REFINE=gc.REFINE,
                    UNIFORM=native.mode == 'uniform', SEARCH=native.search, CONTEXT=produce_context,
                    LEAN=False, ONES=layer == 0 or not native.ours[layer - 1])
    for stream in streams or []:
        current.wait_stream(stream)
    if current is not None:
        for stream in [current] + (streams or []) + (bufs.get('gemm_streams') or []):
            buf.record_stream(stream)
            starts.record_stream(stream)
            words.record_stream(stream)
    return words
