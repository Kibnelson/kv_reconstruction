"""Reuse static partner channels in the exact one-step GPU decoder.

The 128-channel head is two 64-channel halves. Its RoPE partner is a static
half swap of values already held in registers. This kernel preserves the
frozen decoder's FP32 operations, integer coding, probabilities and wire;
it avoids the duplicate partner prediction and decoded-word memory reloads.

The source snapshot stays untouched. An instance-local launch adapter replaces
only its layer decoder kernel. All matrix products, features, surprise sums,
stream scheduling, packet validation and output ownership retain their existing
implementations and arithmetic.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .team_lookup import TeamLookupCodec
from .team_onestep import layerwise as lw
from .team_onestep import fused as fu
from .team_onestep import gpu_coder as gc

triton, tl = gc.triton, gc.tl
_feature = lw._feature if triton is not None else None

if triton is not None:
    @triton.jit
    def _half_swap(values, H2: tl.constexpr):
        # A compile-time layout operation, rather than a dynamic gather.
        return tl.reshape(tl.flip(tl.reshape(values, (2, H2)), dim=0), (2 * H2,))

    @triton.jit
    def _register_lane_mu(acc_row, own, sao, bo, cs, sn, low, key, H2: tl.constexpr):
        # Preserve the original FP32 scale/add and RoPE operation order. The
        # partner's rounded scaled value already occurs in this complete head.
        ao = tl.load(acc_row + own).to(tl.float32) * sao + bo
        ap = _half_swap(ao, H2)
        rot = tl.where(low, ao * cs - ap * sn, ao * cs + ap * sn)
        return tl.where(key, rot, ao)

    @triton.jit
    def _register_ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, t_base, T, W, G, HG, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, FEAT: tl.constexpr, LEAN: tl.constexpr, ONES: tl.constexpr):
        """Decode one layer of one token block x head (pointers are the layer's: out_ptr, sa_ptr, ...).
        FEAT: make the next layer's features here (else a separate pass does); LEAN: reload the per-channel
        constants every token instead of keeping them in registers; FIRST: the stream states come from the stream
        (the first layer this codec decodes); ONES: prior 1 (no previous layer of ours to take surprise from)."""
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
            spl = tl.load(sp_ptr + own)
            invl = tl.load(inv_ptr + own)
            isxl = tl.load(isx_ptr + own)
        for i in range(0, TB):
            t = t0 + i
            if LEAN:
                sao = tl.load(sa_ptr + own)
                bo = tl.load(b_ptr + own)
                spl = tl.load(sp_ptr + own)
                invl = tl.load(inv_ptr + own)
                isxl = tl.load(isx_ptr + own)
            cs = tl.load(cos_ptr + t * H2 + hh)
            sn = tl.load(sin_ptr + t * H2 + hh)
            mu = _register_lane_mu(acc_ptr + (t - t_base) * W, own, sao, bo, cs, sn, low, key, H2)
            s = spl * tl.load(smult + i)
            x, rp, w = gc._dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, LUT_SHIFT, LUT_W,
                                     REFINE, UNIFORM, 0, SEARCH)
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            v = gc._word_val(w)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            if FEAT:
                # Match int16 store/reload bits without the global reload
                # or its barrier; the complete decoded head remains in w.
                vp = gc._word_val(_half_swap(w.to(tl.int32) & 0xFFFF, H2))
                tl.store(xout_ptr + t * W + own, _feature(v, vp, cs, sn, low, key, isxl))
        if not LAST:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, rp)


def register_source_identity():
    """Source-only receipt; no fitted tensors or GPU state are loaded."""
    root = Path(__file__).resolve().parents[1]
    names = ("kv_compare/team_register.py", "kv_compare/team_onestep/layerwise.py",
             "kv_compare/team_onestep/gpu_coder.py", "kv_compare/team_onestep/fused.py")
    return dict(format="team_register_kernel_v1",
        sources={n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in names},
        reference_kernel="_ldec_kernel", candidate_kernel="_register_ldec_kernel",
        transformation="static half swap of scaled prediction and decoded-word partner",
        probabilities="unchanged fixed CDF and BF16 symbols")


def kernel_provenance_for(*, decoder_mode, register_partner, decoder_warps=1):
    if decoder_mode not in ("graph", "direct") or type(register_partner) is not bool:
        raise ValueError("decoder mode must be graph/direct and register_partner must be bool")
    if type(decoder_warps) is not int or decoder_warps not in (1, 2, 4, 8):
        raise ValueError("decoder_warps must be 1, 2, 4 or 8")
    return dict(register_source_identity(), decoder_mode=decoder_mode,
        register_partner=register_partner, decoder_warps=decoder_warps,
        selected_kernel="_register_ldec_kernel" if register_partner else "_ldec_kernel",
        enable_fp_fusion=False, inverse_algorithm="coarse_binary_search")


class TeamRegisterCodec(TeamLookupCodec):
    """Same checked packet API, with reference or static-partner decoding.

    This is still a sequential public API. The inherited stream ordering,
    borrowed-input alignment/lifetime checks and owned outputs remain active.
    The selected launch configuration is fixed for the lifetime of the codec.
    """
    def __init__(self, state_path, device, arm, *, decoder_mode="graph",
                 register_partner=True, decoder_warps=1):
        proof = kernel_provenance_for(decoder_mode=decoder_mode,
            register_partner=register_partner, decoder_warps=decoder_warps)
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode, lookup=False)
        self.register_partner = register_partner
        self.decoder_warps = decoder_warps
        self.kernel_provenance = proof
        original_launch = self.native._launch
        target = _register_ldec_kernel if register_partner else lw._ldec_kernel
        selected_warps = decoder_warps

        def launch(kernel, grid, *args, warps=1, **kwargs):
            if kernel is lw._ldec_kernel:
                return original_launch(target, grid, *args, warps=selected_warps, **kwargs)
            return original_launch(kernel, grid, *args, warps=warps, **kwargs)

        self.native._launch = launch

    def accounting(self, packet):
        return dict(super().accounting(packet), register_partner=self.register_partner,
                    decoder_warps=self.decoder_warps,
                    decoder_kernel=self.kernel_provenance["selected_kernel"])
