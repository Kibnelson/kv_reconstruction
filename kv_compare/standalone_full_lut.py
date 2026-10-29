"""Exact stage-1 inverse LUT for the standalone lossless GPU decoder.

This local execution kernel derives from the unchanged team_onestep
layerwise._ldec_kernel and gpu_coder._dec_value. It changes only stage-1
CDF inversion: one UINT16 lookup replaces the coarse-table binary search.
Probability tables, prediction arithmetic, rANS state updates, word bounds,
uniform/exact in-slice coding, escapes and reconstructed BF16 bits are unchanged.
The independent proof covers every slot of every original fixed CDF class.
No source snapshot or global function is modified.
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .team_onestep import gpu_coder as gc
from .team_onestep import layerwise as lw

triton, tl = gc.triton, gc.tl
fu = lw.fu


def _tensor_receipt(value):
    array = value.detach().cpu().numpy().astype('<i4', copy=False)
    return dict(dtype='int32', shape=list(array.shape),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


@functools.lru_cache(maxsize=1)
def _cached_inverse():
    tables = gc.slice_tables('cpu')
    slots = np.arange(gc.M, dtype=np.int64)
    inverse = np.empty((gc.CLASSES, gc.M), dtype='<u2')
    binary = np.empty_like(inverse)
    reference = np.empty_like(inverse)
    for c in range(gc.CLASSES):
        off = int(tables['off'][c]); count = int(tables['K'][c]) + 2
        cum = tables['cum'][off:off + count].numpy().astype(np.int64)
        expected = np.searchsorted(cum, slots, side='right') - 1
        if np.any(expected < 0) or np.any(expected > np.iinfo(np.uint16).max):
            raise AssertionError('stage-1 symbol does not fit UINT16')
        # Simulate the original packed coarse-table binary search independently
        # of searchsorted, including per-lane intervals and packed CDF loads.
        row = tables['lut'][c].numpy().astype(np.int64)
        lp = row[slots >> (16 - gc.LUT_BITS)]
        klo = lp & 0xffff
        khi = klo + (lp >> 16)
        kop = int(row[gc.LUT_W - 3]); packed_off = kop >> 10
        packed = tables['cumf'].numpy().astype(np.int64)
        iterations = 0
        while np.max(khi - klo) > 1:
            mid = (klo + khi) // 2
            go = ((packed[packed_off + mid] >> 16) & 0xffff) <= slots
            klo = np.where(go, mid, klo)
            khi = np.where(go, khi, mid)
            iterations += 1
            if iterations > 16:
                raise AssertionError('original stage-1 search did not converge')
        alternate = gc.stage1_reference(tables, c, slots)
        if not np.array_equal(expected, klo) or not np.array_equal(expected, alternate):
            raise AssertionError('stage-1 inverse differs from original CDF algorithms')
        inverse[c] = expected
        binary[c] = klo
        reference[c] = alternate
    payload = inverse.tobytes(order='C')
    source_root = Path(__file__).resolve().parent
    proof = dict(format='standalone_full_inverse_v1', verified=True,
        classes=gc.CLASSES, slots_per_class=gc.M, checked_slots=gc.CLASSES * gc.M,
        dtype='uint16', shape=[gc.CLASSES, gc.M], table_bytes=len(payload),
        table_sha256=hashlib.sha256(payload).hexdigest(),
        coarse_search_sha256=hashlib.sha256(binary.tobytes(order='C')).hexdigest(),
        central_tail_reference_sha256=hashlib.sha256(reference.tobytes(order='C')).hexdigest(),
        fixed_table_sha256={name:_tensor_receipt(value) for name,value in sorted(tables.items())},
        sources={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                 for name in ('team_onestep/gpu_coder.py','team_onestep/layerwise.py')},
        scope='All fixed stage-1 slots only; complete BF16 and packet parity require separate replay checks')
    return payload, json.dumps(proof, sort_keys=True, separators=(',', ':'))


def full_inverse_proof():
    """Return an independently owned copy of the exhaustive CPU proof."""
    return json.loads(_cached_inverse()[1])


def build_full_inverse(device='cpu'):
    """Create owned contiguous UINT16[classes, slots] storage and its proof."""
    payload, proof_json = _cached_inverse()
    array = np.frombuffer(payload, dtype='<u2').copy().reshape(gc.CLASSES, gc.M)
    table = torch.from_numpy(array).to(device)
    return table, json.loads(proof_json)


if triton is not None:
    @triton.jit
    def _dec_value_full(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, inverse_ptr, LUT_SHIFT: tl.constexpr,
                   LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr, SEG: tl.constexpr,
                   SEARCH: tl.constexpr):
        """Decode one word per lane; returns (state, read position, word)."""
        c = gc._width_class(mu, s)
        row = lut_ptr + c * LUT_W
        kop = tl.load(row + (LUT_W - 3))
        K = kop & 1023
        offf = kop >> 10
        slot = (x & 0xFFFF).to(tl.int32)
        # Exhaustively proven inverse of the same discrete stage-1 CDF.
        k = tl.load(inverse_ptr + c * 65536 + slot).to(tl.int32)
        p = tl.load(cumf_ptr + offf + k)
        x, rp = gc._take(x, (p >> 16) & 0xFFFF, p & 0xFFFF, valid, buf_ptr, rp, SEG)
        esc = k == K
        lo, hi = gc._bounds(c, tl.where(esc, 0, k), mu, s)
        first = gc._first_ge(lo)
        last = gc._first_ge(hi) - 1
        n = last - first + 1
        slot2 = (x & 0xFFFF).to(tl.int32)
        if UNIFORM:
            q = gc._q_of(n)
            j = gc._j_of(slot2, q, n)
            o = tl.where(valid & (~esc) & (n > 1), first + j, first)
            cum2 = tl.where(esc, slot2, j * q)
            f2 = tl.where(esc, 1, tl.where(j == n - 1, 65536 - (n - 1) * q, q))
        else:
            base, R, width = gc._shares(first, last, n)
            # estimate the word from the slot, refine it for the one-slot-per-word term, then a binary search on
            # the cumulative slots (the loop runs to the slowest lane of the program, so a good estimate matters)
            u = base + slot2.to(tl.float32) * (width * (1.0 / 65536.0))
            oe = tl.minimum(tl.maximum(gc._ord(gc._bf16_word(u)), first), last)
            inv_r = width * gc._rcp((65536 - n).to(tl.float32))                 # an estimate, searched below
            for _r in tl.static_range(REFINE):
                u = base + (slot2 - (oe - first)).to(tl.float32) * inv_r
                oe = tl.minimum(tl.maximum(gc._ord(gc._bf16_word(u)), first), last)
            ce0 = gc._cw(oe, first, last, base, R)
            ce1 = gc._cw(oe + 1, first, last, base, R)
            search = valid & (~esc) & (n > 1)
            below = search & (ce0 > slot2)
            above = search & (ce1 <= slot2)
            olo = tl.where(below, first, tl.where(above, oe + 1, oe))
            ohi = tl.where(below, oe, tl.where(above, last + 1, oe + 1))
            clo = tl.where(below, 0, tl.where(above, ce1, ce0))              # C(first) = 0
            chi = tl.where(below, ce0, tl.where(above, 65536, ce1))          # C(last + 1) = 65536
            while tl.max(ohi - olo, 0) > 1:
                mid = (olo + ohi) // 2
                cm = gc._cw(mid, first, last, base, R)
                go = cm <= slot2
                olo = tl.where(go, mid, olo)
                clo = tl.where(go, cm, clo)
                ohi = tl.where(go, ohi, mid)
                chi = tl.where(go, chi, cm)
            o = tl.where(search, olo, first)
            cum2 = tl.where(esc, slot2, clo)
            f2 = tl.where(esc, 1, chi - clo)
        two = valid & (esc | (n > 1))
        x, rp = gc._take(x, cum2, f2, two, buf_ptr, rp, SEG)
        w = tl.where(esc, slot2, gc._word(o))
        return x, rp, w

    @triton.jit
    def _full_lut_ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, inverse_ptr, t_base, T, W, G, HG, PRIOR_SCALE,
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
            mu = lw._lane_mu(acc_ptr + (t - t_base) * W, own, part, sao, bo, sap, bp, cs, sn, low, key)
            s = spl * tl.load(smult + i)
            x, rp, w = _dec_value_full(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, inverse_ptr, LUT_SHIFT, LUT_W,
                                     REFINE, UNIFORM, 0, SEARCH)
            tl.store(out_ptr + t * W + own, w.to(tl.int16))
            v = gc._word_val(w)
            tl.store(ssum_out_ptr + g * T + t, tl.sum(fu._sq(v, mu, invl), 0))
            if FEAT:
                tl.debug_barrier()                   # the partner lane's word, for the de-rotated features
                vp = gc._word_val(tl.load(out_ptr + t * W + part).to(tl.int32) & 0xFFFF)
                tl.store(xout_ptr + t * W + own, lw._feature(v, vp, cs, sn, low, key, isxl))
        if not LAST:
            tl.store(state_ptr + seg * HD + cc, x.to(tl.int32, bitcast=True))
            tl.store(pos_ptr + seg, rp)

    @triton.jit
    def _probe_inverse_kernel(inverse_ptr, output_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(inverse_ptr + offs, mask=offs < N, other=0)
        tl.store(output_ptr + offs, values, mask=offs < N)
