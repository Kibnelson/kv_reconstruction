"""Exact same-CDF decoder cores: compact packed inverse or masked search.

Only stage-1 CDF inversion changes. Prediction, rANS updates, BF16 intervals,
word-in-slice coding, escapes and all original source files remain unchanged.
The packed coarse table stores a base and one local cutoff for simple buckets;
36 complex buckets use tiny exact 64-slot tables. The separate masked arm
keeps the original binary search and avoids loads for already-resolved lanes.
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


def _binary_inverse(tables, c, slots, *, masked):
    """Independent NumPy mirror of the original loop and its masked-load arm."""
    row = tables['lut'][c].numpy().astype(np.int64)
    packed = tables['cumf'].numpy().astype(np.int64)
    lp = row[slots >> (16 - gc.LUT_BITS)]
    low, high = lp & 0xffff, (lp & 0xffff) + (lp >> 16)
    offset = int(row[gc.LUT_W - 3]) >> 10
    steps = loads = 0
    while np.max(high - low) > 1:
        unresolved = (high - low) > 1
        middle = (low + high) // 2
        if masked:
            loaded = np.zeros_like(middle)
            loaded[unresolved] = packed[offset + middle[unresolved]]
            loads += int(unresolved.sum())
        else:
            loaded = packed[offset + middle]
            loads += len(slots)
        go = ((loaded >> 16) & 0xffff) <= slots
        low, high = np.where(go, middle, low), np.where(go, high, middle)
        steps += 1
        if steps > 16:
            raise AssertionError('fixed CDF search failed to converge')
    return low, dict(iterations=steps, simulated_loads=loads)


def _packed_inverse(coarse, sparse, c, slots):
    entries = coarse[c, slots >> 6]
    rare = (entries >> 31) != 0
    local = slots & 63
    fast = (entries & 1023) + (local >= ((entries >> 10) & 127))
    address = np.where(rare, entries & 0x7fffffff, 0) + local
    return np.where(rare, sparse[address], fast)


@functools.lru_cache(maxsize=1)
def _cached_tables():
    if (gc.CLASSES, gc.M, gc.LUT_BITS) != (10, 65536, 10):
        raise ValueError('fixed 10-class 16-bit CDF geometry required')
    tables = gc.slice_tables('cpu')
    slots = np.arange(gc.M, dtype=np.int64)
    expected = np.empty((gc.CLASSES, gc.M), dtype='<u2')
    coarse = np.empty((gc.CLASSES, 1024), dtype='<u4')
    sparse_values, exception_counts, searches = [], [], []
    for c in range(gc.CLASSES):
        off, count = int(tables['off'][c]), int(tables['K'][c]) + 2
        cumulative = tables['cum'][off:off + count].numpy().astype(np.int64)
        if cumulative[0] != 0 or cumulative[-1] != gc.M or np.any(np.diff(cumulative) < 1):
            raise ValueError('strictly increasing original fixed CDF required')
        truth = np.searchsorted(cumulative, slots, side='right') - 1
        if np.any(truth < 0) or np.any(truth >= 1024):
            raise AssertionError('fixed slice symbol does not fit 10-bit base')
        expected[c] = truth
        rare_count = 0
        for bucket in range(1024):
            values = truth[bucket * 64:(bucket + 1) * 64]
            base = int(values[0]); span = int(values[-1]) - base + 1
            if span <= 2:
                cutoff = int(np.flatnonzero(values != base)[0]) if span == 2 else 64
                coarse[c, bucket] = base | (cutoff << 10)
            else:
                offset = len(sparse_values)
                if offset + 64 > (1 << 31):
                    raise AssertionError('sparse offset exceeds packed representation')
                coarse[c, bucket] = (1 << 31) | offset
                sparse_values.extend(values)
                rare_count += 1
        binary, old_stats = _binary_inverse(tables, c, slots, masked=False)
        masked, masked_stats = _binary_inverse(tables, c, slots, masked=True)
        reference = gc.stage1_reference(tables, c, slots)
        if any(not np.array_equal(truth, other) for other in (binary, masked, reference)):
            raise AssertionError('fixed inverse differs from original algorithms')
        exception_counts.append(rare_count)
        searches.append(dict(width_class=c, original=old_stats, masked=masked_stats))
    sparse = np.asarray(sparse_values, dtype='<u2')
    for c in range(gc.CLASSES):
        if not np.array_equal(_packed_inverse(coarse, sparse, c, slots), expected[c]):
            raise AssertionError('packed inverse differs from original CDF')
    if exception_counts != [16, 6, 4, 2, 2, 2, 2, 1, 1, 0]:
        raise AssertionError('original fixed CDF exception geometry changed')
    coarse_payload, sparse_payload = coarse.tobytes(), sparse.tobytes()
    digest = hashlib.sha256(expected.tobytes()).hexdigest()
    source_root = Path(__file__).resolve().parent
    proof = dict(format='standalone_inverse_core_v1', verified=True,
        classes=gc.CLASSES, slots_per_class=gc.M, checked_slots=gc.CLASSES * gc.M,
        coarse_dtype='int32', coarse_shape=[10, 1024], coarse_bytes=len(coarse_payload),
        sparse_dtype='uint16', sparse_shape=list(sparse.shape), sparse_bytes=len(sparse_payload),
        table_bytes=len(coarse_payload) + len(sparse_payload),
        coarse_sha256=hashlib.sha256(coarse_payload).hexdigest(),
        sparse_sha256=hashlib.sha256(sparse_payload).hexdigest(),
        complex_buckets=sum(exception_counts), exception_counts_by_class=exception_counts,
        reference_inverse_sha256=digest, coarse_search_sha256=digest,
        masked_search_sha256=digest, packed_inverse_sha256=digest,
        original_fixed_tables_bytes=sum(v.numel() * v.element_size() for v in tables.values()),
        fixed_table_sha256={name:_tensor_receipt(value) for name,value in sorted(tables.items())},
        searches=searches,
        sources={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                 for name in ('team_onestep/gpu_coder.py','team_onestep/layerwise.py')},
        scope='Exhaustive fixed stage-1 inverse only; simulated loads are algorithm counts, not GPU timings')
    return coarse_payload, sparse_payload, json.dumps(proof, sort_keys=True, separators=(',', ':'))


def inverse_proof():
    """Return an independently owned copy of both exhaustive inverse proofs."""
    return json.loads(_cached_tables()[2])


def build_inverse_tables(device='cpu'):
    """Build independently owned contiguous INT32 coarse and UINT16 sparse tables."""
    coarse_bytes, sparse_bytes, proof_json = _cached_tables()
    coarse = np.frombuffer(coarse_bytes, dtype='<u4').copy().view('<i4').reshape(10, 1024)
    sparse = np.frombuffer(sparse_bytes, dtype='<u2').copy()
    return torch.from_numpy(coarse).to(device), torch.from_numpy(sparse).to(device), json.loads(proof_json)


if triton is not None:
    @triton.jit
    def _dec_value_core(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, coarse_ptr, sparse_ptr, LUT_SHIFT: tl.constexpr,
                   LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr, SEG: tl.constexpr,
                   SEARCH: tl.constexpr, PACKED: tl.constexpr):
        """Decode one word per lane; returns (state, read position, word)."""
        c = gc._width_class(mu, s)
        row = lut_ptr + c * LUT_W
        kop = tl.load(row + (LUT_W - 3))
        K = kop & 1023
        offf = kop >> 10
        slot = (x & 0xFFFF).to(tl.int32)
        if PACKED:
            entry = tl.load(coarse_ptr + c * 1024 + (slot >> 6)).to(tl.uint32)
            exceptional = (entry & 0x80000000) != 0
            local = slot & 63
            base = (entry & 1023).to(tl.int32)
            cutoff = ((entry >> 10) & 127).to(tl.int32)
            fast = base + (local >= cutoff).to(tl.int32)
            # Form an in-bounds address even for masked nonexception lanes.
            sparse_offset = tl.where(exceptional, entry & 0x7fffffff, 0).to(tl.int32)
            rare = tl.load(sparse_ptr + sparse_offset + local,
                           mask=exceptional, other=0).to(tl.int32)
            k = tl.where(exceptional, rare, fast)
        else:
            lp = tl.load(row + (slot >> LUT_SHIFT))
            if SEARCH:                       # stage 1: coarse table, then binary search for the largest k, cum[k] <= slot
                klo = lp & 0xFFFF
                khi = klo + (lp >> 16)
                while tl.max(khi - klo, 0) > 1:
                    mid = (klo + khi) // 2
                    go = ((tl.load(cumf_ptr + offf + mid, mask=(khi - klo) > 1, other=0) >> 16) & 0xFFFF) <= slot
                    klo = tl.where(go, mid, klo)
                    khi = tl.where(go, khi, mid)
                k = klo
            else:                            # stage 1 without a search: tail slots from their table; a central slot is
                kc0 = tl.load(row + (LUT_W - 2))     # the bucket's first central symbol or the next (see slice_tables)
                kc1 = tl.load(row + (LUT_W - 1))
                lo = kc0 & 0xFFFF
                hm1 = (kc0 >> 16) & 0xFFFF
                in_lo = slot < lo
                tail = in_lo | (slot > hm1)
                kt = tl.load(tail_ptr + (kc1 >> 10) + tl.where(in_lo, slot, slot - hm1 - 1 + lo), mask=tail, other=0)
                kb = tl.maximum(lp & 0xFFFF, kc1 & 1023)
                nxt = (tl.load(cumf_ptr + offf + kb + 1, mask=~tail, other=0) >> 16) & 0xFFFF
                k = tl.where(tail, kt, kb + (nxt <= slot).to(tl.int32))
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
    def _core_ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, coarse_ptr, sparse_ptr, t_base, T, W, G, HG, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, FEAT: tl.constexpr, LEAN: tl.constexpr, ONES: tl.constexpr, PACKED: tl.constexpr):
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
            x, rp, w = _dec_value_core(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, coarse_ptr, sparse_ptr, LUT_SHIFT, LUT_W,
                                     REFINE, UNIFORM, 0, SEARCH, PACKED)
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
    def _probe_inverse_kernel(cumf_ptr, lut_ptr, coarse_ptr, sparse_ptr, output_ptr,
                              N: tl.constexpr, BLOCK: tl.constexpr, PACKED: tl.constexpr, MIXED: tl.constexpr = False):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = index < N
        if MIXED:
            c = tl.where(active, index % 10, 0)
            slot = tl.where(active, ((index // 10) * 4051 + 65535) & 65535, 0)
        else:
            c = tl.where(active, index // 65536, 0)
            slot = tl.where(active, index & 65535, 0)
        if PACKED:
            entry = tl.load(coarse_ptr + c * 1024 + (slot >> 6)).to(tl.uint32)
            rare = (entry & 0x80000000) != 0
            local = slot & 63
            fast = (entry & 1023).to(tl.int32) + (local >= ((entry >> 10) & 127)).to(tl.int32)
            offset = tl.where(rare, entry & 0x7fffffff, 0).to(tl.int32)
            symbol = tl.load(sparse_ptr + offset + local, mask=rare, other=0).to(tl.int32)
            result = tl.where(rare, symbol, fast)
        else:
            row = lut_ptr + c * 1028
            offset = tl.load(row + 1025) >> 10
            lp = tl.load(row + (slot >> 6))
            low = lp & 0xffff
            high = low + (lp >> 16)
            while tl.max(high - low, 0) > 1:
                middle = (low + high) // 2
                packed = tl.load(cumf_ptr + offset + middle, mask=(high - low) > 1, other=0)
                go = ((packed >> 16) & 0xffff) <= slot
                low = tl.where(go, middle, low)
                high = tl.where(go, high, middle)
            result = low
        tl.store(output_ptr + index, result.to(tl.uint16), mask=active)
