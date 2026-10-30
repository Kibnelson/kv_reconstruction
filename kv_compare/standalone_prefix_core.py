"""Fixed-wire warp-ballot decoder refill prefixes, on the exact packed inverse.

One program has 128 logical channels on one physical SM86 warp. Reshape and
split explicitly expose four vectors of 32 channels. A pack=1 collective
consumes one bit from each vector per thread; popcounts and ordered group
carries reproduce the original 128-channel inclusive prefix. The compiled
32-element lane layout is checked before executing a new production variant.
This is an execution experiment: no packet, symbol, probability or prediction
changes. Actual CUDA correctness and performance require the NUC run.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import re

import numpy as np
import torch

from .team_onestep import gpu_coder as gc
from .team_onestep import layerwise as lw
from . import standalone_inverse_core as inverse

triton, tl = gc.triton, gc.tl
fu = lw.fu


def _numpy_slots(mask_words, bases):
    """Independent ballot/popcount mirror; words shape [rows,4], bases [rows]."""
    words = np.asarray(mask_words, dtype=np.uint32)
    lane = np.arange(32, dtype=np.uint32)
    prefix_masks = np.uint32(0xffffffff) >> (np.uint32(31) - lane)
    inclusive = np.bitwise_count(words[:, :, None] & prefix_masks).astype(np.int32)
    counts = np.bitwise_count(words).astype(np.int32)
    carries = np.cumsum(counts, axis=1, dtype=np.int32) - counts
    return (bases[:, None] + (inclusive + carries[:, :, None]).reshape(-1,128) - 1,
            counts.sum(axis=1, dtype=np.int32))


def prefix_cases(random_rows=1024):
    """Structured boundaries and seeded independent full-128 masks; packed words."""
    cc = np.arange(128)
    masks = [np.zeros(128, dtype=np.int32), np.ones(128, dtype=np.int32)]
    for index in range(128):
        masks += [(cc == index).astype(np.int32), (cc != index).astype(np.int32)]
    for end in range(129):
        masks += [(cc < end).astype(np.int32), (cc >= end).astype(np.int32)]
    for period in (2,3,4,7,8,16,31,32,33,64,127):
        masks += [((cc % period) == offset).astype(np.int32) for offset in (0,1)]
    for left, right in ((0,127),(30,31),(31,32),(63,64),(95,96),(126,127)):
        masks.append(((cc == left) | (cc == right)).astype(np.int32))
    rng = np.random.default_rng(0xC0DEC0DE)
    masks.extend(rng.integers(0,2,size=(int(random_rows),128),dtype=np.int32))
    values = np.stack(masks).reshape(-1,4,32).astype(np.uint64)
    words = np.sum(values << np.arange(32,dtype=np.uint64),axis=2,dtype=np.uint64).astype(np.uint32)
    return words


@functools.lru_cache(maxsize=1)
def _prefix_proof_cached():
    # Exhaust both independent 16-bit halves of a warp. Popcount is additive
    # across disjoint halves; do not call this exhaustive over 2**128 masks.
    checked = 0
    digest = hashlib.sha256()
    for shift in (0,16):
        for begin in range(0,65536,4096):
            words = np.zeros((4096,4),dtype=np.uint32)
            words[:,0] = np.arange(begin,begin+4096,dtype=np.uint32) << np.uint32(shift)
            words[:,1] = ~words[:,0]
            words[:,2] = np.uint32(0xffffffff)
            bases = np.full(len(words),1024,dtype=np.int32)
            actual, total = _numpy_slots(words,bases)
            bits = ((words[:,:,None] >> np.arange(32,dtype=np.uint32)) & 1).reshape(-1,128).astype(np.int32)
            expected = bases[:,None] + np.cumsum(bits,axis=1,dtype=np.int32) - 1
            if not np.array_equal(actual,expected) or not np.array_equal(total,bits.sum(axis=1)):
                raise AssertionError('host ballot algebra differs from exact full-row prefix')
            digest.update(actual.astype('<i4').tobytes()); checked += len(words)
    words = prefix_cases()
    bases = np.asarray([0,1,31,1024,1<<30,(1<<31)-129],dtype=np.int32)[np.arange(len(words))%6]
    actual,total = _numpy_slots(words,bases)
    bits = ((words[:,:,None] >> np.arange(32,dtype=np.uint32)) & 1).reshape(-1,128).astype(np.int32)
    if not np.array_equal(actual,bases[:,None]+np.cumsum(bits,axis=1,dtype=np.int32)-1):
        raise AssertionError('structured/random host prefix differs')
    if not np.array_equal(total,bits.sum(axis=1)):
        raise AssertionError('structured/random total differs')
    return json.dumps(dict(format='standalone_prefix_warp_v1',verified=True,
        channels=128,logical_groups=4,lanes_per_warp=32,warps=1,
        required_triton='3.6.0',required_cuda_capability=[8,6],required_segment=0,
        host_halfwarp_masks_checked=checked,host_fullrow_masks_checked=len(words),
        host_prefix_sha256=digest.hexdigest(),additional_shared_device_bytes=0,
        gpu_execution_verified=False,
        scope='CPU algebra proof only; not exhaustive over all 128-bit masks and not CUDA layout proof'),sort_keys=True)


def prefix_proof():
    return json.loads(_prefix_proof_cached())


def validate_compiled_prefix(module, expected_calls=2):
    """Fail closed on the actual optimized production TTGIR lane distribution.

    inline_asm pack does not specify element/thread mapping. We accept only
    explicit 32-element identity-lane operands and outputs; no register, warp
    or CTA distribution is allowed for those vectors. This check must run on
    each compiled production variant before its first launch.
    """
    if triton.__version__ != '3.6.0':
        raise RuntimeError('warp prefix requires pinned Triton 3.6.0')
    ttgir,ptx = module.asm['ttgir'],module.asm['ptx']
    for required in ('"ttg.num-ctas" = 1 : i32','"ttg.num-warps" = 1 : i32',
                     'ttg.target = "cuda:86"','"ttg.threads-per-warp" = 32 : i32'):
        if required not in ttgir:
            raise RuntimeError('unsupported prefix compiler geometry: '+required)
    if '.target sm_86' not in ptx or not re.search(r'\.reqntid\s+32(?:\s|,)',ptx):
        raise RuntimeError('prefix PTX must launch exactly one SM86 warp')
    layouts = dict(re.findall(r'^(#\w+) = (.+)$',ttgir,flags=re.M))
    sites = [line for line in ttgir.splitlines()
             if 'tt.elementwise_inline_asm' in line and 'kv_prefix_warp32_v1' in line]
    if len(sites) != expected_calls:
        raise RuntimeError('unexpected production prefix call count')
    accepted = []
    for line in sites:
        if 'packed_element = 1 : i32, pure = false' not in line:
            raise RuntimeError('prefix collective must be pack=1 and non-pure')
        # Every one of five inputs and five outputs must have the same
        # explicit identity-lane tensor32 layout. Regex does not rely on SSA names.
        types = re.findall(r'tensor<(\d+)xi32, (#[A-Za-z0-9_]+)>',line)
        if len(types) != 10 or any(size != '32' for size,_ in types):
            raise RuntimeError('prefix collective requires ten explicit tensor32 types')
        for _, alias in types:
            layout = layouts.get(alias,'').replace(' ','')
            identity = ('#ttg.linear<{register=[],lane=[[1],[2],[4],[8],[16]],warp=[],block=[]}>'
                        == layout)
            blocked = ('#ttg.blocked<{sizePerThread=[1],threadsPerWarp=[32],warpsPerCTA=[1],order=[0]}>'
                       == layout)
            if not (identity or blocked):
                raise RuntimeError('unsupported physical prefix lane layout: '+layout)
            accepted.append(layouts[alias])
    if ptx.count('vote.ballot.sync.b32') != 4*expected_calls:
        raise RuntimeError('prefix PTX must contain four full-warp ballots per call')
    return dict(verified=True,target='cuda sm86 warp32',prefix_call_sites=len(sites),
        ballots=ptx.count('vote.ballot.sync.b32'),pack=1,pure=False,
        layouts=sorted(set(accepted)),
        ttgir_sha256=hashlib.sha256(ttgir.encode()).hexdigest(),
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
        compiled_hash=str(module.hash))


if triton is not None:
    @triton.jit
    def _slots_warp(e, pos, SEG: tl.constexpr):
        tl.static_assert(SEG == 0, 'warp prefix only supports one 128-channel segment')
        tl.static_assert(e.shape[0] == 128, 'warp prefix requires 128 channels')
        # Expose four independent logical32 vectors, avoiding pack=4's
        # unspecified grouping. Their actual identity-lane layout is checked.
        split = tl.permute(tl.reshape(e,[2,2,32]),(2,0,1))
        e02,e13 = tl.split(split)
        e0,e2 = tl.split(e02)
        e1,e3 = tl.split(e13)
        lane = tl.arange(0,32)
        p0,p1,p2,p3,total32 = tl.inline_asm_elementwise(
            """{
            // kv_prefix_warp32_v1
            .reg .pred a,b,c,d;
            .reg .b32 ba,bb,bc,bd,ca,cb,cc,cd,m,sh,ma,mb,mc,md;
            setp.ne.u32 a, $5, 0;
            setp.ne.u32 b, $6, 0;
            setp.ne.u32 c, $7, 0;
            setp.ne.u32 d, $8, 0;
            vote.ballot.sync.b32 ba, a, 0xffffffff;
            vote.ballot.sync.b32 bb, b, 0xffffffff;
            vote.ballot.sync.b32 bc, c, 0xffffffff;
            vote.ballot.sync.b32 bd, d, 0xffffffff;
            sub.u32 sh,31,$9;
            mov.u32 m,0xffffffff;
            shr.u32 m,m,sh;
            and.b32 ma,ba,m;
            and.b32 mb,bb,m;
            and.b32 mc,bc,m;
            and.b32 md,bd,m;
            popc.b32 $0,ma;
            popc.b32 $1,mb;
            popc.b32 $2,mc;
            popc.b32 $3,md;
            popc.b32 ca,ba;
            popc.b32 cb,bb;
            popc.b32 cc,bc;
            popc.b32 cd,bd;
            add.u32 $1,$1,ca;
            add.u32 cb,cb,ca;
            add.u32 $2,$2,cb;
            add.u32 cc,cc,cb;
            add.u32 $3,$3,cc;
            add.u32 $4,cc,cd;
            }""",
            '=r,=r,=r,=r,=r,r,r,r,r,r',[e0,e1,e2,e3,lane],
            dtype=(tl.int32,tl.int32,tl.int32,tl.int32,tl.int32),is_pure=False,pack=1)
        joined = tl.join(tl.join(p0,p2),tl.join(p1,p3))
        inclusive = tl.reshape(tl.permute(joined,(1,2,0)),[128])
        total = tl.sum(tl.gather(total32,tl.full([1],0,tl.int32),axis=0),0)
        return pos + inclusive - 1, total

    @triton.jit
    def _take_prefix(x, cum, f, active, buf_ptr, rp, SEG: tl.constexpr):
        x = tl.where(active, f.to(tl.uint32) * (x >> 16) + (x & 0xFFFF) - cum.to(tl.uint32), x)
        need = active & (x < 65536)
        pos, total = _slots_warp(need.to(tl.int32), rp, SEG)
        w = (tl.load(buf_ptr + pos, mask=need, other=0).to(tl.int32) & 0xFFFF).to(tl.uint32)
        x = tl.where(need, (x << 16) | w, x)
        return x, rp + total

    @triton.jit
    def _prefix_dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, coarse_ptr, sparse_ptr, LUT_SHIFT: tl.constexpr,
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
        x, rp = _take_prefix(x, (p >> 16) & 0xFFFF, p & 0xFFFF, valid, buf_ptr, rp, SEG)
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
        x, rp = _take_prefix(x, cum2, f2, two, buf_ptr, rp, SEG)
        w = tl.where(esc, slot2, gc._word(o))
        return x, rp, w

    @triton.jit
    def _prefix_ldec_kernel(buf_ptr, start_ptr, state_ptr, pos_ptr, out_ptr, acc_ptr, xout_ptr, sa_ptr, b_ptr, sp_ptr,
                     inv_ptr, isx_ptr, cos_ptr, sin_ptr, ssum_in_ptr, ssum_out_ptr, smult_ptr, cumf_ptr, lut_ptr,
                     tail_ptr, coarse_ptr, sparse_ptr, t_base, T, W, G, HG, PRIOR_SCALE,
                     FIRST: tl.constexpr, LAST: tl.constexpr, TB: tl.constexpr, H2: tl.constexpr,
                     LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr,
                     SEARCH: tl.constexpr, FEAT: tl.constexpr, LEAN: tl.constexpr, ONES: tl.constexpr, PACKED: tl.constexpr):
        """Decode one layer of one token block x head (pointers are the layer's: out_ptr, sa_ptr, ...).
        FEAT: make the next layer's features here (else a separate pass does); LEAN: reload the per-channel
        constants every token instead of keeping them in registers; FIRST: the stream states come from the stream
        (the first layer this codec decodes); ONES: prior 1 (no previous layer of ours to take surprise from)."""
        tl.static_assert(H2 == 64, "warp prefix requires HD128")
        tl.static_assert(PACKED, "prefix experiment retains packed inverse")
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
            x, rp, w = _prefix_dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, coarse_ptr, sparse_ptr, LUT_SHIFT, LUT_W,
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
    def _probe_prefix_kernel(mask_ptr, base_ptr, output_ptr, reference_ptr, STEPS: tl.constexpr):
        row = tl.program_id(0)
        cc = tl.arange(0,128)
        base = tl.load(base_ptr + row)
        for step in range(0,STEPS):
            word = tl.load(mask_ptr + (row * STEPS + step) * 4 + cc // 32).to(tl.uint32)
            e = ((word >> (cc & 31)) & 1).to(tl.int32)
            actual, count = _slots_warp(e,base,0)
            expected, total = gc._slots(e,base,0)
            offset = (row * STEPS + step) * 129
            tl.store(output_ptr + offset + cc,actual)
            tl.store(reference_ptr + offset + cc,expected)
            tl.store(output_ptr + offset + 128,count)
            tl.store(reference_ptr + offset + 128,total)


    @triton.jit
    def _probe_take_kernel(x_ptr, cum_ptr, frequency_ptr, active_ptr, buffer_ptr, output_ptr, reference_ptr):
        row = tl.program_id(0)
        cc = tl.arange(0,128)
        offset = row * 128 + cc
        x = tl.load(x_ptr + offset).to(tl.uint32)
        cumulative = tl.load(cum_ptr + offset)
        frequency = tl.load(frequency_ptr + offset)
        active = tl.load(active_ptr + offset) != 0
        rp = row * 128
        actual,position = _take_prefix(x,cumulative,frequency,active,buffer_ptr,rp,0)
        expected,end = gc._take(x,cumulative,frequency,active,buffer_ptr,rp,0)
        out = row * 129
        tl.store(output_ptr + out + cc,actual.to(tl.int32,bitcast=True))
        tl.store(reference_ptr + out + cc,expected.to(tl.int32,bitcast=True))
        tl.store(output_ptr + out + 128,position)
        tl.store(reference_ptr + out + 128,end)


def _run_prefix_probe(device, words, *, steps=3):
    """Actual-CUDA gate only. All tensor scratch is freed before returning."""
    words = np.asarray(words,dtype=np.uint32).reshape(-1,4)
    mask_steps = np.stack([np.roll(words,shift,axis=0) ^ np.uint32(
        (0x9e3779b9*shift)&0xffffffff) for shift in range(steps)],axis=1)
    n = len(words)
    bases = np.asarray([0,1,31,1024,1<<30,(1<<31)-129],dtype=np.int32)[np.arange(n)%6]
    masks_gpu = torch.from_numpy(mask_steps.copy().view(np.int32)).to(device)
    bases_gpu = torch.from_numpy(bases).to(device)
    output = torch.empty((n,steps,129),dtype=torch.int32,device=device)
    reference = torch.empty_like(output)
    module = _probe_prefix_kernel.warmup(masks_gpu,bases_gpu,output,reference,
        STEPS=steps,grid=(n,),num_warps=1,num_stages=3,enable_fp_fusion=False)
    layout = validate_compiled_prefix(module,expected_calls=1)
    _probe_prefix_kernel[(n,)](masks_gpu,bases_gpu,output,reference,
        STEPS=steps,num_warps=1,num_stages=3,enable_fp_fusion=False)
    torch.cuda.synchronize(device)
    host = output.cpu().numpy()
    original = reference.cpu().numpy()
    if not np.array_equal(host,original):
        raise AssertionError('CUDA warp prefix differs from original 128-channel prefix')
    flattened = mask_steps.reshape(-1,4)
    cpu_pos,cpu_total = _numpy_slots(flattened,np.repeat(bases,steps))
    if (not np.array_equal(host[:,:,:128].reshape(-1,128),cpu_pos)
            or not np.array_equal(host[:,:,128].reshape(-1),cpu_total)):
        raise AssertionError('CUDA warp prefix differs from independent exact host algebra')
    receipt = dict(verified=True,gpu_execution_verified=True,mask_rows=n,
        changing_steps=steps,prefix_calls=n*steps,all_channel_positions_compared=n*steps*128,
        output_sha256=hashlib.sha256(host.astype('<i4').tobytes()).hexdigest(),
        compiler_layout=layout)
    del masks_gpu,bases_gpu,output,reference
    return receipt


@functools.lru_cache(maxsize=None)
def _runtime_prefix_proof_cached(index):
    if (os.environ.get('TRITON_INTERPRET') == '1' or not torch.cuda.is_available()
            or triton.__version__ != '3.6.0'):
        raise RuntimeError('warp prefix requires actual CUDA with Triton 3.6.0')
    device = torch.device('cuda',index)
    if torch.cuda.get_device_capability(device) != (8,6):
        raise RuntimeError('warp prefix experiment requires SM86')
    with torch.cuda.device(device):
        receipt = _run_prefix_probe(device,prefix_cases(),steps=3)
    receipt.update(device=str(device),device_name=torch.cuda.get_device_name(device),
        capability=list(torch.cuda.get_device_capability(device)),triton=triton.__version__,
        additional_shared_device_bytes=0,
        scope='Actual CUDA structured/random prefix algebra and compiled probe layout; production layout separately checked before launch')
    return json.dumps(receipt,sort_keys=True)


def runtime_prefix_proof(device):
    dev = torch.device(device)
    if dev.type != 'cuda':
        raise RuntimeError('runtime prefix proof requires actual CUDA')
    index = torch.cuda.current_device() if dev.index is None else dev.index
    return json.loads(_runtime_prefix_proof_cached(index))
