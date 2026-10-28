"""Value-slice coder for BF16 words on the GPU (Triton): lossless, given a prediction and a spread per value.

The word's value v is coded against Logistic(mu, s) in two symbols, with 16-bit rANS on many interleaved streams:

  1. Slice. With the width class c = clamp(floor(log2(|mu| + 2s)) - floor(log2 s) - 3, 0, 9), slices of width
     d = 2^(c-3) spreads cover t = (v - mu)/s on [-24, 24): slice k has bounds lo = mu + t0*s, hi = mu + t1*s
     with t0 = -24 + k*d, t1 = min(t0 + d, 24), and holds the words whose ordered index o satisfies
     first(lo) <= o < first(hi), where first(x) is the ordered index of x rounded up to BF16 (bit operations).
     The slice symbol uses one static table per class (logistic mass of the slice, every symbol at least
     1/65536); an escape symbol (outside the range, Inf or NaN) is followed by the raw 16-bit word.
  2. Word in the slice, n = first(hi) - first(lo) words, j = o - first(lo):
       mode "uniform": equal shares, q = floor(65536 / n) slots each, the last word also takes the remainder;
                       decoded as j = min(slot // q, n - 1)
       mode "exact":   shares follow the words' rounding intervals [BD(o), BD(o + 1)), every word at least one
                       slot: C(o) = floor((BD(o) - BD(first)) * R) + (o - first), R = (65536 - n) / width;
                       decoded from an estimate refined twice, then a bounded binary search
     A slice with one word needs no second symbol.

Encoder and decoder run the same integer and float32 operations (add, multiply, correctly rounded division; the
kernels are compiled without fused multiply-add), so they agree bit for bit on one GPU type. Quotients computed
with approximate float division are corrected to the exact integer result. Shipping the slice tables as
constants (computed once here, in float64) makes the stream portable across GPUs; that portability is not
tested here.

Streams: lane j of a Triton program codes values j, j + S, j + 2S, ... (S lanes in all), so loads are coalesced.
A program's lanes share one output segment: at each symbol the lanes that emit or read a 16-bit word take
consecutive positions (prefix sum over the lanes), so the bitstream is read and written contiguously.
"""
from __future__ import annotations

import math

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


def _interpreter_shim():
    """TRITON_INTERPRET=1 (CPU testing): Triton's interpreter converts its 1-element scalar arrays with int() and
    bool(), which newer NumPy rejects for arrays with ndim > 0. Convert through .item() instead."""
    import os
    if triton is None or os.environ.get("TRITON_INTERPRET") != "1":
        return
    try:
        import triton.runtime.interpreter as ti
    except ImportError:  # pragma: no cover
        return
    if getattr(ti, "_kv_compare_shim", False):
        return
    original = ti._patch_lang_tensor

    def patched(tensor, scope):
        original(tensor, scope)
        scalar = lambda d: np.asarray(d).reshape(-1)[0].item()
        scope.set_attr(tensor, "__index__", lambda self: int(scalar(self.handle.data)))
        scope.set_attr(tensor, "__bool__", lambda self: bool(scalar(self.handle.data)) if np.asarray(self.handle.data).size == 1 else True)
    ti._patch_lang_tensor = patched
    ti._kv_compare_shim = True


_interpreter_shim()

M = 1 << 16
CLASSES = 10
RANGE = 24.0
LUT_BITS = 10                    # coarse slot -> symbol table: 2^10 buckets per class
LUT_N = (1 << LUT_BITS) + 1
LUT_W = LUT_N + 3                # a lut row: the buckets' entries, then the class's ko, kc0, kc1 (one row address)
REFINE = 2                       # exact mode: refinements of the word estimate before its binary search


def slice_tables(device):
    """Static stage-1 tables.
    cum [sum(K_c + 2)] int32: per class, cumulative counts of the K_c slices, the escape, then 65536 (mirror);
    off [10], K [10] int32 (mirror); and the packed tables the kernels read:
    cumf: per class, for symbols 0..K: (cum << 16) | frequency;  ko [10]: (start in cumf << 10) | K;
    lut [10, LUT_W]: (first symbol that can hold bucket b's slots) | (number of candidates << 16) for the
    2^LUT_BITS + 1 buckets, then the class's ko, kc0, kc1 (so the decoder reads its class constants from one row).
    Decoder without a search: the symbols with at least 2^(16 - LUT_BITS) slots (the bucket size) form one central
    run k_lo..k_hi (the slice masses are unimodal), so a central bucket meets at most two of them and one
    comparison picks the symbol; the slots of the tails (below cum[k_lo], from cum[k_hi + 1]) have a direct table:
    kc [10, 2]: (cum[k_lo] | (cum[k_hi + 1] - 1) << 16, k_lo | start in tail << 10);  tail: symbol per tail slot."""
    cums, off, Ks, cumfs, kos, luts, kcs, tails = [], [], [], [], [], [], [], []
    pos = posf = post = 0
    bucket = 1 << (16 - LUT_BITS)
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    for c in range(CLASSES):
        d = 2.0 ** (c - 3)
        K = math.ceil(2 * RANGE / d)
        t = [-RANGE + k * d for k in range(K)] + [RANGE]
        P = [sig(t[k + 1]) - sig(t[k]) for k in range(K)]
        f = [1 + math.floor(p * (M - K - 1)) for p in P] + [1]
        f[int(np.argmax(P))] += M - sum(f)
        cum = np.concatenate([[0], np.cumsum(f)]).astype(np.int64)
        assert cum[-1] == M and min(f) >= 1 and max(f) < M and K < 1024
        sym = [int(np.searchsorted(cum, b << (16 - LUT_BITS), side="right") - 1) for b in range(1 << LUT_BITS)] + [K]
        luts.append([sym[b] | ((sym[b + 1] + 1 - sym[b]) << 16) for b in range(1 << LUT_BITS)] + [K | (1 << 16)])
        cumfs.append([(int(cum[k]) << 16) | int(cum[k + 1] - cum[k]) for k in range(K + 1)])
        kos.append((posf << 10) | K)
        big = [k for k in range(K + 1) if f[k] >= bucket]
        k_lo, k_hi = big[0], big[-1]
        assert big == list(range(k_lo, k_hi + 1)) and k_hi < K, "central symbols must be one run"
        lo, hi = int(cum[k_lo]), int(cum[k_hi + 1])
        slots = np.concatenate([np.arange(0, lo), np.arange(hi, M)])
        tails.append(np.searchsorted(cum, slots, side="right") - 1)
        kcs += [lo | ((hi - 1) << 16), k_lo | (post << 10)]
        luts[-1] += [kos[-1]] + kcs[-2:]
        assert post < (1 << 21)
        post += len(slots)
        cums.append(cum)
        off.append(pos)
        Ks.append(K)
        pos += len(cum)
        posf += K + 1
    as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64).astype(np.uint32).view(np.int32), device=device)
    return {"cum": as_t(np.concatenate(cums)), "off": as_t(off), "K": as_t(Ks),
            "cumf": as_t(np.concatenate([np.asarray(x, dtype=np.int64) for x in cumfs])), "ko": as_t(kos),
            "lut": as_t(np.asarray(luts, dtype=np.int64)), "kc": as_t(kcs), "tail": as_t(np.concatenate(tails))}


def stage1_reference(tables, c, slot):
    """NumPy mirror of the decoder's search-free stage 1 (for tests): symbol of each slot of class c."""
    g = lambda k: {n: np.asarray(v.cpu()).view(np.uint32).astype(np.int64) for n, v in tables.items()}[k]
    kc, lut, cumf, ko, tail = g("kc"), g("lut").reshape(CLASSES, LUT_W), g("cumf"), g("ko"), g("tail")
    assert np.array_equal(lut[:, LUT_W - 3], ko) and np.array_equal(lut[:, LUT_W - 2:].reshape(-1), kc)
    lo, hm1 = kc[2 * c] & 0xFFFF, (kc[2 * c] >> 16) & 0xFFFF
    klo, toff = kc[2 * c + 1] & 1023, kc[2 * c + 1] >> 10
    offf = ko[c] >> 10
    in_lo, in_hi = slot < lo, slot > hm1
    ti = np.where(in_lo, slot, slot - (hm1 + 1) + lo)
    kt = tail[toff + np.where(in_lo | in_hi, ti, 0)]
    kb = np.maximum(lut[c, slot >> (16 - LUT_BITS)] & 0xFFFF, klo)
    nxt = (cumf[offf + kb + 1] >> 16) & 0xFFFF
    return np.where(in_lo | in_hi, kt, kb + (nxt <= slot))


if triton is not None:
    import os as _os
    # GPU: approximate reciprocals (one MUFU.RCP) where the caller corrects the integer result exactly or only needs
    # an estimate; the interpreter (CPU tests) cannot run PTX and divides instead (same corrected results).
    _ASM = tl.constexpr(_os.environ.get("TRITON_INTERPRET") != "1")

    @triton.jit
    def _rcp(y):
        """Approximate 1/y (float32, within an ulp or two)."""
        if _ASM:
            return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [y], dtype=tl.float32, is_pure=True,
                                             pack=1)
        else:
            return 1.0 / y

    @triton.jit
    def _word_val(w):
        """BF16 word (int32, 0..65535) -> float32 value."""
        return (w.to(tl.uint32) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _ord(w):
        return tl.where((w & 0x8000) != 0, (~w) & 0xFFFF, w ^ 0x8000)

    @triton.jit
    def _word(o):
        return tl.where((o & 0x8000) != 0, o ^ 0x8000, (~o) & 0xFFFF)

    @triton.jit
    def _bf16_word(x):
        """float32 -> nearest BF16 word (ties to even), int32."""
        b = x.to(tl.uint32, bitcast=True)
        return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF).to(tl.int32)

    @triton.jit
    def _first_ge(x):
        """Ordered index of x rounded up to BF16 (finite x): truncate the bit pattern, plus one step for a positive
        value with nonzero low bits (truncating a negative value already moves it up). For x = +0 this is +0."""
        b = x.to(tl.uint32, bitcast=True)
        w = (b >> 16).to(tl.int32)
        up = ((b >> 31) == 0) & ((b & 0xFFFF) != 0)
        return _ord(tl.where(up, w + 1, w))

    @triton.jit
    def _bd(o):
        """Lower edge of ordered word o's rounding interval (exact in float32)."""
        return (_word_val(_word(o - 1)) + _word_val(_word(o))) * 0.5

    @triton.jit
    def _width_class(mu, s):
        a = tl.abs(mu) + s * 2.0
        ea = ((a.to(tl.int32, bitcast=True) >> 23) & 0xFF)
        es = ((s.to(tl.int32, bitcast=True) >> 23) & 0xFF)
        return tl.minimum(tl.maximum(ea - es - 3, 0), 9)

    @triton.jit
    def _bounds(c, k, mu, s):
        d = ((c + 124) << 23).to(tl.float32, bitcast=True)          # 2^(c-3)
        t0 = k.to(tl.float32) * d - 24.0
        t1 = tl.minimum(t0 + d, 24.0)
        return mu + t0 * s, mu + t1 * s

    @triton.jit
    def _shares(first, last, n):
        """Exact mode: base edge and slot scale of a slice."""
        base = _bd(first)
        width = _bd(last + 1) - base
        R = tl.math.div_rn((65536 - n).to(tl.float32), width)
        return base, R, width

    @triton.jit
    def _cw(o, first, last, base, R):
        c = tl.floor((_bd(o) - base) * R).to(tl.int32) + (o - first)
        return tl.where(o > last, 65536, c)

    @triton.jit
    def _q_of(n):
        """Uniform mode: q = floor(65536 / n) (n >= 2), from an approximate float quotient corrected exactly."""
        nn = tl.maximum(n, 2)
        q = tl.floor(_rcp(nn.to(tl.float32)) * 65536.0).to(tl.int32)
        q = tl.where(q * nn > 65536, q - 1, q)
        q = tl.where((q + 1) * nn <= 65536, q + 1, q)
        return q

    @triton.jit
    def _j_of(slot, q, n):
        """Uniform mode: j = min(slot // q, n - 1), exactly."""
        j = tl.floor(slot.to(tl.float32) * _rcp(q.to(tl.float32))).to(tl.int32)
        j = tl.where(j * q > slot, j - 1, j)
        j = tl.where((j + 1) * q <= slot, j + 1, j)
        return tl.minimum(j, n - 1)

    # Stream segments (SEG): 0 = all lanes of a program share one segment (scalar write / read position);
    # R > 0 = the lanes [R * N] are R rows of N lanes, each row its own segment (positions [R]; prefix sums stay
    # within a row, so a row that is one warp needs no cross-warp step).
    @triton.jit
    def _slots(e, pos, SEG: tl.constexpr):
        """Consecutive positions from pos for the lanes with e = 1 (in lane order); returns (positions, count)."""
        if SEG > 0:
            N: tl.constexpr = e.shape[0] // SEG
            e2 = tl.reshape(e, [SEG, N])
            total = tl.sum(e2, 1)
            return tl.reshape(pos[:, None] + tl.cumsum(e2, 1) - 1, [SEG * N]), total
        else:
            return pos + tl.cumsum(e, 0) - 1, tl.sum(e, 0)

    @triton.jit
    def _put(x, cum, f, active, out_ptr, wp, SEG: tl.constexpr):
        """rANS-encode one symbol per active lane; emitted 16-bit words take consecutive positions below wp."""
        fu = f.to(tl.uint32)
        emit = active & (x >= (fu << 16))
        e = emit.to(tl.int32)
        if SEG > 0:
            N: tl.constexpr = e.shape[0] // SEG
            e2 = tl.reshape(e, [SEG, N])
            total = tl.sum(e2, 1)
            pos = tl.reshape((wp - total)[:, None] + tl.cumsum(e2, 1) - 1, [SEG * N])
        else:
            total = tl.sum(e, 0)
            pos = wp - total + tl.cumsum(e, 0) - 1
        tl.store(out_ptr + pos, (x & 0xFFFF).to(tl.int16), mask=emit)
        x = tl.where(emit, x >> 16, x)
        x = tl.where(active, ((x // fu) << 16) + (x % fu) + cum.to(tl.uint32), x)
        return x, wp - total

    @triton.jit
    def _take(x, cum, f, active, buf_ptr, rp, SEG: tl.constexpr):
        """rANS-decode step for one symbol (cum, f) per active lane, then refill from consecutive positions."""
        x = tl.where(active, f.to(tl.uint32) * (x >> 16) + (x & 0xFFFF) - cum.to(tl.uint32), x)
        need = active & (x < 65536)
        pos, total = _slots(need.to(tl.int32), rp, SEG)
        w = (tl.load(buf_ptr + pos, mask=need, other=0).to(tl.int32) & 0xFFFF).to(tl.uint32)
        x = tl.where(need, (x << 16) | w, x)
        return x, rp + total

    @triton.jit
    def _enc_value(x, wp, w, mu, s, valid, out_ptr, cumf_ptr, ko_ptr, UNIFORM: tl.constexpr, SEG: tl.constexpr):
        """Encode one word per lane (both symbols, in rANS order: word-in-slice first, then the slice)."""
        c = _width_class(mu, s)
        kop = tl.load(ko_ptr + c)
        K = kop & 1023
        offf = kop >> 10
        o = _ord(w)
        lo0, _u = _bounds(c, c * 0, mu, s)
        _l, hiK = _bounds(c, K - 1, mu, s)
        esc = (((w >> 7) & 0xFF) == 0xFF) | (o < _first_ge(lo0)) | (o >= _first_ge(hiK))
        # slice: estimate from the value, then move until first(lo) <= o < first(hi) (exactly the decoder's sets)
        v = _word_val(w)
        d_inv = ((130 - c) << 23).to(tl.float32, bitcast=True)            # 2^(3-c)
        t = (v - mu) * _rcp(s)                                            # an estimate: corrected below
        k = tl.floor((t + 24.0) * d_inv)
        k = tl.where(esc, 0.0, tl.minimum(tl.maximum(k, 0.0), (K - 1).to(tl.float32))).to(tl.int32)
        lo, hi = _bounds(c, k, mu, s)
        first = _first_ge(lo)
        nxt = _first_ge(hi)
        move = (~esc) & ((o < first) | (o >= nxt))
        while tl.max(move.to(tl.int32), 0) > 0:
            k = tl.where(move & (o < first), k - 1, tl.where(move & (o >= nxt), k + 1, k))
            lo, hi = _bounds(c, k, mu, s)
            first = _first_ge(lo)
            nxt = _first_ge(hi)
            move = (~esc) & ((o < first) | (o >= nxt))
        k = tl.where(esc, K, k)
        p = tl.load(cumf_ptr + offf + k)
        cum1 = (p >> 16) & 0xFFFF                                # packed int32: shift is arithmetic
        f1 = p & 0xFFFF
        last = nxt - 1
        n = nxt - first
        j = o - first
        if UNIFORM:                      # equal shares; the last word also takes the remainder
            q = _q_of(n)
            c0 = j * q
            f0 = tl.where(j == n - 1, 65536 - (n - 1) * q, q)
        else:                            # shares follow the words' rounding intervals
            base, R, width = _shares(first, last, n)
            c0 = _cw(o, first, last, base, R)
            f0 = _cw(o + 1, first, last, base, R) - c0
        two = valid & (esc | (n > 1))
        cum2 = tl.where(esc, w, c0)
        f2 = tl.where(esc, 1, f0)
        x, wp = _put(x, cum2, f2, two, out_ptr, wp, SEG)
        x, wp = _put(x, cum1, f1, valid, out_ptr, wp, SEG)
        return x, wp

    @triton.jit
    def _dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, LUT_SHIFT: tl.constexpr,
                   LUT_W: tl.constexpr, REFINE: tl.constexpr, UNIFORM: tl.constexpr, SEG: tl.constexpr,
                   SEARCH: tl.constexpr):
        """Decode one word per lane; returns (state, read position, word)."""
        c = _width_class(mu, s)
        row = lut_ptr + c * LUT_W
        kop = tl.load(row + (LUT_W - 3))
        K = kop & 1023
        offf = kop >> 10
        slot = (x & 0xFFFF).to(tl.int32)
        lp = tl.load(row + (slot >> LUT_SHIFT))
        if SEARCH:                       # stage 1: coarse table, then binary search for the largest k, cum[k] <= slot
            klo = lp & 0xFFFF
            khi = klo + (lp >> 16)
            while tl.max(khi - klo, 0) > 1:
                mid = (klo + khi) // 2
                go = ((tl.load(cumf_ptr + offf + mid) >> 16) & 0xFFFF) <= slot
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
        x, rp = _take(x, (p >> 16) & 0xFFFF, p & 0xFFFF, valid, buf_ptr, rp, SEG)
        esc = k == K
        lo, hi = _bounds(c, tl.where(esc, 0, k), mu, s)
        first = _first_ge(lo)
        last = _first_ge(hi) - 1
        n = last - first + 1
        slot2 = (x & 0xFFFF).to(tl.int32)
        if UNIFORM:
            q = _q_of(n)
            j = _j_of(slot2, q, n)
            o = tl.where(valid & (~esc) & (n > 1), first + j, first)
            cum2 = tl.where(esc, slot2, j * q)
            f2 = tl.where(esc, 1, tl.where(j == n - 1, 65536 - (n - 1) * q, q))
        else:
            base, R, width = _shares(first, last, n)
            # estimate the word from the slot, refine it for the one-slot-per-word term, then a binary search on
            # the cumulative slots (the loop runs to the slowest lane of the program, so a good estimate matters)
            u = base + slot2.to(tl.float32) * (width * (1.0 / 65536.0))
            oe = tl.minimum(tl.maximum(_ord(_bf16_word(u)), first), last)
            inv_r = width * _rcp((65536 - n).to(tl.float32))                 # an estimate, searched below
            for _r in tl.static_range(REFINE):
                u = base + (slot2 - (oe - first)).to(tl.float32) * inv_r
                oe = tl.minimum(tl.maximum(_ord(_bf16_word(u)), first), last)
            ce0 = _cw(oe, first, last, base, R)
            ce1 = _cw(oe + 1, first, last, base, R)
            search = valid & (~esc) & (n > 1)
            below = search & (ce0 > slot2)
            above = search & (ce1 <= slot2)
            olo = tl.where(below, first, tl.where(above, oe + 1, oe))
            ohi = tl.where(below, oe, tl.where(above, last + 1, oe + 1))
            clo = tl.where(below, 0, tl.where(above, ce1, ce0))              # C(first) = 0
            chi = tl.where(below, ce0, tl.where(above, 65536, ce1))          # C(last + 1) = 65536
            while tl.max(ohi - olo, 0) > 1:
                mid = (olo + ohi) // 2
                cm = _cw(mid, first, last, base, R)
                go = cm <= slot2
                olo = tl.where(go, mid, olo)
                clo = tl.where(go, cm, clo)
                ohi = tl.where(go, ohi, mid)
                chi = tl.where(go, chi, cm)
            o = tl.where(search, olo, first)
            cum2 = tl.where(esc, slot2, clo)
            f2 = tl.where(esc, 1, chi - clo)
        two = valid & (esc | (n > 1))
        x, rp = _take(x, cum2, f2, two, buf_ptr, rp, SEG)
        w = tl.where(esc, slot2, _word(o))
        return x, rp, w

    @triton.jit
    def _encode_kernel(words_ptr, mu_ptr, s_ptr, out_ptr, start_ptr, cumf_ptr, ko_ptr,
                       nvals, S, V, CAP, BLOCK: tl.constexpr, UNIFORM: tl.constexpr):
        pid = tl.program_id(0)
        lane = pid * BLOCK + tl.arange(0, BLOCK)
        x = tl.full([BLOCK], 65536, tl.uint32)
        wp = (pid + 1) * CAP
        for step in range(0, V):
            i = V - 1 - step
            idx = i * S + lane
            valid = (lane < S) & (idx < nvals)
            w = tl.load(words_ptr + idx, mask=valid, other=0).to(tl.int32) & 0xFFFF
            mu = tl.load(mu_ptr + idx, mask=valid, other=0.0)
            s = tl.load(s_ptr + idx, mask=valid, other=1.0)
            x, wp = _enc_value(x, wp, w, mu, s, valid, out_ptr, cumf_ptr, ko_ptr, UNIFORM, 0)
        r = tl.arange(0, BLOCK)
        tl.store(out_ptr + (wp - 2 * BLOCK + 2 * r), (x >> 16).to(tl.int16))
        tl.store(out_ptr + (wp - 2 * BLOCK + 2 * r + 1), (x & 0xFFFF).to(tl.int16))
        tl.store(start_ptr + pid, wp - 2 * BLOCK)

    @triton.jit
    def _decode_kernel(buf_ptr, start_ptr, mu_ptr, s_ptr, words_ptr, cumf_ptr, lut_ptr, tail_ptr,
                       nvals, S, V, BLOCK: tl.constexpr, LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr,
                       REFINE: tl.constexpr, UNIFORM: tl.constexpr, SEARCH: tl.constexpr):
        pid = tl.program_id(0)
        lane = pid * BLOCK + tl.arange(0, BLOCK)
        r = tl.arange(0, BLOCK)
        rp = tl.load(start_ptr + pid)
        hi16 = (tl.load(buf_ptr + (rp + 2 * r)).to(tl.int32) & 0xFFFF).to(tl.uint32)
        lo16 = (tl.load(buf_ptr + (rp + 2 * r + 1)).to(tl.int32) & 0xFFFF).to(tl.uint32)
        x = (hi16 << 16) | lo16
        rp = rp + 2 * BLOCK
        for i in range(0, V):
            idx = i * S + lane
            valid = (lane < S) & (idx < nvals)
            mu = tl.load(mu_ptr + idx, mask=valid, other=0.0)
            s = tl.load(s_ptr + idx, mask=valid, other=1.0)
            x, rp, w = _dec_value(x, rp, mu, s, valid, buf_ptr, cumf_ptr, lut_ptr, tail_ptr, LUT_SHIFT, LUT_W, REFINE,
                                  UNIFORM, 0, SEARCH)
            tl.store(words_ptr + idx, w.to(tl.int16), mask=valid)


class SliceCoder:
    """Encode/decode BF16 words given float32 predictions and spreads (all on one device)."""

    def __init__(self, device, values_per_stream=2048, block=128, warps=None, fuse=False, mode="exact", search=False):
        """block: lanes (streams) per Triton program; warps: warps per program (default one thread per lane;
        fewer warps give each thread several lanes, i.e. independent rANS chains to interleave). search: decode
        stage 1 with the coarse table and a binary search (the earlier decoder) instead of without a search (the
        streams are the same either way)."""
        if triton is None:
            raise RuntimeError("Triton is required for the GPU coder")
        self.device = torch.device(device)
        self.V, self.BLOCK = int(values_per_stream), int(block)
        self.warps = max(1, self.BLOCK // 32) if warps is None else int(warps)
        if mode not in ("exact", "uniform"):
            raise ValueError("mode is 'exact' (word shares follow rounding intervals) or 'uniform' (equal shares)")
        self.mode = mode
        self.search = bool(search)
        self.tables = slice_tables(self.device)
        self.opts = {} if fuse else {"enable_fp_fusion": False}

    def layout(self, n):
        S = -(-n // self.V)                       # lanes (streams)
        programs = -(-S // self.BLOCK)
        cap = self.BLOCK * (2 * self.V + 2)
        return S, programs, cap

    def _launch(self, kernel, grid, *args, **kw):
        try:                                     # explicit warps: Triton's default of 4 replicates small blocks
            return kernel[grid](*args, **kw, num_warps=self.warps, **self.opts)
        except TypeError:                        # launch options unsupported (e.g. interpreter): plain launch
            return kernel[grid](*args, **kw)

    def encode(self, words, mu, s, out=None):
        """words: int16/int32 [n]; mu, s: float32 [n]. Returns (buffer int16 holding 16-bit words, starts int32, n).
        out: an int16 buffer of at least programs x capacity words to reuse (its old contents do not matter)."""
        n = words.numel()
        S, P, cap = self.layout(n)
        w = words.reshape(-1).to(torch.int32).contiguous()
        if out is None:
            out = torch.empty(P * cap, dtype=torch.int16, device=self.device)
        starts = torch.empty(P, dtype=torch.int32, device=self.device)
        t = self.tables
        self._launch(_encode_kernel, (P,), w, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous(), out, starts,
                     t["cumf"], t["ko"], n, S, self.V, cap, BLOCK=self.BLOCK, UNIFORM=self.mode == "uniform")
        return out, starts, n

    def decode(self, buf, starts, mu, s, n, out=None):
        S, P, cap = self.layout(n)
        words = torch.empty(n, dtype=torch.int16, device=self.device) if out is None else out
        t = self.tables
        self._launch(_decode_kernel, (P,), buf, starts, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous(),
                     words, t["cumf"], t["lut"], t["tail"], n, S, self.V, BLOCK=self.BLOCK,
                     LUT_SHIFT=16 - LUT_BITS, LUT_W=LUT_W, REFINE=REFINE, UNIFORM=self.mode == "uniform",
                     SEARCH=self.search)
        return words

    def bits(self, starts, n):
        """Coded size in bits: every program's used words (states included) plus a 32-bit start per program."""
        S, P, cap = self.layout(n)
        ends = (torch.arange(P, device=starts.device, dtype=torch.int64) + 1) * cap
        used = int((ends - starts.to(torch.int64)).sum())
        return 16 * used + 32 * P

    def ideal_bits(self, words, mu, s):
        """Sum of -log2(f / 65536) over the coder's symbols (no stream overhead), computed in PyTorch with the
        same float32 operations: the size the rANS stream should reach up to its state and rounding overhead."""
        t = self.tables
        return float(symbol_costs(words, mu, s, t["cum"], t["off"], t["K"], self.mode).sum())


# ---------------------------------------------------------------------------------------------- reference model
def _f32(x):
    return x.to(torch.float32)


def _word_val_t(w):
    return ((w.to(torch.int64) & 0xFFFF) << 16).to(torch.int32).view(torch.float32)


def _ord_t(w):
    w = w.to(torch.int64) & 0xFFFF
    return torch.where((w & 0x8000) != 0, (~w) & 0xFFFF, w ^ 0x8000)


def _word_t(o):
    o = o.to(torch.int64)
    return torch.where((o & 0x8000) != 0, o ^ 0x8000, (~o) & 0xFFFF)


def _bf16_word_t(x):
    b = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return (((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF)


def _first_ge_t(x):
    b = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    w = b >> 16
    up = ((b >> 31) == 0) & ((b & 0xFFFF) != 0)
    return _ord_t(torch.where(up, w + 1, w))


def _bd_t(o):
    return (_word_val_t(_word_t(o - 1)) + _word_val_t(_word_t(o))) * 0.5


def symbol_costs(words, mu, s, cum, off, K, mode="exact"):
    """Per-value bits of the value-slice coder (both symbols), PyTorch float32 mirror of the kernels."""
    w = words.reshape(-1).to(torch.int64) & 0xFFFF
    mu, s = _f32(mu.reshape(-1)), _f32(s.reshape(-1))
    a = mu.abs() + s * 2.0
    ea = (a.view(torch.int32) >> 23) & 0xFF
    es = (s.view(torch.int32) >> 23) & 0xFF
    c = (ea - es - 3).clamp(0, 9).to(torch.int64)
    Kc = K.to(torch.int64)[c]
    offc = off.to(torch.int64)[c]
    d = ((c + 124) << 23).to(torch.int32).view(torch.float32)
    v = _word_val_t(w)
    o = _ord_t(w)

    def bounds(k):
        t0 = k.to(torch.float32) * d - 24.0
        t1 = torch.minimum(t0 + d, torch.full_like(t0, 24.0))
        return mu + t0 * s, mu + t1 * s
    lo0, _ = bounds(torch.zeros_like(Kc))
    _, hiK = bounds(Kc - 1)
    esc = (((w >> 7) & 0xFF) == 0xFF) | (o < _first_ge_t(lo0)) | (o >= _first_ge_t(hiK))
    k = torch.floor(((v - mu) / s + 24.0) * ((130 - c) << 23).to(torch.int32).view(torch.float32))
    k = torch.where(esc, torch.zeros_like(k), torch.minimum(k.clamp_min(0), (Kc - 1).to(torch.float32))).to(torch.int64)
    lo, hi = bounds(k)
    first, nxt = _first_ge_t(lo), _first_ge_t(hi)
    for _ in range(64):
        move = ~esc & ((o < first) | (o >= nxt))
        if not bool(move.any()):
            break
        k = torch.where(move & (o < first), k - 1, torch.where(move & (o >= nxt), k + 1, k))
        lo, hi = bounds(k)
        first, nxt = _first_ge_t(lo), _first_ge_t(hi)
    cumt = cum.to(torch.int64)
    ks = torch.where(esc, Kc, k)
    f1 = cumt[offc + ks + 1] - cumt[offc + ks]
    last = nxt - 1
    n = nxt - first
    j = o - first
    if mode == "uniform":
        q = torch.div(65536, n.clamp_min(2), rounding_mode="floor")
        share = torch.where(j == n - 1, 65536 - (n - 1) * q, q)
    else:
        base = _bd_t(first)
        R = (65536 - n).to(torch.float32) / (_bd_t(last + 1) - base)

        def cw(qq):
            cc = torch.floor((_bd_t(qq) - base) * R).to(torch.int64) + (qq - first)
            return torch.where(qq > last, torch.full_like(cc, 65536), cc)
        share = cw(o + 1) - cw(o)
    f2 = torch.where(esc, torch.ones_like(n), torch.where(n > 1, share, torch.full_like(n, 65536)))
    return 16 - torch.log2(f1.double()) + 16 - torch.log2(f2.double())
