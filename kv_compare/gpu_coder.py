"""Value-slice coder for BF16 words on the GPU (Triton): lossless, given a prediction and a spread per value.

The word's value v is coded against Logistic(mu, s) in two symbols, with 16-bit rANS on many interleaved streams:

  1. Slice. With the width class c = clamp(floor(log2(|mu| + 2s)) - floor(log2 s) - 3, 0, 9), slices of width
     d = 2^(c-3) spreads cover t = (v - mu)/s on [-24, 24): slice k is [mu + t0*s, mu + t1*s) with
     t0 = -24 + k*d, t1 = min(t0 + d, 24). The slice symbol uses one static table per class (logistic mass of the
     slice, every symbol at least 1/65536); an escape symbol (value outside the range, Inf or NaN) is followed by
     the raw 16-bit word.
  2. Word in the slice. The words of the slice, first..last in ordered index, share 65,536 slots: word o gets
     C(o) = floor((BD(o) - BD(first)) * R) + (o - first), R = (65536 - n) / (BD(last + 1) - BD(first)), so its
     share follows its rounding interval [BD(o), BD(o + 1)) and every word has at least one slot. A slice with
     one word needs no second symbol. The decoder estimates the word from the slot, then corrects it with a
     bounded binary search.

Encoder and decoder run the same float32 operations (add, multiply, correctly rounded division; the kernels are
compiled without fused multiply-add) and integer operations, so they agree bit for bit on one GPU type. Shipping
the slice tables as constants (they are computed once here, in float64) and keeping the float32 lines unfused
makes the stream portable across GPUs; that portability is not tested here.

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
REFINE = 2                       # decoder: refinements of the word estimate before its binary search


def slice_tables(device):
    """Static stage-1 tables. Returns cum [sum(K_c + 2)] int32 (per class: K_c slices, the escape, then 65536),
    off [10] int32 (class start in cum), K [10] int32, lut [10, 2^LUT_BITS + 1] int32 (symbol holding slot
    b * 2^(16 - LUT_BITS); the last entry is the escape)."""
    cums, off, Ks, luts = [], [], [], []
    pos = 0
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))
    for c in range(CLASSES):
        d = 2.0 ** (c - 3)
        K = math.ceil(2 * RANGE / d)
        t = [-RANGE + k * d for k in range(K)] + [RANGE]
        P = [sig(t[k + 1]) - sig(t[k]) for k in range(K)]
        f = [1 + math.floor(p * (M - K - 1)) for p in P] + [1]
        f[int(np.argmax(P))] += M - sum(f)
        cum = np.concatenate([[0], np.cumsum(f)]).astype(np.int64)
        assert cum[-1] == M and min(f) >= 1
        lut = [int(np.searchsorted(cum, b << (16 - LUT_BITS), side="right") - 1) for b in range(1 << LUT_BITS)]
        luts.append(lut + [K])
        cums.append(cum)
        off.append(pos)
        Ks.append(K)
        pos += len(cum)
    as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int32), device=device)
    return as_t(np.concatenate(cums)), as_t(off), as_t(Ks), as_t(np.asarray(luts))


if triton is not None:

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
        """Smallest ordered index whose value is >= x (x finite)."""
        o = _ord(_bf16_word(x))
        o = tl.where(_word_val(_word(o)) < x, o + 1, o)
        o = tl.where(_word_val(_word(o - 1)) >= x, o - 1, o)
        return o

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
    def _bounds(c, k, K, mu, s):
        d = ((c + 124) << 23).to(tl.float32, bitcast=True)          # 2^(c-3)
        t0 = k.to(tl.float32) * d - 24.0
        t1 = tl.minimum(t0 + d, 24.0)
        lo = mu + t0 * s
        hi = mu + t1 * s
        return lo, hi

    @triton.jit
    def _slots(lo, hi):
        first = _first_ge(lo)
        last = _first_ge(hi) - 1
        n = last - first + 1
        base = _bd(first)
        width = _bd(last + 1) - base
        R = tl.math.div_rn((65536 - n).to(tl.float32), width)
        return first, last, n, base, R, width

    @triton.jit
    def _cw(o, first, last, base, R):
        c = tl.floor((_bd(o) - base) * R).to(tl.int32) + (o - first)
        return tl.where(o > last, 65536, c)

    @triton.jit
    def _first_last(lo, hi):
        first = _first_ge(lo)
        last = _first_ge(hi) - 1
        return first, last, last - first + 1

    @triton.jit
    def _cu(j, n):
        """Equal-share cumulative count: ceil(j * 65536 / n), unsigned 32-bit (j <= n <= 65535)."""
        ju = j.to(tl.uint32)
        nu = tl.maximum(n, 1).to(tl.uint32)
        return ((ju << 16) + nu - 1) // nu

    @triton.jit
    def _put(x, cum, f, active, out_ptr, wp):
        """rANS-encode one symbol per active lane; emitted 16-bit words take consecutive positions below wp."""
        fu = f.to(tl.uint32)
        emit = active & (x >= (fu << 16))
        e = emit.to(tl.int32)
        cnt = tl.cumsum(e, 0)
        total = tl.sum(e, 0)
        tl.store(out_ptr + (wp - total + cnt - 1), (x & 0xFFFF).to(tl.int16), mask=emit)
        x = tl.where(emit, x >> 16, x)
        x = tl.where(active, ((x // fu) << 16) + (x % fu) + cum.to(tl.uint32), x)
        return x, wp - total

    @triton.jit
    def _take(x, cum, f, active, buf_ptr, rp):
        """rANS-decode step for one symbol (cum, f) per active lane, then refill from consecutive positions."""
        x = tl.where(active, f.to(tl.uint32) * (x >> 16) + (x & 0xFFFF) - cum.to(tl.uint32), x)
        need = active & (x < 65536)
        e = need.to(tl.int32)
        cnt = tl.cumsum(e, 0)
        total = tl.sum(e, 0)
        w = (tl.load(buf_ptr + (rp + cnt - 1), mask=need, other=0).to(tl.int32) & 0xFFFF).to(tl.uint32)
        x = tl.where(need, (x << 16) | w, x)
        return x, rp + total

    @triton.jit
    def _encode_kernel(words_ptr, mu_ptr, s_ptr, out_ptr, start_ptr, cum_ptr, off_ptr, K_ptr,
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
            c = _width_class(mu, s)
            K = tl.load(K_ptr + c)
            off = tl.load(off_ptr + c)
            v = _word_val(w)
            finite = ((w >> 7) & 0xFF) != 0xFF                     # exponent field all ones: Inf or NaN
            lo0, _u = _bounds(c, tl.zeros([BLOCK], tl.int32), K, mu, s)
            _l, hiK = _bounds(c, K - 1, K, mu, s)
            esc = (~finite) | (v < lo0) | (v >= hiK)
            # slice: estimate, then move until lo <= v < hi (the decoder sees exactly these bounds)
            d_inv = ((130 - c) << 23).to(tl.float32, bitcast=True)            # 2^(3-c)
            t = tl.math.div_rn(v - mu, s)
            k = tl.floor((t + 24.0) * d_inv)
            k = tl.where(esc, 0.0, tl.minimum(tl.maximum(k, 0.0), (K - 1).to(tl.float32))).to(tl.int32)
            lo, hi = _bounds(c, k, K, mu, s)
            move = (~esc) & ((v < lo) | (v >= hi))
            while tl.max(move.to(tl.int32), 0) > 0:
                k = tl.where(move & (v < lo), k - 1, tl.where(move & (v >= hi), k + 1, k))
                lo, hi = _bounds(c, k, K, mu, s)
                move = (~esc) & ((v < lo) | (v >= hi))
            k = tl.where(esc, K, k)
            cum1 = tl.load(cum_ptr + off + k)
            f1 = tl.load(cum_ptr + off + k + 1) - cum1
            o = _ord(w)
            if UNIFORM:                      # every word of the slice gets an equal share of the 65,536 slots
                first, last, n = _first_last(lo, hi)
                c0 = _cu(o - first, n).to(tl.int32)
                c1 = _cu(o - first + 1, n).to(tl.int32)
            else:                            # shares follow the words' rounding intervals
                first, last, n, base, R, width = _slots(lo, hi)
                c0 = _cw(o, first, last, base, R)
                c1 = _cw(o + 1, first, last, base, R)
            two = valid & (esc | (n > 1))
            cum2 = tl.where(esc, w, c0)
            f2 = tl.where(esc, 1, c1 - c0)
            x, wp = _put(x, cum2, f2, two, out_ptr, wp)
            x, wp = _put(x, cum1, f1, valid, out_ptr, wp)
        r = tl.arange(0, BLOCK)
        tl.store(out_ptr + (wp - 2 * BLOCK + 2 * r), (x >> 16).to(tl.int16))
        tl.store(out_ptr + (wp - 2 * BLOCK + 2 * r + 1), (x & 0xFFFF).to(tl.int16))
        tl.store(start_ptr + pid, wp - 2 * BLOCK)

    @triton.jit
    def _decode_kernel(buf_ptr, start_ptr, mu_ptr, s_ptr, words_ptr, cum_ptr, off_ptr, K_ptr, lut_ptr,
                       nvals, S, V, BLOCK: tl.constexpr, LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr,
                       REFINE: tl.constexpr, UNIFORM: tl.constexpr):
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
            c = _width_class(mu, s)
            K = tl.load(K_ptr + c)
            off = tl.load(off_ptr + c)
            # stage 1: coarse table, then binary search for the largest k with cum[k] <= slot
            slot = (x & 0xFFFF).to(tl.int32)
            b = slot >> LUT_SHIFT
            klo = tl.load(lut_ptr + c * LUT_W + b)
            khi = tl.load(lut_ptr + c * LUT_W + b + 1) + 1
            while tl.max(khi - klo, 0) > 1:
                mid = (klo + khi) // 2
                go = tl.load(cum_ptr + off + mid) <= slot
                klo = tl.where(go, mid, klo)
                khi = tl.where(go, khi, mid)
            k = klo
            cum1 = tl.load(cum_ptr + off + k)
            f1 = tl.load(cum_ptr + off + k + 1) - cum1
            x, rp = _take(x, cum1, f1, valid, buf_ptr, rp)
            esc = k == K
            lo, hi = _bounds(c, tl.where(esc, 0, k), K, mu, s)
            if UNIFORM:
                first, last, n = _first_last(lo, hi)
                slot2 = (x & 0xFFFF).to(tl.int32)
                j = ((slot2.to(tl.uint32) * tl.maximum(n, 1).to(tl.uint32)) >> 16).to(tl.int32)   # exact inverse
                o = tl.where(valid & (~esc) & (n > 1), first + j, first)
                two = valid & (esc | (n > 1))
                cu0 = _cu(o - first, n).to(tl.int32)
                cu1 = _cu(o - first + 1, n).to(tl.int32)
                cum2 = tl.where(esc, slot2, cu0)
                f2 = tl.where(esc, 1, cu1 - cu0)
            else:
                first, last, n, base, R, width = _slots(lo, hi)
                slot2 = (x & 0xFFFF).to(tl.int32)
                # stage 2: estimate the word from the slot, refine it for the one-slot-per-word term, then a binary
                # search on the cumulative slots (which lanes need it, and how far, varies; the loop runs to the
                # slowest lane of the program, so a good estimate matters)
                u = base + slot2.to(tl.float32) * (width * (1.0 / 65536.0))
                oe = tl.minimum(tl.maximum(_ord(_bf16_word(u)), first), last)
                inv_r = tl.math.div_rn(width, (65536 - n).to(tl.float32))
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
                two = valid & (esc | (n > 1))
                cum2 = tl.where(esc, slot2, clo)
                f2 = tl.where(esc, 1, chi - clo)
            x, rp = _take(x, cum2, f2, two, buf_ptr, rp)
            w = tl.where(esc, slot2, _word(o))
            tl.store(words_ptr + idx, w.to(tl.int16), mask=valid)


class SliceCoder:
    """Encode/decode BF16 words given float32 predictions and spreads (all on one device)."""

    def __init__(self, device, values_per_stream=2048, block=128, warps=None, fuse=False, mode="exact"):
        """block: lanes (streams) per Triton program; warps: warps per program (default one thread per lane;
        fewer warps give each thread several lanes, i.e. independent rANS chains to interleave)."""
        if triton is None:
            raise RuntimeError("Triton is required for the GPU coder")
        self.device = torch.device(device)
        self.V, self.BLOCK = int(values_per_stream), int(block)
        self.warps = max(1, self.BLOCK // 32) if warps is None else int(warps)
        if mode not in ("exact", "uniform"):
            raise ValueError("mode is 'exact' (word shares follow rounding intervals) or 'uniform' (equal shares)")
        self.mode = mode
        self.cum, self.off, self.K, self.lut = slice_tables(self.device)
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

    def encode(self, words, mu, s):
        """words: int16/int32 [n]; mu, s: float32 [n]. Returns (buffer int16 holding 16-bit words, starts int32, n)."""
        n = words.numel()
        S, P, cap = self.layout(n)
        w = words.reshape(-1).to(torch.int32).contiguous()
        out = torch.zeros(P * cap, dtype=torch.int16, device=self.device)
        starts = torch.zeros(P, dtype=torch.int32, device=self.device)
        self._launch(_encode_kernel, (P,), w, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous(), out, starts,
                     self.cum, self.off, self.K, n, S, self.V, cap, BLOCK=self.BLOCK, UNIFORM=self.mode == "uniform")
        return out, starts, n

    def decode(self, buf, starts, mu, s, n):
        S, P, cap = self.layout(n)
        words = torch.empty(n, dtype=torch.int16, device=self.device)
        self._launch(_decode_kernel, (P,), buf, starts, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous(),
                     words, self.cum, self.off, self.K, self.lut, n, S, self.V, BLOCK=self.BLOCK,
                     LUT_SHIFT=16 - LUT_BITS, LUT_W=LUT_N, REFINE=REFINE, UNIFORM=self.mode == "uniform")
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
        return float(symbol_costs(words, mu, s, self.cum, self.off, self.K, self.mode).sum())


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
    o = _ord_t(_bf16_word_t(x))
    o = torch.where(_word_val_t(_word_t(o)) < x, o + 1, o)
    o = torch.where(_word_val_t(_word_t(o - 1)) >= x, o - 1, o)
    return o


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
    d = 2.0 ** (c.to(torch.float32) - 3)
    v = _word_val_t(w)

    def bounds(k):
        t0 = k.to(torch.float32) * d - 24.0
        t1 = torch.minimum(t0 + d, torch.full_like(t0, 24.0))
        return mu + t0 * s, mu + t1 * s
    lo0, _ = bounds(torch.zeros_like(Kc))
    _, hiK = bounds(Kc - 1)
    esc = ~torch.isfinite(v) | (v < lo0) | (v >= hiK)
    k = torch.floor(((v - mu) / s + 24.0) * (2.0 ** (3 - c.to(torch.float32))))
    k = torch.where(esc, torch.zeros_like(k), torch.minimum(k.clamp_min(0), (Kc - 1).to(torch.float32))).to(torch.int64)
    lo, hi = bounds(k)
    for _ in range(64):
        move = ~esc & ((v < lo) | (v >= hi))
        if not bool(move.any()):
            break
        k = torch.where(move & (v < lo), k - 1, torch.where(move & (v >= hi), k + 1, k))
        lo, hi = bounds(k)
    cumt = cum.to(torch.int64)
    ks = torch.where(esc, Kc, k)
    f1 = cumt[offc + ks + 1] - cumt[offc + ks]
    first, last = _first_ge_t(lo), _first_ge_t(hi) - 1
    n = last - first + 1
    base = _bd_t(first)
    R = (65536 - n).to(torch.float32) / (_bd_t(last + 1) - base)
    o = _ord_t(w)

    def cw(q):
        cc = torch.floor((_bd_t(q) - base) * R).to(torch.int64) + (q - first)
        return torch.where(q > last, torch.full_like(cc, 65536), cc)
    if mode == "uniform":
        cu = lambda j: torch.div((j << 16) + n.clamp_min(1) - 1, n.clamp_min(1), rounding_mode="floor")
        share = cu(o - first + 1) - cu(o - first)
    else:
        share = cw(o + 1) - cw(o)
    f2 = torch.where(esc, torch.ones_like(n), torch.where(n > 1, share, torch.full_like(n, 65536)))
    return 16 - torch.log2(f1.double()) + 16 - torch.log2(f2.double())
