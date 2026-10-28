"""GPU residual rANS and GPU compaction for the causal packet experiment.

Lossless words use the team's unchanged value-slice model. Lossy symbols are
signed quantized residuals, with an exact original-BF16 escape for tails.
Only the CPU test harness enables the Triton interpreter.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
import torch
from .gpu_coder import SliceCoder, _put, _take
import triton
import triton.language as tl

LIMIT = 127
ESC = 2 * LIMIT + 1
NCDF = ESC + 2


def residual_tables(kappa):
    if not math.isfinite(kappa) or not 0 < kappa <= 16:
        raise ValueError("invalid residual step")
    rows = []
    for j in range(65):
        mult = 2.0 ** ((j - 32) / 16)
        edges = (np.arange(-LIMIT, LIMIT + 2, dtype=np.float64) - .5) * kappa / mult
        cdf = 1 / (1 + np.exp(-np.clip(edges, -700, 700)))
        p = np.r_[np.diff(cdf), max(0., 1 - (cdf[-1] - cdf[0]))]
        f = 1 + np.floor(p * (65536 - len(p))).astype(np.int64)
        f[np.argmax(p)] += 65536 - int(f.sum())
        if f.min() < 1:
            raise ValueError("invalid residual frequency table")
        rows.append(np.r_[0, np.cumsum(f)])
    return np.asarray(rows, dtype=np.int32)


@triton.jit
def _res_encode(symp, rawp, idp, cdfp, outp, startp, N, S, V, CAP, COLS,
                BLOCK: tl.constexpr, NCDF_: tl.constexpr, ESC_: tl.constexpr):
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.full([BLOCK], 65536, tl.uint32)
    wp = (pid + 1) * CAP
    for it in range(V):
        idx = (V - 1 - it) * S + lane
        valid = (lane < S) & (idx < N)
        sym = tl.load(symp + idx, valid, 0)
        raw = tl.load(rawp + idx, valid, 0).to(tl.int32) & 65535
        table = tl.load(idp + idx // COLS, valid, 32)
        c0 = tl.load(cdfp + table * NCDF_ + sym)
        f = tl.load(cdfp + table * NCDF_ + sym + 1) - c0
        x, wp = _put(x, raw, tl.full([BLOCK], 1, tl.int32),
                     valid & (sym == ESC_), outp, wp)
        x, wp = _put(x, c0, f, valid, outp, wp)
    r = tl.arange(0, BLOCK)
    tl.store(outp + wp - 2 * BLOCK + 2 * r, (x >> 16).to(tl.int16))
    tl.store(outp + wp - 2 * BLOCK + 2 * r + 1, (x & 65535).to(tl.int16))
    tl.store(startp + pid, wp - 2 * BLOCK)


@triton.jit
def _res_decode(bufp, startp, idp, cdfp, symp, rawp, N, S, V, COLS,
                BLOCK: tl.constexpr, NCDF_: tl.constexpr, ESC_: tl.constexpr):
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)
    r = tl.arange(0, BLOCK)
    rp = tl.load(startp + pid)
    hi = (tl.load(bufp + rp + 2 * r).to(tl.int32) & 65535).to(tl.uint32)
    lo = (tl.load(bufp + rp + 2 * r + 1).to(tl.int32) & 65535).to(tl.uint32)
    x = (hi << 16) | lo
    rp += 2 * BLOCK
    for it in range(V):
        idx = it * S + lane
        valid = (lane < S) & (idx < N)
        table = tl.load(idp + idx // COLS, valid, 32)
        slot = (x & 65535).to(tl.int32)
        left = tl.zeros([BLOCK], tl.int32)
        right = tl.full([BLOCK], ESC_ + 1, tl.int32)
        while tl.max(right - left, 0) > 1:
            mid = (left + right) // 2
            go = tl.load(cdfp + table * NCDF_ + mid) <= slot
            left = tl.where(go, mid, left)
            right = tl.where(go, right, mid)
        sym = left
        c0 = tl.load(cdfp + table * NCDF_ + sym)
        f = tl.load(cdfp + table * NCDF_ + sym + 1) - c0
        x, rp = _take(x, c0, f, valid, bufp, rp)
        raw = (x & 65535).to(tl.int32)
        x, rp = _take(x, raw, tl.full([BLOCK], 1, tl.int32),
                      valid & (sym == ESC_), bufp, rp)
        tl.store(symp + idx, sym, valid)
        tl.store(rawp + idx, tl.where(sym == ESC_, raw, 0).to(tl.int16), valid)


@triton.jit
def _compact(srcp, srcstartp, lengthp, dststartp, dstp, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    a = tl.load(srcstartp + pid)
    n = tl.load(lengthp + pid)
    b = tl.load(dststartp + pid)
    r = tl.arange(0, BLOCK)
    for i in range(tl.cdiv(n, BLOCK)):
        off = i * BLOCK + r
        v = tl.load(srcp + a + off, off < n, 0)
        tl.store(dstp + b + off, v, off < n)


def compact(staging, starts, bases, ends):
    """One scalar synchronization for the final size; all stream copies on GPU."""
    source_starts = starts + bases
    lengths = ends - source_starts
    offsets = torch.cumsum(lengths, 0) - lengths
    size = int((offsets[-1] + lengths[-1]).item())
    packed = torch.empty(size, dtype=torch.int16, device=staging.device)
    _compact[(starts.numel(),)](staging, source_starts, lengths, offsets, packed,
                              BLOCK=256, num_warps=1)
    return packed, offsets


class ResidualCoder:
    def __init__(self, device, kappa, values_per_stream=128, block=64):
        self.layout_owner = SliceCoder(device, values_per_stream=values_per_stream, block=block, warps=1)
        self.device, self.V, self.BLOCK = torch.device(device), values_per_stream, block
        array = residual_tables(kappa)
        self.table_sha256 = hashlib.sha256(array.astype("<i4").tobytes()).hexdigest()
        self.cdf = torch.from_numpy(array).to(device)

    def layout(self, n):
        return self.layout_owner.layout(n)

    def encode(self, symbols, raw, ids, columns, *, out, starts):
        n = symbols.numel()
        S, P, cap = self.layout(n)
        _res_encode[(P,)](symbols.reshape(-1).contiguous(), raw.reshape(-1).contiguous(),
            ids.contiguous(), self.cdf, out, starts, n, S, self.V, cap, columns,
            BLOCK=self.BLOCK, NCDF_=NCDF, ESC_=ESC, num_warps=1)

    def decode(self, payload, starts, ids, columns, n):
        S, P, cap = self.layout(n)
        sym = torch.empty(n, dtype=torch.int32, device=self.device)
        raw = torch.empty(n, dtype=torch.int16, device=self.device)
        _res_decode[(P,)](payload, starts, ids.contiguous(), self.cdf, sym, raw,
            n, S, self.V, columns, BLOCK=self.BLOCK, NCDF_=NCDF, ESC_=ESC, num_warps=1)
        return sym, raw
