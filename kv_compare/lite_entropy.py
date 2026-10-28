"""Exact-wire decoder ablations: complete slice lookup and word-guess refinements.

The encoder and probability map stay inherited. The full lookup is derived
solely from the original cumulative tables; REFINE changes only a word guess,
which is checked and corrected by the unchanged exact cumulative search.
"""
import hashlib
import time

import numpy as np
import torch
import triton
import triton.language as tl

from .gpu_coder import (CLASSES, LUT_BITS, LUT_N, SliceCoder, _decode_kernel,
                        _width_class, _take, _bounds, _slots, _ord, _bf16_word,
                        _cw, _word)
from .lite_finish import FinishEngine


FULL_LUT_WIDTH = 65536


def full_slice_lookup(cum, off, counts):
    """Build [class, 16-bit slot] -> slice as uint16 from existing cums only."""
    def host(tensor):
        return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    cum, off, counts = (host(value) for value in (cum, off, counts))
    if cum.ndim != 1 or off.shape != (CLASSES,) or counts.shape != (CLASSES,):
        raise ValueError('invalid original slice table geometry')
    slots = np.arange(FULL_LUT_WIDTH, dtype=np.int32)
    result = np.empty((CLASSES, FULL_LUT_WIDTH), dtype=np.uint16)
    for c in range(CLASSES):
        offset, count = int(off[c]), int(counts[c])
        row = cum[offset:offset + count + 2]
        if (offset < 0 or count < 1 or count > np.iinfo(np.uint16).max or
                row.size != count + 2 or row[0] != 0 or row[-1] != FULL_LUT_WIDTH or
                np.any(np.diff(row.astype(np.int64)) <= 0)):
            raise ValueError('invalid original slice cumulative intervals')
        result[c] = np.searchsorted(row, slots, side='right') - 1
    return result


@triton.jit
def _decode_full_kernel(buf_ptr, start_ptr, mu_ptr, s_ptr, words_ptr, cum_ptr, off_ptr, K_ptr, lut_ptr,
                        nvals, S, V, BLOCK: tl.constexpr, LUT_SHIFT: tl.constexpr, LUT_W: tl.constexpr,
                        REFINE: tl.constexpr):
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
        # Stage 1 changes only the exact slot -> slice lookup.
        slot = (x & 0xFFFF).to(tl.int32)
        k = tl.load(lut_ptr + c * LUT_W + slot).to(tl.int32)
        cum1 = tl.load(cum_ptr + off + k)
        f1 = tl.load(cum_ptr + off + k + 1) - cum1
        x, rp = _take(x, cum1, f1, valid, buf_ptr, rp)
        esc = k == K
        lo, hi = _bounds(c, tl.where(esc, 0, k), K, mu, s)
        first, last, n, base, R, width = _slots(lo, hi)
        slot2 = (x & 0xFFFF).to(tl.int32)
        # Stage 2 remains identical to the original kernel, including both
        # rounded divisions and the exact search after the optional estimates.
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
        clo = tl.where(below, 0, tl.where(above, ce1, ce0))
        chi = tl.where(below, ce0, tl.where(above, 65536, ce1))
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


class EntropySliceCoder(SliceCoder):
    """Share the original coder's tables and encoder; change only decoding."""
    def __init__(self, original, lookup='coarse', refine=2):
        if lookup not in ('coarse', 'full') or type(refine) is not int or refine not in (0, 1, 2):
            raise ValueError('entropy lookup must be coarse/full and refine must be 0/1/2')
        # No second SliceCoder construction or duplicate original table tensors.
        self.__dict__.update(original.__dict__)
        self.lookup, self.refine = lookup, refine
        self.full_lut = None
        self.full_lut_sha256 = None
        elapsed = 0.
        if lookup == 'full':
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            begin = time.perf_counter()
            array = full_slice_lookup(self.cum, self.off, self.K)
            self.full_lut_sha256 = hashlib.sha256(array.astype('<u2', copy=False).tobytes()).hexdigest()
            self.full_lut = torch.from_numpy(array).to(self.device)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - begin) * 1000
        original_bytes = sum(t.numel() * t.element_size() for t in (self.cum, self.off, self.K, self.lut))
        full_bytes = 0 if self.full_lut is None else self.full_lut.numel() * self.full_lut.element_size()
        self.metadata = dict(lookup=lookup, refine=refine, full_lut_bytes=full_bytes,
            full_lut_sha256=self.full_lut_sha256, retained_original_table_bytes=original_bytes,
            total_entropy_table_bytes=original_bytes + full_bytes, extra_shared_bytes=full_bytes,
            full_lut_construction_wall_ms=elapsed,
            scope='fixed tables, separately from predictor bytes; full LUT construction includes host work, upload and CUDA synchronization on CUDA')

    def decode(self, buf, starts, mu, s, n):
        if self.lookup == 'coarse' and self.refine == 2:
            return super().decode(buf, starts, mu, s, n)
        S, P, _ = self.layout(n)
        words = torch.empty(n, dtype=torch.int16, device=self.device)
        kernel = _decode_full_kernel if self.lookup == 'full' else _decode_kernel
        table = self.full_lut if self.lookup == 'full' else self.lut
        width = FULL_LUT_WIDTH if self.lookup == 'full' else LUT_N
        self._launch(kernel, (P,), buf, starts, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous(),
            words, self.cum, self.off, self.K, table, n, S, self.V, BLOCK=self.BLOCK,
            LUT_SHIFT=16 - LUT_BITS, LUT_W=width, REFINE=self.refine)
        return words


class EntropyEngine(FinishEngine):
    """FinishEngine with an independently accounted, exact-wire decoder arm."""
    def __init__(self, *args, entropy_lookup='coarse', entropy_refine=2, **kwargs):
        if (entropy_lookup not in ('coarse', 'full') or type(entropy_refine) is not int or
                entropy_refine not in (0, 1, 2)):
            raise ValueError('entropy lookup must be coarse/full and refine must be 0/1/2')
        super().__init__(*args, **kwargs)
        self.lossless = EntropySliceCoder(self.lossless, entropy_lookup, entropy_refine)
        self.entropy_lookup, self.entropy_refine = entropy_lookup, entropy_refine
        self.entropy_metadata = dict(self.lossless.metadata)
        self.entropy_shared_bytes = self.entropy_metadata['extra_shared_bytes']
        self.entropy_table_bytes = self.entropy_metadata['total_entropy_table_bytes']
        self.entropy_table_sha256 = self.lossless.full_lut_sha256
