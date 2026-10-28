"""Execution-only Lite ablations: unique-column stores and one-kernel RoPE.

The reference predictor, entropy kernels, validation and wire format stay in
causal_packet.py. _run_columns is its _run with four indexed assignments
replaced by scatter calls; an AST comparison test locks that narrow difference.
Columns come only from the fitted permutations validated by Engine.__init__.
"""
import torch
import triton
import triton.language as tl

from .causal_packet import Packet
from .causal_entropy import ESC, LIMIT, compact
from .core import LN2X2, words_to_float
from .lite_fast import FastEngine
import numpy as np


@triton.jit
def _scatter_columns(src, columns, dst, n, width: tl.constexpr,
                     sr: tl.constexpr, sc: tl.constexpr,
                     dr: tl.constexpr, dc: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // width, i % width
    target = tl.load(columns + col, mask=i < n, other=0)
    value = tl.load(src + row * sr + col * sc, mask=i < n, other=0)
    # Each destination is unique. No atomics, summation or ordering change.
    tl.store(dst + row * dr + target * dc, value, mask=i < n)


def scatter_columns(dst, columns, src):
    """Internal store for Engine's checked, unique, in-bounds fitted columns."""
    if (src.ndim != 2 or dst.ndim != 2 or src.shape[0] != dst.shape[0] or
            src.shape[1] != columns.numel() or columns.dtype != torch.int64 or
            src.dtype != dst.dtype or src.device != dst.device or columns.device != dst.device):
        raise ValueError("invalid unique-column store geometry")
    if src.numel():
        _scatter_columns[(triton.cdiv(src.numel(), 256),)](src, columns, dst,
            src.numel(), src.shape[1], *src.stride(), *dst.stride(), 256, num_warps=4)


@triton.jit
def _rotate_group(x, cos, sin, freq, mate, key, sign, out, n,
                  width: tl.constexpr, frequencies: tl.constexpr,
                  xr: tl.constexpr, xc: tl.constexpr,
                  inverse: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // width, i % width
    f = tl.load(freq + col, mask=i < n, other=0)
    partner = tl.load(mate + col, mask=i < n, other=0)
    is_key = tl.load(key + col, mask=i < n, other=False)
    sg = tl.load(sign + col, mask=i < n, other=1.)
    c = tl.load(cos + row * frequencies + f, mask=i < n, other=1.)
    s = tl.load(sin + row * frequencies + f, mask=i < n, other=0.)
    if inverse:
        # Triton 3.6 unary minus lowers through 0-s, which changes +0 to
        # +0. Match torch.neg, including the negative zero at position zero.
        s = (s.to(tl.uint64, bitcast=True) ^ 0x8000000000000000).to(tl.float64, bitcast=True)
    xd = tl.load(x + row * xr + col * xc, mask=i < n, other=0.).to(tl.float64)
    other = tl.load(x + row * xr + partner * xc, mask=i < n, other=0.).to(tl.float64)
    # Preserve reference float64 operation order. FMA is disabled at launch.
    rotated = xd * c + (other * s) * sg
    value = tl.where(is_key, rotated, xd).to(tl.float32)
    tl.store(out + i, value, mask=i < n)


class FusedEngine(FastEngine):
    def __init__(self, *args, fuse_scatter=True, fuse_rope=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fuse_scatter, self.fuse_rope = fuse_scatter, fuse_rope

    def scatter(self, dst, columns, src):
        if self.fuse_scatter:
            scatter_columns(dst, columns, src)
        else:
            dst[:, columns] = src

    def rotate(self, x, g, positions, inverse=False):
        # The measured Llama state uses scale=1. Other attention scales retain
        # the exact reference path; no approximate division is introduced.
        if not self.fuse_rope or self.rope.scale != 1.:
            return super().rotate(x, g, positions, inverse)
        cos, sin = self.rope.tables(positions)
        out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        _rotate_group[(triton.cdiv(x.numel(), 256),)](x, cos, sin, g["freq"],
            g["mate"], g["key"], g["sign"], out, x.numel(), x.shape[1], cos.shape[1],
            *x.stride(), inverse, 256, num_warps=4, enable_fp_fusion=False)
        return out

    def _run(self, shape, start, kappa, source=None, packet=None, calibrate=False):
        with self.rope.request(start, shape[1], self.device):
            return _run_columns(self, shape, start, kappa, source, packet, calibrate)


def _run_columns(self, shape, start, kappa, source=None, packet=None, calibrate=False):
    L, N, W = shape
    positions = torch.arange(start, start + N, device=self.device)
    descriptions, capacity, program_count = self.layout(N)
    if calibrate:
        calibrated = []
    elif source is not None:
        staging = torch.empty(capacity, dtype=torch.int16, device=self.device)
        starts = torch.empty(program_count, dtype=torch.int64, device=self.device)
        self.last_reconstruction = (torch.empty(tuple(shape), dtype=torch.int16, device=self.device)
                                    if kappa else source)
    else:
        recovered = torch.empty(tuple(shape), dtype=torch.int16, device=self.device)
    features = {}
    prior = torch.ones(N, dtype=torch.float32, device=self.device)
    ci = 0
    for l, layer in enumerate(self.layers):
        if calibrate:
            layer_spread = torch.empty(W, dtype=torch.float32, device=self.device)
        external = []
        for kind in layer["kinds"]:
            external.append(features[{"L0": 0, "P": l - 1, "P2": l - 2}[kind]])
        if external:
            Xe = torch.cat(external, 1).to(self.dtype)
            base_mu = (Xe @ layer["ext"]).float() + layer["bias"]
        else:
            base_mu = layer["bias"].expand(N, W)
        cur = torch.zeros(N, W, dtype=torch.float32, device=self.device)
        surprise = torch.zeros(N, dtype=torch.float32, device=self.device)
        for g in layer["groups"]:
            st, ch, pos = g["start"], g["ch"], g["pos"]
            md = base_mu[:, pos]
            if st:
                md = md + (cur[:, :st].to(self.dtype) @ g["current"]).float()
            m = self.rotate(md, g, positions)
            m = torch.nan_to_num(m, nan=0., posinf=1e30, neginf=-1e30).clamp(-1e30, 1e30)
            spread = layer["spread"][ch][None, :]
            mult = ((64 * prior + surprise + 16) / (64 + st + 16)).clamp(.25, 4.)
            if not self.meta["adaptive"]:
                mult = torch.ones_like(mult)
            base, off, P, cap = descriptions[ci]; ci += 1
            n = N * g["columns"]
            if calibrate:
                rec = source[l][:, ch]
                layer_spread[ch] = (words_to_float(rec) - m.double()).abs().mean(0).float() / LN2X2 + 1e-12
            elif kappa:
                ids = (torch.round(torch.log2(mult) * 16).to(torch.int32) + 32).clamp(0, 64)
                step = (spread * kappa).clamp_min(1e-30)
                coder = self.residual(kappa)
                if source is not None:
                    raw = source[l][:, ch]
                    x = words_to_float(raw).float()
                    q = torch.round((x - m) / step)
                    inside = torch.isfinite(q) & (q.abs() <= LIMIT)
                    sym = torch.where(inside, q, 0).to(torch.int32) + LIMIT
                    sym = torch.where(inside, sym, ESC).contiguous()
                    coder.encode(sym, raw, ids, g["columns"],
                        out=staging[base:base + P * cap], starts=starts[off:off + P])
                    rec = self.reconstruct(m, sym, raw, step)
                else:
                    sym, raw = coder.decode(packet.payload, packet.starts[off:off + P], ids, g["columns"], n)
                    rec = self.reconstruct(m, sym, raw, step)
            else:
                s = (spread * mult[:, None]).expand_as(m).contiguous()
                if source is not None:
                    rec = source[l][:, ch].contiguous()
                    self.lossless.encode(rec, m.contiguous(), s,
                        out=staging[base:base + P * cap], starts=starts[off:off + P])
                else:
                    rec = self.lossless.decode(packet.payload, packet.starts[off:off + P],
                                               m.contiguous(), s, n).reshape(N, -1)
            xrec = words_to_float(rec)
            self.scatter(cur, pos, self.rotate(xrec, g, positions, inverse=True))
            surprise += ((xrec.float() - m).abs() / (LN2X2 * spread)).sum(1)
            if source is None:
                self.scatter(recovered[l], ch, rec)
            elif kappa:
                self.scatter(self.last_reconstruction[l], ch, rec)
        f = torch.empty_like(cur); self.scatter(f, layer["perm"], cur)
        features[l] = f
        for j in [j for j in features if j < l - 1 and j != 0]:
            del features[j]
        prior = surprise / W
        if calibrate:
            calibrated.append(layer_spread)
    if calibrate:
        return torch.stack(calibrated)
    if source is None:
        return recovered
    if N not in self.packing_tables:
        # Static geometry: two uploads once, instead of thousands of tiny
        # tensor allocations/kernel launches on every packet.
        bases = np.concatenate([np.full(P, base, dtype=np.int64) for base, _, P, cap in descriptions])
        ends = np.concatenate([base + (np.arange(P, dtype=np.int64) + 1) * cap
                               for base, _, P, cap in descriptions])
        self.packing_tables[N] = (torch.from_numpy(bases).to(self.device), torch.from_numpy(ends).to(self.device))
    bases, ends = self.packing_tables[N]
    payload, offsets = compact(staging, starts, bases, ends)
    return Packet(self.header(shape, start, kappa), offsets, payload)
