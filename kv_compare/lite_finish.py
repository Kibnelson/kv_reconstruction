"""Fuse exact post-group conversion, inverse RoPE and unique-column writes.

The adaptive-statistics expression and Torch reduction order stay unchanged.
_run_finish is _run_columns with only its post-group conversion/write cluster
replaced by finish_group; tests can compare their ASTs for that narrow change.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from .causal_packet import Packet
from .causal_entropy import ESC, LIMIT, compact
from .core import LN2X2, words_to_float
from .lite_fused import FusedEngine


@triton.jit
def _finish_kernel(rec, cos, sin, freq, mate, key, sign, pos, ch,
                   cur, recovered, xrec, n,
                   width: tl.constexpr, frequencies: tl.constexpr,
                   rr: tl.constexpr, rc: tl.constexpr,
                   cr: tl.constexpr, cc: tl.constexpr,
                   orow: tl.constexpr, oc: tl.constexpr,
                   WRITE_WORDS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // width, i % width
    w = tl.load(rec + row * rr + col * rc, mask=i < n, other=0).to(tl.int32) & 0xFFFF
    value = (w.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    partner = tl.load(mate + col, mask=i < n, other=0)
    other_word = tl.load(rec + row * rr + partner * rc, mask=i < n, other=0).to(tl.int32) & 0xFFFF
    other_value = (other_word.to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    f = tl.load(freq + col, mask=i < n, other=0)
    is_key = tl.load(key + col, mask=i < n, other=False)
    sg = tl.load(sign + col, mask=i < n, other=1.)
    c = tl.load(cos + row * frequencies + f, mask=i < n, other=1.)
    s = tl.load(sin + row * frequencies + f, mask=i < n, other=0.)
    # Match torch.neg bit-for-bit, including sine +0 -> -0 at position zero.
    s = (s.to(tl.uint64, bitcast=True) ^ 0x8000000000000000).to(tl.float64, bitcast=True)
    xd = value.to(tl.float64)
    rotated = xd * c + (other_value.to(tl.float64) * s) * sg
    feature = tl.where(is_key, rotated, xd).to(tl.float32)
    target = tl.load(pos + col, mask=i < n, other=0)
    tl.store(cur + row * cr + target * cc, feature, mask=i < n)
    tl.store(xrec + i, value, mask=i < n)
    if WRITE_WORDS:
        target_word = tl.load(ch + col, mask=i < n, other=0)
        tl.store(recovered + row * orow + target_word * oc, w.to(tl.int16), mask=i < n)


class FinishEngine(FusedEngine):
    """Byte-compatible post-group fusion on top of the measured fused engine."""
    def __init__(self, *args, fusedfinish=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusedfinish = bool(fusedfinish)

    def finish_group(self, rec, cur, g, positions, recovered=None):
        """Write exact recovered words/features and return BF16 values as fp32.

        The recovered destination is optional for lossless encoding. Column
        indices are the same unique fitted permutations validated by Engine.
        """
        if (rec.ndim != 2 or cur.ndim != 2 or rec.shape[0] != cur.shape[0] or
                rec.shape[1] != g['columns'] or rec.dtype != torch.int16 or
                cur.dtype != torch.float32 or rec.device != cur.device or
                (recovered is not None and (recovered.ndim != 2 or
                 recovered.shape != cur.shape or recovered.dtype != torch.int16 or
                 recovered.device != cur.device))):
            raise ValueError('invalid finish-group geometry/dtypes/device')
        cos, sin = self.rope.tables(positions)
        xrec = torch.empty(rec.shape, dtype=torch.float32, device=rec.device)
        output = rec if recovered is None else recovered
        output_strides = (0, 0) if recovered is None else recovered.stride()
        _finish_kernel[(triton.cdiv(rec.numel(), 256),)](rec, cos, sin,
            g['freq'], g['mate'], g['key'], g['sign'], g['pos'], g['ch'],
            cur, output, xrec, rec.numel(), rec.shape[1], cos.shape[1],
            *rec.stride(), *cur.stride(), *output_strides,
            recovered is not None, 256, num_warps=4, enable_fp_fusion=False)
        return xrec

    def _run(self, shape, start, kappa, source=None, packet=None, calibrate=False):
        # Calibration retains its complete original float64 path. Nonunit RoPE
        # scales and disabled inherited fusions retain their reference behavior.
        if (not self.fusedfinish or calibrate or self.rope.scale != 1. or
                not self.fuse_scatter or not self.fuse_rope):
            return super()._run(shape, start, kappa, source, packet, calibrate)
        with self.rope.request(start, shape[1], self.device):
            return _run_finish(self, shape, start, kappa, source, packet, calibrate)


def _run_finish(self, shape, start, kappa, source=None, packet=None, calibrate=False):
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
            xrec = self.finish_group(rec, cur, g, positions,
                recovered[l] if source is None else self.last_reconstruction[l] if kappa else None)
            surprise += ((xrec.float() - m).abs() / (LN2X2 * spread)).sum(1)
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
