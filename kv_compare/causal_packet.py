"""Complete causal GPU packets for matched lossless/lossy predictor policies.

Receiver inputs are a packet and fixed fitted state. Predictions, uncertainty
and earlier-group features are rebuilt from recovered words. BF16 products
are tested on the same GPU/software; cross-GPU portability is not claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import numpy as np
import torch
from .causal_entropy import ESC, LIMIT, ResidualCoder, compact
from .core import LN2X2, Rope, float_to_words, words_to_float
from .gpu_coder import SliceCoder

MAGIC = b"KVC2"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def word_sha(words):
    return hashlib.sha256(words.detach().cpu().numpy().astype("<u2").tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


@dataclass
class Packet:
    header: dict
    starts: torch.Tensor
    payload: torch.Tensor

    def export(self):
        h = json_bytes(self.header)
        directory = self.starts.detach().cpu().numpy().astype("<i8").tobytes()
        body = self.payload.detach().cpu().numpy().astype("<i2").tobytes()
        record = MAGIC + struct.pack("<IQQ", len(h), len(directory), len(body)) + h + directory + body
        return record + hashlib.sha256(record).digest()

    @classmethod
    def restore(cls, wire, device):
        if len(wire) < 60 or wire[:4] != MAGIC or hashlib.sha256(wire[:-32]).digest() != wire[-32:]:
            raise ValueError("bad packet checksum or magic")
        nh, nd, nb = struct.unpack("<IQQ", wire[4:24])
        if nh > 65536 or nd % 8 or nb % 2 or 24 + nh + nd + nb + 32 != len(wire):
            raise ValueError("invalid packet lengths")
        header = json.loads(wire[24:24 + nh])
        a = 24 + nh
        starts = torch.from_numpy(np.frombuffer(wire[a:a + nd], dtype="<i8").copy()).to(device)
        payload = torch.from_numpy(np.frombuffer(wire[a + nd:-32], dtype="<i2").copy()).to(device)
        return cls(header, starts, payload)


class Engine:
    def __init__(self, state_dir, device, *, predictor_format="bf16", values_per_stream=128, block=64):
        self.device = torch.device(device)
        if self.device.type != "cuda" and os.environ.get("TRITON_INTERPRET") != "1":
            raise RuntimeError("CPU operation is reserved for explicit Triton interpreter software tests")
        if predictor_format not in ("fp32", "bf16") or values_per_stream < 1 or block not in (8, 16, 32, 64, 128):
            raise ValueError("invalid engine options")
        self.dtype = torch.float32 if predictor_format == "fp32" else torch.bfloat16
        self.predictor_format, self.V, self.B = predictor_format, values_per_stream, block
        state_dir = Path(state_dir)
        self.meta = json.loads((state_dir / "state.json").read_text())
        state = dict(self.meta); expected = state.pop("state_sha256")
        if hashlib.sha256(json_bytes(state)).hexdigest() != expected:
            raise ValueError("state manifest digest mismatch")
        self.state_sha256 = expected
        self.W = 2 * self.meta["kv_heads"] * self.meta["head_dim"]
        self.C = self.W // 2
        self.L = len(self.meta["files"])
        runtime_spread = None
        if self.meta.get("runtime_spread"):
            entry = self.meta["runtime_spread"]
            if entry["predictor_format"] != predictor_format:
                raise ValueError("runtime spreads were calibrated for another arithmetic format")
            path = state_dir / entry["path"]
            if sha(path) != entry["sha256"]:
                raise ValueError("runtime spread file changed")
            runtime_spread = np.load(path, allow_pickle=False)
            if (runtime_spread.shape != (self.L, self.W) or not np.isfinite(runtime_spread).all() or
                    np.any(runtime_spread <= 0)):
                raise ValueError("invalid runtime spreads")
        self.rope = Rope(self.meta["inv_freq"], self.meta["kv_heads"], self.meta["head_dim"],
                         self.meta.get("rope_layout", "half"), self.meta.get("attention_scaling", 1.))
        self.layers = []
        self.shared_bytes = 0
        for layer_index, entry in enumerate(self.meta["files"]):
            path = state_dir / entry["path"]
            if sha(path) != entry["sha256"]:
                raise ValueError("fitted layer file changed")
            with np.load(path, allow_pickle=False) as a:
                A, b, spread = a["A"], a["b"], a["spread"]
                perm, starts = a["perm"].astype(np.int64), a["starts"].astype(np.int64)
                kinds = a["kinds"].tolist()
            if runtime_spread is not None:
                spread = runtime_spread[layer_index]
            ne = A.shape[1] - self.W
            if (A.shape[0] != self.W or sorted(perm.tolist()) != list(range(self.W)) or
                    any(k not in ("P", "P2", "L0") for k in kinds) or
                    ne != len(kinds) * self.W or np.any(starts < 0) or np.any(starts > np.arange(self.W)) or
                    not np.isfinite(A).all() or not np.isfinite(b).all() or not np.isfinite(spread).all()):
                raise ValueError("invalid causal fitted state")
            layer = dict(kinds=kinds, perm=self.tensor(perm, torch.long),
                ext=self.tensor(A[:, :ne].T.copy(), self.dtype),
                bias=self.tensor(b, torch.float32), spread=self.tensor(np.maximum(spread, 1e-12), torch.float32), groups=[])
            for st in sorted(set(starts.tolist())):
                pos = np.flatnonzero(starts == st); ch = perm[pos]
                partners = np.r_[self.rope.pair_partner(), np.arange(self.C, self.W)]
                lookup = {int(c): i for i, c in enumerate(ch)}
                if any(int(partners[c]) not in lookup for c in ch):
                    raise ValueError("a RoPE pair was split across decode groups")
                d = ch % self.meta["head_dim"]
                key = ch < self.C
                if self.rope.layout == "half":
                    freq = d % (self.meta["head_dim"] // 2)
                    sign = np.where(d < self.meta["head_dim"] // 2, -1., 1.)
                else:
                    freq = d // 2; sign = np.where(d % 2 == 0, -1., 1.)
                layer["groups"].append(dict(start=int(st), columns=len(pos),
                    pos=self.tensor(pos, torch.long), ch=self.tensor(ch, torch.long),
                    current=self.tensor(A[pos, ne:ne + st].T.copy(), self.dtype),
                    mate=self.tensor([lookup[int(partners[c])] for c in ch], torch.long),
                    key=self.tensor(key, torch.bool), freq=self.tensor(freq, torch.long),
                    sign=self.tensor(sign, torch.float64)))
            self.layers.append(layer)
        self.lossless = SliceCoder(self.device, values_per_stream=self.V, block=self.B, warps=1)
        table_data = b"".join(t.cpu().numpy().astype("<i4").tobytes()
                              for t in (self.lossless.cum, self.lossless.off, self.lossless.K))
        self.slice_table_sha = hashlib.sha256(table_data).hexdigest()
        self.residuals = {}
        self.layouts = {}
        self.packing_tables = {}

    def tensor(self, value, dtype):
        t = torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=self.device)
        self.shared_bytes += t.numel() * t.element_size()
        return t

    def residual(self, kappa):
        if kappa not in self.residuals:
            self.residuals[kappa] = ResidualCoder(self.device, kappa, self.V, self.B)
        return self.residuals[kappa]

    def header(self, shape, start, kappa, raw=False):
        return dict(format="causal_kv_v1", state_sha256=self.state_sha256,
            shape=list(shape), start=int(start), kappa=float(kappa), raw=bool(raw),
            predictor_format=self.predictor_format, values_per_stream=self.V, block=self.B,
            table_sha256=self.residual(kappa).table_sha256 if kappa else self.slice_table_sha)

    def layout(self, tokens):
        if tokens in self.layouts:
            return self.layouts[tokens]
        descriptions = []; total_capacity = 0; programs = 0
        for layer in self.layers:
            for g in layer["groups"]:
                n = tokens * g["columns"]
                _, P, cap = self.lossless.layout(n)
                descriptions.append((total_capacity, programs, P, cap))
                total_capacity += P * cap; programs += P
        self.layouts[tokens] = (descriptions, total_capacity, programs)
        return self.layouts[tokens]

    def rotate(self, x, g, positions, inverse=False):
        cos, sin = self.rope.tables(positions)
        c = cos[:, g["freq"]]; s = sin[:, g["freq"]]
        if inverse:
            s = -s
            if self.rope.scale != 1.:
                c, s = c / self.rope.scale ** 2, s / self.rope.scale ** 2
        xd = x.to(torch.float64)
        rotated = xd * c + xd[:, g["mate"]] * s * g["sign"]
        return torch.where(g["key"][None, :], rotated, xd).to(torch.float32)

    @staticmethod
    def reconstruct(m, sym, raw, step):
        sym = sym.reshape_as(m); raw = raw.reshape_as(m)
        normal = float_to_words(m + (sym - LIMIT).to(torch.float32) * step)
        return torch.where(sym == ESC, raw.to(torch.int32) & 65535, normal).to(torch.int16)

    def validate(self, packet):
        h = packet.header
        if set(h) != set(self.header((self.L, 1, self.W), 0, 0)):
            raise ValueError("unexpected packet fields")
        shape = h["shape"]
        if (not isinstance(shape, list) or len(shape) != 3 or any(type(x) is not int for x in shape) or
                shape[0] != self.L or shape[2] != self.W or not 0 < shape[1] <= 8192 or
                type(h["start"]) is not int or not 0 <= h["start"] <= 8192 - shape[1] or
                type(h["raw"]) is not bool or not isinstance(h["kappa"], (int, float)) or
                not math.isfinite(h["kappa"]) or not 0 <= h["kappa"] <= 16):
            raise ValueError("invalid packet geometry or mode")
        if h != self.header(shape, h["start"], h["kappa"], h["raw"]):
            raise ValueError("packet and decoder state/options differ")
        if packet.starts.dtype != torch.int64 or packet.payload.dtype != torch.int16:
            raise ValueError("invalid packet buffer types")
        if h["raw"]:
            if packet.starts.numel() or packet.payload.numel() != math.prod(shape):
                raise ValueError("invalid raw fallback")
        else:
            desc, _, count = self.layout(shape[1])
            offsets = packet.starts.detach().cpu().numpy()
            if len(offsets) != count or offsets[0] != 0:
                raise ValueError("invalid program directory")
            lengths = np.diff(np.r_[offsets, packet.payload.numel()])
            capacities = np.concatenate([np.full(P, cap) for _, _, P, cap in desc])
            if np.any(lengths < 2 * self.B) or np.any(lengths > capacities):
                raise ValueError("invalid stream bounds")

    def encode(self, words, *, start=0, kappa=0.):
        if tuple(words.shape)[::2] != (self.L, self.W) or words.ndim != 3 or not 0 < words.shape[1] <= 8192:
            raise ValueError("wrong input geometry")
        if not 0 <= start <= 8192 - words.shape[1] or not math.isfinite(kappa) or not 0 <= kappa <= 16:
            raise ValueError("invalid input start or step")
        words = words.to(device=self.device, dtype=torch.int16)
        # Preserve every BF16 pattern. Exceptional/extreme packets take a raw,
        # fully accounted fallback instead of propagating NaNs through features.
        extreme = bool(((((words.to(torch.int32) & 0x7F80) >> 7) >= 190)).any())
        if extreme:
            self.last_reconstruction = words
            return Packet(self.header(words.shape, start, kappa, True),
                          torch.empty(0, dtype=torch.int64, device=self.device), words.reshape(-1).clone())
        return self._run(words.shape, start, kappa, source=words)

    def decode(self, packet, *, validate=True):
        if validate:
            self.validate(packet)
        h = packet.header
        if h["raw"]:
            return packet.payload.reshape(h["shape"]).clone()
        return self._run(h["shape"], h["start"], h["kappa"], packet=packet)

    def calibrate(self, words):
        """Offline scales on the separate scale document using actual runtime arithmetic."""
        return self._run(words.shape, 0, 0., source=words.to(self.device, torch.int16), calibrate=True)

    def _run(self, shape, start, kappa, source=None, packet=None, calibrate=False):
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
                cur[:, pos] = self.rotate(xrec, g, positions, inverse=True)
                surprise += ((xrec.float() - m).abs() / (LN2X2 * spread)).sum(1)
                if source is None:
                    recovered[l][:, ch] = rec
                elif kappa:
                    self.last_reconstruction[l][:, ch] = rec
            f = torch.empty_like(cur); f[:, layer["perm"]] = cur
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
