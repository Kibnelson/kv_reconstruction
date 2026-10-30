"""Transfer codecs of the PD protocol: every one turns a request's KV words (int16 [layers, tokens, channels], keys
then values, on the sender's GPU) into tensors for the wire, and the receiver's tensors back into words, one layer at
a time through a sink (so the receiver installs layer by layer and never holds the whole cache twice).

  raw            BF16 as it is (16 bits per value; no codec work)
  ours-<mode>    T: our layer-wise one-step codec (kv_compare.layerwise), modes dense, kv (keys then values), 4g (4
                 head groups), hbh (head by head), rank256 (the reduced-rank predictor); only the used tail of each
                 stream segment travels (wire.compact_segments)
  splitzip       the authors' SplitZip codec (code book from WikiText-2 train KV, their protocol), per layer
  adacode        the authors' AdaCode CompressorBitMap, per layer for keys and values, code length and window from
                 their rule on the calibration KV

Constants come from one file written by scripts/pd_export_codecs.py (our predictors, RoPE tables, SplitZip's code
book, AdaCode's parameters), so sender and receiver build identical codecs. Our codec pads the token count up to a
multiple of 256 (zero words at the next positions; the receiver drops them) so any prompt length codes with the same
kernels as the measured 8K windows.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

OUR_MODES = {"dense": None, "kv": 2, "4g": 4, "hbh": 16, "rank256": "rank"}
PAD = 256
_TABLES = {}


def _cache_slice_tables():
    """Our codec builds its static coding tables (about 10 ms of Python) per codec object; a transfer makes one per
    request, so keep them per device."""
    from .. import gpu_coder as g
    if getattr(g.slice_tables, "_pd_cached", False):
        return
    orig = g.slice_tables

    def cached(device):
        k = str(device)
        if k not in _TABLES:
            _TABLES[k] = orig(device)
        return _TABLES[k]
    cached._pd_cached = True
    g.slice_tables = cached


class Raw:
    name = "raw"

    def encode(self, words, pos0=0):
        return {"words": words.contiguous()}, {"codec": self.name, "T": int(words.shape[1])}

    def decode(self, parts, meta, sink):
        w = parts["words"]
        for l in range(w.shape[0]):
            sink(l, w[l])


class Ours:
    """T in one mode. bundle: the dict of pd_export_codecs (tensors on any device; moved here)."""

    def __init__(self, bundle, mode, device):
        from .. import layerwise as lw
        from .. import fused as fu
        _cache_slice_tables()
        if mode not in OUR_MODES:
            raise ValueError(f"mode is one of {sorted(OUR_MODES)}")
        self.lw, self.fu = lw, fu
        self.name = f"ours-{mode}"
        self.mode = mode
        dev = torch.device(device)
        mv = lambda d: {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}
        self.base = mv(bundle["consts"])
        self.rank = mv(bundle["rank"]) if mode == "rank256" else None
        g = OUR_MODES[mode]
        self.group = mv(bundle["groups"][g]) if isinstance(g, int) else None
        self.cos, self.sin = bundle["cos"].to(dev), bundle["sin"].to(dev)
        self.C, self.hd, self.L, self.W = (int(bundle["meta"][k]) for k in ("C", "hd", "L", "W"))
        self.device = dev

    def params(self, T):
        """Token count padded, tokens per program, chunks and rank tiles: the choices of the measured sweep
        (kv_compare.context_sweep.measure_case), a function of the padded count only."""
        Tp = -(-T // PAD) * PAD
        ng = int(self.group.get("groups", 1)) if self.group is not None else 1
        TB, ch = (32, 2) if ng <= 4 else (16, 4)
        while Tp % (TB * ch):
            ch //= 2
        r = int(self.rank["rank"]) if self.rank is not None else 128
        tiles = (128, 128, 64, 4) if (Tp % 256 == 0 and r % 128 == 0 and self.W % 64 == 0) else (16, 16, 32, 4)
        return Tp, TB, ch, tiles

    def codec(self, T, pos0=0):
        Tp, TB, ch, tiles = self.params(T)
        if pos0 + Tp > self.cos.shape[0]:
            raise ValueError(f"positions {pos0}..{pos0 + Tp} beyond the exported RoPE tables ({self.cos.shape[0]})")
        base = self.group if self.group is not None else self.base
        kp = dict(base, cos=self.cos[pos0:pos0 + Tp].contiguous(), sin=self.sin[pos0:pos0 + Tp].contiguous())
        c = self.lw.LayerwiseOneStep(kp, self.C, self.hd, tokens_per_program=TB, mode="uniform", search=True,
                                     chunks=ch, rank=self.rank, requant="epilogue" if self.rank is not None else "kernel",
                                     rank_tiles=tiles)
        return c, kp, Tp

    def encode(self, words, pos0=0):
        from .wire import compact_segments
        L, T, W = words.shape
        codec, kp, Tp = self.codec(T, pos0)
        wp = words if Tp == T else torch.cat(
            [words, torch.zeros(L, Tp - T, W, dtype=words.dtype, device=words.device)], 1)
        wp = wp.contiguous()
        if self.rank is None and self.group is None:
            xall, prior = self.fu.FusedOneStep(kp, self.C, self.hd, tokens_per_program=32, rows=4, warps=4).model(wp)
        else:
            xall, prior = codec.model(wp)
        buf, starts = codec.encode(wp, xall, prior)
        _, cap = codec.layout(Tp)
        packed = compact_segments(buf, starts, cap, self.hd)
        del buf, xall, prior, wp
        return ({"words": packed, "starts": starts},
                {"codec": self.name, "T": int(T), "Tpad": int(Tp), "cap": int(cap), "pos0": int(pos0)})

    def decode(self, parts, meta, sink):
        from .wire import expand_segments
        T, pos0 = int(meta["T"]), int(meta.get("pos0", 0))
        codec, _, Tp = self.codec(T, pos0)
        _, cap = codec.layout(Tp)
        if cap != int(meta["cap"]) or Tp != int(meta["Tpad"]):
            raise ValueError("the sender's codec layout differs from the receiver's (different constants?)")
        starts = parts["starts"]
        buf = expand_segments(parts["words"], starts, cap, self.hd,
                              out=torch.empty(starts.numel() * cap, dtype=torch.int16, device=self.device))
        codec.decode(buf, starts, Tp, sink=lambda l, w: sink(l, w[:T]))
        del buf


class SplitZip:
    """The authors' ChunkLocalSplitZipGPU with the exported code book, one encode per layer."""
    name = "splitzip"

    def __init__(self, bundle, source, device):
        from ..mixed import load_splitzip
        self.sz, self.commit = load_splitzip(source, device)
        luts = bundle["splitzip"]
        for k in ("enc_lut", "enc_lut_marked", "dec_lut", "common_lut"):
            setattr(self.sz, k, luts[k].to(device))
        mod = sys.modules[type(self.sz).__module__]
        self.Encoded = mod.ChunkLocalGPUEncoded
        self.device = torch.device(device)

    def encode(self, words, pos0=0):
        parts, ns, nesc = {}, [], []
        for l in range(words.shape[0]):
            e = self.sz.encode(words[l].contiguous().view(torch.bfloat16))
            for f in ("pk", "sm", "chunk_id", "local_pos", "esc_val"):
                parts[f"{f}{l}"] = getattr(e, f)
            ns.append(int(e.n))
            nesc.append(int(e.n_esc))
            cs = int(e.chunk_size)
        return parts, {"codec": self.name, "T": int(words.shape[1]), "W": int(words.shape[2]), "n": ns,
                       "n_esc": nesc, "chunk_size": cs}

    def decode(self, parts, meta, sink):
        T, W = int(meta["T"]), int(meta["W"])
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        for l, (n, ne) in enumerate(zip(meta["n"], meta["n_esc"])):
            e = self.Encoded(parts[f"pk{l}"], parts[f"sm{l}"], empty, empty, parts[f"chunk_id{l}"],
                             parts[f"local_pos{l}"], parts[f"esc_val{l}"], n, ne, int(meta["chunk_size"]))
            sink(l, self.sz.decode(e).view(torch.int16).view(T, W))


def load_adacode(source):
    """AdaCode's CompressorBitMap, loaded by file (its package __init__ pulls in experiment scripts)."""
    from ..more_baselines import ADACODE_COMMIT, ADACODE_REPO, checkout
    src = checkout(ADACODE_REPO, ADACODE_COMMIT, source)
    spec = importlib.util.spec_from_file_location("adacode_compress_bitmap_merge", Path(src) / "compress_bitmap_merge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CompressorBitMap


class AdaCode:
    """The authors' CompressorBitMap per layer for keys and values; (right, nbits) exported from calibration."""
    name = "adacode"
    SCALARS = ("right", "nbits", "left", "left_nbits", "blocksize", "invalid_block_size", "n_elements", "n_exp_data")
    TENSORS = ("sign_mass_data", "exp_data", "invalid_exp", "block_base")

    def __init__(self, bundle, source, device, C):
        self.Compressor = load_adacode(source)
        self.params = bundle["adacode"]
        self.C = int(C)
        self.device = torch.device(device)

    def encode(self, words, pos0=0):
        parts, scal = {}, []
        L, T, W = words.shape
        for l in range(L):
            for j, part in enumerate((slice(0, self.C), slice(self.C, None))):
                right, nbits = self.params[l][j]
                x = words[l, :, part].contiguous().view(torch.bfloat16).reshape(-1)
                c = self.Compressor(right=right, nbits=nbits)
                c.compress(x)
                st = c.export_state()
                s = {k: st[k] for k in self.SCALARS}
                s["shape"] = list(st["original_shape"])
                for k in self.TENSORS:
                    s["has_" + k] = st[k] is not None
                    if st[k] is not None:
                        parts[f"{k}{l}{'kv'[j]}"] = st[k]
                scal.append(s)
        return parts, {"codec": self.name, "T": int(T), "W": int(W), "states": scal}

    def decode(self, parts, meta, sink):
        T, W, C = int(meta["T"]), int(meta["W"]), self.C
        L = len(meta["states"]) // 2
        for l in range(L):
            halves = []
            for j in range(2):
                s = meta["states"][2 * l + j]
                st = {k: s[k] for k in self.SCALARS}
                st["original_shape"] = torch.Size(s["shape"])
                st["device"] = self.device
                for k in self.TENSORS:
                    st[k] = parts[f"{k}{l}{'kv'[j]}"] if s["has_" + k] else None
                c = self.Compressor.from_state(st)
                halves.append(c.uncompress().view(torch.int16).reshape(T, -1))
            sink(l, torch.cat(halves, 1))


def build(bundle_path, device, names, splitzip_source=None, adacode_source=None, log=print):
    """{name: codec} for the requested names (raw, ours-<mode>, splitzip, adacode); a codec that cannot load is
    left out with a log line (its requests fail loudly at the sender)."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False) if bundle_path else None
    out, errors = {}, {}
    for n in names:
        try:
            if n == "raw":
                out[n] = Raw()
            elif n.startswith("ours-"):
                out[n] = Ours(bundle, n[5:], device)
            elif n == "splitzip":
                out[n] = SplitZip(bundle, splitzip_source, device)
            elif n == "adacode":
                out[n] = AdaCode(bundle, adacode_source, device, bundle["meta"]["C"])
            else:
                raise ValueError(f"unknown codec {n}")
        except Exception as exc:
            errors[n] = repr(exc)[:300]
            log(f"pd codecs: {n} not available ({errors[n]})")
    return out, errors, (bundle["meta"] if bundle else {})
