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
  exactkv        the authors' exactkv page encoder and tile decoder (Triton), per layer for keys and values: its
                 buffers and the raw rows it used
  nvcomp-<alg>   nVIDIA's nvCOMP (Python package, CUDA 13 build in vLLM's environment): the request's bytes as one
                 buffer, algorithms ans, zstd, bitcomp, lz4, cascaded, gdeflate (default options; 2-byte elements
                 for bitcomp and cascaded)
  exans          the authors' ExANS library (libopenlake_expans.so, built by scripts/pd_setup.sh) through ctypes as
                 in their own vLLM connector: blocks of 488 KiB, each block's record cut to its used bytes; blocks
                 that do not fit and the tail travel raw
  t2-<mode>      T2 (kv_compare.t2): T in that mode with a constant set picked per request (T's int8 product on 1,024
                 sampled tokens of layers 1-2) and layer 0 recomputed by the receiver from the token ids and the
                 sender's prefill steps (kv_compare.pd.layer0); when the sender's own recompute differs from its
                 cache, layer 0 is coded as in T (the header says which)
  ibp            the authors' Invariant Bit Packing, for the CPU tier only (kv_compare.pd.tier): compressed in pinned
                 host memory, decompressed on the way to the GPU

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


_ON = {}


def _on(t, dev):
    """t on dev, one copy per (tensor, device) for the process: every mode and constant set of T and T2 shares the
    same base constants and RoPE tables (the bundle stays loaded, so the ids stay valid)."""
    k = (id(t), str(dev))
    hit = _ON.get(k)
    if hit is None or hit[0] is not t:
        hit = _ON[k] = (t, t.to(dev))
    return hit[1]


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
        mv = lambda d: {k: (_on(v, dev) if torch.is_tensor(v) else v) for k, v in d.items()}
        self.base = mv(bundle["consts"])
        self.rank = mv(bundle["rank"]) if mode == "rank256" else None
        g = OUR_MODES[mode]
        self.group = mv(bundle["groups"][g]) if isinstance(g, int) else None
        self.cos, self.sin = _on(bundle["cos"], dev), _on(bundle["sin"], dev)
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

    def codec(self, T, pos0=0, cls=None):
        """The codec for T tokens at positions pos0.. (cls: LayerwiseOneStep, or kv_compare.t2.Layer0Recomputed)."""
        Tp, TB, ch, tiles = self.params(T)
        if pos0 + Tp > self.cos.shape[0]:
            raise ValueError(f"positions {pos0}..{pos0 + Tp} beyond the exported RoPE tables ({self.cos.shape[0]})")
        base = self.group if self.group is not None else self.base
        kp = dict(base, cos=self.cos[pos0:pos0 + Tp].contiguous(), sin=self.sin[pos0:pos0 + Tp].contiguous())
        cls = cls or self.lw.LayerwiseOneStep
        c = cls(kp, self.C, self.hd, tokens_per_program=TB, mode="uniform", search=True,
                chunks=ch, rank=self.rank, requant="epilogue" if self.rank is not None else "kernel",
                rank_tiles=tiles)
        return c, kp, Tp

    def model(self, codec, kp, wp):
        """xall and prior of the padded words (the fused one-step kernel in dense mode)."""
        if self.rank is None and self.group is None:
            return self.fu.FusedOneStep(kp, self.C, self.hd, tokens_per_program=32, rows=4, warps=4).model(wp)
        return codec.model(wp)

    def encode(self, words, pos0=0):
        from .wire import compact_segments
        L, T, W = words.shape
        codec, kp, Tp = self.codec(T, pos0)
        wp = words if Tp == T else torch.cat(
            [words, torch.zeros(L, Tp - T, W, dtype=words.dtype, device=words.device)], 1)
        wp = wp.contiguous()
        xall, prior = self.model(codec, kp, wp)
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


def _pad_tokens(words, Tp):
    L, T, W = words.shape
    if Tp == T:
        return words.contiguous()
    return torch.cat([words, torch.zeros(L, Tp - T, W, dtype=words.dtype, device=words.device)], 1).contiguous()


class ExactKV:
    """The authors' exactkv (CompressedKV.prefill_slot, decode_to_bf16) per layer for keys and values; tokens padded
    to its 16-token pages (zero rows, dropped by the receiver)."""
    name = "exactkv"
    BUFFERS = ("sm", "code", "base", "rowmask", "rowoff")

    def __init__(self, source, device, hd):
        from ..more_baselines import EXACTKV_COMMIT, EXACTKV_REPO, checkout
        src = checkout(EXACTKV_REPO, EXACTKV_COMMIT, source)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from exactkv.cache import CompressedKV
        from exactkv.kernels import P, decode_to_bf16
        if int(hd) != 128:
            raise ValueError(f"exactkv codes head_dim 128 only, not {hd}")
        self.CompressedKV, self.decode_to_bf16, self.P = CompressedKV, decode_to_bf16, int(P)
        self.hd = int(hd)
        self.device = torch.device(device)

    def encode(self, words, pos0=0):
        L, T, W = words.shape
        C = W // 2
        H = C // self.hd
        Tp = -(-T // self.P) * self.P
        wp = _pad_tokens(words, Tp)
        parts, used = {}, []
        for l in range(L):
            for j, part in enumerate((slice(0, C), slice(C, None))):
                x = wp[l, :, part].contiguous().view(torch.bfloat16).view(1, Tp, H, self.hd)
                kv = self.CompressedKV(1, Tp, H, self.hd, device=words.device)
                kv.prefill_slot(0, x)
                n = int(kv.cnt.item())
                tag = f"{l}{'kv'[j]}"
                for k in self.BUFFERS:
                    parts[k + tag] = getattr(kv, k)
                parts["raw" + tag] = kv.raw[:max(n, 1)]
                used.append(n)
        return parts, {"codec": self.name, "T": int(T), "Tpad": int(Tp), "H": int(H), "raw_rows": used}

    def decode(self, parts, meta, sink):
        T, Tp, H = int(meta["T"]), int(meta["Tpad"]), int(meta["H"])
        L = len(meta["raw_rows"]) // 2
        for l in range(L):
            halves = []
            for j in range(2):
                tag = f"{l}{'kv'[j]}"
                c = {k: parts[k + tag] for k in self.BUFFERS}
                c.update(shape=(1, Tp, H, self.hd), raw=parts["raw" + tag])
                y = self.decode_to_bf16(c, Tp)
                halves.append(y.view(torch.int16).reshape(Tp, H * self.hd)[:T])
            sink(l, torch.cat(halves, 1))


NVCOMP = {"ans": ("ANS", {}), "zstd": ("Zstd", {}), "bitcomp": ("Bitcomp", {"data_type": "<u2"}),
          "lz4": ("LZ4", {}), "cascaded": ("Cascaded", {"data_type": "<u2"}), "gdeflate": ("GDeflate", {})}


class Nvcomp:
    """nvCOMP's Codec (from nvidia import nvcomp) on the request's bytes as one buffer; nvCOMP's own header in the
    stream (the receiver decodes the bytes as they come)."""

    def __init__(self, alg, device):
        from nvidia import nvcomp
        if alg not in NVCOMP:
            raise ValueError(f"nvCOMP algorithm is one of {sorted(NVCOMP)}")
        self.nv = nvcomp
        self.name = f"nvcomp-{alg}"
        self.device = torch.device(device)
        name, opts = NVCOMP[alg]
        self.stream = torch.cuda.current_stream(self.device)
        self.codec = nvcomp.Codec(algorithm=name, device_id=self.device.index or 0,
                                  cuda_stream=self.stream.cuda_stream, **opts)
        self.version = getattr(nvcomp, "__version__", None)

    def encode(self, words, pos0=0):
        L, T, W = words.shape
        raw = words.contiguous().view(torch.uint8).reshape(-1)
        enc = self.codec.encode(self.nv.as_array(raw, cuda_stream=self.stream.cuda_stream))
        size = getattr(enc, "buffer_size", None)
        n = int(size if size is not None else enc.size * enc.item_size)
        coded = torch.from_dlpack(enc).reshape(-1).view(torch.uint8)[:n].clone()
        return {"coded": coded}, {"codec": self.name, "T": int(T), "L": int(L), "W": int(W), "nvcomp": self.version}

    def decode(self, parts, meta, sink):
        L, T, W = int(meta["L"]), int(meta["T"]), int(meta["W"])
        arr = self.nv.as_array(parts["coded"].contiguous(), cuda_stream=self.stream.cuda_stream)
        dec = torch.from_dlpack(self.codec.decode(arr)).contiguous().reshape(-1).view(torch.uint8)
        if dec.numel() != L * T * W * 2:
            raise ValueError(f"nvCOMP decoded {dec.numel()} bytes, expected {L * T * W * 2}")
        w = dec.view(torch.int16).view(L, T, W)
        for l in range(L):
            sink(l, w[l])


class ExANS:
    """The authors' ExANS through ctypes (kv_compare.exans_official's loader and constants): the request's words in
    memory order cut into blocks; each block that fits its record travels as the record's used bytes, the others
    and the tail raw. info: the build information JSON scripts/pd_setup.sh writes."""
    name = "exans"

    def __init__(self, info_path, device, max_blocks=1 << 16):
        import ctypes
        import json as _json
        from .. import exans_official as ex
        self.ex, self.ct = ex, ctypes
        info = _json.loads(Path(info_path).read_text())
        for d in ex._lib_dirs(Path(info["nvcomp_root"]), info["cuda_home"]):   # the CUDA 12 runtime it links,
            for so in sorted(d.glob("libcudart.so.12*")):                       # beside vLLM's CUDA 13 torch
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
        self.lib = ex._load(info)
        self.device = torch.device(device)
        self.handles = {}                            # capacity in blocks -> handle (made as requests need them)
        self.max_blocks = int(max_blocks)
        self.version = info.get("commit")
        self._handle(64)                             # fails here, not at the first request, if the library cannot

    def _handle(self, nb):
        """A handle for at least nb blocks (capacities 64, 128, ...: the library's scratch grows with it)."""
        ct = self.ct
        cap = 64
        while cap < nb:
            cap *= 2
        h = self.handles.get(cap)
        if h is None:
            h = ct.c_void_p()
            seg = (ct.c_uint32 * 1)(self.ex.BLOCK_BYTES)
            st = self.lib.openlake_expans_create(self.ex.BLOCK_BYTES, self.ex.RECORD_BYTES, seg, 1, cap, ct.byref(h))
            if st:
                raise RuntimeError("ExANS create failed: " + (self.lib.openlake_expans_last_error(h) or b"?").decode())
            self.handles[cap] = h
        return h

    def _u(self, kind, xs):
        return (kind * len(xs))(*xs)

    def encode(self, words, pos0=0):
        ex, ct = self.ex, self.ct
        L, T, W = words.shape
        flat = words.contiguous().view(-1)
        nb = flat.numel() // ex.BLOCK_WORDS
        if nb > self.max_blocks:
            raise ValueError(f"{nb} ExANS blocks, more than the handle's {self.max_blocks}")
        tail = flat[nb * ex.BLOCK_WORDS:]
        parts = {"tail": tail.clone() if tail.numel() else torch.zeros(1, dtype=torch.int16, device=flat.device)}
        meta = {"codec": self.name, "T": int(T), "L": int(L), "W": int(W), "blocks": int(nb), "tail": int(tail.numel())}
        if nb == 0:
            meta.update(fits=[], sizes=[])
            return parts, meta
        data = flat[:nb * ex.BLOCK_WORDS].view(nb, ex.BLOCK_WORDS)
        records = torch.zeros(nb * ex.RECORD_BYTES, dtype=torch.uint8, device=flat.device)
        fits = (ct.c_uint8 * nb)()
        addrs = self._u(ct.c_uint64, [data.data_ptr() + i * ex.BLOCK_BYTES for i in range(nb)])
        sizes = self._u(ct.c_uint32, [ex.BLOCK_BYTES] * nb)
        offs = self._u(ct.c_uint32, list(range(nb + 1)))
        stream = torch.cuda.current_stream(flat.device)
        handle = self._handle(nb)
        st = self.lib.openlake_expans_encode(handle, addrs, sizes, offs, nb, records.data_ptr(), fits,
                                             stream.cuda_stream)
        if st:
            raise RuntimeError("ExANS encode failed: " + self.lib.openlake_expans_last_error(handle).decode())
        hdr = records.view(nb, ex.RECORD_BYTES)[:, :ex.HEADER_BYTES].cpu()
        used = hdr[:, 20:24].contiguous().view(torch.int32).flatten().tolist()
        fit = [bool(fits[i]) for i in range(nb)]
        rec = records.view(nb, ex.RECORD_BYTES)
        kept = [rec[i, :used[i]] for i in range(nb) if fit[i]]
        raw = [data[i] for i in range(nb) if not fit[i]]
        parts["records"] = torch.cat(kept) if kept else torch.zeros(1, dtype=torch.uint8, device=flat.device)
        if raw:
            parts["raw"] = torch.cat(raw)
        meta.update(fits=fit, sizes=[int(used[i]) if fit[i] else 0 for i in range(nb)])
        return parts, meta

    def decode(self, parts, meta, sink):
        ex, ct = self.ex, self.ct
        L, T, W, nb = int(meta["L"]), int(meta["T"]), int(meta["W"]), int(meta["blocks"])
        out = torch.empty(L * T * W, dtype=torch.int16, device=self.device)
        if nb:
            fit, sizes = meta["fits"], meta["sizes"]
            blocks = out[:nb * ex.BLOCK_WORDS].view(nb, ex.BLOCK_WORDS)
            idx = [i for i in range(nb) if fit[i]]
            if idx:
                records = torch.zeros(nb * ex.RECORD_BYTES, dtype=torch.uint8, device=self.device)
                rec = records.view(nb, ex.RECORD_BYTES)
                o = 0
                for i in idx:
                    rec[i, :sizes[i]] = parts["records"][o:o + sizes[i]]
                    o += sizes[i]
                oaddrs = self._u(ct.c_uint64, [blocks.data_ptr() + i * ex.BLOCK_BYTES for i in idx])
                handle = self._handle(nb)
                st = self.lib.openlake_expans_decode(handle, records.data_ptr(), self._u(ct.c_uint32, idx),
                                                     len(idx), oaddrs, self._u(ct.c_uint32, [ex.BLOCK_BYTES] * len(idx)),
                                                     self._u(ct.c_uint32, list(range(len(idx) + 1))),
                                                     torch.cuda.current_stream(self.device).cuda_stream)
                if st:
                    raise RuntimeError("ExANS decode failed: " + self.lib.openlake_expans_last_error(handle).decode())
            raw = [i for i in range(nb) if not fit[i]]
            if raw:
                blocks[torch.as_tensor(raw, device=self.device)] = parts["raw"].view(len(raw), ex.BLOCK_WORDS)
        if int(meta["tail"]):
            out[nb * ex.BLOCK_WORDS:] = parts["tail"][:int(meta["tail"])]
        w = out.view(L, T, W)
        for l in range(L):
            sink(l, w[l])


class T2Codec:
    """T2 in one mode over the bundle's constant sets (bundle["t2"]): the set picked per request, layer 0 recomputed
    by the receiver when the sender's check passes. ctx (kv_compare.pd.layer0.Layer0Context): the request's token
    ids and prefill steps and the engine's own layer-0 modules; without one, layer 0 is coded."""
    needs_ctx = True

    def __init__(self, bundle, mode, device):
        from .. import t2
        t = bundle.get("t2")
        if not t:
            raise ValueError("no T2 constant sets in the codec file (scripts/pd_export_codecs.py --t2-sets)")
        self.t2 = t2
        self.name = f"t2-{mode}"
        self.mode = mode
        self.names = list(t["order"])
        self.sets = {n: Ours(dict(bundle, consts=t["sets"][n]["consts"], rank=t["sets"][n].get("rank"),
                                  groups=t["sets"][n].get("groups", {})), mode, device) for n in self.names}
        first = self.sets[self.names[0]]
        self.C, self.hd, self.L, self.W, self.device = first.C, first.hd, first.L, first.W, first.device
        self.pick_tokens, self.pick_layers = int(t.get("pick_tokens", 1024)), tuple(t.get("pick_layers", (1, 2)))

    def pick(self, words, pos0):
        L, T, W = words.shape
        cos, sin = self.sets[self.names[0]].cos, self.sets[self.names[0]].sin
        sets = [dict(self.sets[n].base, cos=cos[pos0:pos0 + T], sin=sin[pos0:pos0 + T]) for n in self.names]
        rows = self.t2.sample_rows(T, self.pick_tokens).to(words.device)
        cost = self.t2.proxy_costs(sets, words, self.C, self.hd, self.pick_layers, rows)
        return self.names[min(range(len(cost)), key=cost.__getitem__)], cost

    def encode(self, words, pos0=0, ctx=None):
        from .wire import compact_segments
        L, T, W = words.shape
        name, cost = self.pick(words, pos0) if len(self.names) > 1 else (self.names[0], [0.0])
        o = self.sets[name]
        ok = bool(ctx is not None and ctx.covers(pos0, T) and ctx.matches(words[0], pos0))
        codec, kp, Tp = o.codec(T, pos0, cls=self.t2.Layer0Recomputed if ok else None)
        wp = _pad_tokens(words, Tp)
        xall, prior = o.model(codec, kp, wp)
        buf, starts = codec.encode(wp, xall, prior)
        _, cap = codec.layout(Tp)
        packed = compact_segments(buf, starts, cap, self.hd)
        del buf, xall, prior, wp
        meta = {"codec": self.name, "set": name, "layer0": "recomputed" if ok else "coded", "T": int(T),
                "Tpad": int(Tp), "cap": int(cap), "pos0": int(pos0),
                "steps": ctx.steps_for(pos0, T) if ok else None,
                "layer0_check": None if ctx is None else ctx.last_check}
        return {"words": packed, "starts": starts}, meta

    def decode(self, parts, meta, sink, ctx=None):
        from .wire import expand_segments
        T, pos0 = int(meta["T"]), int(meta.get("pos0", 0))
        o = self.sets[meta["set"]]
        rec = meta["layer0"] == "recomputed"
        codec, _, Tp = o.codec(T, pos0, cls=self.t2.Layer0Recomputed if rec else None)
        _, cap = codec.layout(Tp)
        if cap != int(meta["cap"]) or Tp != int(meta["Tpad"]):
            raise ValueError("the sender's codec layout differs from the receiver's (different constants?)")
        starts = parts["starts"]
        buf = expand_segments(parts["words"], starts, cap, self.hd,
                              out=torch.empty(starts.numel() * cap, dtype=torch.int16, device=self.device))
        if rec:
            if ctx is None:
                raise ValueError("the stream's layer 0 is recomputed by the receiver, which has no layer-0 context")
            l0 = torch.zeros(Tp, self.W, dtype=torch.int16, device=self.device)
            l0[:T] = ctx.layer0(pos0, T, meta["steps"])
            codec.decode(buf, starts, Tp, l0, sink=lambda l, w: sink(l, w[:T]))
        else:
            codec.decode(buf, starts, Tp, sink=lambda l, w: sink(l, w[:T]))
        del buf


class IBP:
    """The authors' Invariant Bit Packing (package ibp) for the CPU tier: per layer, rows of 64-bit words
    preprocessed on the GPU, compressed in place in pinned host memory, fetched and decompressed by the GPU (its
    PCIe path). Not a wire codec: tier_store / tier_load only. The host buffer keeps its raw size (compression is
    in place), so the tier counts that size."""
    name = "ibp"
    tier_only = True

    def __init__(self, device):
        import ibp
        self.ibp = ibp
        self.device = torch.device(device)

    def tier_store(self, words):
        ibp = self.ibp
        L, T, W = words.shape
        layers, nbytes, coded = [], 0, 0
        for l in range(L):
            data = words[l].contiguous().view(torch.int64)
            mask, bitval = ibp.preprocess(data)
            sizes = ibp.get_compress_size(data, mask, bitval)
            comp_len = ibp.get_single_compress_len(data, mask, bitval)
            host = data.cpu().pin_memory()
            bitmask = ibp.compress_inplace(host, mask, bitval)
            layers.append((host, mask, bitval, bitmask, comp_len))
            nbytes += host.numel() * 8 + sum(t.numel() * t.element_size() for t in (mask, bitval, bitmask))
            coded += int(sizes.sum().item())
        return {"layers": layers, "T": int(T), "W": int(W)}, nbytes, coded

    def tier_load(self, state, sink):
        T, W = state["T"], state["W"]
        for l, (host, mask, bitval, bitmask, comp_len) in enumerate(state["layers"]):
            out = torch.empty(T, W // 4, dtype=torch.int64, device=self.device)
            self.ibp.decompress_fetch(host, mask, bitval, bitmask, self.device, comp_len=comp_len, output_tensor=out)
            sink(l, out.view(torch.int16).view(T, W))


def build(bundle_path, device, names, splitzip_source=None, adacode_source=None, log=print, exactkv_source=None,
          exans_info=None):
    """{name: codec} for the requested names (raw, ours-<mode>, t2-<mode>, splitzip, adacode, exactkv,
    nvcomp-<alg>, exans, ibp); a codec that cannot load is left out with a log line (its requests fail loudly at
    the sender)."""
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False) if bundle_path else None
    out, errors = {}, {}
    for n in names:
        try:
            if n == "raw":
                out[n] = Raw()
            elif n.startswith("ours-"):
                out[n] = Ours(bundle, n[5:], device)
            elif n.startswith("t2-"):
                out[n] = T2Codec(bundle, n[3:], device)
            elif n == "splitzip":
                out[n] = SplitZip(bundle, splitzip_source, device)
            elif n == "adacode":
                out[n] = AdaCode(bundle, adacode_source, device, bundle["meta"]["C"])
            elif n == "exactkv":
                out[n] = ExactKV(exactkv_source, device, bundle["meta"]["hd"])
            elif n.startswith("nvcomp-"):
                out[n] = Nvcomp(n[7:], device)
            elif n == "exans":
                if not exans_info:
                    raise ValueError("no exans_info (scripts/pd_setup.sh builds ExANS and writes it)")
                out[n] = ExANS(exans_info, device)
            elif n == "ibp":
                out[n] = IBP(device)
            else:
                raise ValueError(f"unknown codec {n}")
        except Exception as exc:
            errors[n] = repr(exc)[:300]
            log(f"pd codecs: {n} not available ({errors[n]})")
    return out, errors, (bundle["meta"] if bundle else {})
