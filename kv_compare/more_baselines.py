"""The other bit-exact KV codecs of the literature (docs/lossless_related_work.md), run on our captured KV the way
SplitZip and ExANS are (kv_compare.context_sweep.measure_case): bits per value, a bit-exact round trip, and encode
and decode times on this GPU.

  nvCOMP          NVIDIA's GPU codecs (LZ4, Cascaded, Bitcomp, ANS, Zstd): SplitZip's Table 2 baselines. Python
                  package nvidia-nvcomp-cu12; the KV bytes of a stream as one buffer, default options except the
                  element type (2 bytes) for Cascaded and Bitcomp.
  AdaCode         EMNLP 2026 (github.com/lichangshiwhu/Adacode, ADACODE_COMMIT): its CompressorBitMap codec, unchanged,
                  per layer for keys and values. Code length (2, 3 or 4 bits) and exponent window from the authors'
                  rule (analysis_best_bits, reproduced below) on the calibration KV; "own data" picks them from the
                  coded KV itself (what the online adjuster tends to).
  exactkv         github.com/zyfyz666/exactkv (EXACTKV_COMMIT): its page encoder and tile decoder (Triton), per layer
                  for keys and values; bits from the buffers used, raw rows counted as used.
  IBP             EuroSys '26 (github.com/AKKamath/InvariantBitPacking, IBP_COMMIT): invariant bit packing per layer
                  (rows of 64-bit words); decode is its fetch from pinned host memory to the GPU, decompressing on the
                  way (the PCIe path it is built for), timed beside a plain copy of the raw bytes.
  exponent bounds computed, not codecs: 8 + the Huffman code length of the exponent (the DFloat11 / ZipNN format,
                  table excluded) and 8 + the exponent entropy, per layer for keys and values: what any coder of the
                  exponent alone, with sign and mantissa raw, cannot beat.

The authors' code is loaded from a checkout at the pinned commit (scripts/a5000_setup.sh clones them). Each method
reports an error row instead of stopping the sweep when its package or build is missing.
"""
from __future__ import annotations

import heapq
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

ADACODE_REPO = "https://github.com/lichangshiwhu/Adacode.git"
ADACODE_COMMIT = "d46cf6cbf7e5de497a7e79bf0d1766023a366854"
EXACTKV_REPO = "https://github.com/zyfyz666/exactkv.git"
EXACTKV_COMMIT = "d90376b84e8c116424298e98027995bdbbf0e8ee"
IBP_REPO = "https://github.com/AKKamath/InvariantBitPacking.git"
IBP_COMMIT = "b2f71003113defb4e387caf18843b241f6280ba9"
NVCOMP_ALGORITHMS = {"LZ4": {}, "Cascaded": {"data_type": "<u2"}, "Bitcomp": {"data_type": "<u2"}, "ANS": {},
                     "Zstd": {}}


def checkout(repo, commit, path, log=print):
    """A clone of repo at commit in path (cloned if missing)."""
    path = Path(path)
    if not (path / ".git").exists():
        log(f"  cloning {repo} into {path}")
        subprocess.run(["git", "clone", "-q", repo, str(path)], check=True)
    if subprocess.run(["git", "-C", str(path), "cat-file", "-e", commit], capture_output=True).returncode:
        subprocess.run(["git", "-C", str(path), "fetch", "-q", "origin", commit], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", commit], check=True)
    return path


def _exp_hist(w):
    """Counts of the 256 exponents of int16 BF16 words."""
    e = (w.reshape(-1).to(torch.int32) >> 7) & 0xFF
    return torch.bincount(e, minlength=256)


def huffman_lengths(counts):
    """Huffman code lengths of the symbols with nonzero counts (a list of ints) -> {symbol: length}."""
    sym = [(int(c), i) for i, c in enumerate(counts) if c > 0]
    if len(sym) == 1:
        return {sym[0][1]: 1}
    heap = [(c, k, [s]) for k, (c, s) in enumerate(sym)]
    heapq.heapify(heap)
    length = {s: 0 for _, s in sym}
    k = len(heap)
    while len(heap) > 1:
        c1, _, a = heapq.heappop(heap)
        c2, _, b = heapq.heappop(heap)
        for s in a + b:
            length[s] += 1
        heapq.heappush(heap, (c1 + c2, k, a + b))
        k += 1
    return length


def exponent_bounds(words, C, spans):
    """Bits per value of 8 + Huffman(exponent) and 8 + H(exponent), each layer's keys and values with their own
    statistics, over the spans (each span its own table, like a stream)."""
    huff = ent = 0.0
    n = 0
    L = words.shape[0]
    for a, b in spans:
        for l in range(L):
            for part in (slice(0, C), slice(C, None)):
                h = _exp_hist(words[l, a:b, part]).cpu().tolist()
                tot = sum(h)
                lens = huffman_lengths(h)
                huff += sum(h[s] * ln for s, ln in lens.items())
                ent += -sum(c * math.log2(c / tot) for c in h if c)
                n += tot
    return {"huffman": 8 + huff / n, "entropy": 8 + ent / n}


# ------------------------------------------------------------------------------------------------------------ nvCOMP
class Nvcomp:
    """nvCOMP's GPU codecs through its Python package (from nvidia import nvcomp)."""

    def __init__(self, device, algorithms=tuple(NVCOMP_ALGORITHMS)):
        from nvidia import nvcomp
        self.nv = nvcomp
        self.device = device
        self.stream = torch.cuda.current_stream(device)
        self.codecs = {}
        self.errors = {}
        for a in algorithms:
            try:
                self.codecs[a] = nvcomp.Codec(algorithm=a, device_id=device.index or 0,
                                              cuda_stream=self.stream.cuda_stream, **NVCOMP_ALGORITHMS.get(a, {}))
            except Exception as exc:                 # an algorithm this nvCOMP build does not offer
                self.errors[a] = repr(exc)[:300]
        self.version = getattr(nvcomp, "__version__", None)

    def measure(self, alg, words, spans, timer):
        codec = self.codecs[alg]
        ok, nbytes, te, td = True, 0, 0.0, 0.0
        for a, b in spans:
            raw = words[:, a:b].contiguous().view(torch.uint8).reshape(-1)
            arr = self.nv.as_array(raw, cuda_stream=self.stream.cuda_stream)
            enc = codec.encode(arr)
            dec = codec.decode(enc)
            back = torch.from_dlpack(dec).reshape(-1)
            ok = ok and back.numel() == raw.numel() and bool(torch.equal(back.view(torch.uint8), raw))
            nbytes += int(enc.buffer_size)
            te += timer(lambda: codec.encode(arr))
            td += timer(lambda: codec.decode(enc))
            del raw, arr, enc, dec, back
        return {"bit_exact": ok, "bits": 8 * nbytes / (words.numel()), "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
                "streams": len(spans), "nvcomp": self.version}


# ----------------------------------------------------------------------------------------------------------- AdaCode
def adacode_best_bits(data_freq):
    """The authors' parameter rule (Adacode float_cache_utils.analysis_best_bits, ADACODE_COMMIT): for 2, 3 and 4
    bits, the 2^n - 1 most frequent exponents are coded and the rest stored as 8-bit outliers; the cheapest n wins,
    and the window ends at the largest exponent among those coded. Returns (right, nbits)."""
    data_freq_list = [int(v) for v in data_freq]
    total_numbers = sum(data_freq_list)
    indexed = sorted(enumerate(data_freq_list), key=lambda x: x[1], reverse=True)
    best_compressed_memory = total_numbers * 8
    best_indices, best_nbit = [], 0
    for nbit in [2, 3, 4]:
        topk = indexed[:2 ** nbit - 1]
        outlier_values = total_numbers - sum(v for _, v in topk)
        compressed_memory = total_numbers * nbit + outlier_values * 8
        if compressed_memory < best_compressed_memory:
            best_compressed_memory = compressed_memory
            best_indices = [i for i, _ in topk]
            best_nbit = nbit
    best_indices.sort()
    return best_indices[-1], best_nbit


class AdaCode:
    """AdaCode's codec (compress_bitmap_merge.CompressorBitMap) loaded from the authors' checkout."""

    def __init__(self, source, log=print):
        src = checkout(ADACODE_REPO, ADACODE_COMMIT, source, log)
        # load the codec module by file: the package's __init__ pulls in its experiment scripts
        spec = importlib.util.spec_from_file_location("adacode_compress_bitmap_merge", src / "compress_bitmap_merge.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.Compressor = mod.CompressorBitMap
        self.params = None

    def calibrate(self, cal_words, C):
        """Per layer, keys and values: (right, nbits) from the calibration KV (int16 [L, T, W])."""
        self.params = [[adacode_best_bits(_exp_hist(cal_words[l, :, part]).cpu().tolist())
                        for part in (slice(0, C), slice(C, None))] for l in range(cal_words.shape[0])]
        return self.params

    def measure(self, words, spans, C, timer, own=False):
        L = words.shape[0]
        parts = (slice(0, C), slice(C, None))
        ok, nbytes, te, td = True, 0, 0.0, 0.0
        chosen = {}
        for a, b in spans:
            xs, comps = [], []
            for l in range(L):
                for j, part in enumerate(parts):
                    x = words[l, a:b, part].contiguous().view(torch.bfloat16).reshape(-1)
                    right, nbits = (adacode_best_bits(_exp_hist(x.view(torch.int16)).cpu().tolist()) if own
                                    else self.params[l][j])
                    chosen[f"{l}{'kv'[j]}"] = (right, nbits)
                    c = self.Compressor(right=right, nbits=nbits)
                    c.compress(x)
                    xs.append(x)
                    comps.append(c)

            def encode():
                for x, c in zip(xs, comps):
                    c.compress(x)

            def decode():
                return [c.uncompress() for c in comps]
            outs = decode()
            ok = ok and all(bool(torch.equal(y.view(torch.int16).reshape(-1), x.view(torch.int16))) for x, y in zip(xs, outs))
            nbytes += sum(int(c.get_memory()["compressed_memory"]) for c in comps)
            te += timer(encode)
            td += timer(decode)
            del xs, comps, outs
        nb = sorted({v[1] for v in chosen.values()})
        return {"bit_exact": ok, "bits": 8 * nbytes / words.numel(), "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
                "streams": len(spans), "code_bits": nb, "launches": 2 * L * len(spans)}


# ----------------------------------------------------------------------------------------------------------- exactkv
class ExactKV:
    """exactkv's page encoder and decoder (Triton) from the authors' checkout; head_dim 128 only."""

    def __init__(self, source, log=print):
        src = checkout(EXACTKV_REPO, EXACTKV_COMMIT, source, log)
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from exactkv.cache import CompressedKV
        from exactkv.kernels import decode_to_bf16, P
        self.CompressedKV, self.decode_to_bf16, self.P = CompressedKV, decode_to_bf16, P

    def measure(self, words, spans, C, hd, timer):
        L = words.shape[0]
        H = C // hd
        ok, nbytes, te, td = True, 0, 0.0, 0.0
        for a, b in spans:
            T = b - a
            if T % self.P:
                raise ValueError(f"exactkv codes pages of {self.P} tokens; {T} tokens")
            items = []
            for l in range(L):
                for part in (slice(0, C), slice(C, None)):
                    x = words[l, a:b, part].contiguous().view(torch.bfloat16).view(1, T, H, hd)
                    kv = self.CompressedKV(1, T, H, hd, device=words.device)
                    kv.prefill_slot(0, x)
                    items.append((x, kv))

            def encode():
                st = (items[0][0].stride(0), self.P * items[0][0].stride(1), items[0][0].stride(1), items[0][0].stride(2))
                for x, kv in items:
                    kv.cnt.zero_()
                    kv._encode(x, st, 0, 0, 1, T // self.P)

            def decode():
                return [self.decode_to_bf16(kv.buffers(), T) for _, kv in items]
            outs = decode()
            ok = ok and all(bool(torch.equal(y.view(torch.int16), x.view(torch.int16))) for (x, _), y in zip(items, outs))
            for _, kv in items:
                used_raw = int(kv.cnt.item())
                nbytes += sum(t.numel() * t.element_size() for t in (kv.sm, kv.code, kv.base, kv.rowmask, kv.rowoff))
                nbytes += used_raw * hd * 2
            te += timer(encode)
            td += timer(decode)
            del items, outs
        return {"bit_exact": ok, "bits": 8 * nbytes / words.numel(), "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
                "streams": len(spans), "launches": 2 * L * len(spans)}


# --------------------------------------------------------------------------------------------------------------- IBP
class IBP:
    """Invariant bit packing (the authors' ibp package, built from IBP_COMMIT by scripts/a5000_setup.sh)."""

    def __init__(self):
        import ibp
        self.ibp = ibp

    def measure(self, words, spans, device, timer):
        ibp = self.ibp
        L = words.shape[0]
        ok, nbytes, t_enc, td, traw = True, 0, 0.0, 0.0, 0.0
        for a, b in spans:
            for l in range(L):
                data = words[l, a:b].contiguous().view(torch.int64)          # rows of 64-bit words, as their API
                t0 = time.perf_counter()
                mask, bitval = ibp.preprocess(data)
                sizes = ibp.get_compress_size(data, mask, bitval)
                torch.cuda.synchronize(device)
                comp_len = ibp.get_single_compress_len(data, mask, bitval)
                host = data.cpu().pin_memory()
                bitmask = ibp.compress_inplace(host, mask, bitval)
                torch.cuda.synchronize(device)
                t_enc += time.perf_counter() - t0
                nbytes += int(sizes.sum().item()) + sum(t.numel() * t.element_size() for t in (mask, bitval, bitmask))
                out = torch.empty_like(data)

                def fetch():
                    ibp.decompress_fetch(host, mask, bitval, bitmask, device, comp_len=comp_len, output_tensor=out)
                fetch()
                torch.cuda.synchronize(device)
                ok = ok and bool(torch.equal(out, data))
                raw_host = data.cpu().pin_memory()
                td += timer(fetch)
                traw += timer(lambda: out.copy_(raw_host, non_blocking=True))
                del data, mask, bitval, sizes, host, bitmask, out, raw_host
        return {"bit_exact": ok, "bits": 8 * nbytes / words.numel(), "encode_ms": 1e3 * t_enc, "decode_ms": 1e3 * td,
                "raw_copy_ms": 1e3 * traw, "streams": len(spans),
                "path": "pinned host memory -> GPU over PCIe, decompressing on the way (decode includes the copy)"}


def build(args_like, device, cal_words, C, log=print):
    """The available extra codecs: {name: object}, and {name: error} for those that could not load."""
    have, errors = {}, {}
    want = set(args_like.extra_baselines)
    root = Path(args_like.baseline_sources)
    if "nvcomp" in want:
        try:
            nv = Nvcomp(device)
            have["nvcomp"] = nv
            errors.update({f"nvCOMP {a}": e for a, e in nv.errors.items()})
        except Exception as exc:
            errors["nvcomp"] = repr(exc)[:300]
    if "adacode" in want:
        try:
            ada = AdaCode(root / "adacode", log)
            ada.calibrate(cal_words, C)
            have["adacode"] = ada
        except Exception as exc:
            errors["adacode"] = repr(exc)[:300]
    if "exactkv" in want:
        try:
            have["exactkv"] = ExactKV(root / "exactkv", log)
        except Exception as exc:
            errors["exactkv"] = repr(exc)[:300]
    if "ibp" in want:
        try:
            have["ibp"] = IBP()
        except Exception as exc:
            errors["ibp"] = repr(exc)[:300]
    for k, e in errors.items():
        log(f"  extra baseline {k} not available ({e})")
    return have, errors
