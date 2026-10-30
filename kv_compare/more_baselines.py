"""The other bit-exact KV codecs of the literature (docs/lossless_related_work.md), run on our captured KV the way
SplitZip and ExANS are (kv_compare.context_sweep.measure_case): bits per value, a bit-exact round trip, and encode
and decode times on this GPU.

  nvCOMP          NVIDIA's GPU codecs (LZ4, Cascaded, Bitcomp, ANS, Zstd): SplitZip's Table 2 baselines. Python
                  package nvidia-nvcomp-cu12; the KV bytes of a stream as one buffer, default options except the
                  element type (2 bytes) for Cascaded and Bitcomp. Run in a child process (IsolatedCodec): the
                  package crashed whole runs at random inside its encode.
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
import signal
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

    def measure_span(self, alg, raw, timer):
        """One stream, raw: its bytes (uint8 [n] on the device) -> (bit_exact, coded bytes, encode s, decode s,
        what differed or None)."""
        codec = self.codecs[alg]
        arr = self.nv.as_array(raw, cuda_stream=self.stream.cuda_stream)
        enc = codec.encode(arr)
        dec = codec.decode(enc)
        # the decoded array keeps the codec's element type (2 bytes for Cascaded and Bitcomp): compare bytes
        t = torch.from_dlpack(dec)
        back = t.contiguous().reshape(-1).view(torch.uint8)
        same = back.numel() == raw.numel() and bool(torch.equal(back, raw))
        diag = None
        if not same:                                   # what differed, for the report
            n = min(back.numel(), raw.numel())
            diag = {"decoded_dtype": str(t.dtype), "decoded_bytes": back.numel(), "raw_bytes": raw.numel(),
                    "differing_bytes_in_common_prefix": int((back[:n] != raw[:n]).sum())}
        size = getattr(enc, "buffer_size", None)
        nbytes = int(size if size is not None else enc.size * enc.item_size)
        del t, back, dec
        te = timer(lambda: codec.encode(arr))
        td = timer(lambda: codec.decode(enc))
        return same, nbytes, te, td, diag

    def measure(self, alg, words, spans, timer):
        return measure_streams(self.measure_span, alg, words, spans, timer, nvcomp=self.version)


def measure_streams(span_fn, alg, words, spans, timer, **extra):
    """A codec's row over the streams spans of words (int16 [L, T, W]): span_fn(alg, raw bytes of a stream, timer)
    -> (bit_exact, coded bytes, encode s, decode s, mismatch or None), summed over the streams."""
    ok, nbytes, te, td, diag = True, 0, 0.0, 0.0, None
    for a, b in spans:
        raw = words[:, a:b].contiguous().view(torch.uint8).reshape(-1)
        same, nb, e, d, dg = span_fn(alg, raw, timer)
        del raw
        ok, nbytes, te, td = ok and same, nbytes + nb, te + e, td + d
        diag = diag or dg
    out = {"bit_exact": ok, "bits": 8 * nbytes / words.numel(), "encode_ms": 1e3 * te, "decode_ms": 1e3 * td,
           "streams": len(spans), **extra}
    if diag:
        out["mismatch"] = diag
    return out


# ------------------------------------------------------------------------------------- a codec in a child process
class ChildCrash(RuntimeError):
    """A codec's child process died (a crash in native code, or no answer in time) on the same stream again."""


def _exit_name(code):
    if code is None:
        return "no answer in time"
    if code < 0:
        try:
            return signal.Signals(-code).name
        except ValueError:
            return f"signal {-code}"
    return f"exit {code}"


def _codec_worker(conn, factory, kwargs, device, warmup, reps):
    """The child of IsolatedCodec: builds factory(device, **kwargs) ("module:name"), sends what it offers, then
    measures one stream per request ((alg, bytes in shared host memory)) until it receives None."""
    import faulthandler
    from .gpu_speed import timer as _timer
    faulthandler.enable()                            # a crash's Python stack goes to the run's console log
    dev = torch.device(device)
    gpu = dev.type == "cuda"
    if gpu:
        torch.cuda.set_device(dev)
    try:
        mod, _, name = factory.partition(":")
        codec = getattr(importlib.import_module(mod), name)(dev, **kwargs)
        conn.send(("ready", {"codecs": list(codec.codecs), "errors": dict(getattr(codec, "errors", {})),
                             "version": getattr(codec, "version", None)}))
    except Exception as exc:
        conn.send(("error", repr(exc)[:300]))
        return
    tm = lambda fn: _timer(fn, dev, warmup=warmup if gpu else 0, reps=reps if gpu else 1)
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        alg, host = msg
        try:
            raw = host.to(dev) if gpu else host
            conn.send(("ok", codec.measure_span(alg, raw, tm)))
        except Exception as exc:
            conn.send(("error", repr(exc)[:300]))
        raw = host = msg = None
        if gpu:
            torch.cuda.empty_cache()


class IsolatedCodec:
    """A codec (factory "module:name", the class's measure_span) run in a child process: nvCOMP's Python package
    crashed the NUC's full runs inside Codec.encode (SIGSEGV, at random: the same codec on the same KV passed on
    another attempt), taking every later measurement with it. Each stream goes to the child through shared host
    memory; a crash costs that stream one retry in a new child, and a second crash this codec's row of this case
    (an error row), never the run. Timed in the child as in the parent (CUDA events, warmup and reps)."""
    isolated = True

    def __init__(self, factory, device, kwargs=None, warmup=2, reps=5, retries=1, timeout=1800.0, log=print):
        self.factory, self.device, self.kwargs = factory, torch.device(device), dict(kwargs or {})
        self.warmup, self.reps, self.retries, self.timeout, self.log = warmup, reps, retries, timeout, log
        self._proc = self._conn = None
        self.crashes = 0
        info = self._start()
        self.codecs = {a: None for a in info["codecs"]}
        self.errors = info["errors"]
        self.version = info["version"]

    def _start(self):
        ctx = torch.multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_codec_worker, daemon=True,
                           args=(child, self.factory, self.kwargs, str(self.device), self.warmup, self.reps))
        proc.start()
        child.close()
        self._proc, self._conn = proc, parent
        r = self._answer(600.0)
        if r is None or r[0] != "ready":
            if r is None:
                proc.join(5)
            why = _exit_name(proc.exitcode) if r is None else r[1]
            self.close()
            raise RuntimeError(f"the codec's child process did not start ({why})")
        return r[1]

    def _answer(self, timeout):
        """The child's reply, or None when it died (or gave no answer within timeout seconds)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._conn.poll(0.2):
                try:
                    return self._conn.recv()
                except (EOFError, OSError):
                    return None
            if not self._proc.is_alive():
                if self._conn.poll(0):
                    try:
                        return self._conn.recv()
                    except (EOFError, OSError):
                        return None
                return None
        return None

    def close(self):
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is None:
            return
        if proc.is_alive():
            try:
                conn.send(None)
            except (OSError, ValueError):
                pass
            proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(10)
        conn.close()

    def __del__(self):                               # a stage's codecs go when it ends: so does the child
        try:
            self.close()
        except Exception:
            pass

    def measure_span(self, alg, raw, timer=None):
        """As the factory's measure_span, in the child (timer unused: the child times with its own)."""
        host = torch.empty(raw.numel(), dtype=torch.uint8).share_memory_()
        host.copy_(raw.reshape(-1))
        if raw.is_cuda:
            torch.cuda.empty_cache()                 # room for the child's buffers
        for attempt in range(1 + self.retries):
            if self._proc is None or not self._proc.is_alive():
                self.close()
                self._start()
            self._conn.send((alg, host))
            r = self._answer(self.timeout)
            if r is not None:
                break
            self._proc.join(5)                       # its end may not be collected yet (the pipe closes first)
            code = self._proc.exitcode
            self.crashes += 1
            self.log(f"      {alg}: its child process died ({_exit_name(code)})"
                     + ("; once more in a new one" if attempt < self.retries else ""))
            self.close()
        else:
            raise ChildCrash(f"{alg} took its child process down {1 + self.retries} times on the same stream "
                             f"({_exit_name(code)}); see the console log")
        if r[0] != "ok":
            raise RuntimeError(r[1])
        return tuple(r[1])

    def measure(self, alg, words, spans, timer=None):
        before = self.crashes
        out = measure_streams(self.measure_span, alg, words, spans, timer, nvcomp=self.version, isolated=True)
        out["child_crashes"] = self.crashes - before
        return out


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
            # in a child process on the GPU (IsolatedCodec: its crashes cost a row, not the run);
            # KV_NVCOMP_IN_PROCESS=1 runs it in this process as before
            if device.type == "cuda" and os.environ.get("KV_NVCOMP_IN_PROCESS") != "1":
                nv = IsolatedCodec(f"{__name__}:Nvcomp", device, log=log)
            else:
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
