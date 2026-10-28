"""The official ExANS codec (openlake-project/openlake, crates/openlake_kv_client/cuda, ExpANS) on our captured KV.

ExANS is built from the authors' source at EXANS_COMMIT with one build fix: the allocation macro of
openlake_expans_create passes its std::unique_ptr where the error helper takes a raw pointer (21 compile errors
in nvcc 12.8 / GCC 13; the earlier official run stopped there). The fix changes `fail_cuda(codec, ...)` to
`fail_cuda(&*codec, ...)` inside that macro only; the codec itself is unchanged. The library is called through
ctypes exactly as the authors' vLLM connector does (external/connectors/vllm/openlake_expans.py).

ExANS codes fixed raw blocks of 122 x 4096 BF16 words (999,424 bytes): sign+mantissa bytes raw, exponent bytes
(with a per-128-value phase shift chosen on the GPU) compressed with nvCOMP's batched ANS, a 64-byte header and a
phase bitmap. A record holds one block; its compressed size is the header's actual_bytes (a block that does not
fit the record slot would travel raw). Here a block is 244 consecutive tokens x 2,048 channels of one layer (the
document's [layers, tokens, channels] words cut in order), and every block of the document is coded in one batch.
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import time

EXANS_REPO = "https://github.com/openlake-project/openlake.git"
EXANS_COMMIT = "de6c38bcc5438475bd1abf1e9b3d154cef414ef6"
BLOCK_WORDS = 122 * 4096
BLOCK_BYTES = 2 * BLOCK_WORDS
HEADER_BYTES = 64
PHASE_BYTES = BLOCK_WORDS // 128 // 8 + ((122 * 2 + 31) // 32) * 4
FRAME_OFFSET = ((HEADER_BYTES + BLOCK_WORDS + PHASE_BYTES + 15) // 16) * 16
RECORD_BYTES = BLOCK_BYTES - 16                  # the largest slot the codec accepts (record < raw block)
FIX_OLD = "return fail_cuda(codec, status__, #member);"
FIX_NEW = "return fail_cuda(&*codec, status__, #member);"


def _run(cmd, log, cwd=None, env=None):
    log("  $ " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} exited {r.returncode}:\n{(r.stdout + r.stderr)[-4000:]}")
    return r.stdout + r.stderr


def nvcomp_root():
    spec = importlib.util.find_spec("nvidia.libnvcomp")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("nvCOMP not found: pip install nvidia-libnvcomp-cu12 (the native SDK package)")
    return Path(next(iter(spec.submodule_search_locations)))


def build(work, source, cuda_home, log=print):
    """Copy the codec's CUDA directory from `source` (an openlake checkout at EXANS_COMMIT; cloned into work if
    missing), apply the build fix, build libopenlake_expans.so and the authors' test, run the test.
    Returns build information."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    source = Path(source) if source else work / "openlake"
    if not (source / ".git").exists():
        log(f"  cloning {EXANS_REPO} into {source}")
        _run(["git", "clone", "--filter=blob:none", EXANS_REPO, source], log)
    head = subprocess.run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != EXANS_COMMIT:
        if subprocess.run(["git", "-C", source, "cat-file", "-e", EXANS_COMMIT]).returncode:
            _run(["git", "-C", source, "fetch", "origin", EXANS_COMMIT], log)
        src_root = work / "openlake_at_commit"
        if src_root.exists():
            shutil.rmtree(src_root)
        _run(["git", "-C", source, "worktree", "add", "--detach", src_root, EXANS_COMMIT], log)
    else:
        src_root = source
    cuda_src = work / "src"
    if cuda_src.exists():
        shutil.rmtree(cuda_src)
    shutil.copytree(src_root / "crates/openlake_kv_client/cuda", cuda_src)
    cu = cuda_src / "src/expans_codec.cu"
    text = cu.read_text()
    if text.count(FIX_OLD) != 1:
        raise RuntimeError("the ExANS source differs from the expected commit (build-fix line not found once)")
    cu.write_text(text.replace(FIX_OLD, FIX_NEW))
    (work / "build_fix.diff").write_text(f"--- {cu.name} (in the EXPANS_ALLOC macro)\n-{FIX_OLD}\n+{FIX_NEW}\n")
    cuda_home = Path(cuda_home)
    env = dict(os.environ, CUDA_HOME=str(cuda_home), CUDACXX=str(cuda_home / "bin/nvcc"),
               PATH=str(cuda_home / "bin") + os.pathsep + os.environ.get("PATH", ""))
    root = nvcomp_root()
    bdir = work / "build"
    _run(["cmake", "-S", cuda_src, "-B", bdir, "-DOPENLAKE_EXPANS_BUILD=ON", "-DOPENLAKE_CUDA_CODEC_BUILD_TESTS=ON",
          "-DCMAKE_CUDA_ARCHITECTURES=86", f"-DNVCOMP_ROOT={root}", f"-Dnvcomp_DIR={_nvcomp_cmake(root)}",
          f"-DCUDAToolkit_ROOT={cuda_home}", "-DCMAKE_BUILD_TYPE=Release"], log, env=env)
    _run(["cmake", "--build", bdir, "--target", "openlake_expans", "openlake_expans_test", "-j", "4"], log, env=env)
    lib = next(bdir.rglob("libopenlake_expans.so"))
    test = next(bdir.rglob("openlake_expans_test"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(p) for p in _lib_dirs(root, cuda_home)] +
                                             [os.environ.get("LD_LIBRARY_PATH", "")])
    out = _run([test], log, env=env)
    (work / "official_test.log").write_text(out)
    log("  the authors' ExANS test passed")
    return {"commit": EXANS_COMMIT, "source": str(src_root), "library": str(lib), "nvcomp_root": str(root),
            "cuda_home": str(cuda_home), "build_fix": f"{FIX_OLD} -> {FIX_NEW}", "official_test": "passed"}


def _nvcomp_cmake(root):
    for c in ("lib/cmake/nvcomp", "lib64/cmake/nvcomp"):
        if (root / c).is_dir():
            return root / c
    return root


def _lib_dirs(root, cuda_home):
    return [d for d in (root / "lib", root / "lib64", Path(cuda_home) / "lib64", Path(cuda_home) / "lib")
            if d.is_dir()]


def _load(info):
    import torch
    torch.cuda.init()                                # the CUDA runtime first (torch's), then nvCOMP, then ExANS
    for d in _lib_dirs(Path(info["nvcomp_root"]), info["cuda_home"]):
        for so in sorted(d.glob("libnvcomp*.so*")):
            ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL(info["library"])
    vp = ctypes.c_void_p
    lib.openlake_expans_create.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32),
                                           ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(vp)]
    lib.openlake_expans_create.restype = ctypes.c_int
    lib.openlake_expans_destroy.argtypes = [vp]
    lib.openlake_expans_last_error.argtypes = [vp]
    lib.openlake_expans_last_error.restype = ctypes.c_char_p
    lib.openlake_expans_encode.argtypes = [vp, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
                                           ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_uint64,
                                           ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
    lib.openlake_expans_encode.restype = ctypes.c_int
    lib.openlake_expans_decode.argtypes = [vp, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32,
                                           ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32),
                                           ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint64]
    lib.openlake_expans_decode.restype = ctypes.c_int
    return lib


def measure(info, words, reps=5, log=print):
    """Encode and decode the words (int16 tensor on the GPU, any shape; cut into whole blocks in memory order) with
    the official codec. Returns bits per value, exactness and timings (wall clock of the codec calls, which
    synchronise their stream, and CUDA events)."""
    import torch
    lib = _load(info)
    flat = words.contiguous().view(-1)
    nb = flat.numel() // BLOCK_WORDS
    if nb == 0:
        raise ValueError("fewer values than one ExANS block")
    data = flat[: nb * BLOCK_WORDS].view(nb, BLOCK_WORDS)
    handle = ctypes.c_void_p()
    seg = (ctypes.c_uint32 * 1)(BLOCK_BYTES)
    status = lib.openlake_expans_create(BLOCK_BYTES, RECORD_BYTES, seg, 1, nb, ctypes.byref(handle))
    if status:
        raise RuntimeError("ExANS create failed: " + (lib.openlake_expans_last_error(handle) or b"?").decode())
    try:
        records = torch.zeros(nb * RECORD_BYTES, dtype=torch.uint8, device=words.device)
        out = torch.empty_like(data)
        u64 = lambda xs: (ctypes.c_uint64 * len(xs))(*xs)
        u32 = lambda xs: (ctypes.c_uint32 * len(xs))(*xs)
        addrs = u64([data.data_ptr() + i * BLOCK_BYTES for i in range(nb)])
        oaddrs = u64([out.data_ptr() + i * BLOCK_BYTES for i in range(nb)])
        sizes, offs, idx = u32([BLOCK_BYTES] * nb), u32(list(range(nb + 1))), u32(list(range(nb)))
        fits = (ctypes.c_uint8 * nb)()
        stream = torch.cuda.current_stream(words.device)

        def enc():
            s = lib.openlake_expans_encode(handle, addrs, sizes, offs, nb, records.data_ptr(), fits, stream.cuda_stream)
            if s:
                raise RuntimeError("ExANS encode failed: " + lib.openlake_expans_last_error(handle).decode())

        def dec():
            s = lib.openlake_expans_decode(handle, records.data_ptr(), idx, nb, oaddrs, sizes, offs, stream.cuda_stream)
            if s:
                raise RuntimeError("ExANS decode failed: " + lib.openlake_expans_last_error(handle).decode())

        def timed(fn):
            fn()                                     # warm-up
            walls, evs = [], []
            for _ in range(reps):
                torch.cuda.synchronize(words.device)
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                a.record(stream)
                fn()
                b.record(stream)
                b.synchronize()
                walls.append(time.perf_counter() - t0)
                evs.append(a.elapsed_time(b) / 1e3)
            walls.sort(); evs.sort()
            return walls[len(walls) // 2], evs[len(evs) // 2]

        enc()
        nfit = sum(int(fits[i]) for i in range(nb))
        hdr = records.view(nb, RECORD_BYTES)[:, :HEADER_BYTES].cpu()
        actual = hdr[:, 20:24].contiguous().view(torch.int32).flatten().to(torch.int64)
        magic = hdr[:, 0:4].contiguous().view(torch.int32).flatten()
        out.zero_()
        dec()
        exact = bool(torch.equal(out, data))
        fit_mask = torch.tensor([bool(fits[i]) for i in range(nb)])
        nbytes = int(torch.where(fit_mask, actual, torch.full_like(actual, BLOCK_BYTES)).sum())
        te_wall, te_ev = timed(enc)
        td_wall, td_ev = timed(dec)
        raw = nb * BLOCK_BYTES
        res = {"blocks": nb, "values": nb * BLOCK_WORDS, "fit_blocks": nfit, "record_bytes": RECORD_BYTES,
               "magic_ok": bool((magic == 0x31505845).all()), "bit_exact": exact,
               "bits_per_value": 8 * nbytes / (nb * BLOCK_WORDS),
               "header_and_phase_bits_per_value": 8 * FRAME_OFFSET / BLOCK_WORDS - 8,
               "encode_ms_wall": 1e3 * te_wall, "decode_ms_wall": 1e3 * td_wall,
               "encode_ms_events": 1e3 * te_ev, "decode_ms_events": 1e3 * td_ev,
               "encode_gbps_wall": raw / te_wall / 1e9, "decode_gbps_wall": raw / td_wall / 1e9,
               "encode_gbps_events": raw / te_ev / 1e9, "decode_gbps_events": raw / td_ev / 1e9}
        log(f"  ExANS (official): {res['bits_per_value']:.4f} bits/value over {nb} blocks ({nfit} fit their record), "
            f"exact={exact}; encode {res['encode_gbps_wall']:.1f} GB/s, decode {res['decode_gbps_wall']:.1f} GB/s "
            "(wall clock of the codec calls)")
        return res
    finally:
        lib.openlake_expans_destroy(handle)
