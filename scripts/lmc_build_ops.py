#!/usr/bin/env python3
"""LMCache's compiled CUDA ops (lmcache.cuda_ops) built from our copy (third_party/lmcache/csrc) against the torch of
the environment this runs in, with the nvcc on PATH (CUDA_HOME).

  CUDA_HOME=TK PATH=TK/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6 python scripts/lmc_build_ops.py --src third_party/lmcache \
      --out BUILD_DIR

Why: the release wheel's lmcache.cuda_ops was built against another torch and does not load beside vLLM 0.32's
torch 2.13 (undefined symbol c10::cuda::CUDAErrorLogCapture, job 30688); LMCache then runs on its torch baseline,
and its server crashed there in both of job 30694's runs (a segmentation fault copying a chunk to an engine's GPU).

The extension is the one LMCache's own build makes (setup_extensions/build_profiles/cuda.py: its source list, read
from that file, and its flags), built alone: none of LMCache's Python is installed, and nothing is imported from it.
Writes BUILD_DIR/lmcache/cuda_ops*.so and BUILD_DIR/BUILD.json (sources' hashes, torch, nvcc).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pathlib
import subprocess
import sys


def sources(src):
    """cuda_sources in CudaProfile.build (our copy's cuda.py), as LMCache's build lists them."""
    tree = ast.parse((src / "setup_extensions" / "build_profiles" / "cuda.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "cuda_sources" for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise SystemExit("STOP: no cuda_sources in setup_extensions/build_profiles/cuda.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="our copy of LMCache (third_party/lmcache)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, out = pathlib.Path(a.src).resolve(), pathlib.Path(a.out).resolve()
    files = sources(src)
    missing = [f for f in files if not (src / f).is_file()]
    if missing:
        raise SystemExit(f"STOP: missing sources {missing}")
    import torch
    from setuptools import setup
    from torch.utils import cpp_extension
    abi = "-D_GLIBCXX_USE_CXX11_ABI=1"
    nvcc = os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc")
    nvcc_v = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    jobs = os.environ.setdefault("MAX_JOBS", str(max(1, min(8, len(os.sched_getaffinity(0))))))
    print(f"   building lmcache.cuda_ops from {len(files)} sources: torch {torch.__version__} (CUDA "
          f"{torch.version.cuda}), {nvcc_v[0] if nvcc_v else nvcc}, arch {os.environ.get('TORCH_CUDA_ARCH_LIST')}, "
          f"{jobs} jobs", flush=True)
    ext = cpp_extension.CUDAExtension(
        "lmcache.cuda_ops", sources=[str(src / f) for f in files], include_dirs=[str(src / "csrc")],
        extra_compile_args={"cxx": [abi], "nvcc": [abi]})
    tmp = out / "tmp"
    setup(name="lmcache-cuda-ops-build", ext_modules=[ext], cmdclass={"build_ext": cpp_extension.BuildExtension},
          script_args=["-q", "build_ext", "--build-lib", str(out), "--build-temp", str(tmp)])
    built = sorted((out / "lmcache").glob("cuda_ops*.so"))
    if len(built) != 1:
        raise SystemExit(f"STOP: the build made {built}")
    so = built[0]
    info = {"module": "lmcache.cuda_ops", "file": so.name, "sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
            "from": "third_party/lmcache/csrc", "sources": {f: hashlib.sha256((src / f).read_bytes()).hexdigest()
                                                            for f in files},
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "nvcc": nvcc_v[0] if nvcc_v else None,
            "arch": os.environ.get("TORCH_CUDA_ARCH_LIST"), "python": sys.version.split()[0]}
    (out / "BUILD.json").write_text(json.dumps(info, indent=1))
    print(f"   built {so} ({so.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
