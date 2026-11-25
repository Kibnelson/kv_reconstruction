#!/usr/bin/env python3
"""LMCache's compiled modules built from our copy (third_party/lmcache/csrc) against the torch of the environment
this runs in, with the nvcc on PATH (CUDA_HOME): all four the release wheel has, as LMCache's own build makes them.

  CUDA_HOME=TK PATH=TK/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6 python scripts/lmc_build_ops.py --src third_party/lmcache \
      --out BUILD_DIR

  lmcache.cuda_ops        the GPU kernels and transfer plans (setup_extensions/build_profiles/cuda.py)
  lmcache.lmcache_native  bitmaps, folds, locks, and the shared plan types (PageBufferShapeDesc, ...)
  lmcache.lmcache_fs      the file-system storage connector     (these three: setup_extensions/common_cpp.py)
  lmcache.lmcache_redis   the Redis storage connector

Why all four: the release wheel's were built against another torch, and its cuda_ops does not load beside torch 2.13
(job 30688); LMCache's torch fallback then crashed its server (job 30694). Building cuda_ops alone (job 30722) left
it unable to take lmcache_native's types: pybind11 shares a type between modules only when they are built alike, so
every store failed ("DirectCopyGroupSpec(): incompatible constructor arguments" for the wheel's PageBufferShapeDesc).

Each module is the one LMCache's build makes (its source lists and flags, read from those two files), built here
alone: none of LMCache's Python is installed, and nothing is imported from it. Writes BUILD_DIR/lmcache/<module>*.so
and BUILD_DIR/BUILD.json (each module's file and hash, the sources' hashes, torch, nvcc).
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

ABI = "-D_GLIBCXX_USE_CXX11_ABI=1"


def cuda_sources(src):
    """cuda_sources in CudaProfile.build (our copy's cuda.py), as LMCache's build lists them."""
    tree = ast.parse((src / "setup_extensions" / "build_profiles" / "cuda.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "cuda_sources" for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise SystemExit("STOP: no cuda_sources in setup_extensions/build_profiles/cuda.py")


def common_specs(src):
    """COMMON_EXTENSIONS of our copy's common_cpp.py: [(name, sources, include_dirs)]."""
    tree = ast.parse((src / "setup_extensions" / "common_cpp.py").read_text())
    for node in ast.walk(tree):
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
        if getattr(target, "id", None) != "COMMON_EXTENSIONS" or node.value is None:
            continue
        out = []
        for call in node.value.elts:
            kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            out.append((kw["name"], kw["sources"], kw["include_dirs"]))
        return out
    raise SystemExit("STOP: no COMMON_EXTENSIONS in setup_extensions/common_cpp.py")


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="our copy of LMCache (third_party/lmcache)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    src, out = pathlib.Path(a.src).resolve(), pathlib.Path(a.out).resolve()
    cuda_files = cuda_sources(src)
    common = common_specs(src)
    every = cuda_files + [f for _n, fs, _i in common for f in fs]
    missing = [f for f in every if not (src / f).is_file()]
    if missing:
        raise SystemExit(f"STOP: missing sources {missing}")
    import torch
    from setuptools import setup
    from torch.utils import cpp_extension
    nvcc = os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc")
    nvcc_v = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]
    jobs = os.environ.setdefault("MAX_JOBS", str(max(1, min(8, len(os.sched_getaffinity(0))))))
    exts = [cpp_extension.CppExtension(
                "lmcache." + name, sources=[str(src / f) for f in files], include_dirs=[str(src / d) for d in incs],
                extra_compile_args={"cxx": [ABI, "-O3"]})
            for name, files, incs in common]
    exts.append(cpp_extension.CUDAExtension(
        "lmcache.cuda_ops", sources=[str(src / f) for f in cuda_files], include_dirs=[str(src / "csrc")],
        extra_compile_args={"cxx": [ABI], "nvcc": [ABI]}))
    names = [e.name for e in exts]
    print(f"   building {', '.join(names)} from {len(every)} sources: torch {torch.__version__} (CUDA "
          f"{torch.version.cuda}), {nvcc_v[0] if nvcc_v else nvcc}, arch {os.environ.get('TORCH_CUDA_ARCH_LIST')}, "
          f"{jobs} jobs", flush=True)
    setup(name="lmcache-native-build", ext_modules=exts, cmdclass={"build_ext": cpp_extension.BuildExtension},
          script_args=["-q", "build_ext", "--build-lib", str(out), "--build-temp", str(out / "tmp")])
    modules = []
    for name in names:
        mod = name.rsplit(".", 1)[1]
        built = sorted((out / "lmcache").glob(mod + ".*.so")) + sorted((out / "lmcache").glob(mod + ".so"))
        if len(built) != 1:
            raise SystemExit(f"STOP: the build made {built} for {name}")
        modules.append({"module": name, "file": built[0].name, "sha256": sha(built[0])})
        print(f"   built {built[0]} ({built[0].stat().st_size / 1e6:.1f} MB)")
    info = {"modules": modules, "from": "third_party/lmcache/csrc", "sources": {f: sha(src / f) for f in every},
            "torch": torch.__version__, "torch_cuda": torch.version.cuda, "nvcc": nvcc_v[0] if nvcc_v else None,
            "arch": os.environ.get("TORCH_CUDA_ARCH_LIST"), "python": sys.version.split()[0]}
    (out / "BUILD.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
