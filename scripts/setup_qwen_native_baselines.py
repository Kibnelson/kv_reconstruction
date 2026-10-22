#!/usr/bin/env python3
"""Fetch pinned AQUA/OpenLake sources; install Quanto and build native ExANS.

Does not replace Torch, Triton, NumPy, Transformers or their dependencies.
ExANS needs a CUDA toolkit with nvcc; the GPU driver alone is insufficient.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from qwen_kv_bench.native_baseline_sources import (
    AQUA_COMMIT,AQUA_FILES,EXANS_COMMIT,EXANS_FILES,EXANS_CUDA,QUANTO_VERSION,sha_file,verify_sources)
from qwen_kv_bench.matched_transfer import NVCOMP_VERSION


def fetch_sources(repository,commit,files,directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name in files:
        path=directory/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():continue
        url=f'https://raw.githubusercontent.com/{repository}/{commit}/{name}'
        data=urllib.request.urlopen(url,timeout=60).read()
        temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
    verify_sources(directory,files)


def find_nvcc(explicit=None):
    candidates=[explicit,shutil.which('nvcc')]
    for key in ['CUDA_HOME','CUDA_PATH']:
        if os.environ.get(key):candidates.append(str(Path(os.environ[key])/'bin/nvcc'))
    candidates+=['/usr/local/cuda/bin/nvcc']
    return next((str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()),None)


def exans_build_source(directory):
    # The pinned release's allocation macro passes its local unique_ptr to
    # fail_cuda(openlake_expans_codec*). The macro is used only inside create.
    # Correct that error-path argument in a separate build copy; no kernel,
    # layout, allocation size or compression/decompression path is changed.
    original=directory/EXANS_CUDA/'src/expans_codec.cu'
    source=original.read_text();before='return fail_cuda(codec, status__, #member);'
    if source.count(before)!=1:raise ValueError('ExANS build-fix context differs from pinned source')
    source=source.replace(before,'return fail_cuda(codec.get(), status__, #member);')
    target=directory/'build/expans_codec_compat.cu';target.parent.mkdir(exist_ok=True);target.write_text(source)
    return target


def build_exans(directory,nvcc,arch):
    if not nvcc:raise RuntimeError('nvcc not found; install a CUDA 12 toolkit or set CUDA_HOME (driver/Torch alone do not provide nvcc)')
    if importlib.metadata.version('nvidia-libnvcomp-cu12')!=NVCOMP_VERSION:
        raise RuntimeError('native ExANS requires pinned nvidia-libnvcomp-cu12=='+NVCOMP_VERSION)
    spec=importlib.util.find_spec('nvidia.libnvcomp')
    if spec is None or not spec.submodule_search_locations:raise RuntimeError('install pinned nvidia-libnvcomp-cu12=='+NVCOMP_VERSION)
    runtime=Path(next(iter(spec.submodule_search_locations)));include=runtime/'include';libdir=runtime/'lib64'
    library=libdir/'libnvcomp.so.5'
    if not (include/'nvcomp/ans.h').is_file() or not library.is_file():raise RuntimeError('nvCOMP 5.3 headers/runtime missing')
    compiler=subprocess.check_output([nvcc,'--version'],text=True)
    source=directory/EXANS_CUDA;target=directory/'libopenlake_expans.so';temporary=target.with_suffix('.part.so')
    build_source=exans_build_source(directory)
    command=[nvcc,'-std=c++17','-O3','--shared','-Xcompiler=-fPIC','--expt-relaxed-constexpr',
        '-gencode',f'arch=compute_{arch},code=sm_{arch}',
        '-I'+str(source/'include'),'-I'+str(source/'src'),'-I'+str(include),
        str(build_source),'-L'+str(libdir),'-l:libnvcomp.so.5',
        '-Xlinker=-rpath','-Xlinker='+str(libdir),'-o',str(temporary)]
    completed=subprocess.run(command,capture_output=True,text=True)
    log=dict(command=command,returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
    (directory/'build.log.json').write_text(json.dumps(log,indent=2)+'\n')
    if completed.returncode:raise RuntimeError('native ExANS compile failed: '+completed.stderr[-3000:])
    temporary.replace(target)
    manifest=dict(commit=EXANS_COMMIT,source_sha256=EXANS_FILES,compiler=compiler,arch=arch,
        nvcomp=NVCOMP_VERSION,nvcomp_library_sha256=sha_file(library),library_sha256=sha_file(target),command=command,
        host_build_fix='allocation-error macro uses codec.get() for unique_ptr; one error-path argument only',
        build_source_sha256=sha_file(build_source),kernels_unmodified=True)
    (directory/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install',action='store_true');p.add_argument('--nvcc');p.add_argument('--arch',default='86')
    args=p.parse_args()
    if not args.arch.isdigit():p.error('--arch must be a CUDA architecture number, for example 86')
    root=ROOT/'runs/_matched_baselines';report={}
    if args.install:
        packages=['optimum-quanto=='+QUANTO_VERSION,'ninja==1.11.1.3']
        r=subprocess.run([sys.executable,'-m','pip','install','--no-deps',*packages])
        report['packages']=dict(packages=packages,returncode=r.returncode)
    for name,repository,commit,files in [
        ('aqua','goodevening13/aquakv',AQUA_COMMIT,AQUA_FILES),
        ('exans','openlake-project/openlake',EXANS_COMMIT,EXANS_FILES)]:
        try:
            fetch_sources(repository,commit,files,root/name)
            info=dict(status='ready',commit=commit,source_sha256=files)
            if name=='exans':info['build']=build_exans(root/name,find_nvcc(args.nvcc),args.arch)
            else:
                from qwen_kv_bench.matched_aqua import load_upstream
                load_upstream(root/name)
            report[name]=info
        except Exception as error:report[name]=dict(status='unavailable',commit=commit,reason=repr(error))
    root.mkdir(parents=True,exist_ok=True)
    (root/'native_setup.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if all(report[n]['status']=='ready' for n in ['aqua','exans']) else 1


if __name__=='__main__':sys.exit(main())
