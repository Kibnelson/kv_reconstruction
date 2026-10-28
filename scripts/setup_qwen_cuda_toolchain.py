#!/usr/bin/env python3
"""Prepare CUDA 12.8 build tools locally; preserve the driver and Python env.

Uses an existing complete CUDA 12.8 toolkit, or SHA256-pinned NVIDIA nvcc,
cudart and CCCL archives (~81 MB download) under the ignored runs/ directory.
No administrator access, system installation or Python package changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
BASE_URL='https://developer.download.nvidia.com/compute/cuda/redist/'
PACKAGES={
    'cuda_nvcc': dict(version='12.8.93',bytes=79015464,
        sha256='9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3'),
    'cuda_cudart': dict(version='12.8.90',bytes=1354240,
        sha256='8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68'),
    'cuda_cccl': dict(version='12.8.90',bytes=928524,
        sha256='0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b')}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def compiler_version(nvcc):
    r=subprocess.run([str(nvcc),'--version'],text=True,capture_output=True,timeout=30)
    match=re.search(r'release (\d+\.\d+)',r.stdout)
    return (match.group(1) if r.returncode==0 and match else None),r.stdout+r.stderr


def existing_toolkit():
    paths=[os.environ.get('CUDA_HOME'),os.environ.get('CUDA_PATH'),'/usr/local/cuda-12.8','/usr/local/cuda']
    on_path=shutil.which('nvcc')
    if on_path:paths.insert(0,str(Path(on_path).absolute().parent.parent))
    for path in paths:
        if not path:continue
        root=Path(path).absolute();nvcc=root/'bin/nvcc'
        if (not nvcc.is_file() or not (root/'include/cuda_runtime.h').is_file()
                or not any((root/p/'libcudart_static.a').is_file() for p in ['lib64','lib'])):continue
        version,banner=compiler_version(nvcc)
        if version=='12.8':return root,banner
    return None,None


def download_package(name, info, cache):
    filename=f"{name}-linux-x86_64-{info['version']}-archive.tar.xz"
    path=cache/filename;cache.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==info['bytes'] and sha_file(path)==info['sha256']:return path
    url=BASE_URL+f'{name}/linux-x86_64/'+filename
    print(f"Downloading NVIDIA {name} {info['version']} ({info['bytes']/1e6:.1f} MB)",flush=True)
    temporary=path.with_suffix('.part')
    with urllib.request.urlopen(url,timeout=60) as response,temporary.open('wb') as f:
        shutil.copyfileobj(response,f,length=1<<20)
    if temporary.stat().st_size!=info['bytes'] or sha_file(temporary)!=info['sha256']:
        raise ValueError('NVIDIA archive byte count/SHA256 mismatch: '+name)
    temporary.replace(path);return path


def extract_package(archive, destination):
    # data_filter rejects traversal, escaping links and special files. Build
    # only from verified official archives; restore executable tool modes
    # explicitly because tar hard links may otherwise lose executable bits.
    with tarfile.open(archive,'r:xz') as t:
        members=t.getmembers()
        roots={Path(m.name).parts[0] for m in members if Path(m.name).parts}
        if len(roots)!=1 or any(m.name.startswith('/') for m in members):raise ValueError('invalid CUDA archive root')
        destination.mkdir(parents=True,exist_ok=True)
        t.extractall(destination,filter='data')
    root=destination/next(iter(roots))
    for binary in ['bin/nvcc','bin/nvlink','bin/ptxas','bin/cudafe++','bin/fatbinary','bin/bin2c',
                   'bin/__nvcc_device_query','nvvm/bin/cicc']:
        p=root/binary
        if p.is_file():p.chmod(0o755)
    return root


def compose_toolkit(components, destination):
    destination.mkdir(parents=True,exist_ok=True)
    for component in components:
        for source in sorted(component.rglob('*')):
            if not source.is_file():continue
            relative=source.relative_to(component)
            # NVIDIA tarballs use lib/; nvcc and Torch also search lib64/.
            if relative.parts[0]=='lib':relative=Path('lib64',*relative.parts[1:])
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
            if target.is_symlink() and target.resolve()==source.resolve():continue
            if target.exists() or target.is_symlink():continue
            target.symlink_to(source.resolve())
    alias=destination/'lib'
    if not alias.exists():alias.symlink_to('lib64',target_is_directory=True)
    for name in ['bin/nvcc','bin/nvlink','bin/ptxas','nvvm/bin/cicc','include/cuda_runtime.h',
                 'lib64/libcudadevrt.a','lib64/libcudart_static.a','include/cub/cub.cuh']:
        if not (destination/name).is_file():raise ValueError('incomplete local toolkit: '+name)
    return destination


def write_env(path, root):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('export CUDA_HOME='+shlex.quote(str(root))+'\n'
        'export CUDA_PATH="$CUDA_HOME"\n'
        'export PATH="$CUDA_HOME/bin:$PATH"\n'
        'export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n')


def prepare(directory, env_path, *, force_local=False):
    directory=Path(directory).absolute();directory.mkdir(parents=True,exist_ok=True)
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','AMD64'):
        raise RuntimeError('this pinned toolchain bootstrap targets Linux x86_64')
    cxx=shutil.which(os.environ.get('CXX','c++'))
    if not cxx:raise RuntimeError('host C++ compiler missing; install build-essential or set CXX')
    root,banner=(None,None) if force_local else existing_toolkit()
    origin='existing CUDA 12.8 toolkit'
    if root is None:
        print('Preparing project-local CUDA 12.8 build tools; no driver or Python packages are changed.',flush=True)
        components=[]
        for name,info in PACKAGES.items():
            archive=download_package(name,info,directory/'downloads')
            components.append(extract_package(archive,directory/'components'/name))
        root=compose_toolkit(components,directory/'toolkit')
        version,banner=compiler_version(root/'bin/nvcc')
        if version!='12.8':raise RuntimeError('prepared compiler does not report CUDA 12.8')
        origin='project-local NVIDIA archives'
    manifest=dict(format='qwen_cuda_toolchain_v1',status='ready',origin=origin,cuda_home=str(root),
        compiler=banner,host_cxx=cxx,packages=PACKAGES if origin.startswith('project-local') else {},
        scope='build tool provisioning only; not GPU execution or a GPU measurement')
    (directory/'toolchain.json').write_text(json.dumps(manifest,indent=2)+'\n')
    write_env(env_path,root);print('CUDA_HOME='+str(root),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'runs/_matched_baselines/cuda12_8')
    p.add_argument('--output-env',type=Path,default=ROOT/'runs/_matched_baselines/cuda_env.sh')
    p.add_argument('--force-local',action='store_true');args=p.parse_args()
    prepare(args.directory,args.output_env,force_local=args.force_local)


if __name__=='__main__':main()
