#!/usr/bin/env python3
"""Read-only cloud inventory. No allocation, installation, download or benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def command(args):
    if shutil.which(args[0]) is None:return dict(available=False,argv=args)
    try:
        r=subprocess.run(args,capture_output=True,text=True,timeout=10)
        return dict(available=True,argv=args,exit_code=r.returncode,stdout=r.stdout[:16000],stderr=r.stderr[:2000])
    except subprocess.TimeoutExpired:return dict(available=True,argv=args,error="timed out after 10s")


def package(name):
    try:return metadata.version(name)
    except metadata.PackageNotFoundError:return None


def host_memory():
    physical=os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")
    limits=[]
    for name in ("/sys/fs/cgroup/memory.max","/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value=Path(name).read_text().strip()
            if value.isdigit() and int(value)>0:limits.append(int(value))
        except OSError:pass
    return dict(physical_bytes=physical,effective_limit_bytes=min([physical]+limits))


def inventory(work_dir):
    calls={
        "nvidia_smi":["nvidia-smi","--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id,pstate","--format=csv,noheader"],
        "gpu_topology":["nvidia-smi","topo","-m"],"nvcc":["nvcc","--version"],
        "efa":["fi_info","-p","efa"],"verbs":["ibv_devinfo"],"cpu":["lscpu"],
        "git_head":["git","-C",str(ROOT),"rev-parse","HEAD"]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=dict(zip(calls,pool.map(command,calls.values())))
    pkgs={name:package(name) for name in ("torch","triton","transformers","flashinfer-python", "nvidia-libnvcomp-cu12","nvidia-nvcomp-cu12")}
    gpu=[];torch_error=None;torch_cuda=None
    try:
        import torch
        torch_cuda=torch.version.cuda
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p=torch.cuda.get_device_properties(i)
                gpu.append(dict(index=i,name=p.name,total_memory_bytes=p.total_memory,sm=10*p.major+p.minor))
    except Exception as e:torch_error=repr(e)
    return dict(python=platform.python_version(),python_executable=sys.executable,platform=platform.platform(),
        visible_cpu_affinity_count=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count(),
        memory=host_memory(),work_directory=str(work_dir),disk_free_bytes=shutil.disk_usage(work_dir).free,
        packages=pkgs,torch_cuda_runtime=torch_cuda,gpus=gpu,torch_probe_error=torch_error,commands=results,
        triton_interpret=os.environ.get("TRITON_INTERPRET","0"))


def assess(plan,profile_name,observed,root=ROOT):
    profile=plan["cloud_profiles"][profile_name];runtime=plan["runtime_profiles"]["matched_t"]
    checks={"python":observed["python"].startswith(runtime["python_major_minor"]+"."),
        "torch":observed["packages"].get("torch")==runtime["torch"],
        "triton":observed["packages"].get("triton")==runtime["triton"],
        "compiled_execution_selected":observed["triton_interpret"] in ("","0"),
        "CUDA_visible":bool(observed["gpus"]),
        "Hopper_gpu_visible":any(g["sm"]==90 and g["total_memory_bytes"]>=74_000_000_000 for g in observed["gpus"]),
        "GPU_family_matches_profile":any(profile["gpu"].split()[0].upper() in g["name"].upper() for g in observed["gpus"]),
        "CUDA_runtime":observed["torch_cuda_runtime"]==runtime["cuda_toolkit_major_minor"],
        "CUDA_toolkit_present":observed["commands"]["nvcc"].get("exit_code")==0,
        "CUDA_toolkit_version":"release "+runtime["cuda_toolkit_major_minor"]+"," in observed["commands"]["nvcc"].get("stdout",""),
        "ExANS_compatible_nvCOMP_SDK":observed["packages"].get(runtime["nvcomp_sdk_package"])==runtime["nvcomp_sdk_version"],
        "ExANS_compatible_nvCOMP_Python":observed["packages"].get(runtime["nvcomp_python_package"])==runtime["nvcomp_python_version"]}
    source=root/"kv_compare/exans_official.py"
    hardcoded_ampere=source.exists() and "CMAKE_CUDA_ARCHITECTURES=86" in source.read_text()
    blockers=["Failed static check: "+key for key,value in checks.items() if not value]
    if hardcoded_ampere:blockers.append("Existing ExANS build helper hardcodes SM86; a recorded native SM90 cloud rebuild is required.")
    blockers += ["Cloud-native kernel/packet validation has not run.","Two-node GPU transport and actual payload bandwidth have not been measured."]
    if profile.get("gpudirect_rdma_supported") is False:
        blockers.append("This profile lacks GPUDirect RDMA; host-staged transfer tests are a separate, limited-bandwidth track.")
    return dict(profile=profile_name,advertised_profile=profile,static_checks=checks,
        static_runtime_prerequisites_met=all(checks.values()),native_SM90_rebuild_required=hardcoded_ampere,
        full_environment_validated=False,network_performance_validated=False,cloud_resource_created=False,
        GPU_benchmark_executed=False,blockers=blockers,
        note="Static inventory is not a benchmark pass, a quota/capacity check or a promise of published throughput.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=ROOT/"env/lossless_cloud_plan.json")
    p.add_argument("--profile",default="aws_h100_codec")
    p.add_argument("--work-dir",type=Path,default=Path.cwd())
    p.add_argument("--output",type=Path)
    args=p.parse_args();plan=json.loads(args.plan.read_text())
    if args.profile not in plan["cloud_profiles"]:p.error("unknown profile; see env/lossless_cloud_plan.json")
    if args.output and args.output.exists():p.error("refusing to overwrite an existing inventory")
    observed=inventory(args.work_dir.resolve())
    result=dict(schema="lossless_cloud_inventory_v1",plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        observed=observed,assessment=assess(plan,args.profile,observed))
    encoded=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as f:f.write(encoded)
        print("Inventory:",args.output)
    else:print(encoded,end="")


if __name__=="__main__":main()
