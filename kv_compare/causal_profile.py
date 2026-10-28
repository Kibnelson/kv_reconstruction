"""Diagnostic instrumentation around the unchanged causal codec.

Profiles are deliberately separate from ordinary synchronized wall timings.
Scope device times are inclusive and nested, and must not be added together.
"""
from contextlib import contextmanager
from functools import wraps
import gzip
from pathlib import Path
import shutil
import time
import torch
from . import causal_packet


@contextmanager
def scopes(engine, kappa):
    saved = []
    def wrap(obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def call(*args, **kwargs):
            tag = label(*args, **kwargs) if callable(label) else label
            with torch.profiler.record_function(tag):
                return original(*args, **kwargs)
        had = name in getattr(obj, "__dict__", {})
        saved.append((obj,name,original,had))
        setattr(obj,name,call)
    try:
        wrap(engine,"rotate",lambda *a,**kw: "kv/rope_inverse" if kw.get("inverse") else "kv/rope_forward")
        wrap(engine,"reconstruct","kv/residual_reconstruction")
        wrap(engine,"validate","kv/directory_validation")
        wrap(causal_packet,"compact","kv/compaction")
        codec = engine.residual(kappa) if kappa else engine.lossless
        wrap(codec,"encode","kv/entropy_encode")
        wrap(codec,"decode","kv/entropy_decode")
        yield
    finally:
        for obj,name,original,had in reversed(saved):
            if had:
                setattr(obj,name,original)
            else:
                delattr(obj,name)


def event_row(event):
    def value(*names):
        for name in names:
            if hasattr(event,name):
                return float(getattr(event,name))
        return 0.
    return dict(name=event.key, calls=int(event.count),
        self_cpu_ms=value("self_cpu_time_total")/1000,
        cpu_total_ms=value("cpu_time_total")/1000,
        self_device_ms=value("self_device_time_total","self_cuda_time_total")/1000,
        device_total_ms=value("device_time_total","cuda_time_total")/1000)


def profile_call(fn, engine, kappa, device, trace):
    device = torch.device(device)
    activities=[torch.profiler.ProfilerActivity.CPU]
    if device.type=="cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize(device)
    with scopes(engine,kappa), torch.profiler.profile(activities=activities,
            record_shapes=True, profile_memory=False, with_stack=False) as prof:
        begin=time.perf_counter()
        with torch.profiler.record_function("kv/request"):
            result=fn()
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        wall_ms=(time.perf_counter()-begin)*1000
    trace=Path(trace); trace.parent.mkdir(parents=True,exist_ok=True)
    raw=trace.with_suffix("")
    prof.export_chrome_trace(str(raw))
    with raw.open("rb") as source,gzip.open(trace,"wb") as target:
        shutil.copyfileobj(source,target)
    raw.unlink()
    rows=[event_row(e) for e in prof.key_averages()]
    events=prof.events()
    kernels=[e for e in events if e.device_type==torch.autograd.DeviceType.CUDA]
    if device.type=="cuda" and not kernels:
        raise RuntimeError("CUDA executed but profiler captured no device events; check CUPTI / profiler access")
    return result,dict(gpu_measured=device.type=="cuda",profiled_wall_ms=wall_ms,
        trace=str(trace),cuda_device_event_count=len(kernels),
        semantic_scopes=[r for r in rows if r["name"].startswith("kv/")],
        top_self_device=sorted(rows,key=lambda r:r["self_device_ms"],reverse=True)[:30],
        top_self_cpu=sorted(rows,key=lambda r:r["self_cpu_ms"],reverse=True)[:30],
        operators=[r for r in rows if r["name"].startswith(("aten::mm","aten::bmm","aten::matmul","cudaLaunch","cudaStreamSynchronize","cudaMemcpy"))],
        caveat="diagnostic run has profiler overhead; scopes are nested/inclusive, not additive; use paired unprofiled wall times")
