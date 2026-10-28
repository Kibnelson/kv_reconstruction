"""Separate diagnostic traces; CUDA kernels and host operators stay separate."""
from contextlib import contextmanager
import gzip
from pathlib import Path
import shutil
import time

import torch


@contextmanager
def regions(engine, kappa):
    saved = []
    def wrap(obj, name, label):
        original = getattr(obj, name)
        had = name in getattr(obj, '__dict__', {})
        def call(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        saved.append((obj,name,original,had)); setattr(obj,name,call)
    try:
        wrap(engine,'rotate','lite/rotation')
        if hasattr(engine,'scatter'): wrap(engine,'scatter','lite/column_store')
        wrap(engine,'validate','lite/directory_validation')
        if kappa: wrap(engine,'reconstruct','lite/residual_reconstruction')
        coder = engine.residual(kappa) if kappa else engine.lossless
        wrap(coder,'encode','lite/entropy_encode'); wrap(coder,'decode','lite/entropy_decode')
        yield
    finally:
        for obj,name,original,had in reversed(saved):
            if had: setattr(obj,name,original)
            else: delattr(obj,name)


def summarize(prof, wall_ms):
    from .causal_profile import event_row
    # Raw CUDA activities only: do not add aten operators or nested user ranges
    # to kernel durations. Parent operator totals include those same kernels.
    by_name = {}; memory = {}; intervals = []
    for event in prof.events():
        if (event.device_type != torch.autograd.DeviceType.CUDA or
                getattr(event,'is_user_annotation',False) or event.name.startswith(('lite/','kv/'))):
            continue
        is_memory = event.name.lower().startswith(('memcpy','memset','[memory]','[out of memory]'))
        dest = memory if is_memory else by_name
        row = dest.setdefault(event.name,dict(name=event.name,calls=0,self_device_ms=0.))
        row['calls'] += 1; row['self_device_ms'] += event.self_device_time_total / 1000
        if not is_memory: intervals.append((event.time_range.start,event.time_range.end))
    if not by_name:
        raise RuntimeError('no CUDA kernels captured; check CUPTI/profiler access')
    merged = []
    for lo,hi in sorted(intervals):
        if merged and lo <= merged[-1][1]: merged[-1][1] = max(hi,merged[-1][1])
        else: merged.append([lo,hi])
    averages = [event_row(e) for e in prof.key_averages()]
    kernels = sorted(by_name.values(),key=lambda x:x['self_device_ms'],reverse=True)
    return dict(gpu_measured=True, profiled_wall_ms=wall_ms,
        cuda_kernel_calls=sum(r['calls'] for r in kernels),
        cuda_kernel_duration_sum_ms=sum(r['self_device_ms'] for r in kernels),
        cuda_kernel_busy_union_ms=sum(b-a for a,b in merged)/1000,
        first_to_last_kernel_span_ms=(merged[-1][1]-merged[0][0])/1000,
        kernels=kernels,
        device_transfers_and_memsets=list(memory.values()),
        host_operators=sorted((r for r in averages if r['name'].startswith(('aten::','cuda'))),
                              key=lambda x:x['self_cpu_ms'],reverse=True),
        regions=[r for r in averages if r['name'].startswith('lite/')],
        caveat='instrumented diagnostic, not benchmark latency; CUDA activities are separate from inclusive operator/range totals; kernel union excludes host work and idle gaps')


def profile(fn, engine, kappa, device, trace):
    if torch.device(device).type != 'cuda':
        raise RuntimeError('actual CUDA required for timing diagnostics')
    torch.cuda.synchronize(device)
    with regions(engine,kappa), torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,with_stack=False,profile_memory=False) as prof:
        begin = time.perf_counter()
        value = fn(); torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter()-begin)*1000
    trace = Path(trace); trace.parent.mkdir(parents=True,exist_ok=True)
    raw = trace.with_suffix(''); prof.export_chrome_trace(str(raw))
    with raw.open('rb') as source,gzip.open(trace,'wb') as target: shutil.copyfileobj(source,target)
    raw.unlink()
    return value,summarize(prof,wall_ms)
