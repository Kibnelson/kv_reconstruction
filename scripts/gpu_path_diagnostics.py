"""Optional CUDA bottleneck controls; profiler records no tensor contents."""
from collections import defaultdict
from contextlib import contextmanager
import time
import numpy as np
from qwen_kv_bench import transfer_t1 as t1

PROFILE_SCHEMA='leaf_cuda_activities_v2'


def summarize_profile(profiler,torch,wall_seconds):
    # CUDA user-annotation ranges also appear as device events in Kineto.
    # Exclude them as well as CPU parents to avoid counting nested intervals.
    kernels=defaultdict(lambda:dict(count=0,total_ms=0.))
    excluded=0
    for event in profiler.events():
        if event.device_type!=torch.autograd.DeviceType.CUDA:continue
        if getattr(event,'is_user_annotation',False) or event.name.startswith('T1/'):
            excluded+=1;continue
        row=kernels[event.name];row['count']+=1
        row['total_ms']+=float(event.device_time_total)/1000.
    kernel_rows=[dict(name=name,**row) for name,row in kernels.items()]
    kernel_rows.sort(key=lambda x:x['total_ms'],reverse=True)
    cpu=[];regions=[]
    for event in profiler.key_averages():
        if event.device_type!=torch.autograd.DeviceType.CPU:continue
        row=dict(name=event.key,count=int(event.count),self_CPU_ms=float(event.self_cpu_time_total)/1000.,
            inclusive_CPU_ms=float(event.cpu_time_total)/1000.,inclusive_device_ms=float(event.device_time_total)/1000.)
        if event.key.startswith('T1/'):regions.append(row)
        elif event.key.startswith('aten::'):cpu.append(row)
    cpu.sort(key=lambda x:x['self_CPU_ms'],reverse=True)
    return dict(profile_summary_schema=PROFILE_SCHEMA,profiled_synchronized_wall_ms=wall_seconds*1000,
        CUDA_activity_observed=bool(kernel_rows),CUDA_device_event_count=sum(x['count'] for x in kernel_rows),
        excluded_CUDA_annotation_event_count=excluded,
        CUDA_device_event_total_ms=sum(x['total_ms'] for x in kernel_rows),top_device_events=kernel_rows[:25],
        Torch_operator_call_count=sum(x['count'] for x in cpu),top_Torch_operators=cpu[:25],regions=regions,
        scope='One instrumented call, not an uninstrumented latency measurement. Leaf device activities include kernels and copies; CUDA user annotations and CPU parents are excluded. Durations may overlap and are not a wall-time estimate. Nested region times must not be added. No tensor values, stacks, trace file or shapes are exported.')


@contextmanager
def stage_regions(engine,torch):
    originals={name:getattr(engine,name) for name in ['_walk','_mean','_context','_parameters','_advance','_encode_prepared']}
    existing={name:engine.__dict__.get(name) for name in originals}
    def wrap(name,fn):
        def run(*args,**kwargs):
            with torch.profiler.record_function('T1/'+name.lstrip('_')):
                return fn(*args,**kwargs)
        return run
    try:
        for name,fn in originals.items():setattr(engine,name,wrap(name,fn))
        yield
    finally:
        for name in originals:
            if existing[name] is None:engine.__dict__.pop(name,None)
            else:setattr(engine,name,existing[name])


def profile_cuda(fn,engine,torch,device):
    activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        return None,dict(profile_summary_schema=PROFILE_SCHEMA,status='CUDA_profiler_unavailable',CUDA_activity_observed=False,
            scope='CUDA component timing controls still run; no profiler-based bottleneck conclusion is available.')
    torch.cuda.synchronize(device)
    with stage_regions(engine,torch),torch.profiler.profile(activities=activities,record_shapes=False,
            profile_memory=False,with_stack=False) as profiler:
        begin=time.perf_counter();result=fn();torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-begin
    report=summarize_profile(profiler,torch,elapsed)
    report['status']='completed' if report['CUDA_activity_observed'] else 'CUDA_activity_not_observed'
    return result,report


def run(engine,words,start,identity,args,torch,device,measure):
    """Keep actual causal decode separate from known-source encoder controls."""
    if device.type!='cuda' or engine.interpreter:raise RuntimeError('diagnostic measurements require CUDA')
    raw_bytes=words.numel()*2
    packet,complete_timing=measure(lambda:engine.encode(words,start),args,torch,device,raw_bytes)
    wire=t1.export_packet(packet,identity)
    decoded,decode_timing=measure(lambda:engine.decode(packet),args,torch,device,raw_bytes)
    recovered,error,_=decoded
    if bool(error.any().item()):raise AssertionError('diagnostic actual causal decode failed')
    np.testing.assert_array_equal(recovered.cpu().numpy(),words.cpu().numpy())
    del decoded,recovered,error
    parameters,parameter_timing=measure(lambda:engine._walk(words,start),args,torch,device,raw_bytes)
    prepared,entropy_timing=measure(lambda:engine._encode_prepared(words,start,parameters),args,torch,device,raw_bytes)
    if t1.export_packet(prepared,identity)!=wire:raise AssertionError('prepared entropy control changed T1 bytes')
    del prepared,parameters
    profiled,encode_profile=profile_cuda(lambda:engine.encode(words,start),engine,torch,device)
    if profiled is not None:
        if t1.export_packet(profiled,identity)!=wire:raise AssertionError('profiling changed T1 bytes')
        del profiled
    profiled,decode_profile=profile_cuda(lambda:engine.decode(packet),engine,torch,device)
    if profiled is not None:
        decoded,error,_=profiled
        if bool(error.any().item()):raise AssertionError('profiled causal decode failed')
        np.testing.assert_array_equal(decoded.cpu().numpy(),words.cpu().numpy())
        del profiled,decoded,error
    return dict(complete_encode_gpu=complete_timing,actual_causal_decode_gpu=decode_timing,
        parameter_replay_gpu=parameter_timing,entropy_encode_with_prepared_parameters_gpu=entropy_timing,
        prepared_entropy_bytes_match_complete_encode=True,encode_profile=encode_profile,decode_profile=decode_profile,
        uninstrumented_measurements_completed_before_profiling=True,
        scope='Source-derived parameter replay and prepared entropy encoding are separate encoder timing controls. Actual causal decoding uses the packet and saved state only. Controls retain different tensors and their memory peaks are not additive. CUDA intervals can include host dispatch gaps; profiler activity is reported separately.')
