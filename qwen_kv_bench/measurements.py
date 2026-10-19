"""Bounded, record-streamed CPU/GPU KV representation measurements.

CPU reference bytes are never described as GPU-decodable. Tensor-native
packets have a separate path. Neither path simulates concurrent serving.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import struct
import tempfile
import time
import numpy as np

_OUTER_HEADER_BYTES = 32
_TIMING_RECORD_LIMIT = 32


def _inputs(matrices, tile_tokens, max_records, layer_indices):
    if tile_tokens < 1: raise ValueError('tile_tokens must be positive')
    if isinstance(matrices, dict):
        indices=sorted(int(k.split('_')[1]) for k in matrices if k.endswith('_K'))
        pairs=[(matrices[f'layer_{l:02d}_K'],matrices[f'layer_{l:02d}_V']) for l in indices]
        if layer_indices is not None and list(layer_indices)!=indices: raise ValueError('layer_indices contradict mapping')
    else:
        pairs=list(matrices)
        indices=list(range(len(pairs))) if layer_indices is None else list(layer_indices)
    if not pairs or len(indices)!=len(pairs): raise ValueError('empty or inconsistent layer selection')
    shape=np.asarray(pairs[0][0]).shape
    if len(shape)!=2 or min(shape)<1: raise ValueError('nonempty [tokens,channels] matrices required')
    for pair in pairs:
        if len(pair)!=2: raise ValueError('each layer needs K and V')
        for words in pair:
            if np.asarray(words).dtype!=np.uint16 or np.asarray(words).shape!=shape:
                raise ValueError('all KV matrices must be matching uint16 arrays')
    tokens,channels=shape
    ntiles=(tokens+tile_tokens-1)//tile_tokens
    total=len(pairs)*2*ntiles
    if max_records is not None and max_records<1: raise ValueError('max_records must be positive')
    count=total if max_records is None else min(total,int(max_records))
    selected=set(np.linspace(0,total-1,count,dtype=np.int64).tolist())
    timing=set(np.linspace(0,count-1,min(count,_TIMING_RECORD_LIMIT),dtype=np.int64).tolist())
    def records():
        ordinal=0; measured=0
        for layer,pair in zip(indices,pairs):
            for kind,words in zip(('K','V'),pair):
                for start in range(0,tokens,tile_tokens):
                    if ordinal in selected:
                        yield measured,layer,kind,start,np.ascontiguousarray(words[start:start+tile_tokens]),measured in timing
                        measured+=1
                    ordinal+=1
    meta=dict(layers=len(pairs),layer_indices=indices,tokens=tokens,channels=channels,tile_tokens=tile_tokens,
        records_total=total,records_measured=count,coverage='full' if count==total else 'sampled',
        size_is_estimated=count!=total,full_raw_payload_bytes=sum(np.asarray(w).nbytes for p in pairs for w in p))
    return records,meta


def _numeric_diagnostics(stats):
    return {k:{str(kk):vv.item() if isinstance(vv,np.generic) else vv for kk,vv in value.items()} for k,value in stats.items()
        if isinstance(value,dict) and value and all(isinstance(v,(int,float,np.integer,np.floating)) for v in value.values())}


def _add_diagnostics(target,stats):
    for name,values in _numeric_diagnostics(stats).items(): target.setdefault(name,Counter()).update(values)


def _encode(adapter,words,layer,kind):
    if getattr(adapter,'parent',None) is None: return adapter.encode(words,layer,kind)
    return adapter.encode_from_parent(_encode(adapter.parent,words,layer,kind))


def _check(actual,expected):
    actual=np.asarray(actual)
    if actual.dtype!=np.uint16 or actual.shape!=expected.shape or not np.array_equal(actual,expected):
        raise AssertionError('KV reconstruction differs from original uint16 BF16 words')


def _summary(times, payload_bytes=None):
    result=dict(seconds=[float(x) for x in times],median_seconds=float(statistics.median(times)),repeats=len(times))
    if payload_bytes is not None:
        result['payload_bytes']=int(payload_bytes)
        result['MiB_per_second']=payload_bytes/2**20/result['median_seconds'] if result['median_seconds'] else None
    return result


def _sizes(meta,payload_bytes,observed_raw,state_bytes,components):
    count=meta['records_measured']; total=meta['records_total']
    overhead=_OUTER_HEADER_BYTES+8*(count+1)
    observed=payload_bytes+overhead
    components=Counter(components); components['cache_header']+=_OUTER_HEADER_BYTES;components['record_offset_table']+=8*(count+1)
    if sum(components.values())!=observed: raise AssertionError('incomplete serialized byte accounting')
    estimated=payload_bytes*total/count+_OUTER_HEADER_BYTES+8*(total+1)
    return dict(raw_payload_bytes=observed_raw,serialized_bytes=observed,component_bytes=dict(components),
        actual_record_payload_bytes=payload_bytes,calculated_outer_framing_bytes=overhead,
        decoder_state_bytes=int(state_bytes),estimated_full_cache_serialized_bytes=estimated if meta['size_is_estimated'] else None,
        payload_space_saving_percent=100*(1-observed/observed_raw),
        savings_including_state_percent=100*(1-(observed+state_bytes)/observed_raw),
        estimated_full_savings_including_state_percent=100*(1-(estimated+state_bytes)/meta['full_raw_payload_bytes']) if meta['size_is_estimated'] else None,
        byte_accounting='Actual selected-record payload bytes plus calculated costs for the specified 32-byte cache header and 8*(record_count+1) index bytes. A complete outer cache container is not serialized or persisted by this measurement; the framing cost is included arithmetically. Full decoder state charged once. Sampled full-cache estimates assume measured bytes per record and are not full-cache measurements.')


class _Device:
    def __init__(self,device):
        import torch
        self.torch=torch;self.device=torch.device(device or 'cpu');self.kind=self.device.type
        self.api=getattr(torch,self.kind,None) if self.kind in ('cuda','xpu') else None
        self.accelerated=self.api is not None and bool(self.api.is_available())
        if self.kind!='cpu' and not self.accelerated: raise RuntimeError('requested accelerator is unavailable: '+str(self.device))
    def sync(self):
        if self.accelerated: self.api.synchronize(self.device)
    def timed(self,call):
        self.sync();start_event=end_event=None
        if self.accelerated and hasattr(self.api,'Event'):
            try:
                start_event=self.api.Event(enable_timing=True);end_event=self.api.Event(enable_timing=True)
                start_event.record(self.api.current_stream(self.device))
            except (RuntimeError,TypeError,AttributeError): start_event=end_event=None
        start=time.perf_counter();result=call()
        if end_event is not None: end_event.record(self.api.current_stream(self.device))
        self.sync();elapsed=time.perf_counter()-start
        event_elapsed=float(start_event.elapsed_time(end_event))/1000 if start_event is not None else None
        return result,elapsed,event_elapsed
    def memory_start(self):
        if not self.accelerated or not hasattr(self.api,'memory_allocated'): return None
        self.sync();baseline=int(self.api.memory_allocated(self.device))
        if hasattr(self.api,'reset_peak_memory_stats'): self.api.reset_peak_memory_stats(self.device)
        return baseline
    def memory_end(self,baseline):
        if baseline is None: return dict(measured=False,reason='no accelerator allocator statistics')
        self.sync();peak=int(self.api.max_memory_allocated(self.device))
        return dict(measured=True,baseline_allocated_bytes=baseline,peak_allocated_bytes=peak,
            peak_additional_allocated_bytes=max(0,peak-baseline),
            scope='Allocator allocated bytes during this bounded operation, excluding pre-existing model/state baseline; reserved memory and host allocations are separate.')


def _io(blob,outputdir,repeats):
    path=None;writes=[];reads=[];checksum=hashlib.sha256(blob).hexdigest()
    import os
    try:
        with tempfile.NamedTemporaryFile(prefix='kv_io_',suffix='.bin',dir=outputdir,delete=False) as handle:
            path=Path(handle.name)
        for _ in range(repeats):
            start=time.perf_counter()
            with path.open('wb') as handle: handle.write(blob);handle.flush();os.fsync(handle.fileno())
            writes.append(time.perf_counter()-start)
            start=time.perf_counter();actual=path.read_bytes();reads.append(time.perf_counter()-start)
            if hashlib.sha256(actual).hexdigest()!=checksum: raise AssertionError('buffered IO checksum mismatch')
        return dict(measured=True,bytes=len(blob),sha256=checksum,write_fsync=_summary(writes,len(blob)),
            read=_summary(reads,len(blob)),scope='Bounded concatenated encoded-record sample. Buffered filesystem write+fsync and read; reads can hit page cache. No direct IO, cache eviction, SSD/device identity or physical media bandwidth claim. Outer cache index is excluded from this transfer sample.')
    finally:
        if path is not None: path.unlink(missing_ok=True)


def _reference_transfers(samples,adapter,device,repeats):
    d=_Device(device)
    if not d.accelerated: return dict(measured=False,reason='CPU functional run; no GPU transfer timings')
    torch=d.torch
    raw=np.concatenate([expected.reshape(-1) for _,expected,_,_ in samples])
    encoded=b''.join(blob for blob,_,_,_ in samples)
    baseline=d.memory_start();results={}
    for pinned in (False,True):
        mode='pinned' if pinned else 'pageable'
        try:
            raw_source=torch.from_numpy(raw.copy());encoded_source=torch.from_numpy(np.frombuffer(encoded,dtype=np.uint8).copy())
            if pinned: raw_source=raw_source.pin_memory();encoded_source=encoded_source.pin_memory()
            raw_dest=torch.empty(raw_source.shape,dtype=raw_source.dtype,device=d.device)
            encoded_dest=torch.empty(encoded_source.shape,dtype=encoded_source.dtype,device=d.device)
            encoded_back=torch.empty_like(encoded_source,pin_memory=pinned)
            result={};event_results={}
            for name,src,dst in (('raw_h2d',raw_source,raw_dest),('encoded_h2d',encoded_source,encoded_dest),('encoded_d2h',encoded_dest,encoded_back)):
                call=lambda src=src,dst=dst:dst.copy_(src,non_blocking=pinned)
                d.timed(call);times=[];events=[]
                for _ in range(repeats):
                    _,wall,event=d.timed(call);times.append(wall)
                    if event is not None: events.append(event)
                result[name]=_summary(times,src.numel()*src.element_size())
                if events: event_results[name]=_summary(events,src.numel()*src.element_size())
            if not np.array_equal(raw_dest.cpu().numpy(),raw) or encoded_back.numpy().tobytes()!=encoded:
                raise AssertionError('GPU transfer changed original bytes')
            # Bounded batch materialization followed by one bulk copy, using a
            # reusable host buffer and GPU destination. This does not overlap.
            times=[];decode_pack=[];h2d_events=[]
            def end_to_end():
                start=time.perf_counter();cursor=0
                for blob,expected,layer,kind in samples:
                    words=adapter.decode(blob,layer,kind)
                    source=torch.from_numpy(np.ascontiguousarray(words).reshape(-1))
                    raw_source[cursor:cursor+source.numel()].copy_(source);cursor+=source.numel()
                decode_pack.append(time.perf_counter()-start)
                return raw_dest.copy_(raw_source,non_blocking=pinned)
            end_to_end();d.sync();decode_pack.clear()
            for _ in range(repeats):
                _,wall,event=d.timed(end_to_end);times.append(wall)
                if event is not None: h2d_events.append(event)
                if not np.array_equal(raw_dest.cpu().numpy(),raw): raise AssertionError('CPU decode+H2D differs')
            result['cpu_decode_pack_then_h2d']=_summary(times,raw.nbytes)
            result['cpu_decode_and_host_pack']=_summary(decode_pack,raw.nbytes)
            if h2d_events: event_results['cpu_decode_pack_then_h2d']=_summary(h2d_events,raw.nbytes)
            results[mode]=dict(measured=True,wall=result,device_event_elapsed=event_results,
                event_scope='Device stream elapsed time between events. For CPU decode+pack then copy, events can include CPU launch gaps and must not be read as pure copy-kernel time.')
        except RuntimeError as error:
            if not pinned: raise
            results[mode]=dict(measured=False,reason=str(error))
    memory=d.memory_end(baseline)
    return dict(measured=True,device=str(d.device),buffers=results,memory=memory,
        sampled_records=len(samples),raw_sample_bytes=raw.nbytes,encoded_sample_bytes=len(encoded),
        exact_byte_transfer=True,scope='Bounded sampled transfers with synchronized reusable device buffers. CPU reference encoded bytes are staging only: no GPU decoder for these original wire formats. End-to-end CPU decode+host pack then bulk H2D is serialized, with no attention or serving benchmark. Device-event durations can include host launch gaps.')


def measure_reference(matrices,adapter,device,outputdir,repeats=3,io_bytes_limit=32<<20,
                      tile_tokens=16,max_records=None,layer_indices=None,measure_io=False):
    """Stream a reference codec; size/exactness cover every selected record.

    max_records selects evenly spaced records and explicitly reports estimates.
    Repeated CPU decode and optional IO/transfers use at most 32 representative
    records and at most io_bytes_limit combined encoded+raw sample bytes.
    """
    if repeats<1 or io_bytes_limit<0: raise ValueError('invalid repeats or byte limit')
    outputdir=Path(outputdir);outputdir.mkdir(parents=True,exist_ok=True)
    records,meta=_inputs(matrices,tile_tokens,max_records,layer_indices)
    components=Counter();diagnostics={};payload=raw=0;encode_seconds=decode_seconds=0.0
    sample_index=[];sample_bytes=0;all_start=time.perf_counter()
    with tempfile.TemporaryFile(dir=outputdir) as spool:
        for index,layer,kind,start,words,want_sample in records():
            t=time.perf_counter();record=_encode(adapter,words,layer,kind);encode_seconds+=time.perf_counter()-t
            blob=bytes(record.data)
            t=time.perf_counter();restored=adapter.decode(blob,layer,kind);decode_seconds+=time.perf_counter()-t
            _check(restored,words)
            payload+=len(blob);raw+=words.nbytes
            parts=record.stats.get('component_bytes',{'unclassified_complete_record':len(blob)})
            if sum(parts.values())!=len(blob): raise AssertionError('codec component bytes do not sum')
            components.update(parts);_add_diagnostics(diagnostics,record.stats)
            if want_sample and sample_bytes+len(blob)+words.nbytes<=io_bytes_limit:
                offset=spool.tell();spool.write(blob);spool.write(words.astype('<u2').tobytes())
                sample_index.append((offset,len(blob),words.shape,words.nbytes,layer,kind,index,start))
                sample_bytes+=len(blob)+words.nbytes
        samples=[]
        for offset,length,shape,nbytes,layer,kind,index,start in sample_index:
            spool.seek(offset);blob=spool.read(length);expected=np.frombuffer(spool.read(nbytes),dtype='<u2').reshape(shape).copy()
            samples.append((blob,expected,layer,kind))
    result=dict(name=adapter.name,execution='cpu_reference',**meta,
        **_sizes(meta,payload,raw,getattr(adapter,'state_bytes',0),components),
        decoder_state_info=getattr(adapter,'state_info',{}),traits=getattr(adapter,'traits',{}),
        roundtrip_exact=True,exact_verified=True,records_checked=meta['records_measured'],diagnostics={k:dict(v) for k,v in diagnostics.items()},
        encode_seconds_total=encode_seconds,decode_seconds_total=decode_seconds,
        one_pass_total_wall_seconds=time.perf_counter()-all_start,
        dependency=getattr(adapter,'dependency',None),wrapper_parent_bytes_reused=bool(getattr(adapter,'parent',None)),
        sample=dict(records=len(samples),combined_encoded_raw_bytes=sample_bytes,byte_limit=io_bytes_limit,
            record_limit=_TIMING_RECORD_LIMIT,selection='evenly spaced selected-record ordinals, subject to combined byte budget'))
    if samples:
        for blob,expected,layer,kind in samples: _check(adapter.decode(blob,layer,kind),expected)
        times=[]
        for _ in range(repeats):
            elapsed=0.0
            for blob,expected,layer,kind in samples:
                t=time.perf_counter();restored=adapter.decode(blob,layer,kind);elapsed+=time.perf_counter()-t
                _check(restored,expected)
            times.append(elapsed)
        result['sample_decode']=dict(**_summary(times,sum(w.nbytes for _,w,_,_ in samples)),
            exact=True,scope='CPU codec decode calls only; sample already in host memory, equality checking excluded; one warmup pass.')
        result['transfers']=_reference_transfers(samples,adapter,device,repeats)
        result['buffered_io']=_io(b''.join(b for b,_,_,_ in samples),outputdir,repeats) if measure_io else dict(measured=False,reason='not requested')
    else:
        result['sample_decode']=dict(measured=False,reason='sample byte budget retains no records')
        result['transfers']=dict(measured=False,reason='no sampled records')
        result['buffered_io']=dict(measured=False,reason='no sampled records')
    return result


def measure_native(matrices,native_spec,state,device,outputdir,repeats=3,io_bytes_limit=32<<20,
                   tile_tokens=16,max_records=None,layer_indices=None,measure_io=False):
    """Measure separately named gpu_ops tensor packets, one tile at a time.

    CPU executes the same tensor functions for functional checking only. State
    physical tensor bytes are charged once after lazy rotations are created.
    """
    from . import gpu_ops
    if repeats<1 or io_bytes_limit<0: raise ValueError('invalid repeats or byte limit')
    outputdir=Path(outputdir);outputdir.mkdir(parents=True,exist_ok=True)
    records,meta=_inputs(matrices,tile_tokens,max_records,layer_indices)
    d=_Device(device);torch=d.torch
    state=state or gpu_ops.NativeState()
    name=native_spec.get('name',native_spec['method'])
    options=dict(native_spec.get('options',{}))
    components=Counter();payload=raw=0;encode_times=[];decode_times=[];input_times=[];encode_events=[];decode_events=[]
    samples=[];sample_bytes=0;baseline=d.memory_start();all_start=time.perf_counter()
    runtime_diagnostics={};entropy_backend=None;preloaded_rotation_ids=dict(state.rotation_ids)
    prefill_recipe_totals=Counter();prefill_recipe_description={}
    for index,layer,kind,start,words,want_sample in records():
        source=torch.from_numpy(words)
        tensor,input_elapsed,_=d.timed(lambda:source.to(device=d.device,copy=True))
        input_times.append(input_elapsed)
        key=f'layer_{layer:02d}_{kind}'
        packet,elapsed,event=d.timed(lambda:gpu_ops.encode(tensor,state=state,method=native_spec['method'],key=key,**options))
        encode_times.append(elapsed)
        if event is not None: encode_events.append(event)
        restored,elapsed,event=d.timed(lambda:gpu_ops.decode(packet,state))
        decode_times.append(elapsed)
        if event is not None: decode_events.append(event)
        _check(restored.detach().cpu().numpy(),words)
        stats=packet.stats
        if 'prefill_recipe' in stats:
            for key,value in stats['prefill_recipe'].items():
                if isinstance(value,(int,float)):
                    prefill_recipe_totals[key]+=value
                else:
                    prefill_recipe_description[key]=value
        runtime_diagnostics.update(stats.get('runtime_diagnostics',{}))
        entropy_backend=stats.get('native_entropy_backend')
        parts=stats['component_bytes']
        if sum(parts.values())!=stats['serialized_bytes']: raise AssertionError('native packet accounting differs')
        payload+=stats['serialized_bytes'];raw+=words.nbytes;components.update(parts)
        if want_sample and sample_bytes+stats['serialized_bytes']+words.nbytes<=io_bytes_limit:
            cpu_packet=gpu_ops.packet_to_cpu(packet)
            samples.append((cpu_packet,words.copy()))
            sample_bytes+=stats['serialized_bytes']+words.nbytes
        del packet,restored,tensor
    memory=d.memory_end(baseline)
    result=dict(name=name,execution='tensor_native_accelerator' if d.accelerated else 'tensor_native_cpu_functional',
        **meta,**_sizes(meta,payload,raw,state.physical_nbytes,components),roundtrip_exact=True,exact_verified=True,
        records_checked=meta['records_measured'],original_cpu_format=False,
        input_staging_seconds_total=sum(input_times),encode_seconds_total=sum(encode_times),decode_seconds_total=sum(decode_times),
        one_pass_total_wall_seconds=time.perf_counter()-all_start,memory=memory,
        native_state_physical_bytes=state.physical_nbytes,state_id=state.state_id,
        runtime_diagnostics=runtime_diagnostics,native_entropy_backend=entropy_backend,
        host_metadata_prefetched=bool(runtime_diagnostics.get('prefetched_validated_host_launch_descriptor',False)),
        rotation_ids=[dict(dimension=int(key[0]),transform=key[1],seed=int(key[2]),sha256=value,
            preloaded_before_measurement=key in preloaded_rotation_ids) for key,value in sorted(state.rotation_ids.items())],
        rotation_basis='Caller-supplied explicit float64 matrix; the Qwen runner supplies the same NumPy CPU reference rotation.' if preloaded_rotation_ids else 'No preloaded rotation; any generated gpu_ops basis is identified by its recorded SHA256.',
        decoder_state_info=dict(accounting='Actual distinct tensor storage bytes, including lazily created rotations; Python metadata and allocator reserved bytes are not included.'),
        device=str(d.device),sample=dict(records=len(samples),combined_packet_raw_bytes=sample_bytes,byte_limit=io_bytes_limit),
        device_event_elapsed=dict(encode_seconds_total=sum(encode_events) if encode_events else None,
                                  decode_seconds_total=sum(decode_events) if decode_events else None),
        scope='Separately named tensor packet format, not a GPU implementation of the CPU entropy wire formats. Record input CPU→device staging, device encode and device decode are separately synchronized. Full cache is not retained on accelerator. Packet JSON headers and all physical tensor payload bytes are measured; specified 32-byte cache framing and offset costs are added arithmetically, with no complete outer container persisted. Prefetched validated host launch descriptors and CPU descriptor guards are included in native execution; authoritative JSON header parsing/validation occurs during explicit packet transfer/restore. CPU runs establish functionality only. No attention, concurrency, or admission benefit is measured.')
    if prefill_recipe_description:
        result['prefill_recipe']=dict(prefill_recipe_description,**prefill_recipe_totals)
    if samples:
        # Sample packet transfer happens outside isolated decode timing.
        sample_times=[];sample_events=[];restore_times=[]
        for repeat in range(repeats+1):
            elapsed_total=0.;event_total=0.;restore_total=0.
            for cpu_packet,words in samples:
                packet,transfer_elapsed,_=d.timed(lambda:gpu_ops.packet_to_device(cpu_packet,d.device))
                restored,elapsed,event=d.timed(lambda:gpu_ops.decode(packet,state))
                _check(restored.detach().cpu().numpy(),words)
                elapsed_total+=elapsed;event_total+=event or 0.;restore_total+=transfer_elapsed+elapsed
                del packet,restored
            if repeat:
                sample_times.append(elapsed_total);restore_times.append(restore_total)
                if d.accelerated: sample_events.append(event_total)
        result['sample_decode']=dict(**_summary(sample_times,sum(w.nbytes for _,w in samples)),exact=True,
            device_event_elapsed=_summary(sample_events) if sample_events else None,
            scope='Sample packet already on target device for each isolated decode call; explicit packet copy and authoritative header parsing excluded. Decode includes CPU launch-descriptor guards. One warmup.')
        result['sample_packet_transfer_then_decode']=dict(**_summary(restore_times,sum(w.nbytes for _,w in samples)),
            device_transfer_measured=d.accelerated,scope='Explicit per-field pageable packet CPU→device copy plus synchronized device decode, allocation and authoritative JSON header validation included. CPU functional copy when no accelerator. Reconstructed-word validation D2H excluded.')
        if measure_io:
            blob=b''.join(p.header.detach().cpu().numpy().tobytes()+b''.join(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() for v in p.fields.values()) for p,_ in samples)
            result['buffered_io']=_io(blob,outputdir,repeats)
        else: result['buffered_io']=dict(measured=False,reason='not requested')
    else:
        result['sample_decode']=dict(measured=False,reason='sample byte budget retains no packets')
        result['sample_packet_transfer_then_decode']=dict(measured=False,reason='no sampled packets')
        result['buffered_io']=dict(measured=False,reason='no sampled packets')
    return result
