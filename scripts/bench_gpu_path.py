#!/usr/bin/env python3
"""Batch P1 attention, T1 GPU entropy and external T/P bridge measurements.

This component suite does not integrate a native serving cache, generate model
tokens, or establish multi-request capacity / cold SSD / network throughput.
"""
import argparse
import dataclasses
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import bench_pool_p1 as common
from qwen_kv_bench import pool_p1 as p,pool_attention as attention,transfer_reference as ref,transfer_probability as prob,transfer_t1 as t1,gpu_kv_bridge as bridge


def log(message):print(time.strftime('[%H:%M:%S] ')+message,flush=True)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def load_frozen_model(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());name=m['state_file'];file=root/name
    if file.is_file():
        item=m['files'][name]
        if file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:raise ValueError('frozen predictor checksum mismatch')
        model=ref.load_model(file)
    else:
        chunks=[]
        for part in m['distribution']['parts']:
            f=common.safe_file(root,part['path'])
            if f.stat().st_size!=part['bytes'] or sha(f)!=part['sha256']:raise ValueError('frozen predictor part checksum mismatch')
            chunks.append(f.read_bytes())
        data=b''.join(chunks)
        if len(data)!=m['distribution']['reassembled_bytes'] or hashlib.sha256(data).hexdigest()!=m['distribution']['reassembled_sha256']:raise ValueError('reassembled predictor mismatch')
        model=ref.load_model(io.BytesIO(data))
    if model.state_sha256!=m['state']['state_sha256']:raise ValueError('wrong frozen mean predictor identity')
    return model


def datasets(args):
    result=[]
    cases,identity=common.fixture_cases(args.fixture,args.tile_tokens)
    base=load_frozen_model(args.fixture)
    result.append(dict(name='frozen_smol',cases=cases,identity=identity,base=base,
        states={'A':prob.empty_state(base,'A')},
        scope='Frozen model words; fixed-width A. The fixture contains no fitted B adapter.'))
    if args.captures_from:
        cases,identity=common.captured_cases(args.captures_from,args.tile_tokens,args.page_tokens,args.page_starts)
        calibration=Path(args.captures_from)/'calibration'
        base=ref.load_model(calibration/'base.npz')
        states={arm:prob.load_state(calibration/(arm+'_decoder.npz')) for arm in args.arms.split(',')}
        for arm,state in states.items():
            if state.base_sha256!=base.state_sha256 or state.arm!=arm:raise ValueError('mismatched saved Qwen probability state')
        identity['base_file_sha256']=sha(calibration/'base.npz')
        identity['adapter_file_sha256']={a:sha(calibration/(a+'_decoder.npz')) for a in states}
        result.append(dict(name='qwen_reuse',cases=cases,identity=identity,base=base,states=states,
            scope='Reuses existing exploratory TEST pages and frozen states; not fresh holdout or full-model Qwen.'))
    return result


def extract_previous(path):
    r=json.loads((Path(path)/'results.json').read_text())
    if r.get('status')!='completed' or not r.get('coverage',{}).get('gpu_measured'):raise ValueError('previous P1 result must be completed actual CUDA')
    rows=[]
    for x in r['rows']:
        if x['software_case']:continue
        rows.append(dict(name=x['name'],calibration_sha256=x['calibration_sha256'],
            arenas=[dict(ratio=a['arena_ratio_of_padded_raw'],stats=a['stats'],kv_to_p=a['kv_to_p'],
                full_sector_unpack=a['full_sector_unpack'],selected_tile_reads=a['selected_tile_reads']) for a in x['arenas']],p0=x.get('p0_control')))
    return dict(path=str(path),results_sha256=sha(Path(path)/'results.json'),sources=r['sources'],device=r['device'],rows=rows,
        scope='Historical source hashes retained; current fused loader refactor is separately tested. No timing is substituted for a new run.')


def checked_status(status,label):
    if np.any(status.detach().cpu().numpy()!=0):raise AssertionError(label+' returned invalid status')


def pool(case,torch,device,ratio):
    begin=time.perf_counter()
    ds=p.to_cuda(case.state,device);source=torch.from_numpy(case.words.copy()).to(device)
    capacity=max(16,int(np.ceil(case.layout.padded_raw_bytes*ratio/16))*16)
    requested=capacity
    arena=p.allocate_arena(case.layout,ds,capacity);p.encode_into(source,ds,arena)
    stats=p.arena_statistics(arena,ds)
    rejected=stats['unpublished_sectors']
    if rejected and capacity<case.layout.padded_raw_bytes:
        # A declared raw-capacity retry, not a TEST-selected optimal budget.
        # No rejected partial pool is exposed to attention. Original GPU words
        # remain live until the complete retry has passed its status check.
        del arena
        capacity=case.layout.padded_raw_bytes
        arena=p.allocate_arena(case.layout,ds,capacity);p.encode_into(source,ds,arena)
        stats=p.arena_statistics(arena,ds)
    if stats['unpublished_sectors']:raise RuntimeError('P1 cannot admit complete chunk even at the full raw-capacity control')
    stats.update(requested_arena_capacity_bytes=requested,requested_capacity_rejected=bool(rejected),
        first_attempt_unpublished_sectors=rejected,cold_pool_setup_wall_seconds=time.perf_counter()-begin)
    return ds,source,arena,stats


def measure(fn,args,torch,device,raw_bytes,attention_case=False):
    value,timing=common.measure_gpu(fn,torch,device,args.warmup,args.attention_repeats if attention_case else args.repeats,raw_bytes)
    timing['scope']='Resident inputs and shared state; allocations made by this callback are included. CUDA stream intervals may include host dispatch gaps. Synchronized wall time and Torch allocation/reservation samples are recorded. On-chip registers/shared memory and driver allocations are outside Torch counters.'
    return value,timing


def parameter_oracle(engine,source,case,start):
    """Validation outside codec timings; CPU reference is never a decoder input."""
    begin=time.perf_counter();g=case.layout;half=g.channels//2
    capture=dict(K_bits=case.words[:,:,:half],V_bits=case.words[:,:,half:],Hk=np.array(g.heads),
        hd=np.array(g.head_dim),base=np.array(engine.base.geometry['base']),positions=np.arange(start,start+g.tokens))
    # trace needs bucket thresholds even for the fixed-width A control.
    check_state=engine.adapter if engine.adapter.arm=='B' else prob.empty_state(engine.base,'B')
    rows=prob.trace(engine.base,capture,check_state)
    records=engine._walk(source,start)
    if len(rows)!=len(records):raise AssertionError('parameter trace length mismatch')
    histogram=np.zeros((case.layout.layers,33),np.int64)
    for (layer,_,mu,inverse,off),row in zip(records,rows):
        expected_off=prob.offsets(engine.adapter,row['context'],row['layer'],row['group'])
        np.testing.assert_array_equal(mu.cpu().numpy(),row['mu'])
        np.testing.assert_array_equal(off.cpu().numpy(),expected_off)
        np.testing.assert_array_equal(inverse.cpu().numpy(),row['inverse']*engine.adapter.scale_table[expected_off+prob.MAX_OFFSET,None])
        histogram[layer]+=np.bincount(off.cpu().numpy()+prob.MAX_OFFSET,minlength=33)
    return dict(mean_scale_offset_exact=True,groups=len(rows),wall_seconds=time.perf_counter()-begin,
        GPU_width_offset_histogram_by_layer=histogram.tolist(),
        scope='CPU parameter oracle outside timings. Proves the clipping safeguard did not change the frozen A/B parameters on this source page.')


def attention_case(case,args,torch,device,tail_tokens):
    g=case.layout
    if tail_tokens:
        prefix=(g.tokens//g.tile_tokens-1)*g.tile_tokens
        if prefix<g.tile_tokens:raise ValueError('attention raw-tail control needs at least two complete source tiles')
        words=case.words[:,:prefix].copy();tail=case.words[:,prefix:prefix+tail_tokens].copy()
        layout=p.Layout(g.layers,prefix,g.heads,g.head_dim,g.tile_tokens)
    else:words=case.words;tail=np.empty((g.layers,0,g.channels),np.uint16);layout=g
    pc=dataclasses.replace(case,words=words,layout=layout)
    ds,source,arena,stats=pool(pc,torch,device,args.arena_ratio)
    rng=np.random.default_rng(902);q=rng.normal(size=(g.layers,args.gqa_multiple*g.heads,g.head_dim)).astype(np.float32)
    query=torch.from_numpy(q).to(device);rawtail=torch.from_numpy(tail).to(device)
    expected=attention.reference(q,words,layout,tail)
    begin=time.perf_counter();cold,status=attention.packed_attention(query,arena,ds,rawtail);torch.cuda.synchronize(device)
    cold_seconds=time.perf_counter()-begin;checked_status(status,'cold P1 attention');del cold,status
    # Release the paired pool before dense timing, then recreate it before
    # packed timing. Each primary memory baseline owns its actual cache form.
    capacity=arena.payload.numel();del arena,ds
    dense,dt=measure(lambda:attention.dense_attention(query,source,layout,rawtail),args,torch,device,layout.raw_bytes+tail.nbytes,True)
    dout,dstatus=dense;checked_status(dstatus,'dense attention');dense_np=dout.cpu().numpy().copy();del dense,dout,dstatus
    dense_sequence,ds_timing=measure(lambda:[attention.dense_attention(query[l:l+1],source,layout,rawtail,layer_start=l) for l in range(g.layers)],args,torch,device,layout.raw_bytes+tail.nbytes,True)
    for l,(o,s) in enumerate(dense_sequence):
        checked_status(s,'sequential dense attention');np.testing.assert_allclose(o.cpu().numpy()[0],dense_np[l],rtol=3e-6,atol=3e-6)
    del dense_sequence,o,s
    ds=p.to_cuda(pc.state,device);arena=p.allocate_arena(layout,ds,capacity)
    p.encode_into(source,ds,arena);torch.cuda.synchronize(device)
    # The timed packed path has no original GPU cache buffer to consult.
    del source
    packed,pt=measure(lambda:attention.packed_attention(query,arena,ds,rawtail),args,torch,device,layout.raw_bytes+tail.nbytes,True)
    pout,pstatus=packed;checked_status(pstatus,'packed attention');actual=pout.cpu().numpy()
    np.testing.assert_allclose(actual,dense_np,rtol=3e-6,atol=3e-6)
    np.testing.assert_allclose(actual,expected,rtol=4e-6,atol=4e-6)
    del packed,pout,pstatus
    sequence_max=0.;sequence_reference_max=0.
    sequence,ps_timing=measure(lambda:[attention.packed_attention(query[l:l+1],arena,ds,rawtail,layer_start=l) for l in range(g.layers)],args,torch,device,layout.raw_bytes+tail.nbytes,True)
    for l,(o,s) in enumerate(sequence):
        checked_status(s,'sequential P1 attention');np.testing.assert_allclose(o.cpu().numpy()[0],dense_np[l],rtol=3e-6,atol=3e-6)
        sequence_max=max(sequence_max,float(np.max(np.abs(o.cpu().numpy()[0]-dense_np[l]))))
        sequence_reference_max=max(sequence_reference_max,float(np.max(np.abs(o.cpu().numpy()[0]-expected[l]))))
    del sequence,o,s
    return dict(name=case.name,provenance=case.provenance,query_seed=902,tail_tokens=tail.shape[1],prefix_tokens=layout.tokens,layout=layout.description(),
        pool=stats,query_kind='seeded FLOAT32 component queries, not captured model queries',gqa_multiple=args.gqa_multiple,
        packed_cold_seconds_including_JIT=cold_seconds,dense=ds_timing,packed=ps_timing,all_layers_batched_dense=dt,all_layers_batched_packed=pt,
        maximum_output_abs_difference=sequence_max,maximum_reference_abs_difference=sequence_reference_max,
        all_layers_batched_maximum_output_abs_difference=float(np.max(np.abs(actual-dense_np))),
        original_dense_GPU_input_released_for_packed_timing=True,
        paired_P1_released_for_dense_timing=True,
        scope='Primary times are sequential layer calls, with the same FP32 arithmetic. All-layer batching is a separate throughput diagnostic. No whole-cache BF16 unpack inside packed attention. Not logits, generated tokens or a serving capacity result.')


def isolated_decode(packet_path,identity,source_spec,output,tag,args,expected_words_sha256):
    identity_path=output/'storage'/(tag+'_identity.json');save(identity_path,identity)
    child_output=output/'isolated'/tag
    cmd=[sys.executable,str(Path(__file__).resolve()),'--decode-child',str(packet_path.resolve()),
        '--identity-file',str(identity_path.resolve()),'--output',str(child_output.resolve()),'--device',args.device]
    if source_spec['kind']=='fixture':cmd+=['--fixture',str(Path(source_spec['path']).resolve())]
    else:cmd+=['--base-state-file',str(Path(source_spec['base']).resolve()),'--adapter-state-file',str(Path(source_spec['adapter']).resolve())]
    begin=time.perf_counter();result=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    wall=time.perf_counter()-begin;logfile=Path(str(child_output)+'.log');logfile.parent.mkdir(parents=True,exist_ok=True);logfile.write_text(result.stdout)
    if result.returncode:raise RuntimeError('fresh-process GPU decode failed: '+result.stdout[-4000:])
    checked=json.loads((child_output/'decoder.json').read_text())
    if checked['status']!='completed' or not checked['gpu_measured'] or checked['decoded_words_sha256']!=expected_words_sha256:raise AssertionError('fresh-process GPU decode hash/coverage mismatch')
    return dict(**checked,process_wall_seconds=wall,original_KV_supplied=False,
        scope='Fresh process receives only saved state, expected model identity and T packet; emits a decoded-word hash, not raw KV.')


def transfer_case(case,engine,args,torch,device,output,tag,model_identity,source_spec=None,*,diagnose=False):
    g=case.layout;start=int(case.provenance['token_start'])
    ds,source,source_pool,stats=pool(case,torch,device,args.arena_ratio)
    oracle_report=parameter_oracle(engine,source,case,start)
    begin=time.perf_counter();cold=engine.encode(source,start);torch.cuda.synchronize(device)
    cold_seconds=time.perf_counter()-begin;checked_status(cold.encoder_error,'cold T1 encode');del cold
    packet,encode_timing=measure(lambda:engine.encode(source,start),args,torch,device,g.raw_bytes)
    checked_status(packet.encoder_error,'T1 encode')
    del source
    decoded,decode_timing=measure(lambda:engine.decode(packet),args,torch,device,g.raw_bytes)
    words,error,_=decoded;checked_status(error,'T1 decode');np.testing.assert_array_equal(words.cpu().numpy(),case.words);del decoded,words,error
    begin=time.perf_counter();wire=t1.export_packet(packet,model_identity);export_seconds=time.perf_counter()-begin
    accounting=t1.packet_accounting(wire)
    # A matched framing control: identical entropy payload, lane directory and
    # identity binding; only the representation of that identity changes.
    legacy_wire=t1.export_packet(packet,model_identity,identity_mode='full')
    legacy_accounting=t1.packet_accounting(legacy_wire)
    legacy_import=t1.import_packet(legacy_wire,engine,model_identity)
    if not torch.equal(legacy_import.payload,packet.payload) or not torch.equal(legacy_import.directory,packet.directory):
        raise AssertionError('legacy identity envelope changed the encoded stream')
    del legacy_wire,legacy_import
    pt,pto_timing=measure(lambda:bridge.encode_pool(engine,source_pool,ds,start),args,torch,device,g.raw_bytes)
    checked_status(pt.encoder_error,'P1 to T1')
    # Same original words, state and lane order must give identical packets.
    if t1.export_packet(pt,model_identity)!=wire:raise AssertionError('P1-to-T1 changed the encoded words/parameters')
    del pt
    begin=time.perf_counter();imported=t1.import_packet(wire,engine,model_identity);torch.cuda.synchronize(device)
    import_seconds=time.perf_counter()-begin
    installed,tto_p_timing=measure(lambda:bridge.install_pool(engine,imported,ds,g,source_pool.payload.numel()),args,torch,device,g.raw_bytes)
    target,error=installed;checked_status(error,'T1 to P1');del installed,error
    ids=torch.arange(g.sectors,device=device,dtype=torch.int64);tiles,status=p.unpack_tiles(target,ds,ids)
    checked_status(status,'installed pool')
    np.testing.assert_array_equal(tiles.cpu().numpy(),p.original_tiles(case.words,g,np.arange(g.sectors)))
    del tiles,status,ids,target
    raw_result,raw_unpack_timing=measure(lambda:bridge.unpack_words(source_pool,ds),args,torch,device,g.raw_bytes)
    raw_words,raw_status=raw_result;checked_status(raw_status,'raw transfer control')
    raw_pool,raw_pack_timing=measure(lambda:bridge.pack_words(raw_words,ds,g,source_pool.payload.numel()),args,torch,device,g.raw_bytes)
    if p.arena_statistics(raw_pool,ds)['unpublished_sectors']:raise AssertionError('raw transfer control failed to install')
    del raw_result,raw_words,raw_status,raw_pool
    coding=dict(raw=packet.raw,payload=packet.payload.numel(),directory=packet.directory.numel()*8,
        workspace=packet.encoder_workspace_capacity_bytes)
    capacity=source_pool.payload.numel()
    del packet,imported
    torch.cuda.synchronize(device)
    baseline=torch.cuda.memory_allocated(device);reserved_baseline=torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    # Measure the actual serial cycle directly, rather than summing medians.
    begin=time.perf_counter();cycle_packet=bridge.encode_pool(engine,source_pool,ds,start)
    external=t1.export_packet(cycle_packet,model_identity)
    del cycle_packet
    recovered,storage=bridge.filesystem_roundtrip(external,output/'storage'/(tag+'.t1'))
    # The file and its directory are fsynced. No readers/writer run concurrently
    # in this component, so source ownership can be released before restore.
    del source_pool
    restored=t1.import_packet(recovered,engine,model_identity)
    final,error=bridge.install_pool(engine,restored,ds,g,capacity)
    torch.cuda.synchronize(device);cycle_seconds=time.perf_counter()-begin;checked_status(error,'external install')
    cycle_memory=dict(baseline_allocated_bytes=baseline,baseline_reserved_bytes=reserved_baseline,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        incremental_peak_allocated_bytes=torch.cuda.max_memory_allocated(device)-baseline,
        source_pool_released_after_file_and_directory_fsync=True)
    q=np.random.default_rng(903).normal(size=(g.layers,args.gqa_multiple*g.heads,g.head_dim)).astype(np.float32)
    query=torch.from_numpy(q).to(device)
    answer,status=attention.packed_attention(query,final,ds);checked_status(status,'restored attention')
    np.testing.assert_allclose(answer.cpu().numpy(),attention.reference(q,case.words,g),rtol=4e-6,atol=4e-6)
    final_stats=p.arena_statistics(final,ds)
    result=dict(name=case.name,start=start,provenance=case.provenance,restored_attention_query_seed=903,parameter_oracle=oracle_report,original_words=int(case.words.size),original_bytes=g.raw_bytes,
        raw_fallback=coding['raw'],T_payload_bytes=coding['payload'],T_directory_bytes=coding['directory'],
        complete_external_T_bytes=len(wire),complete_T_bpv=len(wire)*8/case.words.size,
        T_packet_accounting=accounting,legacy_full_identity_accounting=legacy_accounting,
        identity_compaction_saved_bytes=legacy_accounting['complete_bytes']-len(wire),
        external_saving_fraction=1-len(wire)/g.raw_bytes,P=stats,restored_P=final_stats,
        encoder_lane_workspace_capacity_bytes=coding['workspace'],
        cold_first_encode_seconds_including_JIT=cold_seconds,kv_to_T_gpu_body=encode_timing,
        T_to_KV_gpu=decode_timing,P_to_T_gpu_body=pto_timing,T_to_P_gpu_composed=tto_p_timing,
        T_export_D2H_and_CPU_envelope_seconds=export_seconds,T_import_CPU_envelope_and_H2D_seconds=import_seconds,
        raw_control_P_to_words=raw_unpack_timing,raw_control_words_to_P=raw_pack_timing,
        serial_P_T_filesystem_T_P_seconds=cycle_seconds,serial_cycle_memory=cycle_memory,filesystem=storage,exact=True,restored_attention_matches_dense=True,
        scope='GPU probabilities/entropy and causal contexts. CPU reads final byte count for compaction allocation; CPU export/import frames and checksums compressed bytes. At-most-32-token word scratch. Composed bridge, not fused two-layer scratch. No network or serving integration.')
    if all('wall_median_ms' in x for x in [pto_timing,tto_p_timing,raw_unpack_timing,raw_pack_timing]):
        extra=pto_timing['wall_median_ms']+tto_p_timing['wall_median_ms']-raw_unpack_timing['wall_median_ms']-raw_pack_timing['wall_median_ms']
        saved=g.raw_bytes-len(wire)
        result['serial_link_diagnostic']=dict(extra_GPU_codec_ms=extra,bytes_saved_vs_unframed_raw=saved,
            break_even_MB_per_second=(saved/(extra*1000) if extra>0 and saved>0 else None),
            scope='Diagnostic serial model only: extra GPU codec time versus P-to-raw/raw-to-P, unframed raw bytes, no overlap. Excludes CPU envelope/staging differences. Not a measured network result.')
    if source_spec is not None and not getattr(args,'skip_isolated',False):
        expected=hashlib.sha256(case.words.tobytes()).hexdigest()
        result['isolated_GPU_decoder']=isolated_decode(output/'storage'/(tag+'.t1'),model_identity,source_spec,output,tag,args,expected)
    if args.cpu_t0_control:
        begin=time.perf_counter();old,info=prob.encode_words(engine.base,engine.adapter,case.words,start)
        restored,decode_info=prob.decode(engine.base,engine.adapter,old)
        np.testing.assert_array_equal(restored,case.words)
        result['paired_CPU_T0']=dict(bytes=len(old),bpv=len(old)*8/case.words.size,encode_and_decode_wall_seconds=time.perf_counter()-begin,
            scope='CPU reference; same original words and frozen A/B state, different framing/finite-precision coder. Not GPU timing.')
    if diagnose:
        from scripts import gpu_path_diagnostics
        diagnostic_words,diagnostic_error=bridge.unpack_words(final,ds)
        checked_status(diagnostic_error,'diagnostic source pool')
        result['bottleneck_diagnostics']=gpu_path_diagnostics.run(engine,diagnostic_words,start,model_identity,args,torch,device,measure)
    return result


def child(args):
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',gpu_measured=False)
    try:
        if os.environ.get('TRITON_INTERPRET')=='1':raise RuntimeError('GPU child refuses CPU interpreter')
        torch,device=p._cuda_torch(args.device);torch.cuda.set_device(device);torch.set_num_threads(1)
        begin=time.perf_counter()
        if args.base_state_file:
            base=ref.load_model(args.base_state_file);state=prob.load_state(args.adapter_state_file)
        else:
            base=load_frozen_model(args.fixture);state=prob.empty_state(base,'A')
        report['CPU_state_load_seconds']=time.perf_counter()-begin
        begin=time.perf_counter();engine=t1.Engine(base,state,device);torch.cuda.synchronize(device)
        report['cold_engine_initialize_seconds']=time.perf_counter()-begin
        identity=json.loads(args.identity_file.read_text());data=args.decode_child.read_bytes()
        begin=time.perf_counter();packet=t1.import_packet(data,engine,identity);torch.cuda.synchronize(device)
        report['import_and_H2D_seconds']=time.perf_counter()-begin
        begin=time.perf_counter();words,error,_=engine.decode(packet);torch.cuda.synchronize(device)
        report['cold_decode_wall_seconds_including_JIT_cache_load']=time.perf_counter()-begin
        checked_status(error,'fresh-process T1 decode')
        report.update(status='completed',gpu_measured=True,decoded_words_sha256=hashlib.sha256(words.cpu().numpy().tobytes()).hexdigest(),
            original_words=words.numel(),raw_fallback=packet.raw,device=torch.cuda.get_device_name(device),
            state=engine.statistics(),peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        code=0
    except Exception as e:report.update(status='failed',error=str(e),traceback=traceback.format_exc());code=1
    save(args.output/'decoder.json',report);print(json.dumps(report),flush=True);return code


def limited(cases,n):
    if n==0 or len(cases)<=n:return cases
    return [cases[i] for i in np.linspace(0,len(cases)-1,n,dtype=np.int64)]


def highlights(r):
    def timing(x):
        return dict(wall_median_ms=x['wall_median_ms'],cuda_interval_median_ms=x['cuda_interval_median_ms'],
            incremental_peak_allocated_bytes=x['incremental_peak_allocated_bytes'],
            maximum_total_allocated_bytes=max(y['peak_allocated_bytes'] for y in x['samples']),
            maximum_total_reserved_bytes=max(y['peak_reserved_bytes'] for y in x['samples']))
    h=dict(status=r['status'],coverage=r['coverage'],attention=[],transfer=[],B_vs_A=[],historical_P1=[])
    for x in r['attention']:
        h['attention'].append(dict(dataset=x['dataset'],name=x['name'],tail_tokens=x['tail_tokens'],dense=timing(x['dense']),packed=timing(x['packed']),
            P1_packet_bpv=x['pool']['resident_packet_allocated_bpv'],P1_shared_calibration_bytes=x['pool']['calibration_device_bytes'],
            maximum_output_abs_difference=x['maximum_output_abs_difference']))
    for x in r['transfer']:
        row=dict(dataset=x['dataset'],name=x['name'],arm=x['arm'],complete_T_bpv=x['complete_T_bpv'],
            external_saving_fraction=x['external_saving_fraction'],raw_fallback=x['raw_fallback'],exact=x['exact'],
            composed_filesystem_cycle_ms=x['serial_P_T_filesystem_T_P_seconds']*1000)
        for key in ['kv_to_T_gpu_body','T_to_KV_gpu','P_to_T_gpu_body','T_to_P_gpu_composed','raw_control_P_to_words','raw_control_words_to_P']:row[key]=timing(x[key])
        row['serial_link_diagnostic']=x.get('serial_link_diagnostic')
        row['serial_cycle_memory']=x['serial_cycle_memory']
        for key in ['T_packet_accounting','legacy_full_identity_accounting','identity_compaction_saved_bytes','bottleneck_diagnostics']:
            if key in x:row[key]=x[key]
        h['transfer'].append(row)
        if x['arm']=='B':
            paired=next((a for a in r['transfer'] if a['arm']=='A' and a['dataset']==x['dataset'] and a['name']==x['name']),None)
            if paired:h['B_vs_A'].append(dict(dataset=x['dataset'],name=x['name'],A_bytes=paired['complete_external_T_bytes'],B_bytes=x['complete_external_T_bytes'],
                B_packet_saving_fraction=1-x['complete_external_T_bytes']/paired['complete_external_T_bytes'],
                scope='Paired T1 coder and source words; width adapter changes only. Separate from T0 and P1 rates.'))
    for x in r.get('previous_P1',{}).get('rows',[]):
        p0=x.get('p0') or {}
        for ar in x['arenas']:
            row=dict(name=x['name'],arena_ratio=ar['ratio'],P1_allocated_packet_bpv=ar['stats']['allocated_packet_bpv'],
                P1_pack=timing(ar['kv_to_p']),P1_full_unpack=timing(ar['full_sector_unpack']),
                tiles=[dict(sectors=t['sectors'],**timing(t['timing'])) for t in ar['selected_tile_reads']])
            if p0.get('measured'):row.update(P0_pack=timing(p0['kv_to_p']),P0_full_unpack=timing(p0['full_decode']))
            h['historical_P1'].append(row)
    h['transfer_states']=r.get('transfer_states',[])
    return h


def summary(r):
    lines=['# GPU KV path component batch','',f"Status: **{r['status']}**. GPU measured: **{r['coverage']['gpu_measured']}**.",'']
    if r.get('error'):lines+=['Error: '+r['error'],'']
    lines+=['| Attention case | Tail | Dense wall ms | P1 wall ms | Max output difference |',
        '|---|---:|---:|---:|---:|']
    for x in r['attention']:
        lines.append(f"| {x['dataset']}/{x['name']} | {x['tail_tokens']} | {x['dense']['wall_median_ms']:.5f} | {x['packed']['wall_median_ms']:.5f} | {x['maximum_output_abs_difference']:.3g} |")
    lines+=['','| Transfer case | Arm | T complete bpv | Raw fallback | KV to T ms | T to KV ms | P to T ms | T to P ms | Exact |',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    for x in r['transfer']:
        lines.append(f"| {x['dataset']}/{x['name']} | {x['arm']} | {x['complete_T_bpv']:.5f} | {x['raw_fallback']} | {x['kv_to_T_gpu_body']['wall_median_ms']:.3f} | {x['T_to_KV_gpu']['wall_median_ms']:.3f} | {x['P_to_T_gpu_body']['wall_median_ms']:.3f} | {x['T_to_P_gpu_composed']['wall_median_ms']:.3f} | {x['exact']} |")
    accounting_rows=[x for x in r['transfer'] if 'T_packet_accounting' in x]
    if accounting_rows:
        lines+=['','| Group | Complete bpv | Payload bpv | Directory + envelope bpv | Legacy full-identity bpv |',
            '|---|---:|---:|---:|---:|']
        for dataset,arm in sorted({(x['dataset'],x['arm']) for x in accounting_rows}):
            rows=[x for x in accounting_rows if (x['dataset'],x['arm'])==(dataset,arm)]
            words=sum(x['original_words'] for x in rows)
            def rate(field):return sum(x['T_packet_accounting'][field] for x in rows)*8/words
            old=sum(x['legacy_full_identity_accounting']['complete_bytes'] for x in rows)*8/words
            lines.append(f"| {dataset}/{arm} | {rate('complete_bytes'):.6f} | {rate('payload_bytes'):.6f} | {rate('directory_bytes')+rate('envelope_bytes'):.6f} | {old:.6f} |")
    diagnostic_rows=[x for x in r['transfer'] if 'bottleneck_diagnostics' in x]
    if diagnostic_rows:
        lines+=['','| Diagnostic encoder control | Parameters ms | Prepared entropy ms | Encode device events | Decode device events |',
            '|---|---:|---:|---:|---:|']
        for x in diagnostic_rows:
            d=x['bottleneck_diagnostics'];ep=d['encode_profile'];dp=d['decode_profile']
            lines.append(f"| {x['dataset']}/{x['arm']}/{x['name']} | {d['parameter_replay_gpu']['wall_median_ms']:.3f} | {d['entropy_encode_with_prepared_parameters_gpu']['wall_median_ms']:.3f} | {ep.get('CUDA_device_event_count','unavailable')} | {dp.get('CUDA_device_event_count','unavailable')} |")
        lines+=['','Prepared entropy uses parameters computed from known source KV; it is an encoder timing control. Actual causal decode remains packet-only. Profiled event totals are separate from the uninstrumented medians; missing CUDA profiler activity is explicit.']
    lines+=['','T1 is a new 12-bit binary-rANS stream with per-token restarts. Complete external bytes include the envelope, model/state identity digests, directory, checksum and payload. The independently expected full model identity is hashed and verified, and legacy full-identity envelopes remain readable. Shared state and measured scratch/peaks are separate. Raw fallback is explicitly identified.',
        '', 'T rates do not inherit the earlier CPU ~44% result. P rates and T rates are separate representations. GPU-body codec times exclude separately reported compressed-byte export/import and CPU envelope/checksum. The actual filesystem cycle is measured directly.',
        '', 'Attention uses seeded component queries, not model logits. The bridge temporarily materializes one bounded chunk; it is not a fused two-layer-scratch converter. Filesystem reads may hit the page cache. Native cache-writer/generation integration, concurrency, cold SSD and real network throughput remain unimplemented.']
    if r['failures']:
        lines+=['','## Failed components','']+[f"- {x['stage']}: {x['error']}" for x in r['failures']]
    return '\n'.join(lines)+'\n'


def parser():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--fixture',type=Path,default=ROOT/'benchmark_fixtures/team_transfer_smol_replay_v1')
    a.add_argument('--captures-from',type=Path);a.add_argument('--previous-p1',type=Path)
    a.add_argument('--device',default='cuda:0');a.add_argument('--arms',default='A,B')
    a.add_argument('--tile-tokens',type=int,default=16);a.add_argument('--page-tokens',type=int,default=32)
    a.add_argument('--page-starts',type=lambda x:[int(v) for v in x.split(',')],default=[0,512,1024,-1])
    a.add_argument('--gqa-multiple',type=int,default=4);a.add_argument('--arena-ratio',type=float,default=.8)
    a.add_argument('--warmup',type=int,default=1);a.add_argument('--repeats',type=int,default=3);a.add_argument('--attention-repeats',type=int,default=20)
    a.add_argument('--attention-limit',type=int,default=2);a.add_argument('--transfer-limit',type=int,default=2,help='0 means all source pages; otherwise evenly samples first through last')
    a.add_argument('--cpu-t0-control',action='store_true');a.add_argument('--skip-tests',action='store_true')
    a.add_argument('--validate-inputs-only',action='store_true',help='explicit CPU input/state validation; no codec/GPU measurement')
    a.add_argument('--skip-isolated',action='store_true',help='skip the fresh-process packet-only GPU decoder control')
    a.add_argument('--diagnostics',action='store_true',help='measure encoder substeps and collect CUDA operation summaries on selected pages')
    a.add_argument('--diagnostic-limit',type=int,default=2,help='diagnostic pages per dataset/arm, evenly sampled; 0 means all')
    a.add_argument('--decode-child',type=Path);a.add_argument('--identity-file',type=Path)
    a.add_argument('--base-state-file',type=Path);a.add_argument('--adapter-state-file',type=Path)
    return a


def main(argv=None):
    args=parser().parse_args(argv)
    if args.decode_child:return child(args)
    if args.output.exists():raise ValueError('fresh output directory required')
    if args.repeats<1 or args.attention_repeats<1 or args.warmup<0 or args.attention_limit<0 or args.transfer_limit<0 or args.diagnostic_limit<0 or args.gqa_multiple<1 or not np.isfinite(args.arena_ratio) or args.arena_ratio<=0:raise ValueError('invalid measurement bounds')
    if not args.arms or any(x not in ('A','B') for x in args.arms.split(',')):raise ValueError('this baseline batch supports A,B only')
    args.output.mkdir(parents=True)
    sources=['scripts/bench_gpu_path.py','scripts/gpu_path_diagnostics.py','scripts/bench_pool_p1.py','qwen_kv_bench/pool_p1.py','qwen_kv_bench/pool_p1_kernels.py','qwen_kv_bench/pool_attention.py','qwen_kv_bench/pool_attention_kernels.py','qwen_kv_bench/transfer_t1.py','qwen_kv_bench/transfer_t1_kernels.py','qwen_kv_bench/gpu_kv_bridge.py','qwen_kv_bench/transfer_reference.py','qwen_kv_bench/transfer_probability.py','qwen_kv_bench/transfer_gpu_predictor.py']
    r=dict(format='gpu_kv_path_batch_v1',status='running',runtime=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        sources=[dict(path=f,sha256=sha(ROOT/f)) for f in sources],datasets=[],attention=[],transfer=[],failures=[],
        coverage=dict(gpu_measured=False,attention_component=False,gpu_entropy=False,external_T_P_bridge=False,
            native_generation=False,serving_concurrency=False,cold_SSD=False,real_network=False))
    def checkpoint():
        save(args.output/'results.json',r)
        save(args.output/'highlights.json',highlights(r))
        (args.output/'summary.md').write_text(summary(r))
    def failure(stage,e):r['failures'].append(dict(stage=stage,error=str(e),traceback=traceback.format_exc()));log(stage+' FAILED: '+str(e));checkpoint()
    checkpoint()
    try:
        if not args.validate_inputs_only:
            if os.environ.get('TRITON_INTERPRET')=='1':raise RuntimeError('CUDA benchmark refuses the CPU interpreter')
            torch,device=p._cuda_torch(args.device);import triton
            torch.cuda.set_device(device);torch.set_num_threads(1)
            r['device']=dict(name=torch.cuda.get_device_name(device),torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,capability=list(torch.cuda.get_device_capability(device)))
            if not args.skip_tests:
                log('running actual CUDA component tests')
                cmd=[sys.executable,'-m','unittest','tests.test_pool_p1.CudaTests','tests.test_pool_attention.CudaTests','tests.test_transfer_t1.CudaTests','tests.test_gpu_kv_bridge.CudaTests','-v']
                checked=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (args.output/'cuda_tests.log').write_text(checked.stdout);print(checked.stdout,flush=True)
                if checked.returncode or 'skipped' in checked.stdout or not re.search(r'Ran [1-9][0-9]* tests',checked.stdout):raise RuntimeError('CUDA component tests failed or skipped')
                r['cuda_tests']=dict(passed=True,count=int(re.search(r'Ran ([0-9]+) tests',checked.stdout).group(1)))
        log('verifying source pages and frozen states')
        begin=time.perf_counter();data=datasets(args);r['offline_input_and_state_load_seconds']=time.perf_counter()-begin
        r['datasets']=[dict(name=d['name'],identity=d['identity'],scope=d['scope'],pages=len(d['cases']),arms=list(d['states']),base_sha256=d['base'].state_sha256,adapter_sha256={a:s.state_sha256 for a,s in d['states'].items()}) for d in data]
        if args.previous_p1:r['previous_P1']=extract_previous(args.previous_p1)
        save(args.output/'source_manifest.json',dict(sources=r['sources'],datasets=r['datasets']))
        if args.validate_inputs_only:r['status']='inputs_validated_cpu_only';checkpoint();return 0
        for d in data:
            for case in limited(d['cases'],args.attention_limit):
                for tail in [0,3]:
                    stage=f"attention/{d['name']}/{case.name}/tail{tail}";log(stage)
                    try:r['attention'].append(dict(dataset=d['name'],**attention_case(case,args,torch,device,tail)))
                    except Exception as e:failure(stage,e)
                    checkpoint()
        r['coverage']['attention_component']=bool(r['attention'])
        for d in data:
            for arm,state in d['states'].items():
                log(f"loading {d['name']} T1 arm {arm} state")
                begin=time.perf_counter();engine=t1.Engine(d['base'],state,device);torch.cuda.synchronize(device)
                r.setdefault('transfer_states',[]).append(dict(dataset=d['name'],arm=arm,cold_initialize_seconds=time.perf_counter()-begin,**engine.statistics()))
                transfer_cases=limited(d['cases'],args.transfer_limit)
                diagnostic_ids={id(c) for c in limited(transfer_cases,args.diagnostic_limit)} if args.diagnostics else set()
                for case in transfer_cases:
                    tag=re.sub(r'[^A-Za-z0-9_.-]+','_',d['name']+'_'+arm+'_'+case.name);stage='transfer/'+tag;log(stage)
                    try:
                        identity=case.provenance.get('model_identity',d['identity'])
                        spec=dict(kind='fixture',path=args.fixture) if d['name']=='frozen_smol' else dict(kind='saved',base=args.captures_from/'calibration/base.npz',adapter=args.captures_from/'calibration'/(arm+'_decoder.npz'))
                        r['transfer'].append(dict(dataset=d['name'],arm=arm,**transfer_case(case,engine,args,torch,device,args.output,tag,identity,spec,diagnose=id(case) in diagnostic_ids)))
                    except Exception as e:failure(stage,e)
                    checkpoint()
                del engine
        r['coverage'].update(gpu_measured=bool(r['attention'] or r['transfer']),gpu_entropy=bool(r['transfer']),external_T_P_bridge=bool(r['transfer']),
            model_coded_chunks=sum(not x['raw_fallback'] for x in r['transfer']),model_raw_fallback_chunks=sum(x['raw_fallback'] for x in r['transfer']))
        r['status']='failed' if r['failures'] else 'completed'
    except Exception as e:r['status']='failed';r['error']=str(e);r['traceback']=traceback.format_exc()
    checkpoint();log(r['status']+': '+str(args.output/'summary.md'))
    return 0 if r['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
