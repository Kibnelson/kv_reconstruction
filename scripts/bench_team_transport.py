#!/usr/bin/env python3
"""Profile and reduce host transport overhead on frozen exact one-step packets.

No fitting, raw KV capture or encoding occurs. Reference and optimized byte I/O
must reproduce the existing wire exactly and recover the original BF16 words.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','16')
os.environ.setdefault('OPENBLAS_NUM_THREADS','16')
import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import pstats
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_team_packet import canonical, identity, cuda_check, own_codec, check_original
from scripts.bench_lite_baselines import digest, launch, write
from scripts.bench_llama_causal import log, timed
from scripts.bench_lite_allocation import runtime_snapshot, pcie_metadata

ARMS=('reference','optimized')
METHODS=('team_dense_uniform','team_rank256_uniform')
LENGTHS=(512,8192)
LINKS=(1.,100.,200.,300.,400.)
METRICS=('export','import','receive','roundtrip')
FILES=('scripts/bench_team_transport.py','scripts/package_team_transport.py',
       'scripts/run_team_transport_a4500.sh','tests/test_team_transport_runner.py',
       'kv_compare/team_transport.py','tests/test_team_transport.py')


def arm_order(index):
    if type(index) is not int or index<0: raise ValueError('nonnegative order index required')
    shift=index%len(ARMS);order=ARMS[shift:]+ARMS[:shift]
    return order if (index//len(ARMS))%2==0 else order[::-1]


def implementations(arm):
    from kv_compare.team_transport import reference_export,fast_export,reference_import,fast_import
    if arm=='reference':return reference_export,reference_import
    if arm=='optimized':return fast_export,fast_import
    raise ValueError('unknown transport arm')


def checked_wire(blob,entry):
    if type(blob) is not bytes or len(blob)!=entry['packet_bytes'] or hashlib.sha256(blob).hexdigest()!=entry['packet_sha256']:
        raise AssertionError('transport changed original immutable wire bytes')
    return True


def valid_receiver_request(q):
    if set(q)!={'codec','entries','output','arms'} or tuple(q['arms'])!=ARMS:
        raise ValueError('receiver requires only fixed state, packet assertions and predeclared arms')
    if any({'inputs','captures','source_run','predictions','decoded','path'}&set(e) for e in q['entries']):
        raise ValueError('receiver contains source KV or prediction data')
    return True


def samples_record():
    return dict(samples=[],warmup_samples=[])


def finish_samples(record):
    result=dict(record)
    if not result['samples']:raise ValueError('actual samples required')
    result['wall_median_ms']=statistics.median(s['wall_ms'] for s in result['samples'])
    return result


def aggregate(rows):
    grouped={}
    for r in rows:grouped.setdefault((r['tokens'],r['method'],r['arm']),[]).append(r)
    result=[]
    for (n,method,arm),group in sorted(grouped.items()):
        value=dict(tokens=n,method=method,arm=arm,documents=[r['document'] for r in group],
            packet_bytes=statistics.mean(r['packet_bytes'] for r in group),complete_bpv=statistics.mean(r['complete_bpv'] for r in group),
            gpu_resident_decode_ms=statistics.mean(r['gpu_resident_decode']['wall_median_ms'] for r in group),
            original_exact=all(r['original_exact'] for r in group))
        for metric in METRICS:value[metric+'_ms']=statistics.mean(r[metric]['wall_median_ms'] for r in group)
        result.append(value)
    return result


def link_models(aggregates,links=LINKS):
    if any(not math.isfinite(g) or g<=0 for g in links):raise ValueError('positive finite links required')
    result=[]
    for a in aggregates:
        for g in links:
            wire=8*a['packet_bytes']/(g*1e6)
            result.append(dict(tokens=a['tokens'],method=a['method'],arm=a['arm'],effective_gbps=g,
                wire_ms=wire,host_blob_fetch_ms=wire+a['receive_ms'],
                gpu_compressed_egress_ms=wire+a['roundtrip_ms'],
                component_egress_diagnostic_ms=wire+a['export_ms']+a['receive_ms'],
                gpu_direct_context_ms=wire+a['gpu_resident_decode_ms'],measured_network=False))
    return dict(measured_network=False,links=list(links),rows=result,
        scope='host blob fetch = wire + measured import-and-decode; GPU compressed egress = wire + measured export-import-decode sequence; no raw KV encoding, disk, NIC or measured arrival pipeline')


def read_parent(parent_dir,methods=METHODS,lengths=LENGTHS):
    """Check packet/state/receiver receipts without reading any capture or KV array."""
    parent=json.loads((parent_dir/'results.json').read_text())
    if parent.get('format')!='team_packet_v1' or parent['status']!='completed' or not parent.get('gpu_measured'):
        raise ValueError('completed actual-GPU one-step packet parent required')
    recipe=parent['recipe']
    if identity(recipe)!=parent['identity']:raise ValueError('parent recipe identity changed')
    inherited={**recipe['unchanged_code'],**recipe['code']}
    for path,wanted in inherited.items():
        if digest(ROOT/path)!=wanted:raise ValueError('frozen measured source changed: '+path)
    jobs=[];receipts={}
    for method in methods:
        for n in lengths:
            name=f'{method}_t{n}';job=parent['jobs'][name]
            for path,wanted in job['artifacts'].items():
                p=(parent_dir/path).resolve()
                if not p.is_relative_to(parent_dir) or digest(p)!=wanted:raise ValueError('parent worker receipt changed')
                receipts[path]=wanted
            q=json.loads((parent_dir/name/'request.json').read_text())
            data=json.loads((parent_dir/name/'worker_results.json').read_text())
            proof=json.loads((parent_dir/name/'receiver.json').read_text())
            checks={c['id']:c for c in proof['checks']}
            if (job['rows']!=data['rows'] or data['codec_provenance']!=q['codec'] or
                    not data.get('gpu_measured') or not proof.get('gpu_measured') or
                    not proof.get('no_source_kv_or_predictions') or digest(parent_dir/name/'receiver.json')!=data['receiver_sha256']):
                raise ValueError('parent independent receiver proof differs')
            entries=[]
            for r in job['rows']:
                c=checks[r['id']]
                if (r['method']!=method or r['shape']!=[28,n,2048] or r['status']!='verified' or
                        not r['original_exact'] or r['original_word_sha256']!=r['decoded_word_sha256'] or
                        c['word_sha256']!=r['original_word_sha256'] or c['packet_sha256']!=r['packet_sha256'] or not c['verified']):
                    raise ValueError('parent original-word coverage differs')
                packet=Path(r['packet']).resolve()
                if (not packet.is_relative_to(parent_dir) or packet.is_symlink() or digest(packet)!=r['packet_sha256'] or
                        packet.stat().st_size!=r['packet_bytes']):raise ValueError('parent serialized packet changed')
                entries.append({k:r[k] for k in ('id','document','shape','values','packet','packet_sha256','packet_bytes',
                                                 'original_word_sha256','complete_bpv','state_sha256')})
            if len(entries)!=2 or {e['document'] for e in entries}!={f'test_text_t{n}',f'test_code_t{n}'}:
                raise ValueError('both original TEST documents required')
            if digest(Path(q['codec']['state'])/'manifest.json')!=q['codec']['manifest_sha256']:
                raise ValueError('parent shared fixed state changed')
            jobs.append((name,dict(method=method,tokens=n,codec=q['codec'],entries=entries)))
    from kv_compare.team_packet import load_state
    _,_,manifest=load_state(jobs[0][1]['codec']['state'],'cpu')
    if manifest['state_sha256']!=jobs[0][1]['codec']['state_sha256']:raise ValueError('parent fixed tensors changed')
    return parent,inherited,jobs,receipts


def receive(q,torch,device):
    valid_receiver_request(q)
    before=runtime_snapshot(torch);codec=own_codec(q['codec'],device)
    blobs=[Path(e['packet']).read_bytes() for e in q['entries']]
    for b,e in zip(blobs,q['entries']):checked_wire(b,e)
    checks={};held={}
    for i in (0,1,0):
        entry=q['entries'][i]
        for arm in q['arms']:
            exporter,importer=implementations(arm)
            packet,_=importer(blobs[i],device)
            value=codec.graph_decode(packet)
            original=check_original(value,entry)
            exported,_=exporter(packet);checked_wire(exported,entry)
            key=arm+'_'+entry['id']
            checks[key]=dict(id=key,arm=arm,parent_id=entry['id'],verified=True,
                packet_sha256=entry['packet_sha256'],word_sha256=original,shape=entry['shape'])
            if i==0 and arm not in held:held[arm]=value
            elif i==1:check_original(held[arm],q['entries'][0])
            del packet,exported,value
    after=runtime_snapshot(torch)
    if before!=after:raise AssertionError('receiver runtime changed')
    if codec.graph_capture_count!=1:raise AssertionError('alternating compact packets caused graph recapture')
    write(q['output'],dict(gpu_measured=True,no_source_kv_or_predictions=True,
        alternating_packets=True,independent_returned_outputs=True,
        graph_capture_count=codec.graph_capture_count,
        inputs='parent immutable packets and fixed shared state only; expected digests are assertions',
        runtime_policy=before,runtime_policy_after=after,checks=list(checks.values())))


def worker(q,torch,device):
    out=Path(q['output']);out.mkdir(parents=True,exist_ok=True)
    before=runtime_snapshot(torch);begin=time.perf_counter();codec=own_codec(q['codec'],device)
    torch.cuda.synchronize(device);cold=(time.perf_counter()-begin)*1000
    blobs=[];reads=[];packets=[]
    reference_export,reference_import=implementations('reference')
    for e in q['entries']:
        begin=time.perf_counter();blob=Path(e['packet']).read_bytes();reads.append((time.perf_counter()-begin)*1000)
        checked_wire(blob,e);blobs.append(blob)
        packet,_=reference_import(blob,device);packets.append(packet)
    begin=time.perf_counter();codec.graph_prepare(packets[0]);torch.cuda.synchronize(device)
    setup_ms=(time.perf_counter()-begin)*1000
    data={(arm,i):{m:samples_record() for m in METRICS} for arm in ARMS for i in range(2)}
    resident=[samples_record() for _ in range(2)];order=[];held=None
    def import_decode(importer,blob):
        packet,_=importer(blob,device)
        return codec.graph_decode(packet)
    def roundtrip(exporter,importer,packet):
        blob,_=exporter(packet);imported,_=importer(blob,device)
        return codec.graph_decode(imported),blob
    for rep in range(q['warmup']+q['repeats']):
        docs=(0,1) if rep%2==0 else (1,0)
        index=rep if rep<q['warmup'] else rep-q['warmup']
        arms=arm_order(index);order.append(dict(documents=[q['entries'][i]['document'] for i in docs],arms=list(arms)))
        key='warmup_samples' if rep<q['warmup'] else 'samples'
        for i in docs:
            e=q['entries'][i]
            for arm in arms:
                exporter,importer=implementations(arm)
                exported,sample=timed(lambda:exporter(packets[i])[0],torch,device,0,1)
                checked_wire(exported,e);data[arm,i]['export'][key].append(sample['samples'][0]);del exported
                imported,sample=timed(lambda:importer(blobs[i],device)[0],torch,device,0,1)
                check_original(codec.graph_decode(imported),e)
                data[arm,i]['import'][key].append(sample['samples'][0]);del imported
                value,sample=timed(lambda:import_decode(importer,blobs[i]),torch,device,0,1)
                check_original(value,e);data[arm,i]['receive'][key].append(sample['samples'][0])
                if rep==0 and i==0 and held is None:held=value
                elif rep==0 and i==1 and held is not None:check_original(held,q['entries'][0]);held=None
                del value
                returned,sample=timed(lambda:roundtrip(exporter,importer,packets[i]),torch,device,0,1)
                value,exported=returned;check_original(value,e);checked_wire(exported,e)
                data[arm,i]['roundtrip'][key].append(sample['samples'][0]);del value,exported,returned
            value,sample=timed(lambda:codec.graph_decode(packets[i]),torch,device,0,1)
            check_original(value,e);resident[i][key].append(sample['samples'][0]);del value
    rows=[]
    for arm in ARMS:
        for i,e in enumerate(q['entries']):
            row=dict(e,id=arm+'_'+e['id'],parent_id=e['id'],method=q['method'],arm=arm,tokens=q['tokens'],
                original_exact=True,decoded_word_sha256=e['original_word_sha256'],status='verified',
                gpu_resident_decode=finish_samples(resident[i]))
            row.update({m:finish_samples(data[arm,i][m]) for m in METRICS});rows.append(row)
            log(f"{row['id']}: export {row['export']['wall_median_ms']:.3f}ms; import {row['import']['wall_median_ms']:.3f}ms; receive {row['receive']['wall_median_ms']:.3f}ms")
    after=runtime_snapshot(torch)
    if before!=after:raise AssertionError('transport worker runtime changed')
    req=dict(codec=q['codec'],entries=q['entries'],arms=list(ARMS),output=str(out/'receiver.json'))
    write(out/'receiver_request.json',req)
    graph_count=codec.graph_capture_count;accounting=codec.accounting(packets[0])
    if graph_count!=1:raise AssertionError('paired transport calls unexpectedly recaptured the decoder graph')
    del codec,packets,blobs,packet;gc.collect();torch.cuda.empty_cache()
    launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--receive',out/'receiver_request.json'],out/'receiver.log')
    proof=json.loads((out/'receiver.json').read_text());checks={c['id']:c for c in proof['checks']}
    if (not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
            set(checks)!={r['id'] for r in rows} or proof['runtime_policy']!=before or
            not proof.get('alternating_packets') or not proof.get('independent_returned_outputs')):
        raise ValueError('fresh receiver coverage/runtime differs')
    for r in rows:
        c=checks[r['id']]
        if not c['verified'] or c['word_sha256']!=r['original_word_sha256'] or c['packet_sha256']!=r['packet_sha256']:
            raise AssertionError('fresh receiver exactness differs')
    write(out/'worker_results.json',dict(status='completed',gpu_measured=True,rows=rows,
        codec_provenance=q['codec'],device=torch.cuda.get_device_name(device),
        software={n:importlib.metadata.version(n) for n in ('torch','triton','numpy')},
        runtime_policy=before,runtime_policy_after=after,receiver_sha256=digest(out/'receiver.json'),
        cold_state_initialization_wall_ms=cold,graph_setup_wall_ms=setup_ms,graph_capture_count=graph_count,
        file_read_wall_ms=reads,accounting=accounting,sample_order=order,
        scope='actual public byte APIs, no phase timers; full roundtrip executes export then import then unchanged graph decoder with an owned output; checksum verification remains enabled'))


def profile_worker(q,torch,device):
    """A separate, explicitly instrumented process; never supplies benchmark medians."""
    out=Path(q['output']);out.mkdir(parents=True,exist_ok=True)
    entry=q['entries'][0];blob=Path(entry['packet']).read_bytes();checked_wire(blob,entry)
    codec=own_codec(q['codec'],device);_,ref_import=implementations('reference')
    packet,_=ref_import(blob,device);codec.graph_prepare(packet)
    diagnostics={};assertions=[]
    from torch.profiler import ProfilerActivity,profile,record_function
    cpu=cProfile.Profile()
    trace=out/'trace.json'
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,
                 profile_memory=True,with_stack=False) as prof:
        cpu.enable()
        for arm in ARMS:
            export_fn,import_fn=implementations(arm)
            with record_function('team_transport/'+arm+'/export'):
                exported,egress=export_fn(packet,profile=True,cuda_events=True)
            with record_function('team_transport/'+arm+'/import'):
                imported,ingress=import_fn(exported,device,profile=True,cuda_events=True)
            with record_function('team_transport/'+arm+'/decode'):
                value=codec.graph_decode(imported)
            diagnostics[arm]=dict(export=egress,import_record=ingress)
            # Assertion-only SHA and output D2H are excluded from both
            # profilers, just as they are excluded from ordinary timings.
            assertions.append((exported,value))
            del exported,imported,value
        cpu.disable();torch.cuda.synchronize(device)
    for exported,value in assertions:
        checked_wire(exported,entry);check_original(value,entry)
    del assertions,exported,value
    prof.export_chrome_trace(str(trace))
    functions=[]
    for (filename,line,name),(primitive,total,self_seconds,cumulative,callers) in pstats.Stats(cpu).stats.items():
        functions.append(dict(file=filename,line=line,function=name,primitive_calls=primitive,total_calls=total,
            self_ms=1000*self_seconds,cumulative_ms=1000*cumulative))
    functions.sort(key=lambda r:r['self_ms'],reverse=True)
    operators=[]
    for e in prof.key_averages():
        operators.append(dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
            cpu_total_ms=e.cpu_time_total/1000,
            self_device_ms=getattr(e,'self_device_time_total',getattr(e,'self_cuda_time_total',0))/1000,
            device_total_ms=getattr(e,'device_time_total',getattr(e,'cuda_time_total',0))/1000,
            cpu_memory_bytes=e.cpu_memory_usage,device_memory_bytes=getattr(e,'device_memory_usage',0)))
    operators.sort(key=lambda r:r['self_cpu_ms']+r['self_device_ms'],reverse=True)
    write(out/'profile.json',dict(status='completed',gpu_measured=True,method=q['method'],tokens=q['tokens'],
        document=entry['document'],packet_sha256=entry['packet_sha256'],codec_provenance=q['codec'],
        diagnostics=diagnostics,cprofile_top_self=functions[:40],operators=operators[:60],
        trace_path=str(trace),trace_sha256=digest(trace),trace_bytes=trace.stat().st_size,
        runtime_policy=runtime_snapshot(torch),
        assertions_outside_profilers=True,
        memory_scope='Torch operator allocation fields exclude Python bytes, NumPy buffers and other non-Torch host allocations; these are not full process memory peaks',
        scope='separate phase/cProfile/Torch CPU+CUDA diagnostic; instruments synchronization and allocations; not included in unprofiled paired medians; full trace stays local'))


def report(r,out):
    lines=['# Frozen one-step packet transport overhead', '',
        f"Status: **{r['status']}**. GPU measured: **{r.get('gpu_measured',False)}**.", '',
        'Existing compact packets and fitted state are reused unchanged. Every exported byte string must match the parent SHA256 and every decoded BF16 word must match the parent original-word digest.',
        'All timings below are means of per-document synchronized wall medians. Reference and optimized arms use balanced rotation/reversal and alternate two documents. CUDA intervals around host code can include host gaps; public wall latency is the primary metric.', '',
        '| Tokens | Method | I/O arm | Export ms | Import ms | Import + decode ms | Export + import + decode ms | GPU-resident decode ms |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for a in r.get('aggregates',[]):
        lines.append(f"| {a['tokens']} | {a['method']} | {a['arm']} | {a['export_ms']:.3f} | {a['import_ms']:.3f} | {a['receive_ms']:.3f} | {a['roundtrip_ms']:.3f} | {a['gpu_resident_decode_ms']:.3f} |")
    lines+=['', 'Calculated serial link models. Host fetch starts with an immutable compressed blob in host DRAM. GPU egress starts with an already-compressed GPU packet and charges the measured full export/import/decode sequence plus wire time. This step does not encode raw KV or measure a network.', '',
        '| Tokens | Method | Gb/s | Host fetch reference ms | Host fetch optimized ms | GPU egress reference ms | GPU egress optimized ms |',
        '|---:|---|---:|---:|---:|---:|---:|']
    models=r.get('link_models',{}).get('rows',[])
    for n in r.get('recipe',{}).get('lengths',[]):
        for method in r.get('recipe',{}).get('methods',[]):
            for g in LINKS:
                found={x['arm']:x for x in models if x['tokens']==n and x['method']==method and x['effective_gbps']==g}
                if set(found)!=set(ARMS):continue
                cells=[found[a][k] for k in ('host_blob_fetch_ms','gpu_compressed_egress_ms') for a in ARMS]
                lines.append(f'| {n} | {method} | {g:g} | '+' | '.join(f'{v:.3f}' for v in cells)+' |')
    if r.get('profiles'):
        lines+=['', 'Separate profiled 8K text phase diagnostics (wall ms). These instrumented calls run in different processes from ordinary paired timing; their phase totals are not additive ordinary latency. An em dash means the phase is absent, rather than a measured zero. CUDA copy intervals and top cProfile/Torch functions remain in profile.json; full Chrome traces stay local.', '',
            '| Method | I/O arm | D2H | Dtype/layout copy | Payload tobytes | Frame assembly | Checksum | Checksum append |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        def phase_cell(record,name):
            phase=record.get('phases',{}).get(name)
            return '\u2014' if phase is None else f"{phase['wall_ms']:.3f}"
        for _name,p in sorted(r['profiles'].items()):
            for arm in ARMS:
                phase=p['diagnostics'][arm]['export']
                cells=[phase_cell(phase,k) for k in ('D2H','dtype_layout_copy','payload_tobytes','framing_assembly','checksum','checksum_append')]
                lines.append(f"| {p['method']} | {arm} | "+' | '.join(cells)+' |')
        lines+=['', '| Method | I/O arm | Checksum input copy | Checksum | Payload bytes slice | Owned CPU copy | H2D | Validation |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for _name,p in sorted(r['profiles'].items()):
            for arm in ARMS:
                phase=p['diagnostics'][arm]['import_record']
                cells=[phase_cell(phase,k) for k in ('checksum_input_copy','checksum','payload_bytes_slice','payload_cpu_copy','H2D','packet_validation')]
                lines.append(f"| {p['method']} | {arm} | "+' | '.join(cells)+' |')
    if r.get('baseline_anchors'):
        lines+=['', 'Earlier GPU-resident decode anchors, reused without retiming. Their export/import paths were not measured here, so these are not matched end-to-end transport competitors.', '',
            '| Tokens | Earlier method | Complete bpv | Earlier GPU-resident decode ms |', '|---:|---|---:|---:|']
        for a in r['baseline_anchors']:lines.append(f"| {a['tokens']} | {a['method']} | {a['complete_bpv']:.5f} | {a['decode_ms']:.3f} |")
    if r.get('error'):lines+=['','Failure: '+r['error']]
    lines+=['', 'Cold state/graph setup and file-read samples are separate. No pinned reusable workspace is introduced by these byte I/O arms. Native codec arithmetic, prediction, compressed attention, pool conversion and serving remain outside this change.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,default=ROOT/'runs/team_packet_a4500_20261028_122017_2834427')
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--no-profile',action='store_false',dest='profile',default=True)
    for name in ('worker','receive','profile-worker'):p.add_argument('--'+name,type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    request=a.worker or a.receive or a.profile_worker
    if request:
        torch,device=cuda_check(a.device);q=json.loads(request.read_text())
        (worker if a.worker else receive if a.receive else profile_worker)(q,torch,device);return 0
    if a.output is None:p.error('--output required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    result=dict(format='team_transport_v1',status='running',gpu_measured=False,jobs={},profiles={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[row for job in result['jobs'].values() if job.get('status')=='completed' for row in job['rows']]
        result['aggregates']=aggregate(result['rows']);result['link_models']=link_models(result['aggregates'])
        write(out/'results.json',result);report(result,out)
    try:
        torch,device=cuda_check(a.device)
        if a.warmup<1 or a.repeats<2 or a.repeats%2:raise ValueError('warmup>=1 and positive even repeats>=2 required')
        parent_dir=a.parent_run.resolve();parent,inherited,plan,receipts=read_parent(parent_dir)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']):raise ValueError('original software/GPU required')
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_artifacts=receipts,unchanged_code=inherited,code={n:digest(ROOT/n) for n in FILES},
            methods=list(METHODS),lengths=list(LENGTHS),arms=list(ARMS),warmup=a.warmup,repeats=a.repeats,
            profile=a.profile,software=versions,device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,
            selections={name:q for name,q in plan})
        wanted=identity(recipe)
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=wanted:raise ValueError('resume identity differs; preserve this output and choose another')
            result=old;writable=True
        result.update(identity=wanted,recipe=recipe,status='running',gpu_measured=True,
            hardware=dict(parent['hardware'],pcie_query=pcie_metadata()),
            baseline_anchors=[dict(x,retimed=False,scope='reused earlier GPU-resident decode; unmatched host I/O') for x in parent['aggregates'] if x['method'] in
                ('raw_bf16','lite_current_graph','splitzip_official_framed','exans_official_fixed_framed')])
        result.pop('error',None);result.pop('traceback',None)
        write(out/'source_manifest.json',recipe);save()
        result['selected_jobs']=[name for name,_ in plan];save()
        for name,base in plan:
            directory=out/name;directory.mkdir(exist_ok=True)
            q=dict(base,output=str(directory),warmup=a.warmup,repeats=a.repeats)
            old=result['jobs'].get(name,{})
            if old.get('status')=='completed':
                for path,wanted_sha in old['artifacts'].items():
                    if digest(out/path)!=wanted_sha:raise ValueError('completed job artifact changed')
                if json.loads((directory/'request.json').read_text())!=q:raise ValueError('worker request changed')
            else:
                write(directory/'request.json',q)
                launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--worker',directory/'request.json'],directory/'worker.log')
                data=json.loads((directory/'worker_results.json').read_text())
                if not data.get('gpu_measured') or data['status']!='completed' or data['codec_provenance']!=q['codec']:
                    raise ValueError('actual-GPU worker required')
                result['jobs'][name]=dict(status='completed',rows=data['rows'],artifacts={str((directory/f).relative_to(out)):digest(directory/f)
                    for f in ('request.json','worker_results.json','receiver.json','receiver_request.json')});save()
            if a.profile and base['tokens']==8192:
                directory=out/'profiles'/name;directory.mkdir(parents=True,exist_ok=True)
                if name in result['profiles']:
                    if digest(directory/'profile.json')!=result['profiles'][name]['summary_sha256']:raise ValueError('profile receipt changed')
                else:
                    q=dict(base,entries=[next(e for e in base['entries'] if e['document'].startswith('test_text'))],output=str(directory))
                    write(directory/'request.json',q)
                    launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--profile-worker',directory/'request.json'],directory/'profile.log')
                    data=json.loads((directory/'profile.json').read_text())
                    result['profiles'][name]=dict(data,summary_sha256=digest(directory/'profile.json'));save()
        result['status']='completed';save()
        from scripts.package_team_transport import verify
        verify(out,result);log('completed: '+str(out/'summary.md'));return 0
    except Exception as error:
        if writable:result.update(status='failed',error=str(error),traceback=traceback.format_exc());save()
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
