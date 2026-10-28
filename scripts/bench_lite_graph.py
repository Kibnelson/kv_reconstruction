#!/usr/bin/env python3
"""Paired packet-only eager/graph decode, with separate phase diagnostics."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','16')
os.environ.setdefault('OPENBLAS_NUM_THREADS','16')
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_llama_causal import device_check,log,timed
from scripts.bench_lite_baselines import digest,launch,write
from scripts.bench_llama_baselines import confined
from kv_compare.lite_scoreboard import validate_comparison
from kv_compare.lite_stream_layout import fetch_budgets

FILES=('scripts/bench_lite_graph.py','scripts/package_lite_graph.py',
       'scripts/run_lite_graph_a4500.sh','kv_compare/lite_graph.py','tests/test_lite_graph.py')
BASELINES=('raw_bf16','splitzip_official_framed','exans_official_fixed_framed')
LAYOUTS=(('lite_v128_b8','lite_v128_b8_before'),('lite_v64_b32','lite_v64_b32'))
ARMS=('eager','graph')


def aggregate_decode(rows):
    grouped={}
    for row in rows: grouped.setdefault(row['method'],[]).append(row)
    result=[]
    for method,group in grouped.items():
        mean=lambda key:statistics.mean(r[key] for r in group)
        result.append(dict(method=method,kappa=0.,mode='lossless',documents=[r['document'] for r in group],
            packet_bytes=mean('packet_bytes'),values=mean('values'),complete_bpv=mean('complete_bpv'),
            saved_percent=mean('saved_percent'),decode_ms=statistics.mean(r['decode']['wall_median_ms'] for r in group),
            original_exact=all(r['original_exact'] for r in group)))
    return result


def fetch_model(aggregates,links=(1.,2.5,5.,10.)):
    return dict(measured_network=False,scope='serial precompressed transfer+decode model; no fresh encode projection, transport, staging, disk, serialization, overlap or serving measurement',
        links=[dict(effective_gbps=g,estimates=[dict(method=r['method'],
            transfer_ms=8*r['packet_bytes']/(g*1e6),fetch_ms=r['decode_ms']+8*r['packet_bytes']/(g*1e6))
            for r in aggregates]) for g in links])


def load_packets(q,device):
    from kv_compare.causal_packet import Packet
    packets=[]; uploads=[]
    for e in q['entries']:
        if digest(e['packet'])!=e['packet_sha256']: raise ValueError('received packet changed')
        wire=Path(e['packet']).read_bytes(); begin=time.perf_counter()
        p=Packet.restore(wire,device)
        import torch
        torch.cuda.synchronize(device)
        uploads.append(dict(document=e['document'],serialized_upload_wall_ms=(time.perf_counter()-begin)*1000))
        if p.header['raw'] or p.header['kappa']!=0: raise ValueError('this graph experiment needs nonraw lossless packets')
        if tuple(p.header['shape'])!=tuple(e['shape']) or len(wire)!=e['packet_bytes']:
            raise ValueError('packet shape/byte scope changed')
        packets.append(p)
    if len(packets)!=2 or len({e['packet_sha256'] for e in q['entries']})!=2:
        raise ValueError('two different packets required')
    if len({p.payload.numel() for p in packets})!=2:
        raise ValueError('changed-payload-length replay check requires two payload lengths')
    headers=[p.header for p in packets]
    if headers[0]!=headers[1]: raise ValueError('paired replay packets must share graph-bound metadata')
    return packets,uploads


def make_decoder(q,device):
    from kv_compare.lite_fused import FusedEngine
    from kv_compare.lite_graph import GraphDecoder
    engine=FusedEngine(q['codec']['state'],device,**q['codec']['engine_options'])
    return engine,GraphDecoder(engine,warmup=3)


def check_output(output,entry):
    from kv_compare.causal_packet import word_sha
    actual=word_sha(output)
    if actual!=entry['decoded_word_sha256'] or actual!=entry['original_word_sha256']:
        raise AssertionError('decoded packet differs from original BF16 words')
    return actual


def receive(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA graph receiver required')
    packets,_=load_packets(q,device); engine,graph=make_decoder(q,device)
    graph.prepare(packets[0]); checks={}; held=None
    for i in (0,1,0):
        for arm in ARMS:
            value=(engine.decode if arm=='eager' else graph.decode)(packets[i])
            actual=check_output(value,q['entries'][i]); ident=q['method']+'_'+arm+'_'+q['entries'][i]['document']
            checks[ident]=dict(id=ident,verified=True,packet_sha256=q['entries'][i]['packet_sha256'],
                word_sha256=actual,shape=list(value.shape))
            if arm=='graph' and i==0 and held is None: held=value
            elif arm=='graph' and i==1: check_output(held,q['entries'][0])
    if graph.capture_count!=1: raise AssertionError('payload length change unexpectedly recaptured graph')
    write(q['output'],dict(gpu_measured=True,no_source_kv_or_predictions=True,checks=list(checks.values()),
        inputs='fixed fitted state + original serialized packets; expected hashes are assertions',
        alternating_packets=True,distinct_payload_lengths=True,independent_returned_outputs=True,
        graph_capture_count=graph.capture_count))


def worker(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA required; graph timing has no CPU substitute')
    out=Path(q['output']); out.mkdir(parents=True,exist_ok=True)
    packets,uploads=load_packets(q,device)
    begin=time.perf_counter(); engine,graph=make_decoder(q,device); torch.cuda.synchronize(device)
    init_ms=(time.perf_counter()-begin)*1000
    setup=graph.prepare(packets[0])
    measurements={(arm,i):dict(samples=[],warmup_samples=[]) for arm in ARMS for i in range(2)}
    order=[]; held=None
    for round_index in range(q['warmup']+q['repeats']):
        docs=(0,1) if round_index%2==0 else (1,0)
        arms=ARMS if round_index%2==0 else ARMS[::-1]
        order.append(dict(documents=[q['entries'][i]['document'] for i in docs],arms=list(arms)))
        for i in docs:
            for arm in arms:
                fn=lambda arm=arm,i=i:(engine.decode if arm=='eager' else graph.decode)(packets[i])
                value,sample=timed(fn,torch,device,0,1)
                # Hashing is outside every timed invocation, for both arms.
                check_output(value,q['entries'][i])
                measurements[arm,i]['warmup_samples' if round_index<q['warmup'] else 'samples'].append(sample['samples'][0])
                if arm=='graph' and i==0 and held is None: held=value
                elif arm=='graph' and i==1 and held is not None: check_output(held,q['entries'][0])
                del value
    if graph.capture_count!=1: raise AssertionError('alternating valid payload lengths caused recapture')
    rows=[]
    for arm in ARMS:
        for i,e in enumerate(q['entries']):
            dec=measurements[arm,i]; dec['wall_median_ms']=statistics.median(s['wall_ms'] for s in dec['samples'])
            method=q['method']+'_'+arm
            row={k:e[k] for k in ('document','packet','packet_sha256','original_word_sha256','decoded_word_sha256',
                'shape','values','packet_bytes','complete_bpv','saved_percent','state_sha256')}
            row.update(id=method+'_'+e['document'],method=method,kappa=0.,mode='lossless',original_exact=True,
                decode=dec,engine_options=q['codec']['engine_options'],shared_predictor_gpu_bytes=engine.shared_bytes)
            rows.append(row); log(f"{row['id']}: decode {dec['wall_median_ms']:.3f} ms, exact")
    graph_capture_count=graph.capture_count; setups=graph.setups
    del held,graph,engine; gc.collect(); torch.cuda.empty_cache()
    request=dict(q,output=str(out/'receiver.json'))
    # Request contains state, packet paths and assertions only; never source KV.
    write(out/'receiver_request.json',request)
    launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--receive',out/'receiver_request.json'],out/'receiver.log')
    proof=json.loads((out/'receiver.json').read_text()); checks={r['id']:r for r in proof['checks']}
    if (not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
            set(checks)!={r['id'] for r in rows} or len(checks)!=len(rows) or proof['graph_capture_count']!=1):
        raise ValueError('fresh graph receiver coverage differs')
    for row in rows:
        c=checks[row['id']]
        if not c['verified'] or c['word_sha256']!=row['decoded_word_sha256'] or c['packet_sha256']!=row['packet_sha256']:
            raise AssertionError('fresh graph receiver changed words')
        row['status']='verified'
    write(out/'worker_results.json',dict(status='completed',gpu_measured=True,rows=rows,
        device=torch.cuda.get_device_name(device),software={n:importlib.metadata.version(n) for n in ('torch','triton','numpy')},
        codec_provenance=q['codec'],cold_state_initialization_wall_ms=init_ms,graph_setup=setup,graph_setups=setups,
        graph_capture_count=graph_capture_count,sample_order=order,serialized_uploads=uploads,
        receiver_sha256=digest(out/'receiver.json')))


def diagnostic(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA phase diagnostics required')
    packets,_=load_packets(q,device); engine,graph=make_decoder(q,device); graph.prepare(packets[0])
    data=[]
    for packet,e in zip(packets,q['entries']):
        measured=graph.measure_phases(packet,repeats=q.get('phase_repeats',3))
        check_output(graph.decode(packet),e)
        data.append(dict(document=e['document'],packet_sha256=e['packet_sha256'],measurement=measured))
    if graph.capture_count!=1: raise AssertionError('diagnostic changed graph geometry')
    write(Path(q['output'])/'phases.json',dict(status='completed',gpu_measured=True,phases=data,
        codec_provenance=q['codec'],graph_setups=graph.setups,
        scope='separate synchronized diagnostic process; phase times are not ordinary latency and must not be added to ordinary timings'))


def report(result,out):
    lines=['# Paired Lite CUDA graph decoder experiment','',
        f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured',False)}**.",'',
        'Same original packet bytes and fixed fitted state. Eager and graph calls alternate AB/BA and alternate two packets with different payload lengths.',
        'Full graph decode includes received-directory validation, staging copies, replay and an independently owned output clone.','',
        '| Method | Saved | Complete bpv | Decode ms | Exact |','|---|---:|---:|---:|---|']
    for r in result.get('aggregates',[]):
        lines.append(f"| {r['method']} | {r['saved_percent']:.2f}% | {r['complete_bpv']:.5f} | {r['decode_ms']:.3f} | {r['original_exact']} |")
    setups=[(name,setup) for name,job in result.get('jobs',{}).items()
            for setup in job.get('graph_setups',[])]
    if setups:
        mib=lambda value:f"{value/(1<<20):.3f}"
        lines+=['','Cold setup and resident memory. Allocated/reserved columns include the resident engine and packets; before/after refer to graph preparation.','',
            '| Layout | Cold setup ms | Allocated before → after MiB | Reserved before → after MiB | Peak allocated / reserved MiB | Staging payload MiB | Directory KiB | Captured output MiB |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,s in setups:
            lines.append(f"| {name} | {s['cold_synchronized_wall_ms']:.3f} | "
                f"{mib(s['baseline_torch_bytes'])} → {mib(s['persistent_torch_bytes'])} | "
                f"{mib(s['baseline_torch_reserved_bytes'])} → {mib(s['persistent_torch_reserved_bytes'])} | "
                f"{mib(s['peak_torch_bytes'])} / {mib(s['peak_torch_reserved_bytes'])} | "
                f"{mib(s['staging_payload_bytes'])} | {s['staging_directory_bytes']/1024:.3f} | {mib(s['captured_output_bytes'])} |")
    lines+=['','Calculated serial precompressed fetch, not measured transport. No fresh encode projection is produced.','',
        '| Gb/s | Method | Transfer + decode ms |','|---:|---|---:|']
    for link in result.get('fetch_model',{}).get('links',[]):
        for e in link['estimates']:
            lines.append(f"| {link['effective_gbps']:g} | {e['method']} | {e['fetch_ms']:.3f} |")
    if result.get('fetch_budgets'):
        lines+=['','Decode budgets against retimed SplitZip, using each complete packet size.','',
            '| Candidate | Gb/s | Decode budget ms | Current decode ms |','|---|---:|---:|---:|']
        for e in result['fetch_budgets']:
            lines.append(f"| {e['method']} | {e['effective_gbps']:g} | {e['decode_budget_ms']:.3f} | {e['current_decode_ms']:.3f} |")
    if result.get('diagnostics'):
        lines+=['','Separate synchronized phase diagnostics; do not add these to ordinary paired timings. Raw wall/CUDA samples remain in JSON.','',
            '| Layout / document | Validation wall ms | Input-copy CUDA ms | Replay CUDA ms | Replay CPU submit ms | Output-clone CUDA ms |',
            '|---|---:|---:|---:|---:|---:|']
        for name,job in result['diagnostics'].items():
            for e in job['phases']:
                p=e['measurement']['phases']
                lines.append(f"| {name} / {e['document']} | {p['validation']['wall_median_ms']:.3f} | "
                    f"{p['input_copy']['cuda_median_ms']:.3f} | {p['replay']['cuda_median_ms']:.3f} | "
                    f"{p['replay']['submit_cpu_median_ms']:.3f} | {p['output_clone']['cuda_median_ms']:.3f} |")
    if result.get('error'): lines+=['','Failure: '+result['error']]
    lines+=['','Cold graph warmup/capture, persistent staging, graph-private Torch allocations, sample distributions and allocation peaks are in worker_results.json. Driver graph metadata is not fully covered by Torch accounting.',
        'Both arms share a resident fitted engine and graph buffers during paired timing; allocation baselines include that setup. Uploading serialized packets is measured separately.',
        'Fresh GPU receivers get only fixed state and packet bytes, and verify alternating packets and output ownership. No source KV or reference predictions are used.',
        'Original official baseline workers/environments are unchanged. No fitting, model download, native rebuild, actual network, SSD, pool attention or serving-concurrency measurement is performed.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def native_receipt(parent):
    current=parent
    while 'nvcomp' not in current['recipe']:
        recipe=current['recipe']; path=Path(recipe['parent_run'])/'results.json'
        if digest(path)!=recipe['parent_results_sha256']: raise ValueError('ancestor evidence changed')
        current=json.loads(path.read_text())
    return current['recipe']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,default=ROOT/'runs/lite_streams_a4500_20261028_085102_2556937')
    p.add_argument('--output',type=Path); p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=2); p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--phase-repeats',type=int,default=3)
    for name in ('worker','receive','diagnostic'): p.add_argument('--'+name,type=Path,help=argparse.SUPPRESS)
    p.add_argument('--software-check',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    request=args.worker or args.receive or args.diagnostic
    if request:
        torch,device=device_check(args.device,args.software_check); q=json.loads(request.read_text())
        (worker if args.worker else receive if args.receive else diagnostic)(q,torch,device); return 0
    if args.output is None: p.error('--output required')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=dict(format='lite_graph_v1',status='running',gpu_measured=False,jobs={},diagnostics={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        result['aggregates']=aggregate_decode(result['rows']); result['fetch_model']=fetch_model(result['aggregates'])
        result['fetch_budgets']=fetch_budgets(result['aggregates']); write(out/'results.json',result); report(result,out)
    try:
        torch,device=device_check(args.device)
        if args.warmup<1 or args.repeats<2 or args.phase_repeats<2: raise ValueError('warmup>=1, repeats/phase-repeats>=2 required')
        parent_dir=args.parent_run.resolve(); parent=json.loads((parent_dir/'results.json').read_text())
        from scripts.package_lite_streams import verify as verify_parent
        if parent['status']!='completed' or not parent.get('gpu_measured'): raise ValueError('completed GPU parent required')
        verify_parent(parent_dir,parent)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']): raise ValueError('original GPU/software required')
        inherited={**parent['recipe']['unchanged_code'],**parent['recipe']['code']}
        for name,wanted in inherited.items():
            if digest(ROOT/name)!=wanted: raise ValueError('measured source changed: '+name)
        receipt=native_receipt(parent)
        names=(*BASELINES,*(old for _,old in LAYOUTS))
        requests={name:json.loads((parent_dir/name/'request.json').read_text()) for name in names}
        for name,q in requests.items():
            data=json.loads((parent_dir/name/'worker_results.json').read_text())
            if q['codec']!=data['codec_provenance']: raise ValueError('parent codec request changed')
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_requests_sha256={n:digest(parent_dir/n/'request.json') for n in names},
            code={n:digest(ROOT/n) for n in FILES},unchanged_code=inherited,
            warmup=args.warmup,repeats=args.repeats,phase_repeats=args.phase_repeats,
            software=versions,device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,
            layouts=[list(x) for x in LAYOUTS])
        identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=identity: raise ValueError('resume identity differs; choose fresh output')
            result=old; writable=True
        result.update(identity=identity,recipe=recipe,status='running',gpu_measured=True,model=parent['model'],hardware=parent['hardware'])
        result.pop('error',None); result.pop('traceback',None); write(out/'source_manifest.json',recipe)
        plan=[]
        for name in BASELINES:
            q=dict(requests[name]); q.update(output=str(out/name),warmup=args.warmup,repeats=args.repeats)
            python=sys.executable; env=None
            if name=='splitzip_official_framed': python=str(ROOT/'runs/_llama_official/envs/splitzip/bin/python')
            if name=='exans_official_fixed_framed':
                from kv_compare.profile_recovery import runtime_environment
                spec=importlib.util.find_spec('nvidia.libnvcomp')
                if spec is None or importlib.metadata.version('nvidia-libnvcomp-cu12')!=receipt['nvcomp']:
                    raise ValueError('pinned nvCOMP runtime required')
                path=Path(q['codec']['library']).parent.parent/'toolchain.json'
                if digest(path)!=receipt['toolchain_sha256']: raise ValueError('native toolchain receipt changed')
                toolchain=json.loads(path.read_text())
                env=runtime_environment(os.environ,toolchain['cuda_home'],[Path(s)/'lib' for s in spec.submodule_search_locations])
            plan.append((name,q,python,ROOT/'scripts/bench_lite_baselines.py',env))
        for name,oldname in LAYOUTS:
            spec=requests[oldname]['codec']; entries=[]
            for r in parent['jobs'][oldname]['rows']:
                entries.append({k:r[k] for k in ('document','packet','packet_sha256','original_word_sha256',
                    'decoded_word_sha256','shape','values','packet_bytes','complete_bpv','saved_percent','state_sha256')})
            q=dict(method=name,codec=spec,entries=entries,output=str(out/name),warmup=args.warmup,
                repeats=args.repeats,phase_repeats=args.phase_repeats)
            plan.append((name,q,sys.executable,Path(__file__).resolve(),None))
        result['selected_jobs']=[x[0] for x in plan]; save()
        for name,q,python,script,env in plan:
            directory=out/name; directory.mkdir(exist_ok=True); old=result['jobs'].get(name,{})
            if old.get('status')=='completed':
                for path,wanted in old['artifacts'].items():
                    if digest(confined(out,path))!=wanted: raise ValueError('completed evidence changed')
                for row in old['rows']:
                    if digest(row['packet'])!=row['packet_sha256']: raise ValueError('completed packet changed')
                log('reuse completed job: '+name); continue
            write(directory/'request.json',q)
            launch([python,script,'--device',str(device),'--worker',directory/'request.json'],directory/'worker.log',env)
            data=json.loads((directory/'worker_results.json').read_text())
            if data['status']!='completed' or not data['gpu_measured'] or data['device']!=recipe['device']:
                raise ValueError('worker did not complete on original CUDA device')
            if script==Path(__file__).resolve() and data['software']!={n:versions[n] for n in ('torch','triton','numpy')}:
                raise ValueError('worker software differs')
            result['jobs'][name]=dict(status='completed',rows=data['rows'],
                artifacts={str((directory/f).relative_to(out)):digest(directory/f) for f in ('worker_results.json','receiver.json')})
            if name.startswith('lite_'):
                result['jobs'][name]['graph_setups']=data['graph_setups']
            save()
        validate_comparison(result['rows'])
        for name,_ in LAYOUTS:
            directory=out/'diagnostics'/name; directory.mkdir(parents=True,exist_ok=True)
            if name in result['diagnostics']:
                if digest(directory/'phases.json')!=result['diagnostics'][name]['summary_sha256']:
                    raise ValueError('phase diagnostics changed')
                continue
            q=dict(next(q for n,q,_,_,_ in plan if n==name),output=str(directory))
            write(directory/'request.json',q)
            launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--diagnostic',directory/'request.json'],directory/'worker.log')
            data=json.loads((directory/'phases.json').read_text())
            if data['status']!='completed' or not data['gpu_measured']: raise ValueError('GPU phase diagnostics incomplete')
            data['summary_sha256']=digest(directory/'phases.json'); result['diagnostics'][name]=data; save()
        result['status']='completed'; save(); log('completed: '+str(out/'summary.md')); return 0
    except Exception as error:
        if writable:
            result.update(status='failed',error=str(error),traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
