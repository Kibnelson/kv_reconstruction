#!/usr/bin/env python3
"""Complete lossless Lite stream tuning with unchanged predictors and kernels."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_llama_causal import device_check, log, timed
from scripts.bench_lite_baselines import digest, launch, write
from scripts.bench_llama_baselines import confined, verified_inputs
from kv_compare.lite_scoreboard import aggregate, network_model, validate_comparison
from kv_compare.lite_stream_layout import geometry, fetch_budgets

FILES=('scripts/bench_lite_streams.py','scripts/package_lite_streams.py',
       'scripts/run_lite_streams_a4500.sh','kv_compare/lite_stream_layout.py','tests/test_lite_streams.py')
CONFIGURATIONS=(('lite_v128_b8_before',128,8),('lite_v64_b8',64,8),
                ('lite_v32_b8',32,8),('lite_v64_b32',64,32),
                ('lite_v32_b32',32,32),('lite_v128_b8_after',128,8))
BASELINES=('raw_bf16','splitzip_official_framed','exans_official_fixed_framed')


def codec(spec,device):
    from kv_compare.lite_fused import FusedEngine
    return FusedEngine(spec['state'],device,**spec['engine_options'])


def receive(q,torch,device):
    from kv_compare.causal_packet import Packet,word_sha
    engine=codec(q['codec'],device); checks=[]
    for e in q['entries']:
        if digest(e['packet'])!=e['packet_sha256']: raise ValueError('receiver packet changed')
        output=engine.decode(Packet.restore(Path(e['packet']).read_bytes(),device))
        actual=word_sha(output)
        if actual!=e['decoded_word_sha256']: raise AssertionError('fresh lossless receiver changed words')
        checks.append(dict(id=e['id'],verified=True,packet_sha256=e['packet_sha256'],word_sha256=actual,shape=list(output.shape)))
    write(q['output'],dict(gpu_measured=device.type=='cuda',no_source_kv_or_predictions=True,
        inputs='fixed fitted state + newly encoded packet bytes; expected digests are assertions',checks=checks))


def worker(q,torch,device):
    import numpy as np
    from kv_compare.causal_packet import Packet,word_sha
    out=Path(q['output']); out.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter(); engine=codec(q['codec'],device)
    if device.type=='cuda': torch.cuda.synchronize(device)
    cold=(time.perf_counter()-begin)*1000; rows=[]; checks=[]
    for e in q['entries']:
        if digest(e['input'])!=e['input_file_sha256'] or digest(e['reference_packet'])!=e['reference_packet_sha256']:
            raise ValueError('original input/reference packet changed')
        words=torch.from_numpy(np.load(e['input'],allow_pickle=False).view(np.int16).copy()).to(device)
        if word_sha(words)!=e['original_word_sha256'] or e['decoded_word_sha256']!=e['original_word_sha256']:
            raise ValueError('lossless original input identity differs')
        start=e['start']
        packet,enc=timed(lambda:engine.encode(words,start=start,kappa=0.),torch,device,q['warmup'],q['repeats'])
        if word_sha(engine.last_reconstruction)!=e['decoded_word_sha256']:
            raise AssertionError('sender changed lossless words')
        engine.last_reconstruction=None
        decoded,dec=timed(lambda:engine.decode(packet),torch,device,q['warmup'],q['repeats'])
        if word_sha(decoded)!=e['decoded_word_sha256'] or not torch.equal(decoded,words):
            raise AssertionError('complete receiver differs from original BF16 words')
        begin=time.perf_counter(); wire=packet.export(); export_ms=(time.perf_counter()-begin)*1000
        packet_sha=hashlib.sha256(wire).hexdigest(); matches=packet_sha==e['reference_packet_sha256']
        if q.get('require_original_packet') and (not matches or wire!=Path(e['reference_packet']).read_bytes()):
            raise AssertionError('V128/B8 control changed original fused packet bytes')
        begin=time.perf_counter(); imported=Packet.restore(wire,device)
        if device.type=='cuda': torch.cuda.synchronize(device)
        upload_ms=(time.perf_counter()-begin)*1000
        if word_sha(engine.decode(imported))!=e['decoded_word_sha256']:
            raise AssertionError('serialized receiver changed words')
        ident=q['method']+'_'+e['document']; path=out/(ident+'.kvc'); path.write_bytes(wire)
        geometries=[]
        if not packet.header['raw']:
            for width,count in sorted(Counter(g['columns'] for layer in engine.layers for g in layer['groups']).items()):
                geometries.append(dict(groups=count,columns=width,**geometry(words.shape[1]*width,engine.V,engine.B)))
        state_bytes=sum(g['groups']*g['state_bytes'] for g in geometries)
        directory_bytes=packet.starts.numel()*8; payload_bytes=packet.payload.numel()*2
        row=dict(id=ident,method=q['method'],document=e['document'],kappa=0.,mode='lossless',
            shape=list(words.shape),values=words.numel(),start=start,original_word_sha256=e['original_word_sha256'],
            decoded_word_sha256=e['decoded_word_sha256'],original_exact=True,packet=str(path),packet_sha256=packet_sha,
            packet_matches_original=matches,reference_packet_sha256=e['reference_packet_sha256'],
            reference_packet_bytes=Path(e['reference_packet']).stat().st_size,
            packet_bytes=len(wire),complete_bpv=8*len(wire)/words.numel(),saved_percent=100*(1-len(wire)/(words.numel()*2)),
            state_sha256=packet.header['state_sha256'],engine_options=q['codec']['engine_options'],
            encode=enc,decode=dec,serialized_export_wall_ms=export_ms,upload_wall_ms=upload_ms,
            shared_predictor_gpu_bytes=engine.shared_bytes,quality=e.get('quality'),
            payload_bytes=payload_bytes,directory_bytes=directory_bytes,mandatory_stream_state_bytes=state_bytes,
            framing_bytes=len(wire)-directory_bytes-payload_bytes,coder_geometry=geometries,
            quality_scope='identical original BF16 words; existing raw continuation evidence only')
        rows.append(row)
        checks.append(dict(id=ident,packet=str(path),packet_sha256=packet_sha,decoded_word_sha256=e['decoded_word_sha256']))
        log(f"{ident}: {row['complete_bpv']:.5f} bpv; encode {enc['wall_median_ms']:.3f} ms; decode {dec['wall_median_ms']:.3f} ms")
        del words,packet,decoded,wire,imported; gc.collect()
    del engine; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    request=dict(codec=q['codec'],entries=checks,output=str(out/'receiver.json'))
    write(out/'receiver_request.json',request)
    argv=[sys.executable,Path(__file__).resolve(),'--device',str(device),'--receive',out/'receiver_request.json']
    if device.type!='cuda': argv.append('--software-check')
    launch(argv,out/'receiver.log')
    proof=json.loads((out/'receiver.json').read_text()); byid={r['id']:r for r in proof['checks']}
    if (len(byid)!=len(rows) or set(byid)!={r['id'] for r in rows} or
            proof['gpu_measured']!=(device.type=='cuda') or not proof['no_source_kv_or_predictions']):
        raise ValueError('independent receiver coverage/device differs')
    for row in rows:
        if (not byid[row['id']]['verified'] or byid[row['id']]['word_sha256']!=row['decoded_word_sha256'] or
                byid[row['id']]['packet_sha256']!=row['packet_sha256']): raise AssertionError('independent replay differs')
        row['status']='verified'
    write(out/'worker_results.json',dict(status='completed' if device.type=='cuda' else 'software_checked',
        gpu_measured=device.type=='cuda',rows=rows,cold_initialization_wall_ms=cold,codec_provenance=q['codec'],
        device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU interpreter',
        software={n:importlib.metadata.version(n) for n in ('torch','triton','numpy')},
        receiver_sha256=digest(out/'receiver.json')))


def report(result,out):
    lines=['# Complete lossless Lite stream-length tuning','',
        f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured',False)}**.",'',
        'Same original BF16 words, fused execution, fitted predictor and entropy kernels. Only stream length and lanes per program change.',
        'Means average per-document synchronized wall medians. Complete packets charge framing, all stream states and INT64 directories.','',
        '| Method | Complete bpv | Saved | Encode ms | Decode ms | Exact |','|---|---:|---:|---:|---:|---|']
    for r in result.get('aggregates',[]):
        lines.append(f"| {r['method']} | {r['complete_bpv']:.5f} | {r['saved_percent']:.2f}% | {r['encode_ms']:.3f} | {r['decode_ms']:.3f} | {r['original_exact']} |")
    if result.get('network_model'):
        lines+=['','Calculated serial network model; these are not measured network or serving latencies.','',
            '| Effective Gb/s | Best fresh encode + transfer + decode | Best precompressed fetch + decode |','|---:|---|---|']
        for e in result['network_model']['links']:
            lines.append(f"| {e['effective_gbps']:g} | {e['best_fresh']} | {e['best_precompressed']} |")
    if result.get('fetch_budgets'):
        lines+=['','Candidate decode budgets use each new complete packet size and the retimed official SplitZip result.','',
            '| Candidate | Gb/s | Decode budget ms | Current decode ms | Modeled fetch ms | SplitZip fetch ms |','|---|---:|---:|---:|---:|---:|']
        for e in result['fetch_budgets']:
            lines.append(f"| {e['method']} | {e['effective_gbps']:g} | {e['decode_budget_ms']:.3f} | {e['current_decode_ms']:.3f} | {e['modeled_fetch_ms']:.3f} | {e['baseline_modeled_fetch_ms']:.3f} |")
    if result.get('profiles'):
        lines+=['','Separate optional diagnostic processes ran after ordinary timings. Full traces stay in runs/.','',
            '| Profile | Direction | GPU kernels | Sum kernel ms |','|---|---|---:|---:|']
        for name,job in result['profiles'].items():
            for p in job['profiles']:
                lines.append(f"| {name} | {p['direction']} | {p['cuda_kernel_calls']} | {p['cuda_kernel_duration_sum_ms']:.3f} |")
    if result.get('error'): lines+=['','Failure: '+result['error']]
    lines+=['','V128/B8 controls before/after the sweep must match the previous fused packet bytes. Other layouts preserve every original BF16 word but have newly measured packet bytes.',
        'Retimed raw BF16, official SplitZip and official ExANS retain their original environments, fixed state and accounting. No AQUA adapter is substituted; its official run remains blocked.',
        'Every method has a fresh GPU packet-only receiver. Timing workers never profile. Input upload/export, cold initialization and shared predictor state are separate.',
        'No fitting, model download, baseline rebuild, actual network, SSD, pool conversion, attention or serving measurement is performed.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,default=ROOT/'runs/lite_execution_a4500_20261028_081432_1775498')
    p.add_argument('--output',type=Path); p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=2); p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--profile',action='store_true',help='separate reference and lowest-latency candidate text diagnostics after timings')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS); p.add_argument('--receive',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--software-check',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.worker or args.receive:
        torch,device=device_check(args.device,args.software_check)
        q=json.loads((args.worker or args.receive).read_text()); (worker if args.worker else receive)(q,torch,device); return 0
    if args.output is None: p.error('--output required')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=dict(format='lite_streams_v1',status='running',gpu_measured=False,jobs={},profiles={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        result['aggregates']=aggregate(result['rows']); result['network_model']=network_model(result['aggregates'])
        result['fetch_budgets']=fetch_budgets(result['aggregates'])
        write(out/'results.json',result); report(result,out)
    try:
        torch,device=device_check(args.device)
        if args.warmup<1 or args.repeats<2: raise ValueError('warmup>=1 and repeats>=2 required')
        parent_dir=args.parent_run.resolve(); parent=json.loads((parent_dir/'results.json').read_text())
        from scripts.package_lite_execution import verify as verify_parent
        if parent['status']!='completed' or not parent.get('gpu_measured'): raise ValueError('completed GPU parent required')
        verify_parent(parent_dir,parent)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']): raise ValueError('original GPU/software required')
        inherited={**parent['recipe']['unchanged_code'],**parent['recipe']['code']}
        for name,wanted in inherited.items():
            if digest(ROOT/name)!=wanted: raise ValueError('original measured source changed: '+name)
        grand_dir=Path(parent['recipe']['parent_run']); grand=json.loads((grand_dir/'results.json').read_text())
        if digest(grand_dir/'results.json')!=parent['recipe']['parent_results_sha256']: raise ValueError('ancestor evidence changed')
        base_dir=Path(grand['recipe']['base_run']); base,inputs=verified_inputs(base_dir)
        if digest(base_dir/'results.json')!=grand['recipe']['base_results_sha256']: raise ValueError('original input evidence changed')
        names=(*BASELINES,'lite_b8_fused')
        requests={n:json.loads((parent_dir/n/'request.json').read_text()) for n in names}
        for name,q in requests.items():
            data=json.loads((parent_dir/name/'worker_results.json').read_text())
            if q['codec']!=data['codec_provenance']: raise ValueError('parent worker codec changed')
            if name in BASELINES and q['inputs']!=inputs: raise ValueError('parent baseline inputs changed')
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_requests_sha256={n:digest(parent_dir/n/'request.json') for n in names},
            code={n:digest(ROOT/n) for n in FILES},unchanged_code=inherited,
            warmup=args.warmup,repeats=args.repeats,profiles=args.profile,software=versions,
            device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,
            configurations=[list(c) for c in CONFIGURATIONS])
        identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=identity: raise ValueError('resume identity differs; preserve run and choose fresh output')
            result=old; writable=True
        result.update(identity=identity,recipe=recipe,status='running',gpu_measured=True,
            model=parent['model'],hardware=parent['hardware'])
        result.pop('error',None); result.pop('traceback',None); write(out/'source_manifest.json',recipe)
        plan=[]
        for name in BASELINES:
            q=dict(requests[name]); q.update(output=str(out/name),warmup=args.warmup,repeats=args.repeats)
            python=sys.executable; env=None
            if name=='splitzip_official_framed': python=str(ROOT/'runs/_llama_official/envs/splitzip/bin/python')
            if name=='exans_official_fixed_framed':
                from kv_compare.profile_recovery import runtime_environment
                spec=importlib.util.find_spec('nvidia.libnvcomp')
                if spec is None or importlib.metadata.version('nvidia-libnvcomp-cu12')!=grand['recipe']['nvcomp']:
                    raise ValueError('original pinned nvCOMP runtime required')
                toolchain_path=Path(q['codec']['library']).parent.parent/'toolchain.json'
                if digest(toolchain_path)!=grand['recipe']['toolchain_sha256']: raise ValueError('ExANS toolchain receipt changed')
                toolchain=json.loads(toolchain_path.read_text())
                env=runtime_environment(os.environ,toolchain['cuda_home'],[Path(s)/'lib' for s in spec.submodule_search_locations])
            plan.append((name,q,python,ROOT/'scripts/bench_lite_baselines.py',env))
        entries=[]
        ref=requests['lite_b8_fused']['codec']
        for r in parent['jobs']['lite_b8_fused']['rows']:
            if r['kappa']!=0 or not r['original_exact']: raise ValueError('lossless parent rows required')
            e=next(e for e in inputs if e['document']==r['document'])
            if digest(r['packet'])!=r['packet_sha256']: raise ValueError('reference packet changed')
            with Path(r['packet']).open('rb') as stream:
                head=stream.read(24); nh=struct.unpack('<I',head[4:8])[0]; header=json.loads(stream.read(nh))
            entries.append(dict(document=r['document'],input=e['path'],input_file_sha256=e['file_sha256'],
                reference_packet=r['packet'],reference_packet_sha256=r['packet_sha256'],
                original_word_sha256=r['original_word_sha256'],decoded_word_sha256=r['decoded_word_sha256'],
                start=header['start'],quality=r.get('quality')))
        for name,v,b in CONFIGURATIONS:
            q=dict(method=name,codec=dict(state=ref['state'],engine_options=dict(ref['engine_options'],values_per_stream=v,block=b)),
                entries=entries,output=str(out/name),warmup=args.warmup,repeats=args.repeats,require_original_packet=v==128 and b==8)
            plan.append((name,q,sys.executable,Path(__file__).resolve(),None))
        result['selected_jobs']=[x[0] for x in plan]; save()
        for name,q,python,script,env in plan:
            directory=out/name; directory.mkdir(exist_ok=True); old=result['jobs'].get(name,{})
            if old.get('status')=='completed':
                for path,wanted in old['artifacts'].items():
                    if digest(confined(out,path))!=wanted: raise ValueError('completed evidence changed')
                for row in old['rows']:
                    if digest(row['packet'])!=row['packet_sha256']: raise ValueError('completed packet changed')
                log('reuse completed method: '+name); continue
            write(directory/'request.json',q)
            launch([python,script,'--device',str(device),'--worker',directory/'request.json'],directory/'worker.log',env)
            data=json.loads((directory/'worker_results.json').read_text())
            if data['status']!='completed' or not data['gpu_measured'] or data['device']!=recipe['device']:
                raise ValueError('worker did not complete on the original CUDA device')
            if script==Path(__file__).resolve() and data['software']!={n:versions[n] for n in ('torch','triton','numpy')}:
                raise ValueError('worker software changed')
            result['jobs'][name]=dict(status='completed',rows=data['rows'],
                artifacts={str((directory/f).relative_to(out)):digest(directory/f) for f in ('worker_results.json','receiver.json')})
            save()
        validate_comparison(result['rows'])
        if args.profile:
            candidates=[r for r in result['aggregates'] if r['method'].startswith('lite_') and not r['method'].endswith(('before','after'))]
            selected=['lite_v128_b8_before',min(candidates,key=lambda r:r['decode_ms'])['method']]
            result['selected_profiles']=selected; save()
            for name in selected:
                directory=out/'profiles'/name; directory.mkdir(parents=True,exist_ok=True)
                if name in result['profiles']:
                    if digest(directory/'profile.json')!=result['profiles'][name]['summary_sha256']: raise ValueError('profile evidence changed')
                    continue
                q=next(q for n,q,_,_,_ in plan if n==name)
                profile_entries=[]
                for r in result['jobs'][name]['rows']:
                    if r['document']!='test_text': continue
                    e=next(e for e in entries if e['document']==r['document'])
                    profile_entries.append(dict(document=e['document'],input=e['input'],input_file_sha256=e['input_file_sha256'],
                        packet=r['packet'],packet_sha256=r['packet_sha256'],original_word_sha256=r['original_word_sha256'],
                        decoded_word_sha256=r['decoded_word_sha256']))
                if len(profile_entries)!=1: raise ValueError('one text diagnostic case required')
                pq=dict(codec=dict(q['codec'],arm='fused'),entries=profile_entries,output=str(directory))
                write(directory/'request.json',pq)
                launch([sys.executable,ROOT/'scripts/bench_lite_execution.py','--device',str(device),'--profile-worker',directory/'request.json'],directory/'worker.log')
                data=json.loads((directory/'profile.json').read_text()); data['summary_sha256']=digest(directory/'profile.json')
                result['profiles'][name]=data; save()
        result['status']='completed'; save(); log('completed: '+str(out/'summary.md')); return 0
    except Exception as error:
        if writable:
            result.update(status='failed',error=str(error),traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
