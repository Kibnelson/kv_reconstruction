#!/usr/bin/env python3
"""Profile current Lite and test byte-compatible execution ablations on CUDA."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_llama_causal import device_check, log, timed
from scripts.bench_lite_baselines import digest, launch, write
from scripts.bench_llama_baselines import verified_inputs, confined
from kv_compare.lite_scoreboard import aggregate, network_model, validate_comparison

FILES = ('kv_compare/lite_fused.py','kv_compare/lite_execution_profile.py',
    'scripts/bench_lite_execution.py','scripts/package_lite_execution.py',
    'scripts/run_lite_execution_a4500.sh','tests/test_lite_fused.py','kv_compare/causal_profile.py')


def codec(spec, device):
    if spec['arm'] not in ('reference','scatter','rope','fused'):
        raise ValueError('unknown execution ablation')
    if spec['arm']=='reference':
        from kv_compare.lite_fast import FastEngine
        return FastEngine(spec['state'],device,**spec['engine_options'])
    from kv_compare.lite_fused import FusedEngine
    return FusedEngine(spec['state'],device,**spec['engine_options'],
        fuse_scatter=spec['arm'] in ('scatter','fused'),fuse_rope=spec['arm'] in ('rope','fused'))


def load_entry(entry, torch, device):
    import numpy as np
    from kv_compare.causal_packet import Packet, word_sha
    if digest(entry['input']) != entry['input_file_sha256'] or digest(entry['packet']) != entry['packet_sha256']:
        raise ValueError('frozen input/packet changed')
    words = torch.from_numpy(np.load(entry['input'],allow_pickle=False).view(np.int16).copy()).to(device)
    if word_sha(words) != entry['original_word_sha256']:
        raise ValueError('input is not the original matched BF16 words')
    wire = Path(entry['packet']).read_bytes()
    packet = Packet.restore(wire,device)
    return words,wire,packet


def receive(q, torch, device):
    from kv_compare.causal_packet import Packet, word_sha
    engine = codec(q['codec'],device); checks=[]
    for e in q['entries']:
        if digest(e['packet'])!=e['packet_sha256']: raise ValueError('receiver packet changed')
        output = engine.decode(Packet.restore(Path(e['packet']).read_bytes(),device))
        actual = word_sha(output)
        if actual!=e['decoded_word_sha256']: raise AssertionError('fresh receiver changed words')
        checks.append(dict(id=e['id'],verified=True,packet_sha256=e['packet_sha256'],word_sha256=actual,shape=list(output.shape)))
    write(q['output'],dict(gpu_measured=device.type=='cuda',no_source_kv_or_predictions=True,
        inputs='fixed codec state + existing packet bytes; expected digests are assertions',checks=checks))


def worker(q, torch, device):
    from kv_compare.causal_packet import word_sha
    out=Path(q['output']); out.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter(); engine=codec(q['codec'],device)
    if device.type=='cuda': torch.cuda.synchronize(device)
    cold=(time.perf_counter()-begin)*1000; rows=[]; checks=[]
    for e in q['entries']:
        words,wire,packet=load_entry(e,torch,device)
        kappa=packet.header['kappa']
        if kappa: engine.residual(kappa)
        encoded,enc=timed(lambda:engine.encode(words,start=packet.header['start'],kappa=kappa),
            torch,device,q['warmup'],q['repeats'])
        if encoded.export()!=wire or word_sha(engine.last_reconstruction)!=e['decoded_word_sha256']:
            raise AssertionError('execution change altered original packet/reconstruction')
        engine.last_reconstruction=None; del encoded
        recovered,dec=timed(lambda:engine.decode(packet),torch,device,q['warmup'],q['repeats'])
        if word_sha(recovered)!=e['decoded_word_sha256']: raise AssertionError('decode changed words')
        exact=bool(torch.equal(recovered,words))
        if not kappa and not exact: raise AssertionError('lossless original words changed')
        row=dict(id=f"{q['method']}_k{kappa:g}_{e['document']}",method=q['method'],document=e['document'],
            kappa=kappa,mode='lossy' if kappa else 'lossless',shape=list(words.shape),values=words.numel(),
            original_word_sha256=e['original_word_sha256'],decoded_word_sha256=e['decoded_word_sha256'],
            original_exact=exact,packet_bytes=len(wire),complete_bpv=8*len(wire)/words.numel(),
            saved_percent=100*(1-len(wire)/(2*words.numel())),packet=e['packet'],packet_sha256=e['packet_sha256'],
            packet_matches_original=True,state_sha256=packet.header['state_sha256'],encode=enc,decode=dec,
            shared_predictor_gpu_bytes=engine.shared_bytes,quality=e.get('quality'),
            quality_scope='prior short continuation evidence; identical reconstructed words required')
        rows.append(row)
        checks.append(dict(id=row['id'],packet=e['packet'],packet_sha256=e['packet_sha256'],decoded_word_sha256=e['decoded_word_sha256']))
        log(f"{row['id']}: byte-identical; encode {enc['wall_median_ms']:.3f} ms; decode {dec['wall_median_ms']:.3f} ms")
        del words,wire,packet,recovered; gc.collect()
    del engine; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    request=dict(codec=q['codec'],entries=checks,output=str(out/'receiver.json'))
    write(out/'receiver_request.json',request)
    argv=[sys.executable,Path(__file__).resolve(),'--device',str(device),'--receive',out/'receiver_request.json']
    if device.type!='cuda': argv.append('--software-check')
    launch(argv,out/'receiver.log')
    proof=json.loads((out/'receiver.json').read_text()); byid={c['id']:c for c in proof['checks']}
    if set(byid)!={r['id'] for r in rows} or proof['gpu_measured']!=(device.type=='cuda'):
        raise ValueError('independent receiver device/coverage differs')
    for row in rows:
        if not byid[row['id']]['verified'] or byid[row['id']]['word_sha256']!=row['decoded_word_sha256']:
            raise AssertionError('fresh replay differs')
        row['status']='verified'
    write(out/'worker_results.json',dict(status='completed' if device.type=='cuda' else 'software_checked',
        gpu_measured=device.type=='cuda',rows=rows,cold_initialization_wall_ms=cold,codec_provenance=q['codec'],
        device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU interpreter',
        software={n:importlib.metadata.version(n) for n in ('torch','triton','numpy')},
        receiver_sha256=digest(out/'receiver.json')))


def profile_worker(q, torch, device):
    from kv_compare.causal_packet import word_sha
    from kv_compare.lite_execution_profile import profile
    out=Path(q['output']); out.mkdir(parents=True,exist_ok=True)
    engine=codec(q['codec'],device); data=[]
    for e in q['entries']:
        words,wire,packet=load_entry(e,torch,device); kappa=packet.header['kappa']
        # Warm compilation and geometry caches in this diagnostic process only.
        warm=engine.encode(words,start=packet.header['start'],kappa=kappa)
        if warm.export()!=wire: raise AssertionError('profile warmup changed packet')
        if word_sha(engine.decode(packet))!=e['decoded_word_sha256']: raise AssertionError('profile warmup changed words')
        del warm; engine.last_reconstruction=None
        for direction,fn in (('encode',lambda:engine.encode(words,start=packet.header['start'],kappa=kappa)),
                             ('decode',lambda:engine.decode(packet))):
            trace=out/f"{direction}.json.gz"
            value,result=profile(fn,engine,kappa,device,trace)
            if direction=='encode':
                if value.export()!=wire or word_sha(engine.last_reconstruction)!=e['decoded_word_sha256']:
                    raise AssertionError('profile changed encoder output')
                engine.last_reconstruction=None
            elif word_sha(value)!=e['decoded_word_sha256']: raise AssertionError('profile changed decoded words')
            result.update(direction=direction,document=e['document'],kappa=kappa,
                packet_sha256=e['packet_sha256'],trace=str(trace),trace_sha256=digest(trace))
            data.append(result); del value
    write(out/'profile.json',dict(gpu_measured=True,status='completed',codec_provenance=q['codec'],profiles=data))


def report(result,out):
    lines=['# Lite execution study','',f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured',False)}**.",'',
        'Complete original packets, unchanged fit and entropy kernels. Timings come from fresh unprofiled processes.',
        'Reference B8 is repeated before/after the lossless ablations. Reference B32 brackets the lossy comparison.','',
        '| Method | Kappa | Saved | Encode ms | Decode ms | Exact original |','|---|---:|---:|---:|---:|---|']
    for row in result.get('aggregates',[]):
        lines.append(f"| {row['method']} | {row['kappa']:g} | {row['saved_percent']:.2f}% | {row['encode_ms']:.3f} | {row['decode_ms']:.3f} | {row['original_exact']} |")
    if result.get('profiles'):
        lines+=['','Separate instrumented diagnostics. Kernel times exclude inclusive aten/user ranges; do not add host and device totals.','',
            '| Profile | Direction | GPU kernel calls | Sum kernel ms | GPU busy union ms | First-to-last kernel span ms |','|---|---|---:|---:|---:|---:|']
        for name,job in result['profiles'].items():
            for p in job['profiles']:
                lines.append(f"| {name} | {p['direction']} | {p['cuda_kernel_calls']} | {p['cuda_kernel_duration_sum_ms']:.3f} | {p['cuda_kernel_busy_union_ms']:.3f} | {p['first_to_last_kernel_span_ms']:.3f} |")
        for name,job in result['profiles'].items():
            for p in job['profiles']:
                top=p['kernels'][:5]
                lines+=['',f"Top kernels for {name} / {p['direction']}:",'']
                lines += [f"- `{k['name']}`: {k['self_device_ms']:.3f} ms, {k['calls']} calls." for k in top]
    baseline=next((r for r in result.get('aggregates',[]) if r['method']=='splitzip_official_framed'),None)
    if baseline:
        lines+=['','Lossless fetch targets against retimed official SplitZip. The decode budget includes the candidate packet-size advantage.','',
            '| Candidate | Decode ms | Required decode ms at 5 Gb/s | Current / required |','|---|---:|---:|---:|']
        for r in result.get('aggregates',[]):
            if r['mode']!='lossless' or not r['method'].startswith('lite_'): continue
            budget=baseline['decode_ms']+8*(baseline['packet_bytes']-r['packet_bytes'])/(5*1e6)
            lines.append(f"| {r['method']} | {r['decode_ms']:.3f} | {budget:.3f} | "+
                (f"{r['decode_ms']/budget:.2f}x" if budget>0 else 'no positive budget')+' |')
    if result.get('network_model'):
        lines+=['','Calculated serial link model, not measured network latency.','',
            '| Gb/s | Best fresh encode + transfer + decode | Best precompressed transfer + decode |','|---:|---|---|']
        for row in result['network_model']['links']:
            lines.append(f"| {row['effective_gbps']:g} | {row['best_fresh']} | {row['best_precompressed']} |")
    if result.get('error'): lines+=['','Failure: '+result['error']]
    lines+=['','Every Lite row must match its previous packet bytes and intended BF16 reconstruction. Nonzero kappa remains lossy.',
        'Raw BF16, FP8, official SplitZip and official ExANS are retimed using the previous checked worker requests and unchanged baseline harness.',
        'Shared state, cold initialization and per-call allocation peaks are additional. Kernel fusion currently uses the reference path for attention scaling other than 1.',
        'Lossy quality is the prior short continuation check, not full task validation. Official AQUA remains blocked.',
        'Profiles run in separate processes after ordinary measurements. Full traces stay in runs/. No SSD, actual network, pool attention or serving result is measured.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run',type=Path,default=ROOT/'runs/lite_baselines_a4500_20261028_070735_342774')
    parser.add_argument('--output',type=Path); parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--warmup',type=int,default=2); parser.add_argument('--repeats',type=int,default=6)
    parser.add_argument('--no-profile',action='store_true')
    for name in ('worker','receive','profile-worker'): parser.add_argument('--'+name,type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--software-check',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    request=args.worker or args.receive or args.profile_worker
    if request:
        torch,device=device_check(args.device,args.software_check)
        q=json.loads(request.read_text())
        (worker if args.worker else profile_worker if args.profile_worker else receive)(q,torch,device)
        return 0
    if args.output is None: parser.error('--output required')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=dict(format='lite_execution_v1',status='running',gpu_measured=False,jobs={},profiles={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        result['aggregates']=aggregate(result['rows']); result['network_model']=network_model(result['aggregates'])
        write(out/'results.json',result); report(result,out)
    try:
        torch,device=device_check(args.device)
        if args.warmup<1 or args.repeats<2: raise ValueError('warmup>=1 and repeats>=2 required')
        parent_dir=args.parent_run.resolve(); parent=json.loads((parent_dir/'results.json').read_text())
        from scripts.package_lite_baselines import verify
        if parent['status']!='completed' or not parent.get('gpu_measured'): raise ValueError('completed GPU parent required')
        verify(parent_dir,parent)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']):
            raise ValueError('use the original GPU/software for the controlled comparison')
        for name,wanted in dict(parent['recipe']['original_code'],**parent['recipe']['code']).items():
            if digest(ROOT/name)!=wanted: raise ValueError('original measured source changed: '+name)
        base_dir=Path(parent['recipe']['base_run']); base,inputs=verified_inputs(base_dir)
        if digest(base_dir/'results.json')!=parent['recipe']['base_results_sha256']: raise ValueError('base evidence changed')
        requests={name:json.loads((parent_dir/name/'request.json').read_text()) for name in parent['selected_jobs']}
        for name,q in requests.items():
            data=json.loads((parent_dir/name/'worker_results.json').read_text())
            if q['codec']!=data['codec_provenance']: raise ValueError('original worker codec changed')
            if q['inputs']!=inputs: raise ValueError('original worker inputs differ')
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_requests_sha256={name:digest(parent_dir/name/'request.json') for name in requests},
            code={name:digest(ROOT/name) for name in FILES},unchanged_code=dict(parent['recipe']['original_code'],**parent['recipe']['code']),
            warmup=args.warmup,repeats=args.repeats,profiles=not args.no_profile,software=versions,
            device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda)
        identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=identity: raise ValueError('resume identity differs; choose a fresh output')
            result=old; writable=True
        result.update(identity=identity,recipe=recipe,status='running',gpu_measured=True,
            model=parent['model'],hardware=parent['hardware'])
        result.pop('error',None); result.pop('traceback',None)
        write(out/'source_manifest.json',recipe)
        plan=[]
        for name in ('raw_bf16','splitzip_official_framed','fp8_e4m3fn_saturating','exans_official_fixed_framed'):
            q=dict(requests[name]); q.update(output=str(out/name),warmup=args.warmup,repeats=args.repeats)
            python=sys.executable; env=None
            if name=='splitzip_official_framed': python=str(ROOT/'runs/_llama_official/envs/splitzip/bin/python')
            if name=='exans_official_fixed_framed':
                from kv_compare.profile_recovery import runtime_environment
                spec=importlib.util.find_spec('nvidia.libnvcomp')
                if spec is None: raise ValueError('pinned nvCOMP runtime missing')
                if importlib.metadata.version('nvidia-libnvcomp-cu12')!=parent['recipe']['nvcomp']:
                    raise ValueError('nvCOMP version changed')
                build_dir=Path(requests[name]['codec']['library']).parent
                toolchain_path=build_dir.parent/'toolchain.json'
                if digest(toolchain_path)!=parent['recipe']['toolchain_sha256']: raise ValueError('ExANS toolchain receipt changed')
                toolchain=json.loads(toolchain_path.read_text())
                env=runtime_environment(os.environ,toolchain['cuda_home'],[Path(p)/'lib' for p in spec.submodule_search_locations])
            plan.append((name,q,python,ROOT/'scripts/bench_lite_baselines.py',env))
        configurations=(('b8_ref_before',8,'reference',(0.,)),('b8_scatter',8,'scatter',(0.,)),
            ('b8_rope',8,'rope',(0.,)),('b8_fused',8,'fused',(0.,)),('b8_ref_after',8,'reference',(0.,)),
            ('b32_ref_before',32,'reference',(.5,1.)),('b32_fused',32,'fused',(.5,1.)),('b32_ref_after',32,'reference',(.5,1.)))
        for tag,block,arm,kappas in configurations:
            oldname=f'lite_b{block}'; spec=requests[oldname]['codec']
            entries=[]
            for r in parent['jobs'][oldname]['rows']:
                if r['kappa'] not in kappas: continue
                e=next(e for e in inputs if e['document']==r['document'])
                entries.append(dict(document=r['document'],input=e['path'],input_file_sha256=e['file_sha256'],
                    packet=r['packet'],packet_sha256=r['packet_sha256'],original_word_sha256=r['original_word_sha256'],
                    decoded_word_sha256=r['decoded_word_sha256'],quality=r.get('quality')))
            name='lite_'+tag
            q=dict(method=name,codec=dict(arm=arm,state=spec['state'],engine_options=spec['engine_options']),
                entries=entries,output=str(out/name),warmup=args.warmup,repeats=args.repeats)
            plan.append((name,q,sys.executable,Path(__file__).resolve(),None))
        result['selected_jobs']=[p[0] for p in plan]; save()
        for name,q,python,script,env in plan:
            directory=out/name; directory.mkdir(exist_ok=True)
            old=result['jobs'].get(name,{})
            if old.get('status')=='completed':
                for path,wanted in old['artifacts'].items():
                    if digest(confined(out,path))!=wanted: raise ValueError('completed evidence changed')
                for row in old['rows']:
                    if digest(row['packet'])!=row['packet_sha256']: raise ValueError('completed packet changed')
                log('reuse completed job: '+name); continue
            write(directory/'request.json',q)
            launch([python,script,'--device',str(device),'--worker',directory/'request.json'],directory/'worker.log',env)
            data=json.loads((directory/'worker_results.json').read_text())
            if data['status']!='completed' or not data['gpu_measured']: raise ValueError('worker did not complete on CUDA')
            if data['device']!=recipe['device']: raise ValueError('worker GPU differs')
            if script==Path(__file__).resolve() and data['software']!={n:versions[n] for n in ('torch','triton','numpy')}:
                raise ValueError('worker software differs')
            result['jobs'][name]=dict(status='completed',rows=data['rows'],
                artifacts={str((directory/f).relative_to(out)):digest(directory/f) for f in ('worker_results.json','receiver.json')})
            save()
        validate_comparison(result['rows'])
        if not args.no_profile:
            for name in ('lite_b8_ref_before','lite_b8_fused','lite_b32_ref_before','lite_b32_fused'):
                q=dict(next(q for n,q,_,_,_ in plan if n==name))
                q['entries']=[e for e in q['entries'] if e['document']=='test_text' and
                    next(r for r in result['jobs'][name]['rows'] if r['packet']==e['packet'])['kappa'] in (0.,.5)]
                if len(q['entries'])!=1: raise ValueError('expected one text diagnostic case')
                directory=out/'profiles'/name; q['output']=str(directory); directory.mkdir(parents=True,exist_ok=True)
                if name in result['profiles']:
                    path=directory/'profile.json'
                    if digest(path)!=result['profiles'][name]['summary_sha256']: raise ValueError('profile receipt changed')
                    continue
                write(directory/'request.json',q)
                launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--profile-worker',directory/'request.json'],directory/'worker.log')
                data=json.loads((directory/'profile.json').read_text()); data['summary_sha256']=digest(directory/'profile.json')
                result['profiles'][name]=data; save()
        result['status']='completed'; save(); log('completed: '+str(out/'summary.md')); return 0
    except Exception as error:
        if writable:
            result.update(status='failed',error=str(error),traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
