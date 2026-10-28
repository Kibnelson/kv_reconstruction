#!/usr/bin/env python3
"""Isolate uninitialized-allocation fills; native baseline runtimes stay original."""
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
import math
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_llama_causal import device_check,log,timed
from scripts.bench_lite_baselines import digest,launch,write
from scripts.bench_llama_baselines import confined,device_check as baseline_device_check
from kv_compare.lite_allocation import allocation_policy
from scripts.bench_lite_graph import aggregate_decode,load_packets,check_output,native_receipt
from kv_compare.lite_scoreboard import validate_comparison

FILES=('scripts/bench_lite_allocation.py','scripts/package_lite_allocation.py','scripts/run_lite_allocation_a4500.sh',
       'kv_compare/lite_allocation.py','tests/test_lite_allocation.py','tests/test_lite_allocation_runner.py')
BASELINES=('raw_bf16','splitzip_official_framed','exans_official_fixed_framed')
ARMS=('fill_on_graph','fill_off_graph')
POLICIES={'fill_on_graph':True,'fill_off_graph':False}
LINKS=(1.,2.5,5.,10.,25.,50.,100.,200.,400.)


def pcie_metadata():
    fields='name,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max'
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader'],
            capture_output=True,text=True,timeout=15)
        return dict(status='recorded' if p.returncode==0 else 'unavailable',fields=fields,
            stdout=p.stdout.strip(),stderr=p.stderr.strip(),returncode=p.returncode,
            scope='interface metadata only; current generation may be lower at idle; not measured transfer bandwidth')
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(status='unavailable',reason=str(error),scope='optional read-only interface metadata')


def fetch_model(aggregates,links=LINKS):
    if any(not math.isfinite(g) or g<=0 for g in links): raise ValueError('positive finite effective links required')
    return dict(measured_network=False,scope='serial precompressed transfer+decode model; no fresh encode projection, actual transport, staging, disk, serialization, overlap or serving measurement',
        links=[dict(effective_gbps=g,estimates=[dict(method=r['method'],transfer_ms=8*r['packet_bytes']/(g*1e6),
            fetch_ms=r['decode_ms']+8*r['packet_bytes']/(g*1e6)) for r in aggregates]) for g in links])


def budgets(aggregates,links=LINKS):
    controls={r['method']:r for r in aggregates if r['method'] in ('raw_bf16','splitzip_official_framed')}
    result=[]
    for row in aggregates:
        if not row['method'].startswith('lite_'): continue
        for name,control in controls.items():
            for g in links:
                if not math.isfinite(g) or g<=0: raise ValueError('positive finite effective link required')
                limit=control['decode_ms']+8*(control['packet_bytes']-row['packet_bytes'])/(g*1e6)
                result.append(dict(method=row['method'],baseline=name,effective_gbps=g,decode_budget_ms=limit,
                    current_decode_ms=row['decode_ms'],required_reduction_factor=row['decode_ms']/limit if limit>0 else None,
                    modeled_fetch_ms=row['decode_ms']+8*row['packet_bytes']/(g*1e6),
                    baseline_modeled_fetch_ms=control['decode_ms']+8*control['packet_bytes']/(g*1e6),measured_network=False))
    return result


def make_engine(q,device,arm):
    from kv_compare.lite_finish import FinishEngine
    return policy_call(arm,FinishEngine,q['codec']['state'],device,**q['codec']['engine_options'])


def runtime_snapshot(torch):
    deterministic=bool(torch.are_deterministic_algorithms_enabled())
    fill=bool(torch.utils.deterministic.fill_uninitialized_memory)
    return dict(deterministic_algorithms_enabled=deterministic,
        deterministic_warn_only=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        fill_uninitialized_memory=fill,effective_uninitialized_fill=deterministic and fill)


def policy_call(arm,fn,*args,**kwargs):
    if arm not in POLICIES: raise ValueError('unknown fill-policy arm')
    with allocation_policy(POLICIES[arm]) as policy:
        if (not policy['deterministic_algorithms_enabled'] or
                policy['fill_uninitialized_memory'] is not POLICIES[arm]):
            raise AssertionError('requested allocation policy was not active')
        return fn(*args,**kwargs)


def policy_records():
    records={}
    for arm,fill in POLICIES.items():
        with allocation_policy(fill) as policy:
            records[arm]=dict(policy,effective_uninitialized_fill=policy['deterministic_algorithms_enabled'] and policy['fill_uninitialized_memory'])
    return records


def baseline_receive(q,torch,device):
    from scripts import bench_lite_baselines as original
    before=runtime_snapshot(torch)
    original.receive(q,torch,device)
    after=runtime_snapshot(torch)
    if before!=after: raise AssertionError('native receiver changed its original runtime flags')
    data=json.loads(Path(q['output']).read_text())
    data.update(runtime_policy=before,runtime_origin='unchanged original baseline_device_check; no allocation_policy context')
    write(q['output'],data)


def baseline_worker(q,torch,device):
    from scripts import bench_lite_baselines as original
    out=Path(q['output']); before=runtime_snapshot(torch)
    # Delegate the official codec/timing path without changing its flags.
    original.worker(q,torch,device)
    after=runtime_snapshot(torch)
    if before!=after: raise AssertionError('native worker changed its original runtime flags')
    original_data=json.loads((out/'worker_results.json').read_text())
    for src,dst in (('worker_results.json','original_worker_results.json'),('receiver.json','original_receiver.json')):
        shutil.copyfile(out/src,out/dst)
    request=json.loads((out/'receiver_request.json').read_text())
    argv=[sys.executable,Path(__file__).resolve(),'--device',str(device),'--baseline-receive',out/'receiver_request.json']
    if device.type!='cuda': argv.append('--software-check')
    launch(argv,out/'runtime_receiver.log')
    proof=json.loads(Path(request['output']).read_text())
    if proof['runtime_policy']!=before: raise AssertionError('fresh native receiver runtime differs')
    original_data.update(receiver_sha256=digest(out/'receiver.json'),runtime_policy=before,
        runtime_policy_after=after,runtime_origin='unchanged original baseline_device_check; no allocation_policy context',
        original_worker_results_sha256=digest(out/'original_worker_results.json'),
        original_receiver_sha256=digest(out/'original_receiver.json'),
        underlying_worker='scripts/bench_lite_baselines.py',
        proof_scope='original proof retained; additional same-runtime packet-only receiver records effective flags')
    write(out/'worker_results.json',original_data)


def make_graphs(q,device):
    from kv_compare.lite_graph import GraphDecoder
    engines={}; graphs={}; cold={}
    for arm in ARMS:
        begin=time.perf_counter(); engines[arm]=make_engine(q,device,arm)
        import torch
        torch.cuda.synchronize(device); cold[arm]=(time.perf_counter()-begin)*1000
    for arm in ARMS: graphs[arm]=policy_call(arm,GraphDecoder,engines[arm],warmup=3)
    return engines,graphs,cold


def receive(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA packet-only receiver required')
    packets,_=load_packets(q,device); engines,graphs,_=make_graphs(q,device)
    policies=policy_records()
    for arm in ARMS: policy_call(arm,graphs[arm].prepare,packets[0])
    checks={}; held={}
    for i in (0,1,0):
        for arm in ARMS:
            value=policy_call(arm,graphs[arm].decode,packets[i]); actual=check_output(value,q['entries'][i])
            ident=q['method']+'_'+arm+'_'+q['entries'][i]['document']
            checks[ident]=dict(id=ident,verified=True,packet_sha256=q['entries'][i]['packet_sha256'],
                word_sha256=actual,shape=list(value.shape))
            if i==0 and arm not in held: held[arm]=value
            elif i==1: check_output(held[arm],q['entries'][0])
    counts={arm:g.capture_count for arm,g in graphs.items()}
    if any(c!=1 for c in counts.values()): raise AssertionError('payload-size change caused recapture')
    write(q['output'],dict(gpu_measured=True,no_source_kv_or_predictions=True,checks=list(checks.values()),
        inputs='fixed state + unchanged serialized packets only; expected digests are assertions; no encoding',
        alternating_packets=True,distinct_payload_lengths=True,independent_returned_outputs=True,
        graph_capture_counts=counts,allocation_policies=policies))


def worker(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA required; no CPU graph substitute')
    out=Path(q['output']); out.mkdir(parents=True,exist_ok=True)
    initial_policy=runtime_snapshot(torch); policies=policy_records()
    packets,uploads=load_packets(q,device); engines,graphs,cold=make_graphs(q,device)
    # Sender parity is outside decode timings. Fresh receivers below do not
    # perform this step and never receive these temporary source arrays.
    parity=[]
    for packet,e in zip(packets,q['entries']):
        source=policy_call('fill_on_graph',engines['fill_on_graph'].decode,packet); check_output(source,e)
        for arm in ARMS:
            encoded=policy_call(arm,engines[arm].encode,source,start=packet.header['start'],kappa=0.)
            if encoded.export()!=Path(e['packet']).read_bytes(): raise AssertionError('allocation policy changed original encoded packet bytes')
            check_output(engines[arm].last_reconstruction,e); engines[arm].last_reconstruction=None
            parity.append(dict(document=e['document'],arm=arm,packet_sha256=e['packet_sha256'],byte_identical=True))
            del encoded
        del source
    for arm in ARMS: policy_call(arm,graphs[arm].prepare,packets[0])
    data={(arm,i):dict(samples=[],warmup_samples=[]) for arm in ARMS for i in range(2)}
    order=[]; held={}
    for round_index in range(q['warmup']+q['repeats']):
        docs=(0,1) if round_index%2==0 else (1,0); arms=ARMS if round_index%2==0 else ARMS[::-1]
        order.append(dict(documents=[q['entries'][i]['document'] for i in docs],arms=list(arms)))
        for i in docs:
            for arm in arms:
                value,sample=timed(lambda arm=arm,i=i:policy_call(arm,graphs[arm].decode,packets[i]),torch,device,0,1)
                check_output(value,q['entries'][i])
                data[arm,i]['warmup_samples' if round_index<q['warmup'] else 'samples'].append(sample['samples'][0])
                if i==0 and arm not in held: held[arm]=value
                elif i==1 and arm in held: check_output(held[arm],q['entries'][0])
                del value
    counts={arm:g.capture_count for arm,g in graphs.items()}
    if any(c!=1 for c in counts.values()): raise AssertionError('alternating packets recaptured graph')
    rows=[]
    for arm in ARMS:
        for i,e in enumerate(q['entries']):
            dec=data[arm,i]; dec['wall_median_ms']=statistics.median(s['wall_ms'] for s in dec['samples'])
            method=q['method']+'_'+arm
            row={k:e[k] for k in ('document','packet','packet_sha256','original_word_sha256','decoded_word_sha256',
                'shape','values','packet_bytes','complete_bpv','saved_percent','state_sha256')}
            row.update(id=method+'_'+e['document'],method=method,kappa=0.,mode='lossless',original_exact=True,
                decode=dec,engine_options=q['codec']['engine_options'],shared_predictor_gpu_bytes=engines[arm].shared_bytes,
                allocation_policy=policies[arm])
            rows.append(row); log(f"{row['id']}: decode {dec['wall_median_ms']:.3f} ms, exact")
    setups={arm:g.setups for arm,g in graphs.items()}
    for arm in ARMS: policy_call(arm,graphs[arm].close)
    del held,graphs,engines; gc.collect(); torch.cuda.empty_cache()
    request=dict(q,output=str(out/'receiver.json')); write(out/'receiver_request.json',request)
    launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--receive',out/'receiver_request.json'],out/'receiver.log')
    proof=json.loads((out/'receiver.json').read_text()); checks={c['id']:c for c in proof['checks']}
    if (not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
            set(checks)!={r['id'] for r in rows} or len(checks)!=len(rows) or proof['graph_capture_counts']!=counts or
            proof['allocation_policies']!=policies):
        raise ValueError('fresh paired receiver coverage differs')
    for row in rows:
        c=checks[row['id']]
        if not c['verified'] or c['word_sha256']!=row['decoded_word_sha256'] or c['packet_sha256']!=row['packet_sha256']:
            raise AssertionError('fresh receiver changed original words')
        row['status']='verified'
    if runtime_snapshot(torch)!=initial_policy: raise AssertionError('allocation policy did not restore worker flags')
    write(out/'worker_results.json',dict(status='completed',gpu_measured=True,rows=rows,
        device=torch.cuda.get_device_name(device),software={n:importlib.metadata.version(n) for n in ('torch','triton','numpy')},
        codec_provenance=q['codec'],cold_state_initialization_wall_ms=cold,graph_setups=setups,graph_capture_counts=counts,
        encoder_byte_parity=parity,sample_order=order,serialized_uploads=uploads,allocation_policies=policies,
        runtime_policy_before=initial_policy,runtime_policy_after=runtime_snapshot(torch),
        receiver_sha256=digest(out/'receiver.json'),memory_scope='both fitted engines and both graph buffers are resident during paired timing'))


def diagnostic(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA diagnostics required')
    from kv_compare.lite_graph import GraphDecoder
    packets,_=load_packets(q,device); engine=make_engine(q,device,q['arm']); graph=GraphDecoder(engine,warmup=3)
    with allocation_policy(POLICIES[q['arm']]) as active:
        policy=dict(active,effective_uninitialized_fill=active['deterministic_algorithms_enabled'] and active['fill_uninitialized_memory'])
    policy_call(q['arm'],graph.prepare,packets[0]); results=[]
    for packet,e in zip(packets,q['entries']):
        measured=policy_call(q['arm'],graph.measure_phases,packet,repeats=q.get('phase_repeats',3))
        check_output(policy_call(q['arm'],graph.decode,packet),e)
        results.append(dict(document=e['document'],packet_sha256=e['packet_sha256'],measurement=measured))
    if graph.capture_count!=1: raise AssertionError('phase diagnostic recaptured')
    write(Path(q['output'])/'phases.json',dict(status='completed',gpu_measured=True,arm=q['arm'],phases=results,
        codec_provenance=q['codec'],graph_setups=graph.setups,allocation_policy=policy,
        scope='separate synchronized one-arm diagnostic; do not add phase sums to ordinary paired latency'))


def profile_worker(q,torch,device):
    if device.type!='cuda': raise RuntimeError('actual CUDA eager kernel profile required')
    from kv_compare.lite_execution_profile import profile
    packets,_=load_packets(q,device); engine=make_engine(q,device,q['arm'])
    i=next(i for i,e in enumerate(q['entries']) if e['document']=='test_text'); packet=packets[i]; entry=q['entries'][i]
    for _ in range(3): check_output(policy_call(q['arm'],engine.decode,packet),entry)
    out=Path(q['output']); out.mkdir(parents=True,exist_ok=True); trace=out/'eager_decode.json.gz'
    with allocation_policy(POLICIES[q['arm']]) as active:
        policy=dict(active,effective_uninitialized_fill=active['deterministic_algorithms_enabled'] and active['fill_uninitialized_memory'])
        value,data=profile(lambda:engine.decode(packet),engine,0.,device,trace)
    check_output(value,entry)
    fills=[k for k in data['kernels'] if 'FillFunctor' in k['name']]
    entropy=[k for k in data['kernels'] if k['name']=='_decode_kernel']
    data['fill_kernel_summary']=dict(calls=sum(k['calls'] for k in fills),device_ms=sum(k['self_device_ms'] for k in fills),
        scope='all observed Torch FillFunctor kernels; includes explicit zeros/ones that remain required')
    data['entropy_kernel_summary']=dict(calls=sum(k['calls'] for k in entropy),device_ms=sum(k['self_device_ms'] for k in entropy))
    data.update(trace=str(trace),trace_sha256=digest(trace),document=entry['document'],packet_sha256=entry['packet_sha256'])
    write(out/'profile.json',dict(status='completed',gpu_measured=True,arm=q['arm'],profile=data,
        codec_provenance=q['codec'],allocation_policy=policy,
        scope='separate instrumented eager decode, not CUDA graph replay latency'))


def report(result,out):
    lines=['# Controlled uninitialized-allocation fill policy','',
        f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured',False)}**.",'',
        'The same FinishEngine and V64/B32 packets run under fill-on and fill-off graphs. Deterministic algorithms stay ON in both Lite arms; only empty-allocation initialization changes.',
        'Both engines and both graph buffers are resident during paired timing. Full decode includes validation, staging copies, replay and an independently owned output clone.','',
        '| Method | Saved | Complete bpv | Decode ms | Exact |','|---|---:|---:|---:|---|']
    for r in result.get('aggregates',[]): lines.append(f"| {r['method']} | {r['saved_percent']:.2f}% | {r['complete_bpv']:.5f} | {r['decode_ms']:.3f} | {r['original_exact']} |")
    policies=[]
    for name,job in result.get('jobs',{}).items():
        if 'runtime_policy' in job: policies.append((name,job['runtime_policy'],'original native runtime'))
        for arm,p in job.get('allocation_policies',{}).items(): policies.append((arm,p,'controlled Lite allocation context'))
    if policies:
        lines+=['','Effective measured runtime flags. Native baselines keep their ORIGINAL device setup; allocation_policy is never applied to them. If deterministic mode is OFF, its empty-allocation fill is inactive regardless of the fill flag.','',
            '| Method / arm | Deterministic algorithms | Fill flag | Effective uninitialized fill | Runtime origin |','|---|---|---|---|---|']
        for name,p,origin in policies:
            active=p['deterministic_algorithms_enabled'] and p['fill_uninitialized_memory']
            lines.append(f"| {name} | {p['deterministic_algorithms_enabled']} | {p['fill_uninitialized_memory']} | {active} | {origin} |")
    pcie=result.get('hardware',{}).get('pcie_query',{})
    if pcie.get('status')=='recorded':
        lines+=['','Interface metadata only; current generation can be lower at idle. This is not a transfer-bandwidth measurement.','',
            '`'+pcie['fields']+'`','','```text',pcie['stdout'],'```']
    setups=[(arm,s) for job in result.get('jobs',{}).values() for arm,records in job.get('graph_setups',{}).items() for s in records]
    if setups:
        mib=lambda b:f"{b/(1<<20):.3f}"
        lines+=['','Cold graph setup and cumulative paired-setup memory; Torch accounting excludes some driver metadata.','',
            '| Arm | Cold ms | Allocated before → after MiB | Reserved before → after MiB | Peak allocated / reserved MiB | Staging payload MiB | Directory KiB | Output MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm,s in setups:
            lines.append(f"| {arm} | {s['cold_synchronized_wall_ms']:.3f} | {mib(s['baseline_torch_bytes'])} → {mib(s['persistent_torch_bytes'])} | {mib(s['baseline_torch_reserved_bytes'])} → {mib(s['persistent_torch_reserved_bytes'])} | {mib(s['peak_torch_bytes'])} / {mib(s['peak_torch_reserved_bytes'])} | {mib(s['staging_payload_bytes'])} | {s['staging_directory_bytes']/1024:.3f} | {mib(s['captured_output_bytes'])} |")
    if result.get('diagnostics'):
        lines+=['','Separate synchronized phase diagnostics; raw samples remain JSON and are not ordinary paired latencies.','',
            '| Arm / document | Validation wall ms | Input-copy CUDA ms | Replay CUDA ms | Replay CPU submit ms | Output-clone CUDA ms |','|---|---:|---:|---:|---:|---:|']
        for arm,data in result['diagnostics'].items():
            for e in data['phases']:
                p=e['measurement']['phases']
                lines.append(f"| {arm} / {e['document']} | {p['validation']['wall_median_ms']:.3f} | {p['input_copy']['cuda_median_ms']:.3f} | {p['replay']['cuda_median_ms']:.3f} | {p['replay']['submit_cpu_median_ms']:.3f} | {p['output_clone']['cuda_median_ms']:.3f} |")
    if result.get('profiles'):
        lines+=['','Separate instrumented eager profiles; these identify kernel work, not captured-graph latency. Fill totals include required explicit zeros/ones.','',
            '| Arm | Kernel calls | Sum kernel ms | Fill-kernel calls | Fill-kernel ms | Entropy calls | Entropy ms |','|---|---:|---:|---:|---:|---:|---:|']
        for arm,data in result['profiles'].items():
            p=data['profile']; fill=p['fill_kernel_summary']; entropy=p['entropy_kernel_summary']
            lines.append(f"| {arm} | {p['cuda_kernel_calls']} | {p['cuda_kernel_duration_sum_ms']:.3f} | {fill['calls']} | {fill['device_ms']:.3f} | {entropy['calls']} | {entropy['device_ms']:.3f} |")
        for arm,data in result['profiles'].items():
            p=data['profile']
            lines+=['',f'Top device kernels for {arm}:','']+[f"- `{k['name']}`: {k['self_device_ms']:.3f} ms, {k['calls']} calls." for k in p['kernels'][:5]]
    lines+=['','Calculated serial precompressed fetch, not measured network/serving latency. No fresh encoder projection.','',
        '| Gb/s | Method | Transfer + decode ms |','|---:|---|---:|']
    for link in result.get('fetch_model',{}).get('links',[]):
        for e in link['estimates']: lines.append(f"| {link['effective_gbps']:g} | {e['method']} | {e['fetch_ms']:.3f} |")
    if result.get('budgets'):
        lines+=['','Required decoder budgets against raw BF16 and official SplitZip, using complete packet sizes. The 400 Gb/s row is a compute-gap target, not an achieved link rate.','',
            '| Candidate | Baseline | Gb/s | Decode budget ms | Current decode ms | Further reduction |','|---|---|---:|---:|---:|---:|']
        for e in result['budgets']:
            factor='no positive budget' if e['required_reduction_factor'] is None else f"{e['required_reduction_factor']:.2f}x"
            lines.append(f"| {e['method']} | {e['baseline']} | {e['effective_gbps']:g} | {e['decode_budget_ms']:.3f} | {e['current_decode_ms']:.3f} | {factor} |")
    if result.get('error'): lines+=['','Failure: '+result['error']]
    lines+=['','Both allocation policies must reproduce the complete original packet bytes outside timing. Fresh GPU receivers decode fixed state plus unchanged packets and verify policy, new data and output ownership.',
        'The allocation flag is process-global; these are exclusive benchmark-worker contexts with restoration, not a concurrent serving API. Required explicit zeros/ones, deterministic algorithms and codec mathematics are unchanged.',
        'Diagnostics/profiles run in separate processes after all ordinary measurements. Full traces remain local. Packet upload, cold capture, shared state and graph memory are additional.',
        'Pinned official baseline methods retain their original worker/runtime policies. No fitting, model download, native build, actual network, SSD, pool attention or serving-concurrency measurement is performed.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,default=ROOT/'runs/lite_finish_a4500_20261028_101451_154264')
    p.add_argument('--output',type=Path); p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=2); p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--phase-repeats',type=int,default=3); p.add_argument('--profile',action='store_true')
    for name in ('worker','receive','diagnostic','profile-worker','baseline-worker','baseline-receive'): p.add_argument('--'+name,type=Path,help=argparse.SUPPRESS)
    p.add_argument('--software-check',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv); request=args.worker or args.receive or args.diagnostic or args.profile_worker or args.baseline_worker or args.baseline_receive
    if request:
        torch,device=(baseline_device_check if args.baseline_worker or args.baseline_receive else device_check)(args.device,args.software_check)
        q=json.loads(request.read_text())
        (baseline_worker if args.baseline_worker else baseline_receive if args.baseline_receive else worker if args.worker else receive if args.receive else diagnostic if args.diagnostic else profile_worker)(q,torch,device); return 0
    if args.output is None: p.error('--output required')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=dict(format='lite_allocation_v1',status='running',gpu_measured=False,jobs={},diagnostics={},profiles={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        result['aggregates']=aggregate_decode(result['rows']); result['fetch_model']=fetch_model(result['aggregates'])
        result['budgets']=budgets(result['aggregates']); write(out/'results.json',result); report(result,out)
    try:
        torch,device=device_check(args.device)
        if args.warmup<1 or args.repeats<2 or args.phase_repeats<2: raise ValueError('warmup>=1 and repeats/phase-repeats>=2 required')
        parent_dir=args.parent_run.resolve(); parent=json.loads((parent_dir/'results.json').read_text())
        from scripts.package_lite_finish import verify as verify_parent
        if parent['status']!='completed' or not parent.get('gpu_measured'): raise ValueError('completed GPU parent required')
        verify_parent(parent_dir,parent)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or torch.cuda.get_device_name(device)!=parent['hardware']['name']):
            raise ValueError('original GPU/software required')
        inherited={**parent['recipe']['unchanged_code'],**parent['recipe']['code']}
        for name,wanted in inherited.items():
            if digest(ROOT/name)!=wanted: raise ValueError('original source changed: '+name)
        receipt=native_receipt(parent); names=(*BASELINES,'lite_finish')
        requests={n:json.loads((parent_dir/n/'request.json').read_text()) for n in names}
        for name,q in requests.items():
            data=json.loads((parent_dir/name/'worker_results.json').read_text())
            if q['codec']!=data['codec_provenance']: raise ValueError('parent codec request changed')
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_requests_sha256={n:digest(parent_dir/n/'request.json') for n in names},code={n:digest(ROOT/n) for n in FILES},
            unchanged_code=inherited,warmup=args.warmup,repeats=args.repeats,phase_repeats=args.phase_repeats,profile=args.profile,
            software=versions,device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,arms=list(ARMS))
        identity=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=identity: raise ValueError('resume identity differs; choose fresh output')
            result=old; writable=True
        result.update(identity=identity,recipe=recipe,status='running',gpu_measured=True,model=parent['model'],
            hardware=dict(parent['hardware'],pcie_query=pcie_metadata()))
        result.pop('error',None); result.pop('traceback',None); write(out/'source_manifest.json',recipe)
        plan=[]
        for name in BASELINES:
            q=dict(requests[name]); q.update(output=str(out/name),warmup=args.warmup,repeats=args.repeats)
            python=sys.executable; env=None
            if name=='splitzip_official_framed': python=str(ROOT/'runs/_llama_official/envs/splitzip/bin/python')
            if name=='exans_official_fixed_framed':
                from kv_compare.profile_recovery import runtime_environment
                spec=importlib.util.find_spec('nvidia.libnvcomp')
                if spec is None or importlib.metadata.version('nvidia-libnvcomp-cu12')!=receipt['nvcomp']: raise ValueError('pinned nvCOMP required')
                path=Path(q['codec']['library']).parent.parent/'toolchain.json'
                if digest(path)!=receipt['toolchain_sha256']: raise ValueError('native toolchain receipt changed')
                toolchain=json.loads(path.read_text()); env=runtime_environment(os.environ,toolchain['cuda_home'],[Path(s)/'lib' for s in spec.submodule_search_locations])
            plan.append((name,q,python,Path(__file__).resolve(),env))
        seen={}
        for r in parent['jobs']['lite_finish']['rows']:
            e={k:r[k] for k in ('document','packet','packet_sha256','original_word_sha256','decoded_word_sha256','shape','values','packet_bytes','complete_bpv','saved_percent','state_sha256')}
            if r['document'] in seen and seen[r['document']]!=e: raise ValueError('parent graph/eager packets differ')
            seen[r['document']]=e
        q=dict(method='lite_allocation',codec=requests['lite_finish']['codec'],entries=list(seen.values()),
            output=str(out/'lite_allocation'),warmup=args.warmup,repeats=args.repeats,phase_repeats=args.phase_repeats)
        if q['codec']['engine_options']['values_per_stream']!=64 or q['codec']['engine_options']['block']!=32:
            raise ValueError('finish experiment requires current V64/B32 layout')
        plan.append(('lite_allocation',q,sys.executable,Path(__file__).resolve(),None)); result['selected_jobs']=[x[0] for x in plan]; save()
        for name,request,python,script,env in plan:
            directory=out/name; directory.mkdir(exist_ok=True); old=result['jobs'].get(name,{})
            if old.get('status')=='completed':
                for path,wanted in old['artifacts'].items():
                    if digest(confined(out,path))!=wanted: raise ValueError('completed evidence changed')
                for r in old['rows']:
                    if digest(r['packet'])!=r['packet_sha256']: raise ValueError('completed packet changed')
                log('reuse completed method: '+name); continue
            write(directory/'request.json',request)
            launch([python,script,'--device',str(device),'--baseline-worker' if name in BASELINES else '--worker',directory/'request.json'],directory/'worker.log',env)
            data=json.loads((directory/'worker_results.json').read_text())
            if data['status']!='completed' or not data['gpu_measured'] or data['device']!=recipe['device']: raise ValueError('original GPU worker required')
            if name=='lite_allocation' and data['software']!={n:versions[n] for n in ('torch','triton','numpy')}: raise ValueError('worker software changed')
            result['jobs'][name]=dict(status='completed',rows=data['rows'],artifacts={str((directory/f).relative_to(out)):digest(directory/f) for f in ('worker_results.json','receiver.json')})
            if name=='lite_allocation':
                result['jobs'][name]['graph_setups']=data['graph_setups']
                result['jobs'][name]['allocation_policies']=data['allocation_policies']
            else:
                result['jobs'][name]['runtime_policy']=data['runtime_policy']
                result['jobs'][name]['artifacts'].update({str((directory/f).relative_to(out)):digest(directory/f)
                    for f in ('original_worker_results.json','original_receiver.json')})
            save()
        validate_comparison(result['rows'])
        for arm in ARMS:
            directory=out/'diagnostics'/arm; directory.mkdir(parents=True,exist_ok=True)
            if arm not in result['diagnostics']:
                dq=dict(q,arm=arm,output=str(directory)); write(directory/'request.json',dq)
                launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--diagnostic',directory/'request.json'],directory/'worker.log')
                data=json.loads((directory/'phases.json').read_text()); data['summary_sha256']=digest(directory/'phases.json')
                result['diagnostics'][arm]=data; save()
            elif digest(directory/'phases.json')!=result['diagnostics'][arm]['summary_sha256']: raise ValueError('phase diagnostics changed')
        if args.profile:
            for arm in ARMS:
                directory=out/'profiles'/arm; directory.mkdir(parents=True,exist_ok=True)
                if arm in result['profiles']:
                    if digest(directory/'profile.json')!=result['profiles'][arm]['summary_sha256']: raise ValueError('profile receipt changed')
                    continue
                pq=dict(q,arm=arm,output=str(directory)); write(directory/'request.json',pq)
                launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--profile-worker',directory/'request.json'],directory/'worker.log')
                data=json.loads((directory/'profile.json').read_text()); data['summary_sha256']=digest(directory/'profile.json')
                result['profiles'][arm]=data; save()
        result['status']='completed'; save(); log('completed: '+str(out/'summary.md')); return 0
    except Exception as error:
        if writable:
            result.update(status='failed',error=str(error),traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
