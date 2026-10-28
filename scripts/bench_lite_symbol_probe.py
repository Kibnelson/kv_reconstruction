#!/usr/bin/env python3
"""TRAIN-fitted BF16 rank-delta rate probe; no new codec or decoder-speed claim."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','16')
os.environ.setdefault('OPENBLAS_NUM_THREADS','16')
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_llama_causal import device_check,log,timed
from scripts.bench_lite_baselines import digest,write
from scripts.bench_lite_graph import aggregate_decode
from scripts.bench_lite_allocation import runtime_snapshot
from kv_compare.lite_allocation import allocation_policy

FILES=('scripts/bench_lite_symbol_probe.py','scripts/package_lite_symbol_probe.py',
       'scripts/run_lite_symbol_probe_a4500.sh','kv_compare/lite_symbol_probe.py',
       'tests/test_lite_symbol_probe.py','tests/test_lite_symbol_probe_runner.py')
RADII=(127,255,511,1023,2047)
CONTEXTS=(1,32)
LINKS=(1.,100.,200.,300.,400.)
BASELINES=('raw_bf16','splitzip_official_framed','exans_official_fixed_framed')
REFERENCE='lite_entropy_reference_graph'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def key(radius,contexts): return f'r{radius}_c{contexts}'


def walk_causal(parent):
    """Follow checked parent/base-result hashes until the original causal protocol."""
    directory=Path(parent).resolve(); seen=set(); chain=[]
    while True:
        if directory in seen: raise ValueError('cyclic parent lineage')
        seen.add(directory); path=directory/'results.json'; data=json.loads(path.read_text())
        if data['status']!='completed' or not data.get('gpu_measured'): raise ValueError('completed GPU lineage required')
        chain.append(dict(run=str(directory),results_sha256=digest(path)))
        if data.get('protocol') and data.get('state_directories'):
            if hashlib.sha256(canonical(data['protocol'])).hexdigest()!=data['identity']: raise ValueError('causal protocol identity differs')
            return data,chain
        recipe=data['recipe']; field='parent' if 'parent_run' in recipe else 'base'
        directory=Path(recipe[field+'_run']).resolve()
        if digest(directory/'results.json')!=recipe[field+'_results_sha256']: raise ValueError('ancestor results changed')


def candidate_header(original,radius,contexts,table_sha256):
    header=dict(original,format='causal_kv_rank_delta_v1',table_sha256=table_sha256,
                symbol_radius=radius,symbol_contexts=contexts)
    return header,24+len(canonical(header))+32


def projected_packet(cost,geometry,original_header,radius,contexts,table_sha256):
    header,framing=candidate_header(original_header,radius,contexts,table_sha256)
    bits=cost['total_bits']+8*(geometry['state_bytes']+geometry['directory_bytes']+framing)
    return dict(estimated_packet_bits=bits,estimated_packet_bytes=bits/8,
        estimated_bpv=bits/geometry['values'],estimated_saved_percent=100*(1-bits/(16*geometry['values'])),
        symbol_bits=cost['symbol_bits'],escape_raw_bits=cost['escape_bits'],
        projected_state_bytes=geometry['state_bytes'],directory_bytes=geometry['directory_bytes'],
        hypothetical_framing_bytes=framing,hypothetical_header=header,
        actual_new_packet=False,scope='ideal normalized-frequency cost plus projected rANS states, original INT64 directory and hypothetical framing; not actual rANS size or a bound')


def select_candidate(candidates):
    """Validation only; deterministic ties prefer smaller radius then fewer contexts."""
    if not candidates: raise ValueError('no validation candidates')
    return min(candidates,key=lambda r:(r['validation_total_bits'],r['radius'],r['contexts']))


def required_budgets(estimates,baselines,links=LINKS):
    rows=[]
    for estimate in estimates:
        for baseline in baselines:
            for gbps in links:
                if not math.isfinite(gbps) or gbps<=0: raise ValueError('positive finite link required')
                rows.append(dict(method=estimate['method'],baseline=baseline['method'],effective_gbps=gbps,
                    decode_budget_ms=baseline['decode_ms']+8*(baseline['packet_bytes']-estimate['estimated_packet_bytes'])/(gbps*1e6),
                    candidate_decode_measured=False,candidate_packet_estimated=True,
                    inherited_baseline_decode_ms=baseline['decode_ms'],baseline_retimed=False,
                    scope='required decoder latency to tie inherited baseline after a precompressed transfer; no candidate latency or fetch measurement'))
    return rows


def load_words(root,name,tokens,device):
    import torch
    from kv_compare.predict import Doc
    from kv_compare.causal_packet import word_sha
    doc=Doc(root,name)
    if tokens>doc.tokens: raise ValueError('capture prefix too short')
    words=torch.stack([doc.words(l,'cpu',tokens) for l in range(doc.layers)]).to(torch.int16)
    receipt=dict(document=name,tokens=tokens,shape=list(words.shape),values=words.numel(),
        original_word_sha256=word_sha(words),
        token_ids_sha256=hashlib.sha256(doc.ids[:tokens].astype('<i8').tobytes()).hexdigest())
    begin=time.perf_counter(); words=words.to(device); torch.cuda.synchronize(device)
    receipt['upload_wall_ms']=(time.perf_counter()-begin)*1000
    return words,receipt


def collect(engine,words,torch,device,selected_radius=None):
    """Predictor-only and histogram diagnostics, explicitly not a decoder benchmark."""
    from kv_compare.lite_symbol_probe import replay,symbol_histograms,symbolize,reconstruct,escape_diagnostics
    replay(engine,words,0,lambda *args:None); torch.cuda.synchronize(device)
    _,predictor=timed(lambda:replay(engine,words,0,lambda *args:None),torch,device,0,1)
    counts=torch.zeros((engine.L,32,2*max(RADII)+2),dtype=torch.int64,device=device)
    escape_counts=torch.zeros((engine.L,len(RADII),8),dtype=torch.int64,device=device)
    events=[]; callback_submit=0.; groups=0; roundtrip_checked=0
    def callback(layer,group,original,mu,scale):
        nonlocal callback_submit,groups
        begin=time.perf_counter(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
        counts[layer].add_(symbol_histograms(original,mu,scale,radii=(max(RADII),),validate=False)[max(RADII)])
        escape_counts[layer].add_(escape_diagnostics(original,mu,radii=RADII))
        b.record(); events.append((a,b)); callback_submit+=(time.perf_counter()-begin)*1000; groups+=1
    torch.cuda.synchronize(device); begin=time.perf_counter()
    replay(engine,words,0,callback); torch.cuda.synchronize(device)
    wall=(time.perf_counter()-begin)*1000; hist_cuda=sum(a.elapsed_time(b) for a,b in events)
    proof_ms=None
    if selected_radius is not None:
        def proof(layer,group,original,mu,scale):
            nonlocal roundtrip_checked
            symbols,raw=symbolize(original,mu,selected_radius)
            if not torch.equal(reconstruct(symbols,mu,raw,selected_radius),original):
                raise AssertionError('rank-delta reconstruction changed BF16 words')
            roundtrip_checked+=original.numel()
        begin=time.perf_counter(); replay(engine,words,0,proof); torch.cuda.synchronize(device)
        proof_ms=(time.perf_counter()-begin)*1000
    if int(counts.sum())!=words.numel(): raise AssertionError('histograms must cover every word of all layers')
    begin=time.perf_counter(); array=counts.cpu().numpy(); escape_array=escape_counts.cpu().numpy(); copy_ms=(time.perf_counter()-begin)*1000
    return array,escape_array,dict(gpu_measured=True,predictor_only=predictor,
        predictor_plus_histogram_wall_ms=wall,histogram_callbacks_cuda_interval_ms=hist_cuda,
        histogram_callbacks_cpu_submit_ms=callback_submit,count_copy_to_host_wall_ms=copy_ms,
        groups=groups,symbol_roundtrip_checked_values=roundtrip_checked,symbol_roundtrip_separate_replay_wall_ms=proof_ms,
        scope='known-original-KV replay; histogram/explanation CUDA intervals can include host gaps; TEST roundtrip proof is a separate pass; no entropy encode/decode or receiver latency')


def table_metadata(table):
    return {k:v for k,v in table.items() if k not in ('freq','cdf')}


def layer_insights(counts,conditional,unconditional,radius,escape_counts=None):
    import numpy as np
    from kv_compare.lite_symbol_probe import fold_counts,cost
    folded=fold_counts(counts,radius,contexts=32); output=[]
    delta=np.arange(-radius,radius+1)
    for layer,hist in enumerate(folded):
        values=int(hist.sum()); regular=hist[:,:-1].sum(0); inside=int(regular.sum())
        selected_hist=hist if conditional['freq'].shape[0]==32 else hist.sum(0,keepdims=True)
        c=cost(selected_hist,conditional); u=cost(hist.sum(0,keepdims=True),unconditional)
        output.append(dict(layer=layer,values=values,escape_percent=100*int(hist[:,-1].sum())/values,
            central_31_percent=100*int(regular[np.abs(delta)<=31].sum())/values,
            central_127_percent=100*int(regular[np.abs(delta)<=127].sum())/values,
            mean_absolute_in_radius_delta=float((regular*np.abs(delta)).sum()/inside) if inside else None,
            selected_bpv=c['bpv'],unconditional_bpv=u['bpv'],context_gain_bpv=u['bpv']-c['bpv'],
            raw_escape_bpv=c['escape_bits']/values))
        if escape_counts is not None:
            from kv_compare.lite_symbol_probe import ESCAPE_DIAGNOSTIC_LABELS
            output[-1]['escape_reasons']={name:int(value) for name,value in zip(ESCAPE_DIAGNOSTIC_LABELS,escape_counts[layer,RADII.index(radius)])}
            output[-1]['escape_reasons_scope']='sign-crossing and predicted zero/subnormal counts overlap; explanatory only'
    return output


def report(result,out):
    lines=['# Exact BF16 rank-delta symbol rate probe','',f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured',False)}**.",'',
        'All candidate rates are estimates from known-original-KV replay. This experiment does not implement new compressed packets or measure a candidate decoder. The existing coarse/2 graph decoder remains the current implementation.',
        'The fixed predictor, original BF16 captures and V64/B32 geometry are reused. Symbol frequency tables use TRAIN only. Radius/context selection uses validation only; TEST is scored after the selection receipt is written.',
        'Validation reuses the original separate scale-calibration document; the predictor and its spreads stay frozen. The 32 context bins are the fixed clamp(es−ep+7,0,31) rule using scale and BF16-prediction exponent fields. All 28 layers, including layer zero, are coded in the projection.','']
    if result.get('selection'):
        s=result['selection']; lines += [f"Selected on validation: radius **{s['radius']}**, **{s['contexts']}** contexts; table `{s['table_sha256']}`.",'',
            '| Validation candidate | Ideal symbol+escape bpv |','|---|---:|']
        for c in result['candidates']: lines.append(f"| {key(c['radius'],c['contexts'])} | {c['validation_bpv']:.5f} |")
    lines+=['','| TEST method | Estimated bpv incl. projected overhead | Estimated saved | Actual packet |','|---|---:|---:|---|']
    for r in result.get('estimates',[]): lines.append(f"| {r['method']} | {r['estimated_bpv']:.5f} | {r['estimated_saved_percent']:.2f}% | No |")
    lines+=['','Ideal costs use positive 16-bit normalized frequencies and 16 raw bits for every escape. Estimates add all B-lane rANS states, INT64 program directories and a hypothetical header carrying the new format, radius, context count and table hash. Finite-state coding/alignment overhead remains unmeasured; these estimates are neither actual packet sizes nor bounds. Shared tables are separate.','',
        '| Reused measured method | Complete bpv | Decode ms |','|---|---:|---:|']
    for b in result.get('baselines',[]): lines.append(f"| {b['method']} | {b['complete_bpv']:.5f} | {b['decode_ms']:.3f} |")
    lines+=['','Native controls and current coarse decoder above are inherited GPU measurements, **not retimed**. Their original model/data, timestamps, runtime policies and source report hash remain in the receipt. No estimated candidate time is combined with an old decoder time.']
    if result.get('table_accounting'):
        lines+=['','| Selected shared table accounting | Bytes |','|---|---:|']
        for k,v in result['table_accounting'].items():
            if isinstance(v,(int,float)): lines.append(f'| {k} | {v} |')
    if result.get('budgets'):
        lines+=['','Required candidate decoder latency to tie a baseline after precompressed movement; these are planning targets. Negative values mean the estimated candidate bytes already exceed the full baseline transfer/decode budget.','',
            '| Candidate | Baseline | Effective Gb/s | Required decode ≤ ms |','|---|---|---:|---:|']
        for r in result['budgets']: lines.append(f"| {r['method']} | {r['baseline']} | {r['effective_gbps']:g} | {r['decode_budget_ms']:.5f} |")
    if result.get('insights'):
        lines+=['','Layer patterns for the selected configuration versus the predeclared unconditional control. Positive context gain means scale/ULP bins help; the gain is zero if validation selects one context.','',
            '| Layer | Escape % | Delta within ±31 % | Context gain bpv | Raw escape bpv |','|---|---:|---:|---:|---:|']
        for r in result['insights']: lines.append(f"| {r['layer']} | {r['escape_percent']:.3f} | {r['central_31_percent']:.3f} | {r['context_gain_bpv']:.5f} | {r['raw_escape_bpv']:.5f} |")
        lines+=['','Escape explanations: sign crossings and zero/subnormal predictions can overlap. Their counts describe the fixed selected radius and never enter selection.','',
            '| Layer | Escapes | Nonfinite escapes | Sign-cross escapes | Zero/subnormal-prediction escapes | Same-sign normal escapes |','|---|---:|---:|---:|---:|---:|']
        for r in result['insights']:
            e=r['escape_reasons']; lines.append(f"| {r['layer']} | {e['escapes']} | {e['escape_nonfinite']} | {e['escape_finite_sign_crossing']} | {e['escape_finite_predicted_zero_subnormal']} | {e['escape_finite_same_sign_normal']} |")
    lines+=['','| Replay document | Role | Predictor-only wall ms | Instrumented replay+hist wall ms | Histogram + explanation CUDA intervals ms |','|---|---|---:|---:|---:|']
    for name,j in result.get('documents',{}).items():
        m=j['measurement']; lines.append(f"| {name} | {j['role']} | {m['predictor_only']['wall_median_ms']:.3f} | {m['predictor_plus_histogram_wall_ms']:.3f} | {m['histogram_callbacks_cuda_interval_ms']:.3f} |")
    lines+=['','Histogram observer CUDA intervals include observer operations and potential host gaps. TEST symbol roundtrip proofs run in an additional pass, recorded separately in JSON. Replay timings use known original KV and instrumentation. They do not measure entropy coding, GPU receiver decoding, network/SSD, compressed attention or serving. Source verification, uploads, cold initialization and host table fitting are recorded separately. No predictor fitting, model download or native build is performed.']
    if result.get('error'): lines+=['',f"Error: {result['error']}"]
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run',type=Path,default=ROOT/'runs/lite_entropy_a4500_20261028_110019_1125591')
    parser.add_argument('--source-run',type=Path,default=ROOT/'runs/llama32_3b_compare')
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args(argv); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=dict(format='lite_symbol_probe_v1',status='running',gpu_measured=False,documents={})
    writable=not (out/'results.json').exists()
    def save(): write(out/'results.json',result); report(result,out)
    try:
        torch,device=device_check(args.device)
        from scripts.package_lite_entropy import verify as verify_parent
        from kv_compare.causal_study import source_inputs
        from kv_compare.causal_packet import Packet,word_sha
        from kv_compare.lite_entropy import EntropyEngine
        from kv_compare.lite_symbol_probe import fold_counts,fit_tables,cost
        import numpy as np
        parent_dir=args.parent_run.resolve(); parent=json.loads((parent_dir/'results.json').read_text()); verify_parent(parent_dir,parent)
        causal,chain=walk_causal(parent_dir); protocol=causal['protocol']
        inherited={**parent['recipe']['unchanged_code'],**parent['recipe']['code']}
        for name,wanted in inherited.items():
            if digest(ROOT/name)!=wanted: raise ValueError('frozen source changed: '+name)
        software={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (software!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']): raise ValueError('original GPU/runtime required')
        begin=time.perf_counter(); root,train,validation,tests,source=source_inputs(args.source_run,out,len(protocol['train']),log)
        verify_ms=(time.perf_counter()-begin)*1000
        if source!=protocol['source'] or train!=protocol['train']: raise ValueError('original causal source/TRAIN identity differs')
        request=parent_dir/'lite_entropy/request.json'; q=json.loads(request.read_text()); codec=q['codec']
        if Path(codec['state']).resolve()!=Path(causal['state_directories']['lite']).resolve(): raise ValueError('original Lite fitted state differs')
        entries={e['document']:e for e in q['entries']}
        if set(entries)!=set(tests): raise ValueError('original TEST packet coverage differs')
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_request_sha256=digest(request),ancestor_chain=chain,source_run=str(args.source_run.resolve()),source=source,
            original_protocol=protocol,codec=codec,code={n:digest(ROOT/n) for n in FILES},unchanged_code=inherited,
            software=software,device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,
            radii=list(RADII),contexts=list(CONTEXTS),links=list(LINKS),
            selection='minimum validation total ideal symbol+escape bits; ties smaller radius, then fewer contexts',
            test_controls='selected configuration and predeclared unconditional control at selected radius only')
        identity=hashlib.sha256(canonical(recipe)).hexdigest()
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=identity: raise ValueError('resume identity differs; choose fresh output')
            result=old; writable=True
        result.update(identity=identity,recipe=recipe,status='running',gpu_measured=True,source_verification_wall_ms=verify_ms,
            model=parent['model'],hardware=parent['hardware']); result.pop('error',None); result.pop('traceback',None)
        baselines=aggregate_decode([r for r in parent['rows'] if r['method'] in (*BASELINES,REFERENCE)])
        if {b['method'] for b in baselines}!=set((*BASELINES,REFERENCE)): raise ValueError('inherited measured control coverage differs')
        result['baselines']=baselines
        result['baseline_receipt']=dict(source_run=str(parent_dir),results_sha256=digest(parent_dir/'results.json'),
            measurement_timestamp=parent_dir.name,retimed=False,methods=[b['method'] for b in baselines],
            runtime_policies={n:parent['jobs'][n]['runtime_policy'] for n in BASELINES})
        result['runtime_policy_before']=runtime_snapshot(torch)
        with allocation_policy(False) as policy:
            begin=time.perf_counter(); engine=EntropyEngine(codec['state'],device,**codec['engine_options'],entropy_lookup='coarse',entropy_refine=2)
            torch.cuda.synchronize(device); result['cold_engine_wall_ms']=(time.perf_counter()-begin)*1000
            result['allocation_policy']=policy; result['state_sha256']=engine.state_sha256
            if any(e['state_sha256']!=engine.state_sha256 for e in entries.values()): raise ValueError('parent packet fitted state differs')
            if engine.V!=64 or engine.B!=32: raise ValueError('fixed V64/B32 required')
            write(out/'source_manifest.json',recipe); save()
            def document(name,role,tokens,radius=None):
                words,prefix=load_words(root,name,tokens,device); path=out/'counts'/f'{name}.npz'; path.parent.mkdir(exist_ok=True)
                previous=result['documents'].get(name)
                if previous:
                    if previous['prefix']!=prefix and {k:v for k,v in previous['prefix'].items() if k!='upload_wall_ms'}!={k:v for k,v in prefix.items() if k!='upload_wall_ms'}:
                        raise ValueError('resumed source prefix changed')
                    if previous['role']!=role or digest(path)!=previous['counts_sha256']: raise ValueError('resumed count evidence changed')
                    with np.load(path,allow_pickle=False) as saved:
                        counts=saved['counts']; escape_counts=saved['escape_counts']
                    if role=='TEST' and (previous.get('selected_radius')!=radius or previous.get('selection_sha256')!=result.get('selection_sha256')):
                        raise ValueError('TEST checkpoint selection changed')
                    return counts,escape_counts
                proof=None
                if role=='TEST':
                    e=entries[name]
                    if prefix['original_word_sha256']!=e['original_word_sha256'] or prefix['shape']!=e['shape'] or digest(e['packet'])!=e['packet_sha256']:
                        raise ValueError('original TEST prefix/packet changed')
                    packet=Packet.restore(Path(e['packet']).read_bytes(),device)
                    decoded=engine.decode(packet)
                    if word_sha(decoded)!=e['original_word_sha256']: raise AssertionError('original coarse packet reconstruction differs')
                    proof=dict(packet=e['packet'],packet_sha256=e['packet_sha256'],original_word_sha256=e['original_word_sha256'],
                        state_sha256=e['state_sha256'],original_packet_decode_exact=True,new_symbol_packet_exists=False)
                    del decoded,packet
                log(f'known-KV rate replay: {role}/{name}')
                counts,escape_counts,measurement=collect(engine,words,torch,device,radius); tmp=path.with_suffix('.part.npz')
                np.savez_compressed(tmp,counts=counts,escape_counts=escape_counts); tmp.replace(path)
                result['documents'][name]=dict(role=role,prefix=prefix,counts=str(path.relative_to(out)),counts_sha256=digest(path),
                    counts_shape=list(counts.shape),counts_values=int(counts.sum()),measurement=measurement,
                    original_packet_proof=proof,selected_radius=radius,selection_sha256=result.get('selection_sha256') if role=='TEST' else None); save()
                return counts,escape_counts
            train_counts=None
            for name in train:
                c,_=document(name,'TRAIN',protocol['fit_tokens']); train_counts=c.copy() if train_counts is None else train_counts+c
            validation_counts,_=document(validation,'validation',protocol['scale_tokens'])
            tables={}; candidates=[]; arrays={}; begin=time.perf_counter()
            for radius in RADII:
                for contexts in CONTEXTS:
                    ident=key(radius,contexts); table=fit_tables(fold_counts(train_counts.sum(0),radius,contexts=contexts)); tables[ident]=table
                    arrays[ident+'_freq']=table['freq']; arrays[ident+'_cdf']=table['cdf']
                    v=cost(fold_counts(validation_counts.sum(0),radius,contexts=contexts),table)
                    t=cost(fold_counts(train_counts.sum(0),radius,contexts=contexts),table)
                    candidates.append(dict(radius=radius,contexts=contexts,table_sha256=table['table_sha256'],
                        validation_total_bits=v['total_bits'],validation_bpv=v['bpv'],train_bpv=t['bpv'],table=table_metadata(table)))
            result['host_table_fit_wall_ms']=(time.perf_counter()-begin)*1000
            table_path=out/'train_frequency_tables.npz'
            recorded_hash=result.get('train_table_sha256')
            if not recorded_hash and (out/'selection.json').exists():
                recorded_hash=json.loads((out/'selection.json').read_text())['table_file_sha256']
            if table_path.exists():
                if recorded_hash and digest(table_path)!=recorded_hash: raise ValueError('saved TRAIN table archive changed')
                with np.load(table_path,allow_pickle=False) as saved:
                    if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[n],a) for n,a in arrays.items()):
                        raise ValueError('saved frequency arrays differ from freshly derived TRAIN-only tables')
            else:
                if recorded_hash: raise ValueError('recorded TRAIN table archive is missing')
                temporary=table_path.with_suffix('.part.npz'); np.savez_compressed(temporary,**arrays); temporary.replace(table_path)
            selected=select_candidate(candidates); selection=dict(selected,train_documents=train,validation_document=validation,
                table_file_sha256=digest(table_path),test_not_scored_for_selection=True,tie_break=recipe['selection'])
            selection_path=out/'selection.json'
            if selection_path.exists() and json.loads(selection_path.read_text())!=selection: raise ValueError('selection receipt changed')
            write(selection_path,selection); result.update(candidates=candidates,selection=selection,selection_sha256=digest(selection_path),
                train_table_sha256=digest(table_path)); save()
            radius,contexts=selected['radius'],selected['contexts']; test_counts=None; test_escapes=None; rows=[]
            selected_table=tables[key(radius,contexts)]; unconditional=tables[key(radius,1)]
            for name in tests:
                c,esc=document(name,'TEST',protocol['tokens'],radius); test_counts=c.copy() if test_counts is None else test_counts+c
                test_escapes=esc.copy() if test_escapes is None else test_escapes+esc
                e=entries[name]; packet=Packet.restore(Path(e['packet']).read_bytes(),'cpu'); descriptions,_,programs=engine.layout(protocol['tokens'])
                geometry=dict(values=e['values'],state_bytes=4*engine.B*programs,directory_bytes=8*programs)
                for label,ct in (('selected',contexts),('unconditional_control',1)):
                    table=tables[key(radius,ct)]; value=cost(fold_counts(c.sum(0),radius,contexts=ct),table)
                    rows.append(dict(document=name,method='rank_delta_'+label,radius=radius,contexts=ct,
                        table_sha256=table['table_sha256'],values=e['values'],cost=value,
                        **projected_packet(value,geometry,packet.header,radius,ct,table['table_sha256'])))
            estimates=[]
            for method in ('rank_delta_selected','rank_delta_unconditional_control'):
                group=[r for r in rows if r['method']==method]
                estimates.append(dict(method=method,estimated_packet_bytes=statistics.mean(r['estimated_packet_bytes'] for r in group),
                    estimated_bpv=statistics.mean(r['estimated_bpv'] for r in group),
                    estimated_saved_percent=statistics.mean(r['estimated_saved_percent'] for r in group),actual_new_packet=False))
            result.update(rows=rows,estimates=estimates,budgets=required_budgets(estimates,[b for b in baselines if b['method'] in BASELINES]),
                insights=layer_insights(test_counts,selected_table,unconditional,radius,test_escapes),
                table_accounting=dict(actual_frequency_array_bytes=int(selected_table['freq'].nbytes),
                    actual_cdf_array_bytes=int(selected_table['cdf'].nbytes),actual_table_arrays_bytes=int(selected_table['freq'].nbytes+selected_table['cdf'].nbytes),
                    all_candidate_npz_bytes=table_path.stat().st_size,
                    projected_gpu_frequency_bytes=int(selected_table['freq'].nbytes),projected_gpu_cdf_bytes=int(selected_table['cdf'].nbytes),
                    projected_gpu_inverse_uint16_bytes=contexts*65536*2,
                    projected_gpu_total_table_bytes=int(selected_table['freq'].nbytes+selected_table['cdf'].nbytes)+contexts*65536*2,
                    predictor_gpu_bytes=engine.shared_bytes,scope='selected physical frequency/CDF arrays; projected GPU frequency+CDF+uint16 inverse layout; NPZ container holds all candidate arrays; inverse table not allocated'))
        result['runtime_policy_after']=runtime_snapshot(torch)
        if result['runtime_policy_before']!=result['runtime_policy_after']: raise AssertionError('allocation policy did not restore runtime flags')
        result['status']='completed'; save(); log('completed: '+str(out/'summary.md')); return 0
    except Exception as error:
        if writable: result.update(status='failed',error=str(error),traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__=='__main__': raise SystemExit(main())
