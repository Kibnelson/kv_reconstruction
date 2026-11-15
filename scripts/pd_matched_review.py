#!/usr/bin/env python3
"""Report missing methods and failed requests separately from campaign closeout."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shlex
import statistics

OURS={'t-fast','t-dense-cached','t-compact','t-compact-global'}


def audit(ev):
    ev=Path(ev)
    load=lambda p:json.loads(p.read_text())
    campaign=load(ev/'campaign.json')
    run=load(ev/'run.json')
    issues=[];skipped={};rows=[];rounds=set();scopes={};identities={}
    readiness=list((ev/'setup').glob('*pd_matched_checks.json'))
    ready=bool(readiness) and all(load(p).get('ok') and load(p).get('gpu_executed') and
        len(load(p).get('fresh_receivers',[]))==4 for p in readiness)
    if not ready:issues.append('Compiled GPU / independent receiver readiness evidence is missing or failed.')
    specs=load(ev/'node_specs'/'specs.json') if (ev/'node_specs'/'specs.json').is_file() else {}
    bw=specs.get('bandwidth',{})
    patterns=bw.get('patterns',[])
    survey=bool(run.get('specs_exit')==0 and specs.get('system') and patterns and
        all(p.get('correct') for p in patterns) and not specs.get('bandwidth_error') and
        not bw.get('diagnostic_only') and not bw.get('probe',{}).get('diagnostic_only'))
    if not survey:issues.append('Same-allocation hardware and bandwidth survey is missing or incomplete.')
    matrices=list((ev/'matrix').glob('pd_matrix_*/matrix.json'))
    matrix=load(matrices[-1]) if matrices else {}
    if matrix.get('runs')!=campaign.get('runs'):
        issues.append('Matrix did not execute the exact run list frozen at create.')
    if matrix.get('failed') or run.get('status')!='completed':
        issues.append('One or more setup or matrix runs failed.')
    actual=[]
    for folder in matrix.get('folders',[]):
        p=ev/'matrix'/folder
        if folder.startswith('pd_matrix_'):continue
        if not (p/'results.json').is_file():
            issues.append('Missing results: '+folder);continue
        result=load(p/'results.json')
        env=load(p/'env.json') if (p/'env.json').is_file() else result.get('env',{})
        actual.append((folder,env,result))
        parameters=result.get('benchmark_parameters',{})
        if not parameters:issues.append('Missing declared benchmark point grid: '+folder)
        skipped[folder]=result.get('skipped',{})
        if result.get('skipped'):
            issues.append('Requested codecs skipped in '+folder+': '+','.join(result['skipped']))
        rr=int(env.get('matched_round',-1));rounds.add(rr)
        if not env.get('model_revision') or not env.get('prompt_sha256'):
            issues.append('Missing model/prompt identity: '+folder)
        identity=tuple(json.dumps(env.get(k),sort_keys=True) for k in ('model','model_revision','vllm','torch','prompt_sha256','eval_text_sha256','memory_profiler_estimate_cudagraphs'))
        cohort=(env.get('layout'),env.get('transport'),bool(env.get('eager')),env.get('connector','tkv'),env.get('engine_group'))
        if cohort in identities and identities[cohort]!=identity:
            issues.append('Model, runtime or prompt identity changed within a comparison: '+folder)
        identities[cohort]=identity
        if env.get('layout')!='colo4' and not env.get('fresh_PD_prefix_cache_disabled'):
            issues.append('Fresh PD prefix-cache policy was not recorded: '+folder)
        for codec,c in result.get('check',{}).items():
            st=c.get('transfers',{})
            if not c.get('ok') or (codec!='nixl' and (not c.get('prompts') or st.get('exact')!=c['prompts'] or st.get('install_errors') or st.get('inexact'))):
                issues.append('Correctness failed: '+folder+'/'+codec)
        for name,s in result.get('sweep',{}).items():
            if not isinstance(s,dict):continue
            codec=s.get('codec');b=s.get('bench',{});st=s.get('transfers',{})
            requested=s.get('requested_num_prompts',0)
            ok=s.get('exit')==0 and requested>0 and b.get('completed')==requested and b.get('failed') in (0,None)
            for metric in ('request_throughput','request_goodput','median_ttft_ms','p99_ttft_ms'):
                v=b.get(metric)
                ok=ok and isinstance(v,(int,float)) and math.isfinite(v) and v>=0
            if codec!='nixl':
                ok=ok and st.get('installs',0)>=b.get('completed',0) and not any(st.get(k) for k in ('send_errors','install_errors','timeouts','inexact'))
            if codec in OURS and not result.get('check',{}).get(codec,{}).get('ok'):ok=False
            if not ok:issues.append('Incomplete sweep: '+folder+'/'+name)
            scope=(env.get('layout'),env.get('transport'),bool(env.get('eager')),str(s.get('link_gbps')),str(s.get('rate')),env.get('engine_group'))
            scopes.setdefault(scope,{}).setdefault(rr,set()).add(codec)
            rows.append(dict(folder=folder,codec=codec,round=rr,ok=bool(ok),scope=list(scope),bench=b,
                payload_bpv=st.get('bits'),codec_frame_bpv=st.get('codec_frame_bits')))
        for name,s in result.get('tier',{}).items():
            if not isinstance(s,dict) or 'codec' not in s:continue
            st=s.get('tier',{})
            ok=s.get('exit')==0 and s.get('bench',{}).get('requests',0)>0 and not st.get('load_errors')
            if not ok:
                issues.append('Incomplete tier run: '+folder+'/'+name)
            if s.get('codec') in OURS and not st.get('chunks_loaded'):
                issues.append('No actual tier retrieval for '+folder+'/'+name)
                ok=False
            if any(isinstance(x,dict) and x.get('error') for x in s.get('resets',[])):
                issues.append('Prefix cache reset failed: '+folder+'/'+name)
                ok=False
            scope=(env.get('layout'),env.get('transport'),bool(env.get('eager')),'tier','clients='+str(s.get('clients')),env.get('engine_group'))
            scopes.setdefault(scope,{}).setdefault(rr,set()).add(s['codec'])
            rows.append(dict(folder=folder,codec=s['codec'],round=rr,ok=bool(ok),scope=list(scope),
                bench=s.get('bench',{}),payload_bpv=st.get('bits'),codec_frame_bpv=None))
        if env.get('layout')!='colo4':
            expected={(c,str(l),str(rate)) for c in parameters.get('codecs',[]) for l in parameters.get('links',[]) for rate in parameters.get('rates',[])}
            observed={(s.get('codec'),str(s.get('link_gbps')),str(s.get('rate'))) for s in result.get('sweep',{}).values() if isinstance(s,dict)}
        else:
            expected={(c,int(n)) for c in parameters.get('tier_codecs',[]) for n in parameters.get('tier_clients',[])}
            observed={(s.get('codec'),int(s.get('clients',0))) for s in result.get('tier',{}).values() if isinstance(s,dict)}
        if expected-observed:issues.append('Declared method/load points missing: '+folder+': '+str(sorted(expected-observed)))
    for scope,by_round in scopes.items():
        if not any(OURS & cs for cs in by_round.values()):continue
        for rr,cs in by_round.items():
            if not OURS<=cs:issues.append('A required matched T method is absent at '+str((scope,rr)))
            if 'raw' not in cs:issues.append('Raw control is absent at '+str((scope,rr)))
    declared=campaign.get('runs',[])
    for spec in declared:
        kv={k:v for k,v in campaign.get('benchmark_overrides',{}).items() if v is not None}
        kv.update(x.split('=',1) for x in shlex.split(spec) if '=' in x)
        expected=OURS & set((kv.get('CODECS') or kv.get('TIER_CODECS') or '').split(','))
        if not expected:continue
        candidates=[(folder,e,r) for folder,e,r in actual if e.get('layout')==kv.get('LAYOUT','1p3d') and
            e.get('connector','tkv')==kv.get('CONNECTOR','tkv') and
            e.get('transport','ring')==kv.get('TRANSPORT','ring') and
            bool(e.get('eager'))==(kv.get('ENGINE_EAGER','0')=='1') and
            e.get('matched_round')==int(kv.get('MATCHED_ROUND','0')) and
            expected<=set(e.get('requested_codecs',[]))]
        if len(candidates)!=1:
            issues.append('Expected exactly one results folder for '+spec);continue
        folder,e,r=candidates[0]
        observed={v.get('codec') for stage in ('sweep','tier') for v in r.get(stage,{}).values() if isinstance(v,dict)}
        if not expected<=observed:
            issues.append('Required methods lack timed serving rows: '+folder+': '+','.join(sorted(expected-observed)))
    if not any(any(n in spec for n in OURS) for spec in declared):
        issues.append('Campaign declares no matched T methods.')
    issues=list(dict.fromkeys(issues))
    full=campaign.get('matrix')!='smoke' and len(rounds)>=3 and min(rounds,default=-1)>=0 and bool(scopes) and all(len(r)>=3 for r in scopes.values())
    return dict(format='pd_matched_campaign_review_v1',baseline_source_pin='313bc6de96c750a4a297a1a2e7610b9ac464cd90',
        readiness_passed=bool(ready),hardware_bandwidth_survey_passed=survey,declared_campaign_coverage_passed=not issues,
        full_repeated_comparison_validated=bool(full and not issues),smoke_only=campaign.get('matrix')=='smoke',
        rounds=sorted(rounds),sweep_rows=rows,skipped=skipped,issues=issues,
        ring_network_emulated=True,real_NIC_network_measured=False,automatic_merge=False)


def save_review(ev):
    report=audit(ev)
    (Path(ev)/'review'/'matched_comparison.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    groups={}
    for row in report['sweep_rows']:
        if row['ok']:groups.setdefault(tuple(row['scope']),{}).setdefault(row['codec'],[]).append(row)
    lines=['# Matched PD comparison','',
        'Declared coverage passed: '+str(report['declared_campaign_coverage_passed'])+'. Full repeated comparison validated: '+str(report['full_repeated_comparison_validated'])+'.','',
        'Times below are medians of the available per-repeat metrics. Median TTFT p50 means the median of per-repeat p50 values, not a pooled request percentile. Different engine groups remain separate. Smoke is a functional pilot.','']
    if report['issues']:lines+=['Coverage issues:']+['- '+x for x in report['issues']]+['']
    for scope,methods in sorted(groups.items(),key=lambda x:str(x[0])):
        lines+=['## '+str(scope),'', '| Codec | Repeats | Throughput req/s | Goodput req/s | TTFT p50 ms | TTFT p99 ms | Payload bpv | Codec-frame bpv |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for codec,rr in sorted(methods.items()):
            def median(key,bench=True):
                values=[(r['bench'] if bench else r).get(key) for r in rr]
                values=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
                return '-' if not values else f'{statistics.median(values):.3f}'
            lines.append('| '+codec+' | '+str(len({r['round'] for r in rr}))+' | '+' | '.join([
                median('request_throughput'),median('request_goodput'),median('median_ttft_ms'),median('p99_ttft_ms'),median('payload_bpv',False),median('codec_frame_bpv',False)])+' |')
        lines.append('')
    lines+=['Ring bandwidth is paced per prefill link; 2P has two such links. This is not a measured NIC experiment. Codec-frame bytes include alignment and JSON metadata; transport control handshakes remain separate. Native NIXL has no custom codec-frame/checksum counters.','']
    (Path(ev)/'review'/'matched_comparison.md').write_text('\n'.join(lines))
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    a=p.parse_args()
    r=save_review(a.campaign)
    print(json.dumps({k:r[k] for k in ('readiness_passed','declared_campaign_coverage_passed','full_repeated_comparison_validated','issues')},sort_keys=True))
