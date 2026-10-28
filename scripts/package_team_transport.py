#!/usr/bin/env python3
"""Verify frozen packet transport receipts; package small evidence, not traces."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_team_transport import (ARMS,METHODS,LENGTHS,METRICS,aggregate,arm_order,
    identity,link_models,read_parent,valid_receiver_request)
from scripts.bench_lite_baselines import digest


def checked(root,name):
    raw=root/name;path=raw.resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or raw.is_symlink():
        raise ValueError('missing/outside-run/symlink receipt: '+str(name))
    return path


def check_samples(record,warmup,repeats):
    if (len(record['samples'])!=repeats or len(record['warmup_samples'])!=warmup or
            any(s['wall_ms']<=0 or s['cuda_interval_ms']<0 for s in record['samples']) or
            record['wall_median_ms']!=statistics.median(s['wall_ms'] for s in record['samples'])):
        raise ValueError('actual CUDA sample count/median differs')


def verify(run,result):
    run=Path(run).resolve()
    if result.get('format')!='team_transport_v1':raise ValueError('wrong transport report format')
    completed=result['status']=='completed'
    if completed and not result.get('gpu_measured'):raise ValueError('CPU evidence cannot be a completed GPU transport run')
    selected={};parent=None
    if result.get('recipe'):
        recipe=result['recipe']
        if identity(recipe)!=result.get('identity'):raise ValueError('transport recipe identity differs')
        parent_dir=Path(recipe['parent_run']).resolve()
        if digest(parent_dir/'results.json')!=recipe['parent_results_sha256']:raise ValueError('parent report changed')
        parent,inherited,plan,receipts=read_parent(parent_dir)
        selected=dict(plan)
        if (recipe['unchanged_code']!=inherited or recipe['parent_artifacts']!=receipts or
                recipe['selections']!=selected or recipe['methods']!=list(METHODS) or
                recipe['lengths']!=list(LENGTHS) or recipe['arms']!=list(ARMS)):
            raise ValueError('frozen packet selections changed')
        for name,wanted in {**recipe['unchanged_code'],**recipe['code']}.items():
            if digest(ROOT/name)!=wanted:raise ValueError('measured source changed: '+name)
        if json.loads(checked(run,'source_manifest.json').read_text())!=recipe:raise ValueError('source receipt differs')
        anchors=[dict(x,retimed=False,scope='reused earlier GPU-resident decode; unmatched host I/O') for x in parent['aggregates'] if x['method'] in
            ('raw_bf16','lite_current_graph','splitzip_official_framed','exans_official_fixed_framed')]
        if result['baseline_anchors']!=anchors:raise ValueError('reused baseline anchors changed')
        if completed and (set(result['selected_jobs'])!=set(selected) or set(result['jobs'])!=set(selected)):
            raise ValueError('selected transport job coverage differs')
    elif completed:raise ValueError('completed run has no recipe')
    rows=[]
    for name,job in result.get('jobs',{}).items():
        if job.get('status')!='completed' or name not in selected:raise ValueError('unexpected/incomplete job')
        for path,wanted in job['artifacts'].items():
            if digest(checked(run,path))!=wanted:raise ValueError('worker artifact changed')
        q=json.loads(checked(run,name+'/request.json').read_text())
        expected=dict(selected[name],output=str(run/name),warmup=recipe['warmup'],repeats=recipe['repeats'])
        if q!=expected:raise ValueError('worker request differs from selected parent packets')
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json');proof=json.loads(proof_path.read_text())
        rq=json.loads(checked(run,name+'/receiver_request.json').read_text());valid_receiver_request(rq)
        if rq!=dict(codec=q['codec'],entries=q['entries'],arms=list(ARMS),output=str(run/name/'receiver.json')):
            raise ValueError('fresh receiver request differs')
        checks={c['id']:c for c in proof['checks']}
        entries={e['id']:e for e in q['entries']}
        expected_ids={arm+'_'+e['id'] for arm in ARMS for e in q['entries']}
        if (data['rows']!=job['rows'] or data['codec_provenance']!=q['codec'] or not data.get('gpu_measured') or
                data['status']!='completed' or not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
                not proof.get('alternating_packets') or not proof.get('independent_returned_outputs') or
                proof.get('graph_capture_count')!=1 or data['graph_capture_count']!=1 or
                set(checks)!=expected_ids or len(checks)!=len(proof['checks']) or
                {r['id'] for r in job['rows']}!=expected_ids or len(job['rows'])!=len(expected_ids) or
                data['receiver_sha256']!=digest(proof_path)):
            raise ValueError('independent actual-GPU receiver coverage differs')
        if (data['runtime_policy']!=data['runtime_policy_after'] or proof['runtime_policy']!=data['runtime_policy'] or
                proof['runtime_policy_after']!=data['runtime_policy'] or data['runtime_policy']['deterministic_algorithms_enabled']):
            raise ValueError('unchanged team runtime policy differs')
        if len(data['sample_order'])!=recipe['warmup']+recipe['repeats']:
            raise ValueError('paired round coverage differs')
        for i,order in enumerate(data['sample_order']):
            index=i if i<recipe['warmup'] else i-recipe['warmup']
            docs=(0,1) if i%2==0 else (1,0)
            if order!=dict(arms=list(arm_order(index)),documents=[q['entries'][j]['document'] for j in docs]):
                raise ValueError('balanced arm/document schedule differs')
        for row in job['rows']:
            e=entries.get(row['parent_id']);c=checks[row['id']]
            if (e is None or row['id']!=row['arm']+'_'+e['id'] or row['arm'] not in ARMS or
                    row['method']!=q['method'] or row['tokens']!=q['tokens'] or row['status']!='verified' or
                    not row['original_exact'] or row['decoded_word_sha256']!=e['original_word_sha256'] or
                    any(row[k]!=e[k] for k in ('document','shape','values','packet','packet_sha256','packet_bytes','original_word_sha256','complete_bpv','state_sha256')) or
                    not c['verified'] or c['word_sha256']!=e['original_word_sha256'] or
                    c['packet_sha256']!=e['packet_sha256'] or c['shape']!=e['shape']):
                raise ValueError('wire/original BF16 exactness differs')
            for metric in (*METRICS,'gpu_resident_decode'):check_samples(row[metric],recipe['warmup'],recipe['repeats'])
            rows.append(row)
    if rows!=result.get('rows',[]) or aggregate(rows)!=result.get('aggregates') or link_models(result.get('aggregates',[]))!=result.get('link_models'):
        raise ValueError('derived paired comparison differs')
    profiles=result.get('profiles',{})
    expected_profiles={name for name,q in selected.items() if q['tokens']==8192} if result.get('recipe',{}).get('profile') else set()
    if completed and set(profiles)!=expected_profiles:raise ValueError('separate profile coverage differs')
    for name,profile in profiles.items():
        path=checked(run,'profiles/'+name+'/profile.json')
        if (digest(path)!=profile['summary_sha256'] or json.loads(path.read_text())!={k:v for k,v in profile.items() if k!='summary_sha256'} or
                profile['status']!='completed' or not profile.get('gpu_measured') or
                profile['codec_provenance']!=selected[name]['codec'] or set(profile['diagnostics'])!=set(ARMS)):
            raise ValueError('separate GPU profile receipt differs')
        trace=Path(profile['trace_path']).resolve()
        if not trace.is_relative_to(run) or not trace.is_file() or digest(trace)!=profile['trace_sha256'] or trace.stat().st_size!=profile['trace_bytes']:
            raise ValueError('local full profiler trace changed')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args(argv);run=a.run.resolve();dest=a.destination
    if dest.exists():raise ValueError('destination exists')
    result=json.loads((run/'results.json').read_text());verify(run,result)
    dest.mkdir(parents=True)
    names=['results.json','summary.md','source_manifest.json','console.log','cuda_tests.log']
    names += [str(p.relative_to(run)) for name in result.get('jobs',{}) for p in (run/name).iterdir() if p.suffix in ('.json','.log')]
    names += [str(p.relative_to(run)) for name in result.get('profiles',{}) for p in (run/'profiles'/name).iterdir()
              if p.suffix in ('.json','.log') and p.name!='trace.json']
    manifest=dict(format='team_transport_small_report_v1',source_run=str(run),status=result['status'],files={},
        excluded='parent packets and fitted state are referenced, not copied; no raw KV, input arrays or full profiler traces are packaged')
    for name in sorted(set(names)):
        src=run/name
        if not src.is_file():continue
        checked(run,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
        manifest['files'][name]=dict(bytes=target.stat().st_size,sha256=digest(target))
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md');return 0


if __name__=='__main__':raise SystemExit(main())
