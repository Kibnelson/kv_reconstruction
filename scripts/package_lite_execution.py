#!/usr/bin/env python3
"""Publish small checked execution evidence; keep full traces/packets local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_lite_baselines import checked, sha
from kv_compare.lite_scoreboard import aggregate, network_model, validate_comparison


def verify(run,result):
    if result['status']=='completed':
        if not result.get('gpu_measured'): raise ValueError('CPU evidence cannot be a completed GPU benchmark')
        if not result.get('selected_jobs') or set(result['jobs'])!=set(result['selected_jobs']):
            raise ValueError('completed job coverage differs')
        if result['recipe']['profiles'] and set(result['profiles'])!={
                'lite_b8_ref_before','lite_b8_fused','lite_b32_ref_before','lite_b32_fused'}:
            raise ValueError('profile coverage differs')
        recipe=result['recipe']
        if hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()!=result['identity']:
            raise ValueError('recipe digest differs')
        rows=[r for job in result['jobs'].values() if job.get('status')=='completed' for r in job['rows']]
        if rows!=result['rows']: raise ValueError('completed worker rows differ')
        validate_comparison(rows)
        if aggregate(rows)!=result['aggregates'] or network_model(result['aggregates'])!=result['network_model']:
            raise ValueError('derived comparison differs')
        parent_path=Path(recipe['parent_run'])/'results.json'
        if sha(parent_path)!=recipe['parent_results_sha256']: raise ValueError('parent evidence changed')
        parent=json.loads(parent_path.read_text())
        expected={r['packet_sha256']:r for r in parent['rows'] if r['method'] in ('lite_b8','lite_b32')}
        for r in rows:
            if r['method'].startswith('lite_'):
                e=expected.get(r['packet_sha256'])
                if e is None or any(r[k]!=e[k] for k in ('document','kappa','shape','decoded_word_sha256','packet_bytes')):
                    raise ValueError('Lite row differs from original checked packet')
    for name,job in result.get('jobs',{}).items():
        if job.get('status')!='completed': raise ValueError('unfinished job recorded')
        for path,wanted in job['artifacts'].items():
            if sha(checked(run,path))!=wanted: raise ValueError('worker evidence changed')
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json'); proof=json.loads(proof_path.read_text())
        checks={c['id']:c for c in proof['checks']}
        if (not data.get('gpu_measured') or not proof.get('gpu_measured') or
                not proof.get('no_source_kv_or_predictions') or data['rows']!=job['rows'] or
                sha(proof_path)!=data['receiver_sha256'] or set(checks)!={r['id'] for r in job['rows']} or
                len(checks)!=len(job['rows'])): raise ValueError('independent CUDA replay differs')
        parent_dir=Path(result['recipe']['parent_run'])
        for r in job['rows']:
            p=Path(r['packet'])
            if (p.is_symlink() or not (p.resolve().is_relative_to(run) or p.resolve().is_relative_to(parent_dir)) or
                    sha(p)!=r['packet_sha256'] or p.stat().st_size!=r['packet_bytes']):
                raise ValueError('packet changed/outside checked runs')
            c=checks[r['id']]
            if (not c['verified'] or c['packet_sha256']!=r['packet_sha256'] or
                    c['word_sha256']!=r['decoded_word_sha256'] or c['shape']!=r['shape']):
                raise ValueError('receiver words differ')
            if name.startswith('lite_') and not r['packet_matches_original']: raise ValueError('Lite packet changed')
            if r['mode']=='lossless' and r['decoded_word_sha256']!=r['original_word_sha256']:
                raise ValueError('lossless original words differ')
            if abs(r['complete_bpv']-8*r['packet_bytes']/r['values'])>1e-10:
                raise ValueError('packet byte accounting differs')
    for name,data in result.get('profiles',{}).items():
        path=checked(run,'profiles/'+name+'/profile.json')
        if sha(path)!=data['summary_sha256'] or not data.get('gpu_measured') or data['status']!='completed':
            raise ValueError('GPU profile evidence differs')
        if {k:v for k,v in data.items() if k!='summary_sha256'}!=json.loads(path.read_text()):
            raise ValueError('root profile summary differs from diagnostic evidence')
        for p in data['profiles']:
            trace=Path(p['trace'])
            if (trace.is_symlink() or not trace.resolve().is_relative_to(run) or
                    sha(trace)!=p['trace_sha256']): raise ValueError('diagnostic trace changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path); p.add_argument('--destination',required=True,type=Path)
    args=p.parse_args(argv); run=args.run.resolve(); dest=args.destination
    if dest.exists(): raise ValueError('fresh report destination required')
    result=json.loads(checked(run,'results.json').read_text()); verify(run,result)
    names=[n for n in ('results.json','summary.md','source_manifest.json','console.log') if (run/n).exists()]
    for name in result.get('jobs',{}):
        names += [name+'/'+f for f in ('worker_results.json','receiver.json','receiver.log','worker.log') if (run/name/f).exists()]
    for name in result.get('profiles',{}): names += ['profiles/'+name+'/'+f for f in ('profile.json','worker.log')]
    sources={n:checked(run,n) for n in names}; dest.mkdir(parents=True)
    manifest=dict(status=result['status'],gpu_measured=result.get('gpu_measured',False),files={},
        scope='small evidence only; full traces, original KV, model weights, fitted state and packets stay local')
    for name,src in sources.items():
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,target)
        manifest['files'][name]=dict(bytes=src.stat().st_size,sha256=sha(src))
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md'); return 0


if __name__=='__main__': raise SystemExit(main())
