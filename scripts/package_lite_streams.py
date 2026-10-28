#!/usr/bin/env python3
"""Package checked stream-tuning evidence without copying KV, states or traces."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_lite_baselines import checked,sha
from kv_compare.lite_scoreboard import aggregate,network_model,validate_comparison
from kv_compare.lite_stream_layout import geometry,fetch_budgets


def verify(run,result):
    run=Path(run).resolve()
    if result['status']=='completed':
        if not result.get('gpu_measured'): raise ValueError('CPU evidence cannot be a completed GPU benchmark')
        if not result.get('selected_jobs') or set(result['selected_jobs'])!=set(result['jobs']):
            raise ValueError('completed job coverage differs')
        if result['recipe']['profiles'] and (len(result.get('selected_profiles',[]))!=2 or
                set(result['profiles'])!=set(result['selected_profiles'])): raise ValueError('profile coverage differs')
        recipe=result['recipe']
        if hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()!=result['identity']:
            raise ValueError('recipe digest differs')
        rows=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        if rows!=result['rows']: raise ValueError('root rows differ from workers')
        validate_comparison(rows)
        if (aggregate(rows)!=result['aggregates'] or network_model(result['aggregates'])!=result['network_model'] or
                fetch_budgets(result['aggregates'])!=result['fetch_budgets']): raise ValueError('derived comparison differs')
        parent_path=Path(recipe['parent_run'])/'results.json'
        if sha(parent_path)!=recipe['parent_results_sha256']: raise ValueError('parent evidence changed')
        parent=json.loads(parent_path.read_text())
        expected={r['document']:r for r in parent['jobs']['lite_b8_fused']['rows']}
        configurations={name:(v,b) for name,v,b in recipe['configurations']}
        for r in rows:
            if not r['method'].startswith('lite_'): continue
            e=expected.get(r['document'])
            if e is None or any(r[k]!=e[k] for k in ('shape','values','original_word_sha256','decoded_word_sha256','state_sha256')):
                raise ValueError('candidate differs from original lossless input/state')
            if r['reference_packet_sha256']!=e['packet_sha256'] or r['reference_packet_bytes']!=e['packet_bytes']:
                raise ValueError('candidate reference packet changed')
            v,b=configurations[r['method']]
            if (r['engine_options']['values_per_stream'],r['engine_options']['block'])!=(v,b):
                raise ValueError('candidate stream geometry differs')
            if (v,b)==(128,8) and (not r['packet_matches_original'] or r['packet_sha256']!=e['packet_sha256']):
                raise ValueError('V128/B8 control differs from historical fused bytes')
    for name,job in result.get('jobs',{}).items():
        if job.get('status')!='completed': raise ValueError('unfinished job recorded')
        for path,wanted in job['artifacts'].items():
            if sha(checked(run,path))!=wanted: raise ValueError('worker evidence changed')
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json'); proof=json.loads(proof_path.read_text())
        checks={c['id']:c for c in proof['checks']}
        if (not data.get('gpu_measured') or not proof.get('gpu_measured') or
                not proof.get('no_source_kv_or_predictions') or sha(proof_path)!=data['receiver_sha256'] or
                data['rows']!=job['rows'] or len(checks)!=len(job['rows']) or set(checks)!={r['id'] for r in job['rows']}):
            raise ValueError('independent CUDA receiver coverage differs')
        for r in job['rows']:
            p=Path(r['packet'])
            if (p.is_symlink() or not p.resolve().is_relative_to(run) or sha(p)!=r['packet_sha256'] or
                    p.stat().st_size!=r['packet_bytes']): raise ValueError('packet changed/outside run')
            c=checks[r['id']]
            if (not c['verified'] or c['packet_sha256']!=r['packet_sha256'] or c['word_sha256']!=r['decoded_word_sha256'] or
                    c['shape']!=r['shape']): raise ValueError('receiver words differ')
            if r['mode']!='lossless' or r['kappa']!=0 or not r['original_exact'] or r['decoded_word_sha256']!=r['original_word_sha256']:
                raise ValueError('lossless original BF16 words differ')
            if abs(r['complete_bpv']-8*r['packet_bytes']/r['values'])>1e-10: raise ValueError('complete size differs')
            if name.startswith('lite_'):
                if r['packet_bytes']!=r['payload_bytes']+r['directory_bytes']+r['framing_bytes']:
                    raise ValueError('wire accounting differs')
                if (r['mandatory_stream_state_bytes']!=sum(g['groups']*g['state_bytes'] for g in r['coder_geometry']) or
                        r['directory_bytes']!=sum(g['groups']*g['directory_bytes'] for g in r['coder_geometry'])):
                    raise ValueError('stream-state/directory accounting differs')
                for g in r['coder_geometry']:
                    expected=geometry(g['values'],g['values_per_stream'],g['lanes_per_program'])
                    if any(g[k]!=v for k,v in expected.items()): raise ValueError('stream geometry accounting differs')
    for name,data in result.get('profiles',{}).items():
        path=checked(run,'profiles/'+name+'/profile.json')
        if sha(path)!=data['summary_sha256'] or not data.get('gpu_measured') or data['status']!='completed':
            raise ValueError('GPU profile evidence differs')
        if {k:v for k,v in data.items() if k!='summary_sha256'}!=json.loads(path.read_text()):
            raise ValueError('root profile differs from diagnostic receipt')
        for p in data['profiles']:
            trace=Path(p['trace'])
            if trace.is_symlink() or not trace.resolve().is_relative_to(run) or sha(trace)!=p['trace_sha256']:
                raise ValueError('diagnostic trace changed')


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
        scope='small evidence only; no model weights, fitted state, source KV, compressed packets or full traces')
    for name,src in sources.items():
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,target)
        manifest['files'][name]=dict(bytes=src.stat().st_size,sha256=sha(src))
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md'); return 0


if __name__=='__main__': raise SystemExit(main())
