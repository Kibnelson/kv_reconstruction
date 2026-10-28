#!/usr/bin/env python3
"""Package small verified graph-decoder evidence; keep KV and graphs local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_lite_baselines import checked,sha
from scripts.bench_lite_graph import aggregate_decode,fetch_model,LAYOUTS
from kv_compare.lite_scoreboard import validate_comparison
from kv_compare.lite_stream_layout import fetch_budgets


def verify(run,result):
    run=Path(run).resolve()
    if result['status']=='completed':
        if not result.get('gpu_measured'): raise ValueError('CPU evidence cannot be a completed GPU benchmark')
        if not result.get('selected_jobs') or set(result['jobs'])!=set(result['selected_jobs']):
            raise ValueError('completed job coverage differs')
        if set(result.get('diagnostics',{}))!={name for name,_ in LAYOUTS}:
            raise ValueError('phase diagnostic coverage differs')
        recipe=result['recipe']
        if hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()!=result['identity']:
            raise ValueError('recipe identity differs')
        rows=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        if rows!=result['rows']: raise ValueError('root rows differ from completed workers')
        validate_comparison(rows)
        if (aggregate_decode(rows)!=result['aggregates'] or fetch_model(result['aggregates'])!=result['fetch_model'] or
                fetch_budgets(result['aggregates'])!=result['fetch_budgets']): raise ValueError('derived decoder scoreboard differs')
        parent_path=Path(recipe['parent_run'])/'results.json'
        if sha(parent_path)!=recipe['parent_results_sha256']: raise ValueError('parent evidence changed')
        parent=json.loads(parent_path.read_text())
        for name,oldname in LAYOUTS:
            expected={r['document']:r for r in parent['jobs'][oldname]['rows']}
            for row in result['jobs'][name]['rows']:
                e=expected.get(row['document'])
                if e is None or any(row[k]!=e[k] for k in ('packet_sha256','packet_bytes','shape','values',
                        'original_word_sha256','decoded_word_sha256','state_sha256')):
                    raise ValueError('graph/eager packet differs from checked original packet')
    for name,job in result.get('jobs',{}).items():
        if job.get('status')!='completed': raise ValueError('unfinished job recorded')
        for path,wanted in job['artifacts'].items():
            if sha(checked(run,path))!=wanted: raise ValueError('completed worker evidence changed')
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json'); proof=json.loads(proof_path.read_text())
        checks={c['id']:c for c in proof['checks']}
        if (not data.get('gpu_measured') or not proof.get('gpu_measured') or
                not proof.get('no_source_kv_or_predictions') or data['rows']!=job['rows'] or
                sha(proof_path)!=data['receiver_sha256'] or len(checks)!=len(job['rows']) or
                set(checks)!={r['id'] for r in job['rows']}): raise ValueError('independent GPU receiver coverage differs')
        is_lite=name.startswith('lite_')
        if is_lite:
            if job.get('graph_setups')!=data['graph_setups']:
                raise ValueError('root graph setup/memory records differ from worker evidence')
            if (data['graph_capture_count']!=1 or proof['graph_capture_count']!=1 or
                    not proof.get('alternating_packets') or not proof.get('distinct_payload_lengths') or
                    not proof.get('independent_returned_outputs')): raise ValueError('graph replay coverage differs')
            if {r['method'] for r in job['rows']}!={name+'_eager',name+'_graph'}:
                raise ValueError('paired graph/eager arm coverage differs')
            if len(data['sample_order'])!=result['recipe']['warmup']+result['recipe']['repeats']:
                raise ValueError('paired sample-order coverage differs')
        for r in job['rows']:
            packet=Path(r['packet']); parent=Path(result['recipe']['parent_run'])
            if (packet.is_symlink() or not (packet.resolve().is_relative_to(run) or packet.resolve().is_relative_to(parent)) or
                    sha(packet)!=r['packet_sha256'] or packet.stat().st_size!=r['packet_bytes']):
                raise ValueError('completed packet changed/outside checked runs')
            c=checks[r['id']]
            if (not c['verified'] or c['packet_sha256']!=r['packet_sha256'] or c['word_sha256']!=r['decoded_word_sha256'] or
                    c['shape']!=r['shape']): raise ValueError('fresh receiver words differ')
            if r['mode']!='lossless' or not r['original_exact'] or r['original_word_sha256']!=r['decoded_word_sha256']:
                raise ValueError('original BF16 words differ')
            if abs(r['complete_bpv']-8*r['packet_bytes']/r['values'])>1e-10: raise ValueError('complete byte accounting differs')
            if is_lite and (len(r['decode']['samples'])!=result['recipe']['repeats'] or
                    len(r['decode']['warmup_samples'])!=result['recipe']['warmup']): raise ValueError('paired sample coverage differs')
    for name,data in result.get('diagnostics',{}).items():
        path=checked(run,'diagnostics/'+name+'/phases.json')
        if (sha(path)!=data['summary_sha256'] or data['status']!='completed' or not data.get('gpu_measured') or
                {k:v for k,v in data.items() if k!='summary_sha256'}!=json.loads(path.read_text())):
            raise ValueError('separate GPU phase diagnostics differ')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path); p.add_argument('--destination',required=True,type=Path)
    args=p.parse_args(argv); run=args.run.resolve(); dest=args.destination
    if dest.exists(): raise ValueError('fresh destination required')
    result=json.loads(checked(run,'results.json').read_text()); verify(run,result)
    names=[n for n in ('results.json','summary.md','source_manifest.json','console.log') if (run/n).exists()]
    for name in result.get('jobs',{}):
        names += [name+'/'+f for f in ('worker_results.json','receiver.json','receiver.log','worker.log') if (run/name/f).exists()]
    for name in result.get('diagnostics',{}):
        names += ['diagnostics/'+name+'/'+f for f in ('phases.json','worker.log')]
    sources={name:checked(run,name) for name in names}; dest.mkdir(parents=True)
    manifest=dict(status=result['status'],gpu_measured=result.get('gpu_measured',False),files={},
        scope='small evidence only; no source KV, packets, state, graph buffers or model weights')
    for name,src in sources.items():
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,target)
        manifest['files'][name]=dict(bytes=src.stat().st_size,sha256=sha(src))
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md'); return 0


if __name__=='__main__': raise SystemExit(main())
