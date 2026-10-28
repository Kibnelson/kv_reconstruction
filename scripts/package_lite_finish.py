#!/usr/bin/env python3
"""Package checked finish-kernel evidence without source KV or graph buffers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_lite_baselines import checked,sha
from scripts.bench_lite_finish import aggregate_decode,fetch_model,budgets,ARMS
from kv_compare.lite_scoreboard import validate_comparison


def verify(run,result):
    run=Path(run).resolve(); allowed=set(); parent=None
    if result['status']=='completed' and not result.get('gpu_measured'):
        raise ValueError('CPU evidence cannot be a completed GPU benchmark')
    if result.get('recipe'):
        path=Path(result['recipe']['parent_run'])/'results.json'
        if sha(path)!=result['recipe']['parent_results_sha256']: raise ValueError('parent evidence changed')
        parent=json.loads(path.read_text())
        allowed={(str(Path(r['packet']).resolve()),r['packet_sha256']) for r in parent['rows']}
    if result['status']=='completed':
        if not result.get('selected_jobs') or set(result['selected_jobs'])!=set(result['jobs']): raise ValueError('job coverage differs')
        if set(result.get('diagnostics',{}))!=set(ARMS): raise ValueError('phase diagnostic coverage differs')
        if result['recipe']['profile'] and set(result.get('profiles',{}))!=set(ARMS): raise ValueError('eager profile coverage differs')
        if hashlib.sha256(json.dumps(result['recipe'],sort_keys=True).encode()).hexdigest()!=result['identity']:
            raise ValueError('recipe identity differs')
        rows=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        if rows!=result['rows']: raise ValueError('root rows differ from worker rows')
        validate_comparison(rows)
        if (aggregate_decode(rows)!=result['aggregates'] or fetch_model(result['aggregates'])!=result['fetch_model'] or
                budgets(result['aggregates'])!=result['budgets']): raise ValueError('derived comparisons differ')
        expected={r['document']:r for r in parent['jobs']['lite_v64_b32']['rows']}
        for r in result['jobs']['lite_finish']['rows']:
            e=expected.get(r['document'])
            if e is None or any(r[k]!=e[k] for k in ('shape','values','packet_sha256','packet_bytes',
                    'original_word_sha256','decoded_word_sha256','state_sha256')):
                raise ValueError('finish packet differs from original exact packet')
    for name,job in result.get('jobs',{}).items():
        if job.get('status')!='completed': raise ValueError('unfinished job recorded')
        for path,wanted in job['artifacts'].items():
            if sha(checked(run,path))!=wanted: raise ValueError('completed worker evidence changed')
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json'); proof=json.loads(proof_path.read_text())
        checks={c['id']:c for c in proof['checks']}
        if (not data.get('gpu_measured') or not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
                data['rows']!=job['rows'] or sha(proof_path)!=data['receiver_sha256'] or
                len(checks)!=len(job['rows']) or set(checks)!={r['id'] for r in job['rows']}):
            raise ValueError('fresh GPU receiver coverage differs')
        if name=='lite_finish':
            if job.get('graph_setups')!=data['graph_setups']: raise ValueError('root graph memory records differ')
            if (data['graph_capture_counts']!={a:1 for a in ARMS} or proof['graph_capture_counts']!={a:1 for a in ARMS} or
                    not proof.get('alternating_packets') or not proof.get('distinct_payload_lengths') or
                    not proof.get('independent_returned_outputs')): raise ValueError('graph replay checks differ')
            if {r['method'] for r in job['rows']}!={'lite_finish_'+a for a in ARMS}: raise ValueError('paired arm coverage differs')
            expected={r['document']:r for r in job['rows']}
            parity={p['document']:p for p in data['encoder_byte_parity']}
            if len(parity)!=len(data['encoder_byte_parity']) or set(parity)!=set(expected): raise ValueError('encoder byte parity coverage differs')
            if any(not p['byte_identical'] or p['packet_sha256']!=expected[d]['packet_sha256'] for d,p in parity.items()):
                raise ValueError('encoder changed original bytes')
            if len(data['sample_order'])!=result['recipe']['warmup']+result['recipe']['repeats']:
                raise ValueError('paired sample order differs')
        for r in job['rows']:
            packet=Path(r['packet']); own=packet.resolve().is_relative_to(run)
            original=(str(packet.resolve()),r['packet_sha256']) in allowed
            if (packet.is_symlink() or not (own or original) or sha(packet)!=r['packet_sha256'] or
                    packet.stat().st_size!=r['packet_bytes']): raise ValueError('packet changed or is not parent-allowlisted')
            c=checks[r['id']]
            if (not c['verified'] or c['packet_sha256']!=r['packet_sha256'] or c['word_sha256']!=r['decoded_word_sha256'] or
                    c['shape']!=r['shape']): raise ValueError('receiver original words differ')
            if r['mode']!='lossless' or not r['original_exact'] or r['original_word_sha256']!=r['decoded_word_sha256']:
                raise ValueError('lossless original words differ')
            if abs(r['complete_bpv']-8*r['packet_bytes']/r['values'])>1e-10: raise ValueError('complete byte accounting differs')
            if name=='lite_finish' and (len(r['decode']['samples'])!=result['recipe']['repeats'] or
                    len(r['decode']['warmup_samples'])!=result['recipe']['warmup']): raise ValueError('paired sample coverage differs')
    for arm,data in result.get('diagnostics',{}).items():
        path=checked(run,'diagnostics/'+arm+'/phases.json')
        if (sha(path)!=data['summary_sha256'] or data['status']!='completed' or not data.get('gpu_measured') or
                {k:v for k,v in data.items() if k!='summary_sha256'}!=json.loads(path.read_text())):
            raise ValueError('separate GPU phase evidence differs')
    for arm,data in result.get('profiles',{}).items():
        path=checked(run,'profiles/'+arm+'/profile.json')
        if (sha(path)!=data['summary_sha256'] or data['status']!='completed' or not data.get('gpu_measured') or
                {k:v for k,v in data.items() if k!='summary_sha256'}!=json.loads(path.read_text())):
            raise ValueError('separate eager profile evidence differs')
        trace=Path(data['profile']['trace'])
        if trace.is_symlink() or not trace.resolve().is_relative_to(run) or sha(trace)!=data['profile']['trace_sha256']:
            raise ValueError('local profiler trace changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path); p.add_argument('--destination',required=True,type=Path)
    args=p.parse_args(argv); run=args.run.resolve(); dest=args.destination
    if dest.exists(): raise ValueError('fresh destination required')
    result=json.loads(checked(run,'results.json').read_text()); verify(run,result)
    names=[n for n in ('results.json','summary.md','source_manifest.json','console.log') if (run/n).exists()]
    for name in result.get('jobs',{}):
        names += [name+'/'+f for f in ('worker_results.json','receiver.json','receiver.log','worker.log') if (run/name/f).exists()]
    for name in result.get('diagnostics',{}): names += ['diagnostics/'+name+'/'+f for f in ('phases.json','worker.log')]
    for name in result.get('profiles',{}): names += ['profiles/'+name+'/'+f for f in ('profile.json','worker.log')]
    sources={name:checked(run,name) for name in names}; dest.mkdir(parents=True)
    manifest=dict(status=result['status'],gpu_measured=result.get('gpu_measured',False),files={},
        scope='small evidence only; no source KV, packets, fixed state, graph buffers or full traces')
    for name,src in sources.items():
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,target)
        manifest['files'][name]=dict(bytes=src.stat().st_size,sha256=sha(src))
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md'); return 0


if __name__=='__main__': raise SystemExit(main())
