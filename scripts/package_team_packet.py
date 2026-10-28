#!/usr/bin/env python3
"""Verify and package small one-step packet receipts; no KV, streams or fits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_team_packet import identity, normalize_row, aggregate, transfer_models, METHODS, check_fit_metadata
from scripts.bench_lite_baselines import digest


def checked(root, name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or (root/name).is_symlink():
        raise ValueError('missing/outside-run/symlink artifact: '+str(name))
    return path


def verify(run, result):
    run=Path(run).resolve()
    if result.get('format')!='team_packet_v1': raise ValueError('wrong report format')
    completed=result['status']=='completed'
    if completed and not result.get('gpu_measured'): raise ValueError('CPU evidence cannot be a completed GPU report')
    if result.get('recipe'):
        recipe=result['recipe']
        if identity(recipe)!=result.get('identity'): raise ValueError('recipe identity differs')
        parent=Path(recipe['parent_run'])/'results.json'
        if digest(parent)!=recipe['parent_results_sha256']: raise ValueError('parent evidence changed')
        for name,wanted in recipe['parent_request_hashes'].items():
            if digest(Path(recipe['parent_run'])/name/'request.json')!=wanted:
                raise ValueError('parent codec request changed')
        for name,wanted in {**recipe['unchanged_code'],**recipe['code']}.items():
            if digest(ROOT/name)!=wanted: raise ValueError('measured source changed: '+name)
        if json.loads(checked(run,'source_manifest.json').read_text())!=recipe: raise ValueError('source receipt differs')
        expected={f'{m}_t{n}' for m in METHODS for n in recipe['lengths']}
        if completed and (set(result['selected_jobs'])!=expected or set(result['jobs'])!=expected):
            raise ValueError('method/length coverage differs')
    elif completed: raise ValueError('completed report has no recipe')
    fit=result.get('fit')
    if fit:
        path=checked(run,'fit.json')
        if digest(path)!=fit['receipt_sha256'] or json.loads(path.read_text())!={k:v for k,v in fit.items() if k!='receipt_sha256'}:
            raise ValueError('calibration receipt differs')
        state=run/'team_state'
        if digest(checked(run,'team_state/manifest.json'))!=fit['manifest_sha256']:
            raise ValueError('shared team state changed')
        from kv_compare.team_packet import load_state
        _,_,manifest=load_state(state,'cpu')
        check_fit_metadata(manifest,fit['metadata'])
        if (manifest['state_sha256']!=fit['state_sha256'] or
                fit['metadata']['recipe_identity']!=result['identity'] or not fit.get('gpu_measured')):
            raise ValueError('shared state/calibration source identity differs')
    elif completed: raise ValueError('completed run has no verified calibration')
    inputs={e['document']:e for e in result.get('inputs',[])}
    if completed:
        expected_docs={f'{d}_t{n}' for n in recipe['lengths'] for d in ('test_text','test_code')}
        if set(inputs)!=expected_docs: raise ValueError('matched TEST coverage differs')
    for e in inputs.values():
        path=Path(e['path']).resolve()
        if not path.is_relative_to(run) or digest(path)!=e['file_sha256']:
            raise ValueError('matched TEST input changed')
        import numpy as np
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        if a.dtype!=np.uint16 or list(a.shape)!=e['shape'] or a.shape!=(28,e['tokens'],2048):
            raise ValueError('matched BF16 geometry differs')
        h=hashlib.sha256()
        for layer in a: h.update(np.asarray(layer,dtype='<u2').tobytes())
        if h.hexdigest()!=e['original_word_sha256']: raise ValueError('matched original-word digest differs')
    reconstructed=[]
    for name,job in result.get('jobs',{}).items():
        if job.get('status')!='completed': raise ValueError('unfinished job recorded')
        for artifact,wanted in job['artifacts'].items():
            if digest(checked(run,artifact))!=wanted: raise ValueError('worker artifact changed')
        q=json.loads(checked(run,name+'/request.json').read_text())
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json');proof=json.loads(proof_path.read_text())
        checks={c['id']:c for c in proof['checks']}
        if (q['method'] not in METHODS or len(q['inputs'])!=2 or
                len({e['tokens'] for e in q['inputs']})!=1 or
                {e['source_document'] for e in q['inputs']}!={'test_text','test_code'} or
                name!=q['method']+'_t'+str(q['inputs'][0]['tokens']) or
                {e['document'] for e in q['inputs']}!={r['document'] for r in job['rows']} or
                any(r['method']!=q['method'] for r in job['rows'])):
            raise ValueError('method/length/document labels differ')
        if (not data.get('gpu_measured') or not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
                data['status']!='completed' or data['rows']!=job['rows'] or data['codec_provenance']!=q['codec'] or
                data['receiver_sha256']!=digest(proof_path) or len(checks)!=len(job['rows']) or
                set(checks)!={r['id'] for r in job['rows']}):
            raise ValueError('independent actual-GPU receiver coverage differs')
        if q['method'].startswith('team_') or q['method']=='lite_current_graph':
            if (data['runtime_policy']!=data['runtime_policy_after'] or proof['runtime_policy']!=data['runtime_policy'] or
                    not proof.get('alternating_packets') or not proof.get('independent_returned_outputs')):
                raise ValueError('owned output/runtime evidence differs')
            receiver=json.loads(checked(run,name+'/receiver_request.json').read_text())
            if set(receiver)!={'codec','entries','output'} or receiver['codec']!=q['codec']:
                raise ValueError('packet-only receiver request differs')
            if any({'path','inputs','captures','source_run','predictions','decoded'} & set(e) for e in receiver['entries']):
                raise ValueError('receiver contains source data')
            det=data['runtime_policy']['deterministic_algorithms_enabled']
            if det != (q['method']=='lite_current_graph'): raise ValueError('intended team/Lite runtime differs')
        else:
            if (data['runtime_policy']!=data['runtime_policy_after'] or proof['runtime_policy']!=data['runtime_policy'] or
                    data['runtime_policy']['deterministic_algorithms_enabled'] or
                    data.get('underlying_worker')!='scripts/bench_lite_baselines.py'):
                raise ValueError('original native runtime/implementation differs')
            old=checked(run,name+'/original_worker_results.json')
            oldproof=checked(run,name+'/original_receiver.json')
            if digest(old)!=data['original_worker_results_sha256'] or digest(oldproof)!=data['original_receiver_sha256']:
                raise ValueError('original baseline proof changed')
        for r in job['rows']:
            e=inputs.get(r['document']);c=checks[r['id']]
            if e is None or r['shape']!=e['shape'] or r['values']!=e['values']:
                raise ValueError('method did not receive the complete matched words')
            packet=Path(r['packet']).resolve()
            if (not packet.is_relative_to(run) or packet.is_symlink() or digest(packet)!=r['packet_sha256'] or
                    packet.stat().st_size!=r['packet_bytes']): raise ValueError('complete packet changed')
            if (r['mode']!='lossless' or r['kappa']!=0 or not r['original_exact'] or r['status']!='verified' or
                    r['original_word_sha256']!=e['original_word_sha256'] or
                    r['decoded_word_sha256']!=e['original_word_sha256'] or not c['verified'] or
                    c['word_sha256']!=e['original_word_sha256'] or c['shape']!=e['shape'] or
                    c['packet_sha256']!=r['packet_sha256']): raise ValueError('original exactness proof differs')
            if abs(r['complete_bpv']-8*r['packet_bytes']/r['values'])>1e-10:
                raise ValueError('complete byte accounting differs')
            for direction in ('encode','decode'):
                record=r[direction]
                if len(record['samples'])!=recipe['repeats'] or len(record['warmup_samples'])!=recipe['warmup']:
                    raise ValueError('timing sample coverage differs')
                if any(s['wall_ms']<=0 or s['cuda_interval_ms']<0 for s in record['samples']):
                    raise ValueError('actual positive CUDA timing required')
                if record['wall_median_ms']!=statistics.median(s['wall_ms'] for s in record['samples']):
                    raise ValueError('reported timing median differs from samples')
            if r['method'].startswith('team_'):
                from kv_compare.team_packet import TeamPacket
                packet_obj=TeamPacket.import_bytes(packet.read_bytes())
                acc=r['accounting']
                if (packet_obj.packet_bytes!=r['packet_bytes'] or acc['packet_bytes']!=r['packet_bytes'] or
                        acc['compact_payload_bytes']+acc['directory_bytes']+acc['header_checksum_bytes']!=r['packet_bytes'] or
                        packet_obj.header['state_sha256']!=fit['state_sha256']):
                    raise ValueError('compact team packet/state accounting differs')
            reconstructed.append(normalize_row(r))
    if reconstructed!=result.get('rows',[]): raise ValueError('root rows differ from completed workers')
    if aggregate(reconstructed)!=result.get('aggregates') or transfer_models(result.get('aggregates',[]))!=result.get('transfer_models'):
        raise ValueError('derived comparisons differ')
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--destination',type=Path,required=True)
    args=p.parse_args(argv);run=args.run.resolve();dest=args.destination
    if dest.exists(): raise ValueError('destination already exists')
    result=json.loads((run/'results.json').read_text());verify(run,result)
    dest.mkdir(parents=True)
    names=['results.json','summary.md','source_manifest.json','fit.json','fit.log','console.log','fit_request.json']
    names += [str(p.relative_to(run)) for name in result.get('jobs',{})
        for p in (run/name).iterdir() if p.suffix in ('.json','.log')]
    manifest=dict(format='team_packet_small_report_v1',source_run=str(run),status=result['status'],
        excluded='captured KV, input arrays, compressed packets, predictor tensors, shared-state NPZs and CUDA buffers',files={})
    for name in sorted(set(names)):
        src=run/name
        if not src.is_file(): continue
        checked(run,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target);manifest['files'][name]=dict(bytes=target.stat().st_size,sha256=digest(target))
    write_path=dest/'package_manifest.json'
    write_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md');return 0


if __name__=='__main__': raise SystemExit(main())
