#!/usr/bin/env python3
"""Verify same-packet register-partner decode and package small GPU evidence."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.bench_team_register import (ARMS,BASELINES,OWN,METHODS,LENGTHS,FILES,aggregate,
    identity,link_models,profile_kernels,profile_kernel_resources,read_parent,sample_order,
    selected_plan,valid_receiver_request)
from scripts.bench_lite_baselines import digest


def checked(root,name):
    root=Path(root).resolve();raw=root/name;path=raw.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('missing/outside-run receipt: '+str(name))
    relative=raw.relative_to(root)
    if any(root.joinpath(*relative.parts[:i]).is_symlink() for i in range(1,len(relative.parts)+1)):
        raise ValueError('symlink receipt: '+str(name))
    return path


def check_samples(record,warmup,repeats):
    samples=record['samples'];warm=record['warmup_samples']
    if len(samples)!=repeats or len(warm)!=warmup:
        raise ValueError('actual CUDA sample coverage differs')
    for sample in (*warm,*samples):
        if (not math.isfinite(sample['wall_ms']) or sample['wall_ms']<=0 or
                not math.isfinite(sample['cuda_interval_ms']) or sample['cuda_interval_ms']<0):
            raise ValueError('finite positive CUDA timing required')
    if record['wall_median_ms']!=statistics.median(s['wall_ms'] for s in samples):
        raise ValueError('reported timing median differs from samples')


def check_packet_immutability(records,entries):
    expected=[dict(id=e['id'],packet_sha256=e['packet_sha256'],
                   bytes=e['packet_bytes'],unchanged=True) for e in entries]
    if records!=expected:
        raise ValueError('unchanged compressed-input proof differs')


def check_inverse_proof(proof):
    from kv_compare.team_lookup import cdf_inverse_proof
    if proof!=dict(cdf_inverse_proof(),native_tables_verified=True):
        raise ValueError('exhaustive native CDF inverse proof differs')


def expected_kernel_provenance(arm,decoder_warps=1):
    from kv_compare.team_register import kernel_provenance_for
    return kernel_provenance_for(decoder_mode='direct' if arm.startswith('direct_') else 'graph',
        register_partner=arm.endswith('_register'),decoder_warps=decoder_warps)


def check_kernel_provenance(proof,arm,decoder_warps=1):
    if proof!=expected_kernel_provenance(arm,decoder_warps):
        raise ValueError('decoder kernel implementation provenance differs')


def check_accounting(record,row,decoder_warps=1):
    direct=row['arm'].startswith('direct_')
    output_bytes=2*math.prod(row['shape'])
    if (record['decoder_mode']!=('direct' if direct else 'graph') or
            record['lookup'] is not False or
            record['register_partner']!=row['arm'].endswith('_register') or
            record['decoder_warps']!=decoder_warps or
            record['decoder_kernel']!=expected_kernel_provenance(row['arm'],decoder_warps)['selected_kernel'] or
            record['borrowed_payload']!=direct or record['arena_bytes']!=0 or
            record['output_bytes']!=output_bytes or record['packet_bytes']!=row['packet_bytes'] or
            record['output_allocation_bytes_per_call']!=output_bytes or
            record['persistent_output_bytes']!=(0 if direct else output_bytes) or
            record['output_clone_bytes_per_call']!=(0 if direct else output_bytes) or
            record['compact_payload_bytes']+record['directory_bytes']+
                record['header_checksum_bytes']!=row['packet_bytes'] or
            (direct and record['compact_staging_bytes']!=0) or
            (not direct and record['compact_staging_bytes']<record['compact_payload_bytes'])):
        raise ValueError('declared decoder copy/output accounting differs')


def verify(run,result):
    run=Path(run).resolve()
    if result.get('format')!='team_register_v1':raise ValueError('wrong GPU decode report format')
    completed=result['status']=='completed'
    if completed and not result.get('gpu_measured'):
        raise ValueError('CPU evidence cannot be a completed GPU run')
    plan={};recipe=result.get('recipe')
    if recipe:
        if identity(recipe)!=result.get('identity'):raise ValueError('GPU decode recipe identity differs')
        if type(recipe.get('decoder_warps')) is not int or recipe['decoder_warps'] not in (1,2):
            raise ValueError('decoder warp count must be one or two')
        parent_dir=Path(recipe['parent_run']).resolve()
        if digest(parent_dir/'results.json')!=recipe['parent_results_sha256']:
            raise ValueError('frozen parent report changed')
        parent,inherited,selections,receipts=read_parent(parent_dir)
        plan=selected_plan(selections)
        if not isinstance(plan,dict):plan=dict(plan)
        if (recipe['unchanged_code']!=inherited or recipe['parent_artifacts']!=receipts or
                recipe['selections']!=selections or recipe['plan']!=list(plan) or
                recipe['methods']!=list(METHODS) or recipe['lengths']!=list(LENGTHS) or
                recipe['arms']!=list(ARMS) or recipe['native_receipt']!=parent['recipe']['native_receipt'] or
                recipe['software']!=parent['recipe']['software'] or
                recipe['cuda_runtime']!=parent['recipe']['cuda_runtime'] or
                recipe['device']!=parent['hardware']['name']):
            raise ValueError('frozen GPU packet selection/order changed')
        if recipe['code']!={name:digest(ROOT/name) for name in FILES}:
            raise ValueError('new measured GPU source changed')
        for name,wanted in recipe['unchanged_code'].items():
            if digest(ROOT/name)!=wanted:raise ValueError('frozen measured source changed: '+name)
        if json.loads(checked(run,'source_manifest.json').read_text())!=recipe:
            raise ValueError('source manifest differs')
        if recipe['warmup']<1 or recipe['repeats']<8 or recipe['repeats']%8:
            raise ValueError('insufficient warm CUDA sample coverage')
        if completed and (result['selected_jobs']!=list(plan) or set(result['jobs'])!=set(plan)):
            raise ValueError('GPU comparison job coverage differs')
    elif completed:raise ValueError('completed GPU report has no recipe')
    if completed:
        gate=checked(run,'cuda_tests.log').read_text()
        last_gate=gate.rsplit('\nRan ',1)[-1]
        first=last_gate.split(maxsplit=1)[0] if last_gate.split() else ''
        if ('test_' not in gate or '\nRan ' not in gate or '\nOK' not in last_gate or
                not first.isdigit() or int(first)<1 or 'FAILED' in last_gate or
                'skipped' in last_gate or 'expected failures' in last_gate):
            raise ValueError('completed report lacks a successful actual-CUDA correctness gate')
    rows=[]
    for name,job in result.get('jobs',{}).items():
        if name not in plan or job.get('status')!='completed':raise ValueError('unexpected/incomplete GPU job')
        expected_artifacts={name+'/'+f for f in ('request.json','worker_results.json','receiver.json','receiver_request.json')}
        if set(job['artifacts'])!=expected_artifacts:raise ValueError('GPU worker artifact coverage differs')
        for path,wanted in job['artifacts'].items():
            if digest(checked(run,path))!=wanted:raise ValueError('worker receipt changed')
        q=json.loads(checked(run,name+'/request.json').read_text())
        expected=dict(plan[name],output=str(run/name),warmup=recipe['warmup'],repeats=recipe['repeats'],
            decoder_warps=recipe['decoder_warps'])
        if q!=expected:raise ValueError('GPU request differs from selected frozen packets')
        data=json.loads(checked(run,name+'/worker_results.json').read_text())
        proof_path=checked(run,name+'/receiver.json');proof=json.loads(proof_path.read_text())
        receiver=json.loads(checked(run,name+'/receiver_request.json').read_text())
        valid_receiver_request(receiver)
        wanted_receiver=dict(method=q['method'],codec=q['codec'],entries=q['entries'],arms=q['arms'],
            round=q['round'],decoder_warps=recipe['decoder_warps'],output=str(run/name/'receiver.json'))
        if receiver!=wanted_receiver:raise ValueError('packet-only receiver request differs')
        entries={e['id']:e for e in q['entries']}
        expected_ids={q['round']+'_'+arm+'_'+e['id'] for arm in q['arms'] for e in q['entries']}
        checks={c['id']:c for c in proof['checks']}
        own=q['method'] in OWN
        wanted_graphs={arm:int(arm.startswith('graph_')) for arm in ARMS} if own else {}
        if (data['status']!='completed' or not data.get('gpu_measured') or not proof.get('gpu_measured') or
                data['method']!=q['method'] or data['round']!=q['round'] or
                data['rows']!=job['rows'] or data['codec_provenance']!=q['codec'] or
                data['receiver_sha256']!=digest(proof_path) or not proof.get('no_source_kv_or_predictions') or
                not proof.get('alternating_packets') or not proof.get('independent_returned_outputs') or
                data['graph_capture_counts']!=wanted_graphs or proof['graph_capture_counts']!=wanted_graphs or
                set(checks)!=expected_ids or len(proof['checks'])!=len(expected_ids) or
                {r['id'] for r in job['rows']}!=expected_ids or len(job['rows'])!=len(expected_ids)):
            raise ValueError('independent actual-GPU receiver coverage differs')
        before=data['runtime_policy']
        if (data['runtime_policy_after']!=before or proof['runtime_policy']!=before or
                proof['runtime_policy_after']!=before or before['deterministic_algorithms_enabled']):
            raise ValueError('intended codec runtime policy was not preserved')
        if data['sample_order']!=sample_order(q['entries'],q['arms'],recipe['warmup'],recipe['repeats']):
            raise ValueError('GPU arm/document schedule differs')
        check_packet_immutability(data['packet_immutability'],q['entries'])
        check_packet_immutability(proof['packet_immutability'],q['entries'])
        expected_inverse_arms=set(q['arms']) if own else set()
        if (set(data['inverse_proofs'])!=expected_inverse_arms or
                set(proof['inverse_proofs'])!=expected_inverse_arms or
                set(data['kernel_provenance'])!=expected_inverse_arms or
                set(proof['kernel_provenance'])!=expected_inverse_arms):
            raise ValueError('native inverse-proof arm coverage differs')
        for arm in expected_inverse_arms:
            check_inverse_proof(data['inverse_proofs'][arm])
            check_inverse_proof(proof['inverse_proofs'][arm])
            check_kernel_provenance(data['kernel_provenance'][arm],arm,recipe['decoder_warps'])
            check_kernel_provenance(proof['kernel_provenance'][arm],arm,recipe['decoder_warps'])
        for row in job['rows']:
            e=entries.get(row['parent_id']);c=checks[row['id']]
            if (e is None or row['arm'] not in q['arms'] or
                    row['id']!=q['round']+'_'+row['arm']+'_'+e['id'] or
                    row['method']!=q['method'] or row['round']!=q['round'] or row['tokens']!=q['tokens'] or
                    row['status']!='verified' or not row['original_exact'] or
                    row['decoded_word_sha256']!=e['original_word_sha256'] or
                    any(row[k]!=e[k] for k in ('document','shape','values','packet','packet_sha256',
                        'packet_bytes','original_word_sha256','complete_bpv','state_sha256')) or
                    not c['verified'] or c['word_sha256']!=e['original_word_sha256'] or
                    c['packet_sha256']!=e['packet_sha256'] or c['shape']!=e['shape']):
                raise ValueError('same-wire/original BF16 exactness differs')
            if row['complete_bpv']!=8*row['packet_bytes']/row['values']:
                raise ValueError('complete packet accounting differs')
            check_samples(row['decode'],recipe['warmup'],recipe['repeats'])
            if own and row['arm'].startswith('graph_'):
                check_samples(row['kernel_decode'],1,recipe['repeats'])
            elif 'kernel_decode' in row:
                raise ValueError('direct/native decode cannot claim captured graph timing')
            if own:check_accounting(row['accounting'],row,recipe['decoder_warps'])
            rows.append(row)
    if (rows!=result.get('rows',[]) or aggregate(rows)!=result.get('aggregates') or
            link_models(result.get('aggregates',[]))!=result.get('link_models')):
        raise ValueError('derived GPU comparisons differ')
    profiles=result.get('profiles',{})
    if recipe:
        expected_profiles={m+'_'+a+'_t8192' for m in OWN for a in ARMS}
        expected_profiles|={m+'_native_t8192' for m in BASELINES if m!='raw_bf16'}
        if not recipe.get('profile'):expected_profiles=set()
        if completed and set(profiles)!=expected_profiles:raise ValueError('separate GPU profile coverage differs')
    for name,profile in profiles.items():
        path=checked(run,'profiles/'+name+'/profile.json')
        request_path=checked(run,'profiles/'+name+'/request.json')
        method=profile['method'];arm=profile['arm']
        if name!=method+'_'+arm+'_t8192' or method not in METHODS or method=='raw_bf16':
            raise ValueError('unexpected GPU profile label')
        base=selections[method+'_t8192']
        if arm not in base['arms']:raise ValueError('GPU profile arm differs')
        entry=next(e for e in base['entries'] if e['document']=='test_text_t8192')
        wanted_request=dict(base,round='profile',arms=[arm],entries=[entry],
            decoder_warps=recipe['decoder_warps'],output=str(run/'profiles'/name))
        if digest(request_path)!=profile['request_sha256'] or json.loads(request_path.read_text())!=wanted_request:
            raise ValueError('GPU profile request/selection differs')
        if (digest(path)!=profile['summary_sha256'] or
                json.loads(path.read_text())!={k:v for k,v in profile.items() if k not in ('summary_sha256','request_sha256')} or
                profile['status']!='completed' or not profile.get('gpu_measured') or
                not profile.get('assertions_outside_profilers') or profile['tokens']!=8192 or
                profile['document']!=entry['document'] or profile['packet_sha256']!=entry['packet_sha256'] or
                profile['original_word_sha256']!=entry['original_word_sha256'] or
                profile['codec_provenance']!=base['codec'] or profile['profile_calls']!=3 or
                profile['runtime_policy']!=profile['runtime_policy_after'] or
                profile['runtime_policy']['deterministic_algorithms_enabled'] or
                profile['graph_capture_counts']!=({arm:int(arm.startswith('graph_'))} if method in OWN else {})):
            raise ValueError('separate GPU profile summary differs')
        expected_diagnostics=({'full_decode','stage','native_eager','output_clone'}
            if method in OWN and arm.startswith('graph_') else
            {'full_decode','stage'} if method in OWN else {'full_decode'})
        if set(profile['diagnostics'])!=expected_diagnostics:raise ValueError('GPU profile diagnostic coverage differs')
        check_packet_immutability(profile['packet_immutability'],[entry])
        if method in OWN:
            check_inverse_proof(profile['inverse_proof'])
            check_kernel_provenance(profile['kernel_provenance'],arm,recipe['decoder_warps'])
        elif profile.get('inverse_proof') is not None:
            raise ValueError('native baseline cannot claim one-step inverse proof')
        elif profile.get('kernel_provenance') is not None:
            raise ValueError('native baseline cannot claim register-kernel provenance')
        for record in profile['diagnostics'].values():check_samples(record,0,1)
        trace=Path(profile['trace_path']).resolve()
        if (not trace.is_relative_to(run) or checked(run,str(trace.relative_to(run)))!=trace or
                digest(trace)!=profile['trace_sha256'] or trace.stat().st_size!=profile['trace_bytes']):
            raise ValueError('local full GPU trace changed')
        if profile['device_kernels']!=profile_kernels(trace)[:60]:
            raise ValueError('GPU kernel-event attribution differs from full trace')
        if profile['device_kernel_resources']!=profile_kernel_resources(trace):
            raise ValueError('GPU kernel-resource fields differ from full trace')
    return True


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args(argv);run=args.run.resolve();dest=args.destination
    if dest.exists():raise ValueError('destination exists')
    result=json.loads((run/'results.json').read_text());verify(run,result)
    dest.mkdir(parents=True)
    names=['results.json','summary.md','source_manifest.json','console.log','cuda_tests.log']
    names += [str(p.relative_to(run)) for name in result.get('jobs',{})
        for p in (run/name).iterdir() if p.suffix in ('.json','.log')]
    names += [str(p.relative_to(run)) for name in result.get('profiles',{})
        for p in (run/'profiles'/name).iterdir() if p.suffix in ('.json','.log') and p.name!='trace.json']
    manifest=dict(format='team_register_small_report_v1',source_run=str(run),status=result['status'],files={},
        excluded='referenced original packets, fitted tensors, captured KV, input arrays and full profiler traces')
    for name in sorted(set(names)):
        src=run/name
        if not src.is_file():continue
        checked(run,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,target);manifest['files'][name]=dict(bytes=target.stat().st_size,sha256=digest(target))
    (dest/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md');return 0


if __name__=='__main__':raise SystemExit(main())
