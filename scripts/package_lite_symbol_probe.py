#!/usr/bin/env python3
"""Check and package small rate-probe evidence; never publish captures/counts/tables."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_lite_baselines import checked,sha
from scripts.bench_lite_symbol_probe import (FILES,RADII,CONTEXTS,LINKS,BASELINES,REFERENCE,canonical,key,
    walk_causal,select_candidate,required_budgets,projected_packet,layer_insights,table_metadata)
from scripts.bench_lite_graph import aggregate_decode


def verify(run,result):
    run=Path(run).resolve()
    if result['status']=='completed' and not result.get('gpu_measured'):
        raise ValueError('CPU evidence cannot publish a completed GPU rate probe')
    if not result.get('recipe'):
        if result['status']=='completed': raise ValueError('recipe required')
        return
    recipe=result['recipe']; parent_path=Path(recipe['parent_run'])/'results.json'
    if sha(parent_path)!=recipe['parent_results_sha256']: raise ValueError('parent report changed')
    parent=json.loads(parent_path.read_text()); causal,chain=walk_causal(recipe['parent_run'])
    if chain!=recipe['ancestor_chain'] or causal['protocol']!=recipe['original_protocol']:
        raise ValueError('checked causal lineage differs')
    if hashlib.sha256(canonical(recipe)).hexdigest()!=result['identity']: raise ValueError('recipe identity differs')
    if tuple(recipe['radii'])!=RADII or tuple(recipe['contexts'])!=CONTEXTS or tuple(recipe['links'])!=LINKS:
        raise ValueError('predeclared probe grid/links differ')
    if set(recipe['code'])!=set(FILES): raise ValueError('new source coverage differs')
    for name,wanted in {**recipe['unchanged_code'],**recipe['code']}.items():
        if sha(ROOT/name)!=wanted: raise ValueError('probe or inherited source changed: '+name)
    request=Path(recipe['parent_run'])/'lite_entropy/request.json'
    if sha(request)!=recipe['parent_request_sha256']: raise ValueError('parent request changed')
    q=json.loads(request.read_text()); entries={e['document']:e for e in q['entries']}
    if q['codec']!=recipe['codec']: raise ValueError('original predictor request differs')
    protocol=causal['protocol']
    if (recipe['source']!=protocol['source'] or result['model']!=parent['model'] or
            recipe['software']!=parent['recipe']['software'] or recipe['device']!=parent['hardware']['name'] or
            recipe['cuda_runtime']!=parent['recipe']['cuda_runtime']): raise ValueError('source/model/runtime differs')
    if Path(q['codec']['state']).resolve()!=Path(causal['state_directories']['lite']).resolve():
        raise ValueError('original fitted state path differs')
    if result['status']!='completed': return
    expected_baselines=aggregate_decode([r for r in parent['rows'] if r['method'] in (*BASELINES,REFERENCE)])
    if result['baselines']!=expected_baselines or result['baseline_receipt']['retimed'] is not False:
        raise ValueError('reused baseline rows changed')
    receipt=result['baseline_receipt']
    if (receipt['source_run']!=recipe['parent_run'] or receipt['results_sha256']!=recipe['parent_results_sha256'] or
            receipt['runtime_policies']!={n:parent['jobs'][n]['runtime_policy'] for n in BASELINES}):
        raise ValueError('inherited native measurement provenance differs')
    if result['runtime_policy_before']!=result['runtime_policy_after']: raise ValueError('runtime flags were not restored')
    policy=result['allocation_policy']
    if policy['deterministic_algorithms_enabled'] is not True or policy['fill_uninitialized_memory'] is not False:
        raise ValueError('known-KV replay runtime differs')
    import numpy as np
    from kv_compare.lite_symbol_probe import fold_counts,fit_tables,cost,ESCAPE_DIAGNOSTIC_LABELS
    counts={}; escape={}; roles={}
    for name,document in result['documents'].items():
        path=checked(run,document['counts'])
        if sha(path)!=document['counts_sha256']: raise ValueError('resumed count checkpoint changed')
        with np.load(path,allow_pickle=False) as saved: c=saved['counts']; esc=saved['escape_counts']
        prefix=document['prefix']; n=prefix['tokens']; role=document['role']; roles.setdefault(role,[]).append(name)
        expected_tokens=protocol['fit_tokens'] if role=='TRAIN' else protocol['scale_tokens'] if role=='validation' else protocol['tokens']
        if (n!=expected_tokens or prefix['shape']!=[28,n,2048] or c.dtype!=np.int64 or c.shape!=(28,32,4096) or
                esc.dtype!=np.int64 or esc.shape!=(28,len(RADII),len(ESCAPE_DIAGNOSTIC_LABELS)) or np.any(c<0) or np.any(esc<0) or
                np.any(c.sum(axis=(1,2))!=n*2048) or np.any(esc[:,:,0]!=n*2048) or
                int(c.sum())!=prefix['values'] or document['counts_values']!=prefix['values'] or document['counts_shape']!=list(c.shape)):
            raise ValueError('count/prefix geometry or word coverage differs')
        for radius_index,radius in enumerate(RADII):
            if not np.array_equal(fold_counts(c,radius)[:,:,-1].sum(1),esc[:,radius_index,3]):
                raise ValueError('escape reason counters differ from symbol counts')
        if not document['measurement'].get('gpu_measured'): raise ValueError('actual GPU replay required')
        if role=='TEST':
            e=entries.get(name); proof=document['original_packet_proof']
            if (e is None or prefix['original_word_sha256']!=e['original_word_sha256'] or prefix['shape']!=e['shape'] or
                    result['state_sha256']!=e['state_sha256'] or sha(e['packet'])!=e['packet_sha256'] or
                    not proof['original_packet_decode_exact'] or proof['new_symbol_packet_exists'] is not False or
                    proof['packet']!=e['packet'] or proof['packet_sha256']!=e['packet_sha256'] or
                    proof['original_word_sha256']!=e['original_word_sha256'] or proof['state_sha256']!=e['state_sha256'] or
                    document['measurement']['symbol_roundtrip_checked_values']!=prefix['values'] or
                    document['selected_radius']!=result['selection']['radius'] or
                    document['selection_sha256']!=result['selection_sha256']):
                raise ValueError('original TEST packet/word/selection receipt differs')
        elif document['selected_radius'] is not None or document['original_packet_proof'] is not None:
            raise ValueError('TRAIN/validation roles contaminated with TEST proof')
        counts[name]=c; escape[name]=esc
    if result['status']!='completed': return
    if set(roles)!=set(('TRAIN','validation','TEST')) or roles['TRAIN']!=protocol['train'] or set(roles['TEST'])!=set(entries) or len(roles['validation'])!=1:
        raise ValueError('TRAIN/validation/TEST coverage differs')
    if set(result['documents'])!=set(protocol['train']+roles['validation']+list(entries)):
        raise ValueError('unexpected scored documents')
    train=sum((counts[n] for n in protocol['train'])); validation=counts[roles['validation'][0]]
    table_path=checked(run,'train_frequency_tables.npz')
    if sha(table_path)!=result['train_table_sha256']: raise ValueError('TRAIN frequency arrays changed')
    candidates=[]; tables={}
    with np.load(table_path,allow_pickle=False) as saved:
        if set(saved.files)!={key(r,c)+'_'+part for r in RADII for c in CONTEXTS for part in ('freq','cdf')}:
            raise ValueError('frequency array coverage differs')
        for radius in RADII:
            for contexts in CONTEXTS:
                ident=key(radius,contexts); table=fit_tables(fold_counts(train.sum(0),radius,contexts=contexts)); tables[ident]=table
                if not np.array_equal(saved[ident+'_freq'],table['freq']) or not np.array_equal(saved[ident+'_cdf'],table['cdf']):
                    raise ValueError('frequencies are not exact TRAIN-only fit')
                v=cost(fold_counts(validation.sum(0),radius,contexts=contexts),table)
                t=cost(fold_counts(train.sum(0),radius,contexts=contexts),table)
                candidates.append(dict(radius=radius,contexts=contexts,table_sha256=table['table_sha256'],
                    validation_total_bits=v['total_bits'],validation_bpv=v['bpv'],train_bpv=t['bpv'],table=table_metadata(table)))
    if candidates!=result['candidates']: raise ValueError('validation grid/costs differ')
    selected=select_candidate(candidates); selection=result['selection']
    if any(selection[k]!=v for k,v in selected.items()) or selection['test_not_scored_for_selection'] is not True:
        raise ValueError('selected config is not validation-only minimum')
    selection_path=checked(run,'selection.json')
    if (json.loads(selection_path.read_text())!=selection or sha(selection_path)!=result['selection_sha256'] or
            selection['table_file_sha256']!=result['train_table_sha256'] or selection['train_documents']!=protocol['train'] or
            selection['validation_document']!=roles['validation'][0]): raise ValueError('frozen selection receipt differs')
    radius,contexts=selected['radius'],selected['contexts']
    expected_ids={(n,'rank_delta_'+label) for n in entries for label in ('selected','unconditional_control')}
    if len(result['rows'])!=len(expected_ids) or {(r['document'],r['method']) for r in result['rows']}!=expected_ids:
        raise ValueError('TEST scoring includes unselected configurations')
    for row in result['rows']:
        ct=contexts if row['method']=='rank_delta_selected' else 1; table=tables[key(radius,ct)]
        actual=cost(fold_counts(counts[row['document']].sum(0),radius,contexts=ct),table)
        if row['cost']!=actual or row['radius']!=radius or row['contexts']!=ct or row['table_sha256']!=table['table_sha256']:
            raise ValueError('selected TEST symbol cost differs')
        geometry=dict(values=row['values'],state_bytes=row['projected_state_bytes'],directory_bytes=row['directory_bytes'])
        if geometry['values']!=entries[row['document']]['values'] or geometry['state_bytes']!=16*geometry['directory_bytes']:
            raise ValueError('V64/B32 rANS state/directory geometry differs')
        from kv_compare.causal_packet import Packet
        packet=Packet.restore(Path(entries[row['document']]['packet']).read_bytes(),'cpu')
        if geometry['directory_bytes']!=packet.starts.numel()*8: raise ValueError('program directory projection differs')
        projected=projected_packet(actual,geometry,packet.header,radius,ct,table['table_sha256'])
        if any(row[k]!=v for k,v in projected.items()): raise ValueError('hypothetical packet accounting differs')
    import statistics
    estimates=[]
    for method in ('rank_delta_selected','rank_delta_unconditional_control'):
        group=[r for r in result['rows'] if r['method']==method]
        estimates.append(dict(method=method,estimated_packet_bytes=statistics.mean(r['estimated_packet_bytes'] for r in group),
            estimated_bpv=statistics.mean(r['estimated_bpv'] for r in group),
            estimated_saved_percent=statistics.mean(r['estimated_saved_percent'] for r in group),actual_new_packet=False))
    if estimates!=result['estimates'] or required_budgets(estimates,[b for b in expected_baselines if b['method'] in BASELINES])!=result['budgets']:
        raise ValueError('estimated averages/required decoder budgets differ')
    test_counts=sum((counts[n] for n in entries)); test_escapes=sum((escape[n] for n in entries))
    if layer_insights(test_counts,tables[key(radius,contexts)],tables[key(radius,1)],radius,test_escapes)!=result['insights']:
        raise ValueError('per-layer insight counters differ')
    selected_table=tables[key(radius,contexts)]; accounting=result['table_accounting']
    expected=dict(actual_frequency_array_bytes=int(selected_table['freq'].nbytes),actual_cdf_array_bytes=int(selected_table['cdf'].nbytes),
        actual_table_arrays_bytes=int(selected_table['freq'].nbytes+selected_table['cdf'].nbytes),all_candidate_npz_bytes=table_path.stat().st_size,
        projected_gpu_frequency_bytes=int(selected_table['freq'].nbytes),projected_gpu_cdf_bytes=int(selected_table['cdf'].nbytes),
        projected_gpu_inverse_uint16_bytes=contexts*65536*2,
        projected_gpu_total_table_bytes=int(selected_table['freq'].nbytes+selected_table['cdf'].nbytes)+contexts*65536*2)
    if any(accounting[k]!=v for k,v in expected.items()): raise ValueError('actual/projected table accounting differs')
    predictor_bytes={r['shared_predictor_gpu_bytes'] for r in parent['rows'] if r['method']==REFERENCE}
    if predictor_bytes!={accounting['predictor_gpu_bytes']}: raise ValueError('frozen predictor memory differs')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    parser.add_argument('--destination',required=True,type=Path); args=parser.parse_args(argv)
    run=args.run.resolve(); dest=args.destination
    if dest.exists(): raise ValueError('fresh destination required')
    result=json.loads(checked(run,'results.json').read_text()); verify(run,result)
    names=[n for n in ('results.json','summary.md','source_manifest.json','selection.json','console.log') if (run/n).exists()]
    sources={name:checked(run,name) for name in names}; dest.mkdir(parents=True)
    manifest=dict(status=result['status'],gpu_measured=result.get('gpu_measured',False),files={},
        scope='small estimated-rate evidence only; no source captures, counts, tables, packets, fixed state, models or traces')
    for name,source in sources.items():
        target=dest/name; shutil.copyfile(source,target)
        manifest['files'][name]=dict(bytes=source.stat().st_size,sha256=sha(source))
    write_path=dest/'package_manifest.json'; write_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(dest/'summary.md'); return 0


if __name__=='__main__': raise SystemExit(main())
