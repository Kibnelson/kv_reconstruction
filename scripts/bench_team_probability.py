#!/usr/bin/env python3
"""Calibrate fixed-prior head spreads using actual GPU packet bytes.

Only the separate TABLES document selects the spreads. Predictor tensors,
inverse surprise scales, native implementations and the decoder remain fixed.
TEST packets are encoded only after the selected state has been committed to
an immutable, hash-bound state directory.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_team_packet import identity, cuda_check, check_original, collect_source_inputs
from scripts.bench_team_register import (BASELINES, OWN, METHODS, LENGTHS, LINKS,
    aggregate, link_models, sample_order, valid_receiver_request)
from scripts.bench_team_gpu import worker_environment
from scripts.bench_lite_baselines import digest, launch, write
from scripts.bench_llama_causal import log, timed
from scripts.bench_lite_allocation import runtime_snapshot, pcie_metadata

FILES = ('scripts/bench_team_probability.py', 'scripts/package_team_probability.py',
         'scripts/run_team_probability_a4500.sh', 'tests/test_team_probability_runner.py',
         'kv_compare/team_probability.py', 'tests/test_team_probability.py')
VARIANTS = ('control', 'calibrated')


def decode_arm(tokens):
    if tokens not in LENGTHS:
        raise ValueError('only the frozen 512/8192 request lengths are allowed')
    return 'graph_reference' if tokens == 512 else 'direct_reference'


def read_parent(directory):
    from scripts.package_team_register import verify as verify_register
    directory = Path(directory).resolve()
    parent = json.loads((directory/'results.json').read_text())
    if (parent.get('format') != 'team_register_v1' or parent.get('status') != 'completed' or
            not parent.get('gpu_measured') or parent.get('recipe', {}).get('decoder_warps') != 1):
        raise ValueError('completed actual-GPU one-warp register parent required')
    verify_register(directory, parent)
    recipe = parent['recipe']; inherited = {**recipe['unchanged_code'], **recipe['code']}
    if len(inherited) != 90:
        raise ValueError('all 90 frozen source identities are required')
    selections = json.loads(json.dumps(recipe['selections']))
    for q in selections.values():
        q['arms'] = [decode_arm(q['tokens'])] if q['method'] in OWN else ['native']
    receipts = {p:s for j in parent['jobs'].values() for p,s in j['artifacts'].items()}
    receipts.update({n:digest(directory/n) for n in ('source_manifest.json', 'cuda_tests.log')})
    return parent, inherited, selections, receipts


def capture_recipe(source_run, original_manifest):
    """Read metadata only here; no TEST KV is opened before state selection."""
    from kv_compare.causal_study import MODEL, REVISION
    root = Path(source_run).resolve()/'captures'
    source = original_manifest['metadata']['capture_source']
    if digest(root/'meta.json') != source['meta_sha256']:
        raise ValueError('original capture metadata changed')
    meta = json.loads((root/'meta.json').read_text())
    if (meta.get('model') != MODEL or meta.get('revision') != REVISION or meta.get('layers') != 28 or
            source['model'] != MODEL or source['revision'] != REVISION or
            set(source['documents']) != {*(f'train{i}' for i in range(6)), 'tables', 'test_text', 'test_code'} or
            any(d['tokens'] != 8192 for d in source['documents'].values())):
        raise ValueError('original six TRAIN / TABLES / two TEST Llama captures required')
    return dict(root=str(root), source=source, calibration_document='tables',
                calibration_tokens=8192, tests=['test_text', 'test_code'],
                test_selection='none; fixed spread vectors committed before TEST arrays are opened')


def calibration_arrays(capture):
    import numpy as np
    from kv_compare.causal_study import array_word_digest
    root = Path(capture['root']); name = capture['calibration_document']
    if name != 'tables' or digest(root/'meta.json') != capture['source']['meta_sha256']:
        raise ValueError('calibration must use the frozen TABLES document')
    K = np.load(root/(name+'_K_bits.npy'), mmap_mode='r', allow_pickle=False)
    V = np.load(root/(name+'_V_bits.npy'), mmap_mode='r', allow_pickle=False)
    ids = np.load(root/(name+'_ids.npy'), allow_pickle=False)
    expected = capture['source']['documents'][name]
    if (K.dtype != np.uint16 or V.dtype != np.uint16 or K.shape != (28,8192,1024) or V.shape != K.shape or
            len(ids) != 8192 or array_word_digest(K,V) != expected['words_sha256'] or
            hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() != expected['token_ids_sha256']):
        raise ValueError('calibration words or token IDs changed')
    h=hashlib.sha256()
    for layer in range(28):
        for start in range(0,8192,512):
            h.update(np.concatenate((K[layer,start:start+512],V[layer,start:start+512]),axis=1).astype('<u2',copy=False).tobytes())
    entry = dict(document='tables', shape=[28,8192,2048], values=28*8192*2048,
                 original_word_sha256=h.hexdigest())
    return K,V,entry


def load_calibration_words(q, torch, device):
    import numpy as np
    K,V,entry=calibration_arrays(q['capture'])
    words = np.empty((28,8192,2048), dtype='<u2'); words[:,:,:1024] = K; words[:,:,1024:] = V
    return torch.from_numpy(words.view(np.int16)).to(device), entry


def _same_tensors(a, b, torch):
    if set(a) != set(b):
        return False
    return all((type(a[k]) is type(b[k]) and a[k] == b[k] if not torch.is_tensor(a[k]) else
                torch.is_tensor(b[k]) and a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and
                torch.equal(a[k].contiguous().view(torch.uint8),b[k].contiguous().view(torch.uint8))) for k in a)


def save_or_check_state(path, consts, rank, metadata, torch):
    from kv_compare.team_packet import save_state, load_state, source_identity
    path = Path(path)
    if (path/'manifest.json').exists():
        c, r, manifest = load_state(path, 'cpu')
        wanted = dict(metadata, format='team_onestep_state_v1', vendor_source=source_identity(), rank=256)
        if (manifest['metadata'] != wanted or not _same_tensors(c,consts,torch) or
                not _same_tensors(r,rank,torch)):
            raise ValueError('resumed probability state differs')
        return manifest
    return save_state(path, consts, rank=rank, metadata=metadata)


def codec_spec(path, manifest, method):
    return dict(kind='team', state=str(Path(path).resolve()), arm=method.removeprefix('team_'),
                state_sha256=manifest['state_sha256'], manifest_sha256=digest(Path(path)/'manifest.json'))


def state_storage(path):
    """Actual cold-state files, separate from warm per-request packet bytes."""
    path=Path(path)
    files={name:dict(bytes=(path/name).stat().st_size,sha256=digest(path/name))
           for name in ('tensors.npz','manifest.json')}
    return dict(path=str(path.resolve()),files=files,total_bytes=sum(x['bytes'] for x in files.values()))


def probability_codec(spec, device, tokens=8192):
    from kv_compare.team_lookup import TeamLookupCodec
    if digest(Path(spec['state'])/'manifest.json') != spec['manifest_sha256']:
        raise ValueError('probability state manifest changed')
    return TeamLookupCodec(spec['state'], device, spec['arm'],
                           decoder_mode='graph' if tokens == 512 else 'direct', lookup=False)


def _release(torch):
    """Collect storage after the caller synchronizes and drops its codec.

    TeamLookupCodec owns ordinary tensors and graphs, with no close() API.
    The caller must drop its last reference before collecting cached storage.
    """
    gc.collect()
    torch.cuda.empty_cache()


def segment_receipt(packet):
    """Hash real segment bytes outside timing; retain no bulk host payload."""
    packet.validate()
    payload=packet.payload.detach().contiguous().cpu().numpy()
    raw=memoryview(payload).cast('B');offsets=packet.offsets.tolist()
    return dict(segment_words=[b-a for a,b in zip(offsets,offsets[1:])],
        segment_sha256=[hashlib.sha256(raw[2*a:2*b]).hexdigest() for a,b in zip(offsets,offsets[1:])],
        relative_starts=packet.relative_starts.tolist())


def selected_segments(observations, chosen, heads=16):
    by_factor={o['factor']:o for o in observations}
    if len(chosen)!=heads:raise ValueError('selected factor/head count differs')
    count=len(next(iter(by_factor.values()))['segment_words'])
    return {key:[by_factor[chosen[index%heads]][key][index] for index in range(count)]
            for key in ('segment_words','segment_sha256','relative_starts')}


def calibrate_worker(q, torch, device):
    from kv_compare.team_packet import load_state
    from kv_compare.team_probability import FACTORS, scale_state, assert_spread_only, head_word_costs, choose_head_factors
    before = runtime_snapshot(torch); begin = time.perf_counter()
    out = Path(q['output']).parent; original = q['original_codec']
    if digest(Path(original['state'])/'manifest.json') != original['manifest_sha256']:
        raise ValueError('original calibration state changed')
    consts, rank, original_manifest = load_state(original['state'], 'cpu')
    G = consts['sp'].shape[1]//128
    if tuple(q['factors']) != tuple(FACTORS) or G != 16:
        raise ValueError('predeclared factor grid and 16 original heads required')
    words, entry = load_calibration_words(q, torch, device)
    states = {}
    for index, factor in enumerate(FACTORS):
        if factor == 1.:
            states[factor] = (Path(original['state']), original_manifest)
        else:
            c, r = scale_state(consts,rank,[factor]*G,[factor]*G)
            assert_spread_only(consts,rank,c,r)
            path = out/'calibration_states'/f'factor_{index}'
            metadata = dict(original_manifest['metadata'])
            for reserved in ('format','vendor_source','rank'): metadata.pop(reserved,None)
            metadata['probability_calibration'] = dict(recipe_identity=q['recipe_identity'],
                original_state_sha256=original_manifest['state_sha256'], stage='factor_grid', factor=factor,
                inverse_scales_frozen=True, calibration_document='tables')
            states[factor] = (path,save_or_check_state(path,c,r,metadata,torch))
    observations = {}; selections = {}; prior_receipts = {}
    for method in OWN:
        observations[method] = []; reference_features = reference_prior = None
        # Run the exact original factor first so all equality checks bind it.
        for factor in (1.,)+tuple(f for f in FACTORS if f != 1.):
            path, manifest = states[factor]; spec = codec_spec(path,manifest,method)
            codec = probability_codec(spec,device)
            log(f'calibrate TABLES actual bytes: {method}, factor {factor:g}')
            features, prior = codec.model(words)
            if reference_prior is None:
                reference_features, reference_prior = features.clone(), prior.clone()
                prior_receipts[method] = hashlib.sha256(prior.detach().cpu().numpy().tobytes()).hexdigest()
            elif not torch.equal(features,reference_features) or not torch.equal(prior,reference_prior):
                raise AssertionError('spread-only intervention changed features or adaptive priors')
            packet = codec.encode_prepared(words,features,prior)
            value = codec.decode(packet); check_original(value,entry)
            segments = segment_receipt(packet)
            observations[method].append(dict(factor=factor, head_words=head_word_costs(packet),
                **segments, complete_packet_bytes=packet.packet_bytes,
                payload_bytes=packet.payload_bytes, state_sha256=manifest['state_sha256'],
                original_exact=True, features_equal_control=True, priors_equal_control=True))
            del value,packet,features,prior
            torch.cuda.synchronize(device)
            del codec
            _release(torch)
        del reference_features,reference_prior
        selections[method] = choose_head_factors(observations[method],factors=FACTORS)
    c, r = scale_state(consts,rank,selections[OWN[0]]['factors'],selections[OWN[1]]['factors'])
    assert_spread_only(consts,rank,c,r)
    metadata = dict(original_manifest['metadata'])
    for reserved in ('format','vendor_source','rank'): metadata.pop(reserved,None)
    metadata['probability_calibration'] = dict(recipe_identity=q['recipe_identity'],
        original_state_sha256=original_manifest['state_sha256'], stage='selected_head_factors',
        factor_grid=list(FACTORS), selections=selections, observation_sha256=identity(observations),
        inverse_scales_frozen=True, calibration_document='tables', test_selection=False)
    selected_path = out/'calibrated_state'
    selected_manifest = save_or_check_state(selected_path,c,r,metadata,torch)
    selected_checks = {}
    for method in OWN:
        spec = codec_spec(selected_path,selected_manifest,method); codec = probability_codec(spec,device)
        packet = codec.encode(words); value = codec.decode(packet); check_original(value,entry)
        chosen = selections[method]['factors']; by_factor = {o['factor']:o for o in observations[method]}
        expected = selected_segments(observations[method],chosen,G)
        actual = segment_receipt(packet)
        baseline = by_factor[1.]['complete_packet_bytes']
        if actual != expected or packet.packet_bytes > baseline:
            raise AssertionError('head-selected reencode differs from independent calibrated streams')
        selected_checks[method] = dict(codec=spec, **actual, head_words=head_word_costs(packet),
            complete_packet_bytes=packet.packet_bytes, payload_bytes=packet.payload_bytes,
            calibration_control_packet_bytes=baseline, exact_segment_composition=True, original_exact=True)
        del packet,value
        torch.cuda.synchronize(device)
        del codec
        _release(torch)
    del words
    after = runtime_snapshot(torch)
    if before != after: raise AssertionError('calibration runtime policy changed')
    write(q['output'],dict(status='completed',gpu_measured=True,recipe_identity=q['recipe_identity'],
        calibration_entry=entry, calibration_source=q['capture'], factor_grid=list(FACTORS),
        observations=observations, selections=selections, selected_checks=selected_checks,
        adaptive_prior_sha256=prior_receipts, inverse_scales_frozen=True, predictor_tensors_frozen=True,
        no_test_kv_loaded=True, selected_state=str(selected_path),
        selected_state_sha256=selected_manifest['state_sha256'], selected_manifest_sha256=digest(selected_path/'manifest.json'),
        state_storage=dict(original=state_storage(original['state']),selected=state_storage(selected_path),
            grid={str(f):state_storage(path) for f,(path,_manifest) in states.items()}),
        runtime_policy=before,runtime_policy_after=after,wall_ms=(time.perf_counter()-begin)*1000))


def encode_worker(q, torch, device):
    import numpy as np
    from kv_compare.team_packet import TeamPacket
    before = runtime_snapshot(torch); out = Path(q['output']); out.mkdir(parents=True,exist_ok=True)
    if digest(Path(q['selected_state'])/'manifest.json') != q['selected_manifest_sha256']:
        raise ValueError('selected probability state changed before TEST encoding')
    records = []
    for variant in VARIANTS:
        spec = q['codecs'][variant]; codec = probability_codec(spec,device,q['tokens'])
        for e in q['inputs']:
            if digest(e['path']) != e['file_sha256']: raise ValueError('original TEST array changed')
            source = torch.from_numpy(np.load(e['path'],allow_pickle=False).view(np.int16).copy()).to(device)
            check_original(source,e)
            packet,measurement = timed(lambda:codec.encode(source),torch,device,1,3)
            start = time.perf_counter(); blob = packet.export(); export_ms=(time.perf_counter()-start)*1000
            actual = hashlib.sha256(blob).hexdigest(); old = next(x for x in q['control_entries'] if x['document']==e['document'])
            if variant == 'control' and (actual != old['packet_sha256'] or len(blob) != old['packet_bytes']):
                raise AssertionError('fresh control encode differs from original frozen packet')
            path = out/(variant+'_'+e['document']+'.kvp')
            if path.exists() and (digest(path) != actual or path.stat().st_size != len(blob)):
                raise ValueError('resumed TEST packet changed')
            if not path.exists(): path.write_bytes(blob)
            entry = dict(id=q['method']+'_'+variant+'_'+e['document'], document=e['document'],
                shape=e['shape'],values=e['values'],packet=str(path),packet_sha256=actual,packet_bytes=len(blob),
                original_word_sha256=e['original_word_sha256'],complete_bpv=8*len(blob)/e['values'],
                state_sha256=spec['state_sha256'])
            imported = TeamPacket.import_bytes(blob,device=device)
            if q['tokens'] == 512: codec.graph_prepare(imported)
            value=codec.decode(imported);check_original(value,e)
            records.append(dict(variant=variant,entry=entry,encode=measurement,serialized_export_wall_ms=export_ms,
                original_exact=True,control_matches_frozen=(variant=='control'),codec=spec))
            log(f"TEST {q['method']} / {variant} / {e['document']}: {entry['complete_bpv']:.6f} bpv")
            del packet,source,value,imported,blob;gc.collect();torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        del codec
        _release(torch)
    after=runtime_snapshot(torch)
    if before != after:raise AssertionError('TEST encode runtime policy changed')
    write(out/'encoder_results.json',dict(status='completed',gpu_measured=True,recipe_identity=q['recipe_identity'],
        method=q['method'],tokens=q['tokens'],records=records,runtime_policy=before,runtime_policy_after=after,
        selected_manifest_sha256=q['selected_manifest_sha256'],selection_was_frozen_before_test=True,
        scope='full encode API including model pass and compact segment packing; serialized export is separately timed; fixed state fitting/calibration excluded'))


def selected_plan(selections, encoded):
    natives=[(name,q) for name,q in selections.items() if q['method'] in BASELINES]
    own=[]
    for index,(name,q) in enumerate((n,q) for n,q in selections.items() if q['method'] in OWN):
        order=VARIANTS if index%2==0 else VARIANTS[::-1]
        records=encoded[name]['records']
        for variant in order:
            chosen=[r for r in records if r['variant']==variant]
            if len(chosen)!=2:raise ValueError('two encoded TEST documents per variant required')
            codec=chosen[0]['codec']
            if any(r['codec']!=codec for r in chosen):raise ValueError('variant state differs across documents')
            own.append((variant+'_'+name,dict(method=q['method'],tokens=q['tokens'],codec=codec,
                entries=[r['entry'] for r in chosen],arms=[decode_arm(q['tokens'])],round=variant)))
    return ([(f'before_{n}',dict(q,round='before')) for n,q in natives]+own+
            [(f'after_{n}',dict(q,round='after')) for n,q in reversed(natives)])


def report(r,out):
    lines=['# Fixed-prior head-spread calibration versus GPU baselines','',
        f"Status: **{r['status']}**. GPU measured: **{r.get('gpu_measured',False)}**.",'',
        'Llama-3.2-3B, all 28 original BF16 layers. Only TABLES selects per-head spread factors from 0.5/0.75/1/1.25/1.5 using actual compact GPU packet words. Predictor tensors, inverse surprise scales and decoder kernel structure/launch configuration are unchanged. Data-dependent coding work may change. TEST never selects parameters.',
        'Both variants use the one-warp reference binary-search decoder: graph for 512 tokens, direct for 8192. Native controls retain their original implementations and settings.','',
        '| Tokens | Method | Variant / round | Complete bpv | Packet MiB | Decode ms | Exact |',
        '|---:|---|---|---:|---:|---:|---|']
    for a in r.get('aggregates',[]):
        lines.append(f"| {a['tokens']} | {a['method']} | {a['round']} | {a['complete_bpv']:.6f} | {a['packet_bytes']/2**20:.3f} | {a['decode_ms']:.3f} | {a['original_exact']} |")
    if r.get('calibration'):
        lines += ['','TABLES-selected fixed factors (K heads then V heads; shared state, no per-request hints):']
        for method,item in r['calibration']['selections'].items():lines.append(f"- {method}: `{item['factors']}`")
        storage=r['calibration']['state_storage']
        lines += ['',f"Cold fixed-state archive and manifest: control {storage['original']['total_bytes']/2**20:.3f} MiB; calibrated {storage['selected']['total_bytes']/2**20:.3f} MiB. Warm link models below assume this state is already shared; a cold state transfer must additionally charge these bytes."]
    lines += ['','Calculated serial wire + measured full GPU decode, milliseconds. Networking is not measured. Ideal overlap remains a separate bound in results.json.','',
        '| Tokens | Method | Variant / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for a in r.get('aggregates',[]):
        costs=[a['decode_ms']+8*a['packet_bytes']/(g*1e6) for g in LINKS]
        lines.append(f"| {a['tokens']} | {a['method']} | {a['round']} | "+' | '.join(f'{x:.3f}' for x in costs)+' |')
    if r.get('encodes'):
        lines += ['','Own full encode means of two document medians (export separately recorded; native encode not remeasured):','',
            '| Tokens | Method | Variant | Encode ms |','|---:|---|---|---:|']
        for name,e in r['encodes'].items():
            for variant in VARIANTS:
                records=[x for x in e['records'] if x['variant']==variant]
                lines.append(f"| {e['tokens']} | {e['method']} | {variant} | {statistics.mean(x['encode']['wall_median_ms'] for x in records):.3f} |")
    if r.get('error'):lines += ['','Failure: '+r['error']]
    lines += ['','Primary decode starts with compressed payload on GPU and includes normal API validation/metadata work and an independently owned exact output. Bulk host hashing/import/upload, disk, real networking and calibration are excluded. Complete packet bytes include framing, directories, states, alignment and checksum. Fixed shared state and scratch are additional.',
        'No fresh end-to-end native encode comparison, NIC/SSD, compressed attention, T-to-P conversion or serving result is claimed. Smaller TEST packets are a hypothesis, not guaranteed by TABLES selection.']
    (Path(out)/'summary.md').write_text('\n'.join(lines)+'\n')


def _artifacts(directory,names,out):
    return {str((directory/n).relative_to(out)):digest(directory/n) for n in names}


def verify_job(run,name,job,base,recipe):
    from scripts.package_team_register import (checked,check_samples,check_packet_immutability,
        check_inverse_proof,check_kernel_provenance,check_accounting)
    expected_artifacts={name+'/'+f for f in ('request.json','worker_results.json','receiver.json','receiver_request.json')}
    if job.get('status')!='completed' or set(job['artifacts'])!=expected_artifacts:
        raise ValueError('GPU job receipt coverage differs')
    for p,wanted in job['artifacts'].items():
        if digest(checked(run,p))!=wanted:raise ValueError('GPU job artifact changed')
    q=json.loads(checked(run,name+'/request.json').read_text())
    expected=dict(base,decoder_warps=1,output=str(run/name),warmup=recipe['warmup'],repeats=recipe['repeats'])
    if q!=expected:raise ValueError('GPU job request differs from selected packets')
    data=json.loads(checked(run,name+'/worker_results.json').read_text())
    proof_path=checked(run,name+'/receiver.json');proof=json.loads(proof_path.read_text())
    receiver=json.loads(checked(run,name+'/receiver_request.json').read_text());valid_receiver_request(receiver)
    if receiver!={**{k:q[k] for k in ('method','codec','entries','arms','round','decoder_warps')},'output':str(run/name/'receiver.json')}:
        raise ValueError('packet-only receiver request differs')
    own=q['method'] in OWN; graphs={a:int(a.startswith('graph_')) for a in q['arms']} if own else {}
    expected_ids={q['round']+'_'+a+'_'+e['id'] for a in q['arms'] for e in q['entries']}
    checks={c['id']:c for c in proof['checks']};entries={e['id']:e for e in q['entries']}
    if (data['status']!='completed' or not data.get('gpu_measured') or not proof.get('gpu_measured') or
            data['codec_provenance']!=q['codec'] or data['rows']!=job['rows'] or
            data['method']!=q['method'] or data['round']!=q['round'] or data['device']!=recipe['device'] or
            data['receiver_sha256']!=digest(proof_path) or
            data['graph_capture_counts']!=graphs or proof['graph_capture_counts']!=graphs or
            not proof.get('no_source_kv_or_predictions') or not proof.get('alternating_packets') or
            not proof.get('independent_returned_outputs') or set(checks)!=expected_ids or
            len(proof['checks'])!=len(expected_ids) or {r['id'] for r in job['rows']}!=expected_ids or
            len(job['rows'])!=len(expected_ids)):
        raise ValueError('actual-GPU independent receiver proof differs')
    policy=data['runtime_policy']
    if (data['runtime_policy_after']!=policy or proof['runtime_policy']!=policy or proof['runtime_policy_after']!=policy or
            policy['deterministic_algorithms_enabled']):raise ValueError('codec runtime policy changed')
    if data['sample_order']!=sample_order(q['entries'],q['arms'],recipe['warmup'],recipe['repeats']):
        raise ValueError('declared document/sample order differs')
    for obj in (data,proof):
        check_packet_immutability(obj['packet_immutability'],q['entries'])
        wanted=set(q['arms']) if own else set()
        if set(obj['inverse_proofs'])!=wanted or set(obj['kernel_provenance'])!=wanted:
            raise ValueError('inverse/kernel proof coverage differs')
        for arm in wanted:
            check_inverse_proof(obj['inverse_proofs'][arm]);check_kernel_provenance(obj['kernel_provenance'][arm],arm,1)
    for row in job['rows']:
        entry=entries[row['parent_id']];check=checks[row['id']]
        if (row['method']!=q['method'] or row['round']!=q['round'] or row['arm'] not in q['arms'] or
                row['id']!=q['round']+'_'+row['arm']+'_'+entry['id'] or
                row['tokens']!=q['tokens'] or not row.get('original_exact') or row['status']!='verified' or
                row['decoded_word_sha256']!=entry['original_word_sha256'] or
                any(row[k]!=entry[k] for k in ('document','shape','values','packet','packet_sha256','packet_bytes',
                    'original_word_sha256','complete_bpv','state_sha256')) or
                not check['verified'] or check['word_sha256']!=entry['original_word_sha256'] or
                check['packet_sha256']!=entry['packet_sha256'] or check['shape']!=entry['shape'] or
                row['complete_bpv']!=8*row['packet_bytes']/row['values']):
            raise ValueError('exact TEST words or complete packet bytes differ')
        check_samples(row['decode'],recipe['warmup'],recipe['repeats'])
        if own and row['arm'].startswith('graph_'):check_samples(row['kernel_decode'],1,recipe['repeats'])
        elif 'kernel_decode' in row:raise ValueError('direct/native cannot claim captured graph timing')
        if own:check_accounting(row['accounting'],row,1)
    return True


def verify_calibration(run,calibration,request):
    from kv_compare.team_probability import FACTORS, choose_head_factors, scale_state, assert_spread_only
    from kv_compare.team_packet import load_state
    import torch
    if (calibration['status']!='completed' or not calibration.get('gpu_measured') or
            calibration['recipe_identity']!=request['recipe_identity'] or
            calibration['calibration_source']!=request['capture'] or calibration['factor_grid']!=list(FACTORS) or
            not calibration.get('no_test_kv_loaded') or not calibration.get('inverse_scales_frozen') or
            not calibration.get('predictor_tensors_frozen') or
            calibration['runtime_policy']!=calibration['runtime_policy_after'] or
            calibration['runtime_policy']['deterministic_algorithms_enabled']):
        raise ValueError('calibration protocol or actual-GPU proof differs')
    consts,rank,original=load_state(request['original_codec']['state'],'cpu')
    _K,_V,entry=calibration_arrays(request['capture'])
    if calibration['calibration_entry']!=entry:
        raise ValueError('calibration original-word digest differs from frozen TABLES')
    del _K,_V
    if Path(calibration['selected_state']).resolve()!=run/'calibrated_state':
        raise ValueError('selected state must be in the current run')
    if calibration['state_storage']['original']!=state_storage(request['original_codec']['state']):
        raise ValueError('original cold-state storage receipt differs')
    checks=calibration['selected_checks'];selections=calibration['selections']
    if set(checks)!=set(OWN) or set(selections)!=set(OWN) or set(calibration['observations'])!=set(OWN):
        raise ValueError('calibration method coverage differs')
    for method in OWN:
        observations=calibration['observations'][method]
        if choose_head_factors(observations,factors=FACTORS)!=selections[method]:
            raise ValueError('selected factors differ from actual TABLES words')
        by_factor={o['factor']:o for o in observations};chosen=selections[method]['factors']
        for item in observations:
            seg=item['segment_words'];heads=[sum(seg[g::16]) for g in range(16)]
            if (len(seg)!=4096 or item['head_words']!=heads or
                    any(type(v) is not int or v<256 for v in seg) or
                    len(item['segment_sha256'])!=4096 or len(item['relative_starts'])!=4096 or
                    any(type(v) is not int or not 256<=v<=263 for v in item['relative_starts']) or
                    any(not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h) for h in item['segment_sha256']) or
                    item['payload_bytes']!=2*sum(seg) or item['complete_packet_bytes']<=item['payload_bytes'] or
                    not item.get('original_exact') or not item.get('features_equal_control') or not item.get('priors_equal_control')):
                raise ValueError('actual calibration byte/exactness coverage differs')
        expected=selected_segments(observations,chosen)
        check=checks[method]
        if (any(check[k]!=v for k,v in expected.items()) or
                check['head_words']!=[sum(expected['segment_words'][g::16]) for g in range(16)] or
                check['payload_bytes']!=2*sum(expected['segment_words']) or
                check['complete_packet_bytes']>by_factor[1.]['complete_packet_bytes'] or
                check['calibration_control_packet_bytes']!=by_factor[1.]['complete_packet_bytes'] or
                not check.get('exact_segment_composition') or not check.get('original_exact')):
            raise ValueError('selected head composition differs')
    for factor in FACTORS:
        storage=calibration['state_storage']['grid'].get(str(factor))
        path=Path(request['original_codec']['state']) if factor==1. else run/'calibration_states'/f'factor_{list(FACTORS).index(factor)}'
        if storage!=state_storage(path):raise ValueError('grid cold-state storage receipt differs')
        c,r,manifest=load_state(path,'cpu');expected_c,expected_r=scale_state(consts,rank,[factor]*16,[factor]*16)
        assert_spread_only(consts,rank,c,r)
        if not _same_tensors(c,expected_c,torch) or not _same_tensors(r,expected_r,torch):
            raise ValueError('grid state differs from fixed spread factor')
        for method in OWN:
            item=next(o for o in calibration['observations'][method] if o['factor']==factor)
            if item['state_sha256']!=manifest['state_sha256']:raise ValueError('grid observation state binding differs')
    c,r,manifest=load_state(calibration['selected_state'],'cpu')
    expected_c,expected_r=scale_state(consts,rank,selections[OWN[0]]['factors'],selections[OWN[1]]['factors'])
    assert_spread_only(consts,rank,c,r)
    if (not _same_tensors(c,expected_c,torch) or not _same_tensors(r,expected_r,torch) or
            manifest['state_sha256']!=calibration['selected_state_sha256'] or
            digest(Path(calibration['selected_state'])/'manifest.json')!=calibration['selected_manifest_sha256']):
        raise ValueError('selected state tensors differ from calibrated factors')
    if calibration['state_storage']['selected']!=state_storage(calibration['selected_state']):
        raise ValueError('selected cold-state storage receipt differs')
    meta=manifest['metadata']['probability_calibration']
    if (meta['recipe_identity']!=request['recipe_identity'] or meta['original_state_sha256']!=original['state_sha256'] or
            meta['selections']!=selections or meta['observation_sha256']!=identity(calibration['observations']) or
            meta['test_selection'] is not False or meta['inverse_scales_frozen'] is not True):
        raise ValueError('immutable selected-state calibration binding differs')
    for method in OWN:
        if checks[method]['codec']!=codec_spec(calibration['selected_state'],manifest,method):
            raise ValueError('selected codec state differs')
    return True


def verify(run,result):
    from scripts.package_team_register import checked,check_samples
    run=Path(run).resolve();completed=result['status']=='completed';recipe=result.get('recipe')
    if result.get('format')!='team_probability_v1':raise ValueError('wrong probability report format')
    if completed and not result.get('gpu_measured'):raise ValueError('CPU run cannot claim completed GPU evidence')
    if not recipe:
        if completed:raise ValueError('completed probability run lacks recipe')
        return True
    if identity(recipe)!=result.get('identity') or json.loads(checked(run,'source_manifest.json').read_text())!=recipe:
        raise ValueError('probability recipe binding differs')
    if recipe['code']!={name:digest(ROOT/name) for name in FILES}:
        raise ValueError('new probability sources changed')
    for name,wanted in recipe['unchanged_code'].items():
        if digest(ROOT/name)!=wanted:raise ValueError('frozen measured source changed: '+name)
    parent,inherited,selections,receipts=read_parent(recipe['parent_run'])
    if (digest(Path(recipe['parent_run'])/'results.json')!=recipe['parent_results_sha256'] or
            inherited!=recipe['unchanged_code'] or selections!=recipe['selections'] or receipts!=recipe['parent_artifacts'] or
            recipe['software']!=parent['recipe']['software'] or recipe['native_receipt']!=parent['recipe']['native_receipt'] or
            recipe['decoder_warps']!=1 or recipe['lengths']!=list(LENGTHS) or recipe['variants']!=list(VARIANTS) or
            recipe['warmup']<1 or recipe['repeats']<8):
        raise ValueError('frozen parent/protocol differs')
    if completed:
        gate=checked(run,'cuda_tests.log').read_text();last=gate.rsplit('\nRan ',1)[-1]
        if '\nRan ' not in gate or '\nOK' not in last or 'skipped' in last or 'FAILED' in last:
            raise ValueError('actual CUDA gate must pass without skips')
    calibration=result.get('calibration')
    if calibration:
        if digest(checked(run,'calibration.json'))!=result['calibration_sha256']:
            raise ValueError('calibration receipt changed')
        request=json.loads(checked(run,'calibration_request.json').read_text())
        if request!=calibration_request(recipe,result['identity'],run):raise ValueError('calibration request changed')
        if json.loads(checked(run,'calibration.json').read_text())!=calibration:raise ValueError('calibration result differs')
        verify_calibration(run,calibration,request)
    elif completed:raise ValueError('completed run lacks actual-byte calibration')
    encoded={}
    for name,item in result.get('encodes',{}).items():
        path=checked(run,'encode_'+name+'/encoder_results.json')
        if digest(path)!=item['receipt_sha256']:raise ValueError('TEST encoder receipt changed')
        data=json.loads(path.read_text())
        if data!={k:v for k,v in item.items() if k!='receipt_sha256'} or not data.get('gpu_measured') or data['status']!='completed':
            raise ValueError('actual GPU TEST encode differs')
        if (data['selected_manifest_sha256']!=calibration['selected_manifest_sha256'] or
                not data['selection_was_frozen_before_test'] or data['recipe_identity']!=result['identity'] or
                data['runtime_policy']!=data['runtime_policy_after']):raise ValueError('TEST selection binding differs')
        request=json.loads(checked(run,'encode_'+name+'/request.json').read_text())
        expected=encode_request(name,selections[name],calibration,recipe,result['inputs'],result['identity'],run)
        if request!=expected:raise ValueError('TEST encoder request changed')
        wanted_pairs={(v,e['document']) for v in VARIANTS for e in request['inputs']}
        actual_pairs={(r['variant'],r['entry']['document']) for r in data['records']}
        if (len(data['records'])!=len(wanted_pairs) or actual_pairs!=wanted_pairs or
                data['method']!=selections[name]['method'] or data['tokens']!=selections[name]['tokens']):
            raise ValueError('TEST encode Cartesian coverage differs')
        for record in data['records']:
            e=record['entry'];variant=record['variant'];original=next(x for x in request['inputs'] if x['document']==e['document'])
            if (variant not in VARIANTS or not record.get('original_exact') or record['codec']!=request['codecs'][variant] or
                    e['state_sha256']!=record['codec']['state_sha256'] or
                    e['original_word_sha256']!=original['original_word_sha256'] or e['shape']!=original['shape'] or
                    e['values']!=original['values'] or e['complete_bpv']!=8*e['packet_bytes']/e['values'] or
                    Path(e['packet']).stat().st_size!=e['packet_bytes'] or digest(e['packet'])!=e['packet_sha256']):
                raise ValueError('TEST packet/original word identity differs')
            check_samples(record['encode'],1,3)
            if variant=='control':
                original=next(x for x in request['control_entries'] if x['document']==e['document'])
                if e['packet_sha256']!=original['packet_sha256'] or e['packet_bytes']!=original['packet_bytes']:
                    raise ValueError('control TEST packet differs from frozen source')
        encoded[name]=data
    expected_names={name for name,q in selections.items() if q['method'] in OWN}
    if completed and set(encoded)!=expected_names:raise ValueError('TEST encoder job coverage differs')
    plan=dict(selected_plan(selections,encoded)) if set(encoded)==expected_names else {}
    if completed and (result.get('selected_jobs')!=list(plan) or set(result['jobs'])!=set(plan)):
        raise ValueError('GPU comparison job coverage differs')
    rows=[]
    for name,job in result.get('jobs',{}).items():
        if name not in plan:raise ValueError('unexpected GPU comparison job')
        verify_job(run,name,job,plan[name],recipe);rows.extend(job['rows'])
    if rows!=result['rows'] or aggregate(rows)!=result['aggregates'] or link_models(result['aggregates'])!=result['link_models']:
        raise ValueError('derived probability comparisons differ')
    return True


def calibration_request(recipe,wanted,out):
    from kv_compare.team_probability import FACTORS
    return dict(original_codec=recipe['selections'][OWN[0]+'_t512']['codec'], capture=recipe['capture'],
        factors=list(FACTORS), recipe_identity=wanted, output=str(out/'calibration.json'))


def encode_request(name,base,calibration,recipe,inputs,wanted,out):
    return dict(method=base['method'],tokens=base['tokens'],control_entries=base['entries'],
        codecs=dict(control=base['codec'],calibrated=calibration['selected_checks'][base['method']]['codec']),
        inputs=[e for e in inputs if e['tokens']==base['tokens']],selected_state=calibration['selected_state'],
        selected_manifest_sha256=calibration['selected_manifest_sha256'],recipe_identity=wanted,
        output=str(out/('encode_'+name)))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,default=ROOT/'runs/team_register_a4500_20261028_145318_1925338')
    p.add_argument('--source-run',type=Path,default=ROOT/'runs/llama32_3b_compare')
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda:0')
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--calibrate-worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--encode-worker',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args(argv);request=a.calibrate_worker or a.encode_worker
    if request:
        torch,device=cuda_check(a.device);q=json.loads(request.read_text())
        (calibrate_worker if a.calibrate_worker else encode_worker)(q,torch,device);return 0
    if a.output is None:p.error('--output required')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    result=dict(format='team_probability_v1',status='running',gpu_measured=False,jobs={},encodes={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[r for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        result['aggregates']=aggregate(result['rows']);result['link_models']=link_models(result['aggregates'])
        write(out/'results.json',result);report(result,out)
    try:
        torch,device=cuda_check(a.device)
        if a.warmup<1 or a.repeats<8:raise ValueError('warmup>=1 and repeats>=8 required')
        parent_dir=a.parent_run.resolve();parent,inherited,selections,receipts=read_parent(parent_dir)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']):raise ValueError('original main GPU/software required')
        from kv_compare.team_packet import load_state
        original=selections[OWN[0]+'_t512']['codec'];_,_,manifest=load_state(original['state'],'cpu')
        capture=capture_recipe(a.source_run,manifest)
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_artifacts=receipts,unchanged_code=inherited,code={n:digest(ROOT/n) for n in FILES},
            software=versions,device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,
            native_receipt=parent['recipe']['native_receipt'],selections=selections,capture=capture,
            variants=list(VARIANTS),lengths=list(LENGTHS),decoder_warps=1,warmup=a.warmup,repeats=a.repeats,
            selection_protocol='actual TABLES segment words; fixed inv; per-head factors; deterministic ties prefer1; TEST never selects',
            decode_scope='compressed payload already on GPU; full checked API through owned exact BF16 output',
            fresh_encode_scope='own codecs only, measured separately; no native fresh-encode ranking')
        wanted=identity(recipe)
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=wanted:raise ValueError('resume recipe differs; use a new output directory')
            result=old;writable=True
        result.update(identity=wanted,recipe=recipe,status='running',gpu_measured=True,
            hardware=dict(parent['hardware'],pcie_query=pcie_metadata()))
        result.pop('error',None);result.pop('traceback',None)
        write(out/'source_manifest.json',recipe);save()
        q=calibration_request(recipe,wanted,out)
        if not result.get('calibration'):
            write(out/'calibration_request.json',q)
            launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--calibrate-worker',out/'calibration_request.json'],out/'calibration.log')
            data=json.loads((out/'calibration.json').read_text());verify_calibration(out,data,q)
            result['calibration']=data;result['calibration_sha256']=digest(out/'calibration.json');save()
        else:
            if digest(out/'calibration.json')!=result['calibration_sha256']:raise ValueError('completed calibration receipt changed')
            verify_calibration(out,result['calibration'],q)
        # TEST arrays are first opened here, after the immutable selected state.
        inputs=collect_source_inputs(Path(capture['root']),capture['tests'],LENGTHS,out)
        for e in inputs:
            frozen=next(x for x in selections[OWN[0]+f"_t{e['tokens']}"]['entries'] if x['document']==e['document'])
            if e['original_word_sha256']!=frozen['original_word_sha256'] or e['shape']!=frozen['shape']:
                raise ValueError('TEST source differs from original matched words')
        if result.get('inputs') and result['inputs']!=inputs:raise ValueError('resumed TEST inputs differ')
        result['inputs']=inputs;save()
        for name,base in selections.items():
            if base['method'] not in OWN:continue
            directory=out/('encode_'+name);directory.mkdir(exist_ok=True)
            q=encode_request(name,base,result['calibration'],recipe,inputs,wanted,out)
            if name in result['encodes']:
                if digest(directory/'encoder_results.json')!=result['encodes'][name]['receipt_sha256']:
                    raise ValueError('completed TEST encoder receipt changed')
                if json.loads((directory/'request.json').read_text())!=q:raise ValueError('resumed encoder request differs')
                continue
            write(directory/'request.json',q)
            launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--encode-worker',directory/'request.json'],directory/'encoder.log')
            data=json.loads((directory/'encoder_results.json').read_text())
            result['encodes'][name]=dict(data,receipt_sha256=digest(directory/'encoder_results.json'));save()
        plan=selected_plan(selections,result['encodes']);result['selected_jobs']=[n for n,_ in plan];save()
        for name,base in plan:
            directory=out/name;directory.mkdir(exist_ok=True)
            q=dict(base,decoder_warps=1,output=str(directory),warmup=a.warmup,repeats=a.repeats)
            if name in result['jobs']:
                verify_job(out,name,result['jobs'][name],base,recipe);continue
            write(directory/'request.json',q);python,env=worker_environment(base,recipe['native_receipt'])
            launch([python,ROOT/'scripts/bench_team_register.py','--device',str(device),'--worker',directory/'request.json'],directory/'worker.log',env)
            data=json.loads((directory/'worker_results.json').read_text())
            job=dict(status='completed',rows=data['rows'],artifacts=_artifacts(directory,
                ('request.json','worker_results.json','receiver.json','receiver_request.json'),out))
            verify_job(out,name,job,base,recipe);result['jobs'][name]=job;save()
        result['status']='completed';save();verify(out,result)
        log('completed: '+str(out/'summary.md'));return 0
    except Exception as error:
        if writable:
            result.update(status='failed',error=str(error),traceback=traceback.format_exc());save()
        traceback.print_exc();return 1


if __name__=='__main__':raise SystemExit(main())
