#!/usr/bin/env python3
"""Matched compact packets for the team's frozen one-step codec and current Lite.

Actual CUDA is mandatory. The team arithmetic is vendored byte-for-byte; this
runner adds transport framing, isolated workers and independent receivers.
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
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_lite_baselines import digest, launch, write
from scripts.bench_llama_causal import log, timed
from scripts.bench_lite_allocation import runtime_snapshot, pcie_metadata

LINKS = (1., 100., 200., 300., 400.)
TEAM_ARMS = ('dense_exact', 'dense_uniform', 'rank256_uniform')
METHODS = ('raw_bf16', 'splitzip_official_framed', 'exans_official_fixed_framed',
           'lite_current_graph', *(f'team_{a}' for a in TEAM_ARMS))
FILES = ('scripts/bench_team_packet.py', 'scripts/package_team_packet.py',
         'scripts/run_team_packet_a4500.sh', 'tests/test_team_packet_runner.py',
         'kv_compare/team_packet.py', 'tests/test_team_packet.py',
         'kv_compare/team_onestep/__init__.py', 'kv_compare/team_onestep/core.py',
         'kv_compare/team_onestep/predict.py', 'kv_compare/team_onestep/fused.py',
         'kv_compare/team_onestep/layerwise.py', 'kv_compare/team_onestep/gpu_coder.py',
         'kv_compare/team_onestep/source_manifest.json')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def check_fit_metadata(manifest, expected):
    """save_state adds provenance fields; preserve and compare caller metadata."""
    meta=manifest.get('metadata', {})
    if (set(meta)!=set(expected)|{'format','vendor_source','rank'} or
            any(meta.get(k)!=v for k,v in expected.items()) or
            meta.get('format')!='team_onestep_state_v1' or meta.get('rank')!=256):
        raise ValueError('cached team fit belongs to another source/recipe')
    return True


def validate_lengths(value):
    lengths=tuple(int(v) for v in value.split(','))
    # The unchanged rank epilogue has unmasked128-row tiles per half-length
    # chunk. Restrict geometry before calibration rather than changing it.
    if not lengths or len(set(lengths))!=len(lengths) or any(n<256 or n>8192 or n%256 for n in lengths):
        raise ValueError('distinct lengths in256..8192 divisible by256 required')
    return lengths


def normalize_row(row):
    row = dict(row)
    row['tokens'] = int(row['shape'][1])
    row['source_document'] = row['document'].rsplit('_t', 1)[0]
    return row


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['tokens'], row['method']), []).append(row)
    result = []
    for (tokens, method), group in sorted(groups.items()):
        result.append(dict(tokens=tokens, method=method,
            documents=[r['source_document'] for r in group],
            values=statistics.mean(r['values'] for r in group),
            packet_bytes=statistics.mean(r['packet_bytes'] for r in group),
            complete_bpv=statistics.mean(r['complete_bpv'] for r in group),
            saved_percent=statistics.mean(r['saved_percent'] for r in group),
            encode_ms=statistics.mean(r['encode']['wall_median_ms'] for r in group),
            decode_ms=statistics.mean(r['decode']['wall_median_ms'] for r in group),
            kernel_decode_ms=(statistics.mean(statistics.median(s['cuda_interval_ms'] for s in r['kernel_decode']['samples']) for r in group)
                              if all('kernel_decode' in r for r in group) else None),
            original_exact=all(r['original_exact'] for r in group)))
    return result


def transfer_models(aggregates, links=LINKS):
    if any(not math.isfinite(g) or g <= 0 for g in links):
        raise ValueError('positive finite links required')
    rows = []
    for a in aggregates:
        for g in links:
            wire = 8 * a['packet_bytes'] / (g * 1e6)
            rows.append(dict(tokens=a['tokens'], method=a['method'], effective_gbps=g,
                wire_ms=wire, precompressed_serial_ms=wire+a['decode_ms'],
                ideal_overlap_ms=max(wire, a['decode_ms']),
                fresh_serial_ms=a['encode_ms']+wire+a['decode_ms'], measured_network=False))
    return dict(measured_network=False,
        scope='calculated effective-link models; full GPU receiver latency; serialization/export, host upload, storage and real overlap excluded',
        links=list(links), rows=rows)


def check_original(output, entry):
    from kv_compare.causal_packet import word_sha
    actual = word_sha(output)
    if list(output.shape) != entry['shape'] or actual != entry['original_word_sha256']:
        raise AssertionError('decoded words differ from original BF16: '+entry['document'])
    return actual


def cuda_check(name, lite=False):
    if lite:
        from scripts.bench_llama_causal import device_check
    else:
        from scripts.bench_llama_baselines import device_check
    torch, device = device_check(name)
    if device.type != 'cuda':
        raise RuntimeError('actual CUDA required before fitting or benchmark work')
    return torch, device


def with_lite_policy(fn, *args, **kwargs):
    from kv_compare.lite_allocation import allocation_policy
    with allocation_policy(False):
        return fn(*args, **kwargs)


def fit_worker(q, torch, device):
    """The unchanged team's six-document calibration; no TEST words fitted."""
    from kv_compare.team_onestep.predict import Study
    from kv_compare.team_onestep.fused import prepare
    from kv_compare.team_onestep.layerwise import prepare_rank
    from kv_compare.team_packet import save_state, load_state
    path = Path(q['state']); begin = time.perf_counter()
    if (path/'manifest.json').exists():
        _, _, manifest = load_state(path, 'cpu')
        check_fit_metadata(manifest, q['metadata'])
        reused = True
    else:
        study = Study(q['captures'], device, q['train'], q['tables'], q['tests'], log=log)
        if len(study.train) != 6 or any(d.tokens != 8192 for d in study.train) or study.tables.tokens != 8192:
            raise ValueError('unchanged team calibration requires six full8192 TRAIN documents and8192 scale tokens')
        log('fit unchanged team one-step predictor: six full TRAIN documents')
        consts = prepare(study, log)
        rank = prepare_rank(study, consts, 256, log)
        manifest = save_state(path, consts, rank=rank, metadata=q['metadata'])
        torch.cuda.synchronize(device)
        reused = False
    write(q['output'], dict(status='completed', gpu_measured=True, reused=reused,
        manifest_sha256=digest(path/'manifest.json'), state=str(path),
        state_sha256=manifest['state_sha256'], metadata=q['metadata'],
        wall_ms=(time.perf_counter()-begin)*1000, runtime_policy=runtime_snapshot(torch)))


def own_codec(q, device):
    if q['kind'] == 'team':
        from kv_compare.team_packet import TeamCodec
        if digest(Path(q['state'])/'manifest.json') != q['manifest_sha256']:
            raise ValueError('team state manifest changed')
        return TeamCodec(q['state'], device, q['arm'])
    from kv_compare.lite_finish import FinishEngine
    from kv_compare.lite_graph import GraphDecoder
    engine = with_lite_policy(FinishEngine, q['state'], device, **q['engine_options'])
    if engine.state_sha256 != q['state_sha256']:
        raise ValueError('original Lite fitted state changed')
    return engine, with_lite_policy(GraphDecoder, engine, warmup=3)


def restore_wire(spec, entry, device):
    blob = Path(entry['packet']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry['packet_sha256'] or len(blob) != entry['packet_bytes']:
        raise ValueError('received packet bytes changed')
    begin = time.perf_counter()
    if spec['kind'] == 'team':
        from kv_compare.team_packet import TeamPacket
        packet = TeamPacket.import_bytes(blob, device=device)
    else:
        from kv_compare.causal_packet import Packet
        packet = Packet.restore(blob, device)
    import torch
    torch.cuda.synchronize(device)
    return packet, (time.perf_counter()-begin)*1000


def setup_decode(spec, codec, packet):
    if spec['kind'] == 'team':
        codec.graph_prepare(packet)
        return dict(graph_cold_wall_ms=codec.graph_cold_wall_ms,
                    graph_capture_count=codec.graph_capture_count)
    return with_lite_policy(codec[1].prepare, packet)


def decode_owned(spec, codec, packet):
    if spec['kind'] == 'team':
        return codec.graph_decode(packet)
    return with_lite_policy(codec[1].decode, packet)


def receive(q, torch, device):
    """Request deliberately has no source/capture path, encoding or prediction inputs."""
    if set(q) != {'codec', 'entries', 'output'}:
        raise ValueError('receiver request must contain only fixed codec state and serialized packets/assertions')
    forbidden = {'inputs', 'captures', 'source_run', 'predictions', 'decoded', 'path'}
    if any(forbidden & set(e) for e in q['entries']):
        raise ValueError('source data is forbidden in independent receiver')
    before = runtime_snapshot(torch); codec = own_codec(q['codec'], device)
    packets = [restore_wire(q['codec'], e, device)[0] for e in q['entries']]
    setup_decode(q['codec'], codec, packets[0]); checks = {}; held = None
    for i in (0, 1, 0):
        value = decode_owned(q['codec'], codec, packets[i])
        actual = check_original(value, q['entries'][i])
        entry = q['entries'][i]
        checks[entry['id']] = dict(id=entry['id'], verified=True, word_sha256=actual,
            packet_sha256=entry['packet_sha256'], shape=list(value.shape))
        if i == 0 and held is None:
            held = value
        elif i == 1:
            check_original(held, q['entries'][0])
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('receiver runtime policy was not restored')
    write(q['output'], dict(gpu_measured=True, no_source_kv_or_predictions=True,
        inputs='fixed state and serialized compact packets only; expected original digests are assertions',
        alternating_packets=True, independent_returned_outputs=True,
        runtime_policy=before, runtime_policy_after=after, checks=list(checks.values())))


def worker(q, torch, device):
    import numpy as np
    from kv_compare.causal_packet import word_sha
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    before = runtime_snapshot(torch); begin = time.perf_counter()
    codec = own_codec(q['codec'], device)
    torch.cuda.synchronize(device)
    cold = (time.perf_counter()-begin)*1000
    entries = []; packets = []; rows = []; setup = None
    for e in q['inputs']:
        if digest(e['path']) != e['file_sha256']:
            raise ValueError('matched input file changed')
        source = torch.from_numpy(np.load(e['path'], allow_pickle=False).view(np.int16).copy()).to(device)
        check_original(source, e)
        if q['codec']['kind'] == 'team':
            encode_fn = lambda: codec.encode(source, start=0)
        else:
            encode_fn = lambda: with_lite_policy(codec[0].encode, source, start=0, kappa=0.)
        packet, enc = timed(encode_fn, torch, device, q['warmup'], q['repeats'])
        if q['codec']['kind'] == 'lite':
            codec[0].last_reconstruction = None
        begin = time.perf_counter(); blob = packet.export(); export_ms = (time.perf_counter()-begin)*1000
        packet_path = out/(e['document']+'.kvp'); packet_path.write_bytes(blob)
        entry = dict(id=q['method']+'_'+e['document'], document=e['document'],
            shape=e['shape'], values=e['values'], original_word_sha256=e['original_word_sha256'],
            packet=str(packet_path), packet_sha256=hashlib.sha256(blob).hexdigest(), packet_bytes=len(blob))
        imported, upload_ms = restore_wire(q['codec'], entry, device)
        if q['codec']['kind'] == 'team':
            accounting = codec.accounting(imported)
        else:
            accounting = dict(compact_payload_bytes=imported.payload.numel()*2,
                directory_bytes=imported.starts.numel()*8, shared_predictor_bytes=codec[0].shared_bytes)
        row=dict(entry, method=q['method'], kappa=0., mode='lossless', original_exact=True,
            decoded_word_sha256=e['original_word_sha256'], state_sha256=q['codec']['state_sha256'],
            complete_bpv=8*len(blob)/e['values'], saved_percent=100*(1-len(blob)/(2*e['values'])),
            encode=enc, serialized_export_wall_ms=export_ms, upload_wall_ms=upload_ms,
            accounting=accounting, shared_predictor_gpu_bytes=accounting['shared_predictor_bytes'])
        if q['codec']['kind']=='team':
            prepared,row['model_only_diagnostic']=timed(lambda:codec.model(source),torch,device,1,3)
            prepared_packet,row['encode_prepared_diagnostic']=timed(
                lambda:codec.encode_prepared(source,*prepared),torch,device,1,3)
            # Diagnostic components are measured independently, and are not
            # added together or substituted for the full encode measurement.
            if prepared_packet.export()!=blob:
                raise AssertionError('diagnostic prepared encode changed compact packet bytes')
            del prepared,prepared_packet
        rows.append(row)
        entries.append(entry); packets.append(imported)
        del source, packet, blob; gc.collect(); torch.cuda.empty_cache()
    if len(entries) != 2 or entries[0]['shape'] != entries[1]['shape']:
        raise ValueError('two distinct documents at one length required per worker')
    begin = time.perf_counter(); setup = setup_decode(q['codec'], codec, packets[0])
    torch.cuda.synchronize(device); setup_ms = (time.perf_counter()-begin)*1000
    measurements = [dict(samples=[], warmup_samples=[]) for _ in entries]
    held = None
    for rep in range(q['warmup']+q['repeats']):
        for i in ((0, 1) if rep % 2 == 0 else (1, 0)):
            value, sample = timed(lambda i=i: decode_owned(q['codec'], codec, packets[i]), torch, device, 0, 1)
            check_original(value, entries[i])
            measurements[i]['warmup_samples' if rep < q['warmup'] else 'samples'].append(sample['samples'][0])
            if rep == 0 and i == 0:
                held = value
            elif rep == 0 and i == 1:
                check_original(held, entries[0]); held = None
            del value
    for i, row in enumerate(rows):
        dec = measurements[i]; dec['wall_median_ms'] = statistics.median(s['wall_ms'] for s in dec['samples'])
        row['decode'] = dec
        if q['codec']['kind'] == 'team':
            # Separate diagnostic; full receiver timing above includes staging,
            # sparse-arena expansion and an independently owned output clone.
            decode_owned(q['codec'], codec, packets[i])
            native_output, row['kernel_decode'] = timed(codec.graph_replay_native, torch, device, 1, q['repeats'])
            check_original(codec.output, entries[i])
            del native_output
            row['accounting'] = codec.accounting(packets[i])
        log(f"{row['id']}: {row['complete_bpv']:.5f} complete bpv; D={dec['wall_median_ms']:.3f}ms")
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('worker runtime policy was not restored')
    request = dict(codec=q['codec'], entries=entries, output=str(out/'receiver.json'))
    write(out/'receiver_request.json', request)
    del packets, imported, codec; gc.collect(); torch.cuda.empty_cache()
    launch([sys.executable, Path(__file__).resolve(), '--device', str(device), '--receive', out/'receiver_request.json'], out/'receiver.log')
    proof = json.loads((out/'receiver.json').read_text()); checks = {c['id']:c for c in proof['checks']}
    if (not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
            set(checks) != {r['id'] for r in rows} or proof['runtime_policy'] != before or
            not proof.get('alternating_packets') or not proof.get('independent_returned_outputs')):
        raise ValueError('independent receiver coverage/runtime differs')
    for row in rows:
        c = checks[row['id']]
        if not c['verified'] or c['word_sha256'] != row['original_word_sha256'] or c['packet_sha256'] != row['packet_sha256']:
            raise AssertionError('fresh receiver differs from original words')
        row['status'] = 'verified'
    write(out/'worker_results.json', dict(status='completed', gpu_measured=True, rows=rows,
        codec_provenance=q['codec'], device=torch.cuda.get_device_name(device),
        software={n:importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')},
        cold_initialization_wall_ms=cold, graph_setup_wall_ms=setup_ms, graph_setup=setup,
        runtime_policy=before, runtime_policy_after=after, receiver_sha256=digest(out/'receiver.json'),
        decode_scope='GPU-resident compact packet validation, staging/expansion, graph decode and owned output; file read, checksum parse and host upload recorded separately'))


def report(result, out):
    lines = ['# Matched one-step compact packets versus Lite and official controls', '',
        f"Status: **{result['status']}**. GPU measured: **{result.get('gpu_measured', False)}**.", '',
        'Identical original BF16 TEST words for all methods; all 28 layers, start 0; 512 and 8,192 tokens reported separately.',
        'Team arithmetic is unchanged. Team fit uses six 8,192-token TRAIN documents and 8,192 scale tokens; Lite reuses its four 1,024-token fit. This first comparison does not isolate calibration or conditioning.',
        'Complete packets include framing, actual directories and checksums. Fixed predictor/table state is additional. ExANS keeps its original fixed-slot padding/overflow policy.',
        'D is a synchronized wall median. Team/Lite include GPU-resident packet validation, staging/expansion, graph decode and owned output. Native controls retain their original framed decode path and runtime. Export/checksum parsing/host upload are separate.', '',
        '| Tokens | Method | Complete bpv | Saved | Full encode ms | Full decode ms | Native graph CUDA ms | Exact |',
        '|---:|---|---:|---:|---:|---:|---:|---|']
    for r in result.get('aggregates', []):
        kd = 'unmeasured' if r['kernel_decode_ms'] is None else f"{r['kernel_decode_ms']:.3f}"
        lines.append(f"| {r['tokens']} | {r['method']} | {r['complete_bpv']:.5f} | {r['saved_percent']:.2f}% | {r['encode_ms']:.3f} | {r['decode_ms']:.3f} | {kd} | {r['original_exact']} |")
    team_rows=[r for r in result.get('rows',[]) if r['method'].startswith('team_')]
    if team_rows:
        lines += ['', 'Mean per-document accounting in MiB. The compact packet is the actual exported transfer object. The native arena, full BF16 output and scratch are additional working memory; these are not a resident-pool capacity result.', '',
            '| Tokens | Team arm | Complete packet | Used native stream | Native arena | Compact staging | Directory staging | BF16 output | Workspace tensors | Shared predictors | Coder tables |',
            '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for n,m in sorted({(r['shape'][1],r['method']) for r in team_rows}):
            rs=[r for r in team_rows if r['shape'][1]==n and r['method']==m]
            cells=[statistics.mean(r['accounting'][k] for r in rs)/2**20 for k in
                ('packet_bytes','native_used_stream_bytes','arena_bytes','compact_staging_bytes','directory_staging_bytes','output_bytes','decoder_workspace_tensor_bytes','shared_predictor_bytes','coder_table_bytes')]
            lines.append(f"| {n} | {m} | "+' | '.join(f'{v:.3f}' for v in cells)+' |')
        lines += ['', 'Separate encode diagnostics (wall medians, ms); the full encode above includes both model generation and encoding/compaction. These separately synchronized components are not additive.', '',
            '| Tokens | Team arm | Model only | Encoding + compaction with prepared model | Export to host bytes | Import + host-to-device upload |',
            '|---:|---|---:|---:|---:|---:|']
        for n,m in sorted({(r['shape'][1],r['method']) for r in team_rows}):
            rs=[r for r in team_rows if r['shape'][1]==n and r['method']==m]
            a=statistics.mean(r['model_only_diagnostic']['wall_median_ms'] for r in rs)
            b=statistics.mean(r['encode_prepared_diagnostic']['wall_median_ms'] for r in rs)
            c=statistics.mean(r['serialized_export_wall_ms'] for r in rs)
            d=statistics.mean(r['upload_wall_ms'] for r in rs)
            lines.append(f'| {n} | {m} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} |')
    models = result.get('transfer_models', {}).get('rows', [])
    lines += ['', 'Calculated precompressed serial wire + full receiver latency (ms), for every requested link target. Complete compact packets are charged; no network measured.', '',
        '| Tokens | Gb/s | Lite | Team dense uniform | Team rank 256 uniform | SplitZip | ExANS fixed records |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for tokens in result.get('recipe', {}).get('lengths', []):
        for g in LINKS:
            rs={r['method']:r for r in models if r['tokens']==tokens and r['effective_gbps']==g}
            if not rs:continue
            cells=[f"{rs[m]['precompressed_serial_ms']:.3f}" if m in rs else 'pending' for m in
                ('lite_current_graph','team_dense_uniform','team_rank256_uniform','splitzip_official_framed','exans_official_fixed_framed')]
            lines.append(f'| {tokens} | {g:g} | '+' | '.join(cells)+' |')
    lines += ['', 'Calculated link models; no network, arrival pipeline or SSD measured.', '',
        '| Tokens | Gb/s | Best precompressed wire + D | Best ideal max(wire,D) | Best fresh E + wire + D |',
        '|---:|---:|---|---|---|']
    for tokens in result.get('recipe', {}).get('lengths', []):
        for g in LINKS:
            rs = [r for r in models if r['tokens'] == tokens and r['effective_gbps'] == g]
            if not rs: continue
            cells=[]
            for key in ('precompressed_serial_ms', 'ideal_overlap_ms', 'fresh_serial_ms'):
                best=min(rs, key=lambda r:r[key]); cells.append(f"{best['method']} ({best[key]:.3f}ms)")
            lines.append(f"| {tokens} | {g:g} | "+' | '.join(cells)+' |')
    if result.get('error'): lines += ['', 'Failure: '+result['error']]
    lines += ['', 'No lossy quality, AQUA adaptation, attention-facing pool, T-to-P conversion or serving throughput is measured. Native allocation peaks outside Torch remain unmeasured. Ideal overlap is a lower-bound model, not a measured transport result.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')


def collect_source_inputs(root, names, lengths, out):
    import numpy as np
    from kv_compare.team_onestep.predict import Doc
    records=[]; folder=out/'inputs'; folder.mkdir(exist_ok=True)
    for name in names:
        d=Doc(root,name)
        for n in lengths:
            if n > d.tokens: raise ValueError('requested TEST length exceeds original capture')
            words=np.empty((d.layers,n,d.columns*2), dtype='<u2')
            words[:,:,:d.columns]=np.asarray(d.K[:,:n])
            words[:,:,d.columns:]=np.asarray(d.V[:,:n])
            expected=hashlib.sha256(words.tobytes()).hexdigest()
            path=folder/f'{name}_t{n}.npy'
            if path.exists():
                old=np.load(path,mmap_mode='r',allow_pickle=False)
                if old.dtype != np.uint16 or old.shape != words.shape or hashlib.sha256(old.tobytes()).hexdigest()!=expected:
                    raise ValueError('resumed matched TEST array changed')
            else: np.save(path, words, allow_pickle=False)
            records.append(dict(document=f'{name}_t{n}',source_document=name,path=str(path),
                file_sha256=digest(path),original_word_sha256=expected,shape=list(words.shape),values=words.size,tokens=n))
            del words; gc.collect()
    return records


def inherited_requests(parent_dir, parent):
    result={}
    for name in ('raw_bf16','splitzip_official_framed','exans_official_fixed_framed','lite_entropy'):
        path=parent_dir/name/'request.json'
        q=json.loads(path.read_text())
        data=json.loads((parent_dir/name/'worker_results.json').read_text())
        if q['codec'] != data['codec_provenance']:
            raise ValueError('parent codec request changed: '+name)
        result[name]=q
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run',type=Path,default=ROOT/'runs/lite_entropy_a4500_20261028_110019_1125591')
    p.add_argument('--source-run',type=Path,default=ROOT/'runs/llama32_3b_compare')
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--lengths',default='512,8192')
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    for name in ('fit-worker','worker','receive'): p.add_argument('--'+name,type=Path,help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    request=args.fit_worker or args.worker or args.receive
    if request:
        q=json.loads(request.read_text())
        torch,device=cuda_check(args.device, lite=q.get('codec',{}).get('kind')=='lite')
        (fit_worker if args.fit_worker else worker if args.worker else receive)(q,torch,device)
        return 0
    if args.output is None: p.error('--output required')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    result=dict(format='team_packet_v1',status='running',gpu_measured=False,jobs={},rows=[])
    writable=not (out/'results.json').exists()
    def save():
        result['rows']=[normalize_row(r) for j in result['jobs'].values() if j.get('status')=='completed' for r in j['rows']]
        result['aggregates']=aggregate(result['rows']); result['transfer_models']=transfer_models(result['aggregates'])
        write(out/'results.json',result); report(result,out)
    try:
        torch,device=cuda_check(args.device)
        lengths=validate_lengths(args.lengths)
        if args.warmup<1 or args.repeats<2:
            raise ValueError('warmup>=1 and repeats>=2 required')
        parent_dir=args.parent_run.resolve(); parent=json.loads((parent_dir/'results.json').read_text())
        from scripts.package_lite_entropy import verify as verify_parent
        if parent['status']!='completed' or not parent.get('gpu_measured'): raise ValueError('completed actual-GPU parent required')
        verify_parent(parent_dir,parent)
        inherited={**parent['recipe']['unchanged_code'],**parent['recipe']['code']}
        for name,wanted in inherited.items():
            if digest(ROOT/name)!=wanted: raise ValueError('original source changed: '+name)
        versions={n:importlib.metadata.version(n) for n in ('torch','triton','numpy','transformers')}
        if (versions!=parent['recipe']['software'] or torch.version.cuda!=parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device)!=parent['hardware']['name']): raise ValueError('original GPU/software required')
        from kv_compare.causal_study import source_inputs
        root,train,tables,tests,source=source_inputs(args.source_run,out,6,log)
        if train!=[f'train{i}' for i in range(6)] or tables!='tables' or tests!=['test_text','test_code']:
            raise ValueError('this unchanged team experiment requires the original PEP/CPython source roles')
        if any(d['tokens']!=8192 for d in source['documents'].values()): raise ValueError('full original8192-token captures required')
        from scripts.bench_lite_graph import native_receipt
        receipt=native_receipt(parent); requests=inherited_requests(parent_dir,parent)
        recipe=dict(parent_run=str(parent_dir),parent_results_sha256=digest(parent_dir/'results.json'),
            parent_request_hashes={n:digest(parent_dir/n/'request.json') for n in requests},
            source=source,source_run=str(args.source_run.resolve()),code={n:digest(ROOT/n) for n in FILES},
            unchanged_code=inherited,lengths=list(lengths),warmup=args.warmup,repeats=args.repeats,
            methods=list(METHODS),software=versions,device=torch.cuda.get_device_name(device),cuda_runtime=torch.version.cuda,
            native_receipt=receipt,team_calibration=dict(train=train,train_tokens=8192,tables=tables,scale_tokens=8192),
            lite_calibration=dict(train=[f'train{i}' for i in range(4)],fit_tokens=1024,scale_tokens=1024,refitted=False))
        wanted=identity(recipe)
        if not writable:
            old=json.loads((out/'results.json').read_text())
            if old.get('identity')!=wanted: raise ValueError('resume recipe differs; preserve output and choose a new run')
            result=old; writable=True
        result.update(identity=wanted,recipe=recipe,status='running',gpu_measured=True,
            hardware=dict(parent['hardware'],pcie_query=pcie_metadata()))
        result.pop('error',None);result.pop('traceback',None)
        write(out/'source_manifest.json',recipe);save()
        fit_q=dict(state=str(out/'team_state'),captures=str(root),train=train,tables=tables,tests=tests,
            metadata=dict(recipe_identity=wanted,capture_source=source,calibration=recipe['team_calibration']),output=str(out/'fit.json'))
        if not result.get('fit'):
            write(out/'fit_request.json',fit_q)
            launch([sys.executable,Path(__file__).resolve(),'--device',str(device),'--fit-worker',out/'fit_request.json'],out/'fit.log')
            result['fit']=json.loads((out/'fit.json').read_text());result['fit']['receipt_sha256']=digest(out/'fit.json');save()
        elif digest(out/'fit.json')!=result['fit']['receipt_sha256']:
            raise ValueError('completed calibration receipt changed')
        if digest(out/'team_state/manifest.json')!=result['fit']['manifest_sha256']: raise ValueError('team state manifest changed')
        inputs=collect_source_inputs(root,tests,lengths,out);result['inputs']=inputs;save()
        plans=[]
        for n in lengths:
            selected=[e for e in inputs if e['tokens']==n]
            for method in METHODS:
                job=f'{method}_t{n}';directory=out/job;directory.mkdir(exist_ok=True)
                native=method in METHODS[:3]
                if native:
                    spec=dict(requests[method]['codec'])
                    if method=='exans_official_fixed_framed': spec['blocks']=math.ceil(2*max(e['values'] for e in selected)/999424)
                    q=dict(method=method,codec=spec,inputs=selected,kappas=[0.],
                        expected={e['document']+':0':dict(decoded_word_sha256=e['original_word_sha256'],start=0) for e in selected},
                        output=str(directory),warmup=args.warmup,repeats=args.repeats)
                    script=ROOT/'scripts/bench_lite_allocation.py';option='--baseline-worker';python=sys.executable;env=None
                    if method=='splitzip_official_framed': python=str(ROOT/'runs/_llama_official/envs/splitzip/bin/python')
                    if method=='exans_official_fixed_framed':
                        from kv_compare.profile_recovery import runtime_environment
                        nv=importlib.util.find_spec('nvidia.libnvcomp')
                        if nv is None or importlib.metadata.version('nvidia-libnvcomp-cu12')!=receipt['nvcomp']: raise ValueError('pinned nvCOMP required')
                        tool=Path(spec['library']).parent.parent/'toolchain.json'
                        if digest(tool)!=receipt['toolchain_sha256']: raise ValueError('native toolchain receipt changed')
                        env=runtime_environment(os.environ,json.loads(tool.read_text())['cuda_home'],[Path(s)/'lib' for s in nv.submodule_search_locations])
                else:
                    if method=='lite_current_graph':
                        spec=dict(requests['lite_entropy']['codec'],kind='lite')
                        spec['state_sha256']=parent['jobs']['lite_entropy']['rows'][0]['state_sha256']
                    else:
                        spec=dict(kind='team',arm=method.removeprefix('team_'),state=str(out/'team_state'),
                            state_sha256=result['fit']['state_sha256'],manifest_sha256=result['fit']['manifest_sha256'])
                    q=dict(method=method,codec=spec,inputs=selected,output=str(directory),warmup=args.warmup,repeats=args.repeats)
                    script=Path(__file__).resolve();option='--worker';python=sys.executable;env=None
                plans.append((job,q,python,script,option,env))
        result['selected_jobs']=[x[0] for x in plans];save()
        for name,q,python,script,option,env in plans:
            directory=out/name;old=result['jobs'].get(name,{})
            if old.get('status')=='completed':
                for path,wanted_sha in old['artifacts'].items():
                    if digest(out/path)!=wanted_sha: raise ValueError('completed job artifact changed')
                if json.loads((directory/'request.json').read_text())!=q: raise ValueError('completed worker request changed')
                for r in old['rows']:
                    if digest(r['packet'])!=r['packet_sha256']: raise ValueError('completed packet changed')
                log('reuse completed job: '+name);continue
            write(directory/'request.json',q)
            launch([python,script,'--device',str(device),option,directory/'request.json'],directory/'worker.log',env)
            data=json.loads((directory/'worker_results.json').read_text())
            if data['status']!='completed' or not data.get('gpu_measured') or data['device']!=recipe['device'] or data['codec_provenance']!=q['codec']:
                raise ValueError('actual original-GPU worker required')
            result['jobs'][name]=dict(status='completed',rows=data['rows'],
                artifacts={str((directory/f).relative_to(out)):digest(directory/f) for f in ('request.json','worker_results.json','receiver.json','receiver_request.json')})
            save()
        from scripts.package_team_packet import verify
        result['status']='completed';save();verify(out,result)
        log('completed: '+str(out/'summary.md'));return 0
    except Exception as error:
        if writable:
            result.update(status='failed',error=str(error),traceback=traceback.format_exc());save()
        traceback.print_exc();return 1


if __name__=='__main__': raise SystemExit(main())
