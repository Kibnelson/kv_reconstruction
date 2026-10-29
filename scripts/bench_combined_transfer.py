#!/usr/bin/env python3
"""Combined lossless T comparison; calibration-only mixed policy, unchanged controls.

Parent mode imports only the standard library. GPU APIs, source words, and codec
imports live in isolated workers. All link results are calculated serial costs.
"""
from __future__ import annotations
import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pstats
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KGRID = (0, 2, 8, 14, 28)
LENGTHS = (512, 8192)
LINKS = (1, 100, 200, 300, 400)
COMPACT = 'exans_official_compact_framed'
MIXED = tuple(f'mixed_rank256_k{k}' for k in KGRID)
NEW_METHODS = MIXED + (COMPACT,)
CALIBRATION_SOURCE = 'benchmark_results/team_packet_a4500_20261028_122017_2834427_report_20261028_122819'
META_SHA = 'de33b22243969fe8d37be0dbc90f835bda263ba8091aee717c4e0ad53b5c9d2a'
TABLES_SHA = '95cb73a219785c65209da7b48582919a113ff17317c016037e235e4d4bc5a3ce'
SCOPE = ('Lossless complete GPU transfer adapters on frozen original Llama words and fitted state. '
         'Mixed masks and link policies use TABLES calibration only, not TEST. '
         'TABLES also calibrated spreads; it is not an independent validation document. '
         'Native source benchmarks and lossy quality remain separate. No P attention, T-to-P, NIC, or SSD measurement.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def rank_layer_order(sz_bytes, rans_words):
    """Stable calibration ranking; negative benefits remain visible."""
    if len(sz_bytes) != 28 or len(rans_words) != 28:
        raise ValueError('all 28 calibration layers required')
    if any(type(x) is not int or x < 0 for x in [*sz_bytes, *rans_words]):
        raise ValueError('nonnegative integer byte/word costs required')
    savings = [8 * (sz_bytes[i] - 2 * rans_words[i]) for i in range(28)]
    return sorted(range(28), key=lambda i: (-savings[i], i)), savings


def freeze_policy(rows):
    """Freeze each objective/length/link using only measured TABLES full calls."""
    if (len(rows) != len(LENGTHS) * len(KGRID)
        or any(type(r.get('tokens')) is not int or r['tokens'] not in LENGTHS
               or type(r.get('k')) is not int or r['k'] not in KGRID for r in rows)):
        raise ValueError('exactly ten declared TABLES length/k rows required')
    policies = []
    for n in LENGTHS:
        group = [r for r in rows if r['tokens'] == n]
        if {r['k'] for r in group} != set(KGRID) or len(group) != len(KGRID):
            raise ValueError('one calibration row per frozen k/length required')
        for r in group:
            if r.get('document') != 'tables' or type(r['packet_bytes']) is not int or r['packet_bytes'] <= 0:
                raise ValueError('policy selection must use actual TABLES packets')
            if any(not math.isfinite(r[x]) or r[x] <= 0 for x in ('encode_ms', 'decode_ms')):
                raise ValueError('positive finite measured calibration latency required')
        for link in LINKS:
            for objective in ('precompressed', 'fresh'):
                def cost(r):
                    return 8 * r['packet_bytes'] / (link * 1e6) + r['decode_ms'] + (r['encode_ms'] if objective == 'fresh' else 0)
                chosen = min(group, key=lambda r: (cost(r), r['k']))
                policies.append(dict(tokens=n, effective_gbps=link, objective=objective,
                    method=f"mixed_rank256_k{chosen['k']}", k=chosen['k'], calibration_serial_ms=cost(chosen)))
    return dict(format='combined_mixed_policy_v1', selection_document='tables',
        selection_scope='Frozen TABLES document; reused spread calibration, not independent validation; no TEST selection',
        k_grid=list(KGRID), links_gbps=list(LINKS), calibration_timing=dict(warmup=1, repeats=3, basis='isolated_wall_median'),
        policies=policies)


def select_policy_rows(rows, policy):
    """Apply frozen choices to TEST rows without searching TEST outcomes."""
    result = []
    for p in policy['policies']:
        selected = [r for r in rows if r['tokens'] == p['tokens'] and r['method'] == p['method']]
        if len(selected) != 2 or len({r['document'] for r in selected}) != 2:
            raise ValueError('frozen policy needs both TEST rows')
        for r in selected:
            for basis, key in (('batched', 'wall_mean_ms'), ('isolated', 'wall_median_ms')):
                e, d = (r[op][basis][key] for op in ('encode', 'decode'))
                wire = 8 * r['packet_bytes'] / (p['effective_gbps'] * 1e6)
                result.append(dict(p, document=r['document'], timing_basis=basis,
                    packet_bytes=r['packet_bytes'], wire_ms=wire, decode_ms=d, encode_ms=e,
                    serial_ms=wire + d + (e if p['objective'] == 'fresh' else 0), measured_network=False))
    return result


def validate_codec_spec(method, spec, tokens):
    if method in MIXED:
        allowed = {'kind', 'state', 'manifest_sha256', 'predictor_state_sha256', 'splitzip_source',
                   'splitzip_state', 'splitzip_state_sha256', 'mask', 'arm'}
        mask = spec.get('mask', [])
        if (set(spec) != allowed or spec['kind'] != 'mixed_transfer' or spec['arm'] != 'rank256_uniform'
            or len(mask) != 28 or any(type(x) is not bool for x in mask)
            or sum(mask) != int(method.rsplit('k', 1)[1])):
            raise ValueError('mixed receiver codec allowlist/mask differs')
    elif method == COMPACT:
        if (set(spec) != {'kind', 'source', 'library', 'library_sha256', 'blocks'} or spec['kind'] != 'compact_exans'
            or spec['blocks'] != (59 if tokens == 512 else 941)):
            raise ValueError('compact ExANS receiver codec allowlist/geometry differs')
    else:
        raise ValueError('unknown new codec')


def validate_receiver_request(q):
    if set(q) != {'method', 'tokens', 'codec', 'entries', 'execution', 'output'} or q['method'] not in NEW_METHODS:
        raise ValueError('receiver accepts only new codec, serialized packets, and expected assertions')
    if q['tokens'] not in LENGTHS or q['execution'] != 'direct':
        raise ValueError('receiver length/execution differs')
    validate_codec_spec(q['method'], q['codec'], q['tokens'])
    allowed = {'id', 'document', 'shape', 'values', 'packet', 'packet_sha256', 'packet_bytes', 'original_word_sha256'}
    if len(q['entries']) != 2 or any(set(e) != allowed for e in q['entries']):
        raise ValueError('receiver entries must contain only complete packet assertions; raw/source KV forbidden')
    if {e['document'] for e in q['entries']} != {f"test_text_t{q['tokens']}", f"test_code_t{q['tokens']}"}:
        raise ValueError('receiver must alternate the two frozen TEST documents')
    for e in q['entries']:
        if e['shape'] != [28, q['tokens'], 2048] or e['values'] != 28 * q['tokens'] * 2048:
            raise ValueError('receiver geometry differs')


def source_receipts():
    # Bind the unchanged upstream mixed extension, not just today's local files.
    manifest_path = ROOT / 'kv_compare/team_extensions/source_manifest.json'
    m = json.loads(manifest_path.read_text())
    if (m.get('format') != 'team_mixed_extension_source_v1'
        or m.get('commit') != '8a40dd98282ed364a8fe1077d171a1922277c1ec'
        or m.get('core_manifest_sha256') != '9f7b644b2ea6a2c7ac1e9632a1320e6e02d4910b2e615f7fdafa546f72b44617'
        or set(m.get('files', {})) != {'__init__.py', 'mixed.py', 'layerwise.py'}
        or m['files']['mixed.py'] != '3193e510df697087ea300e233217b04303f1370b543ce10bf8f87a2f577442e7'
        or any(sha(manifest_path.parent / n) != h for n, h in m['files'].items())):
        raise ValueError('unchanged author mixed extension source identity differs')
    names = ['docs/combined_transfer_v1.md', 'scripts/run_combined_transfer_a4500.sh', 'tests/test_combined_transfer.py', 'scripts/bench_combined_transfer.py', 'kv_compare/mixed_transfer.py', 'kv_compare/compact_exans.py',
        'scripts/bench_frozen_transfer.py', 'scripts/frozen_transfer_worker.py',
        'kv_compare/team_onestep/source_manifest.json', 'kv_compare/team_extensions/source_manifest.json',
        'tests/test_mixed_transfer.py', 'tests/test_compact_exans.py',
        'scripts/check_nuc_baseline_readiness.py', 'scripts/run_splitzip_official_native.py',
        'scripts/run_exans_official_preflight.py']
    names += ['kv_compare/team_extensions/' + n for n in ('__init__.py', 'mixed.py', 'layerwise.py')]
    return {name: sha(ROOT / name) for name in names}


def new_spec(method, selections, masks):
    rank = selections['team_rank256_uniform_t512']['codec']
    sz = selections['splitzip_official_framed_t512']['codec']
    if method in MIXED:
        k = int(method.rsplit('k', 1)[1])
        return dict(kind='mixed_transfer', state=rank['state'], manifest_sha256=rank['manifest_sha256'],
            predictor_state_sha256=rank['state_sha256'], splitzip_source=sz['source'], splitzip_state=sz['state'],
            splitzip_state_sha256=sz['state_sha256'], mask=masks[str(k)], arm='rank256_uniform')
    return dict(selections['exans_official_fixed_framed_t512']['codec'], kind='compact_exans')


def new_job_plan(selections, policy):
    jobs = []
    for n in LENGTHS:
        for method in NEW_METHODS:
            target = selections[f'team_rank256_uniform_t{n}']
            spec = new_spec(method, selections, policy['masks'])
            if method == COMPACT:
                spec['blocks'] = selections[f'exans_official_fixed_framed_t{n}']['codec']['blocks']
            jobs.append((f'{method}_t{n}', dict(method=method, tokens=n, codec=spec,
                entries=target['entries'], raw_entries=selections[f'raw_bf16_t{n}']['entries'],
                splitzip_entries=selections[f'splitzip_official_framed_t{n}']['entries'],
                execution='direct', round='paired', policy_sha256=policy['policy_sha256'],
                timing_source=str(ROOT / 'runs/_llama_official/sources/splitzip/bench_codec_throughput.py'))))
    return jobs


def spawn(argv, log, env=None):
    with Path(log).open('w') as stream:
        child = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                try: code = child.wait(timeout=30); break
                except subprocess.TimeoutExpired: print('Still running; log: ' + str(log), flush=True)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise
    if code: raise RuntimeError('exit ' + str(code) + '; see ' + str(log))


def worker_argv(py, request, method):
    if method in NEW_METHODS:
        return [str(py), str(Path(__file__).resolve()), '--worker', str(request), '--device', 'cuda:0']
    return [str(py), str(ROOT / 'scripts/frozen_transfer_worker.py'), '--request', str(request), '--device', 'cuda:0']


def codec_for(q, device):
    spec = q['codec']
    if q['method'] in MIXED:
        from kv_compare.mixed_transfer import MixedTransfer
        if sha(Path(spec['state']) / 'manifest.json') != spec['manifest_sha256']:
            raise ValueError('frozen predictor manifest changed')
        codec = MixedTransfer(spec['state'], spec['splitzip_source'], spec['splitzip_state'],
            spec['splitzip_state_sha256'], spec['mask'], device, arm=spec['arm'])
        if any(spec['mask']) and codec.predictor_state_sha256 != spec['predictor_state_sha256']:
            raise ValueError('fixed predictor tensor identity differs')
        return codec
    from kv_compare.compact_exans import CompactExans
    if sha(spec['library']) != spec['library_sha256']:
        raise ValueError('frozen ExANS library changed')
    return CompactExans(spec['source'], spec['library'], device, spec['blocks'])


def restore_entries(entries, device):
    from kv_compare.causal_packet import Packet
    packets = []
    for e in entries:
        wire = Path(e['packet']).read_bytes()
        if len(wire) != e['packet_bytes'] or hashlib.sha256(wire).hexdigest() != e['packet_sha256']:
            raise ValueError('compressed packet changed')
        packets.append(Packet.restore(wire, device))
    return packets


def packet_checks(packets, entries):
    result = []
    for p, e in zip(packets, entries):
        wire = p.export()
        if len(wire) != e['packet_bytes'] or hashlib.sha256(wire).hexdigest() != e['packet_sha256']:
            raise AssertionError('compressed input changed')
        result.append(dict(id=e['id'], unchanged=True, sha256=e['packet_sha256'], bytes=len(wire)))
    return result


def close(codec):
    if hasattr(codec, 'close'): codec.close()


def receive(q, torch, device):
    from scripts.frozen_transfer_worker import runtime_snapshot
    from scripts.bench_team_packet import check_original
    validate_receiver_request(q); before = runtime_snapshot(torch)
    codec = codec_for(q, device); packets = restore_entries(q['entries'], device); checks = {}; held = []
    try:
        for i in (0, 1, 0):
            value = codec.decode(packets[i]); digest = check_original(value, q['entries'][i]); e = q['entries'][i]
            checks[e['id']] = dict(id=e['id'], verified=True, word_sha256=digest,
                shape=list(value.shape), packet_sha256=e['packet_sha256'])
            for old_value, old_entry in held: check_original(old_value, old_entry)
            held.append((value, e))
        immutable = packet_checks(packets, q['entries'])
    finally: close(codec)
    after = runtime_snapshot(torch)
    if before != after: raise AssertionError('receiver changed runtime flags')
    write(q['output'], dict(status='completed', gpu_measured=True, no_source_kv_or_predictions=True,
        alternating_packets=True, independent_returned_outputs=True, checks=list(checks.values()),
        packet_immutability=immutable, runtime_policy=before, runtime_policy_after=after))


def diagnostics(codec, source, packet, out, torch, device):
    """Separate diagnostic calls; these do not replace ordinary timing samples."""
    profile = cProfile.Profile()
    profile.enable()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True) as prof:
        for _ in range(3):
            p = codec.encode(source); value = codec.decode(p)
            torch.cuda.synchronize(device); del p, value
    profile.disable(); prof.export_chrome_trace(str(out / 'full_trace.json'))
    stats = pstats.Stats(profile)
    cpu = sorted([dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1],
        self_seconds=v[2], cumulative_seconds=v[3]) for k, v in stats.stats.items()], key=lambda r: -r['cumulative_seconds'])[:50]
    keys = [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
        self_gpu_us=getattr(e, 'self_device_time_total', 0), cpu_memory_bytes=e.cpu_memory_usage,
        gpu_memory_bytes=getattr(e, 'device_memory_usage', 0)) for e in prof.key_averages()]
    kernels = {}
    for e in prof.events():
        if str(e.device_type).endswith('CUDA'):
            r = kernels.setdefault(e.name, dict(name=e.name, calls=0, gpu_us=0.0)); r['calls'] += 1
            r['gpu_us'] += float(getattr(e, 'device_time_total', 0))
    result = dict(calls=3, scope='Separate full E/D diagnostics, profiler overhead; component times are nonadditive',
        key_averages=keys, kernels=sorted(kernels.values(), key=lambda r: -r['gpu_us']), cprofile=cpu,
        allocation_ops=[r for r in keys if any(x in r['name'] for x in ('empty', 'resize', 'zeros', 'alloc'))])
    from scripts.bench_llama_causal import timed
    if hasattr(codec, 'unpack') and hasattr(codec, 'decode_native'):
        parsed, a = timed(lambda: codec.unpack(packet), torch, device, 1, 3)
        value, b = timed(lambda: codec.decode_native(parsed), torch, device, 1, 3); del value
        result['nonadditive_stages'] = dict(unpack=a, native_decode=b)
    elif hasattr(codec, 'compact_fixed'):
        # Expand validation/scatter is measured independently of the full decoder.
        _, a = timed(lambda: codec.expand(packet), torch, device, 1, 3)
        result['nonadditive_stages'] = dict(expand=a)
    write(out / 'diagnostics.json', result)
    return result


def measure(q, torch, device):
    from scripts.frozen_transfer_worker import runtime_snapshot, load_author_bench, timing_samples
    from scripts.bench_team_gpu import load_packets
    from scripts.bench_team_packet import check_original
    from scripts.bench_llama_causal import timed
    from kv_compare.framed_baselines import Baseline
    from kv_compare.causal_packet import Packet
    out = Path(q['output']); before = runtime_snapshot(torch)
    bench, timing_receipt = load_author_bench(q['timing_source'], torch)
    begin = time.perf_counter(); codec = codec_for(q, device); torch.cuda.synchronize(device)
    cold = 1000 * (time.perf_counter() - begin)
    raw_packets = load_packets(dict(method='raw_bf16', entries=q['raw_entries']), device)
    raw = Baseline('raw_bf16', device); bydoc = {e['document']: (p, e) for e, p in zip(q['raw_entries'], raw_packets)}
    rows = []; entries = []; packets = []
    try:
        for target in q['entries']:
            rp, re = bydoc[target['document']]
            if any(re[k] != target[k] for k in ('shape', 'values', 'original_word_sha256')):
                raise ValueError('raw/target frozen word identities differ')
            source = raw.decode(rp); check_original(source, target)
            encoded = codec.encode(source); check_original(codec.decode(encoded), target)
            wire = encoded.export(); e = {k: target[k] for k in ('id', 'document', 'shape', 'values', 'original_word_sha256')}
            e.update(packet=str(out / (target['id'] + '_encoded.kvc')), packet_bytes=len(wire), packet_sha256=hashlib.sha256(wire).hexdigest())
            if q['method'] == 'mixed_rank256_k0':
                reference = next(x for x in q['splitzip_entries'] if x['document'] == target['document'])
                if e['packet_bytes'] != reference['packet_bytes'] or e['packet_sha256'] != reference['packet_sha256']:
                    raise AssertionError('zero-mask whole-request SplitZip delegation changed bytes')
            Path(e['packet']).write_bytes(wire); packet = Packet.restore(wire, device)
            check_original(codec.decode(packet), target); packets.append(packet); entries.append(e); del encoded, wire
            eb = timing_samples([bench(lambda: codec.encode(source), str(device), 10, 50) for _ in range(5)], 2 * target['values'])
            ep, ei = timed(lambda: codec.encode(source), torch, device, 2, 5)
            check_original(codec.decode(ep), target); del ep
            db = timing_samples([bench(lambda: codec.decode(packet), str(device), 10, 50) for _ in range(5)], 2 * target['values'])
            value, di = timed(lambda: codec.decode(packet), torch, device, 2, 5)
            check_original(value, target); del value
            accounting = codec.accounting(packet) if hasattr(codec, 'accounting') else {}
            if 'packet_bytes' in accounting and accounting['packet_bytes'] != e['packet_bytes']:
                raise AssertionError('accounting differs from actual complete serialized bytes')
            row = dict(e, method=q['method'], tokens=q['tokens'], round=q['round'], execution=q['execution'],
                original_word_exact=True, decoded_word_sha256=target['original_word_sha256'],
                encode=dict(batched=eb, isolated=ei), decode=dict(batched=db, isolated=di), accounting=accounting,
                policy_sha256=q['policy_sha256'])
            if q['tokens'] == 8192 and 'text' in target['document'] and q['method'] in ('mixed_rank256_k14', COMPACT):
                diagnostics(codec, source, packet, out, torch, device); row['diagnostics'] = str(out / 'diagnostics.json')
            check_original(source, target); rows.append(row); del source; gc.collect()
            print(target['document'] + ': complete packet and original words verified', flush=True)
        immutable = packet_checks(packets, entries)
        from scripts.frozen_transfer_worker import packet_checks as old_checks
        raw_immutable = old_checks(raw_packets, q['raw_entries'])
    finally: close(codec)
    after = runtime_snapshot(torch)
    if before != after: raise AssertionError('measurement changed runtime flags')
    return dict(rows=rows, entries=entries, packet_immutability=immutable, raw_packet_immutability=raw_immutable,
        runtime_policy=before, runtime_policy_after=after, cold_state_initialization_wall_ms=cold,
        author_timing=timing_receipt, graph_capture_count=0)


def worker(q, torch, device):
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    data = measure(q, torch, device)
    request = {k: q[k] for k in ('method', 'tokens', 'codec', 'execution')}
    request.update(entries=data.pop('entries'), output=str(out / 'receiver.json'))
    validate_receiver_request(request); write(out / 'receiver_request.json', request)
    torch.cuda.synchronize(device); gc.collect(); torch.cuda.empty_cache()
    spawn([sys.executable, __file__, '--device', str(device), '--receive', out / 'receiver_request.json'], out / 'receiver.log')
    proof = json.loads((out / 'receiver.json').read_text())
    if (not all(proof.get(k) for k in ('gpu_measured', 'no_source_kv_or_predictions', 'alternating_packets', 'independent_returned_outputs'))
        or proof['runtime_policy'] != data['runtime_policy'] or proof['runtime_policy_after'] != data['runtime_policy']):
        raise AssertionError('fresh receiver ownership/runtime proof failed')
    if len(proof['packet_immutability']) != 2 or not all(x['unchanged'] for x in proof['packet_immutability']):
        raise AssertionError('fresh receiver compressed inputs changed')
    checks = {c['id']: c for c in proof['checks']}
    for r in data['rows']:
        c = checks[r['id']]
        if not c['verified'] or c['word_sha256'] != r['original_word_sha256'] or c['packet_sha256'] != r['packet_sha256']:
            raise AssertionError('fresh receiver original words differ')
    data.update(status='completed', gpu_measured=True, receiver_sha256=sha(out / 'receiver.json'),
        device=torch.cuda.get_device_name(device), software={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')},
        scope=SCOPE, memory_scope='Isolated timed() samples Torch baseline and incremental peaks including output/scratch; native allocations outside Torch remain unmeasured')
    write(out / 'worker_results.json', data)


def calibration_words(q, torch, device):
    import numpy as np
    from kv_compare.causal_study import array_word_digest
    root = Path(q['capture']); source = q['capture_source']
    if sha(root / 'meta.json') != META_SHA or source['meta_sha256'] != META_SHA:
        raise ValueError('frozen capture metadata changed')
    k = np.load(root / 'tables_K_bits.npy', mmap_mode='r', allow_pickle=False)
    v = np.load(root / 'tables_V_bits.npy', mmap_mode='r', allow_pickle=False)
    ids = np.load(root / 'tables_ids.npy', allow_pickle=False)
    if (k.dtype != np.uint16 or v.dtype != np.uint16 or k.shape != (28, 8192, 1024) or v.shape != k.shape
        or array_word_digest(k, v) != TABLES_SHA or source['documents']['tables']['words_sha256'] != TABLES_SHA
        or hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest() != source['documents']['tables']['token_ids_sha256']):
        raise ValueError('frozen TABLES calibration words/token IDs changed')
    array = np.concatenate((k, v), axis=2)
    return torch.from_numpy(array.view(np.int16)).to(device)


def select_worker(q, torch, device):
    from scripts.bench_llama_causal import timed
    from scripts.frozen_transfer_worker import runtime_snapshot
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True); before = runtime_snapshot(torch)
    words = calibration_words(q, torch, device)
    alltrue = dict(q['mixed_codec'], mask=[True] * 28)
    codec = codec_for(dict(method='mixed_rank256_k28', codec=alltrue), device)
    try:
        # MixedOneStep's optional layer_words argument is not implemented. Use
        # the original frozen Rank256 encoder to charge states/alignment to l0.
        from kv_compare.team_onestep.layerwise import LayerwiseOneStep
        native = LayerwiseOneStep(codec.consts, codec.C, codec.hd, tokens_per_program=32, chunks=2,
            mode='uniform', search=True, rank=codec.rank, requant='epilogue', rank_tiles=(128, 128, 64, 4))
        xall, prior = native.model(words); counts = torch.zeros(28, dtype=torch.int64, device=device)
        native.encode(words, xall, prior, layer_words=counts)
        rank_words = [int(x) for x in counts.cpu().tolist()]
        sz_bytes, sz_escapes = [], []
        for i in range(28):
            encoded = codec.other.codec.encode(words[i].view(torch.bfloat16))
            sz_bytes.append(int(encoded.compressed_bytes)); sz_escapes.append(int(encoded.n_esc)); del encoded
        del xall, prior, counts, native
    finally: close(codec)
    order, savings = rank_layer_order(sz_bytes, rank_words)
    masks = {str(k): [i in order[:k] for i in range(28)] for k in KGRID}; rows = []
    for n in LENGTHS:
        source = words[:, :n].contiguous()
        source_sha = hashlib.sha256(source.cpu().numpy().astype('<i2', copy=False).tobytes()).hexdigest()
        for k in KGRID:
            spec = dict(q['mixed_codec'], mask=masks[str(k)])
            codec = codec_for(dict(method=f'mixed_rank256_k{k}', codec=spec), device)
            try:
                p, e = timed(lambda: codec.encode(source), torch, device, 1, 3)
                value, d = timed(lambda: codec.decode(p), torch, device, 1, 3)
                if not torch.equal(value, source): raise AssertionError('TABLES mixed roundtrip differs')
                wire = p.export()
                rows.append(dict(document='tables', tokens=n, k=k, original_word_sha256=source_sha, packet_bytes=len(wire), packet_sha256=hashlib.sha256(wire).hexdigest(),
                    encode_ms=e['wall_median_ms'], decode_ms=d['wall_median_ms'], encode=e, decode=d,
                    accounting=codec.accounting(p) if hasattr(codec, 'accounting') else {}))
                if hashlib.sha256(source.cpu().numpy().astype('<i2', copy=False).tobytes()).hexdigest() != source_sha:
                    raise AssertionError('calibration source words changed')
                del value, p, wire
            finally: close(codec)
            gc.collect(); torch.cuda.empty_cache()
    after = runtime_snapshot(torch)
    if before != after: raise AssertionError('calibration changed runtime flags')
    policy = freeze_policy(rows)
    policy.update(masks=masks, layer_order=order, layer_order_tokens=8192,
        layer_order_scope='One TABLES8192 stable native-bit-benefit ranking reused for512 and8192; actual full API policy calibration is per length', layer_diagnostics=[dict(layer=i, splitzip_native_bytes=sz_bytes[i],
        rans_used_words=rank_words[i], benefit_bits=savings[i], splitzip_escape_count=sz_escapes[i]) for i in range(28)], calibration_rows=rows,
        capture_source=q['capture_source'], calibration_source_receipt=q['calibration_source_receipt'],
        fixed_state=q['mixed_codec'], runtime_policy=before, runtime_policy_after=after, gpu_measured=True)
    policy['policy_sha256'] = digest_json(policy); write(out / 'selection.json', policy)


def validate_new_worker(data, q, job):
    if data.get('status') != 'completed' or not data.get('gpu_measured') or len(data['rows']) != 2:
        raise ValueError('new worker did not complete two CUDA rows')
    targets = {e['document']: e for e in q['entries']}
    if {r['document'] for r in data['rows']} != set(targets): raise ValueError('TEST coverage changed')
    proof_path = job / 'receiver.json'
    if sha(proof_path) != data['receiver_sha256']: raise ValueError('receiver receipt changed')
    rq = json.loads((job / 'receiver_request.json').read_text()); validate_receiver_request(rq)
    if any(rq[k] != q[k] for k in ('method', 'tokens', 'codec', 'execution')): raise ValueError('receiver configuration changed')
    proof = json.loads(proof_path.read_text()); checks = {c['id']: c for c in proof['checks']}
    if (not all(proof.get(k) for k in ('gpu_measured', 'no_source_kv_or_predictions', 'alternating_packets', 'independent_returned_outputs'))
        or data['runtime_policy'] != data['runtime_policy_after'] or proof['runtime_policy'] != data['runtime_policy']
        or proof['runtime_policy_after'] != data['runtime_policy']): raise ValueError('ownership/runtime proof incomplete')
    if len(proof['packet_immutability']) != 2 or not all(x['unchanged'] for x in proof['packet_immutability']):
        raise ValueError('fresh receiver compressed inputs changed')
    for row in data['rows']:
        t = targets[row['document']]; check = checks[row['id']]
        if any(row[k] != t[k] for k in ('shape', 'values', 'original_word_sha256')) or not row['original_word_exact']:
            raise ValueError('new worker original identity differs')
        if any(row[k] != q[k] for k in ('method', 'tokens', 'round', 'execution', 'policy_sha256')):
            raise ValueError('new worker frozen execution differs')
        path = Path(row['packet']).resolve()
        if not path.is_relative_to(job.resolve()) or sha(path) != row['packet_sha256'] or path.stat().st_size != row['packet_bytes']:
            raise ValueError('actual encoded wire changed')
        if check['word_sha256'] != t['original_word_sha256'] or check['packet_sha256'] != row['packet_sha256'] or not check['verified']:
            raise ValueError('receiver encoded wire proof differs')
        for op in ('encode', 'decode'):
            b, i = row[op]['batched'], row[op]['isolated']
            if any(b.get(k) != v for k, v in dict(warmup=10, iters=50, repeats=5).items()) or len(b['samples_seconds']) != 5 or len(i['samples']) != 5 or len(i['warmup_samples']) != 2:
                raise ValueError('timing convention changed')
            if any(not math.isfinite(x) or x <= 0 for x in [*b['samples_seconds'], b['wall_mean_ms'], i['wall_median_ms']]):
                raise ValueError('invalid CUDA timing')
    for key in ('packet_immutability', 'raw_packet_immutability'):
        if len(data[key]) != 2 or not all(x['unchanged'] for x in data[key]): raise ValueError('compressed inputs changed')
    return data['rows']


def report(result, out):
    lines = ['# Combined lossless transfer comparison', '', '**Status: ' + result['status'] + '.** GPU measured: **' + str(result['gpu_measured']) + '**.', '', SCOPE, '']
    if result.get('error'): lines += ['Stopped: `' + result['error'].replace('`', "'") + '`. Outputs are preserved.', '']
    for basis in ('batched', 'isolated'):
        lines += ['Timing: **' + basis + '**.', '', '| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |', '|---:|---|---|---:|---:|---:|']
        for r in result.get('aggregates', []):
            lines.append(f"| {r['tokens']} | {r['method']} | {r['round']} | {r['complete_bpv']:.5f} | {r[basis]['encode_ms']:.3f} | {r[basis]['decode_ms']:.3f} |")
        lines.append('')
    policy = result.get('selection')
    if policy:
        lines += ['Frozen mixed policy selected on TABLES before any TEST encode/decode.', '', '| Tokens | Gb/s | Precompressed k | Fresh k |', '|---:|---:|---:|---:|']
        for n in LENGTHS:
            for link in LINKS:
                ps = {p['objective']: p['k'] for p in policy['policies'] if p['tokens'] == n and p['effective_gbps'] == link}
                lines.append(f"| {n} | {link} | {ps['precompressed']} | {ps['fresh']} |")
        lines.append('')
    modeled = result.get('link_models', {}).get('rows', [])
    applied = result.get('frozen_policy_test', [])
    if modeled and applied:
        fixed = ('raw_bf16', 'splitzip_official_framed', 'exans_official_fixed_framed', COMPACT,
                 'team_dense_uniform', 'team_rank256_uniform')
        for basis in ('batched', 'isolated'):
            for objective in ('precompressed', 'fresh'):
                lines += [f'Calculated serial {objective} latency ({basis}), ms. Fixed baseline controls use the faster full before/after round.', '',
                    '| Tokens | Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 | Frozen mixed policy |',
                    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
                for n in LENGTHS:
                    for link in LINKS:
                        cells = []
                        for method in fixed:
                            values = [r[objective + '_serial_ms'] for r in modeled if
                                (r['tokens'], r['effective_gbps'], r['timing_basis'], r['method']) == (n, link, basis, method)]
                            cells.append(f'{min(values):.3f}' if values else 'pending')
                        choices = [r['serial_ms'] for r in applied if
                            (r['tokens'], r['effective_gbps'], r['timing_basis'], r['objective']) == (n, link, basis, objective)]
                        cells.append(f'{statistics.mean(choices):.3f}' if len(choices) == 2 else 'pending')
                        lines.append(f'| {n} | {link} | ' + ' | '.join(cells) + ' |')
                lines.append('')
    lines += ['`results.json` retains every fixed-k TEST result, frozen-policy TEST costs, and calculated 1/100/200/300/400 Gb/s serial wire + decode and encode + wire + decode costs. No TEST argmin, network measurement, or overlap speedup claim is made.', '',
        'Batched full GPU APIs use unchanged author timing bodies: 10 warmups × 50 calls × five repeats. Isolated calls use two warmups and five samples. Packet export, disk, upload, fitting, and cold setup are excluded. Complete bytes come from actual serialized files; shared predictors and native memory reservations are additional.', '',
        'Before/after raw, native SplitZip, and fixed ExANS controls run through the unchanged frozen worker in their original environments. Dense and Rank256 remain paired controls. New mixed and compact ExANS arms are local framed adapters, not unchanged author serving evaluations. Diagnostic profiler/component samples are separate and nonadditive. All results are lossless T; lossy quality and resident P remain follow-on work.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def parent(args):
    from scripts import bench_frozen_transfer as frozen
    out = frozen.confined(args.output); out.mkdir(parents=True, exist_ok=False)
    result = dict(format='combined_lossless_transfer_v1', status='failed', gpu_measured=False, scope=SCOPE, jobs={}, rows=[])
    try:
        result['measurement_source_before'] = source_receipts()
        selections, receipt = frozen.verified_inputs(); result['verification'] = receipt; result['source_before'] = frozen.source_checks()
        # Require actual CUDA first; unittest skip decorators cannot turn a missing
        # GPU into a successful readiness gate. Keep the frozen ExANS library path.
        smoke_py, smoke_env = frozen.worker_environment(selections['exans_official_fixed_framed_t512'], receipt)
        tests = ['tests.test_mixed_transfer.CudaTests', 'tests.test_compact_exans.CudaTests']
        smoke_code = ("import sys, unittest, torch; "
            "assert torch.cuda.is_available(), 'actual CUDA required'; "
            "assert 'RTX A4500' in torch.cuda.get_device_name(0) and torch.version.cuda == '12.8'; "
            "suite=unittest.defaultTestLoader.loadTestsFromNames(" + repr(tests) + "); "
            "r=unittest.TextTestRunner(verbosity=2).run(suite); "
            "sys.exit(0 if r.wasSuccessful() and not r.skipped and r.testsRun == 3 else 1)")
        spawn([smoke_py, '-c', smoke_code], out / 'cuda_smoke.log', smoke_env)
        result['cuda_smoke'] = dict(tests=tests, passed=True, log_sha256=sha(out / 'cuda_smoke.log'))
        sourcepath = ROOT / CALIBRATION_SOURCE / 'results.json'; old = json.loads(sourcepath.read_text())
        capture_source = old['recipe']['source']
        if capture_source['meta_sha256'] != META_SHA or capture_source['documents']['tables']['words_sha256'] != TABLES_SHA:
            raise ValueError('published TABLES source receipt differs')
        spec = new_spec('mixed_rank256_k28', selections, {'28': [True] * 28})
        selout = out / 'calibration'; selout.mkdir()
        sq = dict(output=str(selout), mixed_codec=spec, capture=str(ROOT / 'runs/llama32_3b_compare/captures'),
            capture_source=capture_source, calibration_source_receipt=dict(path=str(sourcepath), sha256=sha(sourcepath)))
        write(selout / 'request.json', sq)
        spawn([smoke_py, __file__, '--select-worker', selout / 'request.json'], selout / 'worker.log', smoke_env)
        policy = json.loads((selout / 'selection.json').read_text()); identity = policy.pop('policy_sha256')
        if digest_json(policy) != identity or not policy.get('gpu_measured'): raise ValueError('calibration policy receipt differs')
        policy['policy_sha256'] = identity; result['selection'] = policy
        write(out / 'policy.json', policy); result['policy_file_sha256'] = sha(out / 'policy.json')
        original_jobs = frozen.job_plan(selections)
        before = [j for j in original_jobs if j[1]['round'] == 'before']
        paired = [j for j in original_jobs if j[1]['round'] == 'paired']
        after = [j for j in original_jobs if j[1]['round'] == 'after']
        for name, q in before + paired + new_job_plan(selections, policy) + after:
            job = out / name; job.mkdir(); q['output'] = str(job); write(job / 'request.json', q)
            new = q['method'] in NEW_METHODS
            envq = q if not new else dict(q, method='exans_official_fixed_framed' if q['method'] == COMPACT else 'raw_bf16')
            py, env = frozen.worker_environment(envq, receipt)
            argv = worker_argv(py, job / 'request.json', q['method'])
            started = time.monotonic(); print('Running ' + name, flush=True); spawn(argv, job / 'worker.log', env)
            data = json.loads((job / 'worker_results.json').read_text())
            rows = validate_new_worker(data, q, job) if new else frozen.validate_worker(data, q, job)
            result['rows'].extend(rows); result['gpu_measured'] = True
            result['jobs'][name] = dict(argv=argv, returncode=0, command_wall_seconds=time.monotonic() - started,
                artifacts={str(p.relative_to(job)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in job.rglob('*')
                    if p.is_file() and p.suffix in ('.json', '.log') and p.name != 'full_trace.json'})
            result['aggregates'] = frozen.aggregate(result['rows']); write(out / 'results.json', result); report(result, out)
        result['source_after'] = frozen.source_checks()
        from scripts.check_nuc_baseline_readiness import frozen_sources
        result['frozen_source_after'] = frozen_sources(ROOT); result['measurement_source_after'] = source_receipts()
        if result['source_before'] != result['source_after'] or not result['frozen_source_after']['all_match'] or result['measurement_source_before'] != result['measurement_source_after']:
            raise ValueError('frozen, author, or measurement source changed during run')
        if sha(out / 'policy.json') != result['policy_file_sha256']: raise ValueError('frozen policy changed during TEST')
        result['link_models'] = frozen.link_models(result['aggregates'])
        result['frozen_policy_test'] = select_policy_rows(result['rows'], policy); result['status'] = 'completed'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        (out / 'failure.log').write_text(traceback.format_exc()); print(result['error'], file=sys.stderr, flush=True)
    finally:
        write(out / 'results.json', result); report(result, out); print('Report: ' + str(out / 'summary.md'), flush=True)
    return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path); mode.add_argument('--select-worker', type=Path)
    mode.add_argument('--worker', type=Path); mode.add_argument('--receive', type=Path)
    args = parser.parse_args()
    if args.output: return parent(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '16'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
    from scripts.bench_team_packet import cuda_check
    torch, device = cuda_check(args.device)
    if 'RTX A4500' not in torch.cuda.get_device_name(device) or torch.version.cuda != '12.8':
        raise RuntimeError('frozen NUC setting requires RTX A4500 / CUDA12.8')
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')}
    if versions != dict(torch='2.11.0+cu128', triton='3.6.0', numpy='2.3.5'):
        raise ValueError('frozen new-worker environment changed: ' + str(versions))
    path = args.select_worker or args.worker or args.receive
    q = json.loads(path.read_text())
    (select_worker if args.select_worker else worker if args.worker else receive)(q, torch, device)
    return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception: traceback.print_exc(); raise SystemExit(1)
