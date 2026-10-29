#!/usr/bin/env python3
"""Lossless critical-first fallback/rANS overlap ablation on frozen BF16 words.

The parent imports no tensor runtime. Masks, coefficients and author controls
are frozen; this experiment changes execution, never BF16 reconstruction.
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
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bench_combined_transfer as common
from scripts import bench_lossless_decode as previous
from scripts import bench_frozen_transfer as frozen

sha, write, digest_json, spawn = common.sha, common.write, common.digest_json, common.spawn
LENGTHS = (512, 8192)
KGRID = (2, 14)
MODES = ('reference', 'overlap')
FLAGS = dict(reference=False, overlap=True)
CUDA_TESTS = ('tests.test_mapped_decode_partition.CudaTests', 'tests.test_mixed_decode_overlap.CudaTests')
LINKS = (1, 5, 10, 25, 50, 75, 100, 200, 300, 400)
METHODS = tuple(f'overlap_{m}_k{k}' for k in KGRID for m in MODES)
PREVIOUS = 'benchmark_results/lossless_decode_a4500_20261029_125616_867717'
PREVIOUS_HASHES = dict(results='932544c4fee02fea73537879e10cbfc59d7af2f5de36c7e1162b3d0df3e2a9f4',
    package='632a624ac26083ae7b8ddb33fb250f5c2ff4b9b01e485b15bdeba3817a4d6296',
    policy='e07fc481ba6f8f7cbeb5650b7517b81354d50c7b65e94eec2bee5f61003fd502')
SCOPE = ('Lossless GPU transfer ablation: decode critical fallback predecessors first, then overlap remaining '
    'SplitZip fallback decode with rANS. The reference is unchanged mapped-I/O plus selective-context decoding. '
    'Identical original BF16 words, fitted state and frozen TABLES k2/k14 masks; both arms must preserve '
    'complete prior bundled packets. No refitting or TEST policy selection. These two documents are reused '
    'for development, not independent generalization evidence. Complete GPU calls and actual serialized '
    'bytes; serial link calculations are not NIC/SSD, resident P, lossy quality or serving measurements.')


def method_parts(method):
    match = re.fullmatch(r'overlap_(reference|overlap)_k(2|14)', method)
    if match is None: raise ValueError('unknown lossless execution arm')
    return match[1], int(match[2])


def verify_previous():
    calibration, calibration_receipt = previous.verify_previous()
    root = ROOT / PREVIOUS; package = frozen.verify_package(root)
    if package['files'] != 214: raise ValueError('completed decoder package file count changed')
    for name, key in (('results.json', 'results'), ('package_manifest.json', 'package'), ('policy.json', 'policy')):
        if sha(root / name) != PREVIOUS_HASHES[key]: raise ValueError('completed decoder receipt changed: ' + name)
    old = json.loads((root / 'results.json').read_text())
    policy = json.loads((root / 'policy.json').read_text()); identity = policy.pop('policy_sha256')
    if digest_json(policy) != identity: raise ValueError('original fixed policy digest changed')
    policy['policy_sha256'] = identity
    if (old['status'] != 'completed' or not old['gpu_measured'] or len(old['rows']) != 68
        or len(old['jobs']) != 34 or not old['cuda_smoke']['passed'] or old['cuda_smoke']['expected_tests'] != 3
        or not old['frozen_source_after']['all_match'] or old['frozen_masks'] != policy or policy != calibration
        or old['measurement_source_before'] != old['measurement_source_after']
        or old['source_before'] != old['source_after'] or old['shared_artifacts_before'] != old['shared_artifacts_after']
        or old['native_runtime_before']['runtime_libraries'] != old['native_runtime_after']['runtime_libraries']
        or old['native_runtime_before']['library_sha256'] != old['native_runtime_after']['library_sha256']
        or sha(root / 'cuda_smoke.log') != old['cuda_smoke']['log_sha256']
        or sha(root / 'policy.json') != old['policy_file_sha256']
        or any(not x['original_word_exact'] or x['decoded_word_sha256'] != x['original_word_sha256'] for x in old['rows'])):
        raise ValueError('previous exact CUDA study is incomplete')
    if previous.source_receipts() != old['measurement_source_after']:
        raise ValueError('previous decoder measurement sources changed')
    if len(old['packet_equivalence']) != 8 or not all(all(e[k] for k in
        ('all_rans_fields_equal', 'all_wire_equal')) for e in old['packet_equivalence']):
        raise ValueError('previous wire equivalence proof incomplete')
    for name in old['jobs']:
        worker = json.loads((root / name / 'worker_results.json').read_text())
        proof_path = root / name / 'receiver.json'; proof = json.loads(proof_path.read_text())
        if (sha(proof_path) != worker['receiver_sha256'] or len(proof['checks']) != 2
            or not all(proof.get(k) for k in ('gpu_measured', 'no_source_kv_or_predictions',
                'alternating_packets', 'independent_returned_outputs')) or not all(c['verified'] for c in proof['checks'])):
            raise ValueError('previous independent receiver proof incomplete: ' + name)
    masks = {str(k): policy['masks'][str(k)] for k in KGRID}
    if any(len(masks[str(k)]) != 28 or any(type(x) is not bool for x in masks[str(k)]) or sum(masks[str(k)]) != k for k in KGRID):
        raise ValueError('frozen masks differ')
    refs = {f'decode_both_k{k}_t{n}': [r for r in old['rows'] if r['method'] == f'decode_both_k{k}' and r['tokens'] == n]
            for k in KGRID for n in LENGTHS}
    if any(len(rows) != 2 for rows in refs.values()): raise ValueError('previous bundled packet identities missing')
    fixed = dict(format='lossless_overlap_v3', masks=masks, modes=list(MODES), k_grid=list(KGRID),
        links_gbps=list(LINKS), previous_policy_sha256=identity, previous_results_sha256=PREVIOUS_HASHES['results'],
        selection_scope='Original TABLES masks reused; no new policy search or TEST selection', reference_packets=refs)
    fixed['policy_sha256'] = digest_json(fixed)
    return fixed, dict(path=str(root), package=package, hashes=PREVIOUS_HASHES, calibration_chain=calibration_receipt)


def source_receipts():
    names = ('scripts/bench_lossless_overlap.py', 'scripts/run_lossless_overlap_a4500.sh',
        'docs/lossless_overlap_v3.md', 'kv_compare/mixed_decode_overlap.py',
        'kv_compare/mapped_decode_partition.py', 'tests/test_mapped_decode_partition.py',
        'tests/test_mixed_decode_overlap.py', 'tests/test_lossless_overlap.py')
    return dict(frozen_decode=previous.source_receipts(), new={name: sha(ROOT / name) for name in names})


def native_runtime_receipts(selections, receipt):
    return previous.native_runtime_receipts(selections, receipt)


def job_plan(selections, policy):
    jobs = []
    for n in LENGTHS:
        for k in KGRID:
            for mode in MODES:
                method = f'overlap_{mode}_k{k}'
                spec = common.new_spec(f'mixed_rank256_k{k}', selections, policy['masks'])
                spec.update(kind='mixed_decode_overlap', selective=True, bundle=True,
                    mapped_io=True, selective_decode=True, overlap_decode=FLAGS[mode])
                target = selections[f'team_rank256_uniform_t{n}']
                jobs.append((f'{method}_t{n}', dict(method=method, tokens=n, codec=spec, entries=target['entries'],
                    raw_entries=selections[f'raw_bf16_t{n}']['entries'],
                    reference_entries=policy.get('reference_packets', {}).get(f'decode_both_k{k}_t{n}', []),
                    execution='direct', round='paired', policy_sha256=policy['policy_sha256'],
                    timing_source=str(ROOT / 'runs/_llama_official/sources/splitzip/bench_codec_throughput.py'))))
    return jobs


def compact_plan(selections, policy):
    return previous.compact_plan(selections, policy)


def validate_receiver_request(q):
    if set(q) != {'method', 'tokens', 'codec', 'entries', 'execution', 'output'}:
        raise ValueError('receiver accepts only codec state, packets and expected assertions')
    mode, k = method_parts(q['method']); spec = q['codec']; base = dict(spec)
    if (spec.get('kind') != 'mixed_decode_overlap' or spec.get('selective') is not True
        or spec.get('bundle') is not True or spec.get('mapped_io') is not True
        or spec.get('selective_decode') is not True or type(spec.get('overlap_decode')) is not bool
        or spec['overlap_decode'] != FLAGS[mode]):
        raise ValueError('receiver overlap optimization flags differ')
    base.pop('overlap_decode'); base['kind'] = 'mixed_decode_fast'
    previous.validate_receiver_request(dict(q, method=f'decode_both_k{k}', codec=base))


def codec_for(q, device):
    mode, _ = method_parts(q['method']); spec = q['codec']
    if sha(Path(spec['state']) / 'manifest.json') != spec['manifest_sha256']:
        raise ValueError('frozen predictor manifest changed')
    args = (spec['state'], spec['splitzip_source'], spec['splitzip_state'], spec['splitzip_state_sha256'], spec['mask'], device)
    if mode == 'reference':
        from kv_compare.mixed_decode_fast import DecodeFastMixedTransfer
        codec = DecodeFastMixedTransfer(*args, arm=spec['arm'], selective=True, bundle=True,
            mapped_io=True, selective_decode=True)
    else:
        from kv_compare.mixed_decode_overlap import OverlapMixedTransfer
        codec = OverlapMixedTransfer(*args, arm=spec['arm'], selective=True, bundle=True,
            mapped_io=True, selective_decode=True, overlap_decode=True)
    if codec.predictor_state_sha256 != spec['predictor_state_sha256']: raise ValueError('predictor tensors changed')
    return codec


def rans_fields(packet):
    return previous.rans_fields(packet)


def validate_packet_equivalence(rows, complete=True):
    groups = {}
    for row in rows:
        if row['method'] not in METHODS: continue
        mode, k = method_parts(row['method']); key = (row['tokens'], k, row['document'])
        group = groups.setdefault(key, {})
        if mode in group: raise ValueError('duplicate packet-equivalence row')
        group[mode] = row
    proofs = []
    for (n, k, doc), group in sorted(groups.items()):
        if set(group) != set(MODES):
            if complete: raise ValueError('both paired overlap arms required')
            continue
        reference = group['reference']
        if any(group[m]['rans_fields'] != reference['rans_fields'] for m in MODES):
            raise ValueError('execution changed rANS symbols, states or directories')
        if any(any(group[m].get(key) != reference.get(key) for key in
            ('packet_bytes', 'packet_sha256', 'original_word_sha256')) for m in MODES):
            raise ValueError('decode execution changed complete wire bytes or original words')
        proofs.append(dict(tokens=n, k=k, document=doc, all_rans_fields_equal=True, all_wire_equal=True))
    if complete and len(proofs) != len(LENGTHS) * len(KGRID) * 2: raise ValueError('paired packet coverage incomplete')
    return proofs


def receive(q, torch, device):
    from scripts.frozen_transfer_worker import runtime_snapshot
    from scripts.bench_team_packet import check_original
    validate_receiver_request(q); before = runtime_snapshot(torch)
    codec = codec_for(q, device); packets = common.restore_entries(q['entries'], device); held = []; checks = {}
    try:
        for i in (0, 1, 0):
            e = q['entries'][i]; value = codec.decode(packets[i]); digest = check_original(value, e)
            for old_value, old_entry in held: check_original(old_value, old_entry)
            held.append((value, e)); checks[e['id']] = dict(id=e['id'], verified=True, word_sha256=digest,
                shape=list(value.shape), packet_sha256=e['packet_sha256'])
        immutable = common.packet_checks(packets, q['entries'])
    finally: common.close(codec)
    after = runtime_snapshot(torch)
    if before != after: raise AssertionError('receiver runtime flags changed')
    write(q['output'], dict(status='completed', gpu_measured=True, no_source_kv_or_predictions=True,
        alternating_packets=True, independent_returned_outputs=True, checks=list(checks.values()),
        packet_immutability=immutable, runtime_policy=before, runtime_policy_after=after))


def interval_overlap_ms(left, right):
    """Intersection length of two finite ordered CUDA event intervals."""
    if (not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple))
        or len(left) != 2 or len(right) != 2):
        raise ValueError('two ordered intervals required')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (*left, *right)):
        raise ValueError('finite interval endpoints required')
    if left[1] < left[0] or right[1] < right[0]:
        raise ValueError('interval ends before it starts')
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def schedule_diagnostics(codec, source, packet, parsed, torch, device):
    """Observe an already-warm candidate schedule outside performance samples.

    Events mark stream intervals, including waits and host enqueue gaps. Their
    intersection demonstrates overlapping scheduled intervals, not concurrent
    SM execution or additive kernel work. Production methods are restored even
    if a diagnostic call fails.
    """
    from kv_compare import mixed_decode_overlap as overlap_module
    if not codec.overlap_decode or not codec.execution_plan['overlap_decode_active']:
        raise ValueError('schedule diagnostics require the overlap candidate')
    original_subset = overlap_module.decode_subset_into
    original_native = codec.native.decode_mixed
    before = hashlib.sha256(packet.export()).hexdigest()
    captured = []

    def record_call(name, function, *args, **kwargs):
        stream = torch.cuda.current_stream(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        value = function(*args, **kwargs)
        end.record(stream)
        captured.append((name, start, end, stream.cuda_stream))
        return value

    def subset(*args, **kwargs):
        name = kwargs.get('partition')
        if name not in ('critical', 'bulk'):
            raise ValueError('unknown diagnostic fallback partition')
        return record_call(name, original_subset, *args, **kwargs)

    def native(*args, **kwargs):
        # The caller records the end after the native controller's rANS joins,
        # before OverlapMixedTransfer enqueues its final bulk-stream join.
        return record_call('rans_controller', original_native, *args, **kwargs)

    samples = []
    try:
        overlap_module.decode_subset_into = subset
        codec.native.decode_mixed = native
        for _ in range(3):
            captured.clear()
            caller = torch.cuda.current_stream(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(caller)
            value = codec.decode_native(parsed)
            end.record(caller)
            torch.cuda.synchronize(device)
            if not torch.equal(value, source):
                raise AssertionError('diagnostic schedule changed original BF16 words')
            intervals = {}
            streams = {}
            for name, begin, finish, stream in captured:
                if name in intervals:
                    raise AssertionError('diagnostic stage was called more than once')
                intervals[name] = [start.elapsed_time(begin), start.elapsed_time(finish)]
                streams[name] = stream
            if set(intervals) != {'critical', 'bulk', 'rans_controller'}:
                raise AssertionError('diagnostic schedule stages missing')
            samples.append(dict(full_ms=start.elapsed_time(end),
                caller_stream=caller.cuda_stream, stream_ids=streams,
                critical_interval_ms=intervals['critical'], bulk_interval_ms=intervals['bulk'],
                rans_controller_interval_ms=intervals['rans_controller'],
                bulk_rans_interval_intersection_ms=interval_overlap_ms(intervals['bulk'], intervals['rans_controller']),
                original_words_exact=True))
            del value
    finally:
        overlap_module.decode_subset_into = original_subset
        codec.native.decode_mixed = original_native
    after = hashlib.sha256(packet.export()).hexdigest()
    if before != after:
        raise AssertionError('diagnostic schedule mutated compressed packet')
    return dict(calls=3, already_warm=True, samples=samples, packet_sha256_before=before,
        packet_sha256_after=after, packet_unchanged=True,
        scope='Separate instrumented native decode diagnostics; all intervals relative to caller start, '
              'include stream waits and enqueue gaps, are nonadditive, and do not establish concurrent SM execution')


def diagnostics(codec, source, packet, out, torch, device):
    """Warm profiler registration outside the codec CPU profile."""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities):
        p = codec.encode(source); value = codec.decode(p); torch.cuda.synchronize(device)
        del p, value
    cpu_profile = cProfile.Profile()
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        cpu_profile.enable()
        for _ in range(3):
            p = codec.encode(source); value = codec.decode(p); torch.cuda.synchronize(device); del p, value
        cpu_profile.disable()
    prof.export_chrome_trace(str(out / 'full_trace.json'))
    stats = pstats.Stats(cpu_profile)
    cpu = sorted([dict(file=k[0], line=k[1], function=k[2], primitive_calls=v[0], calls=v[1],
        self_seconds=v[2], cumulative_seconds=v[3]) for k, v in stats.stats.items()], key=lambda x: -x['cumulative_seconds'])[:50]
    keys = [dict(name=e.key, calls=e.count, self_cpu_us=e.self_cpu_time_total,
        self_gpu_us=getattr(e, 'self_device_time_total', 0), cpu_memory_bytes=e.cpu_memory_usage,
        gpu_memory_bytes=getattr(e, 'device_memory_usage', 0)) for e in prof.key_averages()]
    kernels = {}
    for event in prof.events():
        if str(event.device_type).endswith('CUDA'):
            r = kernels.setdefault(event.name, dict(name=event.name, calls=0, gpu_us=0.0))
            r['calls'] += 1; r['gpu_us'] += float(getattr(event, 'device_time_total', 0))
    from scripts.bench_llama_causal import timed
    parsed, unpack = timed(lambda: codec.unpack(packet), torch, device, 1, 3)
    value, native = timed(lambda: codec.decode_native(parsed), torch, device, 1, 3); del value
    data = dict(calls=3, profiler_registration_warmup_calls=1, cpu_profile_started_after_profiler_entry=True,
        scope='Separate full E/D diagnostics; accumulated kernel work and stage samples are nonadditive',
        kernels=sorted(kernels.values(), key=lambda x: -x['gpu_us']), key_averages=keys, cprofile=cpu,
        allocation_ops=[r for r in keys if any(x in r['name'] for x in ('empty', 'resize', 'zeros', 'alloc'))],
        nonadditive_stages=dict(unpack=unpack, native_decode=native))
    if getattr(codec, 'overlap_decode', False):
        data['schedule_intervals'] = schedule_diagnostics(codec, source, packet, parsed, torch, device)
    write(out / 'diagnostics.json', data)


def measure(q, torch, device):
    from scripts.frozen_transfer_worker import runtime_snapshot, load_author_bench, timing_samples, packet_checks as old_checks
    from scripts.bench_team_gpu import load_packets
    from scripts.bench_team_packet import check_original
    from scripts.bench_llama_causal import timed
    from kv_compare.framed_baselines import Baseline
    from kv_compare.causal_packet import Packet
    out = Path(q['output']); before = runtime_snapshot(torch)
    bench, timing_receipt = load_author_bench(q['timing_source'], torch)
    start = time.perf_counter(); codec = codec_for(q, device); torch.cuda.synchronize(device)
    cold = 1000 * (time.perf_counter() - start)
    raw_packets = load_packets(dict(method='raw_bf16', entries=q['raw_entries']), device)
    raw = Baseline('raw_bf16', device); bydoc = {e['document']: (p, e) for e, p in zip(q['raw_entries'], raw_packets)}
    references = {e['document']: e for e in q['reference_entries']}; rows = []; entries = []; packets = []
    mode, k = method_parts(q['method'])
    try:
        for target in q['entries']:
            rp, re = bydoc[target['document']]
            if any(re[x] != target[x] for x in ('shape', 'values', 'original_word_sha256')):
                raise ValueError('raw/target original word identity differs')
            source = raw.decode(rp); check_original(source, target)
            encoded = codec.encode(source); check_original(codec.decode(encoded), target)
            wire = encoded.export(); entry = {x: target[x] for x in ('id', 'document', 'shape', 'values', 'original_word_sha256')}
            entry.update(packet=str(out / (target['id'] + '_encoded.kvc')), packet_bytes=len(wire),
                packet_sha256=hashlib.sha256(wire).hexdigest())
            previous = references[target['document']]
            if any(previous[x] != target[x] for x in ('shape', 'values', 'original_word_sha256')):
                raise ValueError('previous mixed words differ')
            if any(entry[x] != previous[x] for x in ('packet_bytes', 'packet_sha256')):
                raise AssertionError('unchanged-frame execution differs from previous original mixed packet')
            Path(entry['packet']).write_bytes(wire); packet = Packet.restore(wire, device)
            check_original(codec.decode(packet), target); packets.append(packet); entries.append(entry); del encoded, wire
            eb = timing_samples([bench(lambda: codec.encode(source), str(device), 10, 50) for _ in range(5)], 2 * target['values'])
            ep, ei = timed(lambda: codec.encode(source), torch, device, 2, 5)
            check_original(codec.decode(ep), target); del ep
            db = timing_samples([bench(lambda: codec.decode(packet), str(device), 10, 50) for _ in range(5)], 2 * target['values'])
            value, di = timed(lambda: codec.decode(packet), torch, device, 2, 5); check_original(value, target); del value
            accounting = codec.accounting(packet)
            if accounting['packet_bytes'] != entry['packet_bytes']: raise AssertionError('complete wire accounting differs')
            row = dict(entry, method=q['method'], tokens=q['tokens'], round=q['round'], execution=q['execution'],
                original_word_exact=True, decoded_word_sha256=target['original_word_sha256'],
                encode=dict(batched=eb, isolated=ei), decode=dict(batched=db, isolated=di),
                rans_fields=rans_fields(packet), accounting=accounting, policy_sha256=q['policy_sha256'],
                execution_plan=codec.execution_plan)
            if q['tokens'] == 8192 and 'text' in target['document'] and k in KGRID:
                diagnostics(codec, source, packet, out, torch, device); row['diagnostics'] = str(out / 'diagnostics.json')
            check_original(source, target); rows.append(row); del source; gc.collect()
            print(target['document'] + ': full wire and original words verified', flush=True)
        immutable = common.packet_checks(packets, entries); raw_immutable = old_checks(raw_packets, q['raw_entries'])
    finally: common.close(codec)
    after = runtime_snapshot(torch)
    if before != after: raise AssertionError('worker runtime flags changed')
    return dict(rows=rows, entries=entries, packet_immutability=immutable, raw_packet_immutability=raw_immutable,
        runtime_policy=before, runtime_policy_after=after, author_timing=timing_receipt,
        graph_capture_count=0, cold_state_initialization_wall_ms=cold)


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
    checks = {c['id']: c for c in proof['checks']}
    for row in data['rows']:
        c = checks[row['id']]
        if not c['verified'] or c['word_sha256'] != row['original_word_sha256'] or c['packet_sha256'] != row['packet_sha256']:
            raise AssertionError('fresh receiver original words differ')
    data.update(status='completed', gpu_measured=True, receiver_sha256=sha(out / 'receiver.json'),
        device=torch.cuda.get_device_name(device), software={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')},
        scope=SCOPE, memory_scope='Isolated samples include Torch baseline/incremental peaks, output and scratch; native allocations outside Torch unmeasured')
    write(out / 'worker_results.json', data)


def validate_worker(data, q, job):
    if data.get('status') != 'completed' or not data.get('gpu_measured') or len(data['rows']) != 2:
        raise ValueError('new CUDA worker did not complete both TEST rows')
    targets = {e['document']: e for e in q['entries']}
    if {r['document'] for r in data['rows']} != set(targets): raise ValueError('TEST coverage differs')
    proof_path = job / 'receiver.json'
    if sha(proof_path) != data['receiver_sha256']: raise ValueError('receiver receipt changed')
    rq = json.loads((job / 'receiver_request.json').read_text()); validate_receiver_request(rq)
    if any(rq[k] != q[k] for k in ('method', 'tokens', 'codec', 'execution')): raise ValueError('receiver configuration changed')
    proof = json.loads(proof_path.read_text()); checks = {c['id']: c for c in proof['checks']}
    if (not all(proof.get(k) for k in ('gpu_measured', 'no_source_kv_or_predictions', 'alternating_packets', 'independent_returned_outputs'))
        or data['runtime_policy'] != data['runtime_policy_after'] or proof['runtime_policy'] != data['runtime_policy']
        or proof['runtime_policy_after'] != data['runtime_policy'] or data['graph_capture_count'] != 0):
        raise ValueError('runtime/ownership proof incomplete')
    if len(proof['packet_immutability']) != 2 or not all(x['unchanged'] for x in proof['packet_immutability']):
        raise ValueError('fresh receiver compressed inputs changed')
    for row in data['rows']:
        target = targets[row['document']]; c = checks[row['id']]
        if (any(row[k] != target[k] for k in ('shape', 'values', 'original_word_sha256'))
            or not row['original_word_exact'] or row['decoded_word_sha256'] != target['original_word_sha256']):
            raise ValueError('original BF16 words differ')
        if any(row[k] != q[k] for k in ('method', 'tokens', 'round', 'execution', 'policy_sha256')):
            raise ValueError('frozen execution changed')
        path = Path(row['packet']).resolve()
        if not path.is_relative_to(job.resolve()) or sha(path) != row['packet_sha256'] or path.stat().st_size != row['packet_bytes']:
            raise ValueError('actual encoded file differs')
        if c['word_sha256'] != target['original_word_sha256'] or c['packet_sha256'] != row['packet_sha256'] or not c['verified']:
            raise ValueError('receiver proof differs')
        for op in ('encode', 'decode'):
            b, i = row[op]['batched'], row[op]['isolated']
            if (any(b.get(k) != v for k, v in dict(warmup=10, iters=50, repeats=5).items())
                or len(b['samples_seconds']) != 5 or len(i['samples']) != 5 or len(i['warmup_samples']) != 2
                or any(not math.isfinite(x) or x <= 0 for x in [*b['samples_seconds'], b['wall_mean_ms'], i['wall_median_ms']])):
                raise ValueError('timing convention or measurements differ')
        if set(row['rans_fields']) != {'rans_words', 'rans_offsets', 'rans_relative'}: raise ValueError('rANS equality fields missing')
    for key in ('packet_immutability', 'raw_packet_immutability'):
        if len(data[key]) != 2 or not all(x['unchanged'] for x in data[key]): raise ValueError('compressed inputs changed')
    return data['rows']


def link_models(aggregates):
    return previous.link_models(aggregates)


def paired_insights(aggregates):
    rows = []
    for n in LENGTHS:
        for k in KGRID:
            group = {method_parts(a['method'])[0]: a for a in aggregates if a['tokens'] == n and a['method'] in METHODS and method_parts(a['method'])[1] == k}
            if set(group) != set(MODES): continue
            for mode in MODES[1:]:
                for basis in ('batched', 'isolated'):
                    ref, test = group['reference'], group[mode]
                    rows.append(dict(tokens=n, k=k, mode=mode, timing_basis=basis,
                        complete_bytes_delta=test['packet_bytes']-ref['packet_bytes'],
                        encode_speedup=ref[basis]['encode_ms']/test[basis]['encode_ms'],
                        decode_speedup=ref[basis]['decode_ms']/test[basis]['decode_ms'],
                        encode_delta_ms=test[basis]['encode_ms']-ref[basis]['encode_ms'],
                        decode_delta_ms=test[basis]['decode_ms']-ref[basis]['decode_ms']))
    return rows


def report(result, out):
    lines = ['# Lossless critical-first fallback/rANS overlap comparison', '', '**Status: ' + result['status'] + '.** GPU measured: **' + str(result['gpu_measured']) + '**.', '', SCOPE, '']
    if result.get('error'): lines += ['Stopped: `' + result['error'].replace('`', "'") + '`. Outputs preserved.', '']
    for basis in ('batched', 'isolated'):
        lines += ['Timing: **' + basis + '**.', '', '| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |', '|---:|---|---|---:|---:|---:|']
        for a in result.get('aggregates', []):
            lines.append(f"| {a['tokens']} | {a['method']} | {a['round']} | {a['complete_bpv']:.5f} | {a[basis]['encode_ms']:.3f} | {a[basis]['decode_ms']:.3f} |")
        lines.append('')
    if result.get('paired_insights'):
        lines += ['Paired changes relative to the unchanged mapped-I/O and selective-context execution. Speedup below 1 means slower.', '',
            '| Tokens | k | Change | Basis | Wire delta bytes | Encode speedup | Decode speedup |', '|---:|---:|---|---|---:|---:|---:|']
        for a in result['paired_insights']:
            lines.append(f"| {a['tokens']} | {a['k']} | {a['mode']} | {a['timing_basis']} | {a['complete_bytes_delta']:.0f} | {a['encode_speedup']:.3f} | {a['decode_speedup']:.3f} |")
        lines.append('')
    modeled = result.get('link_models', {}).get('rows', [])
    if modeled:
        lines += ['Link summary below uses **8K isolated** timings only. All lengths and both timing bases remain in `results.json`. Baseline controls use the faster complete before/after round for each objective. No TEST policy selection.', '']
        for objective in ('precompressed', 'fresh'):
            def cell(method, link):
                values = [a[objective+'_serial_ms'] for a in modeled if
                    (a['tokens'], a['effective_gbps'], a['timing_basis'], a['method']) == (8192, link, 'isolated', method)]
                return f'{min(values):.3f}' if values else 'pending'
            controls = ('raw_bf16', 'splitzip_official_framed', 'exans_official_fixed_framed',
                        common.COMPACT, 'team_dense_uniform', 'team_rank256_uniform')
            lines += [f'Calculated serial {objective} cost, **8K isolated**, ms.', '',
                '| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |',
                '|---:|---:|---:|---:|---:|---:|---:|']
            for link in LINKS:
                lines.append(f'| {link} | ' + ' | '.join(cell(method, link) for method in controls) + ' |')
            lines.append('')
            for k in KGRID:
                lines += [f'Fixed k{k} execution arms, serial {objective}, **8K isolated**, ms.', '',
                    '| Gb/s | SplitZip | Mapped reference | Critical-first overlap |', '|---:|---:|---:|---:|']
                methods = ('splitzip_official_framed',) + tuple(f'overlap_{mode}_k{k}' for mode in MODES)
                for link in LINKS:
                    lines.append(f'| {link} | ' + ' | '.join(cell(method, link) for method in methods) + ' |')
                lines.append('')
    lines += ['Both arms must preserve the complete prior bundled packet, including identical compact rANS words and directories. Fresh receivers see only compressed packets and state, alternate 0/1/0, retain previous outputs, and verify original-word hashes.', '',
        'Full GPU E/D calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls have two warmups and five samples. Export, disk, upload, fitting and cold setup are excluded. Complete bytes use actual serialized files. Shared state and native reservations are additional.', '',
        'Original author sources, baseline environments, predictor state, TABLES masks and all prior experiment source files stay unchanged. Partitioned mapped decoding and its overlap schedule are local integration changes, not the official baseline implementation. New-arm CPU/CUDA profiles are warmed, separate diagnostics and nonadditive; the compact control retains its original diagnostic procedure. Full traces remain local. No measured network, resident P, lossy quality, or serving speed claim.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def parent(args):
    out = frozen.confined(args.output); out.mkdir(parents=True, exist_ok=False)
    result = dict(format='lossless_overlap_v3', status='failed', gpu_measured=False, scope=SCOPE, rows=[], jobs={})
    try:
        policy, old_receipt = verify_previous(); result['previous'] = old_receipt
        result['measurement_source_before'] = source_receipts()
        selections, receipt = frozen.verified_inputs(); result['verification'] = receipt; result['source_before'] = frozen.source_checks()
        result['native_runtime_before'] = native_runtime_receipts(selections, receipt)
        fixed_files = {p: item['sha256'] for p, item in receipt['fixed_artifacts'].items() if not p.endswith(('.kvc', '.kvp'))}
        result['shared_artifacts_before'] = {p: sha(p) for p in fixed_files}
        if result['shared_artifacts_before'] != fixed_files: raise ValueError('frozen shared artifacts differ')
        write(out / 'policy.json', policy); result['policy_file_sha256'] = sha(out / 'policy.json'); result['frozen_masks'] = policy
        py, env = frozen.worker_environment(selections['exans_official_fixed_framed_t512'], receipt)
        tests = list(CUDA_TESTS)
        smoke = ("import sys,unittest,torch; assert torch.cuda.is_available(),'actual CUDA required'; "
            "assert 'RTX A4500' in torch.cuda.get_device_name(0) and torch.version.cuda=='12.8'; "
            "s=unittest.defaultTestLoader.loadTestsFromNames(" + repr(tests) + "); "
            "r=unittest.TextTestRunner(verbosity=2).run(s); sys.exit(0 if r.wasSuccessful() and not r.skipped and r.testsRun==2 else 1)")
        spawn([py, '-c', smoke], out / 'cuda_smoke.log', env)
        result['cuda_smoke'] = dict(tests=tests, expected_tests=2, passed=True, log_sha256=sha(out / 'cuda_smoke.log'))
        originals = frozen.job_plan(selections)
        before = [j for j in originals if j[1]['round'] == 'before']; after = [j for j in originals if j[1]['round'] == 'after']
        paired = [j for j in originals if j[1]['round'] == 'paired']
        compact = compact_plan(selections, policy)
        for name, q in before + paired + job_plan(selections, policy) + compact + after:
            job = out / name; job.mkdir(); q['output'] = str(job); write(job / 'request.json', q)
            new = q['method'] in METHODS; is_compact = q['method'] == common.COMPACT
            envq = dict(q, method='raw_bf16') if new else dict(q, method='exans_official_fixed_framed') if is_compact else q
            py, env = frozen.worker_environment(envq, receipt)
            script = __file__ if new else common.__file__ if is_compact else ROOT / 'scripts/frozen_transfer_worker.py'
            argv = [str(py), str(script), '--worker' if new or is_compact else '--request', str(job / 'request.json'), '--device', 'cuda:0']
            start = time.monotonic(); print('Running ' + name, flush=True); spawn(argv, job / 'worker.log', env)
            data = json.loads((job / 'worker_results.json').read_text())
            validator = validate_worker if new else common.validate_new_worker if is_compact else frozen.validate_worker
            result['rows'].extend(validator(data, q, job)); result['gpu_measured'] = True
            result['jobs'][name] = dict(argv=argv, returncode=0, command_wall_seconds=time.monotonic()-start,
                artifacts={str(p.relative_to(job)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in job.rglob('*')
                           if p.is_file() and p.suffix in ('.json', '.log') and p.name != 'full_trace.json'})
            result['packet_equivalence'] = validate_packet_equivalence(result['rows'], complete=False)
            result['aggregates'] = frozen.aggregate(result['rows']); result['paired_insights'] = paired_insights(result['aggregates'])
            write(out / 'results.json', result); report(result, out)
        if len(result['jobs']) != 26 or len(result['rows']) != 52:
            raise ValueError('complete paired experiment requires 26 workers and 52 rows')
        result['packet_equivalence'] = validate_packet_equivalence(result['rows'])
        result['source_after'] = frozen.source_checks(); result['measurement_source_after'] = source_receipts()
        result['shared_artifacts_after'] = {p: sha(p) for p in fixed_files}
        result['native_runtime_after'] = native_runtime_receipts(selections, receipt)
        from scripts.check_nuc_baseline_readiness import frozen_sources
        result['frozen_source_after'] = frozen_sources(ROOT)
        if (result['source_before'] != result['source_after'] or not result['frozen_source_after']['all_match']
            or result['measurement_source_before'] != result['measurement_source_after']
            or result['shared_artifacts_before'] != result['shared_artifacts_after']
            or result['native_runtime_before']['runtime_libraries'] != result['native_runtime_after']['runtime_libraries']
            or result['native_runtime_before']['library_sha256'] != result['native_runtime_after']['library_sha256']
            or sha(out / 'policy.json') != result['policy_file_sha256']):
            raise ValueError('author/frozen/measurement/shared source or fixed policy changed')
        result['link_models'] = link_models(result['aggregates']); result['status'] = 'completed'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        (out / 'failure.log').write_text(traceback.format_exc()); print(result['error'], file=sys.stderr, flush=True)
    finally:
        write(out / 'results.json', result); report(result, out); print('Report: ' + str(out / 'summary.md'), flush=True)
    return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--device', default='cuda:0')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--output', type=Path); modes.add_argument('--worker', type=Path); modes.add_argument('--receive', type=Path)
    args = parser.parse_args()
    if args.output: return parent(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '16'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
    from scripts.bench_team_packet import cuda_check
    torch, device = cuda_check(args.device)
    if 'RTX A4500' not in torch.cuda.get_device_name(device) or torch.version.cuda != '12.8':
        raise RuntimeError('frozen setting requires RTX A4500 / CUDA12.8')
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')}
    if versions != dict(torch='2.11.0+cu128', triton='3.6.0', numpy='2.3.5'):
        raise ValueError('frozen main-worker environment changed: ' + str(versions))
    q = json.loads((args.worker or args.receive).read_text())
    (worker if args.worker else receive)(q, torch, device); return 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except Exception: traceback.print_exc(); raise SystemExit(1)
