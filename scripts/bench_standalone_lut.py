#!/usr/bin/env python3
"""Standalone lossless full-CDF inverse lookup on unchanged original BF16 packets.

All 28 layers use our predictive coder. SplitZip and ExANS are independent
unchanged controls; neither is a component of the candidate codec.
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
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bench_frozen_transfer as frozen
from scripts import bench_combined_transfer as common
from scripts import bench_lossless_decode as previous
from scripts import bench_lossless_mixed_fast as stability
from scripts.frozen_transfer_worker import runtime_snapshot, load_author_bench, timing_samples

sha, write, digest_json, spawn = common.sha, common.write, common.digest_json, common.spawn
LENGTHS = (512, 8192)
ARMS = ('dense_uniform', 'rank256_uniform')
OWN = tuple('standalone_lut_' + arm for arm in ARMS)
METHODS = OWN
LINKS = (1, 5, 10, 25, 50, 75, 100, 200, 300, 400)
CUDA_TESTS = ('tests.test_standalone_full_lut.CudaTests', 'tests.test_standalone_lut_codec.CudaTests')
PREVIOUS = 'benchmark_results/lossless_decode_a4500_20261029_125616_867717'
PREVIOUS_HASHES = dict(results='932544c4fee02fea73537879e10cbfc59d7af2f5de36c7e1162b3d0df3e2a9f4',
    package='632a624ac26083ae7b8ddb33fb250f5c2ff4b9b01e485b15bdeba3817a4d6296')
FULL_LUT_BYTES = 10 * 65536 * 2
SCOPE = ('Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. '
    'The candidate changes only fixed-CDF inversion; complete original packets, fitted coefficients, '
    'prediction arithmetic and BF16 words stay fixed. SplitZip and ExANS are separate unchanged controls, '
    'not candidate components. Existing compact-search graph execution at 512 tokens and direct-search '
    'execution at 8192 tokens are retained. These two documents are reused engineering evidence, not '
    'independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.')


def method_arm(method):
    if method not in OWN:
        raise ValueError('unknown standalone lossless LUT method')
    return method.removeprefix('standalone_lut_')


def reference_method(method):
    return 'team_' + method_arm(method)


def fixed_selections():
    old = json.loads((ROOT / frozen.FROZEN / 'results.json').read_text())
    return frozen.validate_selections(old['recipe']['selections'])


def verify_previous():
    root = ROOT / PREVIOUS
    package = frozen.verify_package(root)
    if package['files'] != 214:
        raise ValueError('completed source comparison package count changed')
    for name, key in (('results.json', 'results'), ('package_manifest.json', 'package')):
        if sha(root / name) != PREVIOUS_HASHES[key]:
            raise ValueError('completed comparison receipt changed: ' + name)
    old = json.loads((root / 'results.json').read_text())
    if (old['status'] != 'completed' or not old['gpu_measured'] or len(old['rows']) != 68
        or len(old['jobs']) != 34 or not old['cuda_smoke']['passed'] or old['cuda_smoke']['expected_tests'] != 3
        or not old['frozen_source_after']['all_match'] or old['source_before'] != old['source_after']
        or old['measurement_source_before'] != old['measurement_source_after']
        or old['shared_artifacts_before'] != old['shared_artifacts_after']
        or old['native_runtime_before']['runtime_libraries'] != old['native_runtime_after']['runtime_libraries']
        or old['native_runtime_before']['library_sha256'] != old['native_runtime_after']['library_sha256']
        or sha(root / 'cuda_smoke.log') != old['cuda_smoke']['log_sha256']
        or any(not r['original_word_exact'] or r['decoded_word_sha256'] != r['original_word_sha256'] for r in old['rows'])):
        raise ValueError('previous actual-CUDA comparison is incomplete')
    if previous.source_receipts() != old['measurement_source_after']:
        raise ValueError('previous source files changed')
    for name in old['jobs']:
        worker = json.loads((root / name / 'worker_results.json').read_text())
        path = root / name / 'receiver.json'
        proof = json.loads(path.read_text())
        if (sha(path) != worker['receiver_sha256'] or len(proof['checks']) != 2
            or not all(proof.get(k) for k in ('gpu_measured', 'no_source_kv_or_predictions',
                'alternating_packets', 'independent_returned_outputs'))
            or not all(c['verified'] for c in proof['checks'])):
            raise ValueError('previous receiver proof incomplete: ' + name)
    refs = {f'team_{arm}_t{n}': [r for r in old['rows'] if r['method'] == 'team_' + arm and r['tokens'] == n]
        for arm in ARMS for n in LENGTHS}
    if any(len(rows) != 2 for rows in refs.values()):
        raise ValueError('standalone reference packet coverage missing')
    selections = fixed_selections()
    for key, rows in refs.items():
        entries = {e['document']: e for e in selections[key]['entries']}
        for row in rows:
            expected = entries[row['document']]
            if any(row[k] != expected[k] for k in ('packet_bytes', 'packet_sha256', 'original_word_sha256', 'shape', 'values')):
                raise ValueError('previous standalone row differs from original frozen packet')
    policy = dict(format='standalone_full_lut_v1', methods=list(OWN), lengths=list(LENGTHS),
        links_gbps=list(LINKS), full_lut_bytes=FULL_LUT_BYTES,
        selection_scope='Fixed Dense and Rank256 arms; no TEST policy or method selection',
        previous_results_sha256=PREVIOUS_HASHES['results'], reference_packets=refs)
    policy['policy_sha256'] = digest_json(policy)
    return policy, dict(path=str(root), package=package, hashes=PREVIOUS_HASHES)


def source_receipts():
    names = ('scripts/bench_standalone_lut.py', 'scripts/run_standalone_lut_a4500.sh',
        'docs/standalone_lut_v1.md', 'kv_compare/standalone_full_lut.py',
        'kv_compare/standalone_lut_codec.py', 'tests/test_standalone_full_lut.py',
        'tests/test_standalone_lut_codec.py', 'tests/test_standalone_lut.py')
    return dict(frozen_decode=previous.source_receipts(), new={name: sha(ROOT / name) for name in names})


def execution_for(tokens):
    if type(tokens) is not int or tokens not in LENGTHS:
        raise ValueError('frozen token length required')
    return 'compact_search' if tokens == 512 else 'direct_search'


def job_plan(selections, policy):
    jobs = []
    for n in LENGTHS:
        for arm in ARMS:
            method = 'standalone_lut_' + arm
            target = selections[f'team_{arm}_t{n}']
            spec = dict(target['codec'], kind='standalone_lut', full_lut=True)
            jobs.append((f'{method}_t{n}', dict(method=method, tokens=n, codec=spec,
                entries=target['entries'], raw_entries=selections[f'raw_bf16_t{n}']['entries'],
                reference_entries=policy['reference_packets'][f'team_{arm}_t{n}'],
                execution=execution_for(n), round='paired', policy_sha256=policy['policy_sha256'],
                timing_source=str(ROOT / 'runs/_llama_official/sources/splitzip/bench_codec_throughput.py'))))
    return jobs


def compact_plan(selections, policy):
    return stability.compact_plan(selections, policy)


def validate_receiver_request(q):
    if set(q) != {'method', 'tokens', 'codec', 'entries', 'execution', 'output'}:
        raise ValueError('receiver accepts only state, packet inputs and expected assertions')
    arm = method_arm(q['method'])
    if q['execution'] != execution_for(q['tokens']):
        raise ValueError('standalone execution mode changed')
    spec = q['codec']
    required = {'kind', 'state', 'state_sha256', 'manifest_sha256', 'arm', 'full_lut'}
    if (set(spec) != required or spec['kind'] != 'standalone_lut' or spec['full_lut'] is not True
        or spec['arm'] != arm):
        raise ValueError('standalone receiver codec allowlist differs')
    original = fixed_selections()[f'team_{arm}_t{q["tokens"]}']
    base = dict(spec, kind='team'); base.pop('full_lut')
    if base != original['codec']:
        raise ValueError('standalone receiver fixed state differs')
    allowed = {'id', 'document', 'shape', 'values', 'packet', 'packet_sha256', 'packet_bytes',
        'original_word_sha256', 'complete_bpv', 'state_sha256'}
    required_entry = {'id', 'document', 'shape', 'values', 'packet', 'packet_sha256', 'packet_bytes', 'original_word_sha256'}
    entries = q['entries']
    if not isinstance(entries, list) or len(entries) != 2:
        raise ValueError('both original documents required')
    expected = {e['document']: e for e in original['entries']}
    if len({e.get('document') for e in entries}) != 2:
        raise ValueError('receiver documents repeated')
    for entry in entries:
        if not required_entry.issubset(entry) or not set(entry).issubset(allowed) or entry['document'] not in expected:
            raise ValueError('source KV/predictions or unknown packet assertion forbidden')
        target = expected[entry['document']]
        if any(entry[k] != target[k] for k in ('id', 'shape', 'values', 'packet_bytes', 'packet_sha256', 'original_word_sha256')):
            raise ValueError('receiver packet changed original frozen identity')
        if 'state_sha256' in entry and entry['state_sha256'] != spec['state_sha256']:
            raise ValueError('receiver packet fixed state differs')


def codec_for(q, device, *, reference=False):
    arm = method_arm(q['method']); spec = q['codec']
    if sha(Path(spec['state']) / 'manifest.json') != spec['manifest_sha256']:
        raise ValueError('fixed state manifest changed')
    options = dict(decoder_mode='graph' if q['tokens'] == 512 else 'direct')
    if reference:
        from kv_compare.team_lookup import TeamLookupCodec
        codec = TeamLookupCodec(spec['state'], device, arm, lookup=False, **options)
    else:
        from kv_compare.standalone_lut_codec import StandaloneLUTCodec
        codec = StandaloneLUTCodec(spec['state'], device, arm, full_lut=True, **options)
    if codec.state_sha256 != spec['state_sha256']:
        raise ValueError('fixed predictor tensors changed')
    return codec


def load_packets(q, device):
    from scripts.bench_team_gpu import load_packets as original
    projected = dict(q, method=reference_method(q['method'])) if q['method'] in OWN else q
    return original(projected, device)


def validate_packet_equivalence(rows, complete=True):
    proofs = []
    for n in LENGTHS:
        for arm in ARMS:
            for doc in (f'test_text_t{n}', f'test_code_t{n}'):
                ref = [r for r in rows if (r['tokens'], r['method'], r['document']) == (n, 'team_' + arm, doc)]
                test = [r for r in rows if (r['tokens'], r['method'], r['document']) == (n, 'standalone_lut_' + arm, doc)]
                if len(ref) > 1 or len(test) > 1:
                    raise ValueError('duplicate standalone parity row')
                if not ref or not test:
                    if complete: raise ValueError('standalone paired coverage incomplete')
                    continue
                if any(ref[0][k] != test[0][k] for k in ('packet_bytes', 'packet_sha256', 'original_word_sha256', 'shape', 'values')):
                    raise ValueError('standalone optimization changed complete original packet')
                proofs.append(dict(tokens=n, arm=arm, document=doc, original_wire_equal=True))
    return proofs



def prepare(q, codec, packets, torch, device):
    if q['method'] not in OWN:
        return {}
    begin = time.perf_counter(); largest = max(packets, key=lambda p: p.payload.numel())
    codec.prepare(largest)
    if q['execution'] == 'compact_search':
        codec.reserve_payload_words(largest.payload.numel()); codec.graph_prepare(largest)
    torch.cuda.synchronize(device)
    return dict(wall_ms=1000 * (time.perf_counter() - begin), graph_capture_count=graph_count(q, codec))


def graph_count(q, codec):
    count = codec.graph_capture_count if q['method'] in OWN else 0
    if count != int(q['execution'] == 'compact_search'):
        raise AssertionError('unexpected graph capture count')
    return count


def packet_checks(packets, entries):
    checks = []
    for packet, e in zip(packets, entries):
        wire = packet.export(); actual = hashlib.sha256(wire).hexdigest()
        if actual != e['packet_sha256'] or len(wire) != e['packet_bytes']:
            raise AssertionError('compressed input packet changed')
        checks.append(dict(id=e['id'], sha256=actual, bytes=len(wire), unchanged=True))
    return checks


def close(codec):
    if hasattr(codec, 'close'):
        codec.close()


def _measure(q, torch, device):
    from scripts.bench_team_packet import check_original
    from scripts.bench_llama_causal import timed
    from kv_compare.framed_baselines import Baseline
    from kv_compare.causal_packet import Packet
    from kv_compare.team_packet import TeamPacket
    out = Path(q['output']); before = runtime_snapshot(torch)
    bench, timing_receipt = load_author_bench(q['timing_source'], torch)
    begin = time.perf_counter(); codec = codec_for(q, device)
    torch.cuda.synchronize(device); cold_ms = 1000 * (time.perf_counter() - begin)
    frozen = load_packets(q, device); setup = prepare(q, codec, frozen, torch, device)
    raw_q = dict(method='raw_bf16', entries=q['raw_entries'])
    raw_packets = load_packets(raw_q, device); raw_codec = Baseline('raw_bf16', device)
    raw_by_doc = {e['document']: (p, e) for e, p in zip(q['raw_entries'], raw_packets)}
    actual_packets = []; actual_entries = []; rows = []
    try:
        for i, entry in enumerate(q['entries']):
            raw_packet, raw_entry = raw_by_doc[entry['document']]
            if raw_entry['original_word_sha256'] != entry['original_word_sha256']:
                raise ValueError('raw control and target original words differ')
            source = raw_codec.decode(raw_packet); check_original(source, entry)
            # Full encode predicts from the original input; no prepared prediction arrays.
            encoded = codec.encode(source); restored = codec.decode(encoded)
            check_original(restored, entry); del restored
            wire = encoded.export(); actual_sha = hashlib.sha256(wire).hexdigest()
            same = actual_sha == entry['packet_sha256']
            if len(wire) != entry['packet_bytes'] or encoded.header != frozen[i].header:
                raise AssertionError('re-encode changed packet size/geometry/state')
            if q['method'] != 'exans_official_fixed_framed' and not same:
                raise AssertionError('deterministic frozen re-encode bytes differ')
            if q['method'] == 'exans_official_fixed_framed':
                if codec.accounting(encoded) != codec.accounting(frozen[i]):
                    raise AssertionError('ExANS fit/fallback accounting changed')
            actual_entry = dict(entry, packet=str(out / (entry['id'] + '_encoded.kvc')),
                                packet_sha256=actual_sha)
            Path(actual_entry['packet']).write_bytes(wire)
            actual = (TeamPacket.import_bytes(wire, device=device) if q['method'] in OWN else Packet.restore(wire, device))
            actual_packets.append(actual); actual_entries.append(actual_entry)
            check_original(codec.decode(actual), entry)
            del encoded, wire
            enc_batches = timing_samples([bench(lambda: codec.encode(source), str(device), 10, 50) for _ in range(5)], 2 * entry['values'])
            enc_packet, enc_isolated = timed(lambda: codec.encode(source), torch, device, 2, 5)
            check_original(codec.decode(enc_packet), entry); del enc_packet
            dec_batches = timing_samples([bench(lambda: codec.decode(actual), str(device), 10, 50) for _ in range(5)], 2 * entry['values'])
            value, dec_isolated = timed(lambda: codec.decode(actual), torch, device, 2, 5)
            check_original(value, entry); del value
            check_original(source, entry)
            accounting = (codec.accounting(actual) if hasattr(codec, 'accounting') else
                          dict(shared_predictor_bytes=getattr(codec, 'shared_bytes', None)))
            if accounting.get('full_inverse_shared_bytes') != FULL_LUT_BYTES:
                raise AssertionError('actual standalone inverse-table shared memory differs')
            rows.append(dict(actual_entry, method=q['method'], execution=q['execution'], round=q['round'],
                tokens=q['tokens'], original_exact=True, original_word_exact=True, status='verified',
                expected_sha=entry['original_word_sha256'], decoded_word_sha256=entry['original_word_sha256'],
                frozen_packet_sha256=entry['packet_sha256'], encode_packet_sha256=actual_sha,
                frozen_packet_bytes=entry['packet_bytes'],
                encode_packet_matches_frozen=same,
                encode=dict(batched=enc_batches, isolated=enc_isolated),
                decode=dict(batched=dec_batches, isolated=dec_isolated), accounting=accounting))
            print(entry['document'] + ': encoded/restored exact; batch and isolated timings complete', flush=True)
            del source; gc.collect()
        if q['tokens'] == 8192:
            profile_pair(q, codec, actual_packets, raw_by_doc, raw_codec, torch, device, out)
        count = graph_count(q, codec)
        immutable = packet_checks(actual_packets, actual_entries)
        frozen_immutable = packet_checks(frozen, q['entries'])
        raw_immutable = packet_checks(raw_packets, q['raw_entries'])
    finally:
        close(codec)
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('codec measurement changed runtime policy')
    return dict(rows=rows, entries=actual_entries, graph_capture_count=count, graph_setup=setup,
        packet_immutability=immutable, frozen_packet_immutability=frozen_immutable,
        raw_packet_immutability=raw_immutable, runtime_policy=before, runtime_policy_after=after,
        cold_state_initialization_wall_ms=cold_ms, author_timing=timing_receipt)


def diagnostics(codec, source, packet, out, torch, device):
    """Separate warmed full-API profiles and nonadditive stage diagnostics."""
    out.mkdir(parents=True, exist_ok=False)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities):
        p = codec.encode(source); value = codec.decode(p); torch.cuda.synchronize(device)
        del p, value
    cpu_profile = cProfile.Profile()
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        cpu_profile.enable()
        for _ in range(3):
            p = codec.encode(source); value = codec.decode(p); torch.cuda.synchronize(device)
            del p, value
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
            item = kernels.setdefault(event.name, dict(name=event.name, calls=0, gpu_us=0.0))
            item['calls'] += 1; item['gpu_us'] += float(getattr(event, 'device_time_total', 0))
    from scripts.bench_llama_causal import timed
    _, stage = timed(lambda: codec.stage(packet), torch, device, 1, 3)

    def native_call():
        output = torch.empty(codec.shape, dtype=torch.int16, device=device)
        codec.native.decode(packet.payload, codec.starts, codec.shape[1], out=output, bufs=codec.buffers)
        return output

    output, native = timed(native_call, torch, device, 1, 3)
    if not torch.equal(output, source):
        raise AssertionError('diagnostic native output changed original words')
    del output
    write(out / 'diagnostics.json', dict(calls=3, profiler_registration_warmup_calls=1,
        cpu_profile_started_after_profiler_entry=True,
        scope='Separate full encode/decode diagnostics; summed GPU work and stage samples are nonadditive',
        kernels=sorted(kernels.values(), key=lambda x: -x['gpu_us']), key_averages=keys, cprofile=cpu,
        accounting=codec.accounting(packet),
        allocation_ops=[x for x in keys if any(word in x['name'] for word in ('empty', 'resize', 'zeros', 'alloc'))],
        nonadditive_stages=dict(stage_checked_packet=stage, native_decode_with_output_allocation=native),
        full_trace_sha256=sha(out / 'full_trace.json')))


def profile_pair(q, candidate, packets, raw_by_doc, raw_codec, torch, device, out):
    """Both old and new kernels are profiled after ordinary timing finishes."""
    from scripts.bench_team_packet import check_original
    index = next(i for i, entry in enumerate(q['entries']) if entry['document'] == 'test_text_t8192')
    entry = q['entries'][index]
    raw_packet, _ = raw_by_doc[entry['document']]
    source = raw_codec.decode(raw_packet); check_original(source, entry)
    packet = packets[index]
    before = hashlib.sha256(packet.export()).hexdigest()
    reference = codec_for(q, device, reference=True)
    try:
        prepare(q, reference, packets, torch, device)
        diagnostics(reference, source, packet, out / 'profiles' / 'reference', torch, device)
    finally:
        close(reference)
    del reference
    gc.collect()
    diagnostics(candidate, source, packet, out / 'profiles' / 'full_lut', torch, device)
    if before != hashlib.sha256(packet.export()).hexdigest():
        raise AssertionError('profiling changed original compressed packet')
    check_original(source, entry)


def receive(q, torch, device):
    from scripts.bench_team_packet import check_original
    validate_receiver_request(q)
    before = runtime_snapshot(torch)
    codec = codec_for(q, device); packets = load_packets(q, device)
    setup = prepare(q, codec, packets, torch, device)
    held = []; checks = {}
    try:
        for index in (0, 1, 0):
            entry = q['entries'][index]
            value = codec.decode(packets[index]); digest = check_original(value, entry)
            for old, expected in held:
                check_original(old, expected)
            held.append((value, entry))
            checks[entry['id']] = dict(id=entry['id'], shape=list(value.shape), verified=True,
                word_sha256=digest, packet_sha256=entry['packet_sha256'])
        immutable = packet_checks(packets, q['entries']); count = graph_count(q, codec)
    finally:
        close(codec)
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('fresh receiver changed runtime flags')
    write(q['output'], dict(status='completed', gpu_measured=True, no_source_kv_or_predictions=True,
        alternating_packets=True, independent_returned_outputs=True, checks=list(checks.values()),
        packet_immutability=immutable, graph_capture_count=count, graph_setup=setup,
        runtime_policy=before, runtime_policy_after=after))


def worker(q, torch, device):
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    projected = {k: q[k] for k in ('method', 'tokens', 'codec', 'execution')}
    projected.update(entries=q['entries'], output=str(out / 'receiver.json'))
    validate_receiver_request(projected)
    data = _measure(q, torch, device)
    request = dict(projected, entries=data.pop('entries'))
    validate_receiver_request(request); write(out / 'receiver_request.json', request)
    torch.cuda.synchronize(device); gc.collect(); torch.cuda.empty_cache()
    spawn([sys.executable, __file__, '--device', str(device), '--receive', out / 'receiver_request.json'], out / 'receiver.log')
    proof = json.loads((out / 'receiver.json').read_text()); checks = {c['id']: c for c in proof['checks']}
    if (not all(proof.get(k) for k in ('gpu_measured', 'no_source_kv_or_predictions',
            'alternating_packets', 'independent_returned_outputs'))
        or proof['runtime_policy'] != data['runtime_policy'] or proof['runtime_policy_after'] != data['runtime_policy']
        or proof['graph_capture_count'] != data['graph_capture_count']):
        raise AssertionError('fresh receiver runtime/ownership proof differs')
    for row in data['rows']:
        item = checks[row['id']]
        if (not item['verified'] or item['word_sha256'] != row['original_word_sha256']
            or item['packet_sha256'] != row['packet_sha256']):
            raise AssertionError('fresh receiver original words differ')
    data.update(status='completed', gpu_measured=True, method=q['method'], tokens=q['tokens'], round=q['round'],
        codec_provenance=q['codec'], device=torch.cuda.get_device_name(device),
        software={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')},
        receiver_sha256=sha(out / 'receiver.json'), scope=SCOPE,
        profiles=[str(p.relative_to(out)) for p in sorted(out.rglob('diagnostics.json'))],
        full_lut_shared_bytes_expected=FULL_LUT_BYTES,
        memory_scope='Torch incremental API peaks include output/scratch; fixed LUT is additional shared state; native allocations outside Torch unmeasured')
    write(out / 'worker_results.json', data)


def validate_worker(data, q, job):
    validate_receiver_request(json.loads((job / 'receiver_request.json').read_text()))
    if data.get('full_lut_shared_bytes_expected') != FULL_LUT_BYTES:
        raise ValueError('full inverse table memory declaration differs')
    if any(row['accounting'].get('full_inverse_shared_bytes') != FULL_LUT_BYTES for row in data['rows']):
        raise ValueError('actual candidate inverse table memory differs')
    if q['tokens'] == 8192 and set(data.get('profiles', [])) != {
        'profiles/reference/diagnostics.json', 'profiles/full_lut/diagnostics.json'}:
        raise ValueError('reference/candidate standalone profiles missing')
    return frozen.validate_worker(data, q, job)


def link_models(aggregates):
    return previous.link_models(aggregates)


def paired_insights(aggregates):
    result = []
    for n in LENGTHS:
        for method in OWN:
            ref = [a for a in aggregates if a['method'] == reference_method(method) and a['tokens'] == n]
            test = [a for a in aggregates if a['method'] == method and a['tokens'] == n]
            if len(ref) != 1 or len(test) != 1:
                continue
            ref, test = ref[0], test[0]
            for basis in ('batched', 'isolated'):
                result.append(dict(tokens=n, arm=method_arm(method), timing_basis=basis,
                    complete_bytes_delta=test['packet_bytes'] - ref['packet_bytes'],
                    encode_delta_ms=test[basis]['encode_ms'] - ref[basis]['encode_ms'],
                    decode_delta_ms=test[basis]['decode_ms'] - ref[basis]['decode_ms'],
                    encode_speedup=ref[basis]['encode_ms'] / test[basis]['encode_ms'],
                    decode_speedup=ref[basis]['decode_ms'] / test[basis]['decode_ms']))
    return result


def report(result, out):
    lines = ['# Standalone lossless full-CDF inverse lookup', '',
        '**Status: ' + result['status'] + '.** GPU measured: **' + str(result['gpu_measured']) + '**.', '', SCOPE, '']
    if result.get('error'):
        lines += ['Stopped: `' + result['error'].replace('`', "'") + '`. Preserve outputs.', '']
    for basis in ('batched', 'isolated'):
        lines += ['Timing: **' + basis + '**.', '',
            '| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |',
            '|---:|---|---|---:|---:|---:|']
        for item in result.get('aggregates', []):
            lines.append(f"| {item['tokens']} | {item['method']} | {item['round']} | {item['complete_bpv']:.5f} | {item[basis]['encode_ms']:.3f} | {item[basis]['decode_ms']:.3f} |")
        lines.append('')
    if result.get('paired_insights'):
        lines += ['Fixed standalone candidate versus its unchanged standalone reference. Speedup below 1 means slower.', '',
            '| Tokens | Predictor | Basis | Wire delta bytes | Encode speedup | Decode speedup |',
            '|---:|---|---|---:|---:|---:|']
        for item in result['paired_insights']:
            lines.append(f"| {item['tokens']} | {item['arm']} | {item['timing_basis']} | {item['complete_bytes_delta']:.0f} | {item['encode_speedup']:.3f} | {item['decode_speedup']:.3f} |")
        lines.append('')
    modeled = result.get('link_models', {}).get('rows', [])
    if modeled:
        lines += ['The link tables use **8K isolated** API timing. Both lengths and both timing bases remain in results.json. '
            'Baseline cells use the faster complete before/after round independently for each objective. '
            'Dense and Rank remain fixed arms; no TEST-selected deployment policy is produced.', '']
        methods = ('raw_bf16', 'splitzip_official_framed', 'exans_official_fixed_framed', common.COMPACT,
            'team_dense_uniform', 'standalone_lut_dense_uniform', 'team_rank256_uniform', 'standalone_lut_rank256_uniform')
        for objective in ('precompressed', 'fresh'):
            lines += [f'Calculated serial **{objective}**, milliseconds.', '',
                '| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Dense LUT | Rank old | Rank LUT |',
                '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            for link in LINKS:
                cells = []
                for method in methods:
                    values = [r[objective + '_serial_ms'] for r in modeled if
                        (r['tokens'], r['effective_gbps'], r['timing_basis'], r['method']) == (8192, link, 'isolated', method)]
                    cells.append(f'{min(values):.3f}' if values else 'pending')
                lines.append(f'| {link} | ' + ' | '.join(cells) + ' |')
            lines.append('')
    lines += ['Every standalone candidate uses our predictive/rANS coder for all 28 layers. No SplitZip fallback, layer mask, '
        'AQUA adapter or lossy reconstruction is involved. The new inverse LUT is 1,310,720 bytes (1.25 MiB) of '
        'additional shared GPU state; packet bytes and shared predictor coefficients stay fixed.', '',
        'Whole encode/decode calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. '
        'Separate isolated calls use two warmups and five samples. Export, disk, upload, offline fitting and cold '
        'setup are outside warm measurements. Actual serialized encoder output is restored in a fresh process, '
        'alternating packets 0/1/0 with retained independently owned outputs.', '',
        'The 512-token path retains compact staging, graph replay and owned output clone; the 8192-token path '
        'retains borrowed compact input and fresh output allocation. Candidate dispatch is installed before graph capture. '
        'Original author controls, native libraries, environments and all frozen sources remain unchanged.', '',
        'Separate warmed profiles compare current standalone reference and candidate. Summed kernel work and stage '
        'timings are nonadditive; full traces stay local. Native allocator peaks outside Torch are unmeasured. '
        'Link scenarios are calculated, not measured NIC/SSD/serving or compressed-resident attention results.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def parent(args):
    out = frozen.confined(args.output); out.mkdir(parents=True, exist_ok=False)
    result = dict(format='standalone_lut_v1', status='failed', gpu_measured=False, scope=SCOPE, rows=[], jobs={})
    try:
        policy, old_receipt = verify_previous(); result['previous'] = old_receipt
        result['measurement_source_before'] = source_receipts()
        selections, receipt = frozen.verified_inputs(); result['verification'] = receipt
        result['source_before'] = frozen.source_checks()
        result['native_runtime_before'] = stability.native_runtime_receipts(selections, receipt)
        fixed_files = {p: item['sha256'] for p, item in receipt['fixed_artifacts'].items() if not p.endswith(('.kvc', '.kvp'))}
        result['shared_artifacts_before'] = {p: sha(p) for p in fixed_files}
        if result['shared_artifacts_before'] != fixed_files:
            raise ValueError('fixed shared artifacts differ')
        write(out / 'policy.json', policy); result['policy_file_sha256'] = sha(out / 'policy.json')
        result['fixed_experiment'] = policy
        py, env = frozen.worker_environment(selections['exans_official_fixed_framed_t512'], receipt)
        smoke = ("import sys,unittest,torch; assert torch.cuda.is_available(),'actual CUDA required'; "
            "assert 'RTX A4500' in torch.cuda.get_device_name(0) and torch.version.cuda=='12.8'; "
            "s=unittest.defaultTestLoader.loadTestsFromNames(" + repr(list(CUDA_TESTS)) + "); "
            "r=unittest.TextTestRunner(verbosity=2).run(s); sys.exit(0 if r.wasSuccessful() and not r.skipped and r.testsRun==2 else 1)")
        spawn([py, '-c', smoke], out / 'cuda_smoke.log', env)
        result['cuda_smoke'] = dict(tests=list(CUDA_TESTS), expected_tests=2, passed=True,
            log_sha256=sha(out / 'cuda_smoke.log'))
        originals = frozen.job_plan(selections)
        before = [j for j in originals if j[1]['round'] == 'before']
        after = [j for j in originals if j[1]['round'] == 'after']
        paired = [j for j in originals if j[1]['round'] == 'paired']
        for name, q in before + paired + job_plan(selections, policy) + compact_plan(selections, policy) + after:
            job = out / name; job.mkdir(); q['output'] = str(job); write(job / 'request.json', q)
            new = q['method'] in OWN; compact = q['method'] == common.COMPACT
            envq = dict(q, method='raw_bf16') if new else dict(q, method='exans_official_fixed_framed') if compact else q
            py, env = frozen.worker_environment(envq, receipt)
            script = __file__ if new else common.__file__ if compact else ROOT / 'scripts/frozen_transfer_worker.py'
            argv = [str(py), str(script), '--worker' if new or compact else '--request', str(job / 'request.json'), '--device', 'cuda:0']
            start = time.monotonic(); print('Running ' + name, flush=True)
            spawn(argv, job / 'worker.log', env)
            data = json.loads((job / 'worker_results.json').read_text())
            validator = validate_worker if new else common.validate_new_worker if compact else frozen.validate_worker
            result['rows'].extend(validator(data, q, job)); result['gpu_measured'] = True
            result['jobs'][name] = dict(argv=argv, returncode=0, command_wall_seconds=time.monotonic() - start,
                artifacts={str(p.relative_to(job)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in job.rglob('*')
                    if p.is_file() and p.suffix in ('.json', '.log') and p.name != 'full_trace.json'})
            result['packet_equivalence'] = validate_packet_equivalence(result['rows'], complete=False)
            result['aggregates'] = frozen.aggregate(result['rows'])
            result['paired_insights'] = paired_insights(result['aggregates'])
            write(out / 'results.json', result); report(result, out)
        if len(result['jobs']) != 22 or len(result['rows']) != 44:
            raise ValueError('complete standalone comparison requires 22 workers and 44 rows')
        result['packet_equivalence'] = validate_packet_equivalence(result['rows'])
        result['source_after'] = frozen.source_checks(); result['measurement_source_after'] = source_receipts()
        result['shared_artifacts_after'] = {p: sha(p) for p in fixed_files}
        result['native_runtime_after'] = stability.native_runtime_receipts(selections, receipt)
        from scripts.check_nuc_baseline_readiness import frozen_sources
        result['frozen_source_after'] = frozen_sources(ROOT)
        if (result['source_before'] != result['source_after'] or not result['frozen_source_after']['all_match']
            or result['measurement_source_before'] != result['measurement_source_after']
            or result['shared_artifacts_before'] != result['shared_artifacts_after']
            or result['native_runtime_before']['runtime_libraries'] != result['native_runtime_after']['runtime_libraries']
            or result['native_runtime_before']['library_sha256'] != result['native_runtime_after']['library_sha256']
            or sha(out / 'policy.json') != result['policy_file_sha256']):
            raise ValueError('frozen source/state/runtime/experiment receipts changed')
        result['link_models'] = link_models(result['aggregates']); result['status'] = 'completed'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        (out / 'failure.log').write_text(traceback.format_exc())
        print(result['error'], file=sys.stderr, flush=True)
    finally:
        write(out / 'results.json', result); report(result, out)
        print('Report: ' + str(out / 'summary.md'), flush=True)
    return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--device', default='cuda:0')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--output', type=Path); modes.add_argument('--worker', type=Path); modes.add_argument('--receive', type=Path)
    args = parser.parse_args()
    if args.output:
        return parent(args)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '16'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
    from scripts.bench_team_packet import cuda_check
    torch, device = cuda_check(args.device)
    if 'RTX A4500' not in torch.cuda.get_device_name(device) or torch.version.cuda != '12.8':
        raise RuntimeError('fixed setting requires actual RTX A4500 / CUDA12.8')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'triton', 'numpy')}
    if versions != dict(torch='2.11.0+cu128', triton='3.6.0', numpy='2.3.5'):
        raise ValueError('frozen candidate environment changed: ' + str(versions))
    q = json.loads((args.worker or args.receive).read_text())
    (worker if args.worker else receive)(q, torch, device)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
