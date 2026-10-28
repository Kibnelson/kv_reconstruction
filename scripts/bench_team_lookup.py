#!/usr/bin/env python3
"""GPU-resident exact decode: lookup and borrowed-input execution ablations.

Frozen packets and fixed calibration are reused. No encode, raw capture, model
download, host export/hash/upload, storage or actual network is timed.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
import argparse
import cProfile
import gc
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import pstats
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_team_packet import canonical, identity, cuda_check, check_original
from scripts.bench_lite_baselines import digest, launch, write, make_codec
from scripts.bench_llama_causal import log, timed
from scripts.bench_lite_allocation import runtime_snapshot, pcie_metadata

BASELINES = ('raw_bf16', 'splitzip_official_framed', 'exans_official_fixed_framed')
OWN = ('team_dense_uniform', 'team_rank256_uniform')
METHODS = BASELINES + OWN
ARMS = ('compact_search', 'compact_lookup', 'direct_search', 'direct_lookup')
LENGTHS = (512, 8192)
LINKS = (1., 100., 200., 300., 400.)
FILES = ('scripts/bench_team_lookup.py', 'scripts/package_team_lookup.py',
         'scripts/run_team_lookup_a4500.sh', 'tests/test_team_lookup_runner.py',
         'kv_compare/team_lookup.py', 'tests/test_team_lookup.py')



def arm_order(index):
    # Eight balanced orders: each arm occupies every position twice, with
    # reversals reducing directional carryover. Fixed before TEST timings.
    rotation = index % len(ARMS)
    order = ARMS[rotation:] + ARMS[:rotation]
    return order if (index // len(ARMS)) % 2 == 0 else order[::-1]


def sample_order(entries, arms, warmup, repeats):
    result = []
    for rep in range(warmup + repeats):
        docs = (0, 1) if rep % 2 == 0 else (1, 0)
        index = rep if rep < warmup else rep - warmup
        order = arm_order(index) if tuple(arms) == ARMS else tuple(arms)
        result.append(dict(documents=[entries[i]['document'] for i in docs], arms=list(order)))
    return result


def finish_samples(record):
    result = dict(record)
    result['wall_median_ms'] = statistics.median(s['wall_ms'] for s in result['samples'])
    return result


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['tokens'], row['method'], row['arm'], row['round']), []).append(row)
    result = []
    for (tokens, method, arm, phase), group in sorted(groups.items()):
        result.append(dict(tokens=tokens, method=method, arm=arm, round=phase,
            documents=[r['document'] for r in group], packet_bytes=statistics.mean(r['packet_bytes'] for r in group),
            complete_bpv=statistics.mean(r['complete_bpv'] for r in group),
            saved_percent=statistics.mean(100 * (1-r['packet_bytes']/(2*r['values'])) for r in group),
            decode_ms=statistics.mean(r['decode']['wall_median_ms'] for r in group),
            decode_cuda_ms=statistics.mean(statistics.median(s['cuda_interval_ms'] for s in r['decode']['samples']) for r in group),
            kernel_decode_ms=(statistics.mean(r['kernel_decode']['wall_median_ms'] for r in group)
                              if all('kernel_decode' in r for r in group) else None),
            original_exact=all(r['original_exact'] for r in group)))
    return result


def link_models(aggregates, links=LINKS):
    if any(not math.isfinite(g) or g <= 0 for g in links):
        raise ValueError('positive finite effective links required')
    result = []
    targets = ('splitzip_official_framed', 'exans_official_fixed_framed')
    for a in aggregates:
        controls = {m: min((b for b in aggregates if b['tokens'] == a['tokens'] and b['method'] == m),
                           key=lambda b: b['decode_ms'], default=None) for m in targets}
        for g in links:
            wire = 8*a['packet_bytes']/(g*1e6)
            item = dict(tokens=a['tokens'], method=a['method'], arm=a['arm'], round=a['round'],
                effective_gbps=g, wire_ms=wire, precompressed_serial_ms=wire+a['decode_ms'],
                ideal_overlap_ms=max(wire, a['decode_ms']), measured_network=False,
                baseline_targets={})
            for method, control in controls.items():
                if control is None:
                    continue
                budget = control['decode_ms']+8*(control['packet_bytes']-a['packet_bytes'])/(g*1e6)
                total = control['decode_ms']+8*control['packet_bytes']/(g*1e6)
                item['baseline_targets'][method] = dict(round=control['round'],
                    decode_ms=control['decode_ms'], serial_ms=total, decode_budget_ms=budget,
                    speedup=total/item['precompressed_serial_ms'],
                    required_reduction_factor=a['decode_ms']/budget if budget > 0 else None)
            result.append(item)
    return dict(measured_network=False, links=list(links), rows=result,
        scope='calculated GPU-resident precompressed wire+full-decode; faster before/after native control is an explicit conservative target; ideal max(wire,decode) is unmeasured overlap; no encode, host export/hash/upload, NIC, SSD or serving')


def read_parent(parent_dir):
    """Bind the completed compact-GPU run and its original packets/state.

    The frozen verifier checks all 78 earlier source files, native sources and
    environments, packet hashes, original-word proofs and timing receipts. It
    never reads source KV captures or raw input arrays.
    """
    from scripts.package_team_gpu import verify as verify_gpu
    parent_dir = Path(parent_dir).resolve()
    parent = json.loads((parent_dir/'results.json').read_text())
    if parent.get('format') != 'team_gpu_v1' or parent['status'] != 'completed' or not parent.get('gpu_measured'):
        raise ValueError('completed actual-GPU compact comparison required')
    verify_gpu(parent_dir, parent)
    recipe = parent['recipe']
    inherited = {**recipe['unchanged_code'], **recipe['code']}
    if len(inherited) != 78:
        raise ValueError('expected all 78 frozen source identities')
    selections = json.loads(json.dumps(recipe['selections']))
    for q in selections.values():
        q['arms'] = list(ARMS if q['method'] in OWN else ('native',))
    receipts = {path: wanted for job in parent['jobs'].values() for path, wanted in job['artifacts'].items()}
    receipts.update({name:digest(parent_dir/name) for name in ('cuda_tests.log', 'source_manifest.json')})
    # Small profiler summaries/request hashes are bound too. Full traces remain
    # validated by the frozen parent verifier and stay on the GPU machine.
    for name, profile in parent.get('profiles', {}).items():
        receipts[f'profiles/{name}/profile.json'] = profile['summary_sha256']
        receipts[f'profiles/{name}/request.json'] = profile['request_sha256']
    return parent, inherited, selections, receipts


from scripts.bench_team_gpu import worker_environment, load_packets, profile_kernels


def selected_plan(selections):
    natives = [(name, q) for name, q in selections.items() if q['method'] in BASELINES]
    own = [(name, q) for name, q in selections.items() if q['method'] in OWN]
    return ([(f'before_{name}', dict(q, round='before')) for name, q in natives] +
            [(name, dict(q, round='paired')) for name, q in own] +
            [(f'after_{name}', dict(q, round='after')) for name, q in reversed(natives)])




def build_codecs(q, device):
    if q['method'] not in OWN:
        return {'native': make_codec(q['codec'], device)}
    from kv_compare.team_lookup import TeamLookupCodec
    if digest(Path(q['codec']['state'])/'manifest.json') != q['codec']['manifest_sha256']:
        raise ValueError('fixed team manifest changed')
    return {arm: TeamLookupCodec(q['codec']['state'], device, q['codec']['arm'],
                decoder_mode='graph' if arm.startswith('compact_') else 'direct',
                lookup=arm.endswith('_lookup')) for arm in q['arms']}


def prepare_codecs(q, codecs, packets, torch, device):
    if q['method'] not in OWN:
        return {}
    setup = {}; largest = max(packets, key=lambda p:p.payload.numel())
    for arm, codec in codecs.items():
        begin = time.perf_counter(); codec.prepare(largest)
        if arm.startswith('compact_'):
            codec.reserve_payload_words(largest.payload.numel())
            codec.graph_prepare(largest)
        torch.cuda.synchronize(device)
        setup[arm] = dict(wall_ms=(time.perf_counter()-begin)*1000,
                          graph_capture_count=codec.graph_capture_count)
    return setup


def decode_call(q, codec, packet):
    return codec.decode(packet)


def expected_graphs(arms):
    return {arm: int(arm.startswith('compact_')) for arm in arms}


def graph_counts(q, codecs):
    counts = {arm:codec.graph_capture_count for arm, codec in codecs.items()} if q['method'] in OWN else {}
    wanted = expected_graphs(q['arms']) if q['method'] in OWN else {}
    if counts != wanted:
        raise AssertionError('decoder graph captures differ from declared execution mode')
    return counts


def check_packets(packets, entries):
    """Validate input immutability outside every ordinary/profiler interval."""
    import hashlib
    checks = []
    for packet, entry in zip(packets, entries):
        blob = packet.export()
        actual = hashlib.sha256(blob).hexdigest()
        if len(blob) != entry['packet_bytes'] or actual != entry['packet_sha256']:
            raise AssertionError('decoder mutated a frozen compressed packet')
        checks.append(dict(id=entry['id'], packet_sha256=actual, bytes=len(blob), unchanged=True))
    return checks


def close_codecs(codecs):
    for codec in codecs.values():
        if hasattr(codec, 'close'):
            codec.close()


def valid_receiver_request(q):
    if set(q) != {'method', 'codec', 'entries', 'arms', 'output', 'round'}:
        raise ValueError('receiver must contain only codec and frozen packets/assertions')
    if any({'path', 'inputs', 'captures', 'source_run', 'predictions', 'decoded'} & set(e) for e in q['entries']):
        raise ValueError('source data forbidden in independent receiver')


def row_id(q, arm, entry):
    return q['round']+'_'+arm+'_'+entry['id']


def receive(q, torch, device):
    valid_receiver_request(q)
    before = runtime_snapshot(torch); codecs = build_codecs(q, device)
    packets = load_packets(q, device); prepare_codecs(q, codecs, packets, torch, device)
    checks = {}; held = {}
    try:
        for i in (0, 1, 0):
            for arm, codec in codecs.items():
                value = decode_call(q, codec, packets[i]); actual = check_original(value, q['entries'][i])
                e = q['entries'][i]; ident = row_id(q, arm, e)
                checks[ident] = dict(id=ident, arm=arm, parent_id=e['id'], verified=True,
                    packet_sha256=e['packet_sha256'], word_sha256=actual, shape=list(value.shape))
                if i == 0 and arm not in held:
                    held[arm] = value
                elif i == 1:
                    check_original(held[arm], q['entries'][0])
        counts = graph_counts(q, codecs)
        packet_checks = check_packets(packets, q['entries'])
        inverse_proofs = {arm:codec.inverse_proof for arm,codec in codecs.items()} if q['method'] in OWN else {}
    finally:
        close_codecs(codecs)
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('receiver runtime policy changed')
    write(q['output'], dict(gpu_measured=True, no_source_kv_or_predictions=True,
        alternating_packets=True, independent_returned_outputs=True, graph_capture_counts=counts,
        packet_immutability=packet_checks, inverse_proofs=inverse_proofs,
        runtime_policy=before, runtime_policy_after=after, checks=list(checks.values()),
        inputs='fixed codec state and unchanged serialized packets only; expected original digests are assertions'))


def worker(q, torch, device):
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    before = runtime_snapshot(torch); begin = time.perf_counter(); codecs = build_codecs(q, device)
    torch.cuda.synchronize(device); cold = (time.perf_counter()-begin)*1000
    packets = load_packets(q, device); setup = prepare_codecs(q, codecs, packets, torch, device)
    measurements = {(arm, i):dict(samples=[], warmup_samples=[]) for arm in q['arms'] for i in range(2)}
    order = sample_order(q['entries'], q['arms'], q['warmup'], q['repeats']); held = {}
    try:
        for rep, schedule in enumerate(order):
            by_doc = {e['document']:i for i, e in enumerate(q['entries'])}
            for name in schedule['documents']:
                i = by_doc[name]
                for arm in schedule['arms']:
                    value, sample = timed(lambda arm=arm, i=i:decode_call(q, codecs[arm], packets[i]),
                                           torch, device, 0, 1)
                    check_original(value, q['entries'][i])
                    record = measurements[arm, i]
                    record['warmup_samples' if rep < q['warmup'] else 'samples'].append(sample['samples'][0])
                    if rep == 0 and i == 0:
                        held[arm] = value
                    elif rep == 0 and i == 1:
                        check_original(held[arm], q['entries'][0])
                        del held[arm]
                    del value
        rows = []
        for arm, codec in codecs.items():
            for i, e in enumerate(q['entries']):
                row = dict(e, id=row_id(q, arm, e), parent_id=e['id'], method=q['method'], arm=arm,
                    round=q['round'], tokens=q['tokens'], status='verified', original_exact=True,
                    decoded_word_sha256=e['original_word_sha256'], decode=finish_samples(measurements[arm, i]))
                if q['method'] in OWN:
                    value = decode_call(q, codec, packets[i]); check_original(value, e); del value
                    if arm.startswith('compact_'):
                        _, row['kernel_decode'] = timed(codec.graph_replay_native, torch, device, 1, q['repeats'])
                        check_original(codec.output, e)
                    row['accounting'] = codec.accounting(packets[i])
                elif q['method'] == 'exans_official_fixed_framed':
                    row['accounting'] = codec.accounting(packets[i])
                else:
                    row['accounting'] = dict(shared_predictor_bytes=getattr(codec, 'shared_bytes', None))
                rows.append(row)
                log(f"{row['id']}: {row['complete_bpv']:.5f}bpv, GPU-resident D={row['decode']['wall_median_ms']:.3f}ms")
        counts = graph_counts(q, codecs)
        packet_checks = check_packets(packets, q['entries'])
        inverse_proofs = {arm:codec.inverse_proof for arm,codec in codecs.items()} if q['method'] in OWN else {}
    finally:
        close_codecs(codecs)
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('worker runtime policy changed')
    req = {k:q[k] for k in ('method', 'codec', 'entries', 'arms', 'round')}
    req['output'] = str(out/'receiver.json'); write(out/'receiver_request.json', req)
    codecs.clear(); del packets, codec; gc.collect(); torch.cuda.empty_cache()
    launch([sys.executable, Path(__file__).resolve(), '--device', str(device), '--receive', out/'receiver_request.json'], out/'receiver.log')
    proof = json.loads((out/'receiver.json').read_text()); checks = {c['id']:c for c in proof['checks']}
    if (not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
            not proof.get('alternating_packets') or not proof.get('independent_returned_outputs') or
            proof['runtime_policy'] != before or proof['runtime_policy_after'] != before or
            proof['graph_capture_counts'] != counts or set(checks) != {r['id'] for r in rows}):
        raise ValueError('fresh receiver runtime/ownership/coverage differs')
    for row in rows:
        c = checks[row['id']]
        if not c['verified'] or c['word_sha256'] != row['original_word_sha256'] or c['packet_sha256'] != row['packet_sha256']:
            raise AssertionError('fresh receiver exact words differ')
    write(out/'worker_results.json', dict(status='completed', gpu_measured=True, rows=rows,
        method=q['method'], round=q['round'], codec_provenance=q['codec'], device=torch.cuda.get_device_name(device),
        software={n:importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')},
        cold_state_initialization_wall_ms=cold, graph_setup=setup, graph_capture_counts=counts,
        sample_order=order, runtime_policy=before, runtime_policy_after=after,
        packet_immutability=packet_checks, inverse_proofs=inverse_proofs,
        receiver_sha256=digest(out/'receiver.json'),
        scope='GPU-resident compressed packet through unchanged/full public decode to independently owned original BF16 output; tiny metadata validation/transfers remain part of each method API',
        memory_scope='four-arm process contains all own codecs; per-codec buffers are separate; global Torch peaks are not standalone serving capacity'))




def profile_worker(q, torch, device):
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    before = runtime_snapshot(torch); codecs = build_codecs(q, device)
    packets = load_packets(q, device); prepare_codecs(q, codecs, packets, torch, device)
    arm = q['arms'][0]; codec = codecs[arm]; entry = q['entries'][0]; packet = packets[0]
    for _ in range(3):
        value = decode_call(q, codec, packet)
    check_original(value, entry); del value; torch.cuda.synchronize(device)
    from torch.profiler import ProfilerActivity, profile, record_function
    cpu = cProfile.Profile(); retained = []; trace = out/'trace.json'; calls = 3
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                 profile_memory=True, with_stack=False) as prof:
        cpu.enable()
        for _ in range(calls):
            with record_function('team_lookup/'+q['method']+'/'+arm+'/full_decode'):
                retained.append(decode_call(q, codec, packet))
            torch.cuda.synchronize(device)
        cpu.disable()
    # All digest/D2H assertions are excluded from cProfile and Torch profiling.
    for value in retained:
        check_original(value, entry)
    del retained, value
    prof.export_chrome_trace(str(trace))
    diagnostics = {}
    value, diagnostics['full_decode'] = timed(lambda:decode_call(q, codec, packet), torch, device, 0, 1)
    check_original(value, entry); del value
    if q['method'] in OWN:
        _, diagnostics['stage'] = timed(lambda:codec.stage(packet), torch, device, 0, 1)
        if arm.startswith('compact_'):
            value, diagnostics['native_eager'] = timed(codec.native_decode, torch, device, 0, 1)
            check_original(value, entry)
            value, diagnostics['output_clone'] = timed(lambda:codec.output.clone(), torch, device, 0, 1)
            check_original(value, entry); del value
    packet_checks = check_packets(packets, q['entries'])
    inverse_proof = codec.inverse_proof if q['method'] in OWN else None
    functions = []
    for (filename, line, name), (primitive, total, self_seconds, cumulative, _callers) in pstats.Stats(cpu).stats.items():
        functions.append(dict(file=filename, line=line, function=name, primitive_calls=primitive,
            total_calls=total, self_ms=1000*self_seconds, cumulative_ms=1000*cumulative))
    functions.sort(key=lambda r:r['self_ms'], reverse=True)
    operators = []
    for event in prof.key_averages():
        operators.append(dict(name=event.key, calls=event.count, self_cpu_ms=event.self_cpu_time_total/1000,
            cpu_total_ms=event.cpu_time_total/1000,
            self_device_ms=getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0))/1000,
            device_total_ms=getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0))/1000,
            cpu_memory_bytes=event.cpu_memory_usage, device_memory_bytes=getattr(event, 'device_memory_usage', 0)))
    operators.sort(key=lambda r:r['self_cpu_ms']+r['self_device_ms'], reverse=True)
    counts = graph_counts(q, codecs); close_codecs(codecs); after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('profile runtime policy changed')
    write(out/'profile.json', dict(status='completed', gpu_measured=True, method=q['method'],
        arm=arm, tokens=q['tokens'], document=entry['document'], packet_sha256=entry['packet_sha256'],
        original_word_sha256=entry['original_word_sha256'], codec_provenance=q['codec'], profile_calls=calls,
        diagnostics=diagnostics, cprofile_top_self=functions[:40], operators=operators[:60],
        device_kernels=profile_kernels(trace)[:60], trace_path=str(trace), trace_sha256=digest(trace),
        trace_bytes=trace.stat().st_size, assertions_outside_profilers=True, packet_immutability=packet_checks, inverse_proof=inverse_proof,
        runtime_policy=before, runtime_policy_after=after, graph_capture_counts=counts,
        kernel_scope='Chrome CUDA kernel events for three full public decode calls; durations sum device work and may overlap across streams, so they are not wall percentages or an additive critical path; empty list means no kernel attribution captured',
        diagnostic_scope='stage/native eager/clone measured separately after profilers; not additive decompositions of ordinary full graph decode',
        memory_scope='profile retains three owned outputs for assertions outside profiler; diagnostic memory fields are not ordinary inference/serving peaks',
        scope='separate cProfile/Torch diagnostic, excluded from primary unprofiled paired medians; full trace stays local'))


def report(r, out):
    lines = ['# GPU lookup and borrowed-input exact KV decode', '',
        f"Status: **{r['status']}**. GPU measured: **{r.get('gpu_measured', False)}**.", '',
        'Same frozen Llama-3.2-3B BF16 words, all 28 layers, text/code TEST documents, 512 and 8192 tokens. Packets, predictor coefficients, CDFs and official native controls are unchanged.',
        'Primary timing starts with compressed payload on GPU and covers validation, normal API metadata work, decode and an independently owned output. Host bulk hashing/import/upload, encoding, disk and real networking are excluded.',
        'compact_search is the current staged CUDA-graph decoder; compact_lookup selects the existing exact inverse-CDF lookup. direct_search/direct_lookup borrow the read-only GPU payload and allocate their own output directly. Four-arm rotation/reversal is predeclared; controls run before and after.', '',
        '| Tokens | Method | Arm / round | Complete bpv | Saved | Full decode ms | CUDA interval ms | Exact |',
        '|---:|---|---|---:|---:|---:|---:|---|']
    for a in r.get('aggregates', []):
        lines.append(f"| {a['tokens']} | {a['method']} | {a['arm']} / {a['round']} | {a['complete_bpv']:.5f} | {a['saved_percent']:.2f}% | {a['decode_ms']:.3f} | {a['decode_cuda_ms']:.3f} | {a['original_exact']} |")
    lines += ['', 'Calculated serial wire + full GPU decode, milliseconds. All methods/arms are shown; native before/after rounds remain separate. Ideal overlap max(wire,decode) is a separate unmeasured bound in results.json.', '',
        '| Tokens | Method | Arm / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for a in r.get('aggregates', []):
        costs = [8*a['packet_bytes']/(g*1e6)+a['decode_ms'] for g in LINKS]
        lines.append(f"| {a['tokens']} | {a['method']} | {a['arm']} / {a['round']} | "+' | '.join(f'{v:.3f}' for v in costs)+' |')
    lines += ['', 'Per-link budgets and speed ratios versus the faster SplitZip and ExANS control rounds are in results.json. No routing or method selection on TEST is deployed by this sweep.']
    if r.get('profiles'):
        lines += ['', 'Separate 8K text profiles record CPU/Torch operators and actual CUDA kernels. Summed kernel work may overlap across streams; it is not wall latency. Stage/eager/clone diagnostics are separate calls and must not be added as a graph decomposition.', '',
            '| Method | Arm | Full diagnostic ms | Stage ms | Native eager ms | Output clone ms |',
            '|---|---|---:|---:|---:|---:|']
        for _name, item in sorted(r['profiles'].items()):
            cells = ['—' if key not in item['diagnostics'] else f"{item['diagnostics'][key]['wall_median_ms']:.3f}"
                     for key in ('full_decode', 'stage', 'native_eager', 'output_clone')]
            lines.append(f"| {item['method']} | {item['arm']} | "+' | '.join(cells)+' |')
    if r.get('error'):
        lines += ['', 'Failure: '+r['error']]
    lines += ['', 'All arms require original-word equality, unchanged wire hashes, held-output ownership and mode-specific graph counts. Fixed shared state and scratch are additional to packet bytes. ExANS native allocation peaks are unmeasured.',
        'AQUA-KV is lossy and remains a separate official evaluation. No compressed attention, T-to-P conversion, fresh encode, SSD/network or serving performance is measured here.']
    (Path(out)/'summary.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', type=Path, default=ROOT/'runs/team_gpu_a4500_20261028_131628_4041134')
    p.add_argument('--output', type=Path); p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup', type=int, default=2); p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--no-profile', action='store_false', dest='profile', default=True)
    for name in ('worker', 'receive', 'profile-worker'):
        p.add_argument('--'+name, type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv); request = a.worker or a.receive or a.profile_worker
    if request:
        torch, device = cuda_check(a.device); q = json.loads(request.read_text())
        (worker if a.worker else receive if a.receive else profile_worker)(q, torch, device); return 0
    if a.output is None:
        p.error('--output required')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    result = dict(format='team_lookup_v1', status='running', gpu_measured=False, jobs={}, profiles={}, rows=[])
    writable = not (out/'results.json').exists()
    def save():
        result['rows'] = [row for job in result['jobs'].values() if job.get('status') == 'completed' for row in job['rows']]
        result['aggregates'] = aggregate(result['rows']); result['link_models'] = link_models(result['aggregates'])
        write(out/'results.json', result); report(result, out)
    try:
        torch, device = cuda_check(a.device)  # Before parent/model/state reads.
        if a.warmup < 1 or a.repeats < 8 or a.repeats % 8:
            raise ValueError('warmup>=1 and repeats a positive multiple of8 required')
        parent_dir = a.parent_run.resolve(); parent, inherited, selections, receipts = read_parent(parent_dir)
        versions = {n:importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy', 'transformers')}
        if (versions != parent['recipe']['software'] or torch.version.cuda != parent['recipe']['cuda_runtime'] or
                torch.cuda.get_device_name(device) != parent['hardware']['name']):
            raise ValueError('original main GPU/software required')
        plan = selected_plan(selections)
        recipe = dict(parent_run=str(parent_dir), parent_results_sha256=digest(parent_dir/'results.json'),
            parent_artifacts=receipts, unchanged_code=inherited, code={name:digest(ROOT/name) for name in FILES},
            methods=list(METHODS), lengths=list(LENGTHS), arms=list(ARMS), warmup=a.warmup, repeats=a.repeats,
            profile=a.profile, software=versions, device=torch.cuda.get_device_name(device),
            cuda_runtime=torch.version.cuda, native_receipt=parent['recipe']['native_receipt'],
            selections=selections, plan=[name for name, _base in plan])
        wanted = identity(recipe)
        if not writable:
            old = json.loads((out/'results.json').read_text())
            if old.get('identity') != wanted:
                raise ValueError('resume recipe differs; preserve output and choose a new directory')
            result = old; writable = True
        result.update(identity=wanted, recipe=recipe, status='running', gpu_measured=True,
            hardware=dict(parent['hardware'], pcie_query=pcie_metadata()), selected_jobs=recipe['plan'])
        result.pop('error', None); result.pop('traceback', None)
        write(out/'source_manifest.json', recipe); save()
        for name, base in plan:
            directory = out/name; directory.mkdir(exist_ok=True)
            q = dict(base, output=str(directory), warmup=a.warmup, repeats=a.repeats)
            old = result['jobs'].get(name, {})
            if old.get('status') == 'completed':
                for path, wanted_sha in old['artifacts'].items():
                    if digest(out/path) != wanted_sha:
                        raise ValueError('completed worker artifact changed')
                if json.loads((directory/'request.json').read_text()) != q:
                    raise ValueError('completed worker request differs')
                continue
            write(directory/'request.json', q)
            python, env = worker_environment(base, recipe['native_receipt'])
            launch([python, Path(__file__).resolve(), '--device', str(device), '--worker', directory/'request.json'], directory/'worker.log', env)
            data = json.loads((directory/'worker_results.json').read_text())
            if not data.get('gpu_measured') or data['status'] != 'completed' or data['codec_provenance'] != q['codec']:
                raise ValueError('actual-GPU worker required')
            result['jobs'][name] = dict(status='completed', rows=data['rows'], artifacts={
                str((directory/f).relative_to(out)):digest(directory/f)
                for f in ('request.json', 'worker_results.json', 'receiver.json', 'receiver_request.json')})
            save()
        # Profiles run after all ordinary measurements, in separate processes.
        if a.profile:
            for base in selections.values():
                if base['tokens'] != 8192 or base['method'] == 'raw_bf16':
                    continue
                for arm in base['arms']:
                    name = f"{base['method']}_{arm}_t8192"; directory = out/'profiles'/name
                    directory.mkdir(parents=True, exist_ok=True)
                    q = dict(base, round='profile', arms=[arm],
                        entries=[next(e for e in base['entries'] if e['document'] == 'test_text_t8192')], output=str(directory))
                    if name in result['profiles']:
                        if (digest(directory/'profile.json') != result['profiles'][name]['summary_sha256'] or
                                json.loads((directory/'request.json').read_text()) != q):
                            raise ValueError('completed profile receipt changed')
                        continue
                    write(directory/'request.json', q)
                    python, env = worker_environment(base, recipe['native_receipt'])
                    launch([python, Path(__file__).resolve(), '--device', str(device), '--profile-worker', directory/'request.json'], directory/'profile.log', env)
                    data = json.loads((directory/'profile.json').read_text())
                    result['profiles'][name] = dict(data, summary_sha256=digest(directory/'profile.json'),
                        request_sha256=digest(directory/'request.json')); save()
        result['status'] = 'completed'; save()
        from scripts.package_team_lookup import verify
        verify(out, result); log('completed: '+str(out/'summary.md')); return 0
    except Exception as error:
        if writable:
            result.update(status='failed', error=str(error), traceback=traceback.format_exc()); save()
        traceback.print_exc(); return 1


if __name__ == '__main__':
    raise SystemExit(main())
