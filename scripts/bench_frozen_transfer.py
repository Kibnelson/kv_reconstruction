#!/usr/bin/env python3
"""Matched lossless GPU transfer calls on frozen 810b516 words and state.

The official native WikiText procedure and ExANS synthetic preflight are gates,
not comparable workloads. This harness measures local complete transfer adapters.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN = 'benchmark_results/team_lookup_a4500_20261028_141701_1145261_report_20261028_143153'
SPLITZIP = 'benchmark_results/splitzip_official_native_a4500_20261028_212622_1967778'
EXANS = 'benchmark_results/exans_official_preflight_a4500_20261028_213554_2171122'
INVENTORY = 'benchmark_results/nuc_baseline_readiness_20261028_200201_158793/results.json'
RESULT_HASHES = {
    FROZEN: 'bf0035394d0660aa392c0629ab4bb8a0dd8594e3b4d7b03cc750f58ba3a1e120',
    SPLITZIP: '23df669d125c688e6ee3c838b959ccaf90b325501fda9abfb0602ef157757f20',
    EXANS: '4e36e3a8dcc05d2fae6debafb1151294f60979af1cd67bf3c6676e581040a99d',
}
BASELINES = ('raw_bf16', 'splitzip_official_framed', 'exans_official_fixed_framed')
OWN = ('team_dense_uniform', 'team_rank256_uniform')
METHODS = BASELINES + OWN
LENGTHS = (512, 8192)
LINKS = (1, 100, 200, 300, 400)
SCOPE = ('Matched complete lossless transfer adapters on frozen PEP/CPython KV. '
    'No source/model/predictor change. Native WikiText results are separate. '
    'ExANS retains the original 59/941-block C-ABI adapter, fixed slots, padding and raw fallback; '
    'this is not its maximum-64-block, stop-on-overflow vLLM serving procedure.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def confined(path, root=ROOT):
    p = Path(path).resolve()
    if not p.is_relative_to(root.resolve()) or p == root.resolve():
        raise ValueError('path must be contained in this repository: ' + str(path))
    return p


def verify_package(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'package_manifest.json').read_text())
    if not isinstance(manifest, dict):
        raise ValueError('package manifest must be an object')
    files = manifest.get('files', manifest)
    if not isinstance(files, dict) or not files:
        raise ValueError('package manifest must contain file receipts')
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not isinstance(expected, dict):
            raise ValueError('package manifest has an invalid relative file path/receipt')
        original = directory / rel
        if any(p.is_symlink() for p in (original, *original.parents) if p != directory and p.is_relative_to(directory)):
            raise ValueError('package evidence must not use symlinks')
        path = confined(original, directory)
        if type(expected.get('bytes')) is not int or expected['bytes'] < 0 or not isinstance(expected.get('sha256'), str) or len(expected['sha256']) != 64:
            raise ValueError('invalid package length/hash receipt')
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('published evidence changed: ' + str(path))
    return dict(files=len(files), manifest_sha256=sha(directory / 'package_manifest.json'))


def validate_selections(selections):
    if set(selections) != {f'{method}_t{n}' for n in LENGTHS for method in METHODS}:
        raise ValueError('frozen method/length selections changed')
    for n in LENGTHS:
        expected = {}
        for method in METHODS:
            item = selections[f'{method}_t{n}']
            if item['method'] != method or item['tokens'] != n or len(item['entries']) != 2:
                raise ValueError('selection identity differs')
            documents = set()
            for entry in item['entries']:
                doc = entry['document']; documents.add(doc)
                if entry['shape'] != [28, n, 2048] or entry['values'] != 28 * n * 2048:
                    raise ValueError('frozen KV geometry differs')
                identity = (entry['original_word_sha256'], tuple(entry['shape']), entry['values'])
                if doc in expected and expected[doc] != identity:
                    raise ValueError('methods do not receive identical original words')
                expected[doc] = identity
            if documents != {f'test_text_t{n}', f'test_code_t{n}'}:
                raise ValueError('frozen TEST documents changed or repeated')
    return selections


def source_checks():
    from scripts.run_splitzip_official_native import source_validation
    from scripts.run_exans_official_preflight import source_receipt
    return dict(splitzip=source_validation(ROOT / 'runs/_llama_official/sources/splitzip'),
                exans=source_receipt(ROOT / 'runs/_llama_official/sources/exans'))


def measurement_source_receipts():
    names = ('scripts/bench_frozen_transfer.py', 'scripts/frozen_transfer_worker.py',
             'scripts/check_nuc_baseline_readiness.py', 'scripts/run_splitzip_official_native.py',
             'scripts/run_exans_official_preflight.py')
    return {name:sha(ROOT / name) for name in names}


def verified_inputs():
    from scripts.check_nuc_baseline_readiness import frozen_sources
    packages = {name: verify_package(ROOT / path) for name, path in (
        ('frozen', FROZEN), ('splitzip', SPLITZIP), ('exans', EXANS))}
    for path, expected in RESULT_HASHES.items():
        if sha(ROOT / path / 'results.json') != expected:
            raise ValueError('committed completed result identity changed: ' + path)
    old = json.loads((ROOT / FROZEN / 'results.json').read_text())
    sz = json.loads((ROOT / SPLITZIP / 'results.json').read_text())
    ex = json.loads((ROOT / EXANS / 'results.json').read_text())
    if old['status'] != 'completed' or not old['gpu_measured']:
        raise ValueError('frozen reference is not a completed CUDA study')
    if sz['status'] != 'completed' or not sz['gpu_measured'] or not sz['validation']['original_word_exact'] or sz['source_before'] != sz['source_after']:
        raise ValueError('official SplitZip native gate is incomplete')
    if ex['status'] != 'completed' or not ex['gpu_correctness_test_passed'] or ex['gpu_performance_measured'] or ex['source_before'] != ex['source_after']:
        raise ValueError('official ExANS correctness gate is incomplete')
    inv = json.loads((ROOT / INVENTORY).read_text())
    if Path(inv['root']).resolve() != ROOT:
        raise ValueError('this is not the inventory-bound NUC checkout')
    code = frozen_sources(ROOT)
    if not code['all_match']:
        raise ValueError('the 84 frozen source files changed')
    selections = validate_selections(old['recipe']['selections'])
    checked = {}
    for item in selections.values():
        spec = item['codec']
        if spec['kind'] == 'splitzip_official_framed':
            path = confined(spec['state'])
            if sha(path) != spec['state_sha256']:
                raise ValueError('frozen TRAIN SplitZip LUTs changed')
            checked[str(path)] = dict(sha256=spec['state_sha256'], bytes=path.stat().st_size)
        elif spec['kind'] == 'exans':
            path = confined(spec['library'])
            if spec['library_sha256'] != old['recipe']['native_receipt']['exans_library_sha256'] or sha(path) != spec['library_sha256']:
                raise ValueError('original frozen ExANS library changed; do not substitute the fresh preflight build')
            if spec['blocks'] != (59 if item['tokens'] == 512 else 941):
                raise ValueError('original ExANS component batch geometry changed')
            checked[str(path)] = dict(sha256=spec['library_sha256'], bytes=path.stat().st_size)
        elif spec['kind'] == 'team':
            state = confined(spec['state']); path = state / 'manifest.json'
            manifest = json.loads(path.read_text())
            if sha(path) != spec['manifest_sha256'] or manifest['state_sha256'] != spec['state_sha256'] or sha(state / 'tensors.npz') != manifest['archive_sha256']:
                raise ValueError('frozen predictor manifest/archive changed')
            checked[str(path)] = dict(sha256=spec['manifest_sha256'])
            checked[str(state / 'tensors.npz')] = dict(sha256=manifest['archive_sha256'])
        for entry in item['entries']:
            path = confined(entry['packet'])
            if path.stat().st_size != entry['packet_bytes'] or sha(path) != entry['packet_sha256']:
                raise ValueError('frozen transfer packet changed: ' + str(path))
            checked[str(path)] = dict(sha256=entry['packet_sha256'], bytes=entry['packet_bytes'])
    return selections, dict(packages=packages, frozen_source_check=code,
        author_sources=source_checks(), fixed_artifacts=checked, native_receipt=old['recipe']['native_receipt'],
        exans_runtime=ex['runtime'], official_splitzip_native=sz['native'])


def job_plan(selections):
    native = [(f'{method}_t{n}', selections[f'{method}_t{n}']) for n in LENGTHS for method in BASELINES]
    own = [(f'{method}_t{n}', selections[f'{method}_t{n}']) for n in LENGTHS for method in OWN]
    plan = [(f'before_{name}', dict(q, round='before')) for name, q in native]
    plan += [(name, dict(q, round='paired')) for name, q in own]
    plan += [(f'after_{name}', dict(q, round='after')) for name, q in reversed(native)]
    return [(name, dict(q, raw_entries=selections[f"raw_bf16_t{q['tokens']}"]['entries'],
        execution=('compact_search' if q['tokens'] == 512 else 'direct_search') if q['method'] in OWN else 'native',
        timing_source=str(ROOT / 'runs/_llama_official/sources/splitzip/bench_codec_throughput.py')))
        for name, q in plan]


def worker_environment(request, receipt):
    py = ROOT / ('runs/_llama_official/envs/splitzip/bin/python' if request['method'] == BASELINES[1] else '.venv/bin/python')
    env = dict(os.environ)
    for key in ('TRITON_INTERPRET', 'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS='16', OPENBLAS_NUM_THREADS='16', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    if request['method'] == BASELINES[2]:
        library = Path(request['codec']['library'])
        tool = library.parent.parent / 'toolchain.json'
        if sha(tool) != receipt['native_receipt']['toolchain_sha256']:
            raise ValueError('frozen ExANS toolchain receipt changed')
        home = Path(json.loads(tool.read_text())['cuda_home']); sdk = Path(receipt['exans_runtime']['sdk'])
        paths = [library.parent, home / 'lib64', home / 'lib', sdk / 'lib64', sdk / 'lib']
        env['LD_LIBRARY_PATH'] = os.pathsep.join(str(p) for p in paths if p.is_dir())
    return py, env


def validate_worker(result, request, jobdir):
    if result['status'] != 'completed' or not result['gpu_measured'] or len(result['rows']) != 2:
        raise ValueError('worker did not complete two actual CUDA TEST rows')
    entries = {entry['document']: entry for entry in request['entries']}
    seen = set()
    for row in result['rows']:
        entry = entries[row['document']]; seen.add(row['document'])
        if (row['method'] != request['method'] or row['tokens'] != request['tokens'] or row['round'] != request['round']
            or row['execution'] != request['execution']
            or row['shape'] != entry['shape'] or row['values'] != entry['values']
            or row['original_word_sha256'] != entry['original_word_sha256'] or not row['original_word_exact']
            or row['decoded_word_sha256'] != entry['original_word_sha256']
            or row['frozen_packet_sha256'] != entry['packet_sha256']
            or row['frozen_packet_bytes'] != entry['packet_bytes'] or row['packet_bytes'] != entry['packet_bytes']):
            raise ValueError('worker row changed frozen input/packet identity or exactness')
        same_wire = row['packet_sha256'] == entry['packet_sha256']
        if row['encode_packet_matches_frozen'] != same_wire or (request['method'] != BASELINES[2] and not same_wire):
            raise ValueError('frozen deterministic packet bytes changed')
        encoded_path = confined(row['packet'], Path(jobdir))
        if encoded_path.stat().st_size != row['packet_bytes'] or sha(encoded_path) != row['packet_sha256']:
            raise ValueError('actual serialized encoder output changed')
        for operation in ('encode', 'decode'):
            batched = row[operation]['batched']
            isolated = row[operation]['isolated']
            if (any(batched.get(k) != v for k, v in dict(warmup=10, iters=50, repeats=5).items())
                or len(batched.get('samples_seconds', [])) != 5
                or any(not math.isfinite(x) or x <= 0 for x in batched['samples_seconds'])
                or len(isolated.get('samples', [])) != 5 or len(isolated.get('warmup_samples', [])) != 2):
                raise ValueError('worker timing convention differs')
            for basis, key in (('batched', 'wall_mean_ms'), ('isolated', 'wall_median_ms')):
                value = row[operation][basis][key]
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('invalid matched timing')
    if seen != set(entries):
        raise ValueError('worker repeated or omitted a document')
    proof = Path(jobdir) / 'receiver.json'
    if sha(proof) != result['receiver_sha256']:
        raise ValueError('independent receiver receipt changed')
    proof_data = json.loads(proof.read_text())
    if (not all(proof_data.get(key) for key in ('gpu_measured', 'no_source_kv_or_predictions', 'alternating_packets', 'independent_returned_outputs'))
        or result['runtime_policy'] != result['runtime_policy_after']
        or proof_data['runtime_policy'] != result['runtime_policy'] or proof_data['runtime_policy_after'] != result['runtime_policy']):
        raise ValueError('receiver ownership/runtime proof is incomplete')
    expected_graphs = int(request['execution'] == 'compact_search')
    if result['graph_capture_count'] != expected_graphs or proof_data['graph_capture_count'] != expected_graphs:
        raise ValueError('declared decoder execution differs from measured graph count')
    checks = {item['id']:item for item in proof_data['checks']}
    for row in result['rows']:
        check = checks.get(row['id'], {})
        if not check.get('verified') or check.get('shape') != row['shape'] or check.get('word_sha256') != row['original_word_sha256'] or check.get('packet_sha256') != row['packet_sha256']:
            raise ValueError('independent receiver did not verify actual encoder output')
    for key in ('packet_immutability', 'frozen_packet_immutability', 'raw_packet_immutability'):
        if len(result[key]) != 2 or not all(item['unchanged'] for item in result[key]):
            raise ValueError('worker compressed inputs changed')
    receiver_request = json.loads((Path(jobdir) / 'receiver_request.json').read_text())
    from scripts.frozen_transfer_worker import valid_receiver_request
    valid_receiver_request(receiver_request)
    if set(receiver_request) != {'method', 'tokens', 'codec', 'entries', 'execution', 'output'}:
        raise ValueError('fresh receiver was supplied undeclared source inputs')
    for key in ('method', 'tokens', 'codec', 'execution'):
        if receiver_request[key] != request[key]:
            raise ValueError('fresh receiver configuration differs')
    if len(receiver_request['entries']) != 2:
        raise ValueError('fresh receiver packet coverage differs')
    return result['rows']


def aggregate(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['tokens'], row['method'], row['round']), []).append(row)
    result = []
    for (tokens, method, phase), group in sorted(grouped.items()):
        if len(group) != 2 or len({r['document'] for r in group}) != 2:
            raise ValueError('aggregate requires both TEST documents exactly once')
        item = dict(tokens=tokens, method=method, round=phase, documents=[r['document'] for r in group],
            packet_bytes=statistics.mean(r['packet_bytes'] for r in group),
            complete_bpv=statistics.mean(8 * r['packet_bytes'] / r['values'] for r in group),
            saved_percent=statistics.mean(100 * (1 - r['packet_bytes'] / (2 * r['values'])) for r in group))
        for basis, key in (('batched', 'wall_mean_ms'), ('isolated', 'wall_median_ms')):
            item[basis] = {op + '_ms': statistics.mean(r[op][basis][key] for r in group) for op in ('encode', 'decode')}
        result.append(item)
    return result


def link_models(aggregates):
    rows = []
    for item in aggregates:
        for basis in ('batched', 'isolated'):
            e, d = item[basis]['encode_ms'], item[basis]['decode_ms']
            for gbps in LINKS:
                wire = 8 * item['packet_bytes'] / (gbps * 1e6)
                row = dict(tokens=item['tokens'], method=item['method'], round=item['round'], timing_basis=basis,
                    effective_gbps=gbps, wire_ms=wire, precompressed_serial_ms=wire + d,
                    fresh_serial_ms=e + wire + d, ideal_precompressed_lower_bound_ms=max(wire, d),
                    ideal_fresh_lower_bound_ms=max(e, wire, d), measured_network=False, baseline_targets={})
                for method in BASELINES[1:]:
                    controls = [a for a in aggregates if a['tokens'] == item['tokens'] and a['method'] == method]
                    if not controls:
                        continue
                    # Independently choose the faster full before/after cost for each objective.
                    targets = {}
                    for objective in ('precompressed', 'fresh'):
                        def cost(a):
                            return 8 * a['packet_bytes'] / (gbps * 1e6) + a[basis]['decode_ms'] + (a[basis]['encode_ms'] if objective == 'fresh' else 0)
                        target = min(controls, key=cost)
                        own_cost = row[objective + '_serial_ms']
                        targets[objective] = dict(round=target['round'], serial_ms=cost(target), speedup=cost(target) / own_cost)
                    row['baseline_targets'][method] = targets
                rows.append(row)
    return dict(measured_network=False, rows=rows, scope='Calculated serial GPU encode + ideal wire + full decode and precompressed wire + decode; max bounds are unmeasured perfect overlap. No NIC/SSD/export/upload/serving measurement.')


def report(result, out):
    lines = ['# Frozen lossless transfer comparison', '', '**Status: ' + result['status'] + '.** GPU measured: **' + str(result['gpu_measured']) + '**.', '', SCOPE, '']
    if result.get('error'):
        lines += ['Stopped: `' + result['error'].replace('`', "'") + '`. Partial outputs are preserved.', '']
    for basis in ('batched', 'isolated'):
        lines += ['Timing basis: **' + basis + '**.', '', '| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |', '|---:|---|---|---:|---:|---:|']
        for row in result.get('aggregates', []):
            lines.append(f"| {row['tokens']} | {row['method']} | {row['round']} | {row['complete_bpv']:.5f} | {row[basis]['encode_ms']:.3f} | {row[basis]['decode_ms']:.3f} |")
        lines.append('')
    modeled = result.get('link_models', {}).get('rows', [])
    if modeled:
        lines += ['Calculated **precompressed wire + full GPU decode**, milliseconds. Baseline cells use '
            'the faster before/after round as a predeclared conservative target; ours uses its paired round.', '']
        for basis in ('batched', 'isolated'):
            lines += ['Timing basis: **' + basis + '**.', '',
                '| Tokens | Gb/s | Raw | SplitZip | ExANS adapter | Dense | Rank256 |',
                '|---:|---:|---:|---:|---:|---:|---:|']
            for n in LENGTHS:
                for gbps in LINKS:
                    cells = []
                    for method in METHODS:
                        choices = [r['precompressed_serial_ms'] for r in modeled if
                            (r['tokens'], r['effective_gbps'], r['timing_basis'], r['method']) == (n, gbps, basis, method)]
                        cells.append(f'{min(choices):.3f}' if choices else 'pending')
                    lines.append(f'| {n} | {gbps} | ' + ' | '.join(cells) + ' |')
            lines.append('')
        lines += ['Fresh **encode + wire + decode** and perfect-overlap lower bounds are recorded separately '
            'for both timing bases in `results.json`; overlap is not measured.', '']
    lines += ['Batched timing uses the pinned author SplitZip bench loop: 10 warmups, 50 calls between synchronizations, five repeats. Isolated timing records per-call synchronized medians separately. Both are complete GPU adapter calls; source data/uploads, packet serialization, fitting and cold setup are excluded.', '',
        'Link models at 1/100/200/300/400 Gb/s are calculated, not measured network transfers. Faster before/after baseline controls are conservative targets; every round remains visible. Do not interpret a timing-basis change from 810b516 as a codec optimization.', '',
        'All results here are lossless. Official lossy AQUA quality, resident P attention and T-to-P conversion remain separate experiments.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); out = confined(args.output)
    out.mkdir(parents=True, exist_ok=False)
    result = dict(format='frozen_transfer_matched_v1', status='failed', gpu_measured=False, scope=SCOPE, jobs={}, rows=[])
    try:
        result['measurement_source_before'] = measurement_source_receipts()
        selections, receipt = verified_inputs(); result['verification'] = receipt
        result['source_before'] = source_checks()
        write(out / 'results.json', result)
        for name, request in job_plan(selections):
            job = out / name; job.mkdir()
            request['output'] = str(job); write(job / 'request.json', request)
            py, env = worker_environment(request, receipt)
            argv = [str(py), str(ROOT / 'scripts/frozen_transfer_worker.py'), '--request', str(job / 'request.json'), '--device', 'cuda:0']
            print('Running ' + name, flush=True); started = time.monotonic()
            with (job / 'worker.log').open('w') as stream:
                child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while True:
                        try:
                            code = child.wait(timeout=30); break
                        except subprocess.TimeoutExpired:
                            print('Still running ' + name + '; log: ' + str(job / 'worker.log'), flush=True)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    raise
            result['jobs'][name] = dict(argv=argv, returncode=code, command_wall_seconds=time.monotonic() - started)
            if code: raise RuntimeError(name + ' exited ' + str(code) + '; see worker.log')
            worker = json.loads((job / 'worker_results.json').read_text())
            result['rows'].extend(validate_worker(worker, request, job)); result['gpu_measured'] = True
            result['jobs'][name]['artifacts'] = {str(p.relative_to(job)):dict(bytes=p.stat().st_size, sha256=sha(p)) for p in job.rglob('*') if p.is_file() and p.suffix in ('.json', '.log')}
            result['aggregates'] = aggregate(result['rows']); write(out / 'results.json', result); report(result, out)
        result['source_after'] = source_checks()
        if result['source_before'] != result['source_after']:
            raise ValueError('author sources changed during measurement')
        from scripts.check_nuc_baseline_readiness import frozen_sources
        result['frozen_source_after'] = frozen_sources(ROOT)
        if not result['frozen_source_after']['all_match']:
            raise ValueError('frozen sources changed during measurement')
        result['measurement_source_after'] = measurement_source_receipts()
        if result['measurement_source_before'] != result['measurement_source_after']:
            raise ValueError('measurement source changed during measurement')
        result['link_models'] = link_models(result['aggregates']); result['status'] = 'completed'
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        (out / 'failure.log').write_text(traceback.format_exc()); print(result['error'], file=sys.stderr, flush=True)
    finally:
        result['runner_sha256'] = sha(__file__)
        result['worker_sha256'] = sha(ROOT / 'scripts/frozen_transfer_worker.py')
        write(out / 'results.json', result); report(result, out); print('Report: ' + str(out / 'summary.md'), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
