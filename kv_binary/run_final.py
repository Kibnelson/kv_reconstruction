"""Run frozen final cases in independent CPU jobs and verify a complete merge.

Parallel jobs are evaluation orchestration, not concurrent LLM serving.
Worker wall times overlap and share CPU resources; do not compare them as clean
serial codec benchmarks. No candidate selection or training occurs here.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from kv_stacked.run_eval import aggregate


def verify_freeze(path):
    if path is None:
        return None
    freeze = json.loads(pathlib.Path(path).read_text())
    frozen = freeze['source_and_model_sha256']
    if not frozen:
        raise ValueError('empty source/model freeze')
    for name, expected in frozen.items():
        actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError('frozen source/model changed: ' + name)
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--data-dir')
    p.add_argument('--freeze')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--generation-candidate', action='append', default=[])
    p.add_argument('--save-sample', action='store_true')
    p.add_argument('--model-dir', default='.smoke_model')
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        raise ValueError('protocol permits one to three CPU jobs')
    freeze_sha256 = verify_freeze(args.freeze)
    config = json.loads(pathlib.Path(args.config).read_text())
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    if args.freeze and manifest.get('selection_sha256') != freeze_sha256:
        raise AssertionError('capture was not made under this source/configuration freeze')
    if len(manifest['cases']) != 6:
        raise ValueError('final protocol requires six prespecified cases')
    output = pathlib.Path(args.output)
    output.parent.mkdir(exist_ok=True, parents=True)
    started = time.perf_counter()
    def run(index, case):
        partial = output.parent / f'final_case_{index}.json'
        log = output.parent / f'final_case_{index}.log'
        cmd = [sys.executable, '-u', '-m', 'kv_binary.run_eval',
            '--config', args.config, '--manifest', args.manifest,
            '--output', str(partial), '--only-index', str(index), '--model-dir', args.model_dir]
        if args.data_dir:
            cmd += ['--data-dir', args.data_dir]
        if index == 0:
            for name in args.generation_candidate:
                cmd += ['--generation-candidate', name]
            if args.save_sample:
                cmd.append('--save-sample')
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        with log.open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)
        print('completed ' + case['name'], flush=True)
        return index, partial
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, i, case) for i, case in enumerate(manifest['cases'])]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
    partials = [json.loads(path.read_text()) for _, path in sorted(completed)]
    if len(partials) != 6:
        raise AssertionError('missing final cases')
    names = set(config['candidates'])
    seen = set()
    for case, partial in zip(manifest['cases'], partials):
        if partial['config'] != config or partial['capture_manifest'] != manifest:
            raise AssertionError('configuration/capture manifest mismatch')
        if partial['state_bytes'] != partials[0]['state_bytes'] or len(partial['cases']) != 1:
            raise AssertionError('decoder state/case count mismatch')
        row = partial['cases'][0]
        if row['name'] != case['name'] or row['name'] in seen or set(row['codecs']) != names:
            raise AssertionError('case identity/candidate mismatch')
        seen.add(row['name'])
        if not all(s['roundtrip_exact'] and s['independent_later_tile'] for s in row['codecs'].values()):
            raise AssertionError('failed final verification')
    if verify_freeze(args.freeze) != freeze_sha256:
        raise AssertionError('freeze file changed during final evaluation')
    result = partials[0]
    result['cases'] = [p['cases'][0] for p in partials]
    result['aggregate'] = aggregate(result['cases'], result['state_bytes'], list(config['candidates']))
    result['test_job_execution'] = dict(workers=args.workers,
        elapsed_seconds=time.perf_counter() - started, freeze_sha256=freeze_sha256,
        orchestration_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        note='Independent CPU correctness/storage jobs. Timings overlap and share CPU resources; not a clean serial benchmark or concurrent LLM serving.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('completed all six final cases: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
