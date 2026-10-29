#!/usr/bin/env python3
"""Run the pinned, unchanged SplitZip native entry point after local verification.

Only the small preflight imports this harness in the author environment. The
performance child is the author's script itself, with all workload/timing defaults.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PIN = '42c743118a29db3270ae6a921a4a4eecb20cad04'
MODEL = 'unsloth/Llama-3.2-3B'
REVISION = 'd4446454d87d51aa42e1fb174f25acc5f8762331'
INVENTORY = 'benchmark_results/nuc_baseline_readiness_20261028_200201_158793/results.json'
AUTHOR_HASHES = {
    'bench_codec_throughput.py': '7f1936a13ee3b2bbb04a35b71ca14a88bc407fe9b67e2c128d2ec48df2d018e2',
    'codec_gpu.py': 'e28fe3263e8906b26d59789cbc5bccf14edc8391b0dae986aafaf34adb2d392c',
}
SCOPE = ('Unchanged released SplitZip native procedure on Llama-3.2-3B / A4500; '
         'model/hardware variation from Qwen3-32B / H200. Native codec bytes and '
         'original-byte API throughput, not framed transfer, NIC, SSD or resident attention.')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def output_path(path, root=ROOT):
    candidate = Path(path)
    p = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if p == root.resolve() or not p.is_relative_to(root.resolve()):
        raise ValueError('output must be a new directory inside the repository')
    if p.exists():
        raise ValueError('output already exists; preserve it and use a new directory')
    return p


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def source_validation(source):
    if git(source, 'rev-parse', 'HEAD') != PIN:
        raise ValueError('SplitZip author checkout is not the pinned commit')
    if git(source, 'diff', '--name-only') or git(source, 'diff', '--cached', '--name-only'):
        raise ValueError('tracked SplitZip author files were changed')
    files = {n: sha(source / n) for n in AUTHOR_HASHES}
    if files != AUTHOR_HASHES:
        raise ValueError('SplitZip source hashes differ from the reviewed pin')
    # An untracked Python file could shadow imported dependencies inside this cwd.
    extra = git(source, 'ls-files', '--others', '--exclude-standard').splitlines()
    if any(n.endswith(('.py', '.so')) for n in extra):
        raise ValueError('untracked executable Python/native files in author checkout')
    return dict(commit=PIN, author_sources_unchanged=True, files=files,
                untracked_files=extra)


def model_receipt(snapshot, entry):
    if snapshot.name != REVISION or snapshot.parent.name != 'snapshots':
        raise ValueError('model snapshot path does not bind the pinned revision')
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    weights = sorted(set(index['weight_map'].values()))
    names = ['config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json']
    names += [n for n in ('special_tokens_map.json', 'generation_config.json') if (snapshot / n).is_file()]
    expected_sizes = {w['name']: w['bytes'] for w in entry['weights']}
    out = {}
    for name in names + weights:
        path = snapshot / name
        if Path(name).name != name or not path.is_file():
            raise ValueError('missing or invalid model file: ' + name)
        size = path.stat().st_size
        if name in expected_sizes and size != expected_sizes[name]:
            raise ValueError('model weight size differs from readiness receipt: ' + name)
        digest = sha(path)
        blob = path.resolve().name if path.is_symlink() else None
        expected = blob if blob and re.fullmatch(r'[0-9a-f]{64}', blob) else None
        if expected and digest != expected:
            raise ValueError('cached Hugging Face blob content hash failed: ' + name)
        out[name] = dict(bytes=size, sha256=digest, hf_blob_sha256=expected,
                         hf_blob_hash_verified=bool(expected))
    if set(weights) != set(expected_sizes):
        raise ValueError('model index weights differ from readiness receipt')
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer_config.json'):
        prior = entry['files'][name]
        if prior.get('sha256') != out[name]['sha256']:
            raise ValueError('model metadata differs from readiness receipt: ' + name)
    return dict(model=MODEL, revision=REVISION, snapshot=str(snapshot), files=out,
                scope='Full local file hashes; HF content-addressed symlink targets checked where present. '
                      'Ordinary-file hashes alone are not a remote model authenticity check.')


def native_result_validation(r, snapshot):
    for key, value in dict(device='cuda:0', numel=268435456, chunk_size=1024,
                           raw_bytes=536870912, warmup=10, iters=50, repeats=5).items():
        if r.get(key) != value:
            raise ValueError('official output default mismatch: ' + key)
    expected = (r['numel'] + 1) // 2 + r['numel'] + 7 * r['n_escapes']
    if not 0 <= r['n_escapes'] <= r['numel'] or r['compressed_bytes'] != expected:
        raise ValueError('native compressed byte accounting mismatch')
    calibration = r['calibration']
    for key, value in dict(calibration_source='model_forward', model=str(snapshot),
                           dataset='wikitext', dataset_config='wikitext-2-raw-v1',
                           split='train', max_prompts=4, max_values=4194304).items():
        if calibration.get(key) != value:
            raise ValueError('calibration protocol mismatch: ' + key)
    benchmark = r['benchmark_source']
    for key, value in dict(source='model_forward', model=str(snapshot), dataset='wikitext',
                           dataset_config='wikitext-2-raw-v1', split='test', max_prompts=4).items():
        if benchmark.get(key) != value:
            raise ValueError('benchmark protocol mismatch: ' + key)
    for operation in ('encode', 'decode'):
        seconds = r[operation + '_seconds']['mean']
        gbs = r[operation + '_gbs']['mean']
        if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(gbs) or gbs <= 0:
            raise ValueError('invalid native timing')
    return dict(original_word_exact=True,
                exactness_evidence='Pinned author entrypoint exited zero after its int16 equality gate.',
                native_bpv=8 * r['compressed_bytes'] / r['numel'],
                saved_percent=100 * (1 - r['compressed_bytes'] / r['raw_bytes']))


def worker(request_path):
    """Untimed correctness and dataset/token receipts, never performance inputs."""
    import importlib.metadata as metadata
    import importlib.util
    from types import SimpleNamespace
    request = json.loads(Path(request_path).read_text())
    versions = {n: metadata.version(n) for n in request['versions']}
    if versions != request['versions']:
        raise ValueError('SplitZip environment versions changed since inventory')
    import torch
    import numpy as np
    from datasets import load_dataset
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA is required; CPU/interpreter results are forbidden')
    torch.cuda.set_device(0)
    if 'RTX A4500' not in torch.cuda.get_device_name(0):
        raise RuntimeError('this named NUC procedure requires the RTX A4500; preserve other hardware as a separate setting')
    source = Path(request['source'])
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('splitzip_author_benchmark', source / 'bench_codec_throughput.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Includes every BF16 bit pattern; all comparisons operate on original words.
    words = torch.arange(65536, dtype=torch.int32, device='cuda:0').to(torch.int16)
    codec = module.ChunkLocalSplitZipGPU(device='cuda:0', chunk_size=1024)
    codec.calibrate(words.view(torch.bfloat16))
    recovered = codec.decode(codec.encode(words.roll(37).view(torch.bfloat16)))
    torch.cuda.synchronize()
    if not torch.equal(recovered.view(torch.int16), words.roll(37)):
        raise AssertionError('untimed author CUDA codec word equality failed')
    args = SimpleNamespace(model=request['snapshot'], cache_dir=None, dataset_cache_dir=None,
                           dataset='wikitext', dataset_config='wikitext-2-raw-v1', min_text_chars=80,
                           local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                              local_files_only=True)
    receipts = {}
    for split in ('train', 'test'):
        ds = load_dataset(args.dataset, args.dataset_config, split=split, cache_dir=None)
        texts = module.load_wikitext_texts(args, split, 4)
        content = hashlib.sha256()
        chosen = []
        for row_id, row in enumerate(ds):
            raw = row['text'].encode('utf-8')
            content.update(len(raw).to_bytes(8, 'little')); content.update(raw)
            normalized = ' '.join(row['text'].split())
            if len(normalized) >= 80 and len(chosen) < 4:
                ids = tokenizer(normalized, return_tensors='pt')['input_ids']
                chosen.append(dict(row=row_id, text_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                                   tokens=ids.numel(), token_ids_sha256=hashlib.sha256(
                                       ids.numpy().astype(np.dtype('<i8'), copy=False).tobytes()).hexdigest()))
        expected_texts = [' '.join(ds[c['row']]['text'].split()) for c in chosen]
        if texts != expected_texts:
            raise AssertionError('receipt prompt selection differs from author loader')
        caches = []
        for item in ds.cache_files:
            p = Path(item['filename'])
            caches.append(dict(filename=p.name, bytes=p.stat().st_size, sha256=sha(p)))
        receipts[split] = dict(rows=len(ds), fingerprint=ds._fingerprint,
                               text_rows_length_prefixed_sha256=content.hexdigest(),
                               selected_prompts=chosen, cache_files=caches)
    props = torch.cuda.get_device_properties(0)
    result = dict(status='verified_preflight', gpu_correctness_measured=True,
                  performance_measured=False, original_word_exact=True, bf16_patterns=65536,
                  versions=versions, cuda=torch.version.cuda,
                  gpu=dict(name=props.name, total_memory=props.total_memory,
                           capability=list(torch.cuda.get_device_capability(0))),
                  dataset=receipts,
                  scope='Untimed native correctness and separate data/token receipts; '
                        'these fixtures are not the native performance workload.')
    write(request['output'], result)


def summary(result, out):
    lines = ['# Official SplitZip native procedure on the NUC', '',
             '**Status: ' + result['status'] + '.** GPU performance measured: **' + str(result['gpu_measured']) + '**.', '', SCOPE, '']
    if result.get('error'):
        lines += ['Stopped: `' + result['error'].replace('`', "'") + '`. Logs and partial receipts are preserved.', '']
    if result['status'] == 'completed' and result.get('native') and result.get('validation'):
        r = result['native']; v = result['validation']
        lines += ['| Native bpv | Saved | Encode GB/s | Decode GB/s | Original-word exact |',
                  '|---:|---:|---:|---:|---|',
                  f"| {v['native_bpv']:.6f} | {v['saved_percent']:.2f}% | {r['encode_gbs']['mean']:.3f} | {r['decode_gbs']['mean']:.3f} | True |", '']
    lines += ['Official defaults: WikiText2 raw TRAIN/TEST, four usable prompts, 4,194,304 calibration values, '
              '268,435,456 TEST values (512 MiB BF16), 1,024-value chunks, 10 warmups, 50 iterations and five repeats.', '',
              'The author repeats/truncates the collected flat KV prefix to the requested value count. '
              'This is not the structured workload of our previous cross-layer predictor study; no cross-workload speed ranking is made.', '',
              'Native compressed bytes include seven bytes per escape. Shared calibration, framing and installed receiver state '
              'are outside this native metric. Link scenarios, compressed attention and T-to-P conversion are not measured.', '',
              'AQUA-KV remains a separate lossy-quality experiment. ExANS serving/paper reproduction is not established by this run.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=ROOT / INVENTORY)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.output is None:
        parser.error('--output is required')
    out = output_path(args.output); out.mkdir(parents=True)
    result = dict(format='splitzip_official_native_nuc_v1', status='failed', gpu_measured=False,
                  scope=SCOPE, commands=[], inventory=str(args.inventory.resolve()),
                  native_performance_attempted=False,
                  model_placement=dict(requested_device_map='auto', actual_module_placement='unobserved in unchanged author process'),
                  performance_input_word_sha256=None,
                  performance_input_identity_scope='Native output does not expose the collected/repeated KV tensor digest; data/token receipts are separate preflight evidence.')
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    for name in ('TRITON_INTERPRET', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    source = ROOT / 'runs/_llama_official/sources/splitzip'
    py = ROOT / 'runs/_llama_official/envs/splitzip/bin/python'
    def run(name, argv, cwd):
        record = dict(name=name, argv=[str(x) for x in argv], cwd=str(cwd))
        result['commands'].append(record)
        write(out / 'results.json', result)
        print('Running ' + name, flush=True)
        start = time.monotonic()
        with (out / (name + '.log')).open('w') as log:
            p = subprocess.run(record['argv'], cwd=cwd, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=False)
        record.update(returncode=p.returncode, command_wall_seconds=time.monotonic() - start)
        if p.returncode:
            raise RuntimeError(name + ' exited ' + str(p.returncode) + '; see ' + name + '.log')
    try:
        inv = json.loads(args.inventory.read_text())
        if inv['format'] != 'nuc_official_baseline_readiness_v1' or inv['model']['revision'] != REVISION or inv['model']['model'] != MODEL:
            raise ValueError('unexpected readiness inventory/model identity')
        if Path(inv['root']).resolve() != ROOT or Path(inv['authors']['splitzip']['path']).resolve() != source or Path(inv['environments']['splitzip']['path']).resolve() != py.resolve():
            raise ValueError('inventory paths do not bind this repository/source/environment')
        result['inventory_sha256'] = sha(args.inventory)
        result['source_before'] = source_validation(source)
        versions = {n: v for n, v in inv['environments']['splitzip']['metadata']['versions'].items() if v is not None}
        snapshots = [s for s in inv['model']['snapshots'] if s['cached_files_present'] and Path(s['path']).is_dir()]
        if len(snapshots) != 1:
            raise ValueError('expected one existing exact-revision model snapshot')
        snapshot = Path(snapshots[0]['path']).absolute()
        print('Hashing full local model files', flush=True)
        write(out / 'model_receipt.json', model_receipt(snapshot, snapshots[0]))
        write(out / 'preflight_request.json', dict(source=str(source), snapshot=str(snapshot), versions=versions,
                                                  output=str(out / 'preflight.json')))
        run('preflight', [py, Path(__file__).resolve(), '--worker', out / 'preflight_request.json'], ROOT)
        result['preflight'] = json.loads((out / 'preflight.json').read_text())
        source_validation(source)
        result['native_performance_attempted'] = True
        run('official_native', [py, source / 'bench_codec_throughput.py', '--device', 'cuda:0',
                               '--model', snapshot, '--local-files-only', '--output', out / 'splitzip_native.json'], source)
        result['native'] = json.loads((out / 'splitzip_native.json').read_text())
        result['validation'] = native_result_validation(result['native'], snapshot)
        result['source_after'] = source_validation(source)
        result.update(status='completed', gpu_measured=True)
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        (out / 'failure.log').write_text(traceback.format_exc())
        print(result['error'], file=sys.stderr, flush=True)
    finally:
        result['runner_sha256'] = sha(__file__)
        write(out / 'results.json', result)
        summary(result, out)
        print('Report: ' + str(out / 'summary.md'), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
