#!/usr/bin/env python3
"""Matched frozen lossless T APIs; author batch timing and isolated call timing.

Raw inputs are reconstructed from verified raw-control packets outside timing.
The receiver is a separate process allowed only fixed state and serialized T.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OWN = ('team_dense_uniform', 'team_rank256_uniform')
AUTHOR_BENCH_SHA = '7f1936a13ee3b2bbb04a35b71ca14a88bc407fe9b67e2c128d2ec48df2d018e2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def timing_samples(seconds, raw_bytes):
    if len(seconds) != 5 or any(not math.isfinite(x) or x <= 0 for x in seconds) or type(raw_bytes) is not int or raw_bytes <= 0:
        raise ValueError('five positive finite author-batch samples and positive raw bytes required')
    rates = [raw_bytes / x / 1e9 for x in seconds]
    def stats(values):
        return dict(mean=statistics.mean(values), min=min(values), max=max(values), std=statistics.stdev(values))
    return dict(samples_seconds=list(seconds), seconds=stats(seconds), raw_gbs_samples=rates,
                raw_gbs=stats(rates), wall_mean_ms=1000 * statistics.mean(seconds),
                warmup=10, iters=50, repeats=5,
                scope='Unchanged author bench body; whole API calls between synchronizations; five per-call batch averages.')


def load_author_bench(source, torch):
    path = Path(source)
    if path.name != 'bench_codec_throughput.py':
        path = path / 'bench_codec_throughput.py'
    if sha(path) != AUTHOR_BENCH_SHA:
        raise ValueError('author timing source differs from reviewed SplitZip pin')
    parsed = ast.parse(path.read_text(), filename=str(path))
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in ('synchronize', 'bench')]
    if len(nodes) != 2 or {n.name for n in nodes} != {'synchronize', 'bench'}:
        raise ValueError('author timing definitions are missing/duplicated')
    namespace = dict(torch=torch, time=time)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['bench'], dict(source=str(path), source_sha256=AUTHOR_BENCH_SHA,
        extracted_definitions=[n.name for n in nodes], unchanged_ast=True,
        ast_sha256=hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
        scope='Only author synchronize/bench definitions; not a full author workload or serving reproduction.')


def valid_receiver_request(q):
    if set(q) != {'method', 'tokens', 'codec', 'entries', 'execution', 'output'}:
        raise ValueError('receiver accepts only codec, frozen/encoded packets and expected assertions')
    allowed = {'id', 'document', 'shape', 'values', 'packet', 'packet_sha256', 'packet_bytes',
               'original_word_sha256', 'complete_bpv', 'state_sha256'}
    if len(q['entries']) != 2 or any(not set(e).issubset(allowed) for e in q['entries']):
        raise ValueError('source KV forbidden in receiver')


def runtime_snapshot(torch):
    from scripts.bench_lite_allocation import runtime_snapshot as original
    return dict(original(torch), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        float32_matmul_precision=torch.get_float32_matmul_precision())


def codec_for(q, device):
    if q['method'] not in OWN:
        if q['execution'] != 'native':
            raise ValueError('native baseline execution mode differs')
        from scripts.bench_lite_baselines import make_codec
        return make_codec(q['codec'], device)
    from kv_compare.team_lookup import TeamLookupCodec
    if q['execution'] != ('compact_search' if q['tokens'] == 512 else 'direct_search'):
        raise ValueError('predeclared own execution differs')
    if sha(Path(q['codec']['state']) / 'manifest.json') != q['codec']['manifest_sha256']:
        raise ValueError('fixed predictor state manifest changed')
    codec = TeamLookupCodec(q['codec']['state'], device, q['codec']['arm'],
        decoder_mode='graph' if q['execution'] == 'compact_search' else 'direct', lookup=False)
    if codec.state_sha256 != q['codec']['state_sha256']:
        raise ValueError('fixed predictor tensors differ')
    return codec


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


def receive(q, torch, device):
    from scripts.bench_team_gpu import load_packets
    from scripts.bench_team_packet import check_original
    valid_receiver_request(q); before = runtime_snapshot(torch)
    codec = codec_for(q, device); packets = load_packets(q, device); setup = prepare(q, codec, packets, torch, device)
    checks = {}; held = None
    try:
        for i in (0, 1, 0):
            value = codec.decode(packets[i]); digest = check_original(value, q['entries'][i])
            e = q['entries'][i]
            checks[e['id']] = dict(id=e['id'], verified=True, word_sha256=digest,
                packet_sha256=e['packet_sha256'], shape=list(value.shape))
            if i == 0 and held is None:
                held = value
            elif i == 1:
                check_original(held, q['entries'][0])
        immutable = packet_checks(packets, q['entries']); count = graph_count(q, codec)
    finally:
        close(codec)
    after = runtime_snapshot(torch)
    if before != after:
        raise AssertionError('receiver changed runtime policy')
    write(q['output'], dict(status='completed', gpu_measured=True, no_source_kv_or_predictions=True,
        alternating_packets=True, independent_returned_outputs=True, checks=list(checks.values()),
        packet_immutability=immutable, graph_capture_count=count, graph_setup=setup,
        runtime_policy=before, runtime_policy_after=after))


def _measure(q, torch, device):
    from scripts.bench_team_gpu import load_packets
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


def worker(q, torch, device):
    from scripts.bench_lite_baselines import launch
    out = Path(q['output']); out.mkdir(parents=True, exist_ok=True)
    data = _measure(q, torch, device)
    request = {key: q[key] for key in ('method', 'tokens', 'codec', 'execution')}
    request.update(entries=data.pop('entries'), output=str(out / 'receiver.json'))
    write(out / 'receiver_request.json', request)
    # _measure has returned; no codec, packet, raw KV, tensor or closure survives.
    torch.cuda.synchronize(device); gc.collect(); torch.cuda.empty_cache()
    launch([sys.executable, Path(__file__).resolve(), '--device', str(device), '--receive', out / 'receiver_request.json'], out / 'receiver.log')
    proof = json.loads((out / 'receiver.json').read_text()); checks = {c['id']: c for c in proof['checks']}
    if (not proof.get('gpu_measured') or not proof.get('no_source_kv_or_predictions') or
            not proof.get('alternating_packets') or not proof.get('independent_returned_outputs') or
            proof['runtime_policy'] != data['runtime_policy'] or proof['runtime_policy_after'] != data['runtime_policy'] or
            proof['graph_capture_count'] != data['graph_capture_count'] or set(checks) != {r['id'] for r in data['rows']}):
        raise AssertionError('fresh receiver runtime/ownership/coverage differs')
    for row in data['rows']:
        check = checks[row['id']]
        if not check['verified'] or check['word_sha256'] != row['original_word_sha256'] or check['packet_sha256'] != row['packet_sha256']:
            raise AssertionError('fresh receiver differs from actual encoded packet')
    data.update(status='completed', gpu_measured=True, method=q['method'], tokens=q['tokens'], round=q['round'],
        codec_provenance=q['codec'], device=torch.cuda.get_device_name(device),
        software={n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')},
        receiver_sha256=sha(out / 'receiver.json'),
        scope='Matched framed GPU-resident full encode/decode APIs; actual serialized encoder output restored independently. '
              'Author batch timing is separate from isolated synchronized latency; no disk/NIC/host serialization timed.',
        memory_scope='Torch incremental peaks include API output/scratch; native allocator peaks outside Torch are unmeasured.')
    write(out / 'worker_results.json', data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--worker', '--request', dest='worker', type=Path)
    group.add_argument('--receive', type=Path)
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '16'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
    sys.path.insert(0, str(ROOT))
    from scripts.bench_team_packet import cuda_check
    torch, device = cuda_check(args.device)
    if 'RTX A4500' not in torch.cuda.get_device_name(device) or torch.version.cuda != '12.8':
        raise RuntimeError('the frozen NUC setting requires RTX A4500 / CUDA12.8')
    q = json.loads((args.worker or args.receive).read_text())
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'triton', 'numpy')}
    expected = (dict(torch='2.9.0', triton='3.5.0', numpy='1.26.4') if q['method'] == 'splitzip_official_framed'
                else dict(torch='2.11.0+cu128', triton='3.6.0', numpy='2.3.5'))
    if versions != expected:
        raise ValueError('frozen worker environment changed: ' + str(versions))
    (worker if args.worker else receive)(q, torch, device)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc(); raise SystemExit(1)
