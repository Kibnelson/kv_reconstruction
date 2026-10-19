"""Reproducible CPU decode comparison on the frozen learned KV byte format.

Prefer final learned64_sample.bin plus its matching raw capture. If those are
not available, encode the first 16 rows of the earlier real-KV smoke sample;
that fallback is explicitly labeled. No fitting or byte-format changes occur.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from kv_research import container, learned_codec
from kv_research.native_decode import NativeDecoder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--encoded',type=Path)
    parser.add_argument('--raw',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--repetitions',type=int,default=3)
    args=parser.parse_args()
    if args.repetitions<1: parser.error('repetitions must be positive')
    root=Path(__file__).resolve().parent
    project=root.parent
    model_path=args.model or root/'models/learned64.bin'
    encoded_path=args.encoded or root/'results/learned64_sample.bin'
    output=args.output or root/'results/native_benchmark.json'
    model=learned_codec.load_model(model_path)
    if encoded_path.exists():
        raw_path=args.raw or root/'results/sample_raw.npz'
        if not raw_path.exists():
            manifest_path=project/'.kv_research_data/test_manifest.json'
            if not manifest_path.exists():
                raise FileNotFoundError('Final encoded sample exists but matching raw sample is unavailable; provide --raw')
            manifest=json.loads(manifest_path.read_text())
            raw_path=manifest_path.parent/manifest['cases'][0]['file']
        source_label='final frozen learned64 encoded cache and matching raw KV capture'
        with np.load(raw_path) as data:
            keys=sorted(data.files)
            nlayers=sum(k.endswith('_K') for k in keys)
            matrices=[tuple(data[f'layer_{l:02d}_{k}'].copy() for k in ('K','V')) for l in range(nlayers)]
        blob=encoded_path.read_bytes()
        encoded_hash=hashlib.sha256(blob).hexdigest()
    else:
        if args.encoded: raise FileNotFoundError(encoded_path)
        raw_path=args.raw or project/'kv_smoke/results/sample_raw_cache.npz'
        with np.load(raw_path) as data:
            nlayers=sum(k.endswith('_K') for k in data.files)
            matrices=[tuple(data[f'layer_{l:02d}_{k}'][:16].copy() for k in ('K','V')) for l in range(nlayers)]
        encoded=container.encode_cache(matrices,lambda w,l,k:learned_codec.encode_matrix(w,model,f'layer_{l:02d}_{k}'))
        blob=encoded.data
        encoded_hash=hashlib.sha256(blob).hexdigest()
        source_label='fallback: first16 tokens from previous real SmolLM2 KV smoke capture, encoded with unchanged frozen learned64'
        encoded_path=None

    expected_words=sum(w.size for pair in matrices for w in pair)
    raw_bytes=sum(w.nbytes for pair in matrices for w in pair)
    raw_word_hash=hashlib.sha256(b''.join(w.astype('<u2').tobytes() for p in matrices for w in p)).hexdigest()
    shape=container.shape(blob)
    native=NativeDecoder(model)  # Build/load C++ and hash snapshot before timing.
    cached=SimpleNamespace(codebooks=model.codebooks,n_centroids=model.n_centroids,model_id=model.model_id)
    decoders={
        'python_reference':lambda b,l,k:learned_codec.decode_matrix(b,model),
        'python_cached_model_id':lambda b,l,k:learned_codec.decode_matrix(b,cached),
        'native_cpu':lambda b,l,k:native.decode_matrix(b),
    }
    def check(recovered):
        if len(recovered)!=len(matrices): raise AssertionError('decoded layer count differs')
        for expected_pair,actual_pair in zip(matrices,recovered):
            for expected,actual in zip(expected_pair,actual_pair):
                if not np.array_equal(expected,actual): raise AssertionError('native/reference BF16 mismatch')
    # Warm up every complete path and validate all stored BF16 words.
    for decoder in decoders.values(): check(container.decode_cache(blob,decoder))
    timings={name:[] for name in decoders}
    equality_checks={name:1 for name in decoders}
    # Rotate order per pass to reduce systematic order effects. These remain
    # local CPU timings, not controlled GPU or serving measurements.
    names=list(decoders)
    for repetition in range(args.repetitions):
        order=names[repetition%len(names):]+names[:repetition%len(names)]
        for name in order:
            start=time.perf_counter()
            recovered=container.decode_cache(blob,decoders[name])
            elapsed=time.perf_counter()-start
            check(recovered)  # Equality checking excluded from elapsed time.
            equality_checks[name]+=1
            timings[name].append(elapsed)
    measurements={name:{'seconds':times,'median_seconds':statistics.median(times),
        'raw_MiB_per_second':raw_bytes/2**20/statistics.median(times),
        'all_bf16_words_exact':True,'complete_equality_checks':equality_checks[name]} for name,times in timings.items()}
    native_seconds=measurements['native_cpu']['median_seconds']
    compiler=subprocess.run(['g++','--version'],check=True,text=True,capture_output=True).stdout.splitlines()[0]
    result={
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source':source_label,'raw_path':str(raw_path),'encoded_path':str(encoded_path) if encoded_path else None,
        'model_path':str(model_path),'shape':shape,'bf16_words_per_pass':expected_words,
        'raw_payload_bytes':raw_bytes,'encoded_bytes':len(blob),'repetitions':args.repetitions,
        'hashes':{'model_sha256':sha(model_path),'raw_npz_sha256':sha(raw_path),'raw_word_sha256':raw_word_hash,
                  'encoded_sha256':encoded_hash,'native_cpp_sha256':sha(root/'native_decode.cpp'),
                  'native_wrapper_sha256':sha(root/'native_decode.py'),'benchmark_sha256':sha(Path(__file__))},
        'runtime':{'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor(),
                   'logical_cpus':os.cpu_count(),'compiler':compiler,'compile_flags':['-O3','-std=c++17','-shared','-fPIC'],
                   'final_jobs_completed':(root/'results/final.json').exists()},
        'measurements':measurements,
        'native_speedup_vs_python_reference':measurements['python_reference']['median_seconds']/native_seconds,
        'native_speedup_vs_cached_model_id_python':measurements['python_cached_model_id']['median_seconds']/native_seconds,
        'scope':'CPU full-cache reconstruction only, using identical preencoded records. Includes container traversal, Python/C++ calls, output allocation and concatenation. Excludes model build/load/hash, shared-object compile/load, encoding, source read and equality checking. Every full reconstruction is bitwise checked after timing. Cached-model-id proxy isolates repeated serialization/hashing from Python unpack/lookup cost. Output arrays hold the whole cache; no bounded-memory attention, GPU, migration, admission or serving performance is demonstrated. Timings are a local implementation comparison, not a statistically controlled throughput claim.',
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
