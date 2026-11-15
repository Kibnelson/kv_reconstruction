#!/usr/bin/env python3
"""Exact peer-copy diagnosis through CUDA runtime APIs, without importing torch.

Uses the CUDA 12 runtime distributed with the pinned codec environment. No
codec, model, timing claim, system-setting change or host-route fallback.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


ROUTE = 'cudart_peer'


def runtime_library():
    candidates = set()
    for package in ('nvidia-cuda-runtime-cu12', 'torch'):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        for file in distribution.files or ():
            if re.search(r'(^|/)libcudart\.so\.12(?:\.\d+)*$', str(file)):
                path = Path(distribution.locate_file(file)).resolve()
                if path.is_file():
                    candidates.add(path)
    if len(candidates) != 1:
        raise RuntimeError(f'expected one bundled CUDA 12 runtime, found {sorted(map(str, candidates))}')
    return candidates.pop()


class Runtime:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        integer, pointer, size = ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
        signatures = {
            'cudaGetDeviceCount': [ctypes.POINTER(integer)],
            'cudaRuntimeGetVersion': [ctypes.POINTER(integer)],
            'cudaDriverGetVersion': [ctypes.POINTER(integer)],
            'cudaSetDevice': [integer], 'cudaDeviceSynchronize': [],
            'cudaMalloc': [ctypes.POINTER(pointer), size], 'cudaFree': [pointer],
            'cudaMemcpy': [pointer, pointer, size, integer],
            'cudaMemcpyPeer': [pointer, integer, pointer, integer, size],
            'cudaDeviceCanAccessPeer': [ctypes.POINTER(integer), integer, integer],
            'cudaDeviceEnablePeerAccess': [integer, ctypes.c_uint],
            'cudaGetLastError': [],
            'cudaDeviceGetPCIBusId': [ctypes.c_char_p, integer, integer],
        }
        for name, arguments in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = arguments, integer
        self.lib.cudaGetErrorString.argtypes = [integer]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            error = self.lib.cudaGetErrorString(status)
            raise RuntimeError(f'{name}: CUDA {status}: {error.decode() if error else "unknown error"}')

    def number(self, name):
        value = ctypes.c_int()
        self.check(name, ctypes.byref(value))
        return value.value

    def pci_bus(self, device):
        output = ctypes.create_string_buffer(32)
        self.check('cudaDeviceGetPCIBusId', output, len(output), device)
        return output.value.decode()

    def select(self, device):
        self.check('cudaSetDevice', device)

    def sync(self, device):
        self.select(device)
        self.check('cudaDeviceSynchronize')

    def allocate(self, device, nbytes):
        self.select(device)
        value = ctypes.c_void_p()
        self.check('cudaMalloc', ctypes.byref(value), nbytes)
        return value.value

    def free(self, device, pointer):
        self.select(device)
        self.check('cudaFree', pointer)

    def upload(self, device, pointer, data):
        self.select(device)
        self.check('cudaMemcpy', pointer, data.ctypes.data, data.nbytes, 1)

    def download(self, device, pointer, nbytes):
        self.select(device)
        data = np.empty(nbytes, dtype=np.uint8)
        self.check('cudaMemcpy', data.ctypes.data, pointer, nbytes, 2)
        return data

    def local_copy(self, device, destination, source, nbytes):
        self.select(device)
        self.check('cudaMemcpy', destination, source, nbytes, 3)

    def peer_supported(self, source, destination):
        value = ctypes.c_int()
        self.check('cudaDeviceCanAccessPeer', ctypes.byref(value), source, destination)
        return bool(value.value)

    def enable_peer(self, source, destination):
        self.select(source)
        status = self.lib.cudaDeviceEnablePeerAccess(destination, 0)
        if status == 704:  # cudaErrorPeerAccessAlreadyEnabled; clear its thread error.
            self.lib.cudaGetLastError()
            return 'already_enabled'
        if status:
            error = self.lib.cudaGetErrorString(status)
            raise RuntimeError(f'cudaDeviceEnablePeerAccess: CUDA {status}: {error.decode()}')
        return 'enabled'

    def peer_copy(self, source_device, source, destination_device, destination, nbytes):
        self.select(destination_device)
        self.check('cudaMemcpyPeer', destination, destination_device, source, source_device, nbytes)


def comparison(expected, received):
    if (expected.dtype != np.uint8 or received.dtype != np.uint8 or expected.ndim != 1
            or expected.shape != received.shape):
        raise ValueError('full one-dimensional uint8 buffers of the same shape are required')
    different = received != expected
    count = int(np.count_nonzero(different))
    first = int(different.argmax()) if count else None
    return {'correct': count == 0, 'bytes': expected.nbytes, 'mismatched_bytes': count,
            'first_mismatch': first,
            'first_expected': int(expected[first]) if count else None,
            'first_received': int(received[first]) if count else None,
            'expected_sha256': hashlib.sha256(expected).hexdigest(),
            'received_sha256': hashlib.sha256(received).hexdigest(),
            'untouched_poison_bytes': int(np.count_nonzero(received == np.bitwise_not(expected)))}


def probe(runtime, source, destination, nbytes, reps, record):
    allocations, buffers = [], {}
    record.update(host_controls=[], peer_enable=[], trials=[], requested_reps=reps, completed_reps=0)
    row = {'src': source, 'dst': destination, 'correct': False, ROUTE: {'correct': False}}
    record['pairs'] = [row]
    try:
        for device in (source, destination):
            buffers[device] = []
            for _ in range(2):
                pointer = runtime.allocate(device, nbytes)
                allocations.append((device, pointer))
                buffers[device].append(pointer)
        for repetition in range(reps):
            seed = 81073 + 1000 * source + 100 * destination + repetition
            expected = np.random.default_rng(seed).integers(0, 256, nbytes, dtype=np.uint8)
            expected.setflags(write=False)
            poison = np.bitwise_not(expected)
            # Both devices must first pass full H2D, local D2D and D2H controls.
            for device in (source, destination):
                data = expected if device == source else poison
                first, second = buffers[device]
                runtime.upload(device, first, data)
                runtime.local_copy(device, second, first, nbytes)
                runtime.sync(device)
                check = comparison(data, runtime.download(device, second, nbytes))
                record['host_controls'].append(dict(check, gpu=device, repetition=repetition))
                if not check['correct']:
                    raise RuntimeError(f'local CUDA memory/host control failed on GPU {device}')
            if repetition == 0:
                for a, b in ((source, destination), (destination, source)):
                    if not runtime.peer_supported(a, b):
                        raise RuntimeError(f'peer access not advertised for {a}->{b}; no fallback attempted')
                    status = runtime.enable_peer(a, b)
                    record['peer_enable'].append({'src': a, 'dst': b, 'status': status})
            src, dst = buffers[source][0], buffers[destination][1]
            runtime.sync(source)
            runtime.sync(destination)
            print(f'RUNTIME_PEER_BEGIN pair={source}->{destination} repetition={repetition} bytes={nbytes}', flush=True)
            runtime.peer_copy(source, src, destination, dst, nbytes)
            runtime.sync(source)
            runtime.sync(destination)
            parity = comparison(expected, runtime.download(destination, dst, nbytes))
            source_parity = comparison(expected, runtime.download(source, src, nbytes))
            trial = dict(parity, repetition=repetition, seed=seed, source_unchanged=source_parity['correct'])
            record['trials'].append(trial)
            print(f'RUNTIME_PEER_RESULT pair={source}->{destination} mismatched_bytes={parity["mismatched_bytes"]} '
                  f'first_mismatch={parity["first_mismatch"]} source_unchanged={trial["source_unchanged"]}', flush=True)
            if not parity['correct'] or not trial['source_unchanged']:
                raise RuntimeError('CUDA runtime peer copy or source-preservation check failed')
            record['completed_reps'] += 1
        row['correct'] = row[ROUTE]['correct'] = True
    finally:
        for device, pointer in reversed(allocations):
            try:
                runtime.free(device, pointer)
            except Exception as exc:
                record.setdefault('cleanup_errors', []).append(str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mib', type=int, default=64)
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--paths', choices=[ROUTE], default=ROUTE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[0-3]-[0-3]', args.pairs) or min(args.mib, args.reps) < 1:
        parser.error('one directed pair in 0..3 and positive sizes/reps required')
    source, destination = map(int, args.pairs.split('-'))
    if source == destination or os.environ.get('CUDA_LAUNCH_BLOCKING') != '1':
        parser.error('different devices and CUDA_LAUNCH_BLOCKING=1 are required')
    record = {'format': 't_pd_cudart_peer_v1', 'runtime_api_only': True, 'torch_imported': 'torch' in sys.modules,
              'gpu_executed': False, 'diagnostic_only': True, 'paths': [ROUTE], 'mib': args.mib,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'status': 'failed'}
    try:
        if record['torch_imported']:
            raise RuntimeError('the runtime-only probe must run in a fresh process without importing torch')
        record['torch'] = importlib.metadata.version('torch')
        if record['torch'] != '2.11.0+cu128':
            raise RuntimeError('the installed codec package must be torch 2.11.0+cu128')
        library = runtime_library()
        record['runtime_library'] = str(library)
        runtime = Runtime(library)
        version = runtime.number('cudaRuntimeGetVersion')
        record['cuda_runtime'] = f'{version // 1000}.{version % 1000 // 10}'
        record['cuda_driver_api_version'] = runtime.number('cudaDriverGetVersion')
        if version != 12080 or runtime.number('cudaGetDeviceCount') != 4:
            raise RuntimeError('exactly four visible GPUs and CUDA runtime 12.8 required')
        record['gpus'] = [runtime.pci_bus(i) for i in range(4)]
        record['gpu_executed'] = True
        probe(runtime, source, destination, args.mib << 20, args.reps, record)
        if record.get('cleanup_errors'):
            raise RuntimeError('CUDA cleanup reported errors')
        record['status'] = 'passed'
    except Exception as exc:
        record['error'] = str(exc)
        print(f'RUNTIME_PEER_FAILED {type(exc).__name__}: {exc}', flush=True)
    finally:
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
