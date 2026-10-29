#!/usr/bin/env python3
"""Build and execute the pinned author's synthetic ExANS correctness test.

Uses only the existing local CUDA/nvCOMP installation. No model, download,
installation, throughput benchmark or serving evaluation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PIN = 'de6c38bcc5438475bd1abf1e9b3d154cef414ef6'
INVENTORY = 'benchmark_results/nuc_baseline_readiness_20261028_200201_158793/results.json'
TOOLCHAIN_RECEIPT = 'benchmark_results/llama_profile_official_a4500_20261027_194103_2478724_report_20261027_203958/toolchain.json'
TOOLCHAIN_SHA = '8effe586da002eea6ef9d9f10f543712340e232b171b7049f30964c089152d94'
COMPAT_SHA = 'a1c9a5b2c6621b6f929a481250067e4618f8dd3d46084fa1be1a783dc2a70182'
CUDA_SUBDIR = 'crates/openlake_kv_client/cuda'
AUTHOR_HASHES = {
    'CMakeLists.txt': 'a9344135a11d3ebd37e4e407ffcb0004e870224b3c5e4c926325637cd7ec6410',
    'tests/expans_codec_test.cu': 'f2d2c9c8c4410bcba561bda7d54220e9d67dfe205fcb4152c7a7d42d2d02f6a0',
    'src/expans_codec.cu': '4e43cd11f0ed294b467e90b96597988a9f26a84942e44de9031d769357d24dff',
    'src/expans_kernels.cuh': '7380e32e8cdd5e9958ac58b68085aef0de8a4f08c21b2dc2d529e28aa86f18d0',
}
SCOPE = ('Released author synthetic ExANS correctness/overflow test on RTX A4500. '
         'No codec throughput, Llama workload, NIC, SSD, vLLM serving or resident attention measurement.')
REMOVE_ENV = ('TRITON_INTERPRET', 'PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'LD_AUDIT',
    'CMAKE_GENERATOR', 'CMAKE_GENERATOR_PLATFORM', 'CMAKE_GENERATOR_TOOLSET', 'CMAKE_TOOLCHAIN_FILE',
    'CMAKE_PREFIX_PATH', 'CMAKE_MODULE_PATH', 'CC', 'CXX', 'CUDAHOSTCXX', 'CFLAGS', 'CXXFLAGS',
    'CUDAFLAGS', 'LDFLAGS', 'CPATH', 'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'NVCOMP_ROOT')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def output_path(path, root=ROOT):
    target = Path(path).resolve()
    if target == root.resolve() or not target.is_relative_to(root.resolve()):
        raise ValueError('output must be a new repository-contained directory')
    if target.exists():
        raise ValueError('output already exists; preserve it and use a new directory')
    return target


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def source_receipt(source):
    if git(source, 'rev-parse', 'HEAD') != PIN:
        raise ValueError('ExANS source is not the reviewed pin')
    if git(source, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError('tracked ExANS author sources were changed')
    directory = source / CUDA_SUBDIR
    for name, expected in AUTHOR_HASHES.items():
        if sha(directory / name) != expected:
            raise ValueError('reviewed author source hash changed: ' + name)
    names = git(source, 'ls-files', '--', CUDA_SUBDIR).splitlines()
    names = [name for name in names if Path(name).suffix in ('.cu', '.cuh', '.h', '.hpp')
             or Path(name).name == 'CMakeLists.txt']
    extras = git(source, 'ls-files', '--others', '--exclude-standard', '--', CUDA_SUBDIR).splitlines()
    if any(Path(name).suffix in ('.cu', '.cuh', '.h', '.hpp', '.cmake')
           or Path(name).name == 'CMakeLists.txt' for name in extras):
        raise ValueError('untracked build/source files could affect the author build')
    return dict(commit=PIN, author_sources_unchanged=True,
                files={name: dict(bytes=(source / name).stat().st_size, sha256=sha(source / name))
                       for name in names}, untracked_files=extras)


def runtime_environment(env, home, sdk, build):
    env = dict(env)
    for key in REMOVE_ENV:
        env.pop(key, None)
    paths = [build, home / 'lib64', home / 'lib', sdk / 'lib64', sdk / 'lib']
    # Bind this fresh build and the recorded runtime; do not reuse unrelated LD paths.
    env['LD_LIBRARY_PATH'] = os.pathsep.join(str(p) for p in paths if p.is_dir())
    env.update(CUDA_HOME=str(home), CUDA_PATH=str(home), CUDACXX=str(home / 'bin/nvcc'),
               PATH=str(home / 'bin') + os.pathsep + env.get('PATH', ''),
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    return env


def validate_runtime_links(text, build, home, sdk):
    if 'not found' in text:
        raise ValueError('a required ExANS runtime library was not found')
    libraries = {}
    candidates = {
        'libopenlake_expans.so': [build / 'libopenlake_expans.so'],
        'libcudart.so.12': [home / part / 'libcudart.so.12' for part in ('lib64', 'lib')],
        'libnvcomp.so.5': [sdk / part / 'libnvcomp.so.5' for part in ('lib64', 'lib')],
    }
    for name, choices in candidates.items():
        match = re.search(r'^\s*' + re.escape(name) + r'\s+=>\s+(/\S+)', text, re.MULTILINE)
        if not match:
            raise ValueError('runtime link receipt is missing ' + name)
        path = Path(match.group(1)).resolve()
        # The locally assembled toolkit links to sibling component directories.
        # Bind exact resolved candidates instead of requiring containment in home.
        expected = {candidate.resolve() for candidate in choices if candidate.is_file()}
        if path not in expected:
            raise ValueError('runtime loaded an unexpected library: ' + name)
        libraries[name] = str(path)
    return libraries


def worker(request_path):
    """Actual CUDA/runtime discovery in the already-installed project environment."""
    import importlib.metadata as metadata
    import importlib.util
    request = json.loads(Path(request_path).read_text())
    versions = {name: metadata.version(name) for name in request['versions']}
    if versions != request['versions'] or metadata.version('nvidia-libnvcomp-cu12') != '5.3.0.16':
        raise ValueError('project environment differs from the recorded pinned installation')
    spec = importlib.util.find_spec('nvidia.libnvcomp')
    if not spec or not spec.submodule_search_locations:
        raise ValueError('installed nvCOMP SDK could not be located')
    locations = list(spec.submodule_search_locations)
    if len(locations) != 1:
        raise ValueError('ambiguous installed nvCOMP SDK')
    sdk = Path(locations[0]).resolve()
    if not (sdk / 'include/nvcomp.h').is_file():
        raise ValueError('nvCOMP development headers missing')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('actual CUDA is required; no CPU/interpreter substitution')
    torch.cuda.set_device(0)
    if 'RTX A4500' not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (8, 6):
        raise RuntimeError('this NUC setting requires an RTX A4500 / SM86')
    value = (torch.arange(16, device='cuda:0') + 1).sum().item()
    torch.cuda.synchronize()
    if value != 136:
        raise AssertionError('CUDA runtime probe failed')
    write(request['output'], dict(versions=versions, sdk=str(sdk), cuda=torch.version.cuda,
        gpu=dict(name=torch.cuda.get_device_name(0), capability=[8, 6]),
        cuda_runtime_probe_passed=True, codec_performance_measured=False))


def summary(result, out):
    lines = ['# Official ExANS synthetic correctness preflight', '',
        '**Status: ' + result['status'] + '.** GPU performance measured: **False**.', '', SCOPE, '']
    if result.get('error'):
        lines += ['Stopped: `' + result['error'].replace('`', "'") + '`. Partial receipts and logs are preserved.', '']
    if result.get('gpu_correctness_test_passed'):
        lines += ['The freshly built, unchanged author test exited zero: the synthetic compressible block '
            'round-tripped bit-exactly and the wide-exponent fixture correctly reported overflow.', '']
    lines += ['Author pin: `' + PIN + '`. Build target: `openlake_expans_test`, SM86, '
        'CUDA12.8 and nvCOMP5.3.0.16. The existing compiler-injected unique_ptr error-handler '
        'compatibility overload is disclosed; author files are unchanged.', '',
        'The test uses a 999,424-byte synthetic input and a 770,048-byte fixed record. '
        'It has no model/dataset or timing loop and is silent on success; exit zero is the success evidence.', '',
        'Fresh executable/library hashes, checked source files, runtime links and exact build/test commands '
        'are recorded. No prior test-executable checksum was assumed. Command wall times are provenance, '
        'not codec performance measurements.', '',
        'This does not reproduce the author post\'s H100 throughput or establish compatible Llama/vLLM serving geometry. '
        'Keep matched framed transfer results and lossy AQUA quality results separate.', '']
    (out / 'summary.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.output is None:
        parser.error('--output is required')
    out = output_path(args.output); out.mkdir(parents=True)
    result = dict(format='exans_official_synthetic_preflight_v1', status='failed', scope=SCOPE,
        gpu_measured=False, gpu_performance_measured=False, gpu_correctness_measured=False,
        gpu_correctness_test_passed=False, official_test_attempted=False, commands=[])
    env = dict(os.environ)
    for key in REMOVE_ENV:
        env.pop(key, None)
    result['removed_inherited_environment_keys'] = [key for key in REMOVE_ENV if key in os.environ]
    def run(name, argv, cwd=ROOT):
        record = dict(name=name, argv=list(map(str, argv)), cwd=str(cwd))
        result['commands'].append(record); write(out / 'results.json', result)
        print('Running ' + name, flush=True)
        begin = time.monotonic()
        with (out / (name + '.log')).open('w') as log:
            process = subprocess.run(record['argv'], cwd=cwd, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, check=False)
        record.update(returncode=process.returncode, command_wall_seconds=time.monotonic() - begin)
        if process.returncode:
            raise RuntimeError(name + ' exited ' + str(process.returncode) + '; see ' + name + '.log')
        return (out / (name + '.log')).read_text()
    try:
        inventory_path = ROOT / INVENTORY
        inventory = json.loads(inventory_path.read_text())
        source = ROOT / 'runs/_llama_official/sources/exans'
        py = ROOT / '.venv/bin/python'
        if inventory['format'] != 'nuc_official_baseline_readiness_v1' or Path(inventory['root']).resolve() != ROOT:
            raise ValueError('readiness inventory does not bind this NUC checkout')
        if Path(inventory['authors']['exans']['path']).resolve() != source.resolve() or Path(inventory['environments']['exans']['path']).resolve() != py.resolve():
            raise ValueError('inventory source/environment paths changed')
        result['inventory_sha256'] = sha(inventory_path)
        result['source_before'] = source_receipt(source)
        compat = ROOT / 'native/exans_build_compat.h'
        if sha(compat) != COMPAT_SHA:
            raise ValueError('reviewed compatibility header changed')
        result['compatibility'] = dict(header=str(compat), sha256=COMPAT_SHA,
            scope='Compiler-injected overload forwards allocation-error unique_ptr to the existing raw-pointer handler; author source is unchanged.')
        receipt = ROOT / TOOLCHAIN_RECEIPT
        if sha(receipt) != TOOLCHAIN_SHA:
            raise ValueError('published CUDA toolchain receipt changed')
        toolchain = json.loads(receipt.read_text())
        home = ROOT / 'runs/_matched_baselines/cuda12_8/toolkit'
        if Path(toolchain['cuda_home']).resolve() != home.resolve() or not (home / 'bin/nvcc').is_file():
            raise ValueError('existing cached CUDA toolkit missing; no automatic installation is permitted')
        env.update(CUDA_HOME=str(home), CUDA_PATH=str(home), CUDACXX=str(home / 'bin/nvcc'),
                   PATH=str(home / 'bin') + os.pathsep + env.get('PATH', ''))
        result['toolchain'] = dict(receipt_sha256=TOOLCHAIN_SHA, cuda_home=str(home),
                                  nvcc_sha256=sha(home / 'bin/nvcc'))
        version = run('nvcc_version', [home / 'bin/nvcc', '--version'])
        if 'release 12.8, V12.8.93' not in version:
            raise ValueError('cached CUDA compiler differs from the recorded version')
        versions = {name: value for name, value in inventory['environments']['exans']['metadata']['versions'].items() if value is not None}
        write(out / 'runtime_request.json', dict(versions=versions, output=str(out / 'runtime.json')))
        run('runtime_probe', [py, Path(__file__).resolve(), '--worker', out / 'runtime_request.json'])
        result['runtime'] = runtime = json.loads((out / 'runtime.json').read_text())
        sdk = Path(runtime['sdk']); build = out / 'build'
        env = runtime_environment(env, home, sdk, build)
        run('cmake_version', ['cmake', '--version'])
        run('configure', ['cmake', '-S', source / CUDA_SUBDIR, '-B', build,
            '-DOPENLAKE_EXPANS_BUILD=ON', '-DOPENLAKE_CUDA_CODEC_BUILD_TESTS=ON',
            '-DCMAKE_CUDA_ARCHITECTURES=86', '-DCMAKE_CUDA_COMPILER=' + str(home / 'bin/nvcc'),
            '-DCMAKE_CXX_COMPILER=/usr/bin/c++', '-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/c++',
            '-DCUDAToolkit_ROOT=' + str(home), '-DNVCOMP_ROOT=' + str(sdk),
            '-DCMAKE_CUDA_FLAGS=--pre-include=' + str(compat)])
        run('build', ['cmake', '--build', build, '--target', 'openlake_expans_test', '-j', '2'])
        env = runtime_environment(env, home, sdk, build)
        binary = build / 'openlake_expans_test'; library = build / 'libopenlake_expans.so'
        links = validate_runtime_links(run('runtime_links', ['ldd', binary]), build, home, sdk)
        result['build'] = dict(executable_sha256=sha(binary), library_sha256=sha(library),
            cmake_cache_sha256=sha(build / 'CMakeCache.txt'),
            runtime_libraries={name:dict(path=path, sha256=sha(path)) for name, path in links.items()})
        cache_values = {}
        for line in (build / 'CMakeCache.txt').read_text().splitlines():
            if not line or line.startswith(('#', '//')) or '=' not in line:
                continue
            key, value = line.split('=', 1)
            name = key.split(':', 1)[0]
            if name.startswith(('CMAKE_CUDA_', 'CMAKE_CXX_FLAGS', 'OPENLAKE_')) or name in (
                    'CMAKE_BUILD_TYPE', 'CMAKE_CXX_COMPILER', 'CUDAToolkit_ROOT', 'NVCOMP_ROOT', 'nvcomp_DIR'):
                cache_values[key] = value
        result['build']['cmake_cache_values'] = cache_values
        result['build']['target_flags'] = {
            str(path.relative_to(build)): dict(sha256=sha(path), contents=path.read_text())
            for target in ('openlake_expans_test', 'openlake_expans')
            for path in [build / 'CMakeFiles' / (target + '.dir') / 'flags.make'] if path.is_file()}
        result['official_test_attempted'] = True
        run('author_test', [binary])
        result['source_after'] = source_receipt(source)
        if result['source_before'] != result['source_after'] or sha(compat) != COMPAT_SHA:
            raise ValueError('author/build compatibility sources changed during the test')
        result.update(status='completed', gpu_correctness_measured=True, gpu_correctness_test_passed=True)
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        (out / 'failure.log').write_text(traceback.format_exc())
        print(result['error'], file=sys.stderr, flush=True)
    finally:
        result['runner_sha256'] = sha(__file__)
        write(out / 'results.json', result); summary(result, out)
        print('Report: ' + str(out / 'summary.md'), flush=True)
    return 0 if result['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
