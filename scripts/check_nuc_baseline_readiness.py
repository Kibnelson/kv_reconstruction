#!/usr/bin/env python3
"""Read-only local inventory for official baseline preflights, not a benchmark.

No Torch import, CUDA kernel, installation, download, training or evaluation is
performed. Only a new output directory receives JSON and a Markdown summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'unsloth/Llama-3.2-3B'
REVISION = 'd4446454d87d51aa42e1fb174f25acc5f8762331'
FROZEN_COMMIT = '810b5166b3668e26f0ecf8e60f2a1fa079dfb32b'
FROZEN_REPORT = 'benchmark_results/team_lookup_a4500_20261028_141701_1145261_report_20261028_143153'
PINS = dict(splitzip='42c743118a29db3270ae6a921a4a4eecb20cad04',
            exans='de6c38bcc5438475bd1abf1e9b3d154cef414ef6',
            aqua='e744183f2f3a4a90381f20a6975456871fd1ae29')
AUTHOR_FILES = dict(
    splitzip=('README.md', 'requirements.txt', 'bench_codec_throughput.py', 'codec_gpu.py'),
    exans=('README.md', 'crates/openlake_kv_client/cuda/CMakeLists.txt',
           'crates/openlake_kv_client/cuda/src/expans_codec.cu',
           'crates/openlake_kv_client/cuda/src/expans_kernels.cuh',
           'crates/openlake_kv_client/cuda/tests/expans_codec_test.cu',
           'external/connectors/vllm/openlake_expans.py'),
    aqua=('README.md', 'requirements.txt', 'train_predictors.py', 'evaluate_perplexity.py', 'aquakv/modelutils.py',
          'aquakv/cache_utils.py', 'aquakv/quantizers.py', 'aquakv/datautils.py'))
ENV_PACKAGES = ('torch', 'triton', 'numpy', 'transformers', 'datasets', 'accelerate',
                'nvidia-libnvcomp-cu12', 'nvidia-nvcomp-cu12')
ACTIVATION_LOWER_BOUND = 2 * 256 * 8192 * 3072 * 2


def command(argv, timeout=20):
    """A missing command or failed probe is evidence, not a fatal dependency error."""
    try:
        p = subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                           timeout=timeout, check=False)
        return dict(argv=[str(a) for a in argv], returncode=p.returncode,
                    stdout=p.stdout.strip(), stderr=p.stderr.strip(), status='recorded')
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(argv=[str(a) for a in argv], returncode=None,
                    stdout='', stderr=str(error), status='unavailable')


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def parse_meminfo(text):
    out = {key: None for key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')}
    for line in (text or '').splitlines():
        name, sep, value = line.partition(':')
        if sep and name in out:
            parts = value.split()
            if len(parts) != 2 or parts[1] != 'kB' or int(parts[0]) < 0:
                raise ValueError('invalid Linux memory field: ' + name)
            out[name] = int(parts[0]) * 1024
    return out


def parse_limit(value):
    """None denotes Linux's max/-1/huge sentinel, not zero available bytes."""
    if value is None or value.strip() in ('max', '-1'):
        return None
    n = int(value)
    if n < 0:
        raise ValueError('negative resource limit')
    return None if n >= 1 << 60 else n


def cgroup_directory(mountpoint, mountroot, membership):
    """Translate /proc membership into the mounted hierarchy without traversal."""
    mount = PurePosixPath(mountpoint)
    root = PurePosixPath(mountroot)
    member = PurePosixPath(membership)
    if any('..' in p.parts for p in (mount, root, member)) or not all(p.is_absolute() for p in (mount, root, member)):
        raise ValueError('invalid cgroup path')
    if member == PurePosixPath('/'):
        return str(mount)  # A cgroup namespace can expose its own root as /.
    try:
        rel = member.relative_to(root)
    except ValueError:
        raise ValueError('cgroup membership is outside the mounted hierarchy')
    return str(mount / rel)


def cgroup_locations(membership_text, mountinfo_text):
    memberships = []
    for line in (membership_text or '').splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3:
            memberships.append((parts[1].split(','), parts[2]))
    found = []
    for line in (mountinfo_text or '').splitlines():
        before, sep, after = line.partition(' - ')
        a, b = before.split(), after.split()
        if not sep or len(a) < 5 or len(b) < 3:
            continue
        version = 2 if b[0] == 'cgroup2' else 1 if b[0] == 'cgroup' and 'memory' in b[2].split(',') else None
        if version is None:
            continue
        member = next((m for controls, m in memberships
                       if (version == 2 and controls == ['']) or (version == 1 and 'memory' in controls)), None)
        if member is not None:
            found.append(dict(version=version, mountpoint=a[4], mountroot=a[3],
                              membership=member, directory=cgroup_directory(a[4], a[3], member)))
    return found


def cgroup_inventory(proc=Path('/proc'), filesystem=Path('/')):
    try:
        locations = cgroup_locations(read_text(proc / 'self/cgroup'), read_text(proc / 'self/mountinfo'))
    except ValueError as error:
        return dict(status='unavailable', error=str(error), hierarchies=[], effective_limit_bytes=None,
                    effective_available_bytes=None)
    rows = []
    for place in locations:
        directory, mount = Path(place['directory']), Path(place['mountpoint'])
        names = (('memory.max', 'memory.current', 'memory.swap.max', 'memory.swap.current')
                 if place['version'] == 2 else
                 ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.memsw.limit_in_bytes', 'memory.memsw.usage_in_bytes'))
        levels = []
        for _ in range(64):
            level = dict(directory=str(directory), values={})
            for name in names:
                raw = read_text(filesystem / str(directory).lstrip('/') / name)
                try:
                    value = parse_limit(raw)
                    level['values'][name] = dict(raw=raw, bytes=value,
                        status='missing' if raw is None else 'unlimited' if value is None else 'recorded')
                except ValueError:
                    level['values'][name] = dict(raw=raw, bytes=None, status='invalid')
            levels.append(level)
            if directory == mount:
                break
            directory = directory.parent
        rows.append(dict(place, levels=levels))
    limits, available = [], []
    for row in rows:
        limit_name, current_name = (('memory.max', 'memory.current') if row['version'] == 2 else
                                    ('memory.limit_in_bytes', 'memory.usage_in_bytes'))
        for level in row['levels']:
            limit = level['values'][limit_name]['bytes']
            current = level['values'][current_name]['bytes']
            if limit is not None:
                limits.append(limit)
                if current is not None:
                    available.append(max(0, limit - current))
    return dict(status='recorded' if rows else 'unavailable', hierarchies=rows,
                effective_limit_bytes=min(limits) if limits else None,
                effective_available_bytes=min(available) if available else None)


def file_receipt(path, max_hash_bytes=64 << 20):
    path = Path(path)
    try:
        size = path.stat().st_size
        sha = hashlib.sha256(path.read_bytes()).hexdigest() if size <= max_hash_bytes else None
        return dict(path=str(path), status='recorded', bytes=size, sha256=sha,
                    content_verified=sha is not None)
    except OSError as error:
        return dict(path=str(path), status='missing', error=str(error), content_verified=False)


def author_inventory(root, name):
    path = root / 'runs/_llama_official/sources' / name
    head = command(['git', '-C', path, 'rev-parse', 'HEAD']) if path.is_dir() else dict(returncode=None, stdout='', status='missing')
    staged = command(['git', '-C', path, 'diff', '--cached', '--name-only']) if path.is_dir() else head
    unstaged = command(['git', '-C', path, 'diff', '--name-only']) if path.is_dir() else head
    untracked = command(['git', '-C', path, 'ls-files', '--others', '--exclude-standard']) if path.is_dir() else head
    clean = all(r.get('returncode') == 0 for r in (staged, unstaged)) and not staged.get('stdout') and not unstaged.get('stdout')
    names = untracked.get('stdout', '').splitlines()
    return dict(path=str(path), expected_commit=PINS[name], head=head.get('stdout'), head_probe=head,
        tracked_staged=staged.get('stdout', '').splitlines(), tracked_unstaged=unstaged.get('stdout', '').splitlines(),
        untracked_files=names[:100], untracked_count=len(names),
        untracked_scope='reported separately; untracked files do not imply modified author code',
        pinned_tracked_clean=head.get('returncode') == 0 and head.get('stdout') == PINS[name] and clean,
        files={n:file_receipt(path / n) for n in AUTHOR_FILES[name]})


def environment_inventory(python):
    code = ('import importlib.metadata as m,json,sys\n'
            'out={"python":sys.version.split()[0],"versions":{}}\n'
            'for n in ' + repr(ENV_PACKAGES) + ':\n'
            ' try: out["versions"][n]=m.version(n)\n'
            ' except m.PackageNotFoundError: out["versions"][n]=None\n'
            'print(json.dumps(out))\n')
    probe = command([python, '-c', code])
    try:
        data = json.loads(probe['stdout']) if probe['returncode'] == 0 else {}
    except ValueError:
        data = {}
    return dict(path=str(python), exists=Path(python).exists(), probe=probe,
                metadata=data, torch_imported=False,
                scope='package metadata only; package imports and CUDA execution not tested')


def cache_roots(environ, home):
    values = [environ.get('HF_HUB_CACHE'), environ.get('HUGGINGFACE_HUB_CACHE'), environ.get('TRANSFORMERS_CACHE')]
    if environ.get('HF_HOME'):
        values.append(str(Path(environ['HF_HOME']) / 'hub'))
    values.append(str(Path(environ.get('XDG_CACHE_HOME', str(home / '.cache'))) / 'huggingface/hub'))
    return list(dict.fromkeys(str(Path(p).expanduser()) for p in values if p))


def snapshot_inventory(path):
    path = Path(path)
    files = {n:file_receipt(path / n, 1 << 20) for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json')}
    index = path / 'model.safetensors.index.json'
    weights, error = [], None
    if index.is_file():
        files[index.name] = file_receipt(index, 1 << 20)
        try:
            names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
            for name in names:
                rel = PurePosixPath(name)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('weight index points outside snapshot')
                p = path / name
                weights.append(dict(name=name, exists=p.is_file(), bytes=p.stat().st_size if p.is_file() else None))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            error = str(exc)
    else:
        for name in ('model.safetensors', 'pytorch_model.bin'):
            p = path / name
            if p.is_file():
                weights.append(dict(name=name, exists=True, bytes=p.stat().st_size))
    present = (files['config.json']['status'] == 'recorded' and weights and
               all(w['exists'] and w['bytes'] > 0 for w in weights) and error is None)
    return dict(path=str(path), exists=path.is_dir(), files=files, weights=weights,
        cached_files_present=bool(present), weight_contents_verified=False, error=error,
        scope='exact revision directory and file presence/size only; large model weights are not hashed or loaded')


def model_inventory(root, environ):
    snapshots = [snapshot_inventory(Path(p) / ('models--' + MODEL.replace('/', '--')) / 'snapshots' / REVISION)
                 for p in cache_roots(environ, Path.home())]
    manifests = [root / 'runs/llama32_3b_compare/captures/meta.json',
                 root / 'runs/llama32_3b_compare/model_manifest.json']
    manifests += sorted((root / 'runs').glob('llama32_official*/model_manifest.json'))[:20]
    return dict(model=MODEL, revision=REVISION, snapshots=snapshots,
                cached_files_present=any(s['cached_files_present'] for s in snapshots),
                weight_contents_verified=False, original_manifests=[file_receipt(p, 1 << 20) for p in manifests],
                network_probed=False)


def gpu_inventory():
    probe = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'])
    rows = []
    if probe['returncode'] == 0:
        for fields in csv.reader(probe['stdout'].splitlines()):
            if len(fields) != 4:
                continue
            try:
                rows.append(dict(name=fields[0].strip(), total_vram_mib=int(fields[1]),
                                 free_vram_mib=int(fields[2]), driver=fields[3].strip()))
            except ValueError:
                continue
    pcie = command(['nvidia-smi', '--query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max', '--format=csv,noheader,nounits'])
    return dict(probe=probe, devices=rows, visible=bool(rows), pcie_probe=pcie,
                scope='driver-reported inventory; no CUDA kernel or bandwidth measurement')


def readiness(memory, cgroups, limits, gpu, authors, envs, model, exans):
    caps = [v for v in (memory.get('MemTotal'), cgroups.get('effective_limit_bytes'), limits.get('RLIMIT_AS', {}).get('soft_bytes')) if v is not None]
    avail = [v for v in (memory.get('MemAvailable'), cgroups.get('effective_available_bytes')) if v is not None]
    cap, available = min(caps) if caps else None, min(avail) if avail else None
    def base(name, packages, needs_model=True):
        reasons = []
        if not gpu.get('visible'): reasons.append('no GPU visible through nvidia-smi')
        if not authors[name]['pinned_tracked_clean']: reasons.append('official source pin/clean tracked files not verified')
        versions = envs[name]['metadata'].get('versions', {})
        if any(not versions.get(p) for p in packages): reasons.append('required package metadata unavailable in the baseline environment')
        if needs_model and not model['cached_files_present']: reasons.append('pinned model snapshot files not located locally')
        return reasons
    splitzip = base('splitzip', ('torch', 'triton', 'numpy', 'transformers', 'datasets', 'accelerate'))
    exans_reasons = base('exans', ('torch', 'nvidia-libnvcomp-cu12'), needs_model=False)
    if not exans.get('library_verified'): exans_reasons.append('existing ExANS library/build receipt not verified')
    aqua = base('aqua', ('torch', 'numpy', 'transformers', 'datasets', 'accelerate'))
    if cap is None or available is None: aqua.append('effective host memory capacity/availability not fully known')
    elif cap < ACTIVATION_LOWER_BOUND or available < ACTIVATION_LOWER_BOUND:
        aqua.append('effective host capacity or current availability is below the 24 GiB activation-only lower bound')
    aqua.extend(['original RedPajama calibration provenance is not verified by this inventory',
                 'a matching complete official predictor receipt is not verified by this inventory'])
    def decision(reasons):
        return dict(status='blocked' if reasons else 'eligible_for_verified_preflight', blockers=reasons,
                    performance_measured=False)
    return dict(splitzip=decision(splitzip), exans=decision(exans_reasons), aqua=decision(aqua),
        effective_host_limit_bytes=cap, effective_host_available_bytes=available,
        aqua_activation_lower_bound_bytes=ACTIVATION_LOWER_BOUND,
        aqua_activation_formula='2 * 256 * 8192 * 3072 * 2 bytes = 24 GiB',
        aqua_memory_scope='host activation arrays only; excludes model weights, covariance/regression state, Python, other activations and overhead; exceeding this bound does not guarantee that training fits',
        original_calibration_provenance='unverified locally; historical original-endpoint failures are not a fresh availability probe',
        backup_calibration='explicit provenance variation, not established equivalent to the original rows/order',
        automatic_aqua_retry=False,
        exans_scope='released synthetic codec correctness preflight only; no model needed; serving block geometry and end-to-end vLLM/RDMA reproduction remain unverified; no official performance script is claimed',
        preflight_scope='eligibility is not correctness or performance; model contents, token/data identity, CUDA imports and official entrypoint must be checked next')


def existing_exans(root):
    run = root / 'runs/llama_profile_official_a4500_20261027_194103_2478724'
    path = run / 'exans_build.json'
    receipt = file_receipt(path, 1 << 20)
    out = dict(build_receipt=receipt, library_verified=False)
    try:
        data = json.loads(path.read_text())
        library = file_receipt(data['library'])
        out.update(library=library, expected_library_sha256=data.get('library_sha256'),
                   source_commit=data.get('commit'), prior_author_smoke=data.get('gpu_author_smoke_passed'),
                   compatibility_header_sha256=data.get('compatibility_header_sha256'))
        out['library_verified'] = bool(data.get('commit') == PINS['exans'] and library.get('sha256') and
                                       library['sha256'] == data.get('library_sha256'))
    except (OSError, ValueError, KeyError, TypeError) as error:
        out['error'] = str(error)
    return out


def frozen_sources(root):
    """Check the 84 source receipts recorded by the locked completed report."""
    path = root / FROZEN_REPORT / 'results.json'
    try:
        recipe = json.loads(path.read_text())['recipe']
        expected = dict(recipe['unchanged_code'], **recipe['code'])
        records = {}
        for name, wanted in expected.items():
            source = (root / name).resolve()
            if not source.is_relative_to(root):
                raise ValueError('frozen source is outside repository')
            record = file_receipt(source)
            records[name] = dict(expected_sha256=wanted, sha256=record.get('sha256'),
                                 matches=record.get('sha256') == wanted)
        return dict(status='recorded', count=len(records), expected_count=84,
                    all_match=len(records) == 84 and all(r['matches'] for r in records.values()), files=records)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(status='unavailable', all_match=False, error=str(error))


def collect(root):
    root = Path(root).resolve()
    warnings = []
    try:
        memory = parse_meminfo(read_text('/proc/meminfo'))
    except ValueError as error:
        memory = parse_meminfo(None); warnings.append(str(error))
    cg = cgroup_inventory()
    limits = {}
    for name in ('RLIMIT_MEMLOCK', 'RLIMIT_AS'):
        soft, hard = resource.getrlimit(getattr(resource, name))
        limits[name] = dict(soft_raw=soft, hard_raw=hard,
                            soft_bytes=parse_limit(str(soft)), hard_bytes=parse_limit(str(hard)))
    authors = {name:author_inventory(root, name) for name in PINS}
    envs = {name:environment_inventory(root / ('runs/_llama_official/envs/' + name + '/bin/python'
            if name != 'exans' else '.venv/bin/python')) for name in PINS}
    gpu = gpu_inventory(); model = model_inventory(root, os.environ); exans = existing_exans(root)
    nvcc = shutil.which('nvcc')
    if nvcc is None and os.environ.get('CUDA_HOME'):
        candidate = Path(os.environ['CUDA_HOME']) / 'bin/nvcc'
        if candidate.is_file(): nvcc = str(candidate)
    network = []
    for path in sorted(Path('/sys/class/net').glob('*/speed')):
        raw = read_text(path)
        try: speed = int(raw) if raw is not None and int(raw) > 0 else None
        except ValueError: speed = None
        network.append(dict(interface=path.parent.name, speed_mbps=speed,
                            status='recorded' if speed is not None else 'unavailable'))
    disk = shutil.disk_usage(root)
    result = dict(format='nuc_official_baseline_readiness_v1', status='readiness_report',
        gpu_measured=False, benchmark_performed=False, root=str(root),
        root_git=command(['git', '-C', root, 'rev-parse', 'HEAD']),
        frozen_reference=dict(commit=FROZEN_COMMIT, files={name:file_receipt(root / FROZEN_REPORT / name)
            for name in ('results.json', 'summary.md', 'source_manifest.json', 'package_manifest.json')}),
        frozen_810b516_source_check=frozen_sources(root),
        host_memory_bytes=memory, cgroups=cg, resource_limits=limits,
        host_memory_scope='MemTotal is host RAM; nvidia-smi memory is GPU VRAM; swap is separate and is not treated as guaranteed training capacity',
        gpu=gpu, nvcc=dict(path=nvcc, version=command([nvcc, '--version']) if nvcc else None),
        disk_bytes=dict(total=disk.total, used=disk.used, free=disk.free),
        network_link_inventory=network,
        link_scope='interface speed and PCIe metadata only; 1/100/200/300/400 Gb/s remain calculated link scenarios, not measured NUC transfer rates',
        authors=authors, environments=envs, model=model, exans_build=exans,
        calibration_receipts={name:file_receipt(root / 'runs/llama_profile_official_a4500_20261027_194103_2478724' / name)
            for name in ('aqua_calibration_backup.json', 'aqua_train_official.json')}, warnings=warnings)
    result['next_official_splitzip'] = dict(
        command_template='runs/_llama_official/envs/splitzip/bin/python runs/_llama_official/sources/splitzip/bench_codec_throughput.py --device cuda:0 --model <VERIFIED_ABSOLUTE_CACHED_SNAPSHOT> --local-files-only --output <NEW_OFFICIAL_RUN_DIR>/splitzip_native.json',
        defaults=dict(prompts=4, calibration_values=4194304, test_values=268435456,
                      chunk_size=1024, warmup=10, iterations=50, repeats=5),
        scope='documented official procedure with supported Llama model/hardware variation, not paper Qwen3-32B/H200 numbers; placeholders require verified local receipts and this command is not executed')
    result['readiness'] = readiness(memory, cg, limits, gpu, authors, envs, model, exans)
    return result


def summary(result):
    def gib(value): return 'unknown' if value is None else f'{value / 2**30:.2f} GiB'
    r = result['readiness']; mem = result['host_memory_bytes']
    lines = ['# Official baseline readiness inventory', '',
             '**Readiness report only. No benchmark or GPU performance measurement was run.**', '',
             f'Host RAM: {gib(mem["MemTotal"])} total; {gib(mem["MemAvailable"])} available. '
             f'Swap: {gib(mem["SwapTotal"])} total; {gib(mem["SwapFree"])} free.',
             f'Effective host limit: {gib(r["effective_host_limit_bytes"])}; '
             f'effective currently available: {gib(r["effective_host_available_bytes"])}.', '',
             '| Baseline | Readiness | Blockers |', '|---|---|---|']
    for name in PINS:
        item = r[name]
        lines.append(f'| {name} | {item["status"]} | ' + ('; '.join(item['blockers']) or 'None found by metadata inventory; verified preflight still required') + ' |')
    lines.extend(['',
        'AQUA host activation arrays alone require at least **24 GiB**: '
        '`2 × 256 × 8192 × 3072 × 2` bytes. Model weights, regression/covariance state, '
        'Python and other allocations are additional. Passing this lower bound does not guarantee training fits.', '',
        'Original calibration data and a matching complete predictor receipt remain unverified. '
        'Existing backup calibration is a disclosed provenance variation. This script does not retry AQUA training.', '',
        'The next feasible step is the official SplitZip preflight with the pinned source/environment, '
        'then its documented benchmark entrypoint using a verified local model snapshot. '
        'Bind the discovered snapshot and model/token/data receipts before replacing the placeholders:', '',
        '```text', result['next_official_splitzip']['command_template'], '```', '',
        'Keep the official defaults: four prompts, 4,194,304 calibration values, '
        '268,435,456 TEST values, chunk size1,024, 10 warmups, 50 iterations and five repeats. '
        'Using supported Llama-3.2-3B on the NUC is a disclosed model/hardware variation, '
        'not a reproduction of the paper\'s Qwen3-32B/H200 performance. '
        'See `docs/nuc_baseline_reproduction_plan.md`.', '',
        'ExANS readiness refers only to its released synthetic codec correctness preflight, '
        'which needs no model. Its serving geometry and end-to-end vLLM/RDMA procedure remain unverified.', '',
        'Model cache checks inspect exact revision paths and file presence/sizes; large weights are neither hashed nor loaded. '
        'Environment checks query package metadata without importing Torch. GPU visibility comes from nvidia-smi.', '',
        'PCIe and interface link speeds are inventory, not transfer measurements. '
        'The 1–400 Gb/s comparisons remain calculated scenarios. The A4500 is not a measured 400 Gb/s NUC ingress path.', ''])
    return '\n'.join(lines)


def new_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError('output must be a new directory contained in the repository')
    if output.exists():
        raise FileExistsError('output directory already exists; choose a new run directory')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = new_output(ROOT, args.output)
    result = collect(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'results.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    (output / 'summary.md').write_text(summary(result))
    print(output / 'summary.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
