#!/usr/bin/env python3
"""Verify a fixed-prior probability calibration run and package small evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bench_lite_baselines import digest


def checked(root, name):
    """Receipts must be ordinary files contained in the local run directory."""
    root = Path(root).resolve()
    raw = root / name
    path = raw.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('missing/outside-run receipt: ' + str(name))
    relative = raw.relative_to(root)
    if any(root.joinpath(*relative.parts[:i]).is_symlink()
           for i in range(1, len(relative.parts) + 1)):
        raise ValueError('symlink receipt: ' + str(name))
    return path


def small_receipts(run):
    """Select evidence only; packets, fitted tensors and input arrays stay local."""
    run = Path(run).resolve()
    names = []
    for path in run.rglob('*'):
        if path.suffix not in ('.json', '.log', '.md') or path.name == 'trace.json':
            continue
        name = str(path.relative_to(run))
        checked(run, name)
        names.append(name)
    return sorted(set(names))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    dest = args.destination.resolve()
    if dest.exists():
        raise ValueError('destination exists')
    if dest.is_relative_to(run):
        raise ValueError('report destination must be outside the source run')
    result = json.loads(checked(run, 'results.json').read_text())
    from scripts.bench_team_probability import verify
    verify(run, result)
    names = small_receipts(run)
    if 'results.json' not in names or 'summary.md' not in names:
        raise ValueError('results and summary are required')
    dest.mkdir(parents=True)
    manifest = dict(format='team_probability_small_report_v1', source_run=str(run),
        status=result['status'], files={},
        excluded='original and recalibrated packet binaries, fitted tensor archives, '
                 'captured KV, input arrays and full profiler traces; '
                 'state manifests and recorded binary hashes are retained')
    for name in names:
        src = checked(run, name)
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        manifest['files'][name] = dict(bytes=target.stat().st_size, sha256=digest(target))
    (dest / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(dest / 'summary.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
