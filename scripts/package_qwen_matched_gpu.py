#!/usr/bin/env python3
"""Publishable report allowlist; excludes tokens, KV, calibration and T files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

FILES=('results.json','summary.md','source_manifest.json','cuda_tests.log','baseline_setup.json')


def package(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    report=json.loads((source/'results.json').read_text())
    if report.get('format')!='qwen_matched_gpu_report_v1':raise ValueError('wrong report format')
    if destination.exists() or destination.with_suffix('.zip').exists():raise ValueError('fresh package destination required')
    for name in ('results.json','summary.md'):
        if not (source/name).is_file():raise ValueError('missing '+name)
    destination.mkdir(parents=True);entries=[]
    for name in FILES:
        p=source/name
        if not p.exists():continue
        if p.is_symlink():raise ValueError('symlink in report')
        shutil.copyfile(p,destination/name)
        entries.append(dict(path=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (destination/'package_manifest.json').write_text(json.dumps(dict(format='qwen_matched_small_report_v1',
        status=report['status'],gpu_measured=report['gpu_measured'],files=entries,
        scope='report allowlist only; no raw KV, token IDs, compressed KV files or calibration arrays'),indent=2)+'\n')
    with zipfile.ZipFile(destination.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destination.iterdir()):z.write(p,arcname=destination.name+'/'+p.name)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--destination',required=True)
    args=p.parse_args();print(package(args.source,args.destination))
