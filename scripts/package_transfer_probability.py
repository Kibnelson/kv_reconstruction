#!/usr/bin/env python3
"""Export small four-arm evidence; keep captures, base state and packets local."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile


def validate_run(source,result):
    """Reconcile saved report with actual packet files before sharing evidence."""
    checks=0
    if result['status']!='completed':return dict(checks=0,scope='failed/incomplete run; no codec pass claimed')
    if set(result['aggregate'])!=set('ABCD'):raise ValueError('missing experiment arm')
    for row in result['test_rows']:
        p=Path(row['packet'])
        if p.is_absolute() or '..' in p.parts:raise ValueError('invalid packet path')
        data=(source/p).read_bytes();s=row['encode']
        if len(data)!=s['packet_bytes'] or hashlib.sha256(data).hexdigest()!=row['packet_sha256']:
            raise ValueError('packet bytes/hash mismatch')
        if len(data)!=s['main_stream_bytes']+s['hint_bytes']+s['framing_bytes']:
            raise ValueError('packet component ledger mismatch')
        if not all(row[k] for k in ('bit_exact','parameter_parity','packet_only_process','hint_parity')):
            raise ValueError('failed exactness or parity check')
        expected=row['cpu_decode']['parameter_sha256']
        if row['packet_only_process_report']['parameter_sha256']!=expected or any(x['parameter_sha256']!=expected for x in row['decode_samples']):
            raise ValueError('causal parameter digest mismatch')
        if row['arm']=='A' and not s['raw_fallback'] and not row['legacy_A_entropy_identical']:
            raise ValueError('baseline A entropy stream changed')
        checks+=6
    for arm,a in result['aggregate'].items():
        rows=[r for r in result['test_rows'] if r['arm']==arm]
        for key in ('packet_bytes','hint_bytes','framing_bytes','main_stream_bytes'):
            if sum(r['encode'][key] for r in rows)!=a[key]:raise ValueError('aggregate byte ledger mismatch')
        if sum(r['values'] for r in rows)!=a['values'] or not math.isclose(a['bits_per_value'],8*a['packet_bytes']/a['values'],rel_tol=1e-12):
            raise ValueError('aggregate rate mismatch')
        checks+=5
    return dict(checks=checks,scope='packet hashes, component/rate arithmetic and recorded parity/independent-process evidence; no new GPU execution or packet re-decode')


def package(source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve();archive=destination.with_suffix('.zip')
    if destination.exists() or archive.exists():raise ValueError('use a fresh destination and ZIP')
    result=json.loads((source/'results.json').read_text())
    if result.get('format')!='block32_probability_arms_v1':raise ValueError('not a four-arm result')
    required=['results.json','summary.md']
    if result['status']=='completed':required+=['insights.json','layer_diagnostics.csv','diagnostics.json.gz','capture_manifest.json','input_ids.json']
    for name in required:
        if not (source/name).is_file():raise ValueError('missing '+name)
    validation=validate_run(source,result)
    destination.mkdir(parents=True)
    for name in required:
        if name=='results.json':
            with gzip.GzipFile(filename=str(destination/'results.json.gz'),mode='wb',mtime=0) as f:f.write((source/name).read_bytes())
        else:shutil.copy2(source/name,destination/name)
    if result['status']=='completed':
        (destination/'calibration').mkdir()
        for p in sorted((source/'calibration').glob('*')):
            if p.name!='base.npz' and p.suffix in ('.npz','.json'):shutil.copy2(p,destination/'calibration'/p.name)
        for p in sorted(source.glob('hints_*.npz')):shutil.copy2(p,destination/p.name)
    files=[]
    for p in sorted(destination.rglob('*')):
        if p.is_file():files.append(dict(path=p.relative_to(destination).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(format='block32_probability_evidence_v1',status=result['status'],files=files,
        validation=validation,
        omitted=['raw KV/hidden captures','baseline base.npz','complete packet files'],
        scope='Small evidence and hint/adapter artifacts only. Retain the full source run for packet replay and independent causal revalidation.')
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(destination.parent))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');p.add_argument('--destination',required=True)
    a=p.parse_args();m=package(a.run,a.destination);print(f"Packaged {len(m['files'])} files: {a.destination}")


if __name__=='__main__':main()
