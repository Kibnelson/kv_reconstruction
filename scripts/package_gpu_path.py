#!/usr/bin/env python3
"""Package small GPU-path reports only. Never include T files or raw captures."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


FILES=('results.json','summary.md','highlights.json','source_manifest.json','cuda_tests.log')


def package(source,destination,allow_failed=False):
    source=Path(source).resolve();destination=Path(destination).resolve()
    r=json.loads((source/'results.json').read_text())
    if r.get('format')!='gpu_kv_path_batch_v1':raise ValueError('wrong report format')
    if r.get('status')!='completed' and not allow_failed:raise ValueError('failed/partial report requires --allow-failed')
    if destination.exists() or destination.with_suffix('.zip').exists():raise ValueError('fresh destination required')
    for name in ['results.json','summary.md']:
        if not (source/name).is_file():raise ValueError('missing '+name)
    destination.mkdir(parents=True)
    ledger=[]
    for name in FILES:
        f=source/name
        if not f.is_file():continue
        if f.is_symlink():raise ValueError('report symlinks are not allowed')
        shutil.copyfile(f,destination/name)
        ledger.append(dict(path=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    (destination/'package_manifest.json').write_text(json.dumps(dict(format='gpu_path_small_report_v1',status=r['status'],gpu_measured=r['coverage']['gpu_measured'],files=ledger,
        scope='Fixed report allowlist; original captures, encoded T files and storage directory are excluded.'),indent=2)+'\n')
    with zipfile.ZipFile(destination.with_suffix('.zip'),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(destination.iterdir()):z.write(f,arcname=destination.name+'/'+f.name)
    return destination


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('source',type=Path);a.add_argument('--destination',type=Path,required=True);a.add_argument('--allow-failed',action='store_true');args=a.parse_args()
    print(package(args.source,args.destination,args.allow_failed))
