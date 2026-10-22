#!/usr/bin/env python3
"""Fetch pinned, unmodified SplitZip and optionally install nvCOMP.

Downloaded upstream sources stay under ignored runs/. This does not change
the existing Torch, Triton or Transformers installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from qwen_kv_bench.matched_transfer import SPLITZIP_COMMIT,SPLITZIP_SHA256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--install-nvcomp',action='store_true');args=p.parse_args()
    directory=ROOT/'runs/_matched_baselines/splitzip';directory.mkdir(parents=True,exist_ok=True)
    report=dict(splitzip=dict(commit=SPLITZIP_COMMIT,status='unavailable'),nvcomp=dict(status='not_requested'))
    try:
        for name in ['codec_gpu.py','LICENSE']:
            url=f'https://raw.githubusercontent.com/Intelligent-Microsystems-Lab/SplitZip/{SPLITZIP_COMMIT}/{name}'
            path=directory/name
            data=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=60).read()
            if name=='codec_gpu.py' and hashlib.sha256(data).hexdigest()!=SPLITZIP_SHA256:
                raise ValueError('upstream source hash mismatch')
            path.write_bytes(data)
        report['splitzip'].update(status='ready',source_sha256=SPLITZIP_SHA256)
    except Exception as error:report['splitzip']['reason']=repr(error)
    if args.install_nvcomp:
        # This CUDA-12 release supports Python 3.12 on the A4500 host and
        # does not depend on or replace Torch/Triton.
        r=subprocess.run([sys.executable,'-m','pip','install','--no-deps','nvidia-nvcomp-cu12==4.2.0.14'])
        report['nvcomp']=dict(status='installed' if r.returncode==0 else 'unavailable',version='4.2.0.14',returncode=r.returncode)
    (directory.parent/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    # Optional native baseline failures are recorded; the core experiment
    # still runs and cannot substitute a CPU estimate for an unavailable GPU.


if __name__=='__main__':main()
