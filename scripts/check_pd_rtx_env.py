#!/usr/bin/env python3
"""Check the isolated vLLM runtime with a compiled CUDA word-copy kernel."""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check():
    import torch
    import triton
    import triton.language as tl
    import vllm

    if os.environ.get('TRITON_INTERPRET','0') not in ('','0'):
        raise RuntimeError('compiled CUDA is required; unset TRITON_INTERPRET')
    if vllm.__version__ != '0.32.0':
        raise RuntimeError('this stage is pinned to vLLM 0.32.0')
    if not torch.cuda.is_available():
        raise RuntimeError('the separate vLLM runtime cannot access CUDA')

    @triton.jit
    def copy_words(source, destination, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
        x = tl.load(source+i,i<N,other=0)
        tl.store(destination+i,x,i<N)

    device = torch.device('cuda',0)
    original = torch.arange(65536,dtype=torch.int32).to(torch.int16)
    source = original.to(device)
    restored = torch.empty_like(source)
    copy_words[(triton.cdiv(source.numel(),1024),)](source,restored,source.numel(),1024)
    torch.cuda.synchronize(device)
    if not torch.equal(restored.cpu(),original):
        raise AssertionError('compiled CUDA word copy changed one or more BF16 bit patterns')

    from vllm.config import KVTransferConfig
    from kv_compare.pd.connector import TKVConnector
    if inspect.isabstract(TKVConnector):
        raise RuntimeError('TKVConnector does not implement the installed vLLM abstract API')
    config = KVTransferConfig(kv_connector='TKVConnector',
        kv_connector_module_path='kv_compare.pd.connector',kv_role='kv_producer',
        kv_load_failure_policy='fail')
    if config.kv_load_failure_policy != 'fail':
        raise AssertionError('fail-on-transfer-error configuration was not retained')
    p = torch.cuda.get_device_properties(device)
    return dict(vllm=vllm.__version__,torch=torch.__version__,triton=triton.__version__,
        torch_cuda_runtime=torch.version.cuda,visible_GPUs=torch.cuda.device_count(),
        GPU=dict(name=p.name,total_memory_bytes=p.total_memory,compute_capability=[p.major,p.minor]),
        compiled_CUDA_word_copy=True,word_patterns_checked=65536,connector_imported=True,
        connector_abstract_API_satisfied=True,fail_on_transfer_error_configured=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite an existing environment check')
    report = dict(schema='pd_rtx_environment_v1',status='failed',python=platform.python_version(),
        python_executable=sys.executable,real_vllm_engine_started=False,PD_handoff_validated=False,
        T_integration_validated=False,network_performance_validated=False,automatic_merge=False)
    try:
        report['checks'] = check()
        r = subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
        report['source_commit'] = r.stdout.strip() if r.returncode==0 else None
        report['status'] = 'passed'
    except Exception:
        report['error'] = traceback.format_exc()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
