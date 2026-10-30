"""Offline SM86 compiler check; no GPU is initialized or timed."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent
os.environ.pop('TRITON_INTERPRET', None)
os.environ['TRITON_CACHE_DIR'] = str(BASE / 'inverse_compile_cache')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from kv_compare import standalone_inverse_core as core

print('Offline compiler correctness check only; no GPU execution or performance measurement.', flush=True)
print('Command: PYTHONPATH=<existing CPU/Triton validation dependencies>:<repo> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python ' + __file__, flush=True)
print('Triton: ' + triton.__version__, flush=True)
target = GPUTarget('cuda', 86, 32)
print('Explicit target: ' + repr(target), flush=True)
parameters = inspect.signature(core._core_ldec_kernel.fn).parameters
ptr_types = dict(buf_ptr='*i16', start_ptr='*i32', state_ptr='*i32', pos_ptr='*i32',
    out_ptr='*i16', acc_ptr='*i32', xout_ptr='*i8', sa_ptr='*fp32', b_ptr='*fp32',
    sp_ptr='*fp32', inv_ptr='*fp32', isx_ptr='*fp32', cos_ptr='*fp32', sin_ptr='*fp32',
    ssum_in_ptr='*i32', ssum_out_ptr='*i32', smult_ptr='*fp32', cumf_ptr='*i32',
    lut_ptr='*i32', tail_ptr='*i32', coarse_ptr='*i32', sparse_ptr='*u16')
constants = dict(TB=32,H2=64,LUT_SHIFT=6,LUT_W=1028,REFINE=2,
    SEARCH=True,FEAT=True,LEAN=False)
signature = {name:ptr_types.get(name,'fp32' if name == 'PRIOR_SCALE' else 'i32')
    for name in parameters if name not in set(constants) | {'FIRST','LAST','ONES','PACKED','UNIFORM'}}
rows=[]
for packed in (False,True):
    mode_signature = dict(signature, sparse_ptr='*u16' if packed else '*i32')
    variants = [('first',True,False,True),('middle',False,False,True),
                ('last',False,True,True),('exact_middle',False,False,False)]
    for label, first, last, uniform in variants:
        constexprs = dict(constants,FIRST=first,LAST=last,ONES=first,PACKED=packed,UNIFORM=uniform)
        begin=time.perf_counter()
        module = triton.compile(ASTSource(core._core_ldec_kernel,mode_signature,constexprs=constexprs),
            target=target,options=dict(num_warps=1,num_stages=3,enable_fp_fusion=False))
        ptx=module.asm['ptx']
        row=dict(algorithm='packed' if packed else 'masked',variant=label,
            sparse_pointer_type=mode_signature['sparse_ptr'],
            compiled=True,ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
            ptx_bytes=len(ptx.encode()),compiler_wall_seconds=time.perf_counter()-begin,
            shared_bytes=module.metadata.shared,num_warps=module.metadata.num_warps)
        rows.append(row)
        print(json.dumps(row,sort_keys=True),flush=True)
(BASE / 'inverse_local_compile.json').write_text(json.dumps(
    dict(status='completed',gpu_executed=False,gpu_performance_measured=False,
         target='cuda sm86 warp32',triton=triton.__version__,variants=rows),indent=2)+'\n')
print('PASS: compiled all eight real decoder variants. Compile wall times are not GPU timings.',flush=True)
