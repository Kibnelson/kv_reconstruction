"""Reproducible offline SM86 compiler/layout gate; never initializes a GPU."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

BASE=Path(__file__).resolve().parent
os.environ.pop('TRITON_INTERPRET',None)
os.environ['TRITON_CACHE_DIR']=str(Path(tempfile.gettempdir())/'kv_standalone_prefix_compile_cache')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from kv_compare import standalone_prefix_core as core
from kv_compare import standalone_inverse_core as inverse

print('Offline SM86 compiler/layout correctness only; no GPU execution or performance measurement.',flush=True)
print('Triton: '+triton.__version__,flush=True)
target=GPUTarget('cuda',86,32)
options=dict(num_warps=1,num_stages=3,enable_fp_fusion=False)
parameters=inspect.signature(core._prefix_ldec_kernel.fn).parameters
ptr_types=dict(buf_ptr='*i16',start_ptr='*i32',state_ptr='*i32',pos_ptr='*i32',
    out_ptr='*i16',acc_ptr='*i32',xout_ptr='*i8',sa_ptr='*fp32',b_ptr='*fp32',
    sp_ptr='*fp32',inv_ptr='*fp32',isx_ptr='*fp32',cos_ptr='*fp32',sin_ptr='*fp32',
    ssum_in_ptr='*i32',ssum_out_ptr='*i32',smult_ptr='*fp32',cumf_ptr='*i32',
    lut_ptr='*i32',tail_ptr='*i32',coarse_ptr='*i32',sparse_ptr='*u16')
constants=dict(TB=32,H2=64,LUT_SHIFT=6,LUT_W=1028,REFINE=2,
    SEARCH=True,FEAT=True,LEAN=False,PACKED=True)
signature={name:ptr_types.get(name,'fp32' if name=='PRIOR_SCALE' else 'i32')
    for name in parameters if name not in set(constants)|{'FIRST','LAST','ONES','UNIFORM'}}
rows=[]
for uniform in (True,False):
    for label,first,last,ones in [('first',True,False,True),('middle',False,False,False),
                                ('last',False,True,False),('single',True,True,True)]:
        constexprs=dict(constants,FIRST=first,LAST=last,ONES=ones,UNIFORM=uniform)
        module=triton.compile(ASTSource(core._prefix_ldec_kernel,signature,constexprs=constexprs),
            target=target,options=options)
        proof=core.validate_compiled_prefix(module,expected_calls=2)
        rows.append(dict(variant=('uniform_' if uniform else 'exact_')+label,
            compiled=True,shared_bytes=module.metadata.shared,**proof))
        print(json.dumps(rows[-1],sort_keys=True),flush=True)
for label,kernel,probe_signature,constexprs in (
        ('changing_mask_probe',core._probe_prefix_kernel,
         {'mask_ptr':'*i32','base_ptr':'*i32','output_ptr':'*i32','reference_ptr':'*i32'},{'STEPS':3}),
        ('take_state_probe',core._probe_take_kernel,
         {'x_ptr':'*i32','cum_ptr':'*i32','frequency_ptr':'*i32','active_ptr':'*i32',
          'buffer_ptr':'*i16','output_ptr':'*i32','reference_ptr':'*i32'},{})):
    module=triton.compile(ASTSource(kernel,probe_signature,constexprs=constexprs),target=target,options=options)
    proof=core.validate_compiled_prefix(module,expected_calls=1)
    rows.append(dict(variant=label,compiled=True,shared_bytes=module.metadata.shared,**proof))
    print(json.dumps(rows[-1],sort_keys=True),flush=True)
# PTX instruction counts describe the compiled program, not GPU time.
constexprs=dict(constants,FIRST=False,LAST=False,ONES=False,UNIFORM=True)
reference=triton.compile(ASTSource(inverse._core_ldec_kernel,signature,constexprs=constexprs),
    target=target,options=options)
candidate=triton.compile(ASTSource(core._prefix_ldec_kernel,signature,constexprs=constexprs),
    target=target,options=options)
counts={}
for name,module in [('packed_reference',reference),('warp_prefix',candidate)]:
    ptx=module.asm['ptx']
    counts[name]=dict(warp_shuffle_sites=ptx.count('shfl.sync.'),
        ballot_sites=ptx.count('vote.ballot.sync.b32'),popcount_sites=ptx.count('popc.b32'),
        shared_bytes=module.metadata.shared,ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest())
result=dict(status='completed',gpu_executed=False,gpu_performance_measured=False,
    target='cuda sm86 warp32',triton=triton.__version__,variants=rows,
    uniform_middle_compiled_instruction_counts=counts,
    source_sha256={name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                   for name,module in [('standalone_prefix_core',core),('standalone_inverse_core',inverse)]},
    scope='Generic offline specializations only; exact runtime-specialized production layouts are checked cold on the GPU before launch')
(BASE/'offline_compile.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(counts,sort_keys=True),flush=True)
print('PASS: ten real prefix variants/probes compiled and passed exact lane-layout checks. No GPU was executed.',flush=True)
