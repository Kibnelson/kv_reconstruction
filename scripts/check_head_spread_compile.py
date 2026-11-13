"""Offline CUDA compilation checks; no device execution or speed claims."""
import argparse
import hashlib,json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from kv_compare.head_spread_kernels import _head_prior_kernel,_head_ldec_kernel,_head_lenc_kernel
from kv_compare.head_spread_wire import _copy_tails
from kv_compare import gpu_coder as gc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arch', type=int, default=86)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if os.environ.get('TRITON_INTERPRET', '0') != '0':
    parser.error('unset TRITON_INTERPRET for native compilation')
rows=[]
def compile_one(fn, constants, types, label, warps=1):
    signature={name:('constexpr' if name in constants else types.get(name,'i32')) for name in fn.arg_names}
    kernel=triton.compile(ASTSource(fn,signature,constants),target=GPUTarget('cuda',args.arch,32),options={'num_warps':warps,'enable_fp_fusion':False})
    rows.append(dict(name=fn.__name__,label=label,cubin_bytes=len(kernel.asm['cubin']),cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),shared_bytes=kernel.metadata.shared))
    print(label,'compiled',flush=True)

for kind in (0,1,2):
    c=dict(KIND=kind,ALPHA=0.5 if kind else 0.0,SHIFT=0,BLOCK=128)
    compile_one(_head_prior_kernel,c,dict(ssum='*i32',prior='*fp32',mix='*fp32',bias='*fp32',PRIOR_SCALE='fp32'),'prior-kind-'+str(kind))
for kind in (0,1,2):
    c=dict(FIRST=False,LAST=False,TB=128,H2=64,LUT_SHIFT=16-gc.LUT_BITS,LUT_W=gc.LUT_W,REFINE=gc.REFINE,UNIFORM=True,SEARCH=True,FEAT=True,LEAN=False,ONES=False,POLICY_KIND=kind,ALPHA=0.5 if kind else 0.0,HEAD_SHIFT=0)
    types={name:'*fp32' for name in ('sa_ptr','b_ptr','sp_ptr','inv_ptr','isx_ptr','cos_ptr','sin_ptr','smult_ptr','mix_ptr','bias_ptr')}
    types.update({name:'*i32' for name in ('start_ptr','state_ptr','pos_ptr','acc_ptr','ssum_in_ptr','ssum_out_ptr','cumf_ptr','lut_ptr','tail_ptr')})
    types.update(buf_ptr='*i16',out_ptr='*i16',xout_ptr='*i8',PRIOR_SCALE='fp32')
    compile_one(_head_ldec_kernel,c,types,'decoder-kind-'+str(kind))
    if kind in (0,2):
        c.update(FIRST=kind==0,LAST=kind==2,ONES=kind==0)
        compile_one(_head_ldec_kernel,c,types,'decoder-edge-kind-'+str(kind))
for flush in (False,True):
    c=dict(FLUSH=flush,TB=128,H2=64,UNIFORM=True)
    types={name:'*fp32' for name in ('prior_ptr','sa_ptr','b_ptr','sp_ptr','cos_ptr','sin_ptr','smult_ptr')}
    types.update({name:'*i32' for name in ('state_ptr','pos_ptr','start_ptr','acc_ptr','cumf_ptr','ko_ptr')})
    types.update(words_ptr='*i16',out_ptr='*i16')
    compile_one(_head_lenc_kernel,c,types,'encoder-flush-'+str(flush))
for pack in (False,True):
    compile_one(_copy_tails,dict(PACK=pack,BLOCK=1024),dict(source='*i16',destination='*i16',offsets='*i64',bases='*i64',lengths='*i64'),'wire-pack-'+str(pack),warps=4)
report=dict(target='sm_'+str(args.arch),gpu_executed=False,scope='Offline native CUDA compilation only; no kernel executed on a GPU',triton=triton.__version__,rows=rows)
args.output.write_text(json.dumps(report,indent=2)+'\n')
