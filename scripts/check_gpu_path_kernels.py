#!/usr/bin/env python3
"""Offline SM86 compilation of the attention/T1 kernels. No CUDA execution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def compile_all():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from qwen_kv_bench.pool_attention_kernels import attention
    from qwen_kv_bench import transfer_t1_kernels as k
    rows=[]
    def check(fn,sig,const,warps=4,label=''):
        c=triton.compile(ASTSource(fn,sig,constexprs=const),target=GPUTarget('cuda',86,32),
            options={'num_warps':warps,'enable_fp_fusion':False})
        rows.append(dict(kernel=fn.__name__,case=label,shared_bytes=c.metadata.shared,
            cubin_bytes=len(c.asm['cubin']),cubin_sha256=hashlib.sha256(c.asm['cubin']).hexdigest()))
        print(fn.__name__,label,'compiled SM86',flush=True)
    for d,h,tile in [(64,3,16),(128,8,16),(16,2,4)]:
        sig=dict(Q='*fp32',WORDS='*u16',RAWTAIL='*u16',A8='*u8',A16='*u16',A32='*i32',DESC='*i64',R='*u8',B='*u8',OUT='*fp32',STATUS='*i32',LSTART='i32')
        for packed in [False,True]:
            for tail in [0,3]:
                c=dict(PACKED=packed,CAPACITY=1000000,T=32,C=2*h*d,HEADS=h,QHEADS=4*h,D=d,TILE=tile,PAGES=(32+tile-1)//tile,N=tile*d,SECTORS=2*h*((32+tile-1)//tile)*2,TAIL=tail,SCALE=d**-.5)
                check(attention,sig,c,label=f'D{d}_packed{packed}_tail{tail}')
    common=dict(MU='*fp64',INV='*fp64',BD='*fp64',FT='*i64')
    for group in [8,32,64]:
        check(k.encode_group,dict(W='*u16',STATE='*i64',COUNT='*i64',BUFFER='*u8',ERROR='*i32',LAYER='i32',CH='i32',**common),dict(T=32,C=384,GROUP=group,CAP=400000),1,f'group{group}')
        check(k.prepare_frequencies,dict(W='*u16',F0='*u16',LAYER='i32',CH='i32',**common),dict(T=32,C=384,GROUP=group,BLOCK=256),4,f'group{group}')
        check(k.encode_group_cached,dict(W='*u16',F0='*u16',STATE='*i64',COUNT='*i64',BUFFER='*u8',ERROR='*i32',LAYER='i32',CH='i32'),dict(T=32,C=384,GROUP=group,CAP=400000),1,f'group{group}')
        for reuse in [False,True]:
            check(k.decode_group,dict(DATA='*u8',DIRECTORY='*i64',STATE='*i64',POSITION='*i64',ERROR='*i32',OUT='*u16',LAYER='i32',CH='i32',SIZE='i64',**common),dict(T=32,C=384,GROUP=group,REUSE_MASS=reuse),1,f'group{group}_reuse{reuse}')
    check(k.compact_lanes,dict(BUFFER='*u8',STATE='*i64',COUNT='*i64',ENDS='*i64',DIRECTORY='*i64',DATA='*u8'),dict(CAP=400000,BLOCK=256))
    check(k.initialize_decoder,dict(DATA='*u8',DIRECTORY='*i64',STATE='*i64',POSITION='*i64',ERROR='*i32',SIZE='i64'),{},1)
    check(k.finish_decoder,dict(STATE='*i64',POSITION='*i64',DIRECTORY='*i64',ERROR='*i32'),{},1)
    check(k.probe_frequency,dict(RANK='*i32',MU='*fp64',INV='*fp64',BD='*fp64',FT='*i64',OUT='*i64'),dict(COUNT=257,PLANE=15,BLOCK=256))
    return dict(status='compiled',cuda_executed=False,target='SM86',triton=triton.__version__,kernels=rows)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    r=compile_all();args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2)+'\n')
