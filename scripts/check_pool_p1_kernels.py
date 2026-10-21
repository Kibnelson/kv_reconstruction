#!/usr/bin/env python3
"""Compile P1 kernels for SM86 without executing CUDA. Not GPU validation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def compile_kernels():
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from qwen_kv_bench import pool_p1_kernels as k
    from qwen_kv_bench.pool_p1 import Layout
    rows=[]
    for label,g in [('smol_fixture',Layout(30,32,3,64)),('qwen_four_layers',Layout(4,32,8,128)),('partial_tail',Layout(2,7,2,16,4))]:
        c=dict(T=g.tokens,C=g.channels,HEADS2=2*g.heads,D=g.head_dim,TILE=g.tile_tokens,PAGES=g.pages,N=g.sector_words)
        capacity=g.padded_raw_bytes
        items=[(k.count_sectors,{'W':'*u16','R':'*u8','B':'*u8','DESC':'*i64'},c),
          (k.place_offsets,{'DESC':'*i64','PREFIX':'*i64'},dict(SECTORS=g.sectors,BLOCK=256)),
          (k.write_sectors,{'W':'*u16','R':'*u8','B':'*u8','DESC':'*i64','A8':'*u8','A16':'*u16','A32':'*i32'},dict(c,CAPACITY=capacity)),
          (k.unpack_sectors,{'A8':'*u8','A16':'*u16','A32':'*i32','DESC':'*i64','R':'*u8','B':'*u8','IDS':'*i64','OUT':'*u16','STATUS':'*i32'},dict(c,CAPACITY=capacity,SECTORS=g.sectors))]
        for fn,sig,const in items:
            compiled=triton.compile(ASTSource(fn,sig,constexprs=const),target=GPUTarget('cuda',86,32),options={'num_warps':4})
            rows.append(dict(kernel=fn.__name__,case=label,head_dim=g.head_dim,tile_tokens=g.tile_tokens,
                             shared_bytes=compiled.metadata.shared,cubin_bytes=len(compiled.asm['cubin']),
                             cubin_sha256=hashlib.sha256(compiled.asm['cubin']).hexdigest()))
            print(fn.__name__,'SM86',label,'compiled',flush=True)
    return dict(status='compiled',cuda_executed=False,triton=triton.__version__,target='NVIDIA SM86',
                kernels=rows,scope='Offline compilation only; no CUDA execution, timing, or occupancy measurement')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=compile_kernels();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
