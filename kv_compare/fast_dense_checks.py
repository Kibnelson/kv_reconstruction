"""New dense forward pass versus original T; CUDA or explicit CPU interpreter."""
from __future__ import annotations

import argparse
import json
import os


def check(device, all_patterns=False):
    import math
    import torch
    from .fast_dense_encode import FastDenseEncode, CachedFusedDense
    from . import layerwise as lw, fused as fu

    if device.type == "cuda" and (not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET", "0") != "0"):
        raise RuntimeError("actual compiled CUDA is required")
    torch.set_num_threads(1)
    L, T, hd, Hk = (2, 128, 128, 1) if all_patterns else (2, 32, 16, 2)
    C, W = Hk * hd, 2 * Hk * hd
    gen = torch.Generator().manual_seed(923)
    phase = torch.arange(T, dtype=torch.float64)[:, None] * torch.exp(-torch.arange(hd//2, dtype=torch.float64))[None, :]
    k = dict(Aq=torch.randint(-3, 4, (L, W, W), dtype=torch.int8, generator=gen),
        sa=torch.full((L, W), .002), b=torch.randn((L, W), generator=gen)*.03,
        sp=torch.full((L, W), .3), isx=torch.full((L, W), 12.),
        cos=phase.cos().float().contiguous(), sin=phase.sin().float().contiguous())
    k["Aq"][0].zero_()
    k["inv"] = (fu.QS / (2*math.log(2)*k["sp"].double())).float()
    k = {key: value.to(device) for key, value in k.items()}
    if all_patterns:
        words = torch.arange(65536, dtype=torch.int64).to(torch.int16).reshape(L,T,W).to(device)
    else:
        words = (.5*torch.randn((L,T,W),generator=gen)).bfloat16().view(torch.int16).to(device)
        words[0,0,:5] = torch.tensor([0,-32768,32640,32705,-128],dtype=torch.int16,device=device)
    original = lw.LayerwiseOneStep(k,C,hd,tokens_per_program=8)
    fast = FastDenseEncode(k,C,hd,tokens_per_program=8)
    cached = CachedFusedDense(k,C,hd,tokens_per_program=8)
    a, p = original.model(words)
    b, q = fast.model(words)
    if not torch.equal(a[:-1],b[:-1]) or not torch.equal(p,q):
        raise AssertionError("cuBLAS forward changed an encoder feature or prior")
    c, r = cached.model(words)
    if not torch.equal(a,c) or not torch.equal(p,r):
        raise AssertionError("cached original forward changed features or priors")
    segments,cap=original.layout(T)
    bo,bs=original.encode(words,a,p,out=torch.zeros(segments*cap,dtype=torch.int16,device=device))
    for name,codec in [("fast",fast),("cached",cached)]:
        fo,fs=codec.encode(words,out=torch.zeros_like(bo))
        if not torch.equal(bs,fs) or not torch.equal(bo,fo):
            raise AssertionError(name+" dense encoder changed original stream bytes")
        if not torch.equal(original.decode(fo,fs,T),words):
            raise AssertionError("the original decoder failed on the "+name+" encoder's stream")
    return dict(device=str(device),gpu_executed=device.type=="cuda",values=words.numel(),
        all_patterns=all_patterns,features_and_priors_exact=True,original_stream_bytes_exact=True,
        cached_features_and_priors_exact=True,cached_original_stream_bytes_exact=True,
        original_decoder_exact=True,decoder_method_unchanged=FastDenseEncode.decode is lw.LayerwiseOneStep.decode
            and CachedFusedDense.decode is lw.LayerwiseOneStep.decode)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--interpret",action="store_true")
    parser.add_argument("--all-patterns",action="store_true")
    args=parser.parse_args()
    if args.device=="cpu":
        if not args.interpret:
            parser.error("CPU validation requires explicit --interpret")
        os.environ["TRITON_INTERPRET"]="1"
    elif args.interpret or os.environ.get("TRITON_INTERPRET","0")!="0":
        parser.error("unset TRITON_INTERPRET for CUDA validation")
    import torch
    print(json.dumps(check(torch.device(args.device),args.all_patterns),sort_keys=True))


if __name__=="__main__":
    main()
