#!/usr/bin/env python3
"""Time the decoders (and encoders) of AQUA-KV and KIVI with the authors' own code, for one document's KV of
Llama-3.2-3B (28 layers, 8,192 tokens, 8 KV heads x 128), on the GPU.

  aqua   (run with the AQUA-KV environment: torch 2.5.1, fast-hadamard-transform) the read path of
         aquakv/cache_utils.py SingleChunkQuantizedCacheWithPredictors, layer by layer: keys = BF16 predictor of the
         previous layer's keys + HiggsQuantizer.dequantize; values = predictor of [keys, previous values] + dequantize;
         RoPE re-applied to the keys (apply_rotary_pos_emb). Layer 0 from the 4-bit grid without a predictor. The
         write path (predict + quantize) as the encoder. Predictor weights are random (the time does not depend on
         them); residuals are Gaussian with the codes HIGGS assigns them.
  kivi   (project environment) quant/new_pack.py: quant_and_pack_kcache / _vcache (encoder) and
         unpack_and_dequant_kcache / _vcache (decoder) with the paper's group size 32, all tokens of every layer.
         KIVI's attention kernels can also read the packed form directly; this times getting BF16 KV back.

    ~/kv_reconstruction/runs/_llama_official/envs/aqua/bin/python scripts/time_baseline_decoders.py aqua \\
        --source ~/kv_reconstruction/runs/_llama_official/sources/aqua --output runs/llama32_3b_compare/time_aqua.json
    python scripts/time_baseline_decoders.py kivi --source <KIVI checkout> --output runs/llama32_3b_compare/time_kivi.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:
        return f"unknown ({exc!r})"


def timed(fn, reps=5, warmup=2):
    import time
    import torch
    for _ in range(warmup):
        fn()
    if not torch.cuda.is_available():                  # CPU smoke test: wall clock
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def aqua(source, L, T, heads, hd, device):
    import torch
    sys.path.insert(0, str(source))
    from aquakv.quantizers import HiggsQuantizer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    C = heads * hd
    g = torch.Generator(device="cpu").manual_seed(0)
    bf = torch.bfloat16
    inv = 1.0 / (500000.0 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    ang = torch.arange(T, dtype=torch.float32)[:, None] * inv[None, :]
    emb = torch.cat([ang, ang], -1)
    cos, sin = emb.cos().to(device, bf)[None], emb.sin().to(device, bf)[None]
    kp = [torch.nn.Linear(C, C, dtype=bf, device=device) for _ in range(L)]
    vp = [torch.nn.Linear(2 * C, C, dtype=bf, device=device) for _ in range(L)]
    res = {"source_commit": commit(source), "torch": torch.__version__, "layers": L, "tokens": T, "channels": C,
           "values": L * T * 2 * C, "bits": {}}
    q0 = HiggsQuantizer(hadamard_groupsize=1024, edenn_d=2, edenn_n=256, channel_size=C)
    for bits, n in ((2, 16), (3, 64), (4, 256)):
        q = HiggsQuantizer(hadamard_groupsize=1024, edenn_d=2, edenn_n=n, channel_size=C)
        resid = [(torch.randn(T, C, generator=g) * 0.1).to(device, bf) for _ in range(2)]
        codes = []
        for l in range(L):                            # codes kept on the GPU (their prototype returns them on the CPU)
            qq = q0 if l == 0 else q
            codes.append(tuple(type(c)(c.idx.to(device), c.scales) for c in (qq.quantize(resid[0]), qq.quantize(resid[1]))))

        def rot(k):                                   # [T, C] -> heads, RoPE, back
            kh = k.view(1, T, heads, hd).transpose(1, 2)
            _, kr = apply_rotary_pos_emb(kh[..., :1, :], kh, cos, sin)
            return kr

        @torch.no_grad()
        def decode():
            pk = pv = None
            for l in range(L):
                qq = q0 if l == 0 else q
                dk = qq.dequantize(codes[l][0]).to(bf)
                dv = qq.dequantize(codes[l][1]).to(bf)
                k = dk if l == 0 else kp[l](pk) + dk
                v = dv if l == 0 else vp[l](torch.cat([k, pv], -1)) + dv
                rot(k)
                pk, pv = k, v

        x = [(torch.randn(T, C, generator=g)).to(device, bf) for _ in range(2)]

        @torch.no_grad()
        def encode():
            pk = pv = None
            for l in range(L):
                qq = q0 if l == 0 else q
                pr = torch.zeros_like(x[0]) if l == 0 else kp[l](pk)
                ck = qq.quantize(x[0] - pr)
                k = pr + qq.dequantize(ck).to(bf)
                pr = torch.zeros_like(x[1]) if l == 0 else vp[l](torch.cat([k, pv], -1))
                cv = qq.quantize(x[1] - pr)
                v = pr + qq.dequantize(cv).to(bf)
                pk, pv = k, v

        td = timed(decode)
        te = timed(encode, reps=3, warmup=1)
        res["bits"][str(bits)] = {"decode_ms": 1e3 * td, "encode_ms": 1e3 * te,
                                  "decode_gvalues_per_s": res["values"] / td / 1e9}
        print(f"AQUA-KV {bits}-bit: decode {1e3 * td:.1f} ms, encode {1e3 * te:.1f} ms", flush=True)
        del codes, resid
    return res


def kivi(source, L, T, heads, hd, device):
    import torch
    sys.path.insert(0, str(source))
    from quant.new_pack import quant_and_pack_kcache, quant_and_pack_vcache, unpack_and_dequant_kcache, \
        unpack_and_dequant_vcache
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {"source_commit": commit(source), "torch": torch.__version__, "layers": L, "tokens": T,
           "channels": heads * hd, "values": L * T * 2 * heads * hd, "group": 32, "bits": {}}
    k = (torch.randn(1, heads, T, hd, generator=g)).to(device, torch.bfloat16)
    v = (torch.randn(1, heads, T, hd, generator=g)).to(device, torch.bfloat16)
    for bits in (2, 4):
        ck = quant_and_pack_kcache(k.clone(), 32, bits)
        cv = quant_and_pack_vcache(v.clone(), 32, bits)

        def decode():
            for _ in range(L):
                unpack_and_dequant_kcache(*ck, 32, bits)
                unpack_and_dequant_vcache(*cv, 32, bits)

        def encode():
            for _ in range(L):
                quant_and_pack_kcache(k.clone(), 32, bits)
                quant_and_pack_vcache(v.clone(), 32, bits)

        td = timed(decode)
        te = timed(encode, reps=3, warmup=1)
        res["bits"][str(bits)] = {"decode_ms": 1e3 * td, "encode_ms": 1e3 * te,
                                  "decode_gvalues_per_s": res["values"] / td / 1e9}
        print(f"KIVI {bits}-bit: decode {1e3 * td:.1f} ms, encode {1e3 * te:.1f} ms", flush=True)
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=["aqua", "kivi"])
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=28)
    p.add_argument("--tokens", type=int, default=8192)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    fn = aqua if a.method == "aqua" else kivi
    res = fn(a.source.resolve(), a.layers, a.tokens, a.kv_heads, a.head_dim, a.device)
    a.output.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
