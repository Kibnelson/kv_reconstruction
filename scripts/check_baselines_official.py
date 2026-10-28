#!/usr/bin/env python3
"""Check our AQUA-KV and KIVI re-implementations (kv_compare/baselines.py) against the authors' own code.

Runs the authors' functions and ours on the same random tensors and records the differences:

  aqua   (run with the AQUA-KV environment: torch 2.5.1, fast-hadamard-transform, optimum-quanto)
         fast_hadamard_transform.hadamard_transform vs our hadamard; aquakv.quantizers.HiggsQuantizer
         quantize_dequantize vs our higgs_qdq for the 2-, 3- and 4-bit grids; aquakv.linear_utils.
         fit_linear_regression vs our _aqua_lsq
  kivi   (run with the project environment) quant/new_pack.py quant_and_pack_kcache + unpack_and_dequant_kcache
         and the value-cache pair vs our _kivi_groups, 2 and 4 bits, group size 32

    ~/kv_reconstruction/runs/_llama_official/envs/aqua/bin/python scripts/check_baselines_official.py aqua \\
        --source ~/kv_reconstruction/runs/_llama_official/sources/aqua --output runs/llama32_3b_compare/check_aqua.json
    python scripts/check_baselines_official.py kivi --source <KIVI checkout> --output runs/llama32_3b_compare/check_kivi.json

The authors' sources are imported from --source (their commit is recorded); nothing is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commit(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:                                   # not a git checkout
        return f"unknown ({exc!r})"


def check_aqua(source, device):
    import torch
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(ROOT))
    from fast_hadamard_transform import hadamard_transform
    from aquakv.quantizers import HiggsQuantizer
    from aquakv.linear_utils import fit_linear_regression
    from kv_compare import baselines as bl
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {"source_commit": commit(source), "torch": torch.__version__}
    x = torch.randn(4096, 1024, generator=g).to(device)
    ref = hadamard_transform(x)
    ours = bl.hadamard(x)
    res["hadamard_max_rel_diff"] = float((ref - ours).abs().max() / ref.abs().max())
    # residual-like data: heavy-tailed, per-token scale varies
    r = (torch.randn(8192, 1024, generator=g) * torch.rand(8192, 1, generator=g) * 0.5).to(device).to(torch.bfloat16)
    res["higgs"] = {}
    for bits, n in bl.AQUA_GRID.items():
        hq = HiggsQuantizer(hadamard_groupsize=1024, edenn_d=2, edenn_n=n, channel_size=1024)
        theirs = hq.quantize_dequantize(r).to(torch.float64)             # their output dtype: the input's (BF16)
        mine = bl.to_bf16(bl.higgs_qdq(r.to(torch.float64), n))
        q = hq.quantize(r)
        err_t = float(((theirs - r.double()).norm() / r.double().norm()))
        err_m = float(((mine - r.double()).norm() / r.double().norm()))
        res["higgs"][f"{bits}-bit"] = {
            "grid_points": n,
            "values_equal_fraction": float((theirs == mine).double().mean()),
            "max_abs_diff_over_rms": float((theirs - mine).abs().max() / r.double().pow(2).mean().sqrt()),
            "relative_error_theirs": err_t, "relative_error_ours": err_m,
            "index_bits_per_value": float(torch.log2(torch.tensor(float(n))) / 2),
            "scale_dtype": str(q.scales.dtype)}
    X = torch.randn(20000, 64, generator=g).to(device)
    Y = X @ torch.randn(64, 32, generator=g).to(device) + 0.1 * torch.randn(20000, 32, generator=g).to(device) + 3.0
    W, b = fit_linear_regression(X, Y, reg_rate=bl.AQUA_DAMP, fit_intercept=True, compute_device=device,
                                 compute_dtype=torch.float32, chunk_size=4096)
    Wo, bo = bl._aqua_lsq([(X.double(), Y.double())])
    res["regression_max_abs_diff_weight"] = float((bl.to_bf16(W.double()) - Wo).abs().max())
    res["regression_max_abs_diff_bias"] = float((bl.to_bf16(b.double()) - bo).abs().max())
    return res


def check_kivi(source, device):
    import torch
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(ROOT))
    from quant.new_pack import quant_and_pack_kcache, quant_and_pack_vcache, unpack_and_dequant_kcache, \
        unpack_and_dequant_vcache
    from kv_compare import baselines as bl
    g = torch.Generator(device="cpu").manual_seed(0)
    res = {"source_commit": commit(source), "torch": torch.__version__, "cases": {}}
    k = (torch.randn(1, 8, 256, 128, generator=g) * (1 + 4 * torch.rand(1, 8, 1, 128, generator=g))).to(device)
    v = torch.randn(1, 8, 256, 128, generator=g).to(device)
    for dt in (torch.bfloat16,):                       # the KV dtype here; ours rounds scales to BF16
        kk, vv = k.to(dt), v.to(dt)
        for bits in (2, 4):
            code, scale, mn = quant_and_pack_kcache(kk.clone(), 32, bits)
            kt = unpack_and_dequant_kcache(code, scale, mn, 32, bits).to(torch.float64)
            # ours: per channel over 32-token groups (tokens on the last axis of each group)
            x = kk.to(torch.float64)[0]                                       # [nh, T, D]
            grp = x.permute(0, 2, 1).reshape(8, 128, 256 // 32, 32)           # [nh, D, groups, 32]
            km = bl._kivi_groups(grp, bits).reshape(8, 128, 256).permute(0, 2, 1)
            code, scale, mn = quant_and_pack_vcache(vv.clone(), 32, bits)
            vt = unpack_and_dequant_vcache(code, scale, mn, 32, bits).to(torch.float64)
            vm = bl._kivi_groups(vv.to(torch.float64)[0].reshape(8, 256, 4, 32), bits).reshape(8, 256, 128)
            rms_k, rms_v = kk.double().pow(2).mean().sqrt(), vv.double().pow(2).mean().sqrt()
            res["cases"][f"{str(dt).split('.')[-1]} {bits}-bit"] = {
                "keys_max_abs_diff_over_rms": float((kt[0] - km).abs().max() / rms_k),
                "keys_relative_error_theirs": float((kt[0] - x).norm() / x.norm()),
                "keys_relative_error_ours": float((km - x).norm() / x.norm()),
                "values_max_abs_diff_over_rms": float((vt[0] - vm).abs().max() / rms_v),
                "values_relative_error_theirs": float((vt[0] - vv.double()[0]).norm() / vv.double()[0].norm()),
                "values_relative_error_ours": float((vm - vv.double()[0]).norm() / vv.double()[0].norm())}
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method", choices=["aqua", "kivi"])
    p.add_argument("--source", type=Path, required=True, help="the authors' checkout")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    res = check_aqua(args.source.resolve(), args.device) if args.method == "aqua" else \
        check_kivi(args.source.resolve(), args.device)
    args.output.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
