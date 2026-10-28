#!/usr/bin/env python3
"""Bits of cheaper one-step predictor products on the captured KV (lossless, exact costs like the lossless stage).

The one-step predictor's product (previous layer's 8-bit features times an 8-bit W x W matrix, per layer) is
about half of the measured whole-document decode time. This probe asks what halving it would cost in bits:

  dense int8 x int8         the predictor the GPU codecs use (reference)
  2:4 sparse int8 x int8    per output row, 2 of every 4 inputs kept (largest |a| x std), refitted by least
                            squares: runs on sparse tensor cores at twice the dense rate
  rank r int8               reduced-rank regression A ~ U (U^T A) as two 8-bit products, the intermediate
                            requantized to 8 bits per channel: 2 x W x r multiply-adds per token (half at r = W/4)
  rank r float              the same without quantization (a lower bound)

    python scripts/probe_cheap_products.py --captures runs/llama32_3b_compare/captures --device cuda \\
        --output runs/llama32_3b_compare/cheap_products.json

Every variant is scored on both test documents with the same adaptive spread as the lossless stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN = [f"train{i}" for i in range(6)]
TESTS = ["test_text", "test_code"]


def main(argv=None):
    import torch
    from kv_compare.core import LN2X2, exact_bits, words_to_float
    from kv_compare.predict import Study
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--ranks", default="4,8", help="ranks as fractions W/k: k values")
    p.add_argument("--rows-per-solve", type=int, default=32, help="2:4 refit: output rows solved per batch")
    args = p.parse_args(argv)
    st = Study(args.captures, args.device, TRAIN, "tables", TESTS, args.max_layers, log=lambda m: None)
    W, dev, f64 = st.W, st.device, torch.float64
    fracs = [int(k) for k in args.ranks.split(",") if k]

    def qrow(x, bits=8):
        m = 2 ** (bits - 1) - 1
        s = (x.abs().amax(1, keepdim=True) / m).clamp_min(1e-30)
        return torch.clamp(torch.round(x / s), -m, m) * s

    def qcols(x, amax, bits=8):
        m = 2 ** (bits - 1) - 1
        s = (amax / m).clamp_min(1e-30)
        return torch.clamp(torch.round(x / s), -m, m) * s

    variants = ["dense int8 x int8", "2:4 sparse int8 x int8"] + [f"rank W/{k} int8" for k in fracs] + \
               [f"rank W/{k} float" for k in fracs]
    bits = {v: {d.name: [] for d in st.tests} for v in variants}
    prior = {v: {d.name: torch.ones(d.tokens, dtype=f64, device=dev) for d in st.tests} for v in variants}
    t0 = time.time()
    for l in range(1, st.L):
        mu, S = st.covariance(l, ["P"])
        S.diagonal().add_(st.ridge * float(torch.trace(S)) / S.shape[0])
        Sxx, Syx = S[:W, :W], S[W:, :W]
        mx, my = mu[:W], mu[W:]
        A = torch.linalg.solve(Sxx, Syx.T).T
        # 2:4 sparse refit
        sal = (A.abs() * Sxx.diagonal().sqrt()[None, :]).reshape(W, W // 4, 4)
        keep = torch.zeros_like(sal, dtype=torch.bool).scatter_(-1, sal.argsort(-1, descending=True)[..., :2], True)
        idx = keep.reshape(W, W).nonzero()[:, 1].reshape(W, W // 2)          # kept inputs per row, ascending
        A24 = torch.zeros_like(A)
        for r0 in range(0, W, args.rows_per_solve):
            k = idx[r0:r0 + args.rows_per_solve]                              # [R, W/2]
            M = Sxx[k[:, :, None], k[:, None, :]]                             # [R, W/2, W/2]
            rhs = torch.gather(Syx[r0:r0 + args.rows_per_solve], 1, k)[..., None]
            A24[torch.arange(r0, r0 + k.shape[0], device=dev)[:, None], k] = torch.linalg.solve(M, rhs)[..., 0]
            del M, rhs
        amax_x = torch.stack([st.feats(d, l - 1).abs().amax(0).to(f64) for d in st.train]).amax(0)
        C = A @ Sxx @ A.T
        _, U = torch.linalg.eigh(C)
        del C
        maps = {"dense int8 x int8": ("lin", qrow(A)), "2:4 sparse int8 x int8": ("lin", qrow(A24))}
        for k in fracs:
            Ur = U[:, -(W // k):]
            B = Ur.T @ A                                                     # [r, W]: first product
            maps[f"rank W/{k} float"] = ("lin", Ur @ B)
            Bq, Uq = qrow(B), qrow(Ur)
            amax_z = torch.stack([((qcols(st.feats(d, l - 1).to(f64), amax_x) - mx) @ Bq.T).abs().amax(0)
                                  for d in st.train]).amax(0)
            maps[f"rank W/{k} int8"] = ("rank", (Bq, Uq, amax_z))

        def predict(d, how, M):
            x = qcols(st.feats(d, l - 1).to(f64), amax_x) - mx
            if how == "lin":
                md = x @ M.T + my
            else:
                Bq, Uq, amax_z = M
                md = qcols(x @ Bq.T, amax_z) @ Uq.T + my
            return st.rope.rotate_row(md, st.positions(d.tokens))

        for name, (how, M) in maps.items():
            xt = words_to_float(st.tables.words(l, dev))
            sp = (xt - predict(st.tables, how, M)).abs().mean(0) / LN2X2 + 1e-12
            for d in st.tests:
                w = d.words(l, dev)
                x = words_to_float(w)
                m = predict(d, how, M)
                q = (x - m).abs() / (LN2X2 * sp)
                mult = ((64 * prior[name][d.name][:, None] + 16) / 80).clamp(0.25, 4.0)
                bits[name][d.name].append(float(exact_bits(w, m, sp * mult).mean()))
                prior[name][d.name] = q.mean(1)
        st.forget(l - 1)
        print(f"layer {l} done ({time.time() - t0:.0f} s)", flush=True)
    res = {"layers": st.L, "W": W, "variants": {}}
    base = None
    print("| Product | Multiply-adds per token | Bits per value (text / code) | Mean | vs dense int8 |")
    print("| --- | --- | --- | --- | --- |")
    for v in variants:
        per = {d: sum(b) / len(b) for d, b in bits[v].items()}
        mean = sum(per.values()) / len(per)
        base = mean if base is None else base
        macs = W * W // 2 if v.startswith("2:4") else (2 * W * (W // int(v.split("/")[1].split()[0])) if v.startswith("rank") else W * W)
        res["variants"][v] = {"bits": per, "mean": mean, "delta": mean - base, "macs_per_token": macs, "per_layer": bits[v]}
        print(f"| {v} | {macs:,} | {per['test_text']:.3f} / {per['test_code']:.3f} | {mean:.3f} | {mean - base:+.3f} |")
    print("(layer 0 is predicted by its mean in every variant and is left out)")
    args.output.write_text(json.dumps(res, indent=1))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
