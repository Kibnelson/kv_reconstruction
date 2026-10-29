"""How many bits a learned transform of the prediction error could save, before building it.

For the one-step predictor (layer l-1 of the same token, the GPU codec's), the prediction error of every layer is
measured in the de-rotated domain where the predictor works. Its second-moment matrix per head (128 channels; 8 key
heads and 8 value heads per layer) and per layer half (all 1,024 keys, all 1,024 values) is estimated on the
calibration documents; the transform is the eigenbasis of that matrix (a KLT / PCA, learned from our data). The
gain is then measured out of sample, on the test documents' errors projected on the calibration basis.

Gaussian rate-distortion estimates (they rank the options; real codes sit about 0.25 bits per value above them):

- "now": our current shape, the step proportional to each channel's spread, so every channel has the same
  signal-to-error ratio: at R bits per value, channel error d_i 2^(-2R).
- the same total error with the bits moved between channels (reverse water-filling: error min(theta, d_i) per
  channel, rate 1/2 log2(d_i / theta) where d_i > theta), within each head or over all lanes of all layers;
- the same within each head or each layer half after the learned transform (eigen-variances in place of d_i).

The high-rate gain of the transform alone, 1/2 mean log2(d_i) - 1/2 mean log2(v_i), is reported too.
"""
from __future__ import annotations

import math

import torch

from .predict import CONFIGS


def residual(study, F, doc, layer):
    """De-rotated prediction error [T, W] (float64) of the fit F on a document's layer."""
    Xe = study.extra(doc, layer, F.kinds)
    cur = study.feats(doc, layer)
    md_perm = torch.cat([Xe, cur[:, F.perm]], 1) @ F.A.T + F.b
    md = torch.empty_like(md_perm)
    md[:, F.perm] = md_perm
    return cur.double() - md.double()


def second_moment(rows, groups, size):
    """Per group of `size` consecutive channels: E[r r^T] [G, size, size] (float64)."""
    n = rows.shape[0]
    g = rows.reshape(n, groups, size).transpose(0, 1)
    return torch.bmm(g.transpose(1, 2), g) / n


def water_rate(lam, d_total, iters=100):
    """Bits per value of reverse water-filling over variances lam that reaches total error d_total."""
    lam = lam.clamp_min(1e-30)
    if d_total >= float(lam.sum()):
        return 0.0
    lo, hi = 0.0, float(lam.max())
    for _ in range(iters):
        th = 0.5 * (lo + hi)
        if float(torch.minimum(lam, torch.tensor(th, dtype=lam.dtype, device=lam.device)).sum()) > d_total:
            hi = th
        else:
            lo = th
    th = max(0.5 * (lo + hi), 1e-300)
    return float(0.5 * torch.log2((lam / th).clamp_min(1.0)).sum()) / lam.numel()


def measure(study, rates=(2.0, 3.0), log=print):
    """Per layer and in total: variances per channel and per transform coefficient, out of sample, and the rate
    estimates at the total error our current shape has at each rate in `rates`."""
    F_cfg = CONFIGS["prev"]
    hd, C, W = study.hd, study.C, study.W
    per_layer, lanes = [], {"channel": [], "klt_head": [], "klt_half": []}
    for layer in range(study.L):
        F = study.fit(layer, F_cfg)
        cal_h = cal_q = None
        n = 0
        for d in study.train:
            r = residual(study, F, d, layer)
            m_h, m_q = second_moment(r, W // hd, hd), second_moment(r, 2, C)
            cal_h = m_h * r.shape[0] if cal_h is None else cal_h + m_h * r.shape[0]
            cal_q = m_q * r.shape[0] if cal_q is None else cal_q + m_q * r.shape[0]
            n += r.shape[0]
        cal_h, cal_q = cal_h / n, cal_q / n
        rt = torch.cat([residual(study, F, d, layer) for d in study.tests])
        tst_h, tst_q = second_moment(rt, W // hd, hd), second_moment(rt, 2, C)
        d_ch = torch.diagonal(tst_h, dim1=1, dim2=2).reshape(-1)                  # [W] per channel
        Uh = torch.linalg.eigh(cal_h)[1]
        Uq = torch.linalg.eigh(cal_q)[1]
        v_h = torch.diagonal(Uh.transpose(1, 2) @ tst_h @ Uh, dim1=1, dim2=2).reshape(-1)
        v_q = torch.diagonal(Uq.transpose(1, 2) @ tst_q @ Uq, dim1=1, dim2=2).reshape(-1)
        lg = lambda x: torch.log2(x.clamp_min(1e-30))
        row = {"layer": layer,
               "klt_head_high_rate_gain": float(0.5 * (lg(d_ch).mean() - lg(v_h).mean())),
               "klt_half_high_rate_gain": float(0.5 * (lg(d_ch).mean() - lg(v_q).mean())),
               "error_rms": float(d_ch.mean().sqrt())}
        for R in rates:
            heads = d_ch.reshape(-1, hd)
            vh = v_h.reshape(-1, hd)
            dt = heads.sum(1) * 2.0 ** (-2 * R)
            row[f"R{R:g}"] = {
                "now": R,
                "water_within_head": sum(water_rate(heads[g], float(dt[g])) for g in range(heads.shape[0])) / heads.shape[0],
                "klt_head_water_within_head": sum(water_rate(vh[g], float(dt[g])) for g in range(heads.shape[0])) / heads.shape[0],
                "water_within_layer": water_rate(d_ch, float(dt.sum())),
                "klt_head_water_within_layer": water_rate(v_h, float(dt.sum())),
                "klt_half_water_within_layer": water_rate(v_q, float(dt.sum())),
            }
        per_layer.append(row)
        lanes["channel"].append(d_ch.cpu())
        lanes["klt_head"].append(v_h.cpu())
        lanes["klt_half"].append(v_q.cpu())
        if log:
            log(f"    transform gain, layer {layer}: per-head KLT {row['klt_head_high_rate_gain']:+.3f} bits, "
                f"per-half {row['klt_half_high_rate_gain']:+.3f} (high rate); at 2 bits now: within-head water "
                f"{row['R2']['water_within_head']:.3f}, + KLT {row['R2']['klt_head_water_within_head']:.3f}")
        study.forget(layer + 1)
    total = {}
    d_all = torch.cat(lanes["channel"])
    for R in rates:
        dt = float(d_all.sum()) * 2.0 ** (-2 * R)
        total[f"R{R:g}"] = {"now": R,
                            "water_all_lanes": water_rate(d_all, dt),
                            "klt_head_water_all_lanes": water_rate(torch.cat(lanes["klt_head"]), dt),
                            "klt_half_water_all_lanes": water_rate(torch.cat(lanes["klt_half"]), dt)}
    for key in ("water_within_head", "klt_head_water_within_head", "water_within_layer",
                "klt_head_water_within_layer", "klt_half_water_within_layer"):
        for R in rates:
            total[f"R{R:g}"][key] = sum(r[f"R{R:g}"][key] for r in per_layer) / len(per_layer)
    total["klt_head_high_rate_gain"] = sum(r["klt_head_high_rate_gain"] for r in per_layer) / len(per_layer)
    total["klt_half_high_rate_gain"] = sum(r["klt_half_high_rate_gain"] for r in per_layer) / len(per_layer)
    return {"predictor": "one-step (layer l-1), the GPU codec's", "rates": list(rates), "layers": per_layer,
            "total": total}
