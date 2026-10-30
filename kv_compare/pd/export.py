"""The codec constants file both ends of a transfer load (scripts/pd_export_codecs.py writes it).

  meta      model, revision, layers L, channels W, key channels C, head dim hd, rank, groups, calibration set
  consts    our one-step predictor (kv_compare.fused.prepare)
  rank      the reduced-rank predictor (kv_compare.layerwise.prepare_rank)
  groups    {2, 4, 16: the head-group predictors (kv_compare.grouped.prepare_groups)}
  cos, sin  RoPE tables of positions 0 .. max_positions - 1 (float32, as kv_compare.context_sweep.rope_consts)
  splitzip  the authors' code book (their calibrate on the WikiText-2 train KV): enc_lut, enc_lut_marked, dec_lut,
            common_lut
  adacode   [(right, nbits) of keys, of values] per layer, the authors' rule on the same KV
"""
from __future__ import annotations

import torch


def _cpu(d):
    return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}


def make_bundle(study, consts, rank, groups, max_positions, splitzip=None, adacode=None, meta=None):
    pos = torch.arange(int(max_positions), device=consts["Aq"].device)
    cos, sin = study.rope.tables(pos)
    b = {
        "meta": dict(meta or {}, L=int(study.L), W=int(study.W), C=int(study.C), hd=int(study.hd),
                     rank=int(rank["rank"]) if rank is not None else None, groups=sorted(int(g) for g in groups),
                     max_positions=int(max_positions)),
        "consts": _cpu(consts),
        "rank": _cpu(rank) if rank is not None else None,
        "groups": {int(g): _cpu(c) for g, c in groups.items()},
        "cos": cos.to(torch.float32).contiguous().cpu(),
        "sin": sin.to(torch.float32).contiguous().cpu(),
    }
    if splitzip is not None:
        b["splitzip"] = {k: getattr(splitzip, k).detach().cpu()
                         for k in ("enc_lut", "enc_lut_marked", "dec_lut", "common_lut")}
    if adacode is not None:
        b["adacode"] = [[(int(r), int(n)) for r, n in layer] for layer in adacode]
    return b
