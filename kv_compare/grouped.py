"""Head-group predictor for the per-layer GPU codec: a layer is decoded in `groups` groups of heads, and each group
is predicted from the previous layer and from the groups of the same layer already decoded.

With 2 groups the keys come first and the values are predicted also from the decoded keys; with 4, four heads at a
time; with 16, head by head. These are the frontier's "natural" block-causal predictors (kv_compare.frontier
gpu_variants): exact lossless costs 8.56 / 8.30 / 8.12 bits per value on PEP 8 against 8.90 for one step.

Constants (prepare_groups), per layer l:
    mu_derot = sa * (Aq . x_q[l-1] + Acq . x_q[l]) + b          (exact int32 sums, then float32)
x_q[l] are the layer's own 8-bit features (de-rotated, the scale sx[l] the next layer uses too), and Acq is block
lower-triangular: the rows of group j only read the channels of groups before j. Both parts share one scale per
output channel, so one int32 sum carries both. Layer 0 has only the within-layer part. The spread is refitted on the
scales document with this predictor (mean absolute residual, as in fused.prepare).

The encoder knows every layer exactly and adds both products at once; the decoder adds the within-layer part group
by group, after decoding the groups before it (kv_compare.layerwise, groups > 1).
"""
from __future__ import annotations

import torch

from .core import LN2X2, words_to_float
from .fused import QS, _feat_ref


def prepare_groups(study, consts, groups, log=print):
    """A copy of the one-step constants (fused.prepare) with the head-group predictor: Aq (previous layer), Acq
    (same layer, block lower-triangular), sa, b, sp, inv; plus "groups". groups: an int, or a list of ints (then a
    dict groups -> constants; each layer's covariance is computed once for all of them)."""
    many = not isinstance(groups, int)
    gl = [int(g) for g in (groups if many else [groups])]
    L, W, hd, C, dev = study.L, study.W, study.hd, study.C, study.device
    G = W // hd
    for g in gl:
        if g < 1 or G % g:
            raise ValueError(f"groups must divide the {G} heads of a layer")
    f32 = torch.float32
    sx = consts["sx"]
    parts = {g: {"Aq": torch.zeros(L, W, W, dtype=torch.int8, device=dev),
                 "Acq": torch.zeros(L, W, W, dtype=torch.int8, device=dev),
                 "sa": torch.ones(L, W, dtype=f32, device=dev), "b": torch.zeros(L, W, dtype=f32, device=dev)}
             for g in gl}
    for l in range(L):
        kinds = ["P"] if l > 0 else []
        mu, S0 = study.covariance(l, kinds)
        for g in gl:
            F = study.solve(mu, [S0.clone()], kinds, g, "natural", keep_mu=True)
            ne = F.ne
            A = F.A.double()                                       # [W, ne + W]; natural order: perm = identity
            fold_c = A[:, ne:] * sx[l].double()[None, :]
            fold_p = A[:, :ne] * sx[l - 1].double()[None, :] if ne else None
            amax = fold_c.abs().amax(1)
            if fold_p is not None:
                amax = torch.maximum(amax, fold_p.abs().amax(1))
            s_row = (amax / 127).clamp_min(1e-30)
            qc = torch.clamp(torch.round(fold_c / s_row[:, None]), -127, 127)
            mu_c = F.mu[ne:].double()
            bias = mu_c - s_row * (qc @ (mu_c / sx[l].double()))
            if fold_p is not None:
                qp = torch.clamp(torch.round(fold_p / s_row[:, None]), -127, 127)
                parts[g]["Aq"][l] = qp.to(torch.int8)
                bias = bias - s_row * (qp @ (F.mu[:ne].double() / sx[l - 1].double()))
            parts[g]["Acq"][l] = qc.to(torch.int8)
            parts[g]["sa"][l] = s_row.to(f32)
            parts[g]["b"][l] = bias.to(f32)
            del F, A
        del S0
        study.forget(l - 1)
    outs = {}
    pos = torch.arange(study.tables.tokens, device=dev)
    for g in gl:
        out = dict(consts, **parts[g], groups=g)
        sp = torch.zeros(L, W, dtype=f32, device=dev)
        prev = None
        for l in range(L):
            x = words_to_float(study.tables.words(l, dev))
            cur = _feat_ref(x, study.rope, pos, consts["isx"][l])
            m = reference_mu_grouped(out, l, prev, cur, pos, C, hd)
            sp[l] = ((x - m.double()).abs().mean(0) / LN2X2 + 1e-12).to(f32)
            prev = cur
        out["sp"] = sp
        out["inv"] = (QS / (LN2X2 * sp.double())).to(f32)
        log(f"  head-group predictor ({g} groups of {G // g} heads) prepared for {L} layers")
        outs[g] = out
    return outs if many else outs[gl[0]]


def reference_mu_grouped(consts, l, prev_feat, cur_feat, positions, C, hd):
    """Torch reference of the head-group prediction for one layer (both products exact integers)."""
    n = positions.numel()
    acc = (cur_feat.double() @ consts["Acq"][l].double().T)
    if l > 0 and prev_feat is not None:
        acc = acc + prev_feat.double() @ consts["Aq"][l].double().T
    acc = acc.to(torch.float32)
    mud = acc * consts["sa"][l][None, :] + consts["b"][l][None, :]
    h2 = hd // 2
    k = mud[:, :C].reshape(n, -1, hd)
    a, bb = k[..., :h2], k[..., h2:]
    cs, sn = consts["cos"][positions][:, None, :], consts["sin"][positions][:, None, :]
    rot = torch.cat([a * cs - bb * sn, bb * cs + a * sn], -1).reshape(n, C)
    return torch.cat([rot, mud[:, C:]], 1)
