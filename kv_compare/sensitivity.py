"""How sensitive the model's loss is to an error in each KV lane, for shaping the lossy codec's step.

A lane is one layer x one channel of the cache row (keys after RoPE, then values, as the captures hold them). Its
sensitivity is the diagonal Fisher information of the cached value: F[l, c] = mean over tokens of
(dLoss / dkv[l, t, c])^2, where Loss is the summed next-token loss of a prefill and the gradient flows through
everything that reads the cache (all later tokens, all later layers). A small error e in that value raises the
expected loss by about F e^2 / 2.

Shaping: for a fixed rate, the weighted error sum_c F_c D_c of entropy-coded uniform quantizers is smallest with
the same F_c step_c^2 on every lane (high-rate theory), so step_c is proportional to F_c^(-1/2). A lane whose step
exceeds its spread then gets almost no bits (reverse water-filling). Our step is kappa_c x spread_c, so
kappa_c = kappa x scale_c with scale_c proportional to 1 / (spread_c sqrt(F_c)), normalized to a geometric mean of 1
(the rate stays about where kappa puts it) and clamped. Either over all lanes ("global": also moves bits between
layers, heads, keys and values) or within each head of each layer ("head": only between the channels of a head).

The first `skip` tokens of each prefill (the attention sinks, which the protocols keep exact) are left out of the
mean.

Measured in AQUA-KV's protocol (8ae8d22), this shaping hurts: at step 2, +10.1% perplexity over all lanes and
+5.9% within heads, against +0.54% unshaped. The gradient of a prefill is a local, first-order measure: it is
dominated by nearby queries, which in the protocol still read exact values (the recent buffer), and the errors
there are not small.

Uniform steps (uniform_scales): the same absolute step for every lane of a group (a head, or all the keys or all
the values of a layer), as a rotation followed by one scale per group gives: the error spread evenly over the
channels instead of in proportion to each channel's spread. For equally important channels this is the
minimum-error allocation (reverse water-filling): channels whose spread is below the step get almost no bits.
"""
from __future__ import annotations

import math

import torch
from transformers.cache_utils import DynamicCache


class _GradCache(DynamicCache):
    """A cache that keeps the tensors attention reads, with their gradients."""

    def __init__(self):
        super().__init__()
        self.kept = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        k, v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        k.retain_grad()
        v.retain_grad()
        self.kept[layer_idx] = (k, v)
        return k, v


def chunks_of(ids, tokens, bos):
    """Consecutive prefills of `tokens` tokens from a token list, each starting with BOS like a request."""
    body = [t for t in ids if t != bos] if bos is not None else list(ids)
    step = tokens - (1 if bos is not None else 0)
    return [([bos] if bos is not None else []) + body[s:s + step] for s in range(0, len(body) - step + 1, step)]


def kv_grads(model, ids, device):
    """Gradient of the summed next-token loss of one prefill with respect to the cached KV: [L, T, W] (keys after
    RoPE then values, heads flattened), and the mean loss per token. Weights get no gradient."""
    params = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in params:
        p.requires_grad_(False)
    try:
        x = torch.tensor([ids], device=device)
        e = model.get_input_embeddings()(x).detach().requires_grad_(True)
        cache = _GradCache()
        with torch.enable_grad():
            out = model(inputs_embeds=e, use_cache=True, past_key_values=cache)
            lab, lg = x[0, 1:], out.logits[0, :-1]
            lg = lg.float() if lg.dtype in (torch.bfloat16, torch.float16) else lg
            loss = sum(torch.nn.functional.cross_entropy(lg[j:j + 512], lab[j:j + 512], reduction="sum")
                       for j in range(0, lab.shape[0], 512))
            del out, lg
            loss.backward()
        g = torch.stack([torch.cat([k.grad[0].transpose(0, 1).reshape(k.shape[2], -1),
                                    v.grad[0].transpose(0, 1).reshape(v.shape[2], -1)], 1)
                         for k, v in (cache.kept[l] for l in range(len(cache.kept)))])
        return g, float(loss.detach()) / (len(ids) - 1)
    finally:
        for p, r in params:
            p.requires_grad_(r)


def fisher(model, prefills, device, skip=4, log=None):
    """Diagonal Fisher [L, W] (float64, on the CPU) of the cached KV over the prefills (lists of token ids), and
    the number of tokens averaged."""
    F, n = None, 0
    for i, ids in enumerate(prefills):
        g, loss = kv_grads(model, ids, device)
        s = g[:, skip:].double().pow(2).sum(1).cpu()
        F = s if F is None else F + s
        n += len(ids) - skip
        if log:
            log(f"    sensitivity: prefill {i + 1}/{len(prefills)} ({len(ids)} tokens), loss {loss:.3f} per token")
        del g
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return F / n, n


def lane_scales(F, spread, mode="global", hd=128, clamp=16.0):
    """kappa multipliers [L, W] (float64): proportional to 1 / (spread sqrt(F)), geometric mean 1 over all lanes
    ("global") or within each head of each layer ("head"), clamped to [1 / clamp, clamp]."""
    F = torch.as_tensor(F, dtype=torch.float64).cpu()
    sp = torch.as_tensor(spread, dtype=torch.float64).cpu()
    if F.shape != sp.shape:
        raise ValueError(f"sensitivity {tuple(F.shape)} and spreads {tuple(sp.shape)} differ")
    tiny = float(F[F > 0].min()) if bool((F > 0).any()) else 1e-30
    logw = -torch.log(sp.clamp_min(1e-30)) - 0.5 * torch.log(F.clamp_min(tiny))
    if mode == "global":
        logw = logw - logw.mean()
    elif mode == "head":
        L, W = logw.shape
        h = logw.reshape(L, W // hd, hd)
        logw = (h - h.mean(2, keepdim=True)).reshape(L, W)
    else:
        raise ValueError("mode is 'global' or 'head'")
    c = math.log(clamp)
    return torch.exp(logw.clamp(-c, c))


def uniform_scales(spread, group, clamp=16.0):
    """kappa multipliers [L, W] (float64) that give every lane of a group of `group` consecutive lanes the same
    absolute step: proportional to 1 / spread, geometric mean 1 within the group, clamped to [1 / clamp, clamp]."""
    sp = torch.as_tensor(spread, dtype=torch.float64).cpu()
    L, W = sp.shape
    if W % group:
        raise ValueError(f"{W} lanes do not split into groups of {group}")
    logw = -torch.log(sp.clamp_min(1e-30)).reshape(L, W // group, group)
    logw = (logw - logw.mean(2, keepdim=True)).reshape(L, W)
    c = math.log(clamp)
    return torch.exp(logw.clamp(-c, c))


def summary(F, scales=None, C=None):
    """Per layer: the keys' and values' mean sensitivity (and mean log2 of the scales, if given)."""
    F = torch.as_tensor(F, dtype=torch.float64)
    C = F.shape[1] // 2 if C is None else C
    out = []
    for l in range(F.shape[0]):
        row = {"layer": l, "keys": float(F[l, :C].mean()), "values": float(F[l, C:].mean())}
        if scales is not None:
            row["log2_scale_keys"] = float(torch.log2(scales[l, :C]).mean())
            row["log2_scale_values"] = float(torch.log2(scales[l, C:]).mean())
        out.append(row)
    return out
