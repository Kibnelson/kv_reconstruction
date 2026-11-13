"""Tests of five ideas for T, our layer-wise one-step transfer codec, measured without changing T.

Every number is an exact cost (-log2 P of each BF16 word, the same accounting as kv_compare.predict) or a GPU time,
and every idea pays for what it sends: side information is counted in bits per value.

  1. adapt    Per-request adaptation. T's constants come from WikiText-2 train, and other text costs more (the
              dataset gap). The kernels already read a per-channel bias b and spread sp per layer, so a request
              can bring its own: forward (the sender measures them and sends them, counted), backward (the receiver
              measures them on a pilot eighth of the tokens decoded first; nothing sent), and an oracle bound (a
              one-step predictor fitted on the request itself; not sendable, it says how much any adaptation can win).
  2. layer0   Layer 0 computed on the decode side. A token's layer-0 KV is a function of its id and position
              (embedding, norm, projections, RoPE: no attention), and the receiver has the token ids and the model.
              T codes layer 0 against its calibration mean, its most expensive layer. Measured: how many layer-0
              words the receiver recomputes bit for bit, and the cost of coding layer 0 against the recomputed values.
  3. anchor   Anchors with key frames. The strongest no-hint predictor used layer l at the token's most recent
              earlier occurrence, which forces sequential waves. Two-wave forms: anchor at the first occurrence
              (first occurrences decode first, repeats second), and key frames every K tokens with the decoder
              choosing each other token's anchor as the key frame nearest in layer l-1 (no side information).
  4. classes  Token classes with their own predictors: known to the decoder (attention-sink positions, repeated
              tokens, k-means on layer l-1), or chosen by the encoder among k predictors (log2 k bits per token and
              layer, counted).
  5. gpu      Per-layer mode allocation and the decoder profile: per layer and mode (dense, keys then values, 4 head
              groups, head by head, rank 256) the coded bits and the decode time, the products alone and the coding
              alone, and the coding kernel's memory floor on this GPU; an offline allocation then picks a mode per
              layer for each link speed.

T's predictor here is its own int8 arithmetic (kv_compare.fused.prepare's constants, fused.reference_mu), so the
adapt costs start from T's exact-model bits. The anchor and class studies use the float one-step predictor of
kv_compare.predict (the same fit before 8-bit rounding) and report their gains against it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time

import numpy as np
import torch

from . import fused as fu
from .core import LN2X2, exact_bits, words_to_float

QS = fu.QS
CAP = 524288.0


# ---------------------------------------------------------------------------------------------- T in torch
def _rot(m, consts, pos, C, hd):
    """Keys of a de-rotated prediction [n, W] (float32) rotated to their positions with T's float32 tables."""
    n = m.shape[0]
    h2 = hd // 2
    k = m[:, :C].reshape(n, -1, hd)
    a, b = k[..., :h2], k[..., h2:]
    cs, sn = consts["cos"][pos][:, None, :], consts["sin"][pos][:, None, :]
    return torch.cat([torch.cat([a * cs - b * sn, b * cs + a * sn], -1).reshape(n, C), m[:, C:]], 1)


def _surprise_prior(x32, mu32, inv):
    """The next layer's prior, as T's kernels compute it: mean over channels of floor(|x - mu| * inv) / QS."""
    q = torch.floor((x32 - mu32).abs() * inv[None, :])
    q = torch.where(q == q, q, torch.full_like(q, CAP)).clamp_max(CAP)
    return (q.to(torch.float64).sum(1) / (q.shape[1] * QS)).to(torch.float32)


def _mult(prior):
    return ((prior * 64.0 + 16.0) * 0.0125).clamp(0.25, 4.0)


ADAPT = {
    "T": "T as it is (calibration constants)",
    "fwd-spread/ch": "forward: the request's own spread per channel and layer (6 bits each, sent)",
    "fwd-bias+spread/ch": "forward: the request's own bias (8 bits) and spread (6 bits) per channel and layer, sent",
    "fwd-spread/head": "forward: one spread scale per head and layer (6 bits each, sent)",
    "bwd-pilot8": "backward: bias and spread per channel measured on every 8th token, decoded first (nothing sent)",
}


def t_costs(study, consts, doc, variants=tuple(ADAPT), pilot=8):
    """T's exact-model bits per layer for one document [L] (mean over its values), for T and each adaptation
    variant; and the side bits each variant sends for the document. Features are the exact previous layer's
    (lossless), so every variant sees the same inputs; each variant keeps its own spread prior."""
    dev, L, W, C, hd = study.device, study.L, study.W, study.C, study.hd
    N = doc.tokens
    pos = torch.arange(N, device=dev)
    f32 = torch.float32
    G = W // hd
    out = {v: torch.zeros(L, dtype=torch.float64) for v in variants}
    side = {v: 0.0 for v in variants}
    prior = {v: torch.ones(N, dtype=f32, device=dev) for v in variants}
    pilot_rows = (pos % pilot) == 0
    prev = None
    for l in range(L):
        w = doc.words(l, dev)
        x = words_to_float(w)
        x32 = x.to(f32)
        acc = torch.zeros(N, W, dtype=f32, device=dev) if prev is None else \
            (prev.double() @ consts["Aq"][l].double().T).to(f32)
        mud = acc * consts["sa"][l][None, :] + consts["b"][l][None, :]
        mu0 = _rot(mud, consts, pos, C, hd)
        sp = consts["sp"][l]
        rd = None
        for v in variants:
            c, g, rows = None, None, None
            if v != "T":
                sel = pilot_rows if v.startswith("bwd") else None
                if "bias" in v or v.startswith("bwd"):
                    if rd is None:
                        rd = study.rope.derotate_row(x, pos).to(f32) - mud
                    r = rd if sel is None else rd[sel]
                    c = r.mean(0)
                    if v.startswith("bwd"):
                        n = float(sel.sum())
                        c = c * (n / (n + 16.0))
                    else:
                        step = sp / 8
                        c = torch.clamp(torch.round(c / step), -127, 127) * step
                        side[v] += 8 * W
                mu1 = mu0 if c is None else mu0 + _rot(c[None, :].expand(N, W).contiguous(), consts, pos, C, hd)
                a = (x32 - mu1).abs() if sel is None else (x32 - mu1)[sel].abs()
                ratio = a.mean(0) / (LN2X2 * sp.double()).to(f32)
                if v.endswith("/head"):
                    ratio = ratio.reshape(G, hd).mean(1).repeat_interleave(hd)
                if v.startswith("bwd"):
                    n = float(sel.sum())
                    g = torch.exp2(torch.log2(ratio.clamp_min(1e-6)) * (n / (n + 16.0)))
                    rows = ~sel
                else:
                    g = torch.exp2(torch.clamp(torch.round(8 * torch.log2(ratio.clamp_min(1e-6))), -31, 31) / 8)
                    side[v] += 6 * (G if v.endswith("/head") else W)
            else:
                mu1 = mu0
            spv = sp if g is None else sp * g
            if rows is not None:                      # pilot rows keep the calibration constants
                mu = torch.where(rows[:, None], mu1, mu0)
                spr = torch.where(rows[:, None], spv[None, :], sp[None, :])
            else:
                mu, spr = mu1, spv[None, :]
            s = spr * _mult(prior[v])[:, None]
            out[v][l] = float(exact_bits(w, mu.double(), s.double()).mean())
            inv = (QS / (LN2X2 * spr.double())).to(f32)
            if rows is None:
                prior[v] = _surprise_prior(x32, mu, inv[0])
            else:
                q = torch.floor((x32 - mu).abs() * inv)
                q = torch.where(q == q, q, torch.full_like(q, CAP)).clamp_max(CAP)
                prior[v] = (q.to(torch.float64).sum(1) / (W * QS)).to(f32)
        prev = fu._feat_ref(x, study.rope, pos, consts["isx"][l])
        del w, x, x32, acc, mud, mu0, rd
    values = L * N * W
    return out, {v: side[v] / values for v in variants}


# ---------------------------------------------------------------------------------------------- layer 0
def _first_kv(cache):
    if hasattr(cache, "layers") and cache.layers is not None:
        layer = cache.layers[0]
        return layer.keys, layer.values
    if hasattr(cache, "key_cache"):
        return cache.key_cache[0], cache.value_cache[0]
    return tuple(cache)[0]


def layer0_words(model, ids, device, chunk_tokens=None):
    """Layer 0's cached K and V words [n, Hk*hd] (uint16 numpy, cache order) recomputed from the token ids with the
    model's first decoder layer only, through the same code path and prefill chunks as kv_compare.capture."""
    inner = model.model
    full = inner.layers
    inner.layers = torch.nn.ModuleList([full[0]])
    try:
        with torch.inference_mode():
            n = len(ids)
            if chunk_tokens and n > int(chunk_tokens):
                from transformers import DynamicCache
                cache = DynamicCache()
                for a in range(0, n, int(chunk_tokens)):
                    part = torch.tensor([list(ids[a:a + int(chunk_tokens)])], dtype=torch.long, device=device)
                    cache = model(part, past_key_values=cache, use_cache=True, logits_to_keep=1).past_key_values
            else:
                cache = model(torch.tensor([list(ids)], dtype=torch.long, device=device), use_cache=True,
                              logits_to_keep=1).past_key_values
            k, v = _first_kv(cache)
            mats = []
            for t in (k, v):
                wds = t[0].permute(1, 0, 2).contiguous().view(torch.int16)
                mats.append(wds.reshape(t.shape[2], -1).cpu().numpy().view(np.uint16).copy())
    finally:
        inner.layers = full
    return mats


def layer0_costs(study, model, docs, scale_doc, device, chunk_tokens=None, t_layer0=None, timer=None):
    """Per document: the fraction of layer-0 words the receiver recomputes bit for bit, and the bits per layer-0
    value of coding the captured words against the recomputed ones: a flag (equal or not, probability from the
    scales document), then for unequal words the word against Logistic(recomputed, s) with s from the scales
    document's differences. t_layer0: {doc: T's layer-0 bits per value} for the saving."""
    def recompute(doc):
        K, V = layer0_words(model, doc.ids[: doc.tokens].tolist(), device, chunk_tokens)
        return torch.cat([torch.as_tensor(K.view(np.int16)), torch.as_tensor(V.view(np.int16))], 1) \
            .to(torch.int32).to(study.device) & 0xFFFF

    def stats(doc):
        got = doc.words(0, study.device)
        ref = recompute(doc)
        if ref.shape != got.shape:
            raise ValueError(f"recomputed layer 0 has shape {tuple(ref.shape)}, the capture {tuple(got.shape)}")
        return got, ref, got == ref

    got, ref, eq = stats(scale_doc)
    n = eq.numel()
    p_eq = (float(eq.sum()) + 1) / (n + 2)                       # add-one: never certain
    d = (words_to_float(got) - words_to_float(ref)).abs()[~eq]
    s_mm = float(d.mean()) / LN2X2 if d.numel() else 1e-3
    out = {"scale_doc": scale_doc.name, "scale_equal_fraction": float(eq.float().mean()), "flag_p_equal": p_eq,
           "mismatch_spread": s_mm, "docs": {}}
    for doc in docs:
        got, ref, eq = stats(doc)
        bits = torch.full(eq.shape, -math.log2(p_eq), dtype=torch.float64, device=eq.device)
        if (~eq).any():
            x = words_to_float(ref)
            mm = exact_bits(got, x, torch.full_like(x, s_mm))
            bits = torch.where(eq, bits, mm - math.log2(1 - p_eq))
        row = {"equal_fraction": float(eq.float().mean()), "bits_per_layer0_value": float(bits.mean())}
        if t_layer0 and doc.name in t_layer0:
            row["t_layer0_bits"] = t_layer0[doc.name]
            row["saving_per_value_all_layers"] = (t_layer0[doc.name] - row["bits_per_layer0_value"]) / study.L
        if timer is not None:
            row["recompute_ms"] = 1e3 * timer(lambda: layer0_words(model, doc.ids[: doc.tokens].tolist(), device,
                                                                     chunk_tokens))
        out["docs"][doc.name] = row
    return out


# ---------------------------------------------------------------------------------------------- layer 0, lean
class _Stop(Exception):
    pass


def _kv_from(attn, hs, cos, sin):
    """The attention module's first lines: K and V projections (K norm where the model has one) and RoPE, through
    the module's own functions. hs [B, n, hidden] -> K, V [B, heads, n, hd]."""
    import sys
    shape = (*hs.shape[:-1], -1, attn.head_dim)
    k = attn.k_proj(hs).view(shape)
    if getattr(attn, "k_norm", None) is not None:
        k = attn.k_norm(k)
    k = k.transpose(1, 2)
    v = attn.v_proj(hs).view(shape).transpose(1, 2)
    rot = sys.modules[type(attn).__module__].apply_rotary_pos_emb
    _, k = rot(k, k, cos, sin)
    return k, v


def _words(t):
    """[1, heads, n, hd] -> int32 words [n, heads*hd] in cache order."""
    n = t.shape[2]
    return (t[0].permute(1, 0, 2).contiguous().view(torch.int16).reshape(n, -1).to(torch.int32)) & 0xFFFF


L0_VARIANTS = {
    "same": "one pass of the request, as the capture ran it",
    "chunks of 2048": "the request in prefill chunks of 2,048 tokens",
    "chunks of 512": "the request in prefill chunks of 512 tokens",
    "behind 1 token": "one token of another request ahead of it in the same pass (vLLM flattens a batch)",
    "behind 2048 tokens": "2,048 tokens of another request ahead of it in the same pass",
}


def _run_passes(model, device, passes):
    """Layer-0 K and V words of the kept rows of each pass (tokens, positions, first kept row, kept rows): the model
    runs up to its first attention module, whose hook takes the inputs and stops the pass."""
    attn = model.model.layers[0].self_attn
    grab = {}

    def hook(mod, args, kwargs):
        grab["h"] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        grab["pe"] = kwargs["position_embeddings"] if "position_embeddings" in kwargs else args[1]
        raise _Stop

    h = attn.register_forward_pre_hook(hook, with_kwargs=True)
    ks, vs = [], []
    try:
        with torch.inference_mode():
            for tok, pos, keep, n in passes:
                try:
                    model.model(input_ids=torch.tensor([list(tok)], device=device),
                                position_ids=torch.tensor([list(pos)], device=device), use_cache=False)
                except _Stop:
                    pass
                cos, sin = grab["pe"]
                k, v = _kv_from(attn, grab["h"], cos, sin)
                ks.append(_words(k)[keep:keep + n])
                vs.append(_words(v)[keep:keep + n])
    finally:
        h.remove()
    return torch.cat(ks, 0), torch.cat(vs, 0)


def layer0_kv(model, ids, device, variant="same", other=()):
    """Layer 0's K and V words [n, Hk*hd] (int32 on device) computed with the embedding, the first layer's input
    norm, the K and V projections and RoPE only (no attention, no MLP). `variant` (L0_VARIANTS) sets the prefill
    shape; `other` are another request's token ids for the batched variants."""
    n = len(ids)
    if variant == "same":
        passes = [(list(ids), list(range(n)), 0, n)]
    elif variant.startswith("chunks of "):
        c = int(variant.split()[-1])
        passes = [(list(ids[a:a + c]), list(range(a, min(a + c, n))), 0, min(c, n - a)) for a in range(0, n, c)]
    elif variant.startswith("behind "):
        m = int(variant.split()[1])
        if not len(other):
            raise ValueError(f"{variant} needs another request's tokens")
        other = (list(other) * (m // len(other) + 1))[:m]          # repeated if shorter (only the shape matters)
        passes = [(list(other[:m]) + list(ids), list(range(m)) + list(range(n)), m, n)]
    else:
        raise ValueError(f"variant is one of {sorted(L0_VARIANTS)}")
    return _run_passes(model, device, passes)


# Simulated vLLM prefill steps: (own tokens per step, other tokens before them, other tokens after them). The step's
# matrix product has own + before + after rows; vLLM caps a step (max_num_batched_tokens) at 2,048 in our PD setup.
L0_SCHEDULES = {
    "2048 own": (2048, 0, 0),
    "1024 own after 1024 other": (1024, 1024, 0),
    "1536 own before 512 other": (1536, 0, 512),
    "512 own after 1536 other": (512, 1536, 0),
}
L0_RECEIVERS = {
    "one pass": "the receiver computes the request in one pass",
    "same chunks": "the sender's chunks, without the other request's rows",
    "padded to the step": "the sender's chunks padded with dummy rows to the sender's step size, at its offset",
    "padded, own rows first": "padded to the step size, the request's rows first",
}


def layer0_schedule(model, ids, device, schedule, receiver=None, other=(), dummy=0):
    """Layer-0 words of a request computed in prefill steps. schedule (own, before, after) from L0_SCHEDULES. The
    sender (receiver None) fills the other rows with another request's tokens `other`; a receiver (L0_RECEIVERS)
    recomputes with what the sender told it (the step sizes), filling with the dummy token at position 0."""
    own, before, after = schedule
    n = len(ids)
    if receiver == "one pass":
        return layer0_kv(model, ids, device)
    other = list(other) or [dummy]
    other = (other * ((before + after) // len(other) + 1))[:before + after]
    passes = []
    for a in range(0, n, own):
        m = min(own, n - a)
        mine, pos = list(ids[a:a + m]), list(range(a, a + m))
        if receiver is None:
            tok = other[:before] + mine + other[before:before + after]
            ps = list(range(before)) + pos + list(range(after))
            passes.append((tok, ps, before, m))
        elif receiver == "same chunks":
            passes.append((mine, pos, 0, m))
        elif receiver == "padded to the step":
            passes.append(([dummy] * before + mine + [dummy] * after, [0] * before + pos + [0] * after, before, m))
        elif receiver == "padded, own rows first":
            passes.append((mine + [dummy] * (before + after), pos + [0] * (before + after), 0, m))
        else:
            raise ValueError(f"receiver is one of {sorted(L0_RECEIVERS)}")
    return _run_passes(model, device, passes)


def _ordered(w):
    """BF16 words as signed magnitudes (monotone in the value; +0 and -0 both 0)."""
    return torch.where(w >= 0x8000, -(w & 0x7FFF), w & 0x7FFF)


def ulp_table(got, ref, R=8):
    """Probabilities of the coded symbol d = ordered(got) - ordered(ref) in [-R, R], and of an escape (|d| > R, or
    equal order but different words), add-one smoothed."""
    d = _ordered(got) - _ordered(ref)
    esc = (d.abs() > R) | ((d == 0) & (got != ref))
    counts = torch.bincount((d[~esc] + R).reshape(-1).long(), minlength=2 * R + 1).double()
    counts = torch.cat([counts, torch.tensor([float(esc.sum())], dtype=torch.float64, device=counts.device)]) + 1
    return counts / counts.sum()


def ulp_bits(got, ref, table):
    """Bits per value of coding got against ref with the symbol table (an escape then costs the 16-bit word)."""
    R = (table.numel() - 2) // 2
    d = _ordered(got) - _ordered(ref)
    esc = (d.abs() > R) | ((d == 0) & (got != ref))
    sym = torch.where(esc, torch.full_like(d, 2 * R + 1), d + R).long()
    bits = -torch.log2(table.to(sym.device)[sym]) + 16.0 * esc
    return float(bits.mean())


def value_table(got, ref):
    """The value coder's parameters from a scales document: the chance a word is recomputed exactly, and per channel
    the spread of the unequal words around the recomputed value (mean absolute difference / 2 ln 2)."""
    eq = got == ref
    n = eq.numel()
    p_eq = (float(eq.sum()) + 1) / (n + 2)
    d = (words_to_float(got) - words_to_float(ref)).abs()
    d = torch.where(eq, torch.zeros_like(d), d)
    cnt = (~eq).sum(0).to(torch.float64)
    pooled = float(d.sum() / max(float(cnt.sum()), 1.0))
    s = torch.where(cnt > 0, d.sum(0) / cnt.clamp_min(1), torch.full_like(cnt, pooled)) / LN2X2
    return {"p_eq": p_eq, "s": s.clamp_min(1e-30)}


def value_bits(got, ref, table):
    """Bits per value of coding got against ref: a flag (equal or not), then an unequal word against a logistic
    around the recomputed value with the channel's spread (exact -log2 P of the BF16 word; a sign flip next to zero
    is a small difference here, unlike a difference in BF16 steps)."""
    eq = got == ref
    p = table["p_eq"]
    bits = torch.full(eq.shape, -math.log2(p), dtype=torch.float64, device=eq.device)
    if (~eq).any():
        mm = exact_bits(got, words_to_float(ref), table["s"].to(eq.device)[None, :].expand(eq.shape))
        bits = torch.where(eq, bits, mm - math.log2(1 - p))
    return float(bits.mean())


def layer0_matched(study, model, docs, scale_doc, device, t_layer0=None, schedules=tuple(L0_SCHEDULES),
                   receivers=tuple(L0_RECEIVERS)):
    """For each simulated prefill schedule (the sender) and each receiver: the share of layer-0 K and V words the
    receiver recomputes bit for bit, and the bits per layer-0 value with the step coder and the value coder (tables
    from the scales document under the same pair)."""
    C = study.C
    out = {"schedules": {k: list(L0_SCHEDULES[k]) for k in schedules}, "receivers": {k: L0_RECEIVERS[k] for k in receivers},
           "docs": {}}
    other_of = {d.name: scale_doc.ids[:2048].tolist() for d in docs}
    other_of[scale_doc.name] = docs[0].ids[:2048].tolist()

    def both(doc, sch, rec):
        ids = doc.ids[: doc.tokens].tolist()
        Ks, Vs = layer0_schedule(model, ids, device, L0_SCHEDULES[sch], None, other_of[doc.name])
        Kr, Vr = layer0_schedule(model, ids, device, L0_SCHEDULES[sch], rec)
        return torch.cat([Ks, Vs], 1), torch.cat([Kr, Vr], 1)

    tables = {}
    for sch in schedules:
        for rec in receivers:
            got, ref = both(scale_doc, sch, rec)
            tables[(sch, rec)] = (ulp_table(got, ref), value_table(got, ref))
    for d in docs:
        rows = {}
        for sch in schedules:
            for rec in receivers:
                got, ref = both(d, sch, rec)
                ut, vt = tables[(sch, rec)]
                r = {"keys_equal": float((got[:, :C] == ref[:, :C]).float().mean()),
                     "values_equal": float((got[:, C:] == ref[:, C:]).float().mean()),
                     "step_bits": ulp_bits(got, ref, ut), "value_bits": value_bits(got, ref, vt)}
                r["bits_per_layer0_value"] = min(r["step_bits"], r["value_bits"])
                if t_layer0 and d.name in t_layer0:
                    r["saving_per_value_all_layers"] = (t_layer0[d.name] - r["bits_per_layer0_value"]) / study.L
                rows[f"{sch} | {rec}"] = r
        out["docs"][d.name] = rows
    return out


def layer0_robust(study, model, docs, scale_doc, device, t_layer0=None, timer=None, variants=tuple(L0_VARIANTS)):
    """Per prefill shape: the share of layer-0 K and V words the lean recompute gives bit for bit, the largest
    difference in BF16 steps, and the bits per layer-0 value of coding the captured words against it (the symbol
    table from the scales document under the same shape). t_layer0: {doc: T's layer-0 bits} for the saving.
    timer(fn) -> seconds: the lean recompute's time per document."""
    C = study.C
    out = {"variants": {v: L0_VARIANTS[v] for v in variants}, "docs": {}, "tables": {}}
    others = {d.name: (scale_doc if d.name != scale_doc.name else docs[0]).ids[:2048].tolist() for d in docs}
    others[scale_doc.name] = docs[0].ids[:2048].tolist()
    tables = {}
    for v in variants:
        got = scale_doc.words(0, study.device)
        K, V = layer0_kv(model, scale_doc.ids[: scale_doc.tokens].tolist(), device, v, others[scale_doc.name])
        tables[v] = ulp_table(got, torch.cat([K, V], 1).to(got.device))
        out["tables"][v] = {"equal": float((got == torch.cat([K, V], 1).to(got.device)).float().mean())}
    for d in docs:
        got = d.words(0, study.device)
        row = {}
        for v in variants:
            K, V = layer0_kv(model, d.ids[: d.tokens].tolist(), device, v, others[d.name])
            ref = torch.cat([K, V], 1).to(got.device)
            diff = (_ordered(got) - _ordered(ref)).abs()
            r = {"keys_equal": float((got[:, :C] == ref[:, :C]).float().mean()),
                 "values_equal": float((got[:, C:] == ref[:, C:]).float().mean()),
                 "max_steps": int(diff.max()), "bits_per_layer0_value": ulp_bits(got, ref, tables[v])}
            if t_layer0 and d.name in t_layer0:
                r["saving_per_value_all_layers"] = (t_layer0[d.name] - r["bits_per_layer0_value"]) / study.L
            row[v] = r
        if timer is not None:
            ids = d.ids[: d.tokens].tolist()
            row["lean_ms"] = 1e3 * timer(lambda: layer0_kv(model, ids, device))
            row["first_layer_ms"] = 1e3 * timer(lambda: layer0_words(model, ids, device))
        out["docs"][d.name] = row
    return out


# ---------------------------------------------------------------------------------------------- calibration sets
CAL_DOMAINS = {"code": "MBPP, all splits (Python problems and solutions; HumanEval's kind of text)",
               "math": "GSM8K train",
               "exam": "MMLU dev and validation (MMLU's prompt format)",
               "news": "Penn Treebank train"}


def cal_sets(windows=3):
    """{set: (calibration documents, scales document, note)}. Calibration window i of a domain is cal_<domain>_<i>;
    every domain set fits on windows 0..windows-1 and takes window `windows` as its scales document. 'mixed' has 6
    documents like 'wikitext' (two WikiText-2 train, one of each domain), 'mixed-all' every calibration document;
    both take cal_mixed_<windows> as scales document, an equal share of each source after those windows."""
    dom = list(CAL_DOMAINS)
    mixed_t = f"cal_mixed_{windows}"
    sets = {"wikitext": ([f"wtrain{i}" for i in range(6)], "wtables", "T's constants: WikiText-2 train"),
            "mixed": (["wtrain0", "wtrain1"] + [f"cal_{d}_0" for d in dom], mixed_t,
                      "6 documents: 2 WikiText-2 train, 1 of each domain"),
            "mixed-all": ([f"wtrain{i}" for i in range(windows)] + [f"cal_{d}_{i}" for d in dom for i in range(windows)],
                          mixed_t, f"{windows * (len(dom) + 1)} documents: {windows} of every source")}
    for d in dom:
        sets[d] = ([f"cal_{d}_{i}" for i in range(windows)], f"cal_{d}_{windows}", CAL_DOMAINS[d])
    return sets


HELD_OUT = {"wikitext2": "wikitext", "humaneval": "code", "gsm8k": "math", "mmlu": "exam", "ptb": "news"}


def cal_loo_sets(windows=3):
    """{held-out source: (calibration documents, scales document, note)}: 'mixed-all' without one source (its
    documents and its share of the scales document), for testing on that source's datasets. The scales document
    cal_mixedx_<source>_<windows> has an equal share of every other source."""
    out = {}
    for m in ["wikitext"] + list(CAL_DOMAINS):
        docs = ([f"wtrain{i}" for i in range(windows)] if m != "wikitext" else []) + \
               [f"cal_{d}_{i}" for d in CAL_DOMAINS if d != m for i in range(windows)]
        out[m] = (docs, f"cal_mixedx_{m}_{windows}", f"mixed-all without {m}: {len(docs)} documents")
    return out


def pick_sets(costs, sets, layers=None, cost=None, side_bits=None):
    """Per-request choice among constant sets. costs {set: {doc: per-layer bits [L]}}. For every document the set
    with the fewest bits over `layers` (None: all) is chosen; its cost is cost(per-layer bits) (default: the mean),
    plus side_bits(doc) when given (the choice itself). Returns {doc: (set, bits)}."""
    cost = cost or (lambda v: float(np.mean(v)))
    out = {}
    docs = set.intersection(*(set(costs[s]) for s in sets))
    for d in sorted(docs):
        def score(s):
            v = np.asarray(costs[s][d], dtype=np.float64)
            return float(v[layers].sum() if layers is not None else v.sum())
        s = min(sets, key=score)
        out[d] = (s, cost(costs[s][d]) + (side_bits(d) if side_bits else 0.0))
    return out


# ---------------------------------------------------------------------------------------------- float predictor
@dataclass(frozen=True)
class Spec:
    """A float one-step variant: anchor ("", "recent", "first", "nnK") and/or classes ("", "sink", "repeat",
    "kmeansK", "bestK")."""
    name: str
    anchor: str = ""
    classes: str = ""
    em: int = 1
    note: str = ""


ANCHORS = [Spec("one-step (float)", note="layer l-1 only: T's predictor before 8-bit rounding"),
           Spec("anchor: most recent occurrence", anchor="recent",
                note="+ layer l at the token's most recent earlier occurrence (waves = longest repeat chain)"),
           Spec("anchor: first occurrence", anchor="first", note="+ layer l at the first occurrence; 2 waves"),
           Spec("anchor: nearest key frame, K=8", anchor="nn8",
                note="key frames t % 8 == 0 first; others + layer l of the key frame nearest in layer l-1; 2 waves"),
           Spec("anchor: nearest key frame, K=4", anchor="nn4", note="as K=8 with key frames t % 4 == 0")]
CLASSES = [Spec("classes: sink tokens", classes="sink", note="tokens 0-3 their own predictor (decoder knows)"),
           Spec("classes: repeated tokens", classes="repeat",
                note="first occurrences and repeats separate predictors, no anchor input (decoder knows)"),
           Spec("classes: k-means of layer l-1, k=4", classes="kmeans4",
                note="decoder assigns each token by its decoded layer l-1 (no side bits)"),
           Spec("classes: encoder picks 1 of 2", classes="best2",
                note="2 predictors (k-means start, one refit); 1 bit per token and layer sent"),
           Spec("classes: encoder picks 1 of 4", classes="best4",
                note="4 predictors; 2 bits per token and layer sent")]


def _first_index(doc):
    out, first = np.full(doc.tokens, -1), {}
    for t, v in enumerate(doc.ids[: doc.tokens].tolist()):
        if v in first:
            out[t] = first[v]
        else:
            first[v] = t
    return out


def waves(anchor_idx):
    """Decode waves a chain of anchors forces: 1 + the longest chain."""
    depth = np.zeros(len(anchor_idx), dtype=np.int64)
    for t, a in enumerate(anchor_idx):
        if a >= 0:
            depth[t] = depth[a] + 1
    return int(depth.max()) + 1 if len(depth) else 1


class _Ctx:
    """Per-spec state: anchors per document (token-id kinds), k-means centroids per layer."""

    def __init__(self, study, spec):
        self.study, self.spec = study, spec
        self._ids = {}
        self.km = None

    def anchors(self, doc, layer):
        st, kind = self.study, self.spec.anchor
        dev, N = st.device, doc.tokens
        if not kind:
            return None
        if kind in ("recent", "first"):
            if doc.name not in self._ids:
                a = st.rep_index(doc).cpu().numpy() if kind == "recent" else _first_index(doc)
                self._ids[doc.name] = torch.as_tensor(a, device=dev)
            return self._ids[doc.name]
        K = int(kind[2:])
        if layer == 0:
            return torch.full((N,), -1, dtype=torch.long, device=dev)
        f = st.feats(doc, layer - 1)
        f = (f - f.mean(0)) / f.std(0).clamp_min(1e-6)            # the receiver's own statistics of layer l-1
        f = torch.nn.functional.normalize(f, dim=1)
        kf = torch.arange(0, N, K, device=dev)
        best = torch.empty(N, dtype=torch.long, device=dev)
        for r0 in range(0, N, 2048):
            best[r0:r0 + 2048] = kf[(f[r0:r0 + 2048] @ f[kf].T).argmax(1)]
        best[kf] = -1
        return best

    def labels(self, doc, layer, anchors):
        """Decoder-known class of every row (the encoder-chosen kinds start from k-means)."""
        st, cl, N = self.study, self.spec.classes, doc.tokens
        dev = st.device
        if self.spec.anchor:
            return (anchors >= 0).long()
        if not cl:
            return torch.zeros(N, dtype=torch.long, device=dev)
        if cl == "sink":
            return (torch.arange(N, device=dev) < 4).long()
        if cl == "repeat":
            return (st.rep_index(doc) >= 0).long()
        if layer == 0:
            return torch.zeros(N, dtype=torch.long, device=dev)
        mean, istd, cent = self.km[layer]
        f = (st.feats(doc, layer - 1) - mean) * istd
        return torch.cdist(f, cent).argmin(1)

    def k(self):
        cl = self.spec.classes
        if self.spec.anchor or cl in ("sink", "repeat"):
            return 2
        return int(cl.lstrip("kmeansbest")) if cl else 1

    def fit_kmeans(self, docs, layer, k, iters=10, sample=32768, seed=0):
        st = self.study
        X = torch.cat([st.feats(d, layer - 1) for d in docs], 0)
        mean, istd = X.mean(0), 1.0 / X.std(0).clamp_min(1e-6)
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(X.shape[0], generator=g)[:sample].to(X.device)
        Z = (X[idx] - mean) * istd
        cent = Z[torch.randperm(Z.shape[0], generator=g)[:k].to(X.device)].clone()
        for _ in range(iters):
            a = torch.cdist(Z, cent).argmin(1)
            for c in range(k):
                m = a == c
                if m.any():
                    cent[c] = Z[m].mean(0)
        return mean, istd, cent


def _inputs(st, doc, layer, anchors, with_anchor):
    parts = [st.feats(doc, layer - 1)] if layer >= 1 else []
    if with_anchor:
        parts.append(st.feats(doc, layer)[anchors.clamp_min(0)])
    N = doc.tokens
    return torch.cat(parts, 1) if parts else torch.zeros(N, 0, device=st.device)


def _class_fits(st, docs, layer, k, has_anchor, rows_of, inputs_of):
    """Conditional-mean fit per class from the rows of that class in docs (two passes, as Study.covariance)."""
    dev, f64 = st.device, torch.float64
    prod = torch.float32 if dev.type == "cuda" else f64
    s1, n = [None] * k, [0] * k
    for d in docs:
        z = rows_of(d)
        for c in range(k):
            m = z == c
            if not m.any():
                continue
            X = torch.cat([inputs_of(d, has_anchor[c])[m], st.feats(d, layer)[m]], 1)
            part = X.to(f64).sum(0)
            s1[c] = part if s1[c] is None else s1[c] + part
            n[c] += int(m.sum())
    fits = [None] * k
    for c in range(k):
        if n[c] < 2:
            continue
        mu = s1[c] / n[c]
        dim = mu.numel()
        S = torch.zeros(dim, dim, dtype=f64, device=dev)
        mp = mu.to(prod)
        for d in docs:
            z = rows_of(d)
            m = z == c
            if not m.any():
                continue
            X = torch.cat([inputs_of(d, has_anchor[c])[m], st.feats(d, layer)[m]], 1).to(prod) - mp
            S += (X.T @ X).to(f64)
        S /= n[c]
        ne = dim - st.W
        fits[c] = st.solve(mu, [S], ["in"] * (1 if ne else 0), 1, "natural")
    return fits, n


def _predict(st, F, Xe, N):
    ne = F.ne
    md = (Xe @ F.A[:, :ne].T + F.b) if ne else F.b[None, :].expand(N, -1)
    return st.rope.rotate_row(md.to(torch.float64), st.positions(N))


def float_costs(study, spec, cal_docs, scale_doc, test_docs, log=print, fit_rows=None, score_rows=None):
    """Exact bits per layer [L] of every test document for one float variant, the side bits per value, and per
    document the fraction of rows in class 1 (rows with an anchor, for anchors) and the decode waves.
    fit_rows {doc name: bool [N]}: only these rows of a calibration or scales document are fitted on;
    score_rows {doc name: bool [N]}: only these rows of a test document are counted (cross-fitting)."""
    st = study
    L, W, dev = st.L, st.W, st.device
    ctx = _Ctx(st, spec)
    k = ctx.k()
    encoder_choice = spec.classes.startswith("best")
    kmeans = spec.classes.startswith("kmeans") or encoder_choice
    has_anchor = [False, bool(spec.anchor)] if spec.anchor else [False] * k
    res = {d.name: torch.zeros(L, dtype=torch.float64) for d in test_docs}
    frac = {d.name: [] for d in test_docs}
    side = {d.name: 0.0 for d in test_docs}
    prior = {d.name: torch.ones(d.tokens, dtype=torch.float64, device=dev) for d in test_docs}
    for layer in range(L):
        if kmeans and layer >= 1:
            ctx.km = {layer: ctx.fit_kmeans(cal_docs, layer, k)}
        anchors = {}

        def anc(d):
            if d.name not in anchors:
                anchors[d.name] = ctx.anchors(d, layer)
            return anchors[d.name]

        cache_in = {}

        def inputs_of(d, wa):
            key = (d.name, wa)
            if key not in cache_in:
                cache_in[key] = _inputs(st, d, layer, anc(d), wa)
            return cache_in[key]

        labels = {}

        def rows_of(d):
            if d.name not in labels:
                labels[d.name] = ctx.labels(d, layer, anc(d))
            return labels[d.name]

        def fit_rows_of(d):
            z = rows_of(d)
            if fit_rows and d.name in fit_rows:
                z = torch.where(fit_rows[d.name].to(z.device), z, torch.full_like(z, -1))
            return z

        kk = k if (layer >= 1 or not kmeans) else 1
        fits, counts = _class_fits(st, cal_docs, layer, kk, has_anchor, fit_rows_of, inputs_of)
        fallback = max(range(kk), key=lambda c: counts[c] if not has_anchor[c] else -1)

        def fit_of(c):
            return (fits[c], has_anchor[c]) if fits[c] is not None else (fits[fallback], has_anchor[fallback])

        def preds(d):
            """Prediction of every row under each class's fit, a list of k [N, W]."""
            out = []
            for c in range(kk):
                F, wa = fit_of(c)
                out.append(_predict(st, F, inputs_of(d, wa), d.tokens))
            return out

        def spreads(items):
            """Per-class spread (mean absolute residual / 2 ln 2) over (x, z, predictions) items; a class with
            fewer than 16 rows takes the pooled spread."""
            sums = [torch.zeros(W, dtype=torch.float64, device=dev) for _ in range(kk)]
            cnt, tot, n = [0] * kk, torch.zeros(W, dtype=torch.float64, device=dev), 0
            for x, z, ms in items:
                for c in range(kk):
                    m = z == c
                    if m.any():
                        r = (x[m] - ms[c][m]).abs().sum(0)
                        sums[c] += r
                        tot += r
                        cnt[c] += int(m.sum())
                        n += int(m.sum())
            pooled = tot / max(n, 1) / LN2X2 + 1e-12
            return [sums[c] / cnt[c] / LN2X2 + 1e-12 if cnt[c] >= 16 else pooled for c in range(kk)]

        def choose(x, ms, sps):
            """The encoder's class per row: the fit with the smallest spread-normalised absolute residual."""
            return torch.stack([((x - ms[c]).abs() / sps[c][None, :]).sum(1) for c in range(kk)]).argmin(0)

        if encoder_choice and kk > 1:                        # hard EM: rows move to the class that codes them best
            for _ in range(spec.em):
                sps = spreads((words_to_float(d.words(layer, dev)), fit_rows_of(d), preds(d)) for d in cal_docs)
                labels.update({d.name: choose(words_to_float(d.words(layer, dev)), preds(d), sps) for d in cal_docs})
                fits, counts = _class_fits(st, cal_docs, layer, kk, has_anchor, fit_rows_of, inputs_of)
                fallback = max(range(kk), key=lambda c: counts[c])
        ms = preds(scale_doc)
        xs = words_to_float(scale_doc.words(layer, dev))
        z = fit_rows_of(scale_doc)
        if encoder_choice and kk > 1:
            z = torch.where(z >= 0, choose(xs, ms, spreads([(xs, z, ms)])), z)
        sp = spreads([(xs, z, ms)])
        del ms, xs
        for d in test_docs:
            w = d.words(layer, dev)
            x = words_to_float(w)
            ms = preds(d)
            mult = ((64 * prior[d.name] + 16) / 80).clamp(0.25, 4.0)[:, None]
            if encoder_choice and kk > 1:
                bits = torch.stack([exact_bits(w, ms[c], sp[c][None, :] * mult).sum(1) for c in range(kk)])
                z = bits.argmin(0)
                tot = bits.gather(0, z[None, :])[0]
                side[d.name] += math.log2(kk) * d.tokens
            else:
                z = rows_of(d)
                tot = torch.zeros(d.tokens, dtype=torch.float64, device=dev)
                for c in range(kk):
                    m = z == c
                    if m.any():
                        tot[m] = exact_bits(w[m], ms[c][m], sp[c][None, :] * mult[m]).sum(1)
            if score_rows and d.name in score_rows:
                keep = score_rows[d.name].to(tot.device)
                res[d.name][layer] = float(tot[keep].sum()) / (int(keep.sum()) * W)
            else:
                res[d.name][layer] = float(tot.sum()) / (d.tokens * W)
            frac[d.name].append(float((z > 0).float().mean()))
            q = torch.empty_like(x)
            for c in range(kk):
                m = z == c
                q[m] = (x[m] - ms[c][m]).abs() / (LN2X2 * sp[c][None, :])
            prior[d.name] = q.mean(1)
        cache_in.clear()
        st.forget(layer)
        log(f"    {spec.name}: layer {layer} done")
    info = {}
    for d in test_docs:
        row = {"class1_fraction": float(np.mean(frac[d.name])), "side_bits_per_value": side[d.name] / (L * d.tokens * W)}
        if spec.anchor in ("recent", "first"):
            row["waves"] = waves(ctx.anchors(d, 0).cpu().numpy())
        elif spec.anchor:
            row["waves"] = 2
            K = int(spec.anchor[2:])
            row["nn_macs_per_value"] = (d.tokens / K) * (L - 1) / L         # N x N/K x W per layer / (N x W)
        info[d.name] = row
    st._f.clear()
    return res, info


def oracle_costs(study, doc, block=512, log=print):
    """The float one-step predictor fitted on the document itself: in-sample (covariance, mean and spreads from all
    its rows; optimistic, a bound) and cross-fitted (blocks of `block` tokens in two alternating folds, each fold
    coded with the fit of the other: what a predictor fitted on this kind of text would cost). Neither is sendable
    (a predictor is W x W coefficients per layer); they say how much room any adaptation has."""
    inside, _ = float_costs(study, Spec("oracle"), [doc], doc, [doc], log=log)
    block = max(1, min(block, doc.tokens // 4))           # both folds get rows
    fold = (torch.arange(doc.tokens, device=study.device) // block) % 2 == 0
    cross = torch.zeros(study.L, dtype=torch.float64)
    for f in (fold, ~fold):
        part, _ = float_costs(study, Spec("oracle"), [doc], doc, [doc], log=log, fit_rows={doc.name: ~f},
                              score_rows={doc.name: f})
        cross += part[doc.name] * float(f.float().mean())
    return inside[doc.name], cross


# ---------------------------------------------------------------------------------------------- GPU: modes
MODE_GROUPS = {"dense": 1, "kv": 2, "4g": 4, "hbh": None, "rank256": 1}


def mode_codec(consts, rank, groups, mode, C, hd, T, cls=None, layers=None):
    """T in one mode, with the launch choices of the PD codec (tokens per program, chunks, rank tiles). cls: the
    codec class (default T's LayerwiseOneStep; kv_compare.t2.Layer0Recomputed for T2); layers: cut the constants
    to the first n layers."""
    from .layerwise import LayerwiseOneStep
    cls = cls or LayerwiseOneStep
    W = consts["Aq"].shape[1]
    ng = MODE_GROUPS[mode] or W // hd
    base = groups[ng] if ng > 1 else consts
    if layers is not None:
        from .t2 import slice_layers
        L = consts["Aq"].shape[0]
        base = slice_layers(base, layers, L)
        consts = slice_layers(consts, layers, L)
        rank = slice_layers(rank, layers, L) if rank is not None else None
    TB, ch = (32, 2) if ng <= 4 else (16, 4)
    while ch > 1 and T % (TB * ch):
        ch //= 2
    rk = rank if mode == "rank256" else None
    r = int(rank["rank"]) if rk is not None else 128
    tiles = (128, 128, 64, 4) if (T % 256 == 0 and r % 128 == 0 and W % 64 == 0) else (16, 16, 32, 4)
    kp = dict(base, cos=consts["cos"][:T].contiguous(), sin=consts["sin"][:T].contiguous())
    return cls(kp, C, hd, tokens_per_program=TB, mode="uniform", search=True, chunks=ch, rank=rk,
               requant="epilogue" if rk is not None else "kernel", rank_tiles=tiles)


def copy_bandwidth(device, mib=1024, reps=5):
    """Device memory bandwidth (bytes read + written per second) of a large copy."""
    n = mib * 2 ** 20 // 4
    a = torch.empty(n, dtype=torch.float32, device=device)
    b = torch.empty_like(a)
    for _ in range(2):
        b.copy_(a)
    times = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        b.copy_(a)
        e1.record()
        e1.synchronize()
        times.append(e0.elapsed_time(e1) / 1e3)
    del a, b
    return 2 * n * 4 / statistics.median(times)


def mode_rows(study, consts, rank, groups, doc, modes, device, timer, log=print, reps=5, multiple=256):
    """Per mode on one document: bits per layer, per-layer decode time (events at each layer's hand-over),
    whole decode, the products alone, the coding alone, and the coding kernel's memory floor. The document is cut
    to a multiple of `multiple` tokens (256: the PD codec's padding)."""
    L, W, C, hd = study.L, study.W, study.C, study.hd
    T = doc.tokens - doc.tokens % multiple
    words = torch.empty(L, T, W, dtype=torch.int16, device=device)
    for l in range(L):
        w = doc.words(l, device, rows=T)
        words[l] = (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
    n = L * T * W
    bw = copy_bandwidth(device) if device.type == "cuda" else None
    rows = []
    for mode in modes:
        row = {"mode": mode, "doc": doc.name, "tokens": T}
        try:
            codec = mode_codec(consts, rank, groups, mode, C, hd, T)
            xall, prior = codec.model(words)
            lw = torch.zeros(L, dtype=torch.int64, device=device)
            buf, starts = codec.encode(words, xall, prior, layer_words=lw)
            bufs = codec.buffers(T)
            dec = codec.decode(buf, starts, T, bufs=bufs)
            row["bit_exact"] = bool(torch.equal(dec, words))
            row["bits_per_value"] = codec.bits(starts, T) / n
            row["layer_bits"] = [16 * int(v) for v in lw.tolist()]       # per layer, whole document
            row["values_per_layer"] = T * W
            if mode == "dense":                       # the kernel's own model cost: checks t_costs' reference
                tot, pos = 0.0, torch.arange(T, device=device)
                for l in range(L):
                    m = fu.reference_mu(codec.k, l, xall[l - 1].float() if l else None, pos, C, hd)
                    s = codec.k["sp"][l][None, :] * _mult(prior[l])[:, None]
                    tot += float(exact_bits(words[l].to(torch.int32) & 0xFFFF, m.double(), s.double()).sum())
                row["exact_model_bits_per_value"] = tot / n
            row["encode_ms"] = 1e3 * timer(lambda: codec.encode(words, xall, prior), device)
            row["decode_ms"] = 1e3 * timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs), device)
            accs = codec.products(xall)
            row["coding_only_ms"] = 1e3 * timer(lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs, accs=accs),
                                                device)
            row["products_only_ms"] = 1e3 * timer(
                lambda: codec.decode(buf, starts, T, out=dec, bufs=bufs, products_only=True), device)
            accs = None
            if device.type == "cuda":
                ring = torch.empty(2, T, W, dtype=torch.int16, device=device)
                per = []
                for _ in range(reps + 1):
                    evs = []
                    e0 = torch.cuda.Event(enable_timing=True)

                    def sink(l, wl):
                        e = torch.cuda.Event(enable_timing=True)
                        e.record()
                        evs.append(e)
                    e0.record()
                    codec.decode(buf, starts, T, out=ring, bufs=bufs, sink=sink)
                    torch.cuda.synchronize(device)
                    marks = [e0] + evs
                    per.append([marks[i].elapsed_time(marks[i + 1]) for i in range(L)])
                per = per[1:]
                row["layer_ms"] = [statistics.median(p[l] for p in per) for l in range(L)]
                byte_per_value = 4 + 2 + 1 + row["bits_per_value"] / 8   # int32 sums in, words + features out, stream
                row["coding_floor_ms"] = 1e3 * n * byte_per_value / bw
                row["copy_gb_per_s"] = bw / 1e9
                del ring
            log(f"  {mode}: exact={row['bit_exact']}, {row['bits_per_value']:.4f} bits/value, decode "
                f"{row['decode_ms']:.1f} ms (products {row['products_only_ms']:.1f}, coding {row['coding_only_ms']:.1f}"
                + (f", coding floor {row['coding_floor_ms']:.1f}" if "coding_floor_ms" in row else "") + ")")
            codec = xall = prior = buf = starts = bufs = dec = None      # free before the next mode
        except Exception as exc:                      # one mode failing must not lose the others
            row["error"] = repr(exc)[:300]
            log(f"  {mode}: not measured ({row['error']})")
        rows.append(row)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------------------------- allocation
def allocate(rows, gbps, step_ms=0.01):
    """Pick one mode per layer for a link of gbps Gb/s from per-layer bits and times of whole-mode runs.
    serial: minimise sum(link + decode) (per layer independent); pipelined: minimise max(sum link, sum decode),
    by dynamic programming over a time grid (fewest bits for every decode-time budget). Returns both against the
    best single mode."""
    ok = [r for r in rows if "layer_ms" in r and "layer_bits" in r and not r.get("error")]
    if not ok:
        return None
    L = len(ok[0]["layer_bits"])
    bits = np.array([r["layer_bits"] for r in ok], dtype=np.float64)          # [modes, L]
    ms = np.array([r["layer_ms"] for r in ok], dtype=np.float64)
    link = bits / (gbps * 1e9) * 1e3                                            # ms per layer on the link
    names = [r["mode"] for r in ok]
    single_serial = (link + ms).sum(1)
    single_pipe = np.maximum(link.sum(1), ms.sum(1))
    pick_s = (link + ms).argmin(0)
    mix_serial = float((link + ms)[pick_s, np.arange(L)].sum())
    u = np.rint(ms / step_ms).astype(np.int64)
    top = int(u.max(0).sum()) + 1
    best = np.full(top, np.inf)
    best[0] = 0.0
    choice = np.zeros((L, top), dtype=np.int64)
    for l in range(L):
        new = np.full(top, np.inf)
        for m in range(len(ok)):
            shifted = np.full(top, np.inf)
            shifted[u[m, l]:] = best[: top - u[m, l]] + bits[m, l]
            better = shifted < new
            new[better] = shifted[better]
            choice[l, better] = m
        best = new
    tau = np.arange(top) * step_ms
    obj = np.maximum(best / (gbps * 1e9) * 1e3, tau)
    t = int(np.argmin(obj))
    picks = []
    for l in reversed(range(L)):                    # walk back the choices
        m = int(choice[l, t])
        picks.append(names[m])
        t -= u[m, l]
    picks.reverse()
    i_s, i_p = int(single_serial.argmin()), int(single_pipe.argmin())
    return {"gbps": gbps,
            "serial": {"best_single": names[i_s], "single_ms": float(single_serial[i_s]), "mix_ms": mix_serial,
                       "mix_modes": [names[i] for i in pick_s]},
            "pipelined": {"best_single": names[i_p], "single_ms": float(single_pipe[i_p]), "mix_ms": float(obj.min()),
                          "mix_modes": picks}}


def wall_timer(fn, device, warmup=1, reps=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t)
    return statistics.median(times)
