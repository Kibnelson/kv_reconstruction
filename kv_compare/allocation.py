"""Bit allocation measured in AQUA-KV's streaming protocol itself, on our calibration text.

The lanes are split into groups: bands of `band` consecutive layers, keys or values (7 x 2 = 14 groups for 28
layers and bands of 4). From a base run (one kappa everywhere), each group in turn gets its step multiplied by
each move (for example x0.5 and x2), and the run measures what that costs or saves: the change in loss (nats per
scored token, float32 loss, the same windows as the base) and in bits (the real streams). No gradient and no
small-error assumption: the errors are the codec's own, read the way the protocol reads them.

The allocation then picks one move per group (or none) to minimise the bits with no more loss than the base,
assuming the groups' effects add up (they do at step 2: keys + values alone +0.55%, together +0.54%): for a
multiplier lam on bits, each group takes the option minimising d_loss + lam x d_bits; lam is scanned and the
cheapest choice whose summed d_loss is at most 0 is kept. The chosen multipliers become lane scales of the GPU
codec and are applied at several base steps in the WikiText-2 protocol.
"""
from __future__ import annotations

import math

import torch


def groups(L, C, band):
    """[(name, first layer, last layer + 1, half)]: half 0 = keys (lanes 0..C), 1 = values (C..2C)."""
    out = []
    for a in range(0, L, band):
        b = min(a + band, L)
        for h, nm in ((0, "keys"), (1, "values")):
            out.append((f"layers {a}-{b - 1} {nm}", a, b, h))
    return out


def lane_scale(L, C, grps, mults):
    """Kappa multipliers [L, 2C] (float64): mults[i] on the lanes of group i, 1 elsewhere."""
    s = torch.ones(L, 2 * C, dtype=torch.float64)
    for (_, a, b, h), m in zip(grps, mults):
        s[a:b, h * C:(h + 1) * C] = m
    return s


def choose(base, measured, moves):
    """base: {"loss": nats per token, "bits": bits per compressed value}; measured[i][m]: the same for group i with
    move m. Returns (multipliers, summed d_loss, summed d_bits, lam)."""
    opts = []
    for i, per in enumerate(measured):
        o = [(1.0, 0.0, 0.0)]
        for m in moves:
            if m in per:
                o.append((m, per[m]["loss"] - base["loss"], per[m]["bits"] - base["bits"]))
        opts.append(o)
    best = None
    for e in range(-60, 61):
        lam = 10 ** (e / 10)                     # nats per token per bit per value, 1e-6 .. 1e6
        pick = [min(o, key=lambda t: t[1] + lam * t[2]) for o in opts]
        dl, db = sum(p[1] for p in pick), sum(p[2] for p in pick)
        if dl <= 0 and (best is None or db < best[2]):
            best = ([p[0] for p in pick], dl, db, lam)
    if best is None:
        return [1.0] * len(opts), 0.0, 0.0, None
    return best


def change_pct(nats):
    return 100 * (math.exp(nats) - 1)
