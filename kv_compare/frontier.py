"""Bits against compute: many predictor variants fitted from one calibration covariance per layer.

For each layer, the covariance of [layer 0, layer l-2, layer l-1, attention input of layer l-1 | layer l] over
the calibration documents is computed once. Every variant is fitted from its sub-matrix and scored with exact
lossless costs on the test documents, exactly as the lossless stage scores the named configurations. Variants
change four things:

  inputs     which of layer 0, l-2, l-1 and the attention input are used
  steps      channel groups per layer (1 = whole layer at once; more steps use more of the layer's own channels)
  rank       inputs first projected to r directions (reduced-rank regression, outputs weighted by their
             input-only residual variance), so each value costs r multiply-adds plus a shared projection
  formats    coefficients rounded to 8 bits per output channel and part (inputs | earlier groups), features
             rounded to 16 or 8 bits per channel (amax scale from calibration): the integer formats a GPU can run
             exactly on its 8-bit tensor cores (16-bit features = 2 products per multiply-add, 8-bit = 1).
             Projected inputs are kept unrounded.

Multiply-adds per value count the inputs (or the projection plus r), the earlier groups (average group start),
and for the attention input the query map and the windowed attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time

import numpy as np
import torch

from .core import LN2X2, exact_bits, words_to_float
from .predict import Fit, kinds_at

SUPERSET = ("L0", "P2", "P", "ATT")
NAMES = {"L0": "layer 0", "P2": "layer l-2", "P": "layer l-1", "ATT": "attention of l-1"}


@dataclass(frozen=True)
class Variant:
    kinds: tuple
    steps: int
    order: str = "pivot"
    rank: int | None = None
    coef_bits: int | None = None
    feat_bits: int | None = None
    adaptive: bool = True

    @property
    def label(self):
        parts = ["+".join(self.kinds), f"{self.steps} {self.order} step{'s' if self.steps > 1 else ''}"]
        if self.rank:
            parts.append(f"rank {self.rank}")
        if self.coef_bits:
            parts.append(f"int{self.coef_bits} coef x int{self.feat_bits} feat")
        if not self.adaptive:
            parts.append("static spread")
        return ", ".join(parts)

    @property
    def products(self):
        """8-bit integer tensor-core products per multiply-add (None: floating point)."""
        if not self.coef_bits:
            return None
        return (self.coef_bits // 8) * (self.feat_bits // 8)


@dataclass
class VFit:
    F: Fit
    proj: torch.Tensor = None        # [ne, r] float64: z = (x - proj_mu) @ proj
    proj_mu: torch.Tensor = None
    in_scale: torch.Tensor = None    # per input channel (feature rounding)
    cur_scale: torch.Tensor = None   # per current-layer channel, decode order


def default_variants(with_att=True, quick=False):
    sets = [("P",), ("L0", "P"), ("L0", "P2", "P")] + ([("L0", "P2", "P", "ATT")] if with_att else [])
    V = [Variant(("P",), 1, "natural", adaptive=False), Variant(("P",), 16, "natural", adaptive=False)]
    steps = (1, 8) if quick else (1, 4, 8, 16)
    for k in sets:
        V += [Variant(k, st) for st in steps]
    ranks = (128,) if quick else (64, 128, 256, 512)
    for k in sets[2:]:
        V += [Variant(k, 8, rank=r) for r in ranks]
    V += [Variant(("P",), 16 if not quick else 8, rank=r) for r in ((256,) if quick else (128, 256, 512))]
    bases = [Variant(("P",), 16 if not quick else 8), Variant(("L0", "P2", "P"), 8),
             Variant(("L0", "P2", "P"), 8, rank=128 if quick else 256)]
    bases += [Variant(("L0", "P2", "P", "ATT"), 8)] if with_att else []
    for b in bases:
        V += [replace(b, coef_bits=8, feat_bits=16), replace(b, coef_bits=8, feat_bits=8)]
    out, seen = [], set()
    for v in V:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _round_sym(M, bits, dim):
    q = 2 ** (bits - 1) - 1
    sc = M.abs().amax(dim, keepdim=True).clamp_min(1e-30) / q
    return torch.round(M / sc) * sc


def _round_feat(X, scale):
    return torch.round(X / scale) * scale


class Frontier:
    def __init__(self, study, variants, log=print):
        self.s, self.variants, self.log = study, variants, log
        Wd = {"L0": study.W, "P2": study.W, "P": study.W, "ATT": study.Hq * study.hd}
        self.width = Wd

    # ------------------------------------------------------------------ helpers
    def columns(self, kinds_all, kinds):
        off, pos = {}, 0
        for k in kinds_all:
            off[k] = pos
            pos += self.width[k]
        dev = self.s.device
        cols = [torch.arange(off[k], off[k] + self.width[k], device=dev) for k in kinds]
        cols.append(torch.arange(pos, pos + self.s.W, device=dev))
        return torch.cat(cols)

    def reduced(self, mu, S, ne, ranks):
        """Reduced-rank projections of the inputs for every rank in `ranks`, each with the covariance of
        [z | current layer]."""
        st = self.s
        Sxx, Sxc = S[:ne, :ne], S[:ne, ne:]
        avg = float(torch.trace(Sxx)) / ne
        M = Sxx.clone()
        M.diagonal().add_(st.ridge * avg)
        Lx = torch.linalg.cholesky(M)
        del M
        G = torch.linalg.solve_triangular(Lx, Sxc, upper=False)            # whitened cross-covariance [ne, W]
        resid = (S.diagonal()[ne:] - (G * G).sum(0)).clamp_min(1e-30)       # input-only residual variance
        H = G / resid.sqrt()[None, :]
        del G
        evals, evecs = torch.linalg.eigh(H.T @ H)
        out = {}
        for r in ranks:
            if r >= min(ne, st.W):
                continue
            U = H @ evecs[:, -r:]
            U = U / U.norm(dim=0, keepdim=True).clamp_min(1e-30)
            P = torch.linalg.solve_triangular(Lx.T, U, upper=True) * math.sqrt(avg)   # z = (x - mu) P
            dim = r + st.W
            Sz = torch.empty(dim, dim, dtype=S.dtype, device=S.device)
            Sz[:r, :r] = P.T @ Sxx @ P
            Sz[:r, r:] = P.T @ Sxc
            Sz[r:, :r] = Sz[:r, r:].T
            Sz[r:, r:] = S[ne:, ne:]
            muz = torch.cat([torch.zeros(r, dtype=mu.dtype, device=mu.device), mu[ne:]])
            out[r] = (P, mu[:ne].clone(), muz, Sz)
        del H, Lx
        return out

    def solve(self, mu, S, kinds, steps, order, perm_cache, key):
        st = self.s
        if order == "pivot" and key not in perm_cache:
            perm_cache[key] = st.pivot_order(S, S.shape[0] - st.W)
        perm = perm_cache.get(key) if order == "pivot" else None
        return st.solve(mu, [S.clone()], kinds, steps, order, perm=perm, keep_mu=True)

    def formats(self, vf, v, amax_in, amax_cur):
        """Round coefficients (per output channel and part) and features to the variant's integer formats."""
        F = vf.F
        ne = F.ne
        A = F.A.double()
        Aq = torch.cat([_round_sym(A[:, :ne], v.coef_bits, 1) if ne else A[:, :0],
                        _round_sym(A[:, ne:], v.coef_bits, 1)], 1)
        b = F.b.double() + (A - Aq) @ F.mu                                  # keep the conditional mean
        q = 2 ** (v.feat_bits - 1) - 1
        cur_scale = (amax_cur[F.perm] / q).clamp_min(1e-30).to(torch.float32)
        in_scale = (amax_in / q).clamp_min(1e-30).to(torch.float32) if amax_in.numel() else None
        proj = _round_sym(vf.proj, v.coef_bits, 0) if vf.proj is not None else None
        G = Fit(Aq.to(torch.float32), b.to(torch.float32), F.perm, F.starts, F.ne, F.kinds, mu=F.mu)
        return VFit(G, proj, vf.proj_mu, in_scale, cur_scale)

    def predict(self, vf, doc, layer):
        st, F = self.s, vf.F
        Xe = st.extra(doc, layer, F.kinds)
        if vf.in_scale is not None:
            Xe = _round_feat(Xe, vf.in_scale)
        if vf.proj is not None:
            Xe = ((Xe.double() - vf.proj_mu) @ vf.proj).to(torch.float32)
        cur = st.feats(doc, layer)[:, F.perm]
        if vf.cur_scale is not None:
            cur = _round_feat(cur, vf.cur_scale)
        return st.predict(F, Xe, cur)

    def macs(self, v, vf):
        st = self.s
        ne = sum(self.width[k] for k in v.kinds)
        inputs = ne * v.rank / st.W + v.rank if v.rank and v.rank < min(ne, st.W) else ne
        m = inputs + float(vf.F.starts.double().mean())
        if "ATT" in v.kinds:
            Qw = st.Hq * st.hd
            m += (st.W + 1) * Qw / st.W + st.Hq * 65 * st.hd * 2 / st.W
        return m

    # ------------------------------------------------------------------ run
    def run(self):
        st = self.s
        with_att = all(d.Q is not None for d in st.train)
        variants = list(dict.fromkeys(v for v in self.variants if with_att or "ATT" not in v.kinds))
        superset = tuple(k for k in SUPERSET if with_att or k != "ATT")
        bits = {v: {d.name: torch.zeros(st.L, d.tokens, dtype=torch.float32) for d in st.tests} for v in variants}
        prior = {v: {d.name: torch.ones(d.tokens, dtype=torch.float64, device=st.device) for d in st.tests}
                 for v in variants}
        macs = {}
        t0 = time.time()
        for layer in range(st.L):
            kinds_all = kinds_at(superset, layer)
            mu, S, amax = st.covariance(layer, kinds_all, with_amax=True)
            ne_all = S.shape[0] - st.W
            fits, perm_cache = {}, {}
            by_kinds = {}
            for v in variants:
                by_kinds.setdefault(tuple(kinds_at(v.kinds, layer)), []).append(v)
            for kinds, vs in by_kinds.items():
                cols = self.columns(kinds_all, kinds)
                Sv = S if tuple(kinds) == tuple(kinds_all) else S.index_select(0, cols).index_select(1, cols)
                muv, amv = mu[cols], amax[cols]
                ne = Sv.shape[0] - st.W
                ranks = sorted({v.rank for v in vs if v.rank and v.rank < min(ne, st.W)})
                red = self.reduced(muv, Sv, ne, ranks) if ranks else {}
                for v in vs:
                    base = replace(v, coef_bits=None, feat_bits=None)
                    r = v.rank if v.rank and v.rank < min(ne, st.W) else None
                    fkey = (kinds, v.steps, v.order, r)
                    if fkey not in fits:
                        if r is None:
                            F = self.solve(muv, Sv, kinds, v.steps, v.order, perm_cache, (kinds, None))
                            fits[fkey] = VFit(F)
                        else:
                            P, pmu, muz, Sz = red[r]
                            F = self.solve(muz, Sz, kinds, v.steps, v.order, perm_cache, (kinds, r))
                            fits[fkey] = VFit(F, P, pmu)
                    vf = fits[fkey]
                    if v.coef_bits:
                        vf = self.formats(vf, v, amv[:ne], amv[ne:])
                    self.score(v, vf, layer, bits, prior)
                    if layer == st.L - 1:
                        macs[v] = self.macs(v, vf) if (r is not None or not v.rank) else None   # None: rank >= width
                del Sv, red
            del S, fits
            st.forget(layer - 1)
            self.log(f"  frontier: layer {layer} done ({len(variants)} variants, {time.time() - t0:.0f} s)")
        return variants, bits, macs

    def score(self, v, vf, layer, bits, prior):
        st, F = self.s, vf.F
        x = words_to_float(st.tables.words(layer, st.device))
        F.spread = (x - self.predict(vf, st.tables, layer)).abs().mean(0) / LN2X2 + 1e-12
        for d in st.tests:
            w = d.words(layer, st.device)
            x = words_to_float(w)
            m = self.predict(vf, d, layer)
            q = (x - m).abs() / (LN2X2 * F.spread)
            s = F.spread * (st.multiplier(F, q, prior[v][d.name]) if v.adaptive else 1.0)
            bits[v][d.name][layer] = exact_bits(w, m, s).mean(1).to(torch.float32).cpu()
            prior[v][d.name] = q.mean(1)


def summarize(variants, bits, macs, spans):
    rows = []
    for v in variants:
        if macs.get(v) is None:          # reduced rank not below the layer width: same as the full-rank variant
            continue
        per = {d: b.numpy() for d, b in bits[v].items()}
        whole = {d: float(np.mean(b)) for d, b in per.items()}
        chunks = [float(np.mean(b[:, a:e])) for b in per.values() for a, e in spans]
        rows.append({"label": v.label, "kinds": list(v.kinds), "steps": v.steps, "order": v.order, "rank": v.rank,
                     "coef_bits": v.coef_bits, "feat_bits": v.feat_bits, "adaptive": v.adaptive,
                     "products_per_mac": v.products, "macs_per_value": macs.get(v),
                     "whole": whole, "whole_average": float(np.mean(list(whole.values()))),
                     "chunk_average": float(np.mean(chunks)),
                     "per_layer": {d: [float(x) for x in np.mean(b, 1)] for d, b in per.items()}})
    rows.sort(key=lambda r: (r["macs_per_value"] or 0, r["chunk_average"]))
    best = math.inf
    for r in rows:                       # Pareto among floating-point variants: fewer bits than every cheaper one
        r["pareto"] = r["coef_bits"] is None and r["chunk_average"] < best - 1e-9
        if r["coef_bits"] is None:
            best = min(best, r["chunk_average"])
    return rows
