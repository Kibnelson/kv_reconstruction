"""Our predictive KV codec: features, fits, channel order and per-word costs (lossless and closed-loop lossy).

Every BF16 value of layer l is predicted by a linear map of KV the receiver already holds, then coded against a
logistic distribution centred on the prediction. Configurations:

  prev       layer l-1 of the same token; whole layer in one step; static spread    (the "44%" version)
  block      + earlier channel groups of layer l (16 steps of 128 channels on a 2048-wide layer)
  nohint     + layer 0, layer l-2 and layer l at the token's most recent earlier occurrence; adaptive spread
             (the strongest no-hint version; repeated tokens force waves in a real decoder)
  fast       layer 0, l-2, l-1, the recomputed attention output of layer l-1 (queries predicted from the token's
             own layer l-1 KV, window = last 64 tokens + token 0), pivoted channel order, 8 steps, adaptive spread
             (the strong-and-fast version: no waves)
  fast_noatt the fast configuration without the attention input (used when no calibration queries were captured)

Fit: conditional mean of each channel given the inputs and the earlier groups, in closed form from the
calibration covariance (ridge 1e-3 x average variance). Spread: per-channel mean absolute residual on the scales
document / (2 ln 2). Adaptive spread: x clip((64 * previous-layer surprise + earlier-group surprise + 16) /
(64 + n + 16), 0.25, 4), where surprise = |x - prediction| / (2 ln 2 * spread).
Costs are exact -log2 P of the BF16 word (lossless) or of the quantized residual bin (lossy). Prediction runs in
float32 with TF32 disabled; fits in float64.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np
import torch

from .core import LN2X2, Rope, as_words, bin_bits, exact_bits, to_bf16, words_to_float


@dataclass(frozen=True)
class Config:
    name: str
    kinds: tuple
    steps: int
    order: str = "natural"
    adaptive: bool = True


CONFIGS = {
    "prev": Config("prev", ("P",), 1, "natural", False),
    "block": Config("block", ("P",), 16, "natural", False),
    "nohint": Config("nohint", ("L0", "P2", "R", "P"), 16, "natural", True),
    "fast": Config("fast", ("L0", "P2", "P", "ATT"), 8, "pivot", True),
    "fast_noatt": Config("fast_noatt", ("L0", "P2", "P"), 8, "pivot", True),
}
LABELS = {"prev": "Our 44% version (layer l-1)", "block": "Our Block (+ earlier groups)",
          "nohint": "Our strongest no-hint", "fast": "Our strong-and-fast",
          "fast_noatt": "Our strong-and-fast without attention input"}


def kinds_at(kinds, layer):
    out = []
    for k in kinds:
        if k in ("L0", "P2") and layer < 2:
            continue
        if k in ("P", "ATT") and layer < 1:
            continue
        out.append(k)
    return out


class Doc:
    """A captured document: BF16 words K/V [layers, tokens, Hk*hd] (cache order), token ids, optional queries."""

    def __init__(self, root, name):
        root = Path(root)
        self.name = name
        self.K = np.load(root / f"{name}_K_bits.npy", mmap_mode="r")
        self.V = np.load(root / f"{name}_V_bits.npy", mmap_mode="r")
        self.ids = np.load(root / f"{name}_ids.npy")
        q = root / f"{name}_Q_bits.npy"
        self.Q = np.load(q, mmap_mode="r") if q.is_file() else None
        self.layers, self.tokens, self.columns = self.K.shape

    def words(self, layer, device, rows=None):
        sl = slice(None) if rows is None else slice(0, rows)
        return torch.cat([as_words(np.asarray(self.K[layer][sl]), device),
                          as_words(np.asarray(self.V[layer][sl]), device)], 1)


def load_meta(root):
    return json.loads((Path(root) / "meta.json").read_text())


@dataclass
class Fit:
    A: torch.Tensor          # [W, ne+W] float32, rows in decode order, columns [inputs | current layer in decode order]
    b: torch.Tensor          # [W] float32
    perm: torch.Tensor       # decode position -> channel
    starts: torch.Tensor     # decode position -> start position of its group
    ne: int
    kinds: list
    spread: torch.Tensor = field(default=None)   # per channel (channel order)
    mu: torch.Tensor = field(default=None)       # calibration mean of the feature row (decode order), if kept


class Study:
    def __init__(self, root, device, train, tables, tests, max_layers=None, ridge=1e-3, log=print):
        self.root, self.device, self.log, self.ridge = Path(root), torch.device(device), log, ridge
        meta = load_meta(root)
        self.meta = meta
        self.Hk, self.hd, self.Hq = int(meta["kv_heads"]), int(meta["head_dim"]), int(meta["query_heads"])
        self.C = self.Hk * self.hd
        self.W = 2 * self.C
        self.rope = Rope(meta["inv_freq"], self.Hk, self.hd, meta.get("rope_layout", "half"), meta.get("attention_scaling", 1.0))
        self.rope_q = Rope(meta["inv_freq"], self.Hq, self.hd, meta.get("rope_layout", "half"), meta.get("attention_scaling", 1.0))
        self.train = [Doc(root, n) for n in train]
        self.tables = Doc(root, tables)
        self.tests = [Doc(root, n) for n in tests]
        self.L = self.tables.layers if max_layers is None else min(max_layers, self.tables.layers)
        partner = self.rope.pair_partner()
        self.partner = torch.as_tensor(np.concatenate([partner, np.arange(self.C, self.W)]), device=self.device)
        self._f, self._att, self._rep, self._qmap = {}, {}, {}, {}
        self.keep_qmaps = False      # True: forget() keeps the query maps (closed-loop coding reuses every layer's)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------ exact features
    def positions(self, n):
        return torch.arange(n, device=self.device)

    def feats(self, doc, layer, cache=True):
        """De-rotated float32 features of a captured layer. cache=False computes them without keeping them
        (one layer of one 8,192-token document of an 8B-shaped model is 67 MB)."""
        key = (doc.name, layer)
        if key in self._f:
            return self._f[key]
        x = words_to_float(doc.words(layer, self.device))
        f = self.rope.derotate_row(x, self.positions(x.shape[0])).to(torch.float32)
        if cache:
            self._f[key] = f
        return f

    def forget(self, keep_from):
        """Drop cached features of layers below keep_from (layer 0 is always kept) and attention inputs of layers
        up to keep_from (the next layer reads the attention of layer keep_from + 1)."""
        for key in [k for k in self._f if k[1] < keep_from and k[1] != 0]:
            del self._f[key]
        for key in [k for k in self._att if k[1] <= keep_from]:
            del self._att[key]
        if not self.keep_qmaps:
            for key in [k for k in self._qmap if k < keep_from - 1]:
                del self._qmap[key]

    def rep_index(self, doc):
        if doc.name not in self._rep:
            last, out = {}, np.full(doc.tokens, -1)
            for t, v in enumerate(doc.ids[: doc.tokens].tolist()):
                out[t] = last.get(v, -1)
                last[v] = t
            self._rep[doc.name] = torch.as_tensor(out, device=self.device)
        return self._rep[doc.name]

    # ------------------------------------------------------------------ attention input
    def qmap(self, layer):
        """Ridge least squares from the token's de-rotated layer KV to its pre-RoPE queries (calibration)."""
        if layer not in self._qmap:
            if any(d.Q is None for d in self.train):
                raise ValueError("the attention input needs calibration queries (capture with --queries)")
            d1 = self.W + 1
            G = torch.zeros(d1, d1, dtype=torch.float64, device=self.device)
            R = torch.zeros(d1, self.Hq * self.hd, dtype=torch.float64, device=self.device)
            n = 0
            for d in self.train:
                X = torch.cat([self.feats(d, layer, cache=False).to(torch.float64),
                               torch.ones(d.tokens, 1, dtype=torch.float64, device=self.device)], 1)
                Y = words_to_float(as_words(np.asarray(d.Q[layer]), self.device))
                G += X.T @ X
                R += X.T @ Y
                n += d.tokens
            G += 1e-3 * n * torch.eye(d1, dtype=torch.float64, device=self.device)
            self._qmap[layer] = torch.linalg.solve(G, R).to(torch.float32)
        return self._qmap[layer]

    def attention(self, x, f, layer, chunk=512):
        """Windowed attention of layer `layer` with predicted queries. x: post-RoPE KV values [n, W] (float),
        f: the same rows de-rotated. Returns [n, Hq*hd] float32."""
        n = x.shape[0]
        G = self.Hq // self.Hk
        Wq = self.qmap(layer)
        qd = torch.cat([f.to(torch.float32), torch.ones(n, 1, device=self.device)], 1) @ Wq
        Q = self.rope_q.rotate_heads(qd.to(torch.float64), self.positions(n)).to(torch.float32).reshape(n, self.Hk, G, self.hd)
        K = x[:, : self.C].to(torch.float32).reshape(n, self.Hk, self.hd)
        V = x[:, self.C:].to(torch.float32).reshape(n, self.Hk, self.hd)
        out = torch.empty(n, self.Hk, G, self.hd, device=self.device)
        scale = 1.0 / math.sqrt(self.hd)
        for t0 in range(0, n, chunk):
            t = torch.arange(t0, min(n, t0 + chunk), device=self.device)
            idx = t[:, None] - torch.arange(64, device=self.device)[None, :]
            idx = torch.cat([torch.zeros_like(t)[:, None], idx], 1)
            valid = idx >= 0
            valid[:, 1:] &= idx[:, 1:] != 0            # token 0 appears once
            ic = idx.clamp_min(0)
            sc = torch.einsum("nkgd,nwkd->nkgw", Q[t], K[ic]) * scale
            sc = sc.masked_fill(~valid[:, None, None, :], float("-inf"))
            p = torch.softmax(sc, -1)
            out[t] = torch.einsum("nkgw,nwkd->nkgd", p, V[ic])
        return out.reshape(n, self.Hq * self.hd)

    def att_feature(self, doc, layer):
        key = (doc.name, layer)
        if key not in self._att:
            x = words_to_float(doc.words(layer, self.device))
            self._att[key] = self.attention(x, self.feats(doc, layer), layer)
        return self._att[key]

    # ------------------------------------------------------------------ inputs
    def extra(self, doc, layer, kinds, provider=None):
        """Non-current inputs. provider(j) -> (post-RoPE values float64, de-rotated float32) of layer j,
        used for closed-loop (reconstructed) KV; default: exact KV."""
        cols = []
        for k in kinds:
            if k == "L0":
                cols.append(provider(0)[1] if provider else self.feats(doc, 0))
            elif k == "P2":
                cols.append(provider(layer - 2)[1] if provider else self.feats(doc, layer - 2))
            elif k == "P":
                cols.append(provider(layer - 1)[1] if provider else self.feats(doc, layer - 1))
            elif k == "R":
                if provider:
                    raise ValueError("the earlier-occurrence input is not supported in closed-loop coding")
                r = self.rep_index(doc)
                flag = (r >= 0).to(torch.float32)[:, None]
                cols += [self.feats(doc, layer)[r.clamp_min(0)] * flag, flag]
            elif k == "ATT":
                if provider:
                    x, f = provider(layer - 1)
                    cols.append(self.attention(x, f, layer - 1))
                else:
                    cols.append(self.att_feature(doc, layer - 1))
        n = doc.tokens if provider is None else provider(0)[0].shape[0]
        return torch.cat(cols, 1) if cols else torch.zeros(n, 0, device=self.device)

    # ------------------------------------------------------------------ order and groups
    def pivot_order(self, S, ne):
        """Greedy: the unit (RoPE key pair or value channel) that explains the most remaining conditional
        variance of the layer goes next."""
        See, Sec, Scc = S[:ne, :ne], S[:ne, ne:], S[ne:, ne:]
        if ne:
            M = See.clone()                          # See + reg I without a separate identity matrix
            M.diagonal().add_(1e-6 * torch.trace(See) / ne)
            Sc = Scc - Sec.T @ torch.linalg.solve(M, Sec)
            del M
        else:
            Sc = Scc.clone()
        W = self.W
        partner = self.partner.cpu().numpy()
        a_list, b_list = [], []
        for c in range(W):
            if c < self.C:
                if c < partner[c]:
                    a_list.append(c); b_list.append(int(partner[c]))
            else:
                a_list.append(c); b_list.append(-1)
        a = torch.as_tensor(a_list, device=S.device)
        b = torch.as_tensor(b_list, device=S.device)
        pair = b >= 0
        bb = b.clamp_min(0)
        alive = torch.ones(len(a_list), dtype=torch.bool, device=S.device)
        order = []
        tiny = 1e-30
        for _ in range(len(a_list)):
            col2 = (Sc * Sc).sum(0)
            diag = Sc.diagonal().clamp_min(tiny)
            red = col2[a] / diag[a]
            sab = Sc[a, bb]
            cab = (Sc[:, a] * Sc[:, bb]).sum(0)
            det = (diag[a] * diag[bb] - sab * sab).clamp_min(tiny)
            red_pair = (diag[bb] * col2[a] - 2 * sab * cab + diag[a] * col2[bb]) / det
            red = torch.where(pair, red_pair, red)
            red = torch.where(alive, red, torch.full_like(red, -math.inf))
            u = int(torch.argmax(red))
            alive[u] = False
            idx = [int(a[u])] + ([int(b[u])] if int(b[u]) >= 0 else [])
            order += idx
            ix = torch.as_tensor(idx, device=S.device)
            M = Sc[ix][:, ix] + tiny * torch.eye(len(idx), dtype=Sc.dtype, device=Sc.device)
            Sc = Sc - Sc[:, ix] @ torch.linalg.solve(M, Sc[ix, :])
        return torch.as_tensor(order, device=S.device)

    def group_starts(self, perm, steps):
        W = self.W
        gsz = -(-W // steps)
        pos = torch.empty(W, dtype=torch.long, device=perm.device)
        pos[perm] = torch.arange(W, device=perm.device)
        start = (torch.arange(W, device=perm.device) // gsz) * gsz
        partner_pos = pos[self.partner[perm]]                      # decode position of each position's partner
        start = torch.minimum(start, start[partner_pos])           # a key pair shares the earlier group
        return start

    # ------------------------------------------------------------------ fit
    def covariance(self, layer, kinds, chunk=1024, with_amax=False):
        """Mean and covariance of [inputs | current layer] over the calibration documents, one document at a time
        (two passes: sums, then centred products). On CUDA a document's centred product is accumulated in float32
        (TF32 off) and added to the float64 total; on CPU everything is float64. Peak memory is the float64
        matrix plus one float32 matrix and one document's inputs."""
        dev, f64 = self.device, torch.float64
        prod = torch.float32 if dev.type == "cuda" else f64
        n, s1, amax = 0, None, None
        for d in self.train:
            Xe, Xc = self.extra(d, layer, kinds), self.feats(d, layer)
            part = torch.cat([Xe.to(f64).sum(0), Xc.to(f64).sum(0)])
            s1 = part if s1 is None else s1 + part
            if with_amax:
                m = torch.cat([Xe.abs().amax(0), Xc.abs().amax(0)]).to(f64)
                amax = m if amax is None else torch.maximum(amax, m)
            n += d.tokens
            del Xe
        mu = s1 / n
        dim = mu.numel()
        S = torch.zeros(dim, dim, dtype=f64, device=dev)
        mu_p = mu.to(prod)
        for d in self.train:
            Xe, Xc = self.extra(d, layer, kinds), self.feats(d, layer)
            acc = S if prod == f64 else torch.zeros(dim, dim, dtype=prod, device=dev)
            for r0 in range(0, d.tokens, chunk):
                X = torch.cat([Xe[r0:r0 + chunk], Xc[r0:r0 + chunk]], 1).to(prod) - mu_p
                acc.addmm_(X.T, X)
            if acc is not S:
                S.add_(acc)
            del Xe, acc
        S /= n
        return (mu, S, amax) if with_amax else (mu, S)

    def fit(self, layer, cfg):
        """Block-causal conditional-mean coefficients: channel j (decode order) is predicted from the inputs and
        the channels of earlier groups, a_j = L[j, :s] L[:s, :s]^-1 with S = L L^T (ridged). Solved with one
        triangular solve; no explicit inverse."""
        kinds = kinds_at(cfg.kinds, layer)
        mu, S = self.covariance(layer, kinds)
        box = [S]
        del S                                        # solve() frees the matrix after factorising it
        return self.solve(mu, box, kinds, cfg.steps, cfg.order)

    def solve(self, mu, box, kinds, steps, order, perm=None, keep_mu=False):
        """Coefficients from a mean and covariance of [inputs | current layer (channel order)]. box = [S]: the
        matrix is taken out of the list, overwritten and freed after the Cholesky factorisation. perm: a
        precomputed pivoted order (the order depends only on the inputs, not on the number of steps)."""
        dev, f64 = self.device, torch.float64
        S = box.pop()
        dim = S.shape[0]
        ne = dim - self.W
        if perm is None:
            perm = self.pivot_order(S, ne) if order == "pivot" else torch.arange(self.W, device=dev)
        cur = ne + perm
        S[ne:, :] = S[cur, :]                       # current-layer block into decode order, in place
        S[:, ne:] = S[:, cur]
        mu = torch.cat([mu[:ne], mu[cur]])
        S.diagonal().add_(self.ridge * float(torch.trace(S)) / dim)
        Lc = torch.linalg.cholesky(S)
        del S
        starts = self.group_starts(perm, steps)
        # row j keeps L[j, :s_j]; solving X L = that row gives exact zeros beyond s_j (L is lower triangular)
        B = Lc[ne:] * (torch.arange(dim, device=dev)[None, :] < (ne + starts)[:, None])
        A = torch.linalg.solve_triangular(Lc, B, upper=False, left=False)
        del Lc, B
        b = mu[ne:] - A @ mu
        return Fit(A.to(torch.float32), b.to(torch.float32), perm, starts, ne, kinds, mu=mu if keep_mu else None)

    def predict(self, F, Xe, cur_perm):
        """Prediction for every channel (channel order), post-RoPE, float64. cur_perm: current layer features in
        decode order (only earlier groups are used by the block-causal coefficients)."""
        md_perm = torch.cat([Xe, cur_perm], 1) @ F.A.T + F.b
        md = torch.empty_like(md_perm)
        md[:, F.perm] = md_perm
        return self.rope.rotate_row(md.to(torch.float64), self.positions(md.shape[0]))

    def predict_exact(self, F, doc, layer):
        return self.predict(F, self.extra(doc, layer, F.kinds), self.feats(doc, layer)[:, F.perm])

    def static_spread(self, F, layer):
        x = words_to_float(self.tables.words(layer, self.device))
        m = self.predict_exact(F, self.tables, layer)
        return (x - m).abs().mean(0) / LN2X2 + 1e-12

    def multiplier(self, F, q, prior):
        """Adaptive spread multiplier per value. q: surprise [n, W] in channel order; prior: [n]."""
        qp = q[:, F.perm]
        cs = torch.cat([torch.zeros(q.shape[0], 1, dtype=q.dtype, device=q.device), torch.cumsum(qp, 1)[:, :-1]], 1)
        earlier = cs[:, F.starts]
        mult_p = ((64 * prior[:, None] + earlier + 16) / (64 + F.starts[None, :].to(q.dtype) + 16)).clamp(0.25, 4.0)
        mult = torch.empty_like(mult_p)
        mult[:, F.perm] = mult_p
        return mult

    # ------------------------------------------------------------------ lossless costs
    def lossless(self, cfg, fits=None):
        """Per-token mean bits [L, N] for every test document (exact costs)."""
        out = {d.name: torch.zeros(self.L, d.tokens, dtype=torch.float64) for d in self.tests}
        prior = {d.name: torch.ones(d.tokens, dtype=torch.float64, device=self.device) for d in self.tests}
        for layer in range(self.L):
            F = self.fit(layer, cfg)
            F.spread = self.static_spread(F, layer)
            if fits is not None:
                fits[layer] = self.on_device(F, torch.device("cpu"))
            for d in self.tests:
                w = d.words(layer, self.device)
                x = words_to_float(w)
                m = self.predict_exact(F, d, layer)
                q = (x - m).abs() / (LN2X2 * F.spread)
                s = F.spread * (self.multiplier(F, q, prior[d.name]) if cfg.adaptive else 1.0)
                out[d.name][layer] = exact_bits(w, m, s).mean(1).cpu()
                prior[d.name] = q.mean(1)
            self.forget(layer - 1)
            self.log(f"  {cfg.name}: layer {layer} done")
        return out, fits

    def on_device(self, F, device=None):
        device = self.device if device is None else device
        return Fit(F.A.to(device), F.b.to(device), F.perm.to(device), F.starts.to(device), F.ne, F.kinds,
                   None if F.spread is None else F.spread.to(device))

    # ------------------------------------------------------------------ closed-loop lossy coding
    def lossy(self, cfg, fits, doc, rows, kappa):
        """Code the first `rows` tokens of doc closed loop: every prediction uses the receiver's reconstructed KV.
        Residuals are quantized with step kappa * spread (per channel), entropy-coded against the adaptive
        logistic model, and reconstructed to BF16. Returns (reconstructed values [L, rows, W] float32 on CPU,
        bits per value)."""
        if "R" in cfg.kinds:
            raise ValueError("closed-loop coding uses configurations without the earlier-occurrence input")
        rec, frec = {}, {}
        provider = lambda j: (rec[j], frec[j])
        prior = torch.ones(rows, dtype=torch.float64, device=self.device)
        total = 0.0
        pos_all = self.positions(rows)
        out = torch.empty(self.L, rows, self.W, dtype=torch.float32)
        for layer in range(self.L):
            F = self.on_device(fits[layer])
            s = F.spread
            x = words_to_float(doc.words(layer, self.device, rows))
            Xe = self.extra(doc, layer, F.kinds, provider) if F.kinds else torch.zeros(rows, 0, device=self.device)
            xrec = torch.zeros(rows, self.W, dtype=torch.float64, device=self.device)
            cur_perm = torch.zeros(rows, self.W, dtype=torch.float32, device=self.device)
            qsum = torch.zeros(rows, dtype=torch.float64, device=self.device)
            for st in torch.unique(F.starts).tolist():
                pos = torch.nonzero(F.starts == st).flatten()
                ch = F.perm[pos]
                md_pos = torch.cat([Xe, cur_perm], 1) @ F.A[pos].T + F.b[pos]
                md = torch.zeros(rows, self.W, dtype=torch.float64, device=self.device)
                md[:, ch] = md_pos.to(torch.float64)
                m = self.rope.rotate_row(md, pos_all)[:, ch]
                mult = ((64 * prior[:, None] + qsum[:, None] + 16) / (64 + st + 16)).clamp(0.25, 4.0)
                se = s[ch][None, :] * mult
                step = kappa * s[ch][None, :]
                k = torch.round((x[:, ch] - m) / step)
                xh = to_bf16(m + k * step)
                total += float(bin_bits(k, step, se).sum())
                q = (xh - m).abs() / (LN2X2 * s[ch][None, :])
                qsum += q.sum(1)
                xrec[:, ch] = xh
                cur_perm[:, pos] = self.rope.derotate_row(xrec, pos_all)[:, ch].to(torch.float32)
            prior = qsum / self.W
            rec[layer] = xrec
            frec[layer] = self.rope.derotate_row(xrec, pos_all).to(torch.float32)
            for j in [j for j in rec if j < layer - 1 and j != 0]:
                del rec[j], frec[j]
            out[layer] = xrec.to(torch.float32).cpu()
        return out, total / (self.L * rows * self.W)


def macs_per_value(cfg_name, Hk, hd, Hq):
    """Predictor multiply-adds per coded value for a configuration (entropy coder excluded)."""
    W = 2 * Hk * hd
    Qw = Hq * hd
    cfg = CONFIGS[cfg_name]
    gsz = -(-W // cfg.steps)
    earlier = (cfg.steps - 1) / 2 * gsz
    if cfg_name == "prev":
        return W
    if cfg_name == "block":
        return W + earlier
    if cfg_name == "nohint":
        return 4 * W + 1 + earlier
    att = Qw + Qw + Hq * 65 * hd * 2 / W if "ATT" in cfg.kinds else 0
    return 3 * W + earlier + att
