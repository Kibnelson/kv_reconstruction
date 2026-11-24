#!/usr/bin/env python3
"""The region graph's deciding test: on pages with an earlier copy, does T plus the base beat T alone?

  python3 scripts/pd_region.py --capture CAPTURE_DIR --run RUN_DIR --codecs-file pd_codecs.pt --model-dir MODEL \
      --out OUT_DIR

Runs on one GPU (MATRIX=region: after setup, no vLLM engines), over a capture kept by MATRIX=capture and its run's
records. The pages: the capture's distinct complete blocks that have a base (scripts/pd_page_compare.py collect): the
latest earlier copy of the same 16 tokens (keys turned by RoPE to the new place), else the latest earlier turn's page
at the same place. Layers 1..27 of every such page are coded with T's value-slice coder (kv_compare.gpu_coder
.SliceCoder, T's uniform mode) against:

  T, its spread     T's own prediction and spread (each token's layer l from its own layer l-1 and, in head groups,
                    the same layer's earlier groups: kv_compare.layerwise, the kernels' arithmetic), from the page's
                    own words at its own positions: T as it codes now, layer 0 recomputed as T2 does
  T, fitted         the same prediction, the spread fitted as below
  base              the base's words, keys turned (the diff graph's delta), the spread fitted without T's prior (as
                    scripts/pd_diffgraph.py)
  base, prior       the same, the spread also scaled by T's per-token prior
  blend             mu_T + w (mu_base - mu_T), w per layer, K/V and agreement bucket (how long the text before the
                    token has agreed with the base's), least squares; the spread fitted

Fitted spreads: scale per layer, K/V and channel, times a factor per layer, K/V and agreement bucket, times T's
per-token multiplier (its prior: how surprising the token's previous layer was; the decoder has it), from mean |r|
(logistic: 2 ln 2 s). Weights and spreads are fitted on the even conversations; sizes are reported on the odd ones
(held out) and on all. Every coded slice is decoded and checked. Pages are coded BATCH at a time (one call per layer
group and K/V), which also times a batched delta's decode. A check: T through this coder against T2's own size on the
same tokens (whole requests, a multiple of 256 tokens).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pd_delta_analysis as da  # noqa: E402
import pd_diffgraph as dg  # noqa: E402
import pd_page_compare as pg  # noqa: E402

np = da.np
LN2 = math.log(2.0)
BATCH = 64
GROUPS = ((1, 4), (5, 13), (14, 27))         # layer groups, inclusive (layers 1..L-1)
VARIANTS = ("t_own", "t_fit", "base", "base_prior", "blend")
LABEL = {"t_own": "T, its own spread (as now)", "t_fit": "T, fitted spread", "base": "Base (the diff graph's delta)",
         "base_prior": "Base, spread with T's prior", "blend": "Blend: T + w (base - T)"}
CHECK_REQUESTS = 6
OTHER_TEXT = len(dg.BUCKETS)                 # the agreement buckets, then: the base at the same place, other text


def mult(prior):
    """T's spread multiplier per token from its prior (kv_compare.fused._mult)."""
    return ((prior * 64.0 + 16.0) * 0.0125).clamp(0.25, 4.0)


def groups_of(L):
    out = []
    for a, b in GROUPS:
        a, b = max(a, 1), min(b, L - 1)
        if a <= b:
            out.append((a, b))
    return out


class TPredictor:
    """T's prediction and spread for tokens at any positions, from their own words, batched: every layer (layer 0
    from the bias alone, and in head groups its earlier groups, as T codes it when it is not recomputed)."""

    def __init__(self, ours):
        from kv_compare import layerwise as lw
        from kv_compare.pd import codecs as cdm
        self.o, self.lw, self.cdm = ours, lw, cdm

    def __call__(self, words, positions):
        """words [L, n, W] int16 and positions [n] (long) on the device -> mu, s [L, n, W] float32 and T's spread
        multiplier per token [L, n] (its prior)."""
        import torch
        o = self.o
        L, n, W = words.shape
        pad = self.cdm.PAD                           # the codec's own padding (a multiple of its programs)
        Tp = -(-n // pad) * pad
        pos = torch.zeros(Tp, dtype=torch.long, device=words.device)
        pos[:n] = positions
        cos, sin = o.cos[pos].contiguous(), o.sin[pos].contiguous()
        c, _kp, Tp2 = o.codec(Tp, 0, rope=(cos, sin))
        if Tp2 != Tp:
            raise ValueError(f"T padded {Tp} tokens to {Tp2}")
        wp = words if Tp == n else torch.cat([words, torch.zeros(L, Tp - n, W, dtype=words.dtype,
                                                                 device=words.device)], 1)
        xall, prior = c.model(wp.contiguous())
        acc = torch.zeros(Tp, W, dtype=torch.int32, device=words.device)
        mu = torch.empty(L, n, W, dtype=torch.float32, device=words.device)
        s = torch.empty_like(mu)
        m = torch.empty(L, n, dtype=torch.float32, device=words.device)
        k = c.k
        for layer in range(L):
            c._acc_full(xall, layer, acc)
            mu[layer] = self.lw._mu_from_acc(acc, k["sa"][layer], k["b"][layer], k["cos"], k["sin"], o.C, o.hd)[:n]
            m[layer] = mult(prior[layer, :n])
            s[layer] = k["sp"][layer][None, :].float() * m[layer][:, None]
        return mu, s, m


class Fit:
    """Blend weights and fitted spreads, accumulated on the device (float64)."""

    def __init__(self, L1, C, nb, device):
        import torch
        z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=device)
        self.L1, self.C, self.nb = L1, C, nb
        self.num, self.den = z(L1, 2, nb), z(L1, 2, nb)
        self.sums = {v: (z(L1, 2, C), z(L1, 2, nb), z(L1, 2, nb)) for v in ("t_fit", "base", "base_prior", "blend")}
        self.cnt = collections.Counter()
        self.w = None

    @staticmethod
    def _kv(x, C):                                   # [L1, n, W] -> [L1, n, 2, C]
        return x.reshape(x.shape[0], x.shape[1], 2, C)

    def add_weights(self, tgt, mu_t, mu_b, buckets):
        """tgt [L1, n, W] float32 (the target's values), buckets [n] long."""
        r, d = self._kv(tgt - mu_t, self.C).double(), self._kv(mu_b - mu_t, self.C).double()
        num, den = (r * d).sum(3), (d * d).sum(3)        # [L1, n, 2]
        self.num.index_add_(2, buckets, num.permute(0, 2, 1))
        self.den.index_add_(2, buckets, den.permute(0, 2, 1))

    def solve(self):
        import torch
        self.w = torch.where(self.den > 0, self.num / self.den.clamp_min(1e-300), torch.zeros_like(self.num))
        return self.w

    def blend(self, mu_t, mu_b, buckets):
        w = self.w[:, :, buckets].permute(0, 2, 1).float()          # [L1, n, 2]
        d = self._kv(mu_b - mu_t, self.C)
        return (self._kv(mu_t, self.C) + w[..., None] * d).reshape(mu_t.shape)

    def add_spread(self, v, tgt, mu, m, buckets):
        """m [L1, n]: T's multiplier (1 for the variant fitted without it)."""
        import torch
        sc, bs_, bc = self.sums[v]
        r = (self._kv(tgt - mu, self.C).abs() / m[:, :, None, None]).double()   # [L1, n, 2, C]
        sc += r.sum(1)
        mm = r.mean(3)                                                       # [L1, n, 2]
        bs_.index_add_(2, buckets, mm.permute(0, 2, 1))
        bc.index_add_(2, buckets, torch.ones_like(mm).permute(0, 2, 1))
        self.cnt[v] += r.shape[1]

    def spread(self, v, m, buckets, floor=1e-7):
        import torch
        sc, bs_, bc = self.sums[v]
        scale = (sc / max(self.cnt[v], 1) / (2 * LN2)).clamp_min(floor)            # [L1, 2, C]
        mean_all = (bs_.sum(2) / bc.sum(2).clamp_min(1))[:, :, None]
        factor = torch.where(bc > 0, (bs_ / bc.clamp_min(1)) / mean_all.clamp_min(1e-30), torch.ones_like(bs_))
        f = factor[:, :, buckets].permute(0, 2, 1)                                    # [L1, n, 2]
        s = scale[:, None, :, :] * f[..., None] * m[:, :, None, None].double()
        return s.reshape(self.L1, len(buckets), 2 * self.C).float()


def code_slices(coder, tgt_w, mu, s, L, C, timer, tag):
    """Bytes per (K/V, layer group) and exactness of tgt_w [L, n, W] (int16, layers 1.. coded) against mu, s
    [L-1, n, W]: one coder call per slice."""
    import torch
    out, ok = {}, True
    for a, b in groups_of(L):
        for kv, sl in (("K", slice(0, C)), ("V", slice(C, 2 * C))):
            w = tgt_w[a:b + 1, :, sl].reshape(-1).contiguous()
            m_ = mu[a - 1:b, :, sl].reshape(-1).contiguous()
            s_ = s[a - 1:b, :, sl].reshape(-1).contiguous()
            with timer(f"{tag} encode"):
                buf, starts, n = coder.encode(w, m_, s_)
            nbytes = coder.bits(starts, n) / 8
            with timer(f"{tag} decode"):
                back = coder.decode(buf, starts, m_, s_, n)
            ok = ok and bool(torch.equal(back.view(torch.int16), w.view(torch.int16)))
            out[f"{kv} {a}-{b}"] = nbytes
    return out, ok


def analyse(cap, run_dir, codecs_file, model_dir, out_dir, names=("t2-hbh",), device="cuda", batch=BATCH,
            check=CHECK_REQUESTS, predictor=None, stream_values=None, log=print):
    import torch
    from kv_compare.pd import codecs as cdm
    from kv_compare import gpu_coder as gc
    dev = torch.device(device)
    stores, runs, err = pg.collect(cap, run_dir)
    if err:
        return {"error": err}
    st0 = stores["P"][0]
    L, bs, W = st0.L, st0.bs, st0.W
    C = W // 2
    inv, hd = da.rope_inv_freq(model_dir, 128)
    heads = C // hd
    words = da.Words(stores["P"])
    fetch = lambda loc: torch.from_numpy(words.get(loc).transpose(1, 0, 2).copy()).to(dev)   # [L, n, W]
    built, errors, _meta = cdm.build(codecs_file, dev, list(names), log=log)
    if errors:
        log(f"codecs not available: {errors}")
    coder = gc.SliceCoder(dev, mode="uniform", **({"values_per_stream": stream_values} if stream_values else {}))
    timer = dg.Timer()
    req = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request" and r.get("missing", 0) == 0:
                req[r["key"]] = (s.pid, r["rows"], r["T"])
    out = {"meta": {"layers": L, "block_size": bs, "page_bytes": L * bs * W * 2, "codecs": list(built),
                    "codec_errors": errors, "batch": batch, "groups": groups_of(L)}, "runs": {}}
    vals_page = L * bs * W                           # values per page: bits over all layers (layer 0 recomputed: 0)
    for mode, codec in built.items():
        sets = getattr(codec, "sets", None) or {getattr(codec, "name", mode): codec}
        for name, run in runs.items():
            reqs, pages = run["requests"], run["pages"]
            cand = []
            for i, p in enumerate(pages):
                rule = "text" if "text" in p["bases"] else ("position" if "position" in p["bases"] else None)
                if rule is None:
                    continue
                b = p["bases"][rule]
                s0 = p["block"] * bs
                ids_t, ids_b = reqs[p["r"]][1]["ids"], reqs[b["req"]][1]["ids"]
                if rule == "position" and list(ids_t[s0:s0 + bs]) != list(ids_b[s0:s0 + bs]):
                    buckets = np.full(bs, OTHER_TEXT)        # the same place, other text: a bucket of its own
                else:
                    buckets = dg.bucket_of(dg.agreement(ids_t, s0, ids_b, s0 - b["shift"], bs))
                cand.append({"i": i, "rule": rule, "b": b, "s0": s0, "buckets": buckets, "conv": p["conv"]})
            convs = sorted({c["conv"] for c in cand})
            train = {c for k, c in enumerate(convs) if k % 2 == 0}
            log(f"== {mode} {name}: {len(cand)} pages with a base ({len(train)} of {len(convs)} conversations to fit)")
            res = {"pages_with_base": len(cand), "by_rule": dict(collections.Counter(c["rule"] for c in cand)),
                   "train_conversations": sorted(train),
                   "train_pages": sum(c["conv"] in train for c in cand)}
            if not cand:
                out["runs"][f"{mode} {name}"] = res
                continue

            def batches(sel):
                xs = [c for c in cand if sel(c)]
                for k in range(0, len(xs), batch):
                    yield xs[k:k + batch]

            def tensors(bt, pred):
                tw = torch.cat([fetch(pages[c["i"]]["loc"]) for c in bt], 1)              # [L, n, W]
                pos = torch.cat([torch.arange(c["s0"], c["s0"] + bs) for c in bt]).to(dev)
                mu_t, s_t, m = (x[1:] for x in pred(tw, pos))                         # layers 1..
                mu_b = torch.cat([dg.prediction(fetch(c["b"]["loc"]), c["b"]["shift"], inv, heads, hd, C)
                                  for c in bt], 1)
                bk = torch.as_tensor(np.concatenate([c["buckets"] for c in bt]), dtype=torch.long, device=dev)
                tgt = tw[1:].view(torch.bfloat16).to(torch.float32)
                return tw, tgt, mu_t, s_t, m, mu_b, bk

            # T's constant set: the one coding the first training batches smallest (T2 picks one per request)
            set_bytes = {}
            first = [bt for _, bt in zip(range(2), batches(lambda c: c["conv"] in train))]
            for sname, o in sets.items():
                pred = predictor(o) if predictor else TPredictor(o)
                tot = 0
                for bt in first:
                    tw, _tgt, mu_t, s_t, _m, _mu_b, _bk = tensors(bt, pred)
                    sz, _ = code_slices(coder, tw, mu_t, s_t, L, C, timer, "set choice")
                    tot += sum(sz.values())
                set_bytes[sname] = tot
            chosen = min(set_bytes, key=set_bytes.get)
            pred = predictor(sets[chosen]) if predictor else TPredictor(sets[chosen])
            res["set_bytes"], res["set"] = set_bytes, chosen
            # fit: the blend's weights, then every fitted spread, on the training pages
            fit = Fit(L - 1, C, len(dg.BUCKETS) + 1, dev)
            for bt in batches(lambda c: c["conv"] in train):
                _tw, tgt, mu_t, _s, _m, mu_b, bk = tensors(bt, pred)
                fit.add_weights(tgt, mu_t, mu_b, bk)
            w = fit.solve()
            for bt in batches(lambda c: c["conv"] in train):
                _tw, tgt, mu_t, _s, m, mu_b, bk = tensors(bt, pred)
                one = torch.ones_like(m)
                fit.add_spread("t_fit", tgt, mu_t, m, bk)
                fit.add_spread("base", tgt, mu_b, one, bk)
                fit.add_spread("base_prior", tgt, mu_b, m, bk)
                fit.add_spread("blend", tgt, fit.blend(mu_t, mu_b, bk), m, bk)
            num, den = fit.num.cpu().numpy(), fit.den.cpu().numpy()                     # [L1, 2, buckets]
            res["weights"] = {f"{kv} {a}-{b}": float(num[a - 1:b, j].sum() / max(den[a - 1:b, j].sum(), 1e-300))
                              for a, b in groups_of(L) for j, kv in enumerate("KV")}    # least squares over the group
            res["weights_by_bucket"] = {kv: [float(num[:, j, q].sum() / den[:, j, q].sum()) if den[:, j, q].sum() > 0
                                             else None for q in range(num.shape[2])] for j, kv in enumerate("KV")}
            # code every page with a base: each variant, per slice, decoded and checked
            acc = {part: {v: collections.Counter() for v in VARIANTS} for part in ("heldout", "train")}
            pages_n = collections.Counter()
            exact, calls = collections.Counter(), collections.Counter()
            for bt in batches(lambda c: True):
                tw, tgt, mu_t, s_t, m, mu_b, bk = tensors(bt, pred)
                one = torch.ones_like(m)
                mus = {"t_own": (mu_t, s_t), "t_fit": (mu_t, fit.spread("t_fit", m, bk)),
                       "base": (mu_b, fit.spread("base", one, bk)), "base_prior": (mu_b, fit.spread("base_prior", m, bk)),
                       "blend": (fit.blend(mu_t, mu_b, bk), fit.spread("blend", m, bk))}
                held = [c["conv"] not in train for c in bt]
                # per page parts: split the batch in held-out and train pages (one call each)
                for part, keep in (("heldout", True), ("train", False)):
                    idx = [k for k, h in enumerate(held) if h == keep]
                    if not idx:
                        continue
                    tok = torch.cat([torch.arange(k * bs, (k + 1) * bs) for k in idx]).to(dev)
                    pages_n[part] += len(idx)
                    for v, (mu_, s_) in mus.items():
                        sz, ok = code_slices(coder, tw[:, tok], mu_[:, tok], s_[:, tok], L, C, timer, f"{v}")
                        acc[part][v].update(sz)
                        exact[v] += ok
                        calls[v] += 1
            res["pages"] = dict(pages_n)
            out["meta"]["pages_coded"] = out["meta"].get("pages_coded", 0) + sum(pages_n.values())
            res["bytes"] = {part: {v: dict(c) for v, c in d.items()} for part, d in acc.items()}
            res["exact"], res["calls"] = dict(exact), dict(calls)
            res["values_per_page"] = vals_page
            # the check: T through this coder against the codec's own size, whole requests (no padding)
            chk = []
            for _t, d, k in reqs:
                if len(chk) >= check:
                    break
                pid, rows, T = req[k]
                n = T // bs * bs // cdm.PAD * cdm.PAD
                if n == 0:
                    continue
                tw = fetch([(pid, int(rows[t // bs]), t % bs) for t in range(n)])
                n2, ok2, l0 = dg.t2_code(codec, tw, 0, timer, "check codec")
                mu_t, s_t, _m = pred(tw, torch.arange(n, device=dev))
                first = 1 if l0 == "recomputed" else 0              # T2 recomputes layer 0; T codes it
                here, ok3 = 0, True
                for kv, sl in (("K", slice(0, C)), ("V", slice(C, 2 * C))):
                    w = tw[first:, :, sl].reshape(-1).contiguous()
                    m_, s_ = mu_t[first:, :, sl].reshape(-1).contiguous(), s_t[first:, :, sl].reshape(-1).contiguous()
                    with timer("check T here encode"):
                        buf, starts, nv = coder.encode(w, m_, s_)
                    here += coder.bits(starts, nv) / 8
                    ok3 = ok3 and bool(torch.equal(coder.decode(buf, starts, m_, s_, nv).view(torch.int16),
                                                   w.view(torch.int16)))
                chk.append({"tokens": n, "t2_bytes": n2, "t2_layer0": l0, "here_bytes": here, "exact": ok2 and ok3})
            res["check"] = chk
            out["runs"][f"{mode} {name}"] = res
            log(f"   set {chosen}; weights {res['weights']}")
    out["times"] = timer.summary()
    return out


# -------------------------------------------------------------------------------------------------------- report
def markdown(res):
    m = res["meta"]
    L, bs = m["layers"], m["block_size"]
    W = m["page_bytes"] // 2 // bs // L
    gl = [f"{a}-{b}" for a, b in m["groups"]]
    L_ = ["# T plus the base against T alone, on pages with an earlier copy", "",
          f"Layers 1-{L - 1} of every distinct page with a base (layer 0 recomputed, as T2 does), coded with T's "
          "value-slice coder against each prediction, decoded and checked. Fitted on the even conversations. Bits "
          f"per value (16 raw) over all {L} layers, layer 0 at 0 bits (as pd_diffgraph.py's rows), held out (the odd "
          "conversations) and on the training pages; the K and V columns over their own values. Base: the latest "
          "earlier copy of the 16 tokens (keys turned), else the latest earlier turn's page at the same place."]
    for key, r in res["runs"].items():
        L_ += ["", f"## {key}", ""]
        if not r.get("pages"):
            L_.append(f"{r.get('pages_with_base', 0)} pages with a base.")
            continue
        L_.append(f"{r['pages_with_base']:,} pages with a base ({', '.join(f'{k} {v:,}' for k, v in r['by_rule'].items())}); "
                  f"held out {r['pages'].get('heldout', 0):,}, training {r['pages'].get('train', 0):,}. T's "
                  f"constants: set {r['set']} (the smallest on the first training pages).")
        vals_slice = {}
        for a, b in m["groups"]:
            vals_slice[f"{a}-{b}"] = (b - a + 1) * bs * (W // 2)
        L_ += ["", "| Prediction | Held out | Training | " + " | ".join(f"K {g}" for g in gl) + " | " +
               " | ".join(f"V {g}" for g in gl) + " | Decoded exactly |",
               "| --- | ---: | ---: | " + " | ".join("---:" for _ in gl * 2) + " | --- |"]
        base_held = None
        for v in VARIANTS:
            row = []
            for part in ("heldout", "train"):
                n = r["pages"].get(part, 0)
                tot = sum(r["bytes"][part][v].values())
                row.append(f"{8 * tot / (n * r['values_per_page']):.2f}" if n else "-")
            n = r["pages"].get("heldout", 0)
            cells = []
            for kv in "KV":
                for g in gl:
                    b_ = r["bytes"]["heldout"][v].get(f"{kv} {g}", 0)
                    cells.append(f"{8 * b_ / (n * vals_slice[g]):.2f}" if n else "-")
            if v == "t_own":
                base_held = sum(r["bytes"]["heldout"][v].values())
            L_.append(f"| {LABEL[v]} | {row[0]} | {row[1]} | " + " | ".join(cells) +
                      f" | {r['exact'].get(v, 0)} of {r['calls'].get(v, 0)} calls |")
        if base_held:
            bl = sum(r["bytes"]["heldout"]["blend"].values())
            L_ += ["", f"Blend against T as now, held out: {100 * (1 - bl / base_held):+.1f}% "
                   f"({'smaller' if bl < base_held else 'larger'})."]
        L_ += ["", "Blend weights w (0: T alone, 1: the base alone), least squares per layer group: " +
               ", ".join(f"{k} {v:.2f}" for k, v in r["weights"].items()) + ". By agreement bucket (tokens the text "
               "has agreed with the base's: 0-15 / 16-63 / 64-255 / 256+ / the same place, other text), all "
               "layers: " +
               "; ".join(f"{kv} " + " / ".join("-" if x is None else f"{x:.2f}" for x in v)
                         for kv, v in r["weights_by_bucket"].items()) + "."]
        if r.get("check"):
            t2 = sum(c["t2_bytes"] for c in r["check"])
            here = sum(c["here_bytes"] for c in r["check"])
            tok = sum(c["tokens"] for c in r["check"])
            L_ += ["", f"Check, {len(r['check'])} whole requests ({tok:,} tokens, no padding): the codec's own size "
                   f"{8 * t2 / (tok * L * W):.2f} bits per value (layer 0 {r['check'][0]['t2_layer0']}), T's "
                   f"prediction through this coder {8 * here / (tok * L * W):.2f} (the same layers); exact "
                   f"{sum(c['exact'] for c in r['check'])} of {len(r['check'])}."]
    t = res.get("times") or {}
    if t:
        pc = m.get("pages_coded") or 0
        L_ += ["", f"## Times (CUDA-synchronised, one GPU; batches of up to {m['batch']} pages, one call per layer "
               "group and K/V)", "", "| Step | calls | mean ms per call | ms per page | total s |",
               "| --- | ---: | ---: | ---: | ---: |"]
        for k in sorted(t):
            x = t[k]
            per = f"{1e3 * x['total_s'] / pc:.3f}" if pc and k.split(" ")[0] in VARIANTS else "-"
            L_.append(f"| {k} | {x['n']:,} | {x['mean_ms']:.2f} | {per} | {x['total_s']:.1f} |")
    return "\n".join(L_) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--codecs-file", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codecs", default="t2-hbh,t2-rank256")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--check", type=int, default=CHECK_REQUESTS)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    res = analyse(a.capture, a.run, a.codecs_file, a.model_dir, a.out, names=tuple(a.codecs.split(",")),
                  device=a.device, batch=a.batch, check=a.check)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "region_report.json").write_text(json.dumps(res, indent=1, default=str))
    if "error" in res:
        print("STOP:", res["error"])
        return 1
    text = markdown(res)
    (out / "region_report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
