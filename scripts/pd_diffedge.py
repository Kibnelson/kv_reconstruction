#!/usr/bin/env python3
"""Plain diff edges: every distinct page of a capture coded against earlier versions of it; then T and the blend.

  python3 scripts/pd_diffedge.py --capture CAPTURE_DIR --run RUN_DIR --model-dir MODEL --out OUT_DIR \
      [--codecs-file pd_codecs.pt --t-codec t2-hbh]

After the engines of a capture run (MATRIX=capture, STAGES=...,diffedge), or on a kept capture (MATRIX=diffedge), on
one GPU, over the run's own capture and records. "Plain": no codec's prediction (T's is not used). A page is coded
against a prediction made from its earlier versions by plain arithmetic, with the value-slice coder
(kv_compare.gpu_coder.SliceCoder) given that prediction and a fitted spread, then decoded and checked.

The pages P stored (distinct pages: a block P wrote first; a block whose bytes P stored before is a reference):
  twin, equal  D held the same tokens after the same prefix with the same bytes (D's trace): free
  twin         D held them with other bytes (its own decode, or a prefill in another batch): P's page against D's
               copy (D's capture keeps the copy of every held block that differs)
  moved        the same 16 tokens earlier in the conversation after another prefix (keys turned by RoPE to the new
               place): against a prediction from its versions (below)
  new          raw
Edges of a moved page (weights per layer, K/V and agreement bucket, or per channel; least squares):
  first          its first version (a star: every copy against the first)
  latest         its latest version (a chain: each copy against the one before)
  latest_affine  the latest with a scale and an offset per layer, K/V and channel
  second_order   latest + l (latest - previous): the drift between versions goes on (a diff of a diff)
  multi_ref      a latest + b previous + c first
  composite      per layer, K/V, channel and agreement bucket: a latest + b (latest - previous) + c first + d
With the codec constants (--codecs-file), moved pages also get T's edges, layers 1.. with T's value-slice coder in its
uniform mode (layer 0 as the latest edge codes it), as scripts/pd_region.py:
  t              T alone, its own prediction and spread: T as it codes now
  t_fit          T alone, the spread fitted as the blend's (the blend's control)
  t_blend        T + w (latest - T), w per layer, K/V and class (least squares): CacheBlend's reuse, made exact
The agreement of a token: how many tokens up to it the target's text has agreed with the version's (backwards from
the page's start, then the page itself): deep inside a long repeated run the version's context is nearly the same.
Weights and spreads (scale per layer, K/V and channel, a factor per agreement bucket, a multiplier per edge and bucket
chosen by the coder's ideal size) are fitted on the even conversations; every page is coded (per page, by layer group
and K/V), decoded and checked; tables give held-out (odd) conversations, and training ones apart.

The plain graph: references and equal twins free, twins and moved pages with the smallest of their edges (the edge's
number in the page's header) or raw, new pages raw; against raw distinct pages. Chains: each moved page's depth when
coded against its latest version, the pages to decode to rebuild it, and the time against recomputing.

Outputs: diffedge_report.json / .md, diffedge_pages.json (one row per coded page: bytes per edge and slice, its
conversation, turn, kind, versions, shifts and agreement) and diffedge_sample.npz (pages with their versions and twins,
layers 0,1,3,4,12,13,26,27, head 0's keys and values, for analysis elsewhere).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
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
import pd_region as rg  # noqa: E402

np = da.np
LN2 = math.log(2.0)
BUCKETS = (0, 16, 64, 256, 1024)            # tokens of agreement: 0-15, 16-63, 64-255, 256-1023, 1024+
BLABEL = ("0-15", "16-63", "64-255", "256-1023", "1024+")
GROUPS = ((0, 0), (1, 4), (5, 13), (14, 27))  # layer groups, inclusive; layer 0 alone (T2 recomputes it)
EDGES = ("first", "latest", "latest_affine", "second_order", "multi_ref", "composite")
FEATS = {"first": ("first",), "latest": ("latest",), "latest_affine": ("latest", 1),
         "second_order": ("latest", "drift"), "multi_ref": ("latest", "previous", "first"),
         "composite": ("latest", "drift", "first", 1), "twin": ("twin",)}
MODE = {"latest_affine": "channel", "second_order": "bucket", "multi_ref": "bucket", "composite": "channel_bucket"}
LABEL = {"first": "Its first version (star)", "latest": "Its latest version (chain)",
         "latest_affine": "Latest, scale + offset per channel", "second_order": "Second order: latest + l (latest - previous)",
         "multi_ref": "a latest + b previous + c first", "composite": "Composite: a latest + b drift + c first + d, per channel",
         "twin": "Twin: P's page against D's copy", "best": "Best edge per page (or raw)",
         "best_plain": "Best plain edge per page (or raw)", "t": "T alone, its own spread (as now)",
         "t_fit": "T alone, fitted spread", "t_blend": "Blend: T + w (latest - T)"}
T_EDGES = ("t", "t_fit", "t_blend")
MULTS = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0)      # spread multipliers tried per edge and bucket
BATCH = 32
MIN_FIT = 16                                  # samples a weight group needs (else the plain latest)
PRIOR = 1.0                                   # the pull toward the plain latest: as many samples of it
SAMPLE_LAYERS = (0, 1, 3, 4, 12, 13, 26, 27)
SAMPLE_MOVED, SAMPLE_TWINS = 96, 64


def bucket_of(agree):
    return np.searchsorted(np.asarray(BUCKETS[1:]), np.asarray(agree), side="right")


def groups_of(L):
    return [(a, min(b, L - 1)) for a, b in GROUPS if a <= L - 1]


def slices_of(L, C):
    """(name, layers a..b, words) of every coded slice: layer group by K/V."""
    return [(f"{kv} {a}-{b}", a, b, slice(j * C, (j + 1) * C)) for a, b in groups_of(L) for j, kv in enumerate("KV")]


# ---------------------------------------------------------------------------------------------------- the pages
def versions(run, req, bs):
    """Per distinct page with its text earlier in the conversation (an earlier request, or earlier in its own): every
    earlier version (the same 16 tokens at another place or after another prefix), in order, each {loc, shift, req,
    j}. A version holding the page itself is left out."""
    reqs, pages = run["requests"], run["pages"]
    starts, hashes = {}, {}
    for r, (_t, d, _k) in enumerate(reqs):
        hs = da.window_hashes(d["ids"])
        m = collections.defaultdict(list)
        for i, h in enumerate(hs):
            m[h].append(i)
        starts[r], hashes[r] = m, hs
    by_conv = collections.defaultdict(list)
    for r, (_t, d, _k) in enumerate(reqs):
        by_conv[d["conv"]].append(r)
    out = {}
    for i, p in enumerate(pages):
        r, s0 = p["r"], p["block"] * bs
        ids, hs = reqs[r][1]["ids"], hashes[r]
        if s0 >= len(hs):
            continue
        want, seen, vs = list(ids[s0:s0 + bs]), set(), []
        for e in by_conv[p["conv"]]:
            if e > r:
                break
            eids = reqs[e][1]["ids"]
            epid, erows, _eT = req[reqs[e][2]]
            for j in starts[e].get(hs[s0], []):
                if e == r and j >= s0:
                    break
                if list(eids[j:j + bs]) != want:
                    continue
                loc = [(epid, erows[(j + o) // bs], (j + o) % bs) for o in range(bs)]
                if any(x[1] < 0 for x in loc) or any((x[0], x[1]) == tuple(p["page"]) for x in loc):
                    continue
                key = tuple(loc)
                if key in seen:
                    continue
                seen.add(key)
                vs.append({"loc": loc, "shift": s0 - j, "req": e, "j": j})
        if vs:
            out[i] = vs
    return out


def twins(stores, bs):
    """(P pid, P row) -> (D pid, D row): D's own copy of a block P computed again (same tokens, same prefix), from
    D's capture; and how many such blocks D's capture had no room for."""
    pkey = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request":
                pkey[r["key"]] = (s.pid, r["rows"], r["T"])
    out, missing = {}, 0
    for s in stores["D"]:
        for r in s.records:
            if r.get("event") != "held_differs" or r["key"] not in pkey:
                continue
            ppid, prow, T = pkey[r["key"]]
            for j, drow in zip(r["blocks"], r["rows"]):
                jb = r["t0"] // bs + j
                if jb >= len(prow) or prow[jb] < 0 or (jb + 1) * bs > T:
                    continue
                if drow < 0:
                    missing += 1
                    continue
                out.setdefault((ppid, int(prow[jb])), (s.pid, int(drow)))
    return out, missing


def held_equal(stores, run_dir, bs):
    """P's pages D held with the same bytes (D's trace "held" records: every layer equal): free in the graph."""
    from kv_compare.pd.trace import unpack_bits
    pkey = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request":
                pkey[r["key"]] = (s.pid, r["rows"], r["T"])
    out = set()
    for f in sorted(pathlib.Path(run_dir).rglob("trace_D_*.jsonl*")):
        for r in da.jl(f):
            if r.get("event") != "held" or r.get("key") not in pkey or not r.get("equal"):
                continue
            ppid, prow, T = pkey[r["key"]]
            n = int(r["pages"]) // max(int(r["layers"]), 1)
            m = unpack_bits(r["equal"], (int(r["layers"]), n))
            for j in range(n):
                jb = int(r["t0"]) // bs + j
                if m[:, j].all() and jb < len(prow) and prow[jb] >= 0 and (jb + 1) * bs <= T:
                    out.add((ppid, int(prow[jb])))
    return out


# ------------------------------------------------------------------------------------------------------ fitting
class LSQ:
    """Least squares weights of an edge's features per group (layer, K/V, and class and/or channel; a class is an
    agreement bucket and whether the page has one version or more), on the device (float64), pulled toward the
    plain first feature (weight 1, the rest 0) as if by PRIOR samples of it; a group with fewer than MIN_FIT samples
    keeps the plain first feature."""

    def __init__(self, mode, F, L, C, nb, device):
        import torch
        self.mode, self.L, self.C, self.nb = mode, L, C, nb
        shape = {"bucket": (L, 2, nb), "channel": (L, 2, C), "channel_bucket": (L, 2, C, nb)}[mode]
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.A, self.b, self.n = z(*shape, F, F), z(*shape, F), z(*shape)

    def _reduce(self, t, bk):
        """[L, n, 2, C] -> the group shape."""
        import torch
        if self.mode == "channel":
            return t.sum(1)
        if self.mode == "bucket":
            out = torch.zeros(self.L, 2, self.nb, dtype=torch.float64, device=t.device)
            return out.index_add_(2, bk, t.sum(3).permute(0, 2, 1))
        out = torch.zeros(self.L, 2, self.C, self.nb, dtype=torch.float64, device=t.device)
        return out.index_add_(3, bk, t.permute(0, 2, 3, 1))

    def add(self, fs, y, bk):
        """fs: the features [L, n, 2, C] (None: the constant 1); y: the target."""
        import torch
        X = [torch.ones_like(y) if f is None else f for f in fs]
        self.n += self._reduce(torch.ones_like(y), bk)
        for a in range(len(X)):
            self.b[..., a] += self._reduce(X[a] * y, bk)
            for c in range(a + 1):
                v = self._reduce(X[a] * X[c], bk)
                self.A[..., a, c] += v
                if c != a:
                    self.A[..., c, a] += v

    def solve(self):
        import torch
        F = self.A.shape[-1]
        w0 = torch.zeros(F, dtype=torch.float64, device=self.A.device)
        w0[0] = 1.0
        diag = self.A.diagonal(dim1=-2, dim2=-1)
        rho = 1e-9 + PRIOR / self.n.clamp_min(1)
        D = torch.diag_embed(diag * rho[..., None] + 1e-30)
        w = torch.linalg.solve(self.A + D, (self.b + (D @ w0[:, None])[..., 0])[..., None])[..., 0]
        ok = (self.n >= MIN_FIT)[..., None] & torch.isfinite(w).all(-1, keepdim=True)
        return torch.where(ok, w, w0.expand_as(w))


class Spread:
    """s = scale[layer, K/V, channel] * factor[layer, K/V, class] * mult[class] (logistic: mean |r| = 2 ln 2 s)."""

    def __init__(self, L, C, nb, device):
        import torch
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.sc, self.bsum, self.bcnt, self.n, self.L, self.C = z(L, 2, C), z(L, 2, nb), z(L, 2, nb), 0, L, C
        self.mult = torch.ones(nb, dtype=torch.float64, device=device)

    def add(self, r, bk):
        import torch
        a = r.abs().reshape(self.L, r.shape[1], 2, self.C).double()
        self.sc += a.sum(1)
        mm = a.mean(3).permute(0, 2, 1)                                  # [L, 2, n]
        self.bsum.index_add_(2, bk, mm)
        self.bcnt.index_add_(2, bk, torch.ones_like(mm))
        self.n += r.shape[1]

    def get(self, bk, floor=1e-7):
        import torch
        scale = (self.sc / max(self.n, 1) / (2 * LN2)).clamp_min(floor)
        mean_all = (self.bsum.sum(2) / self.bcnt.sum(2).clamp_min(1))[:, :, None]
        fac = torch.where(self.bcnt > 0, self.bsum / self.bcnt.clamp_min(1) / mean_all.clamp_min(1e-30),
                          torch.ones_like(self.bsum))
        f = (fac * self.mult)[:, :, bk].permute(0, 2, 1)                  # [L, n, 2]
        return (scale[:, None] * f[..., None]).reshape(self.L, len(bk), 2 * self.C).float().clamp_min(floor)


def kv4(x, C):
    return x.reshape(x.shape[0], x.shape[1], 2, C)


def feature_list(v, X):
    return [None if f == 1 else X[f] for f in FEATS[v]]


def predict(v, w, X, bk, C):
    """mu [L, n, W] of edge v from the versions X ({name: [L, n, W] float32})."""
    import torch
    fs = feature_list(v, X)
    if v not in MODE:
        return fs[0]
    L, n, W = fs[0].shape
    out = torch.zeros(L, n, 2, C, dtype=torch.float32, device=fs[0].device)
    for a, f in enumerate(fs):
        wa = w[..., a].float()
        if MODE[v] == "channel":
            wa = wa[:, None]                                                  # [L, 1, 2, C]
        elif MODE[v] == "bucket":
            wa = wa[:, :, bk].permute(0, 2, 1)[..., None]                     # [L, n, 2, 1]
        else:
            wa = wa[:, :, :, bk].permute(0, 3, 1, 2)                          # [L, n, 2, C]
        out += wa * (1.0 if f is None else kv4(f, C))
    return out.reshape(L, n, W)


def ideal(coder, w, mu, s, chunk=1 << 23):
    """The coder's ideal bits of flat w against mu, s, in chunks (its PyTorch mirror needs room)."""
    tot = 0.0
    for a in range(0, w.numel(), chunk):
        tot += coder.ideal_bits(w[a:a + chunk], mu[a:a + chunk], s[a:a + chunk])
    return tot


# -------------------------------------------------------------------------------------------------------- coding
def code(coder, w, mu, s, timer=None, tag=""):
    """Bytes of the int16 words w against mu, s (flat), decoded and checked."""
    import torch
    t = timer or (lambda _n: contextlib.nullcontext())
    with t(f"{tag} encode"):
        buf, starts, n = coder.encode(w, mu, s)
    nb = coder.bits(starts, n) / 8
    with t(f"{tag} decode"):
        back = coder.decode(buf, starts, mu, s, n)
    return nb, bool(torch.equal(back.view(torch.int16), w.view(torch.int16)))


def analyse(cap, run_dir, model_dir, out_dir, device="cuda", batch=BATCH, stream_values=None, sample=True,
            log=print, codecs_file=None, t_codec="t2-hbh", predictor=None):
    import torch
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
    pw = da.Words(stores["P"])
    dw = da.Words(stores["D"]) if stores["D"] else None
    fetch = lambda words, loc: torch.from_numpy(words.get(loc).transpose(1, 0, 2).copy()).to(dev)   # [L, n, W]
    sv = {"values_per_stream": stream_values} if stream_values else {}
    coder = gc.SliceCoder(dev, **sv)
    timer = dg.Timer()
    # T's edges: its constant sets (T2 picks one per request: here one per run, the smallest on training pages)
    t_sets, t_info = None, {}
    if t_codec and t_codec != "none" and (codecs_file or predictor is not None):
        from kv_compare.pd import codecs as cdm
        built, errors, _meta = cdm.build(codecs_file, dev, [t_codec], log=log)
        codec = built.get(t_codec)
        if codec is None:
            t_info["error"] = str(errors.get(t_codec, errors))[:300]
            log(f"T's edges left out: {t_info['error']}")
        else:
            t_sets = getattr(codec, "sets", None) or {t_codec: codec}
            t_info["codec"] = t_codec
    make_t = predictor or rg.TPredictor
    coder_u = gc.SliceCoder(dev, mode="uniform", **sv) if t_sets else None
    req = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request" and r.get("missing", 0) == 0:
                req[r["key"]] = (s.pid, r["rows"], r["T"])
    tw_all, tw_missing = twins(stores, bs)
    eq_all = held_equal(stores, run_dir, bs)
    page_bytes = L * bs * W * 2
    SL = slices_of(L, C)
    nb = len(BUCKETS)
    NC = 2 * nb                                     # classes: agreement bucket, and one version or more
    out = {"meta": {"layers": L, "block_size": bs, "page_bytes": page_bytes, "buckets": list(BLABEL),
                    "slices": [x[0] for x in SL], "edges": list(EDGES), "twin_pairs": len(tw_all),
                    "twins_without_room": tw_missing, "held_equal_pages": len(eq_all), "coder": coder.mode,
                    "values_per_stream": coder.V, "t": t_info}, "runs": {}}
    rows_out, smp = [], {"moved": [], "twin": []}
    pages_for = collections.Counter()               # timer tag -> pages its calls covered
    for name, run in runs.items():
        reqs, pages = run["requests"], run["pages"]
        vs = versions(run, req, bs)
        convs = sorted({p["conv"] for p in pages})
        train = {c for k, c in enumerate(convs) if k % 2 == 0}
        kind = {}
        for i, p in enumerate(pages):
            pk = tuple(p["page"])
            kind[i] = ("twin_equal" if pk in eq_all else "twin" if pk in tw_all and dw is not None
                       else "moved" if i in vs else "new")
        twin_i = [i for i in range(len(pages)) if kind[i] == "twin"]
        moved_i = [i for i in range(len(pages)) if kind[i] == "moved"]
        blocks_conv = collections.Counter()
        for _t, d, k in reqs:
            blocks_conv[d["conv"]] += req[k][2] // bs
        res = {"blocks": sum(blocks_conv.values()), "distinct": len(pages),
               "kinds": dict(collections.Counter(kind.values())),
               "versions": dict(collections.Counter(min(len(vs[i]), 3) for i in moved_i)),
               "train_conversations": sorted(map(str, train)), "conversations": len(convs),
               "blocks_by_part": {"heldout": sum(n for c, n in blocks_conv.items() if c not in train),
                                  "train": sum(n for c, n in blocks_conv.items() if c in train)},
               "distinct_by_part": {"heldout": sum(p["conv"] not in train for p in pages),
                                    "train": sum(p["conv"] in train for p in pages)}}
        res["references"] = res["blocks"] - res["distinct"]
        res["kinds_by_part"] = {part: dict(collections.Counter(kind[i] for i, p in enumerate(pages)
                                                               if (p["conv"] in train) == (part == "train")))
                                for part in ("heldout", "train")}
        log(f"== {name}: {len(pages)} distinct pages: {res['kinds']}")

        def agreement(i, v):
            p = pages[i]
            return np.asarray(dg.agreement(reqs[p["r"]][1]["ids"], p["block"] * bs, reqs[v["req"]][1]["ids"], v["j"], bs))

        tpred = [None]                                  # T's predictor for this run (its set chosen below)
        flat = lambda x: x.reshape(-1).contiguous()

        def moved_batch(bt, tp=None):
            tgt_w = torch.cat([fetch(pw, pages[i]["loc"]) for i in bt], 1)                    # [L, n, W]
            X = {"latest": [], "previous": [], "first": []}
            R = {"latest": [], "previous": [], "first": []}
            ag = {"latest": [], "previous": [], "first": []}
            for i in bt:
                v = vs[i]
                for k, ver in (("latest", v[-1]), ("previous", v[-2] if len(v) > 1 else v[-1]), ("first", v[0])):
                    R[k].append((fetch(pw, ver["loc"]), ver["shift"]))
                    ag[k].append(agreement(i, ver))
            with timer("turn versions"):                 # RoPE to the page's place: the versions' keys turned
                for k, xs in R.items():
                    X[k] = torch.cat([dg.prediction(torch.cat([B[:1], B], 0), sh, inv, heads, hd, C) for B, sh in xs], 1)
                X["drift"] = X["latest"] - X["previous"]
            pages_for["turn versions"] += 3 * len(bt)
            R = {k: torch.cat([B for B, _sh in xs], 1) for k, xs in R.items()}
            ag = {k: np.concatenate(x) for k, x in ag.items()}
            multi = torch.as_tensor([len(vs[i]) >= 2 for i in bt], dtype=torch.long).repeat_interleave(bs)
            cls = torch.as_tensor(bucket_of(ag["latest"]), dtype=torch.long) + nb * multi    # what the decoder knows
            tp = tp or tpred[0]
            if tp is not None:                          # T from the page's own words at its own places
                pos = torch.cat([torch.arange(pages[i]["block"] * bs, (pages[i]["block"] + 1) * bs) for i in bt])
                with timer("predict t"):
                    X["t_mu"], X["t_s"], X["t_m"] = tp(tgt_w, pos.to(dev))
                pages_for["predict t"] += len(bt)
            return tgt_w, tgt_w.view(torch.bfloat16).to(torch.float32), X, cls.to(dev), R, ag

        def twin_batch(bt):
            tgt_w = torch.cat([fetch(pw, pages[i]["loc"]) for i in bt], 1)
            twin_w = torch.cat([fetch(dw, [(*tw_all[tuple(pages[i]["page"])], o) for o in range(bs)]) for i in bt], 1)
            X = {"twin": twin_w.view(torch.bfloat16).to(torch.float32)}
            bk = torch.full((tgt_w.shape[1],), nb - 1, dtype=torch.long, device=dev)
            return tgt_w, tgt_w.view(torch.bfloat16).to(torch.float32), X, bk, twin_w

        def chunks(idx, sel=lambda c: True, limit=None):
            xs = [i for i in idx if sel(pages[i]["conv"])]
            for n_, k in enumerate(range(0, len(xs), batch)):
                if limit is not None and n_ >= limit:
                    return
                yield xs[k:k + batch]

        is_train = lambda c: c in train
        # T's set: the one coding the first training batches smallest with its own spread (layers 1..)
        if t_sets:
            first = list(chunks(moved_i, is_train, limit=2))
            sb = {}
            for sname, o in t_sets.items():
                pr = make_t(o)
                sb[sname] = 0.0
                for bt in first:
                    tgt_w, _t, X, _c, _R, _a = moved_batch(bt, pr)
                    for _nm, a, b, sl in SL:
                        if b >= 1:
                            a = max(a, 1)
                            sb[sname] += code(coder_u, flat(tgt_w[a:b + 1, :, sl]), flat(X["t_mu"][a:b + 1, :, sl]),
                                              flat(X["t_s"][a:b + 1, :, sl]))[0]
            res["t_set"] = min(sb, key=sb.get)
            res["t_set_bytes"] = sb
            tpred[0] = make_t(t_sets[res["t_set"]])
        fit_t = rg.Fit(L - 1, C, NC, dev) if tpred[0] is not None else None
        # fit: weights (least squares), then spreads, then a spread multiplier per edge and bucket
        lsq = {v: LSQ(MODE[v], len(FEATS[v]), L, C, NC, dev) for v in EDGES if v in MODE}
        for bt in chunks(moved_i, is_train):
            _w, tgt, X, bk, _R, _a = moved_batch(bt)
            y = kv4(tgt, C).double()
            for v, q in lsq.items():
                q.add([None if f is None else kv4(f, C).double() for f in feature_list(v, X)], y, bk)
            if fit_t is not None:
                fit_t.add_weights(tgt[1:], X["t_mu"][1:], X["latest"][1:], bk)
        weights = {v: q.solve() for v, q in lsq.items()}
        del lsq
        if fit_t is not None:
            fit_t.solve()
        spreads = {v: Spread(L, C, NC, dev) for v in list(EDGES) + ["twin"]}
        for bt in chunks(moved_i, is_train):
            _w, tgt, X, bk, _R, _a = moved_batch(bt)
            for v in EDGES:
                spreads[v].add(tgt - predict(v, weights.get(v), X, bk, C), bk)
            if fit_t is not None:
                mu_t, m_t = X["t_mu"][1:], X["t_m"][1:]
                fit_t.add_spread("t_fit", tgt[1:], mu_t, m_t, bk)
                fit_t.add_spread("blend", tgt[1:], fit_t.blend(mu_t, X["latest"][1:], bk), m_t, bk)
        t_mult = {v: torch.ones(NC, dtype=torch.float32, device=dev) for v in ("t_fit", "t_blend")}

        def t_mus(X, bk, lat):
            """T's edges [L, n, W]: layer 0 as the latest edge codes it (lat: its mu, s), layers 1.. T's."""
            mu_t, s_t, m_t = X["t_mu"][1:], X["t_s"][1:], X["t_m"][1:]
            cat = lambda a, b: torch.cat([a[:1], b], 0)
            sp = lambda v, name: fit_t.spread(name, m_t, bk) * t_mult[v][bk][None, :, None]
            return {"t": (cat(lat[0], mu_t), cat(lat[1], s_t)),
                    "t_fit": (cat(lat[0], mu_t), cat(lat[1], sp("t_fit", "t_fit"))),
                    "t_blend": (cat(lat[0], fit_t.blend(mu_t, X["latest"][1:], bk)), cat(lat[1], sp("t_blend", "blend")))}
        for bt in chunks(twin_i, is_train):
            _w, tgt, X, bk, _t = twin_batch(bt)
            spreads["twin"].add(tgt - X["twin"], bk)
        tried = {v: np.zeros((len(MULTS), NC)) for v in list(spreads) + list(t_mult)}
        for kind_i, maker in ((moved_i, moved_batch), (twin_i, twin_batch)):
            for bt in chunks(kind_i, is_train, limit=4):
                got = maker(bt)
                tgt_w, X, bk = got[0], got[2], got[3]
                cand = {v: (predict(v, weights.get(v), X, bk, C), spreads[v].get(bk), coder, 0)
                        for v in (EDGES if maker is moved_batch else ("twin",))}
                if maker is moved_batch and fit_t is not None:        # layers 1..: T's coder
                    cand.update({v: (mu, s, coder_u, 1) for v, (mu, s) in t_mus(X, bk, cand["latest"][:2]).items()
                                 if v in t_mult})
                for v, (mu, s, cd, l0) in cand.items():
                    for q in torch.unique(bk).tolist():
                        tok = (bk == q).nonzero()[:, 0]
                        w_, m_, s_ = (x[l0:, tok].reshape(-1).contiguous() for x in (tgt_w, mu, s))
                        for a, mlt in enumerate(MULTS):
                            tried[v][a, q] += ideal(cd, w_, m_, s_ * mlt)
        for v, t in tried.items():
            for q in range(NC):
                if t[:, q].any():
                    (spreads[v].mult if v in spreads else t_mult[v])[q] = MULTS[int(np.argmin(t[:, q]))]
        res["spread_mult"] = {v: [float(x) for x in (spreads[v].mult if v in spreads else t_mult[v]).tolist()]
                              for v in tried if v in spreads or fit_t is not None}
        res["edges"] = list(EDGES) + (list(T_EDGES) if fit_t is not None else [])
        if fit_t is not None:                           # w by layer group and class (least squares over the group)
            num, den = fit_t.num.cpu().numpy(), fit_t.den.cpu().numpy()                    # [L-1, 2, NC]
            res["t_blend_weights"] = {f"{kv} {a}-{b}": [float(num[a - 1:b, j, q].sum() / den[a - 1:b, j, q].sum())
                                                         if den[a - 1:b, j, q].sum() > 0 else None for q in range(NC)]
                                      for a, b in groups_of(L) if b >= 1 for j, kv in enumerate("KV")}
        res["weights"] = weights_summary(weights, NC)
        # sample choice: spread over the moved pages (half with two versions or more) and the twins
        rng = np.random.default_rng(7)
        cap_m, cap_t = SAMPLE_MOVED // max(len(runs), 1), SAMPLE_TWINS // max(len(runs), 1)
        multi = [i for i in moved_i if len(vs[i]) >= 2]
        single = [i for i in moved_i if len(vs[i]) < 2]
        pick = lambda xs, n: set(rng.choice(xs, min(n, len(xs)), replace=False).tolist()) if xs and n > 0 else set()
        want_m = pick(multi, cap_m // 2)
        want_m |= pick(single, cap_m - len(want_m))
        want_t = pick(twin_i, cap_t)
        # code every page: per edge and slice, one page at a time, decoded and checked; batches timed whole
        exact, calls = collections.Counter(), collections.Counter()
        equal_tw = collections.Counter()
        batched = collections.defaultdict(lambda: [0.0, 0])     # edge -> bytes and pages, coded a batch at a time

        def row_base(i):
            p = pages[i]
            return {"run": name, "i": i, "conv": str(p["conv"]), "turn": int(p["turn"]), "block": int(p["block"]),
                    "kind": p["kind"], "edge": kind[i], "part": "train" if p["conv"] in train else "heldout"}

        def code_rows(bt, tgt_w, mus, edges):
            rows = []
            for k in range(len(bt)):
                tok = slice(k * bs, (k + 1) * bs)
                by = {}
                for v in edges:
                    mu, s = mus[v]
                    cd = coder_u if v in T_EDGES else coder
                    cell = []
                    for si, (_nm, a, b, sl) in enumerate(SL):
                        if v in T_EDGES and a == 0:     # layer 0 (alone in GROUPS): as the latest edge codes it
                            cell.append(by["latest"][si])
                            continue
                        n_, ok = code(cd, tgt_w[a:b + 1, tok, sl].reshape(-1).contiguous(),
                                      mu[a:b + 1, tok, sl].reshape(-1).contiguous(),
                                      s[a:b + 1, tok, sl].reshape(-1).contiguous())
                        cell.append(n_)
                        exact[v] += ok
                        calls[v] += 1
                    by[v] = cell
                rows.append(by)
            return rows

        def time_batch(tgt_w, mus, edges, n):
            for v in edges:
                mu, s = mus[v]
                _n, ok = code(coder, tgt_w.reshape(-1).contiguous(), mu.reshape(-1).contiguous(),
                              s.reshape(-1).contiguous(), timer, f"batch {v}")
                exact[v] += ok
                calls[v] += 1
                batched[v][0] += _n
                batched[v][1] += n
                pages_for[f"batch {v} encode"] += n
                pages_for[f"batch {v} decode"] += n

        for bt in chunks(moved_i):
            tgt_w, tgt, X, bk, R, ag = moved_batch(bt)
            mus = {}
            for v in EDGES:
                with timer(f"predict {v}"):
                    mus[v] = (predict(v, weights.get(v), X, bk, C), spreads[v].get(bk))
                pages_for[f"predict {v}"] += len(bt)
            if fit_t is not None:
                mus.update(t_mus(X, bk, mus["latest"]))
            time_batch(tgt_w, mus, ("latest", "second_order", "composite"), len(bt))
            vsame = (tgt_w[:, :, C:] == R["latest"][:, :, C:]).reshape(L, len(bt), bs, C)
            for k, (i, by) in enumerate(zip(bt, code_rows(bt, tgt_w, mus, res["edges"]))):
                v = vs[i]
                a = ag["latest"][k * bs:(k + 1) * bs]
                rows_out.append(dict(row_base(i), versions=len(v), shifts=[x["shift"] for x in v][-3:],
                                     agree0=int(a[0]), bucket=int(bucket_of(int(np.median(a)))),
                                     equal_v=float(vsame[:, k].float().mean()), bytes=by))
                if sample and i in want_m:
                    sl = slice(k * bs, (k + 1) * bs)
                    smp["moved"].append({"row": len(rows_out) - 1, "target": _cut(tgt_w[:, sl], C, hd),
                                         **{k2: _cut(R[k2][:, sl], C, hd) for k2 in ("latest", "previous", "first")},
                                         "shift": [v[-1]["shift"], (v[-2] if len(v) > 1 else v[-1])["shift"],
                                                   v[0]["shift"]],
                                         "agree": np.stack([ag[k2][k * bs:(k + 1) * bs] for k2 in
                                                            ("latest", "previous", "first")])})
        for bt in chunks(twin_i):
            tgt_w, tgt, X, bk, twin_w = twin_batch(bt)
            mus = {"twin": (X["twin"], spreads["twin"].get(bk))}
            time_batch(tgt_w, mus, ("twin",), len(bt))
            same = (tgt_w == twin_w).reshape(L, len(bt), bs, W)
            equal_tw["same"] += int(same.sum())
            equal_tw["all"] += same.numel()
            for k, (i, by) in enumerate(zip(bt, code_rows(bt, tgt_w, mus, ("twin",)))):
                rows_out.append(dict(row_base(i), versions=0, bucket=nb - 1, equal=float(same[:, k].float().mean()),
                                     bytes=by))
                if sample and i in want_t:
                    sl = slice(k * bs, (k + 1) * bs)
                    smp["twin"].append({"row": len(rows_out) - 1, "target": _cut(tgt_w[:, sl], C, hd),
                                        "twin": _cut(twin_w[:, sl], C, hd)})
        res["exact"], res["calls"] = dict(exact), dict(calls)
        res["batched"] = {v: {"bytes": b, "pages": n} for v, (b, n) in batched.items()}
        res["twin_equal_words"] = equal_tw["same"] / equal_tw["all"] if equal_tw["all"] else None
        res["chains"] = chains(pages, vs, kind)
        out["runs"][name] = res
        try:
            import pd_tree
            ms_tok, n_ = pd_tree.prefill_rate(run_dir, {k for _t, _d, k in reqs})
            res["recompute_ms_per_page"] = None if ms_tok is None else ms_tok * bs
            res["recompute_requests"] = n_
        except Exception as exc:                     # the comparison only; the sizes stand without it
            res["recompute_error"] = repr(exc)[:200]
    out["times"] = t = timer.summary()
    for k, x in t.items():
        if pages_for.get(k):
            x["ms_per_page"] = 1e3 * x["total_s"] / pages_for[k]
            x["pages"] = pages_for[k]
    summarise(out, rows_out)
    od = pathlib.Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "diffedge_pages.json").write_text(json.dumps({"slices": [x[0] for x in SL], "page_bytes": page_bytes,
                                                        "values_per_page": L * bs * W, "rows": rows_out}))
    if sample and (smp["moved"] or smp["twin"]):
        p = od / "diffedge_sample.npz"
        write_sample(p, smp, inv, hd, C, rows_out)
        out["meta"]["sample"] = {"file": p.name, "moved": len(smp["moved"]), "twins": len(smp["twin"]),
                                 "bytes": p.stat().st_size, "layers": [x for x in SAMPLE_LAYERS if x < L]}
    return out


def weights_summary(weights, nb):
    """Mean weights over layers (and channels): {edge: {K/V: per bucket [weights]}} or {K/V: [weights]}."""
    out = {}
    for v, w in weights.items():
        if MODE[v] == "channel":                                       # [L, 2, C, F]
            out[v] = {kv: [float(w[:, j, :, a].mean()) for a in range(w.shape[-1])] for j, kv in enumerate("KV")}
        elif MODE[v] == "bucket":                                      # [L, 2, nb, F]
            out[v] = {kv: [[float(w[:, j, q, a].mean()) for a in range(w.shape[-1])] for q in range(nb)]
                      for j, kv in enumerate("KV")}
        else:                                                          # [L, 2, C, nb, F]
            out[v] = {kv: [[float(w[:, j, :, q, a].mean()) for a in range(w.shape[-1])] for q in range(nb)]
                      for j, kv in enumerate("KV")}
    return out


def chains(pages, vs, kind):
    """Moved pages coded against their latest version: depth (1 + the deepest page that version spans; a twin is 1,
    a page stored whole 0) and the pages to decode to rebuild it (itself and every page its bases need). Against
    the first version: the same, a star."""
    row_of = {tuple(p["page"]): i for i, p in enumerate(pages)}
    out = {}
    for rule, pick in (("latest", lambda v: v[-1]), ("first", lambda v: v[0])):
        need = {}

        def closure(i):
            if i in need:
                return need[i]
            if kind[i] == "twin":
                need[i] = frozenset([i])
            elif kind[i] != "moved":
                need[i] = frozenset()
            else:
                need[i] = frozenset([i])            # no cycles: a version is earlier; this guards anyway
                s = {i}
                for b in {(x[0], x[1]) for x in pick(vs[i])["loc"]}:
                    if b in row_of:
                        s |= closure(row_of[b])
                need[i] = frozenset(s)
            return need[i]
        depth = {}

        def dep(i):
            if i in depth:
                return depth[i]
            depth[i] = 1 if kind[i] == "twin" else 0
            if kind[i] == "moved":
                depth[i] = 1 + max((dep(row_of[b]) for b in {(x[0], x[1]) for x in pick(vs[i])["loc"]} if b in row_of),
                                   default=0)
            return depth[i]
        moved = [i for i in range(len(pages)) if kind[i] == "moved"]
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(pages) + 1000))
        d = [dep(i) for i in moved]
        n = [len(closure(i)) for i in moved]
        out[rule] = {"depth": dict(sorted(collections.Counter(min(x, 8) for x in d).items())),
                     "decode_pages_mean": float(np.mean(n)) if n else None,
                     "decode_pages_p90": float(np.percentile(n, 90)) if n else None,
                     "decode_pages_max": int(max(n)) if n else None, "depth_max": int(max(d)) if d else None}
    return out


# ----------------------------------------------------------------------------------------------------- summaries
def bits(rows, edge, vpp, part=None):
    """Bits per value of `edge` over rows (that have it)."""
    xs = [sum(r["bytes"][edge]) for r in rows if edge in r["bytes"] and (part is None or r["part"] == part)]
    return (8 * sum(xs) / (len(xs) * vpp), len(xs)) if xs else (None, 0)


def best_of(r, page_bytes, edges=None):
    """The page's smallest size: raw, or its smallest edge (of `edges`, else every edge it has)."""
    return min([page_bytes] + [sum(x) for v, x in r["bytes"].items() if edges is None or v in edges])


def best_row(P, pb, vpp, nb, edges=None):
    pick = lambda r: min((v for v in r["bytes"] if edges is None or v in edges), key=lambda v: sum(r["bytes"][v]))
    return {"all": (8 * sum(best_of(r, pb, edges) for r in P) / (len(P) * vpp)) if P else None,
            "bucket": [(8 * sum(best_of(r, pb, edges) for r in P if r["bucket"] == q) /
                        (sum(r["bucket"] == q for r in P) * vpp)) if any(r["bucket"] == q for r in P)
                       else None for q in range(nb)],
            "chosen": dict(collections.Counter(pick(r) if best_of(r, pb, edges) < pb else "raw" for r in P))}


def summarise(out, rows):
    """Tables from the per-page rows: by part, bucket, versions, kind, slice; the graph by part."""
    m = out["meta"]
    vpp = m["page_bytes"] // 2
    pb = m["page_bytes"]
    nb = len(BLABEL)
    for name, res in out["runs"].items():
        E = res.get("edges") or list(EDGES)
        R = [r for r in rows if r["run"] == name]
        mv = [r for r in R if r["edge"] == "moved"]
        tw = [r for r in R if r["edge"] == "twin"]
        t = {}
        for part in ("heldout", "train"):
            P = [r for r in mv if r["part"] == part]
            T = [r for r in tw if r["part"] == part]
            row = {}
            for v in E:
                row[v] = {"all": bits(P, v, vpp)[0],
                          "bucket": [bits([r for r in P if r["bucket"] == q], v, vpp)[0] for q in range(nb)],
                          "versions": [bits([r for r in P if min(r["versions"], 3) == n], v, vpp)[0] for n in (1, 2, 3)],
                          "slice": {sl: (8 * sum(r["bytes"][v][k] for r in P) /
                                         (len(P) * vpp * slice_share(sl, m["layers"])) if P else None)
                                    for k, sl in enumerate(m["slices"])}}
            kinds = sorted({r["kind"] for r in P})
            for v in E:
                row[v]["kind"] = {k: bits([r for r in P if r["kind"] == k], v, vpp)[0] for k in kinds}
            row["best"] = best_row(P, pb, vpp, nb)
            row["best_plain"] = best_row(P, pb, vpp, nb, EDGES)
            row["twin"] = {"all": bits(T, "twin", vpp)[0],
                           "slice": {sl: (8 * sum(r["bytes"]["twin"][k] for r in T) /
                                          (len(T) * vpp * slice_share(sl, m["layers"])) if T else None)
                                     for k, sl in enumerate(m["slices"])},
                           "equal_words": float(np.mean([r["equal"] for r in T])) if T else None}
            t[part] = {"edges": row, "moved_pages": len(P), "twin_pages": len(T),
                       "moved_by_bucket": [sum(r["bucket"] == q for r in P) for q in range(nb)],
                       "moved_by_versions": [sum(min(r["versions"], 3) == n for r in P) for n in (1, 2, 3)],
                       "moved_by_kind": dict(collections.Counter(r["kind"] for r in P)),
                       "equal_v_latest": float(np.mean([r["equal_v"] for r in P])) if P else None}
        # the cheap rule: per bucket, the edge (or raw) smallest on the training pages, used on the held-out ones
        rule = []
        for q in range(nb):
            Tq = [r for r in mv if r["part"] == "train" and r["bucket"] == q]
            if not Tq:
                rule.append("latest")
                continue
            cost = {v: sum(sum(r["bytes"][v]) for r in Tq) for v in E}
            cost["raw"] = pb * len(Tq)
            rule.append(min(cost, key=cost.get))
        res["bucket_rule"] = rule
        for part in ("heldout", "train"):
            P = [r for r in mv if r["part"] == part]
            cost = lambda r: pb if rule[r["bucket"]] == "raw" else sum(r["bytes"][rule[r["bucket"]]])
            t[part]["edges"]["rule"] = {"all": 8 * sum(cost(r) for r in P) / (len(P) * vpp) if P else None,
                                        "bucket": [8 * sum(cost(r) for r in P if r["bucket"] == q) /
                                                   (sum(r["bucket"] == q for r in P) * vpp)
                                                   if any(r["bucket"] == q for r in P) else None for q in range(nb)]}
        res["tables"] = t
        res["per_page_vs_batched"] = {}
        for v, x in (res.get("batched") or {}).items():
            P = tw if v == "twin" else mv
            if x["pages"] and P:
                res["per_page_vs_batched"][v] = {"per_page": bits(P, v, vpp)[0],
                                                 "batched": 8 * x["bytes"] / (x["pages"] * vpp)}
        res["graph"] = {}
        for part in ("heldout", "train", "all"):
            sel = lambda r: part == "all" or r["part"] == part
            distinct = res["distinct_by_part"][part] if part != "all" else res["distinct"]
            blocks = res["blocks_by_part"][part] if part != "all" else res["blocks"]
            kinds_part = (_sum(res["kinds_by_part"].values()) if part == "all"
                          else collections.Counter(res["kinds_by_part"][part]))
            g = {"blocks": blocks, "distinct": distinct, "raw_all_blocks": blocks * pb, "raw_distinct": distinct * pb,
                 "twin_equal": 0, "twins": sum(best_of(r, pb) for r in tw if sel(r)),
                 "moved_best": sum(best_of(r, pb) for r in mv if sel(r)),
                 "moved_best_plain": sum(best_of(r, pb, EDGES) for r in mv if sel(r)),
                 "moved_latest": sum(min(pb, sum(r["bytes"]["latest"])) for r in mv if sel(r)),
                 "new_raw": kinds_part.get("new", 0) * pb, "pages": dict(kinds_part)}
            g["graph"] = g["twins"] + g["moved_best"] + g["new_raw"]
            g["graph_latest_only"] = g["twins"] + g["moved_latest"] + g["new_raw"]
            g["graph_plain"] = g["twins"] + g["moved_best_plain"] + g["new_raw"]
            res["graph"][part] = g


def _sum(ds):
    c = collections.Counter()
    for d in ds:
        c.update(d)
    return c


def slice_share(sl, L):
    """A slice's share of a page's values: its layers' share times a half (K or V)."""
    a, b = map(int, sl.split(" ")[1].split("-"))
    return (b - a + 1) / L / 2


# -------------------------------------------------------------------------------------------------------- sample
def _cut(x, C, hd, sel=SAMPLE_LAYERS):
    """[L, 16, W] (a torch tensor) -> [len(sel), 16, 2 hd] int16: head 0's keys and values."""
    L = x.shape[0]
    ls = [l for l in sel if l < L]
    y = x[ls].cpu().numpy()
    return np.concatenate([y[:, :, :hd], y[:, :, C:C + hd]], 2).astype(np.int16)


def write_sample(path, smp, inv, hd, C, rows):
    arrs = {"layers": np.asarray(list(SAMPLE_LAYERS)), "head_dim": np.asarray(hd if hd else 128),
            "rope_inv_freq": np.asarray(inv if inv is not None else [], dtype=np.float64),
            "words_note": np.asarray("BF16 words as int16 [pages, layers, 16 tokens, 2 head_dim]: head 0's keys, "
                                     "then its values; layers: those of `layers` the model has")}
    M = smp["moved"]
    if M:
        for k in ("target", "latest", "previous", "first"):
            arrs[f"moved_{k}"] = np.stack([s[k] for s in M])
        arrs["moved_shift"] = np.asarray([s["shift"] for s in M], dtype=np.int64)          # latest, previous, first
        arrs["moved_agree"] = np.stack([s["agree"] for s in M]).astype(np.int32)           # [n, 3, 16]
        arrs["moved_meta"] = np.asarray(json.dumps([_meta(rows[s["row"]]) for s in M]))
    T = smp["twin"]
    if T:
        arrs["twin_target"] = np.stack([s["target"] for s in T])
        arrs["twin_d"] = np.stack([s["twin"] for s in T])
        arrs["twin_meta"] = np.asarray(json.dumps([_meta(rows[s["row"]]) for s in T]))
    nl = (arrs.get("moved_target") if "moved_target" in arrs else arrs["twin_target"]).shape[1]
    arrs["layers"] = arrs["layers"][:nl]                  # those the model has (SAMPLE_LAYERS is in order)
    np.savez_compressed(path, **arrs)


def _meta(r):
    return {k: r[k] for k in ("run", "i", "conv", "turn", "block", "kind", "part", "versions", "shifts", "agree0",
                              "bucket", "bytes") if k in r}


# -------------------------------------------------------------------------------------------------------- report
def markdown(res):
    m = res["meta"]
    L, bs = m["layers"], m["block_size"]
    f2 = lambda x: "-" if x is None else f"{x:.2f}"
    mb = lambda b: f"{b / 1e6:,.1f}"
    out = ["# Diff edges: pages coded against their earlier versions (plain), then with T's prediction", "",
           f"Pages of {bs} tokens x {L} layers. Every value coded with the value-slice coder ({m['coder']} mode) against "
           "a prediction made from the page's earlier versions by plain arithmetic (keys turned by RoPE to the page's "
           "place), decoded and checked. Weights and spreads fitted on the even conversations; held out: the odd ones. "
           "Bits per value (raw 16), all layers coded. Agreement bucket: the median over the page's tokens of how many "
           "tokens up to them its text has agreed with its latest version's."]
    if m.get("twins_without_room"):
        out += ["", f"D's capture had no room for {m['twins_without_room']:,} held blocks that differ (not counted as "
                "twins: raise KVPD_CAPTURE_D_GB)."]
    for name, r in res["runs"].items():
        E = r.get("edges") or list(EDGES)
        withT = any(v in T_EDGES for v in E)
        k = r["kinds"]
        out += ["", f"## {name}", "",
                f"{r['blocks']:,} blocks in {r['conversations']} conversations: {r['references']:,} references (bytes P "
                f"stored before), {r['distinct']:,} distinct pages: {k.get('twin_equal', 0):,} twins with D's bytes, "
                f"{k.get('twin', 0):,} twins with other bytes, {k.get('moved', 0):,} moved (1 / 2 / 3+ earlier "
                "versions: " + " / ".join(f"{r['versions'].get(n, 0):,}" for n in (1, 2, 3)) + f"), "
                f"{k.get('new', 0):,} new."]
        tb = r.get("tables") or {}
        for part in ("heldout", "train"):
            x = tb.get(part)
            if not x or not x["moved_pages"]:
                continue
            e = x["edges"]
            title = "Held out" if part == "heldout" else "Training (fitted on these)"
            out += ["", f"{title}: {x['moved_pages']:,} moved pages, bits per value by agreement bucket (pages)", "",
                    "| Edge | All | " + " | ".join(f"{b} ({n:,})" for b, n in zip(BLABEL, x["moved_by_bucket"])) + " |",
                    "| --- | ---: | " + " | ".join("---:" for _ in BLABEL) + " |"]
            for v in E + (["best_plain", "best"] if withT else ["best"]):
                out.append(f"| {LABEL[v]} | {f2(e[v]['all'])} | " + " | ".join(f2(y) for y in e[v]["bucket"]) + " |")
            out.append(f"| Rule by bucket, from training ({' / '.join(r['bucket_rule'])}) | {f2(e['rule']['all'])} | "
                       + " | ".join(f2(y) for y in e["rule"]["bucket"]) + " |")
            if part == "heldout":
                out += ["", f"Values equal to the latest version's (V; keys are turned): "
                        f"{100 * (x['equal_v_latest'] or 0):.1f}%. Best edge per page: " +
                        ", ".join(f"{a} {b:,}" for a, b in sorted(e["best"]["chosen"].items(), key=lambda z: -z[1])) + "."]
                out += ["", "Held out, by earlier versions (pages): diff of a diff needs two", "",
                        "| Edge | " + " | ".join(f"{n} ({c:,})" for n, c in zip(("1", "2", "3+"), x["moved_by_versions"]))
                        + " |", "| --- | ---: | ---: | ---: |"]
                for v in E:
                    out.append(f"| {LABEL[v]} | " + " | ".join(f2(y) for y in e[v]["versions"]) + " |")
                kinds = sorted(x["moved_by_kind"])
                out += ["", "Held out, by what the page holds (pages)", "",
                        "| Edge | " + " | ".join(f"{kk} ({x['moved_by_kind'][kk]:,})" for kk in kinds) + " |",
                        "| --- | " + " | ".join("---:" for _ in kinds) + " |"]
                for v in E:
                    out.append(f"| {LABEL[v]} | " + " | ".join(f2(e[v]["kind"].get(kk)) for kk in kinds) + " |")
                sls = m["slices"]
                out += ["", "Held out, by layer group and K/V", "", "| Edge | " + " | ".join(sls) + " |",
                        "| --- | " + " | ".join("---:" for _ in sls) + " |"]
                for v in E:
                    out.append(f"| {LABEL[v]} | " + " | ".join(f2(e[v]["slice"][s]) for s in sls) + " |")
                if x["twin_pages"]:
                    out.append(f"| {LABEL['twin']} | " + " | ".join(f2(e["twin"]["slice"][s]) for s in sls) + " |")
            if x["twin_pages"]:
                out += ["", f"{title}: {x['twin_pages']:,} twins: {f2(e['twin']['all'])} bits per value; "
                        f"{100 * (e['twin']['equal_words'] or 0):.1f}% of their words equal to D's."]
        gr = r.get("graph") or {}
        if gr:
            parts = [p for p in ("heldout", "train", "all") if gr.get(p, {}).get("distinct")]
            out += ["", "The graph (MB)" + (": plain edges, then with T's" if withT else "; no codec anywhere"), "",
                    "| | " + " | ".join(
                {"heldout": "Held out", "train": "Training", "all": "All"}[p] for p in parts) + " |",
                "| --- | " + " | ".join("---:" for _ in parts) + " |"]
            rows = (("Raw, every block", "raw_all_blocks"), ("Raw, distinct pages (references free)", "raw_distinct"),
                    ("Twins (D's copy) as diffs, or raw", "twins"), ("Moved pages, best edge or raw", "moved_best"),
                    ("Moved pages, latest edge or raw", "moved_latest"), ("New pages raw", "new_raw"),
                    ("**The plain graph**", "graph_plain"), ("The plain graph, latest edge only", "graph_latest_only"))
            if withT:
                rows = rows[:3] + (("Moved pages, best plain edge or raw", "moved_best_plain"),
                                   ("Moved pages, best edge with T's, or raw", "moved_best")) + rows[4:] + \
                    (("**The graph with T's edges**", "graph"),)
            for lab, key in rows:
                out.append(f"| {lab} | " + " | ".join(mb(gr[p][key]) for p in parts) + " |")
            for lab, key in (("Plain graph under raw distinct pages", "graph_plain"),) + (
                    (("With T's edges, under raw distinct pages", "graph"),) if withT else ()):
                out.append(f"| {lab} | " + " | ".join(
                    f"{100 * (1 - gr[p][key] / gr[p]['raw_distinct']):.1f}%" for p in parts) + " |")
        ch = r.get("chains") or {}
        if ch:
            out += ["", "Chains and stars (moved pages; a twin is depth 1)", "",
                    "| Coded against | depth: pages | pages to decode, mean / p90 / max |", "| --- | --- | --- |"]
            for rule in ("latest", "first"):
                c = ch.get(rule)
                if c and c["depth"]:
                    out.append(f"| {rule} | " + ", ".join(f"{d}: {n:,}" for d, n in c["depth"].items()) +
                               f" | {c['decode_pages_mean']:.1f} / {c['decode_pages_p90']:.0f} / {c['decode_pages_max']} |")
        tm = res.get("times") or {}
        dec = (tm.get("batch latest decode") or {}).get("ms_per_page")
        prd = (tm.get("turn versions") or {}).get("ms_per_page")
        rc = r.get("recompute_ms_per_page")
        if dec is not None:
            out += ["", f"Rebuild: turning a version {f2(prd)} ms per page, decoding {dec:.3f} ms per page (batches of "
                    f"{BATCH}); recomputing: {f2(rc)} ms per page (P's prefill in this run, {r.get('recompute_requests', 0)}"
                    " requests)."]
        w = r.get("weights") or {}
        nbk = len(BLABEL)
        tup = lambda q: "(" + ", ".join(f"{y:.2f}" for y in q) + ")"
        if w.get("second_order"):
            out += ["", "Fitted weights, pages with two versions or more, by agreement bucket (mean over layers): "
                    "second order's l: " + "; ".join(f"{kv} " + " / ".join(f"{q[1]:.2f}" for q in
                                                                          w["second_order"][kv][nbk:]) for kv in "KV")
                    + "."]
        if w.get("multi_ref"):
            out += ["a latest + b previous + c first, (a, b, c): " + "; ".join(
                f"{kv} " + " / ".join(tup(q) for q in w["multi_ref"][kv][nbk:]) for kv in "KV") + "."]
        if w.get("composite"):
            out += ["Composite (a, b, c, d), mean over channels too: " + "; ".join(
                f"{kv} " + " / ".join(tup(q) for q in w["composite"][kv][nbk:]) for kv in "KV") + "."]
        if w.get("latest_affine"):
            out += ["", "Latest's scale and offset (mean over layers and channels): " + "; ".join(
                f"{kv} {w['latest_affine'][kv][0]:.3f}, {w['latest_affine'][kv][1]:+.4f}" for kv in "KV") + "."]
        if r.get("t_set"):
            out += ["", f"T's edges: {m['t'].get('codec')}, set {r['t_set']} (smallest on the first training pages); "
                    "layers 1.. with T's coder (uniform mode), layer 0 as the latest edge codes it."]
            tw_ = r.get("t_blend_weights") or {}
            if tw_:
                out += ["Blend w (0: T alone, 1: the latest version), pages with two versions or more, by agreement "
                        "bucket: " + "; ".join(f"{g} " + " / ".join("-" if x is None else f"{x:.2f}" for x in ws[nbk:])
                                               for g, ws in tw_.items()) + ".",
                        "With one version: " + "; ".join(f"{g} " + " / ".join("-" if x is None else f"{x:.2f}"
                                                                           for x in ws[:nbk]) for g, ws in tw_.items())
                        + "."]
        elif m.get("t", {}).get("error"):
            out += ["", f"T's edges left out: {m['t']['error']}"]
        if r.get("spread_mult"):
            out += ["", "Spread multipliers chosen (by class: buckets with one version, then two or more): " + "; ".join(
                f"{v} " + "/".join(f"{x:g}" for x in xs) for v, xs in r["spread_mult"].items()) + "."]
        pv = r.get("per_page_vs_batched") or {}
        if pv:
            out += ["", "Coded a page and a slice at a time (the tables) against whole batches of " + f"{BATCH} pages, all "
                    "layers (bits per value; the difference is the coder's streams' overhead): " +
                    "; ".join(f"{v} {f2(x['per_page'])} vs {f2(x['batched'])}" for v, x in pv.items()) + "."]
        bad = {v: (r["calls"].get(v, 0) - r["exact"].get(v, 0)) for v in r.get("calls", {})}
        out += ["", "Decoded exactly: " + ", ".join(f"{v} {r['exact'].get(v, 0):,} of {r['calls'][v]:,}"
                                                    for v in r.get("calls", {})) +
                (" (ALL)." if not any(bad.values()) else " (NOT ALL: see the json).")]
    tm = res.get("times") or {}
    if tm:
        out += ["", "## Times (CUDA-synchronised, one GPU)", "", "| Step | calls | ms per page | mean ms per call |",
                "| --- | ---: | ---: | ---: |"]
        for k in sorted(tm):
            x = tm[k]
            out.append(f"| {k} | {x['n']:,} | {f2(x.get('ms_per_page'))} | {x['mean_ms']:.2f} |")
    sm = m.get("sample")
    if sm:
        out += ["", f"Sample for analysis: `{sm['file']}` ({sm['bytes'] / 1e6:.1f} MB): {sm['moved']} moved pages with "
                f"their latest, previous and first versions, {sm['twins']} twins with D's copy; layers "
                f"{', '.join(map(str, sm.get('layers', SAMPLE_LAYERS)))}, head 0's keys and values; per-page sizes in "
                "`diffedge_pages.json`."]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--codecs-file", default=None, help="the codec constants (pd_codecs.pt): T's edges")
    ap.add_argument("--t-codec", default="t2-hbh", help="T's codec for its edges (none: plain edges only)")
    ap.add_argument("--codecs", default=None, help="unused (the runner passes it to every stage)")
    a = ap.parse_args(argv)
    cf = a.codecs_file if a.codecs_file and os.path.isfile(a.codecs_file) else None
    res = analyse(a.capture, a.run, a.model_dir, a.out, device=a.device, batch=a.batch, sample=not a.no_sample,
                  codecs_file=cf, t_codec=a.t_codec)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diffedge_report.json").write_text(json.dumps(res, indent=1, default=str))
    if "error" in res:
        print("STOP:", res["error"])
        return 1
    text = markdown(res)
    (out / "diffedge_report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
