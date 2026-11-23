#!/usr/bin/env python3
"""Our KV Diff Graph against EvidenceStore's, on a kept capture (MATRIX=capture): every page stored exactly, by the
cheapest representation each graph allows, decoded back and checked byte for byte, with sizes and rebuild times.

  python3 scripts/pd_diffgraph.py --capture CAPTURE_DIR --run RUN_DIR --codecs-file pd_codecs.pt \
      --model-dir MODEL --out OUT_DIR

Runs on one GPU (MATRIX=diffgraph: after setup, no vLLM engines). Pages: the capture's distinct complete 16-token
blocks (scripts/pd_page_compare.py collect), in request order, with their base candidates: the latest earlier turn's
page at the same position, and the latest earlier copy of the same 16 tokens anywhere (keys turned by RoPE).

Ours (everything exact; layer 0 recomputed from the token ids in all of them, as T2 does in the PD path: offline the
recompute is taken as exact, which T2's sender checks on every request in the engine):
  T2 per request   the whole prompt coded with T2 (hbh, rank256): the transfer's unit, not page-addressable
  T2 per segment   each run of contiguous new pages of a request coded together (a turn's new KV)
  T2 per page      every page coded alone (16 tokens: page-addressable)
  delta per page   layers 1..L-1 coded with T's value-slice coder (kv_compare.gpu_coder.SliceCoder) against a
                   prediction from the base page: its values, keys turned to the new position; the spread per
                   layer, K/V and channel, times a factor for how long the text before the token has agreed with
                   the base's (fitted on the even conversations, reported also on the odd ones alone)
  graph            per page the smaller of T2-per-page and delta (depth one, bases counted once, as theirs)
Theirs (scripts/pd_page_compare.py, zlib-1): best independent encoding and their depth-one graph, same pages.
Times (CUDA-synchronised): encode and decode per request, per page; delta encode, decode with the base warm on the
GPU, and cold (the base decoded first); a raw page copied from host memory.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import pd_delta_analysis as da  # noqa: E402
import pd_page_compare as pg  # noqa: E402

np = da.np
LN2 = math.log(2.0)


# ----------------------------------------------------------------------------------------------- helpers
def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = collections.defaultdict(list)

    def __call__(self, name):
        tm = self

        class _T:
            def __enter__(self):
                sync()
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                sync()
                tm.t[name].append(time.perf_counter() - self.t0)
        return _T()

    def summary(self):
        return {k: {"n": len(v), "mean_ms": 1e3 * float(np.mean(v)), "p50_ms": 1e3 * float(np.median(v)),
                    "total_s": float(np.sum(v))} for k, v in self.t.items()}


class OfflineLayer0:
    """T2's layer-0 context offline: the captured layer 0 stands for the engine's recompute (which T2's sender
    checks against its cache before leaving layer 0 out)."""

    def __init__(self, words):
        self.words = words                          # [L, T, W] on the device
        self.last_check = "offline: the captured layer 0"

    def covers(self, pos0, T):
        return True

    def matches(self, l0, pos0):
        return True

    def steps_for(self, pos0, T):
        return None

    def layer0(self, pos0, T, steps):
        return self.words[0, pos0:pos0 + T]


def t2_code(codec, words, pos0, timer, tag):
    """(bytes, exact) of words [L, T, W] (on the GPU) coded with a T2 codec, layer 0 recomputed."""
    import torch
    from kv_compare.pd.wire import bundle, payload_bytes
    kw = {}
    if getattr(codec, "needs_ctx", False):         # T2: layer 0 recomputed (here: the captured one)
        kw["ctx"] = OfflineLayer0(torch.cat([torch.zeros(words.shape[0], pos0, words.shape[2], dtype=words.dtype,
                                                         device=words.device), words], 1) if pos0 else words)
    with timer(f"{tag} encode"):
        parts, meta = codec.encode(words, pos0=pos0, **kw)
    header, _blob = bundle(parts, meta)
    n = payload_bytes(header)
    out = torch.empty_like(words)
    with timer(f"{tag} decode"):
        codec.decode(parts, meta, lambda l, w: out[l].copy_(w), **kw)
    return n, bool(torch.equal(out, words)), meta.get("layer0", "coded")


def rotate_f32(keys, shift, inv, heads, hd):
    """Keys [n, heads*hd] (BF16 words, a torch tensor) turned on by `shift` positions, as float32 (not rounded):
    the prediction (the same arithmetic in encoder and decoder)."""
    import torch
    x = keys.view(torch.bfloat16).to(torch.float32).view(-1, heads, hd)
    ang = (float(shift) * torch.as_tensor(inv, dtype=torch.float64)).to(torch.float32).to(keys.device)
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(keys.shape[0], -1)


def prediction(B, shift, inv, heads, hd, C):
    """mu [L-1, 16, W] float32 from base words B [L, 16, W] (layers 1..): values as they are, keys turned."""
    import torch
    mu = B[1:].view(torch.bfloat16).to(torch.float32).clone()
    if inv is not None and shift:
        for l in range(B.shape[0] - 1):
            mu[l, :, :C] = rotate_f32(B[l + 1, :, :C], shift, inv, heads, hd)
    return mu


BUCKETS = (0, 16, 64, 256)


def bucket_of(agree):
    """[16] agreement lengths -> bucket index 0..3."""
    return np.searchsorted(np.array(BUCKETS[1:]), agree, side="right")


def agreement(ids_t, s0, ids_b, j0, bs):
    """Per token of the page: how many tokens up to it the target's text has agreed with the base's (backwards from
    the page start, then the page itself)."""
    g = 0
    while s0 - 1 - g >= 0 and j0 - 1 - g >= 0 and ids_t[s0 - 1 - g] == ids_b[j0 - 1 - g] and g < 4096:
        g += 1
    return np.arange(bs) + g


class Spread:
    """s = scale[l, kv, channel] * factor[l, kv, bucket], fitted from residuals |target - mu| (logistic: mean |r| =
    2 ln 2 * s)."""

    def __init__(self, L1, C):
        self.L1, self.C = L1, C
        self.sum = np.zeros((L1, 2, C))
        self.cnt = 0
        self.bsum = np.zeros((L1, 2, len(BUCKETS)))
        self.bcnt = np.zeros((L1, 2, len(BUCKETS)))

    def add(self, T, mu, buckets):
        import torch
        r = (T[1:].view(torch.bfloat16).to(torch.float32) - mu).abs().double().cpu().numpy()   # [L1, 16, W]
        r = r.reshape(self.L1, r.shape[1], 2, self.C)
        self.sum += r.sum(1)
        self.cnt += r.shape[1]
        m = r.mean(3)                                               # [L1, 16, 2]
        for k in range(len(BUCKETS)):
            sel = buckets == k
            if sel.any():
                self.bsum[:, :, k] += m[:, sel, :].sum(1)
                self.bcnt[:, :, k] += sel.sum()

    def fit(self, floor=1e-7):
        scale = self.sum / max(self.cnt, 1) / (2 * LN2)
        mean_all = (self.bsum.sum(2) / np.maximum(self.bcnt.sum(2), 1))[:, :, None]
        factor = np.where(self.bcnt > 0, (self.bsum / np.maximum(self.bcnt, 1)) / np.maximum(mean_all, 1e-30), 1.0)
        self.scale, self.factor = np.maximum(scale, floor), factor
        return self

    def s(self, buckets, device):
        import torch
        f = self.factor[:, :, buckets]                              # [L1, 2, 16]
        s = self.scale[:, None, :, :] * f.transpose(0, 2, 1)[:, :, :, None]   # [L1, 16, 2, C]
        return torch.as_tensor(s.reshape(self.L1, len(buckets), 2 * self.C), dtype=torch.float32, device=device)


def delta_code(coder, T, mu, s, timer, tag):
    """(bytes, exact) of target layers 1.. coded against mu, s with the slice coder."""
    import torch
    w = T[1:].reshape(-1).contiguous()
    with timer(f"{tag} encode"):
        buf, starts, n = coder.encode(w, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous())
    nbytes = coder.bits(starts, n) / 8
    with timer(f"{tag} decode (base warm)"):
        back = coder.decode(buf, starts, mu.reshape(-1).contiguous(), s.reshape(-1).contiguous(), n)
    return nbytes, bool(torch.equal(back.view(torch.int16), w.view(torch.int16)))


# ----------------------------------------------------------------------------------------------- the run
PAGE_PAD = 32                                       # T's programs take 16 or 32 tokens: a page padded to 32


def analyse(cap, run_dir, codecs_file, model_dir, out_dir, names=("t2-hbh", "t2-rank256"), page_sample=0, seed=0,
            device="cuda", log=print):
    log("== theirs first (scripts/pd_page_compare.py, zlib-1; its worker processes start before CUDA does)")
    theirs = pg.analyse(cap, run_dir, model_dir)
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
    fetch = lambda loc: torch.from_numpy(words.get(loc).transpose(1, 0, 2).copy()).to(dev)   # [L, 16, W]
    built, errors, _meta = cdm.build(codecs_file, dev, list(names), log=log)
    if errors:
        log(f"codecs not available: {errors}")
    coder = gc.SliceCoder(dev)
    timer = Timer()
    req_info = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request" and r.get("missing", 0) == 0:
                req_info[r["key"]] = (s.pid, r["rows"], r["T"])
    out = {"meta": {"layers": L, "block_size": bs, "page_bytes": L * bs * W * 2, "rope": inv is not None,
                    "modes": list(built), "codec_errors": errors}, "runs": {}}
    rng = np.random.default_rng(seed)
    for name, run in runs.items():
        reqs, pages = run["requests"], run["pages"]
        log(f"== {name}: {len(reqs)} requests, {len(pages)} pages")
        res = {"requests": len(reqs), "pages": len(pages), "raw_bytes": len(pages) * out["meta"]["page_bytes"]}
        # references: complete blocks of the requests that were already stored (identical bytes)
        blocks = sum(req_info[k][2] // bs for _t, _d, k in reqs)
        res["references"] = {"blocks": blocks, "distinct_pages": len(pages), "reference_blocks": blocks - len(pages)}
        # T2 per request (the whole prompt) and per segment (contiguous new pages)
        for mode, codec in built.items():
            tot = ok = toks = 0
            for _t, d, k in reqs:
                pid, rows, T = req_info[k]
                w = fetch([(pid, rows[t // bs], t % bs) for t in range(T // bs * bs)])
                if w.shape[1] == 0:
                    continue
                n, exact, l0 = t2_code(codec, w, 0, timer, f"{mode} per request")
                tot += n
                ok += exact
                toks += w.shape[1]
            res[f"{mode}_per_request"] = {"bytes": tot, "exact": ok, "requests": len(reqs), "tokens": toks}
        page_ix = list(range(len(pages)))
        if page_sample and len(page_ix) > page_sample:
            page_ix = sorted(rng.choice(len(page_ix), page_sample, replace=False).tolist())
        for mode, codec in built.items():
            segs = collections.defaultdict(list)
            for i in page_ix:
                p = pages[i]
                segs[p["r"]].append(p)
            tot = ok = n_seg = 0
            for r, ps in segs.items():
                ps.sort(key=lambda p: p["block"])
                runs_ = [[ps[0]]]
                for p in ps[1:]:
                    (runs_[-1].append(p) if p["block"] == runs_[-1][-1]["block"] + 1 else runs_.append([p]))
                for seg in runs_:
                    w = torch.cat([fetch(p["loc"]) for p in seg], 1)
                    n, exact, _ = t2_code(codec, w, seg[0]["block"] * bs, timer, f"{mode} per segment")
                    tot += n
                    ok += exact
                    n_seg += 1
            res[f"{mode}_per_segment"] = {"bytes": tot, "exact": ok, "segments": n_seg}
        # T2 per page (16 tokens, padded to PAGE_PAD, not 256: the padding is in the sizes)
        per_page = {}
        old_pad = cdm.PAD
        cdm.PAD = PAGE_PAD
        try:
            for mode, codec in built.items():          # each mode on its own: one failing leaves the other
                try:
                    sizes, ok = {}, 0
                    for i in page_ix:
                        p = pages[i]
                        n, exact, _ = t2_code(codec, fetch(p["loc"]), p["block"] * bs, timer, f"{mode} per page")
                        sizes[i] = n
                        ok += exact
                    per_page[mode] = sizes
                    res[f"{mode}_per_page"] = {"bytes": sum(sizes.values()), "exact": ok, "pages": len(sizes)}
                except Exception as exc:
                    res.setdefault("per_page_error", f"{mode}: {exc!r}"[:300])
                    log(f"   {mode} per page failed: {exc!r}")
        finally:
            cdm.PAD = old_pad
        # raw page from host memory (pinned)
        if dev.type == "cuda":
            host = torch.empty(L, bs, W, dtype=torch.int16).pin_memory()
            for i in page_ix[:200]:
                with timer("raw page host to GPU"):
                    host.to(dev, non_blocking=True)
        # delta per page: fit the spread on the even conversations, code every page with a base
        convs = sorted({p["conv"] for p in pages})
        train = {c for k, c in enumerate(convs) if k % 2 == 0}
        sp = Spread(L - 1, C)
        cand = {}
        for i in page_ix:
            p = pages[i]
            for rule in ("text", "position"):
                b = p["bases"].get(rule)
                if not b:
                    continue
                ids_t = reqs[p["r"]][1]["ids"]
                ids_b = reqs[b["req"]][1]["ids"]
                s0 = p["block"] * bs
                j0 = s0 - b["shift"]
                buckets = bucket_of(agreement(ids_t, s0, ids_b, j0, bs))
                cand[(i, rule)] = (b, buckets)
                if p["conv"] in train:
                    T = fetch(p["loc"])
                    sp.add(T, prediction(fetch(b["loc"]), b["shift"], inv, heads, hd, C), buckets)
        sp.fit()
        res["spread"] = {"train_conversations": sorted(train), "factor_by_bucket_mean": sp.factor.mean((0, 1)).tolist()}
        dsize, dok = {}, 0
        for (i, rule), (b, buckets) in cand.items():
            p = pages[i]
            T = fetch(p["loc"])
            B = fetch(b["loc"])
            mu = prediction(B, b["shift"], inv, heads, hd, C)
            s = sp.s(buckets, dev)
            n, exact = delta_code(coder, T, mu, s, timer, "delta")
            dsize[(i, rule)] = n + pg.DESC * len(b["pages"])
            dok += exact
        best = {}                                   # per page, its smaller delta (by text or by position)
        for (i, _rule), n in dsize.items():
            best[i] = min(best.get(i, n), n)
        res["delta"] = {"coded": len(dsize), "exact": dok, "pages": len(best), "bytes": int(sum(best.values())),
                        "raw_bytes": len(best) * out["meta"]["page_bytes"],
                        "heldout_bytes": int(sum(n for i, n in best.items() if pages[i]["conv"] not in train)),
                        "heldout_pages": sum(1 for i in best if pages[i]["conv"] not in train)}
        # cold rebuild: the base decoded first (its page with T2, then the delta) for a sample
        mode0 = next(iter(built), None)
        if mode0 and mode0 in per_page:
            old_pad = cdm.PAD
            cdm.PAD = PAGE_PAD
            try:
                by_page = {qq["page"]: qq for qq in pages}
                for (i, rule) in list(dsize)[:100]:
                    b, buckets = cand[(i, rule)]
                    stored = []                     # the base pages as stored (T2 per page), coded beforehand
                    for pr in b["pages"]:
                        q = by_page.get(tuple(pr))
                        if q is not None:
                            w = fetch(q["loc"])
                            ctx = OfflineLayer0(w)
                            kw = {"ctx": ctx} if getattr(built[mode0], "needs_ctx", False) else {}
                            stored.append((w, kw) + built[mode0].encode(w, pos0=0, **kw))
                    T = fetch(pages[i]["loc"])
                    s = sp.s(buckets, dev)
                    mu0 = prediction(fetch(b["loc"]), b["shift"], inv, heads, hd, C)
                    buf, starts, n = coder.encode(T[1:].reshape(-1), mu0.reshape(-1), s.reshape(-1))
                    with timer("rebuild cold (base pages decoded with T2, then the delta)"):
                        for w, kw, parts, meta in stored:
                            back = torch.empty_like(w)
                            built[mode0].decode(parts, meta, lambda l, x, back=back: back[l].copy_(x), **kw)
                        mu = prediction(fetch(b["loc"]), b["shift"], inv, heads, hd, C)
                        coder.decode(buf, starts, mu.reshape(-1), s.reshape(-1), n)
            finally:
                cdm.PAD = old_pad
        # graphs: ours (per page: T2 page or delta, depth one) against T2 per page alone
        for mode in per_page:
            fake = []
            sub = [pages[i] for i in page_ix]
            for k, i in enumerate(page_ix):
                x = {"ind": {"split": per_page[mode][i]}, "cand": {}}
                for rule in ("text", "position"):
                    if (i, rule) in dsize:
                        x["cand"][rule] = (dsize[(i, rule)], "delta")
                fake.append(x)
            g = pg.graph(sub, fake, ("text", "position"))
            odd = [k for k, i in enumerate(page_ix) if pages[i]["conv"] not in train]
            g_odd = pg.graph([sub[k] for k in odd], [fake[k] for k in odd], ("text", "position")) if odd else None
            res[f"graph_{mode}"] = g
            res[f"graph_{mode}_heldout"] = g_odd
        # theirs on the same pages
        out["runs"][name] = res
        log(f"   {name}: " + json.dumps({k: v for k, v in res.items() if k.endswith(("per_request", "per_page"))})[:600])
    out["theirs"] = theirs
    out["times"] = timer.summary()
    pathlib.Path(out_dir, "page_compare.md").write_text(pg.markdown(theirs))
    return out


# ----------------------------------------------------------------------------------------------- report
def markdown(res):
    if res.get("error"):
        return "# KV Diff Graph against EvidenceStore\n\n" + res["error"] + "\n"
    mb = lambda b: f"{b / 1e6:,.1f}"
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    m = res["meta"]
    L = ["# KV Diff Graph against EvidenceStore, on the kept capture", "",
         f"Pages of 16 tokens x {m['layers']} layers ({m['page_bytes']:,} B raw). Ours (t2-*): layer 0 recomputed from the "
         "token ids (taken as exact offline; T2 checks it per request in the engine); per page, T pads a page to "
         f"{PAGE_PAD} tokens (its programs' size), and that padding is in the sizes; the rest with T2 or as a delta "
         "coded by T's value-slice coder against the base (keys turned by RoPE). Every representation decoded and "
         "compared byte for byte. Theirs: zlib level 1 (scripts/pd_page_compare.py).", ""]
    for name, r in res["runs"].items():
        raw = r["raw_bytes"]
        th = (res.get("theirs") or {}).get("runs", {}).get(name, {})
        L += [f"## {name}", "",
              f"{r['requests']} requests; {r['references']['blocks']:,} complete blocks, of them "
              f"{r['references']['reference_blocks']:,} identical to a stored one (references) and "
              f"{r['pages']:,} distinct pages ({mb(raw)} MB raw).", "",
              "Every row covers the distinct pages, except T2 per request: it codes each request's whole prompt, the "
              "pages identical to stored ones included (no page is shared between requests), against that prompt's "
              "own raw size. Bits: per value (16 raw).", "",
              "| Representation | MB | Bits | Saving vs raw | Decoded exactly |",
              "| --- | ---: | ---: | ---: | --- |", f"| Raw | {mb(raw)} | 16.00 | - | - |"]
        bits = lambda b, base: f"{16 * b / base:.2f}"
        if th:
            ind = th['independent']['split']
            L.append(f"| Theirs: best independent (zlib, exponent / sign-fraction streams) | "
                     f"{mb(ind)} | {bits(ind, th['raw_bytes'])} | {pc(1 - ind / th['raw_bytes'])} | - |")
            for k, lab in (("graph_theirs", "Theirs: graph (position, text)"),
                           ("graph_turned", "Theirs: graph + keys turned")):
                g = th[k]
                L.append(f"| {lab} | {mb(g['graph'])} | {bits(g['graph'], th['raw_bytes'])} | "
                         f"{pc(1 - g['graph'] / th['raw_bytes'])} | - |")
        for mode in m["modes"]:
            for unit in ("per_request", "per_segment", "per_page"):
                x = r.get(f"{mode}_{unit}")
                if not x:
                    continue
                cnt = x.get("requests") or x.get("segments") or x.get("pages")
                base = raw if unit != "per_request" else x["tokens"] / 16 * m["page_bytes"]
                lab = (f"{mode} per request (whole prompts: {x['tokens'] / 16:,.0f} blocks, repeats included)"
                       if unit == "per_request" else f"{mode} {unit.replace('_', ' ')}")
                L.append(f"| Ours: {lab} | {mb(x['bytes'])} | {bits(x['bytes'], base)} | {pc(1 - x['bytes'] / base)} | "
                         f"{x['exact']} of {cnt} |")
            g = r.get(f"graph_{mode}")
            if g:
                L.append(f"| Ours: graph ({mode} page or delta): {g['deltas']} deltas on {g['bases']} bases | "
                         f"{mb(g['graph'])} | {bits(g['graph'], raw)} | {pc(1 - g['graph'] / raw)} (beyond {mode} "
                         f"per page: {pc(g['saving_beyond_independent'])}) | deltas {r['delta']['exact']} of "
                         f"{r['delta']['coded']} |")
                h = r.get(f"graph_{mode}_heldout")
                if h:
                    L.append(f"| Ours: the same, odd conversations only (spread fitted on the even ones) | "
                             f"{mb(h['graph'])} | - | beyond {mode} per page: {pc(h['saving_beyond_independent'])} | - |")
        dl = r.get("delta") or {}
        if dl.get("pages"):
            L.append(f"| Ours: delta against the base alone, on the {dl['pages']:,} pages that have one (layer 0 "
                     f"recomputed) | {mb(dl['bytes'])} | {bits(dl['bytes'], dl['raw_bytes'])} | "
                     f"{pc(1 - dl['bytes'] / dl['raw_bytes'])} | {dl['exact']} of {dl['coded']} |")
            if dl.get("heldout_pages"):
                hb = dl["heldout_pages"] * m["page_bytes"]
                L.append(f"| Ours: the same, odd conversations only ({dl['heldout_pages']:,} pages) | "
                         f"{mb(dl['heldout_bytes'])} | {bits(dl['heldout_bytes'], hb)} | "
                         f"{pc(1 - dl['heldout_bytes'] / hb)} | - |")
        if r.get("per_page_error"):
            L.append(f"\nT2 per page did not run: {r['per_page_error']}")
        L.append("")
    t = res.get("times") or {}
    if t:
        L += ["## Times (CUDA-synchronised, one RTX A5000)", "", "| Step | n | mean ms | p50 ms |",
              "| --- | ---: | ---: | ---: |"]
        for k in sorted(t):
            x = t[k]
            L.append(f"| {k} | {x['n']:,} | {x['mean_ms']:.2f} | {x['p50_ms']:.2f} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--codecs-file", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codecs", default="t2-hbh,t2-rank256", help="T codecs to compare (t2-* recompute layer 0)")
    ap.add_argument("--page-sample", type=int, default=0, help="pages measured per run (0: all)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    res = analyse(a.capture, a.run, a.codecs_file, a.model_dir, a.out, tuple(a.codecs.split(",")), a.page_sample,
                  device=a.device)
    text = markdown(res)
    pathlib.Path(a.out, "diffgraph_report.json").write_text(json.dumps(res, indent=1, default=str))
    pathlib.Path(a.out, "diffgraph_report.md").write_text(text)
    print(text)
    return 0 if not res.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
