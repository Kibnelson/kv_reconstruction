#!/usr/bin/env python3
"""The other team's differential-page measures (EvidenceStore's note on job 30547: LMCache + CacheBlend, one
conversation) applied to our captured pages (MATRIX=capture: our connector, vLLM's prefix caching, no Blend), so
their findings can be confirmed or not on our data, by the same methods.

  python3 -I scripts/pd_page_compare.py --capture CAPTURE_DIR --run RUN_DIR [--model-dir MODEL] \
      [--markdown page_compare.md] [--json page_compare.json]

Pages: every complete 16-token block P stored for a captured request (each distinct block once: the first request
that wrote it), all layers, keys and values, laid out as theirs ([layer][K then V][token][row bytes]). A row is the
key or value of one token at one layer across the KV heads (2,048 B).

  independent  zlib level 1 of each page: the raw bytes; byte-shuffled (low bytes, then high bytes); the BF16
               exponent and sign-and-fraction as two streams (their best independent encoding; 4 B per stream)
  delta        a whole-page residual against an earlier page of the same conversation (an earlier turn): XOR or
               the difference mod 2^16, each in the three layouts above, the smallest kept; 48 B of framing and
               64 B per base page. Bases (the latest earlier turn's page by each rule):
                 position  the block at the same token positions (their "absolute start token"); "aligned"
                           when its tokens are the same too (their byte-match pairs)
                 text      the same 16 token ids anywhere (a moved copy; may span two stored pages)
                 turned    the same text, its keys first turned by RoPE to the new position (our addition)
  graph        depth one, as theirs: a page is a delta or a base, never both; deltas chosen greedily by saving;
               totals against the independent encoding (bases counted once, at their independent size)
  bytes        high-byte and low-byte match rates of target and base, keys and values, layer 0 and the rest
  rows         rows of a page byte-identical to a row of an earlier page of its conversation, at the same slot (the
               position base) or at any slot (any earlier page; and earlier turns only)
  layers       each layer as an XOR against the previous one (byte-shuffled zlib, per layer) against each layer alone
numpy and the standard library only (scripts/pd_delta_analysis.py for the store and RoPE).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pd_delta_analysis as da  # noqa: E402

np = da.np
FRAME, DESC, STREAM = 48, 64, 4
_S: dict = {}


# ----------------------------------------------------------------------------------------------- encodings
def their_order(page, C):
    """[L, 16, W] int16 (K then V per token) -> uint16 [L, 2, 16, C] flat, in [layer][K then V][token][row]."""
    L, T, W = page.shape
    return np.ascontiguousarray(page.view(np.uint16).reshape(L, T, 2, C).transpose(0, 2, 1, 3)).ravel()


def zsize(b):
    return len(zlib.compress(b, 1))


def enc(u):
    """zlib-1 sizes of uint16 words u in the three layouts."""
    b = u.view(np.uint8).reshape(-1, 2)
    exp = ((u >> 7) & 0xFF).astype(np.uint8)
    sf = (((u >> 8) & 0x80) | (u & 0x7F)).astype(np.uint8)
    return {"raw": zsize(u.tobytes()),
            "shuffle": zsize(np.ascontiguousarray(b[:, 0]).tobytes() + np.ascontiguousarray(b[:, 1]).tobytes()),
            "split": zsize(exp.tobytes()) + zsize(sf.tobytes()) + 2 * STREAM}


def residual_size(ut, ub, nbases):
    """The smallest whole-page residual of target words ut against base words ub (XOR or mod-2^16 difference, each
    in the three layouts), with framing and base descriptors."""
    best = None
    for name, r in (("xor", ut ^ ub), ("sub", (ut.astype(np.int32) - ub.astype(np.int32)) & 0xFFFF)):
        r = r.astype(np.uint16)
        for lay, n in enc(r).items():
            if best is None or n < best[0]:
                best = (n, f"{name}/{lay}")
    return best[0] + FRAME + DESC * nbases, best[1]


def byte_match(t, b, C):
    """High- and low-byte match rates, keys and values, layer 0 and layers 1+ (t, b: [L, 16, W] int16)."""
    out = {}
    tu, bu = t.view(np.uint16), b.view(np.uint16)
    for kv, sl in (("K", slice(0, C)), ("V", slice(C, 2 * C))):
        for grp, ls in (("L0", slice(0, 1)), ("L1+", slice(1, None))):
            x, y = tu[ls, :, sl], bu[ls, :, sl]
            out[f"{kv} {grp}"] = (float(((x >> 8) == (y >> 8)).mean()), float(((x & 0xFF) == (y & 0xFF)).mean()),
                                  int(x.size))
    return out


def prev_layer(page):
    """(each layer from 1 on as an XOR against the previous one, each layer alone): byte-shuffled zlib-1 sizes."""
    u = page.view(np.uint16)
    xo = al = 0
    for l in range(1, u.shape[0]):
        xo += enc(np.ascontiguousarray(u[l] ^ u[l - 1]).ravel())["shuffle"]
        al += enc(np.ascontiguousarray(u[l]).ravel())["shuffle"]
    return xo, al


def _page_job(k):
    """One target page: its independent sizes, each base rule's delta size and byte matches, the layer check."""
    t = _S["pages"][k]
    C, inv, heads, hd = _S["C"], _S["inv"], _S["heads"], _S["hd"]
    T = _S["fetch"](t["loc"])
    ut = their_order(T, C)
    out = {"ind": enc(ut), "cand": {}, "bytes": {}, "layer0_bytes": int(T[0].size * 2)}
    out["prev"] = prev_layer(T)
    for rule, spec in t["bases"].items():
        B = _S["fetch"](spec["loc"])
        if rule == "turned":
            if inv is None or spec["shift"] == 0:
                continue
            B = B.copy()
            for l in range(B.shape[0]):
                B[l, :, :C] = da.rotate(B[l, :, :C], np.full(B.shape[1], spec["shift"]), inv, heads, hd)
        n, how = residual_size(ut, their_order(B, C), len(spec["pages"]))
        out["cand"][rule] = (n, how)
        out["bytes"][rule] = byte_match(T, B, C)
        if spec.get("same_ids"):                    # aligned: the same tokens at the same place (their pairs)
            out["bytes"]["aligned"] = out["bytes"][rule]
    return out


# ----------------------------------------------------------------------------------------------- the pages
def collect(cap, run_dir):
    """Per run, in request order: the pages (blocks first written by that request) with their base candidates."""
    stores = da.load_stores(cap)
    if not stores["P"]:
        return None, None, "no P store in " + str(cap)
    st0 = stores["P"][0]
    bs = st0.bs
    req = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request" and r.get("missing", 0) == 0:
                req[r["key"]] = (s.pid, r["rows"], r["T"])
    run_dir = pathlib.Path(run_dir)
    drv = [r for f in da.files(run_dir / "runs", "mtrag_*.jsonl") for r in da.jl(f)]
    proxy = [r for f in da.files(run_dir, "proxy.jsonl") for r in da.jl(f)]
    key_of = {p["tag"]: p["key"] for p in proxy if p.get("tag")}
    t_in = {p["key"]: p.get("t_in", 0) for p in proxy}
    runs = collections.defaultdict(list)
    for d in drv:
        k = key_of.get(d.get("tag"))
        if d.get("error") or not d.get("ids") or k not in req:
            continue
        runs[d.get("run") or d["layout"]].append((t_in.get(k, 0), d, k))
    owner = {}                                      # (pid, row) -> the first (run, request) that wrote it
    out = {}
    for run, reqs in sorted(runs.items()):
        reqs.sort(key=lambda x: x[0])
        by_conv = collections.defaultdict(list)    # conversation -> its earlier requests (index in reqs)
        pages = []
        where = {}                                  # request -> 16-token window hash -> its last start
        for r, (_t, d, k) in enumerate(reqs):
            pid, rows, T = req[k]
            ids = d["ids"]
            hs = da.window_hashes(ids)              # hs[i]: the window starting at token i
            where[r] = {h: i for i, h in enumerate(hs)}
            earlier = by_conv[d["conv"]]
            for b in range(T // bs):                # complete blocks only
                row = rows[b]
                if (pid, row) in owner:
                    continue                        # the same bytes stored before: a reference
                owner[(pid, row)] = (run, r)
                s0 = b * bs
                bases = {}
                # position: the latest earlier turn's block at the same tokens
                for e in reversed(earlier):
                    epid, erows, eT = req[reqs[e][2]]
                    if eT >= s0 + bs and erows[b] >= 0 and (epid, erows[b]) != (pid, row):
                        eids = reqs[e][1]["ids"]
                        bases["position"] = {"loc": [(epid, erows[b], o) for o in range(bs)],
                                             "pages": [(epid, erows[b])], "shift": 0, "req": e,
                                             "same_ids": list(eids[s0:s0 + bs]) == list(ids[s0:s0 + bs])}
                        break
                # text: the latest earlier turn holding these 16 token ids anywhere
                want = list(ids[s0:s0 + bs])
                for e in reversed(earlier):
                    eids = reqs[e][1]["ids"]
                    j = where[e].get(hs[s0]) if bs == da.WINDOW and s0 < len(hs) else _find(eids, tuple(want))
                    if j is not None and list(eids[j:j + bs]) != want:
                        j = None                    # a hash collision: not the same text
                    if j is not None:
                        epid, erows, _eT = req[reqs[e][2]]
                        loc = [(epid, erows[(j + o) // bs], (j + o) % bs) for o in range(bs)]
                        if any(x[1] < 0 for x in loc):
                            break
                        pg = sorted({(x[0], x[1]) for x in loc})
                        bases["text"] = {"loc": loc, "pages": pg, "shift": s0 - j, "req": e}
                        bases["turned"] = dict(bases["text"])
                        break
                pages.append({"r": r, "conv": d["conv"], "turn": d["turn"], "block": b, "page": (pid, row),
                              "loc": [(pid, row, o) for o in range(bs)], "bases": bases,
                              "kind": da.kind_at(d["pieces"], s0)})
            earlier.append(r)
        out[run] = {"requests": reqs, "pages": pages}
    return stores, out, None


def _find(seq, want):
    """The last position of `want` in `seq` (a list of ints), or None."""
    n = len(want)
    first = want[0]
    for j in range(len(seq) - n, -1, -1):
        if seq[j] == first and tuple(seq[j:j + n]) == want:
            return j
    return None


def row_matches(stores, run):
    """Rows of each page byte-identical to a row of an earlier page of its conversation: at the same slot (the
    position base), at any slot (any earlier page, or earlier turns only); by layer 0 / 1+ and K / V."""
    words = da.Words(stores["P"])
    C = stores["P"][0].W // 2
    cnt = collections.Counter()
    seen = collections.defaultdict(dict)           # conversation -> row digest -> turn
    for p in run["pages"]:
        T = words.get(p["loc"]).transpose(1, 0, 2)  # [L, 16, W]
        pos = p["bases"].get("position")
        B = words.get(pos["loc"]).transpose(1, 0, 2) if pos else None
        conv = seen[p["conv"]]
        new = {}
        for l in range(T.shape[0]):
            g = "L0" if l == 0 else "L1+"
            for kv, sl in (("K", slice(0, C)), ("V", slice(C, 2 * C))):
                for t in range(T.shape[1]):
                    rb = T[l, t, sl].tobytes()
                    h = hash(rb)
                    cnt[(g, kv, "rows")] += 1
                    if B is not None:
                        cnt[(g, kv, "same_slot_rows")] += 1
                        if rb == B[l, t, sl].tobytes():
                            cnt[(g, kv, "same_slot")] += 1
                    hit = conv.get(h)
                    if hit is not None:
                        cnt[(g, kv, "any_slot")] += 1
                        if hit < p["turn"]:
                            cnt[(g, kv, "any_slot_earlier_turn")] += 1
                    new.setdefault(h, p["turn"])
        for h, turn in new.items():
            conv.setdefault(h, turn)
    return {f"{g} {kv}": {k: cnt[(g, kv, k)] for k in ("rows", "same_slot_rows", "same_slot", "any_slot",
                                                         "any_slot_earlier_turn")}
            for g in ("L0", "L1+") for kv in ("K", "V")}


def graph(pages, res, rules, ind_key="split"):
    """Depth-one selection: (independent total, graph total, deltas, bases, per-page best savings)."""
    ind = [x["ind"][ind_key] for x in res]
    cands = []
    for i, (p, x) in enumerate(zip(pages, res)):
        for rule in rules:
            if rule in x["cand"]:
                n = x["cand"][rule][0]
                if n < ind[i]:
                    cands.append((ind[i] - n, i, tuple(p["bases"][rule]["pages"]), n))
    page_index = {p["page"]: i for i, p in enumerate(pages)}
    cands.sort(reverse=True)
    is_delta, is_base, size = set(), set(), list(ind)
    for saving, i, bpages, n in cands:
        bidx = [page_index.get(q) for q in bpages]
        if i in is_delta or i in is_base or any(b in is_delta for b in bidx if b is not None):
            continue
        is_delta.add(i)
        size[i] = n
        is_base.update(b for b in bidx if b is not None)
    best = []
    for i, x in enumerate(res):
        c = [x["cand"][r][0] for r in rules if r in x["cand"]]
        if c:
            best.append(1 - min(c) / ind[i])
    return {"independent": int(sum(ind)), "graph": int(sum(size)), "deltas": len(is_delta), "bases": len(is_base),
            "saving_beyond_independent": 1 - sum(size) / sum(ind) if sum(ind) else None,
            "pages_with_candidate": len(best), "best_page_saving": _dist(best)}


def _dist(xs):
    if not xs:
        return None
    a = np.sort(np.array(xs))
    return {"median": float(a[len(a) // 2]), "p90": float(a[int(0.9 * (len(a) - 1))]), "max": float(a[-1]),
            "share_positive": float((a > 0).mean())}


def analyse(cap, run_dir, model_dir=None, max_pages=0, seed=0):
    stores, runs, err = collect(cap, run_dir)
    if err:
        return {"error": err}
    st0 = stores["P"][0]
    L, bs, W = st0.L, st0.bs, st0.W
    C = W // 2
    inv, hd = da.rope_inv_freq(model_dir, 128) if model_dir else (None, 128)
    words = da.Words(stores["P"])
    _S.update(C=C, inv=inv, heads=C // hd, hd=hd, fetch=lambda loc: words.get(loc).transpose(1, 0, 2).copy())
    out = {"meta": {"layers": L, "block_size": bs, "row_bytes": C * 2, "page_bytes": L * bs * W * 2,
                    "rope": inv is not None}, "runs": {}}
    rng = np.random.default_rng(seed)
    for name, run in runs.items():
        pages = run["pages"]
        if max_pages and len(pages) > max_pages:
            keep = set(rng.choice(len(pages), max_pages, replace=False).tolist())
            sampled = [p for i, p in enumerate(pages) if i in keep]
        else:
            sampled = pages
        _S["pages"] = sampled
        res = da.pmap(_page_job, range(len(sampled)))
        raw = len(sampled) * out["meta"]["page_bytes"]
        tot = {k: sum(x["ind"][k] for x in res) for k in ("raw", "shuffle", "split")}
        tot["best_per_page"] = sum(min(x["ind"].values()) for x in res)
        r = {"requests": len(run["requests"]), "pages": len(pages), "pages_measured": len(sampled), "raw_bytes": raw,
             "independent": tot,
             "with_base": {rule: sum(1 for x in res if rule in x["cand"]) for rule in ("position", "text", "turned")}}
        r["graph_theirs"] = graph(sampled, res, ("position", "text"))
        r["graph_turned"] = graph(sampled, res, ("position", "text", "turned"))
        r["graph_by_rule"] = {rule: graph(sampled, res, (rule,)) for rule in ("position", "text", "turned")}
        kinds = collections.defaultdict(list)
        for i, p in enumerate(sampled):
            kinds[p["kind"]].append(i)
        r["by_kind"] = {k: graph([sampled[i] for i in ix], [res[i] for i in ix], ("position", "text", "turned"))
                        for k, ix in sorted(kinds.items())}
        bm = collections.defaultdict(lambda: [0.0, 0.0, 0])
        for x in res:
            for rule, d in x["bytes"].items():
                for k, (hi, lo, n) in d.items():
                    a = bm[(rule, k)]
                    a[0] += hi * n
                    a[1] += lo * n
                    a[2] += n
        r["byte_match"] = {f"{rule} {k}": {"high": a[0] / a[2], "low": a[1] / a[2], "values": a[2]}
                           for (rule, k), a in sorted(bm.items()) if a[2]}
        xo = sum(x["prev"][0] for x in res)
        al = sum(x["prev"][1] for x in res)
        r["previous_layer"] = {"xor_bytes": xo, "alone_bytes": al, "change": xo / al - 1 if al else None}
        r["layer0_share"] = 1 / L
        r["rows"] = row_matches(stores, {"pages": sampled})
        out["runs"][name] = r
    return out


# ----------------------------------------------------------------------------------------------- report
def markdown(res):
    if res.get("error"):
        return "# Page comparison\n\n" + res["error"] + "\n"
    mb = lambda b: f"{b / 1e6:,.1f}"
    pc = lambda x: "-" if x is None else f"{100 * x:.1f}%"
    m = res["meta"]
    L = ["# Their page measures on our captured pages", "",
         f"Pages of 16 tokens x {m['layers']} layers ({m['page_bytes']:,} B raw; rows of {m['row_bytes']:,} B); "
         f"zlib level 1 throughout; deltas against an earlier turn of the same conversation, depth one. RoPE turning "
         f"{'on' if m['rope'] else 'off'}. Their run (job 30547, 91 pages): zlib 18.8%, best independent (exponent "
         "and sign-fraction streams) 27.5%, graph 28.9% of raw saved: 1.9% beyond independent.", ""]
    for name, r in res["runs"].items():
        raw = r["raw_bytes"]
        ind = r["independent"]
        L += [f"## {name}", "",
              f"{r['requests']} requests; {r['pages']:,} distinct complete pages ({r['pages_measured']:,} measured), "
              f"{mb(raw)} MB raw. Pages with a base: same position {r['with_base']['position']:,}, same text "
              f"{r['with_base']['text']:,}.", "",
              "| Representation | MB | Saving vs raw |", "| --- | ---: | ---: |",
              f"| Raw | {mb(raw)} | - |"]
        for k, lab in (("raw", "zlib 1"), ("shuffle", "byte shuffle + zlib"), ("split", "exponent / sign-fraction "
                                                                                       "streams + zlib (their best)"),
                       ("best_per_page", "best of the three per page")):
            L.append(f"| {lab} | {mb(ind[k])} | {pc(1 - ind[k] / raw)} |")
        for k, lab in (("graph_theirs", "graph, their bases (position, text)"),
                       ("graph_turned", "graph, + keys turned by RoPE")):
            g = r[k]
            L.append(f"| {lab}: {g['deltas']} deltas on {g['bases']} bases | {mb(g['graph'])} | "
                     f"{pc(1 - g['graph'] / raw)} (beyond independent: {pc(g['saving_beyond_independent'])}) |")
        L += ["", "Per base rule (alone): beyond independent / pages with a candidate / per-page best saving median, "
              "p90, max:", ""]
        for rule, g in r["graph_by_rule"].items():
            d = g["best_page_saving"] or {}
            L.append(f"- {rule}: {pc(g['saving_beyond_independent'])} / {g['pages_with_candidate']:,} / "
                     f"{pc(d.get('median'))}, {pc(d.get('p90'))}, {pc(d.get('max'))}")
        L += ["", "By the kind of text a page starts in (graph with all rules): beyond independent (deltas):", ""]
        for k, g in r["by_kind"].items():
            L.append(f"- {k}: {pc(g['saving_beyond_independent'])} ({g['deltas']} of {g['pages_with_candidate']} "
                     "with a candidate)")
        L += ["", "High / low byte matches with the base (their 10 aligned pairs: K 76% / 8%, V 67% / 5%):", "",
              "| Base | K L0 | K L1+ | V L0 | V L1+ |", "| --- | --- | --- | --- | --- |"]
        for rule in ("aligned", "position", "text", "turned"):
            cells = []
            for k in ("K L0", "K L1+", "V L0", "V L1+"):
                x = r["byte_match"].get(f"{rule} {k}")
                cells.append(f"{pc(x['high'])} / {pc(x['low'])}" if x else "-")
            L.append(f"| {rule} | " + " | ".join(cells) + " |")
        L += ["", "Rows byte-identical to an earlier page's row of the same conversation (their: slot remapping "
              "added 570 matches, all layer-0 V; 150 from earlier turns):", "",
              "| | rows | same slot | any slot | any slot, earlier turns |", "| --- | ---: | ---: | ---: | ---: |"]
        for k, x in r["rows"].items():
            L.append(f"| {k} | {x['rows']:,} | {x['same_slot']:,} of {x['same_slot_rows']:,} | {x['any_slot']:,} | "
                     f"{x['any_slot_earlier_turn']:,} |")
        pl = r["previous_layer"]
        L += ["", f"Each layer as an XOR against the previous one (byte-shuffled zlib per layer): {mb(pl['xor_bytes'])} "
              f"MB against {mb(pl['alone_bytes'])} MB alone ({pc(pl['change'])}; theirs +4.4%).", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-pages", type=int, default=0, help="a random sample of each run's pages (0: all)")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = analyse(a.capture, a.run, a.model_dir, a.max_pages)
    text = markdown(res)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    if a.markdown:
        pathlib.Path(a.markdown).write_text(text)
    print(text)
    return 0 if not res.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
