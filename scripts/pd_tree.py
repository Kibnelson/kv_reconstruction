#!/usr/bin/env python3
"""The conversations' block tree, kept by a rule, the rest recreated by recompute (docs/pd_delta.md, idea 1).

  python3 scripts/pd_tree.py --capture CAPTURE_DIR --run RUN_DIR --codecs-file pd_codecs.pt --model-dir MODEL \
      --out OUT_DIR

Runs on one GPU (MATRIX=tree: after setup, no vLLM engines), over a capture kept by MATRIX=capture and its run's
records. The capture keys every block by its bytes, per layer: two requests' blocks are the same page exactly when
their bytes are. P recomputed every prompt in that run (no prefix cache), so a page seen again is a recompute that came
out byte-identical; D compared the pages it held with P's recompute (trace_report.json, held_recomputed).

Replay: the requests in arrival order. A store holds pages by a rule. A request's blocks found in it from its first
block on are reused (what a prefix cache can reuse: hits); the rest are computed (prefill). Rules:
  everything      every distinct page kept (as EvidenceStore keeps every page)
  newest_path     per conversation, the latest request's pages; the rest dropped
  newest_no_docs  the same without its document pages
  hindsight       a page kept from its first use to its last (it knows the future: the least store for the most hits)
  none            nothing kept
Per rule: its hits, the prefill left (tokens, and seconds at the run's own prefill rate), and the store at its peak:
pages, raw GB, and coded with T2 (hbh, rank256; layer 0 recomputed: offline the capture's layer 0, as in
pd_diffgraph.py). Each run of contiguous kept pages of the request that wrote them is coded together, decoded back and
checked byte for byte, with encode and decode times per page.

Padding (for the next step): per request, T2 as now (zeros up to a multiple of 256) against the same tokens without
padding (the multiple of 256 below them), and against the tail under 256 coded apart, padded to 32 with zeros or with
copies of its last token, and the whole padded to 256 with copies of the last token.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
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
RULES = ("everything", "newest_path", "newest_no_docs", "hindsight", "none")
LABEL = {"everything": "Everything (every distinct page)", "newest_path": "Newest path per conversation",
         "newest_no_docs": "Newest path, no document pages", "hindsight": "Hindsight (kept until its last use)",
         "none": "Nothing kept"}
TAIL_PAD = 32           # T's programs: 32 tokens
PROBE = 24              # requests per layout in the padding probe


# ------------------------------------------------------------------------------------------------------- replay
def token_keys(ids, nb, bs):
    """Per complete block, a hash of the token ids up to its end (what a prefix cache keys a block by)."""
    h, out = hashlib.blake2b(digest_size=16), []
    for b in range(nb):
        h.update(np.asarray(ids[b * bs:(b + 1) * bs], dtype=np.int64).tobytes())
        out.append(h.copy().digest())
    return out


def replay(reqs, req, bs):
    """Every rule over the requests in order: per rule its counts and its store at the peak (a set of pages)."""
    pages_of, kinds = [], []
    for _t, d, k in reqs:
        pid, rows, T = req[k]
        pages_of.append([(pid, int(rows[b])) for b in range(T // bs)])
        kinds.append([da.kind_at(d["pieces"], b * bs) for b in range(T // bs)])
    last = {}
    for r, ps in enumerate(pages_of):
        for p in ps:
            last[p] = r
    out, peaks = {}, {}
    for rule in RULES:
        store, per_conv, seen = set(), {}, set()
        c = {"blocks": 0, "hits": 0, "hits_anywhere": 0, "computed_again": 0, "computed_tokens": 0, "tokens": 0,
             "peak_pages": 0, "peak_after_request": -1}
        peak = set()
        for r, ((_t, d, k), ps) in enumerate(zip(reqs, pages_of)):
            T = req[k][2]
            prefix, nhit = True, 0
            for p in ps:
                inn = p in store
                prefix = prefix and inn
                nhit += prefix
                c["hits_anywhere"] += inn
                c["computed_again"] += (not inn) and p in seen
            c["blocks"] += len(ps)
            c["hits"] += nhit
            c["tokens"] += T
            c["computed_tokens"] += T - nhit * bs
            seen.update(ps)
            if rule == "everything":
                store.update(ps)
            elif rule in ("newest_path", "newest_no_docs"):
                per_conv[d["conv"]] = {p for p, kd in zip(ps, kinds[r]) if rule == "newest_path" or kd != "doc"}
                store = set().union(*per_conv.values())
            elif rule == "hindsight":
                store = {p for p in seen if last[p] > r}
            if len(store) > c["peak_pages"]:
                c["peak_pages"], c["peak_after_request"], peak = len(store), r, set(store)
        out[rule], peaks[rule] = c, peak
    return out, peaks, pages_of


def identity(reqs, req, bs):
    """Token prefixes against the capture's bytes: blocks whose token prefix an earlier block had, with the same bytes
    (an exact recompute) or not (a recompute that differed); and blocks with an earlier block's bytes under another
    token prefix (the records' ids are not what the engine saw)."""
    first_tok, first_row = {}, {}
    same = differ = other_prefix = 0
    for _t, d, k in reqs:
        pid, rows, T = req[k]
        nb = T // bs
        for key, b in zip(token_keys(d["ids"], nb, bs), range(nb)):
            row = (pid, int(rows[b]))
            if key in first_tok:
                same += first_tok[key] == row
                differ += first_tok[key] != row
            elif row in first_row:
                other_prefix += 1
            first_tok.setdefault(key, row)
            first_row.setdefault(row, key)
    return {"same_prefix_same_bytes": same, "same_prefix_other_bytes": differ, "other_prefix_same_bytes": other_prefix}


def kept_runs(peak, pages_of, bs):
    """The kept pages as runs of contiguous blocks of the request that first wrote them: [(pos0, [pages])]."""
    writer = {}
    for r, ps in enumerate(pages_of):
        for b, p in enumerate(ps):
            writer.setdefault(p, (r, b))
    by = sorted((writer[p], p) for p in peak)
    runs = []
    for (r, b), p in by:
        if runs and runs[-1][0] == r and runs[-1][1] + len(runs[-1][2]) == b:
            runs[-1][2].append(p)
        else:
            runs.append([r, b, [p]])
    return [(b * bs, ps) for _r, b, ps in runs]


# ------------------------------------------------------------------------------------------------- the run's rates
def prefill_rate(run_dir, keys):
    """Median ms per prompt token of P's prefill in the capture's run: request arrival (proxy t_in) to the end of its
    prefill (P's send record, finished_at), over the requests replayed. P had no prefix cache there."""
    run_dir = pathlib.Path(run_dir)
    t_in = {p["key"]: p.get("t_in") for f in da.files(run_dir, "proxy.jsonl") for p in da.jl(f)}
    per = []
    for f in da.files(run_dir / "connector", "tkv_Pw_*.jsonl"):
        for r in da.jl(f):
            k = r.get("key")
            if r.get("event") == "send" and k in keys and t_in.get(k) and r.get("finished_at") and r.get("tokens"):
                if int(r.get("t0", 0)) == 0:
                    per.append(1e3 * (r["finished_at"] - t_in[k]) / int(r["tokens"]))
    return (float(np.median(per)) if per else None), len(per)


def held_check(run_dir):
    """D's held pages against P's recompute, per layout (trace_report.json: pages are 16 tokens of one layer)."""
    p = pathlib.Path(run_dir) / "trace_report.json"
    if not p.is_file():
        return {}
    out = {}
    for name, x in json.loads(p.read_text()).items():
        h = (x.get("relations") or {}).get("held_recomputed") if isinstance(x, dict) else None
        if h:
            out[name] = {"layer_pages": h.get("pages"), "differing": h.get("pages_differing")}
    return out


# -------------------------------------------------------------------------------------------------------- analysis
def analyse(cap, run_dir, codecs_file, model_dir, out_dir, names=("t2-hbh", "t2-rank256"), device="cuda",
            probe=PROBE, log=print):
    import torch
    from kv_compare.pd import codecs as cdm
    dev = torch.device(device)
    stores, runs, err = pg.collect(cap, run_dir)
    if err:
        return {"error": err}
    st0 = stores["P"][0]
    L, bs, W = st0.L, st0.bs, st0.W
    page_bytes = L * bs * W * 2
    words = da.Words(stores["P"])
    fetch = lambda loc: torch.from_numpy(words.get(loc).transpose(1, 0, 2).copy()).to(dev)   # [L, n, W]
    built, errors, _meta = cdm.build(codecs_file, dev, list(names), log=log)
    if errors:
        log(f"codecs not available: {errors}")
    req = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request" and r.get("missing", 0) == 0:
                req[r["key"]] = (s.pid, r["rows"], r["T"])
    timer, pages_done = dg.Timer(), collections.Counter()
    keys = {k for run in runs.values() for _t, _d, k in run["requests"]}
    rate, n_rate = prefill_rate(run_dir, keys)
    out = {"meta": {"layers": L, "block_size": bs, "page_bytes": page_bytes, "modes": list(built),
                    "codec_errors": errors, "prefill_ms_per_token": rate, "prefill_requests": n_rate,
                    "held_check": held_check(run_dir)}, "runs": {}}
    coded = {}                                       # (mode, pos0, pages) -> (bytes, exact): runs shared by rules

    def code_run(mode, codec, pos0, ps):
        key = (mode, pos0, tuple(ps))
        if key not in coded:
            w = fetch([(pid, row, o) for pid, row in ps for o in range(bs)])
            n, exact, _ = dg.t2_code(codec, w, pos0, timer, f"{mode} kept")
            pages_done[f"{mode} kept"] += len(ps)
            coded[key] = (n, exact)
        return coded[key]

    for name, run in runs.items():
        reqs = run["requests"]
        log(f"== {name}: {len(reqs)} requests")
        counts, peaks, pages_of = replay(reqs, req, bs)
        res = {"requests": len(reqs), "distinct_pages": len({p for ps in pages_of for p in ps}),
               "identity": identity(reqs, req, bs), "rules": {}}
        for rule in RULES:
            c = dict(counts[rule])
            c["peak_raw_bytes"] = c["peak_pages"] * page_bytes
            runs_ = kept_runs(peaks[rule], pages_of, bs)
            c["peak_runs"] = len(runs_)
            for mode, codec in built.items():
                tot = ok = 0
                for pos0, ps in runs_:
                    n, exact = code_run(mode, codec, pos0, ps)
                    tot += n
                    ok += exact
                c[f"{mode}_bytes"], c[f"{mode}_exact"] = tot, ok
            res["rules"][rule] = c
            log(f"   {rule}: hits {c['hits']} of {c['blocks']}; peak {c['peak_pages']} pages in {len(runs_)} runs")
        res["probe"] = padding_probe(reqs, req, bs, built, fetch, timer, cdm, probe, log)
        out["runs"][name] = res
    t = timer.summary()
    for tag, n in pages_done.items():
        for step in ("encode", "decode"):
            x = t.get(f"{tag} {step}")
            if x and n:
                x["ms_per_page"] = 1e3 * x["total_s"] / n
    out["times"] = t
    return out


def padding_probe(reqs, req, bs, built, fetch, timer, cdm, probe, log):
    """Bytes of the same requests' T2 with the padding as now and without it (see the module's docstring)."""
    import torch
    pick = [k for _t, _d, k in reqs if req[k][2] // bs * bs >= 256][:probe]
    acc = {m: collections.Counter() for m in built}

    def edge(x, mult):
        m = -(-x.shape[1] // mult) * mult - x.shape[1]
        return x if m == 0 else torch.cat([x, x[:, -1:].expand(x.shape[0], m, x.shape[2])], 1).contiguous()
    for k in pick:
        pid, rows, T = req[k]
        n = T // bs * bs
        w = fetch([(pid, int(rows[t // bs]), t % bs) for t in range(n)])
        head = n // 256 * 256
        for mode, codec in built.items():
            a = acc[mode]
            b_now, e1, _ = dg.t2_code(codec, w, 0, timer, f"{mode} probe")
            b_head, e2, _ = dg.t2_code(codec, w[:, :head].contiguous(), 0, timer, f"{mode} probe")
            b_edge, e3, _ = dg.t2_code(codec, edge(w, 256), 0, timer, f"{mode} probe")
            b_tz = b_te = 0
            e4 = e5 = True
            if n > head:
                old = cdm.PAD
                cdm.PAD = TAIL_PAD
                try:
                    tail = w[:, head:].contiguous()
                    b_tz, e4, _ = dg.t2_code(codec, tail, head, timer, f"{mode} probe")
                    b_te, e5, _ = dg.t2_code(codec, edge(tail, TAIL_PAD), head, timer, f"{mode} probe")
                finally:
                    cdm.PAD = old
            a["requests"] += 1
            a["tokens"] += n
            a["head_tokens"] += head
            a["pad_tokens_now"] += -(-n // 256) * 256 - n
            a["pad_tokens_split"] += (-(-(n - head) // TAIL_PAD) * TAIL_PAD - (n - head)) if n > head else 0
            a["now"] += b_now
            a["head_only"] += b_head
            a["split_zeros"] += b_head + b_tz
            a["split_copies"] += b_head + b_te
            a["copies_256"] += b_edge
            a["exact"] += e1 and e2 and e3 and e4 and e5
    out = {m: dict(a) for m, a in acc.items()}
    log(f"   padding probe: {len(pick)} requests")
    return out


# -------------------------------------------------------------------------------------------------------- report
def markdown(res):
    m = res["meta"]
    pb, L, bs = m["page_bytes"], m["layers"], m["block_size"]
    vals_per_tok = pb // 2 // bs
    gb = lambda b: f"{b / 1e9:,.2f}"
    bits = lambda b, n_pages: f"{8 * b / (n_pages * pb / 2):.2f}" if n_pages else "-"
    rate = m.get("prefill_ms_per_token")
    L_ = ["# The conversations' block tree, kept by a rule", "",
          f"Pages of {bs} tokens x {L} layers ({pb:,} B raw). The requests replayed in arrival order; a page is "
          "identified by its bytes (the capture's rows). Hits: blocks found in the store from the request's first "
          "block on (what a prefix cache reuses); the rest are computed. Peak: the store at its largest, coded with "
          "T2 per run of contiguous kept pages (layer 0 recomputed), decoded back and checked byte for byte. Bits: per "
          "value (16 raw)."]
    if rate:
        L_ += ["", f"Prefill in the capture's run: {rate:.3f} ms per token at the median over "
               f"{m['prefill_requests']} requests (arrival to the end of P's prefill, queueing included; P had no "
               f"prefix cache), so {rate * bs:.1f} ms to recompute a page."]
    hc = m.get("held_check") or {}
    if hc:
        L_ += ["", "Recompute is exact: D compared the pages it held with P's recompute of the same prompt prefix: " +
               "; ".join(f"{k}: {v['layer_pages']:,} pages of one layer ({v['layer_pages'] // L:,} blocks), "
                         f"{v['differing']} differing" for k, v in hc.items()) + "."]
    for name, r in res["runs"].items():
        idn = r["identity"]
        L_ += ["", f"## {name}", "",
               f"{r['requests']} requests, {r['distinct_pages']:,} distinct pages. P computed "
               f"{idn['same_prefix_same_bytes']:,} blocks again under a token prefix computed before: "
               f"{idn['same_prefix_same_bytes']:,} with the same bytes, {idn['same_prefix_other_bytes']:,} not. "
               f"Blocks with an earlier block's bytes under another token prefix in the run's records: "
               f"{idn['other_prefix_same_bytes']:,}.", "",
               "| Rule | Hits (blocks reused) | Prefill left (tokens) | Prefill left (s) | Peak pages | Peak raw GB | " +
               " | ".join(f"Peak {md} GB (bits)" for md in m["modes"]) + " | Decoded exactly |",
               "| --- | ---: | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in m["modes"]) + " | --- |"]
        for rule in RULES:
            c = r["rules"][rule]
            sec = f"{c['computed_tokens'] * rate / 1e3:,.1f}" if rate else "-"
            cells = [f"{gb(c[f'{md}_bytes'])} ({bits(c[f'{md}_bytes'], c['peak_pages'])})" if c["peak_pages"] else "-"
                     for md in m["modes"]]
            ex = "; ".join(f"{c[f'{md}_exact']} of {c['peak_runs']} runs" for md in m["modes"]) if c["peak_runs"] \
                else "-"
            L_.append(f"| {LABEL[rule]} | {c['hits']:,} of {c['blocks']:,} | {c['computed_tokens']:,} of "
                      f"{c['tokens']:,} | {sec} | {c['peak_pages']:,} | {gb(c['peak_raw_bytes'])} | " +
                      " | ".join(cells) + f" | {ex} |")
        pr = r.get("probe") or {}
        if pr:
            L_ += ["", "Padding, on the first requests of 256 tokens or more (bits per real value; the padding's "
                   "bytes counted):", "",
                   "| Codec | Requests | As now: zeros to 256 | No padding (the 256-multiple below) | Tail apart, zeros "
                   "to 32 | Tail apart, copies to 32 | Copies of the last token to 256 | Padding tokens now / split | "
                   "Exact |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
            for md, a in pr.items():
                if not a.get("requests"):
                    continue
                bt = lambda b, t: f"{8 * b / (t * vals_per_tok):.2f}"
                L_.append(f"| {md} | {a['requests']} | {bt(a['now'], a['tokens'])} | "
                          f"{bt(a['head_only'], a['head_tokens'])} | {bt(a['split_zeros'], a['tokens'])} | "
                          f"{bt(a['split_copies'], a['tokens'])} | {bt(a['copies_256'], a['tokens'])} | "
                          f"{a['pad_tokens_now']:,} / {a['pad_tokens_split']:,} | {a['exact']} of {a['requests']} |")
    t = res.get("times") or {}
    if t:
        L_ += ["", "## Times (CUDA-synchronised, one GPU)", "", "| Step | n | ms per page | mean ms per call |",
               "| --- | ---: | ---: | ---: |"]
        for k in sorted(t):
            x = t[k]
            mpp = f"{x['ms_per_page']:.3f}" if "ms_per_page" in x else "-"
            L_.append(f"| {k} | {x['n']:,} | {mpp} | {x['mean_ms']:.2f} |")
        if rate:
            L_.append(f"| recompute (the run's prefill) | - | {rate * bs:.3f} | - |")
    return "\n".join(L_) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--codecs-file", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--codecs", default="t2-hbh,t2-rank256")
    ap.add_argument("--probe", type=int, default=PROBE)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    res = analyse(a.capture, a.run, a.codecs_file, a.model_dir, a.out, names=tuple(a.codecs.split(",")),
                  device=a.device, probe=a.probe)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tree_report.json").write_text(json.dumps(res, indent=1, default=str))
    if "error" in res:
        print("STOP:", res["error"])
        return 1
    text = markdown(res)
    (out / "tree_report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
