#!/usr/bin/env python3
"""One report over the runs of a PD matrix (scripts/pd_matrix.sh): every run folder's results.json and env.json.

  python scripts/pd_matrix_report.py --runs NAME,NAME,... --root benchmark_results --output DIR/summary.md

Tables, each codec a row:
  correctness       transfers (or tier chunks) checked bit-exact, per run
  PD sweep          per layout, path and link: requests per second and goodput, TTFT p50 / p99, TPOT, bits; the
                    best codec at each request rate by goodput (then TTFT p50); raw BF16 and vLLM's NIXL as references
  CPU tier          per number of clients: requests per second, TTFT, the tier's loads and bits; LMCache as reference
  T2                how often the receiver recomputed layer 0, and the constant sets picked
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def load(root, names):
    runs = []
    for n in names:
        d = Path(root) / n
        if not (d / "results.json").is_file():
            continue
        res = json.loads((d / "results.json").read_text())
        env = json.loads((d / "env.json").read_text()) if (d / "env.json").is_file() else res.get("env", {})
        runs.append({"name": n, "res": res, "env": env})
    return runs


def path_of(env):
    if env.get("connector") == "nixl":
        return "NIXL (vLLM), every GPU visible" if env.get("nixl_all_gpus") else "NIXL (vLLM), own GPU only"
    if env.get("connector") == "lmcache":
        return "LMCache (vLLM)"
    return {"ring": "host memory", "p2p": "GPU to GPU", "tcp": "TCP"}.get(env.get("transport"), env.get("transport"))


def label(env):
    lay = env.get("layout") or f"{env.get('prefills')}p{env.get('decodes')}d"
    return f"{lay}, {path_of(env)}" + (", one link shared" if env.get("shared_link") else "") + \
        (", eager" if env.get("eager") else "")


def per_gpu(env, link):
    """The link's speed per prefill GPU: the whole link, or its share when the prefill engines share it."""
    try:
        g = float(link)
    except (TypeError, ValueError):
        return None
    return g / max(1, int(env.get("prefills") or 1)) if env.get("shared_link") else g


def moved(st):
    """IBP holds each chunk at its raw size and moves its coded size: both, where they differ."""
    b, m = st.get("bits"), st.get("moved_bits")
    return f" held, {m:.3f} moved" if b is not None and m is not None and abs(b - m) > 0.01 else ""


def lossy(c):
    return str(c).startswith("aquakv-")


def cname(c):
    """A codec's name in the tables: AQUA-KV marked lossy."""
    return f"{c} (lossy)" if lossy(c.split(" #")[0]) else c


LOSSY_NOTE = ("aquakv-<b> is AQUA-KV (arXiv 2501.19392, the authors' algorithm on the GPU), lossy: the KV it "
              "delivers differs from the original (its error is in the correctness table). It is here for speed and "
              "size only, beside the bit-exact codecs.")


def speed_key(x):
    return (1, 0.0) if x in ("native", "None") else (0, float(x))


def capacity(by, codec, rates, share=0.9):
    """(the highest offered request rate whose goodput reached `share` of it, the most requests per second at any
    rate): how many requests a codec carries within the TTFT target. None where no rate qualified."""
    cap = top = None
    for rt in rates:
        b = ((by.get((codec, rt)) or {}).get("bench") or {})
        thr = b.get("request_throughput")
        if thr is not None:
            top = thr if top is None else max(top, thr)
        try:
            r = float(rt)
        except (TypeError, ValueError):
            continue
        if r == float("inf"):
            continue
        g = b.get("request_goodput")
        if g is not None and g >= share * r:
            cap = r if cap is None else max(cap, r)
    return cap, top


def numeric_rates(rates):
    out = []
    for r in rates:
        try:
            if float(r) != float("inf"):
                out.append(r)
        except (TypeError, ValueError):
            pass
    return out


CAPACITY_NOTE = ("Capacity: the highest request rate (per second) at which at least 90 % of the requests met the "
                 "run's TTFT target (goodput at least 0.9 x the rate); *most*: the most requests per second at any "
                 "rate, the target aside.")


def link_tables(key, vs):
    """One layout and path over many link speeds: TTFT p50 per codec and speed (the best in bold), what the link
    carried, the speeds where each codec beats raw, and where a request's time goes."""
    lay, path, eager = key
    L = []
    links = sorted({str(v.get("link_gbps")) for v in vs}, key=speed_key)
    codecs = []
    for v in vs:
        if v["codec"] not in codecs:
            codecs.append(v["codec"])
    for rate in sorted({str(v["rate"]) for v in vs}, key=lambda x: (x == "inf", float(x) if x != "inf" else 0)):
        by = {(v["codec"], str(v.get("link_gbps"))): v for v in vs if str(v["rate"]) == rate}
        ttft = lambda c, l: ((by.get((c, l)) or {}).get("bench") or {}).get("median_ttft_ms")
        head = [("native" if l in ("native", "None") else f"{float(l):g}") for l in links]
        L += [f"## {lay}, {path}{', eager engines' if eager else ''}: link speed sweep, request rate {rate}", "",
              "TTFT p50 in ms per set link speed (Gb/s); the best at each speed in bold. *Carried*: what the link "
              "carried while busy for raw, all transfers together: below the set speed, the node's own path is the "
              "limit, and the faster speeds measure the node.", "",
              "| Codec | Bits | " + " | ".join(head) + " |", "| --- | --- |" + " --- |" * len(links)]
        best = {l: min([t for t in (ttft(c, l) for c in codecs) if t is not None] or [None]) for l in links}
        for c in codecs:
            bits = next((by[(c, l)]["transfers"].get("bits") for l in links
                         if (c, l) in by and (by[(c, l)].get("transfers") or {}).get("bits") is not None), None)
            cells = []
            for l in links:
                t = ttft(c, l)
                st = (by.get((c, l)) or {}).get("transfers") or {}
                bad = (st.get("install_errors") or 0) + (st.get("send_errors") or 0) + (st.get("timeouts") or 0)
                cell = "-" if t is None else (f"**{t:.0f}**" if t == best[l] else f"{t:.0f}")
                cells.append(cell + (" (failed)" if bad else ""))
            L.append(f"| {cname(c)} | {fmt(bits, 2) if c != 'raw' else '16.00'} | " + " | ".join(cells) + " |")
        carried = [fmt(((by.get(("raw", l)) or {}).get("transfers") or {}).get("wire_gbps"), 0) for l in links]
        L += ["| *Carried (raw)* | | " + " | ".join(carried) + " |", ""]
        wins = []
        for c in codecs:
            if c == "raw":
                continue
            w = [h for l, h in zip(links, head)
                 if ttft(c, l) is not None and ttft("raw", l) is not None and ttft(c, l) < ttft("raw", l)]
            wins.append(f"{cname(c)}: " + (", ".join(w) if w else "none"))
        L += ["Faster than raw at: " + "; ".join(wins) + ".", ""]
        # where the time goes, at the slowest and the fastest speed
        L += ["Where a request's transfer time goes (ms, mean per request): *queue* waiting for the sender, *gather* "
              "copying the request's KV out of vLLM's cache (includes waiting for the GPU's queued prefill work: "
              "the same for every codec), *encode* on the prefill GPU, *link* its time on the link (shared with the "
              "other transfers), *install* decoding and writing it into the decode engine's cache.", "",
              "| Codec | Link | Queue | Gather | Encode | Link time | Install | TTFT p50 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for l, h in ((links[0], head[0]), (links[-1], head[-1])):
            for c in codecs:
                v = by.get((c, l))
                if not v:
                    continue
                st = v.get("transfers") or {}
                L.append(f"| {c} | {h} | {fmt(st.get('queue_ms'), 0)} | {fmt(st.get('gather_ms'), 0)} | "
                         f"{fmt(st.get('encode_ms'), 0)} | {fmt(st.get('link_ms'), 0)} | {fmt(st.get('install_ms'), 0)} | "
                         f"{fmt(ttft(c, l), 0)} |")
            if len(links) == 1:
                break
        L.append("")
    rates_all = sorted({str(v["rate"]) for v in vs}, key=lambda x: (x == "inf", float(x) if x != "inf" else 0))
    if len(numeric_rates(rates_all)) >= 3:
        head = [("native" if l in ("native", "None") else f"{float(l):g}") for l in links]
        L += [f"## {lay}, {path}{', eager engines' if eager else ''}: capacity per link speed", "", CAPACITY_NOTE, "",
              "| Codec | " + " | ".join(head) + " |", "|" + " --- |" * (len(links) + 1)]
        for c in codecs:
            cells = []
            for l in links:
                by = {(v["codec"], str(v["rate"])): v for v in vs if str(v.get("link_gbps")) == l}
                cap, _ = capacity(by, c, rates_all)
                cells.append(fmt(cap, 1) if cap else "-")
            L.append(f"| {cname(c)} | " + " | ".join(cells) + " |")
        L.append("")
    return L


FAMILIES = {"sz1": "SplitZip, 1 at a time", "sz16": "SplitZip, 16 at once", "sz1024": "SplitZip, 1,024 tokens",
            "sz32k": "SplitZip, 32K tokens", "ibp": "IBP", "exactkv": "exactkv", "long": "Long prompt, 1 at a time",
            "share": "Shared link, many at once"}
# "same as raw": a TTFT within this share of raw's either way (the run-to-run noise of a point is about 1-2%)
TIE = 0.03


def regime_tables(runs):
    """Both regimes, per codec: at each link speed (per prefill GPU), on how many workloads its TTFT p50 beats raw's
    by more than TIE, matches it, or loses, and its speedup over raw from the worst workload to the best."""
    groups = {}
    for r in runs:
        for v in (r["res"].get("shapes") or {}).values():
            g = per_gpu(r["env"], v.get("link_gbps"))
            key = (label(r["env"]), str(v.get("link_gbps")), g)
            groups.setdefault(key, {}).setdefault(v["codec"], {})[(v["tokens"], v["batch"], v["output"])] = v
    if not groups:
        return []
    codecs = []
    for g in groups.values():
        for c in g:
            if c != "raw" and c not in codecs:
                codecs.append(c)
    L = ["## Where compression helps and where it does not", "",
         f"Per layout and link speed (and its share per prefill GPU when the engines share one link), per codec: the "
         f"workloads (stage shapes) where its TTFT p50 beats raw's by more than {100 * TIE:.0f}% (helps), is within "
         f"{100 * TIE:.0f}% (same) or is slower (hurts), as helps / same / hurts, then its speedup over raw (raw's "
         "TTFT / the codec's) from its worst workload to its best.", "",
         "| Run | Link | Per prefill GPU | " + " | ".join(cname(c) for c in codecs) + " |",
         "| --- | --- | --- |" + " --- |" * len(codecs)]
    for (lab, link, g), by in sorted(groups.items(), key=lambda kv: (kv[0][0], speed_key(kv[0][1]))):
        raw = by.get("raw") or {}
        cells = []
        for c in codecs:
            sp = []
            for k, v in (by.get(c) or {}).items():
                a = ((raw.get(k) or {}).get("bench") or {}).get("median_ttft_ms")
                b = (v.get("bench") or {}).get("median_ttft_ms")
                if a and b:
                    sp.append(a / b)
            if not sp:
                cells.append("-")
                continue
            h = sum(1 for x in sp if x > 1 + TIE)
            t = sum(1 for x in sp if 1 - TIE <= x <= 1 + TIE)
            cells.append(f"{h} / {t} / {len(sp) - h - t}, {min(sp):.2f}-{max(sp):.2f}x")
        link_s = "unpaced" if link in ("native", "None") else f"{float(link):g} Gb/s"
        L.append(f"| {lab} | {link_s} | {'-' if g is None else f'{g:g} Gb/s'} | " + " | ".join(cells) + " |")
    return L + [""]


def workload_tables(runs):
    """The published workloads (scripts/pd_bench.py stage shapes): per layout, path and link, one row per workload
    and one column per codec; TTFT p50 with its speedup over raw, then requests per second with its gain."""
    groups = {}
    for r in runs:
        for v in (r["res"].get("shapes") or {}).values():
            key = (r["env"].get("layout") or "", path_of(r["env"]), bool(r["env"].get("eager")), str(v.get("link_gbps")))
            groups.setdefault(key, []).append((r["name"], v))
    L = []
    order = list(FAMILIES)
    for (lay, path, eager, link), vs in sorted(groups.items(), key=lambda kv: (kv[0][:3], speed_key(kv[0][3]))):
        labels, by, shapes = {}, {}, {}
        for run, v in vs:
            if (run, v["codec"]) not in labels:     # the same codec from a second run (raw beside nvCOMP): "raw #2"
                n = sum(1 for _, c in labels if c == v["codec"])
                labels[(run, v["codec"])] = v["codec"] if n == 0 else f"{v['codec']} #{n + 1}"
            lab = labels[(run, v["codec"])]
            k = (v["tokens"], v["batch"], v["output"])
            fam = shapes.setdefault(k, [])
            fam += [f for f in v.get("families") or [] if f not in fam]
            by[(lab, k)] = v
        cols = list(dict.fromkeys(labels.values()))
        rows = sorted(shapes, key=lambda k: (min([order.index(f) for f in shapes[k] if f in order] or [99]), k[1], k[0]))
        link_s = "unpaced (the machine's own path)" if link in ("native", "None") else f"{float(link):g} Gb/s"
        ref = "raw" if "raw" in cols else None

        def table(metric, unit, better):
            out = ["| Workload | Prompt tokens x at once, out | " + " | ".join(cname(c) for c in cols) + " |",
                   "| --- | --- |" + " --- |" * len(cols)]
            for k in rows:
                vals = {c: ((by.get((c, k)) or {}).get("bench") or {}).get(metric) for c in cols}
                good = [x for x in vals.values() if x]
                best = (min(good) if better == "low" else max(good)) if good else None
                cells = []
                for c in cols:
                    x, v = vals[c], by.get((c, k)) or {}
                    st = v.get("transfers") or {}
                    bad = (st.get("install_errors") or 0) + (st.get("send_errors") or 0) + (st.get("timeouts") or 0)
                    if not x:
                        cells.append("-")
                        continue
                    cell = f"{x:.0f}" if unit == "ms" else f"{x:.2f}"
                    cell = f"**{cell}**" if x == best else cell
                    rx = vals.get(ref) if ref else None
                    if rx and c != ref:
                        cell += f" ({(rx / x) if better == 'low' else (x / rx):.2f}x)"
                    host = (st.get("spilled_to_host") or 0) + (st.get("recv_on_host") or 0)
                    cells.append(cell + (" (failed)" if bad else "") + (" (host)" if host else ""))
                fam = " + ".join(FAMILIES.get(f, f) for f in sorted(shapes[k], key=lambda f: order.index(f)
                                                                         if f in order else 99))
                out.append(f"| {fam} | {k[0]} x {k[1]}, {k[2]} | " + " | ".join(cells) + " |")
            return out
        L += [f"## Published workloads: {lay}, {path}{', eager engines' if eager else ''}, link {link_s}", "",
              "Each baseline's own workload, as its authors reported it (docs/pd_baselines.md, \"Workloads\"), for "
              "every codec: B requests at once, B x R in all, T-token prompts, O tokens out. TTFT p50 in ms, the best "
              "in bold, and in brackets the speedup over raw (raw's time / the codec's). (failed): some of the point's "
              "transfers failed and vLLM computed those prompts; (host): some went through host memory (a transfer "
              "over the GPU budget while others were held), which costs copies the others did not pay.", ""]
        L += table("median_ttft_ms", "ms", "low") + [""]
        L += ["Requests per second, and in brackets the gain over raw:", ""]
        L += table("request_throughput", "req/s", "high") + [""]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--root", default="benchmark_results")
    ap.add_argument("--title", default="PD matrix")
    ap.add_argument("--failed", default="")
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    runs = load(a.root, [n for n in a.runs.split(",") if n])
    L = [f"# {a.title}", ""]
    env0 = runs[0]["env"] if runs else {}
    L.append(f"Model {env0.get('model')}; {env0.get('gpus')}; vLLM {env0.get('vllm')}, torch {env0.get('torch')}. "
             f"{len(runs)} runs: " + ", ".join(f"`{r['name']}`" for r in runs) + ".")
    failed = [f for f in a.failed.split(";") if f.strip()]
    if failed:
        L.append("")
        L.append("Runs that did not finish: " + "; ".join(f"`{f.strip()}`" for f in failed) + ".")
    for r in runs:
        s = r["res"].get("stopped")
        if s:
            eng = s.get("engines") or {}
            L += ["", f"`{r['name']}` stopped early in its {s.get('stage')} stage: {s.get('why')}"
                  + (f" (not answering: {', '.join(sorted(eng))})" if eng else "")
                  + "; its later points were not run."]
    L.append("")

    # correctness
    rows = []
    for r in runs:
        for c, v in (r["res"].get("check") or {}).items():
            st = v.get("transfers") or {}
            ex = ("n/a" if c == "nixl" else
                  f"lossy (KV error {100 * st['rel_err']:.2f}%)" if lossy(c) and st.get("rel_err") is not None else
                  "lossy" if lossy(c) else f"{st.get('exact')}/{v.get('prompts')}")
            rows.append((label(r["env"]), cname(c), ex,
                         f"{v.get('same_text_as_local')}/{v.get('prompts')}", "yes" if v.get("ok") else "NO",
                         fmt(st.get("bits"), 3)))
        for v in (r["res"].get("tier") or {}).values():
            if not isinstance(v, dict) or "tier" not in v:
                continue
            st = v["tier"]
            rows.append((label(r["env"]) + " tier", f"{v['codec']} (c{v['clients']})",
                         f"{st.get('exact_chunks')}/{st.get('chunks_loaded')} chunks", "-",
                         "yes" if not st.get("load_errors") else "NO", fmt(st.get("bits"), 3) + moved(st)))
    if rows:
        L += ["## Correctness", "", "| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |",
              "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {' | '.join(map(str, x))} |" for x in rows]
        L.append("")

    # PD sweeps: many link speeds in one table per layout and path; a table per link when there are few
    groups, speeds = {}, {}
    for r in runs:
        for v in (r["res"].get("sweep") or {}).values():
            key = (r["env"].get("layout") or "", path_of(r["env"]), bool(r["env"].get("eager")), str(v.get("link_gbps")))
            groups.setdefault(key, []).append(v)
            speeds.setdefault(key[:3], set()).add(str(v.get("link_gbps")))
    many = {k for k, ls in speeds.items() if len(ls) > 4}
    for k in sorted(many):
        L += link_tables(k, [v for g, vs in groups.items() if g[:3] == k for v in vs])
    for (lay, path, eager, link), vs in sorted(groups.items()):
        if (lay, path, eager) in many:
            continue
        rates = []
        for v in vs:
            if v["rate"] not in rates:
                rates.append(v["rate"])
        codecs = []
        for v in vs:
            if v["codec"] not in codecs:
                codecs.append(v["codec"])
        by = {(v["codec"], v["rate"]): v for v in vs}
        link_s = "unpaced (the machine's own path)" if link in ("native", "None") else f"{link} Gb/s"
        L += [f"## {lay}, {path}{', eager engines' if eager else ''}: link {link_s}", "",
              "Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.", "",
              "| Codec | Bits | " + " | ".join(f"rate {r}" for r in rates) + " | TPOT p50 (ms) |",
              "| --- | --- |" + " --- |" * len(rates) + " --- |"]
        for c in codecs:
            cells, bits, tpot = [], None, None
            for rt in rates:
                v = by.get((c, rt))
                if v is None:
                    cells.append("-")
                    continue
                b, st = v.get("bench") or {}, v.get("transfers") or {}
                bits = st.get("bits") if st.get("bits") is not None else bits
                tpot = b.get("median_tpot_ms") if b.get("median_tpot_ms") is not None else tpot
                bad = (st.get("install_errors") or 0) + (st.get("send_errors") or 0) + (st.get("timeouts") or 0)
                cells.append(f"{fmt(b.get('request_throughput'))} ({fmt(b.get('request_goodput'))}); "
                             f"{fmt(b.get('median_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)}"
                             + (f"; **{bad} of {st.get('sends') or '?'} transfers failed, computed locally**" if bad else ""))
            L.append(f"| {cname(c)} | {fmt(bits, 3) if c not in ('raw', 'nixl') else '16.000'} | " + " | ".join(cells) +
                     f" | {fmt(tpot, 1)} |")
        L.append("")
        best = []
        for rt in rates:
            cand = [(by[(c, rt)].get("bench") or {}, c) for c in codecs if (c, rt) in by]
            cand = [(b.get("request_goodput") or 0, -(b.get("median_ttft_ms") or 1e18), c) for b, c in cand]
            if cand:
                g, t, c = max(cand)
                best.append(f"rate {rt}: {cname(c)} (goodput {g:.2f}/s, TTFT p50 {-t:.0f} ms)")
        if best:
            L += ["Best by goodput, then TTFT p50: " + "; ".join(best) + ".", ""]
        if len(numeric_rates(rates)) >= 3:
            L += [CAPACITY_NOTE, "", "| Codec | Capacity (req/s) | Most (req/s) | vs raw |", "| --- | --- | --- | --- |"]
            raw_cap = capacity(by, "raw", rates)[0] if "raw" in codecs else None
            for c in codecs:
                cap, top = capacity(by, c, rates)
                rel = (f"{cap / raw_cap:.2f}x" if cap and raw_cap and c != "raw" else "")
                L.append(f"| {cname(c)} | {fmt(cap, 1) if cap else 'below ' + str(numeric_rates(rates)[0])} | "
                         f"{fmt(top)} | {rel} |")
            L.append("")

    L += regime_tables(runs)
    L += workload_tables(runs)

    # CPU tier
    tiers = {}
    for r in runs:
        for v in (r["res"].get("tier") or {}).values():
            if isinstance(v, dict) and "bench" in v:
                tiers.setdefault((r["env"].get("layout"), bool(r["env"].get("eager")), v["clients"]), []).append(
                    (v, path_of(r["env"])))
    for (lay, eager, clients), vs in sorted(tiers.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        L += [f"## CPU tier, {lay}{', eager engines' if eager else ''}: {clients} clients", "",
              "Prefix hits: vLLM's own counters, GPU cache / external (the tier: ours or LMCache's), % of the tokens "
              "asked. Bits: of the whole chunks loaded; Needed: the share of those chunks the requests needed. Loads, "
              "bits and chunks come from our connector's records, so LMCache has none: its external hits show its use.",
              "", "| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Prefix hits GPU / external (%) | Loads | "
              "Tokens loaded | Load ms (copy / decode) | Bits | Needed (%) | Chunks stored |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v, path in sorted(vs, key=lambda x: (x[0]["bench"].get("median_ttft_ms") or 1e18)):
            b, st, hc = v["bench"], v["tier"], v.get("prefix_cache") or {}
            lm = path in ("LMCache (vLLM)",)
            name = v["codec"] if not lm else "LMCache CPU offload (raw)"
            dash = lambda x: "-" if lm else x
            L.append(f"| {name} | {fmt(b.get('request_throughput'))} | {fmt(b.get('median_ttft_ms'), 0)} / "
                     f"{fmt(b.get('p90_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)} | {fmt(b.get('median_tpot_ms'), 1)} | "
                     f"{fmt(hc.get('gpu_hit_pct'), 1)} / {fmt(hc.get('external_hit_pct'), 1)} | "
                     f"{dash(st.get('loads'))} | {dash(st.get('loaded_tokens'))} | {dash(fmt(st.get('load_ms'), 1))} "
                     f"({dash(fmt(st.get('h2d_ms'), 1))} / {dash(fmt(st.get('decode_ms'), 1))}) | "
                     f"{dash(fmt(st.get('bits'), 3) + moved(st))} | {dash(fmt(st.get('needed_pct'), 0))} | "
                     f"{dash(st.get('stored_chunks'))} |")
        L.append("")

    # T2
    t2 = []
    for r in runs:
        for name, v in (r["res"].get("sweep") or {}).items():
            st = v.get("transfers") or {}
            if str(v.get("codec", "")).startswith("t2-") and st.get("layer0_recomputed") is not None:
                t2.append((label(r["env"]), v["codec"], str(v.get("link_gbps")), v["rate"], st["layer0_recomputed"],
                           st["layer0_coded"], st.get("sets"), "; ".join(st.get("layer0_checks") or [])))
    if t2:
        L += ["## T2: layer 0 recomputed by the receiver, and the constant sets picked", "",
              "| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {x[0]} | {x[1]} | {x[2]} | {x[3]} | {x[4]} | {x[5]} | {x[6]} | {x[7][:160]} |" for x in t2]
        L.append("")
    jits = [(label(r["env"]), j) for r in runs for j in (r["res"].get("jit") or []) if j.get("point")]
    if jits:
        L += ["## Kernels compiled inside measured points", "",
              "vLLM compiled these while a point was measured, so that point's latencies include the compile:", ""]
        L += [f"- {lab}: {j['point']} ({j['log']}): {j['text'][:110]}" for lab, j in jits[:30]] + [""]
    skipped = {}
    for r in runs:
        for c, w in (r["res"].get("skipped") or {}).items():
            skipped.setdefault(c, w)
    if skipped:
        L += ["## Codecs that did not load", ""] + [f"- {c}: {w[:200]}" for c, w in sorted(skipped.items())] + [""]
    if any(lossy(c) for r in runs for key in ("check", "sweep", "shapes")
           for c in ([v.get("codec") for v in (r["res"].get(key) or {}).values()] if key != "check"
                     else list(r["res"].get("check") or {}))):
        L += ["## Note on the lossy reference", "", LOSSY_NOTE, ""]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
