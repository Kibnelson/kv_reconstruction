#!/usr/bin/env python3
"""The reuse opportunity in a traced PD run (kv_compare/pd/trace.py, scripts/pd_mtrag.py): which KV pages were
already somewhere, as a prefix, as the same bytes or as the same text, and what P and D did again that the other had.

  python3 -I scripts/pd_trace_report.py RUN_DIR [--markdown trace_report.md] [--json trace_report.json]

RUN_DIR: a run's folder as published (runs/mtrag_*.jsonl.gz, proxy.jsonl.gz, connector/*.jsonl.gz), or during the
run its output folder with --log-dir for the connector's logs. A page is one 16-token block at one layer (keys and
values together, 64 KiB for Llama-3.2-3B).

Per run variant (prompt layout x the proxy's routing):
  Moved again  tokens P sent that the chosen D already held (its prefix cache), and their bytes
  Recomputed   tokens P prefilled that it did not have cached; of them, tokens of an earlier answer (made on D)
  Same bytes?  of the pages P sent again that D held, how many D's own bytes equal, per kind of token (answers,
               documents, the rest) and per layer
  Affinity     how often a conversation's turn went to the D that served its previous turn
Per traced variant (every page's content hash), in the order the requests came:
  1 prefix     a page whose token prefix an earlier request had (what vLLM's and LMCache's prefix caches share)
  2 same bytes not 1, but an earlier page at the same layer has the same bytes (exact reuse a prefix cache misses)
  3 same text  not 1 or 2, but its 16 tokens appeared in an earlier prompt (the same text after another prefix: what
               a differential store could code against; how small such deltas are is measured offline)
  4 new        none of these
and the bytes a store of every page would hold keyed by prefix (LMCache), by content, and by content without
layer 0 (recomputable from the tokens).
Per traced variant, how close related KV is (P's relation probe, kv_compare/pd/trace.py): the tokens whose 16-token
window appeared before (not as a shared prefix), by kind of piece, moved or at the same place after another prefix,
within the conversation or across; and what coding them against their first occurrence costs, in bits per value,
per layer, keys and values apart: xor, diff (ordered integers), rot (keys turned to the new position first) and,
for the same tokens, alone (sign and exponent entropy plus 7 mantissa bits, the exponent coders' floor). For the
pages P sent again that D held and that differed (P's recomputation), the same measures against D's own.
"""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import json
import pathlib
import sys


def jl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                pass


def files(d, pattern):
    d = pathlib.Path(d)
    return sorted(set(d.glob(pattern)) | set(d.glob(pattern + ".gz")))


def u64(s, shape):
    import numpy as np
    return np.frombuffer(base64.b64decode(s), dtype="<u8").reshape(shape)


def bits(s, shape):
    import numpy as np
    n = 1
    for x in shape:
        n *= x
    return np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))[:n].astype(bool).reshape(shape)


def kind_of(pieces, t):
    """The kind of piece token t lies in: answer, doc, or other."""
    for kind, ref, a, b in pieces:
        if a <= t < b:
            return "answer" if kind == "answer" else ("doc" if kind == "doc" else "other")
    return "other"


def load(run, log_dir=None):
    run = pathlib.Path(run)
    logs = pathlib.Path(log_dir) if log_dir else run / "connector"
    drv = [r for f in files(run / "runs", "mtrag_*.jsonl") for r in jl(f)]
    proxy = [r for f in files(run, "proxy.jsonl") for r in jl(f)]
    conn = [r for f in files(logs, "tkv_*.jsonl") for r in jl(f)]
    trace = []
    for f in files(logs, "trace_*.jsonl"):           # trace_<role>_<pid>_<n>: each engine's records apart
        src = "_".join(f.name.split(".")[0].split("_")[1:3])
        trace += [dict(r, _src=src) for r in jl(f)]
    env = json.loads((run / "env.json").read_text()) if (run / "env.json").is_file() else {}
    return drv, proxy, conn, trace, env


def analyse(drv, proxy, conn, trace, env):
    by_tag = {r["tag"]: r for r in drv if "tag" in r}
    key_of = {p["tag"]: p["key"] for p in proxy if p.get("tag")}
    pin = {p["key"]: p for p in proxy}
    pdone = {r["key"]: r for r in conn if r.get("event") == "prefill_done"}
    match = {r["key"]: r for r in conn if r.get("event") == "match"}
    pages = collections.defaultdict(list)
    held = collections.defaultdict(list)
    for r in trace:
        if r.get("event") == "pages":
            pages[r["key"]].append(r)
        elif r.get("event") == "held":
            held[r["key"]].append(r)
    L = int(env.get("layers") or next((r["layers"] for r in trace if "layers" in r), 28))
    page_bytes = None
    variants = collections.defaultdict(list)
    for tag, d in by_tag.items():
        if d.get("error") or tag not in key_of:
            continue
        run = d.get("run") or f"{d['layout']}_{d['policy']}"
        variants[(run, d["layout"], d["policy"])].append((pin.get(key_of[tag], {}).get("t_in", 0), d, key_of[tag]))
    names = collections.Counter((lay, pol) for _r, lay, pol in variants)
    rels = collections.defaultdict(dict)             # P's relation records: engine -> request number -> record
    stats = {}                                       # P's running totals: (engine, group) -> the last record
    unmeasured = collections.Counter()               # requests the probe was behind on, or failed
    for r in trace:
        if r.get("event") in ("relations_skipped", "relations_error"):
            unmeasured[r.get("key")] += 1
        if r.get("event") == "relations":
            rels[r["_src"]][(str(r.get("group")), r.get("r"))] = r
        elif r.get("event") == "delta_stats":
            stats[(r["_src"], r["group"])] = r
    out = {}
    for (run, layout, policy), reqs in sorted(variants.items()):
        reqs.sort(key=lambda x: x[0])
        v = {"requests": len(reqs)}
        # moved again, recomputed, same bytes, affinity
        sent = held_tok = recomputed = ans_total = ans_recomputed = 0
        eqs = collections.defaultdict(lambda: [0, 0])
        eq_layer = None
        last_dest, same_dest, follow = {}, 0, 0
        for t_in, d, key in reqs:
            pd, m = pdone.get(key, {}), match.get(key, {})
            T = int(pd.get("tokens") or d["prompt_tokens"])
            hit = int(m.get("local_hit") or 0)
            sent += T - int(pd.get("t0") or 0)
            held_tok += min(hit, T)
            cached = pd.get("cached")
            cached = int(cached) if cached is not None else 0
            recomputed += max(0, T - cached)
            for kind, ref, a, b in d["pieces"]:
                if kind == "answer":
                    ans_total += b - a
                    ans_recomputed += max(0, b - max(a, cached))
            for h in held.get(key, []):
                m_ = bits(h["equal"], (h["layers"], -(-h["tokens"] // h["bs"])))
                if eq_layer is None:
                    eq_layer = [[0, 0] for _ in range(m_.shape[0])]
                for j in range(m_.shape[1]):
                    k = kind_of(d["pieces"], h["t0"] + j * h["bs"])
                    eqs[k][0] += int(m_[:, j].all())
                    eqs[k][1] += 1
                for l in range(m_.shape[0]):
                    eq_layer[l][0] += int(m_[l].sum())
                    eq_layer[l][1] += m_.shape[1]
            dest = pin.get(key, {}).get("dest")
            c = d["conv"]
            if d["turn"] > 0 and c in last_dest:
                follow += 1
                same_dest += int(last_dest[c] == dest)
            last_dest[c] = dest
        kvt = env.get("kv_bytes_per_token") or 114688
        v["moved_again"] = {"tokens_sent": sent, "tokens_d_held": held_tok, "share": held_tok / sent if sent else None,
                            "gb_d_held": held_tok * kvt / 1e9}
        v["recomputed"] = {"tokens": recomputed, "answer_tokens": ans_total, "answer_tokens_recomputed": ans_recomputed}
        v["same_bytes"] = {k: {"equal": e, "pages": n, "share": e / n if n else None} for k, (e, n) in eqs.items()}
        v["same_bytes_by_layer"] = [e / n if n else None for e, n in (eq_layer or [])]
        v["affinity"] = {"later_turns": follow, "same_d": same_dest, "share": same_dest / follow if follow else None}
        # the content view (traced requests only)
        traced = [(t, d, k) for t, d, k in reqs if pages.get(k) and d.get("ids")]
        if traced:
            seen_prefix, seen_content = set(), [set() for _ in range(L)]
            shingles = set()
            counts = collections.Counter()
            per_layer_same = [0] * L
            text_kind = collections.Counter()
            store_prefix, store_content, store_nol0 = set(), set(), set()
            total = 0
            for t_in, d, key in traced:
                ps = sorted(pages[key], key=lambda r: r["t0"])
                hs = []
                for r in ps:
                    nb = -(-r["tokens"] // r["bs"])
                    hs.append(u64(r["h"], (r["layers"], nb)))
                import numpy as np
                h = np.concatenate(hs, 1) if hs else None
                bs = ps[0]["bs"]
                blocks = pdone.get(key, {}).get("blocks") or []
                ids = d["ids"]
                Lh, nb = h.shape
                for j in range(nb):
                    pid = blocks[j] if j < len(blocks) else None
                    toks = tuple(ids[j * bs:(j + 1) * bs])
                    text_seen = len(toks) == bs and hash(toks) in shingles
                    for l in range(Lh):
                        total += 1
                        c = int(h[l, j])
                        if pid is not None and pid in seen_prefix:
                            counts["1 prefix"] += 1
                        elif c in seen_content[l]:
                            counts["2 same bytes"] += 1
                            per_layer_same[l] += 1
                        elif text_seen:
                            counts["3 same text"] += 1
                            text_kind[kind_of(d["pieces"], j * bs)] += 1
                        else:
                            counts["4 new"] += 1
                        seen_content[l].add(c)
                        store_content.add((l, c))
                        if l > 0:
                            store_nol0.add((l, c))
                        store_prefix.add((l, pid if pid is not None else ("tail", key, j)))
                for j in range(nb):
                    if j < len(blocks):
                        seen_prefix.add(blocks[j])
                for i in range(0, max(0, len(ids) - bs + 1)):
                    shingles.add(hash(tuple(ids[i:i + bs])))
                page_bytes = page_bytes or (kvt * bs // Lh)
            v["pages"] = {"pages": total, "kinds": {k: {"pages": n, "share": n / total} for k, n in sorted(counts.items())},
                          "same_bytes_by_layer": per_layer_same, "same_text_by_piece": dict(text_kind)}
            pb = page_bytes or 65536
            v["store_gb"] = {"by_prefix": len(store_prefix) * pb / 1e9, "by_content": len(store_content) * pb / 1e9,
                             "by_content_without_layer0": len(store_nol0) * pb / 1e9}
        rel = relations(run, reqs, rels, stats, held, kvt, unmeasured)
        if rel:
            v["relations"] = rel
        out[f"{layout} / {policy}" if names[(layout, policy)] == 1 else run] = v
    return out


# ------------------------------------------------------------------------------------------------- relations
def entropy(counts):
    import math
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


def cost(hist, signed):
    """Bits per value of coding values by their bit length (entropy-coded) and the bits below the leading one (and
    a sign, for differences); None without counts."""
    n = sum(hist)
    if not n:
        return None
    return entropy(hist) + sum(c * (max(b - 1, 0) + (1 if signed and b else 0)) for b, c in enumerate(hist)) / n


def add_delta(acc, d):
    """acc += d (delta_stats' or held's histograms)."""
    if acc is None:
        return json.loads(json.dumps({k: d[k] for k in ("hist", "sign_exp", "equal", "values")}))
    for k, h in d["hist"].items():
        for kv in range(2):
            for l in range(len(h[kv])):
                acc["hist"][k][kv][l] = [x + y for x, y in zip(acc["hist"][k][kv][l], h[kv][l])]
    for kv in range(2):
        for l in range(len(d["values"][kv])):
            acc["sign_exp"][kv][l] = [x + y for x, y in zip(acc["sign_exp"][kv][l], d["sign_exp"][kv][l])]
            acc["equal"][kv][l] += d["equal"][kv][l]
            acc["values"][kv][l] += d["values"][kv][l]
    return acc


def delta_summary(acc):
    """Bits per value by measure, keys (K) and values (V), per layer and weighted over them; best: the cheapest
    measure of each layer; saving: 1 - best / alone."""
    if acc is None:
        return None
    out = {"values": 0, "equal": 0, "by_layer": {}, "bits": {}}
    tot_best = tot_alone = 0.0
    for kv, name in ((0, "K"), (1, "V")):
        rows, w = [], 0
        sums = collections.Counter()
        for l in range(len(acc["values"][kv])):
            n = acc["values"][kv][l]
            out["values"] += n
            out["equal"] += acc["equal"][kv][l]
            row = {m: cost(acc["hist"][m][kv][l], m != "xor") for m in ("xor", "diff", "rot")}
            se = acc["sign_exp"][kv][l]
            row["alone"] = entropy(se) + 7 if sum(se) else None
            have = [x for m, x in row.items() if m != "alone" and x is not None]
            row["best"] = min(have) if have else None
            rows.append(row)
            if n and row["best"] is not None:
                w += n
                for m, x in row.items():
                    if x is not None:
                        sums[m] += n * x
        out["by_layer"][name] = rows
        out["bits"][name] = {m: (sums[m] / w if w and m in sums else None) for m in ("xor", "diff", "rot", "alone", "best")}
        tot_best += sums["best"]
        tot_alone += sums["alone"]
    out["equal_share"] = out["equal"] / out["values"] if out["values"] else None
    out["saving"] = 1 - tot_best / tot_alone if tot_alone else None
    out["best_bits"] = tot_best / out["values"] if out["values"] else None
    out["alone_bits"] = tot_alone / out["values"] if out["values"] else None
    return out


def relations(run, reqs, rels, stats, held, kvt, unmeasured=None):
    """The relation section of one variant (None if it was not traced with the probe)."""
    keys = {k for _t, _d, k in reqs}
    conv_of = {k: d["conv"] for _t, d, k in reqs}
    tokens = related = measured = novel = moved = same_place = across = within = 0
    kinds = collections.Counter()
    shifts = []
    found = False
    sample = None
    for src, by_r in rels.items():
        mine = {r: x for (g, r), x in by_r.items() if g == run and x["key"] in keys}
        for r, x in mine.items():
            found = True
            sample = x.get("sample", sample)
            tokens += x["tokens"]
            novel += x.get("novel", 0)
            for kind, (n, m) in x.get("by_kind", {}).items():
                kinds[kind] += n
                related += n
                measured += m
            for s, kind, i0, j0, n in x.get("spans", []):
                shifts.append((i0 - j0, n))
                if i0 == j0:
                    same_place += n
                else:
                    moved += n
                base = by_r.get((run, s))
                if base is not None and base["key"] in conv_of:
                    if conv_of[base["key"]] == conv_of[x["key"]]:
                        within += n
                    else:
                        across += n
    if not found:
        return None
    accs, total = {}, None
    for (src, g), rec in stats.items():
        if not g.startswith(run + "/"):
            continue
        kind = g[len(run) + 1:]
        accs[kind] = add_delta(accs.get(kind), rec)
        total = add_delta(total, rec)
    by_kind = {k: delta_summary(a) for k, a in sorted(accs.items())}
    dacc, dpages, ddiff = None, 0, 0
    for k in keys:
        for h in held.get(k, []):
            dpages += h.get("pages", 0)
            if "delta" in h:
                ddiff += h["pages"] - h.get("equal_pages", 0)
                dacc = add_delta(dacc, h["delta"])
    allk = delta_summary(total)
    shifts.sort(key=lambda x: abs(x[0]))
    half, acc, med = related / 2, 0, None
    for sh, n in shifts:
        acc += n
        if med is None and acc >= half:
            med = abs(sh)
    out = {"tokens": tokens, "related": related, "related_share": related / tokens if tokens else None,
           "measured": measured, "sample": sample, "novel": novel, "by_kind": dict(kinds),
           "requests_unmeasured": sum(1 for k in keys if (unmeasured or {}).get(k)), "moved": moved, "same_place": same_place,
           "within_conversation": within, "across_conversations": across, "median_shift": med,
           "delta": allk, "delta_by_kind": by_kind,
           "held_recomputed": {"pages": dpages, "pages_differing": ddiff, "delta": delta_summary(dacc)}}
    if allk and allk["best_bits"] is not None and allk["alone_bits"]:
        out["gb"] = {"related_raw": related * kvt / 1e9, "related_alone": related * kvt * allk["alone_bits"] / 16 / 1e9,
                     "related_as_deltas": related * kvt * allk["best_bits"] / 16 / 1e9}
    return out


def markdown(res):
    f = lambda x, p=0: "-" if x is None else (f"{100 * x:.{p}f}%" if isinstance(x, float) and x <= 1.0 else f"{x:,}")
    L = ["# The reuse opportunity (traced PD run)", "",
         "Per prompt layout and routing. Moved again: tokens P sent that the chosen D already held. Recomputed: tokens "
         "P prefilled without its cache; answers: tokens of earlier answers (made on D). Same bytes: pages P sent "
         "again that D held and whose bytes D's equal. Affinity: later turns that went to the D of their previous "
         "turn.", "",
         "| Layout / routing | Requests | Moved again (share of sent, GB) | Recomputed tokens | Earlier answers recomputed | Same bytes: answers / docs / rest | Affinity |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in res.items():
        mv, rc, sb, af = v["moved_again"], v["recomputed"], v["same_bytes"], v["affinity"]
        sbs = " / ".join(f((sb.get(k) or {}).get("share")) for k in ("answer", "doc", "other"))
        L.append(f"| {name} | {v['requests']} | {f(mv['share'])} ({mv['gb_d_held']:.1f} GB) | {rc['tokens']:,} | "
                 f"{rc['answer_tokens_recomputed']:,} of {rc['answer_tokens']:,} | {sbs} | {f(af['share'])} |")
    L += ["", "## Every page, by what already existed (traced variants)", "",
          "1 prefix: an earlier request had its token prefix (prefix caches share it). 2 same bytes: not 1, but an "
          "earlier page at its layer has its bytes. 3 same text: not 1 or 2, but its 16 tokens were in an earlier "
          "prompt (a differential store's candidates). 4 new. Store: what holding every page once would take, keyed "
          "by prefix (LMCache), by content, and by content without layer 0 (recomputable from tokens).", "",
          "| Layout / routing | Pages | 1 prefix | 2 same bytes | 3 same text | 4 new | Store by prefix / content / content w/o layer 0 (GB) |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, v in res.items():
        if "pages" not in v:
            continue
        k = v["pages"]["kinds"]
        g = lambda n: f((k.get(n) or {}).get("share"), 1)
        st = v["store_gb"]
        L.append(f"| {name} | {v['pages']['pages']:,} | {g('1 prefix')} | {g('2 same bytes')} | {g('3 same text')} | "
                 f"{g('4 new')} | {st['by_prefix']:.1f} / {st['by_content']:.1f} / {st['by_content_without_layer0']:.1f} |")
    L += ["", "Same bytes (2) by layer, and same text (3) by kind of piece:", ""]
    for name, v in res.items():
        if "pages" in v:
            pl = v["pages"]["same_bytes_by_layer"]
            nz = ", ".join(f"layer {i}: {n:,}" for i, n in enumerate(pl) if n) or "none"
            L.append(f"- {name}: same bytes at {nz}; same text in {v['pages']['same_text_by_piece']}")
    rel = {n: v["relations"] for n, v in res.items() if v.get("relations")}
    if rel:
        b = lambda x: "-" if x is None else f"{x:.2f}"
        L += ["", "## How close related KV is (the differential opportunity)", "",
              "Related: prompt tokens whose 16-token window appeared earlier in the run, not as a shared prefix (so a "
              "prefix cache misses them), at the same place after another prefix or moved; within the conversation or "
              "across. Bits per value of coding each against its first occurrence: xor; diff, the difference of the "
              "words as ordered integers; rot, keys turned to the new position first; best, the cheapest per layer; "
              "alone, the same tokens without a base (sign and exponent entropy + 7 bits, the exponent coders' floor). "
              "Raw BF16 is 16. Measured: a sample of the related tokens, chosen by content (1 in the sample's "
              "number), each against its first occurrence when it was still held.", "",
              "| Layout / routing | Prompt tokens | Related (share) | Docs / answers / questions / other | Same place / moved | Within / across conversations | Median shift |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in rel.items():
            k = r["by_kind"]
            L.append(f"| {name} | {r['tokens']:,} | {r['related']:,} ({f(r['related_share'], 1)}) | "
                     f"{k.get('doc', 0):,} / {k.get('answer', 0):,} / {k.get('question', 0):,} / "
                     f"{k.get('other', 0) + k.get('all', 0):,} | {r['same_place']:,} / {r['moved']:,} | "
                     f"{r['within_conversation']:,} / {r['across_conversations']:,} | {f(r['median_shift'])} |")
        L += ["", "Measured related tokens: " + "; ".join(
            f"{n}: {r['measured']:,} (1 in {r['sample'] or 1}), {r['requests_unmeasured']} requests not measured"
            for n, r in rel.items()), "",
              "| Layout / routing | Kind | Measured values | Equal | K: xor / diff / rot / alone | V: xor / diff / alone | Best vs alone (saving) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in rel.items():
            rows = [("every kind", r["delta"])] + [(k, x) for k, x in r["delta_by_kind"].items() if k != "all"]
            for kind, dsum in rows:
                if not dsum or not dsum["values"]:
                    continue
                K, V = dsum["bits"]["K"], dsum["bits"]["V"]
                L.append(f"| {name} | {kind} | {dsum['values']:,} | {f(dsum['equal_share'], 1)} | "
                         f"{b(K['xor'])} / {b(K['diff'])} / {b(K['rot'])} / {b(K['alone'])} | "
                         f"{b(V['xor'])} / {b(V['diff'])} / {b(V['alone'])} | {b(dsum['best_bits'])} vs "
                         f"{b(dsum['alone_bits'])} ({f(dsum['saving'], 1)}) |")
        L += ["", "Best bits per value by layer (keys, values), against alone:", ""]
        for name, r in rel.items():
            dsum = r["delta"]
            if not dsum or not dsum["values"]:
                continue
            for nm in ("K", "V"):
                rows = dsum["by_layer"][nm]
                L.append(f"- {name}, {nm}: " + ", ".join(f"{l}: {b(x['best'])}/{b(x['alone'])}"
                                                           for l, x in enumerate(rows) if x["best"] is not None))
        L += ["", "Sizes of the related tokens' KV: raw, coded alone, coded as deltas (GB); and D's pages that P "
              "recomputed (it sent them again; D held them):", ""]
        for name, r in rel.items():
            g, hr = r.get("gb"), r["held_recomputed"]
            line = f"- {name}: " + (f"{g['related_raw']:.2f} raw, {g['related_alone']:.2f} alone, "
                                    f"{g['related_as_deltas']:.2f} as deltas" if g else "no measured deltas")
            hd = hr["delta"]
            line += (f"; held pages {hr['pages']:,}, of them {hr['pages_differing']:,} differ" +
                     (f" (best {b(hd['best_bits'])} vs alone {b(hd['alone_bits'])} bits per value, "
                      f"{f(hd['equal_share'], 1)} of the words equal)" if hd else ""))
            L.append(line)
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = analyse(*load(a.run, a.log_dir))
    text = markdown(res)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    if a.markdown:
        pathlib.Path(a.markdown).write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
