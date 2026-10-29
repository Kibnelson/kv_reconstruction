#!/usr/bin/env python3
"""Our best results and the baselines, pinned in one table: scoreboard.json and scoreboard.md from a run folder
of bench_llama_compare.py (or a packaged benchmark_results/ folder).

  Lossless   fewest bits of any predictor (exact cost), without the attention input (the GPU-practical one), as
             a real GPU stream decoded bit-exact, the fastest decode of ours alone; SplitZip and ExANS (the
             authors' codecs); per link speed the fastest mode and the fewest bytes within +25% of its time.
  Lossy      for each baseline row (AQUA-KV 2/3/4-bit, KIVI 4-bit), the fewest bits of ours (CPU coder with the
             strong predictor; GPU codec) at no worse quality: pooled perplexity change at most the baseline's
             plus one standard error, on the WikiText-2 windows (lossy_eval.json; else the two test documents).
  Pool       one cached 8K prefix: memory, prefixes that fit in free GPU memory, time to bring it back.

With --against LOCKED.json, every entry shows the locked value next to the new one, so a later run shows at a
glance what improved and what regressed.

    python scripts/scoreboard.py runs/llama32_3b_compare --against benchmark_results/locked_scoreboard.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LINKS = (1, 10, 50, 100, 200, 400, 800)


def _load(run, name):
    p = Path(run) / name
    return json.loads(p.read_text()) if p.is_file() else None


def _mean(d):
    v = list(d.values())
    return sum(v) / len(v)


def build(run):
    run = Path(run)
    env = _load(run, "environment.json") or {}
    sb = {"run": str(run), "commit": env.get("git_commit"), "gpu": env.get("gpu"), "lossless": {}, "transfer": [],
          "lossy": {}, "pool": []}
    L = sb["lossless"]
    # ---------------------------------------------------------------- lossless bits (exact costs)
    fr = _load(run, "frontier.json")
    ll = _load(run, "lossless.json")
    cands = []
    if fr:
        cands += [(v["label"], v["chunk_average"], v.get("whole", {}), v.get("macs_per_value"), "frontier.json")
                  for v in fr["variants"]]
    if ll:
        cands += [(k, m["chunk_average"], m.get("whole", {}), m.get("macs_per_value"), "lossless.json")
                  for k, m in ll["methods"].items() if not k.endswith("-style")]
        for k in ("SplitZip-style", "ExANS-style"):
            if k in ll["methods"]:
                L[f"{k} (exact cost)"] = {"bits": ll["methods"][k]["chunk_average"], "source": "lossless.json"}
    if cands:
        best = min(cands, key=lambda c: c[1])
        L["fewest bits, any predictor (exact cost)"] = {"mode": best[0], "bits": best[1], "whole": best[2],
                                                        "macs_per_value": best[3], "source": best[4]}
        noatt = [c for c in cands if "ATT" not in c[0] and "fast" not in c[0].lower()]
        if noatt:
            b = min(noatt, key=lambda c: c[1])
            L["fewest bits without the attention input (exact cost)"] = {
                "mode": b[0], "bits": b[1], "whole": b[2], "macs_per_value": b[3], "source": b[4]}
    # ---------------------------------------------------------------- lossless streams and speed
    g = (_load(run, "gpu_speed.json") or {}).get("rows", [])
    z = (_load(run, "szcodec.json") or {}).get("rows", [])
    szr = next((r for r in z + g if r["part"] == "splitzip_official" and r.get("bit_exact")), None)
    n = szr["values"] if szr else None
    ours = []
    for r in g + z:
        if not r.get("bit_exact") or r.get("values") != n or not r.get("decode_gvalues_per_s"):
            continue
        kind = f"rank {r['rank']}" if r.get("rank") else "dense"
        if r["part"] == "fused":
            label = "rANS, fused one-kernel codec"
        elif r["part"] == "layerwise":
            label = f"rANS, per layer, {kind}"
        elif r["part"] == "szcodec":
            label = f"SplitZip format, {kind}, {r.get('layout', 'tile')}"
        elif r["part"] == "mixed" and r["ours_layers"]:
            label = (f"rANS, {r['mode']}" if r["ours_layers"] == len(r["ours_mask"]) else
                     f"rANS + SplitZip, {r['mode']}, {r['ours_layers']} layers ours")
        elif r["part"] == "szcodec_mixed" and r["ours_layers"]:
            label = (f"SplitZip format, {kind}" if r["ours_layers"] == len(r["ours_mask"]) else
                     f"SplitZip format + SplitZip, {kind}, {r['ours_layers']} layers ours")
        else:
            continue
        ours.append({"mode": label, "bits": r["stream_bits_per_value"], "decode_ms": r["decode_ms"],
                     "rate": r["decode_gvalues_per_s"], "mixed": "+ SplitZip" in label,
                     "source": "szcodec.json" if r in z else "gpu_speed.json"})
    alone = [o for o in ours if not o["mixed"]]
    if alone:
        b = min(alone, key=lambda o: (o["bits"], o["decode_ms"]))
        L["fewest bits, real GPU stream (bit-exact)"] = dict(b)
        f = min(alone, key=lambda o: o["decode_ms"])
        mid = min([o for o in alone if o["decode_ms"] <= 2 * f["decode_ms"]], key=lambda o: o["bits"])
        L["fewest bits, real GPU stream, decode within 2x our fastest"] = dict(mid)
        L["fastest decode of ours alone (bit-exact)"] = dict(f)
    if szr:
        L["SplitZip (authors' codec)"] = {"bits": szr["stream_bits_per_value"], "decode_ms": szr["decode_ms"],
                                          "rate": szr["decode_gvalues_per_s"], "source": "official codec, this GPU"}
    ex = _load(run, "exans.json")
    exr = (ex or {}).get("documents", {}).get("test_text")
    if exr:
        L["ExANS (authors' codec)"] = {"bits": exr["bits_per_value"], "decode_ms": 2e3 * n / exr["decode_gbps_wall"] / 1e9
                                       if n else None, "rate": exr["decode_gbps_wall"] / 2,
                                       "source": "official codec, this GPU"}
    # ---------------------------------------------------------------- transfer per link
    if szr and ours:
        t = lambda bits, rate, R: max(bits / R, 1 / rate) * n / 1e9
        mib = lambda bits: bits * n / 8 / 2 ** 20
        for R in LINKS:
            ts = t(szr["stream_bits_per_value"], szr["decode_gvalues_per_s"], R)
            fast = min(ours, key=lambda o: t(o["bits"], o["rate"], R))
            tf = t(fast["bits"], fast["rate"], R)
            if tf >= ts:                             # no layer of ours helps: the best mix is SplitZip itself
                fast = {"mode": "SplitZip itself (no layer of ours helps)", "bits": szr["stream_bits_per_value"],
                        "rate": szr["decode_gvalues_per_s"]}
                tf = ts
            budget = 1.25 * tf
            lean = min([o for o in ours if t(o["bits"], o["rate"], R) <= budget], key=lambda o: o["bits"], default=None)
            row = {"link_gbps": R, "splitzip_ms": 1e3 * ts, "splitzip_mib": mib(szr["stream_bits_per_value"])}
            if exr:
                row.update(exans_ms=1e3 * t(exr["bits_per_value"], exr["decode_gbps_wall"] / 2, R),
                           exans_mib=mib(exr["bits_per_value"]))
            row.update(ours_mode=fast["mode"], ours_ms=1e3 * tf, ours_mib=mib(fast["bits"]), ours_vs_splitzip=ts / tf)
            if lean:
                row.update(lean_mode=lean["mode"], lean_ms=1e3 * t(lean["bits"], lean["rate"], R),
                           lean_mib=mib(lean["bits"]))
            sb["transfer"].append(row)
    # ---------------------------------------------------------------- lossy, quality-matched
    lres, lsrc = _load(run, "lossy_eval.json"), "lossy_eval.json (WikiText-2 windows)"
    if not lres or not lres.get("documents"):
        lres, lsrc = _load(run, "lossy.json"), "lossy.json (two test documents)"
    if lres and lres.get("documents"):
        docs = list(lres["documents"])
        labels = [k for k in lres["documents"][docs[0]] if k != "exact KV" and
                  all(k in lres["documents"][d] for d in docs)]

        def stat(label):
            rs = [lres["documents"][d][label] for d in docs]
            base = [lres["documents"][d]["exact KV"]["nll"] for d in docs]
            pooled = 100 * (math.exp(sum(r["nll"] for r in rs) / len(rs) - sum(base) / len(base)) - 1)
            ch = [r["ppl_change_pct"] for r in rs]
            m = sum(ch) / len(ch)
            se = (sum((c - m) ** 2 for c in ch) / (len(ch) - 1)) ** 0.5 / len(ch) ** 0.5 if len(ch) > 1 else 0.0
            return {"bits": sum(r["bits"] for r in rs) / len(rs), "ppl_change_pct": pooled, "se": se}
        stats = {k: stat(k) for k in labels}
        ta, tk = _load(run, "time_aqua.json"), _load(run, "time_kivi.json")
        lw = [r for r in g if r["part"] == "lossy_layerwise" and r.get("bit_exact") and not r.get("rank")]
        gdec = min((r["decode_ms"] for r in lw), default=None)
        sb["lossy"] = {"source": lsrc, "windows": len(docs), "targets": []}
        for base in ("AQUA-KV, 2-bit", "AQUA-KV, 3-bit", "AQUA-KV, 4-bit", "KIVI, 4-bit"):
            if base not in stats:
                continue
            s = stats[base]
            limit = s["ppl_change_pct"] + s["se"]
            entry = {"baseline": base, "baseline_bits": s["bits"], "baseline_change_pct": s["ppl_change_pct"],
                     "baseline_se": s["se"], "limit_pct": limit}
            src = ta if base.startswith("AQUA") else tk
            bb = base.split(", ")[1].split("-")[0]
            if src and bb in src.get("bits", {}):
                entry["baseline_decode_ms"] = src["bits"][bb]["decode_ms"]
            for who, pick in (("ours_cpu", lambda k: k.startswith("Ours, step")),
                              ("ours_gpu", lambda k: k.startswith("Ours, GPU codec"))):
                ok = [(k, v) for k, v in stats.items() if pick(k) and v["ppl_change_pct"] <= limit]
                if ok:
                    k, v = min(ok, key=lambda kv: kv[1]["bits"])
                    entry[who] = {"mode": k, **v}
                    if who == "ours_gpu" and gdec:
                        entry[who]["decode_ms"] = gdec
            sb["lossy"]["targets"].append(entry)
    # ---------------------------------------------------------------- AQUA-KV's own protocol (streaming perplexity)
    ap = _load(run, "aqua_protocol.json")
    if ap and "exact" in ap.get("methods", {}):
        ms = ap["methods"]
        ex = ms["exact"]
        tpw = ex["tokens_per_window"]

        def change(r):
            d = [math.exp((a - b) / tpw) - 1 for a, b in zip(r["nll_sums_f32"], ex["nll_sums_f32"])]
            m = sum(d) / len(d)
            se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5 if len(d) > 1 else 0.0
            return 100 * (r["perplexity_f32"] / ex["perplexity_f32"] - 1), 100 * se
        rows = {k: dict(bits=r.get("bits_per_compressed_value", 16.0), ppl=r["perplexity"],
                        ppl_f32=r["perplexity_f32"], change_pct=change(r)[0], se=change(r)[1])
                for k, r in ms.items() if k != "exact"}
        seqlen = ap.get("protocol", "seqlen 8192,").split("seqlen ")[-1].split(",")[0]
        cm = _load(run, "captures/meta.json") or _load(run, "capture_meta.json") or {}
        nval = int(seqlen) * cm.get("layers", 0) * 2 * cm.get("kv_heads", 0) * cm.get("head_dim", 0)
        sb["aqua_protocol"] = {"windows": ap["windows"], "seqlen": seqlen, "values_per_window": nval,
                               "exact_ppl": ex["perplexity"],
                               "exact_ppl_f32": ex["perplexity_f32"], "targets": []}
        for base in [k for k in rows if not k.startswith("ours")]:
            b = rows[base]
            ok = [(k, v) for k, v in rows.items() if k.startswith("ours") and v["change_pct"] <= b["change_pct"] + b["se"]]
            entry = {"baseline": base, **{f"baseline_{f}": b[f] for f in ("bits", "ppl", "change_pct", "se")}}
            if ok:
                k, v = min(ok, key=lambda kv: kv[1]["bits"])
                entry["ours"] = {"mode": k, **v}
            sb["aqua_protocol"]["targets"].append(entry)
        off = _load(run, "aqua_official.json")
        if off:
            sb["aqua_protocol"]["authors_scripts"] = off
    # ---------------------------------------------------------------- pool (one cached prefix)
    pool = _load(run, "pool.json")
    if pool and n:
        free = pool["free_after_model_bytes"]
        recompute = (pool.get("prefill_seconds") or {}).get(str(pool["tokens"]))
        sb["pool_recompute_ms"] = 1e3 * recompute if recompute else None

        def pentry(name, bits, decode_ms):
            b = bits * n / 8
            sb["pool"].append({"method": name, "mib": b / 2 ** 20, "prefixes_fit": int(free // b),
                               "restore_ms": decode_ms})
        pentry("raw BF16", 16.0, None)
        if szr:
            pentry("SplitZip", szr["stream_bits_per_value"], szr["decode_ms"])
        if exr:
            pentry("ExANS", exr["bits_per_value"], L["ExANS (authors' codec)"]["decode_ms"])
        for key in ("fewest bits, real GPU stream (bit-exact)", "fewest bits, real GPU stream, decode within 2x our fastest",
                    "fastest decode of ours alone (bit-exact)"):
            if key in L:
                pentry(f"ours lossless ({L[key]['mode']})", L[key]["bits"], L[key]["decode_ms"])
        for tgt in sb.get("lossy", {}).get("targets", []):
            if tgt["baseline"].startswith("AQUA"):
                pentry(tgt["baseline"], tgt["baseline_bits"], tgt.get("baseline_decode_ms"))
                if "ours_gpu" in tgt:
                    pentry(f"ours lossy, GPU codec, at {tgt['baseline']} quality", tgt["ours_gpu"]["bits"],
                           tgt["ours_gpu"].get("decode_ms"))
    return sb


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def markdown(sb, locked=None):
    """The scoreboard as Markdown; locked: an earlier scoreboard to show beside (locked -> now)."""
    lk = locked or {}

    def was(path, field, fmt):
        cur = lk
        for p in path:
            cur = cur.get(p, {}) if isinstance(cur, dict) else {}
        v = cur.get(field) if isinstance(cur, dict) else None
        return f" (locked {format(v, fmt)})" if v is not None and locked else ""
    lines = [f"## Scoreboard: our best against the baselines", "",
             f"Run {sb['run']}, commit {(sb.get('commit') or '?')[:7]}, {sb.get('gpu')}." +
             (f" Locked values (in brackets) from commit {(lk.get('commit') or '?')[:7]}." if locked else ""), "",
             "**Lossless, bits per value** (BF16 = 16; the exact cost of a predictor is what its coder would need; a "
             "stream is a real encoding decoded bit-exact on the GPU):", "",
             "| Entry | Mode | Bits per value | Decode (one document) |", "| --- | --- | --- | --- |"]
    for key, e in sb["lossless"].items():
        lines.append(f"| {key} | {e.get('mode', '')} | {e['bits']:.3f}{was(('lossless', key), 'bits', '.3f')} | "
                     f"{_f(e.get('decode_ms'), '.1f') + ' ms' if e.get('decode_ms') else '-'}"
                     f"{was(('lossless', key), 'decode_ms', '.1f')} |")
    if sb["transfer"]:
        lines += ["", "**Transfer of one document (ideal link, decode overlapped with arrival)**: SplitZip, ExANS, our "
                  "fastest mode, and our fewest bytes within +25% of the fastest time:", "",
                  "| Link | SplitZip | ExANS | Ours, fastest | vs SplitZip | Ours, fewest bytes within +25% |",
                  "| --- | --- | --- | --- | --- | --- |"]
        old = {r["link_gbps"]: r for r in lk.get("transfer", [])}
        for r in sb["transfer"]:
            o = old.get(r["link_gbps"])
            lines.append(f"| {r['link_gbps']} Gb/s | {r['splitzip_ms']:.1f} ms, {r['splitzip_mib']:.0f} MiB | "
                         f"{_f(r.get('exans_ms'), '.1f')} ms, {_f(r.get('exans_mib'), '.0f')} MiB | "
                         f"{r['ours_ms']:.1f} ms, {r['ours_mib']:.0f} MiB ({r['ours_mode']}) | {r['ours_vs_splitzip']:.2f}x"
                         + (f" (locked {o['ours_vs_splitzip']:.2f}x)" if o else "") + " | " +
                         (f"{r['lean_ms']:.1f} ms, {r['lean_mib']:.0f} MiB ({r['lean_mode']})" if r.get("lean_mode") else "-")
                         + " |")
    if sb.get("lossy", {}).get("targets"):
        lo = sb["lossy"]
        old = {t["baseline"]: t for t in lk.get("lossy", {}).get("targets", [])}
        lines += ["", f"**Lossy, at the baselines' quality** ({lo['source']}, {lo['windows']} windows; ours = fewest bits "
                  "with a pooled perplexity change at most the baseline's plus one standard error):", "",
                  "| Baseline | Its bits, change | Ours, CPU coder (strong predictor) | Ours, GPU codec | Decode: baseline / ours GPU |",
                  "| --- | --- | --- | --- | --- |"]
        for t in lo["targets"]:
            o = old.get(t["baseline"], {})

            def cell(who):
                e = t.get(who)
                if not e:
                    return "none at this quality"
                prev = o.get(who, {}).get("bits")
                return (f"{e['bits']:.2f} bits, {e['ppl_change_pct']:+.2f}% ({e['mode'].replace('Ours, GPU codec (one-step, measured stream), ', '').replace('Ours, ', '')})"
                        + (f" (locked {prev:.2f})" if prev is not None and locked else ""))
            lines.append(f"| {t['baseline']} | {t['baseline_bits']:.2f} bits, {t['baseline_change_pct']:+.2f}% ± "
                         f"{t['baseline_se']:.2f} | {cell('ours_cpu')} | {cell('ours_gpu')} | "
                         f"{_f(t.get('baseline_decode_ms'), '.0f')} ms / "
                         f"{_f((t.get('ours_gpu') or {}).get('decode_ms'), '.0f')} ms |")
    if sb.get("aqua_protocol"):
        ap = sb["aqua_protocol"]
        old = {t["baseline"]: t for t in lk.get("aqua_protocol", {}).get("targets", [])}
        lines += ["", f"**Lossy in AQUA-KV's own protocol** (WikiText-2 perplexity with a streaming compressed cache, "
                  f"{ap['windows']} windows of {ap.get('seqlen', '8192')} tokens; perplexity with the authors' BF16 loss, "
                  f"exact cache {ap['exact_ppl']:.3f}, paper 6.98; the change uses a float32 loss; ours = fewest bits "
                  "per compressed value with a change at most the baseline's plus one standard error):", "",
                  "| Baseline | Its bits, perplexity, change | Ours, GPU codec |", "| --- | --- | --- |"]
        nv = ap.get("values_per_window") or 0
        mb = lambda bits: f", {bits * nv / 8 / 2 ** 20:.0f} MiB per window" if nv else ""
        for t in ap["targets"]:
            o = old.get(t["baseline"], {}).get("ours", {}).get("bits")
            e = t.get("ours")
            lines.append(f"| {t['baseline']} | {t['baseline_bits']:.3f} bits{mb(t['baseline_bits'])}, "
                         f"{t['baseline_ppl']:.3f}, {t['baseline_change_pct']:+.2f}% ± {t['baseline_se']:.2f} | " +
                         (f"{e['bits']:.3f} bits{mb(e['bits'])}, {e['ppl']:.3f}, {e['change_pct']:+.2f}% ({e['mode']})" if e else
                          "none at this quality") + (f" (locked {o:.3f})" if o is not None and locked else "") + " |")
        for k, v in (ap.get("authors_scripts") or {}).items():
            lines.append(f"| AQUA-KV, the authors' own scripts: {k} | perplexity {v:.3f} | - |")
    if sb["pool"]:
        lines += ["", f"**Pool: one cached 8K prefix** (recomputing it: {_f(sb.get('pool_recompute_ms'), '.0f')} ms):", "",
                  "| Method | Memory | Prefixes that fit | Bring back |", "| --- | --- | --- | --- |"]
        for p in sb["pool"]:
            lines.append(f"| {p['method']} | {p['mib']:.0f} MiB | {p['prefixes_fit']} | "
                         f"{_f(p['restore_ms'], '.0f') + ' ms' if p['restore_ms'] is not None else 'a copy'} |")
    return lines + [""]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--against", type=Path, default=None, help="a locked scoreboard.json to compare with")
    p.add_argument("--lock", type=Path, default=None, help="also write the scoreboard here (the new locked file)")
    a = p.parse_args(argv)
    sb = build(a.run)
    locked = json.loads(a.against.read_text()) if a.against and a.against.is_file() else None
    (a.run / "scoreboard.json").write_text(json.dumps(sb, indent=1))
    md = "\n".join(markdown(sb, locked))
    (a.run / "scoreboard.md").write_text(md + "\n")
    if a.lock:
        a.lock.write_text(json.dumps(sb, indent=1))
    print(md)


if __name__ == "__main__":
    main()
