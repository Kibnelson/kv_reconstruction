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
        return "NIXL (vLLM)"
    if env.get("connector") == "lmcache":
        return "LMCache (vLLM)"
    return {"ring": "host memory", "p2p": "GPU to GPU", "tcp": "TCP"}.get(env.get("transport"), env.get("transport"))


def label(env):
    lay = env.get("layout") or f"{env.get('prefills')}p{env.get('decodes')}d"
    return f"{lay}, {path_of(env)}" + (", eager" if env.get("eager") else "")


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
    L.append("")

    # correctness
    rows = []
    for r in runs:
        for c, v in (r["res"].get("check") or {}).items():
            st = v.get("transfers") or {}
            rows.append((label(r["env"]), c, f"{st.get('exact')}/{v.get('prompts')}" if c != "nixl" else "n/a",
                         f"{v.get('same_text_as_local')}/{v.get('prompts')}", "yes" if v.get("ok") else "NO",
                         fmt(st.get("bits"), 3)))
        for v in (r["res"].get("tier") or {}).values():
            if not isinstance(v, dict) or "tier" not in v:
                continue
            st = v["tier"]
            rows.append((label(r["env"]) + " tier", f"{v['codec']} (c{v['clients']})",
                         f"{st.get('exact_chunks')}/{st.get('chunks_loaded')} chunks", "-",
                         "yes" if not st.get("load_errors") else "NO", fmt(st.get("bits"), 3)))
    if rows:
        L += ["## Correctness", "", "| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |",
              "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {' | '.join(map(str, x))} |" for x in rows]
        L.append("")

    # PD sweeps
    groups = {}
    for r in runs:
        for v in (r["res"].get("sweep") or {}).values():
            key = (r["env"].get("layout") or "", path_of(r["env"]), bool(r["env"].get("eager")), str(v.get("link_gbps")))
            groups.setdefault(key, []).append(v)
    for (lay, path, eager, link), vs in sorted(groups.items()):
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
                cells.append(f"{fmt(b.get('request_throughput'))} ({fmt(b.get('request_goodput'))}); "
                             f"{fmt(b.get('median_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)}")
            L.append(f"| {c} | {fmt(bits, 3) if c not in ('raw', 'nixl') else '16.000'} | " + " | ".join(cells) +
                     f" | {fmt(tpot, 1)} |")
        L.append("")
        best = []
        for rt in rates:
            cand = [(by[(c, rt)].get("bench") or {}, c) for c in codecs if (c, rt) in by]
            cand = [(b.get("request_goodput") or 0, -(b.get("median_ttft_ms") or 1e18), c) for b, c in cand]
            if cand:
                g, t, c = max(cand)
                best.append(f"rate {rt}: {c} (goodput {g:.2f}/s, TTFT p50 {-t:.0f} ms)")
        if best:
            L += ["Best by goodput, then TTFT p50: " + "; ".join(best) + ".", ""]

    # CPU tier
    tiers = {}
    for r in runs:
        for v in (r["res"].get("tier") or {}).values():
            if isinstance(v, dict) and "bench" in v:
                tiers.setdefault((r["env"].get("layout"), bool(r["env"].get("eager")), v["clients"]), []).append(
                    (v, path_of(r["env"])))
    for (lay, eager, clients), vs in sorted(tiers.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2])):
        L += [f"## CPU tier, {lay}{', eager engines' if eager else ''}: {clients} clients", "",
              "| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Loads | Tokens loaded | Load ms (copy / "
              "decode) | Bits stored | Chunks stored |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for v, path in sorted(vs, key=lambda x: (x[0]["bench"].get("median_ttft_ms") or 1e18)):
            b, st = v["bench"], v["tier"]
            name = v["codec"] if path not in ("LMCache (vLLM)",) else "LMCache CPU offload (raw)"
            L.append(f"| {name} | {fmt(b.get('request_throughput'))} | {fmt(b.get('median_ttft_ms'), 0)} / "
                     f"{fmt(b.get('p90_ttft_ms'), 0)} / {fmt(b.get('p99_ttft_ms'), 0)} | {fmt(b.get('median_tpot_ms'), 1)} | "
                     f"{st.get('loads')} | {st.get('loaded_tokens')} | {fmt(st.get('load_ms'), 1)} ({fmt(st.get('h2d_ms'), 1)}"
                     f" / {fmt(st.get('decode_ms'), 1)}) | {fmt(st.get('bits'), 3)} | {st.get('stored_chunks')} |")
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
    skipped = {}
    for r in runs:
        for c, w in (r["res"].get("skipped") or {}).items():
            skipped.setdefault(c, w)
    if skipped:
        L += ["## Codecs that did not load", ""] + [f"- {c}: {w[:200]}" for c, w in sorted(skipped.items())] + [""]
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
