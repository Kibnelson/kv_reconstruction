"""Where the transfer codecs spend their time: encode and decode of real captured KV on the GPU, without vLLM (run it
in the vLLM environment, as the self-test).

  python -m kv_compare.pd.profile --codecs $RUN/pd_codecs.pt --captures $RUN/captures --doc ds_wikitext2_r0 \
      --tokens 512,2048,8192,16384 --names ours-hbh,t2-hbh,t2-rank256,raw,splitzip,adacode,exactkv,exans \
      --output $RUN/pd_profile.json --markdown $RUN/pd_profile.md

Per codec and token count, for encode and for decode (into a layer sink, as the install):
  wall_ms      the call with the GPU finished (median of --reps)
  issue_ms     the CPU's time until the call returns, the GPU perhaps still at work: near wall_ms, the call is
               bound by issuing launches (what CUDA graphs remove); far below, by the GPU's work
  gpu_ms       the summed time of its GPU kernels (torch.profiler); wall_ms - gpu_ms is the GPU idle in between
  launches     its number of GPU kernels
  kernels      the kernels by kind (our decoder, encoder, features, products, packing, copies ...): count and ms
A prompt longer than the captured document repeats it (the timing is what counts here, not the bits).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import torch

# kernel kinds, by the first of these words in a kernel's name (lower case)
KINDS = [
    ("our decoder", ("_ldec",)),
    ("our encoder", ("_lenc",)),
    ("our features", ("_lfeat", "_lsurp", "_lprior", "_feat")),
    ("products (int8 GEMM)", ("_igemm", "int_mm", "igemm", "cutlass", "gemm", "gemv", "sm80_xmma", "ampere")),
    ("packing", ("_tail_copy", "compact")),
    ("exactkv", ("_dec_only", "_load_ctile", "_encode_pages")),
    ("copies and fills", ("copy", "memcpy", "fill", "memset", "cat", "elementwise", "vectorized", "unrolled",
                          "index", "scatter", "gather", "reduce", "cumsum", "scan", "where")),
]


def kind(name):
    n = name.lower()
    for k, words in KINDS:
        if any(w in n for w in words):
            return k
    return "other"


def _timed(fn, sync, reps):
    """(median wall ms, median issue ms) of fn() over reps calls, after one call that compiles."""
    fn()
    sync()
    walls, issues = [], []
    for _ in range(reps):
        sync()
        t = time.perf_counter()
        fn()
        issues.append(1e3 * (time.perf_counter() - t))
        sync()
        walls.append(1e3 * (time.perf_counter() - t))
    return statistics.median(walls), statistics.median(issues)


def _kernels(fn, sync):
    """(summed GPU ms, launches, {kind: [count, ms]}, [(ms, count, name)] the top kernels) of one fn() call."""
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        sync()
    by_kind, by_name = {}, {}
    total, n = 0.0, 0
    for e in prof.events():
        dt = getattr(e, "device_type", None)
        if dt is None or "CUDA" not in str(dt):
            continue
        ms = (getattr(e, "device_time", None) or getattr(e, "cuda_time", 0) or 0) / 1e3
        total += ms
        n += 1
        k = kind(e.name)
        by_kind.setdefault(k, [0, 0.0])
        by_kind[k][0] += 1
        by_kind[k][1] += ms
        x = by_name.setdefault(e.name[:80], [0, 0.0])
        x[0] += 1
        x[1] += ms
    top = sorted(((ms, c, name) for name, (c, ms) in by_name.items()), reverse=True)[:6]
    return total, n, by_kind, top


def _with(flags, fn):
    """fn() with the codecs' switches (kv_compare.pd.codecs, kv_compare.layerwise) set to flags, then restored."""
    from .. import layerwise as lw
    from . import codecs as cd
    mods = {k: (cd if hasattr(cd, k) else lw) for k in flags}
    old = {k: getattr(m, k) for k, m in mods.items()}
    try:
        for k, v in flags.items():
            setattr(mods[k], k, v)
        return fn()
    finally:
        for k, v in old.items():
            setattr(mods[k], k, v)


def words_of(doc, T, dev):
    """[L, T, 2C] int16 words of the captured document, repeated when T is longer."""
    n = min(T, doc.tokens)
    w = torch.empty(doc.layers, n, 2 * doc.columns, dtype=torch.int16, device=dev)
    for l in range(doc.layers):
        x = doc.words(l, dev, rows=n)
        w[l] = (x - 65536 * (x >= 32768).to(torch.int32)).to(torch.int16)
    if n < T:
        w = w.repeat(1, -(-T // n), 1)[:, :T].contiguous()
    return w


def markdown(rows):
    L = ["# Where the codecs spend their time (encode / decode on one GPU, no vLLM)", "",
         "wall: the call with the GPU finished; issue: the CPU's time to issue it; GPU: its kernels' summed time; "
         "launches: its kernels. Issue near wall and GPU far below it: bound by issuing launches.", "",
         "| Codec | Tokens | Encode wall / issue / GPU ms | launches | Decode wall / issue / GPU ms | launches | Bits |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    before = [r for r in rows if r.get("encode_twice") or r.get("decode_before")]
    f = lambda x: "-" if x is None else f"{x:.1f}"
    for r in rows:
        e, d = r.get("encode") or {}, r.get("decode") or {}
        L.append(f"| {r['codec']} | {r['tokens']} | {f(e.get('wall_ms'))} / {f(e.get('issue_ms'))} / "
                 f"{f(e.get('gpu_ms'))} | {e.get('launches', '-')} | {f(d.get('wall_ms'))} / {f(d.get('issue_ms'))} / "
                 f"{f(d.get('gpu_ms'))} | {d.get('launches', '-')} | {f(r.get('bits'))} |"
                 + (f" {r['error']} |" if r.get("error") else ""))
    if before:
        L += ["", "## T and T2 against their calls as of job 30436 (wall ms)", "",
              "Encode: each layer's sums once / twice (TKV_ENCODE_ONCE). Decode: graphs / launches one by one with "
              "the fused group product / one by one with the column copy, product and add (TKV_DECODE_GRAPHS, "
              "TKV_GROUP_PRODUCT).", "",
              "| Codec | Tokens | Encode once | Encode twice | Decode (graphs) | Decode eager | Decode as before |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        g = lambda r, k: f((r.get(k) or {}).get("wall_ms"))
        for r in before:
            L.append(f"| {r['codec']} | {r['tokens']} | {g(r, 'encode')} | {g(r, 'encode_twice')} | {g(r, 'decode')} | "
                     f"{g(r, 'decode_eager')} | {g(r, 'decode_before')} |")
    L += ["", "## By kernel kind (count, GPU ms)", ""]
    for r in rows:
        for side in ("encode", "decode"):
            s = r.get(side) or {}
            if s.get("kinds"):
                kinds = "; ".join(f"{k} {c} x, {ms:.1f} ms" for k, (c, ms) in
                                  sorted(s["kinds"].items(), key=lambda x: -x[1][1]))
                L.append(f"- {r['codec']}, {r['tokens']} tokens, {side}: {kinds}")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--doc", default="ds_wikitext2_r0")
    ap.add_argument("--tokens", default="512,2048,8192,16384")
    ap.add_argument("--names", default="ours-hbh,t2-hbh,t2-rank256,raw,splitzip,adacode,exactkv,exans")
    ap.add_argument("--splitzip-source", default=None)
    ap.add_argument("--adacode-source", default=None)
    ap.add_argument("--exactkv-source", default=None)
    ap.add_argument("--exans-info", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=None)
    ap.add_argument("--markdown", default=None)
    a = ap.parse_args(argv)
    from ..predict import Doc
    from . import codecs as cd
    from .wire import bundle, payload_bytes, unbundle
    dev = torch.device(a.device)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    built, errors, meta = cd.build(a.codecs, dev, a.names.split(","), a.splitzip_source, a.adacode_source,
                                   exactkv_source=a.exactkv_source, exans_info=a.exans_info)
    doc = Doc(a.captures, a.doc)
    rows = []
    for T in [int(t) for t in a.tokens.split(",")]:
        words = words_of(doc, T, dev)
        back = torch.empty_like(words)
        for name, c in built.items():
            if getattr(c, "tier_only", False):
                continue
            row = {"codec": name, "tokens": T}
            try:
                with torch.inference_mode():
                    enc = lambda: c.encode(words)
                    wall, issue = _timed(enc, sync, a.reps)
                    row["encode"] = {"wall_ms": wall, "issue_ms": issue}
                    parts, m = c.encode(words)
                    header, blob = bundle(parts, m)
                    row["bits"] = 8.0 * payload_bytes(header) / words.numel()
                    p2, m2 = unbundle(header, blob)
                    dec = lambda: c.decode(p2, m2, lambda l, w: back[l].copy_(w))
                    wall, issue = _timed(dec, sync, a.reps)
                    row["decode"] = {"wall_ms": wall, "issue_ms": issue}
                    row["exact"] = None if getattr(c, "lossy", False) else bool(torch.equal(back, words))
                    if name.startswith(("ours-", "t2-")):
                        # the same calls as of job 30436: each layer's sums made twice by the sender; the receiver
                        # issuing its launches one by one (no graphs), with or without the fused group product
                        for label, side, fn, flags in (
                                ("encode_twice", "encode", enc, {"ENCODE_ONCE": False}),
                                ("decode_eager", "decode", dec, {"DECODE_GRAPHS": False}),
                                ("decode_before", "decode", dec, {"DECODE_GRAPHS": False, "GROUP_PRODUCT": "torch"})):
                            wall, issue = _with(flags, lambda: _timed(fn, sync, a.reps))
                            row[label] = {"wall_ms": wall, "issue_ms": issue}
                    if dev.type == "cuda":
                        for side, fn in (("encode", enc), ("decode", dec)):
                            try:
                                gpu, n, kinds, top = _kernels(fn, sync)
                                row[side].update(gpu_ms=gpu, launches=n, kinds=kinds,
                                                 top=[{"ms": ms, "count": k, "name": nm} for ms, k, nm in top])
                            except Exception as exc:     # no profiler here: the timings above stand
                                row[side]["profiler_error"] = repr(exc)[:200]
            except Exception as exc:
                row["error"] = repr(exc)[:300]
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("encode", "decode")} |
                             {s: {k: v for k, v in (row.get(s) or {}).items() if k in ("wall_ms", "issue_ms",
                                                                                       "gpu_ms", "launches")}
                              for s in ("encode", "decode")}), flush=True)
        del words, back
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    out = {"rows": rows, "codec_errors": errors, "meta": meta, "torch": torch.__version__,
           "decode_chunks": cd.DECODE_CHUNKS, "piece_tokens": cd.PIECE_TOKENS, "encode_once": cd.ENCODE_ONCE,
           "decode_graphs": cd.DECODE_GRAPHS, "graph_events": cd.graph_events()}
    if a.output:
        with open(a.output, "w") as f:
            json.dump(out, f, indent=1)
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(markdown(rows))
    bad = [r for r in rows if r.get("error") or r.get("exact") is False]
    print("PROFILE " + ("OK" if not bad else f"with {len(bad)} failures") + f" ({len(rows)} codec runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
