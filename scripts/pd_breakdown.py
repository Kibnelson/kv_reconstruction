#!/usr/bin/env python3
"""Where each request's time went, from a PD run's own logs: the proxy's record (request in, first token out), the
prefill engine's (prefill done, gather, encode, link) and the decode engine's (received, install), joined by the
transfer's key, then the median of each part per benchmark point (one runs/<point>.json).

  python3 -I scripts/pd_breakdown.py RUN_DIR [RUN_DIR ...] [--json out.json] [--markdown out.md]

RUN_DIR: a run's folder as published (benchmark_results/pd_card/<tag>/matrix/<run>/, with runs/, proxy.jsonl.gz,
connector/*.jsonl.gz and env.json). A request sent in pieces counts once (pd_bench.merge_pieces): its link from the
first piece's start to the last one's end, its encode and install summed; those then overlap, so the parts no longer
add up to the TTFT. A point is matched to its requests by time: vLLM's bench names the end of a point in the node's
local time (its "date"), so the offset to the logs' UNIX times is the whole hour under which the most requests fall
in a point of their own codec.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import json
import pathlib
import re
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pd_bench import merge_pieces  # noqa: E402

NAME = re.compile(r"^(shape|sweep)_(.+?)_(native|\d+)gbps_(.*)$")
PARTS = ("prefill", "pick", "gather", "encode", "link_wait", "link", "recv_wait", "install_queued", "install",
         "first_token")


def jl(p):
    with gzip.open(p, "rt") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                pass


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def points(R):
    """The run's benchmark points: kind, codec, link, the rest of the name, end (naive local time), duration."""
    out = []
    for f in sorted((R / "runs").glob("*.json")):
        m = NAME.match(f.stem)
        if not m:
            continue
        try:
            d = json.loads(f.read_text())
        except ValueError:
            continue
        if "date" not in d or "duration" not in d:
            continue
        out.append(dict(kind=m.group(1), codec=m.group(2), gbps=m.group(3), rest=m.group(4),
                        end=dt.datetime.strptime(d["date"], "%Y%m%d-%H%M%S"), duration=float(d["duration"]),
                        ttft=d.get("median_ttft_ms"), tpot=d.get("median_tpot_ms"), completed=d.get("completed"),
                        file=f.name))
    return out


def analyse(R):
    R = pathlib.Path(R)
    if not (R / "proxy.jsonl.gz").exists():
        return []
    proxy = {r["key"]: r for r in jl(R / "proxy.jsonl.gz") if "key" in r}
    recs = [r for f in sorted((R / "connector").glob("*.jsonl.gz")) for r in jl(f)]
    recs = merge_pieces(recs)
    send = {r["key"]: r for r in recs if r.get("event") == "send" and "key" in r}
    inst = {r["key"]: r for r in recs if r.get("event") == "install" and "key" in r}
    pdone = {r["key"]: r for r in recs if r.get("event") == "prefill_done" and "key" in r}
    pts = points(R)
    if not pts or not proxy:
        return []
    t_in = [(p["t_in"], p.get("codec")) for p in proxy.values() if "t_in" in p]

    def window(p, off):
        b = p["end"].replace(tzinfo=dt.timezone.utc).timestamp() - off + 1
        return b - p["duration"] - 3, b
    def hits(h):
        return sum(1 for p in pts for a, b in [window(p, h * 3600)] for t, c in t_in if a <= t <= b and c == p["codec"])
    off = 3600 * max(range(-14, 15), key=hits)
    env = json.loads((R / "env.json").read_text()) if (R / "env.json").exists() else {}
    rows = []
    for p in pts:
        a, b = window(p, off)
        seg = collections.defaultdict(list)
        n = 0
        for k, x in proxy.items():
            if not (a <= x.get("t_in", -1) <= b) or x.get("codec") != p["codec"]:
                continue
            s, i, q = send.get(k), inst.get(k), pdone.get(k)
            if s is None or i is None or "t_first_byte" not in x:
                continue
            n += 1
            seg["ttft"].append(1e3 * (x["t_first_byte"] - x["t_in"]))
            if q is not None:
                seg["prefill"].append(1e3 * (q["time"] - x["t_in"]))
                seg["pick"].append(1e3 * (s["picked_at"] - q["time"]) if "picked_at" in s else None)
            seg["gather"].append(s.get("gather_ms"))
            seg["encode"].append(s.get("encode_ms"))
            if "queued" in s and "send_begin" in s:
                seg["link_wait"].append(1e3 * (s["send_begin"] - s["queued"]))
                seg["link"].append(1e3 * (s["send_end"] - s["send_begin"]))
            if "link_recv_end" in i and "taken_at" in i:
                seg["recv_wait"].append(1e3 * (i["taken_at"] - i["link_recv_end"]))
            seg["install_queued"].append(i.get("install_queued_ms"))
            seg["install"].append(i.get("decode_install_ms"))
            if "done_at" in i:
                seg["first_token"].append(1e3 * (x["t_first_byte"] - i["done_at"]))
            seg["tokens"].append(s.get("tokens"))
            seg["bits"].append(s.get("bits"))
            seg["pieces"].append(s.get("pieces") or 1)
        if n:
            row = dict(run=R.name, transport=env.get("transport"), layout=env.get("layout"),
                       shared_link=env.get("shared_link"), n=n, bench_ttft=p["ttft"],
                       **{k: p[k] for k in ("kind", "codec", "gbps", "rest", "file")})
            row.update({k: med(v) for k, v in seg.items()})
            rows.append(row)
    return rows


def markdown(rows):
    f = lambda v: "-" if v is None else f"{v:.0f}"
    L = ["# Where the time went (ms, medians per point)", "",
         "TTFT: the proxy's, request in to first token (the bench's beside it). Prefill: request in to prefill done. "
         "Pick: until the connector took it. Link wait: queued behind other transfers. Install queued: waiting for "
         "the installer thread. First token: install done to the first token. A request in pieces: link from the "
         "first piece's start to the last one's end, encode and install summed (they overlap).", ""]
    for run in sorted({r["run"] for r in rows}):
        L += [f"## {run}", "", "| Point | Codec | Link | n | Bits | TTFT (bench) | " +
              " | ".join(p.replace("_", " ") for p in PARTS) + " |", "| --- " * (6 + len(PARTS)) + "|"]
        for r in sorted((r for r in rows if r["run"] == run), key=lambda r: (r["kind"], r["rest"], r["gbps"],
                                                                             r["codec"])):
            L.append(f"| {r['kind']} {r['rest']} | {r['codec']} | {r['gbps']} | {r['n']} | "
                     f"{'-' if r.get('bits') is None else format(r['bits'], '.2f')} | "
                     f"{f(r.get('ttft'))} ({f(r.get('bench_ttft'))}) | " + " | ".join(f(r.get(p)) for p in PARTS)
                     + " |")
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--markdown")
    a = ap.parse_args(argv)
    rows = [r for d in a.runs for r in analyse(d)]
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1))
    text = markdown(rows)
    if a.markdown:
        pathlib.Path(a.markdown).write_text(text)
    else:
        print(text)
    print(f"{len(rows)} points", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
