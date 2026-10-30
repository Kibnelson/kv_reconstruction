#!/usr/bin/env python3
"""Reuse opportunities of the PD protocol, measured before any base is sent differently (docs/pd_protocol.md,
phase 2): of the KV the prefill engine sent, how much the decode engine already held, or could have held.

From the connector logs (log_dir/tkv_*.jsonl) of a run, per transferred request:
  on GPU      prompt tokens the receiving decode engine already had in its own prefix cache when the request
              arrived (the decode scheduler's local hit, measured by vLLM: event "match")
  in DRAM     further prefix tokens that engine had held before (blocks of earlier requests routed to it, by the
              chained block hashes of event "prefill_done"), evicted from its GPU since: what a host-memory tier on
              the decode node would have kept (unbounded, and within --dram-gb, least recently used)
  elsewhere   further prefix tokens only another decode engine had held (a peer transfer instead of a resend)
  new         the rest: KV that has to come from the prefill engine
Bytes per category at 16 bits per value (raw) and at the bits each transfer was actually sent with.

  python scripts/pd_opportunity.py --log-dir $RUN/pd/logs --output $RUN/pd/opportunity.json [--window t0,t1 ...]
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import glob
import json
import os


def load(log_dir, windows):
    pre, match, sends = {}, {}, {}
    for p in glob.glob(os.path.join(log_dir, "tkv_*.jsonl")):
        with open(p) as fh:
            lines = fh.read().splitlines()
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            t = r.get("time") or r.get("finished_at")
            if windows and (t is None or not any(a <= t <= b for a, b in windows)):
                continue
            ev = r.get("event")
            if ev == "prefill_done":
                pre[r["key"]] = r
            elif ev == "match":
                match[r["key"]] = r
            elif ev == "send" and r.get("bits"):
                sends[r["key"]] = r
    return pre, match, sends


def prefix_len(blocks, have):
    n = 0
    for h in blocks:
        if h not in have:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", action="append", default=[], help="t0,t1 (epoch seconds) of the runs to count")
    ap.add_argument("--block-tokens", type=int, default=16)
    ap.add_argument("--bytes-per-token", type=float, default=28 * 2048 * 2, help="raw KV bytes per token (3B)")
    ap.add_argument("--dram-gb", type=float, default=64.0, help="host memory per decode node for the bounded tier")
    a = ap.parse_args()
    windows = [tuple(float(x) for x in w.split(",")) for w in a.window]
    pre, match, sends = load(a.log_dir, windows)
    reqs = sorted(pre.values(), key=lambda r: r["time"])
    seen = {}                                        # dest -> every block hash it has held
    lru = {}                                         # dest -> OrderedDict of blocks within the DRAM budget
    cap_blocks = int(a.dram_gb * 2 ** 30 / (a.bytes_per_token * a.block_tokens))
    tot = {k: 0.0 for k in ("tokens", "gpu", "dram", "dram_bounded", "elsewhere", "new")}
    sent_bits = {k: 0.0 for k in tot}
    per = []
    for r in reqs:
        dest, blocks, T = r["dest"], r.get("blocks", []), int(r["tokens"])
        bt = a.block_tokens
        gpu = min(T, int(match.get(r["key"], {}).get("local_hit", 0)))
        own = seen.setdefault(dest, set())
        mem = lru.setdefault(dest, OrderedDict())
        k_own = prefix_len(blocks, own) * bt
        k_mem = prefix_len(blocks, mem) * bt
        k_any = max([prefix_len(blocks, s) * bt for d, s in seen.items()] + [0])
        dram = max(0, min(T, k_own) - gpu)
        dram_b = max(0, min(T, k_mem) - gpu)
        else_ = max(0, min(T, k_any) - max(gpu, k_own))
        new = T - gpu - dram - else_
        bits = sends.get(r["key"], {}).get("bits", 16.0)
        row = {"key": r["key"], "dest": dest, "conv": r.get("conv"), "tokens": T, "gpu": gpu, "dram": dram,
               "dram_bounded": dram_b, "elsewhere": else_, "new": new, "bits": bits}
        per.append(row)
        for k in tot:
            tot[k] += row[k]
            sent_bits[k] += row[k] * bits
        own.update(blocks)
        for h in blocks:
            mem[h] = True
            mem.move_to_end(h)
        while len(mem) > cap_blocks:
            mem.popitem(last=False)
    n = tot["tokens"] or 1
    bpt = a.bytes_per_token
    gb = lambda tokens, bits=16.0: tokens * bpt * bits / 16 / 1e9
    out = {"requests": len(per), "tokens": tot, "fraction": {k: tot[k] / n for k in tot},
           "raw_GB": {k: gb(tot[k]) for k in tot},
           "sent_GB_as_coded": {k: sent_bits[k] * bpt / 16 / 1e9 for k in tot},
           "dram_gb_bound": a.dram_gb, "per_request": per}
    f = out["fraction"]
    out["summary"] = (f"{len(per)} transferred requests, {int(n)} prompt tokens: {100 * f['gpu']:.1f}% were already in "
                      f"the decode engine's GPU cache, {100 * f['dram']:.1f}% more it had held before ("
                      f"{100 * f['dram_bounded']:.1f}% within {a.dram_gb:g} GB of host memory), "
                      f"{100 * f['elsewhere']:.1f}% only another decode engine had held, {100 * f['new']:.1f}% new. "
                      f"Sent today: everything ({out['sent_GB_as_coded']['tokens']:.2f} GB as coded, "
                      f"{out['raw_GB']['tokens']:.2f} GB raw); new KV alone: {out['sent_GB_as_coded']['new']:.2f} GB.")
    with open(a.output, "w") as fh:
        json.dump(out, fh, indent=1)
    print(out["summary"])


if __name__ == "__main__":
    main()
