#!/usr/bin/env python3
"""How fast can one GPU send KV to another on this machine? (prefill/decode disaggregation without a datacenter)

For every ordered pair of GPUs: whether peer-to-peer access is possible, and the measured bandwidth of moving one
document's KV (Llama-3.2-3B, 8,192 tokens: 896 MiB in BF16) three ways:

  direct     dst.copy_(src): peer-to-peer over PCIe or NVLink when the GPUs allow it (otherwise the driver stages it)
  host       through pinned CPU memory, one copy after the other (GPU -> CPU -> GPU)
  host_pipe  through pinned CPU memory in 32 MiB chunks, the two copies overlapped on two streams (what a transfer
             engine does without peer-to-peer)

and what that means for one document: raw BF16 (896 MiB), SplitZip (675 MiB), our lossless stream (504 MiB) and a
2-bit lossy stream (117 MiB), wire time only. Prints `nvidia-smi topo -m` first. Results also go to --output (JSON).

Every GPU's own host legs are measured too, which is all one GPU allows (a one-GPU allocation): device -> pinned
host memory and back (the PCIe legs a transfer to another machine passes through on each side, before and after
the network), and the same from pageable memory (what a plain .cpu() / .to(device) does).

    python scripts/pd_link_probe.py --output pd_link_probe.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch

DOC_MIB = {"raw BF16": 896, "SplitZip": 675, "ours lossless": 504, "ours / AQUA-KV lossy, ~2 bits": 117}


def sync():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def timed(fn, reps):
    """Best wall time of fn over reps runs, every GPU synchronised before and after."""
    fn()
    sync()
    best = float("inf")
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mib", type=int, default=896, help="bytes moved per measurement (default: one document)")
    p.add_argument("--chunk-mib", type=int, default=32)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--pairs", default="", help="e.g. 0-1,0-2 (default: every ordered pair)")
    p.add_argument("--output", default="")
    a = p.parse_args(argv)
    try:
        topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:                          # no nvidia-smi: carry on with the measurements
        topo = f"nvidia-smi topo -m not available ({exc!r})"
    print(topo)
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    print(f"{n} GPUs: {', '.join(names)}; torch {torch.__version__}, CUDA {torch.version.cuda}")
    if n < 1:
        raise SystemExit("no GPU visible")
    nbytes = a.mib << 20
    chunk = a.chunk_mib << 20
    res = {"gpus": names, "topology": topo, "mib": a.mib, "host_legs": [], "pairs": []}
    print(f"\n{'GPU':>4} {'to pinned':>12} {'from pinned':>12} {'to pageable':>12} {'from pageable':>14}")
    for i in range(n):
        src = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{i}").random_(0, 255)
        pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        pageable = torch.empty(nbytes, dtype=torch.uint8)
        row = {"gpu": i, "name": names[i]}
        for k, fn in (("d2h_pinned", lambda: pinned.copy_(src, non_blocking=True)),
                      ("h2d_pinned", lambda: src.copy_(pinned, non_blocking=True)),
                      ("d2h_pageable", lambda: pageable.copy_(src)),
                      ("h2d_pageable", lambda: src.copy_(pageable))):
            t = timed(fn, a.reps)
            row[k] = {"seconds": t, "GB_per_s": nbytes / t / 1e9, "Gb_per_s": 8 * nbytes / t / 1e9}
        res["host_legs"].append(row)
        print(f"{i:>4} " + " ".join(f"{row[k]['GB_per_s']:9.1f} GB/s" for k in
                                     ("d2h_pinned", "h2d_pinned", "d2h_pageable", "h2d_pageable")))
        del src, pinned, pageable
        torch.cuda.empty_cache()
    pairs = [tuple(int(x) for x in s.split("-")) for s in a.pairs.split(",") if s] or \
        [(i, j) for i in range(n) for j in range(n) if i != j]
    if not pairs:
        print("\none GPU: no GPU pairs to measure (allocate two GPUs for the GPU-to-GPU paths)")
    print(f"\n{'pair':>6} {'peer':>5} {'direct':>16} {'host':>16} {'host_pipe':>16}")
    for i, j in pairs:
        src = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{i}").random_(0, 255)
        dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{j}")
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        peer = torch.cuda.can_device_access_peer(i, j)
        t_direct = timed(lambda: dst.copy_(src, non_blocking=True), a.reps)

        def via_host():
            host.copy_(src, non_blocking=True)
            torch.cuda.synchronize(i)
            dst.copy_(host, non_blocking=True)
        t_host = timed(via_host, a.reps)
        s_out = torch.cuda.Stream(device=f"cuda:{i}")
        s_in = torch.cuda.Stream(device=f"cuda:{j}")

        def via_host_pipe():
            evs = []
            for o in range(0, nbytes, chunk):
                with torch.cuda.stream(s_out):
                    host[o:o + chunk].copy_(src[o:o + chunk], non_blocking=True)
                    e = torch.cuda.Event()
                    e.record(s_out)
                evs.append((o, e))
            for o, e in evs:
                e.synchronize()
                with torch.cuda.stream(s_in):
                    dst[o:o + chunk].copy_(host[o:o + chunk], non_blocking=True)
            s_in.synchronize()
        t_pipe = timed(via_host_pipe, a.reps)
        ok = bool(torch.equal(dst.cpu()[:1 << 20], src.cpu()[:1 << 20]))
        row = {"src": i, "dst": j, "peer_access": peer, "correct": ok}
        cells = []
        for k, t in (("direct", t_direct), ("host", t_host), ("host_pipe", t_pipe)):
            gbs = nbytes / t / 1e9
            row[k] = {"seconds": t, "GB_per_s": gbs, "Gb_per_s": 8 * gbs}
            cells.append(f"{gbs:6.1f} GB/s {8 * gbs:4.0f}Gb")
        res["pairs"].append(row)
        print(f"{i:>3}->{j:<2} {str(peer):>5} " + " ".join(f"{c:>16}" for c in cells))
        del src, dst, host
        torch.cuda.empty_cache()
    if res["pairs"]:
        best = max(max(r["direct"]["GB_per_s"], r["host_pipe"]["GB_per_s"]) for r in res["pairs"])
        worst = min(max(r["direct"]["GB_per_s"], r["host_pipe"]["GB_per_s"]) for r in res["pairs"])
        what = "pair"
    else:                                             # one GPU: the slower pinned leg bounds a transfer through host
        legs = [min(r["d2h_pinned"]["GB_per_s"], r["h2d_pinned"]["GB_per_s"]) for r in res["host_legs"]]
        best, worst, what = max(legs), min(legs), "pinned host leg"
    print(f"\nOne 8K document of Llama-3.2-3B KV, wire time at the slowest / fastest measured {what} "
          f"({worst:.1f} / {best:.1f} GB/s):")
    for k, mib in DOC_MIB.items():
        b = mib * (1 << 20)
        print(f"  {k:32s} {mib:5d} MiB  {1e3 * b / worst / 1e9:7.1f} ms / {1e3 * b / best / 1e9:7.1f} ms")
    res["document_wire_ms"] = {k: {"slowest_pair": 1e3 * m * (1 << 20) / worst / 1e9,
                                   "fastest_pair": 1e3 * m * (1 << 20) / best / 1e9} for k, m in DOC_MIB.items()}
    if a.output:
        with open(a.output, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
