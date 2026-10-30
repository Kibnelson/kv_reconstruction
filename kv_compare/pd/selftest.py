"""Self-test of the transfer codecs and the paced transport on real captured KV, without vLLM (run it in the vLLM
environment before starting the engines: same torch and Triton as the connector).

  python -m kv_compare.pd.selftest --codecs $RUN/pd_codecs.pt --captures $RUN/captures --doc ds_wikitext2_r0 \
      --tokens 8192,3000 --splitzip-source $KV_HOME/sources/splitzip --adacode-source $KV_HOME/sources/adacode

Per codec and token count: bit-exact through bundle -> paced transport (ring and tcp, loopback) -> layer sink;
bits per value; encode and decode times. Exit status 1 when any loaded codec is not exact.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--codecs", required=True)
    ap.add_argument("--captures", required=True)
    ap.add_argument("--doc", default="ds_wikitext2_r0")
    ap.add_argument("--tokens", default="8192,3000")
    ap.add_argument("--names", default="raw,ours-dense,ours-kv,ours-4g,ours-hbh,ours-rank256,splitzip,adacode")
    ap.add_argument("--splitzip-source", default=None)
    ap.add_argument("--adacode-source", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gbps", type=float, default=100.0, help="link speed of the transport check")
    ap.add_argument("--output", default=None, help="JSON results")
    a = ap.parse_args(argv)
    from ..predict import Doc
    from . import codecs as cd
    from .transport import Link, Receiver, Sender
    from .wire import bundle, payload_bytes, unbundle, word_checksum
    dev = torch.device(a.device)
    sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
    built, errors, meta = cd.build(a.codecs, dev, a.names.split(","), a.splitzip_source, a.adacode_source)
    doc = Doc(a.captures, a.doc)
    rows, ok_all = [], True
    for T in [int(t) for t in a.tokens.split(",")]:
        T = min(T, doc.tokens)
        words = torch.empty(doc.layers, T, 2 * doc.columns, dtype=torch.int16, device=dev)
        for l in range(doc.layers):
            x = doc.words(l, dev, rows=T)
            words[l] = (x - 65536 * (x >= 32768).to(torch.int32)).to(torch.int16)
        want = word_checksum(words)
        for name, c in built.items():
            row = {"codec": name, "tokens": T}
            try:
                c.encode(words)                      # compile
                sync()
                t = time.perf_counter()
                parts, m = c.encode(words)
                sync()
                row["encode_ms"] = 1e3 * (time.perf_counter() - t)
                header, blob = bundle(parts, m)
                row["bits"] = 8.0 * payload_bytes(header) / words.numel()
                back = torch.empty_like(words)
                p2, m2 = unbundle(header, blob)
                c.decode(p2, m2, lambda l, w: back[l].copy_(w))
                sync()
                t = time.perf_counter()
                c.decode(p2, m2, lambda l, w: back[l].copy_(w))
                sync()
                row["decode_ms"] = 1e3 * (time.perf_counter() - t)
                row["exact"] = bool(torch.equal(back, words)) and word_checksum(back) == want
                ok_all = ok_all and row["exact"]
            except Exception as exc:
                row["error"] = repr(exc)[:300]
                ok_all = False
            rows.append(row)
            print(json.dumps(row), flush=True)
    # the transport: one 8K raw transfer and one coded, ring and tcp, at --gbps
    tr_rows = []
    if built:
        name = next((n for n in ("ours-dense", "raw") if n in built), next(iter(built)))
        parts, m = built[name].encode(words)
        header, blob = bundle(parts, m)
        for mode in ("ring", "tcp"):
            rx = Receiver("127.0.0.1", 0, dev, ring_bytes=256 << 20)
            tx = Sender(Link(gbps=a.gbps), dev, mode=mode)
            recs = []
            t = time.perf_counter()
            tx.send(f"127.0.0.1:{rx.port}", "selftest", header, blob, on_done=recs.append)
            got = rx.wait("selftest", 600)
            wall = time.perf_counter() - t
            ok = got is not None
            if ok:
                back = torch.empty_like(words)
                p2, m2 = unbundle(got[0], got[1] if got[1].device == dev else got[1].to(dev))
                built[name].decode(p2, m2, lambda l, w: back[l].copy_(w))
                ok = bool(torch.equal(back, words))
            r = {"transport": mode, "codec": name, "bytes": int(blob.numel()), "gbps_set": a.gbps, "exact": ok,
                 "wall_s": wall, "achieved_gbps": 8 * blob.numel() / wall / 1e9,
                 "link_s": recs[0].get("link_s") if recs else None}
            ok_all = ok_all and ok
            tr_rows.append(r)
            print(json.dumps(r), flush=True)
            tx.close()
            rx.close()
    out = {"codecs": rows, "transport": tr_rows, "codec_errors": errors, "meta": meta, "ok": ok_all,
           "torch": torch.__version__}
    if a.output:
        with open(a.output, "w") as f:
            json.dump(out, f, indent=1)
    print("SELFTEST " + ("OK" if ok_all else "FAILED") + f" ({len(rows)} codec runs, errors: {errors or 'none'})")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
