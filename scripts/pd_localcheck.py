#!/usr/bin/env python3
"""Did D answer as one engine would on its own? Each traced turn's exact prompt (the token ids the MTRAG driver sent,
mtrag.jsonl, D's earlier answers in it) goes to one plain engine (no P/D, no connector, its own prefill) at
temperature 0 with D's answer length; its answer must be D's, token for token. Greedy decoding of one prompt at a
time is deterministic on one GPU, so a different answer means D's KV was not the KV of its prompt: what P computed
and NIXL (or LMCache) delivered differs from what this engine computes itself.

  python scripts/pd_localcheck.py --mtrag RUN/mtrag.jsonl --engine http://127.0.0.1:PORT --output RUN/local_check.json

Writes {"turns": n, "same": n, "rows": [{tag, conv, turn, prompt_tokens, d_tokens, local_tokens, same,
first_diff, d_text, local_text}], ...} and prints one line a turn. Exit 0 either way (the report reads the file).
"""

import argparse
import json
import sys
import time
import urllib.request


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtrag", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="llama")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    recs = [json.loads(ln) for ln in open(a.mtrag) if ln.strip()]
    rows = []
    for r in recs:
        ids, d_ids = r.get("ids"), r.get("answer_ids")
        if not ids or d_ids is None or r.get("error"):
            continue
        t0 = time.time()
        out = post(a.engine.rstrip("/") + "/v1/completions",
                   {"model": a.model, "prompt": ids, "max_tokens": max(1, len(d_ids)), "temperature": 0,
                    "return_token_ids": True}, a.timeout)
        ch = (out.get("choices") or [{}])[0]
        local = ch.get("token_ids")
        if local is None:                                  # no ids from the engine: compare the texts only
            local_text = ch.get("text") or ""
            same, fd = local_text == (r.get("answer_text") or ""), None
        else:
            local_text = ch.get("text") or ""
            fd = first_diff(d_ids, local)
            same = fd is None
        rows.append({"tag": r.get("tag"), "conv": r.get("conv"), "turn": r.get("turn"),
                     "prompt_tokens": len(ids), "d_tokens": len(d_ids), "local_tokens": len(local or []),
                     "same": same, "first_diff": fd, "local_s": round(time.time() - t0, 3),
                     "d_text": (r.get("answer_text") or "")[:400], "local_text": local_text[:400]})
        print(f"   turn {r.get('conv')}/{r.get('turn')}: {r.get('prompt_tokens')} prompt tokens, D's answer "
              + ("the same" if same else f"differs from token {fd}") + f" ({len(d_ids)} tokens)", flush=True)
    res = {"turns": len(rows), "same": sum(1 for x in rows if x["same"]),
           "first_diff_turn": next((f"{x['conv']}/{x['turn']}" for x in rows if not x["same"]), None),
           "engine": "one plain engine (no P/D, no KV connector), temperature 0", "rows": rows}
    json.dump(res, open(a.output, "w"), indent=1)
    print(f"   local check: {res['same']} of {res['turns']} answers the same as one engine's own", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
