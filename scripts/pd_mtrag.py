#!/usr/bin/env python3
"""MTRAG through the PD proxy: IBM's multi-turn RAG conversations replayed turn by turn, each turn's prompt built
from that turn's passages, the conversation so far (with the model's own earlier answers) and the new question.

  python3 -I scripts/pd_mtrag.py --conversations mtrag/conversations.json --proxy http://127.0.0.1:8000 \
      --tokenize http://127.0.0.1:8201 --model llama --layout docs_first --policy conversation \
      --concurrency 8 --max-tokens 128 [--trace] --output runs/mtrag_x.jsonl --summary runs/mtrag_x.json

The data (MTRAG human, 110 conversations, 7.7 turns on average; github.com/IBM/mt-rag-benchmark, Apache-2.0) is
read as data only: the passages of a turn are those on its agent message ("contexts"), the questions the user
messages. Its own agent answers are not used: the next turn's history holds the answer this model generated, as a
chat system would send it.

Layouts (where a turn's passages go):
  docs_first     system, the turn's passages, the conversation so far, the question (the usual RAG prompt; the
                 passages change from turn to turn, so the history after them is never a cached prefix)
  history_first  system, the conversation so far, the turn's passages, the question
  interleaved    system, then every turn so far as passages, question, answer, then this turn's passages and
                 question: each prompt extends the previous one (append only)

Prompts go as token ids, built from separately tokenized pieces (the engine's /tokenize), so a passage has the same
tokens wherever it appears and the driver knows every piece's place: each record lists them (system, doc <id>,
user <turn>, answer <turn>, ...). A generated answer goes into the history as the engine's own token ids
(return_token_ids), or else its text tokenized again (the record says which). Conversations run `--concurrency` at
a time, each one turn after another; a turn waits for its whole answer.
Standard library and aiohttp only (the vLLM environment), so it runs with python -I.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time

SYSTEM = ("You are a helpful assistant. Answer the user's last question from the documents. If they do not hold "
          "the answer, say so.\n\n")


def turns_of(conv):
    """[(question, [(document id, title, text)])] of one MTRAG conversation: each user message with the passages of
    the agent message that answers it."""
    out, ms = [], conv["messages"]
    for i, m in enumerate(ms):
        if m.get("speaker") != "user":
            continue
        ctx = []
        if i + 1 < len(ms) and ms[i + 1].get("speaker") == "agent":
            for c in ms[i + 1].get("contexts") or []:
                ctx.append((str(c.get("document_id")), str(c.get("title") or ""), str(c.get("text") or "")))
        out.append((str(m.get("text") or ""), ctx))
    return out


class Tok:
    """Token ids of text pieces from the engine's /tokenize, each piece once."""

    def __init__(self, session, url, model):
        self.s, self.url, self.model, self.cache = session, url.rstrip("/"), model, {}

    async def __call__(self, text, special=False):
        k = (text, special)
        if k not in self.cache:
            async with self.s.post(self.url + "/tokenize", json={"model": self.model, "prompt": text,
                                                                 "add_special_tokens": special}) as r:
                r.raise_for_status()
                self.cache[k] = list((await r.json())["tokens"])
        return self.cache[k]


async def build(tok, layout, turns, t, answers):
    """(token ids, pieces [(kind, ref, start, end)]) of the prompt for turn t (0-based); answers[i]: the token ids
    of the answer to turn i."""
    ids, pieces = [], []

    async def add(kind, ref, toks):
        pieces.append((kind, ref, len(ids), len(ids) + len(toks)))
        ids.extend(toks)

    async def docs(i):
        for doc_id, title, text in turns[i][1]:
            await add("doc", doc_id, await tok(f"Document: {title}\n{text}\n\n" if title else f"Document:\n{text}\n\n"))

    async def user(i):
        await add("user", i, await tok(f"User: {turns[i][0]}\n"))

    async def answer(i):
        await add("assistant", i, await tok("Assistant:"))
        await add("answer", i, answers[i])
        await add("newline", i, await tok("\n"))
    await add("system", None, await tok(SYSTEM, special=True))
    if layout == "docs_first":
        await docs(t)
        for i in range(t):
            await user(i)
            await answer(i)
        await user(t)
    elif layout == "history_first":
        for i in range(t):
            await user(i)
            await answer(i)
        await docs(t)
        await user(t)
    elif layout == "interleaved":
        for i in range(t):
            await docs(i)
            await user(i)
            await answer(i)
        await docs(t)
        await user(t)
    else:
        raise ValueError(f"layout {layout}")
    await add("assistant", t, await tok("Assistant:"))
    return ids, pieces


async def ask(session, a, ids, conv, tag, trace, pieces=None, capture=False):
    """Stream one completion: (answer token ids or None, its text, TTFT s, latency s)."""
    body = {"model": a.model, "prompt": ids, "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
            "return_token_ids": True, "conversation_id": conv, "t_tag": tag}
    if trace:
        body["t_trace"] = True
        if pieces:                                   # the prefill engine's relation probe groups by them
            body["t_pieces"] = [[k, s, e] for k, _r, s, e in pieces]
        if capture:                                  # the engines store this request's raw pages
            body["t_capture"] = True
    t0 = time.time()
    ttft, text, toks = None, [], []
    async with session.post(a.proxy.rstrip("/") + "/v1/completions", json=body) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
        buf = b""
        async for chunk in r.content.iter_any():
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                msg = json.loads(data)
                for c in msg.get("choices") or []:
                    if ttft is None and (c.get("text") or c.get("token_ids")):
                        ttft = time.time() - t0
                    text.append(c.get("text") or "")
                    if c.get("token_ids") is not None:
                        toks.extend(c["token_ids"])
    return (toks or None), "".join(text), ttft, time.time() - t0


async def conversation(session, tok, a, k, conv, out, trace):
    turns = turns_of(conv)
    cid = str(conv.get("conversation_id") or conv.get("id") or k)
    answers = []
    for t in range(len(turns)):
        ids, pieces = await build(tok, a.layout, turns, t, answers)
        rec = {"tag": f"{run_of(a)}|{k}|{t}", "run": run_of(a), "layout": a.layout, "policy": a.policy, "conv": k,
               "conv_id": cid, "domain": conv.get("domain"), "turn": t, "turns": len(turns),
               "prompt_tokens": len(ids), "pieces": pieces, "trace": trace}
        if a.max_len and len(ids) + a.max_tokens > a.max_len:
            out.append(dict(rec, error=f"over the model's length ({len(ids)} + {a.max_tokens} > {a.max_len})"))
            break
        if trace:
            rec["ids"] = ids
        try:
            capture = trace and k < a.capture_convs
            rec["capture"] = capture
            got, text, ttft, lat = await ask(session, a, ids, f"{run_of(a)}:{k}", rec["tag"], trace, pieces, capture)
            src = "engine"
            if got is None:
                got, src = await tok(text), "retokenized"
            answers.append(got)
            out.append(dict(rec, answer_tokens=len(got), answer_ids_from=src, ttft=ttft, latency=lat, t_done=time.time()))
        except Exception as exc:
            out.append(dict(rec, error=repr(exc)[:300]))
            break


def run_of(a):
    """The run's name: in every request's tag and conversation id (the prefill engine's relation probe keeps each
    run's requests apart by it)."""
    return a.run or f"{a.layout}_{a.policy}"


def p50(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


async def main_async(a):
    import aiohttp
    convs = json.load(open(a.conversations))
    if a.limit:
        convs = convs[:a.limit]
    out = []
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tok = Tok(session, a.tokenize, a.model)
        sem = asyncio.Semaphore(a.concurrency)

        async def one(k, c):
            async with sem:
                await conversation(session, tok, a, k, c, out, a.trace)
        await asyncio.gather(*(one(k, c) for k, c in enumerate(convs)))
    dur = time.time() - t0
    with open(a.output, "w") as f:
        for r in sorted(out, key=lambda r: (r["conv"], r["turn"])):
            f.write(json.dumps(r) + "\n")
    ok = [r for r in out if not r.get("error")]
    summ = {"run": run_of(a), "layout": a.layout, "policy": a.policy, "conversations": len(convs), "requests": len(ok),
            "errors": len(out) - len(ok), "duration_s": dur, "ttft_p50_ms": None if p50([r["ttft"] for r in ok]) is None
            else 1e3 * p50([r["ttft"] for r in ok]), "latency_p50_s": p50([r["latency"] for r in ok]),
            "prompt_tokens_mean": statistics.mean([r["prompt_tokens"] for r in out]) if out else None,
            "prompt_tokens_max": max([r["prompt_tokens"] for r in out], default=None),
            "answer_tokens_mean": statistics.mean([r["answer_tokens"] for r in ok]) if ok else None,
            "answer_ids_from": {s: sum(1 for r in ok if r["answer_ids_from"] == s) for s in ("engine", "retokenized")},
            "first_errors": [r["error"] for r in out if r.get("error")][:5]}
    with open(a.summary, "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--proxy", required=True)
    ap.add_argument("--tokenize", required=True, help="an engine's URL (its /tokenize)")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--layout", default="docs_first", choices=["docs_first", "history_first", "interleaved"])
    ap.add_argument("--policy", default="conversation", help="the proxy's routing (a label here)")
    ap.add_argument("--run", default="", help="the run's name (default layout_policy)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=0, help="the engines' model length (0: no check)")
    ap.add_argument("--limit", type=int, default=0, help="the first N conversations (0: all)")
    ap.add_argument("--request-timeout", type=float, default=600)
    ap.add_argument("--trace", action="store_true", help="the prefill engine hashes every page (and ids are kept)")
    ap.add_argument("--capture-convs", type=int, default=0,
                    help="with --trace: the first N conversations' raw pages are stored (the engines' capture_dir)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    a = ap.parse_args(argv)
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
