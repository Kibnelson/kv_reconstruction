#!/usr/bin/env python3
"""The KV pool experiment (docs/lmcache_lifecycle.md, "The KV pool experiment"): a frozen fixture of six three-turn
chat conversations (workloads/kv-pool-concurrency-chat-v1.json, Qwen3-4B's tokens), offered n at a time to 1 prefill
and 3 decode engines with small GPU pools, every request on one D engine; then what the trace shows: how many
requests overlapped on each engine, why the others waited (GPU capacity, a KV load in flight, an LMCache lookup, the
request limit), the pool's slots (in use, cached, empty; P's held for NIXL), what LMCache restored, and NIXL per
request.

  python3 scripts/lmc_pool.py check  --fixture F [--tokenizer MODEL_DIR] [--output check.json]
        every prompt: its length, its hash, its blocks and chunks, the prefixes the turns share and the
        conversations do not; with --tokenizer, the model's tokenizer.json (its hash) and every prompt's text
        tokenized again, the same ids (exit 1 otherwise)
  python3 scripts/lmc_pool.py run    --fixture F --count n --proxy URL --run NAME [--decode 0] [--stagger-ms 5]
                                     [--max-tokens 8] --output RUN/mtrag.jsonl --summary RUN/mtrag_summary.json
        the driver (scripts/lmc_run.sh, LMC_WORKLOAD=pool): the first n conversations in the fixture's order; turn k
        of every one sent together (stagger-ms apart, in that order), turn k+1 when all of turn k are done; each
        request names the D engine (--decode, 0: D1) and its stage's id (<key>.<run>.<conversation>.T<k>.P / .D);
        records in mtrag.jsonl's format, with the reference answer checked
  python3 scripts/lmc_pool.py report --run RUN_DIR | --results RESULTS_DIR  --output DIR
        pool.json and pool.md (and pool.md put first in DIR/summary.md when there is one)
  python3 scripts/lmc_pool.py compare --dirs D1,D2,... --output DIR
        every arm's pool.json side by side: pool_compare.md, pool_compare.json
  python3 scripts/lmc_pool.py kv-bytes --config MODEL_DIR/config.json
        the KV bytes a token takes (K and V, every layer)

report, compare and check (without --tokenizer) need only the standard library.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
ENGINES = ("P", "D1", "D2", "D3")
WAIT_ORDER = ("no_free_blocks", "queued_behind_no_free_blocks", "kv_load_in_flight", "connector_lookup_pending",
              "queued_behind_connector_lookup_pending", "max_num_seqs", "queued_behind_max_num_seqs", "token_budget",
              "not_reached")
WAIT_NAMES = {
    "no_free_blocks": "GPU capacity (no free blocks for it)",
    "queued_behind_no_free_blocks": "GPU capacity (behind a request without blocks)",
    "kv_load_in_flight": "its KV load in flight (NIXL's read, or LMCache's)",
    "connector_lookup_pending": "an LMCache lookup pending",
    "queued_behind_connector_lookup_pending": "behind a pending LMCache lookup",
    "max_num_seqs": "the request limit (max-num-seqs)",
    "queued_behind_max_num_seqs": "the request limit (behind)",
    "token_budget": "the step's token budget",
    "not_reached": "not reached this step",
}


def lifecycle():
    """scripts/lmc_lifecycle.py, beside this script (its loading, keys and unpacking)."""
    spec = importlib.util.spec_from_file_location("lmc_lifecycle_for_pool", HERE / "lmc_lifecycle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ids_sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def common(a, b):
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


def q50(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def ms(x):
    return None if x is None else round(1000 * x, 1)


# ----------------------------------------------------------------------------------------------- the fixture
def turns_of(fx, conv):
    return conv["layouts"][fx.get("layout", "chat")]


def check_fixture(fx, tokenizer=None):
    """The fixture's own claims, checked: (result dict, problems list)."""
    problems = []
    B, C = int(fx["block_size"]), int(fx["chunk_size"])
    out_tok, want, maxlen = int(fx["decode_output_tokens"]), int(fx["prompt_tokens"]), int(fx["max_model_len"])
    convs = fx["conversations"]
    keys, first_blocks, stable, rows = set(), {}, {}, 0
    blocks_per_request = set()
    for conv in convs:
        cid = conv["conversation_id"]
        ts = turns_of(fx, conv)
        if [t["turn"] for t in ts] != list(range(1, len(ts) + 1)):
            problems.append(f"{cid}: turns not 1..{len(ts)} in order")
        prev = None
        for t in ts:
            rows += 1
            ids, where = t["token_ids"], f"{cid} turn {t['turn']}"
            n = len(ids)
            if n != t["prompt_tokens"] or n != want:
                problems.append(f"{where}: {n} ids, the fixture says {t['prompt_tokens']} (and {want} for all)")
            if ids_sha(ids) != t["prompt_sha256"]:
                problems.append(f"{where}: the ids' SHA-256 is not prompt_sha256")
            if n // C * C != t.get("measured_prefix_tokens", n // C * C) or n % C != t.get("tail_tokens", n % C):
                problems.append(f"{where}: {n // C} chunks of {C} and {n % C} tail tokens, not the fixture's")
            if n + out_tok > maxlen:
                problems.append(f"{where}: {n} + {out_tok} tokens is over max_model_len {maxlen}")
            blocks_per_request.add(math.ceil((n + out_tok) / B))
            if t["request_key"] in keys:
                problems.append(f"{where}: request_key {t['request_key']} twice")
            keys.add(t["request_key"])
            if prev is not None:
                c = common(prev, ids)
                if t.get("common_prefix_tokens_with_previous_turn") not in (None, c):
                    problems.append(f"{where}: shares {c} tokens with the turn before, the fixture says "
                                    f"{t['common_prefix_tokens_with_previous_turn']}")
            prev = ids
        all_ids = [t["token_ids"] for t in ts]
        stable[cid] = min((common(all_ids[0], x) for x in all_ids[1:]), default=len(all_ids[0]))
        firsts = {tuple(x[:B]) for x in all_ids}
        if len(firsts) != 1:
            problems.append(f"{cid}: its turns start with different first blocks")
        for fb in firsts:
            first_blocks.setdefault(fb, set()).add(cid)
    shared = [sorted(v) for v in first_blocks.values() if len(v) > 1]
    if shared:
        problems.append(f"conversations sharing a first block: {shared}")
    cross = max((common(a["token_ids"], b["token_ids"]) for i, x in enumerate(convs) for y in convs[i + 1:]
                 for a in turns_of(fx, x) for b in turns_of(fx, y)), default=0)
    res = {"conversations": len(convs), "prompts": rows, "prompt_tokens": want, "block_size": B, "chunk_size": C,
           "blocks_per_request": sorted(blocks_per_request), "stable_prefix_tokens": stable,
           "stable_prefix_blocks": {k: v // B for k, v in stable.items()},
           "stable_prefix_chunks": {k: v // C for k, v in stable.items()},
           "longest_prefix_between_conversations": cross, "model": fx.get("model"),
           "model_revision": fx.get("model_revision")}
    if tokenizer is not None:
        tdir = pathlib.Path(tokenizer)
        tj = tdir / "tokenizer.json"
        tres = {"dir": str(tdir)}
        if not tj.is_file():
            problems.append(f"no tokenizer.json in {tdir}")
        else:
            tres["sha256"] = hashlib.sha256(tj.read_bytes()).hexdigest()
            tres["sha256_matches"] = tres["sha256"] == fx.get("tokenizer_sha256")
            if not tres["sha256_matches"]:
                problems.append(f"tokenizer.json's SHA-256 {tres['sha256'][:16]}... is not the fixture's "
                                f"{str(fx.get('tokenizer_sha256'))[:16]}...")
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                problems.append(f"no tokenizers package to tokenize the prompts again: {exc}")
            else:
                tok = Tokenizer.from_file(str(tj))
                same = 0
                for conv in convs:
                    for t in turns_of(fx, conv):
                        got = tok.encode(t["prompt_text"], add_special_tokens=False).ids
                        if got == t["token_ids"]:
                            same += 1
                        else:
                            problems.append(f"{t['request_key']}: the text tokenizes to {len(got)} ids, other than "
                                            f"the fixture's (first difference at {common(got, t['token_ids'])})")
                tres["prompts_tokenized_the_same"] = same
        res["tokenizer"] = tres
    res["ok"], res["problems"] = not problems, problems
    return res, problems


def cmd_check(a):
    fx = json.load(open(a.fixture))
    res, problems = check_fixture(fx, a.tokenizer)
    res["fixture"] = a.fixture
    res["fixture_sha256"] = hashlib.sha256(pathlib.Path(a.fixture).read_bytes()).hexdigest()
    if a.output:
        json.dump(res, open(a.output, "w"), indent=1)
    tk = res.get("tokenizer") or {}
    print(f"   fixture {a.fixture}: {res['conversations']} conversations, {res['prompts']} prompts of "
          f"{res['prompt_tokens']} tokens, {res['blocks_per_request']} blocks a request with its output, stable "
          f"prefixes {sorted(set(res['stable_prefix_tokens'].values()))} tokens, longest between conversations "
          f"{res['longest_prefix_between_conversations']}"
          + (f"; tokenizer.json {'matches' if tk.get('sha256_matches') else 'DIFFERS'}, "
             f"{tk.get('prompts_tokenized_the_same', 0)} of {res['prompts']} prompts tokenized the same" if tk else ""))
    for p in problems:
        print(f"   PROBLEM: {p}")
    return 0 if not problems else 1


def cmd_kv_bytes(a):
    c = json.load(open(a.config))
    c = c.get("text_config", c)
    heads = c["num_attention_heads"]
    dim = c.get("head_dim") or c["hidden_size"] // heads
    size = {"float32": 4}.get(str(c.get("torch_dtype") or c.get("dtype") or "bfloat16"), 2)
    print(2 * c["num_hidden_layers"] * c.get("num_key_value_heads", heads) * dim * size)
    return 0


# ----------------------------------------------------------------------------------------------- the driver
def label_of(run, cid, turn):
    return f"{run}.{cid}.T{turn}"


async def ask(session, a, body):
    """Stream one completion through the proxy: (token ids, text, t_first, t_done)."""
    t_first, text, toks = None, [], []
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
                    if t_first is None and (c.get("text") or c.get("token_ids")):
                        t_first = time.time()
                    text.append(c.get("text") or "")
                    if c.get("token_ids") is not None:
                        toks.extend(c["token_ids"])
    return toks, "".join(text), t_first, time.time()


async def one(session, a, fx, k, conv, row, turn0, n_turns):
    cid = conv["conversation_id"]
    label = label_of(a.run, cid, row["turn"])
    tag = f"{a.run}|{k}|{turn0}"
    rec = {"tag": tag, "run": a.run, "layout": fx.get("layout", "chat"), "policy": f"pinned:D{a.decode + 1}",
           "conv": k, "conv_id": cid, "domain": "synthetic", "docs": "fixture", "turn": turn0, "turns": n_turns,
           "prompt_tokens": len(row["token_ids"]), "pieces": [], "trace": True, "ids": row["token_ids"],
           "request_key": row["request_key"], "label": label, "wave": turn0, "order": k,
           "prompt_sha256_ok": ids_sha(row["token_ids"]) == row["prompt_sha256"],
           "reference_answer": row.get("reference_answer")}
    body = {"model": a.model, "prompt": row["token_ids"], "max_tokens": a.max_tokens, "temperature": 0,
            "ignore_eos": True, "stream": True, "return_token_ids": True, "conversation_id": f"{a.run}:{cid}",
            "t_tag": tag, "t_decode": a.decode, "t_rid": label}
    rec["t_send"] = time.time()
    try:
        toks, text, t_first, t_done = await ask(session, a, body)
    except Exception as exc:
        rec.update(error=repr(exc)[:300], t_done=time.time())
        return rec
    ref = (row.get("reference_answer") or "").strip().lower()
    rec.update(answer_ids=toks, answer_text=text, answer_tokens=len(toks), answer_ids_from="engine",
               ttft=None if t_first is None else t_first - rec["t_send"], latency=t_done - rec["t_send"],
               t_first=t_first, t_done=t_done, output_sha256=ids_sha(toks),
               answer_ok=bool(ref) and ref in text.lower())
    return rec


async def drive(a):
    import asyncio
    import aiohttp
    fx = json.load(open(a.fixture))
    res, problems = check_fixture(fx)
    if problems:
        raise SystemExit("STOP: the fixture: " + "; ".join(problems[:3]))
    convs = fx["conversations"][:a.count]
    if len(convs) < a.count:
        raise SystemExit(f"STOP: the fixture has {len(fx['conversations'])} conversations, not {a.count}")
    n_turns = max(len(turns_of(fx, c)) for c in convs)
    out, waves = [], []
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        for t in range(n_turns):
            ws = time.time()
            tasks = []
            for k, conv in enumerate(convs):
                ts = turns_of(fx, conv)
                if t >= len(ts):
                    continue
                tasks.append(asyncio.create_task(one(session, a, fx, k, conv, ts[t], t, n_turns)))
                if a.stagger_ms and k < len(convs) - 1:
                    await asyncio.sleep(a.stagger_ms / 1000)
                else:
                    await asyncio.sleep(0)            # the task starts (and sends) before the next is made
            got = await asyncio.gather(*tasks)
            out.extend(got)
            waves.append({"turn": t + 1, "requests": len(got), "t_start": ws, "t_end": time.time(),
                          "s": round(time.time() - ws, 3), "errors": sum(1 for r in got if r.get("error"))})
            print(f"   turn {t + 1}: {len(got)} requests together, {waves[-1]['s']:.2f} s, "
                  f"{sum(1 for r in got if r.get('answer_ok'))} answers with the reference, "
                  f"{waves[-1]['errors']} errors", flush=True)
    with open(a.output, "w") as f:
        for r in sorted(out, key=lambda r: (r["turn"], r["conv"])):
            f.write(json.dumps(r) + "\n")
    ok = [r for r in out if not r.get("error")]
    summ = {"run": a.run, "workload": "pool", "fixture": a.fixture, "count": len(convs), "turns": n_turns,
            "decode": f"D{a.decode + 1}", "stagger_ms": a.stagger_ms, "requests": len(ok),
            "errors": len(out) - len(ok), "answers_ok": sum(1 for r in ok if r.get("answer_ok")),
            "duration_s": time.time() - t0, "waves": waves,
            "ttft_p50_ms": ms(q50([r.get("ttft") for r in ok])), "latency_p50_s": q50([r.get("latency") for r in ok]),
            "conversations": [c["conversation_id"] for c in convs], "layout": fx.get("layout"),
            "policy": f"pinned:D{a.decode + 1}", "prompt_tokens_mean": fx.get("prompt_tokens"),
            "answer_ids_from": {"engine": len(ok), "retokenized": 0},
            "first_errors": [r["error"] for r in out if r.get("error")][:5]}
    with open(a.summary, "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: summ[k] for k in ("run", "count", "requests", "errors", "answers_ok", "ttft_p50_ms")}))
    return 0 if ok and len(ok) == len(out) else 1


def cmd_run(a):
    import asyncio
    return asyncio.run(drive(a))


# ----------------------------------------------------------------------------------------------- the report
class Pool:
    """One arm's records: the trace (kvtrace/), the proxy's (proxy.jsonl), the driver's (mtrag.jsonl), env.json."""

    def __init__(self, run):
        L = lifecycle()
        self.L = L
        self.dir = pathlib.Path(run)
        self.env = json.load(open(self.dir / "env.json")) if (self.dir / "env.json").is_file() else {}
        recs, self.load_info = L.load(self.dir)
        self.recs = recs
        self.t_end = recs[-1]["t"] if recs else 0.0
        self.proxy = {r["key"]: r for r in L.read_jsonl(self.dir / "proxy.jsonl") if r.get("key")}
        self.driver = {r["tag"]: r for r in L.read_jsonl(self.dir / "mtrag.jsonl") if r.get("tag")}
        self.summary = (json.load(open(self.dir / "mtrag_summary.json"))
                        if (self.dir / "mtrag_summary.json").is_file() else {})
        self.by = collections.defaultdict(list)                          # (role, ev) -> records
        for r in recs:
            self.by[(r.get("role"), r["ev"])].append(r)
        kvb = self.env.get("kv_bytes_per_token")
        self.block_bytes = 16 * int(kvb) if kvb else None

    def ev(self, role, name):
        return self.by.get((role, name), [])

    def who(self, rid):
        """A request id's key, tag and driver record."""
        key = self.L.rkey(rid)
        px = self.proxy.get(key) or {}
        return key, px.get("tag"), self.driver.get(px.get("tag")) or {}

    def label(self, rid):
        _key, tag, d = self.who(rid)
        if d:
            return f"{d.get('conv_id')} T{d.get('turn', 0) + 1}"
        return tag or str(rid)[:24]

    # ------------------------------------------------------------------------- the scheduler's steps
    def steps(self, role):
        return sorted(self.ev(role, "vllm.sched.step"), key=lambda r: r["t"])

    def engine(self, role):
        st = self.steps(role)
        out = {"steps_recorded": len(st)}
        if not st:
            return out
        span = []
        for i, s in enumerate(st):
            t1 = st[i + 1]["t"] if i + 1 < len(st) else max(s["t"], self.t_end)
            span.append((s, max(0.0, t1 - s["t"])))
        waits = collections.defaultdict(lambda: collections.defaultdict(float))    # rid -> reason -> s
        resident_peak, running_peak, waiting_peak, batch_peak, held_peak = 0, 0, 0, 0, 0
        in_use_peak, free_min, cached_peak = 0, None, 0
        pool_rows = []
        for s, dt in span:
            loading = [w[0] for w in s.get("waiting") or [] if w[1] == "WAITING_FOR_REMOTE_KVS"]
            resident = set(s.get("running") or []) | set(loading) | set(s.get("held") or [])
            resident_peak = max(resident_peak, len(resident))
            running_peak = max(running_peak, len(s.get("running") or []))
            waiting_peak = max(waiting_peak, len([w for w in s.get("waiting") or []
                                                 if w[1] != "WAITING_FOR_REMOTE_KVS"]))
            batch_peak = max(batch_peak, len([1 for v in (s.get("scheduled") or {}).values() if v > 0]))
            held_peak = max(held_peak, len(s.get("held") or []))
            p = s.get("pool") or {}
            if p:
                in_use_peak = max(in_use_peak, p.get("in_use", 0))
                free_min = p.get("free") if free_min is None else min(free_min, p.get("free", free_min))
                cached_peak = max(cached_peak, p.get("cached_free") or 0)
                pool_rows.append([round(s["t"], 4), p.get("in_use"), p.get("cached_free"), p.get("empty"),
                                  len(resident), len(s.get("held") or [])])
            for rid, _status, w in s.get("waiting") or []:
                waits[rid][w.get("reason")] += dt
        by_reason = collections.defaultdict(float)
        reqs_by_reason = collections.Counter()
        for rid, rs in waits.items():
            for reason, sec in rs.items():
                by_reason[reason] += sec
                reqs_by_reason[reason] += 1
        pool0 = (st[0].get("pool") or {})
        out.update(pool_total=pool0.get("total"), resident_peak=resident_peak, running_peak=running_peak,
                   waiting_peak=waiting_peak, batch_peak=batch_peak, held_for_transfer_peak=held_peak,
                   in_use_peak=in_use_peak, free_min=free_min, cached_free_peak=cached_peak,
                   wait_s={k: round(v, 4) for k, v in sorted(by_reason.items(), key=lambda kv: _rank(kv[0]))},
                   requests_waiting={k: reqs_by_reason[k] for k in sorted(reqs_by_reason, key=_rank)},
                   occupancy=pool_rows[:4000])
        self._waits = getattr(self, "_waits", {})
        self._waits[role] = waits
        return out

    # ------------------------------------------------------------------------- per request
    def requests(self):
        rows = {}                                                           # key -> row

        def row(rid):
            key, tag, d = self.who(rid)
            k = key or rid
            if k not in rows:
                rows[k] = {"key": key, "tag": tag, "conv": d.get("conv_id"), "turn": (d.get("turn") or 0) + 1,
                           "order": d.get("order"), "t_send": d.get("t_send"), "t_first": d.get("t_first"),
                           "t_done": d.get("t_done"), "ttft_ms": ms(d.get("ttft")), "answer_ok": d.get("answer_ok"),
                           "answer_text": d.get("answer_text"), "output_sha256": d.get("output_sha256"),
                           "error": d.get("error"), "dest": (self.proxy.get(key) or {}).get("dest"),
                           "t_proxy_in": (self.proxy.get(key) or {}).get("t_in"),
                           "t_prefill_done": (self.proxy.get(key) or {}).get("t_prefill_done")}
            return rows[k]

        for role in ENGINES:
            st = "P" if role == "P" else "D"
            for r in self.ev(role, "vllm.request.added"):
                x = row(r["req"])
                x[f"{st}_engine"] = role
                x.setdefault(f"{st}_rid", r["req"])
                x.setdefault(f"t_{st}_arrive", r["t"])
            for r in self.ev(role, "vllm.gpu.prefix_hit"):
                row(r["req"]).setdefault(f"{st}_gpu_hit_tokens", r.get("tokens"))
            for r in self.ev(role, "vllm.gpu.allocated"):
                x = row(r["req"])
                x.setdefault(f"t_{st}_admit", r["t"])
                x[f"{st}_blocks"] = x.get(f"{st}_blocks", 0) + len(r.get("blocks") or [])
            for r in self.ev(role, "lmc.conn.lookup"):
                x = row(r["req"])
                x[f"{st}_lmc_hit_tokens"], x[f"{st}_lmc_load_tokens"] = r.get("lmc_hit"), r.get("load")
            for r in self.ev(role, "kv.connector.chosen"):
                x = row(r["req"])
                x[f"{st}_loaded_by"] = r.get("chosen")
                x[f"{st}_asked"] = r.get("asked")
            for r in self.ev(role, "lmc.conn.retrieve_planned"):
                x = row(r["req"])
                x[f"{st}_lmc_restored_tokens"] = x.get(f"{st}_lmc_restored_tokens", 0) + \
                    max(0, int(r.get("end", 0)) - int(r.get("start", 0)) - int(r.get("skip", 0)))
            for r in self.ev(role, "vllm.request.preempted"):
                x = row(r["req"])
                x[f"{st}_preempted"] = x.get(f"{st}_preempted", 0) + 1
                x[f"{st}_recomputed_tokens"] = x.get(f"{st}_recomputed_tokens", 0) + int(r.get("n_computed") or 0)
            for r in self.ev(role, "vllm.request.finished"):
                row(r["req"]).setdefault(f"t_{st}_finished", r["t"])
            for r in self.ev(role, "vllm.request.blocks_freed"):
                row(r["req"]).setdefault(f"t_{st}_freed", r["t"])
            for r in self.ev(role, "nixl.request_finished"):
                if r.get("is_p") and r.get("delay"):
                    x = row(r["req"])
                    x["P_held_blocks"] = len(r.get("blocks") or [])
                    x["t_P_held"] = r["t"]
            for r in self.ev(role, "nixl.match"):
                row(r["req"])["D_nixl_tokens"] = r.get("count")
            for r in self.ev(role, "nixl.read"):
                x = row(r["req"])
                x.setdefault("t_nixl_post", r["t"])
                x["nixl_blocks"] = x.get("nixl_blocks", 0) + len(r.get("local") or [])
            for r in self.ev(role, "nixl.done"):
                for rid in r.get("recving") or []:
                    row(rid).setdefault("t_kv_ready", r["t"])
                for rid in r.get("sending") or []:
                    row(rid).setdefault("t_P_released", r["t"])
                for rid in r.get("failed") or []:
                    row(rid)["nixl_failed"] = True
        waits = getattr(self, "_waits", {})
        for x in rows.values():
            for st in ("P", "D"):
                e, rid = x.get(f"{st}_engine"), x.get(f"{st}_rid")
                w = (waits.get(e) or {}).get(rid) or {}
                x[f"{st}_wait_ms"] = {k: ms(v) for k, v in sorted(w.items(), key=lambda kv: _rank(kv[0])) if v > 0}
                a, b = x.get(f"t_{st}_arrive"), x.get(f"t_{st}_admit")
                x[f"{st}_admit_ms"] = ms(b - a) if a is not None and b is not None else None
            if x.get("t_nixl_post") is not None and x.get("t_kv_ready") is not None:
                x["nixl_ms"] = ms(x["t_kv_ready"] - x["t_nixl_post"])
                if self.block_bytes and x.get("nixl_blocks"):
                    x["nixl_bytes"] = x["nixl_blocks"] * self.block_bytes
                    sec = x["t_kv_ready"] - x["t_nixl_post"]
                    x["nixl_gb_s"] = round(x["nixl_bytes"] / sec / 1e9, 3) if sec > 0 else None
            if x.get("t_P_held") is not None and x.get("t_P_released") is not None:
                x["P_held_ms"] = ms(x["t_P_released"] - x["t_P_held"])
        return sorted((x for x in rows.values() if x.get("tag")),
                      key=lambda x: (x.get("turn") or 0, x.get("order") if x.get("order") is not None else 99))

    # ------------------------------------------------------------------------- overlap from intervals
    @staticmethod
    def overlap(rows, start, end):
        ev = []
        for x in rows:
            a, b = x.get(start), x.get(end)
            if a is not None and b is not None and b >= a:
                ev += [(a, 1), (b, -1)]
        peak = cur = 0
        for _t, d in sorted(ev, key=lambda e: (e[0], e[1])):
            cur += d
            peak = max(peak, cur)
        return peak

    def boundary(self, rows):
        """The first request that waited for GPU capacity: where, how long, and the releases that let it in."""
        cap = ("no_free_blocks", "queued_behind_no_free_blocks")
        cands = []
        for x in rows:
            for st in ("P", "D"):
                w = x.get(f"{st}_wait_ms") or {}
                if any(w.get(c) for c in cap) and x.get(f"t_{st}_arrive") is not None:
                    cands.append((x[f"t_{st}_arrive"], st, x))
        if not cands:
            return None
        t_arr, st, x = min(cands, key=lambda c: c[0])
        e = x.get(f"{st}_engine")
        t_admit = x.get(f"t_{st}_admit")
        released = []
        if t_admit is not None:
            for y in rows:
                if y is x or y.get(f"{st}_engine") != e:
                    continue
                # P frees a request's blocks when D has read them (NIXL's done); D when the request finishes
                if st == "P":
                    tr, how = y.get("t_P_released") or y.get("t_P_freed"), "D read it over NIXL, P freed its blocks"
                else:
                    tr, how = y.get("t_D_freed") or y.get("t_D_finished"), "finished on D, its blocks freed"
                if tr is not None and t_arr <= tr <= t_admit:
                    released.append({"request": f"{y.get('conv')} T{y.get('turn')}", "how": how,
                                     "at_ms": ms(tr - t_arr)})
        return {"request": f"{x.get('conv')} T{x.get('turn')}", "stage": st, "engine": e,
                "waited_ms": x.get(f"{st}_wait_ms"), "admitted_after_ms": ms(t_admit - t_arr) if t_admit else None,
                "released_before_admission": sorted(released, key=lambda r: r["at_ms"])}

    def analyse(self):
        eng = {role: self.engine(role) for role in ENGINES}
        rows = self.requests()
        e, pool = self.env, self.env.get("pool") or {}
        usable = pool.get("usable_blocks") or (eng["P"].get("pool_total"))
        # a request's blocks with its output: the longest prompt and the tokens asked for, 16 tokens a block
        longest = max((d.get("prompt_tokens") or 0 for d in self.driver.values()), default=0)
        per_req = math.ceil((longest + int(e.get("max_tokens") or 0)) / 16) if longest else None
        conn = collections.defaultdict(collections.Counter)
        for x in rows:
            for st in ("P", "D"):
                if x.get(f"{st}_engine"):
                    conn[st][str(x.get(f"{st}_loaded_by"))] += 1
        lmc_roles = [r for r in (e.get("server_roles") or ["LMC"])]
        disk_reads = {s: sum(1 for _ in self.ev(s, "lmc.l2.read")) for s in lmc_roles}
        disk_keys = {s: sum(int(r.get("n_keys") or 0) for r in self.ev(s, "lmc.l2.read")) for s in lmc_roles}
        stores = {role: sum(1 for _ in self.ev(role, "lmc.conn.store_submitted")) for role in ENGINES}
        evictions = {role: len(self.ev(role, "vllm.gpu.evicted")) for role in ENGINES}
        preempt = {role: len(self.ev(role, "vllm.request.preempted")) for role in ENGINES}
        d_rows = [x for x in rows if x.get("D_engine")]
        p_rows = [x for x in rows if x.get("P_engine")]
        waves = (self.summary.get("waves") or [])
        wave_order_ok = all(waves[i + 1]["t_start"] >= waves[i]["t_end"] - 1e-6 for i in range(len(waves) - 1))
        dec = pool.get("decode") or "D1"
        checks = {
            "every request answered once": bool(rows) and all(x.get("t_done") and not x.get("error") for x in rows)
                                            and len(rows) == len(self.driver),
            "every prompt's hash the fixture's": all(d.get("prompt_sha256_ok") for d in self.driver.values()),
            "turn k+1 sent after every turn k finished": wave_order_ok,
            f"every request decoded on {dec}": all(x.get("D_engine") == dec for x in d_rows) and bool(d_rows),
            "every D request read its KV over NIXL, none failed": all(x.get("t_kv_ready") and not x.get("nixl_failed")
                                                                     for x in d_rows) and bool(d_rows),
            "LMCache loaded nothing on D": all(not (x.get("D_lmc_restored_tokens") or 0) for x in d_rows),
            "the scheduler's steps are in the trace": all(eng[r].get("steps_recorded") for r in ("P", dec)),
        }
        n_ok = sum(1 for x in rows if x.get("answer_ok"))
        return {
            "run": e.get("run"), "label": self.arm_label(), "env": {k: e.get(k) for k in (
                "mode", "servers", "transfer", "l1_gb", "eviction", "store_policy", "prefetch_policy", "disk_reads",
                "gpu_blocks", "max_num_seqs", "max_model_len", "max_num_batched_tokens", "gpu_mem", "model",
                "model_revision", "kv_bytes_per_token", "routing", "commit", "chunk")},
            "pool": pool, "usable_blocks": usable, "blocks_per_request": per_req,
            "capacity_requests": (usable // per_req) if usable and per_req else None,
            "offered": pool.get("count"), "requests": len(rows),
            "engines": eng, "overlap": {
                "P_admitted_to_released": self.overlap(p_rows, "t_P_admit", "t_P_released"),
                "D_admitted_to_finished": self.overlap(d_rows, "t_D_admit", "t_D_finished"),
                "client_in_flight": self.overlap(rows, "t_send", "t_done")},
            "lmcache": {"loaded_by": {k: dict(v) for k, v in conn.items()},
                        "P_requests_restored": sum(1 for x in p_rows if (x.get("P_lmc_restored_tokens") or 0) > 0),
                        "P_tokens_restored": sum(x.get("P_lmc_restored_tokens") or 0 for x in p_rows),
                        "P_lookups_with_hits": sum(1 for x in p_rows if (x.get("P_lmc_hit_tokens") or 0) > 0),
                        "disk_reads": disk_reads, "disk_chunks_read": disk_keys, "stores_submitted": stores},
            "gpu": {"evictions": evictions, "preemptions": preempt,
                    "P_prefix_hit_tokens": sum(x.get("P_gpu_hit_tokens") or 0 for x in p_rows),
                    "D_prefix_hit_tokens": sum(x.get("D_gpu_hit_tokens") or 0 for x in d_rows),
                    "prompt_tokens": sum((self.driver.get(x["tag"]) or {}).get("prompt_tokens") or 0 for x in rows)},
            "nixl": {"reads": sum(1 for x in d_rows if x.get("nixl_ms") is not None),
                     "blocks": sum(x.get("nixl_blocks") or 0 for x in d_rows),
                     "ms_median": q50([x.get("nixl_ms") for x in d_rows]),
                     "gb_s_median": q50([x.get("nixl_gb_s") for x in d_rows]),
                     "P_held_ms_median": q50([x.get("P_held_ms") for x in p_rows])},
            "answers": {"ok": n_ok, "of": len(rows), "ttft_ms_median": q50([x.get("ttft_ms") for x in rows])},
            "boundary": self.boundary(rows), "checks": checks, "requests_rows": rows,
            "trace": dict(self.load_info),
        }

    def arm_label(self):
        e, p = self.env, self.env.get("pool") or {}
        srv = {"shared": "shared", "per_engine": "per-engine"}.get(e.get("servers"), e.get("servers"))
        return (f"{p.get('count')} offered, {p.get('gpu_blocks') or e.get('gpu_blocks')} blocks, "
                f"{srv} {e.get('mode')}")


def _rank(reason):
    return WAIT_ORDER.index(reason) if reason in WAIT_ORDER else len(WAIT_ORDER)


def _fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _waits_cell(w):
    if not w:
        return "-"
    return ", ".join(f"{WAIT_NAMES.get(k, k).split(' (')[0]} {_fmt(v)} ms" for k, v in w.items())


def write_md(n, out):
    e = n["env"]
    L = [f"# KV pool arm: {n['label']}", "",
         f"Run `{n.get('run')}`, model {e.get('model')} ({str(e.get('model_revision'))[:12]}), commit "
         f"{str(e.get('commit'))[:12]}. GPU pool {e.get('gpu_blocks')} blocks an engine ({n.get('usable_blocks')} "
         f"usable, {n['blocks_per_request']} a request: {n.get('capacity_requests')} requests fit at once), "
         f"max-num-seqs {e.get('max_num_seqs')}, batch budget {e.get('max_num_batched_tokens')} tokens; LMCache "
         f"{e.get('servers')}, {e.get('mode')} (L1 {e.get('l1_gb')} GB, eviction {e.get('eviction')}, store "
         f"{e.get('store_policy')}, disk reads {e.get('disk_reads')}); every request on "
         f"{(n.get('pool') or {}).get('decode')}.", ""]
    L += ["## Checks", ""] + [f"- {'ok' if v else 'NOT'}: {k}" for k, v in n["checks"].items()] + [""]
    L += ["## Engines", "",
          "| Engine | Steps recorded | Peak resident | Peak running | Peak waiting | Peak batch | Peak held for NIXL "
          "| Peak blocks in use | Fewest free | Peak cached free | Evictions | Preemptions | Waiting (s, by reason) |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for role, g in n["engines"].items():
        if not g.get("steps_recorded"):
            continue
        ws = ", ".join(f"{WAIT_NAMES.get(k, k)} {v:.3f} ({g['requests_waiting'].get(k)} requests)"
                       for k, v in (g.get("wait_s") or {}).items() if v > 0) or "-"
        L.append(f"| {role} | {g['steps_recorded']} | {g.get('resident_peak')} | {g.get('running_peak')} | "
                 f"{g.get('waiting_peak')} | {g.get('batch_peak')} | {g.get('held_for_transfer_peak')} | "
                 f"{g.get('in_use_peak')} of {g.get('pool_total')} | {_fmt(g.get('free_min'))} | "
                 f"{_fmt(g.get('cached_free_peak'))} | {n['gpu']['evictions'].get(role)} | "
                 f"{n['gpu']['preemptions'].get(role)} | {ws} |")
    o = n["overlap"]
    L += ["", f"Overlap measured: P held at most {o['P_admitted_to_released']} requests between admission and NIXL's "
          f"release, D at most {o['D_admitted_to_finished']} between admission and finish; the client had at most "
          f"{o['client_in_flight']} in flight. Resident: running, loading (WAITING_FOR_REMOTE_KVS) or finished and "
          "holding blocks for NIXL, at one step.", ""]
    b = n.get("boundary")
    L += ["## The first wait for GPU capacity", ""]
    if b:
        rel = "; ".join(f"{r['request']} ({r['how']}, +{r['at_ms']} ms)" for r in b["released_before_admission"])
        L += [f"{b['request']} arrived at {b['engine']} ({b['stage']} stage) and waited: "
              f"{_waits_cell(b['waited_ms'])}; admitted {b['admitted_after_ms']} ms after it arrived. "
              f"Released in between: {rel or 'nothing traced'}.",
              ""]
    else:
        L += ["No request waited for GPU capacity.", ""]
    lm, gp, nx = n["lmcache"], n["gpu"], n["nixl"]
    L += ["## LMCache, the GPU's prefix cache, NIXL", "",
          f"- Loaded by (MultiConnector's choice): P {lm['loaded_by'].get('P', {})}, D {lm['loaded_by'].get('D', {})}",
          f"- P: {lm['P_lookups_with_hits']} lookups found chunks, {lm['P_requests_restored']} requests restored "
          f"{lm['P_tokens_restored']} tokens from LMCache to the GPU; GPU prefix hits {gp['P_prefix_hit_tokens']} "
          f"tokens (D {gp['D_prefix_hit_tokens']}) of {gp['prompt_tokens']} prompt tokens",
          f"- Disk reads (into LMCache's memory): {lm['disk_reads']} ({lm['disk_chunks_read']} chunks); stores "
          f"submitted {lm['stores_submitted']}",
          f"- NIXL: {nx['reads']} reads, {nx['blocks']} blocks, median {_fmt(nx['ms_median'])} ms "
          f"({_fmt(nx['gb_s_median'], 3)} GB/s); P held its blocks a median {_fmt(nx['P_held_ms_median'])} ms",
          f"- Answers with the reference answer: {n['answers']['ok']} of {n['answers']['of']}; TTFT median "
          f"{_fmt(n['answers']['ttft_ms_median'])} ms", ""]
    L += ["## Requests", "",
          "| Turn | Conversation | P waited | P admitted after (ms) | P GPU hit | LMCache restored | P held (ms) "
          "| D waited | D GPU hit | NIXL blocks | NIXL ms | TTFT ms | Answer |",
          "| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for x in n["requests_rows"]:
        L.append(f"| {x.get('turn')} | {x.get('conv')} | {_waits_cell(x.get('P_wait_ms'))} | "
                 f"{_fmt(x.get('P_admit_ms'))} | {_fmt(x.get('P_gpu_hit_tokens'))} | "
                 f"{_fmt(x.get('P_lmc_restored_tokens'))} | {_fmt(x.get('P_held_ms'))} | "
                 f"{_waits_cell(x.get('D_wait_ms'))} | {_fmt(x.get('D_gpu_hit_tokens'))} | "
                 f"{_fmt(x.get('nixl_blocks'))} | {_fmt(x.get('nixl_ms'))} | {_fmt(x.get('ttft_ms'))} | "
                 f"{'ok' if x.get('answer_ok') else (x.get('error') or repr((x.get('answer_text') or '')[:40]))} |")
    L.append("")
    (out / "pool.md").write_text("\n".join(L))
    return "\n".join(L)


def cmd_report(a):
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    run = a.run
    if a.results:
        run = tempfile.mkdtemp(prefix="pool_unpack_")
        lifecycle().unpack(pathlib.Path(a.results), pathlib.Path(run))
        for f in ("mtrag_summary.json",):
            src = pathlib.Path(a.results) / f
            if src.is_file():
                (pathlib.Path(run) / f).write_bytes(src.read_bytes())
    n = Pool(run).analyse()
    json.dump(n, open(out / "pool.json", "w"), indent=1, default=str)
    md = write_md(n, out)
    s = out / "summary.md"
    if s.is_file() and not s.read_text().startswith("# KV pool arm"):
        s.write_text(md + "\n\n" + s.read_text())
    bad = [k for k, v in n["checks"].items() if not v]
    print(f"   pool report: {n['label']}: peak resident P {n['engines']['P'].get('resident_peak')}, "
          f"D {n['engines'].get((n.get('pool') or {}).get('decode') or 'D1', {}).get('resident_peak')}; "
          f"LMCache restored {n['lmcache']['P_tokens_restored']} tokens on P; answers {n['answers']['ok']} of "
          f"{n['answers']['of']}" + (f"; checks not met: {bad}" if bad else "; every check met"))
    return 0


def cmd_compare(a):
    out = pathlib.Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    arms = []
    for d in [x for x in a.dirs.split(",") if x]:
        f = pathlib.Path(d) / "pool.json"
        if f.is_file():
            n = json.load(open(f))
            n["folder"] = pathlib.Path(d).name
            arms.append(n)
    srv_rank = {"per_engine": 0, "shared": 1}
    mode_rank = {"memory": 0, "disk": 1}
    arms.sort(key=lambda n: (n.get("offered") or 0, n["env"].get("gpu_blocks") or 0,
                             srv_rank.get(n["env"].get("servers"), 9), mode_rank.get(n["env"].get("mode"), 9)))
    rows = []
    for n in arms:
        dec = (n.get("pool") or {}).get("decode") or "D1"
        P, D = n["engines"].get("P", {}), n["engines"].get(dec, {})
        cap = ("no_free_blocks", "queued_behind_no_free_blocks")
        rows.append({
            "offered": n.get("offered"), "blocks": n["env"].get("gpu_blocks"), "fit": n.get("capacity_requests"),
            "servers": n["env"].get("servers"), "mode": n["env"].get("mode"),
            "P_peak": P.get("resident_peak"), "D_peak": D.get("resident_peak"),
            "P_capacity_wait_s": round(sum((P.get("wait_s") or {}).get(c, 0) for c in cap), 3),
            "D_capacity_wait_s": round(sum((D.get("wait_s") or {}).get(c, 0) for c in cap), 3),
            "D_load_wait_s": (D.get("wait_s") or {}).get("kv_load_in_flight"),
            "lookup_wait_s": round(sum((g.get("wait_s") or {}).get("connector_lookup_pending", 0)
                                       for g in n["engines"].values()), 3),
            "P_hit_tokens": n["gpu"]["P_prefix_hit_tokens"], "P_restored_tokens": n["lmcache"]["P_tokens_restored"],
            "disk_reads": sum((n["lmcache"].get("disk_reads") or {}).values()),
            "P_evictions": n["gpu"]["evictions"].get("P"), "D_evictions": n["gpu"]["evictions"].get(dec),
            "preemptions": sum((n["gpu"].get("preemptions") or {}).values()),
            "nixl_ms": n["nixl"].get("ms_median"), "ttft_ms": n["answers"].get("ttft_ms_median"),
            "answers": f"{n['answers']['ok']}/{n['answers']['of']}",
            "checks": all(n["checks"].values()), "folder": n["folder"]})
    json.dump(rows, open(out / "pool_compare.json", "w"), indent=1)
    L = ["# KV pool experiment: the arms side by side", "",
         "Each arm: the fixture's first n conversations offered together turn by turn, every request decoded on one "
         "D engine; GPU pools of 131, 261, 391 blocks fit 2, 4, 6 requests of 65 blocks. Peak: the most requests "
         "resident on the engine at one scheduler step (running, loading, or holding blocks for NIXL). Capacity "
         "wait: seconds requests spent waiting with no free blocks for them (or behind one that had none).", "",
         "| Offered | Blocks (fit) | LMCache | P peak | D peak | P capacity wait s | D capacity wait s | D KV load "
         "wait s | Lookup wait s | P GPU hit tokens | LMCache restored on P | Disk reads | Evictions P/D | "
         "Preemptions | NIXL ms | TTFT ms | Answers | Checks |",
         "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
         "| ---: | ---: | --- |"]
    for r in rows:
        L.append(f"| {r['offered']} | {r['blocks']} ({r['fit']}) | {r['servers']} {r['mode']} | {_fmt(r['P_peak'])} | "
                 f"{_fmt(r['D_peak'])} | {_fmt(r['P_capacity_wait_s'], 3)} | {_fmt(r['D_capacity_wait_s'], 3)} | "
                 f"{_fmt(r['D_load_wait_s'], 3)} | {_fmt(r['lookup_wait_s'], 3)} | {r['P_hit_tokens']} | "
                 f"{r['P_restored_tokens']} | {r['disk_reads']} | {r['P_evictions']}/{r['D_evictions']} | "
                 f"{r['preemptions']} | {_fmt(r['nixl_ms'])} | {_fmt(r['ttft_ms'])} | {r['answers']} | "
                 f"{'ok' if r['checks'] else 'NOT'} |")
    if not rows:
        L.append("| (no arm had a pool.json) |" + " |" * 17)
    L.append("")
    (out / "pool_compare.md").write_text("\n".join(L))
    print(f"   pool comparison: {len(rows)} arms -> {out / 'pool_compare.md'}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--fixture", required=True)
    c.add_argument("--tokenizer", default=None, help="the model's folder (its tokenizer.json)")
    c.add_argument("--output", default=None)
    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--count", type=int, required=True)
    r.add_argument("--proxy", required=True)
    r.add_argument("--model", default="llama", help="the served model name")
    r.add_argument("--run", required=True)
    r.add_argument("--decode", type=int, default=0, help="the D engine every request goes to (0: D1)")
    r.add_argument("--stagger-ms", type=float, default=5.0)
    r.add_argument("--max-tokens", type=int, default=8)
    r.add_argument("--request-timeout", type=float, default=600)
    r.add_argument("--output", required=True)
    r.add_argument("--summary", required=True)
    p = sub.add_parser("report")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--results")
    p.add_argument("--output", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--dirs", required=True)
    m.add_argument("--output", required=True)
    k = sub.add_parser("kv-bytes")
    k.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    return {"check": cmd_check, "run": cmd_run, "report": cmd_report, "compare": cmd_compare,
            "kv-bytes": cmd_kv_bytes}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
