#!/usr/bin/env python3
"""The lives of KV blocks and LMCache chunks in one LMCache lifecycle run (scripts/lmc_run.sh), from its trace.

  python3 -I scripts/lmc_lifecycle.py report --run RUN_DIR --output RESULTS_DIR
  python3 -I scripts/lmc_lifecycle.py compare --dirs DIR,DIR,... --output DIR

report reads RUN_DIR/kvtrace/*.jsonl (every process's hook records, docs/lmcache_lifecycle.md), the proxy's log
(proxy.jsonl: each request's key, conversation and turn), the MTRAG driver's records (mtrag.jsonl: each turn's
prompt pieces and token ids) and env.json, and follows:

  a block    16 tokens in a vLLM engine's GPU KV cache, named by vLLM's block hash: computed and cached on P (its
             GPU block, the request and time), evicted from P's GPU; on D, loaded into a GPU block from LMCache (or
             held from an earlier turn, or computed by D), cached there, evicted
  a chunk    256 tokens (16 blocks) in LMCache, named by LMCache's chunk hash: its tokens and place in the prompt
             (mp.tokens), stored by P's connector (store planned, submitted), written to the server's memory (L1
             write reserved, finished), to disk (L2 stored) and dropped from memory or evicted, looked up by a D
             engine (hit or miss, from L1 or L2), read from disk into memory (L2 accessed), copied to D's GPU
             (retrieve), and whether it was ever read

and writes to RESULTS_DIR: summary.md (what happened, the numbers, one chunk's and one request's whole story),
results.json (the numbers), chunks.jsonl.gz (one row per chunk, every time of its life), blocks.jsonl.gz (one row
per engine, request and block of its prompt: where the block came from and its GPU block), requests.jsonl.gz (one
row per engine request: hits, loads, stores and their times).

compare reads several results folders' results.json (one per mode) and writes compare.md and compare.json.
Standard library only (python -I).
"""
from __future__ import annotations

import argparse
import array
import collections
import gzip
import json
import pathlib
import re
import sys
import zlib

KEY_RE = re.compile(r"[0-9a-f]{32}")
BLOCK = 16
ENGINES = ("P", "D1", "D2", "D3")
L1_EVENTS = {"l1.write.reserved", "l1.write.finished", "l1.write_finished_and_read_reserved", "l1.read.reserved",
             "l1.read.finished", "l1.keys.evicted", "l1.keys.accessed", "l1.allocation.failed", "l1.read.failed"}
L2_EVENTS = {"l2.keys.stored", "l2.keys.accessed", "l2.keys.deleted", "l2.prefetch.failed"}
SM_EVENTS = {"sm.write.reserved", "sm.write.finished", "sm.read.prefetched_finished"}
SESSION_EVENTS = {"mp.request.start", "mp.request.end", "mp.store.submitted", "mp.store.start", "mp.store.end",
                  "mp.retrieve.submitted", "mp.retrieve.start", "mp.retrieve.end", "mp.lookup_prefetch.start",
                  "mp.lookup_prefetch.end", "mp.vllm.end_session"}


def rkey(s):
    """The proxy's key (32 hex) inside a vLLM request id ("cmpl-<key>-0-<random>")."""
    m = KEY_RE.search(str(s or ""))
    return m.group(0) if m else None


def read_jsonl(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    out = []
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    return out


def crc(tokens):
    return zlib.crc32(array.array("i", list(tokens)).tobytes())


def q(xs, p):
    """The p-quantile (0..1) of xs, or None."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def ms(x):
    return None if x is None else round(1000 * x, 2)


def first(xs, cond=lambda r: True):
    for x in xs:
        if cond(x):
            return x
    return None


# ------------------------------------------------------------------------------------------------- loading
def load(run):
    recs, bad, files = [], 0, sorted((run / "kvtrace").glob("kvtrace_*.jsonl"))
    for f in files:
        with open(f, errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    recs.append(json.loads(ln))
                except ValueError:
                    bad += 1                     # a process stopped mid-line
    recs.sort(key=lambda r: r.get("t", 0.0))
    return recs, {"files": len(files), "records": len(recs), "unreadable_lines": bad,
                  "by_role": dict(collections.Counter(r.get("role") for r in recs))}


def dedup_stream(recs):
    """An LMCache event scheduled on a CUDA stream is seen twice: when queued ("via": "stream") and when the GPU
    reached it (the native recorder's copy, "native", or the host callback publishing it again). Keep the second,
    with the queue time as t_queued; a queued one never reached is kept, marked t_is_queue."""
    reached = collections.defaultdict(list)
    for r in recs:
        if r.get("via") != "stream" and r.get("transfer_key") and r["ev"].startswith("lmc."):
            reached[(r["ev"], r.get("session"), r.get("transfer_key"))].append(r)
    out = []
    for r in recs:
        if r.get("via") == "stream":
            hit = reached.get((r["ev"], r.get("session"), r.get("transfer_key")))
            if hit:
                hit[0].setdefault("t_queued", r["t"])
                continue
            r = dict(r, t_is_queue=True)
        out.append(r)
    return out


# ------------------------------------------------------------------------------------------------- analysis
class Run:
    def __init__(self, run):
        self.dir = pathlib.Path(run)
        self.env = json.loads((self.dir / "env.json").read_text()) if (self.dir / "env.json").is_file() else {}
        self.mode = self.env.get("mode", "?")
        self.chunk = int(self.env.get("chunk", 256))
        recs, self.load_info = load(self.dir)
        self.recs = dedup_stream(recs)
        self.t0 = self.recs[0]["t"] if self.recs else 0.0
        self.proxy = {r["key"]: r for r in read_jsonl(self.dir / "proxy.jsonl") if r.get("key")}
        self.turns = {r["tag"]: r for r in read_jsonl(self.dir / "mtrag.jsonl") if r.get("tag")}
        self.reqs = {}                                    # (role, rid) -> request
        self.lives = {e: collections.defaultdict(list) for e in ENGINES}   # vLLM hash -> [life]
        self.chunks = {}                                  # chunk hash -> chunk
        self.sessions = collections.defaultdict(lambda: collections.defaultdict(list))   # server session -> events
        self.prefetch = []                                # L2 prefetch / load task events (by request_id)
        self.errors = collections.Counter()
        self.unknown = collections.Counter()
        self.has_read_hook = False                   # traces from before the lmc.l2.read hook (jobs up to 30728)
        self._engine()
        self._server()

    # -------------------------------------------------------------------------------- per engine
    def R(self, role, rid):
        k = (role, rid)
        if k not in self.reqs:
            key = rkey(rid)
            px = self.proxy.get(key, {})
            turn = self.turns.get(px.get("tag"), {})
            self.reqs[k] = {"role": role, "rid": rid, "key": key, "tag": px.get("tag"), "conv": turn.get("conv"),
                            "turn": turn.get("turn"), "hashes": [], "cached": {}, "lookups": [], "retrieves": [],
                            "stores": [], "allocs": [], "retrieve_submitted": [], "store_submitted": []}
        return self.reqs[k]

    def _engine(self):
        for r in self.recs:
            ev, role, t = r["ev"], r.get("role"), r["t"]
            if ev == "kvtrace.error":
                self.errors[r.get("of")] = max(self.errors[r.get("of")], int(r.get("n", 1)))
                continue
            if role not in ENGINES:
                continue
            if ev == "vllm.request.added":
                q_ = self.R(role, r["req"])
                q_.update(t_added=t, n_prompt=r.get("n_prompt"), hashes=r.get("hashes") or [])
            elif ev == "vllm.gpu.prefix_hit":
                q_ = self.R(role, r["req"])
                q_["local_hit"], q_["t_prefix"] = r.get("tokens"), t
            elif ev == "vllm.gpu.allocated":
                self.R(role, r["req"])["allocs"].append(r)
            elif ev == "vllm.gpu.cached":
                q_ = self.R(role, r["req"])
                for i, b, h in zip(r.get("index") or [], r.get("blocks") or [], r.get("hashes") or []):
                    q_["cached"].setdefault(i, (b, h, t))
                    lives = self.lives[role][h]
                    if not lives or lives[-1].get("t_evicted") is not None:
                        lives.append({"t_cached": t, "rid": r["req"], "gpu": b, "index": i, "t_evicted": None})
            elif ev == "vllm.gpu.evicted":
                for h in r.get("hashes") or []:
                    lives = self.lives[role].get(h)
                    if lives and lives[-1].get("t_evicted") is None:
                        lives[-1]["t_evicted"] = t
            elif ev == "vllm.gpu.reset":
                for lives in self.lives[role].values():
                    if lives and lives[-1].get("t_evicted") is None:
                        lives[-1]["t_evicted"] = t
            elif ev == "vllm.request.blocks_freed":
                self.R(role, r["req"])["t_freed"] = t
            elif ev == "vllm.request.finished":
                q_ = self.R(role, r["req"])
                q_.update(t_finished=t, n_out=r.get("n_out"), status=r.get("status"))
            elif ev == "lmc.conn.lookup":
                self.R(role, r["req"])["lookups"].append(r)
            elif ev == "lmc.conn.alloc":
                self.R(role, r["req"])["alloc"] = r
            elif ev == "lmc.conn.retrieve_planned":
                self.R(role, r["req"])["retrieves"].append(r)
            elif ev == "lmc.conn.store_planned":
                self.R(role, r["req"])["stores"].append(r)
            elif ev in ("lmc.conn.retrieve_submitted", "lmc.conn.store_submitted"):
                k = ev.split(".")[-1]
                for rid in r.get("reqs") or []:
                    self.R(role, rid)[k].append(t)
            elif ev == "lmc.conn.finished":
                for rid in r.get("sending") or []:
                    self.R(role, rid).setdefault("t_sent", t)
                for rid in r.get("receiving") or []:
                    self.R(role, rid).setdefault("t_received", t)
            elif ev == "lmc.conn.request_finished":
                self.R(role, r["req"])["conn_final"] = r
            elif ev.startswith("lmc."):
                self.sessions[r.get("session")]["engine:" + ev[4:]].append(r)   # engine-side LMCache events
            else:
                self.unknown[ev] += 1

    # -------------------------------------------------------------------------------- the server
    def C(self, h):
        if h not in self.chunks:
            self.chunks[h] = {"hash": h, "l1": [], "l2": [], "sm": []}
        return self.chunks[h]

    def _server(self):
        for r in self.recs:
            if r.get("role") != "LMC" or not r["ev"].startswith("lmc."):
                continue
            name, t = r["ev"][4:], r["t"]
            if name == "mp.tokens":
                hs, offs = r.get("chunk_hashes") or [], r.get("token_offsets") or []
                ntok, crcs = r.get("chunk_tokens") or [], r.get("chunk_crc") or []
                par = r.get("parent_hashes") or []
                for i, h in enumerate(hs):
                    c = self.C(h)
                    c.setdefault("bindings", []).append(t)
                    if "offset" not in c:
                        c.update(offset=offs[i] if i < len(offs) else None, ntok=ntok[i] if i < len(ntok) else None,
                                 crc=crcs[i] if i < len(crcs) else None, parent=par[i] if i < len(par) else None,
                                 stored_by=r.get("session"), t_bound=t)
            elif name == "mp.lookup":
                self.sessions[r.get("session")]["mp.lookup"].append(r)
            elif name in SESSION_EVENTS:
                self.sessions[r.get("session")][name].append(r)
            elif name == "l2.read":                  # our hook: a real read from disk
                self.has_read_hook = True
                for h in r.get("keys") or []:
                    self.C(h)["l2"].append(("l2.read", t, None))
            elif name == "touch":                    # our hook: eviction order refreshed, no I/O
                for h in r.get("keys") or []:
                    self.C(h).setdefault("touch", []).append(t)
            elif name in L1_EVENTS:
                sizes = r.get("sizes") or []
                for i, h in enumerate(r.get("keys") or []):
                    self.C(h)["l1"].append((name, t, sizes[i] if i < len(sizes) else None))
            elif name in L2_EVENTS:
                sizes = r.get("sizes") or []
                for i, h in enumerate(r.get("keys") or []):
                    self.C(h)["l2"].append((name, t, sizes[i] if i < len(sizes) else None))
            elif name in SM_EVENTS:
                for k in ("succeeded_keys", "failed_keys"):
                    for h in r.get(k) or []:
                        self.C(h)["sm"].append((name, k.split("_")[0], t))
            elif name.startswith(("l2.prefetch", "l2.load_task", "l2.store")):
                self.prefetch.append(r)
            else:
                self.unknown["lmc." + name] += 1

    # -------------------------------------------------------------------------------- joins
    def engine_of(self, session):
        for e in ENGINES:
            if (e, session) in self.reqs:
                return self.reqs[(e, session)]
        return None

    def rel(self, t):
        return None if t is None else round(t - self.t0, 4)

    def pieces_at(self, key, start, end):
        """The MTRAG pieces (kind, ref) a token range of a request's prompt holds."""
        tag = self.proxy.get(key, {}).get("tag")
        rec = self.turns.get(tag) or {}
        out = []
        for p in rec.get("pieces") or []:
            kind, ref, s, e = (p + [None] * 4)[:4]
            if s is not None and e is not None and s < end and e > start:
                out.append(f"{kind} {ref}" if ref not in (None, "") else str(kind))
        return out

    @staticmethod
    def _derive(c):
        """A chunk's times in memory and on disk, from its key events."""
        l1, l2 = c["l1"], c["l2"]
        c["t_l1_reserved"] = first([t for nm, t, _ in l1 if nm == "l1.write.reserved"])
        c["t_l1_written"] = first([t for nm, t, _ in l1 if nm in ("l1.write.finished",
                                                                  "l1.write_finished_and_read_reserved")])
        c["bytes"] = first([sz for nm, _, sz in l1 + l2 if sz])
        c["t_l2_stored"] = first([t for nm, t, _ in l2 if nm == "l2.keys.stored"])
        c["t_l1_removed"] = [t for nm, t, _ in l1 if nm == "l1.keys.evicted"]
        c["t_l2_accessed"] = [t for nm, t, _ in l2 if nm == "l2.keys.accessed"]
        c["t_l2_read"] = [t for nm, t, _ in l2 if nm == "l2.read"]
        c.setdefault("touch", [])
        c["t_l2_deleted"] = first([t for nm, t, _ in l2 if nm == "l2.keys.deleted"])
        c["reads"] = []

    def analyse(self):
        ch = self.chunk
        # per server session: its lookups, the hit (chunks found), the retrieve
        for sess, evs in self.sessions.items():
            req = self.engine_of(sess)
            if req is not None:
                req["server"] = evs
        # chunks: creation on P, the store, memory and disk, and every read
        for h, c in self.chunks.items():
            sreq = self.engine_of(c.get("stored_by"))
            c["store_role"] = sreq["role"] if sreq else None
            c["key"] = sreq["key"] if sreq else rkey(c.get("stored_by"))
            c["conv"], c["turn"] = (sreq or {}).get("conv"), (sreq or {}).get("turn")
            off, n = c.get("offset"), c.get("ntok")
            if sreq is not None and off is not None and n:
                b0, b1 = off // BLOCK, (off + n) // BLOCK
                blocks = []
                for i in range(b0, b1):
                    vh = sreq["hashes"][i] if i < len(sreq["hashes"]) else None
                    life = first(self.lives[sreq["role"]].get(vh, []))
                    blocks.append({"index": i, "hash": vh, "gpu": life["gpu"] if life else None,
                                   "t_cached": life["t_cached"] if life else None,
                                   "computed_by": rkey(life["rid"]) if life else None})
                c["p_blocks"] = blocks
                tc = [b["t_cached"] for b in blocks if b["t_cached"] is not None]
                c["t_created"] = max(tc) if len(tc) == len(blocks) and tc else None
                st = first(sreq["stores"], lambda s: s["start"] <= off < s["end"])
                c["t_store_planned"] = st["t"] if st else None
                c["t_store_submitted"] = first(sreq["store_submitted"], lambda t: st is not None and t >= st["t"])
                c["pieces"] = self.pieces_at(c["key"], off, off + n)
                ids = (self.turns.get(self.proxy.get(c["key"], {}).get("tag")) or {}).get("ids")
                if ids and c.get("crc") is not None and off + n <= len(ids):
                    c["tokens_match_prompt"] = crc(ids[off:off + n]) == c["crc"]
            self._derive(c)
        # reads: every engine request whose lookup asked for a chunk
        for req in self.reqs.values():
            srv = req.get("server") or {}
            lks = srv.get("mp.lookup") or []
            if not lks:
                continue
            look = lks[-1]
            hashes = look.get("chunk_hashes") or []
            pend = (srv.get("mp.lookup_prefetch.end") or [None])[-1]
            found = (pend or {}).get("found_count")
            conn = (req["lookups"] or [None])[-1] or {}
            ret = (req["retrieves"] or [None])[-1]
            r_end = (srv.get("mp.retrieve.end") or [None])[-1]
            req["looked_up_chunks"], req["found_chunks"] = len(hashes), found
            req["l1_hit_tokens"] = (pend or {}).get("l1_hit_tokens")
            req["l2_hit_tokens"] = (pend or {}).get("l2_hit_tokens")
            for j, h in enumerate(hashes):
                c = self.chunks.get(h)
                if c is None:                            # looked up, never stored (nor seen in memory or on disk)
                    c = self.C(h)
                    self._derive(c)
                    c["unknown_store"] = True
                lo, hi = j * ch, (j + 1) * ch
                hit = found is not None and j < found
                if not hit and c.get("stored_by") == req["rid"]:
                    how = "new: stored by this request"       # P looks its prompt up before it stores it
                elif not hit:
                    tw = c.get("t_l1_written")
                    if tw is None:
                        how = "miss: never stored"
                    elif tw > look["t"]:
                        how = "miss: not yet stored"
                    elif any(t <= look["t"] for t in c["t_l1_removed"]) and c.get("t_l2_stored") is None:
                        how = "miss: evicted"
                    else:
                        how = "miss"
                elif ret is not None and ret["start"] <= lo and hi <= ret["end"] and conn.get("load", 0) > 0:
                    skip_to = ret["start"] + ret.get("skip", 0)
                    how = "pulled" if hi > skip_to else "held (on GPU already)"
                else:
                    how = "held (on GPU already)" if (conn.get("vllm_hit") or 0) >= hi else "hit, not loaded"
                tier = None
                if how == "pulled":
                    t_hi = (r_end or {}).get("t", look["t"] + 600)
                    reads = c["t_l2_read"] if self.has_read_hook else c["t_l2_accessed"]
                    tier = "L2" if any(look["t"] - 0.001 <= t <= t_hi + 0.001 for t in reads) else "L1"
                c["reads"].append({"role": req["role"], "key": req["key"], "conv": req["conv"], "turn": req["turn"],
                                   "index": j, "how": how, "tier": tier, "t_lookup": look["t"],
                                   "t_found": (pend or {}).get("t"), "t_retrieved": (r_end or {}).get("t")
                                   if how == "pulled" else None})
                if how == "pulled" and ret is not None:
                    # D's blocks of this chunk: retrieve op block k covers token block ret.start/16 + k
                    d_blocks = []
                    for i in range(lo // BLOCK, hi // BLOCK):
                        k = i - ret["start"] // BLOCK
                        gpu = ret["blocks"][k] if 0 <= k < len(ret.get("blocks") or []) else None
                        cached = req["cached"].get(i)
                        dh = req["hashes"][i] if i < len(req["hashes"]) else None
                        p_h = None
                        if c.get("p_blocks"):
                            pb = first(c["p_blocks"], lambda b: b["index"] == i)
                            p_h = pb["hash"] if pb else None
                        written = i * BLOCK >= ret["start"] + ret.get("skip", 0)
                        d_blocks.append({"index": i, "gpu": gpu, "written": written, "hash": dh,
                                         "same_as_p": None if p_h is None or dh is None else dh == p_h,
                                         "t_cached": cached[2] if cached else None})
                    c["reads"][-1]["d_blocks"] = d_blocks

    # -------------------------------------------------------------------------------- rows and numbers
    def request_rows(self):
        rows = []
        for (role, rid), r in sorted(self.reqs.items(), key=lambda kv: kv[1].get("t_added") or 0):
            srv = r.get("server") or {}
            conn = (r["lookups"] or [None])[-1] or {}
            ret = (r["retrieves"] or [None])[-1]
            r_start = first(srv.get("mp.retrieve.start") or [])
            r_end = (srv.get("mp.retrieve.end") or [None])[-1]
            s_end = srv.get("mp.store.end") or []
            stored = sum(s["end"] - s["start"] for s in r["stores"])
            n = r.get("n_prompt")
            vh = conn.get("vllm_hit", r.get("local_hit"))
            load = conn.get("load") or 0
            px = self.proxy.get(r["key"], {})
            reads = collections.Counter()
            tiers = collections.Counter()
            for c in self.chunks.values():
                for rd in c["reads"]:
                    if rd["key"] == r["key"] and rd["role"] == role:
                        reads[rd["how"]] += 1
                        if rd["tier"]:
                            tiers[rd["tier"]] += 1
            row = {"role": role, "rid": rid, "key": r["key"], "conv": r["conv"], "turn": r["turn"], "n_prompt": n,
                   "local_hit": vh, "lmc_hit": conn.get("lmc_hit"), "load": load,
                   "computed": None if n is None else max(0, n - max(vh or 0, (vh or 0) + load)),
                   "looked_up_chunks": r.get("looked_up_chunks"), "found_chunks": r.get("found_chunks"),
                   "l1_hit_tokens": r.get("l1_hit_tokens"), "l2_hit_tokens": r.get("l2_hit_tokens"),
                   "chunks": dict(reads), "pulled_by_tier": dict(tiers),
                   "retrieve": None if ret is None else {"start": ret["start"], "end": ret["end"],
                                                         "skip": ret.get("skip", 0), "blocks": ret.get("n_blocks")},
                   "retrieve_bytes": (r_end or {}).get("total_bytes"),
                   "stored_tokens": stored, "store_ops": len(r["stores"]),
                   "store_bytes": sum(s.get("total_bytes") or 0 for s in s_end),
                   "t_added": self.rel(r.get("t_added")), "t_lookup": self.rel(first([x["t"] for x in r["lookups"]])),
                   "t_found": self.rel(first([x["t"] for x in srv.get("mp.lookup_prefetch.end") or []])),
                   "t_retrieve_start": self.rel((r_start or {}).get("t")), "t_retrieve_end": self.rel((r_end or {}).get("t")),
                   "t_received": self.rel(r.get("t_received")),
                   "t_store_planned": self.rel(first([s["t"] for s in r["stores"]])),
                   "t_store_end": self.rel(max([s["t"] for s in s_end], default=None)), "t_sent": self.rel(r.get("t_sent")),
                   "t_finished": self.rel(r.get("t_finished")), "t_freed": self.rel(r.get("t_freed")),
                   "proxy_t_in": self.rel(px.get("t_in")), "proxy_t_prefill_done": self.rel(px.get("t_prefill_done")),
                   "proxy_t_first_byte": self.rel(px.get("t_first_byte"))}
            # the pull: from the server's lookup (its prefetch reads the disk) to the copy on the GPU done
            t_look = first([x["t"] for x in srv.get("mp.lookup") or []]) or first([x["t"] for x in r["lookups"]])
            p_end = first([x["t"] for x in srv.get("mp.lookup_prefetch.end") or []])
            row["t_server_lookup"] = self.rel(first([x["t"] for x in srv.get("mp.lookup") or []]))
            if t_look is not None and p_end is not None:
                row["prefetch_s"] = round(p_end - t_look, 6)
            if t_look is not None and r_end is not None and load:
                row["pull_s"] = round(r_end["t"] - t_look, 6)
                if r_start is not None:
                    row["retrieve_s"] = round(r_end["t"] - r_start["t"], 6)
            if r["stores"] and s_end:
                row["store_s"] = round(max(s["t"] for s in s_end) - r["stores"][0]["t"], 6)
            # D looked this prompt up before P's store of the same turn had landed: what it missed
            if role != "P" and r["lookups"]:
                p = first(self.reqs.values(), lambda x: x["role"] == "P" and x["key"] == r["key"])
                if p is not None:
                    pend = max([s["t"] for s in ((p.get("server") or {}).get("mp.store.end") or [])], default=None)
                    tf = first([x["t"] for x in srv.get("mp.lookup_prefetch.end") or []])
                    if pend is not None and tf is not None:
                        row["looked_up_before_store_end"] = tf < pend
                        row["p_store_end_minus_d_found_s"] = round(pend - tf, 6)
            rows.append(row)
        return rows

    def chunk_rows(self):
        rows = []
        for c in sorted(self.chunks.values(), key=lambda c: (c.get("t_bound") or 9e18, c.get("offset") or 0)):
            row = {k: c.get(k) for k in ("hash", "key", "conv", "turn", "offset", "ntok", "crc", "parent", "pieces",
                                         "tokens_match_prompt", "bytes", "store_role", "unknown_store")}
            for k in ("t_created", "t_store_planned", "t_store_submitted", "t_bound", "t_l1_reserved", "t_l1_written",
                      "t_l2_stored", "t_l2_deleted"):
                row[k] = self.rel(c.get(k))
            row["t_l1_removed"] = [self.rel(t) for t in c.get("t_l1_removed") or []]
            row["t_l2_accessed"] = [self.rel(t) for t in c.get("t_l2_accessed") or []]
            row["t_l2_read"] = [self.rel(t) for t in c.get("t_l2_read") or []]
            row["t_touched"] = [self.rel(t) for t in c.get("touch") or []]
            row["p_blocks"] = [dict(b, t_cached=self.rel(b["t_cached"])) for b in c.get("p_blocks") or []]
            row["reads"] = [dict(rd, t_lookup=self.rel(rd["t_lookup"]), t_found=self.rel(rd["t_found"]),
                                 t_retrieved=self.rel(rd["t_retrieved"]),
                                 d_blocks=[dict(b, t_cached=self.rel(b["t_cached"])) for b in rd.get("d_blocks") or []])
                            for rd in c.get("reads") or []]
            row["events"] = sorted([(self.rel(t), nm) for nm, t, _ in c.get("l1", []) + c.get("l2", [])] +
                                   [(self.rel(t), f"{nm} ({ok})") for nm, ok, t in c.get("sm", [])])
            rows.append(row)
        return rows

    def block_rows(self):
        """One row per engine request and prompt block: its hash, GPU block and where it came from."""
        rows = []
        for (role, rid), r in self.reqs.items():
            conn = (r["lookups"] or [None])[-1] or {}
            vh = conn.get("vllm_hit", r.get("local_hit")) or 0
            ret = (r["retrieves"] or [None])[-1]
            for i, h in enumerate(r["hashes"]):
                lo = i * BLOCK
                if lo < vh:
                    src = "local prefix cache"
                elif ret is not None and ret["start"] + ret.get("skip", 0) <= lo < ret["end"]:
                    src = "LMCache"
                else:
                    src = "computed"
                cached = r["cached"].get(i)
                lives = self.lives[role].get(h) or []
                ev = first([lf["t_evicted"] for lf in lives if lf["t_evicted"] is not None])
                rows.append({"role": role, "key": r["key"], "conv": r["conv"], "turn": r["turn"], "index": i,
                             "hash": h, "chunk": lo // self.chunk, "source": src,
                             "gpu": cached[0] if cached else None, "t_cached": self.rel(cached[2]) if cached else None,
                             "t_evicted": self.rel(ev)})
        return rows

    def numbers(self, reqs, chunks):
        P = [r for r in reqs if r["role"] == "P"]
        D = [r for r in reqs if r["role"] != "P"]
        stored = [c for c in chunks if c.get("store_role")]
        reads = [rd for c in chunks for rd in c["reads"]]
        pulled = [rd for rd in reads if rd["how"] == "pulled"]
        by_tier = collections.Counter(rd["tier"] for rd in pulled)
        never = [c for c in stored if not any(rd["how"] == "pulled" for rd in c["reads"])]
        d_pull = [r for r in D if r.get("load")]
        d_pull_l1 = [r["pull_s"] for r in d_pull if r.get("pull_s") is not None and not r["pulled_by_tier"].get("L2")]
        d_pull_l2 = [r["pull_s"] for r in d_pull if r.get("pull_s") is not None and r["pulled_by_tier"].get("L2")]
        same = [b["same_as_p"] for rd in pulled for b in rd.get("d_blocks") or [] if b["same_as_p"] is not None]
        checks = [c["tokens_match_prompt"] for c in stored if c.get("tokens_match_prompt") is not None]
        chain = [c for c in stored if c.get("t_created") is not None and c.get("t_l1_written") is not None]
        races = [r for r in D if r.get("looked_up_before_store_end")]
        out = {
            "mode": self.mode, "env": {k: self.env.get(k) for k in ("mode", "l1_gb", "eviction", "store_policy",
                                                                    "chunk", "odirect", "d_role", "mtrag_limit",
                                                                    "layout", "store_wait_ms", "gpu_blocks",
                                                                    "d_prefix_cache")},
            "trace": dict(self.load_info, hook_errors=dict(self.errors), unknown_events=dict(self.unknown)),
            "requests": {"P": len(P), "D": len(D), "turns": len(self.turns)},
            "prompt_tokens": {"P": sum(r["n_prompt"] or 0 for r in P), "D": sum(r["n_prompt"] or 0 for r in D)},
            "P": {"local_hit_tokens": sum(r["local_hit"] or 0 for r in P),
                  "lmcache_loaded_tokens": sum(r["load"] or 0 for r in P),
                  "computed_tokens": sum(r["computed"] or 0 for r in P),
                  "stored_tokens": sum(r["stored_tokens"] or 0 for r in P),
                  "store_bytes": sum(r["store_bytes"] or 0 for r in P),
                  "store_s_median": q([r.get("store_s") for r in P], 0.5),
                  "store_s_p90": q([r.get("store_s") for r in P], 0.9)},
            "D": {"local_hit_tokens": sum(r["local_hit"] or 0 for r in D),
                  "lmcache_loaded_tokens": sum(r["load"] or 0 for r in D),
                  "computed_tokens": sum(r["computed"] or 0 for r in D),
                  "requests_pulling": len(d_pull),
                  "retrieve_bytes": sum(r["retrieve_bytes"] or 0 for r in D),
                  "pull_s_median_l1": q(d_pull_l1, 0.5), "pull_s_p90_l1": q(d_pull_l1, 0.9),
                  "pull_s_median_l2": q(d_pull_l2, 0.5), "pull_s_p90_l2": q(d_pull_l2, 0.9),
                  "retrieve_s_median": q([r.get("retrieve_s") for r in d_pull], 0.5),
                  "prefetch_s_median_l1": q([r.get("prefetch_s") for r in d_pull if not r["pulled_by_tier"].get("L2")], 0.5),
                  "prefetch_s_median_l2": q([r.get("prefetch_s") for r in d_pull if r["pulled_by_tier"].get("L2")], 0.5),
                  "looked_up_before_p_store_end": len(races)},
            "chunks": {"stored": len(stored), "in_memory": sum(1 for c in stored if c.get("t_l1_written") is not None),
                       "stored_bytes": sum(c.get("bytes") or 0 for c in stored),
                       "to_disk": sum(1 for c in stored if c.get("t_l2_stored") is not None),
                       "removed_from_memory": sum(1 for c in stored if c.get("t_l1_removed")),
                       "pulled": len(pulled), "pulled_from": dict(by_tier),
                       "read_outcomes": dict(collections.Counter(rd["how"] for rd in reads)),
                       "never_pulled": len(never),
                       "never_pulled_holding_an_answer": sum(1 for c in never if any(
                           str(p).startswith(("answer", "assistant")) for p in c.get("pieces") or [])),
                       "disk_reads": sum(len(c.get("t_l2_read") or []) for c in chunks),
                       "touches": sum(len(c.get("touch") or []) for c in chunks),
                       "reads_hook": self.has_read_hook,
                       "pulls_per_chunk": dict(collections.Counter(
                           sum(1 for rd in c["reads"] if rd["how"] == "pulled") for c in stored)),
                       "created_to_l1_s_median": q([c["t_l1_written"] - c["t_created"] for c in chain], 0.5),
                       "l1_to_l2_s_median": q([c["t_l2_stored"] - c["t_l1_written"] for c in stored
                                               if c.get("t_l2_stored") and c.get("t_l1_written")], 0.5),
                       "lookups_of_unknown_chunks": sum(1 for c in chunks if c.get("unknown_store"))},
            "checks": {"chunk_tokens_match_prompt": f"{sum(checks)} of {len(checks)}",
                       "pulled_blocks_same_hash_as_p": f"{sum(same)} of {len(same)}",
                       "chunks_with_full_chain": f"{len(chain)} of {len(stored)}"},
        }
        return out


# ------------------------------------------------------------------------------------------------- writing
def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def story_chunk(c, reads_hook=False):
    """One chunk's life as rows (time from the run's first record, what happened)."""
    ev = []
    for b in c.get("p_blocks") or []:
        if b.get("t_cached") is not None and b["index"] in (c["p_blocks"][0]["index"], c["p_blocks"][-1]["index"]):
            ev.append((b["t_cached"], f"P computed and cached block {b['index']} (GPU block {b['gpu']}, hash "
                                      f"{b['hash']}, request {str(b['computed_by'])[:8]})"))
    for k, what in (("t_store_planned", "P's connector planned the store"),
                    ("t_store_submitted", "P's worker sent it to the server"),
                    ("t_bound", "the server bound the chunk to its tokens (mp.tokens)"),
                    ("t_l1_reserved", "server memory (L1) reserved"), ("t_l1_written", "written to memory (L1)"),
                    ("t_l2_stored", "written to disk (L2)")):
        if c.get(k) is not None:
            ev.append((c[k], what))
    for t in c.get("t_l1_removed") or []:
        ev.append((t, "removed from memory (L1)"))
    if reads_hook:
        for t in c.get("t_l2_read") or []:
            ev.append((t, "read from disk (L2)"))
        for t in c.get("t_touched") or []:
            ev.append((t, "eviction order refreshed (a touch: no I/O)"))
    else:                                            # a trace without the read hook: a read or a touch
        for t in c.get("t_l2_accessed") or []:
            ev.append((t, "disk tier accessed (a read from disk, or a touch: no I/O)"))
    for rd in c.get("reads") or []:
        ev.append((rd["t_lookup"], f"{rd['role']} looked it up (turn {rd['turn']}): {rd['how']}"
                                   + (f", from {rd['tier']}" if rd.get("tier") else "")))
        if rd.get("t_retrieved") is not None:
            ev.append((rd["t_retrieved"], f"copied to {rd['role']}'s GPU"))
        for b in rd.get("d_blocks") or []:
            if b.get("t_cached") is not None and b["index"] == (rd["d_blocks"][0]["index"]):
                ev.append((b["t_cached"], f"{rd['role']} cached block {b['index']} (GPU block {b['gpu']}; same hash as "
                                          f"P's: {b['same_as_p']})"))
    return sorted((t, w) for t, w in ev if t is not None)


LOG_T = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[,.](\d{3})\]")


def health(run, t0):
    """Whether the LMCache server stayed up, and when each engine's connector gave up on it ("degraded mode"), from
    the run's logs (the run folder's, or a results folder's logs/*_tail.log). Log times are local wall-clock."""
    import datetime
    run = pathlib.Path(run)

    def logs(stem):
        """The run's logs: whole (the run folder's), or as a results folder keeps them (gzipped, else the tail)."""
        found = sorted(run.glob(f"{stem}.log"))
        if not found:
            found = sorted((run / "logs").glob(f"{stem}.log.gz")) or sorted((run / "logs").glob(f"{stem}_tail.log"))
        return found

    def text(f):
        if f.suffix == ".gz":
            with gzip.open(f, "rt", errors="replace") as g:
                return g.read()
        return f.read_text(errors="replace")

    def when(line):
        m = LOG_T.search(line)
        if not m:
            return None
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000
        return round(t - t0, 3)

    out = {"server_fatal": None, "server_last_log_s": None, "degraded": {}, "server_errors": 0,
           "server_error_kinds": {}}
    kinds = collections.Counter()
    for f in logs("lmcache_server"):
        last, pending = None, None
        for ln in text(f).splitlines():
            if LOG_T.search(ln):
                last = when(ln)
            if out["server_fatal"] is None and ("Fatal Python error" in ln or "Segmentation fault" in ln):
                out["server_fatal"] = ln.strip()[:200]
            if "LMCache ERROR" in ln:                  # job 30722: every store failed, the server stayed up
                out["server_errors"] += 1
                msg = re.sub(r"\x1b\[[0-9;]*m", "", ln.split("LMCache ERROR", 1)[1]).strip(" :")
                pending = re.sub(r"\s*\(\S+\.py:\d+:\S+\)\s*$", "", msg)[:120]
                out.setdefault("server_first_error_s", when(ln))
            elif pending and re.match(r"^[A-Za-z_.]*(Error|Exception)\b", ln):
                kinds[f"{pending} -> {ln.strip()[:160]}"] += 1
                pending = None
        out["server_last_log_s"] = last
    out["server_error_kinds"] = dict(kinds.most_common(5))
    for f in logs("engine_*"):
        role = f.name.split("_")[1].split(".")[0]
        for ln in text(f).splitlines():
            if "degraded mode" in ln:
                out["degraded"].setdefault(role, when(ln))
    env = run / "env.json"
    if env.is_file():
        out["server_alive_at_end"] = json.loads(env.read_text()).get("server_alive_at_end")
    out["server_crashed"] = bool(out["server_fatal"]) or out.get("server_alive_at_end") is False
    out["server_failed"] = out["server_crashed"] or out["server_errors"] > 0
    return out


def write_report(run, out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    R = Run(run)
    R.analyse()
    reqs, chunks, blocks = R.request_rows(), R.chunk_rows(), R.block_rows()
    nums = R.numbers(reqs, list(R.chunks.values()))
    hl = nums["health"] = health(run, R.t0)
    seen = [v for v in hl["degraded"].values() if v is not None]
    if seen:                                         # what came after the engines gave up on the server
        hl["requests_after_degraded"] = sum(1 for r in reqs if r["t_added"] is not None and r["t_added"] >= min(seen))
    for name, rows in (("requests", reqs), ("chunks", chunks), ("blocks", blocks)):
        with gzip.open(out / f"{name}.jsonl.gz", "wt") as fh:
            for row in rows:
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    (out / "results.json").write_text(json.dumps(nums, indent=1))
    md = [f"# LMCache lifecycle: {R.mode}", ""]
    if hl["server_errors"] and not hl["server_crashed"]:
        md += [f"**The LMCache server logged {hl['server_errors']} errors** (the first at "
               f"{fmt(hl.get('server_first_error_s'))} s): "
               + "; ".join(f"{k} ({v}x)" for k, v in hl["server_error_kinds"].items())
               + ". What failed there is missing from the lifecycle below.", ""]
    if hl["server_crashed"] or hl["degraded"]:
        md += [f"**The LMCache server did not last the run.** {hl['server_fatal'] or 'It was gone at the end.'} "
               f"Its last log line: {fmt(hl['server_last_log_s'])} s; the engines gave up on it (degraded mode) at "
               + (", ".join(f"{k} {fmt(v)} s" for k, v in sorted(hl["degraded"].items())) or "–")
               + f"; {hl.get('requests_after_degraded', '–')} engine requests came after that and ran without LMCache. "
               "Everything below is what happened before.", ""]
    e = R.env
    md += [f"1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **{R.mode}**"
           f" (memory {e.get('l1_gb')} GB, eviction {e.get('eviction')}, store policy {e.get('store_policy')}, chunk "
           f"{e.get('chunk')} tokens); MTRAG's first {e.get('mtrag_limit')} conversations, {e.get('layout')}; the "
           f"proxy waits {e.get('store_wait_ms')} ms between P's answer and D's request. D's role {e.get('d_role')}. "
           f"Commit {str(e.get('commit'))[:12]}.", ""]
    t = nums["trace"]
    md += [f"Trace: {t['records']} records from {t['files']} processes ({', '.join(f'{k} {v}' for k, v in sorted((t['by_role'] or {}).items(), key=lambda kv: str(kv[0])))}); "
           f"hook errors {sum(t['hook_errors'].values()) if t['hook_errors'] else 0}"
           + (f" ({t['hook_errors']})" if t["hook_errors"] else "") + f"; unreadable lines {t['unreadable_lines']}.", ""]
    ck = nums["chunks"]
    md += ["## What happened", "",
           "| | P | D (all three) |", "| --- | ---: | ---: |",
           f"| requests | {nums['requests']['P']} | {nums['requests']['D']} |",
           f"| prompt tokens | {nums['prompt_tokens']['P']} | {nums['prompt_tokens']['D']} |",
           f"| from the engine's own GPU prefix cache | {nums['P']['local_hit_tokens']} | {nums['D']['local_hit_tokens']} |",
           f"| loaded from LMCache | {nums['P']['lmcache_loaded_tokens']} | {nums['D']['lmcache_loaded_tokens']} |",
           f"| computed | {nums['P']['computed_tokens']} | {nums['D']['computed_tokens']} |",
           f"| stored to LMCache (tokens) | {nums['P']['stored_tokens']} | – |", ""]
    md += ["## Chunks", "",
           f"- bound to their tokens by P's stores: {ck['stored']} chunks; written to the server's memory "
           f"{ck['in_memory']} ({ck['stored_bytes'] / 1e6:.1f} MB); to disk {ck['to_disk']}; removed from memory "
           f"{ck['removed_from_memory']}",
           f"- pulled by D or P: {ck['pulled']} times, from {ck['pulled_from'] or '–'}; never pulled: {ck['never_pulled']}"
           f" ({ck['never_pulled_holding_an_answer']} of them hold an earlier answer, which D generated itself and holds)",
           (f"- reads from disk: {ck['disk_reads']}; touches (eviction order refreshed, no I/O): {ck['touches']}"
            if ck["reads_hook"] else "- reads from disk and touches are one event in this trace (before the read "
            "hook): a pull's tier comes from LMCache's access events (in the disk mode every chunk is on disk only)"),
           f"- every lookup's outcome per chunk: {ck['read_outcomes']}",
           f"- pulls per stored chunk: {dict(sorted(ck['pulls_per_chunk'].items()))}",
           f"- median time from a chunk's last block computed on P to the chunk in memory: "
           f"{fmt(ms(ck['created_to_l1_s_median']))} ms; memory to disk: {fmt(ms(ck['l1_to_l2_s_median']))} ms",
           f"- lookups of chunks never seen stored: {ck['lookups_of_unknown_chunks']}", ""]
    d = nums["D"]
    md += ["## Pulls", "",
           f"- D requests that loaded from LMCache: {d['requests_pulling']}; bytes {d['retrieve_bytes'] / 1e6:.1f} MB",
           f"- the server's lookup to the copy on D's GPU done, all from memory: median {fmt(ms(d['pull_s_median_l1']))} ms, p90 "
           f"{fmt(ms(d['pull_s_p90_l1']))} ms; with chunks from disk: median {fmt(ms(d['pull_s_median_l2']))} ms, "
           f"p90 {fmt(ms(d['pull_s_p90_l2']))} ms",
           f"- of which the server's lookup and prefetch (where a disk read happens): median "
           f"{fmt(ms(d['prefetch_s_median_l1']))} ms from memory, {fmt(ms(d['prefetch_s_median_l2']))} ms with disk",
           f"- and its copy to D's GPU (retrieve start to end): median {fmt(ms(d['retrieve_s_median']))} ms",
           f"- D requests whose lookup finished before P's store of the same turn: {d['looked_up_before_p_store_end']}",
           f"- P's stores (planned to stored): median {fmt(ms(nums['P']['store_s_median']))} ms, p90 "
           f"{fmt(ms(nums['P']['store_s_p90']))} ms", ""]
    md += ["## Checks", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in nums["checks"].items()] + [""]
    md += ["## Requests", "", "| engine | conv | turn | prompt | own GPU | LMCache hit | loaded | computed | "
           "chunks (outcome) | from | pull ms | stored |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for r in reqs:
        md.append(f"| {r['role']} | {fmt(r['conv'])} | {fmt(r['turn'])} | {fmt(r['n_prompt'])} | {fmt(r['local_hit'])} | "
                  f"{fmt(r['lmc_hit'])} | {fmt(r['load'])} | {fmt(r['computed'])} | "
                  f"{', '.join(f'{k} {v}' for k, v in r['chunks'].items()) or '–'} | "
                  f"{', '.join(f'{k} {v}' for k, v in r['pulled_by_tier'].items()) or '–'} | "
                  f"{fmt(ms(r.get('pull_s')))} | {fmt(r['stored_tokens'])} |")
    md.append("")
    pick = max(chunks, key=lambda c: (sum(1 for rd in c["reads"] if rd["how"] == "pulled"), len(c["events"])),
               default=None)
    if pick is not None:
        md += [f"## One chunk's life: {pick['hash']}", "",
               f"Conversation {pick['conv']}, turn {pick['turn']}, tokens {pick['offset']}-"
               f"{(pick['offset'] or 0) + (pick['ntok'] or 0)} ({', '.join(pick.get('pieces') or []) or '?'}); "
               f"{fmt(pick['bytes'])} bytes. Seconds from the run's first record:", "",
               "| s | what |", "| ---: | --- |"]
        md += [f"| {t:.4f} | {w} |" for t, w in story_chunk(pick, R.has_read_hook)]
        md.append("")
    dreq = max((r for r in reqs if r["role"] != "P"), key=lambda r: (r["pulled_by_tier"].get("L2", 0), r["load"] or 0),
               default=None)
    if dreq is not None:
        md += [f"## One D request: conversation {dreq['conv']}, turn {dreq['turn']} on {dreq['role']}", "",
               "| s | what |", "| ---: | --- |"]
        story = []
        for k, what in (("proxy_t_in", "the proxy received the turn"), ("proxy_t_prefill_done", "P answered (stored)"),
                        ("t_added", f"{dreq['role']} took the request"),
                        ("t_server_lookup", "the server got its lookup"),
                        ("t_lookup", "its connector had the lookup's answer"),
                        ("t_found", f"the server found {dreq['found_chunks']} of {dreq['looked_up_chunks']} chunks "
                                    f"(memory {dreq['l1_hit_tokens']}, disk {dreq['l2_hit_tokens']} tokens)"),
                        ("t_retrieve_start", "the server's copy to the GPU began"),
                        ("t_retrieve_end", f"copied ({fmt(dreq['retrieve_bytes'])} bytes)"),
                        ("t_received", "the worker saw the load finished"),
                        ("proxy_t_first_byte", "the first token reached the proxy"), ("t_finished", "finished"),
                        ("t_freed", "its GPU blocks freed (they stay cached)")):
            if dreq.get(k) is not None:
                story.append((dreq[k], what))
        md += [f"| {t:.4f} | {w} |" for t, w in sorted(story, key=lambda x: x[0])]
        md.append("")
    md += ["Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; "
           "docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).", ""]
    (out / "summary.md").write_text("\n".join(md))
    print(f"lifecycle {R.mode}: {nums['requests']} requests, {ck['stored']} chunks stored, {ck['pulled']} pulls "
          f"({ck['pulled_from']}), checks {nums['checks']} -> {out}")
    return nums


def compare(dirs, out):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in dirs:
        p = pathlib.Path(d) / "results.json"
        if p.is_file():
            runs.append((pathlib.Path(d).name, json.loads(p.read_text())))
    rows = [("LMCache server lasted the run", lambda n: not (n.get("health") or {}).get("server_crashed")),
            ("LMCache server errors", lambda n: (n.get("health") or {}).get("server_errors")),
            ("chunks bound by P's stores", lambda n: n["chunks"]["stored"]),
            ("chunks written to memory", lambda n: n["chunks"].get("in_memory")),
            ("stored MB", lambda n: round(n["chunks"]["stored_bytes"] / 1e6, 1)),
            ("chunks to disk", lambda n: n["chunks"]["to_disk"]),
            ("chunks removed from memory", lambda n: n["chunks"]["removed_from_memory"]),
            ("pulls", lambda n: n["chunks"]["pulled"]),
            ("pulled from", lambda n: n["chunks"]["pulled_from"]),
            ("never pulled", lambda n: n["chunks"]["never_pulled"]),
            ("D tokens from own GPU", lambda n: n["D"]["local_hit_tokens"]),
            ("D tokens loaded", lambda n: n["D"]["lmcache_loaded_tokens"]),
            ("D tokens computed", lambda n: n["D"]["computed_tokens"]),
            ("pull ms, memory (median)", lambda n: ms(n["D"]["pull_s_median_l1"])),
            ("pull ms, disk (median)", lambda n: ms(n["D"]["pull_s_median_l2"])),
            ("pull ms, disk (p90)", lambda n: ms(n["D"]["pull_s_p90_l2"])),
            ("lookup and prefetch ms, memory (median)", lambda n: ms(n["D"]["prefetch_s_median_l1"])),
            ("lookup and prefetch ms, disk (median)", lambda n: ms(n["D"]["prefetch_s_median_l2"])),
            ("retrieve ms (median)", lambda n: ms(n["D"]["retrieve_s_median"])),
            ("P store ms (median)", lambda n: ms(n["P"]["store_s_median"])),
            ("memory to disk ms (median)", lambda n: ms(n["chunks"]["l1_to_l2_s_median"])),
            ("D lookups before P's store ended", lambda n: n["D"]["looked_up_before_p_store_end"]),
            ("checks", lambda n: "; ".join(f"{v}" for v in n["checks"].values()))]
    md = ["# LMCache lifecycle: the modes side by side", "", "| | " + " | ".join(n["mode"] for _, n in runs) + " |",
          "| --- |" + " ---: |" * len(runs)]
    table = {}
    for label, fn in rows:
        vals = []
        for _, n in runs:
            try:
                vals.append(fn(n))
            except (KeyError, TypeError):
                vals.append(None)
        table[label] = vals
        md.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + " |")
    md += ["", "Runs: " + ", ".join(name for name, _ in runs), ""]
    (out / "compare.md").write_text("\n".join(md))
    (out / "compare.json").write_text(json.dumps({"runs": [name for name, _ in runs], "table": table}, indent=1))
    print(f"compared {len(runs)} runs -> {out / 'compare.md'}")
    return table


def unpack(results, into):
    """A published results folder (kvtrace.tar.gz, proxy.jsonl.gz, mtrag.jsonl.gz, env.json, logs/) as a run
    folder, to report on it again."""
    import shutil
    import tarfile
    results, into = pathlib.Path(results), pathlib.Path(into)
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(results / "kvtrace.tar.gz") as tf:
        for m in tf.getmembers():                   # only plain files under kvtrace/
            if m.isfile() and m.name.startswith("kvtrace/") and ".." not in m.name.split("/"):
                tf.extract(m, into, filter="data") if hasattr(tarfile, "data_filter") else tf.extract(m, into)
    for name in ("proxy.jsonl", "mtrag.jsonl"):
        if (results / f"{name}.gz").is_file():
            with gzip.open(results / f"{name}.gz", "rb") as g:
                (into / name).write_bytes(g.read())
    for name in ("env.json",):
        if (results / name).is_file():
            shutil.copyfile(results / name, into / name)
    if (results / "logs").is_dir():
        shutil.copytree(results / "logs", into / "logs", dirs_exist_ok=True)
    return into


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--run", help="a run folder (scripts/lmc_run.sh's)")
    r.add_argument("--results", help="or a published results folder, unpacked into --unpack-to first")
    r.add_argument("--unpack-to", help="where to unpack --results (default: OUTPUT/run)")
    r.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dirs", required=True, help="results folders, by commas")
    c.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        run = a.run
        if run is None:
            if not a.results:
                ap.error("report needs --run or --results")
            run = unpack(a.results, a.unpack_to or pathlib.Path(a.output) / "run")
        write_report(pathlib.Path(run), a.output)
    else:
        compare([d for d in a.dirs.split(",") if d], a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
