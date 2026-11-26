#!/usr/bin/env python3
"""The lives of KV blocks and LMCache chunks in one LMCache lifecycle run (scripts/lmc_run.sh), from its trace.

  python3 -I scripts/lmc_lifecycle.py report --run RUN_DIR --output RESULTS_DIR [--conversations MTRAG_FILE]
  python3 -I scripts/lmc_lifecycle.py conversations --run RUN_DIR|--results DIR --output DIR [--conversations FILE]
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
row per engine request: hits, loads, stores and their times), conversations/ (a page per conversation: each turn's
prompt as text, how it grows, its chunks and blocks; the text from mtrag.jsonl, or for a run whose driver kept none,
rebuilt from MTRAG_FILE).

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
# LMCache events the report reads nothing from: the connector's report of vLLM's block allocations, for the server's
# metrics (vLLM's own hooks record the allocations)
LMC_IGNORED = {"mp.vllm.block_allocation"}
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
LAYOUT_NAMES = {"history_first": "chat RAG", "interleaved": "carry-forward", "docs_first": "docs first"}
DOCS_NAMES = {"conversations": "conversations.json's passages", "rag": "the retriever's top 5 (RAG.jsonl)",
              "reference": "the relevant passages (reference.jsonl)",
              "all": "every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once)"}


UCX_TRANSPORT = re.compile(r"\b(tcp|cuda_ipc|cuda_copy|gdr_copy|rc_mlx5|rc_verbs|dc_mlx5|ud_mlx5|ud_verbs|posix|sysv|"
                           r"cma|xpmem|knem|self)/[\w:\[\].-]+")


UCX_IPC_ERROR = re.compile(r"(cuIpc\w*\(\) failed.*|peer access is not supported.*|cuda_ipc.*(?:fail|unreachable|error).*)",
                           re.I)


def ucx_transports(run, errors=None):
    """engine -> {UCX transport: table rows naming it}: what UCX chose, from its protocol tables in the engines' logs
    (UCX_PROTO_INFO=y): the run's own logs, or a published run's logs/*.log.gz. cuda_ipc is GPU to GPU (peer to
    peer over PCIe or NVLink); tcp with cuda_copy is the fallback, GPU to host, TCP, host to GPU. errors (a dict):
    filled with UCX's distinct CUDA IPC failures (UCX_LOG_LEVEL=debug logs why cuda_ipc could not be used)."""
    out = {}
    files = sorted(run.glob("engine_*.log")) + sorted((run / "logs").glob("engine_*.log.gz"))
    for f in files:
        role = f.name.split(".")[0][len("engine_"):]
        if role in out:
            continue
        seen = collections.Counter()
        try:
            with (gzip.open(f, "rt", errors="replace") if f.suffix == ".gz" else open(f, errors="replace")) as fh:
                for ln in fh:
                    if "|" in ln:
                        for m in UCX_TRANSPORT.finditer(ln):
                            seen[m.group(0)] += 1
                    if errors is not None and len(errors) < 8:
                        m = UCX_IPC_ERROR.search(ln)
                        if m:
                            errors.setdefault(m.group(1).strip()[:240], role)
        except OSError:
            continue
        if seen:
            out[role] = dict(seen.most_common())
    return out


def run_label(e, mode=None):
    """A run's name in titles: its mode, and for NIXL its servers, prompt layout and passages (runs that set them)."""
    out = f"{mode or e.get('mode', '?')} mode"
    if (e.get("transfer") or "lmcache") == "lmcache":
        return out
    out += f", NIXL, {(e.get('servers') or 'shared').replace('_', '-')} servers"
    if e.get("layout_label"):
        out += f", {LAYOUT_NAMES.get(e.get('layout'), e.get('layout'))}, {e.get('mtrag_docs')} passages"
    return out


class Run:
    def __init__(self, run):
        self.dir = pathlib.Path(run)
        self.env = json.loads((self.dir / "env.json").read_text()) if (self.dir / "env.json").is_file() else {}
        self.mode = self.env.get("mode", "?")
        self.chunk = int(self.env.get("chunk", 256))
        # how P's KV reaches D (lmcache: through the shared server; nixl: D reads P's GPU) and the LMCache servers
        self.transfer = self.env.get("transfer", "lmcache")
        self.servers = self.env.get("servers", "shared")
        self.label = run_label(self.env, self.mode)
        self.engine_server = self.env.get("engine_server") or {e: "LMC" for e in ENGINES}
        recs, self.load_info = load(self.dir)
        self.recs = dedup_stream(recs)
        self.t0 = self.recs[0]["t"] if self.recs else 0.0
        self.proxy = {r["key"]: r for r in read_jsonl(self.dir / "proxy.jsonl") if r.get("key")}
        self.turns = {r["tag"]: r for r in read_jsonl(self.dir / "mtrag.jsonl") if r.get("tag")}
        self.reqs = {}                                    # (role, rid) -> request
        self.lives = {e: collections.defaultdict(list) for e in ENGINES}   # vLLM hash -> [life]
        self.chunks = {}                                  # chunk hash (server/hash with a server per engine) -> chunk
        self.handshakes = []                              # NIXL handshakes (D's worker meets P's agent)
        self.sessions = collections.defaultdict(lambda: collections.defaultdict(list))   # server session -> events
        self.prefetch = []                                # L2 prefetch / load task events (by request_id)
        self.errors = collections.Counter()
        self.unknown = collections.Counter()
        # the lmc.l2.read hook (not in jobs up to 30728): the overlay's hooked files say so even when nothing read the
        # disk; older runs' records show it by its events
        self.has_read_hook = any(f.endswith("l2_adapters/fs_l2_adapter.py")
                                 for f in (self.env.get("overlay") or {}).get("hooked_files") or [])
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
                q_["prefix_blocks"] = r.get("blocks") or []
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
            elif ev == "nixl.match":                  # D: the prompt tokens it will read from P
                self.R(role, r["req"])["nixl_match"] = r
            elif ev == "nixl.recv_planned":           # D: its blocks the read goes into, P's it reads
                self.R(role, r["req"])["nixl_recv"] = r
            elif ev == "nixl.request_finished":       # P: its blocks held for D
                self.R(role, r["req"])["nixl_finished"] = r
            elif ev == "nixl.read":                   # D's worker posts the read
                self.R(role, r["req"]).setdefault("nixl_reads", []).append(r)
            elif ev == "nixl.handshake":
                self.handshakes.append(r)
            elif ev == "nixl.done":
                for rid in r.get("sending") or []:      # P: D has read its blocks (P frees them)
                    self.R(role, rid).setdefault("t_nixl_sent", t)
                for rid in r.get("recving") or []:      # D: the KV is in its GPU blocks
                    self.R(role, rid).setdefault("t_nixl_received", t)
                for rid in r.get("failed") or []:
                    self.R(role, rid)["nixl_failed"] = t
            elif ev.startswith("lmc."):
                self.sessions[r.get("session")]["engine:" + ev[4:]].append(r)   # engine-side LMCache events
            else:
                self.unknown[ev] += 1

    # -------------------------------------------------------------------------------- the server
    def ck(self, h, srv="LMC"):
        """A chunk's key: its hash, or server/hash when each engine has its own server (each holds its own copy)."""
        return h if self.servers == "shared" else f"{srv}/{h}"

    def C(self, h, srv="LMC"):
        k = self.ck(h, srv)
        if k not in self.chunks:
            self.chunks[k] = {"hash": h, "server": srv, "l1": [], "l2": [], "sm": []}
        return self.chunks[k]

    def chunks_of(self, h):
        """The chunk with this hash on every server that has it."""
        if self.servers == "shared":
            return [self.chunks[h]] if h in self.chunks else []
        return [c for c in self.chunks.values() if c["hash"] == h]

    def chunk_on(self, h, role):
        """The chunk with this hash on the server an engine (P, D1-D3) talks to."""
        return self.chunks.get(self.ck(h, self.engine_server.get(role, "LMC"))) if h else None

    def _server(self):
        for r in self.recs:
            srv = r.get("role") or ""
            if not (srv == "LMC" or srv.startswith("LMC_")) or not r["ev"].startswith("lmc."):
                continue
            name, t = r["ev"][4:], r["t"]
            C = lambda h, srv=srv: self.C(h, srv)                 # noqa: E731
            if name == "mp.tokens":
                hs, offs = r.get("chunk_hashes") or [], r.get("token_offsets") or []
                ntok, crcs = r.get("chunk_tokens") or [], r.get("chunk_crc") or []
                par = r.get("parent_hashes") or []
                for i, h in enumerate(hs):
                    c = C(h)
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
                    C(h)["l2"].append(("l2.read", t, None))
            elif name == "touch":                    # our hook: eviction order refreshed, no I/O
                for h in r.get("keys") or []:
                    C(h).setdefault("touch", []).append(t)
            elif name in L1_EVENTS:
                sizes = r.get("sizes") or []
                for i, h in enumerate(r.get("keys") or []):
                    C(h)["l1"].append((name, t, sizes[i] if i < len(sizes) else None))
            elif name in L2_EVENTS:
                sizes = r.get("sizes") or []
                for i, h in enumerate(r.get("keys") or []):
                    C(h)["l2"].append((name, t, sizes[i] if i < len(sizes) else None))
            elif name in SM_EVENTS:
                for k in ("succeeded_keys", "failed_keys"):
                    for h in r.get(k) or []:
                        C(h)["sm"].append((name, k.split("_")[0], t))
            elif name.startswith(("l2.prefetch", "l2.load_task", "l2.store")):
                self.prefetch.append(r)
            elif name in LMC_IGNORED:
                pass
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
            srv_role = look.get("role") or "LMC"          # the server this request's lookup went to
            for j, h in enumerate(hashes):
                c = self.chunks.get(self.ck(h, srv_role))
                if c is None:                            # looked up, never stored (nor seen in memory or on disk)
                    c = self.C(h, srv_role)
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

    def split(self, q):
        """A request's prompt: (tokens, held on its own GPU, loaded from LMCache, read from P over NIXL, computed)."""
        conn = (q["lookups"] or [None])[-1] or {}
        n = q.get("n_prompt") or 0
        vh = conn.get("vllm_hit", q.get("local_hit")) or 0
        lmc = conn.get("load") or 0
        if self.transfer == "nixl" and not q["retrieves"]:
            lmc = 0                     # MultiConnector gave the load to NIXL: LMCache's lookup loaded nothing
        nx = (q.get("nixl_recv") or {}).get("external") or 0
        return n, vh, lmc, nx, max(0, n - vh - lmc - nx)

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
            _n_, _vh, load, nixl_load, computed = self.split(r)
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
                   "local_hit": vh, "lmc_hit": conn.get("lmc_hit"), "load": load, "nixl_load": nixl_load,
                   "computed": None if n is None else computed,
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
            if self.transfer == "nixl":                  # D read P's blocks; P held them for it
                nr, nf = r.get("nixl_recv") or {}, r.get("nixl_finished") or {}
                t_read = first([x["t"] for x in r.get("nixl_reads") or []])
                row.update(nixl_blocks=len(nr.get("local") or []), t_nixl_read=self.rel(t_read),
                           t_nixl_received=self.rel(r.get("t_nixl_received")), t_nixl_sent=self.rel(r.get("t_nixl_sent")),
                           nixl_held_blocks=len(nf.get("blocks") or []) if nf.get("delay") else 0)
                if t_read is not None and r.get("t_nixl_received") is not None:
                    row["nixl_read_s"] = round(r["t_nixl_received"] - t_read, 6)
                if r.get("t_finished") is not None and r.get("t_nixl_sent") is not None:
                    row["nixl_held_s"] = round(r["t_nixl_sent"] - r["t_finished"], 6)
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
        d_nixl = [r for r in D if r.get("nixl_load")]
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
                                                                    "d_prefix_cache", "transfer", "servers",
                                                                    "routing", "concurrency", "mtrag_docs",
                                                                    "conv_ids", "disk_reads", "prefetch_policy",
                                                                    "layout_label")},
            "trace": dict(self.load_info, hook_errors=dict(self.errors), unknown_events=dict(self.unknown)),
            "requests": {"P": len(P), "D": len(D), "turns": len(self.turns)},
            "prompt_tokens": {"P": sum(r["n_prompt"] or 0 for r in P), "D": sum(r["n_prompt"] or 0 for r in D)},
            "P": {"local_hit_tokens": sum(r["local_hit"] or 0 for r in P),
                  "lmcache_loaded_tokens": sum(r["load"] or 0 for r in P),
                  "computed_tokens": sum(r["computed"] or 0 for r in P),
                  "stored_tokens": sum(r["stored_tokens"] or 0 for r in P),
                  "store_bytes": sum(r["store_bytes"] or 0 for r in P),
                  "store_s_median": q([r.get("store_s") for r in P], 0.5),
                  "store_s_p90": q([r.get("store_s") for r in P], 0.9),
                  "nixl_held_s_median": q([r.get("nixl_held_s") for r in P], 0.5)},
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
                  "looked_up_before_p_store_end": len(races),
                  "nixl_read_tokens": sum(r.get("nixl_load") or 0 for r in D), "requests_reading_nixl": len(d_nixl),
                  "nixl_read_s_median": q([r.get("nixl_read_s") for r in d_nixl], 0.5),
                  "nixl_read_s_p90": q([r.get("nixl_read_s") for r in d_nixl], 0.9),
                  "stored_tokens": sum(r["stored_tokens"] or 0 for r in D)},
            "chunks": {"stored": len(stored), "in_memory": sum(1 for c in stored if c.get("t_l1_written") is not None),
                       "stored_by": dict(collections.Counter(c["store_role"][:1] for c in stored)),
                       "per_server": dict(collections.Counter(c.get("server") for c in stored)),
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
        if self.transfer == "nixl":
            bpt = next((c["bytes"] / self.chunk for c in stored if c.get("bytes")), None)   # KV bytes a token
            rates = [r["nixl_load"] * bpt / r["nixl_read_s"] / 1e9 for r in d_nixl
                     if bpt and r.get("nixl_load") and r.get("nixl_read_s")]
            ipc_errors = {}
            ucx = ucx_transports(self.dir, ipc_errors)
            names = sorted({t.split("/")[0] for v in ucx.values() for t in v})
            out["nixl"] = {"handshakes": sum(1 for h in self.handshakes if h.get("phase") == "done"),
                           "failed_reads": sum(1 for r in self.reqs.values() if r.get("nixl_failed")),
                           "read_gb_s_median": q(rates, 0.5), "ucx_transports": ucx, "ucx_transport_names": names,
                           "gpu_to_gpu": ("cuda_ipc" in names) if ucx else None,
                           "ucx_ipc_errors": [f"{role}: {line}" for line, role in ipc_errors.items()],
                           "all_gpus_visible": bool(self.env.get("nixl_all_gpus")),
                           "ucx_tls": self.env.get("ucx_tls") or None}
        if self.env.get("store_policy") == "skip_l1":   # the disk mode: on disk, then out of memory, read from disk
            written = [c for c in stored if c.get("t_l1_written") is not None]
            order = [c for c in written if c.get("t_l2_stored") is not None and c.get("t_l1_removed")
                     and min(c["t_l1_removed"]) >= c["t_l2_stored"]]
            early = [rd for c in stored for rd in c["reads"] if rd["how"] == "pulled"
                     and (not c.get("t_l1_removed") or rd["t_lookup"] < min(c["t_l1_removed"]))]
            out["checks"]["disk_mode_on_disk_then_out_of_memory"] = f"{len(order)} of {len(written)}"
            out["checks"]["disk_mode_pulls_from_disk"] = f"{by_tier.get('L2', 0)} of {len(pulled)}"
            out["checks"]["disk_mode_pulls_while_still_in_memory"] = len(early)
            # what a read from disk left in memory: with the retain prefetch policy it stays until evicted (LRU),
            # with the default one it is temporary, removed once copied to the GPU (LMCache publishes that removal
            # as l1.keys.evicted too)
            # (traces before the read hook: the pulls from disk; LMCache's access events count lookups too)
            disk_reads = ([(c, t) for c in stored for t in c.get("t_l2_read") or []] if self.has_read_hook else
                          [(c, rd["t_lookup"]) for c in stored for rd in c["reads"]
                           if rd["how"] == "pulled" and rd["tier"] == "L2"])
            gone = [min(x for x in c["t_l1_removed"] if x >= t) - t for c, t in disk_reads
                    if any(x >= t for x in c.get("t_l1_removed") or [])]
            out["checks"]["disk_mode_reads_kept_in_memory"] = (
                f"{len(disk_reads) - len(gone)} of {len(disk_reads)} disk reads still in memory at the end"
                + (f"; {len(gone)} removed later, {ms(q(gone, 0.5))} ms after the read at the median" if gone else "")
                + f" (prefetch policy {self.env.get('prefetch_policy') or 'default'})")
            roles = self.env.get("server_roles") or ["LMC"]
            for srv in roles:                        # each server's folder (the shared one: server/ itself)
                st = self.dir / "server" if self.servers == "shared" else self.dir / "server" / srv
                tag = "" if self.servers == "shared" else f" ({srv})"
                status = next((f for f in (st / "status.txt", st / "status.json") if f.is_file()), None)
                if status is not None:
                    try:
                        l1 = json.loads(status.read_text())["storage_manager"]["l1_managers"]
                        out["checks"][f"disk_mode_server_memory_at_end{tag}"] = {
                            k: f"{v.get('total_object_count')} objects, {v.get('memory_used_bytes')} bytes"
                            for k, v in l1.items()}
                    except (ValueError, KeyError, TypeError):
                        pass
                files = st / "l2_files.tsv"
                if files.is_file():
                    sizes = [int(ln.split("\t", 1)[0]) for ln in files.read_text().splitlines() if ln.strip()]
                    out["checks"][f"disk_mode_files_at_end{tag}"] = f"{len(sizes)} files, {sum(sizes) / 1e9:.2f} GB"
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
    for f in logs("lmcache_server") + logs("lmcache_server_LMC_*"):
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


def write_report(run, out, conversations=None):
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
    md = [f"# LMCache lifecycle: {R.label}", ""]
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
    servers = ("one LMCache server shared by all four" if R.servers == "shared" else
               "an LMCache server per engine (LMC_P, LMC_D1-3)")
    if R.transfer == "nixl":
        how = ("P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU "
               "blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn "
               "with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin")
    else:
        how = (f"P stores its chunks in the server and D loads them from there; the proxy waits "
               f"{e.get('store_wait_ms')} ms between P's answer and D's request, each conversation on one D; D's role "
               f"{e.get('d_role')}")
    md += [f"1 prefill engine (P) and 3 decode engines (D1-D3), {servers}; mode **{R.mode}**"
           f" (memory {e.get('l1_gb')} GB each, eviction {e.get('eviction')}, store policy {e.get('store_policy')}, "
           f"chunk {e.get('chunk')} tokens"
           + (f", prefetch policy {e.get('prefetch_policy')}: what a disk read brings into memory "
              + ("stays until evicted" if e.get("prefetch_policy") == "retain" else "is dropped after the copy")
              if e.get("prefetch_policy") and R.mode != "memory" else "") + "); "
           + (f"MTRAG's conversations {', '.join(c[:8] for c in (e.get('conv_ids') or [])[:e.get('mtrag_limit') or None])}"
              if e.get("conv_ids") else f"MTRAG's first {e.get('mtrag_limit')} conversations")
           + f", {e.get('layout')}"
           + (f" ({LAYOUT_NAMES[e.get('layout')]})" if e.get("layout_label") and e.get("layout") in LAYOUT_NAMES else "")
           + (f", passages: {DOCS_NAMES.get(e.get('mtrag_docs'), e.get('mtrag_docs'))}" if e.get("mtrag_docs") else "")
           + (", one conversation at a time" if e.get("concurrency") == 1 else
              f", {e.get('concurrency')} conversations at a time" if e.get("concurrency") else "")
           + f". {how}. Commit {str(e.get('commit'))[:12]}.", ""]
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
           *([f"| read from P's GPU over NIXL | – | {nums['D']['nixl_read_tokens']} |"] if R.transfer == "nixl" else []),
           f"| computed | {nums['P']['computed_tokens']} | {nums['D']['computed_tokens']} |",
           f"| stored to LMCache (tokens) | {nums['P']['stored_tokens']} | {nums['D']['stored_tokens'] or '–'} |", ""]
    if R.transfer == "nixl":
        dn, nx = nums["D"], nums.get("nixl", {})
        md += ["## NIXL", "",
               f"- D requests that read from P: {dn['requests_reading_nixl']}, {dn['nixl_read_tokens']} tokens; the read "
               f"(D's worker posts it, to the KV in D's GPU blocks): median {fmt(ms(dn['nixl_read_s_median']))} ms, "
               f"p90 {fmt(ms(dn['nixl_read_s_p90']))} ms",
               f"- P held its blocks for D from its request's end to D's read done: median "
               f"{fmt(ms(nums['P']['nixl_held_s_median']))} ms",
               f"- handshakes (D's worker meets P's agent, once a pair): {nx.get('handshakes')}; failed reads: "
               f"{nx.get('failed_reads')}",
               f"- the read's speed: median {fmt(nx.get('read_gb_s_median'))} GB/s (the KV it read over its time)",
               "- UCX's transports (its protocol tables in the engines' logs): "
               + (", ".join(nx.get("ucx_transport_names") or []) or "none logged")
               + ("" if nx.get("gpu_to_gpu") is None else
                  " — GPU to GPU (cuda_ipc)" if nx["gpu_to_gpu"] else
                  " — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU")
               + f"; every GPU visible to each engine: {'yes' if nx.get('all_gpus_visible') else 'no (its own only)'}"
               + (f"; UCX_TLS={nx['ucx_tls']}" if nx.get("ucx_tls") else "; UCX_TLS unset (UCX's default, all)"),
               *([f"- UCX on CUDA IPC (its log): " + "; ".join(f"`{x}`" for x in nx["ucx_ipc_errors"])]
                 if nx.get("ucx_ipc_errors") else []), ""]
    md += ["## Chunks", "",
           f"- bound to their tokens by the engines' stores: {ck['stored']} chunks (by {ck['stored_by'] or '–'}"
           + (f"; per server {ck['per_server']}" if R.servers != "shared" else "") + "); written to memory "
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
    if R.transfer == "nixl":                         # D's KV came over NIXL: LMCache's pulls only if any happened
        md += [f"D's loads from LMCache: {d['requests_pulling']} requests (NIXL gives the load to D first; LMCache "
               "loads only what an engine's own GPU and NIXL do not have).", ""]
    md += [] if R.transfer == "nixl" and not d["requests_pulling"] else ["## Pulls", "",
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
    if R.transfer == "nixl":
        md += [f"P's stores (planned to stored): median {fmt(ms(nums['P']['store_s_median']))} ms, p90 "
               f"{fmt(ms(nums['P']['store_s_p90']))} ms.", ""]
    md += ["## Checks", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in nums["checks"].items()] + [""]
    md += ["## Requests", "", "| engine | conv | turn | prompt | own GPU | LMCache hit | "
           + ("loaded (LMCache / NIXL)" if R.transfer == "nixl" else "loaded") + " | computed | "
           "chunks (outcome) | from | pull ms | stored |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    for r in reqs:
        md.append(f"| {r['role']} | {fmt(r['conv'])} | {fmt(r['turn'])} | {fmt(r['n_prompt'])} | {fmt(r['local_hit'])} | "
                  f"{fmt(r['lmc_hit'])} | {fmt(r['load'])}"
                  + (f" / {fmt(r.get('nixl_load'))}" if R.transfer == "nixl" else "") + f" | {fmt(r['computed'])} | "
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
    md += ["One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.", ""]
    (out / "summary.md").write_text("\n".join(md))
    conversation_views(R, out / "conversations", conversations)
    print(f"lifecycle {R.label}: {nums['requests']} requests, {ck['stored']} chunks stored, {ck['pulled']} pulls "
          f"({ck['pulled_from']}), checks {nums['checks']} -> {out}")
    return nums


# ------------------------------------------------------------------------------------- one conversation's view
def _ranges(ids):
    """GPU block ids as short ranges: 410-425, or 3 ids."""
    ids = [i for i in ids if i is not None]
    if not ids:
        return "–"
    out, start, prev = [], ids[0], ids[0]
    for i in ids[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append(f"{start}-{prev}" if prev != start else f"{start}")
        start = prev = i
    out.append(f"{start}-{prev}" if prev != start else f"{start}")
    return ", ".join(out) if len(out) <= 3 else f"{out[0]} … {out[-1]} ({len(ids)} blocks)"


def _piece(p):
    kind, ref = p[0], p[1]
    if kind == "doc":
        return f"doc {str(ref)[:14]}"
    if kind in ("user", "assistant", "answer", "newline"):
        return {"user": "question", "assistant": "`assistant:`", "answer": "answer", "newline": "newline"}[kind] + \
            f" {ref}"
    return str(kind)


def _gpu_blocks(req):
    """Block index -> GPU block id on the engine, for a request's prompt: its prefix hit, then what it cached."""
    out = {i: b for i, b in enumerate(req.get("prefix_blocks") or [])}
    for i, (b, _h, _t) in (req.get("cached") or {}).items():
        out.setdefault(i, b)
    return out


def _n(k, word):
    return f"{k} {word}" + ("" if k == 1 else "s")


def _driver():
    """scripts/pd_mtrag.py, loaded from beside this script: a prompt's text is rebuilt by the code that built it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pd_mtrag_texts", pathlib.Path(__file__).resolve().parent / "pd_mtrag.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prompt_texts(R, conversations=None):
    """tag -> (each piece's text, parallel to the turn's pieces; where they came from). The driver's own (its --trace
    records keep them, the answers' too), else rebuilt by the driver's build() from the MTRAG file the run read
    (conversations: the same pinned file, SHA-256 checked by scripts/lmc_setup.sh): every piece but the answers,
    which only the engine had. A turn whose rebuilt pieces are not the run's, kind for kind and id for id, gets none."""
    out, need = {}, []
    for tag, t in R.turns.items():
        tx = t.get("texts")
        if tx and len(tx) == len(t.get("pieces") or []):
            out[tag] = (tx, "driver")
        else:
            need.append((tag, t))
    if not need or not conversations:
        return out
    import asyncio
    drv = _driver()
    convs = json.loads(pathlib.Path(conversations).read_text())

    async def tok(text, special=False):              # one placeholder token a piece: only the order and text matter
        return [0]
    for tag, t in need:
        k, turn = t.get("conv"), t.get("turn")
        if not isinstance(k, int) or not 0 <= k < len(convs) or not isinstance(turn, int) or not t.get("pieces"):
            continue
        conv = convs[k]
        if str(conv.get("conversation_id") or conv.get("id") or k) != str(t.get("conv_id")) or \
                conv.get("domain") != t.get("domain"):
            continue
        texts = []
        try:
            _ids, pieces = asyncio.run(drv.build(tok, t.get("layout"), drv.turns_of(conv), turn, [[0]] * turn, texts))
        except (ValueError, IndexError, KeyError):
            continue
        if [(p[0], p[1]) for p in pieces] == [(p[0], p[1]) for p in t["pieces"]]:
            out[tag] = (texts, "rebuilt")
    return out


WRAP = 100


def _fenced(text):
    """A text as sent, in a code block whose fence no run of backticks in it can close; lines over WRAP characters
    wrapped for reading (the text as sent has no such breaks)."""
    import textwrap
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, run + 1)
    lines = []
    for ln in text.rstrip("\n").split("\n"):
        lines += textwrap.wrap(ln, WRAP, break_on_hyphens=False, replace_whitespace=False) if len(ln) > WRAP else [ln]
    return [fence + "text"] + lines + [fence]


def _cell(text, n=70):
    s = " ".join(str(text).split()).replace("|", "\\|")
    return s if len(s) <= n else s[:n - 1] + "…"


def _chunk_span(s, e, ch):
    a, b = s // ch, (e - 1) // ch
    return f"chunk {a}" if a == b else f"chunks {a}-{b}"


def prompt_lines(pieces, texts, new_from, ch, seen_docs, turn):
    """The prompt's pieces from token new_from on, each with its tokens, chunks and text."""
    md = []
    for i, p in enumerate(pieces):
        kind, ref, s, e = p[0], p[1], p[2], p[3]
        if s < new_from:
            continue
        text = texts[i] if texts is not None else None
        where = f"tokens {s}-{e} ({e - s}) · {_chunk_span(s, e, ch)}" if e > s else f"tokens {s}-{e} (0)"
        if kind in ("assistant", "newline"):
            shown = "\\n" if kind == "newline" else (text if text is not None else "Assistant:")
            md += [f"- `{shown}` · {where}", ""]
        elif kind == "doc":
            if ref in seen_docs and seen_docs[ref][0] != turn:
                t0, n0 = seen_docs[ref]
                same = f"the same {n0} tokens" if n0 == e - s else f"NOT the same length ({n0} tokens then)"
                md += [f"- document `{ref}` · {where} · **again**: sent on [turn {t0}](#turn-{t0}) already ({same}), "
                       "here at another place in the prompt, so in other chunks: a prefix cache cannot reuse it", ""]
                continue
            seen_docs.setdefault(ref, (turn, e - s))
            md += [f"<details><summary>document <code>{ref}</code> · {where}</summary>", ""]
            md += (_fenced(text) if text is not None else ["(text not available)"]) + ["", "</details>", ""]
        elif kind == "answer":
            md.append(f"- **D's answer to turn {ref}** · {where}")
            md += [""] + (_fenced(text) if text is not None else
                          ["  its text was not kept by this run: its driver kept only the token ids (later runs keep "
                           "it: scripts/pd_mtrag.py --trace)"]) + [""]
        else:
            label = {"system": "system prompt", "user": f"question {ref}"}.get(kind, f"{kind} {ref}")
            md.append(f"- **{label}** · {where}")
            md += [""] + (_fenced(text) if text is not None else ["  (text not available)"]) + [""]
    return md


def growth_row(t, pieces, texts, prev_len, ch):
    """One row of how the prompt grows: its new tokens by what they are."""
    n = t["prompt_tokens"]
    parts = collections.Counter()
    ndocs, again, again_tok, question = 0, 0, 0, ""
    before = {p[1] for p in pieces if p[0] == "doc" and p[2] < prev_len}
    for i, p in enumerate(pieces):
        if p[2] < prev_len:
            continue
        k = p[0]
        parts[{"answer": "answer", "doc": "docs", "user": "question"}.get(k, "framing")] += p[3] - p[2]
        if k == "doc":
            ndocs += 1
            again += p[1] in before
            again_tok += (p[3] - p[2]) if p[1] in before else 0
        if k == "user" and texts is not None and texts[i]:
            question = texts[i][len("User: "):] if texts[i].startswith("User: ") else texts[i]
    q_txt = f"{parts['question']}" + (f": {_cell(question)}" if question else "")
    return (f"| {t.get('turn')} | {n:,} | +{n - prev_len:,} | {parts['answer'] or '–'} | "
            f"{_n(ndocs, 'doc')}, {parts['docs']:,}{f' ({again} sent before: {again_tok:,})' if again else ''} | {q_txt} | "
            f"{parts['framing']} | {n // ch} (+{n // ch - prev_len // ch}) |"), question


def turn_times(R, P, D, px):
    """One turn's moments, in ms from the proxy sending it to P: P's queue and prefill, LMCache's memory and disk for
    the chunks P stored, and D's start, find (with the read from disk), copy into its GPU, compute, first token, end."""
    T = px.get("t_in")
    if T is None or P is None:
        return None
    ms = lambda t: None if t is None else round((t - T) * 1000, 1)   # noqa: E731
    lo = lambda xs: min(xs) if xs else None                          # noqa: E731
    hi = lambda xs: max(xs) if xs else None                          # noqa: E731

    n, p_held, _p_load, _p_nixl, p_comp = R.split(P)
    ch = [c for c in R.chunks.values() if c.get("key") == P["key"] and c.get("store_role") == "P"]
    w = {"n": n, "p_held": p_held, "p_computed": p_comp, "chunks": len(ch),
         "p_added": ms(P.get("t_added")), "p_start": ms(P["allocs"][0]["t"] if P["allocs"] else None),
         "p_done": ms(px.get("t_prefill_done") or P.get("t_finished")),
         "mem_in": ms(lo([c["t_l1_reserved"] for c in ch if c.get("t_l1_reserved")])),
         "mem_written": ms(hi([c["t_l1_written"] for c in ch if c.get("t_l1_written")])),
         "disk": ms(hi([c["t_l2_stored"] for c in ch if c.get("t_l2_stored")])),
         "mem_out": ms(hi([c["t_l1_removed"][0] for c in ch if c.get("t_l1_removed")])),
         "p_nixl_sent": ms(P.get("t_nixl_sent"))}
    if D is not None:
        srv = D.get("server") or {}
        _n, d_held, d_load, d_nixl, d_comp = R.split(D)
        dch = [c for c in R.chunks.values() if c.get("key") == D["key"] and c.get("store_role") == D["role"]]
        w.update(d_nixl=d_nixl, d_nixl_read=ms(first([x["t"] for x in D.get("nixl_reads") or []])),
                 d_nixl_received=ms(D.get("t_nixl_received")), d_chunks=len(dch),
                 d_mem_written=ms(hi([c["t_l1_written"] for c in dch if c.get("t_l1_written")])),
                 d_disk=ms(hi([c["t_l2_stored"] for c in dch if c.get("t_l2_stored")])))
        tiers = collections.Counter(rd.get("tier") for c in R.chunks.values() for rd in c.get("reads") or []
                                    if rd["key"] == D["key"] and rd["role"] == D["role"] and rd["how"] == "pulled")
        w.update(d_role=D["role"], d_held=d_held, d_loaded=d_load, d_computed=d_comp, n_out=D.get("n_out"),
                 d_from=("disk" if tiers.get("L2") and not tiers.get("L1") else "memory" if tiers.get("L1") and
                         not tiers.get("L2") else "memory and disk" if tiers else None),
                 d_added=ms(D.get("t_added")), d_find=ms(lo([x["t"] for x in srv.get("mp.lookup") or []])),
                 d_found=ms(lo([x["t"] for x in srv.get("mp.lookup_prefetch.end") or []])),
                 d_copy=ms(lo([x["t"] for x in srv.get("mp.retrieve.start") or []])) if d_load else None,
                 d_received=ms(D.get("t_received")) if d_load else None,
                 d_first=ms(px.get("t_first_byte")), d_done=ms(D.get("t_finished") or px.get("t_out")))
    return w


def _f(x):
    return "–" if x is None else f"{x:,.0f}"


def turn_events(R, P, D, px, wait_ms):
    """Every step of one turn's KV, in time order, in ms from the proxy sending the turn to P: P's GPU memory, the
    copy into LMCache (LMCache's server reads P's GPU blocks itself, through CUDA IPC handles P registered: no NIXL
    here), LMCache's memory and disk, D's lookup and read, the copy into D's GPU blocks, and the answer."""
    T = px.get("t_in")
    if T is None or P is None:
        return []
    ch = R.chunk
    ev = []

    def add(t, where, what):
        if t is not None:
            ev.append((t, where, what))

    def gpu_side(q, name):
        conn = (q["lookups"] or [None])[-1] or {}
        vh = conn.get("vllm_hit", q.get("local_hit")) or 0
        held = [q["cached"][i][0] for i in sorted(q["cached"]) if i < vh // BLOCK] if vh else []
        r = _ranges(held or q.get("prefix_blocks") or [])
        add(q.get("t_prefix"), f"{name} GPU", f"its prefix cache holds {vh:,} of the prompt's tokens"
            + ((f": {_n(vh // BLOCK, 'block')} in GPU blocks {r}" if "(" not in r else f": GPU blocks {r}") if vh else ""))
        return conn, vh

    def mb(b):
        return f"{(b or 0) / 1e6:,.0f} MB"
    n = P.get("n_prompt") or 0
    add(T, "proxy → P", f"sends the turn ({n:,} prompt tokens; P is asked for one token: prefill only)")
    add(P.get("t_added"), "P", "the request arrives")
    conn, vh = gpu_side(P, "P")
    if conn:
        add(conn.get("t"), "P ↔ LMCache", f"LMCache has {conn.get('lmc_hit') or 0:,} of its tokens; P loads "
            f"{conn.get('load') or 0:,} and computes the rest")
    for a in P["allocs"]:
        if a.get("blocks"):
            add(a["t"], "P GPU", f"{_n(len(a['blocks']), 'new GPU block')} for the {a.get('new', 0):,} tokens it computes: "
                f"{_ranges(a['blocks'])}; the prefill runs")
    srv = P.get("server") or {}
    s0 = (P["stores"] or [None])[0]
    if s0:
        add(s0["t"], "P → LMCache", f"store planned: tokens {s0['start']:,}-{s0['end']:,}, chunks "
            f"{s0['start'] // ch}-{(s0['end'] - 1) // ch} ({s0.get('n_blocks')} blocks)")
    add(first([x["t"] for x in srv.get("mp.store.submitted") or []]), "LMCache",
        "the store request is in; it waits for P's forward pass to finish (a CUDA event)")
    chunks = [c for c in R.chunks.values() if c.get("key") == P["key"] and c.get("store_role") == "P"]
    res = [c["t_l1_reserved"] for c in chunks if c.get("t_l1_reserved")]
    add(min(res) if res else None, "LMCache memory", f"space reserved for {_n(len(chunks), 'chunk')} "
        f"({mb(sum(c.get('bytes') or 0 for c in chunks))})")
    st0, st1 = first(srv.get("mp.store.start") or []), (srv.get("mp.store.end") or [None])[-1]
    if st0:
        add(st0["t"], "P GPU → LMCache memory", "the copy starts: LMCache reads P's GPU blocks (CUDA IPC)")
    if st1:
        add(st1["t"], "P GPU → LMCache memory", f"copy done: {_n(st1.get('stored_count') or 0, 'chunk')}, "
            f"{mb(st1.get('total_bytes'))}" + (f" in {(st1['t'] - st0['t']) * 1000:.1f} ms" if st0 else ""))
    wr = [c["t_l1_written"] for c in chunks if c.get("t_l1_written")]
    add(max(wr) if wr else None, "LMCache memory", f"{_n(len(wr), 'chunk')} in memory, "
        + ("kept for later requests (D gets this turn's KV from P over NIXL)" if R.transfer == "nixl" else "ready for D")
        + "; P keeps its copy in its GPU blocks")
    nf = P.get("nixl_finished") or {}
    if nf.get("delay"):
        add(nf["t"], "P GPU", f"the request is done: P holds its {_n(len(nf.get('blocks') or []), 'block')} for D to "
            "read over NIXL, and its reply carries them (kv_transfer_params: the blocks, its NIXL agent)")
    add(px.get("t_prefill_done"), "P → proxy", "P answers its one token: the prefill is done")
    disk = [c["t_l2_stored"] for c in chunks if c.get("t_l2_stored")]
    add(max(disk) if disk else None, "LMCache disk", f"{_n(len(disk), 'chunk')} written to disk (one file each)")
    gone = [c["t_l1_removed"][0] for c in chunks if c.get("t_l1_removed") and c.get("t_l2_stored") and
            c["t_l1_removed"][0] >= c["t_l2_stored"]]
    add(max(gone) if gone else None, "LMCache memory",
        f"{len(gone)} memory cop{'y' if len(gone) == 1 else 'ies'} dropped: on disk only now")
    if D is not None:
        role = D["role"]
        if R.transfer == "nixl":
            add(D.get("t_added"), f"proxy → {role}", f"sends the turn to {role} at once, with P's kv_transfer_params "
                "(round-robin; no wait)")
        else:
            add(D.get("t_added"), f"proxy → {role}", f"after the {wait_ms:,.0f} ms store wait, sends the turn to {role}")
        dconn, dvh = gpu_side(D, role)
        dsrv = D.get("server") or {}
        add(first([x["t"] for x in dsrv.get("mp.lookup") or []]), f"{role} ↔ LMCache",
            "asks LMCache for the prompt's chunks")
        pe = first(dsrv.get("mp.lookup_prefetch.end") or [])
        if pe:
            l2 = pe.get("l2_hit_keys") or 0
            add(pe["t"], "LMCache disk → memory" if l2 else "LMCache memory",
                f"found {pe.get('hit_tokens') or 0:,} tokens it lacks"
                + (f": {_n(l2, 'chunk')} read from disk into a memory buffer" if l2 else
                   (f": {_n(pe.get('l1_hit_keys') or 0, 'chunk')} in memory" if pe.get("l1_hit_keys") else "")))
        _n_, _vh, d_lmc, d_nixl, d_comp = R.split(D)
        nm = D.get("nixl_match")
        if nm:
            add(nm["t"], f"{role} ↔ P (NIXL)", f"will read {max(0, nm.get('count') or 0):,} prompt tokens from P's GPU "
                f"(beyond the {nm.get('computed') or 0:,} its own cache holds)")
        if dconn:
            if R.transfer == "nixl":
                add(dconn.get("t"), f"{role} ↔ LMCache", f"LMCache has {dconn.get('lmc_hit') or 0:,} of its tokens; "
                    + ("it loads from LMCache" if d_lmc else "NIXL loads it (MultiConnector: the first connector "
                                                             "with tokens)"))
            else:
                add(dconn.get("t"), role, f"will load {dconn.get('load') or 0:,} tokens and compute "
                    f"{max(0, (D.get('n_prompt') or 0) - dvh - (dconn.get('load') or 0)):,}")
        nr = D.get("nixl_recv")
        if nr:
            add(nr["t"], f"{role} GPU", f"{_n(len(nr.get('local') or []), 'GPU block')} for the KV from P: "
                f"{_ranges(nr.get('local') or [])}")
        for h in R.handshakes:
            if h.get("role") == role and (D.get("t_added") or 0) - 1 <= h["t"] <= (D.get("t_finished") or 9e18):
                add(h["t"], f"{role} ↔ P (NIXL)", "handshake " + ("starts: first contact with P's NIXL agent"
                                                                  if h.get("phase") == "start" else "done"))
        for rd in D.get("nixl_reads") or []:
            add(rd["t"], f"P GPU → {role} GPU (NIXL)", f"{role} reads P's GPU blocks {_ranges(rd.get('remote') or [])} "
                f"into its {_ranges(rd.get('local') or [])} (a NIXL READ)" if rd.get("local") else
                "nothing to read (all on its GPU): notifies P")
        add(D.get("t_nixl_received"), role, f"the KV from P is in its GPU blocks ({d_nixl:,} tokens)")
        add(P.get("t_nixl_sent"), "P GPU", f"{role} has read: P frees its blocks (they stay in its prefix cache)")
        r0 = (D["retrieves"] or [None])[-1]
        if r0:
            skip = (r0.get("skip") or 0) // BLOCK
            add(r0["t"], f"{role} GPU", f"GPU blocks for the loaded KV: {_ranges((r0.get('blocks') or [])[skip:])}")
        rs, re_ = first(dsrv.get("mp.retrieve.start") or []), (dsrv.get("mp.retrieve.end") or [None])[-1]
        if rs:
            add(rs["t"], f"LMCache memory → {role} GPU", f"the copy starts: LMCache writes {role}'s GPU blocks (CUDA IPC)")
        if re_:
            add(re_["t"], f"LMCache memory → {role} GPU", f"copy done: {_n(re_.get('retrieved_count') or 0, 'chunk')}, "
                f"{mb(re_.get('total_bytes'))}" + (f" in {(re_['t'] - rs['t']) * 1000:.1f} ms" if rs else ""))
        t_lo, t_hi = D.get("t_added") or 0, D.get("t_finished") or 9e18
        buf = [t for c in R.chunks.values() for t in (c.get("t_l1_removed") or [])[1:] if t_lo <= t <= t_hi
               and any(rd["key"] == D["key"] and rd["role"] == role for rd in c.get("reads") or [])]
        add(max(buf) if buf else None, "LMCache memory", f"the read buffer dropped ({_n(len(buf), 'chunk')})")
        add(D.get("t_received"), role, "has the KV in its GPU blocks")
        dst = [s_ for s_ in D["stores"]]
        if dst:                                      # D's own LMCache connector stores (kv_both)
            dch = [c for c in R.chunks.values() if c.get("key") == D["key"] and c.get("store_role") == role]
            add(dst[0]["t"], f"{role} → LMCache", f"store planned: tokens {dst[0]['start']:,}-{dst[0]['end']:,}"
                + (f"; {_n(len(dch), 'chunk')} new to its server" if dch else "; its server has them all already"))
            wr_d = [c["t_l1_written"] for c in dch if c.get("t_l1_written")]
            add(max(wr_d) if wr_d else None, "LMCache memory", f"{_n(len(wr_d), 'chunk')} from {role} in memory")
            dk = [c["t_l2_stored"] for c in dch if c.get("t_l2_stored")]
            add(max(dk) if dk else None, "LMCache disk", f"{_n(len(dk), 'chunk')} from {role} written to disk")
        grows = [a for a in D["allocs"] if a.get("blocks")]
        if grows:                                    # the first: the prompt's rest; then one block per 16 answer tokens
            a = grows[0]
            what = (f"the {a['external']:,} tokens it loads" + (f" and the {a.get('new', 0):,} it computes"
                                                                if a.get("new") else "")) if a.get("external") else \
                f"the {a.get('new', 0):,} prompt tokens it computes"
            add(a["t"], f"{role} GPU", f"{_n(len(a['blocks']), 'new GPU block')} for {what}: {_ranges(a['blocks'])}")
            later = [b for g in grows[1:] for b in g["blocks"]]
            if later:
                add(grows[1]["t"], f"{role} GPU", f"decoding: the answer's KV takes {_n(len(later), 'new GPU block')} as it "
                    f"grows, one per 16 tokens: {_ranges(later)}, until {(grows[-1]['t'] - px['t_in']) * 1000:,.0f} ms")
        add(px.get("t_first_byte"), f"{role} → proxy", "the first answer token")
        add(D.get("t_finished"), role, "done" + (f": {D['n_out']} tokens decoded" if D.get("n_out") else "")
            + "; its blocks stay cached in its GPU for the next turn")
        add(px.get("t_out"), "proxy", "the response is complete")
    return [(round((t - T) * 1000, 1), where, what) for t, where, what in sorted(ev, key=lambda e: e[0])]


def turn_story(w, wait_ms, nixl=False):
    """A turn in one sentence: what P did, where its chunks went, what D did with them."""
    if not w:
        return ""
    s = (f"P queued until {_f(w['p_start'])} ms, prefilled {w['p_computed']:,} tokens (held {w['p_held']:,}) and was "
         f"done at {_f(w['p_done'])} ms")
    if w["chunks"]:
        s += (f"; its {_n(w['chunks'], 'new chunk')} were in LMCache's memory at {_f(w['mem_written'])} ms"
              + (f", on disk at {_f(w['disk'])} ms" + (" and out of memory" if w.get("mem_out") else "")
                 if w.get("disk") is not None else ""))
    else:
        s += "; no new chunk to store"
    if w.get("d_role") and w.get("d_nixl") is not None and nixl:
        s += (f". {w['d_role']} started at {_f(w['d_added'])} ms (no wait) holding {w['d_held']:,} tokens, read "
              f"{w['d_nixl']:,} from P's GPU over NIXL (posted at {_f(w.get('d_nixl_read'))} ms, in its GPU at "
              f"{_f(w.get('d_nixl_received'))} ms; P freed its blocks at {_f(w.get('p_nixl_sent'))} ms)")
        if w["d_loaded"]:
            s += f", loaded {w['d_loaded']:,} from LMCache"
        s += (f", computed {w['d_computed']:,} and gave its first token at {_f(w['d_first'])} ms; it decoded "
              + (f"{w['n_out']} tokens " if w.get("n_out") else "") + f"until {_f(w['d_done'])} ms")
        s += (f"; its LMCache connector stored {_n(w['d_chunks'], 'new chunk')}" if w.get("d_chunks") else "") + "."
        return s
    if w.get("d_role"):
        s += (f". After the {wait_ms:,.0f} ms store wait, {w['d_role']} started at {_f(w['d_added'])} ms holding "
              f"{w['d_held']:,} tokens")
        if w["d_loaded"]:
            s += (f", pulled {w['d_loaded']:,} from {w.get('d_from') or 'LMCache'} (found {_f(w['d_find'])}-"
                  f"{_f(w['d_found'])} ms, copied into its GPU {_f(w['d_copy'])}-{_f(w['d_received'])} ms)")
        s += (f", computed {w['d_computed']:,} and gave its first token at {_f(w['d_first'])} ms; it decoded "
              + (f"{w['n_out']} tokens " if w.get("n_out") else "") + f"until {_f(w['d_done'])} ms.")
    return s


TL_COLORS = {"wait": "#D3D1C7", "compute": "#7F77DD", "reserved": "#FAC775", "memory": "#EF9F27", "disk": "#1D9E75",
             "find": "#85B7EB", "copy": "#185FA5", "nixl": "#D85A30", "decode": "#CECBF6"}


def timeline_svg(rows, mode, wait_ms, nixl=False):
    """One conversation's turns as lanes (P, LMCache, D) on one time axis: ms from the turn's send to P. An SVG
    GitHub shows as an image: fixed colors on a white ground."""
    import html
    X0, X1, LANE, GAP = 150, 840, 14, 4
    ends = [w.get("d_first") or w.get("p_done") or 0 for _k, w in rows] + [w.get("disk") or 0 for _k, w in rows]
    A = max(500, int(max(ends, default=0) * 1.08 // 100 + 1) * 100)
    x = lambda v: X0 + (X1 - X0) * min(max(v, 0), A) / A          # noqa: E731
    out, y = [], 16

    def bar(a, b, yy, kind, label=""):
        if a is None or b is None or b <= a:
            return
        xa, xb = x(a), x(b)
        out.append(f'<rect x="{xa:.1f}" y="{yy}" width="{max(xb - xa, 1.5):.1f}" height="{LANE}" rx="2" '
                   f'fill="{TL_COLORS[kind]}"/>')
        if label and xb - xa <= len(label) * 6.2 + 8 and " " in label:
            label = label.rsplit(" ", 1)[1]          # "compute 32" -> "32" where only the number fits
        if label and xb - xa > len(label) * 6.2 + 8:
            fg = "#ffffff" if kind in ("compute", "disk", "copy", "nixl") else "#2C2C2A"
            out.append(f'<text x="{(xa + xb) / 2:.1f}" y="{yy + LANE - 3.5}" text-anchor="middle" font-size="10.5" '
                       f'fill="{fg}">{html.escape(label)}</text>')

    def text(xx, yy, s, size=11, weight=400, anchor="start", fill="#2C2C2A"):
        out.append(f'<text x="{xx:.1f}" y="{yy}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
                   f'fill="{fill}">{html.escape(s)}</text>')
    for turn, w in rows:
        text(10, y + 11, f"turn {turn} · {w['n']:,} tokens", 12, 600)
        y += 18
        text(20, y + LANE - 3, "P")
        bar(0, w.get("p_start"), y, "wait", "queued")
        bar(w.get("p_start"), w.get("p_done"), y, "compute", f"prefill {w['p_computed']:,}")
        y += LANE + GAP
        text(20, y + LANE - 3, "LMCache")
        if w["chunks"]:
            m_end = w["disk"] if w.get("disk") is not None else A
            bar(w.get("mem_in"), w.get("mem_written"), y, "reserved", "reserved")   # space held, P's copy under way
            bar(w.get("mem_written"), m_end, y, "memory", f"{_n(w['chunks'], 'chunk')} in memory")
            if w.get("disk") is not None:
                bar(w["disk"], A, y, "disk", "on disk" + (", out of memory" if w.get("mem_out") else ""))
        else:
            text(X0 + 4, y + LANE - 3, "no new chunk", 10.5, fill="#888780")
        y += LANE + GAP
        if w.get("d_role"):
            text(20, y + LANE - 3, w["d_role"])
            bar(w.get("p_done"), w.get("d_added"), y, "wait", "store wait" if not nixl else "")
            bar(w.get("d_added"), w.get("d_find"), y, "wait")
            bar(w.get("d_find"), w.get("d_found"), y, "find", "find" + ("+read" if mode != "memory" and not nixl else ""))
            bar(w.get("d_found"), w.get("d_copy"), y, "wait")              # D's scheduler picks the answer up
            bar(w.get("d_copy"), w.get("d_received"), y, "copy", "copy")
            bar(w.get("d_nixl_read"), w.get("d_nixl_received"), y, "nixl", "NIXL")
            start = (w.get("d_nixl_received") or w.get("d_received") or w.get("d_found") or w.get("d_added"))
            bar(start, w.get("d_first"), y, "compute", f"compute {w['d_computed']:,}")
            if w.get("d_first") is not None and w.get("d_done") is not None:
                bar(w["d_first"], min(w["d_done"], A), y, "decode")
                text(X1 + 4, y + LANE - 3, f"→ {w['d_done'] / 1000:.2f} s", 10.5, fill="#5F5E5A")
            y += LANE + GAP
            if w.get("d_chunks"):                    # D's own LMCache connector stored chunks too
                text(20, y + LANE - 3, f"LMCache ({w['d_role']})")
                if w.get("d_mem_written") is not None:
                    bar(w["d_mem_written"], w["d_disk"] if w.get("d_disk") is not None else A, y, "memory",
                        f"{_n(w['d_chunks'], 'chunk')} in memory")
                if w.get("d_disk") is not None:
                    bar(w["d_disk"], A, y, "disk", "on disk")
                y += LANE + GAP
        y += 10
    # the axis
    out.append(f'<line x1="{X0}" y1="{y}" x2="{X1}" y2="{y}" stroke="#888780" stroke-width="0.7"/>')
    step = 100 if A <= 1200 else 200
    for v in range(0, A + 1, step):
        out.append(f'<line x1="{x(v):.1f}" y1="{y}" x2="{x(v):.1f}" y2="{y + 4}" stroke="#888780" stroke-width="0.7"/>')
        text(x(v), y + 16, f"{v}", 10.5, anchor="middle", fill="#5F5E5A")
    text(X1, y + 30, "ms from the turn's send to P (decode runs past the edge: its end is on the right)", 10.5,
         anchor="end", fill="#5F5E5A")
    y += 46
    items = [("wait", "waiting (queue, store wait)"), ("compute", "computing (prefill, D's rest)"),
             ("reserved", "LMCache memory reserved (P's copy under way)"), ("memory", "in LMCache memory"), ("disk", "on disk"), ("find", "D asks LMCache (its lookup)" if nixl else
                                                     "D finds it" + (" (reads the disk)" if mode != "memory" else "")),
             ("copy", "copy into D's GPU"), *([("nixl", "D reads P's GPU over NIXL")] if nixl else []),
             ("decode", "decoding the answer")]
    lx = 10
    for kind, label in items:
        if kind == "disk" and mode == "memory":
            continue
        out.append(f'<rect x="{lx}" y="{y - 9}" width="10" height="10" rx="2" fill="{TL_COLORS[kind]}"/>')
        text(lx + 14, y, label, 10.5)
        lx += 14 + len(label) * 5.6 + 18
        if lx > 760:
            lx, y = 10, y + 16
    y += 14
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{y}" viewBox="0 0 900 {y}" '
            f'font-family="Helvetica, Arial, sans-serif"><rect width="900" height="{y}" fill="#ffffff"/>'
            + "".join(out) + "</svg>\n")


def conversation_views(R, out, conversations=None):
    """One page per conversation (and an index): each turn's prompt, piece by piece with its text (the new pieces:
    the rest is the previous turn's prompt), how it grew, its 256-token chunks and their 16-token blocks, and what
    happened to each chunk on P, in LMCache and on D that turn. conversations: the MTRAG file, for a run whose
    driver kept no texts (prompt_texts)."""
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ch = R.chunk
    texts_of = prompt_texts(R, conversations)
    wait_ms = float(R.env.get("store_wait_ms") or 0)
    srcs = collections.Counter(src for _tx, src in texts_of.values())
    if srcs.get("driver"):
        src_txt = ("The prompts' text is the driver's own record of what it sent (answers included); lines over "
                   f"{WRAP} characters are wrapped here for reading.")
    elif srcs.get("rebuilt"):
        src_txt = ("The prompts' text is rebuilt from the MTRAG file the run read, by the driver's own code "
                   "(scripts/pd_mtrag.py build()), and matches the run's record piece for piece; token counts and "
                   "places are the run's. D's answers were not kept by this run's driver, only their token ids. "
                   f"Lines over {WRAP} characters are wrapped here for reading.")
    else:
        src_txt = ("This run kept no prompt text: pass the MTRAG file (--conversations) to rebuild it.")
    by_key = collections.defaultdict(dict)
    for (role, _rid), q in R.reqs.items():
        if q["key"]:
            by_key[q["key"]]["P" if role == "P" else "D"] = q
    convs = collections.defaultdict(list)
    for tag, t in R.turns.items():
        key = next((k for k, px in R.proxy.items() if px.get("tag") == tag), None)
        convs[t.get("conv")].append((t, by_key.get(key, {}), key))
    mode = R.mode
    nixl = R.transfer == "nixl"
    routing = ("round-robin: each turn to the next D engine" if R.env.get("routing") == "round_robin" else
               "routed by conversation")
    index = [f"# The conversations, one by one: {R.label}", "",
             "Each page follows one MTRAG conversation turn by turn: the prompt's pieces, the 256-token chunks LMCache "
             "keeps them in, each chunk's 16 vLLM blocks, and what happened to every chunk on P, in LMCache and on the "
             "conversation's D engine. Built by scripts/lmc_lifecycle.py from the run's trace. " + src_txt, "",
             "| conversation | domain | first question | turns | last prompt | D engine | chunks stored | pulled | "
             "held by D | page |",
             "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for conv in sorted(convs, key=lambda c: (c is None, c)):
        turns = sorted(convs[conv], key=lambda x: x[0].get("turn", 0))
        head = turns[0][0]
        d_roles = sorted({x[1]["D"]["role"] for x in turns if "D" in x[1]})
        md = [f"# Conversation {conv}: {head.get('domain')} ({R.label})", "",
              (f"P's KV reaches D over NIXL (D reads P's GPU blocks); every engine's LMCache connector stores beside "
               f"it, to {'the one shared server' if R.servers == 'shared' else 'its own server'}. " if nixl else "") +
              f"{len(turns)} turns on {', '.join(d_roles) or '–'} ({routing}); prompts grow from "
              f"{head['prompt_tokens']:,} to {turns[-1][0]['prompt_tokens']:,} tokens (interleaved: each turn's prompt "
              "is the previous one, its answer, then this turn's documents and question). Chunks are 256 tokens "
              f"({ch // BLOCK} vLLM blocks of {BLOCK}); a chunk's name is LMCache's hash of its tokens and everything "
              "before them, so the same prefix has the same chunks on every turn and engine.", "",
              "Legend for the strips (one character a chunk): P: `=` on P's GPU from an earlier turn, `#` computed "
              "and stored now, `+` computed, not stored (a partial chunk). D: `=` on D's GPU from an earlier turn, "
              "`M` pulled from memory, `S` pulled from disk, `N` read from P's GPU over NIXL, `+` computed by D (a "
              "partial chunk), `?` missed.", "",
              "GPU blocks (the chunk tables): where each engine keeps the chunk's 16 vLLM blocks in its GPU KV cache, "
              "by vLLM's block ids (vLLM's paged KV cache: a block, or page, holds 16 tokens; block 0 is its reserved "
              "null block), in the prompt's block order, a run of consecutive ids written as a range: `P 1, 138-152` "
              "is the prompt's block 0 in GPU block 1 and its blocks 1-15 in GPU blocks 138-152. An engine takes a "
              "GPU block from its free list when it first computes or loads a block, and the block keeps it while "
              "cached, so a later turn that reuses the block shows the same id. Blocks that are the same in several "
              "prompts (the same tokens and everything before them) share one GPU block once the first is cached: the "
              "system prompt's first block, the same in every conversation, is the usual one. "
              "`16/16 hashes = P's`: of the blocks D pulled, how many have the same vLLM hash on D as on P.", "",
              src_txt, "",
              "| turn | prompt | chunks + tail | P: own GPU / computed / stored | D: own GPU / loaded / computed | "
              "pulled | pull ms | D first token ms |",
              "| ---: | ---: | --- | --- | --- | --- | ---: | ---: |"]
        growth = ["## How the prompt grows", "",
                  "Each turn's prompt is the previous one, token for token, and then: D's answer to the previous turn, "
                  "this turn's documents, its question and the framing (the system prompt on turn 0, `Assistant:` "
                  "and newlines). Only the new tokens can make new chunks: the previous prompt's partial chunk fills "
                  "first. A document sent before comes again at another place, so in other chunks.", "",
                  "| turn | prompt | new | previous answer | documents | question | framing | whole chunks |",
                  "| ---: | ---: | ---: | ---: | --- | --- | ---: | --- |"]
        totals = collections.Counter()
        detail = []
        prev_len = 0
        prev_ids = None
        seen_docs = {}
        first_q = ""
        tl_rows = []
        lineage_done = False
        for t, rq, key in turns:
            n = t["prompt_tokens"]
            pieces = t.get("pieces") or []
            texts = (texts_of.get(t.get("tag")) or (None,))[0]
            P, D = rq.get("P"), rq.get("D")
            nfull, tail = n // ch, n % ch
            # the chunks' names: the server's lookups of this prompt (D's, else P's)
            hashes = []
            for q in (D, P):
                lks = ((q or {}).get("server") or {}).get("mp.lookup") or []
                if lks:
                    hashes = lks[-1].get("chunk_hashes") or []
                    break

            def read_of(c, role_is_p):
                for rd in (c or {}).get("reads") or []:
                    if rd["key"] == key and ((rd["role"] == "P") == role_is_p):
                        return rd
                return None

            def hit(q):
                lk = ((q or {}).get("lookups") or [None])[-1] or {}
                return lk.get("vllm_hit", (q or {}).get("local_hit")) or 0

            p_hit, d_hit = hit(P), hit(D)
            d_load = (((D or {}).get("lookups") or [None])[-1] or {}).get("load") or 0
            ret = ((D or {}).get("retrieves") or [None])[-1]
            p_gpu, d_gpu = _gpu_blocks(P or {}), _gpu_blocks(D or {})
            if ret is not None:                      # D's blocks that LMCache loaded into
                for k, b in enumerate(ret.get("blocks") or []):
                    d_gpu.setdefault(ret["start"] // BLOCK + k, b)
            d_nixl = 0
            nx_read = first((D or {}).get("nixl_reads") or [], lambda r_: r_.get("local"))
            if D is not None:
                _a, _b, d_lmc_, d_nixl, _c = R.split(D)
                if not ret:
                    d_load = d_lmc_
                nx_local = ((D.get("nixl_recv") or {}).get("local") or [])
                for k, b in enumerate(nx_local):     # D's blocks the NIXL read went into, from its first one it lacks
                    d_gpu.setdefault(d_hit // BLOCK + k, b)
            nx_lo, nx_hi = d_hit, d_hit + d_nixl     # the prompt's tokens D read from P
            rows, p_strip, d_strip = [], [], []
            stored_now = pulled = held = 0
            tiers = collections.Counter()
            for j in range(nfull + (1 if tail else 0)):
                lo, hi = j * ch, min((j + 1) * ch, n)
                whole = hi - lo == ch
                h = hashes[j] if whole and j < len(hashes) else None
                c = R.chunk_on(h, "P")                       # the chunk on P's server, and on D's
                c_d = R.chunk_on(h, (D or {}).get("role", "P"))
                inside = []
                for p in pieces:
                    a, b = max(p[2], lo), min(p[3], hi)
                    if a < b:
                        inside.append(f"{_piece(p)}{' (new)' if p[2] >= prev_len and t.get('turn') else ''}: {b - a}")
                prd, drd = read_of(c, True), read_of(c_d, False)
                # P
                if not whole:
                    p_txt, p_ch = "computed (partial chunk: not stored)", "+"
                elif prd and prd["how"].startswith("new"):
                    p_txt, p_ch = "computed, **stored now**", "#"
                    if lo < p_hit < hi:
                        p_txt = f"computed (its first {p_hit - lo} tokens on P's GPU already), **stored now**"
                    stored_now += 1
                elif hi <= p_hit:
                    p_txt, p_ch = "on P's GPU (earlier turn)", "="
                else:
                    p_txt, p_ch = (prd or {}).get("how", "computed"), "#"
                # LMCache: the chunk on each server that has it
                def on_server(cc):
                    sreq = R.engine_of(cc.get("stored_by"))
                    txt = f"stored on turn {(sreq or {}).get('turn')}" + (f" by {sreq['role']}" if nixl and sreq else "")
                    if cc.get("t_l2_stored") is not None:
                        txt += ", on disk" + (", out of memory" if cc.get("t_l1_removed") else "")
                    return txt
                held_by = [cc for cc in R.chunks_of(h) if cc.get("t_l1_written") is not None] if h else []
                if held_by and R.servers == "shared":
                    lmc_txt = f"`{h}` {on_server(held_by[0])}"
                elif held_by:
                    lmc_txt = f"`{h}`: " + "; ".join(f"{cc['server']}: {on_server(cc)}" for cc in held_by)
                else:
                    lmc_txt = "–" if not whole else (f"`{h}` not stored" if h else "–")
                # D
                in_nixl = D is not None and d_nixl and lo < nx_hi and hi > nx_lo
                if in_nixl:
                    d_txt, d_ch = "**read from P's GPU over NIXL**", "N"
                    if lo < nx_lo:
                        d_txt += f" (its first {nx_lo - lo} tokens on D's GPU already)"
                    if hi > nx_hi:
                        d_txt += f" (its last {hi - nx_hi} computed by D)"
                    if not whole:
                        d_txt += " (partial chunk)"
                    pulled += 1
                    tiers["NIXL"] += 1
                elif not whole:
                    d_txt, d_ch = "computed by D (partial chunk)", "+"
                elif drd and drd["how"] == "pulled":
                    tier = drd.get("tier") or "?"
                    d_txt, d_ch = f"**pulled from {'disk' if tier == 'L2' else 'memory'}**", "S" if tier == "L2" else "M"
                    skip_to = (ret or {}).get("start", 0) + (ret or {}).get("skip", 0)
                    if lo < skip_to < hi:
                        d_txt += f" (its first {skip_to - lo} tokens on D's GPU already)"
                    pulled += 1
                    tiers[tier] += 1
                elif hi <= d_hit:
                    d_txt, d_ch = "on D's GPU (earlier turn" + (", its own answer)" if any(
                        s.startswith("answer") for s in inside) else ")"), "="
                    held += 1
                elif drd:
                    d_txt, d_ch = drd["how"], "?"
                else:
                    d_txt, d_ch = "–", "?"
                blocks = f"{lo // BLOCK}-{(hi - 1) // BLOCK}" if (hi - 1) // BLOCK > lo // BLOCK else f"{lo // BLOCK}"
                pg = _ranges([p_gpu.get(i) for i in range(lo // BLOCK, (hi + BLOCK - 1) // BLOCK)])
                dg = _ranges([d_gpu.get(i) for i in range(lo // BLOCK, (hi + BLOCK - 1) // BLOCK)])
                same = ""
                if drd and drd.get("d_blocks"):
                    ok = [b["same_as_p"] for b in drd["d_blocks"] if b["same_as_p"] is not None]
                    same = f"; {sum(ok)}/{len(ok)} hashes = P's" if ok else ""
                elif in_nixl and P is not None:      # D cached the blocks it read under the hashes P gave them
                    idx = [i for i in range(max(lo, nx_lo) // BLOCK, min(hi, nx_hi) // BLOCK)]
                    ok = [D["cached"][i][1] == P["cached"][i][1] for i in idx if i in D["cached"] and i in P["cached"]]
                    same = f"; {sum(ok)}/{len(ok)} hashes = P's" if ok else ""
                rows.append(f"| {j} | {lo}-{hi} | {'; '.join(inside)} | {blocks} | P {pg}; D {dg}{same} | {lmc_txt} | "
                            f"{p_txt} | {d_txt} |")
                p_strip.append(p_ch)
                d_strip.append(d_ch)
                # one chunk, block by block: the first one D read from P over NIXL in this conversation
                if (not lineage_done and in_nixl and whole and lo >= nx_lo and hi <= nx_hi and nx_read and P
                        and D is not None):
                    lineage_done = True
                    first_new = d_hit // BLOCK
                    lin = [f"### One chunk, block by block: chunk {j} of turn {t.get('turn')} (`{h}`)", "",
                           f"Tokens {lo}-{hi} ({'; '.join(inside)}). P computed its blocks; {D['role']} read them from "
                           "P's GPU over NIXL, P's block into D's block, one for one:", "",
                           "| block | tokens | P's GPU block | vLLM hash on P | D's GPU block | same hash on D |",
                           "| ---: | --- | ---: | --- | ---: | --- |"]
                    for i in range(lo // BLOCK, hi // BLOCK):
                        k = i - first_new
                        loc, rem = nx_read.get("local") or [], nx_read.get("remote") or []
                        pc, dc = P["cached"].get(i), D["cached"].get(i)
                        lin.append(f"| {i} | {i * BLOCK}-{(i + 1) * BLOCK} | {fmt(rem[k] if 0 <= k < len(rem) else None)}"
                                   f" | `{pc[1] if pc else '–'}` | {fmt(loc[k] if 0 <= k < len(loc) else None)} | "
                                   f"{(pc[1] == dc[1]) if pc and dc else '–'} |")
                    lin.append("")
                    lineage = lin
                if not lineage_done and drd and drd["how"] == "pulled" and drd.get("d_blocks") and c and c.get("p_blocks"):
                    lineage_done = True
                    pb = {b["index"]: b for b in c["p_blocks"]}
                    lin = [f"### One chunk, block by block: chunk {j} of turn {t.get('turn')} (`{h}`)", "",
                           f"Tokens {lo}-{hi} ({'; '.join(inside)}). P computed its blocks on turn "
                           f"{(R.engine_of(c.get('stored_by')) or {}).get('turn')} and stored them as one chunk; "
                           f"{D['role']} pulled the chunk into these GPU blocks:", "",
                           "| block | tokens | P's GPU block | vLLM hash | D's GPU block | D wrote it | same hash on D |",
                           "| ---: | --- | ---: | --- | ---: | --- | --- |"]
                    for b in drd["d_blocks"]:
                        p = pb.get(b["index"], {})
                        lin.append(f"| {b['index']} | {b['index'] * BLOCK}-{(b['index'] + 1) * BLOCK} | "
                                   f"{fmt(p.get('gpu'))} | `{p.get('hash')}` | {fmt(b.get('gpu'))} | "
                                   f"{'yes' if b.get('written') else 'no (on its GPU already)'} | {b.get('same_as_p')} |")
                    lin.append("")
                    lineage = lin
            pull_ms = None
            if D is not None:
                srv = D.get("server") or {}
                lk = first([x["t"] for x in srv.get("mp.lookup") or []])
                re_ = (srv.get("mp.retrieve.end") or [None])[-1]
                if lk is not None and re_ is not None and d_load:
                    pull_ms = (re_["t"] - lk) * 1000
                t_nr = first([x["t"] for x in D.get("nixl_reads") or []])
                if d_nixl and t_nr is not None and D.get("t_nixl_received") is not None:
                    pull_ms = (D["t_nixl_received"] - t_nr) * 1000      # the NIXL read, posted to done
            px = R.proxy.get(key, {})
            ft = None
            if D is not None and px.get("t_first_byte") and D.get("t_added"):
                ft = (px["t_first_byte"] - D["t_added"]) * 1000
            p_comp = max(0, n - p_hit)
            d_comp = max(0, n - d_hit - d_load - d_nixl)
            md.append(f"| [{t.get('turn')}](#turn-{t.get('turn')}) | {n:,} | {nfull} + {tail} | {p_hit:,} / {p_comp:,} / "
                      f"{_n(stored_now, 'chunk')} | {d_hit:,} / {d_load + d_nixl:,} / {d_comp:,} | {pulled} chunks"
                      f"{' (' + ', '.join(({'L2': 'disk', 'NIXL': 'NIXL'}.get(k, 'memory')) + f' {v}' for k, v in tiers.items()) + ')' if tiers else ''}"
                      f" | {fmt(pull_ms, 1)} | {fmt(ft, 1)} |")
            totals.update(stored=stored_now, pulled=pulled, held=held)
            g_row, question = growth_row(t, pieces, texts, prev_len, ch)
            growth.append(g_row)
            if not t.get("turn"):
                first_q = question
            if t.get("turn") and prev_len:
                ids = t.get("ids")
                same = ("checked: its first {:,} token ids are the previous prompt's".format(prev_len)
                        if ids is not None and prev_ids is not None and ids[:prev_len] == prev_ids else
                        "NOT the previous prompt's ids" if ids is not None and prev_ids is not None else
                        "not checked: the run kept no ids")
                whole = prev_len // ch
                kept = ("no whole chunk yet" if not whole else "chunk 0 is the same chunk" if whole == 1 else
                        f"chunks 0-{whole - 1} are the same chunks")
                carried = [f"Tokens 0-{prev_len:,}: [turn {t.get('turn') - 1}](#turn-{t.get('turn') - 1})'s prompt, "
                           f"unchanged ({same}), so {kept}"
                           + (f"; chunk {whole} (partial then, {prev_len % ch} tokens) fills now" if prev_len % ch
                              else "") + ". New from token {:,}:".format(prev_len), ""]
            else:
                carried = ["The whole prompt, piece by piece:", ""]
            prompt_md = ["### The prompt", ""] + carried + prompt_lines(pieces, texts, prev_len if t.get("turn") else 0,
                                                                        ch, seen_docs, t.get("turn"))
            w = turn_times(R, rq.get("P"), rq.get("D"), R.proxy.get(key, {}))
            if w:
                tl_rows.append((t.get("turn"), w))
            evs = turn_events(R, rq.get("P"), rq.get("D"), R.proxy.get(key, {}), wait_ms)
            how_copies = (
                "P's KV reaches D over NIXL: D's worker reads P's GPU blocks into its own (a NIXL READ, P told when "
                "D is done); LMCache's copies (to and from each server's memory) its servers do themselves through "
                "CUDA IPC handles each engine registered at start." if nixl else
                "The copies are LMCache's: its server reads P's GPU blocks and writes D's through CUDA IPC handles "
                "each engine registered at start (no NIXL in this setup).")
            events_md = (["### Step by step", "",
                          "Every move of this turn's KV, ms from the proxy sending the turn to P. " + how_copies, "",
                          "| ms | where | what |", "| ---: | --- | --- |"]
                         + [f"| {ms:,.1f} | {where} | {what} |" for ms, where, what in evs] + [""]) if evs else []
            detail += [f"## Turn {t.get('turn')}", "",
                       *(["**Timeline:** " + turn_story(w, wait_ms, nixl) + " ([all turns](#timeline))", ""] if w else []),
                       *events_md,
                       f"{n:,} prompt tokens: {_n(nfull, 'whole chunk')} and {tail} tokens over (the partial chunk, computed "
                       "by both engines and stored only when a later turn completes it). "
                       f"P held {p_hit:,} tokens on its GPU, computed {p_comp:,} and stored {_n(stored_now, 'new chunk')}; "
                       f"{(D or {}).get('role', 'D')} held {d_hit:,}, "
                       + (f"read {d_nixl:,} from P over NIXL, " if nixl else "")
                       + f"loaded {d_load:,} from LMCache and computed {d_comp:,}.",
                       ""] + prompt_md + ["### Pieces, chunks and blocks", "",
                                          "| piece | tokens | length | chunks | |", "| --- | --- | ---: | --- | --- |"]
            for p in pieces:
                detail.append(f"| {_piece(p)} | {p[2]}-{p[3]} | {p[3] - p[2]} | "
                              f"{_chunk_span(p[2], p[3], ch).split(' ', 1)[1] if p[3] > p[2] else '–'} | "
                              f"{'new this turn' if p[2] >= prev_len and t.get('turn') else ''} |")
            detail += ["", "```", f"P  {''.join(p_strip)}", f"D  {''.join(d_strip)}", "```", "",
                       "| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | "
                       "D this turn |", "| ---: | --- | --- | --- | --- | --- | --- | --- |"] + rows + [""]
            prev_len = n
            prev_ids = t.get("ids")
        timeline = []
        if tl_rows:
            svg_name = f"conversation_{conv}_timeline.svg"
            (out / svg_name).write_text(timeline_svg(tl_rows, mode, wait_ms, nixl))
            nixl_intro = ("Every turn on one clock: ms from the moment the proxy sent the turn to P. P prefills it (one "
                          "token out) and holds its blocks for D; the proxy sends the turn to the next D engine at once "
                          "with P's reply's kv_transfer_params (no wait), D reads P's blocks over NIXL (orange-red), "
                          "computes the rest and decodes the answer. Beside it every engine's LMCache connector stores "
                          "its new whole chunks (P's lane, and D's own when it stored any). The lanes: P, LMCache (P's "
                          "chunks), D, and LMCache (D's chunks).")
            store_intro = (
                "Every turn on one clock: ms from the moment the proxy sent the turn to P. P prefills it (one "
                "token out) and stores its new whole chunks in LMCache; the proxy waits "
                f"{wait_ms:,.0f} ms after P is done (the store wait) and sends the turn to "
                f"{', '.join(d_roles) or 'D'}, which finds what LMCache has"
                + (" (reading it from disk)" if mode != "memory" else "") + ", copies it into its GPU, "
                "computes the rest and decodes the answer. The lanes: P, LMCache (the chunks P stored this "
                "turn) and D. P computes the KV into its own GPU blocks, where it stays (the next turn reuses "
                "it); its LMCache connector, set to store (P runs as kv_both), copies each new whole chunk "
                "from those blocks to LMCache: LMCache reserves the space when P schedules the prefill (light "
                "orange) and has the chunks as P's prefill completes (orange).")
            timeline = ["## Timeline", "", nixl_intro if nixl else store_intro, "",
                        f"![timeline of conversation {conv}]({svg_name})", ""]
            if nixl:
                timeline += ["| turn | P queued → done | P's chunks: in memory / on disk | D | D starts | NIXL read | "
                             "D: held / from P / from LMCache / computed | D's chunks stored | first token | P freed | "
                             "D done |",
                             "| ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
                for turn, w in tl_rows:
                    timeline.append(
                        f"| {turn} | {_f(w['p_start'])} → {_f(w['p_done'])} | {w['chunks']}: {_f(w['mem_written'])} / "
                        f"{_f(w.get('disk'))} | {w.get('d_role', '–')} | {_f(w.get('d_added'))} | "
                        f"{_f(w.get('d_nixl_read'))}-{_f(w.get('d_nixl_received'))} | {w.get('d_held', 0):,} / "
                        f"{w.get('d_nixl') or 0:,} / {w.get('d_loaded') or 0:,} / {w.get('d_computed', 0):,} | "
                        f"{w.get('d_chunks') or 0} | {_f(w.get('d_first'))} | {_f(w.get('p_nixl_sent'))} | "
                        f"{_f(w.get('d_done'))} |")
                timeline += ["", "All times in ms from the turn's send to P. NIXL read: D's worker posts the read of "
                             "P's blocks to the KV in D's GPU blocks; P freed: D has read, P's blocks go back to its "
                             "free queue (still cached). Each turn's section below tells it in a sentence and step by "
                             "step."]
            else:
                timeline += ["| turn | P queued → done | chunks stored | in memory | on disk | D starts | D finds"
                             + (" (+ disk read)" if mode != "memory" else "") + " | copy into D's GPU | D: held / pulled / "
                             "computed | first token | D done |",
                             "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |"]
                for turn, w in tl_rows:
                    pulled = f"{w.get('d_loaded') or 0:,}" + (f" from {w['d_from']}" if w.get("d_from") else "")
                    timeline.append(
                        f"| {turn} | {_f(w['p_start'])} → {_f(w['p_done'])} | {w['chunks']} | {_f(w['mem_written'])} | "
                        f"{_f(w.get('disk'))}{' (out of memory)' if w.get('mem_out') and w.get('disk') is not None else ''} | "
                        f"{_f(w.get('d_added'))} | {_f(w.get('d_find'))}-{_f(w.get('d_found'))} | "
                        + (f"{_f(w.get('d_copy'))}-{_f(w.get('d_received'))}" if w.get("d_loaded") else "–")
                        + f" | {w.get('d_held', 0):,} / {pulled} / {w.get('d_computed', 0):,} | {_f(w.get('d_first'))} | "
                        f"{_f(w.get('d_done'))} |")
                timeline += ["", "All times in ms from the turn's send to P. \"in memory\": the last of the turn's chunks "
                         "written into LMCache's memory; \"on disk\": the last one written to disk (then dropped from "
                         "memory, in the disk mode); D finds: from LMCache's lookup to its prefetch done (in the disk "
                         "mode that is the read from disk); copy: LMCache's retrieve into D's GPU blocks until D's "
                         "connector has it. Each turn's section below tells it in a sentence."]
        md += [""] + growth + [""] + timeline + [""] + (lineage if lineage_done else []) + detail
        name = f"conversation_{conv}.md"
        (out / name).write_text("\n".join(md))
        index.append(f"| {conv} | {head.get('domain')} | {_cell(first_q, 50) or '–'} | {len(turns)} | "
                     f"{turns[-1][0]['prompt_tokens']:,} | "
                     f"{', '.join(d_roles)} | {totals['stored']} | {totals['pulled']} | {totals['held']} | [{name}]({name}) |")
    (out / "index.md").write_text("\n".join(index + [""]))
    return sorted(c for c in convs if c is not None)


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
            ("P to D", lambda n: (n.get("env") or {}).get("transfer") or "lmcache"),
            ("LMCache servers", lambda n: (n.get("env") or {}).get("servers") or "shared"),
            ("prompt layout", lambda n: (n.get("env") or {}).get("layout")),
            ("passages", lambda n: (n.get("env") or {}).get("mtrag_docs")),
            ("conversations", lambda n: (n.get("env") or {}).get("mtrag_limit")),
            ("disk reads", lambda n: (n.get("env") or {}).get("disk_reads")),
            ("chunks bound by the engines' stores", lambda n: n["chunks"]["stored"]),
            ("chunks stored by", lambda n: n["chunks"].get("stored_by")),
            ("chunks written to memory", lambda n: n["chunks"].get("in_memory")),
            ("stored MB", lambda n: round(n["chunks"]["stored_bytes"] / 1e6, 1)),
            ("chunks to disk", lambda n: n["chunks"]["to_disk"]),
            ("chunks removed from memory", lambda n: n["chunks"]["removed_from_memory"]),
            ("pulls", lambda n: n["chunks"]["pulled"]),
            ("pulled from", lambda n: n["chunks"]["pulled_from"]),
            ("never pulled", lambda n: n["chunks"]["never_pulled"]),
            ("D tokens from own GPU", lambda n: n["D"]["local_hit_tokens"]),
            ("D tokens loaded from LMCache", lambda n: n["D"]["lmcache_loaded_tokens"]),
            ("D tokens read from P over NIXL", lambda n: n["D"].get("nixl_read_tokens")),
            ("NIXL read ms (median)", lambda n: ms(n["D"].get("nixl_read_s_median"))),
            ("NIXL read ms (p90)", lambda n: ms(n["D"].get("nixl_read_s_p90"))),
            ("NIXL read GB/s (median)", lambda n: (n.get("nixl") or {}).get("read_gb_s_median")),
            ("UCX transports", lambda n: ", ".join((n.get("nixl") or {}).get("ucx_transport_names") or []) or None),
            ("P held its blocks for D, ms (median)", lambda n: ms(n["P"].get("nixl_held_s_median"))),
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
    def label(n):
        e = n.get("env") or {}
        out = n["mode"] + ("" if (e.get("transfer") or "lmcache") == "lmcache" else
                           f", NIXL, {(e.get('servers') or 'shared').replace('_', '-')}")
        return out + (f", {LAYOUT_NAMES.get(e.get('layout'), e.get('layout'))}" if e.get("layout_label") else "")
    md = ["# LMCache lifecycle: the runs side by side", "", "| | " + " | ".join(label(n) for _, n in runs) + " |",
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
    for sub in ("logs", "server"):
        if (results / sub).is_dir():
            shutil.copytree(results / sub, into / sub, dirs_exist_ok=True)
    return into


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--run", help="a run folder (scripts/lmc_run.sh's)")
    r.add_argument("--results", help="or a published results folder, unpacked into --unpack-to first")
    r.add_argument("--unpack-to", help="where to unpack --results (default: OUTPUT/run)")
    r.add_argument("--output", required=True)
    r.add_argument("--conversations", help="the MTRAG file the run read: the prompts' text for a run that kept none")
    v = sub.add_parser("conversations", help="only the per-conversation pages")
    v.add_argument("--run")
    v.add_argument("--results")
    v.add_argument("--unpack-to")
    v.add_argument("--output", required=True)
    v.add_argument("--conversations", help="the MTRAG file the run read: the prompts' text for a run that kept none")
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
        write_report(pathlib.Path(run), a.output, a.conversations)
    elif a.cmd == "conversations":
        run = a.run or unpack(a.results, a.unpack_to or pathlib.Path(a.output) / "run")
        R = Run(run)
        R.analyse()
        print("conversations:", conversation_views(R, a.output, a.conversations), "->", a.output)
    else:
        compare([d for d in a.dirs.split(",") if d], a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
