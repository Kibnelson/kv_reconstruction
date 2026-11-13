"""A CPU tier for KV (the role LMCache's CPU backend plays): chunks of earlier requests' KV kept in pinned host
memory, each in a codec's format, within a byte budget (least recently used out first), and loaded back to the GPU
when a later request starts with the same tokens.

  chunks      a request's prompt in chunks of `chunk` tokens (256 as LMCache's default); chunk i is keyed by the
              chained hash of every token up to its end, so a hit means the whole prefix is the same
  store       when a request finishes, its full chunks not in the tier are gathered from the paged cache (inside
              the engine step, then the blocks go back to vLLM), and coded and copied to host memory by a
              background thread (not in the engine's step)
  load        a request whose first chunks are in the tier waits for them (vLLM's WAITING_FOR_REMOTE_KVS) with its
              blocks allocated; the worker copies each chunk to the GPU and decodes it into those blocks
  budget      the stored bytes (the codec's coded size, or what a tier-only codec keeps in host memory) are what the
              budget limits: a smaller format keeps more chunks, which is its gain besides the shorter copy

The engine's scheduler and its worker share one tier object: vLLM's single-GPU executor runs both in the engine
process (the runner passes --distributed-executor-backend uni).
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict

_TIERS = {}
_LOCK = threading.Lock()


def get_tier(name, budget_bytes, chunk):
    """The process's tier of that name (made on first use)."""
    with _LOCK:
        t = _TIERS.get(name)
        if t is None:
            t = _TIERS[name] = HostTier(budget_bytes, chunk)
        return t


def chunk_hashes(tokens, chunk):
    """Chained hashes of the full chunks of tokens: hash i covers tokens 0 .. (i + 1) x chunk - 1."""
    out, h = [], b""
    for a in range(0, len(tokens) - len(tokens) % chunk, chunk):
        h = hashlib.blake2b(h + json.dumps(list(tokens[a:a + chunk])).encode("ascii"), digest_size=12).digest()
        out.append(h.hex())
    return out


class HostTier:
    def __init__(self, budget_bytes, chunk=256):
        self.budget = int(budget_bytes)
        self.chunk = int(chunk)
        self.entries = OrderedDict()                 # hash -> {"codec", "header", "data", "bytes", "t0", "steps"}
        self.bytes = 0
        self.lock = threading.Lock()
        self.stats = {"stored": 0, "evicted": 0, "rejected": 0, "hits": 0, "hit_tokens": 0, "lookups": 0,
                      "loaded_bytes": 0}
        self.generation = 0

    def __contains__(self, h):
        with self.lock:
            return h in self.entries

    def prefix_hits(self, hashes):
        """How many leading chunks are in the tier (touched as recently used)."""
        n = 0
        with self.lock:
            self.stats["lookups"] += 1
            for h in hashes:
                if h not in self.entries:
                    break
                self.entries.move_to_end(h)
                n += 1
        return n

    def put(self, h, entry, nbytes):
        """Keep one chunk; least recently used chunks leave until it fits. False when it alone exceeds the budget
        or is there already."""
        nbytes = int(nbytes)
        with self.lock:
            if h in self.entries:
                self.entries.move_to_end(h)
                return False
            if nbytes > self.budget:
                self.stats["rejected"] += 1
                return False
            while self.bytes + nbytes > self.budget and self.entries:
                _, old = self.entries.popitem(last=False)
                self.bytes -= old["bytes"]
                self.stats["evicted"] += 1
            self.entries[h] = dict(entry, bytes=nbytes)
            self.bytes += nbytes
            self.stats["stored"] += 1
            return True

    def get(self, h):
        with self.lock:
            e = self.entries.get(h)
            if e is not None:
                self.entries.move_to_end(h)
                self.stats["hits"] += 1
                self.stats["loaded_bytes"] += e["bytes"]
            return e

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.bytes = 0
            self.generation += 1
            for k in self.stats:
                self.stats[k] = 0

    def snapshot(self):
        with self.lock:
            return dict(self.stats, chunks=len(self.entries), bytes=self.bytes, budget=self.budget, chunk=self.chunk)
