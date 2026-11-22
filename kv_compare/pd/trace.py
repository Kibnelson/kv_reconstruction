"""Page trace: what KV pages a PD run moves, by content, so the reuse opportunity can be measured afterwards
(scripts/pd_trace_report.py).

A page is one block of `bs` tokens (vLLM's block, 16) at one layer, its keys and values together. Its content hash
is two weighted sums of its int16 words with fixed odd 31-bit weights (exact integers, no overflow: |word| < 2^15,
weight < 2^31, at most 2^15 terms, so each sum stays under 2^61), folded into one 64-bit value. Equal pages always
hash equal; two different pages collide with probability about 2^-60, so a trace of millions of pages has none.

  P (trace on for the request): one "pages" record per piece it sends: the content hash of every page.
  D (trace_pages in its config): one "held" record per piece that carries tokens D already held in its prefix
     cache: for each such page, whether D's own bytes equal what P sent (P computed them again).
The records go to their own files (trace_<role>_<pid>_<n>.jsonl, a new one every ~60 MB), beside the connector log.
"""
from __future__ import annotations

import base64
import json
import os
import threading
import time

import torch

_WEIGHTS: dict = {}
SEED = 0x5EED_7AC3
FOLD = 0x9E3779B97F4A7C15


def _weights(n, device):
    """Two fixed [n] int64 weight vectors, odd and below 2^31 (the same on every run and device)."""
    key = (str(device), int(n))
    w = _WEIGHTS.get(key)
    if w is None:
        g = torch.Generator(device="cpu").manual_seed(SEED)
        w = (torch.randint(0, 2 ** 30, (2, int(n)), generator=g, dtype=torch.int64) * 2 + 1).to(device)
        _WEIGHTS[key] = w
    return w


def page_hashes(words, bs):
    """[L, nb] uint64 (numpy) content hashes of words [L, T, W] (int16), per block of bs tokens and layer; the last
    block may be partial (its tokens only)."""
    import numpy as np
    L, T, W = words.shape
    full, rest = divmod(T, bs)
    nb = full + (1 if rest else 0)
    w = _weights(bs * W, words.device)
    out = torch.empty(2, L, nb, dtype=torch.int64, device=words.device)
    for l in range(L):
        x = words[l]
        if full:
            xb = x[:full * bs].reshape(full, bs * W).to(torch.int64)
            out[0, l, :full] = (xb * w[0]).sum(1)
            out[1, l, :full] = (xb * w[1]).sum(1)
        if rest:
            xr = x[full * bs:].reshape(1, rest * W).to(torch.int64)
            out[0, l, full] = (xr * w[0, :rest * W]).sum(1)[0]
            out[1, l, full] = (xr * w[1, :rest * W]).sum(1)[0]
    h = out.cpu().numpy().astype(np.uint64)
    with np.errstate(over="ignore"):
        return h[0] * np.uint64(FOLD) + h[1]


def equal_pages(a, b, bs):
    """[L, nb] bool (numpy): per block of bs tokens and layer, whether words a and b ([L, T, W], the same T) are
    equal everywhere."""
    L, T, W = a.shape
    full, rest = divmod(T, bs)
    eq = []
    if full:
        eq.append((a[:, :full * bs] == b[:, :full * bs]).reshape(L, full, bs * W).all(-1))
    if rest:
        eq.append((a[:, full * bs:] == b[:, full * bs:]).reshape(L, 1, rest * W).all(-1))
    return torch.cat(eq, 1).cpu().numpy()


def pack_u64(h):
    """base64 of a uint64 array (little endian), row-major."""
    return base64.b64encode(h.astype("<u8").tobytes()).decode("ascii")


def unpack_u64(s, shape):
    import numpy as np
    return np.frombuffer(base64.b64decode(s), dtype="<u8").reshape(shape)


def pack_bits(m):
    import numpy as np
    return base64.b64encode(np.packbits(m.astype(bool), axis=None).tobytes()).decode("ascii")


def unpack_bits(s, shape):
    import numpy as np
    n = int(np.prod(shape))
    return np.unpackbits(np.frombuffer(base64.b64decode(s), dtype=np.uint8))[:n].astype(bool).reshape(shape)


class TraceLog:
    """Trace records, one JSON per line, in files of their own beside the connector log (each under ~60 MB, so
    every packaged file stays far below GitHub's limit)."""

    def __init__(self, log_dir, role, limit=60 << 20):
        self.dir, self.role, self.limit = log_dir, role, int(limit)
        self.n, self.size, self.f = 0, 0, None
        self.lock = threading.Lock()

    def _open(self):
        if self.f is not None:
            self.f.close()
        os.makedirs(self.dir, exist_ok=True)
        self.f = open(os.path.join(self.dir, f"trace_{self.role}_{os.getpid()}_{self.n}.jsonl"), "a")
        self.n += 1
        self.size = 0

    def write(self, rec):
        if not self.dir:
            return
        line = json.dumps(dict(rec, time=rec.get("time", time.time()))) + "\n"
        with self.lock:
            if self.f is None or self.size + len(line) > self.limit:
                self._open()
            self.f.write(line)
            self.f.flush()
            self.size += len(line)


# ------------------------------------------------------------------------------------------- relations between pages
# Whether a request's KV is close to KV an earlier request already made, for the same text at another place (after
# another prefix, at another position): the opportunity of a differential store, which keeps such KV as a small delta
# against its base instead of in full.
#
# P matches every traced request's tokens to the earlier traced requests' (RelationIndex, in the send path, tokens
# only): a token is related when its 16-token window (it and the 15 before it) appeared before; runs of related tokens
# whose bases follow each other are spans, and a span at the same place after the same prefix is left out (a prefix
# cache's hit: kind 1 of the page trace). A thread of its own (RelationProbe) then measures, per span, layer, keys
# and values apart, what coding the request's words against its base's would cost, in bits per value:
#   xor    the XOR of the two words: its bit length entropy-coded (18 lengths) plus the bits below its leading one
#   diff   the difference of the two words as ordered integers (BF16 values one ULP apart differ by 1, across zero
#          too): its bit length, the bits below the leading one and a sign
#   rot    keys only: the base key turned to the new position first (RoPE by the shift, in float32, rounded to BF16;
#          a key is RoPE(position) applied to a vector that depends on the text), then diff
#   alone  without a base: the entropy of sign and exponent (9 bits) plus 7 mantissa bits, about what the exponent
#          coders reach (SplitZip, ExANS, ...), for the same tokens
# Measured is a sample, chosen by content: the tokens whose window hash is 0 modulo trace_sample (8), so a sampled
# token's base was sampled too (the same window). Only those go to the host: as targets (at most trace_max_tokens a
# request) and as bases, kept within trace_store_gb, only tokens whose window was new (a repeat's base is its first
# occurrence), oldest dropped first; a sampled token whose base is gone is counted, not measured. Every related token
# is counted (the token index holds every request whole). When the words waiting for the probe pass trace_queue_gb
# (4), a request is related but not measured ("relations_skipped"): P never waits for the probe.
# Requests are related within their run only (the driver's run name: the conversation id before ":"). Totals per
# group (the run and the kind of piece the token lies in, from the driver's piece list): "delta_stats" records,
# every 20 requests or 30 s and at the end. Per request: "relations".

NBL = 18                                             # bit lengths 0..17
WINDOW = 16


def bit_length(v):
    """Exact bit lengths of non-negative integers below 2^24 (0 for 0), as int64."""
    return torch.frexp(v.to(torch.float32)).exponent.to(torch.int64)


def ordered(w):
    """16-bit BF16 patterns (any integer dtype) as int32 in value order: -0 and +0 next to each other, one ULP 1."""
    u = w.to(torch.int32) & 0xFFFF
    return torch.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)


def rope_inv_freq(source, head_dim):
    """[head_dim / 2] float64 inverse frequencies of the model's RoPE, from its config (a dict, or a folder with
    config.json); Llama-3's scaling applied. None if there is no config or another scaling."""
    import math
    cfg = source
    if not isinstance(source, dict):
        try:
            cfg = json.load(open(os.path.join(source, "config.json")))
        except (OSError, ValueError, TypeError):
            return None
    sc = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    theta = cfg.get("rope_theta") or sc.get("rope_theta")
    if not theta:
        return None
    theta = float(theta)
    inv = [1.0 / theta ** (2 * k / head_dim) for k in range(head_dim // 2)]
    kind = sc.get("rope_type") or sc.get("type") or "default"
    if kind == "llama3":
        factor, low, high = float(sc["factor"]), float(sc["low_freq_factor"]), float(sc["high_freq_factor"])
        orig = float(sc["original_max_position_embeddings"])
        out = []
        for f in inv:
            wl = 2 * math.pi / f
            if wl < orig / high:
                out.append(f)
            elif wl > orig / low:
                out.append(f / factor)
            else:
                s = (orig / wl - low) / (high - low)
                out.append((1 - s) * f / factor + s * f)
        inv = out
    elif kind != "default":
        return None
    return torch.tensor(inv, dtype=torch.float64)


def rotate(words, shift, inv_freq, heads, hd):
    """Keys as BF16 words [n, heads * hd] turned on by `shift` positions (vLLM's neox-style RoPE, Llama's: the two
    halves of each head), in float32, rounded to BF16 words."""
    x = words.view(torch.bfloat16).to(torch.float32).view(-1, heads, hd)
    ang = (float(shift) * inv_freq).to(torch.float32).to(words.device)
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    y = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    return y.to(torch.bfloat16).view(torch.int16).reshape(words.shape)


class DeltaStats:
    """Running histograms per layer, keys (0) and values (1): bit lengths of xor, diff and rot (keys), the target's
    sign and exponent (alone), equal words. On the device; nothing waits for it until record()."""

    def __init__(self, L, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.int64, device=device)
        self.L = L
        self.h = {"xor": z(2, L, NBL), "diff": z(2, L, NBL), "rot": z(2, L, NBL)}
        self.se = z(2, L, 512)
        self.equal = z(2, L)
        self.values = [[0] * L, [0] * L]
        self.requests = self.tokens = self.related = self.measured = self.base_gone = 0

    def _hist(self, name, kv, l, v):
        self.h[name][kv, l] += torch.bincount(bit_length(v).flatten(), minlength=NBL)[:NBL]

    def add(self, l, kv, a, b, rot_b=None):
        """Target words a and base words b ([n, C] int16) of layer l; rot_b: the base turned (keys)."""
        oa = ordered(a)
        x = (a.to(torch.int32) ^ b.to(torch.int32)) & 0xFFFF
        self._hist("xor", kv, l, x)
        self._hist("diff", kv, l, (oa - ordered(b)).abs())
        if rot_b is not None:
            self._hist("rot", kv, l, (oa - ordered(rot_b)).abs())
        self.se[kv, l] += torch.bincount(((a.to(torch.int32) & 0xFFFF) >> 7).flatten(), minlength=512)[:512]
        self.equal[kv, l] += (x == 0).sum()
        self.values[kv][l] += int(x.numel())

    def record(self, group=None):
        c = lambda t: t.cpu().tolist()
        r = {"event": "delta_stats", "requests": self.requests, "tokens": self.tokens, "related": self.related,
             "measured": self.measured, "base_gone": self.base_gone, "layers": self.L,
             "hist": {k: c(v) for k, v in self.h.items()}, "sign_exp": c(self.se), "equal": c(self.equal),
             "values": self.values}
        if group is not None:
            r["group"] = group
        return r


def window_hashes(ids):
    """uint64 hashes of every 16-token window of `ids`, the one ending at token i at [i - 15] (numpy)."""
    import numpy as np
    a = np.asarray(ids, dtype=np.int64)
    if len(a) < WINDOW:
        return np.zeros(0, dtype=np.uint64)
    w = np.lib.stride_tricks.sliding_window_view(a, WINDOW).astype(np.uint64)
    g = np.random.default_rng(SEED).integers(1, 2 ** 63, size=WINDOW, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    with np.errstate(over="ignore"):
        return (w * g).sum(1, dtype=np.uint64) ^ np.uint64(FOLD)


class RelationIndex:
    """The text side, in P's send path: every traced request's tokens, and where each 16-token window first
    appeared. match() relates a new request to the earlier ones and adds it."""

    def __init__(self):
        self.ids = []                                # request number -> its token ids (numpy int32)
        self.keys = []
        self.first = {}                              # window hash -> (request number, position of its last token)
        self.lock = threading.Lock()

    def match(self, key, ids):
        """(request number, spans [(base request, i0, j0, n)]: tokens i0..i0+n are the text of the base's
        j0..j0+n, novel: positions whose window was new (kept as bases), hashes: uint64 window hash of each position
        from 15 on, at [i - 15])."""
        import numpy as np
        ids = np.asarray(ids, dtype=np.int32)
        hs = window_hashes(ids).tolist()
        with self.lock:
            r = len(self.ids)
            runs, cur, novel = [], None, []
            for k, h in enumerate(hs):
                i = k + WINDOW - 1
                hit = self.first.get(h)
                if hit is None:
                    self.first[h] = (r, i)
                    novel.append(i)
                if cur is not None and hit is not None and hit[0] == cur[0] and hit[1] == cur[2] + cur[3]:
                    cur[3] += 1
                    continue
                if cur is not None:
                    runs.append(cur)
                cur = [hit[0], i, hit[1], 1] if hit is not None else None
            if cur is not None:
                runs.append(cur)
            spans = []
            for s, i, j, n in runs:
                i0, j0, n = i - WINDOW + 1, j - WINDOW + 1, n + WINDOW - 1   # the whole first window is shared
                if i0 == j0 and np.array_equal(self.ids[s][:i0 + n], ids[:i0 + n]):
                    continue                         # same place, same prefix: a prefix cache's
                spans.append((s, i0, j0, n))
            self.ids.append(ids)
            self.keys.append(key)
        return r, spans, np.asarray(novel, dtype=np.int64), np.asarray(hs, dtype=np.uint64)


def sampled(hashes, every):
    """Positions whose window hash is 0 modulo `every` (content-defined: the same text, the same choice)."""
    import numpy as np
    if every <= 1:
        return np.arange(len(hashes), dtype=np.int64) + WINDOW - 1
    return np.nonzero(hashes % np.uint64(every) == 0)[0].astype(np.int64) + WINDOW - 1


def kind_of_piece(kind):
    return {"doc": "doc", "answer": "answer", "user": "question"}.get(kind, "other")


def split_spans(spans, pieces):
    """Spans cut at the driver's piece boundaries: [(kind, base, i0, j0, n)]; pieces [(kind, start, end)] (or None:
    every token of kind "all")."""
    if not pieces:
        return [("all", s, i0, j0, n) for s, i0, j0, n in spans]
    cuts = sorted(pieces, key=lambda p: p[1])
    out = []
    for s, i0, j0, n in spans:
        for kind, a, b in cuts:
            lo, hi = max(a, i0), min(b, i0 + n)
            if hi > lo:
                out.append((kind_of_piece(kind), s, lo, j0 + lo - i0, hi - lo))
    return out


class RelationProbe:
    """P's measurement thread (see above). submit(): a request's number, key, group, spans by kind, and the words of
    the positions it needs ([L, len(pos), W] on the host, pos sorted: the span tokens and the novel ones)."""

    def __init__(self, tlog, device, heads, hd, rope=None, store_gb=8.0, every=20, period_s=30.0, queue_gb=4.0):
        import queue
        self.tlog, self.dev, self.heads, self.hd = tlog, device, int(heads), int(hd)
        self.inv = rope_inv_freq(rope, self.hd) if rope else None
        self.budget = int(float(store_gb) * 2 ** 30)
        self.every, self.period = int(every), float(period_s)
        self.queue_budget, self.queued, self.qlock = int(float(queue_gb) * 2 ** 30), 0, threading.Lock()
        self.store = {}                              # request number -> (row of each position or -1, words)
        self.order, self.bytes, self.run = [], 0, None
        self.stats, self.last, self.dirty = {}, {}, set()
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._loop, daemon=True, name="tkv-relations")
        self.t.start()

    def room(self, nbytes):
        """Whether `nbytes` more words fit the queue's budget (else the request is not measured: P runs on)."""
        with self.qlock:
            return self.queued + nbytes <= self.queue_budget or self.queued == 0

    def submit(self, r, key, layout, T, spans, pos, words, novel, info=None):
        with self.qlock:
            self.queued += words.numel() * 2
        self.q.put((r, key, layout, T, spans, pos, words, novel, info or {}))

    def close(self):
        self.q.put(None)
        self.t.join(timeout=120)

    def flush(self):
        for g, st in sorted(self.stats.items()):
            self.tlog.write(st.record(g))

    def _loop(self):
        import queue
        stream = None
        if self.dev.type == "cuda":
            torch.cuda.set_device(self.dev)
            stream = torch.cuda.Stream(self.dev)
        while True:
            try:
                item = self.q.get(timeout=3)
            except queue.Empty:                      # idle: the totals as they stand (a report may be read now)
                for g in sorted(self.dirty):
                    self.tlog.write(self.stats[g].record(g))
                    self.last[g] = time.time()
                self.dirty = set()
                continue
            if item is None:
                self.flush()
                return
            try:
                with torch.inference_mode():
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            self._one(*item)
                    else:
                        self._one(*item)
            except Exception as exc:               # the trace never stops the run
                self.tlog.write({"event": "relations_error", "key": item[1], "error": repr(exc)[:300]})
            finally:
                with self.qlock:
                    self.queued -= item[6].numel() * 2

    def _one(self, r, key, layout, T, spans, pos, words, novel, info):
        import numpy as np
        if layout != self.run:                       # a new run: its own request numbers, its own bases
            self.store, self.order, self.bytes, self.run = {}, [], 0, layout
        L, C = words.shape[0], self.heads * self.hd
        row = np.full(T, -1, dtype=np.int64)
        row[pos] = np.arange(len(pos))
        groups = set()
        rec = {"event": "relations", "r": r, "key": key, "group": layout, "tokens": T, "stored": int(len(novel)),
               "spans": [], "by_kind": {}, **info}
        for kind, s, i0, j0, n in spans:
            g = f"{layout}/{kind}"
            st = self.stats.get(g) or self.stats.setdefault(g, DeltaStats(L, self.dev))
            groups.add(g)
            st.related += n
            bk = rec["by_kind"].setdefault(kind, [0, 0])
            bk[0] += n
            base = self.store.get(s)
            rec["spans"].append([s, kind, i0, j0, n])
            if base is None:
                st.base_gone += int((row[i0:i0 + n] >= 0).sum())
                continue
            ar, br = row[i0:i0 + n], base[0][j0:j0 + n]
            cand = ar >= 0                           # sampled (and within the cap)
            ok = cand & (br >= 0)
            m = int(ok.sum())
            st.base_gone += int(cand.sum()) - m
            if m == 0:
                continue
            ai, bi = torch.from_numpy(ar[ok]), torch.from_numpy(br[ok])
            for l in range(L):
                a = words[l].index_select(0, ai).to(self.dev, non_blocking=False)
                b = base[1][l].index_select(0, bi).to(self.dev, non_blocking=False)
                rb = rotate(b[:, :C], i0 - j0, self.inv, self.heads, self.hd) if self.inv is not None else None
                st.add(l, 0, a[:, :C], b[:, :C], rb)
                st.add(l, 1, a[:, C:], b[:, C:])
            st.measured += m
            bk[1] += m
        for g in groups | {f"{layout}/all"}:
            st = self.stats.get(g) or self.stats.setdefault(g, DeltaStats(L, self.dev))
            st.requests += 1
        self.stats[f"{layout}/all"].tokens += T
        self.tlog.write(rec)
        # this request's new windows: the bases of later ones
        if len(novel):
            keep = torch.from_numpy(row[novel])
            nrow = np.full(T, -1, dtype=np.int64)
            nrow[novel] = np.arange(len(novel))
            w = words.index_select(1, keep).clone()
            self.store[r] = (nrow, w)
            self.order.append(r)
            self.bytes += w.numel() * 2
            while self.bytes > self.budget and len(self.order) > 1:
                old = self.order.pop(0)
                self.bytes -= self.store.pop(old)[1].numel() * 2
        now = time.time()
        for g in groups | {f"{layout}/all"}:
            st = self.stats[g]
            if st.requests % self.every == 0 or now - self.last.get(g, 0) > self.period:
                self.tlog.write(st.record(g))
                self.last[g] = now
                self.dirty.discard(g)
            else:
                self.dirty.add(g)
