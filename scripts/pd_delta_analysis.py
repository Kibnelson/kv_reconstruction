#!/usr/bin/env python3
"""Offline delta analysis of raw KV pages stored by a capture run (MATRIX=capture; kv_compare/pd/trace.py PageStore):
for text that appears again where no prefix cache can reuse it, how its KV relates to the KV of its earlier copy, by
every measure we have, so the differential-storage opportunity is known before any coder is built.

  python3 -I scripts/pd_delta_analysis.py --capture CAPTURE_DIR --run RUN_DIR [--model-dir MODEL] \
      [--markdown delta_report.md] [--json delta_report.json]

CAPTURE_DIR: blocks_P_*.bin / capture_P_*.jsonl / meta_P_*.json (P: every block it sent, each once) and the same for
D (its own copy of held blocks that differ from what P sent again). RUN_DIR: the run's output (runs/mtrag_*.jsonl
with the prompts' token ids and pieces, proxy.jsonl; .gz too). MODEL: config.json, for RoPE (keys turned to the new
position). numpy only.

What it measures, per run (prompt layout):
  related     prompt tokens whose 16-token window appeared earlier in the run, not inside the prefix any earlier
              request shares (so no prefix cache can reuse them); bases: the first earlier copy and the latest one
  lossless    for a sample of related tokens, per layer, keys (base turned by RoPE) and values apart, bits per value:
                alone      sign+exponent entropy + 7 mantissa bits (no base: the exponent coders' floor)
                prev       sign+exponent given the same token's previous layer + 7 (a stand-in for our codec's context)
                xor, diff  the base's XOR / ordered-integer difference, bit length entropy-coded + the bits below
                planes     a bit-plane coder: each bit given the base's bit, whether the planes above all matched,
                           and the plane above (planes0: the same without the base)
                base       sign+exponent given the base's, + the mantissa given the base's when sign+exponent match
                both       sign+exponent given the base's and how the previous layer's compares with it (sign the
                           same or not, exponent below / equal / above), + the mantissa as in base
              entropies cross-validated (counts from half the tokens, cost measured on the other half)
  structure   for whole related chunks: the difference (target - base, in floats) per layer: its size against the
              target, how much of its energy rank r holds (a low-rank patch), how few tokens hold most of it, where
              in the chunk they are, whether the same tokens stay highest from layer 2 on (CacheBlend's selection),
              how it spreads over heads
  held        D's own copy of blocks P computed again (same text, same place): the same lossless measures
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker (forked)
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

# workers: layers and chunks are measured in parallel processes (forked: the arrays are shared, not copied)
WORKERS = max(1, min(int(os.environ.get("DELTA_WORKERS", os.cpu_count() or 1)), 12))
_SHARED: dict = {}


def pmap(fn, items):
    items = list(items)
    if WORKERS > 1 and len(items) > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(min(WORKERS, len(items))) as pool:
            return pool.map(fn, items)
    return [fn(x) for x in items]

WINDOW = 16
SEED = 0x5EED_7AC3
FOLD = 0x9E3779B97F4A7C15
NBL = 18


# ------------------------------------------------------------------------------------------------------- reading
def jl(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            try:
                yield json.loads(line)
            except ValueError:
                pass


def files(d, pattern):
    d = pathlib.Path(d)
    return sorted(set(d.glob(pattern)) | set(d.glob(pattern + ".gz")))


class Store:
    """One engine's stored blocks: rows [row, layer, token, word] int16 (memory-mapped)."""

    def __init__(self, meta_path):
        m = json.loads(pathlib.Path(meta_path).read_text())
        d = pathlib.Path(meta_path).parent
        self.L, self.bs, self.W = m["layers"], m["block_size"], m["words"]
        path = d / m["blocks"]
        n = path.stat().st_size // m["row_bytes"] if path.is_file() else 0
        self.rows = np.memmap(path, dtype=np.int16, mode="r", shape=(n, self.L, self.bs, self.W)) if n else None
        self.n = n
        self.records = list(jl(d / m["index"])) if (d / m["index"]).is_file() else []
        self.role, self.pid = m["role"], m["pid"]


def load_stores(cap):
    out = {"P": [], "D": []}
    for p in sorted(pathlib.Path(cap).glob("meta_*_*.json")):
        s = Store(p)
        out[s.role].append(s)
    return out


def rope_inv_freq(model_dir, hd):
    try:
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
    except (OSError, ValueError, TypeError):
        return None, None
    sc = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
    theta = cfg.get("rope_theta") or sc.get("rope_theta")
    if cfg.get("head_dim"):
        hd = int(cfg["head_dim"])
    elif cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        hd = int(cfg["hidden_size"]) // int(cfg["num_attention_heads"])
    if not theta:
        return None, hd
    inv = [1.0 / float(theta) ** (2 * k / hd) for k in range(hd // 2)]
    kind = sc.get("rope_type") or sc.get("type") or "default"
    if kind == "llama3":
        f, lo, hi, orig = (float(sc[k]) for k in ("factor", "low_freq_factor", "high_freq_factor",
                                                   "original_max_position_embeddings"))
        out = []
        for x in inv:
            wl = 2 * math.pi / x
            if wl < orig / hi:
                out.append(x)
            elif wl > orig / lo:
                out.append(x / f)
            else:
                s = (orig / wl - lo) / (hi - lo)
                out.append((1 - s) * x / f + s * x)
        inv = out
    elif kind != "default":
        return None, hd
    return np.array(inv, dtype=np.float64), hd


# ----------------------------------------------------------------------------------------------------- words, bits
def u16(w):
    return w.view(np.uint16).astype(np.int32)


def to_f32(w):
    return (w.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def to_bf16(x):
    """float32 -> BF16 words (int16), round to nearest even (torch's)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    return r.astype(np.uint16).view(np.int16)


def ordered(w):
    u = u16(w)
    return np.where(u >= 0x8000, 0xFFFF - u, u + 0x8000)


def bit_length(v):
    return np.frexp(v.astype(np.float64))[1].astype(np.int64)


def rotate(keys, shift, inv, heads, hd):
    """Keys [n, heads * hd] (BF16 words) turned on by shift[n] positions (neox halves), rounded to BF16 words."""
    x = to_f32(keys).reshape(len(keys), heads, hd)
    ang = (shift.astype(np.float64)[:, None] * inv[None, :]).astype(np.float32)[:, None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    y = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    return to_bf16(y.reshape(len(keys), -1))


def entropy(c):
    c = np.asarray(c, dtype=np.float64)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log2(p)).sum())


def length_cost(v, signed):
    h = np.bincount(bit_length(v).ravel(), minlength=NBL)[:NBL]
    n = h.sum()
    extra = sum(int(c) * (max(b - 1, 0) + (1 if signed and b else 0)) for b, c in enumerate(h))
    return entropy(h) + extra / n if n else None


def xent(ctx, sym, nctx, nsym, train, alpha=0.5):
    """Cross-entropy (bits per symbol) of sym given ctx on the test half, counts from the train half (additive
    smoothing): an honest estimate of what a context coder with these contexts would spend."""
    t = np.bincount((ctx[train] * nsym + sym[train]).astype(np.int64), minlength=nctx * nsym).reshape(nctx, nsym)
    t = t.astype(np.float64) + alpha
    logp = np.log2(t) - np.log2(t.sum(1, keepdims=True))
    test = ~train
    if not test.any():
        return None
    return float(-logp[ctx[test], sym[test]].mean())


def lossless(a, b, prev, train):
    """Bits per value of coding target words a ([n, C] int16) by each measure; b: the base's words (keys already
    turned), prev: the same tokens' previous-layer words (None at layer 0); train: [n] bool, the counting half."""
    n, C = a.shape
    tr = np.repeat(train, C)
    ua, ub = u16(a).ravel(), u16(b).ravel()
    se_a, se_b = ua >> 7, ub >> 7
    m_a, m_b = ua & 0x7F, ub & 0x7F
    zero = np.zeros_like(se_a)
    out = {}
    out["alone"] = xent(zero, se_a, 1, 512, tr) + 7
    x = (ua ^ ub)
    out["xor"] = length_cost(x, False)
    out["diff"] = length_cost(np.abs(ordered(a).ravel() - ordered(b).ravel()), True)
    out["equal"] = float((x == 0).mean())
    out["same_sign_exp"] = float((se_a == se_b).mean())
    # bit planes, from the sign down
    pl = pl0 = 0.0
    for p in range(15, -1, -1):
        ap, bp = (ua >> p) & 1, (ub >> p) & 1
        above = (x >> (p + 1)) == 0 if p < 15 else np.ones_like(ap, dtype=bool)
        up = (ua >> (p + 1)) & 1 if p < 15 else zero
        pl += xent(bp * 4 + above.astype(np.int64) * 2 + up, ap, 8, 2, tr)
        pl0 += xent(up, ap, 2, 2, tr)
    out["planes"], out["planes0"] = pl, pl0
    # sign+exponent given the base's, the mantissa given the base's when sign+exponent match
    h_se_b = xent(se_b, se_a, 512, 512, tr)
    same = se_a == se_b
    trs, trd = tr & same, tr & ~same
    ms = xent(m_b[same], m_a[same], 128, 128, trs[same]) if same.any() and (~trs[same]).any() else 7.0
    md = xent(zero[~same], m_a[~same], 1, 128, trd[~same]) if (~same).any() and (~trd[~same]).any() else 7.0
    test = ~tr
    ps = float(same[test].mean()) if test.any() else 0.0
    mant = ps * ms + (1 - ps) * md
    out["base"] = h_se_b + mant
    if prev is not None:
        se_p = u16(prev).ravel() >> 7
        out["prev"] = xent(se_p, se_a, 512, 512, tr) + 7
        # the previous layer against the base: the same sign or not, its exponent below, equal to or above the
        # base's (6 classes: a context small enough to learn next to the base's 512 symbols)
        q = (se_p >> 8 == se_b >> 8) * 3 + np.sign((se_p & 0xFF) - (se_b & 0xFF)) + 1
        out["both"] = xent(se_b * 6 + q, se_a, 512 * 6, 512, tr) + mant
    return out


# ------------------------------------------------------------------------------------------------------ relations
def window_hashes(ids):
    a = np.asarray(ids, dtype=np.int64)
    if len(a) < WINDOW:
        return []
    w = np.lib.stride_tricks.sliding_window_view(a, WINDOW).astype(np.uint64)
    g = np.random.default_rng(SEED).integers(1, 2 ** 63, size=WINDOW, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    with np.errstate(over="ignore"):
        return ((w * g).sum(1, dtype=np.uint64) ^ np.uint64(FOLD)).tolist()


def prefix_hit(ids, seen, bs):
    """Tokens of ids's longest block-aligned prefix an earlier request had (`seen`: their prefix hashes); adds ids's."""
    h, hit, alive = 0, 0, True
    for k in range(0, len(ids) // bs):
        h = hash((h, tuple(ids[k * bs:(k + 1) * bs])))
        if alive and h in seen:
            hit = (k + 1) * bs
        else:
            alive = False
        seen.add(h)
    return hit


def spans_of(hs, index, r, hit):
    """Runs of related tokens (i0, base request, j0, n) of a request whose window hashes are hs, against `index`
    (window hash -> (request, position of its last token)), beyond the prefix hit."""
    runs, cur = [], None
    for k, h in enumerate(hs):
        i = k + WINDOW - 1
        b = index.get(h)
        if b is not None and b[0] == r:
            b = None                                  # the same request: not an earlier copy
        if cur is not None and b is not None and b[0] == cur[1] and b[1] == cur[2] + cur[3]:
            cur[3] += 1
            continue
        if cur is not None:
            runs.append(cur)
        cur = [i, b[0], b[1], 1] if b is not None else None
    if cur is not None:
        runs.append(cur)
    out = []
    for i, s, j, n in runs:
        i0, j0, n = i - WINDOW + 1, j - WINDOW + 1, n + WINDOW - 1
        lo = max(i0, hit)
        if lo < i0 + n:
            out.append((lo, s, j0 + lo - i0, i0 + n - lo))
    return out


def kind_at(pieces, t):
    for kind, _ref, a, b in pieces:
        if a <= t < b:
            return {"doc": "doc", "answer": "answer", "user": "question"}.get(kind, "other")
    return "other"


# ------------------------------------------------------------------------------------------------------- analysis
class Words:
    """Token words of captured requests: (request, position) -> [L, W] int16."""

    def __init__(self, stores):
        self.stores = {s.pid: s for s in stores}

    def get(self, pairs):
        """pairs: [(pid, row, offset)] -> [n, L, W]."""
        if not pairs:
            return None
        s0 = next(iter(self.stores.values()))
        out = np.empty((len(pairs), s0.L, s0.W), dtype=np.int16)
        by = collections.defaultdict(list)
        for k, (pid, row, off) in enumerate(pairs):
            by[pid].append((k, row, off))
        for pid, items in by.items():
            st = self.stores[pid]
            ks = np.array([x[0] for x in items])
            rows = np.array([x[1] for x in items])
            offs = np.array([x[2] for x in items])
            order = np.argsort(rows, kind="stable")
            for c in range(0, len(order), 4096):
                sel = order[c:c + 4096]
                out[ks[sel]] = st.rows[rows[sel], :, offs[sel], :]
        return out


def _pairs_layer(l):
    A, B, shift, inv, heads, hd, train = (_SHARED[k] for k in ("A", "B", "shift", "inv", "heads", "hd", "train"))
    C = A.shape[2] // 2
    a, b = A[:, l], B[:, l]
    kb = rotate(b[:, :C], shift, inv, heads, hd) if inv is not None else b[:, :C]
    prev = A[:, l - 1] if l > 0 else None
    k = lossless(a[:, :C], kb, None if prev is None else prev[:, :C], train)
    if inv is not None:
        k["xor_unturned"] = length_cost(u16(a[:, :C]).ravel() ^ u16(b[:, :C]).ravel(), False)
    v = lossless(a[:, C:], b[:, C:], None if prev is None else prev[:, C:], train)
    return dict(k, layer=l), dict(v, layer=l)


def measure_pairs(A, B, shift, inv, heads, hd, layers, seed):
    """Per layer, keys and values: lossless(); A, B [n, L, W] target and base words; shift [n] positions."""
    _SHARED.update(A=A, B=B, shift=shift, inv=inv, heads=heads, hd=hd,
                   train=np.random.default_rng(seed).random(A.shape[0]) < 0.5)
    rows = pmap(_pairs_layer, layers)
    return {"K": [k for k, _v in rows], "V": [v for _k, v in rows]}


def deviation(A, B, shift, inv, heads, hd, offset):
    """Relative deviation |a - b| / |a| per token and layer (keys turned), and its mean by place in the span."""
    n, L, W = A.shape
    C = W // 2
    out = {"K": np.zeros((n, L)), "V": np.zeros((n, L))}
    for l in range(L):
        kb = rotate(B[:, l, :C], shift, inv, heads, hd) if inv is not None else B[:, l, :C]
        for name, a, b in (("K", A[:, l, :C], kb), ("V", A[:, l, C:], B[:, l, C:])):
            fa, fb = to_f32(a).astype(np.float64), to_f32(b).astype(np.float64)
            out[name][:, l] = np.linalg.norm(fa - fb, axis=1) / np.maximum(np.linalg.norm(fa, axis=1), 1e-30)
    buckets = [(0, 16), (16, 64), (64, 256), (256, 10 ** 9)]
    by = {}
    for name in ("K", "V"):
        by[name] = {}
        for lo, hi in buckets:
            m = (offset >= lo) & (offset < hi)
            if m.any():
                by[name][f"{lo}-{hi if hi < 10 ** 9 else ''}"] = {
                    "tokens": int(m.sum()), "by_layer": [float(x) for x in out[name][m].mean(0)]}
    return out, by


def chunk_structure(A, B, shift, inv, heads, hd, ranks=(1, 2, 4, 8, 16, 32, 64), frac=0.15):
    """One chunk's [n, L, W] target and base: per layer, keys (turned) and values: relative size of the
    difference, the share of its energy in rank r, tokens for 50/80/90% of it, the top `frac` tokens' share, overlap
    of the top tokens with layer 2's, the largest head's share."""
    n, L, W = A.shape
    C = W // 2
    res = {"K": [], "V": []}
    top2 = {}
    k_top = max(1, int(round(frac * n)))
    for l in range(L):
        kb = rotate(B[:, l, :C], shift, inv, heads, hd) if inv is not None else B[:, l, :C]
        for name, a, b in (("K", A[:, l, :C], kb), ("V", A[:, l, C:], B[:, l, C:])):
            fa = to_f32(a).astype(np.float64)
            d = fa - to_f32(b).astype(np.float64)
            tot = float((d * d).sum())
            rel = math.sqrt(tot / max(float((fa * fa).sum()), 1e-30))
            s = np.linalg.svd(d.astype(np.float32), compute_uv=False).astype(np.float64)
            e = np.cumsum(s * s) / max(float((s * s).sum()), 1e-30)
            tok = (d * d).sum(1)
            srt = np.sort(tok)[::-1]
            cum = np.cumsum(srt) / max(float(srt.sum()), 1e-30)
            need = {f"{int(q * 100)}": float((np.searchsorted(cum, q) + 1) / n) for q in (0.5, 0.8, 0.9)}
            top = set(np.argsort(tok)[::-1][:k_top].tolist())
            if l == 2:
                top2[name] = top
            hs = (d.reshape(n, heads, -1) ** 2).sum((0, 2))
            res[name].append({"layer": l, "rel": rel, "rank_energy": {str(r): float(e[min(r, len(e)) - 1]) for r in ranks},
                              "tokens_for": need, "top_share": float(srt[:k_top].sum() / max(srt.sum(), 1e-30)),
                              "top_head_share": float(hs.max() / max(hs.sum(), 1e-30)), "top": top})
    for name in ("K", "V"):
        for row in res[name]:
            t2 = top2.get(name)
            row["overlap_with_layer2"] = (len(row["top"] & t2) / max(len(t2), 1)) if t2 else None
            del row["top"]
    return res


def _chunk_job(k):
    A, B, shift = _SHARED["chunks"][k]
    return chunk_structure(A, B, shift, _SHARED["inv"], _SHARED["heads"], _SHARED["hd"])


def mean_rows(rows_list, keys):
    """Mean over chunks of each layer's numbers (nested dicts of floats)."""
    if not rows_list:
        return []
    L = len(rows_list[0])
    out = []
    for l in range(L):
        row = {"layer": rows_list[0][l]["layer"]}
        for k in keys:
            vals = [r[l][k] for r in rows_list if r[l].get(k) is not None]
            if not vals:
                row[k] = None
            elif isinstance(vals[0], dict):
                row[k] = {kk: float(np.mean([v[kk] for v in vals])) for kk in vals[0]}
            else:
                row[k] = float(np.mean(vals))
        out.append(row)
    return out


def analyse(cap, run_dir, model_dir=None, max_pairs=12000, max_chunks=48, chunk_tokens=512, max_held=1500,
            layers=None, seed=0):
    stores = load_stores(cap)
    if not stores["P"]:
        return {"error": f"no P store in {cap}"}
    L, bs, W = stores["P"][0].L, stores["P"][0].bs, stores["P"][0].W
    C = W // 2
    inv, hd = rope_inv_freq(model_dir, 128) if model_dir else (None, 128)
    heads = C // hd
    layers = list(range(L)) if layers is None else [l for l in layers if 0 <= l < L]
    words = Words(stores["P"])
    # captured requests: key -> (pid, rows, T)
    req = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request" and r.get("missing", 0) == 0:
                req[r["key"]] = (s.pid, r["rows"], r["T"])
    run_dir = pathlib.Path(run_dir)
    drv = [r for f in files(run_dir / "runs", "mtrag_*.jsonl") for r in jl(f)]
    proxy = [r for f in files(run_dir, "proxy.jsonl") for r in jl(f)]
    key_of = {p["tag"]: p["key"] for p in proxy if p.get("tag")}
    t_in = {p["key"]: p.get("t_in", 0) for p in proxy}
    runs = collections.defaultdict(list)
    for d in drv:
        k = key_of.get(d.get("tag"))
        if d.get("error") or not d.get("ids") or k not in req:
            continue
        runs[d.get("run") or d["layout"]].append((t_in.get(k, 0), d, k))
    rng = np.random.default_rng(seed)
    out = {"meta": {"layers": L, "block_size": bs, "words": W, "heads": heads, "head_dim": hd,
                    "rope": inv is not None, "stored_gb": sum(s.n for s in stores["P"]) * L * bs * W * 2 / 1e9,
                    "d_stored_gb": sum(s.n for s in stores["D"]) * L * bs * W * 2 / 1e9},
           "runs": {}}
    for run, reqs in sorted(runs.items()):
        reqs.sort(key=lambda x: x[0])
        seen, first, latest = set(), {}, {}
        tokens = prefix = 0
        rel = {"first": [], "latest": []}
        kinds = collections.Counter()
        for r, (_t, d, k) in enumerate(reqs):
            ids = d["ids"]
            tokens += len(ids)
            hit = prefix_hit(ids, seen, bs)
            prefix += hit
            hs = window_hashes(ids)
            for name, index in (("first", first), ("latest", latest)):
                for i0, s, j0, n in spans_of(hs, index, r, hit):
                    rel[name].append((r, i0, s, j0, n))
                    if name == "first":
                        for t in range(i0, i0 + n):
                            kinds[kind_at(d["pieces"], t)] += 1
            for q, h in enumerate(hs):
                first.setdefault(h, (r, q + WINDOW - 1))
                latest[h] = (r, q + WINDOW - 1)

        def loc(r, t):
            pid, rows, _T = req[reqs[r][2]]
            return pid, rows[t // bs], t % bs

        res = {"requests": len(reqs), "tokens": tokens, "prefix_tokens": prefix,
               "related": {n: sum(x[4] for x in v) for n, v in rel.items()}, "by_kind": dict(kinds)}
        res["related_share"] = {n: v / tokens for n, v in res["related"].items()} if tokens else {}
        for name in ("first", "latest"):
            pairs = [(r, i0 + u, s, j0 + u, u) for r, i0, s, j0, n in rel[name] for u in range(n)]
            if not pairs:
                continue
            if len(pairs) > max_pairs:
                pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_pairs, replace=False))]
            A = words.get([loc(r, i) for r, i, s, j, u in pairs])
            B = words.get([loc(s, j) for r, i, s, j, u in pairs])
            shift = np.array([i - j for r, i, s, j, u in pairs], dtype=np.int64)
            res[f"lossless_{name}"] = measure_pairs(A, B, shift, inv, heads, hd, layers, seed)
            res[f"lossless_{name}"]["pairs"] = len(pairs)
            if name == "first":
                _dev, res["deviation_by_place"] = deviation(A, B, shift, inv, heads, hd,
                                                            np.array([u for *_x, u in pairs]))
            del A, B
        # whole chunks
        ch = [x for x in rel["first"] if x[4] >= 64]
        if len(ch) > max_chunks:
            ch = [ch[i] for i in sorted(rng.choice(len(ch), max_chunks, replace=False))]
        rows = {"K": [], "V": []}
        data = []
        for r, i0, s, j0, n in ch:
            n = min(n, chunk_tokens)
            data.append((words.get([loc(r, i0 + u) for u in range(n)]), words.get([loc(s, j0 + u) for u in range(n)]),
                         np.full(n, i0 - j0, dtype=np.int64)))
        _SHARED.update(chunks=data, inv=inv, heads=heads, hd=hd)
        for st in pmap(_chunk_job, range(len(data))):
            rows["K"].append(st["K"])
            rows["V"].append(st["V"])
        del data
        keys = ("rel", "rank_energy", "tokens_for", "top_share", "top_head_share", "overlap_with_layer2")
        res["chunks"] = {"chunks": len(ch), "K": mean_rows(rows["K"], keys), "V": mean_rows(rows["V"], keys)}
        out["runs"][run] = res
    # D's held copies against P's recomputation (same text, same place)
    held = []
    pkey = {}
    for s in stores["P"]:
        for r in s.records:
            if r.get("event") == "request":
                pkey[r["key"]] = (s.pid, r["rows"], r["T"])
    dwords = Words(stores["D"]) if stores["D"] else None
    for s in stores["D"]:
        for r in s.records:
            if r.get("event") != "held_differs" or r["key"] not in pkey:
                continue
            ppid, prow, T = pkey[r["key"]]
            for j, drow in zip(r["blocks"], r["rows"]):
                jb = r["t0"] // bs + j
                if drow >= 0 and jb < len(prow) and prow[jb] >= 0:
                    held.append((ppid, prow[jb], s.pid, drow, min(bs, T - jb * bs)))
    if held and dwords is not None:
        if len(held) > max_held:
            held = [held[i] for i in sorted(rng.choice(len(held), max_held, replace=False))]
        A = np.concatenate([words.get([(p, pr, o) for o in range(nv)]) for p, pr, _d, _dr, nv in held])
        B = np.concatenate([dwords.get([(d, dr, o) for o in range(nv)]) for _p, _pr, d, dr, nv in held])
        out["held"] = {"blocks": len(held), "lossless": measure_pairs(A, B, np.zeros(len(A), dtype=np.int64), None,
                                                                       heads, hd, layers, seed)}
    return out


# ------------------------------------------------------------------------------------------------------- report
MEASURES = ("alone", "prev", "xor", "diff", "planes0", "planes", "base", "both")


def overall(rows, k):
    """Mean over layers 1 on (layer 0 is a function of the tokens: recomputed, not stored)."""
    vals = [r[k] for r in rows if r.get(k) is not None and r["layer"] >= 1]
    return float(np.mean(vals)) if vals else None


def markdown(res):
    f2 = lambda x: "-" if x is None else f"{x:.2f}"
    pc = lambda x: "-" if x is None else f"{100 * x:.0f}%"
    L = ["# Delta analysis of the stored pages", ""]
    if res.get("error"):
        return "\n".join(L + [res["error"], ""])
    m = res["meta"]
    L += [f"{m['layers']} layers, {m['heads']} KV heads of {m['head_dim']}; RoPE {'on' if m['rope'] else 'off'}; "
          f"{m['stored_gb']:.1f} GB of P's blocks, {m['d_stored_gb']:.2f} GB of D's.", "",
          "Bits per value (raw BF16: 16). alone: sign+exponent entropy + 7 mantissa bits (no base). prev: given the "
          "same token's previous layer (a stand-in for our codec's context). xor, diff: against the base, bit "
          "length coded. planes0 / planes: a bit-plane context coder without / with the base. base: sign+exponent "
          "given the base's, mantissa given the base's when they match. both: base and previous layer together. "
          "Keys: the base turned by RoPE to the new place. Cross-validated entropies.", ""]
    for run, r in res["runs"].items():
        rs = r.get("related_share", {})
        L += [f"## {run}", "",
              f"{r['requests']} requests, {r['tokens']:,} prompt tokens; prefix any earlier request shares: "
              f"{r['prefix_tokens']:,} ({pc(r['prefix_tokens'] / r['tokens'] if r['tokens'] else None)}); related "
              f"beyond it: {r['related'].get('first', 0):,} ({pc(rs.get('first'))}), by kind {r['by_kind']}.", ""]
        for name in ("first", "latest"):
            ll = r.get(f"lossless_{name}")
            if not ll:
                continue
            L += [f"Lossless, against the {name} copy ({ll['pairs']:,} tokens sampled), mean over layers 1-27 and "
                  "per layer:", "",
                  "| | " + " | ".join(MEASURES) + " | equal | same sign+exp |",
                  "| --- |" + " --- |" * (len(MEASURES) + 2)]
            for kv in ("K", "V"):
                rows = ll[kv]
                L.append(f"| {kv} mean | " + " | ".join(f2(overall(rows, k)) for k in MEASURES) +
                         f" | {pc(overall(rows, 'equal'))} | {pc(overall(rows, 'same_sign_exp'))} |")
            for kv in ("K", "V"):
                for row in ll[kv]:
                    if row["layer"] in (0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 27):
                        L.append(f"| {kv} {row['layer']} | " + " | ".join(f2(row.get(k)) for k in MEASURES) +
                                 f" | {pc(row.get('equal'))} | {pc(row.get('same_sign_exp'))} |")
            L.append("")
        dv = r.get("deviation_by_place")
        if dv:
            L += ["Relative deviation |target - base| / |target| by place in the related run (tokens since it "
                  "started), layers 1 / 4 / 12 / 27:", ""]
            for kv in ("K", "V"):
                for b, x in dv[kv].items():
                    bl = x["by_layer"]
                    L.append(f"- {kv} {b}: " + " / ".join(f2(bl[l]) for l in (1, 4, 12, 27) if l < len(bl)) +
                             f" ({x['tokens']:,} tokens)")
            L.append("")
        ch = r.get("chunks") or {}
        if ch.get("chunks"):
            L += [f"Whole related chunks ({ch['chunks']}, up to 512 tokens each), per layer (mean): relative size of "
                  "the difference; the energy rank 4 / 16 / 64 holds; tokens for 50% / 90% of it; the top 15% "
                  "tokens' share; overlap of the top 15% with layer 2's; the largest head's share:", "",
                  "| | rel | rank 4 / 16 / 64 | tokens for 50% / 90% | top 15% share | overlap with layer 2 | top head |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
            for kv in ("K", "V"):
                for row in ch[kv]:
                    if row["layer"] in (1, 2, 4, 8, 12, 16, 20, 24, 27):
                        re_ = row["rank_energy"]
                        tf = row["tokens_for"]
                        L.append(f"| {kv} {row['layer']} | {f2(row['rel'])} | {pc(re_['4'])} / {pc(re_['16'])} / "
                                 f"{pc(re_['64'])} | {pc(tf['50'])} / {pc(tf['90'])} | {pc(row['top_share'])} | "
                                 f"{pc(row['overlap_with_layer2'])} | {pc(row['top_head_share'])} |")
            L.append("")
    h = res.get("held")
    if h:
        ll = h["lossless"]
        L += ["## D's held copies against P's recomputation", "",
              f"{h['blocks']:,} blocks whose bytes differ (same text, same place), mean over layers 1-27:", "",
              "| | " + " | ".join(MEASURES) + " | equal |", "| --- |" + " --- |" * (len(MEASURES) + 1)]
        for kv in ("K", "V"):
            L.append(f"| {kv} | " + " | ".join(f2(overall(ll[kv], k)) for k in MEASURES) +
                     f" | {pc(overall(ll[kv], 'equal'))} |")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-pairs", type=int, default=12000)
    ap.add_argument("--max-chunks", type=int, default=48)
    ap.add_argument("--max-held", type=int, default=1500)
    ap.add_argument("--layers", default=None, help="comma-separated layers for the lossless measures (default all)")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    res = analyse(a.capture, a.run, a.model_dir, a.max_pairs, a.max_chunks, max_held=a.max_held, layers=layers)
    text = markdown(res)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    if a.markdown:
        pathlib.Path(a.markdown).write_text(text)
    print(text)
    return 0 if not res.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
