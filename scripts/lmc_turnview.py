#!/usr/bin/env python3
"""One NIXL run's conversation as a turn timeline: for each turn, where the current prompt's KV is, row by row.

Each row is one step of the turn (P accepts the prompt, writes its blocks, offloads LMCache objects, the proxy
dispatches to D, D plans its NIXL read, the disk writes, P releases its partial block, D reports the KV ready, D's
offload, its decode, its release) with its time range in ms from the prompt's arrival at P, and the residency of the
current prompt at the range's end: P's and D's GPU blocks, the blocks NIXL has pending, the LMCache objects in each
server's memory and on its disk (in block equivalents, 16 a 256-token object), and the copies tracked in all. GPU
capacity evictions (vLLM reclaiming a cached block of another prompt for this one) are counted per row and listed,
and a block map gives each logical block's GPU block on P and on D, its generation (how many times that GPU block
has been allocated), and where its KV came from.

  python scripts/lmc_turnview.py --results RESULTS_DIR --output PAGE.md [--conv 0] [--pair OTHER.md] [--title T]

RESULTS_DIR: a published run's folder (kvtrace.tar.gz, proxy.jsonl.gz, mtrag.jsonl.gz, env.json, logs/), or --run
for a run's own folder. Uses scripts/lmc_lifecycle.py's reading of the trace.
"""

import argparse
import collections
import gzip
import math
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import lmc_lifecycle as L  # noqa: E402

BLOCK = L.BLOCK
INF = math.inf
WRITTEN = ("l1.write.finished", "l1.write_finished_and_read_reserved")


def ranges(xs, prefix=""):
    """[0, 1, 2, 5] -> 'L0–2, L5' (prefix 'L')."""
    xs = sorted(set(xs))
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(f"{prefix}{xs[i]}" if i == j else f"{prefix}{xs[i]}–{xs[j]}")
        i = j + 1
    return ", ".join(out)


class Presence:
    """A chunk's times in a server's memory (intervals) and on its disk."""
    def __init__(self, c):
        self.l1, cur = [], None
        for nm, t, _ in sorted(c.get("l1") or [], key=lambda e: e[1]):
            if nm in WRITTEN and cur is None:
                cur = t
            elif nm == "l1.keys.evicted" and cur is not None:
                self.l1.append((cur, t))
                cur = None
        if cur is not None:
            self.l1.append((cur, INF))
        self.l2 = (c.get("t_l2_stored"), c.get("t_l2_deleted") or INF) if c.get("t_l2_stored") else None
        self.times = [t for _, t, _ in (c.get("l1") or []) + (c.get("l2") or [])]

    def cpu(self, t):
        return any(a <= t < b for a, b in self.l1)

    def disk(self, t):
        return self.l2 is not None and self.l2[0] <= t < self.l2[1]


class Engine:
    """One engine's GPU block history: every allocation of a block (its generation) and every eviction."""
    def __init__(self, R, role):
        self.role = role
        allocs = []
        for (rl, _rid), q in R.reqs.items():
            if rl != role:
                continue
            for a in q["allocs"]:
                for b in a.get("blocks") or []:
                    allocs.append((a["t"], b))
            nr = q.get("nixl_recv")
            if nr and not (q["allocs"] and q["allocs"][0].get("blocks")):
                allocs += [(nr["t"], b) for b in nr.get("local") or []]   # async-load blocks: allocated at the plan
        self.allocs = sorted(allocs)
        self.evicted = sorted((r["t"], r.get("block"), (r.get("hashes") or [None])[0]) for r in R.recs
                              if r.get("role") == role and r["ev"] == "vllm.gpu.evicted")
        self.resets = sorted(r["t"] for r in R.recs if r.get("role") == role and r["ev"] == "vllm.gpu.reset")

    def gen(self, block, t):
        return sum(1 for ta, b in self.allocs if b == block and ta <= t + 1e-9)

    def next_alloc(self, block, t):
        return next((ta for ta, b in self.allocs if b == block and ta > t), None)


def turn_pairs(R):
    by_key = collections.defaultdict(dict)
    for (role, _rid), q in R.reqs.items():
        if q["key"]:
            by_key[q["key"]]["P" if role == "P" else "D"] = q
    out = collections.defaultdict(list)
    for tag, t in R.turns.items():
        key = next((k for k, px in R.proxy.items() if px.get("tag") == tag), None)
        pair = by_key.get(key, {})
        if "P" in pair and "D" in pair:
            out[t.get("conv")].append((t, pair["P"], pair["D"], R.proxy[key]))
    return {c: sorted(v, key=lambda x: int(x[0].get("turn", 0))) for c, v in out.items()}


def question(t):
    pieces, texts = t.get("pieces") or [], t.get("texts") or []
    qs = [tx for p, tx in zip(pieces, texts) if p[0] == "user" and tx]
    return re.sub(r"^\s*User:\s*", "", qs[-1] if qs else "").strip()


def owner_of(R, role, h):
    """The turn and token range an old GPU entry (its vLLM hash) held."""
    for life in R.lives[role].get(h, []):
        q = R.reqs.get((role, life["rid"]))
        turn = q.get("turn") if q else None
        i = life["index"]
        return (None if turn is None else int(turn) + 1), (i * BLOCK, (i + 1) * BLOCK), life
    return None, None, None


def kv_pool(run):
    """GPU KV cache size (tokens) from P's log."""
    for f in (run / "engine_P.log", run / "logs" / "engine_P.log.gz"):
        if f.is_file():
            op = gzip.open if f.suffix == ".gz" else open
            with op(f, "rt", errors="replace") as fh:
                for ln in fh:
                    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", ln)
                    if m:
                        return int(m.group(1).replace(",", ""))
    return None


def turn_section(R, eng, k, t, P, D, px, links):
    n = int(P.get("n_prompt") or t.get("prompt_tokens") or 0)
    nb = -(-n // BLOCK)
    full = n // R.chunk
    tail = n - full * R.chunk
    T = P["t_added"]
    msf = lambda x: f"{(x - T) * 1000:,.3f}"               # noqa: E731
    d = D["role"]
    per_engine = R.servers != "shared"
    p_srv, d_srv = R.engine_server.get("P", "LMC"), R.engine_server.get(d, "LMC")
    # the prompt's whole chunks, by LMCache's names (P's lookup on its server)
    look = ((P.get("server") or {}).get("mp.lookup") or [None])[-1] or {}
    hashes = (look.get("chunk_hashes") or [])[:full]
    pres = {srv: [Presence(R.chunks.get(R.ck(h, srv), {"l1": [], "l2": []})) for h in hashes]
            for srv in {p_srv, d_srv}}
    p_local = (P.get("local_hit") or 0) // BLOCK
    d_local = (D.get("local_hit") or 0) // BLOCK
    nr = D.get("nixl_recv") or {}
    n_nixl = len(nr.get("local") or [])
    p_partial = n % BLOCK != 0
    d_last_cached = (nb - 1) in D["cached"]
    t_end = D.get("t_freed") or D.get("t_finished") or T

    p_prompt_hashes = set((P.get("hashes") or [])[:nb])
    d_prompt_hashes = set((D.get("hashes") or [])[:nb])

    def evicted_prompt(e, hs, t_):
        return sum(1 for te, _b, h in e.evicted if T <= te <= t_ and h in hs)

    def reset_by(e, t_):
        return any(T <= tr <= t_ for tr in e.resets)

    def p_gpu(t_):
        if reset_by(eng["P"], t_):
            return 0
        v = p_local + ((nb - p_local) if t_ >= (P.get("t_finished") or INF) else 0)
        if p_partial and t_ >= (P.get("t_freed") or INF):
            v -= 1
        return max(0, v - evicted_prompt(eng["P"], p_prompt_hashes, t_))

    def d_gpu(t_):
        if reset_by(eng[d], t_):
            return 0
        v = d_local + (n_nixl if t_ >= (D.get("t_nixl_received") or INF) else 0)
        if not d_last_cached and t_ >= (D.get("t_freed") or INF):
            v -= 1
        return max(0, v - evicted_prompt(eng[d], d_prompt_hashes, t_))

    def pending(t_):
        return n_nixl if nr and nr["t"] <= t_ < (D.get("t_nixl_received") or INF) else 0

    def cpu(srv, t_):
        return BLOCK * sum(p.cpu(t_) for p in pres[srv])

    def disk(srv, t_):
        return BLOCK * sum(p.disk(t_) for p in pres[srv])

    def state(t_):
        s = {"P GPU": p_gpu(t_), "NIXL pending": pending(t_), "D GPU": d_gpu(t_),
             "P CPU": cpu(p_srv, t_), "P disk": disk(p_srv, t_)}
        if per_engine:
            s.update({"D CPU": cpu(d_srv, t_), "D disk": disk(d_srv, t_)})
        s["copies"] = s["P GPU"] + s["D GPU"] + s["P CPU"] + s["P disk"] + s.get("D CPU", 0) + s.get("D disk", 0)
        return s

    def peak(a, b):
        ts = [a, b] + [x for srv in pres for p in pres[srv] for x in p.times if a <= x <= b]
        return max(state(x)["copies"] for x in ts)

    def stored_chunks(q, srv):
        """The whole chunks of this prompt q's store wrote to srv this turn: indices, times."""
        out = []
        for j, h in enumerate(hashes):
            c = R.chunks.get(R.ck(h, srv))
            if c and c.get("key") == q["key"] and c.get("store_role") == q["role"]:
                out.append((j, c))
        return out

    rows = []                    # (start, end, actor, event, evicted, {column: note})

    def row(a, b, actor, event, evicted=None, notes=None):
        if a is None and b is None:
            return
        a = a if a is not None else b
        b = b if b is not None else a
        rows.append((a, b, actor, event, evicted, notes or {}))

    ev_rows = []
    for role, e in (("P", eng["P"]), (d, eng[d])):
        evs = [x for x in e.evicted if T <= x[0] <= t_end]
        if not evs:
            continue
        # one batch per allocation: evictions within 50 ms of each other
        batch = [evs[0]]
        for x in evs[1:]:
            if x[0] - batch[-1][0] > 0.05:
                ev_rows.append((role, batch))
                batch = []
            batch.append(x)
        ev_rows.append((role, batch))
    row(T, P.get("t_prefix") or (P["allocs"][0]["t"] if P["allocs"] else T), "P",
        f"Accept prompt · {p_local} local", notes={"P GPU": "carried" if p_local else None,
                                                    "D GPU": "carried" if d_local else None})
    for q, actor in ((P, "P"), (D, d)):            # each engine's LMCache lookup (on disk: its prefetch reads)
        qs = q.get("server") or {}
        lk = L.first(qs.get("mp.lookup") or [])
        pe = L.first(qs.get("mp.lookup_prefetch.end") or [])
        if lk and pe:
            found, l2 = pe.get("found_count") or 0, pe.get("l2_hit_keys") or 0
            row(lk["t"], pe["t"], f"{actor} · LMCache", f"Lookup · {found} object{'s' if found != 1 else ''} found"
                + (f", {l2} read from disk into CPU" if l2 else ""))
    p_alloc = next((a for a in P["allocs"] if a.get("blocks")), None)
    if p_alloc:
        row(p_alloc["t"], P.get("t_finished"), "P", f"Allocate slots and write {nb - p_local} blocks",
            notes={"P GPU": f"{p_local} local + {nb - p_local} written"})
    for role, batch in ev_rows:
        kind = "answer KV" if role == d and batch[0][0] > (D.get("t_nixl_received") or INF) else "prompt KV"
        row(batch[0][0], batch[-1][0], role if role == "P" else d, "Evict cached blocks and reallocate slots for "
            + kind, evicted=len(batch))
    row(px.get("t_prefill_done") or P.get("t_finished"), D.get("t_added"), f"Proxy → {d}",
        "Dispatch prompt and metadata · D accepts")
    psrv = P.get("server") or {}
    p_st = stored_chunks(P, p_srv)
    st0, st1 = L.first(psrv.get("mp.store.start") or []), (psrv.get("mp.store.end") or [None])[-1]
    if st0 or st1:
        nobj = (st1 or {}).get("stored_count", len(p_st))
        blocks = [i for j, _ in p_st for i in range(j * 16, j * 16 + 16)]
        # the range ends when LMCache reports the objects written to memory (its events come just after the copy)
        tw = [c["t_l1_written"] for _j, c in p_st if c.get("t_l1_written")]
        row((st0 or st1)["t"], max([(st1 or st0)["t"]] + tw), "P GPU → LMCache",
            f"Offload {nobj} object{'s' if nobj != 1 else ''}", notes={"P CPU": ranges(blocks, "L")} if blocks else {})
    if nr:
        row(D.get("t_prefix") or D.get("t_added"), nr["t"], d, f"Plan fetch · {d_local} local, {n_nixl} NIXL",
            notes={"NIXL pending": ranges(range(d_local, d_local + n_nixl), "L"),
                   "D GPU": f"<br>{n_nixl} slots reserved"})
    if R.mode != "memory" and p_st:
        tw = [c["t_l2_stored"] for _j, c in p_st if c.get("t_l2_stored")]
        tr = [x for _j, c in p_st for x in c.get("t_l1_removed") or [] if c.get("t_l2_stored") and
              x >= c["t_l2_stored"]]
        if tw:
            blocks = [i for j, _ in p_st for i in range(j * 16, j * 16 + 16)]
            row(min(tw), max(tw + tr), "P · LMCache", f"{len(tw)} disk write{'s' if len(tw) != 1 else ''} · release "
                "CPU copies", notes={"P disk": "+" + ranges(blocks, "L"), "peak": True})
    if p_partial and P.get("t_freed"):
        row(P["t_freed"], P["t_freed"], "P", "Release uncached partial prompt block",
            notes={"P GPU": f"releases L{nb - 1}"})
    if D.get("t_nixl_received"):
        row(D["t_nixl_received"], D["t_nixl_received"], "NIXL", "D reports prompt KV ready",
            notes={"NIXL pending": f"{n_nixl} ready", "D GPU": f"{d_local} local + {n_nixl} NIXL"})
    dsrv = D.get("server") or {}
    d_st = stored_chunks(D, d_srv) if per_engine else []
    ds0, ds1 = L.first(dsrv.get("mp.store.start") or []), (dsrv.get("mp.store.end") or [None])[-1]
    if ds0 or ds1:
        nobj = (ds1 or {}).get("stored_count", len(d_st))
        blocks = [i for j, _ in d_st for i in range(j * 16, j * 16 + 16)]
        tw = [c["t_l1_written"] for _j, c in d_st if c.get("t_l1_written")]
        row((ds0 or ds1)["t"], max([(ds1 or ds0)["t"]] + tw), f"{d} GPU → LMCache",
            f"Offload {nobj} object{'s' if nobj != 1 else ''}" + ("" if per_engine or nobj else
                                                                    " (the shared server has them already)"),
            notes={"D CPU": ranges(blocks, "L")} if per_engine and blocks else {})
    dec = [a for a in D["allocs"][1:]]
    row(D.get("t_nixl_received"), D.get("t_finished"), d, "Run logits and write answer KV" if dec else "Run logits")
    if D.get("t_freed"):
        row(D["t_freed"], D["t_freed"], d, "Release request" + ("" if d_last_cached else " · free partial block"))
    if R.mode != "memory" and d_st:
        tw = [c["t_l2_stored"] for _j, c in d_st if c.get("t_l2_stored")]
        tr = [x for _j, c in d_st for x in c.get("t_l1_removed") or [] if c.get("t_l2_stored") and
              x >= c["t_l2_stored"]]
        if tw:
            blocks = [i for j, _ in d_st for i in range(j * 16, j * 16 + 16)]
            row(min(tw), max(tw + tr), f"{d} · LMCache", f"{len(tw)} disk write{'s' if len(tw) != 1 else ''} · "
                "release CPU copies", notes={"D disk": "+" + ranges(blocks, "L"), "peak": True})
    for role in ("P", d):
        for tr in eng[role].resets:
            if T <= tr <= t_end + 1.0:
                row(tr, tr, role, "Reset GPU prefix slots")
    rows.sort(key=lambda r: (r[1], r[0]))

    cols = ["P GPU", "NIXL pending", "D GPU", "P CPU", "P disk"] + (["D CPU", "D disk"] if per_engine else [])
    names = {"P GPU": "P GPU", "NIXL pending": "NIXL pending", "D GPU": f"{d} GPU",
             "P CPU": "P CPU" if per_engine else "LMCache CPU", "P disk": "P disk" if per_engine else "LMCache disk",
             "D CPU": f"{d} CPU", "D disk": f"{d} disk"}
    any_ev = bool(ev_rows)
    head = ["Actor / path", "Elapsed (ms)", "Event"] + (["Old GPU entries evicted"] if any_ev else []) + \
        [names[c] for c in cols] + ["Tracked copies"]
    md_rows = []
    prev = None
    for a, b, actor, event, evicted, notes in rows:
        s = state(b)
        cells = []
        for c in cols:
            v = s[c]
            txt = f"**{v}**" if prev is not None and v != prev[c] else str(v)
            note = notes.get(c)
            if note == "carried":
                txt = f"**{v}** carried" if v else str(v)
            elif note and note.startswith("<br>"):
                txt += note
            elif note:
                txt += f" · {note}"
            cells.append(txt)
        cp = s["copies"]
        ctxt = f"**{cp}**" if prev is not None and cp != prev["copies"] else str(cp)
        if notes.get("peak"):
            pk = peak(a, b)
            if pk > cp:
                ctxt += f" · peak {pk}"
        el = msf(a) if abs(b - a) < 5e-7 else f"{msf(a)}–{msf(b)}"
        md_rows.append([actor, el, event] + ([f"**{evicted}**" if evicted else "—"] if any_ev else []) + cells
                       + [ctxt])
        prev = s

    q_text = question(t)
    nev = {r: sum(len(bt) for rl, bt in ev_rows if rl == r) for r in ("P", d)}
    md = [f"## Turn {k + 1} on {d}", ""]
    if q_text:
        md += [f"> {q_text}", ""]
    md += [f"**{n:,} prompt tokens · {nb} GPU blocks · {full} complete LMCache object{'s' if full != 1 else ''} · "
           f"{tail} tail tokens.**" + (f" {links}" if links else ""), "",
           f"**Capacity evictions on this turn** · P {nev['P']} · {d} {nev[d]}.", "",
           "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in md_rows]
    md += [""]
    if ev_rows:
        allev = [(role, x) for role, bt in ev_rows for x in bt]
        md += ["<details>", f"<summary>All {len(allev)} evicted slots and their new allocations</summary>", "",
               "| Engine | Eviction at ms | Old slot | Old indexed tokens | Reallocated at ms | New slot |",
               "| --- | --- | --- | --- | --- | --- |"]
        for role, (te, b, h) in allev:
            e = eng[role]
            turn, rng, _life = owner_of(R, role, h)
            ta = e.next_alloc(b, te)
            md.append(f"| {role} | {msf(te)} | `#{b} g{e.gen(b, te)}` | "
                      + (f"Turn {turn} · `[{rng[0]}, {rng[1]})`" if rng else "–") + f" | "
                      + (msf(ta) if ta else "–") + f" | " + (f"`#{b} g{e.gen(b, ta)}`" if ta else "–") + " |")
        md += ["", "</details>", ""]
    # the block map
    p_blocks = list(P.get("prefix_blocks") or [])[:p_local]
    for a in P["allocs"]:
        if len(p_blocks) >= nb:
            break
        p_blocks += list(a.get("blocks") or [])
    p_blocks = p_blocks[:nb]
    d_blocks = list(D.get("prefix_blocks") or [])[:d_local] + list(nr.get("local") or [])
    d_blocks = d_blocks[:nb]
    t_pa = p_alloc["t"] if p_alloc else T
    rows_b = []
    for i in range(nb):
        pb = p_blocks[i] if i < len(p_blocks) else None
        db = d_blocks[i] if i < len(d_blocks) else None
        p_src = "Local prefix" if i < p_local else "Model write"
        d_src = "Local prefix" if i < d_local else "NIXL from P"
        pg = eng["P"].gen(pb, T if i < p_local else t_pa) if pb is not None else None
        dg = eng[d].gen(db, D["t_added"] if i < d_local else (nr.get("t") or D["t_added"])) if db is not None else None
        rows_b.append((i, pb, pg, p_src, db, dg, d_src))
    grouped = []
    for r in rows_b:
        g = grouped[-1] if grouped else None
        if (g and g["p_src"] == r[3] and g["d_src"] == r[6] and g["pg"] == r[2] and g["dg"] == r[5]
                and r[1] is not None and r[4] is not None and g["pb1"] is not None and g["db1"] is not None
                and r[1] == g["pb1"] + 1 and r[4] == g["db1"] + 1):
            g.update(i1=r[0], pb1=r[1], db1=r[4])
        else:
            grouped.append(dict(i0=r[0], i1=r[0], pb0=r[1], pb1=r[1], pg=r[2], p_src=r[3], db0=r[4], db1=r[4],
                                dg=r[5], d_src=r[6]))

    def span(a, b):
        return "–" if a is None else str(a) if a == b else f"{a}–{b}"
    md += ["<details>", "<summary>Prompt block map</summary>", "",
           f"| Logical blocks | Prompt tokens | P block IDs / gen | P source | {d} block IDs / gen | {d} source |",
           "| --- | --- | --- | --- | --- | --- |"]
    for g in grouped:
        tok0, tok1 = g["i0"] * BLOCK, min(n, (g["i1"] + 1) * BLOCK)
        md.append(f"| {span(g['i0'], g['i1'])} | `[{tok0}, {tok1})` | {span(g['pb0'], g['pb1'])} / {g['pg']} | "
                  f"{g['p_src']} | {span(g['db0'], g['db1'])} / {g['dg']} | {g['d_src']} |")
    md += ["", "</details>", ""]
    p_disk = sum(1 for _j, c in p_st if c.get("t_l2_stored"))
    d_disk = sum(1 for _j, c in d_st if c.get("t_l2_stored"))
    totals = {"turn": k + 1, "d": d, "n": n, "blocks": nb, "p_local": p_local, "p_written": nb - p_local,
              "d_local": d_local, "nixl": n_nixl, "p_objects": len(p_st), "d_objects": len(d_st),
              "p_disk": p_disk, "d_disk": d_disk, "p_ev": nev["P"], "d_ev": nev[d],
              "nixl_ms": (D["t_nixl_received"] - nr["t"]) * 1000 if nr and D.get("t_nixl_received") else None}
    return md, totals


def page(R, run, conv=None, pair=None, title=None, evidence=None, pair_label=None):
    convs = turn_pairs(R)
    if not convs:
        raise SystemExit("no turns with both P and D in this run")
    conv = conv if conv in convs else sorted(convs, key=lambda c: (c is None, str(c)))[0]
    turns = convs[conv]
    e = R.env
    layout = L.LAYOUT_NAMES.get(e.get("layout"), e.get("layout"))
    ids = e.get("conv_ids") or []
    try:
        cid = ids[int(conv)] if ids else conv
    except (TypeError, ValueError, IndexError):
        cid = conv
    eng = {r: Engine(R, r) for r in L.ENGINES}
    pool = kv_pool(run)
    per_engine = R.servers != "shared"
    topo = ("P and each D have their own LMCache server" if per_engine else
            "P and the three D engines share one LMCache server")
    name = title or (f"NIXL with {'per-engine' if per_engine else 'shared'} LMCache · "
                     f"{R.mode}, {layout} timeline")
    nav = " · ".join(f"[Turn {k + 1}](#turn-{k + 1}-on-{p[2]['role'].lower()})" for k, p in enumerate(turns))
    md = [f"# {name}", "",
          f"Conversation `{cid}`, **{R.mode}, {layout}**, {len(turns)} turns. {topo}; "
          f"{e.get('l1_gb')} GB of LMCache memory a server"
          + (f", the disk tier with O_DIRECT, store policy `{e.get('store_policy')}`, prefetch policy "
             f"`{e.get('prefetch_policy')}`" if R.mode != "memory" else "")
          + (f". GPU KV pool: {pool:,} tokens ({pool // BLOCK:,} blocks) an engine (vLLM's default, no cap)."
             if pool else "."), "",
          (f"[Paired timeline]({pair}) · " if pair else "") + (f"[Run's results]({evidence}) · " if evidence else "")
          + f"Commit `{str(e.get('commit'))[:12]}`, run `{e.get('run') or run.name}`.", "", nav, "",
          "Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current "
          "prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token "
          "LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the "
          "objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk "
          "columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. "
          "Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value "
          "changed in that row. GPU block ids are each engine's own; a block's generation counts its "
          "allocations so far."
          + (" No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks)."
             if not any(eng[r].evicted for r in L.ENGINES) else ""), ""]
    tots = []
    for k, (t, P, D, px) in enumerate(turns):
        link = (f"[Same turn in {pair_label or 'the paired timeline'}]({pair}#turn-{k + 1}-on-{D['role'].lower()})"
                if pair else None)
        sec, tot = turn_section(R, eng, k, t, P, D, px, link)
        md += sec
        tots.append(tot)
    srv_p = "P's server" if per_engine else "the server"
    md += ["## The turns together", "",
           "| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | "
           f"Objects stored: {srv_p}" + (" / D's" if per_engine else "") + " | Disk writes"
           + (": P's / D's" if per_engine else "") + " | Capacity evictions P / D |",
           "| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |"]
    for x in tots:
        md.append(f"| {x['turn']} | {x['d']} | {x['n']:,} | {x['blocks']} | {x['p_local']} / {x['p_written']} | "
                  f"{x['d_local']} / {x['nixl']} | {x['nixl_ms']:,.0f} | {x['p_objects']}"
                  + (f" / {x['d_objects']}" if per_engine else "") + f" | {x['p_disk']}"
                  + (f" / {x['d_disk']}" if per_engine else "") + f" | {x['p_ev']} / {x['d_ev']} |")
    sm = lambda key: sum(x[key] for x in tots)           # noqa: E731
    md.append(f"| all | | {sm('n'):,} | {sm('blocks')} | {sm('p_local')} / {sm('p_written')} | "
              f"{sm('d_local')} / {sm('nixl')} | | {sm('p_objects')}" + (f" / {sm('d_objects')}" if per_engine else "")
              + f" | {sm('p_disk')}" + (f" / {sm('d_disk')}" if per_engine else "") + f" | {sm('p_ev')} / {sm('d_ev')} |")
    md.append("")
    return "\n".join(md) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results")
    ap.add_argument("--run")
    ap.add_argument("--output", required=True)
    ap.add_argument("--conv")
    ap.add_argument("--pair")
    ap.add_argument("--title")
    ap.add_argument("--evidence")
    ap.add_argument("--pair-label")
    a = ap.parse_args(argv)
    if not (a.results or a.run):
        ap.error("--results or --run")
    run = pathlib.Path(a.run) if a.run else L.unpack(a.results, tempfile.mkdtemp(prefix="turnview_"))
    R = L.Run(run)
    R.analyse()
    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(R, run, a.conv, a.pair, a.title, a.evidence, a.pair_label))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
