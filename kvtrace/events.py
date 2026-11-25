"""The records of kvtrace's events: one extractor per event name, from the objects its hook passes.

Names (docs/lmcache_lifecycle.md has each with the file and function of its hook):
  vllm.request.added        a request enters an engine's scheduler: its prompt's block hashes
  vllm.gpu.prefix_hit       the engine's own prefix cache serves the first blocks of a request
  vllm.gpu.allocated        GPU blocks given to a request for tokens to compute or load
  vllm.gpu.cached           full GPU blocks enter the prefix cache (their hash now finds them): "created"
  vllm.gpu.evicted          a cached GPU block's hash is dropped to reuse the block
  vllm.request.blocks_freed a finished request's GPU blocks go back to the free queue (still cached)
  vllm.request.finished     a request leaves the scheduler
  vllm.gpu.reset            the prefix cache emptied
  lmc.conn.lookup           LMCache's connector (in the engine) learns how many tokens LMCache holds
  lmc.conn.alloc            the connector is told the request's blocks; whether it will load into them
  lmc.conn.retrieve_planned the tokens and blocks the connector will load this step
  lmc.conn.store_planned    the tokens and blocks the connector will store this step
  lmc.conn.retrieve_submitted / lmc.conn.store_submitted   the worker sends them to the LMCache server
  lmc.conn.finished         the worker's finished transfers (sending, receiving)
  lmc.conn.request_finished the connector's last word on a request (its vLLM and LMCache hits)
  lmc.<event>               every event LMCache publishes on its event bus (l1.write.finished, l1.keys.evicted,
                            l2.store.completed, l2.prefetch.load.completed, mp.tokens, ...), in the server and in
                            the engines' connectors. "via" says how it came: absent for EventBus.publish, "stream"
                            when scheduled on a CUDA stream (the time it was queued), "native" when the native
                            recorder's copy is drained (the time the GPU reached it)

Hashes and keys are written as 16 hex characters (their first 8 bytes); block lists as block ids.
"""
from __future__ import annotations

import zlib

KEYS_MAX = 4096                                       # keys written per record (the count is always written)


def hx(v) -> str | None:
    """A hash or key as 16 hex characters: bytes (vLLM's sha256, LMCache's chunk hash), an int (LMCache's builtin
    hash, as its TokenHasher.hash_to_bytes turns it into key bytes), or an ObjectKey (its chunk hash)."""
    if v is None:
        return None
    ident = getattr(v, "identity", None)
    if ident is not None:
        v = ident.chunk_hash
    elif hasattr(v, "chunk_hash"):
        v = v.chunk_hash
    if isinstance(v, int):
        v = v.to_bytes(8, byteorder="big", signed=True)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)[:8].hex()
    return str(v)[:32]


def vhx(key) -> str | None:
    """vLLM's BlockHashWithGroupId (the hash, then 4 bytes of group id) or BlockHash, as hx."""
    if key is None:
        return None
    b = bytes(key)
    return (b[:-4] if len(b) in (36, 12) else b)[:8].hex()


def ids(blocks) -> list[int]:
    out = []
    for b in blocks or ():
        bid = getattr(b, "block_id", b)
        if getattr(b, "is_null", False):
            continue
        out.append(int(bid))
    return out


def group0(block_ids):
    """Block ids of the first KV cache group, from a tuple of per-group lists (or a flat list)."""
    if not block_ids:
        return []
    first = block_ids[0]
    if isinstance(first, (list, tuple)):
        return list(first)
    return list(block_ids)


def crc(tokens) -> int:
    import array
    return zlib.crc32(array.array("i", list(tokens)).tobytes())


def req_of(request) -> dict:
    return {"req": getattr(request, "request_id", None)}


# ------------------------------------------------------------------------------------------------------ vLLM
def request_added(request):
    hashes = [vhx(h) for h in (getattr(request, "block_hashes", None) or [])]
    return dict(req_of(request), n_prompt=int(request.num_prompt_tokens), n_hashes=len(hashes),
                hashes=hashes, kvp=sorted((getattr(request, "kv_transfer_params", None) or {}).keys()))


def request_finished(request, delay=None):
    return dict(req_of(request), status=str(getattr(request, "status", "")), n_out=int(request.num_output_tokens),
                n_tokens=int(request.num_tokens), delay_free=None if delay is None else bool(delay))


def gpu_prefix_hit(request, tokens, blocks):
    bl = group0(blocks.get_block_ids()) if hasattr(blocks, "get_block_ids") else ids(blocks)
    return dict(req_of(request), tokens=int(tokens), blocks=bl)


def gpu_allocated(request, new_blocks, num_new_tokens, num_new_computed_tokens, num_external_computed_tokens,
                  delay):
    groups = new_blocks if isinstance(new_blocks, (list, tuple)) else getattr(new_blocks, "blocks", ())
    first = groups[0] if groups and isinstance(groups[0], (list, tuple)) else groups
    return dict(req_of(request), new=int(num_new_tokens), local_hit=int(num_new_computed_tokens),
                external=int(num_external_computed_tokens), delay_cache=bool(delay), blocks=ids(first),
                n_computed=int(getattr(request, "num_computed_tokens", -1)))


def gpu_cached(request, blocks, first, hashes, group, mask=None, block_size=None):
    bl, hs, idx = [], [], []
    for i, blk in enumerate(blocks):
        if blk.is_null or (mask is not None and not mask[i]):
            continue
        bl.append(int(blk.block_id))
        hs.append(vhx(hashes[i]) if i < len(hashes) else None)
        idx.append(first + i)
    if not bl:
        return None
    return dict(req_of(request), group=int(group), first=int(first), index=idx, blocks=bl, hashes=hs,
                block_size=block_size)


def gpu_evicted(block, hashes):
    return {"block": int(block.block_id), "hashes": [vhx(h) for h in hashes]}


def request_blocks_freed(manager, request):
    try:
        bl = group0(manager.get_block_ids(request.request_id))
    except Exception:
        bl = None
    return dict(req_of(request), blocks=bl)


def gpu_reset():
    return {}


# ---------------------------------------------------------------------------------------- LMCache, engine side
def conn_lookup(request, vllm_hit, lmc_hit, load=None):
    return dict(req_of(request), vllm_hit=int(vllm_hit), lmc_hit=int(lmc_hit) if lmc_hit is not None else None,
                load=None if load is None else int(load), n_tokens=int(request.num_tokens))


def conn_alloc(request, new_block_ids, external, tracker):
    return dict(req_of(request), new_blocks=group0(new_block_ids), external=int(external),
                state=getattr(getattr(tracker, "state", None), "name", None))


def _op(meta):
    op = meta.op
    return {"req": meta.request_id, "start": int(op.start), "end": int(op.end),
            "skip": int(getattr(op, "skip_first_n_tokens", 0)), "blocks": list(op.flat_block_ids)[:KEYS_MAX],
            "n_blocks": len(op.flat_block_ids)}


def conn_retrieve_planned(meta):
    return _op(meta)


def conn_store_planned(meta):
    return _op(meta)


def conn_submitted(request_ids):
    return {"reqs": list(request_ids)}


def conn_finished(result):
    sending, receiving = result if result else (None, None)
    if not sending and not receiving:
        return None
    return {"sending": sorted(sending or ()), "receiving": sorted(receiving or ())}


def conn_request_finished(request, tracker, can_store):
    return dict(req_of(request), vllm_hit=getattr(tracker, "num_vllm_hit_tokens", None),
                lmc_hit=getattr(tracker, "num_lmcache_hit_tokens", None),
                stored=getattr(tracker, "num_stored_tokens", None), delay_free=bool(can_store))


# --------------------------------------------------------------------------------------- LMCache, server side
SKIP = {"trace.call", "mp.transfer.phase_samples", "sm.capacity.changed", "l1.eviction.loop_tick"}


def _md(v, depth=0):
    """LMCache event metadata as JSON: keys and hashes as hx; lists of them kept (at most KEYS_MAX)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (bytes, bytearray, memoryview)) or hasattr(v, "identity") or hasattr(v, "chunk_hash"):
        return hx(v)
    if isinstance(v, dict):
        return {str(k): _md(x, depth + 1) for k, x in v.items()} if depth < 3 else repr(v)[:200]
    if isinstance(v, (list, tuple, set, frozenset)):
        v = list(v)
        return [_md(x, depth + 1) for x in v[:KEYS_MAX]] if depth < 3 else len(v)
    return repr(v)[:200]


def lmc_event(event, via=None):
    """An event of LMCache's event bus: published (EventBus.publish), or scheduled on a CUDA stream
    (EventBus.publish_on_stream, via "stream"; without LMCache's native recorder the same event is published again
    when the stream reaches it, so the report keeps one of the two)."""
    name = getattr(event.event_type, "value", str(event.event_type))
    if name in SKIP:
        return None
    md = dict(event.metadata or {})
    # token lists: their size and a checksum per chunk (the run's records have the tokens themselves)
    if "token_chunks" in md:
        tc = md.pop("token_chunks")
        md["chunk_tokens"] = [len(c) for c in tc]
        md["chunk_crc"] = [crc(c) for c in tc]
    if "token_ids" in md:
        md["n_token_ids"] = len(md.pop("token_ids"))
    if isinstance(md.get("meta"), (list, tuple)):    # L1 objects' metadata: their sizes, parallel to the keys
        md["sizes"] = [getattr(m, "size_bytes", None) for m in md.pop("meta")][:KEYS_MAX]
    for k in ("dtypes", "shapes", "stacktrace"):
        md.pop(k, None)
    rec = {"_ev": "lmc." + name, "session": event.session_id or None}
    if via:
        rec["via"] = via
    for k, x in md.items():
        if isinstance(x, (list, tuple, set, frozenset)) and k.endswith(("keys", "hashes")):
            rec["n_" + k] = len(x)
        rec[k] = _md(x)
    return rec


def lmc_native(event_name, session, ts, metadata):
    """An event LMCache recorded on a CUDA stream with its native recorder (EventBus._drain_all): t is when the GPU
    reached it (the recorder's time), t_hook when the bus drained it. Its metadata's non-integers are strings."""
    if event_name in SKIP:
        return None
    rec = {"_ev": "lmc." + event_name, "_t": float(ts), "session": session or None, "via": "native"}
    for k, x in (metadata or {}).items():
        if k not in ("dtypes", "shapes", "stacktrace"):
            rec[k] = _md(x)
    return rec


def generic(**objs):
    return {k: _md(v) for k, v in objs.items()}


EXTRACT = {
    "vllm.request.added": request_added,
    "vllm.request.finished": request_finished,
    "vllm.gpu.prefix_hit": gpu_prefix_hit,
    "vllm.gpu.allocated": gpu_allocated,
    "vllm.gpu.cached": gpu_cached,
    "vllm.gpu.evicted": gpu_evicted,
    "vllm.request.blocks_freed": request_blocks_freed,
    "vllm.gpu.reset": gpu_reset,
    "lmc.conn.lookup": conn_lookup,
    "lmc.conn.alloc": conn_alloc,
    "lmc.conn.retrieve_planned": conn_retrieve_planned,
    "lmc.conn.store_planned": conn_store_planned,
    "lmc.conn.retrieve_submitted": conn_submitted,
    "lmc.conn.store_submitted": conn_submitted,
    "lmc.conn.finished": conn_finished,
    "lmc.conn.request_finished": conn_request_finished,
    "lmc.bus": lmc_event,
    "lmc.bus.stream": lambda event: lmc_event(event, via="stream"),
    "lmc.bus.native": lmc_native,
}

