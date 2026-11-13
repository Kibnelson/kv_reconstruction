"""TKVConnector: the PD protocol as a vLLM 0.32 KV connector (push mode, prefill engines to decode engines), and a
CPU tier for colocated engines (kv_compare.pd.tier: LMCache's role, in every codec's format).

Load it with
  --kv-transfer-config '{"kv_connector": "TKVConnector", "kv_connector_module_path": "kv_compare.pd.connector",
                         "kv_role": "kv_producer" | "kv_consumer", "kv_connector_extra_config": {...}}'
(the repository on PYTHONPATH). Extra config:
  codecs_file      the export of scripts/pd_export_codecs.py (both roles: identical constants)
  codecs           names to load (default: raw, ours-dense, ours-kv, ours-4g, ours-hbh, ours-rank256, splitzip, adacode;
                   also t2-<mode>, exactkv, nvcomp-<alg>, exans, ibp: kv_compare.pd.codecs)
  splitzip_source, adacode_source, exactkv_source   the authors' checkouts (scripts/a5000_setup.sh clones them)
  exans_info       the ExANS build information (scripts/pd_setup.sh)
  listen           decode: "host:port" this engine receives on
  transport        prefill: "ring" (same machine, default), "p2p" (GPU to GPU over CUDA IPC) or "tcp"
  gpu_ring_bytes   decode: the GPU ring per sender of the p2p transport (default 512 MiB)
  tier             colocated engines (kv_role kv_both): {"budget_gb": x, "chunk": 256, "codec": name, "control":
                   JSON file {"generation": n, "codec": name} (a new generation empties the tier and sets its codec),
                   "max_pending_gb": 2}
  link_control     prefill: JSON file {"gbps": x} read when it changes (the benchmark sets the link speed there)
  log_dir          one JSON line per transfer: tkv_<role>_<pid>.jsonl
  recv_timeout_s   decode: a transfer not complete this long after its request was admitted fails (default 600)
  gpu_send_budget  prefill: bytes of coded transfers kept on the GPU while they wait for the link (default 2 GiB);
                   later ones wait in host memory

T2 (t2-<mode>) leaves layer 0 out when the receiver can recompute it bit for bit (kv_compare.pd.layer0): the
prefill engine's scheduler side records each step's row count for every request, the send carries those steps and
the token ids, and both ends recompute with the engine's own modules.

Per request, the proxy (scripts/pd_proxy.py) sets kv_transfer_params:
  prefill  {"t_send": true, "t_key": k, "t_dest": "host:port", "t_codec": name[, "t_conv": conversation,
            "t_verify": true (checksums: the correctness pass; off in timed runs)]}
  decode   {"t_recv": true, "t_key": k}

Prefill side: when the request finishes (after its first token), the scheduler keeps its blocks and hands them to
the worker; the worker gathers the prompt's KV from the paged cache, encodes it, releases the blocks and queues the
coded blob for the paced link. Decode side: the request waits (WAITING_FOR_REMOTE_KVS) with blocks allocated; each
engine step the worker installs every transfer that has arrived - decoded layer by layer straight into those blocks -
and reports it finished, after which vLLM recomputes only the last prompt token. A verified transfer carries a
checksum of the sender's words; the receiver compares it with what it installed (bit-exact or the transfer fails).
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)
DEFAULT_CODECS = ["raw", "ours-dense", "ours-kv", "ours-4g", "ours-hbh", "ours-rank256", "splitzip", "adacode"]


@dataclass
class SendSpec:
    req_id: str
    key: str
    dest: str
    codec: str
    block_ids: list[int]
    t0: int
    t1: int
    conv: str | None = None
    finished_at: float = 0.0
    verify: bool = False
    steps: list | None = None                        # [(start, n, M)] of the prefill steps (T2's layer 0)
    prompt: list | None = None                       # token ids from position 0 (T2's layer 0)
    chunks: list | None = None                       # tier store: [(chunk index, hash)]


@dataclass
class RecvSpec:
    req_id: str
    key: str
    block_ids: list[int]
    t0: int
    t1: int
    admitted_at: float = 0.0
    prompt: list | None = None                       # token ids from position 0 (T2's layer 0)
    chunks: list | None = None                       # tier load: [(chunk index, hash)]


@dataclass
class TKVMetadata(KVConnectorMetadata):
    sends: list[SendSpec] = field(default_factory=list)
    recvs: list[RecvSpec] = field(default_factory=list)
    cancels: list[str] = field(default_factory=list)


def block_hashes(tokens, block_size=16):
    """Chained hashes of full token blocks (what a prefix cache keys on), for the reuse-opportunity log."""
    import hashlib
    out, h = [], b""
    for a in range(0, len(tokens) - len(tokens) % block_size, block_size):
        h = hashlib.blake2b(h + bytes(json.dumps(tokens[a:a + block_size]), "ascii"), digest_size=8).digest()
        out.append(h.hex())
    return out


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Log:
    def __init__(self, log_dir, role):
        self.path = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.path = os.path.join(log_dir, f"tkv_{role}_{os.getpid()}.jsonl")
        self.lock = threading.Lock()

    def write(self, rec):
        if self.path is None:
            return
        with self.lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")


class TKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig" = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.cfg = dict(self._kv_transfer_config.kv_connector_extra_config or {})
        self.block_size = int(vllm_config.cache_config.block_size)
        self.producer = bool(self._kv_transfer_config.is_kv_producer)
        side = "P" if self.producer else "D"
        if role == KVConnectorRole.SCHEDULER:
            self.s = _SchedulerSide(self.cfg, self.block_size, self.producer, _Log(self.cfg.get("log_dir"), side + "s"))
            self.w = None
        else:
            self.s = None
            self.w = _WorkerSide(self.cfg, self.block_size, self.producer, _Log(self.cfg.get("log_dir"), side + "w"))

    # ------------------------------------------------------------------------------------------ worker side
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self.w.register(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, TKVMetadata)
        self.w.on_metadata(meta)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata",
                      **kwargs: Any) -> None:
        return

    def wait_for_save(self):
        return

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        return self.w.results(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self.w.take_failed_blocks()

    def shutdown(self):
        if self.w is not None:
            self.w.close()

    # --------------------------------------------------------------------------------------- scheduler side
    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        return self.s.matched(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        self.s.after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        self.s.record_steps(scheduler_output)
        return self.s.build_meta()

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        self.s.on_output(connector_output)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self.s.finished(request, block_ids)

    def has_pending_push_work(self) -> bool:
        return self.s.pending()

    def reset_cache(self) -> bool | None:
        """vLLM's POST /reset_prefix_cache?reset_external=true: the CPU tier is emptied (colocated engines)."""
        tier = (self.s or self.w).tier
        if tier is None:
            return None
        tier.clear()
        return True


def _tier_of(cfg):
    t = cfg.get("tier")
    if not t:
        return None
    from .tier import get_tier
    return get_tier("tkv", int(float(t.get("budget_gb", 8)) * (1 << 30)), int(t.get("chunk", 256)))


class _SchedulerSide:
    def __init__(self, cfg, block_size, producer, log):
        self.cfg, self.bs, self.producer, self.log = cfg, block_size, producer, log
        self.to_send: dict[str, SendSpec] = {}       # finished prefills not yet handed to the worker
        self.sending: set[str] = set()               # handed over, blocks held until the worker reports
        self.to_recv: dict[str, RecvSpec] = {}
        self.receiving: set[str] = set()
        self.cancels: list[str] = []
        self.steps: dict[str, list] = {}             # request -> [(start, n, M)] of its forward steps
        self.tier = _tier_of(cfg)
        self.hit: dict[str, tuple] = {}              # tier: request -> (computed before, hit end, hashes)

    def record_steps(self, so):
        """Each scheduled request's (first position, tokens, rows of the step): the step's row count M decides
        the matrix product's algorithm, which T2's layer-0 recompute repeats."""
        if so is None:
            return
        try:
            M = int(so.total_num_scheduled_tokens)
            start = {}
            for r in getattr(so, "scheduled_new_reqs", None) or ():
                start[r.req_id] = int(r.num_computed_tokens)
            cached = getattr(so, "scheduled_cached_reqs", None)
            if cached is not None:
                for rid, n in zip(getattr(cached, "req_ids", ()) or (), getattr(cached, "num_computed_tokens", ()) or ()):
                    start[rid] = int(n)
            for rid, n in so.num_scheduled_tokens.items():
                if rid in start:
                    self.steps.setdefault(rid, []).append((start[rid], int(n), M))
        except Exception as exc:                     # a vLLM whose scheduler output differs: T2 codes layer 0
            if not getattr(self, "_steps_warned", False):
                self._steps_warned = True
                logger.warning("TKVConnector: prefill steps not recorded (%r); T2 will code layer 0", exc)

    # decode
    def matched(self, request, num_computed):
        if self.tier is not None:
            return self._tier_matched(request, num_computed)
        p = request.kv_transfer_params or {}
        if self.producer or not p.get("t_recv") or p.get("_t_alloc"):
            return 0, False
        T = len(request.prompt_token_ids or [])
        if not p.get("_t_logged"):                   # what this engine already holds (its own prefix cache)
            p["_t_logged"] = True
            self.log.write({"event": "match", "key": p.get("t_key"), "prompt_tokens": T,
                            "local_hit": int(num_computed), "time": time.time()})
        n = T - num_computed
        return (n, True) if n > 0 else (0, False)

    def after_alloc(self, request, blocks, num_external):
        if self.tier is not None:
            return self._tier_after_alloc(request, blocks, num_external)
        p = request.kv_transfer_params or {}
        if self.producer or not p.get("t_recv") or p.get("_t_alloc") or num_external <= 0:
            return
        T = len(request.prompt_token_ids or [])
        spec = RecvSpec(req_id=request.request_id, key=p["t_key"], block_ids=list(blocks.get_block_ids()[0]),
                        t0=T - num_external, t1=T, admitted_at=time.time(),
                        prompt=list(request.prompt_token_ids or []))
        self.to_recv[request.request_id] = spec
        self.receiving.add(request.request_id)
        p["_t_alloc"] = True                         # a re-entry (preemption) must not register twice

    # prefill
    def finished(self, request, block_ids):
        from vllm.v1.request import RequestStatus
        p = request.kv_transfer_params or {}
        rid = request.request_id
        steps = self.steps.pop(rid, None)
        if self.tier is not None:
            return self._tier_finished(request, block_ids, steps)
        if not self.producer:
            if rid in self.receiving:                # gone while its transfer is pending: the worker drops it
                self.cancels.append(rid)
                return True, None
            return False, None
        if not p.get("t_send"):
            return False, None
        if request.status not in (RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_STOPPED) or \
                not block_ids:
            self.log.write({"event": "not_sent", "key": p.get("t_key"), "status": str(request.status)})
            return False, None
        prompt = list(request.prompt_token_ids or [])
        T = min(len(prompt), int(request.num_computed_tokens))
        t0 = int(p.get("t_start", 0))
        codec = p.get("t_codec", "raw")
        t2 = codec.startswith("t2-")
        spec = SendSpec(req_id=rid, key=p["t_key"], dest=p["t_dest"], codec=codec,
                        block_ids=list(block_ids), t0=t0, t1=T, conv=p.get("t_conv"), finished_at=time.time(),
                        verify=bool(p.get("t_verify", self.cfg.get("verify", False))),
                        steps=steps if t2 else None, prompt=prompt if t2 else None)
        self.to_send[rid] = spec
        self.sending.add(rid)
        self.log.write({"event": "prefill_done", "key": spec.key, "conv": spec.conv, "dest": spec.dest,
                        "codec": spec.codec, "tokens": T, "t0": t0, "time": spec.finished_at,
                        "blocks": block_hashes(prompt[:T], self.bs)})
        return True, {"t_key": spec.key, "t_tokens": T}

    # tier (colocated engines)
    def _tier_matched(self, request, num_computed):
        from .tier import chunk_hashes
        rid = request.request_id
        if rid in self.receiving:
            return 0, False
        prompt = list(request.prompt_token_ids or [])
        hashes = chunk_hashes(prompt, self.tier.chunk)
        end = self.tier.prefix_hits(hashes) * self.tier.chunk
        if end <= num_computed:
            return 0, False
        self.hit[rid] = (int(num_computed), end, hashes)
        return end - int(num_computed), True

    def _tier_after_alloc(self, request, blocks, num_external):
        rid = request.request_id
        got = self.hit.pop(rid, None)
        if got is None or num_external <= 0:
            return
        t0, _, hashes = got
        t1 = t0 + int(num_external)
        c = self.tier.chunk
        spec = RecvSpec(req_id=rid, key=f"tier:{rid}", block_ids=list(blocks.get_block_ids()[0]), t0=t0, t1=t1,
                        admitted_at=time.time(), prompt=list(request.prompt_token_ids or []),
                        chunks=[(i, hashes[i]) for i in range(t0 // c, -(-t1 // c))])
        self.to_recv[rid] = spec
        self.receiving.add(rid)

    def _tier_finished(self, request, block_ids, steps):
        from vllm.v1.request import RequestStatus
        from .tier import chunk_hashes
        rid = request.request_id
        if rid in self.receiving:                    # gone while its load is pending
            self.cancels.append(rid)
            return True, None
        if request.status not in (RequestStatus.FINISHED_LENGTH_CAPPED, RequestStatus.FINISHED_STOPPED) or \
                not block_ids:
            return False, None
        prompt = list(request.prompt_token_ids or [])
        c = self.tier.chunk
        done = min(len(prompt), int(request.num_computed_tokens)) // c
        hashes = chunk_hashes(prompt[:done * c], c)
        todo = [(i, h) for i, h in enumerate(hashes) if h not in self.tier]
        if not todo:
            return False, None
        spec = SendSpec(req_id=rid, key=f"tier:{rid}", dest="tier", codec="tier", block_ids=list(block_ids),
                        t0=todo[0][0] * c, t1=(todo[-1][0] + 1) * c, finished_at=time.time(), steps=steps,
                        prompt=prompt[:done * c], chunks=todo)
        self.to_send[rid] = spec
        self.sending.add(rid)
        return True, None

    def build_meta(self):
        m = TKVMetadata(sends=list(self.to_send.values()), recvs=list(self.to_recv.values()),
                        cancels=list(self.cancels))
        self.to_send.clear()
        self.to_recv.clear()
        self.cancels.clear()
        return m

    def on_output(self, out):
        for r in out.finished_sending or ():
            self.sending.discard(r)
        for r in out.finished_recving or ():
            self.receiving.discard(r)
        for r in getattr(out, "failed_recving", None) or ():
            self.receiving.discard(r)

    def pending(self):
        return bool(self.to_send or self.sending)


class _WorkerSide:
    def __init__(self, cfg, block_size, producer, log):
        self.cfg, self.bs, self.producer, self.log = cfg, block_size, producer, log
        self.paged = None
        self.codecs = {}
        self.meta = {}
        self.sender = self.receiver = None
        self.pending: dict[str, RecvSpec] = {}       # decode: admitted requests waiting for their transfer
        self.sent_ready: set[str] = set()            # prefill: blocks no longer needed
        self.finished_seen: set[str] = set()
        self.recv_ok: set[str] = set()
        self.recv_failed: set[str] = set()
        self.failed_blocks: set[int] = set()
        self.timeout = float(cfg.get("recv_timeout_s", 600))
        self.gpu_budget = int(cfg.get("gpu_send_budget", 2 << 30))
        self.gpu_queued = 0
        self.qlock = threading.Lock()
        self._last_expire = time.time()
        self.tier = _tier_of(cfg)
        self.tier_cfg = dict(cfg.get("tier") or {})
        self.tier_codec = self.tier_cfg.get("codec", "raw")
        self.tier_gen = None
        self._ctl_mtime = None
        self._last_tier_log = time.time()
        self.store_q = None
        self.store_pending = 0

    def _ctx(self, codec, prompt, steps=None):
        """T2's layer-0 context for a request (None for the other codecs, or without the token ids)."""
        if not getattr(codec, "needs_ctx", False) or prompt is None:
            return None
        from .layer0 import Layer0Context
        return Layer0Context(prompt, steps)

    def register(self, kv_caches):
        from .codecs import build
        from .paged import Paged, layer_order
        names = layer_order([n for n, t in kv_caches.items() if torch.is_tensor(t) and t.dim() == 4])
        caches = [kv_caches[n] for n in names]
        device = caches[0].device
        self.codecs, errors, self.meta = build(self.cfg.get("codecs_file"), device,
                                               self.cfg.get("codecs", DEFAULT_CODECS),
                                               self.cfg.get("splitzip_source"), self.cfg.get("adacode_source"),
                                               log=logger.warning, exactkv_source=self.cfg.get("exactkv_source"),
                                               exans_info=self.cfg.get("exans_info"))
        hd = int(self.meta.get("hd", caches[0].shape[3] // 2))
        heads = int(self.meta["C"]) // hd if self.meta else int(caches[0].shape[1])
        if self.meta and len(caches) != int(self.meta["L"]):
            raise ValueError(f"{len(caches)} attention layers, the codec constants have {self.meta['L']}")
        self.paged = Paged(caches, self.bs, heads, hd)
        self.device = device
        warm = self.warm_up()
        self.log.write({"event": "ready", "role": "P" if self.producer else "D", "codecs": sorted(self.codecs),
                        "codec_errors": errors, "layers": len(caches), "view": list(caches[0].shape),
                        "block_size": self.bs, "warm_up": warm, "time": time.time()})
        if self.tier is not None:
            import queue
            self.store_q = queue.Queue()
            self.store_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
            threading.Thread(target=self._store_loop, daemon=True, name="tkv-tier-store").start()
            self._tier_control()
        elif self.producer:
            from .transport import Link, Sender
            link = Link(gbps=self.cfg.get("gbps"), control=self.cfg.get("link_control"))
            self.sender = Sender(link, device, mode=self.cfg.get("transport", "ring"), log=logger.warning)
        else:
            from .transport import Receiver
            host, port = self.cfg["listen"].rsplit(":", 1)
            self.receiver = Receiver(host, int(port), device, ring_bytes=int(self.cfg.get("ring_bytes", 1 << 30)),
                                     log=logger.warning, gpu_budget=self.cfg.get("gpu_recv_budget"),
                                     gpu_ring_bytes=int(self.cfg.get("gpu_ring_bytes", 512 << 20)))
        logger.info("TKVConnector worker ready: %s codecs %s", "prefill" if self.producer else "decode",
                    sorted(self.codecs))

    def warm_up(self):
        """Compile every codec's kernels once (a 256-token round trip of random words) and check exactness."""
        from .wire import bundle, unbundle
        L = len(self.paged.caches)
        W = 2 * self.paged.heads * self.paged.hd
        g = torch.Generator(device="cpu").manual_seed(0)
        words = (torch.randn(L, 256, W, generator=g) * 0.05).to(torch.bfloat16).view(torch.int16).to(self.device)
        out = {}
        for n, c in list(self.codecs.items()):
            if getattr(c, "tier_only", False):
                try:
                    state, _, _ = c.tier_store(words)
                    back = torch.empty_like(words)
                    c.tier_load(state, lambda l, w: back[l].copy_(w))
                    out[n] = bool(torch.equal(back, words))
                except Exception as exc:
                    out[n] = repr(exc)[:200]
                if out[n] is not True:
                    logger.warning("TKVConnector: codec %s failed its warm-up (%s); left out", n, out[n])
                    self.codecs.pop(n, None)
                continue
            try:
                parts, meta = c.encode(words)
                h, blob = bundle(parts, meta)
                p2, m2 = unbundle(h, blob)
                back = torch.empty_like(words)
                c.decode(p2, m2, lambda l, w: back[l].copy_(w))
                out[n] = bool(torch.equal(back, words))
                if not out[n]:
                    raise ValueError("round trip not exact")
            except Exception as exc:
                out[n] = repr(exc)[:200]
                logger.warning("TKVConnector: codec %s failed its warm-up (%s); left out", n, out[n])
                self.codecs.pop(n, None)
        return out

    # ------------------------------------------------------------------------------------------------ prefill
    def _send(self, spec: SendSpec):
        if spec.dest == "tier":
            return self._tier_gather(spec)
        from .wire import bundle, payload_bytes, word_checksum
        rec = {"event": "send", "key": spec.key, "req": spec.req_id, "codec": spec.codec, "dest": spec.dest,
               "tokens": spec.t1 - spec.t0, "t0": spec.t0, "conv": spec.conv, "finished_at": spec.finished_at,
               "picked_at": time.time()}
        codec = self.codecs.get(spec.codec)
        try:
            if codec is None:
                raise ValueError(f"codec {spec.codec} not loaded here")
            sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda *a: None)
            t = time.perf_counter()
            words = self.paged.gather(spec.block_ids, spec.t0, spec.t1)
            sync(self.device)
            rec["gather_ms"] = 1e3 * (time.perf_counter() - t)
            self.sent_ready.add(spec.req_id)         # the blocks are copied out: vLLM may free them
            check = word_checksum(words) if spec.verify else None
            t = time.perf_counter()
            ctx = self._ctx(codec, spec.prompt, spec.steps)
            parts, meta = (codec.encode(words, pos0=spec.t0, ctx=ctx) if ctx is not None
                           else codec.encode(words, pos0=spec.t0))
            header, blob = bundle(parts, dict(meta, checksum=check, key=spec.key, t0=spec.t0, sent_at=time.time()))
            for k in ("set", "layer0", "layer0_check"):
                if k in meta:
                    rec[k] = meta[k]
            sync(self.device)
            rec["encode_ms"] = 1e3 * (time.perf_counter() - t)
            rec["raw_bytes"] = int(words.numel() * 2)
            rec["coded_bytes"] = int(payload_bytes(header))
            rec["blob_bytes"] = int(blob.numel())
            rec["bits"] = 8.0 * rec["coded_bytes"] / words.numel()
            del words, parts
            with self.qlock:
                on_gpu = self.gpu_queued + blob.numel() <= self.gpu_budget
                if on_gpu:
                    self.gpu_queued += blob.numel()
            if not on_gpu:
                blob = blob.cpu()
            rec["queued_on"] = "gpu" if on_gpu else "host"
            nbytes = int(blob.numel())

            def done(r, on_gpu=on_gpu, nbytes=nbytes, base=rec):
                if on_gpu:
                    with self.qlock:
                        self.gpu_queued -= nbytes
                self.log.write(dict(base, **r))
            self.sender.send(spec.dest, spec.key, header, blob, on_done=done)
        except Exception as exc:
            self.sent_ready.add(spec.req_id)
            rec["error"] = repr(exc)[:300]
            self.log.write(rec)
            logger.warning("TKVConnector: send of %s failed: %s", spec.key, rec["error"])

    # ------------------------------------------------------------------------------------------------- decode
    def _install(self, spec: RecvSpec, got):
        from .wire import Checksum, unbundle
        header, blob, trec = got
        rec = {"event": "install", "key": spec.key, "req": spec.req_id, "tokens": spec.t1 - spec.t0,
               "admitted_at": spec.admitted_at, "taken_at": time.time(), **{f"link_{k}": v for k, v in trec.items()}}
        try:
            parts, meta = unbundle(header, blob if blob.device == self.device else blob.to(self.device))
            codec = self.codecs[meta["codec"]]
            rec["codec"] = meta["codec"]
            st0 = int(meta.get("t0", 0))
            if st0 > spec.t0 or st0 + int(meta["T"]) < spec.t1:
                raise ValueError(f"transfer holds tokens {st0}..{st0 + int(meta['T'])}, request needs "
                                 f"{spec.t0}..{spec.t1}")
            a, b = spec.t0 - st0, spec.t1 - st0      # the part this engine does not have yet
            blk, off = self.paged.slots(spec.block_ids, spec.t0, spec.t1)
            want = meta.get("checksum")
            ck = Checksum() if want is not None else None

            def sink(l, w):
                if ck is not None:
                    ck.add(w)
                self.paged.scatter_layer(l, w[a:b], blk, off)
            sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda *x: None)
            t = time.perf_counter()
            ctx = self._ctx(codec, spec.prompt)
            with torch.inference_mode():
                if ctx is not None:
                    codec.decode(parts, meta, sink, ctx=ctx)
                else:
                    codec.decode(parts, meta, sink)
            sync(self.device)
            for k in ("set", "layer0"):
                if k in meta:
                    rec[k] = meta[k]
            rec["decode_install_ms"] = 1e3 * (time.perf_counter() - t)
            rec["exact"] = None if want is None else (ck.value() == want)
            if rec["exact"] is False:
                raise ValueError("checksum mismatch: the installed KV differs from the sender's")
            self.recv_ok.add(spec.req_id)
        except Exception as exc:
            rec["error"] = repr(exc)[:300]
            self.recv_failed.add(spec.req_id)
            self.failed_blocks.update(spec.block_ids[spec.t0 // self.bs:])
            logger.warning("TKVConnector: install of %s failed: %s", spec.key, rec["error"])
        finally:
            if self.receiver is not None:
                self.receiver.release(blob)
        rec["done_at"] = time.time()
        self.log.write(rec)

    # ------------------------------------------------------------------------------------------------- shared
    def on_metadata(self, meta: TKVMetadata):
        for spec in meta.sends:
            self._send(spec)
        for spec in meta.recvs:
            self.pending[spec.req_id] = spec
        for rid in meta.cancels:
            spec = self.pending.pop(rid, None)
            if spec is not None and self.receiver is not None and not spec.chunks:
                self.receiver.drop(spec.key)
                self.log.write({"event": "cancel", "key": spec.key, "req": rid, "time": time.time()})
            self.recv_ok.add(rid)                    # lets vLLM free the blocks of the finished request

    # ---------------------------------------------------------------------------------------------------- tier
    def _tier_control(self):
        """The benchmark's control file: a new generation empties the tier and sets its codec."""
        path = self.tier_cfg.get("control")
        if not path:
            return
        try:
            m = os.stat(path).st_mtime_ns
            if m == self._ctl_mtime:
                return
            self._ctl_mtime = m
            with open(path) as f:
                c = json.loads(f.read())
        except (OSError, ValueError):
            return
        if c.get("generation") != self.tier_gen:
            self.tier_gen = c.get("generation")
            self.tier_codec = c.get("codec", self.tier_codec)
            self.tier.clear()
            self.log.write({"event": "tier_reset", "generation": self.tier_gen, "codec": self.tier_codec,
                            "time": time.time()})

    def _tier_gather(self, spec: SendSpec):
        """In the engine step: copy the chunks to store out of the paged cache (the blocks then go back to vLLM);
        the background thread codes them and puts them in host memory."""
        rec = {"event": "tier_store", "key": spec.key, "req": spec.req_id, "codec": self.tier_codec,
               "chunks": len(spec.chunks or ()), "finished_at": spec.finished_at}
        try:
            cap = int(float(self.tier_cfg.get("max_pending_gb", 2)) * (1 << 30))
            nbytes = (spec.t1 - spec.t0) * 2 * self.paged.heads * self.paged.hd * 2 * len(self.paged.caches)
            if self.store_pending + nbytes > cap:
                raise RuntimeError("the store queue is full (the background coding is behind); chunks not kept")
            t = time.perf_counter()
            words = self.paged.gather(spec.block_ids, spec.t0, spec.t1)
            ev = None
            if self.device.type == "cuda":
                ev = torch.cuda.Event()
                ev.record()
            rec["gather_ms"] = 1e3 * (time.perf_counter() - t)
            with self.qlock:
                self.store_pending += nbytes
            self.store_q.put((spec, words, ev, nbytes, self.tier_codec, self.tier.generation, rec))
        except Exception as exc:
            rec["error"] = repr(exc)[:300]
            self.log.write(rec)
        self.sent_ready.add(spec.req_id)

    def _store_loop(self):
        from .wire import bundle, payload_bytes, word_checksum
        while True:
            item = self.store_q.get()
            if item is None:
                return
            spec, words, ev, nbytes, cname, gen, rec = item
            try:
                codec = self.codecs.get(cname)
                if codec is None:
                    raise ValueError(f"tier codec {cname} not loaded")
                c = self.tier.chunk
                t = time.perf_counter()
                stored = sbytes = 0
                ctx_s = torch.cuda.stream(self.store_stream) if self.store_stream is not None else _null_ctx()
                with ctx_s, torch.inference_mode():
                    if ev is not None:
                        self.store_stream.wait_event(ev)
                    for i, h in spec.chunks:
                        if self.tier.generation != gen:
                            break                     # the benchmark moved on: drop the rest
                        a = i * c - spec.t0
                        w = words[:, a:a + c]
                        check = word_checksum(w) if self.tier_cfg.get("verify", True) else None
                        if getattr(codec, "tier_only", False):
                            state, nb, _ = codec.tier_store(w.contiguous())
                            entry = {"codec": cname, "state": state, "t0": i * c}
                        else:
                            ctx = self._ctx(codec, spec.prompt, spec.steps)
                            parts, meta = (codec.encode(w, pos0=i * c, ctx=ctx) if ctx is not None
                                           else codec.encode(w, pos0=i * c))
                            header, blob = bundle(parts, meta)
                            host = torch.empty(blob.numel(), dtype=torch.uint8, pin_memory=self.device.type == "cuda")
                            host.copy_(blob, non_blocking=True)
                            if self.store_stream is not None:
                                self.store_stream.synchronize()
                            nb = payload_bytes(header)
                            entry = {"codec": cname, "header": header, "data": host, "t0": i * c}
                        entry["checksum"] = check
                        if self.tier.put(h, entry, nb):
                            stored += 1
                            sbytes += nb
                rec.update(store_ms=1e3 * (time.perf_counter() - t), stored=stored, stored_bytes=sbytes,
                           raw_bytes=int(words.numel() * 2), done_at=time.time())
            except Exception as exc:
                rec["error"] = repr(exc)[:300]
                logger.warning("TKVConnector: tier store of %s failed: %s", spec.key, rec["error"])
            finally:
                with self.qlock:
                    self.store_pending -= nbytes
                del words
            self.log.write(rec)

    def _tier_load(self, spec: RecvSpec):
        """A prefix hit: every chunk from host memory to the GPU, decoded into the request's blocks; each chunk's
        decoded words checked against the checksum taken when it was stored (bit-exact or the load fails)."""
        from .wire import Checksum, unbundle
        rec = {"event": "install", "kind": "tier", "key": spec.key, "req": spec.req_id, "tokens": spec.t1 - spec.t0,
               "admitted_at": spec.admitted_at, "taken_at": time.time(), "chunks": len(spec.chunks or ())}
        try:
            c = self.tier.chunk
            sync = torch.cuda.synchronize if self.device.type == "cuda" else (lambda *x: None)
            t = time.perf_counter()
            h2d = dec = 0.0
            nbytes = exact = 0
            for i, h in spec.chunks:
                e = self.tier.get(h)
                if e is None:
                    raise KeyError(f"chunk {i} left the tier before its load")
                a, b = max(spec.t0, i * c), min(spec.t1, (i + 1) * c)
                blk, off = self.paged.slots(spec.block_ids, a, b)
                lo, hi = a - i * c, b - i * c
                codec = self.codecs[e["codec"]]
                ck = Checksum() if e.get("checksum") is not None else None

                def sink(l, w, lo=lo, hi=hi, blk=blk, off=off, ck=ck):
                    if ck is not None:
                        ck.add(w)
                    self.paged.scatter_layer(l, w[lo:hi], blk, off)
                with torch.inference_mode():
                    if getattr(codec, "tier_only", False):
                        t1 = time.perf_counter()
                        codec.tier_load(e["state"], sink)
                        sync(self.device)
                        dec += time.perf_counter() - t1
                    else:
                        t1 = time.perf_counter()
                        blob = e["data"].to(self.device, non_blocking=True)
                        sync(self.device)
                        t2 = time.perf_counter()
                        h2d += t2 - t1
                        parts, meta = unbundle(e["header"], blob)
                        ctx = self._ctx(codec, spec.prompt)
                        if ctx is not None:
                            codec.decode(parts, meta, sink, ctx=ctx)
                        else:
                            codec.decode(parts, meta, sink)
                        sync(self.device)
                        dec += time.perf_counter() - t2
                nbytes += e["bytes"]
                if ck is not None:
                    if ck.value() != e["checksum"]:
                        raise ValueError(f"chunk {i}: the decoded words differ from the stored ones")
                    exact += 1
            rec.update(exact_chunks=exact, codec=self.tier_codec, decode_install_ms=1e3 * (time.perf_counter() - t), h2d_ms=1e3 * h2d,
                       decode_ms=1e3 * dec, loaded_bytes=nbytes, raw_bytes=(spec.t1 - spec.t0) * 2 *
                       self.paged.heads * self.paged.hd * 2 * len(self.paged.caches))
            self.recv_ok.add(spec.req_id)
        except Exception as exc:
            rec["error"] = repr(exc)[:300]
            self.recv_failed.add(spec.req_id)
            self.failed_blocks.update(spec.block_ids[spec.t0 // self.bs:])
            logger.warning("TKVConnector: tier load of %s failed: %s", spec.key, rec["error"])
        rec["done_at"] = time.time()
        self.log.write(rec)

    def results(self, finished_req_ids):
        self.finished_seen |= set(finished_req_ids or ())
        if self.tier is not None:
            self._tier_control()
            for rid, spec in list(self.pending.items()):
                del self.pending[rid]
                self._tier_load(spec)
            now = time.time()
            if now - self._last_tier_log > 30:
                self._last_tier_log = now
                self.log.write(dict(self.tier.snapshot(), event="tier", codec=self.tier_codec, time=now))
        if self.receiver is not None:
            now = time.time()
            for rid, spec in list(self.pending.items()):
                got = self.receiver.take(spec.key)
                if got is not None:
                    del self.pending[rid]
                    self._install(spec, got)
                elif now - spec.admitted_at > self.timeout:
                    del self.pending[rid]
                    self.receiver.drop(spec.key)
                    self.recv_failed.add(rid)
                    self.failed_blocks.update(spec.block_ids[spec.t0 // self.bs:])
                    self.log.write({"event": "timeout", "key": spec.key, "req": rid, "time": now})
            if now - self._last_expire > 30:          # transfers whose decode request never came (aborted)
                self._last_expire = now
                for k in self.receiver.expire(self.timeout, keep={s.key for s in self.pending.values()}):
                    self.log.write({"event": "expired", "key": k, "time": now})
        sent = self.sent_ready & self.finished_seen
        self.sent_ready -= sent
        self.finished_seen -= sent
        ok, bad = self.recv_ok, self.recv_failed
        self.recv_ok, self.recv_failed = set(), set()
        return KVConnectorTransferResults(finished_sending=sent, finished_recving=ok | bad, failed_recving=bad)

    def take_failed_blocks(self):
        b, self.failed_blocks = self.failed_blocks, set()
        return b

    def close(self):
        if self.store_q is not None:
            self.store_q.put(None)
        for x in (self.sender, self.receiver):
            if x is not None:
                try:
                    x.close()
                except Exception:
                    pass
