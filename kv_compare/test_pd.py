"""Tests of the PD protocol (kv_compare/pd) on the CPU: wire formats, the paced transport, the paged gather /
scatter, the vLLM connector's whole request flow between a prefill and a decode engine (against a stand-in of
vLLM 0.32's connector interface when vLLM is not installed), our codec through the wire (Triton interpreter), and
the reuse-opportunity analysis.

  python -m unittest kv_compare.test_pd
"""
from __future__ import annotations

import enum
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install_vllm_stub():
    """The parts of vLLM 0.32 the connector imports (only when vLLM itself is absent)."""
    try:
        import vllm  # noqa: F401
        return False
    except ImportError:
        pass
    import logging
    mods = {}
    for name in ("vllm", "vllm.distributed", "vllm.distributed.kv_transfer", "vllm.distributed.kv_transfer.kv_connector",
                 "vllm.distributed.kv_transfer.kv_connector.v1", "vllm.distributed.kv_transfer.kv_connector.v1.base",
                 "vllm.logger", "vllm.v1", "vllm.v1.request"):
        mods[name] = types.ModuleType(name)
        sys.modules[name] = mods[name]

    class KVConnectorRole(enum.Enum):
        SCHEDULER = 0
        WORKER = 1

    class KVConnectorMetadata:
        pass

    @dataclass
    class KVConnectorTransferResults:
        finished_sending: set = field(default_factory=set)
        finished_recving: set = field(default_factory=set)
        failed_recving: set = field(default_factory=set)

    class KVConnectorBase_V1:
        def __init__(self, vllm_config, role, kv_cache_config=None):
            self._connector_metadata = None
            self._vllm_config = vllm_config
            self._kv_transfer_config = vllm_config.kv_transfer_config
            self._role = role

        def bind_connector_metadata(self, m):
            self._connector_metadata = m

        def clear_connector_metadata(self):
            self._connector_metadata = None

        def _get_connector_metadata(self):
            assert self._connector_metadata is not None
            return self._connector_metadata

    class RequestStatus(enum.Enum):
        FINISHED_STOPPED = 1
        FINISHED_LENGTH_CAPPED = 2
        FINISHED_ABORTED = 3

    b = mods["vllm.distributed.kv_transfer.kv_connector.v1.base"]
    b.KVConnectorRole, b.KVConnectorMetadata = KVConnectorRole, KVConnectorMetadata
    b.KVConnectorTransferResults, b.KVConnectorBase_V1 = KVConnectorTransferResults, KVConnectorBase_V1
    mods["vllm.logger"].init_logger = logging.getLogger
    mods["vllm.v1.request"].RequestStatus = RequestStatus
    return True


class WireTests(unittest.TestCase):
    def test_bundle_round_trip(self):
        from kv_compare.pd.wire import bundle, header_from_bytes, header_to_bytes, payload_bytes, unbundle
        parts = {"a": torch.arange(7, dtype=torch.int16), "b": torch.randn(3, 5).to(torch.bfloat16),
                 "c": torch.tensor([1, 2], dtype=torch.int32), "e": torch.empty(0, dtype=torch.uint8),
                 "u": torch.arange(9, dtype=torch.int32).to(torch.uint16)}
        h, blob = bundle(parts, {"x": 1})
        self.assertEqual(blob.numel() % 16, 0)
        h2 = header_from_bytes(header_to_bytes(h))
        got, meta = unbundle(h2, blob)
        self.assertEqual(meta, {"x": 1})
        for k, v in parts.items():
            self.assertEqual(got[k].dtype, v.dtype)
            self.assertTrue(torch.equal(got[k].view(torch.int16) if v.dtype == torch.bfloat16 else got[k],
                                        v.view(torch.int16) if v.dtype == torch.bfloat16 else v))
        self.assertEqual(payload_bytes(h), 14 + 30 + 8 + 0 + 18)

    def test_segment_tails_round_trip(self):
        from kv_compare.pd.wire import compact_segments, expand_segments, segment_lengths
        S, cap, hd = 37, 200, 8
        g = torch.Generator().manual_seed(1)
        starts = torch.randint(2 * hd + 8, cap - 1, (S,), generator=g).to(torch.int32) + \
            torch.arange(S, dtype=torch.int32) * cap
        buf = torch.randint(-30000, 30000, (S * cap,), generator=g).to(torch.int16)
        packed = compact_segments(buf, starts, cap, hd, rows=5)
        self.assertEqual(packed.numel(), int(segment_lengths(starts, cap, hd).sum()))
        back = expand_segments(packed, starts, cap, hd, rows=7)
        lens = segment_lengths(starts, cap, hd)
        for s in range(S):
            a = (s + 1) * cap - int(lens[s])
            self.assertTrue(torch.equal(back[a:(s + 1) * cap], buf[a:(s + 1) * cap]))

    def test_checksum_sees_one_flipped_word(self):
        from kv_compare.pd.wire import Checksum, word_checksum
        w = torch.randint(-3000, 3000, (2, 9, 6)).to(torch.int16)
        ck = Checksum()
        for l in range(2):
            ck.add(w[l])
        self.assertEqual(ck.value(), word_checksum(w))
        w2 = w.clone()
        w2[1, 4, 2] += 1
        self.assertNotEqual(word_checksum(w2), word_checksum(w))
        w3 = w.clone()
        w3[0, 1, 1], w3[0, 2, 1] = w[0, 2, 1], w[0, 1, 1]   # two words swapped: the sum alone would miss it
        if w[0, 1, 1] != w[0, 2, 1]:
            self.assertNotEqual(word_checksum(w3), word_checksum(w))


class TransportTests(unittest.TestCase):
    def setUp(self):
        from kv_compare.pd import transport as tr
        self.tr = tr
        self.chunk = tr.CHUNK
        tr.CHUNK = 1 << 14

    def tearDown(self):
        self.tr.CHUNK = self.chunk

    def _run(self, mode, gbps, nbytes=60000, k=3, ring=1 << 16):
        tr = self.tr
        rx = tr.Receiver("127.0.0.1", 0, "cpu", ring_bytes=ring)
        ctl = os.path.join(tempfile.mkdtemp(), "link.json")
        Path(ctl).write_text(json.dumps({"gbps": gbps}))
        tx = tr.Sender(tr.Link(control=ctl), "cpu", mode=mode)
        blobs = [torch.randint(0, 255, (nbytes + i,), dtype=torch.uint8) for i in range(k)]
        recs = []
        t = time.monotonic()
        for i, b in enumerate(blobs):
            tx.send(f"127.0.0.1:{rx.port}", f"k{i}", {"meta": {"i": i}}, b, on_done=recs.append)
        got = [rx.wait(f"k{i}", 30) for i in range(k)]
        dt = time.monotonic() - t
        tx.close()
        rx.close()
        for i, g in enumerate(got):
            self.assertIsNotNone(g)
            self.assertTrue(torch.equal(g[1], blobs[i]))
            self.assertEqual(g[0]["meta"]["i"], i)
        return dt, recs, sum(b.numel() for b in blobs)

    def test_ring_and_tcp_deliver_exactly(self):
        for mode in ("ring", "tcp"):
            self._run(mode, None)

    def test_link_paces_at_the_set_speed(self):
        gbps = 0.02                                   # 2.5 MB/s: 3 x 60 kB take about 72 ms
        for mode in ("ring", "tcp"):
            dt, recs, total = self._run(mode, gbps)
            expect = 8 * total / (gbps * 1e9)
            self.assertGreaterEqual(dt, 0.9 * expect, mode)
            self.assertLess(dt, expect + 1.0, mode)
            self.assertAlmostEqual(sum(r["link_s"] for r in recs), expect, delta=0.25 * expect)

    def test_ring_smaller_than_a_transfer_wraps_with_flow_control(self):
        self._run("ring", None, nbytes=50000, k=4, ring=40000)
        self._run("ring", None, nbytes=23001, k=12, ring=37000)     # many wraps at odd offsets

    def test_drop_discards_now_or_on_arrival(self):
        tr = self.tr
        rx = tr.Receiver("127.0.0.1", 0, "cpu", ring_bytes=1 << 16)
        tx = tr.Sender(tr.Link(), "cpu", mode="ring")
        rx.drop("late")
        tx.send(f"127.0.0.1:{rx.port}", "late", {}, torch.zeros(100, dtype=torch.uint8))
        tx.send(f"127.0.0.1:{rx.port}", "kept", {}, torch.ones(100, dtype=torch.uint8))
        self.assertIsNotNone(rx.wait("kept", 10))
        self.assertIsNone(rx.take("late"))
        tx.close()
        rx.close()

    def test_unclaimed_transfers_expire(self):
        tr = self.tr
        rx = tr.Receiver("127.0.0.1", 0, "cpu", ring_bytes=1 << 16)
        tx = tr.Sender(tr.Link(), "cpu", mode="tcp")
        for k in ("old", "kept"):
            tx.send(f"127.0.0.1:{rx.port}", k, {}, torch.zeros(10, dtype=torch.uint8))
        rx.wait("kept", 10)
        tx.send(f"127.0.0.1:{rx.port}", "kept", {}, torch.zeros(10, dtype=torch.uint8))
        end = time.time() + 10
        while len(rx.done) < 2 and time.time() < end:
            time.sleep(0.01)
        time.sleep(0.05)
        self.assertEqual(rx.expire(0.01, keep={"kept"}), ["old"])
        self.assertIsNotNone(rx.take("kept"))
        tx.close()
        rx.close()

    def test_unreachable_receiver_reports_instead_of_raising(self):
        tr = self.tr
        recs = []
        tr.Sender(tr.Link(), "cpu").send("127.0.0.1:1", "x", {}, torch.zeros(4, dtype=torch.uint8),
                                         on_done=recs.append)
        self.assertIn("connect", recs[0]["error"])


class PagedTests(unittest.TestCase):
    def test_gather_scatter_and_kernel_blocks(self):
        from kv_compare.pd.paged import Paged, layer_order
        self.assertEqual(layer_order(["model.layers.10.self_attn.attn", "model.layers.2.self_attn.attn"]),
                         ["model.layers.2.self_attn.attn", "model.layers.10.self_attn.attn"])
        L, H, hd, bs = 2, 3, 4, 8
        for kb in (8, 4):                            # the view's blocks: the scheduler's, or kernel blocks inside them
            caches = [torch.randn(12 * (bs // kb), H, kb, 2 * hd).to(torch.bfloat16) for _ in range(L)]
            p = Paged(caches, bs, H, hd)
            blocks = [5, 0, 7]
            w = p.gather(blocks, 0, 20)
            self.assertEqual(tuple(w.shape), (L, 20, 2 * H * hd))
            t = 13                                   # token 13: scheduler block blocks[1] = 0, offset 5
            kv = caches[1]
            r = bs // kb
            blk, off = 0 * r + 5 // kb, 5 % kb
            x = kv[blk, :, off]
            self.assertTrue(torch.equal(w[1, t, :H * hd].view(torch.bfloat16), x[:, :hd].reshape(-1)))
            self.assertTrue(torch.equal(w[1, t, H * hd:].view(torch.bfloat16), x[:, hd:].reshape(-1)))
            dst = [torch.zeros_like(c) for c in caches]
            q = Paged(dst, bs, H, hd)
            b2, o2 = q.slots([3, 9, 1], 0, 20)
            for l in range(L):
                q.scatter_layer(l, w[l], b2, o2)
            self.assertTrue(torch.equal(q.gather([3, 9, 1], 0, 20), w))


def _vcfg(producer, extra, block_size=16):
    return SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size),
                           kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra,
                                                              is_kv_producer=producer))


def _caches(L, nb, H, bs, hd, seed):
    g = torch.Generator().manual_seed(seed)
    return {f"model.layers.{l}.self_attn.attn": (torch.randn(nb, H, bs, 2 * hd, generator=g) * 0.1).to(torch.bfloat16)
            for l in range(L)}


class ConnectorFlowTests(unittest.TestCase):
    """A prefill engine and a decode engine (scheduler and worker sides each), raw codec, CPU caches: the request's
    prompt KV ends up in the decode engine's blocks exactly, the hooks report what vLLM expects."""

    @classmethod
    def setUpClass(cls):
        install_vllm_stub()
        from kv_compare.pd import connector as cn
        cls.cn = cn
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.v1.request import RequestStatus
        cls.Role, cls.Status = KVConnectorRole, RequestStatus

    def make(self, codecs=("raw",), timeout=600):
        log_dir = tempfile.mkdtemp()
        L, H, bs, hd = 3, 2, 16, 8
        d_extra = {"codecs": list(codecs), "listen": "127.0.0.1:0", "log_dir": log_dir, "recv_timeout_s": timeout}
        p_extra = {"codecs": list(codecs), "log_dir": log_dir, "transport": "ring", "ring_bytes": 1 << 20}
        Ps, Pw = (self.cn.TKVConnector(_vcfg(True, p_extra), r) for r in (self.Role.SCHEDULER, self.Role.WORKER))
        Ds, Dw = (self.cn.TKVConnector(_vcfg(False, dict(d_extra, ring_bytes=1 << 20)), r)
                  for r in (self.Role.SCHEDULER, self.Role.WORKER))
        pc, dc = _caches(L, 20, H, bs, hd, 1), _caches(L, 20, H, bs, hd, 2)
        Pw.register_kv_caches(pc)
        Dw.register_kv_caches(dc)
        dest = f"127.0.0.1:{Dw.w.receiver.port}"
        self.addCleanup(Pw.shutdown)
        self.addCleanup(Dw.shutdown)
        return Ps, Pw, Ds, Dw, pc, dc, dest, log_dir

    def step(self, sched, worker, finished=()):
        worker.bind_connector_metadata(sched.build_connector_meta(None))
        worker.start_load_kv(None)
        r = worker.get_transfer_results(set(finished))
        worker.clear_connector_metadata()
        sched.update_connector_output(SimpleNamespace(finished_sending=r.finished_sending,
                                                      finished_recving=r.finished_recving,
                                                      failed_recving=r.failed_recving))
        return r

    def flow(self, local_hit=0, codec="raw"):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make()
        prompt = list(range(100, 140))                # 40 tokens: 3 blocks of 16
        p_blocks, d_blocks = [5, 2, 9], [1, 7, 3]
        dreq = SimpleNamespace(request_id="d-1", prompt_token_ids=prompt, num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K1"})
        n, asyn = Ds.get_num_new_matched_tokens(dreq, local_hit)
        self.assertEqual((n, asyn), (40 - local_hit, True))
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: (d_blocks,)), n)
        self.assertEqual(Ds.get_num_new_matched_tokens(dreq, local_hit), (0, False))   # registered once
        before = {k: v.clone() for k, v in dc.items()}
        r = self.step(Ds, Dw)
        self.assertEqual(r.finished_recving, set())
        preq = SimpleNamespace(request_id="p-1", prompt_token_ids=prompt, num_computed_tokens=40,
                               status=self.Status.FINISHED_LENGTH_CAPPED,
                               kv_transfer_params={"t_send": True, "t_key": "K1", "t_dest": dest, "t_codec": codec,
                                                   "t_verify": True})
        delay, params = Ps.request_finished(preq, p_blocks)
        self.assertTrue(delay)
        self.assertEqual(params["t_tokens"], 40)
        self.assertTrue(Ps.has_pending_push_work())
        r = self.step(Ps, Pw, finished={"p-1"})
        self.assertEqual(r.finished_sending, {"p-1"})
        self.assertFalse(Ps.has_pending_push_work())
        end = time.time() + 20
        while time.time() < end:
            r = self.step(Ds, Dw)
            if r.finished_recving:
                break
            time.sleep(0.02)
        self.assertEqual(r.finished_recving, {"d-1"})
        self.assertEqual(r.failed_recving, set())
        from kv_compare.pd.paged import Paged
        L = len(pc)
        src = Paged([pc[f"model.layers.{l}.self_attn.attn"] for l in range(L)], 16, 2, 8).gather(p_blocks, 0, 40)
        dst = Paged([dc[f"model.layers.{l}.self_attn.attn"] for l in range(L)], 16, 2, 8)
        self.assertTrue(torch.equal(dst.gather(d_blocks, local_hit, 40), src[:, local_hit:]))
        if local_hit:                                 # what the engine had stays as it was
            self.assertTrue(torch.equal(dst.gather(d_blocks, 0, local_hit),
                                        Paged([before[f"model.layers.{l}.self_attn.attn"] for l in range(L)], 16, 2,
                                              8).gather(d_blocks, 0, local_hit)))
        recs = [json.loads(l) for f in Path(log_dir).glob("tkv_*.jsonl") for l in f.read_text().splitlines()]
        inst = [x for x in recs if x.get("event") == "install"]
        self.assertEqual(len(inst), 1)
        self.assertTrue(inst[0]["exact"])
        self.assertTrue(any(x.get("event") == "match" and x["local_hit"] == local_hit for x in recs))
        self.assertTrue(any(x.get("event") == "prefill_done" and len(x["blocks"]) == 2 for x in recs))
        return recs

    def test_whole_prompt_moves_exactly(self):
        self.flow()

    def test_only_the_missing_tokens_are_installed(self):
        self.flow(local_hit=16)

    def test_aborted_decode_request_is_released_and_its_transfer_dropped(self):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make()
        dreq = SimpleNamespace(request_id="d-2", prompt_token_ids=list(range(20)), num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K2"})
        n, _ = Ds.get_num_new_matched_tokens(dreq, 0)
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: ([4, 6],)), n)
        self.step(Ds, Dw)
        dreq.status = self.Status.FINISHED_ABORTED
        self.assertEqual(Ds.request_finished(dreq, [4, 6]), (True, None))   # blocks held until the worker says
        r = self.step(Ds, Dw, finished={"d-2"})
        self.assertEqual(r.finished_recving, {"d-2"})
        self.assertIn("K2", Dw.w.receiver.dropped)

    def test_transfer_that_never_comes_fails_after_the_timeout(self):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make(timeout=0.2)
        dreq = SimpleNamespace(request_id="d-3", prompt_token_ids=list(range(20)), num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K3"})
        n, _ = Ds.get_num_new_matched_tokens(dreq, 0)
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: ([4, 6],)), n)
        self.step(Ds, Dw)
        time.sleep(0.3)
        r = self.step(Ds, Dw)
        self.assertEqual(r.failed_recving, {"d-3"})
        self.assertEqual(r.finished_recving, {"d-3"})
        self.assertEqual(Dw.get_block_ids_with_load_errors(), {4, 6})

    def test_requests_without_our_params_are_left_alone(self):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make()
        req = SimpleNamespace(request_id="x", prompt_token_ids=[1, 2, 3], num_computed_tokens=3,
                              kv_transfer_params=None, status=self.Status.FINISHED_STOPPED)
        self.assertEqual(Ds.get_num_new_matched_tokens(req, 0), (0, False))
        self.assertEqual(Ps.request_finished(req, [1]), (False, None))
        self.assertFalse(Ps.has_pending_push_work())


OURS_WIRE_CHECK = r"""
import os, sys, json, tempfile
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from kv_compare.capture import capture_ids, model_meta
from kv_compare.predict import Study
from kv_compare import fused, layerwise
from kv_compare.grouped import prepare_groups
from kv_compare.pd import codecs, export, wire
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, max_position_embeddings=2048, rope_theta=500000.0)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
names = [f"train{i}" for i in range(4)] + ["tables", "test"]
root = Path(tempfile.mkdtemp()) / "c"
reports = []
for i, n in enumerate(names):
    ids = [int(v) for v in np.random.default_rng(i).integers(0, 97, size=96)]
    reports.append(capture_ids(model, ids, root, n, torch.device("cpu"), queries=n.startswith("train")))
(root / "meta.json").write_text(json.dumps(dict(model_meta(model, "tiny", None), tokens=96, captures=reports)))
study = Study(root, "cpu", names[:4], "tables", ["test"], log=lambda m: None)
consts = fused.prepare(study, log=lambda m: None)
rc = layerwise.prepare_rank(study, consts, 64, lambda m: None)
groups = prepare_groups(study, consts, [2, 4], lambda m: None)
b = export.make_bundle(study, consts, rc, groups, 512)
p = Path(tempfile.mkdtemp()) / "b.pt"
torch.save(b, p)
codecs.PAD = 64
codecs.OUR_MODES["4g"] = 4
doc = study.tests[0]
L, W = study.L, study.W
w32 = torch.stack([doc.words(l, "cpu") for l in range(L)])
words = (w32 - 65536 * (w32 >= 32768).to(torch.int32)).to(torch.int16)
built, errors, meta = codecs.build(str(p), "cpu", ["ours-dense", "ours-kv", "ours-4g", "ours-rank256"])
assert not errors, errors
for T, pos0 in ((96, 0), (50, 7)):
    w = words[:, pos0:pos0 + T].contiguous()
    for n, c in built.items():
        parts, m = c.encode(w, pos0=pos0)
        h, blob = wire.bundle(parts, m)
        hb = wire.header_from_bytes(wire.header_to_bytes(h))
        p2, m2 = wire.unbundle(hb, blob.clone())
        got = {}
        c.decode(p2, m2, lambda l, x: got.__setitem__(l, x.clone()))
        back = torch.stack([got[l] for l in range(L)])
        assert torch.equal(back, w), (n, T, int((back != w).sum()))
        bits = 8 * wire.payload_bytes(h) / w.numel()
        print("ok", n, T, pos0, round(bits, 3))
print("ok ours wire")
"""


class OursWireTests(unittest.TestCase):
    def test_our_modes_round_trip_through_the_wire_interpreted(self):
        """Every mode of T (dense, keys then values, head groups, reduced rank): encode, only the used segment tails
        on the wire, decode layer by layer; padded token counts and a non-zero start position included."""
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is required")
        r = subprocess.run([sys.executable, "-c", OURS_WIRE_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=7200)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("ok ours wire", r.stdout)


class _FakeEngines:
    """A fake prefill engine and two fake decode engines (aiohttp) that record what the proxy sends them."""

    def __init__(self, prefill_status=200, decode_wait=0.0):
        import asyncio
        import threading
        from aiohttp import web
        self.seen = {"p": [], "d1": [], "d2": []}
        self.ports = {}
        self.prefill_status, self.decode_wait = prefill_status, decode_wait
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        def run():
            asyncio.set_event_loop(self.loop)

            async def start():
                for name in ("p", "d1", "d2"):
                    app = web.Application()
                    app.router.add_post("/v1/completions", self._handler(name))
                    runner = web.AppRunner(app)
                    await runner.setup()
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    self.ports[name] = site._server.sockets[0].getsockname()[1]
                ready.set()
            self.loop.run_until_complete(start())
            self.loop.run_forever()
        threading.Thread(target=run, daemon=True).start()
        ready.wait(10)

    def _handler(self, name):
        import asyncio
        from aiohttp import web

        async def h(request):
            body = await request.json()
            self.seen[name].append(body)
            if name == "p":
                await asyncio.sleep(0.05)
                return web.json_response({"choices": [{"text": "x"}]}, status=self.prefill_status)
            resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
            await resp.prepare(request)
            await asyncio.sleep(self.decode_wait)
            for i in range(3):
                await resp.write(f"data: {{\"i\": {i}}}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return h


@unittest.skipIf(__import__("importlib").util.find_spec("fastapi") is None, "fastapi / uvicorn / aiohttp needed")
class ProxyTests(unittest.TestCase):
    def start(self, **kw):
        import socket
        eng = _FakeEngines(**kw)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        log = Path(tempfile.mkdtemp()) / "proxy.jsonl"
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "pd_proxy.py"), "--port", str(port),
                                 "--prefill", f"http://127.0.0.1:{eng.ports['p']}",
                                 "--decode", f"http://127.0.0.1:{eng.ports['d1']}=10.0.0.1:9301",
                                 "--decode", f"http://127.0.0.1:{eng.ports['d2']}=10.0.0.1:9302",
                                 "--log", str(log), "--codec", "ours-hbh"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        def stop():
            proc.kill()
            proc.wait()
            proc.stderr.close()
        self.addCleanup(stop)
        import urllib.request
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        return eng, url, log

    def req(self, url, body, path="/v1/completions"):
        import urllib.request
        r = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as f:
            return f.read().decode()

    def test_push_routing_params_and_modes(self):
        eng, url, log = self.start()
        out = self.req(url, {"model": "m", "prompt": "hi", "max_tokens": 5, "stream": True})
        self.assertIn("[DONE]", out)
        p, d = eng.seen["p"][0], eng.seen["d1"][0]
        self.assertEqual((p["max_tokens"], p["stream"]), (1, False))
        kp, kd = p["kv_transfer_params"], d["kv_transfer_params"]
        self.assertEqual((kp["t_send"], kp["t_dest"], kp["t_codec"], kp["t_verify"]), (True, "10.0.0.1:9301", "ours-hbh",
                                                                                         False))
        self.assertEqual((kd["t_recv"], kd["t_key"]), (True, kp["t_key"]))
        self.assertEqual(d["max_tokens"], 5)
        self.req(url, {"model": "m", "prompt": "a"})                       # round robin: the second decode engine
        self.assertEqual(len(eng.seen["d2"]), 1)
        cfg = json.loads(self.req(url, {"codec": "raw", "verify": True, "mode": "local"}, "/t/config"))
        self.assertEqual((cfg["codec"], cfg["mode"]), ("raw", "local"))
        n = len(eng.seen["p"])
        self.req(url, {"model": "m", "prompt": "b"})
        self.assertEqual(len(eng.seen["p"]), n)                          # local: no prefill engine, no params
        local = [b for b in eng.seen["d1"] + eng.seen["d2"] if b["prompt"] == "b"]
        self.assertEqual(len(local), 1)
        self.assertNotIn("kv_transfer_params", local[0])
        self.req(url, {"mode": "pd", "policy": "conversation"}, "/t/config")
        for i in range(4):
            self.req(url, {"model": "m", "prompt": f"t{i}", "conversation_id": "c7"})
        dests = [json.loads(l)["dest"] for l in log.read_text().splitlines() if json.loads(l).get("conv") == "c7"]
        self.assertEqual(len(set(dests)), 1)
        self.assertNotIn("conversation_id", eng.seen["p"][-1])

    def test_failed_prefill_ends_the_decode_request(self):
        eng, url, log = self.start(prefill_status=500, decode_wait=20)
        t = time.time()
        try:
            self.req(url, {"model": "m", "prompt": "hi", "stream": True})
        except Exception:
            pass
        self.assertLess(time.time() - t, 10)
        time.sleep(0.3)
        recs = [json.loads(l) for l in log.read_text().splitlines()]
        self.assertIn("prefill engine 500", recs[-1].get("error", ""))


class BenchReportTests(unittest.TestCase):
    def test_transfer_stats_and_report(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        d = Path(tempfile.mkdtemp())
        recs = [{"event": "send", "finished_at": 5.0, "bits": 8.3, "encode_ms": 70.0, "gather_ms": 3.0, "link_s": 0.4,
                 "queued": 5.1, "send_begin": 5.2, "send_end": 5.7, "blob_bytes": 500_000_000, "queued_on": "gpu"},
                {"event": "install", "taken_at": 6.0, "decode_install_ms": 30.0, "admitted_at": 5.0, "done_at": 6.1,
                 "exact": True},
                {"event": "send", "finished_at": 99.0, "bits": 16.0}]
        (d / "tkv_x.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        st = pb.transfer_stats(pb.records(str(d), 0, 10))
        self.assertEqual((st["sends"], st["installs"], st["exact"]), (1, 1, 1))
        self.assertAlmostEqual(st["achieved_gbps"], 8.0)
        self.assertAlmostEqual(st["link_ms"], 400.0)
        res = {"env": {"model": "m", "gpus": "g", "vllm": "0.32.0", "torch": "2.13", "decodes": 3, "dataset": "x"},
               "check": {"raw": {"prompts": 4, "same_text_as_local": 3, "transfers": dict(st, exact=4), "ok": True}},
               "sweep": {"sweep_raw_10gbps_r1.0": {"link_gbps": 10, "codec": "raw", "rate": 1.0, "exit": 0,
                                                   "bench": {"request_throughput": 0.9, "median_ttft_ms": 900,
                                                             "p99_ttft_ms": 1500}, "transfers": st}},
               "colocated": {"colocated_r1.0": {"rate": 1.0, "bench": {"request_throughput": 0.8}}}}
        a = SimpleNamespace(out=d, output_len=128, ttft_slo_ms=2000, codecs=["raw"])
        pb.stage_report(a, res)
        text = (d / "summary.md").read_text()
        self.assertIn("## Link 10 Gb/s", text)
        self.assertIn("| raw | 1.0 | 0.90 |", text)
        self.assertIn("Without disaggregation", text)


class OpportunityTests(unittest.TestCase):
    def test_categories_from_logs(self):
        d = Path(tempfile.mkdtemp())
        h = [f"{i:016x}" for i in range(10)]
        recs = [
            {"event": "prefill_done", "key": "a", "dest": "D1", "tokens": 64, "blocks": h[:4], "time": 1.0},
            {"event": "match", "key": "a", "prompt_tokens": 64, "local_hit": 0, "time": 1.0},
            {"event": "prefill_done", "key": "b", "dest": "D1", "tokens": 96, "blocks": h[:6], "time": 2.0},
            {"event": "match", "key": "b", "prompt_tokens": 96, "local_hit": 32, "time": 2.0},
            {"event": "prefill_done", "key": "c", "dest": "D2", "tokens": 112, "blocks": h[:7], "time": 3.0},
            {"event": "match", "key": "c", "prompt_tokens": 112, "local_hit": 0, "time": 3.0},
        ]
        (d / "tkv_x.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        out = d / "o.json"
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pd_opportunity.py"), "--log-dir", str(d),
                            "--output", str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        o = json.loads(out.read_text())
        per = {p["key"]: p for p in o["per_request"]}
        self.assertEqual((per["a"]["gpu"], per["a"]["dram"], per["a"]["new"]), (0, 0, 64))
        self.assertEqual((per["b"]["gpu"], per["b"]["dram"], per["b"]["new"]), (32, 32, 32))   # held 4 blocks before
        self.assertEqual((per["c"]["elsewhere"], per["c"]["new"]), (96, 16))                    # D1 held 6 of them
        self.assertEqual(o["tokens"]["tokens"], 272)


if __name__ == "__main__":
    unittest.main()
