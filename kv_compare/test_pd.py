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

    def test_blobs_made_in_inference_mode(self):
        # job 30176: vLLM makes the blob, and the sender its ring, in inference mode; the sender thread then
        # failed ("Inplace update to inference tensor outside InferenceMode") and every decode waited 600 s
        tr = self.tr
        for mode in ("ring", "tcp"):
            rx = tr.Receiver("127.0.0.1", 0, "cpu", ring_bytes=1 << 16)
            tx = tr.Sender(tr.Link(), "cpu", mode=mode)
            recs = []
            with torch.inference_mode():
                blob = torch.randint(0, 255, (50000,), dtype=torch.uint8)
                tx.send(f"127.0.0.1:{rx.port}", "k", {"meta": {}}, blob, on_done=recs.append)
            got = rx.wait("k", 30)
            tx.close()
            rx.close()
            self.assertIsNotNone(got, (mode, recs))
            self.assertTrue(torch.equal(got[1], blob), mode)
            with torch.inference_mode():          # and what arrived is usable where vLLM reads it
                self.assertEqual(int(got[1].add(0).sum()), int(blob.sum()))

    def test_a_failed_send_fails_the_receiver_now(self):
        tr = self.tr
        rx = tr.Receiver("127.0.0.1", 0, "cpu", ring_bytes=1 << 16)
        link = tr.Link()
        calls = []

        def reserve(n):
            calls.append(n)
            if len(calls) == 2:
                raise RuntimeError("the second chunk fails")
            return 0.0, 0.0

        link.reserve = reserve
        tx = tr.Sender(link, "cpu", mode="ring")
        recs = []
        t = time.monotonic()
        tx.send(f"127.0.0.1:{rx.port}", "k", {"meta": {}}, torch.zeros(50000, dtype=torch.uint8),
                on_done=recs.append)
        self.assertIsNone(rx.wait("k", 30))
        self.assertLess(time.monotonic() - t, 10, "failed at once, not at the timeout")
        self.assertIn("the second chunk fails", rx.failure("k"))
        self.assertIsNone(rx.failure("k"), "reported once")
        end = time.monotonic() + 5                      # the sender's record follows its abort message
        while not recs and time.monotonic() < end:
            time.sleep(0.01)
        self.assertIn("error", recs[0])
        tx.close()
        rx.close()

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

    def test_a_shared_link_books_every_senders_chunks_in_turn(self):
        """Two senders (prefill engines) behind one link: the second one's chunk starts when the first one's ends."""
        from kv_compare.pd.transport import Link
        path = os.path.join(tempfile.mkdtemp(), "link.bin")
        a, b = Link(gbps=8.0, shared=path), Link(gbps=8.0, shared=path)
        s1, e1 = a.reserve(10 ** 9, now=100.0)       # 1 s at 8 Gb/s
        s2, e2 = b.reserve(10 ** 9, now=100.5)
        s3, e3 = a.reserve(10 ** 9, now=103.0)
        self.assertEqual((s1, e1, s2, e2, s3, e3), (100.0, 101.0, 101.0, 102.0, 103.0, 104.0))
        c = Link(gbps=8.0)                            # a link of its own
        self.assertEqual(c.reserve(10 ** 9, now=100.5), (100.5, 101.5))

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

    def make(self, codecs=("raw",), timeout=600, async_install=False, pipeline=0, async_encode=False,
             gather_budget=2 << 30, trace=False, p_more=None, d_more=None):
        log_dir = tempfile.mkdtemp()
        L, H, bs, hd = 3, 2, 16, 8
        d_extra = {"codecs": list(codecs), "listen": "127.0.0.1:0", "log_dir": log_dir, "recv_timeout_s": timeout,
                   "async_install": async_install, "pipeline_tokens": pipeline, "trace_pages": trace, **(d_more or {})}
        p_extra = {"codecs": list(codecs), "log_dir": log_dir, "transport": "ring", "ring_bytes": 1 << 20,
                   "pipeline_tokens": pipeline, "async_encode": async_encode, "gather_budget": gather_budget,
                   "trace_pages": trace, **(p_more or {})}
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
        with torch.inference_mode():                  # where vLLM's model runner calls the worker side
            worker.bind_connector_metadata(sched.build_connector_meta(None))
            worker.start_load_kv(None)
            r = worker.get_transfer_results(set(finished))
            worker.clear_connector_metadata()
        sched.update_connector_output(SimpleNamespace(finished_sending=r.finished_sending,
                                                      finished_recving=r.finished_recving,
                                                      failed_recving=r.failed_recving))
        return r

    def flow(self, local_hit=0, codec="raw", lossy=False, async_install=False, pipeline=0, async_encode=False,
             gather_budget=2 << 30, pieced=False):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make(async_install=async_install, pipeline=pipeline,
                                                          async_encode=async_encode, gather_budget=gather_budget)
        if pieced:                                    # raw words in T's and T2's own pieces and meta
            from kv_compare.pd import codecs as cdm

            class Pieced(cdm.Raw):
                def encode(self, words, pos0=0):
                    return cdm._encode_pieces(lambda w, p: cdm.Raw.encode(self, w, p), words, pos0, self.name)

                def decode(self, parts, meta, sink):
                    w = next(iter(parts.values()))
                    return cdm._decode_pieces(lambda a, m, k: cdm.Raw.decode(self, a, m, k), parts, meta, sink,
                                              int(w.shape[0]), int(w.shape[-1]), w.device)
            Pw.w.codecs[codec] = Pieced()
            Dw.w.codecs[codec] = Pieced()
        if lossy:                                     # a lossy stand-in for AQUA-KV: every word's lowest bit flipped
            from kv_compare.pd.codecs import Raw

            class Lossy(Raw):
                lossy = True

                def encode(self, words, pos0=0):
                    parts, meta = Raw.encode(self, words ^ 1, pos0)
                    return parts, dict(meta, rel_err=0.0123)
            Pw.w.codecs[codec] = Lossy()
            Dw.w.codecs[codec] = Lossy()
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
        end = time.time() + 20
        while async_encode and not r.finished_sending and time.time() < end:   # once the encoder has gathered it
            time.sleep(0.02)
            r = self.step(Ps, Pw)
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
        self.assertTrue(torch.equal(dst.gather(d_blocks, local_hit, 40), (src ^ 1 if lossy else src)[:, local_hit:]))
        if local_hit:                                 # what the engine had stays as it was
            self.assertTrue(torch.equal(dst.gather(d_blocks, 0, local_hit),
                                        Paged([before[f"model.layers.{l}.self_attn.attn"] for l in range(L)], 16, 2,
                                              8).gather(d_blocks, 0, local_hit)))
        recs = [json.loads(l) for f in Path(log_dir).glob("tkv_*.jsonl") for l in f.read_text().splitlines()]
        inst = sorted((x for x in recs if x.get("event") == "install"), key=lambda x: x.get("piece", 0))
        pieces = -(-40 // pipeline) if pipeline else 1
        self.assertEqual(len(inst), pieces)
        if pipeline:                                  # one transfer per piece; a piece the engine has is skipped
            sends = sorted((x for x in recs if x.get("event") == "send"), key=lambda x: x["piece"])
            self.assertEqual([(x["piece"], x["pieces"], x["t0"], x["tokens"]) for x in sends],
                             [(i, pieces, i * pipeline, min(pipeline, 40 - i * pipeline)) for i in range(pieces)])
            self.assertEqual([x["key"] for x in sends], ["K1"] + [f"K1#{i}" for i in range(1, pieces)])
            self.assertEqual([bool(x.get("skipped")) for x in inst], [(i + 1) * pipeline <= local_hit
                                                                      for i in range(pieces)])
            inst = [x for x in inst if not x.get("skipped")]
            self.assertTrue(all(x["exact"] for x in inst))
            return recs
        if lossy:                                     # no checksum to compare; the sender's own error recorded
            self.assertIsNone(inst[0]["exact"])
            self.assertNotIn("error", inst[0])
            send = next(x for x in recs if x.get("event") == "send")
            self.assertEqual((send["lossy"], send["rel_err"]), (True, 0.0123))
            return recs
        self.assertTrue(inst[0]["exact"])
        self.assertTrue(any(x.get("event") == "match" and x["local_hit"] == local_hit for x in recs))
        self.assertTrue(any(x.get("event") == "prefill_done" and len(x["blocks"]) == 2 for x in recs))
        return recs

    def test_whole_prompt_moves_exactly(self):
        self.flow()

    def test_the_installer_thread_moves_it_exactly_and_reports_it_once_done(self):
        """async_install: the install runs on a thread (and CUDA stream) of its own, so the engine's step goes on;
        the request is reported loaded once the install has finished."""
        recs = self.flow(async_install=True)
        inst = next(x for x in recs if x.get("event") == "install")
        self.assertTrue(inst["async"])
        self.assertGreaterEqual(inst["install_queued_ms"], 0)

    def test_a_request_that_ends_during_its_install_is_reported_after_it(self):
        """Its blocks are being written: vLLM may free them only once the install has finished."""
        import threading
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make(async_install=True)
        gate, started = threading.Event(), threading.Event()
        install = Dw.w._install

        def slow(*a, **k):
            started.set()
            gate.wait(20)
            return install(*a, **k)
        Dw.w._install = slow
        prompt = list(range(100, 140))
        dreq = SimpleNamespace(request_id="d-9", prompt_token_ids=prompt, num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K9"})
        n, _ = Ds.get_num_new_matched_tokens(dreq, 0)
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: ([1, 7, 3],)), n)
        self.step(Ds, Dw)
        preq = SimpleNamespace(request_id="p-9", prompt_token_ids=prompt, num_computed_tokens=40,
                               status=self.Status.FINISHED_LENGTH_CAPPED,
                               kv_transfer_params={"t_send": True, "t_key": "K9", "t_dest": dest, "t_codec": "raw"})
        Ps.request_finished(preq, [5, 2, 9])
        self.step(Ps, Pw, finished={"p-9"})
        end = time.time() + 20
        while not started.is_set() and time.time() < end:
            self.step(Ds, Dw)
            time.sleep(0.02)
        self.assertTrue(started.is_set())
        dreq.status = self.Status.FINISHED_ABORTED
        Ds.request_finished(dreq, [1, 7, 3])
        r = self.step(Ds, Dw, finished={"d-9"})
        self.assertEqual(r.finished_recving, set())             # not while its blocks are being written
        gate.set()
        got = set()
        while time.time() < end and not got:
            got = self.step(Ds, Dw).finished_recving
            time.sleep(0.02)
        self.assertEqual(got, {"d-9"})
        self.assertEqual(Dw.w.installing, set())

    def test_a_long_request_moves_in_pieces_each_installed_as_it_arrives(self):
        """pipeline_tokens: 40 tokens in pieces of 16, 16 and 8, each a transfer of its own; the engine's step
        reports the request once its last piece is installed; on the installer thread too, with the sender's
        encoder on a thread of its own, and with the first piece already in the decode engine's cache."""
        self.flow(pipeline=16)
        recs = self.flow(pipeline=16, async_install=True, async_encode=True)     # copied out in the step
        self.assertTrue(all(x["gathered_in_step"] for x in recs if x.get("event") == "send"))
        recs = self.flow(pipeline=16, async_encode=True, gather_budget=0)         # piece by piece on the encoder
        self.assertFalse(any(x.get("gathered_in_step") for x in recs if x.get("event") == "send"))
        self.flow(pipeline=16, local_hit=16, async_install=True)

    def test_a_codec_with_pieces_of_its_own_installs_a_request_sent_in_pieces(self):
        """Job 30563: every T and T2 install of a request over 4,096 tokens failed ("not enough values to unpack"):
        the connector's piece count, written into the codec's meta as "pieces", stood for T's own list of pieces.
        The transfer's count is now "t_pieces"; a codec in one piece of its own, and in several, installs exactly."""
        from kv_compare.pd import codecs as cdm
        self.flow(pipeline=16, pieced=True)
        self.flow(pipeline=16, pieced=True, async_install=True, local_hit=16)
        one = cdm._pieces
        cdm._pieces = lambda T: [(a, min(a + 8, T)) for a in range(0, T, 8)] if T > 8 else [(0, T)]
        try:
            recs = self.flow(pipeline=16, pieced=True)
        finally:
            cdm._pieces = one
        self.assertEqual(sorted(x["pieces"] for x in recs if x.get("event") == "install"), [3, 3, 3])

    def test_a_trace_hashes_what_p_sends_and_compares_what_d_held(self):
        """trace_pages: P logs every page's content hash of a traced request; D, which held the first two blocks in
        its prefix cache, logs per page whether its own bytes equal P's: equal in the first block, not at layer 1 of
        the second (one word changed); the new block is installed as usual and the held ones are left alone."""
        from kv_compare.pd import trace as tr
        cap = tempfile.mkdtemp()
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make(pipeline=16, trace=True, async_install=True,
                                                          p_more={"capture_dir": cap}, d_more={"capture_dir": cap})
        names = list(pc)
        for n in names:                               # D holds P's blocks 5 and 2 as its blocks 1 and 7
            dc[n][1].copy_(pc[n][5])
            dc[n][7].copy_(pc[n][2])
        dc[names[1]][7].view(-1)[3] += 1              # but layer 1 of the second differs by one word
        held_before = {n: (dc[n][1].clone(), dc[n][7].clone()) for n in names}
        prompt = list(range(100, 140))
        dreq = SimpleNamespace(request_id="d-7", prompt_token_ids=prompt, num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K7", "t_capture": True})
        n, _ = Ds.get_num_new_matched_tokens(dreq, 32)
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: ([1, 7, 3],)), n)
        self.step(Ds, Dw)
        preq = SimpleNamespace(request_id="p-7", prompt_token_ids=prompt, num_computed_tokens=40, num_cached_tokens=16,
                               status=self.Status.FINISHED_LENGTH_CAPPED,
                               kv_transfer_params={"t_send": True, "t_key": "K7", "t_dest": dest, "t_codec": "raw",
                                                   "t_trace": True, "t_capture": True})
        Ps.request_finished(preq, [5, 2, 9])
        self.step(Ps, Pw, finished={"p-7"})
        end = time.time() + 20
        while time.time() < end:
            r = self.step(Ds, Dw)
            if r.finished_recving:
                break
            time.sleep(0.02)
        self.assertEqual((r.finished_recving, r.failed_recving), ({"d-7"}, set()))
        from kv_compare.pd.paged import Paged
        P = Paged([pc[n] for n in names], 16, 2, 8)
        words = P.gather([5, 2, 9], 0, 40)
        D = Paged([dc[n] for n in names], 16, 2, 8)
        self.assertTrue(torch.equal(D.gather([1, 7, 3], 32, 40), words[:, 32:]))
        for n in names:                               # what D held is left as it was
            self.assertTrue(torch.equal(dc[n][1], held_before[n][0]) and torch.equal(dc[n][7], held_before[n][1]))
        recs = [json.loads(l) for f in Path(log_dir).glob("trace_*.jsonl") for l in f.read_text().splitlines()]
        pages = sorted((x for x in recs if x["event"] == "pages"), key=lambda x: x["piece"])
        self.assertEqual([(x["key"], x["t0"], x["tokens"]) for x in pages], [("K7", 0, 16), ("K7", 16, 16), ("K7", 32, 8)])
        for x in pages:
            got = tr.unpack_u64(x["h"], (x["layers"], 1))
            want = tr.page_hashes(words[:, x["t0"]:x["t0"] + x["tokens"]], 16)
            self.assertTrue((got == want).all())
        held = sorted((x for x in recs if x["event"] == "held"), key=lambda x: x["t0"])
        self.assertEqual([(x["t0"], x["tokens"], x["equal_pages"], x["pages"]) for x in held], [(0, 16, 3, 3), (16, 16, 2, 3)])
        self.assertEqual(tr.unpack_bits(held[1]["equal"], (3, 1))[:, 0].tolist(), [True, False, True])
        self.assertNotIn("delta", held[0])           # every page the same: no histograms
        dl = held[1]["delta"]                         # one word of layer 1's keys or values differs
        self.assertEqual([dl["values"][kv][1] for kv in range(2)], [16 * 16] * 2)
        self.assertEqual(dl["equal"][0][1] + dl["equal"][1][1], 2 * 16 * 16 - 1)
        self.assertEqual([dl["equal"][kv][l] for kv in range(2) for l in (0, 2)], [16 * 16] * 4)
        # capture: P stored its three blocks (the last one partial), D its own copy of the block that differs
        Pw.w.cstore.close()
        Dw.w.cstore.close()
        import numpy as np
        caps = {r: [json.loads(l) for f in Path(cap).glob(f"capture_{r}_*.jsonl") for l in f.read_text().splitlines()]
                for r in ("P", "D")}
        meta = json.loads(next(Path(cap).glob("meta_P_*.json")).read_text())
        self.assertEqual((meta["layers"], meta["block_size"], meta["words"]), (3, 16, 32))
        prec = caps["P"][0]
        self.assertEqual((prec["key"], prec["T"], prec["rows"], prec["missing"]), ("K7", 40, [0, 1, 2], 0))
        rows = np.fromfile(next(Path(cap).glob("blocks_P_*.bin")), dtype=np.int16).reshape(-1, 3, 16, 32)
        self.assertTrue(np.array_equal(rows[0], words[:, 0:16].numpy()) and np.array_equal(rows[1], words[:, 16:32].numpy()))
        self.assertTrue(np.array_equal(rows[2][:, :8], words[:, 32:40].numpy()) and not rows[2][:, 8:].any())
        drec = caps["D"][0]
        self.assertEqual((drec["event"], drec["key"], drec["t0"], drec["blocks"], drec["rows"]),
                         ("held_differs", "K7", 16, [0], [0]))
        drows = np.fromfile(next(Path(cap).glob("blocks_D_*.bin")), dtype=np.int16).reshape(-1, 3, 16, 32)
        self.assertTrue(np.array_equal(drows[0], D.gather([1, 7, 3], 16, 32).numpy()))
        rel = [x for x in recs if x["event"] == "relations"]
        self.assertEqual([(x["key"], x["spans"], x["novel"], x["tokens"], x["sample"]) for x in rel],
                         [("K7", [], 25, 40, 8)])
        self.assertLessEqual(rel[0]["stored"], 25)    # the sampled ones (1 in 8, by content)
        log = [json.loads(l) for f in Path(log_dir).glob("tkv_Ps*.jsonl") for l in f.read_text().splitlines()]
        self.assertEqual([x.get("cached") for x in log if x.get("event") == "prefill_done"], [16])

    def test_the_relation_probe_measures_moved_text_against_its_first_occurrence(self):
        """Two traced requests: the second holds the first's document after another 32-token prefix (its blocks are
        the first's, so the words are the same). P relates the document's 32 tokens to their first occurrence (16
        tokens earlier there), measures them against it (every XOR 0; the turned keys are not the same words), and
        writes per group (layout from the conversation id / kind of piece) the totals."""
        mdir = tempfile.mkdtemp()
        Path(mdir, "config.json").write_text(json.dumps({"rope_theta": 10000.0}))
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make(pipeline=16, trace=True,
                                                          p_more={"model_dir": mdir, "trace_sample": 1})
        X, A, Y = list(range(1000, 1016)), list(range(2000, 2032)), list(range(3000, 3032))
        reqs = [("p-1", "K1", X + A, [5, 2, 9], [["system", 0, 16], ["doc", 16, 48]]),
                ("p-2", "K2", Y + A, [3, 4, 2, 9], [["system", 0, 32], ["doc", 32, 64]])]
        for rid, key, prompt, blocks, pieces in reqs:
            req = SimpleNamespace(request_id=rid, prompt_token_ids=prompt, num_computed_tokens=len(prompt),
                                  num_cached_tokens=0, status=self.Status.FINISHED_LENGTH_CAPPED,
                                  kv_transfer_params={"t_send": True, "t_key": key, "t_dest": dest, "t_codec": "raw",
                                                      "t_trace": True, "t_conv": "docs_first:3", "t_pieces": pieces})
            Ps.request_finished(req, blocks)
            self.step(Ps, Pw, finished={rid})
        Pw.w.probe.close()                            # its queue done, the totals written
        recs = [json.loads(l) for f in Path(log_dir).glob("trace_P*.jsonl") for l in f.read_text().splitlines()]
        self.assertEqual([x for x in recs if x["event"] == "relations_error"], [])
        rel = {x["key"]: x for x in recs if x["event"] == "relations"}
        self.assertEqual((rel["K1"]["spans"], rel["K1"]["novel"], rel["K1"]["stored"]), ([], 33, 33))  # 15..47
        self.assertEqual(rel["K2"]["spans"], [[0, "doc", 32, 16, 32]])
        self.assertEqual((rel["K2"]["by_kind"], rel["K2"]["novel"]), ({"doc": [32, 32]}, 32))
        last = {}
        for x in recs:
            if x["event"] == "delta_stats":
                last[x["group"]] = x
        self.assertEqual(sorted(last), ["docs_first/all", "docs_first/doc"])
        self.assertEqual((last["docs_first/all"]["requests"], last["docs_first/all"]["tokens"]), (2, 112))
        d = last["docs_first/doc"]
        self.assertEqual((d["related"], d["measured"], d["base_gone"]), (32, 32, 0))
        C = 2 * 8
        for l in range(3):
            for kv in range(2):
                self.assertEqual(d["hist"]["xor"][kv][l][0], 32 * C)
                self.assertEqual(d["hist"]["diff"][kv][l][0], 32 * C)
                self.assertEqual(d["equal"][kv][l], 32 * C)
                self.assertEqual(sum(d["sign_exp"][kv][l]), 32 * C)
            self.assertEqual(sum(d["hist"]["rot"][0][l]), 32 * C)    # keys turned by 16 positions
            self.assertLess(d["hist"]["rot"][0][l][0], 32 * C)
            self.assertEqual(sum(d["hist"]["rot"][1][l]), 0)

    def test_the_page_store_keeps_each_block_once_within_its_budget(self):
        import numpy as np
        from kv_compare.pd import trace as tr
        d = tempfile.mkdtemp()
        st = tr.PageStore(d, "P", 3 * 2 * 16 * 4 * 2 / 2 ** 30, 2, 16, 4)     # room for three rows
        g = torch.Generator().manual_seed(1)
        w = torch.randint(-3000, 3000, (2, 40, 4), generator=g, dtype=torch.int32).to(torch.int16)
        r1 = tr.capture_blocks(st, w, tr.page_hashes(w, 16), 16)
        r2 = tr.capture_blocks(st, w[:, :32], tr.page_hashes(w[:, :32], 16), 16)   # its first two blocks again
        v = w.clone()
        v[0, 0, 0] += 1                                # a new first block: no room left
        r3 = tr.capture_blocks(st, v[:, :16], tr.page_hashes(v[:, :16], 16), 16)
        st.close()
        self.assertEqual((r1, r2, r3, st.full), ([0, 1, 2], [0, 1], [-1], True))
        rows = np.fromfile(st.path, dtype=np.int16).reshape(-1, 2, 16, 4)
        self.assertEqual(rows.shape[0], 3)
        self.assertTrue(np.array_equal(rows[1], w[:, 16:32].numpy()))
        self.assertTrue(np.array_equal(rows[2][:, :8], w[:, 32:].numpy()) and not rows[2][:, 8:].any())

    def test_relation_measures_are_exact(self):
        from kv_compare.pd import trace as tr
        v = torch.tensor([0, 1, 2, 3, 4, 7, 8, 65535, 65536, (1 << 23) - 1])
        self.assertEqual(tr.bit_length(v).tolist(), [0, 1, 2, 2, 3, 3, 4, 16, 17, 23])
        b = torch.tensor([1.0, -1.0, 0.0, -0.0, 2.0], dtype=torch.bfloat16).view(torch.int16)
        o = tr.ordered(b).tolist()
        self.assertEqual(o[2] - o[3], 1)              # +0 one above -0
        self.assertTrue(o[1] < o[3] < o[2] < o[0] < o[4])
        up = (b.to(torch.int32) + 1).to(torch.int16)  # one ULP away from zero for positive values
        self.assertEqual((tr.ordered(up[[0, 4]]) - tr.ordered(b[[0, 4]])).tolist(), [1, 1])
        st = tr.DeltaStats(2, torch.device("cpu"))
        a = torch.tensor([[0x3F80, 0x3F81, -0x4080]], dtype=torch.int32).to(torch.int16)   # 1.0, next, -1.0
        st.add(1, 0, a, a)
        st.add(1, 1, a, torch.tensor([[0x3F80, 0x3F80, -0x4080 + 3]], dtype=torch.int32).to(torch.int16))
        r = st.record("g")
        self.assertEqual(r["hist"]["xor"][0][1][0], 3)
        self.assertEqual(r["hist"]["xor"][1][1][:3], [1, 1, 1])     # 0; 1; 0b11 (2 bits) -> lengths 0, 1, 2
        self.assertEqual(r["hist"]["diff"][1][1][:3], [1, 1, 1])    # 0; 1 ULP; 3 ULPs
        self.assertEqual((r["equal"][1][1], r["values"][1][1], sum(r["sign_exp"][1][1])), (1, 3, 3))
        self.assertEqual(sum(r["hist"]["xor"][0][0]), 0)            # layer 0 untouched

    def test_turning_a_key_by_the_shift_predicts_it_at_its_new_place(self):
        """A key is RoPE(position) of a vector: the one at position 7, turned by 33, is (to BF16 rounding) the one
        at 40; unturned, it is far off. Llama-3's scaling keeps the fast frequencies and slows the slow ones."""
        from kv_compare.pd import trace as tr
        heads, hd = 2, 64
        inv = tr.rope_inv_freq({"rope_theta": 500000.0}, hd)
        g = torch.Generator().manual_seed(5)
        q = torch.randn(10, heads, hd, generator=g)

        def rope(x, p):
            ang = (float(p) * inv).to(torch.float32)
            c, s = torch.cos(ang), torch.sin(ang)
            x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1).to(torch.bfloat16).view(torch.int16).reshape(10, -1)
        base, want = rope(q, 7), rope(q, 40)
        turned = tr.rotate(base, 33, inv, heads, hd)
        near = (tr.ordered(turned) - tr.ordered(want)).abs()
        far = (tr.ordered(base) - tr.ordered(want)).abs()
        self.assertLessEqual(tr.bit_length(near).float().mean().item(), 1.5)
        self.assertGreater(tr.bit_length(far).float().mean().item(), tr.bit_length(near).float().mean().item() + 4)
        l3 = tr.rope_inv_freq({"rope_theta": 500000.0, "rope_scaling": {"rope_type": "llama3", "factor": 32.0,
                               "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                               "original_max_position_embeddings": 8192}}, hd)
        self.assertAlmostEqual(l3[0].item(), inv[0].item())
        self.assertAlmostEqual(l3[-1].item(), inv[-1].item() / 32)
        self.assertIsNone(tr.rope_inv_freq({"rope_scaling": {"rope_type": "yarn"}, "rope_theta": 1e4}, hd))
        self.assertIsNone(tr.rope_inv_freq({}, hd))

    def test_relation_index_finds_moved_text_and_leaves_out_shared_prefixes(self):
        from kv_compare.pd import trace as tr
        sys_, docA, q1 = list(range(1, 21)), list(range(100, 140)), list(range(200, 210))
        ans, q2, docB, q3 = list(range(300, 310)), list(range(400, 410)), list(range(500, 530)), list(range(600, 610))
        ix = tr.RelationIndex()
        r0, sp0, nv0, h0 = ix.match("a", sys_ + docA + q1)
        self.assertEqual((r0, sp0, nv0.tolist()), (0, [], list(range(15, 70))))
        r1, sp1, nv1, _h = ix.match("b", sys_ + docA + q1 + ans + q2)  # the same prefix, then new: no relation
        self.assertEqual((r1, sp1, nv1.tolist()), (1, [], list(range(70, 90))))
        r2, sp2, nv2, h2 = ix.match("c", sys_ + docB + docA + q3)      # the document 30 tokens later
        self.assertEqual(sp2, [(0, 50, 20, 40)])
        s0, s2 = set(tr.sampled(h0, 4).tolist()), set(tr.sampled(h2, 4).tolist())
        inside = range(65, 90)                        # windows wholly in the document: the same choice there
        self.assertEqual({i for i in inside if i in s2}, {i for i in inside if i - 30 in s0})
        self.assertTrue(0 < len(s2) < len(h2))
        self.assertEqual(tr.sampled(h2, 1).tolist(), list(range(15, 100)))
        self.assertEqual(nv2.tolist(), list(range(20, 65)) + list(range(90, 100)))
        pieces = [("system", 0, 20), ("doc", 20, 50), ("doc", 50, 90), ("user", 90, 100)]
        self.assertEqual(tr.split_spans(sp2, pieces), [("doc", 0, 50, 20, 40)])
        self.assertEqual(tr.split_spans([(0, 45, 15, 50)], pieces), [("doc", 0, 45, 15, 5), ("doc", 0, 50, 20, 40),
                                                                    ("question", 0, 90, 60, 5)])
        self.assertEqual(tr.split_spans(sp2, None), [("all", 0, 50, 20, 40)])

    def test_page_hashes_are_exact_integers_equal_only_for_equal_pages(self):
        from kv_compare.pd import trace as tr
        g = torch.Generator().manual_seed(3)
        w = torch.randint(-32768, 32768, (3, 40, 64), generator=g, dtype=torch.int32).to(torch.int16)
        h = tr.page_hashes(w, 16)
        self.assertEqual(h.shape, (3, 3))             # 16 + 16 + 8 tokens
        self.assertTrue((h == tr.page_hashes(w.clone(), 16)).all())
        v = w.clone()
        v[2, 39, 5] += 1                              # one word of the partial block of layer 2
        d = tr.page_hashes(v, 16) != h
        self.assertEqual(d.tolist(), [[False] * 3, [False] * 3, [False, False, True]])
        ext = torch.full((1, 16, 2048), -32768, dtype=torch.int16)   # the largest sums: still exact (no overflow)
        e1, e2 = tr.page_hashes(ext, 16), tr.page_hashes(ext.clone(), 16)
        self.assertEqual(e1.tolist(), e2.tolist())
        x = tr.unpack_bits(tr.pack_bits(d), d.shape)
        self.assertEqual(x.tolist(), d.tolist())
        self.assertTrue((tr.unpack_u64(tr.pack_u64(h), h.shape) == h).all())

    def test_a_failed_piece_fails_the_request_once_the_others_are_done(self):
        """The second piece's encode fails: the sender tells the receiver and sends no more; the decode engine
        reports the request failed (vLLM computes it), not before the first piece's install has ended."""
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make(pipeline=16, async_install=True)
        raw = Pw.w.codecs["raw"]
        calls = []
        enc = raw.encode

        def flaky(words, pos0=0):
            calls.append(pos0)
            if pos0 == 16:
                raise RuntimeError("piece broke")
            return enc(words, pos0)
        raw.encode = flaky
        prompt = list(range(100, 140))
        dreq = SimpleNamespace(request_id="d-5", prompt_token_ids=prompt, num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K5"})
        n, _ = Ds.get_num_new_matched_tokens(dreq, 0)
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: ([1, 7, 3],)), n)
        self.step(Ds, Dw)
        preq = SimpleNamespace(request_id="p-5", prompt_token_ids=prompt, num_computed_tokens=40,
                               status=self.Status.FINISHED_LENGTH_CAPPED,
                               kv_transfer_params={"t_send": True, "t_key": "K5", "t_dest": dest, "t_codec": "raw"})
        Ps.request_finished(preq, [5, 2, 9])
        r = self.step(Ps, Pw, finished={"p-5"})
        self.assertEqual(r.finished_sending, {"p-5"})
        self.assertEqual(calls, [0, 16])              # no third piece
        end = time.time() + 10
        while time.time() < end:
            r = self.step(Ds, Dw)
            if r.failed_recving:
                break
            time.sleep(0.02)
        self.assertEqual((r.failed_recving, r.finished_recving), ({"d-5"}, {"d-5"}))
        self.assertEqual((Dw.w.installing, Dw.w.req_state), (set(), {}))
        recs = [json.loads(l) for f in Path(log_dir).glob("tkv_*.jsonl") for l in f.read_text().splitlines()]
        self.assertEqual([x.get("piece") for x in recs if x.get("event") == "install"], [0])
        self.assertTrue(any(x.get("event") == "send_failed" and x["key"] == "K5#1" for x in recs))

    def test_warm_up_keeps_a_lossy_codec_that_rebuilds_its_own_reconstruction(self):
        """Job 30293 left AQUA-KV out at the warm-up ("round trip not exact"): a lossy codec passes when the receiver
        rebuilds the sender's reconstruction (the error it reports), and fails when it does not."""
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make()
        from kv_compare.pd.codecs import Raw

        class Lossy(Raw):
            lossy = True

            def __init__(self, report):
                self.report = report

            def encode(self, words, pos0=0):
                bad = words ^ 1
                a, b = bad.view(torch.bfloat16).double(), words.view(torch.bfloat16).double()
                err = float((a - b).norm() / b.norm())
                parts, meta = Raw.encode(self, bad, pos0)
                return parts, dict(meta, rel_err=err * self.report)
        w = Pw.w
        w.codecs = {"raw": w.codecs["raw"], "aquakv-2": Lossy(1.0), "aquakv-3": Lossy(1.5)}
        out = w.warm_up()
        self.assertEqual((out["raw"], out["aquakv-2"]), (True, True))
        self.assertIn("lossy round trip", out["aquakv-3"])
        self.assertEqual(sorted(w.codecs), ["aquakv-2", "raw"])

    def test_a_lossy_codec_installs_without_a_checksum(self):
        """AQUA-KV is lossy: with the check on, its sender sends no checksum and the receiver installs what came."""
        self.flow(lossy=True)

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
        # the request only: vLLM 0.32 asserted when the same step also named its blocks (job 30176)
        self.assertEqual(Dw.get_block_ids_with_load_errors(), set())

    def test_a_failed_encode_fails_the_decode_at_once(self):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make()
        Pw.w.codecs["raw"].encode = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("encode broke"))
        prompt = list(range(100, 140))
        dreq = SimpleNamespace(request_id="d-9", prompt_token_ids=prompt, num_computed_tokens=0,
                               kv_transfer_params={"t_recv": True, "t_key": "K9"})
        n, _ = Ds.get_num_new_matched_tokens(dreq, 0)
        Ds.update_state_after_alloc(dreq, SimpleNamespace(get_block_ids=lambda: ([1, 7, 3],)), n)
        self.step(Ds, Dw)
        preq = SimpleNamespace(request_id="p-9", prompt_token_ids=prompt, num_computed_tokens=40,
                               status=self.Status.FINISHED_LENGTH_CAPPED,
                               kv_transfer_params={"t_send": True, "t_key": "K9", "t_dest": dest, "t_codec": "raw"})
        Ps.request_finished(preq, [5, 2, 9])
        r = self.step(Ps, Pw, finished={"p-9"})
        self.assertEqual(r.finished_sending, {"p-9"}, "the prefill side lets its blocks go")
        end = time.time() + 10
        while time.time() < end:                      # the decode's timeout is 600 s: this is the abort
            r = self.step(Ds, Dw)
            if r.failed_recving:
                break
            time.sleep(0.02)
        self.assertEqual(r.failed_recving, {"d-9"})
        self.assertEqual(Dw.get_block_ids_with_load_errors(), set())
        recs = [json.loads(l) for f in Path(log_dir).glob("tkv_*.jsonl") for l in f.read_text().splitlines()]
        self.assertTrue(any(x.get("event") == "send_failed" and "encode broke" in x.get("error", "") for x in recs))

    def test_requests_without_our_params_are_left_alone(self):
        Ps, Pw, Ds, Dw, pc, dc, dest, log_dir = self.make()
        req = SimpleNamespace(request_id="x", prompt_token_ids=[1, 2, 3], num_computed_tokens=3,
                              kv_transfer_params=None, status=self.Status.FINISHED_STOPPED)
        self.assertEqual(Ds.get_num_new_matched_tokens(req, 0), (0, False))
        self.assertEqual(Ps.request_finished(req, [1]), (False, None))
        self.assertFalse(Ps.has_pending_push_work())


DIFFGRAPH_CHECK = r"""
import os, sys, json, tempfile
os.environ["TRITON_INTERPRET"] = "1"
os.environ["DELTA_WORKERS"] = "1"
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.path.join(sys.argv[1], "scripts"))
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from kv_compare.capture import capture_ids, model_meta
from kv_compare.predict import Study
from kv_compare import fused, layerwise
from kv_compare.grouped import prepare_groups
from kv_compare.pd import codecs, export, trace as tr
import pd_diffgraph as dg
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
bpath = Path(tempfile.mkdtemp()) / "b.pt"
torch.save(b, bpath)
codecs.PAD = 64


def kv_words(ids):
    with torch.no_grad():
        pkv = model(torch.tensor([ids]), use_cache=True).past_key_values
    layers = []
    for l in range(3):
        k, v = (pkv[l] if isinstance(pkv, (list, tuple)) else (pkv.layers[l].keys, pkv.layers[l].values))
        k = k[0].transpose(0, 1).reshape(len(ids), -1)      # [T, heads*hd]
        v = v[0].transpose(0, 1).reshape(len(ids), -1)
        layers.append(torch.cat([k, v], 1).contiguous().view(torch.int16))
    return torch.stack(layers)                              # [L, T, W]


d = Path(tempfile.mkdtemp())
cap, run, mdir, out = d / "cap", d / "run", d / "model", d / "out"
for x in (cap, run / "runs", mdir, out):
    x.mkdir(parents=True)
(mdir / "config.json").write_text(json.dumps({"rope_theta": 500000.0, "head_dim": 32}))
A = [int(v) for v in np.random.default_rng(10).integers(0, 97, size=48)]
B = [int(v) for v in np.random.default_rng(11).integers(0, 97, size=16)]
prompts = [A, B + A]                                         # turn 1: turn 0's text 16 tokens later
st = tr.PageStore(str(cap), "P", 1.0, 3, 16, 128)
recs, prox = [], []
nm = "mtrag_raw_docs_first_conversation"
for t, ids in enumerate(prompts):
    w = kv_words(ids)
    rows = tr.capture_blocks(st, w, tr.page_hashes(w, 16), 16)
    st.record({"event": "request", "key": f"K{t}", "conv": "r:0", "t0": 0, "T": len(ids), "bs": 16, "rows": rows,
               "missing": 0})
    recs.append({"tag": f"{nm}|0|{t}", "run": nm, "layout": "docs_first", "policy": "conversation", "conv": 0,
                 "turn": t, "prompt_tokens": len(ids), "ids": ids, "pieces": [["doc", "D", 0, len(ids)]]})
    prox.append({"key": f"K{t}", "tag": recs[-1]["tag"], "t_in": t})
st.close()
(run / "runs" / f"{nm}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
(run / "proxy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prox))
res = dg.analyse(str(cap), str(run), str(bpath), str(mdir), str(out), names=("ours-kv",), device="cpu",
                 log=lambda m: None)
r = res["runs"][nm]
assert r["pages"] == 3 + 4, r["pages"]
for unit in ("per_request", "per_segment", "per_page"):
    x = r[f"ours-kv_{unit}"]
    cnt = x.get("requests") or x.get("segments") or x.get("pages")
    assert x["exact"] == cnt and x["bytes"] > 0, (unit, x)
assert r["delta"]["coded"] >= 3 and r["delta"]["exact"] == r["delta"]["coded"], r["delta"]
assert r["graph_ours-kv"]["independent"] == r["ours-kv_per_page"]["bytes"]
assert "mtrag_raw_docs_first_conversation" in res["theirs"]["runs"]
text = dg.markdown(res)
assert "KV Diff Graph against EvidenceStore" in text and "rebuild cold" in text, text[:2000]
print("ok diffgraph", json.dumps({k: r[k] for k in ("ours-kv_per_request", "ours-kv_per_page", "delta")}))
"""


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
        if n == "ours-4g" and T == 96:               # the encoder's own decoder chunks (2 here) decode it too
            codecs.DECODE_CHUNKS = 0
            got = {}
            c.decode(p2, m2, lambda l, x: got.__setitem__(l, x.clone()))
            codecs.DECODE_CHUNKS = 1
            assert torch.equal(torch.stack([got[l] for l in range(L)]), w), n
# the GPU packing kernel (here interpreted) packs exactly what the torch loop packs
c = built["ours-4g"]
codec, kp, Tp = c.codec(96, 0)
wp = torch.cat([words, torch.zeros(L, Tp - 96, W, dtype=words.dtype)], 1).contiguous()
xall, prior = c.model(codec, kp, wp)
buf, starts = codec.encode(wp, xall, prior)
_, cap = codec.layout(Tp)
a = wire.compact_segments(buf, starts, cap, c.hd, kernel=False)
b = wire.compact_segments(buf, starts, cap, c.hd, kernel=True)
assert torch.equal(a, b), "tail copy"
sc, n = wire.compact_starts(starts, cap, c.hd)
assert n == a.numel() and int(sc[0]) == int(starts[0]) - int(wire.segment_bases(starts, cap, c.hd)[0])
print("ok tail copy", a.numel())
# a request longer than a piece: coded in pieces, decoded bit-exact
codecs.PIECE_TOKENS = 64
for n, c in built.items():
    w = words[:, :96].contiguous()
    parts, m = c.encode(w, pos0=0)
    assert [x["t0"] for x in m["pieces"]] == [0, 64] and m["T"] == 96, m
    h, blob = wire.bundle(parts, m)
    p2, m2 = wire.unbundle(wire.header_from_bytes(wire.header_to_bytes(h)), blob.clone())
    got = {}
    c.decode(p2, m2, lambda l, x: got.__setitem__(l, x.clone()))
    assert sorted(got) == list(range(L)) and torch.equal(torch.stack([got[l] for l in range(L)]), w), n
    print("ok pieces", n, round(8 * wire.payload_bytes(h) / w.numel(), 3))
codecs.PIECE_TOKENS = 4096
# the sender's sums made once: the same streams as the forward pass, then the coding pass (and T2's layer 1..L-1)
from kv_compare import t2 as t2m
for n in ("ours-kv", "ours-4g", "ours-rank256"):
    c = built[n]
    for cls in (None, t2m.Layer0Recomputed):
        codec, kp, Tp = c.codec(96, 5, cls=cls)
        wp = torch.cat([words[:, 5:], torch.zeros(L, Tp - 91, W, dtype=words.dtype)], 1).contiguous()
        xall, prior = c.model(codec, kp, wp)
        S, cap = codec.layout(Tp)
        # zeroed buffers: the words between a segment's states and its data (alignment) are never written
        b1, s1 = codec.encode(wp, xall, prior, out=torch.zeros(S * cap, dtype=torch.int16))
        b2, s2 = codec.encode_once(wp, out=torch.zeros(S * cap, dtype=torch.int16))
        assert torch.equal(s1, s2) and torch.equal(b1, b2), (n, cls)
        print("ok encode once", n, cls is not None)
# a head group's sums in one Triton launch: the same integers as the column copy, the product and the add
c = built["ours-4g"]
codec, kp, Tp = c.codec(96, 0, chunks=1)
gen = torch.Generator().manual_seed(1)
x = torch.randint(-127, 128, (Tp, W), dtype=torch.int8, generator=gen)
for j in range(1, codec.ng):
    a1 = torch.randint(-1000, 1000, (Tp, W), dtype=torch.int32, generator=gen)
    a2 = a1.clone()
    layerwise.GROUP_PRODUCT = "torch"
    codec._group_acc(x, j, 1, a1)
    layerwise.GROUP_PRODUCT = "fused"
    codec._group_acc(x, j, 1, a2)
    assert torch.equal(a1, a2), j
print("ok group product")


class CpuArena(codecs._Arena):                    # the graph arena with its CUDA calls replaced: a "graph" reruns
    def _current(self):                           # its step
        return type("S", (), {"wait_event": lambda self, e: None})()

    def _mark(self, cur):
        return None

    def _capture(self, steps, cur):
        return [type("G", (), {"replay": staticmethod(f)})() for f in steps]


# the decoder as per-layer steps (what the graphs replay): the same words; through the arena, eagerly at first use,
# then recorded and checked, then replayed
for n in ("ours-kv", "ours-4g", "ours-rank256"):
    c = built[n]
    for cls in (None, t2m.Layer0Recomputed):
        codec, kp, Tp = c.codec(96, 5, cls=cls, chunks=1)
        wp = torch.cat([words[:, 5:], torch.zeros(L, Tp - 91, W, dtype=words.dtype)], 1).contiguous()
        buf, starts = codec.encode_once(wp)
        _, cap = codec.layout(Tp)
        packed = wire.compact_segments(buf, starts, cap, c.hd)
        cs, _ = wire.compact_starts(starts, cap, c.hd)
        a, b = {}, {}
        l0 = wp[0].clone() if cls is not None else None
        if cls is None:
            codec.decode(packed, cs, Tp, sink=lambda l, x: a.__setitem__(l, x.clone()))
            codec.decode_steps(packed, cs, Tp, lambda l, x: b.__setitem__(l, x.clone()))
        else:
            codec.decode(packed, cs, Tp, l0, sink=lambda l, x: a.__setitem__(l, x.clone()))
            codec.decode_steps(packed, cs, Tp, lambda l, x: b.__setitem__(l, x.clone()), l0=l0)
        assert torch.equal(torch.stack([a[l] for l in range(L)]), wp), (n, cls)
        assert torch.equal(torch.stack([b[l] for l in range(L)]), wp), (n, cls)
        if codec.ng > 1:
            codecs.GRAPH_STREAM_SHARE = 1.5           # the tiny model's streams are larger than its raw words
            arena = CpuArena(c, 128)
            make = lambda cv, sn: c.codec(Tp, 0, cls=cls, chunks=1, rope=(cv, sn))[0]
            for rep in range(3):
                got = {}
                ok = arena.decode((n, id(c), Tp, cls is not None), make, packed, cs, Tp, 5, c.cos, c.sin,
                                  lambda l, x: got.__setitem__(l, x.clone()), l0=l0)
                assert ok and torch.equal(torch.stack([got[l] for l in range(L)]), wp), (n, cls, rep)
            assert len(arena.events) == 1 and arena.events[0]["ok"] and arena.events[0]["graphs"] == L - (cls is not None), arena.events
            big = torch.zeros(arena.cap + 1, dtype=torch.int16)      # a stream over the static buffer: eagerly
            assert not arena.decode((n, id(c), Tp, cls is not None), make, big, cs, Tp, 5, c.cos, c.sin, None, l0=l0)
        print("ok steps", n, cls is not None)
print("ok ours wire")
"""


class OursWireTests(unittest.TestCase):
    def test_the_diff_graph_evaluator_codes_and_checks_every_page_interpreted(self):
        """scripts/pd_diffgraph.py end to end on a tiny model's real KV (two turns, the first's text moved by 16
        tokens): T per request, per segment and per page, deltas by the slice coder from the base, every page
        decoded and equal, their zlib graph beside it."""
        import importlib.util
        if importlib.util.find_spec("triton") is None or importlib.util.find_spec("transformers") is None:
            self.skipTest("Triton and transformers are required")
        r = subprocess.run([sys.executable, "-c", DIFFGRAPH_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=7200)
        self.assertEqual(r.returncode, 0, r.stderr[-4000:])
        self.assertIn("ok diffgraph", r.stdout)

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
        # the MTRAG driver's tag (logged with the key) and trace flag (to the prefill engine only), not passed on
        self.req(url, {"model": "m", "prompt": "z", "t_tag": "docs_first|conversation|3|1", "t_trace": True})
        p, d = eng.seen["p"][-1], (eng.seen["d1"] + eng.seen["d2"])[-1]
        self.assertTrue(p["kv_transfer_params"]["t_trace"])
        self.assertNotIn("t_trace", d["kv_transfer_params"])
        self.assertFalse(any("t_tag" in b or "t_trace" in b for b in (p, d)))
        tagged = [json.loads(l) for l in log.read_text().splitlines() if json.loads(l).get("tag")]
        self.assertEqual([(x["tag"], x["key"]) for x in tagged], [("docs_first|conversation|3|1", p["kv_transfer_params"]["t_key"])])
        # capture (with the trace): both engines are told; the pieces go to the prefill engine; nothing else passed on
        self.req(url, {"model": "m", "prompt": "y", "t_trace": True, "t_capture": True, "t_pieces": [["doc", 0, 4]]})
        self.req(url, {"model": "m", "prompt": "x", "t_capture": True})        # without the trace: ignored
        byp = {b["prompt"]: b for b in eng.seen["p"]}
        byd = {b["prompt"]: b for b in eng.seen["d1"] + eng.seen["d2"]}
        self.assertEqual((byp["y"]["kv_transfer_params"]["t_capture"], byp["y"]["kv_transfer_params"]["t_pieces"]),
                         (True, [["doc", 0, 4]]))
        self.assertTrue(byd["y"]["kv_transfer_params"]["t_capture"])
        self.assertNotIn("t_capture", byp["x"]["kv_transfer_params"])
        self.assertNotIn("t_capture", byd["x"]["kv_transfer_params"])
        self.assertFalse(any(k in b for b in (byp["x"], byd["x"], byp["y"], byd["y"]) for k in ("t_capture", "t_pieces")))

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

    def test_a_request_sent_in_pieces_counts_once(self):
        """Its pieces' records make one send and one install: sizes and times summed, the link from the first
        piece's start to the last one's end, done with the last piece."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        base = {"req_key": "K", "pieces": 2}
        recs = [dict(base, event="send", key="K", piece=0, tokens=4096, encode_ms=60.0, gather_ms=7.0, raw_bytes=800,
                     coded_bytes=400, blob_bytes=416, link_s=0.2, queued=1.0, send_begin=1.1, send_end=1.3,
                     queued_on="gpu"),
                dict(base, event="send", key="K#1", piece=1, tokens=1000, encode_ms=20.0, gather_ms=2.0, raw_bytes=200,
                     coded_bytes=150, blob_bytes=160, link_s=0.05, queued=1.2, send_begin=1.3, send_end=1.35,
                     queued_on="host"),
                dict(base, event="install", key="K#1", piece=1, tokens=1000, decode_install_ms=10.0, admitted_at=0.5,
                     done_at=1.5, exact=True),
                dict(base, event="install", key="K", piece=0, tokens=4096, decode_install_ms=40.0, admitted_at=0.5,
                     done_at=1.4, exact=True),
                {"event": "send", "key": "J", "bits": 16.0}]
        m = pb.merge_pieces(recs)
        s = next(r for r in m if r["event"] == "send" and r["key"] == "K")
        i = next(r for r in m if r["event"] == "install")
        self.assertEqual((s["tokens"], s["encode_ms"], s["raw_bytes"], s["blob_bytes"], s["send_begin"], s["send_end"],
                          s["queued_on"]), (5096, 80.0, 1000, 576, 1.1, 1.35, "host"))
        self.assertAlmostEqual(s["bits"], 16.0 * 550 / 1000)
        self.assertEqual((i["tokens"], i["decode_install_ms"], i["done_at"], i["exact"]), (5096, 50.0, 1.5, True))
        st = pb.transfer_stats(recs)
        self.assertEqual((st["sends"], st["installs"]), (2, 1))


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
