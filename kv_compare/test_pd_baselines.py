"""Tests of the PD baselines branch (docs/pd_baselines.md) on the CPU: the CPU tier (budget, prefix hits, the whole
store-and-load flow through the connector with its checksums), the prefill steps the scheduler records, T2's
layer-0 recompute and its sender check, T2 through the wire (Triton interpreter), the proxy's NIXL flow and several
prefill engines, the bench's codec filter and tier statistics, the matrix report and the node-specs summary.

  python -m unittest kv_compare.test_pd_baselines
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from kv_compare.test_pd import ROOT, _caches, _vcfg, install_vllm_stub


class TierTests(unittest.TestCase):
    def test_budget_prefix_hits_and_clear(self):
        from kv_compare.pd.tier import HostTier, chunk_hashes
        a = chunk_hashes(list(range(100)), 16)
        b = chunk_hashes(list(range(64)) + [7] * 36, 16)
        self.assertEqual(len(a), 6)                  # full chunks only
        self.assertEqual(a[:4], b[:4])               # chained: the same prefix, the same hashes
        self.assertNotEqual(a[4], b[4])
        t = HostTier(budget_bytes=300, chunk=16)
        for i, h in enumerate(a[:3]):
            self.assertTrue(t.put(h, {"i": i}, 100))
        self.assertFalse(t.put(a[0], {"i": 0}, 100))  # there already
        self.assertEqual(t.prefix_hits(b), 3)
        t.prefix_hits([a[0]])                        # a[0] most recent: a[1] is the oldest now
        self.assertTrue(t.put(a[3], {"i": 3}, 100))
        self.assertNotIn(a[1], t)
        self.assertIn(a[0], t)
        self.assertEqual(t.prefix_hits(a), 1)        # a[1] gone: the prefix stops there
        self.assertFalse(t.put("big", {}, 301))
        self.assertEqual(t.snapshot()["rejected"], 1)
        g = t.generation
        t.clear()
        self.assertEqual((len(t.entries), t.bytes, t.generation), (0, 0, g + 1))


class StepRecordTests(unittest.TestCase):
    def test_scheduler_records_each_step_rows(self):
        install_vllm_stub()
        from kv_compare.pd import connector as cn
        s = cn._SchedulerSide({}, 16, True, cn._Log(None, "Ps"))
        so1 = SimpleNamespace(total_num_scheduled_tokens=2048,
                              scheduled_new_reqs=[SimpleNamespace(req_id="a", num_computed_tokens=0)],
                              scheduled_cached_reqs=SimpleNamespace(req_ids=["z"], num_computed_tokens=[77]),
                              num_scheduled_tokens={"a": 2047, "z": 1})
        so2 = SimpleNamespace(total_num_scheduled_tokens=1001, scheduled_new_reqs=[],
                              scheduled_cached_reqs=SimpleNamespace(req_ids=["a"], num_computed_tokens=[2047]),
                              num_scheduled_tokens={"a": 1000, "b": 1})
        s.record_steps(so1)
        s.record_steps(so2)
        s.record_steps(None)
        self.assertEqual(s.steps["a"], [(0, 2047, 2048), (2047, 1000, 1001)])
        self.assertEqual(s.steps["z"], [(77, 1, 2048)])
        self.assertNotIn("b", s.steps)               # no start position known: not recorded


class _Embed(torch.nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.w = torch.nn.Parameter((torch.randn(V, H) * 0.5).to(torch.bfloat16))

    def forward(self, ids):
        return self.w[ids]


class _Norm(torch.nn.Module):
    def forward(self, x):
        v = x.float()
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


class _Attn:
    def __init__(self, H, q, kv, hd):
        self.q_size, self.kv_size, self.hd = q, kv, hd
        lin = torch.nn.Linear(H, q + 2 * kv, bias=False).to(torch.bfloat16)
        self.qkv_proj = lambda x: (lin(x), None)

    def rotary_emb(self, pos, q, k):
        n = k.shape[0]
        kk = k.reshape(n, -1, self.hd).float()
        h = self.hd // 2
        f = pos.float()[:, None] * (1e-2 * torch.arange(1, h + 1).float())[None, :]
        c, s = torch.cos(f)[:, None, :], torch.sin(f)[:, None, :]
        a, b = kk[..., :h], kk[..., h:]
        return q, torch.cat([a * c - b * s, b * c + a * s], -1).reshape(n, -1).to(k.dtype)


class Layer0Tests(unittest.TestCase):
    def engine(self):
        from kv_compare.pd.layer0 import EngineLayer0
        torch.manual_seed(0)
        return EngineLayer0(embed=_Embed(50, 32), norm=_Norm(), attn=_Attn(32, 32, 16, 8))

    def test_steps_and_check(self):
        from kv_compare.pd.layer0 import Layer0Context, covered
        e = self.engine()
        tokens = [int(x) for x in torch.randint(0, 50, (40,))]
        steps = [(0, 24, 30), (24, 16, 20)]           # two prefill steps, other requests' rows besides
        full = e.words(tokens, 0, 40, steps)
        self.assertEqual(tuple(full.shape), (40, 32))
        # a window inside: the same rows, computed in products of the same row counts
        part = e.words(tokens, 20, 12, steps)
        self.assertTrue(torch.equal(part, full[20:32]))
        self.assertTrue(covered(steps, 0, 40))
        self.assertFalse(covered([(0, 24, 30)], 0, 40))
        self.assertFalse(covered([(4, 36, 40)], 0, 40))
        ctx = Layer0Context(tokens, steps, engine=e)
        self.assertTrue(ctx.covers(0, 40))
        self.assertTrue(ctx.matches(full, 0))
        self.assertEqual(ctx.last_check, "equal")
        self.assertTrue(torch.equal(ctx.layer0(0, 40, ctx.steps_for(0, 40)), full))
        bad = full.clone()
        bad[3, 5] += 1
        self.assertFalse(ctx.matches(bad, 0))
        self.assertIn("differs in 1 of", ctx.last_check)
        self.assertFalse(Layer0Context(tokens, [(0, 24, 30)], engine=e).covers(0, 40))

    def test_rounded_row_counts_are_found(self):
        """vLLM may pad a step to a CUDA graph size: the sender finds the padded count and sends it."""
        from kv_compare.pd.layer0 import Layer0Context

        class Probe:                                 # a product whose bits depend on the row count, like cuBLAS
            def __init__(self, e):
                self.e = e

            def words(self, tokens, pos0, T, steps):
                out = self.e.words(tokens, pos0, T, steps)
                return out + int(steps[0][2] % 64 != 0)
        e = self.engine()
        tokens = [int(x) for x in torch.randint(0, 50, (20,))]
        cache = e.words(tokens, 0, 20, [(0, 20, 64)])  # the engine ran 64 rows (20 scheduled)
        ctx = Layer0Context(tokens, [(0, 20, 20)], engine=Probe(e))
        self.assertTrue(ctx.matches(cache, 0))
        self.assertEqual(ctx.steps, [(0, 20, 64)])
        self.assertIn("rounded up to 64", ctx.last_check)


class TierFlowTests(unittest.TestCase):
    """One colocated engine with the CPU tier (raw codec, CPU caches): a finished request's full chunks are stored
    by the background thread; a later request with the same first tokens loads them into its own blocks, checked
    bit-exact; a new generation empties the tier."""

    @classmethod
    def setUpClass(cls):
        install_vllm_stub()
        from kv_compare.pd import connector as cn
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.v1.request import RequestStatus
        cls.cn, cls.Role, cls.Status = cn, KVConnectorRole, RequestStatus

    def setUp(self):
        from kv_compare.pd import tier
        tier._TIERS.clear()

    def step(self, s, w, finished=()):
        w.bind_connector_metadata(s.build_connector_meta(None))
        w.start_load_kv(None)
        r = w.get_transfer_results(set(finished))
        w.clear_connector_metadata()
        s.update_connector_output(SimpleNamespace(finished_sending=r.finished_sending,
                                                  finished_recving=r.finished_recving, failed_recving=r.failed_recving))
        return r

    def test_store_then_load(self):
        log_dir = tempfile.mkdtemp()
        ctl = Path(tempfile.mkdtemp()) / "tier.json"
        ctl.write_text(json.dumps({"generation": 1, "codec": "raw"}))
        extra = {"codecs": ["raw"], "log_dir": log_dir, "tier": {"budget_gb": 1, "chunk": 16, "control": str(ctl)}}
        S, W = (self.cn.TKVConnector(_vcfg(False, extra), r) for r in (self.Role.SCHEDULER, self.Role.WORKER))
        caches = _caches(3, 20, 2, 16, 8, 3)
        W.register_kv_caches(caches)
        self.addCleanup(W.shutdown)
        a = list(range(200, 240))                     # 40 tokens: 2 full chunks of 16
        ra = SimpleNamespace(request_id="a", prompt_token_ids=a, num_computed_tokens=40, kv_transfer_params=None,
                             status=self.Status.FINISHED_STOPPED)
        self.assertEqual(S.get_num_new_matched_tokens(ra, 0), (0, False))    # nothing stored yet
        delay, _ = S.request_finished(ra, [5, 2, 9])
        self.assertTrue(delay)
        r = self.step(S, W, finished={"a"})
        self.assertEqual(r.finished_sending, {"a"})
        tier = S.s.tier
        end = time.time() + 20
        while len(tier.entries) < 2 and time.time() < end:
            time.sleep(0.02)
        self.assertEqual(len(tier.entries), 2)
        self.assertEqual(S.request_finished(ra, [5, 2, 9]), (False, None))  # stored already: nothing to do
        b = a[:32] + [9] * 10
        rb = SimpleNamespace(request_id="b", prompt_token_ids=b, num_computed_tokens=0, kv_transfer_params=None,
                             status=self.Status.FINISHED_STOPPED)
        n, asyn = S.get_num_new_matched_tokens(rb, 0)
        self.assertEqual((n, asyn), (32, True))
        S.update_state_after_alloc(rb, SimpleNamespace(get_block_ids=lambda: ([1, 7, 3],)), n)
        r = self.step(S, W)
        self.assertEqual(r.finished_recving, {"b"})
        self.assertEqual(r.failed_recving, set())
        from kv_compare.pd.paged import Paged
        names = [f"model.layers.{l}.self_attn.attn" for l in range(3)]
        pg = Paged([caches[n] for n in names], 16, 2, 8)
        self.assertTrue(torch.equal(pg.gather([1, 7, 3], 0, 32), pg.gather([5, 2, 9], 0, 32)))
        recs = [json.loads(l) for f in Path(log_dir).glob("tkv_*.jsonl") for l in f.read_text().splitlines()]
        loads = [x for x in recs if x.get("event") == "install" and x.get("kind") == "tier"]
        self.assertEqual(len(loads), 1)
        self.assertEqual((loads[0]["exact_chunks"], loads[0]["chunks"]), (2, 2))
        stores = [x for x in recs if x.get("event") == "tier_store"]
        self.assertEqual(stores[0]["stored"], 2)
        # a request whose prefix hit covers part of a chunk: only the full chunks count
        rc = SimpleNamespace(request_id="c", prompt_token_ids=a[:20], num_computed_tokens=0, kv_transfer_params=None)
        self.assertEqual(S.get_num_new_matched_tokens(rc, 0), (16, True))
        self.assertEqual(S.get_num_new_matched_tokens(rc, 16), (0, False))    # the GPU has it already
        ctl.write_text(json.dumps({"generation": 2, "codec": "raw"}))
        time.sleep(0.01)
        self.step(S, W)
        self.assertEqual(len(tier.entries), 0)
        self.assertEqual(S.get_num_new_matched_tokens(rb, 0), (0, False))
        self.assertTrue(S.reset_cache())


T2_WIRE_CHECK = r"""
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
from kv_compare.pd.export import _cpu
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, max_position_embeddings=2048, rope_theta=500000.0)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
names = [f"train{i}" for i in range(6)] + ["tables", "tables2", "test"]
root = Path(tempfile.mkdtemp()) / "c"
reports = []
for i, n in enumerate(names):
    ids = [int(v) for v in np.random.default_rng(i).integers(0, 97 if i % 2 else 40, size=96)]
    reports.append(capture_ids(model, ids, root, n, torch.device("cpu"), queries=n.startswith("train")))
(root / "meta.json").write_text(json.dumps(dict(model_meta(model, "tiny", None), tokens=96, captures=reports)))
sets = {}
for name, tr, tb in (("a", names[0:3], "tables"), ("b", names[3:6], "tables2")):
    st = Study(root, "cpu", tr, tb, ["test"], log=lambda m: None)
    c = fused.prepare(st, log=lambda m: None)
    sets[name] = (st, c, layerwise.prepare_rank(st, c, 64, lambda m: None))
study, consts, rc = sets["a"]
b = export.make_bundle(study, consts, rc, {}, 512)
b["t2"] = {"order": ["a", "b"], "pick_tokens": 32, "pick_layers": [1, 2],
           "sets": {"a": {"consts": b["consts"], "rank": b["rank"], "groups": {}},
                    "b": {"consts": _cpu(sets["b"][1]), "rank": _cpu(sets["b"][2]), "groups": {}}}}
p = Path(tempfile.mkdtemp()) / "b.pt"
torch.save(b, p)
codecs.PAD = 64
doc = study.tests[0]
L, W = study.L, study.W
w32 = torch.stack([doc.words(l, "cpu") for l in range(L)])
words = (w32 - 65536 * (w32 >= 32768).to(torch.int32)).to(torch.int16)

class Ctx:
    # the receiver's layer 0: the true one (the recompute is tested in Layer0Tests)
    def __init__(self, w, ok=True):
        self.w, self.ok, self.last_check = w, ok, None
    def covers(self, pos0, T):
        return True
    def matches(self, l0, pos0):
        self.last_check = "equal" if self.ok else "differs"
        return self.ok and bool(torch.equal(l0, self.w[0, pos0:pos0 + l0.shape[0]]))
    def steps_for(self, pos0, T):
        return [[pos0, T, T]]
    def layer0(self, pos0, T, steps):
        assert steps == [[pos0, T, T]], steps
        return self.w[0, pos0:pos0 + T]

built, errors, meta = codecs.build(str(p), "cpu", ["t2-dense", "t2-rank256", "ours-dense"])
assert not errors, errors
for T, pos0 in ((64, 0), (50, 7)):
    w = words[:, pos0:pos0 + T].contiguous()
    sizes = {}
    for n, c in built.items():
        for ok in ((True, False) if n.startswith("t2-") else (None,)):
            ctx = None if ok is None else Ctx(words, ok)
            parts, m = c.encode(w, pos0=pos0, ctx=ctx) if ctx is not None else c.encode(w, pos0=pos0)
            h, blob = wire.bundle(parts, m)
            hb = wire.header_from_bytes(wire.header_to_bytes(h))
            p2, m2 = wire.unbundle(hb, blob.clone())
            got = {}
            sink = lambda l, x: got.__setitem__(l, x.clone())
            if ctx is not None:
                c.decode(p2, m2, sink, ctx=Ctx(words))
            else:
                c.decode(p2, m2, sink)
            back = torch.stack([got[l] for l in range(L)])
            assert torch.equal(back, w), (n, ok, T, int((back != w).sum()))
            if ok is not None:
                assert m["layer0"] == ("recomputed" if ok else "coded"), m
                assert m["set"] in ("a", "b"), m
            sizes[(n, ok)] = wire.payload_bytes(h)
            print("ok", n, ok, T, pos0, m.get("set"), m.get("layer0"), sizes[(n, ok)])
    assert sizes[("t2-dense", True)] < sizes[("t2-dense", False)], sizes
print("ok t2 wire")
"""


class T2WireTests(unittest.TestCase):
    def test_t2_round_trips_with_and_without_layer0_interpreted(self):
        """T2 through the wire: the set picked per request, layer 0 left out when the check passes (a smaller
        stream) and coded otherwise, both bit-exact; padded token counts and a non-zero start position."""
        import importlib.util
        if importlib.util.find_spec("triton") is None or importlib.util.find_spec("transformers") is None:
            self.skipTest("Triton and transformers are required")
        r = subprocess.run([sys.executable, "-c", T2_WIRE_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=7200)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-3000:])
        self.assertIn("ok t2 wire", r.stdout)


@unittest.skipIf(__import__("importlib").util.find_spec("fastapi") is None, "fastapi / uvicorn / aiohttp needed")
class ProxyBaselineTests(unittest.TestCase):
    def test_nixl_flow_and_two_prefills(self):
        import asyncio
        import socket
        import threading
        import urllib.request
        from aiohttp import web
        seen = {"p1": [], "p2": [], "d": []}
        ports = {}
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def handler(name):
            async def h(request):
                body = await request.json()
                seen[name].append(body)
                if name.startswith("p"):
                    await asyncio.sleep(0.3)
                    return web.json_response({"choices": [{"text": "x"}],
                                              "kv_transfer_params": {"remote_engine_id": name, "remote_block_ids": [1]}})
                return web.json_response({"choices": [{"text": "decoded"}]})
            return h

        def run():
            asyncio.set_event_loop(loop)

            async def start():
                for name in seen:
                    app = web.Application()
                    app.router.add_post("/v1/completions", handler(name))
                    runner = web.AppRunner(app)
                    await runner.setup()
                    site = web.TCPSite(runner, "127.0.0.1", 0)
                    await site.start()
                    ports[name] = site._server.sockets[0].getsockname()[1]
                ready.set()
            loop.run_until_complete(start())
            loop.run_forever()
        threading.Thread(target=run, daemon=True).start()
        ready.wait(10)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "pd_proxy.py"), "--port", str(port),
                                 "--prefill", f"http://127.0.0.1:{ports['p1']}",
                                 "--prefill", f"http://127.0.0.1:{ports['p2']}",
                                 "--decode", f"http://127.0.0.1:{ports['d']}=127.0.0.1:1"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (proc.kill(), proc.wait()))
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)

        def req(body, path="/v1/completions"):
            r = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=30) as f:
                return f.read().decode()
        req({"mode": "nixl"}, "/t/config")
        out = req({"model": "m", "prompt": "hi", "max_tokens": 7})
        self.assertIn("decoded", out)
        p = seen["p1"][0]
        self.assertEqual((p["max_tokens"], p["stream"], p["kv_transfer_params"]["do_remote_decode"]), (1, False, True))
        d = seen["d"][0]
        self.assertEqual(d["kv_transfer_params"], {"remote_engine_id": "p1", "remote_block_ids": [1]})
        self.assertEqual(d["max_tokens"], 7)
        # two requests at once: the second goes to the other prefill engine (fewest in flight)
        ts = [threading.Thread(target=req, args=({"model": "m", "prompt": f"x{i}"},)) for i in range(2)]
        for t in ts:
            t.start()
            time.sleep(0.05)
        for t in ts:
            t.join()
        self.assertGreaterEqual(len(seen["p2"]), 1)
        st = json.loads(urllib.request.urlopen(url + "/t/status", timeout=5).read())
        self.assertEqual(len(st["prefills"]), 2)
        self.assertEqual(st["prefill_inflight"], [0, 0])
        bad = urllib.request.Request(url + "/t/config", data=b'{"mode": "nope"}',
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(Exception):
            urllib.request.urlopen(bad, timeout=5)


class BenchAndReportTests(unittest.TestCase):
    def test_codec_filter_tier_stats_and_matrix_report(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        import pd_matrix_report as mr
        logs = Path(tempfile.mkdtemp())
        (logs / "tkv_Dw_1.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"event": "ready", "codecs": ["raw", "ours-hbh", "t2-hbh"], "codec_errors": {"exans": "no library"},
             "warm_up": {"raw": True, "ours-hbh": True, "t2-hbh": True}},
            {"event": "install", "kind": "tier", "tokens": 512, "raw_bytes": 1000, "loaded_bytes": 500,
             "decode_install_ms": 4.0, "h2d_ms": 1.0, "decode_ms": 3.0, "exact_chunks": 2, "chunks": 2, "taken_at": 5},
            {"event": "tier_store", "stored": 3, "store_ms": 9.0, "finished_at": 5},
            {"event": "tier", "chunks": 11, "evicted": 2, "time": 5}]) + "\n")
        (logs / "tkv_Pw_2.jsonl").write_text(json.dumps({"event": "ready", "codecs": ["raw", "ours-hbh"],
                                                          "warm_up": {"t2-hbh": "ValueError('x')"}}) + "\n")
        a = SimpleNamespace(log_dir=str(logs))
        res = {}
        self.assertEqual(pb.usable(a, res, ["raw", "t2-hbh", "exans", "nixl"]), ["raw", "nixl"])
        self.assertEqual(set(res["skipped"]), {"t2-hbh", "exans"})
        self.assertIn("no library", res["skipped"]["exans"])
        st = pb.tier_stats(pb.records(str(logs), 0, 10))
        self.assertEqual((st["loads"], st["exact_chunks"], st["stored_chunks"], st["bits"]), (1, 2, 3, 8.0))
        self.assertEqual(pb.pct([5, 1, 3, -1, None], 50), 3)
        # the matrix report over two run folders
        root = Path(tempfile.mkdtemp())
        sweep = {"sweep_raw_10gbps_rinf": {"link_gbps": 10, "codec": "raw", "rate": "inf",
                                           "bench": {"request_throughput": 1.0, "request_goodput": 0.5,
                                                     "median_ttft_ms": 900, "p99_ttft_ms": 2000}, "transfers": {}},
                 "sweep_t2-hbh_10gbps_rinf": {"link_gbps": 10, "codec": "t2-hbh", "rate": "inf",
                                              "bench": {"request_throughput": 1.5, "request_goodput": 1.2,
                                                        "median_ttft_ms": 600, "p99_ttft_ms": 1500},
                                              "transfers": {"bits": 7.9, "layer0_recomputed": 40, "layer0_coded": 8,
                                                            "sets": {"code": 30, "math": 18},
                                                            "layer0_checks": ["equal"]}}}
        for name, env, res in (
                ("r1", {"layout": "1p3d", "connector": "tkv", "transport": "ring", "model": "m", "gpus": "A5000"},
                 {"sweep": sweep, "check": {"t2-hbh": {"prompts": 4, "same_text_as_local": 4, "ok": True,
                                                       "transfers": {"exact": 4, "bits": 7.9}}}}),
                ("r2", {"layout": "colo4", "connector": "lmcache"},
                 {"tier": {"tier_lmcache_c8": {"codec": "lmcache", "clients": 8, "bench": {"median_ttft_ms": 300},
                                               "tier": {"loads": 0}}}, "skipped": {"ibp": "No module named 'ibp'"}})):
            d = root / name
            d.mkdir()
            (d / "results.json").write_text(json.dumps(res))
            (d / "env.json").write_text(json.dumps(env))
        out = root / "m" / "summary.md"
        mr.main(["--runs", "r1,r2,missing", "--root", str(root), "--output", str(out), "--failed", "LAYOUT=2p2d;"])
        text = out.read_text()
        self.assertIn("## 1p3d, host memory: link 10 Gb/s", text)
        self.assertIn("rate inf: t2-hbh (goodput 1.20/s", text)
        self.assertIn("| t2-hbh | 7.900 |", text)
        self.assertIn("LMCache CPU offload (raw)", text)
        self.assertIn("| 1p3d, host memory | t2-hbh | 10 | inf | 40 | 8 |", text)
        self.assertIn("ibp: No module named 'ibp'", text)
        self.assertIn("`LAYOUT=2p2d`", text)


class NodeSpecsTests(unittest.TestCase):
    def test_system_part_and_summary(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import node_specs as ns
        out = Path(tempfile.mkdtemp())
        self.assertEqual(ns.main(["--output", str(out), "--skip-bandwidth"]), 0)
        spec = json.loads((out / "specs.json").read_text())
        self.assertIn("host", spec["system"])
        self.assertIn("# Node specs:", (out / "summary.md").read_text())
        leg = lambda g: {"seconds": 0.1, "GB_per_s": g, "Gb_per_s": 8 * g}
        spec["bandwidth"] = {"probe": {"host_legs": [{"gpu": 0, "d2h_pinned": leg(24), "h2d_pinned": leg(25),
                                                      "d2h_pageable": leg(8), "h2d_pageable": leg(9)}],
                                       "pairs": [{"src": 0, "dst": 1, "peer_access": True, "direct": leg(20),
                                                  "host": leg(11), "host_pipe": leg(19)}]},
                             "patterns": [{"pattern": "2p2d, prefill 0,1", "kind": "direct", "legs": ["0->2", "1->3"],
                                           "correct": True, "seconds": 0.1, "aggregate_GB_per_s": 30.0,
                                           "per_leg_GB_per_s": [15, 15]}]}
        text = ns.summary(spec)
        self.assertIn("- 2p2d direct: 2p2d, prefill 0,1 (0->2, 1->3), 30.0 GB/s aggregate", text)
        self.assertIn("| 0 -> 1 | True | 20.0 GB/s | 11.0 GB/s | 19.0 GB/s |", text)


if __name__ == "__main__":
    unittest.main()


FAKE_VLLM = r'''#!/usr/bin/env python3
"""A stand-in for vLLM's CLI in the runner's dry run: `vllm serve` answers /health, /v1/completions (with
kv_transfer_params for NIXL prefills) and /reset_prefix_cache; `vllm bench serve` writes a result file."""
import json, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
if args[:2] == ["bench", "serve"]:
    d = args[args.index("--result-dir") + 1]
    n = args[args.index("--result-filename") + 1]
    json.dump({"completed": 8, "request_throughput": 1.5, "request_goodput": 1.0, "median_ttft_ms": 500.0,
               "p99_ttft_ms": 900.0, "median_tpot_ms": 20.0}, open(os.path.join(d, n), "w"))
    sys.exit(0)
assert args[0] == "serve", args
port = int(args[args.index("--port") + 1])
cfg = json.loads(args[args.index("--kv-transfer-config") + 1])
log = os.environ.get("FAKE_LOG")
with open(log, "a") as f:
    f.write(json.dumps({"port": port, "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "cfg": cfg,
                        "argv": args, "env": {k: v for k, v in os.environ.items() if k.startswith(("LMCACHE", "VLLM_NIXL"))}}) + "\n")
class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        self.reply({"ok": True, "data": [{"id": "llama"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        out = {"choices": [{"text": " same"}], "success": True}
        if (body.get("kv_transfer_params") or {}).get("do_remote_decode"):
            out["kv_transfer_params"] = {"remote_engine_id": str(port)}
        self.reply(out)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

FAKE_SMI = r'''#!/usr/bin/env bash
case "$*" in
  *--query-gpu=index,name*) printf '0, NVIDIA RTX A5000, 1 MiB, 24564 MiB, 595.45\n1, NVIDIA RTX A5000, 1 MiB, 24564 MiB, 595.45\n2, NVIDIA RTX A5000, 1 MiB, 24564 MiB, 595.45\n3, NVIDIA RTX A5000, 1 MiB, 24564 MiB, 595.45\n' ;;
  *--query-gpu=index*) printf '0\n1\n2\n3\n' ;;
  *--query-gpu=name*) printf 'NVIDIA RTX A5000\nNVIDIA RTX A5000\nNVIDIA RTX A5000\nNVIDIA RTX A5000\n' ;;
  *) echo "fake nvidia-smi" ;;
esac
'''


@unittest.skipIf(__import__("importlib").util.find_spec("fastapi") is None, "fastapi / uvicorn / aiohttp needed")
class RunnerDryRunTests(unittest.TestCase):
    """scripts/run_pd_vllm.sh end to end on a throwaway clone with stand-ins for vLLM and nvidia-smi: the engines'
    GPUs, ports and connector configurations for every layout and connector, the proxy, the bench, the packaging."""

    def setUp(self):
        import shutil
        if shutil.which("curl") is None or shutil.which("git") is None:
            self.skipTest("curl and git needed")
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = self.tmp / "repo"
        subprocess.run(["git", "clone", "-q", str(ROOT), str(self.repo)], check=True)
        br = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
        for d in ("scripts", "kv_compare"):          # the working tree's, committed or not
            for f in (ROOT / d).rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts:
                    dst = self.repo / f.relative_to(ROOT)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, dst)
        subprocess.run(["git", "-C", str(self.repo), "add", "-A", "scripts", "kv_compare"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "checkout", "-q", "-B", br], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam",
                        "working tree"], check=False)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "t@t"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "t"], check=True)
        self.kv = self.tmp / "kv"
        stub = self.tmp / "stub"
        (stub / "vllm").mkdir(parents=True)
        (stub / "vllm" / "__init__.py").write_text("__version__ = '0.32.0'\n")
        bindir = self.tmp / "bin"
        bindir.mkdir()
        (bindir / "nvidia-smi").write_text(FAKE_SMI)
        (bindir / "vllm").write_text(FAKE_VLLM.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        for env in ("venv_vllm", "venv_lmcache"):
            b = self.kv / env / "bin"
            b.mkdir(parents=True)
            (b / "python").write_text(f'#!/usr/bin/env bash\nPYTHONPATH="{stub}:${{PYTHONPATH:-}}" exec {sys.executable} "$@"\n')
            (b / "vllm").write_text(FAKE_VLLM.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
        (self.kv / "venv_vllm" / "lib").mkdir()
        (stub / "nixl.py").write_text("")
        for f in list(bindir.iterdir()) + list((self.kv / "venv_vllm" / "bin").iterdir()) + \
                list((self.kv / "venv_lmcache" / "bin").iterdir()):
            f.chmod(0o755)
        run = self.kv / "runs" / "llama32_3b"
        run.mkdir(parents=True)
        (run / "pd_codecs.pt").write_bytes(b"x")
        self.path = f"{bindir}:{Path(sys.executable).parent}:/usr/bin:/bin"

    def run_pd(self, **env):
        import os
        log = self.tmp / f"engines_{len(list(self.tmp.glob('engines_*')))}.jsonl"
        base = dict(os.environ, PATH=self.path, KV_HOME=str(self.kv), ALLOW_NO_SLURM="1", NO_PUSH="1", PD_MATRIX="1",
                    SMOKE="1", MODEL_PATH=str(self.tmp / "model"), FAKE_LOG=str(log), PORT_BASE=str(30000 + 500 *
                                                                                                   len(list(self.tmp.glob("engines_*")))))
        base.pop("CUDA_VISIBLE_DEVICES", None)
        # the prompts the runner would make with the tokenizer
        prompts = self.kv / "runs" / "llama32_3b" / "pd" / "prompts_2048_check_1000_3000_5000_8192"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "check.jsonl").write_text(json.dumps({"prompt": "hello", "output_tokens": 4}) + "\n")
        (prompts / "prompts_mixed.jsonl").write_text(json.dumps({"prompt": "hello", "output_tokens": 4}) + "\n")
        base.update({k: str(v) for k, v in env.items()})
        r = subprocess.run(["bash", "scripts/run_pd_vllm.sh"], cwd=str(self.repo), env=base, capture_output=True,
                           text=True, timeout=600)
        engines = [json.loads(l) for l in log.read_text().splitlines()] if log.is_file() else []
        return r, engines

    def test_layouts_connectors_and_packaging(self):
        r, eng = self.run_pd(LAYOUT="2p2d", TRANSPORT="p2p", CODECS="raw,ours-hbh")
        self.assertIn("NO_PUSH=1: committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        roles = sorted((e["cfg"]["kv_role"], e["cvd"]) for e in eng)
        # p2p: every engine sees every GPU, its own first; prefills on the first two job GPUs
        self.assertEqual(roles, [("kv_consumer", "2,0,1,3"), ("kv_consumer", "3,0,1,2"),
                                 ("kv_producer", "0,1,2,3"), ("kv_producer", "1,0,2,3")])
        p = next(e for e in eng if e["cfg"]["kv_role"] == "kv_producer")
        ex = p["cfg"]["kv_connector_extra_config"]
        self.assertEqual((ex["transport"], ex["codecs"]), ("p2p", ["raw", "ours-hbh"]))
        self.assertIn("--host", p["argv"])
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_*smoke*"))
        self.assertEqual(len(out), 1, list((self.repo / "benchmark_results").glob("*")))
        env = json.loads((out[0] / "env.json").read_text())
        self.assertEqual((env["layout"], env["connector"], env["transport"], env["prefills"]), ("2p2d", "tkv", "p2p", 2))
        res = json.loads((out[0] / "results.json").read_text())
        self.assertIn("raw", res["check"])
        # NIXL: one GPU each, side channels, the proxy's nixl mode through the bench
        r, eng = self.run_pd(LAYOUT="1p3d", CONNECTOR="nixl")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertEqual(sorted(e["cvd"] for e in eng), ["0", "1", "2", "3"])
        self.assertTrue(all(e["cfg"]["kv_connector"] == "NixlConnector" for e in eng))
        self.assertEqual(len({e["env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] for e in eng}), 4)
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_1p3d_nixl_*"))
        res = json.loads((out[0] / "results.json").read_text())
        self.assertTrue(res["check"]["nixl"]["ok"])
        self.assertIn("sweep_nixl_nativegbps_rinf", res["sweep"])
        # colocated with the CPU tier, and LMCache
        r, eng = self.run_pd(LAYOUT="colo4", TIER_CODECS="raw,ours-hbh")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue(all(e["cfg"]["kv_role"] == "kv_both" for e in eng))
        tier = eng[0]["cfg"]["kv_connector_extra_config"]["tier"]
        self.assertEqual((tier["chunk"], tier["budget_gb"]), (256, 8))
        self.assertIn("--num-gpu-blocks-override", eng[0]["argv"])
        r, eng = self.run_pd(LAYOUT="colo4", CONNECTOR="lmcache")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue(all(e["cfg"]["kv_connector"] == "LMCacheConnectorV1" for e in eng))
        self.assertEqual(eng[0]["env"]["LMCACHE_MAX_LOCAL_CPU_SIZE"], "8")
        r, _ = self.run_pd(LAYOUT="colo4", CONNECTOR="nixl")
        self.assertIn("NIXL needs prefill and decode engines", r.stdout)

    def test_matrix_runs_each_and_reports_once(self):
        import os
        bare = self.tmp / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        br = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
        subprocess.run(["git", "-C", str(self.repo), "remote", "set-url", "origin", str(bare)], check=True)
        # one commit with the tree (the local repository may be a partial clone: its history cannot be pushed)
        g = lambda *a: subprocess.run(["git", "-C", str(self.repo), *a], check=True, capture_output=True)
        g("checkout", "-q", "--orphan", "matrix-test")
        g("add", "-A")
        g("commit", "-qm", "tree")
        g("branch", "-M", br)
        subprocess.run(["git", "-C", str(self.repo), "push", "-q", "origin", f"HEAD:{br}"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "fetch", "-q", "origin"], check=True)
        prompts = self.kv / "runs" / "llama32_3b" / "pd" / "prompts_2048_check_1000_3000_5000_8192"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "check.jsonl").write_text(json.dumps({"prompt": "hello", "output_tokens": 4}) + "\n")
        (prompts / "prompts_mixed.jsonl").write_text(json.dumps({"prompt": "hello", "output_tokens": 4}) + "\n")
        env = dict(os.environ, PATH=self.path, KV_HOME=str(self.kv), ALLOW_NO_SLURM="1", NO_PUSH="1", SMOKE="1",
                   MODEL_PATH=str(self.tmp / "model"), FAKE_LOG=str(self.tmp / "m.jsonl"), PORT_BASE="34000",
                   RUNS="LAYOUT=1p3d CODECS=raw;\n   LAYOUT=colo4 CONNECTOR=lmcache;  LAYOUT=colo4 CONNECTOR=nixl")
        env.pop("CUDA_VISIBLE_DEVICES", None)
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=str(self.repo), env=env, capture_output=True,
                           text=True, timeout=900)
        self.assertIn("NO_PUSH=1: committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("run 3 did not finish", r.stdout)
        rep = list((self.repo / "benchmark_results").glob("pd_matrix_smoke_rtxa5000_*/summary.md"))
        self.assertEqual(len(rep), 1)
        text = rep[0].read_text()
        self.assertIn("2 runs:", text)
        self.assertIn("`LAYOUT=colo4 CONNECTOR=nixl`", text)
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--oneline", "-5"], capture_output=True,
                             text=True).stdout
        self.assertIn("PD matrix smoke: 3 runs (2 not finished)", log)   # the stand-ins log no transfers: run 1 fails its check
