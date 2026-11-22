"""Tests of the PD baselines branch (docs/pd_baselines.md) on the CPU: the CPU tier (budget, prefix hits, the whole
store-and-load flow through the connector with its checksums), the prefill steps the scheduler records, T2's
layer-0 recompute and its sender check, T2 through the wire (Triton interpreter), the proxy's NIXL flow and several
prefill engines, the bench's codec filter and tier statistics, the matrix report and the node-specs summary.

  python -m unittest kv_compare.test_pd_baselines
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import types
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


class Layer0FindTests(unittest.TestCase):
    def setUp(self):
        import kv_compare.pd.layer0 as l0
        self.l0 = l0
        l0._MODEL = None
        self.addCleanup(setattr, l0, "_MODEL", None)

    def fake(self):
        inner = torch.nn.Module()
        inner.embed_tokens = torch.nn.Embedding(4, 2)
        inner.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model = torch.nn.Module()
        model.model = inner
        return model

    def test_found_from_the_model_runner_up_the_stack(self):
        model = self.fake()

        class GPUModelRunner:                       # vLLM's runner calls register_kv_caches
            def __init__(self, m):
                self.model = m

            def initialize_kv_cache(self):
                return register()

        def register():                              # the connector's register calls remember_model()
            return self.l0.remember_model()
        self.assertIs(GPUModelRunner(model).initialize_kv_cache(), model)
        self.assertIs(self.l0.find_model(), model)

    def test_found_in_a_frozen_heap(self):
        import gc
        model = self.fake()
        gc.freeze()                                  # what vLLM does after start-up
        try:
            self.assertIs(self.l0.find_model(), model)
            self.assertGreater(gc.get_freeze_count(), 0, "the heap is frozen again")
        finally:
            gc.unfreeze()


class NvcompPieceTests(unittest.TestCase):
    """Our nvCOMP wrapper's pieces over a stand-in codec that, like nvCOMP, checks its input's alignment."""

    def fake(self):
        from kv_compare.pd import codecs

        class Arr:                                   # nvCOMP's array: dlpack, its buffer size
            def __init__(self, t):
                self.t = t
                self.buffer_size = t.numel()

            def __dlpack__(self, **kw):
                return self.t.__dlpack__(**kw)

            def __dlpack_device__(self):
                return self.t.__dlpack_device__()

        class Codec:                                 # stores a piece reversed, with a 4-byte header
            calls = []

            def encode(self, a):
                Codec.calls.append(a.t.numel())
                return Arr(torch.cat([torch.tensor([7, 7, 7, 7], dtype=torch.uint8), a.t.flip(0)]))

            def decode(self, a):
                return Arr(a.t[4:].flip(0))
        seen = []

        def as_array(t, cuda_stream=None):           # job 30252: "not aligned to the required alignment of 16 bytes"
            seen.append(t.data_ptr() % 16)
            return Arr(t)
        nv = codecs.Nvcomp.__new__(codecs.Nvcomp)
        nv.nv = types.SimpleNamespace(as_array=as_array)
        nv.codec, nv.name, nv.version, nv.device = Codec(), "nvcomp-x", "0", torch.device("cpu")
        nv.stream = types.SimpleNamespace(cuda_stream=0)
        nv._enter = nv._leave = lambda: None
        nv.piece = 1000
        words = torch.randint(-3000, 3000, (3, 7, 50), dtype=torch.int16)      # 2,100 bytes: three pieces
        return nv, words, Codec, seen

    def round_trip(self, offset):
        nv, words, Codec, seen = self.fake()
        parts, meta = nv.encode(words)
        self.assertEqual(Codec.calls[-3:], [1000, 1000, 100])
        self.assertEqual(meta["pieces"], [1004, 1004, 104])
        self.assertEqual(parts["coded"].numel(), 1008 + 1008 + 112)            # each piece padded to 16 bytes
        buf = torch.zeros(parts["coded"].numel() + offset, dtype=torch.uint8)  # as received: offset into a buffer
        buf[offset:] = parts["coded"]
        seen.clear()
        got = {}
        nv.decode({"coded": buf[offset:]}, meta, lambda l, w: got.__setitem__(l, w.clone()))
        self.assertTrue(all(torch.equal(got[l], words[l]) for l in range(3)))
        self.assertEqual(seen, [0, 0, 0])                                     # every piece decoded from aligned bytes

    def test_pieces_round_trip(self):
        self.round_trip(0)

    def test_unaligned_received_buffer(self):
        self.round_trip(4)


class ExansBatchTests(unittest.TestCase):
    """Our ExANS wrapper's batches over a stand-in library: uniform blocks fit a 66-byte record, the others go raw."""

    def lib(self, ex, bad=b"", fail=False):
        import ctypes
        calls = []

        class Lib:
            def openlake_expans_create(self, bb, rb, seg, nseg, cap, href):
                calls.append(("create", cap))
                return 0

            def openlake_expans_last_error(self, h):
                return b"stand-in"

            def openlake_expans_encode(self, h, addrs, sizes, offs, n, rec, fits, stream):
                calls.append(("encode", n))
                for i in range(n):
                    blk = ctypes.string_at(addrs[i], sizes[i])
                    if blk == blk[:2] * (len(blk) // 2):
                        head = bytearray(64)
                        head[0:4] = (0 if blk[:2] == bad else 0x31505845).to_bytes(4, "little")   # "EXP1"
                        head[20:24] = (66).to_bytes(4, "little")
                        ctypes.memmove(rec + i * ex.RECORD_BYTES, bytes(head) + blk[:2], 66)
                        fits[i] = 1
                    else:
                        fits[i] = 0
                return 0

            def openlake_expans_decode(self, h, rec, idx, n, oaddrs, osizes, ooffs, stream):
                calls.append(("decode", n))
                if fail:
                    return 1
                for k in range(n):
                    two = ctypes.string_at(rec + idx[k] * ex.RECORD_BYTES + 64, 2)
                    ctypes.memmove(oaddrs[k], two * (osizes[k] // 2), osizes[k])
                return 0
        return Lib(), calls

    def test_batches_round_trip(self):
        from unittest import mock
        from kv_compare import exans_official as ex
        from kv_compare.pd import codecs
        import ctypes
        c = codecs.ExANS.__new__(codecs.ExANS)
        c.ex, c.ct, c.device, c.handles, c.max_blocks, c.batch = ex, ctypes, torch.device("cpu"), {}, 1 << 16, 2
        c.lib, calls = self.lib(ex)
        nb = 5
        flat = torch.randint(-3000, 3000, (nb * ex.BLOCK_WORDS + 37,), dtype=torch.int16)
        for i in (0, 3, 4):                          # uniform blocks: they fit
            flat[i * ex.BLOCK_WORDS:(i + 1) * ex.BLOCK_WORDS] = 1234 + i
        words = flat.view(1, 1, -1)
        with mock.patch("torch.cuda.current_stream", return_value=types.SimpleNamespace(cuda_stream=0)):
            parts, meta = c.encode(words)
            got = {}
            c.decode(parts, meta, lambda l, w: got.__setitem__(l, w.clone()))
        self.assertEqual(meta["fits"], [True, False, False, True, True])
        self.assertEqual([x for x in calls if x[0] == "encode"], [("encode", 2), ("encode", 2), ("encode", 1)])
        self.assertEqual([x for x in calls if x[0] == "decode"], [("decode", 2), ("decode", 1)])
        self.assertTrue(max(x[1] for x in calls if x[0] == "create") <= 64)
        self.assertTrue(torch.equal(got[0], words[0]))
        self.assertEqual(meta["bad_records"], 0)

    def test_a_record_its_header_does_not_describe_goes_raw_and_a_failed_decode_says_where(self):
        """Job 30371: ExANS rejected one of its records ("invalid or incompatible ExpANS record header"), twice, on
        the same prompt. The sender now sends raw any record whose header lacks the magic or a sane size, and a
        failed decode says whether the headers arrived as the sender made them."""
        from unittest import mock
        from kv_compare import exans_official as ex
        from kv_compare.pd import codecs
        import ctypes
        flat = torch.randint(-3000, 3000, (3 * ex.BLOCK_WORDS,), dtype=torch.int16)
        for i in range(3):
            flat[i * ex.BLOCK_WORDS:(i + 1) * ex.BLOCK_WORDS] = 1234 + i
        words = flat.view(1, 1, -1)
        bad = (1235).to_bytes(2, "little", signed=True)
        for fail in (False, True):
            c = codecs.ExANS.__new__(codecs.ExANS)
            c.ex, c.ct, c.device, c.handles, c.max_blocks, c.batch = ex, ctypes, torch.device("cpu"), {}, 1 << 16, 4
            c.lib, calls = self.lib(ex, bad=bad, fail=fail)
            with mock.patch("torch.cuda.current_stream", return_value=types.SimpleNamespace(cuda_stream=0)):
                parts, meta = c.encode(words)
                self.assertEqual((meta["fits"], meta["bad_records"]), ([True, False, True], 1))
                got = {}
                if not fail:
                    c.decode(parts, meta, lambda l, w: got.__setitem__(l, w.clone()))
                    self.assertTrue(torch.equal(got[0], words[0]))
                    continue
                with self.assertRaisesRegex(RuntimeError, "stand-in; the headers arrived as the sender made them"):
                    c.decode(parts, meta, lambda l, w: None)
                parts["records"][70] ^= 1                # the second record's header
                with self.assertRaisesRegex(RuntimeError, "a header changed on the way"):
                    c.decode(parts, meta, lambda l, w: None)


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
        self.assertEqual(loads[0]["moved_bytes"], loads[0]["loaded_bytes"])    # a coded chunk moves what it holds
        stores = [x for x in recs if x.get("event") == "tier_store"]
        self.assertEqual(stores[0]["stored"], 2)
        # a request whose prefix hit covers part of a chunk: only the full chunks count
        rc = SimpleNamespace(request_id="c", prompt_token_ids=a[:20], num_computed_tokens=0, kv_transfer_params=None)
        self.assertEqual(S.get_num_new_matched_tokens(rc, 0), (16, True))
        self.assertEqual(S.get_num_new_matched_tokens(rc, 16), (0, False))    # the GPU has it already
        # a chunk that fails its check on load leaves the tier: vLLM computes it, and the next lookup stops before
        # it instead of asking for it again (job 30316: IBP's chunks, 18,000 loads an engine)
        for e in tier.entries.values():               # every stored chunk wrong: the load fails at the first
            e["checksum"] = json.loads(json.dumps(e["checksum"]).replace("1", "2", 1))
        first = set(tier.entries)
        rd = SimpleNamespace(request_id="d", prompt_token_ids=b, num_computed_tokens=0, kv_transfer_params=None,
                             status=self.Status.FINISHED_STOPPED)
        n, _ = S.get_num_new_matched_tokens(rd, 0)
        self.assertEqual(n, 32)
        S.update_state_after_alloc(rd, SimpleNamespace(get_block_ids=lambda: ([4, 6, 8],)), n)
        r = self.step(S, W)
        self.assertEqual((r.finished_recving, r.failed_recving), ({"d"}, {"d"}))
        self.assertEqual(len(first - set(tier.entries)), 1)                 # the first chunk dropped, only it
        self.assertEqual(S.get_num_new_matched_tokens(rd, 0), (0, False))
        recs = [json.loads(l) for f in Path(log_dir).glob("tkv_*.jsonl") for l in f.read_text().splitlines()]
        bad = [x for x in recs if x.get("event") == "install" and x.get("req") == "d"]
        self.assertEqual((len(bad), bad[0]["dropped_chunk"]), (1, (first - set(tier.entries)).pop()))
        self.assertIn("differ from the stored ones", bad[0]["error"])
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
# in pieces: each piece its own set and layer-0 check; the request's record says recomputed only if every piece was
codecs.PIECE_TOKENS = 64
c = built["t2-rank256"]
for ok in (True, False):
    w = words[:, :96].contiguous()
    parts, m = c.encode(w, pos0=0, ctx=Ctx(words, ok))
    assert len(m["pieces"]) == 2 and m["layer0"] == ("recomputed" if ok else "coded"), m
    assert m["set"] in ("a", "b") and len(m["sets"]) == 2, m
    h, blob = wire.bundle(parts, m)
    p2, m2 = wire.unbundle(wire.header_from_bytes(wire.header_to_bytes(h)), blob.clone())
    got = {}
    c.decode(p2, m2, lambda l, x: got.__setitem__(l, x.clone()), ctx=Ctx(words))
    assert torch.equal(torch.stack([got[l] for l in range(L)]), w), ok
    print("ok t2 pieces", ok, m["sets"])
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
                if body.get("prompt") == "fail":
                    return web.json_response({"error": "no"}, status=500)
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
        # our connector's flow: eight at once spread over both prefill engines (job 30210 had 7 on one)
        req({"mode": "pd", "codec": "raw"}, "/t/config")
        before = {n: len(seen[n]) for n in ("p1", "p2")}
        go = threading.Barrier(8)

        def at_once(i):
            go.wait()
            req({"model": "m", "prompt": f"y{i}", "max_tokens": 3})
        ts = [threading.Thread(target=at_once, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        got = {n: [b for b in seen[n][before[n]:] if b.get("kv_transfer_params", {}).get("t_send")] for n in ("p1", "p2")}
        self.assertEqual((len(got["p1"]), len(got["p2"])), (4, 4))
        with self.assertRaises(Exception):                  # a failed decode request: its prefill is let go
            req({"model": "m", "prompt": "fail", "max_tokens": 3})
        for _ in range(50):
            st = json.loads(urllib.request.urlopen(url + "/t/status", timeout=5).read())
            if st["prefill_inflight"] == [0, 0] and st["inflight"] == [0]:
                break
            time.sleep(0.1)
        self.assertEqual((st["prefill_inflight"], st["inflight"]), ([0, 0], [0]))
        bad = urllib.request.Request(url + "/t/config", data=b'{"mode": "nope"}',
                                     headers={"Content-Type": "application/json"})
        with self.assertRaises(Exception):
            urllib.request.urlopen(bad, timeout=5)


class ProxyBurstTests(unittest.TestCase):
    def test_requests_in_one_burst_spread_over_the_prefills(self):
        """Requests that arrive in one event-loop turn: each is counted on its prefill engine when picked (job
        30210: 7 of 8 went to one engine, as the count waited for the prefill task to run)."""
        import asyncio
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_proxy
        px = pd_proxy.Proxy(["http://p1", "http://p2"], [("http://d", "127.0.0.1:1")], "round_robin", "raw", None)
        sent = []

        class Resp:
            status = 200
            headers = {}

            def __init__(self):
                self.content = self

            async def iter_any(self):
                yield b"{}"

            async def read(self):
                await asyncio.sleep(0.05)
                return b"{}"

            def release(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def __await__(self):
                async def now():
                    return self
                return now().__await__()

        bodies = []

        class Session:
            def post(self, url, json=None, headers=None):
                sent.append(url)
                bodies.append(json)
                return Resp()

        class Req:
            def __init__(self, **extra):
                self.extra = extra

            async def json(self):
                return dict({"model": "m", "prompt": "x", "max_tokens": 2}, **self.extra)

        px.session = Session()

        async def burst():
            outs = await asyncio.gather(*(px.handle(Req(), "/v1/completions") for _ in range(8)))
            for o in outs:                              # the client reads each stream to its end
                async for _ in o.body_iterator:
                    pass
        asyncio.run(burst())
        self.assertEqual((sent.count("http://p1/v1/completions"), sent.count("http://p2/v1/completions")), (4, 4))
        self.assertEqual(px.p_inflight, [0, 0])
        del sent[:], bodies[:]                          # the warm-up's pinned pair, its fields not passed on

        async def pinned():
            o = await px.handle(Req(t_prefill=1, t_decode=0), "/v1/completions")
            async for _ in o.body_iterator:
                pass
        asyncio.run(pinned())
        self.assertIn("http://p2/v1/completions", sent)
        self.assertNotIn("http://p1/v1/completions", sent)
        self.assertFalse(any("t_prefill" in b or "t_decode" in b for b in bodies))


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
        self.assertIn("NIXL (vLLM), own GPU only", mr.path_of({"connector": "nixl"}))
        self.assertIn("every GPU visible", mr.path_of({"connector": "nixl", "nixl_all_gpus": True}))

    def test_a_dead_engine_stops_the_run_and_what_was_measured_is_reported(self):
        """Job 30334: exactkv's install killed all three decode engines at 10:38; the bench then waited 30 minutes
        after each later point. Now a dead engine (or a point still in flight after --idle-timeout) stops the run:
        the points before stay, the report is written, the exit is 3 (packaged as _partial)."""
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        import pd_matrix_report as mr
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}"
        s.close()                                   # nothing listens there: connection refused
        state = {"inflight": [1]}

        class H(BaseHTTPRequestHandler):
            def log_message(self, *x):
                pass

            def do_GET(self):
                if self.path == "/t/status":
                    b = json.dumps({"inflight": state["inflight"], "prefill_inflight": [0], "prefills": [alive],
                                    "decodes": [[dead, 0], [alive, 0]]}).encode()
                elif self.path == "/health":
                    b = b""
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("content-length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        alive = f"http://127.0.0.1:{srv.server_address[1]}"
        a = SimpleNamespace(proxy=alive)
        self.assertEqual(list(pb.dead_engines(a)), [dead])
        t = time.time()
        with self.assertRaises(pb.RunStopped) as cm:     # in flight, and an engine gone: stops at the first look
            pb.wait_idle(a.proxy, timeout=60, health=lambda: pb.dead_engines(a), every=0.2)
        self.assertLess(time.time() - t, 10)
        self.assertEqual(list(cm.exception.engines), [dead])
        ok, stop = pb.after_point(a)                     # recorded first, raised after
        self.assertFalse(ok)
        with self.assertRaises(pb.RunStopped) as cm:
            pb.end_point(a, "shape_exactkv_10gbps_t128_b16_o128", ok, stop)
        self.assertIn("shape_exactkv_10gbps_t128_b16_o128: engines stopped answering", cm.exception.why)
        old = pb.IDLE_TIMEOUT                            # every engine answers, the requests never finish
        self.addCleanup(setattr, pb, "IDLE_TIMEOUT", old)
        pb.IDLE_TIMEOUT = 1.5
        state["inflight"] = [0]
        self.assertTrue(pb.settle(SimpleNamespace(proxy=alive), "p"))
        state["inflight"] = [2]
        with self.assertRaises(pb.RunStopped) as cm:
            pb.stuck(SimpleNamespace(proxy=alive), "sweep_raw_10gbps_rinf")
        self.assertIn("still in flight 1.5 s", cm.exception.why)
        # main: the stage that stops, the stages after it skipped but the report, exit 3
        out = Path(tempfile.mkdtemp())
        calls = []

        def shapes(a, res):
            res.setdefault("shapes", {})["shape_raw_10gbps_t512_b1_o128"] = {
                "codec": "raw", "tokens": 512, "batch": 1, "output": 128, "rounds": 8, "families": ["sz1"],
                "link_gbps": 10, "requests": 8, "exit": 0, "bench": {"median_ttft_ms": 128}, "transfers": {}}
            raise pb.RunStopped("engines not answering before shape_exactkv_10gbps_t1024_b2_o128", {dead: "refused"})
        orig = {k: getattr(pb, k) for k in ("stage_shapes", "stage_sweep", "stage_report")}
        self.addCleanup(lambda: [setattr(pb, k, v) for k, v in orig.items()])
        pb.stage_shapes = shapes
        pb.stage_sweep = lambda a, res, colocated=False: calls.append("sweep")
        pb.stage_report = lambda a, res: calls.append(("report", pb.stopped_note(res["stopped"])))
        argv = sys.argv
        self.addCleanup(setattr, sys, "argv", argv)
        sys.argv = ["pd_bench.py", "--model", "m", "--out", str(out), "--prompts", str(out), "--log-dir", str(out),
                    "--link-control", str(out / "l.json"), "--stages", "shapes,sweep,report"]
        self.assertEqual(pb.main(), 3)
        self.assertEqual(calls[0][0], "report")
        self.assertNotIn("sweep", calls)
        self.assertIn("**The run stopped early** in its shapes stage", calls[0][1])
        self.assertIn(f"(not answering: {dead})", calls[0][1])
        res = json.loads((out / "results.json").read_text())
        self.assertEqual((res["stopped"]["stage"], list(res["shapes"])), ("shapes", ["shape_raw_10gbps_t512_b1_o128"]))
        # the combined report says which run stopped
        root = Path(tempfile.mkdtemp())
        (root / "r").mkdir()
        (root / "r" / "env.json").write_text(json.dumps({"layout": "1p3d", "connector": "tkv", "transport": "ring"}))
        (root / "r" / "results.json").write_text(json.dumps(res))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        self.assertIn("`r` stopped early in its shapes stage: engines not answering", (root / "s.md").read_text())

    def test_the_point_timeout_stops_a_benchmark_that_hangs(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        d = Path(tempfile.mkdtemp())
        fake = d / "vllm"
        fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
        fake.chmod(0o755)
        exe = sys.executable
        self.addCleanup(setattr, sys, "executable", exe)
        sys.executable = str(d / "python")           # bench_serve runs the vllm beside the interpreter
        a = SimpleNamespace(proxy="http://x", model="m", served_name="s", output_len=4, ttft_slo_ms=1, point_timeout=1)
        t = time.time()
        rc, r, t0, t1 = pb.bench_serve(a, "data.jsonl", "inf", 2, d, "p")
        self.assertEqual((rc, r), (124, {}))
        self.assertLess(time.time() - t, 20)
        self.assertIn("--point-timeout 1 s", (d / "p.log").read_text())

    def test_failed_transfers_marked_in_the_sweep(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_matrix_report as mr
        root = Path(tempfile.mkdtemp())
        (root / "r").mkdir()
        (root / "r" / "env.json").write_text(json.dumps({"layout": "1p3d", "connector": "tkv", "transport": "ring"}))
        (root / "r" / "results.json").write_text(json.dumps({"sweep": {"s": {
            "link_gbps": 10, "codec": "nvcomp-zstd", "rate": "inf", "bench": {"median_ttft_ms": 900},
            "transfers": {"sends": 8, "install_errors": 6, "bits": 12.0}}}}))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        self.assertIn("**6 of 8 transfers failed, computed locally**", (root / "s.md").read_text())

    def test_one_transfer_over_the_budget_stays_on_the_gpu_and_host_ones_are_marked(self):
        """Job 30371: raw's 1.88 GB at 16K went through host memory on both ends (budgets 1 and 1.5 GiB) while the
        coded transfers, just under 1.5 GiB, stayed on the decode GPU: their 1.3-1.4x over raw was mostly that."""
        install_vllm_stub()
        sys.path.insert(0, str(ROOT / "scripts"))
        from kv_compare.pd import connector as cn
        import pd_bench as pb
        import pd_matrix_report as mr
        G = 1 << 30
        self.assertTrue(cn._fits(0, 2 * G, G))                    # the only one: on the GPU
        self.assertTrue(cn._fits(G // 2, G // 4, G))
        self.assertFalse(cn._fits(G // 2, G, G))                  # beside others, over the budget: host memory
        self.assertTrue(cn._fits(5 * G, G, None))
        st = pb.transfer_stats([{"log": "tkv_Dw_1.jsonl", "event": "install", "link_on_host": True, "done_at": 2,
                                 "admitted_at": 1, "decode_install_ms": 3},
                                {"log": "tkv_Dw_1.jsonl", "event": "install", "done_at": 2, "admitted_at": 1,
                                 "decode_install_ms": 3}])
        self.assertEqual(st["recv_on_host"], 1)
        root = Path(tempfile.mkdtemp())
        (root / "r").mkdir()
        (root / "r" / "env.json").write_text(json.dumps({"layout": "1p3d", "connector": "tkv", "transport": "ring"}))
        sh = {f"shape_{c}": {"codec": c, "tokens": 16384, "batch": 1, "output": 128, "rounds": 4, "families": ["sz1"],
                             "link_gbps": 100, "requests": 4, "exit": 0, "bench": {"median_ttft_ms": t},
                             "transfers": {"recv_on_host": h, "spilled_to_host": 0}}
              for c, t, h in (("raw", 3743, 4), ("splitzip", 2730, 0))}
        (root / "r" / "results.json").write_text(json.dumps({"shapes": sh}))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        text = (root / "s.md").read_text()
        self.assertIn("| 3743 (host) | **2730** (1.37x) |", text)

    def test_the_profile_sorts_kernels_by_kind_and_repeats_a_short_document(self):
        from kv_compare.pd import profile as pf
        self.assertEqual(pf.kind("_ldec_kernel"), "our decoder")
        self.assertEqual(pf.kind("_lenc_kernel"), "our encoder")
        self.assertEqual(pf.kind("_encode_pages_kernel"), "exactkv")
        self.assertEqual(pf.kind("ampere_int8_gemm_64x64"), "products (int8 GEMM)")
        self.assertEqual(pf.kind("void at::native::vectorized_elementwise_kernel"), "copies and fills")
        self.assertEqual(pf.kind("_tail_copy"), "packing")

        class Doc:
            layers, tokens, columns = 2, 3, 2

            def words(self, l, dev, rows):
                return torch.arange(rows * 4, dtype=torch.int32).view(rows, 4) + 40000 * l
        w = pf.words_of(Doc(), 7, torch.device("cpu"))
        self.assertEqual(tuple(w.shape), (2, 7, 4))
        self.assertTrue(torch.equal(w[:, 3:6], w[:, 0:3]))
        self.assertEqual(int(w[1, 0, 0]), 40000 - 65536)          # a word over 32767 as int16
        calls = []
        wall, issue = pf._timed(lambda: calls.append(1), lambda: None, 3)
        self.assertEqual(len(calls), 4)                            # one to compile, three timed
        self.assertLessEqual(issue, wall + 1e-6)
        md = pf.markdown([{"codec": "ours-hbh", "tokens": 512, "bits": 8.6,
                           "encode": {"wall_ms": 13, "issue_ms": 4, "gpu_ms": 9, "launches": 140},
                           "decode": {"wall_ms": 62, "issue_ms": 58, "gpu_ms": 11, "launches": 1736,
                                      "kinds": {"our decoder": [448, 6.0], "copies and fills": [840, 3.0]}}}])
        self.assertIn("| ours-hbh | 512 | 13.0 / 4.0 / 9.0 | 140 | 62.0 / 58.0 / 11.0 | 1736 | 8.6 |", md)
        self.assertIn("ours-hbh, 512 tokens, decode: our decoder 448 x, 6.0 ms; copies and fills 840 x, 3.0 ms", md)

    def test_capacity_is_the_highest_rate_most_requests_meet_the_target(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_matrix_report as mr
        good = {"raw": {1.0: 1.0, 2.0: 1.95, 3.0: 2.5, 4.0: 1.0}, "ours-hbh": {1.0: 1.0, 2.0: 2.0, 3.0: 2.9, 4.0: 3.7}}
        sw = {f"s_{c}_{r}": {"link_gbps": 10, "codec": c, "rate": r, "bench": {
            "request_goodput": g, "request_throughput": r * 0.98, "median_ttft_ms": 300}, "transfers": {"bits": 8.6}}
            for c, gs in good.items() for r, g in gs.items()}
        root = Path(tempfile.mkdtemp())
        (root / "r").mkdir()
        (root / "r" / "env.json").write_text(json.dumps({"layout": "1p3d", "connector": "tkv", "transport": "ring"}))
        (root / "r" / "results.json").write_text(json.dumps({"sweep": sw}))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        text = (root / "s.md").read_text()
        self.assertIn("| raw | 2.0 | 3.92 |  |", text)          # 2.5 of 3 per second met the target: not 3
        self.assertIn("| ours-hbh | 4.0 | 3.92 | 2.00x |", text)
        # many link speeds: a capacity table across them
        sw = {f"s_{c}_{l}_{r}": dict(v, link_gbps=l) for l in (10, 25, 50, 100, 200)
              for c, gs in good.items() for r, v in ((r, {"codec": c, "rate": r, "bench": {
                  "request_goodput": g, "request_throughput": r, "median_ttft_ms": 300}, "transfers": {}})
                  for r, g in gs.items())}
        (root / "r" / "results.json").write_text(json.dumps({"sweep": sw}))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        text = (root / "s.md").read_text()
        self.assertIn("capacity per link speed", text)
        self.assertIn("| ours-hbh | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |", text)

    def test_wire_rate_and_link_speed_tables(self):
        """Job 30280: three transfers sharing a 10 Gb/s link each ran at about a third of it; the link carried 10."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        import pd_matrix_report as mr
        one = lambda b, e, log="tkv_Ps_1.jsonl": {"log": log, "send_begin": b, "send_end": e, "blob_bytes": 156_250_000}
        self.assertAlmostEqual(pb.wire_gbps([one(0, .375), one(0, .375), one(0, .375), one(1, 1.125)]), 10.0)
        # two prefill engines, two links: their rates add
        self.assertAlmostEqual(pb.wire_gbps([one(0, .125), one(0, .125, "tkv_Ps_2.jsonl")]), 20.0)
        self.assertIsNone(pb.wire_gbps([]))
        root = Path(tempfile.mkdtemp())
        (root / "r").mkdir()
        (root / "r" / "env.json").write_text(json.dumps({"layout": "1p3d", "connector": "tkv", "transport": "p2p"}))
        sw = {}
        for l in [10, 25, 50, 100, 200, 400, "native"]:
            g = 125.0 if l == "native" else float(l)
            for c, fixed in (("raw", 0), ("t2-hbh", 250)):
                sw[f"s_{c}_{l}"] = {"link_gbps": l, "codec": c, "rate": "inf",
                                    "bench": {"median_ttft_ms": 400 + fixed + (24000 if c == "raw" else 12000) / min(g, 125)},
                                    "transfers": {"bits": 16 if c == "raw" else 8.1, "wire_gbps": min(g, 125.0),
                                                  "install_ms": 150 if c != "raw" else 3,
                                                  "install_errors": 1 if (c, l) == ("t2-hbh", 400) else 0}}
        (root / "r" / "results.json").write_text(json.dumps({"sweep": sw}))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        text = (root / "s.md").read_text()
        self.assertIn("## 1p3d, GPU to GPU: link speed sweep, request rate inf", text)
        self.assertIn("| Codec | Bits | 10 | 25 | 50 | 100 | 200 | 400 | native |", text)
        self.assertIn("| *Carried (raw)* | | 10 | 25 | 50 | 100 | 125 | 125 | 125 |", text)
        self.assertIn("Faster than raw at: t2-hbh: 10, 25.", text)
        self.assertIn("(failed)", text)
        self.assertNotIn("link 10 Gb/s", text)                  # no table per speed when there are many

    def test_aquakv_is_reported_as_lossy(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_matrix_report as mr
        import pd_bench as pb
        self.assertTrue(pb.is_lossy("aquakv-2") and not pb.is_lossy("t2-hbh"))
        root = Path(tempfile.mkdtemp())
        (root / "r").mkdir()
        (root / "r" / "env.json").write_text(json.dumps({"layout": "1p3d", "connector": "tkv", "transport": "ring"}))
        st = {"installs": 4, "exact": 0, "bits": 2.13, "rel_err": 0.0311}
        sw = {f"s_{c}": {"link_gbps": 10, "codec": c, "rate": "inf", "bench": {"median_ttft_ms": t},
                         "transfers": {"bits": b}} for c, t, b in (("raw", 2400, 16), ("aquakv-2", 1700, 2.13))}
        (root / "r" / "results.json").write_text(json.dumps({
            "check": {"aquakv-2": {"prompts": 4, "same_text_as_local": 2, "ok": True, "lossy": True, "transfers": st},
                      "raw": {"prompts": 4, "same_text_as_local": 4, "ok": True, "transfers": dict(st, exact=4)}},
            "sweep": sw}))
        mr.main(["--runs", "r", "--root", str(root), "--output", str(root / "s.md")])
        text = (root / "s.md").read_text()
        self.assertIn("| aquakv-2 (lossy) | lossy (KV error 3.11%) | 2/4 | yes |", text)
        self.assertIn("| aquakv-2 (lossy) | 2.130 |", text)
        self.assertIn("## Note on the lossy reference", text)

    def test_checkout_leaves_a_clone_at_its_commit_and_serialises_changes(self):
        """Job 30293: four engines ran `git checkout` on one clone at once and collided on git's index lock."""
        from kv_compare.more_baselines import checkout
        d = Path(tempfile.mkdtemp())
        src = d / "src"
        git = lambda *a, cwd=src: subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=cwd,
                                                 check=True, capture_output=True, text=True).stdout.strip()
        src.mkdir()
        git("init", "-q")
        (src / "f").write_text("1")
        git("add", "f")
        git("commit", "-q", "-m", "1")
        first = git("rev-parse", "HEAD")
        (src / "f").write_text("2")
        git("commit", "-q", "-am", "2")
        clone = d / "clone"
        checkout(str(src), first, clone, log=lambda *a: None)
        self.assertEqual((clone / "f").read_text(), "1")
        mtime = (clone / ".git" / "index").stat().st_mtime_ns
        checkout(str(src), first, clone, log=lambda *a: None)            # already there: nothing touched
        self.assertEqual((clone / ".git" / "index").stat().st_mtime_ns, mtime)
        second = git("rev-parse", "HEAD")
        code = ("import sys; sys.path.insert(0, sys.argv[1]); from kv_compare.more_baselines import checkout; "
                "checkout(sys.argv[2], sys.argv[3], sys.argv[4], log=lambda *a: None)")
        procs = [subprocess.Popen([sys.executable, "-c", code, str(ROOT), str(src), second, str(clone)],
                                  stderr=subprocess.PIPE, text=True) for _ in range(6)]
        errs = [(p.wait(), p.stderr.read()) for p in procs]
        self.assertEqual([e[0] for e in errs], [0] * 6, errs)
        self.assertEqual((clone / "f").read_text(), "2")

    def test_aquakv_hadamard_by_matrix_product(self):
        """On the GPU AQUA-KV's transform is x @ H with H = the butterfly applied to the identity: the same
        transform, but for rounding."""
        from kv_compare import baselines as bl
        H = bl.hadamard(torch.eye(1024, dtype=torch.float32))
        self.assertTrue(torch.equal(H, H.T))
        self.assertTrue(torch.equal(H.abs(), torch.ones_like(H)))
        x = torch.randn(64, 2, 1024, generator=torch.Generator().manual_seed(0))
        self.assertTrue(torch.allclose((x.reshape(-1, 1024) @ H).reshape(x.shape), bl.hadamard(x), rtol=0, atol=1e-4))

    def test_ibp_round_trip_with_a_stand_in(self):
        """Our IBP wrapper's store and load, over a stand-in for the library that keeps the data as it is."""
        from kv_compare.pd import codecs
        from unittest import mock
        calls = []
        fake = types.SimpleNamespace(
            preprocess=lambda d: (torch.zeros(d.shape[1], dtype=d.dtype), torch.zeros(d.shape[1], dtype=d.dtype)),
            get_compress_size=lambda d, m, b: torch.full((d.shape[0],), d.shape[1]),
            get_single_compress_len=lambda d, m, b: d.shape[1],
            compress_inplace=lambda h, m, b: calls.append(h.is_pinned() or True) or torch.zeros(1, dtype=torch.int32),
            decompress_fetch=lambda h, m, b, bm, dev, comp_len=None, output_tensor=None: output_tensor.copy_(h))
        with mock.patch.dict(sys.modules, {"ibp": fake}), mock.patch.object(torch.Tensor, "pin_memory", lambda t: t):
            c = codecs.IBP("cpu")
            words = torch.randint(-3000, 3000, (3, 16, 32), dtype=torch.int16)
            state, nbytes, coded = c.tier_store(words)
            back = torch.empty_like(words)
            c.tier_load(state, lambda l, w: back[l].copy_(w))
        self.assertTrue(torch.equal(back, words))
        self.assertEqual((len(calls), coded), (3, 3 * 16 * 8))
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_matrix_report as mr
        self.assertEqual(mr.moved({"bits": 16.125, "moved_bits": 12.86}), " held, 12.860 moved")
        self.assertEqual(mr.moved({"bits": 11.0, "moved_bits": 11.0}), "")

    def test_aquakv_index_packing(self):
        from kv_compare.pd import codecs as cd
        g = torch.Generator().manual_seed(1)
        for bits in (4, 6, 8):
            idx = torch.randint(0, 2 ** bits, (3, 2, 512), generator=g)
            b = cd._pack(idx, bits)
            self.assertEqual(b.dtype, torch.uint8)
            self.assertEqual(b.numel(), idx.numel() * bits // 8)
            self.assertTrue(torch.equal(cd._unpack(b, bits, idx.numel()).reshape(idx.shape), idx))

    def test_exactkv_checks_its_raw_rows_before_decoding(self):
        """Job 30334: an illegal memory access in exactkv's install killed every decode engine. The receiver now checks
        that every raw row a page names was sent (and the buffers' shapes) before the tile decoder reads them."""
        install_vllm_stub()
        from kv_compare.pd import codecs as cd
        from kv_compare.pd import connector as cn
        for x in (0, 1, 0x8000, 0xFFFF, 0x5A5A, 12345):
            self.assertEqual(int(cd._popc16(torch.tensor([x]))[0]), bin(x).count("1"))
        x = cd.ExactKV.__new__(cd.ExactKV)
        x.P, x.hd = 16, 128
        Tp, H, NP = 32, 2, 2

        def parts(raw_rows=5):
            p = {}
            for t in ("0k", "0v"):
                p["sm" + t] = torch.zeros(1, Tp, H, 128, dtype=torch.uint8)
                p["code" + t] = torch.zeros(1, Tp, H, 64, dtype=torch.uint8)
                p["base" + t] = torch.zeros(1, NP, H, 128, dtype=torch.uint8)
                rm = torch.zeros(1, NP, H, dtype=torch.int32)
                ro = torch.zeros(1, NP, H, dtype=torch.int32)
                rm[0, 0, 0], ro[0, 0, 0] = 0b101, 0          # rows 0 and 2 of page 0, head 0: raw rows 0, 1
                rm[0, 1, 1], ro[0, 1, 1] = 0b1110, 2         # three in page 1, head 1: raw rows 2..4
                ro[0, 1, 0] = 99                             # a page with no raw row: its offset is not read
                p["rowmask" + t], p["rowoff" + t] = rm, ro
                p["raw" + t] = torch.zeros(raw_rows, 128, dtype=torch.bfloat16)
            return p
        x.check(parts(), 1, Tp, H)
        with self.assertRaisesRegex(ValueError, "0k names raw rows up to 5, 4 sent"):
            x.check(parts(4), 1, Tp, H)
        p = parts()
        p["rowmask0v"][0, 0, 1] = 1 << 16
        with self.assertRaisesRegex(ValueError, "0v names raw rows"):
            x.check(p, 1, Tp, H)
        p = parts()
        p["code0k"] = torch.zeros(1, Tp, H, 128, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, r"code0k is \(1, 32, 2, 128\)"):
            x.check(p, 1, Tp, H)
        # a CUDA error that poisons the GPU: not retried, its own message kept
        rec = {}
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        with self.assertRaisesRegex(RuntimeError, "illegal memory access"):
            cn._retry_once(boom, SimpleNamespace(type="cuda"), rec)
        self.assertEqual((calls, rec), ([1], {}))

    def test_ibp_left_out_says_why(self):
        from kv_compare.pd import codecs
        from unittest import mock
        with mock.patch.dict(os.environ, {"IBP_SKIP_REASON": "KVPD_IBP=0 in the login shell"}), \
                mock.patch.dict(sys.modules, {"ibp": None}):
            with self.assertRaisesRegex(ImportError, "IBP left out: KVPD_IBP=0 in the login shell"):
                codecs.IBP("cpu")
        home = Path(tempfile.mkdtemp())                       # the setup's build failed: its reason, from KV_HOME
        (home / "ibp_unavailable.txt").write_text("the CUDA 13 toolkit did not install (setup: cuda13_installer.log)\n")
        env = {k: v for k, v in os.environ.items() if k != "IBP_SKIP_REASON"}
        with mock.patch.dict(os.environ, dict(env, KV_HOME=str(home)), clear=True), \
                mock.patch.dict(sys.modules, {"ibp": None}):
            with self.assertRaisesRegex(ImportError, "IBP left out: the CUDA 13 toolkit did not install"):
                codecs.IBP("cpu")

    def test_compiles_placed_by_log_position(self):
        """Job 30252: a compile in the warm-up, logged in the same whole second as the point's start, was counted
        inside the point; the log's position at the point's start and end places it."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        d = Path(tempfile.mkdtemp())
        f = d / "engine_decode2.log"
        warn = "(EngineCore pid=1) WARNING 11-15 17:36:50 [jit_monitor.py:140] Triton kernel JIT compilation " \
               "during inference: _topp_sb_mask_kernel. This causes a latency spike\n"
        f.write_text("start\n" + warn)
        a = SimpleNamespace(engine_logs=str(d), out=d, rerun=False)
        m0 = pb.log_marks(a)
        f.write_text(f.read_text() + "the point's requests\n" + warn.replace("_topp_sb_mask", "_other"))
        m1 = pb.log_marks(a)
        t = time.mktime((time.localtime().tm_year, 11, 15, 17, 36, 50, 0, 0, -1))
        res = {"sweep": {"p": {"t0": t + 0.9, "t1": t + 12, "logs0": m0, "logs1": m1}}}
        from unittest import mock
        kernel = lambda j: j["text"].split("inference: ")[1].split(".")[0]
        with mock.patch.object(pb, "save", lambda *x: None):
            jit = pb.jit_report(a, res)
            self.assertEqual([(kernel(j), j["point"]) for j in jit],
                             [("_topp_sb_mask_kernel", None), ("_other_kernel", "p")])
            # without the positions (older results): the time, with its one-second slack
            res = {"sweep": {"p": {"t0": t + 0.9, "t1": t + 12}}}
            self.assertEqual([j["point"] for j in pb.jit_report(a, res)], ["p", "p"])


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
    if os.environ.get("FAKE_LOG"):
        with open(os.environ["FAKE_LOG"] + ".bench", "a") as f:
            f.write(json.dumps(args) + "\n")
    json.dump({"completed": 8, "request_throughput": 1.5, "request_goodput": 1.0, "median_ttft_ms": 500.0,
               "p99_ttft_ms": 900.0, "median_tpot_ms": 20.0}, open(os.path.join(d, n), "w"))
    sys.exit(0)
assert args[0] == "serve", args
port = int(args[args.index("--port") + 1])
if os.environ.get("FAKE_DIE"):
    print("INFO starting")
    print("Traceback (most recent call last):")
    print('  File "core.py", line 1, in <module>')
    print("ValueError: the root cause of the stand-in")
    print("Traceback (most recent call last):")
    print("RuntimeError: Engine core initialization failed. See root cause above.")
    sys.exit(1)
cfg = json.loads(args[args.index("--kv-transfer-config") + 1])
log = os.environ.get("FAKE_LOG")
with open(log, "a") as f:
    f.write(json.dumps({"port": port, "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "cfg": cfg,
                        "argv": args, "env": {k: v for k, v in os.environ.items() if k.startswith(("LMCACHE", "VLLM_NIXL", "UCX_"))}}) + "\n")
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
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)   # a clone with every earlier result: 600 MB
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
        (stub / "nixl").mkdir()                    # NIXL's module as vLLM imports it: nixl._api.nixl_agent
        (stub / "nixl" / "__init__.py").write_text("")
        (stub / "nixl" / "_api.py").write_text("nixl_agent = object\n")
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

    def test_published_workloads(self):
        """SHAPES: the baselines' own workloads through the runner and the bench: their prompt files, the context the
        longest needs, B requests at once (vllm bench serve --max-concurrency), O tokens out, one point each."""
        shapes = "1024:2:256:2:sz1024,1024:2:256:3:sz16,20000:1:128:1:sz1"
        h = subprocess.run(["bash", "-c", f"printf %s '{shapes}' | cksum | cut -d' ' -f1"], capture_output=True,
                           text=True).stdout.strip()
        prompts = self.kv / "runs" / "llama32_3b" / "pd" / f"prompts_2048_check_1000_3000_5000_8192_shapes_{h}"
        prompts.mkdir(parents=True, exist_ok=True)
        for f in ("check.jsonl", "prompts_mixed.jsonl", "prompts_1024.jsonl", "prompts_20000.jsonl"):
            (prompts / f).write_text(json.dumps({"prompt": "hello", "output_tokens": 4}) + "\n")
        r, eng = self.run_pd(LAYOUT="1p3d", TRANSPORT="ring", CODECS="raw,ours-hbh", LINKS="10",
                             STAGES="check,shapes,report", SHAPES=shapes)
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("context 20224 tokens for the workloads", r.stdout)
        argv = next(e for e in eng if e["cfg"]["kv_role"] == "kv_producer")["argv"]
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "20224")
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_1p3d_tkv_ring_rtx*smoke*"))[-1]
        res = json.loads((out / "results.json").read_text())
        self.assertEqual(sorted(res["shapes"]), sorted(f"shape_{c}_10gbps_{s}" for c in ("raw", "ours-hbh")
                                                       for s in ("t1024_b2_o256", "t20000_b1_o128")))
        v = res["shapes"]["shape_ours-hbh_10gbps_t1024_b2_o256"]
        self.assertEqual((v["families"], v["rounds"], v["requests"]), (["sz1024", "sz16"], 3, 6))
        log = sorted(self.tmp.glob("engines_*.jsonl"))[-1]
        bench = [json.loads(l) for l in Path(str(log) + ".bench").read_text().splitlines()]
        b = next(x for x in bench if "shape_raw_10gbps_t1024_b2_o256.json" in x)
        self.assertEqual((b[b.index("--max-concurrency") + 1], b[b.index("--num-prompts") + 1],
                          b[b.index("--custom-output-len") + 1]), ("2", "6", "256"))
        self.assertTrue(b[b.index("--dataset-path") + 1].endswith("prompts_1024.jsonl"))
        self.assertIn("## The published workloads", (out / "summary.md").read_text())
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_matrix_report as mr
        mr.main(["--runs", out.name, "--root", str(out.parent), "--output", str(self.tmp / "wl.md")])
        text = (self.tmp / "wl.md").read_text()
        self.assertIn("## Published workloads: 1p3d, host memory, link 10 Gb/s", text)
        self.assertIn("| SplitZip, 16 at once + SplitZip, 1,024 tokens | 1024 x 2, 256 |", text)

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
        # pieces for every codec, the encoder on a thread of its own; a link of its own per prefill engine
        self.assertEqual((ex["pipeline_tokens"], ex["async_encode"]), (4096, True))
        self.assertNotIn("link_shared", ex)
        d = next(e for e in eng if e["cfg"]["kv_role"] == "kv_consumer")["cfg"]["kv_connector_extra_config"]
        self.assertEqual((d["pipeline_tokens"], d["async_install"]), (4096, True))
        # a failed transfer is computed by the decode engine: under vLLM's default ("fail") it killed the engine
        self.assertTrue(all(e["cfg"]["kv_load_failure_policy"] == "recompute" for e in eng))
        self.assertIn("--host", p["argv"])
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_*smoke*"))
        self.assertEqual(len(out), 1, list((self.repo / "benchmark_results").glob("*")))
        env = json.loads((out[0] / "env.json").read_text())
        self.assertEqual((env["layout"], env["connector"], env["transport"], env["prefills"]), ("2p2d", "tkv", "p2p", 2))
        self.assertEqual((env["pipeline_tokens"], env["async_encode"], env["shared_link"], env["encode_once"],
                          env["decode_graphs"], env["group_product"]), (4096, True, False, True, True, "fused"))
        res = json.loads((out[0] / "results.json").read_text())
        self.assertIn("raw", res["check"])
        # three prefill engines behind one link: every one books its chunks in the same file
        r, eng = self.run_pd(LAYOUT="3p1d", TRANSPORT="ring", CODECS="raw,ours-hbh", SHARED_LINK="1",
                             PIPELINE_TOKENS="0", ASYNC_ENCODE="false")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        ps = [e["cfg"]["kv_connector_extra_config"] for e in eng if e["cfg"]["kv_role"] == "kv_producer"]
        self.assertEqual(len(ps), 3)
        self.assertEqual(len({x["link_shared"] for x in ps}), 1)
        self.assertTrue(ps[0]["link_shared"].endswith("link_shared.bin"))
        self.assertEqual({(x["pipeline_tokens"], x["async_encode"]) for x in ps}, {(0, False)})
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_3p1d_tkv_ring_*smoke*"))
        env = json.loads((out[-1] / "env.json").read_text())
        self.assertEqual((env["prefills"], env["decodes"], env["shared_link"]), (3, 1, True))
        # NIXL: one GPU each, side channels, the proxy's nixl mode through the bench
        r, eng = self.run_pd(LAYOUT="1p3d", CONNECTOR="nixl")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertEqual(sorted(e["cvd"] for e in eng), ["0", "1", "2", "3"])
        self.assertTrue(all(e["cfg"]["kv_connector"] == "NixlConnector" for e in eng))
        self.assertEqual(len({e["env"]["VLLM_NIXL_SIDE_CHANNEL_PORT"] for e in eng}), 4)
        self.assertTrue(all(e["env"]["UCX_PROTO_INFO"] == "y" for e in eng))       # UCX logs its transports
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_1p3d_nixl_rtx*"))
        res = json.loads((out[0] / "results.json").read_text())
        self.assertTrue(res["check"]["nixl"]["ok"])
        self.assertIn("sweep_nixl_nativegbps_rinf", res["sweep"])
        self.assertFalse(json.loads((out[0] / "env.json").read_text())["nixl_all_gpus"])
        # the sweeps greedy: vLLM 0.32's bench no longer sends temperature 0 by default
        bench = [json.loads(l) for l in Path(str(self.tmp / sorted(self.tmp.glob("engines_*.jsonl"))[-1].name)
                                                + ".bench").read_text().splitlines()]
        self.assertTrue(bench and all(b[b.index("--temperature") + 1] == "0" for b in bench))
        # NIXL with every GPU visible, as our p2p engines have
        r, eng = self.run_pd(LAYOUT="2p2d", CONNECTOR="nixl", NIXL_ALL_GPUS="1")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertEqual(sorted(e["cvd"] for e in eng), ["0,1,2,3", "1,0,2,3", "2,0,1,3", "3,0,1,2"])
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_2p2d_nixl_allgpus_*"))
        self.assertEqual(len(out), 1)
        self.assertTrue(json.loads((out[0] / "env.json").read_text())["nixl_all_gpus"])
        # the links matrix: 10 to 400 Gb/s in steps of 25, both paths
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="links", PRINT_RUNS="1"))
        runs = r.stdout.split("\n")
        self.assertIn("LINKS=10,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400 RATES=inf", runs[0])
        self.assertIn("TRANSPORT=p2p LINKS=10,25", runs[1])
        self.assertIn(",400,native", runs[1])
        self.assertTrue(all(x.endswith("CODECS=raw,ours-hbh,t2-hbh,t2-rank256,splitzip,exans,adacode,aquakv-2,exactkv")
                            for x in runs[:2]))
        # the workloads matrix: the published shapes in every main run, the 32K ones last on their own
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="workloads", PRINT_RUNS="1"))
        runs = [x for x in r.stdout.split("\n") if x]
        self.assertEqual(len(runs), 6)
        self.assertTrue(all("STAGES=check,sweep,shapes,report" in x for x in runs))
        # ours first; exactkv (its install killed every decode engine in job 30334) in runs of its own, last
        self.assertTrue(all("CODECS=raw,ours-hbh,t2-hbh,t2-rank256,splitzip,exans,adacode,aquakv-2,aquakv-3 "
                            in x for x in runs[:2]))
        self.assertIn("1920:20:128:2:ibp,4096:16:256:2:exactkv", runs[0])
        self.assertIn("1024:64:128:2:sz1024", runs[0])
        self.assertIn("CONNECTOR=nixl NIXL_ALL_GPUS=1", runs[2])
        self.assertIn("CODECS=raw,nvcomp-ans", runs[3])
        self.assertTrue(runs[4].startswith("TRANSPORT=ring LINKS=10,100 CODECS=raw,exactkv "))
        self.assertTrue(runs[5].startswith("TRANSPORT=p2p LINKS=native CODECS=raw,exactkv "))
        self.assertFalse(any("exactkv" in x.split("SHAPES=")[0] for x in runs[:4]))
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="wlong", PRINT_RUNS="1"))
        long = [x for x in r.stdout.split("\n") if x]
        self.assertEqual(len(long), 4)
        self.assertTrue(all("CODECS=raw,exactkv " in x for x in long[2:]))
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="wlours", PRINT_RUNS="1"))
        again = [x for x in r.stdout.split("\n") if x]
        self.assertEqual([x.split(" CODECS=")[0] for x in again], ["TRANSPORT=ring LINKS=10,100", "TRANSPORT=p2p LINKS=native"])
        self.assertTrue(all("CODECS=raw,ours-hbh,t2-hbh,t2-rank256,splitzip,adacode " in x and "SHAPES=" in x
                            for x in again))
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="capacity", PRINT_RUNS="1"))
        cap = [x for x in r.stdout.split("\n") if x]
        self.assertEqual([x.split(" CODECS=")[0] for x in cap],
                         ["TRANSPORT=ring LINKS=10,25,50,100", "TRANSPORT=p2p LINKS=native"])
        self.assertTrue(all("RATES=1,2,3,4,5,6,8 NUM_PROMPTS=96 TTFT_SLO_MS=1000" in x for x in cap))
        self.assertTrue(all("CODECS=raw,ours-hbh,t2-rank256,splitzip,adacode,exans,aquakv-2,exactkv " in x for x in cap))
        self.assertTrue(all("SHAPES=32768:1:128:4:sz1,32768:1:128:4:sz32k,32768:2:128:2:sz32k,32768:4:128:2:sz32k,"
                            "8192:16:128:2:sz16,16384:16:128:2:sz16,1024:128:128:2:sz1024,1024:256:128:2:sz1024" in x
                            for x in long))
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="xfer", PRINT_RUNS="1"))
        xf = [x for x in r.stdout.split("\n") if x]
        self.assertEqual([x.split(" CODECS=")[0] for x in xf],
                         ["TRANSPORT=ring LINKS=10,25,50,100", "TRANSPORT=ring LINKS=10,25,50,100",
                          "TRANSPORT=p2p LINKS=native", "TRANSPORT=ring LINKS=10,25"])
        self.assertTrue(all("65536:1:128:2:long" in x and "P_MEM=0.68 D_MEM=0.66" in x for x in (xf[0], xf[2], xf[3])))
        self.assertIn("LAYOUT=3p1d SHARED_LINK=1", xf[1])
        self.assertTrue(all("CODECS=raw,ours-hbh,ours-rank256,t2-hbh,t2-rank256,splitzip,adacode,exans " in x
                            for x in xf[:3]))
        self.assertIn("CODECS=raw,exactkv ", xf[3])
        r = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="quick", PRINT_RUNS="1"))
        qk = [x for x in r.stdout.split("\n") if x]
        self.assertEqual(qk, ["TRANSPORT=ring LINKS=10,100 CODECS=raw,ours-hbh,t2-hbh,t2-rank256,splitzip LAYOUT=1p3d "
                              "STAGES=check,shapes,report SHAPES=2048:1:128:4:sz1,16384:1:128:2:sz1,2048:16:128:1:sz16 "
                              "RUN_TIMEOUT_S=5400"])
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_bench as pb
        parsed = pb.parse_shapes(runs[0].split("SHAPES=")[1].split()[0])
        self.assertEqual(len(parsed), 19)                         # 21 entries; 2 shared by two families
        self.assertEqual(max(e["tokens"] * e["batch"] for e in parsed), 65536)
        # colocated with the CPU tier, and LMCache
        r, eng = self.run_pd(LAYOUT="colo4", TIER_CODECS="raw,ours-hbh")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue(all(e["cfg"]["kv_role"] == "kv_both" for e in eng))
        tier = eng[0]["cfg"]["kv_connector_extra_config"]["tier"]
        self.assertEqual((tier["chunk"], tier["budget_gb"]), (256, 8))
        self.assertIn("--num-gpu-blocks-override", eng[0]["argv"])
        argv = eng[0]["argv"]
        self.assertEqual(argv[argv.index("--num-gpu-blocks-override") + 1], "300")     # the smoke's tier sizing
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "4096")
        self.assertTrue(all(e["cfg"]["kv_load_failure_policy"] == "recompute" for e in eng))
        r, eng = self.run_pd(LAYOUT="colo4", CONNECTOR="lmcache")
        self.assertIn("committed locally", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertTrue(all(e["cfg"]["kv_connector"] == "LMCacheConnectorV1Dynamic" and
                            e["cfg"]["kv_connector_module_path"] == "lmcache.integration.vllm.lmcache_connector_v1"
                            for e in eng))
        self.assertEqual(eng[0]["env"]["LMCACHE_MAX_LOCAL_CPU_SIZE"], "8")
        r, _ = self.run_pd(LAYOUT="colo4", CONNECTOR="nixl")
        self.assertIn("NIXL needs prefill and decode engines", r.stdout)

    def test_an_engine_that_dies_at_start_is_packaged(self):
        r, _ = self.run_pd(LAYOUT="1p3d", CODECS="raw", FAKE_DIE="1")
        self.assertIn("did not all come up", r.stdout, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("ValueError: the root cause of the stand-in", r.stdout)
        out = list((self.repo / "benchmark_results").glob("llama32_3b_pd_1p3d_tkv_ring_*_partial"))
        self.assertEqual(len(out), 1)
        self.assertIn("the root cause", (out[0] / "engine_prefill1.log").read_text())

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
        if os.environ.get("DEBUG_MATRIX"):
            print(r.stdout[-12000:], "\n---- stderr\n", r.stderr[-6000:])
        rep = list((self.repo / "benchmark_results").glob("pd_matrix_smoke_rtxa5000_*/summary.md"))
        self.assertEqual(len(rep), 1)
        text = rep[0].read_text()
        self.assertIn("2 runs:", text)
        self.assertIn("`LAYOUT=colo4 CONNECTOR=nixl`", text)
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--oneline", "-5"], capture_output=True,
                             text=True).stdout
        self.assertIn("PD matrix smoke: 3 runs (2 not finished)", log)   # the stand-ins log no transfers: run 1 fails its check


class BreakdownTests(unittest.TestCase):
    def test_a_point_breaks_down_from_its_logs(self):
        """scripts/pd_breakdown.py: a request joined across the proxy's and both engines' records by its key, its
        pieces counted once, matched to its point by time (the bench names the point's end in the node's local
        time; the logs are UNIX time, here 5 h ahead); another codec's request in the same window is not counted."""
        import datetime as dt
        import gzip
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_breakdown as bd
        d = Path(tempfile.mkdtemp())
        (d / "runs").mkdir()
        (d / "connector").mkdir()
        (d / "env.json").write_text(json.dumps({"transport": "ring", "layout": "1p3d"}))
        (d / "runs" / "shape_ours-hbh_10gbps_t8192_b1_o128.json").write_text(json.dumps(
            {"date": "20261118-120010", "duration": 10.0, "median_ttft_ms": 900.0}))
        t0 = dt.datetime(2026, 11, 18, 12, 0, 10, tzinfo=dt.timezone.utc).timestamp() + 5 * 3600 - 8

        def gz(path, recs):
            with gzip.open(path, "wt") as f:
                f.write("".join(json.dumps(r) + "\n" for r in recs))
        gz(d / "proxy.jsonl.gz", [{"key": "K", "codec": "ours-hbh", "t_in": t0, "t_first_byte": t0 + 0.9},
                                  {"key": "J", "codec": "raw", "t_in": t0 + 1, "t_first_byte": t0 + 1.2}])
        base = {"req_key": "K", "pieces": 2}
        gz(d / "connector" / "tkv_P_1.jsonl.gz", [
            {"event": "prefill_done", "key": "K", "time": t0 + 0.4},
            dict(base, event="send", key="K", piece=0, tokens=4096, picked_at=t0 + 0.41, gather_ms=5.0, encode_ms=50.0,
                 raw_bytes=4, coded_bytes=2, blob_bytes=2, queued=t0 + 0.47, send_begin=t0 + 0.47, send_end=t0 + 0.6),
            dict(base, event="send", key="K#1", piece=1, tokens=4096, gather_ms=0.0, encode_ms=50.0, raw_bytes=4,
                 coded_bytes=2, blob_bytes=2, queued=t0 + 0.52, send_begin=t0 + 0.6, send_end=t0 + 0.75),
            {"event": "send", "key": "J", "tokens": 2048, "picked_at": t0 + 1.1, "queued": t0 + 1.1,
             "send_begin": t0 + 1.1, "send_end": t0 + 1.15}])
        gz(d / "connector" / "tkv_D_2.jsonl.gz", [
            dict(base, event="install", key="K", piece=0, decode_install_ms=20.0, link_recv_end=t0 + 0.6,
                 taken_at=t0 + 0.61, done_at=t0 + 0.63, exact=True),
            dict(base, event="install", key="K#1", piece=1, decode_install_ms=20.0, link_recv_end=t0 + 0.75,
                 taken_at=t0 + 0.76, done_at=t0 + 0.78, exact=True),
            {"event": "install", "key": "J", "decode_install_ms": 3.0, "done_at": t0 + 1.17}])
        rows = bd.analyse(d)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        got = {k: round(r[k], 3) for k in ("ttft", "prefill", "encode", "link", "install", "first_token", "tokens",
                                           "pieces")}
        self.assertEqual(got, {"ttft": 900.0, "prefill": 400.0, "encode": 100.0, "link": 280.0, "install": 40.0,
                               "first_token": 120.0, "tokens": 8192, "pieces": 2})
        self.assertEqual((r["codec"], r["gbps"], r["n"], r["bench_ttft"]), ("ours-hbh", "10", 1, 900.0))
        self.assertIn("| shape t8192_b1_o128 | ours-hbh | 10 | 1 |", bd.markdown(rows))
