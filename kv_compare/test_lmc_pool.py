"""The KV pool experiment (docs/lmcache_lifecycle.md, "The KV pool experiment"): the fixture's check, the scheduler's
and MultiConnector's trace records (the real extractors on stand-ins of vLLM's objects), the pool report on a
synthetic arm (a third request waits for GPU capacity on P until NIXL lets P free the first), the driver against fake
engines through the real proxy, the matrix preset and the campaign's disk plan for it."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kv_compare"))

from kvtrace import events  # noqa: E402
import lmc_lifecycle  # noqa: E402
import lmc_pool  # noqa: E402
import test_lmc_lifecycle as tl  # noqa: E402

FIXTURE = ROOT / "workloads" / "kv-pool-concurrency-chat-v1.json"


def fixture():
    return json.loads(FIXTURE.read_text())


class FixtureTests(unittest.TestCase):
    def test_the_fixture_is_what_the_plan_says(self):
        res, problems = lmc_pool.check_fixture(fixture())
        self.assertEqual(problems, [])
        self.assertEqual((res["conversations"], res["prompts"], res["prompt_tokens"]), (6, 18, 1032))
        self.assertEqual(res["blocks_per_request"], [65])                   # 1,032 + 8 tokens, 16 a block
        self.assertEqual(set(res["stable_prefix_tokens"].values()), {288})  # 18 blocks, 1 LMCache chunk
        self.assertEqual(set(res["stable_prefix_chunks"].values()), {1})
        self.assertEqual(res["longest_prefix_between_conversations"], 1)    # no shared block between them

    def test_a_changed_prompt_or_a_shared_first_block_is_found(self):
        fx = fixture()
        fx["conversations"][0]["layouts"]["chat"][1]["token_ids"][500] += 1
        fx["conversations"][1]["layouts"]["chat"][0]["token_ids"][:16] = \
            fx["conversations"][2]["layouts"]["chat"][0]["token_ids"][:16]
        _res, problems = lmc_pool.check_fixture(fx)
        self.assertTrue(any("alder-17 turn 2: the ids' SHA-256" in p for p in problems), problems)
        self.assertTrue(any("sharing a first block" in p or "different first blocks" in p for p in problems))

    def test_the_tokenizer_check(self):
        """With a model folder: tokenizer.json's hash, and every prompt's text tokenized again (a stand-in
        tokenizers module here: the real one runs on the node, scripts/lmc_setup.sh and scripts/lmc_run.sh)."""
        fx = fixture()
        d = Path(tempfile.mkdtemp())
        (d / "tokenizer.json").write_text("{}")
        fx["tokenizer_sha256"] = __import__("hashlib").sha256(b"{}").hexdigest()
        by_text = {t["prompt_text"]: t["token_ids"] for c in fx["conversations"] for t in c["layouts"]["chat"]}
        bad_text = fx["conversations"][3]["layouts"]["chat"][2]["prompt_text"]

        class Tok:
            @staticmethod
            def from_file(path):
                return Tok()

            def encode(self, text, add_special_tokens=True):
                assert add_special_tokens is False
                ids = list(by_text[text])
                if text == bad_text:
                    ids[700] = 0
                return types.SimpleNamespace(ids=ids)
        sys.modules["tokenizers"] = types.SimpleNamespace(Tokenizer=Tok)
        self.addCleanup(sys.modules.pop, "tokenizers", None)
        res, problems = lmc_pool.check_fixture(fx, d)
        self.assertTrue(res["tokenizer"]["sha256_matches"])
        self.assertEqual(res["tokenizer"]["prompts_tokenized_the_same"], 17)
        self.assertEqual(len(problems), 1)
        self.assertIn("dune-43:T3", problems[0])
        fx["tokenizer_sha256"] = "0" * 64
        _res, problems = lmc_pool.check_fixture(fx, d)
        self.assertTrue(any("tokenizer.json's SHA-256" in p for p in problems))

    def test_kv_bytes(self):
        d = Path(tempfile.mkdtemp())
        (d / "config.json").write_text(json.dumps({"num_hidden_layers": 36, "num_attention_heads": 32,
                                                   "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2560,
                                                   "torch_dtype": "bfloat16"}))
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "lmc_pool.py"), "kv-bytes", "--config",
                            str(d / "config.json")], capture_output=True, text=True)
        self.assertEqual(r.stdout.strip(), "147456")                        # Qwen3-4B: 2,359,296 B a block
        self.assertEqual(147456 * 16 * 131, 309_067_776)                    # the plan's 2-request pool


# ------------------------------------------------------------------------------------- stand-ins of vLLM's
class St:
    def __init__(self, name):
        self.name = name


class PReq(tl.Req):
    def __init__(self, rid, n=1032, status="WAITING"):
        super().__init__(rid, n, [])
        self.status = St(status)
        self.finished = False

    def is_finished(self):
        return self.finished


class PBlock:
    def __init__(self, i):
        self.block_id, self.is_null, self.ref_cnt, self.block_hash = i, i == 0, 0, None


class Pool:
    def __init__(self, n):
        self.num_gpu_blocks, self.blocks = n, [PBlock(i) for i in range(n)]

    def get_num_free_blocks(self):
        return sum(1 for b in self.blocks if not b.is_null and b.ref_cnt == 0)

    def take(self, k, cached_first=False):
        """k free blocks referenced (the empty ones first, as a new pool's order)."""
        got = [b for b in self.blocks if not b.is_null and b.ref_cnt == 0][:k]
        for b in got:
            b.ref_cnt, b.block_hash = 1, b"h%d" % b.block_id
        return [b.block_id for b in got]

    def free(self, ids):
        for i in ids:
            self.blocks[i].ref_cnt = 0                    # still cached (its hash kept)


class Sched:
    def __init__(self, n_blocks):
        self.running, self.waiting, self.kv_holding_waiting, self.requests = [], [], [], {}
        self.current_step = 0
        self.kv_cache_manager = types.SimpleNamespace(block_pool=Pool(n_blocks))


def step(s, sim, role, t, sched=None):
    s.current_step += 1
    sim.hook(role, t, "vllm.sched.step", scheduler=s, output=types.SimpleNamespace(num_scheduled_tokens=sched or {}),
             token_budget=6240)


class NixlConnector:
    pass


class LMCacheMPConnector:
    pass


def reset_events():
    events._WAIT.clear()
    events._WAIT_STOP.clear()
    events._STEP_LAST[0] = None
    events._MATCH.clear()
    events._CHOSEN.clear()


def simulate_pool(d, servers="shared", mode="memory"):
    """One turn of three conversations offered together, a 131-block pool (2 requests fit): on P, A and B are
    admitted, C waits with no free blocks; D1 reads A over NIXL, P frees A's blocks, C is admitted (A's blocks); D1
    has room for all three. P's LMCache lookups find nothing; on D, NIXL loads."""
    reset_events()
    s = tl.Sim(d)
    fx = fixture()
    convs = fx["conversations"][:3]
    run = f"lmc_{mode}_nixl_{servers}_blocks131_n3"
    keys = [f"{i:x}" * 32 for i in (10, 11, 12)]
    labels = [lmc_pool.label_of(run, c["conversation_id"], 1) for c in convs]
    prid = [f"cmpl-{k}.{lb}.P-0-aaaa000{i}" for i, (k, lb) in enumerate(zip(keys, labels))]
    drid = [f"cmpl-{k}.{lb}.D-0-bbbb000{i}" for i, (k, lb) in enumerate(zip(keys, labels))]
    P, D = Sched(131), Sched(131)
    pr = [PReq(r) for r in prid]
    dr = [PReq(r, status="WAITING") for r in drid]
    t = 100.0
    nixl, lmc = NixlConnector(), LMCacheMPConnector()
    # P: all three arrive (5 ms apart); A and B admitted, C has no free blocks
    for i, r in enumerate(pr):
        s.hook("P", t + .005 * i, "vllm.request.added", request=r)
        P.waiting.append(r)
        s.hook("P", t + .005 * i + .001, "vllm.gpu.prefix_hit", request=r, tokens=0, blocks=tl.Blocks([]))
        s.hook("P", t + .005 * i + .002, "lmc.conn.lookup", request=r, vllm_hit=0, lmc_hit=0, load=0)
        s.hook("P", t + .005 * i + .002, "kv.connector.match", request=r, index=0, connector=nixl, tokens=0)
        s.hook("P", t + .005 * i + .002, "kv.connector.match", request=r, index=1, connector=lmc, tokens=0)
        s.hook("P", t + .005 * i + .002, "kv.connector.chosen", request=r, connectors=[nixl, lmc], chosen=None,
               result=(0, False), computed=0)
    ta = t + .012
    blocks = {}
    for i in (0, 1):
        blocks[prid[i]] = P.kv_cache_manager.block_pool.take(65)
        s.hook("P", ta, "vllm.gpu.allocated", request=pr[i], new_blocks=([tl.Blk(b) for b in blocks[prid[i]]],),
               num_new_tokens=1032, num_new_computed_tokens=0, num_external_computed_tokens=0, delay=False)
        P.waiting.remove(pr[i])
        P.running.append(pr[i])
    s.hook("P", ta, "vllm.sched.wait", request=pr[2], reason="no_free_blocks", stop=True,
           free=P.kv_cache_manager.block_pool.get_num_free_blocks(), hit=0, external=0, new=1032)
    step(P, s, "P", ta, {prid[0]: 1032, prid[1]: 1032})
    # A and B prefilled, finished, held for NIXL (blocks still referenced); C still waits
    tf = t + .090
    for i in (0, 1):
        P.running.remove(pr[i])
        pr[i].finished = True
        P.requests[prid[i]] = pr[i]
        s.hook("P", tf, "nixl.request_finished", request=pr[i], delay=True, blocks=(blocks[prid[i]],), num_tokens=1032,
               is_p=True)
        s.hook("P", tf, "vllm.request.finished", request=pr[i], delay=True)
    s.hook("P", tf, "vllm.sched.wait", request=pr[2], reason="no_free_blocks", stop=True, free=0, hit=0, external=0,
           new=1032)
    step(P, s, "P", tf)
    # D1: A and B arrive, NIXL claims them, admitted (loading), read
    dblocks = {}
    for i in (0, 1):
        td = tf + .005 + .002 * i
        dr[i].kv_transfer_params = {"do_remote_prefill": True, "remote_block_ids": [blocks[prid[i]]],
                                    "remote_engine_id": "engP", "remote_request_id": prid[i]}
        s.hook("D1", td, "vllm.request.added", request=dr[i])
        s.hook("D1", td, "vllm.gpu.prefix_hit", request=dr[i], tokens=0, blocks=tl.Blocks([]))
        s.hook("D1", td + .001, "nixl.match", request=dr[i], count=1032, computed=0)
        s.hook("D1", td + .001, "kv.connector.match", request=dr[i], index=0, connector=nixl, tokens=1032)
        s.hook("D1", td + .001, "kv.connector.match", request=dr[i], index=1, connector=lmc,
               tokens=256 if servers == "shared" else 0)
        s.hook("D1", td + .001, "kv.connector.chosen", request=dr[i], connectors=[nixl, lmc], chosen=0,
               result=(1032, True), computed=0)
        dblocks[drid[i]] = D.kv_cache_manager.block_pool.take(65)
        s.hook("D1", td + .002, "vllm.gpu.allocated", request=dr[i],
               new_blocks=([tl.Blk(b) for b in dblocks[drid[i]]],), num_new_tokens=0, num_new_computed_tokens=0,
               num_external_computed_tokens=1032, delay=True)
        dr[i].status = St("WAITING_FOR_REMOTE_KVS")
        D.kv_holding_waiting.append(dr[i])
        s.hook("D1", td + .003, "nixl.read", req_id=drid[i], local=(dblocks[drid[i]],), remote=(blocks[prid[i]],),
               engine="engP")
    s.hook("D1", tf + .010, "vllm.sched.wait", request=dr[0], reason="blocked")
    s.hook("D1", tf + .010, "vllm.sched.wait", request=dr[1], reason="blocked")
    step(D, s, "D1", tf + .010)
    # A's read done (300 ms): D1 runs A; P frees A, admits C into A's blocks
    tr = tf + .310
    s.hook("D1", tr, "nixl.done", sending=set(), recving={drid[0]}, failed=set())
    s.hook("P", tr + .001, "nixl.done", sending={prid[0]}, recving=set(), failed=set())
    P.kv_cache_manager.block_pool.free(blocks[prid[0]])
    del P.requests[prid[0]]
    s.hook("P", tr + .001, "vllm.request.blocks_freed", manager=types.SimpleNamespace(
        get_block_ids=lambda rid: (blocks[prid[0]],)), request=pr[0])
    step(P, s, "P", tr + .0015)                       # A's 65 blocks free and cached; C not yet tried
    blocks[prid[2]] = P.kv_cache_manager.block_pool.take(65)
    s.hook("P", tr + .002, "vllm.gpu.allocated", request=pr[2], new_blocks=([tl.Blk(b) for b in blocks[prid[2]]],),
           num_new_tokens=1032, num_new_computed_tokens=0, num_external_computed_tokens=0, delay=False)
    for b in blocks[prid[2]]:
        s.hook("P", tr + .002, "vllm.gpu.evicted", block=tl.Blk(b), hashes=[b"h%d" % b + b"\0" * 30])
    P.waiting.remove(pr[2])
    P.running.append(pr[2])
    step(P, s, "P", tr + .002, {prid[2]: 1032})
    D.kv_holding_waiting.remove(dr[0])
    dr[0].status = St("RUNNING")
    D.running.append(dr[0])
    step(D, s, "D1", tr + .003, {drid[0]: 1})
    for k in range(3):                                # identical decode steps: one record
        step(D, s, "D1", tr + .004 + .001 * k, {drid[0]: 1})
    # B's read, C on P then D: done in order (no further waits)
    tb = tr + .300
    s.hook("D1", tb, "nixl.done", sending=set(), recving={drid[1]}, failed=set())
    s.hook("P", tb + .001, "nixl.done", sending={prid[1]}, recving=set(), failed=set())
    P.kv_cache_manager.block_pool.free(blocks[prid[1]])
    del P.requests[prid[1]]
    tc = tr + .080
    P.running.remove(pr[2])
    pr[2].finished = True
    P.requests[prid[2]] = pr[2]
    s.hook("P", tc, "nixl.request_finished", request=pr[2], delay=True, blocks=(blocks[prid[2]],), num_tokens=1032,
           is_p=True)
    s.hook("D1", tc + .005, "vllm.request.added", request=dr[2])
    s.hook("D1", tc + .006, "kv.connector.chosen", request=dr[2], connectors=[nixl, lmc], chosen=0,
           result=(1032, True), computed=0)
    dblocks[drid[2]] = D.kv_cache_manager.block_pool.take(65)
    s.hook("D1", tc + .007, "vllm.gpu.allocated", request=dr[2], new_blocks=([tl.Blk(b) for b in dblocks[drid[2]]],),
           num_new_tokens=0, num_new_computed_tokens=0, num_external_computed_tokens=1032, delay=True)
    s.hook("D1", tc + .008, "nixl.read", req_id=drid[2], local=(dblocks[drid[2]],), remote=(blocks[prid[2]],),
           engine="engP")
    dr[2].status = St("WAITING_FOR_REMOTE_KVS")
    D.kv_holding_waiting.append(dr[2])
    s.hook("D1", tc + .009, "vllm.sched.wait", request=dr[2], reason="blocked")
    s.hook("D1", tc + .009, "vllm.sched.wait", request=dr[1], reason="blocked")
    step(D, s, "D1", tc + .009, {drid[0]: 1})
    s.hook("D1", tb + .200, "nixl.done", sending=set(), recving={drid[2]}, failed=set())
    s.hook("P", tb + .201, "nixl.done", sending={prid[2]}, recving=set(), failed=set())
    for i in range(3):
        s.hook("D1", tb + .250 + .01 * i, "vllm.request.finished", request=dr[i], delay=False)
    step(P, s, "P", tb + .3)
    step(D, s, "D1", tb + .3)
    # the proxy's and the driver's records
    px, drv = [], []
    for i, c in enumerate(convs):
        row = c["layouts"]["chat"][0]
        tag = f"{run}|{i}|0"
        px.append({"key": keys[i], "dest": 0, "prefill": 0, "conv": f"{run}:{c['conversation_id']}", "tag": tag,
                   "rid": labels[i], "t_in": t + .005 * i - .001, "t_prefill_done": tf})
        drv.append({"tag": tag, "run": run, "layout": "chat", "conv": i, "conv_id": c["conversation_id"], "turn": 0,
                    "turns": 3, "prompt_tokens": 1032, "pieces": [], "ids": row["token_ids"], "order": i, "wave": 0,
                    "request_key": row["request_key"], "label": labels[i], "prompt_sha256_ok": True,
                    "reference_answer": row["reference_answer"], "answer_text": " " + row["reference_answer"],
                    "answer_ids": [1, 2, 3, 4, 5, 6, 7, 8], "answer_tokens": 8, "answer_ok": True,
                    "t_send": t + .005 * i - .002, "t_first": tb + .2 + .01 * i, "t_done": tb + .26 + .01 * i,
                    "ttft": tb + .2 + .01 * i - (t + .005 * i - .002), "latency": .9})
    (d / "proxy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in px))
    (d / "mtrag.jsonl").write_text("".join(json.dumps(r) + "\n" for r in drv))
    (d / "mtrag_summary.json").write_text(json.dumps({"waves": [{"turn": 1, "t_start": t - .002, "t_end": tb + .3}]}))
    roles = ["LMC"] if servers == "shared" else ["LMC_P", "LMC_D1", "LMC_D2", "LMC_D3"]
    (d / "env.json").write_text(json.dumps({
        "mode": mode, "chunk": 256, "l1_gb": 8, "eviction": "LRU", "store_policy": "default", "mtrag_limit": 4,
        "layout": "history_first", "store_wait_ms": 0.0, "d_role": "kv_both", "commit": "x", "transfer": "nixl",
        "servers": servers, "server_roles": roles, "run": run, "max_tokens": 8, "gpu_blocks": "131",
        "engine_server": {e: (roles[0] if servers == "shared" else f"LMC_{e}") for e in ("P", "D1", "D2", "D3")},
        "routing": "pinned:D1", "workload": "pool", "model": "Qwen/Qwen3-4B-Instruct-2507",
        "kv_bytes_per_token": 147456, "max_num_seqs": 6, "max_model_len": 1040, "max_num_batched_tokens": 6240,
        "pool": {"count": 3, "decode": "D1", "gpu_blocks": 131, "usable_blocks": 130, "stagger_ms": 5}}))
    return run, convs


class SchedulerRecordTests(unittest.TestCase):
    def test_step_records_fold_identical_steps_and_name_reasons(self):
        d = Path(tempfile.mkdtemp())
        simulate_pool(d)
        recs, _ = lmc_lifecycle.load(d)
        steps = [r for r in recs if r["ev"] == "vllm.sched.step"]
        p = [r for r in steps if r["role"] == "P"]
        d1 = [r for r in steps if r["role"] == "D1"]
        # P's first step: A and B scheduled, 130 of 130 blocks in use, C waiting for blocks
        self.assertEqual(p[0]["pool"], {"total": 130, "in_use": 130, "free": 0, "cached_free": 0, "empty": 0})
        self.assertEqual([w[2]["reason"] for w in p[0]["waiting"]], ["no_free_blocks"])
        self.assertEqual(p[0]["waiting"][0][2]["new"], 1032)
        # after P frees A: 65 free, all of them cached (a prefix hit until reused)
        freed = [r for r in p if r["pool"]["free"] == 65][0]
        self.assertEqual((freed["pool"]["cached_free"], freed["pool"]["empty"]), (65, 0))
        # D1: two loading (blocked: their NIXL reads in flight); the four identical decode steps are one record
        self.assertEqual([w[2]["reason"] for w in d1[0]["waiting"]], ["kv_load_in_flight", "kv_load_in_flight"])
        dec = [i for i, r in enumerate(d1) if list(r["scheduled"].values()) == [1]]
        self.assertEqual(d1[dec[0] + 1]["step"] - d1[dec[0]]["step"], 4)   # its 3 repeats folded into it
        chosen = [r for r in recs if r["ev"] == "kv.connector.chosen" and r["role"] == "D1"]
        self.assertEqual({r["chosen"] for r in chosen}, {"NixlConnector"})
        self.assertEqual([x[1] for x in chosen[0]["asked"]], ["NixlConnector", "LMCacheMPConnector"])
        self.assertTrue(all(r["chosen"] is None for r in recs if r["ev"] == "kv.connector.chosen" and r["role"] == "P"))

    def test_scheduler_off(self):
        reset_events()
        events.SCHED_ON = False
        self.addCleanup(setattr, events, "SCHED_ON", True)
        s = Sched(10)
        self.assertIsNone(events.sched_wait(PReq("r"), "no_free_blocks"))
        self.assertIsNone(events.sched_step(s, types.SimpleNamespace(num_scheduled_tokens={}), 1))

    def test_behind_a_stopped_queue_and_the_token_budget(self):
        reset_events()
        s = Sched(10)
        a, b = PReq("a"), PReq("b")
        s.waiting += [a, b]
        events.sched_wait(a, "no_free_blocks", stop=True)
        rec = events.sched_step(s, types.SimpleNamespace(num_scheduled_tokens={}), 100)
        self.assertEqual([w[2]["reason"] for w in rec["waiting"]], ["no_free_blocks", "queued_behind_no_free_blocks"])
        rec = events.sched_step(s, types.SimpleNamespace(num_scheduled_tokens={"x": 5}), 0)
        self.assertEqual([w[2]["reason"] for w in rec["waiting"]], ["token_budget", "token_budget"])
        self.assertIsNone(events.sched_step(s, types.SimpleNamespace(num_scheduled_tokens={"x": 5}), 0))
        pre = events.request_preempted(PReq("c"))
        self.assertEqual((pre["req"], pre["n_prompt"]), ("c", 1032))


class PoolReportTests(unittest.TestCase):
    def test_the_report_tells_the_boundary(self):
        run, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate_pool(run)
        lmc_lifecycle.write_report(run, out)                     # the lifecycle report reads a pool arm too
        self.assertTrue((out / "summary.md").is_file())
        self.assertEqual(lmc_pool.main(["report", "--run", str(run), "--output", str(out)]), 0)
        n = json.loads((out / "pool.json").read_text())
        self.assertEqual(n["checks"], {k: True for k in n["checks"]}, n["checks"])
        self.assertEqual((n["capacity_requests"], n["blocks_per_request"], n["usable_blocks"]), (2, 65, 130))
        P, D = n["engines"]["P"], n["engines"]["D1"]
        self.assertEqual(P["resident_peak"], 2)                   # never more than 2 on P: the pool's 130 blocks
        self.assertEqual(P["held_for_transfer_peak"], 2)
        self.assertEqual(D["resident_peak"], 3)
        self.assertGreater(P["wait_s"]["no_free_blocks"], 0.3)    # C, until A's read let P free its blocks
        self.assertEqual(P["requests_waiting"]["no_free_blocks"], 1)
        self.assertEqual(n["overlap"]["P_admitted_to_released"], 2)
        b = n["boundary"]
        self.assertEqual((b["request"], b["stage"], b["engine"]), ("cedar-34 T1", "P", "P"))
        self.assertEqual([r["request"] for r in b["released_before_admission"]], ["alder-17 T1"])
        self.assertEqual(n["gpu"]["evictions"]["P"], 65)          # C took A's cached blocks
        self.assertEqual(n["lmcache"]["loaded_by"], {"P": {"None": 3}, "D": {"NixlConnector": 3}})
        self.assertEqual(n["nixl"]["reads"], 3)
        rows = {r["conv"]: r for r in n["requests_rows"]}
        self.assertEqual(rows["alder-17"]["nixl_blocks"], 65)
        self.assertAlmostEqual(rows["alder-17"]["nixl_gb_s"], 65 * 16 * 147456 / 0.302 / 1e9, places=2)
        md = (out / "pool.md").read_text()
        self.assertIn("cedar-34 T1 arrived at P (P stage) and waited: GPU capacity", md)
        self.assertTrue((out / "summary.md").read_text().startswith("# KV pool arm"))
        # the arms side by side
        run2, out2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate_pool(run2, servers="per_engine", mode="disk")
        lmc_pool.main(["report", "--run", str(run2), "--output", str(out2)])
        cmp = Path(tempfile.mkdtemp())
        self.assertEqual(lmc_pool.main(["compare", "--dirs", f"{out},{out2}", "--output", str(cmp)]), 0)
        rows = json.loads((cmp / "pool_compare.json").read_text())
        self.assertEqual([(r["servers"], r["mode"]) for r in rows], [("per_engine", "disk"), ("shared", "memory")])
        self.assertTrue(all(r["P_peak"] == 2 and r["P_capacity_wait_s"] > 0.3 for r in rows))
        # and the lifecycle report's own comparison of the same arms (the matrix runs both)
        lmc_lifecycle.write_report(run2, out2)
        lmc_lifecycle.compare([str(out), str(out2)], str(cmp))
        self.assertTrue((cmp / "compare.md").is_file())

    def test_a_published_arm_reports_again(self):
        import gzip
        import shutil
        import tarfile
        run, res, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate_pool(run)
        with tarfile.open(res / "kvtrace.tar.gz", "w:gz") as tf:
            tf.add(run / "kvtrace", arcname="kvtrace")
        for f in ("proxy.jsonl", "mtrag.jsonl"):
            (res / f"{f}.gz").write_bytes(gzip.compress((run / f).read_bytes()))
        for f in ("env.json", "mtrag_summary.json"):
            shutil.copyfile(run / f, res / f)
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "lmc_pool.py"), "report", "--results",
                            str(res), "--output", str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        n = json.loads((out / "pool.json").read_text())
        self.assertTrue(all(n["checks"].values()), n["checks"])
        self.assertEqual(n["boundary"]["request"], "cedar-34 T1")


class _Engines:
    """Fake P and D1-D3 (aiohttp): P answers NIXL's prefill (kv_transfer_params), each D streams the request's
    reference answer as text and token ids; every body and header recorded with its arrival time."""

    def __init__(self, fx, d_delay=0.05):
        import asyncio
        import threading
        from aiohttp import web
        answers = {tuple(t["token_ids"]): t["reference_answer"] for c in fx["conversations"]
                   for t in c["layouts"]["chat"]}
        self.seen = {n: [] for n in ("p", "d1", "d2", "d3")}
        self.ports = {}
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        def handler(name):
            async def h(request):
                body = await request.json()
                self.seen[name].append((time.time(), dict(request.headers), body))
                if name == "p":
                    return web.json_response({"choices": [{"text": "x"}], "kv_transfer_params": {
                        "do_remote_prefill": True, "remote_block_ids": [[1]], "remote_engine_id": "e",
                        "remote_request_id": request.headers.get("X-Request-Id")}})
                await asyncio.sleep(d_delay)
                resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await resp.prepare(request)
                text = " " + answers.get(tuple(body["prompt"]), "?")
                for i, piece in enumerate((text[:4], text[4:])):
                    msg = {"choices": [{"text": piece, "token_ids": [100 + i]}]}
                    await resp.write(b"data: " + json.dumps(msg).encode() + b"\n\n")
                await resp.write(b"data: [DONE]\n\n")
                return resp
            return h

        def run():
            asyncio.set_event_loop(self.loop)

            async def start():
                for name in self.seen:
                    app = web.Application()
                    app.router.add_post("/v1/completions", handler(name))
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


@unittest.skipIf(importlib.util.find_spec("fastapi") is None, "fastapi / uvicorn / aiohttp needed")
class DriverTests(unittest.TestCase):
    def test_turns_together_then_the_next_all_on_d1(self):
        import urllib.request
        fx = fixture()
        eng = _Engines(fx)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        tmp = Path(tempfile.mkdtemp())
        log = tmp / "proxy.jsonl"
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "pd_proxy.py"), "--port", str(port),
                                 "--prefill", f"http://127.0.0.1:{eng.ports['p']}"]
                                + sum((["--decode", f"http://127.0.0.1:{eng.ports[x]}"] for x in ("d1", "d2", "d3")),
                                      []) + ["--mode", "nixl", "--policy", "round_robin", "--log", str(log)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: (proc.kill(), proc.wait(), proc.stderr.close()))
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        run = "lmc_memory_nixl_shared_blocks131_n3"
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "lmc_pool.py"), "run", "--fixture",
                            str(FIXTURE), "--count", "3", "--proxy", f"http://127.0.0.1:{port}", "--run", run,
                            "--decode", "0", "--stagger-ms", "20", "--max-tokens", "8", "--output",
                            str(tmp / "mtrag.jsonl"), "--summary", str(tmp / "mtrag_summary.json")],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        recs = [json.loads(x) for x in (tmp / "mtrag.jsonl").read_text().splitlines()]
        self.assertEqual([(x["turn"], x["conv_id"]) for x in recs],
                         [(t, c) for t in range(3) for c in ("alder-17", "beacon-26", "cedar-34")])
        self.assertTrue(all(x["answer_ok"] and x["prompt_sha256_ok"] and not x.get("error") for x in recs))
        # every turn's requests after every one of the turn before finished; in the fixture's order, 20 ms apart
        for t in (1, 2):
            self.assertGreaterEqual(min(x["t_send"] for x in recs if x["turn"] == t),
                                    max(x["t_done"] for x in recs if x["turn"] == t - 1))
        sends = [x["t_send"] for x in recs if x["turn"] == 0]
        self.assertEqual(sends, sorted(sends))
        self.assertGreater(sends[1] - sends[0], 0.015)
        # all on D1; P's and D's own ids, the proxy's key first; the driver's fields never reach an engine
        self.assertEqual((len(eng.seen["d1"]), len(eng.seen["d2"]), len(eng.seen["d3"])), (9, 0, 0))
        hp = [{k.lower(): v for k, v in h.items()}["x-request-id"] for _t, h, _b in eng.seen["p"]]
        hd = [{k.lower(): v for k, v in h.items()}["x-request-id"] for _t, h, _b in eng.seen["d1"]]
        self.assertTrue(all(x.endswith(".P") and lmc_lifecycle.rkey(x) == x[:32] for x in hp), hp)
        self.assertEqual(sorted(x[:-2] for x in hp), sorted(x[:-2] for x in hd))
        self.assertIn(f".{run}.alder-17.T1.P", " ".join(hp))
        for _t, _h, b in eng.seen["p"] + eng.seen["d1"]:
            self.assertFalse(any(k.startswith("t_") or k == "conversation_id" for k in b), b)
            self.assertTrue(b["ignore_eos"])
        self.assertTrue(all(b["max_tokens"] == 8 for _t, _h, b in eng.seen["d1"]))
        summ = json.loads((tmp / "mtrag_summary.json").read_text())
        self.assertEqual((summ["requests"], summ["errors"], summ["answers_ok"], len(summ["waves"])), (9, 0, 9, 3))
        px = [json.loads(x) for x in log.read_text().splitlines()]
        self.assertEqual({p["dest"] for p in px}, {0})
        self.assertEqual({p["tag"] for p in px}, {x["tag"] for x in recs})


class PresetTests(unittest.TestCase):
    def runs(self, **env):
        m = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="lmctrace", PRINT_RUNS="1", KVPD_LMC_PRESET="pool", **env))
        return m

    def test_the_72_arms_in_the_plans_order(self):
        m = self.runs()
        self.assertEqual(m.returncode, 0, m.stderr)
        runs = m.stdout.splitlines()
        self.assertEqual(len(runs), 72)                           # 6 offered counts x 3 pools x 4 LMCache settings
        want = [(n, b, sv, x) for n in range(1, 7) for b in (131, 261, 391) for sv in ("per_engine", "shared")
                for x in ("memory", "disk")]
        got = []
        for r in runs:
            kv = dict(p.split("=", 1) for p in r.split())
            got.append((int(kv["POOL_COUNT"]), int(kv["GPU_BLOCKS"]), kv["LMC_SERVERS"], kv["LMC_MODE"]))
            self.assertEqual({k: kv[k] for k in ("RUNNER", "LMC_TRANSFER", "LMC_WORKLOAD", "LMC_MODEL", "MAXLEN",
                                                  "MAXBT", "MAX_NUM_SEQS", "GPU_MEM", "MAX_TOKENS", "LMC_DISK_READS",
                                                  "LMC_LOCAL_CHECK", "POOL_DECODE", "POOL_STAGGER_MS")},
                             {"RUNNER": "lmc_run.sh", "LMC_TRANSFER": "nixl", "LMC_WORKLOAD": "pool",
                              "LMC_MODEL": "qwen3_4b", "MAXLEN": "1040", "MAXBT": "6240", "MAX_NUM_SEQS": "6",
                              "GPU_MEM": "0.75", "MAX_TOKENS": "8", "LMC_DISK_READS": "temporary",
                              "LMC_LOCAL_CHECK": "0", "POOL_DECODE": "0", "POOL_STAGGER_MS": "5"})
            self.assertEqual(kv["LMC_L1_GB"], "8" if kv["LMC_SERVERS"] == "shared" else "4")
        self.assertEqual(got, want)
        # the 6 x 65 + 1 blocks: 2, 4 and 6 requests and vLLM's reserved block
        self.assertEqual([(b - 1) // 65 for b in (131, 261, 391)], [2, 4, 6])
        self.assertEqual(math.ceil((1032 + 8) / 16), 65)

    def test_a_stage_of_it_and_what_it_refuses(self):
        m = self.runs(KVPD_LMC_POOL_COUNTS="1", KVPD_LMC_POOL_BLOCKS="131", KVPD_LMC_SERVERS="shared",
                      KVPD_LMC_MODES="memory")
        self.assertEqual(len(m.stdout.splitlines()), 1, m.stderr)
        self.assertEqual(len(self.runs(KVPD_LMC_POOL_COUNTS="1,2").stdout.splitlines()), 24)
        for k, v in (("KVPD_LMC_POOL_COUNTS", "7"), ("KVPD_LMC_POOL_BLOCKS", "65"), ("KVPD_LMC_MODES", "tiered"),
                     ("KVPD_LMC_TRANSFERS", "lmcache"), ("KVPD_LMC_POOL_DECODE", "3")):
            self.assertEqual(self.runs(**{k: v}).returncode, 2, k)

    def test_the_disk_plan_for_it(self):
        import pd_campaign
        runs = self.runs(KVPD_LMC_POOL_COUNTS="6").stdout.splitlines()
        # the largest arm on disk: 6 conversations, P's chunks and D1's with a server per engine; and the records
        self.assertAlmostEqual(pd_campaign.extra_gb("lmctrace", runs), 6 * 0.4 * 2.0 + 12 * 0.1)
        cache = Path(tempfile.mkdtemp())
        self.assertEqual(pd_campaign.model_gb(cache, runs), 8.2)             # Qwen3-4B not cached yet
        (cache / "hf" / "hub" / "models--Qwen--Qwen3-4B-Instruct-2507").mkdir(parents=True)
        self.assertEqual(pd_campaign.model_gb(cache, runs), 0)
        self.assertEqual(pd_campaign.model_gb(cache, ["RUNNER=lmc_run.sh LMC_MODE=disk MTRAG_LIMIT=4"]), 0)

    def test_the_run_script_takes_it(self):
        script = (ROOT / "scripts" / "lmc_run.sh").read_text()
        for s in ('WORKLOAD=${LMC_WORKLOAD:-mtrag}', 'qwen3_4b) MODEL_REPO=Qwen/Qwen3-4B-Instruct-2507 '
                  'MODEL_REV=cdbee75f17c01a7cc42f958dc650907174af0554', '--max-num-seqs "$MAX_NUM_SEQS"',
                  'scripts/lmc_pool.py check --fixture "$POOL_FIXTURE" --tokenizer "$MP"',
                  'scripts/lmc_pool.py report --run "$OUT" --output "$RES"', 'RES=$RESULTS_ROOT/${LMC_MODEL}_'):
            self.assertIn(s, script)
        r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / "lmc_run.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
