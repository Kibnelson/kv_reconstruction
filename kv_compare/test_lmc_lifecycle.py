"""The LMCache lifecycle trace (docs/lmcache_lifecycle.md): kvtrace and its records, the hooks in our copies of vLLM
and LMCache (third_party/), the overlay over the release wheels, the lifecycle report on a synthetic run (records
made by the real extractors from stand-ins of vLLM's and LMCache's objects), the proxy's store mode, and the
lmctrace matrix."""
from __future__ import annotations

import gzip
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from kvtrace import events  # noqa: E402
import lmc_lifecycle  # noqa: E402
import lmc_overlay  # noqa: E402

HOOK_RE = re.compile(r'kvtrace\.hook\("([^"]+)"(.*)\)\s+# KVTRACE$')


class KvtraceTests(unittest.TestCase):
    def run_py(self, code, env):
        return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT), **env),
                              capture_output=True, text=True, timeout=60)

    def test_off_writes_nothing(self):
        d = Path(tempfile.mkdtemp())
        env = {k: v for k, v in os.environ.items() if k != "KVTRACE_DIR"}
        r = subprocess.run([sys.executable, "-c", "import kvtrace; assert not kvtrace.ON; kvtrace.flush()"], cwd=d,
                           env=dict(env, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(list(d.iterdir()), [])

    def test_records_errors_and_a_forked_child(self):
        d = Path(tempfile.mkdtemp())
        code = (
            "import os, kvtrace\n"
            "assert kvtrace.ON\n"
            "kvtrace.hook('custom.thing', value=3, key=b'\\x01\\x02')\n"
            "for _ in range(25):\n"
            "    kvtrace.hook('vllm.gpu.evicted', block=None, hashes=[])\n"     # its extractor raises: counted
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    kvtrace.hook('custom.child', value=1); kvtrace.flush(); os._exit(0)\n"
            "os.waitpid(pid, 0)\n"
            "kvtrace.hook('custom.parent', value=2)\n")
        r = self.run_py(code, {"KVTRACE_DIR": str(d), "KVTRACE_ROLE": "P"})
        self.assertEqual(r.returncode, 0, r.stderr)
        files = sorted(d.glob("kvtrace_P_*.jsonl"))
        self.assertEqual(len(files), 2)                                  # the parent's and the child's own file
        recs = [json.loads(ln) for f in files for ln in f.read_text().splitlines()]
        by = {}
        for x in recs:
            by.setdefault(x["ev"], []).append(x)
        self.assertEqual(by["custom.thing"][0]["value"], 3)
        self.assertEqual(by["custom.thing"][0]["key"], "0102")
        self.assertEqual(len(by["kvtrace.error"]), 20)                     # at most ERR_MAX per event name
        self.assertEqual(by["kvtrace.error"][-1]["n"], 20)
        self.assertEqual({x["role"] for x in recs}, {"P"})
        self.assertEqual(len({by["custom.child"][0]["pid"], by["custom.parent"][0]["pid"]}), 2)
        child_file = [f for f in files if str(by["custom.child"][0]["pid"]) in f.name][0]
        self.assertNotIn("custom.thing", child_file.read_text())           # the parent's records stay the parent's


# ------------------------------------------------------------------------------------------------- stand-ins
class Blk:
    def __init__(self, i, null=False):
        self.block_id, self.is_null = i, null


class Blocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self):
        return (list(self.ids),)


class Req:
    def __init__(self, rid, n, hashes):
        self.request_id, self.num_prompt_tokens, self.block_hashes = rid, n, hashes
        self.num_output_tokens, self.num_tokens, self.num_computed_tokens = 0, n, 0
        self.kv_transfer_params, self.status = None, "RUNNING"


class Op:
    def __init__(self, start, end, blocks, skip=0):
        self.start, self.end, self.skip_first_n_tokens, self.flat_block_ids = start, end, skip, blocks


class Meta:
    def __init__(self, rid, op):
        self.request_id, self.op = rid, op


class EvType:
    def __init__(self, v):
        self.value = v


class Ev:
    def __init__(self, name, session="", **md):
        self.event_type, self.session_id, self.metadata = EvType(name), session, md


class ObjKey:
    def __init__(self, h):
        self.identity = types.SimpleNamespace(chunk_hash=h, kv_rank=0, object_group_id=0)


class Size:
    def __init__(self, n):
        self.size_bytes = n


def hb(i, n=32):
    return bytes([i]) * n


class Sim:
    """Records as the hooks would write them, at chosen times: the real extractors on stand-in objects."""

    def __init__(self, d):
        self.d = Path(d)
        (self.d / "kvtrace").mkdir(parents=True)
        self.files = {}

    def hook(self, role, t, name, /, **objs):
        fn = events.EXTRACT.get(name)
        rec = fn(**objs) if fn else events.generic(**objs)
        if rec is None:
            return
        rec["ev"] = rec.pop("_ev", name)
        if "_t" in rec:
            rec["t"], rec["t_hook"] = rec.pop("_t"), t
        else:
            rec["t"] = t
        rec["role"], rec["pid"] = role, {"P": 1, "D1": 2, "D2": 3, "D3": 4, "LMC": 5}[role]
        with open(self.d / "kvtrace" / f"kvtrace_{role}_{rec['pid']}.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=events_default) + "\n")


def events_default(o):
    import kvtrace
    return kvtrace._default(o)


def simulate(d, mode="disk", race=False, reads_hook=True):
    """One turn: P computes and stores a 69-token prompt (2 chunks of 32 tokens, 4 blocks of 16), D1 looks it up and
    pulls both chunks (from disk in disk mode); with race, D1 looks up before P's store has landed (a miss)."""
    s = Sim(d)
    key = "ab" * 16
    prid, drid = f"cmpl-{key}-0-11112222", f"cmpl-{key}-0-33334444"
    ids = list(range(1000, 1069))
    hashes = [hb(i + 1) for i in range(4)]
    c0, c1 = hb(0x40, 32), hb(0x41, 32)
    pr, dr = Req(prid, 69, hashes), Req(drid, 69, hashes)
    t = 100.0
    s.hook("P", t, "vllm.request.added", request=pr)
    s.hook("P", t + .001, "vllm.gpu.prefix_hit", request=pr, tokens=0, blocks=Blocks([]))
    s.hook("LMC", t + .0012, "lmc.bus", event=Ev("mp.lookup", prid, request_id=prid, chunk_hashes=[c0, c1],
                                                  model_name="m", chunk_size=32, seq_len=69, token_ids=ids))
    s.hook("LMC", t + .0015, "lmc.bus", event=Ev("mp.lookup_prefetch.end", prid, found_count=0, l1_hit_tokens=0,
                                                  l2_hit_tokens=0, hit_tokens=0))
    s.hook("P", t + .002, "lmc.conn.lookup", request=pr, vllm_hit=0, lmc_hit=0, load=0)
    s.hook("P", t + .003, "vllm.gpu.allocated", request=pr, new_blocks=([Blk(10), Blk(11), Blk(12), Blk(13), Blk(14)],),
           num_new_tokens=69, num_new_computed_tokens=0, num_external_computed_tokens=0, delay=False)
    s.hook("P", t + .004, "lmc.conn.alloc", request=pr, new_block_ids=[[10, 11, 12, 13, 14]], external=0,
           tracker=types.SimpleNamespace(state=types.SimpleNamespace(name="READY")))
    s.hook("P", t + .050, "vllm.gpu.cached", request=pr, blocks=[Blk(10), Blk(11), Blk(12), Blk(13)], first=0,
           hashes=hashes, group=0, mask=None, block_size=16)
    s.hook("P", t + .051, "lmc.conn.store_planned", meta=Meta(prid, Op(0, 64, [10, 11, 12, 13])))
    s.hook("P", t + .052, "lmc.conn.store_submitted", request_ids=[prid])
    s.hook("P", t + .060, "vllm.request.finished", request=pr, delay=True)
    s.hook("P", t + .061, "lmc.conn.request_finished", request=pr,
           tracker=types.SimpleNamespace(num_vllm_hit_tokens=0, num_lmcache_hit_tokens=0, num_stored_tokens=64),
           can_store=True)
    st = 0.5 if race else 0.0                                          # the store lands late
    s.hook("LMC", t + .053, "lmc.bus", event=Ev("mp.store.submitted", prid, device="cuda:0"))
    s.hook("LMC", t + .054, "lmc.bus.stream", event=Ev("mp.store.start", prid, transfer_key=prid + "#1"))
    s.hook("LMC", t + .055, "lmc.bus", event=Ev("mp.tokens", prid, chunk_hashes=[c0, c1],
                                                token_chunks=[ids[0:32], ids[32:64]], token_offsets=[0, 32],
                                                parent_hashes=[None, c0]))
    s.hook("LMC", t + .056 + st, "lmc.bus", event=Ev("l1.write.reserved", keys=[ObjKey(c0), ObjKey(c1)], l1_tag="l1",
                                                     tag=None))
    s.hook("LMC", t + .070 + st, "lmc.bus", event=Ev("l1.write.finished", keys=[ObjKey(c0), ObjKey(c1)], l1_tag="l1",
                                                     meta=[Size(1000), Size(1000)]))
    s.hook("LMC", t + .071 + st, "lmc.bus.stream", event=Ev("mp.store.end", prid, transfer_key=prid + "#1",
                                                            stored_count=2, total_bytes=2000, num_tokens=64))
    s.hook("LMC", t + .099, "lmc.bus.native", event_name="mp.store.end", session=prid, ts=t + .072 + st,
           metadata={"transfer_key": prid + "#1", "stored_count": 2, "total_bytes": 2000, "num_tokens": 64})
    if mode == "disk":
        s.hook("LMC", t + .080 + st, "lmc.bus", event=Ev("l2.keys.stored", keys=[ObjKey(c0), ObjKey(c1)],
                                                         sizes=[1000, 1000], backend="fs", shared=False))
        s.hook("LMC", t + .081 + st, "lmc.bus", event=Ev("l1.keys.evicted", keys=[ObjKey(c0), ObjKey(c1)], l1_tag="l1",
                                                         meta=[Size(1000), Size(1000)]))
    # D1 (500 ms after P's answer, as the proxy waits)
    t2 = t + (0.2 if race else 0.6)
    s.hook("D1", t2, "vllm.request.added", request=dr)
    s.hook("LMC", t2 + .001, "lmc.bus", event=Ev("mp.lookup", drid, request_id=drid, chunk_hashes=[c0, c1],
                                                 model_name="m", chunk_size=32, seq_len=69, token_ids=ids))
    s.hook("LMC", t2 + .002, "lmc.bus", event=Ev("mp.lookup_prefetch.start", drid))
    if race:
        s.hook("LMC", t2 + .003, "lmc.bus", event=Ev("mp.lookup_prefetch.end", drid, found_count=0, l1_hit_tokens=0,
                                                     l2_hit_tokens=0, hit_tokens=0))
        s.hook("D1", t2 + .004, "lmc.conn.lookup", request=dr, vllm_hit=0, lmc_hit=0, load=0)
        s.hook("D1", t2 + .050, "vllm.gpu.cached", request=dr, blocks=[Blk(20), Blk(21), Blk(22), Blk(23)], first=0,
               hashes=hashes, group=0, mask=None, block_size=16)
    else:
        if mode == "disk":
            if reads_hook:                                             # our hook at the file adapter's read
                s.hook("LMC", t2 + .010, "lmc.l2.read", keys=[ObjKey(c0), ObjKey(c1)], task_id=7, requested=2)
            s.hook("LMC", t2 + .010, "lmc.bus", event=Ev("l2.keys.accessed", keys=[ObjKey(c0), ObjKey(c1)],
                                                         backend="fs", shared=False))
        s.hook("LMC", t2 + .020, "lmc.bus", event=Ev("mp.lookup_prefetch.end", drid, found_count=2,
                                                     l1_hit_tokens=0 if mode == "disk" else 64,
                                                     l2_hit_tokens=64 if mode == "disk" else 0, hit_tokens=64))
        s.hook("D1", t2 + .021, "lmc.conn.lookup", request=dr, vllm_hit=0, lmc_hit=64, load=64)
        s.hook("D1", t2 + .022, "vllm.gpu.allocated", request=dr, new_blocks=([Blk(20), Blk(21), Blk(22), Blk(23),
                                                                              Blk(24)],),
               num_new_tokens=5, num_new_computed_tokens=0, num_external_computed_tokens=64, delay=True)
        s.hook("D1", t2 + .023, "lmc.conn.retrieve_planned", meta=Meta(drid, Op(0, 64, [20, 21, 22, 23])))
        s.hook("D1", t2 + .024, "lmc.conn.retrieve_submitted", request_ids=[drid])
        s.hook("LMC", t2 + .025, "lmc.bus", event=Ev("mp.retrieve.start", drid, transfer_key=drid + "#2"))
        s.hook("LMC", t2 + .030, "lmc.bus", event=Ev("l1.read.finished", keys=[ObjKey(c0), ObjKey(c1)], l1_tag="l1"))
        s.hook("LMC", t2 + .031, "lmc.bus", event=Ev("mp.retrieve.end", drid, transfer_key=drid + "#2",
                                                     retrieved_count=2, total_bytes=2000, num_tokens=64))
        s.hook("D1", t2 + .032, "lmc.conn.finished", result=(set(), {drid}))
        s.hook("D1", t2 + .040, "vllm.gpu.cached", request=dr, blocks=[Blk(20), Blk(21), Blk(22), Blk(23)], first=0,
               hashes=hashes, group=0, mask=None, block_size=16)
    s.hook("D1", t2 + .200, "vllm.request.finished", request=dr, delay=False)
    s.hook("D1", t2 + .201, "vllm.request.blocks_freed", manager=types.SimpleNamespace(
        get_block_ids=lambda rid: ([20, 21, 22, 23, 24],)), request=dr)
    s.hook("D1", t2 + 5, "vllm.gpu.evicted", block=Blk(20), hashes=[hashes[0] + b"\0\0\0\0"])
    tag = "lmc_disk|0|0"
    (Path(d) / "proxy.jsonl").write_text(json.dumps({"key": key, "dest": 0, "prefill": 0, "conv": "lmc:0", "tag": tag,
                                                      "t_in": t - .01, "t_prefill_done": t + .065,
                                                      "t_first_byte": t2 + .1}) + "\n")
    (Path(d) / "mtrag.jsonl").write_text(json.dumps({"tag": tag, "conv": 0, "turn": 0, "prompt_tokens": 69,
                                                      "pieces": [["system", "", 0, 10], ["doc", "d1", 10, 60],
                                                                 ["user", 0, 60, 69]], "ids": ids}) + "\n")
    (Path(d) / "env.json").write_text(json.dumps({"mode": mode, "chunk": 32, "l1_gb": 4, "eviction": "noop",
                                                   "store_policy": "skip_l1", "mtrag_limit": 1, "layout": "interleaved",
                                                   "store_wait_ms": 500, "d_role": "kv_consumer", "commit": "x"}))
    return key


class LifecycleReportTests(unittest.TestCase):
    def report(self, **kw):
        run, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate(run, **kw)
        nums = lmc_lifecycle.write_report(run, out)
        rows = {n: [json.loads(ln) for ln in gzip.open(out / f"{n}.jsonl.gz", "rt")]
                for n in ("chunks", "requests", "blocks")}
        return nums, rows, out

    def test_a_chunk_stored_to_disk_and_pulled_from_it(self):
        nums, rows, out = self.report(mode="disk")
        self.assertEqual(nums["chunks"]["stored"], 2)
        self.assertEqual(nums["chunks"]["to_disk"], 2)
        self.assertEqual(nums["chunks"]["removed_from_memory"], 2)
        self.assertEqual((nums["chunks"]["pulled"], nums["chunks"]["pulled_from"]), (2, {"L2": 2}))
        self.assertEqual(nums["chunks"]["never_pulled"], 0)
        self.assertEqual(nums["chunks"]["read_outcomes"], {"pulled": 2, "new: stored by this request": 2})
        self.assertFalse(nums["health"]["server_crashed"])
        self.assertEqual(nums["checks"], {"chunk_tokens_match_prompt": "2 of 2", "pulled_blocks_same_hash_as_p": "4 of 4",
                                          "chunks_with_full_chain": "2 of 2",
                                          "disk_mode_on_disk_then_out_of_memory": "2 of 2",
                                          "disk_mode_pulls_from_disk": "2 of 2",
                                          "disk_mode_pulls_while_still_in_memory": 0})
        self.assertEqual(nums["D"]["lmcache_loaded_tokens"], 64)
        self.assertEqual(nums["D"]["computed_tokens"], 5)
        self.assertEqual(nums["P"]["stored_tokens"], 64)
        self.assertEqual(nums["trace"]["hook_errors"], {})
        self.assertAlmostEqual(nums["D"]["pull_s_median_l2"], 0.031 - 0.001, places=6)   # server lookup to copy done
        self.assertAlmostEqual(nums["D"]["prefetch_s_median_l2"], 0.020 - 0.001, places=6)
        c0 = rows["chunks"][0]
        self.assertEqual((c0["offset"], c0["ntok"], c0["bytes"], c0["pieces"]), (0, 32, 1000, ["system", "doc d1"]))
        self.assertEqual([b["gpu"] for b in c0["p_blocks"]], [10, 11])
        pulled = [rd for rd in c0["reads"] if rd["how"] == "pulled"]
        self.assertEqual(len(pulled), 1)
        self.assertEqual([b["gpu"] for b in pulled[0]["d_blocks"]], [20, 21])
        self.assertEqual(pulled[0]["role"], "D1")
        # the store's end: the GPU's time (native), not the queue's; queued kept
        self.assertAlmostEqual(rows["requests"][0]["t_store_end"], 0.072 + 0.0, places=4)
        src = {(b["role"], b["index"]): b["source"] for b in rows["blocks"]}
        self.assertEqual(src[("P", 0)], "computed")
        self.assertEqual(src[("D1", 3)], "LMCache")
        ev = [b for b in rows["blocks"] if b["role"] == "D1" and b["index"] == 0][0]
        self.assertIsNotNone(ev["t_evicted"])
        md = (out / "summary.md").read_text()
        self.assertIn("One chunk's life", md)
        self.assertIn("read from disk (L2)", md)

    def test_reads_from_disk_and_touches_apart(self):
        """With the read and touch hooks: a pull is from disk when the file adapter read the chunk; a touch (no I/O,
        published by LMCache as l2.keys.accessed too) is not a read."""
        run, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate(run, mode="disk")
        s = Sim.__new__(Sim)
        s.d = run
        c0 = hb(0x40, 32)
        s.hook("LMC", 103.0, "lmc.touch", keys=[ObjKey(c0)])
        s.hook("LMC", 103.0, "lmc.bus", event=Ev("l2.keys.accessed", keys=[ObjKey(c0)], backend="fs", shared=False))
        nums = lmc_lifecycle.write_report(run, out)
        ck = nums["chunks"]
        self.assertTrue(ck["reads_hook"])
        self.assertEqual((ck["disk_reads"], ck["touches"]), (2, 1))
        self.assertEqual(ck["pulled_from"], {"L2": 2})
        md = (out / "summary.md").read_text()
        self.assertIn("a touch: no I/O", md)
        self.assertIn("reads from disk: 2; touches", md)

    def test_a_trace_from_before_the_read_hook(self):
        run, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate(run, mode="disk", reads_hook=False)
        nums = lmc_lifecycle.write_report(run, out)
        self.assertFalse(nums["chunks"]["reads_hook"])
        self.assertEqual(nums["chunks"]["pulled_from"], {"L2": 2})                   # from l2.keys.accessed
        self.assertIn("a read from disk, or a touch", (out / "summary.md").read_text())

    def test_memory_mode_pulls_from_memory(self):
        nums, rows, _ = self.report(mode="memory")
        self.assertEqual((nums["chunks"]["pulled_from"], nums["chunks"]["to_disk"]), ({"L1": 2}, 0))
        self.assertIsNotNone(nums["D"]["pull_s_median_l1"])

    def test_a_lookup_before_the_store_landed_is_a_miss_and_counted(self):
        nums, rows, _ = self.report(mode="disk", race=True)
        self.assertEqual(nums["chunks"]["read_outcomes"], {"miss: not yet stored": 2,
                                                            "new: stored by this request": 2})
        self.assertEqual(nums["D"]["looked_up_before_p_store_end"], 1)
        self.assertEqual(nums["D"]["computed_tokens"], 69)

    def test_a_server_that_died_is_reported(self):
        run, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate(run, mode="memory")
        t0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(100.0 + 2))   # 2 s after the first record
        (run / "lmcache_server.log").write_text(f"[{t0},500] LMCache INFO: Stored 64 tokens\n"
                                                "Fatal Python error: Segmentation fault\n")
        t1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(100.0 + 20))
        (run / "engine_D1.log").write_text(f"[{t1},000] LMCache WARNING: LMCache server is unhealthy — entering "
                                            "degraded mode\n")
        nums = lmc_lifecycle.write_report(run, out)
        h = nums["health"]
        self.assertTrue(h["server_crashed"])
        self.assertAlmostEqual(h["server_last_log_s"], 2.5, places=3)
        self.assertAlmostEqual(h["degraded"]["D1"], 20.0, places=3)
        self.assertIn("did not last the run", (out / "summary.md").read_text())

    def test_server_errors_are_reported(self):
        """Job 30722: every store failed with an exception and the server stayed up."""
        run, out = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate(run, mode="memory")
        t0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(100.0 + 1))
        err = (f"\x1b[31;20m[{t0},000] LMCache ERROR:\x1b[0m Cannot store keys due to exception "
               "\x1b[3m(lmcache_driven_transfer.py:813:lmcache.v1.multiprocess.modules.lmcache_driven_transfer)\x1b[0m\n"
               "Traceback (most recent call last):\n  File x\n"
               "TypeError: __init__(): incompatible constructor arguments.\n")
        (run / "lmcache_server.log").write_text(err * 3)
        nums = lmc_lifecycle.write_report(run, out)
        h = nums["health"]
        self.assertEqual(h["server_errors"], 3)
        self.assertTrue(h["server_failed"])
        self.assertFalse(h["server_crashed"])
        self.assertEqual(list(h["server_error_kinds"].values()), [3])
        self.assertIn("Cannot store keys due to exception -> TypeError", list(h["server_error_kinds"])[0])
        self.assertIn("logged 3 errors", (out / "summary.md").read_text())

    def test_a_published_folder_reports_again(self):
        run, res = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        simulate(run, mode="disk")
        subprocess.run(["tar", "-C", str(run), "-czf", str(res / "kvtrace.tar.gz"), "kvtrace"], check=True)
        for n in ("proxy.jsonl", "mtrag.jsonl"):
            (res / f"{n}.gz").write_bytes(gzip.compress((run / n).read_bytes()))
        (res / "env.json").write_text((run / "env.json").read_text())
        out = Path(tempfile.mkdtemp())
        lmc_lifecycle.main(["report", "--results", str(res), "--output", str(out)])
        self.assertEqual(json.loads((out / "results.json").read_text())["chunks"]["pulled_from"], {"L2": 2})

    def test_compare(self):
        a, b = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        for d, mode in ((a, "memory"), (b, "disk")):
            run = Path(tempfile.mkdtemp())
            simulate(run, mode=mode)
            lmc_lifecycle.write_report(run, d)
        out = Path(tempfile.mkdtemp())
        table = lmc_lifecycle.compare([str(a), str(b)], out)
        self.assertEqual(table["pulled from"], [{"L1": 2}, {"L2": 2}])
        self.assertIn("| pulls | 2 | 2 |", (out / "compare.md").read_text())


class HookTests(unittest.TestCase):
    def test_the_copies_are_the_import_plus_hooks(self):
        problems, hooked = lmc_overlay.check(ROOT)
        self.assertEqual(problems, [])
        self.assertEqual(len(hooked), 7)

    def test_every_hook_calls_its_extractor_with_the_names_it_takes(self):
        n = 0
        for f, line, ev, text in lmc_overlay.hooks(ROOT):
            if ev in ("import", "other"):
                continue
            m = HOOK_RE.search(text)
            self.assertIsNotNone(m, text)
            passed = set(re.findall(r",\s*(\w+)=", m.group(2)))
            fn = events.EXTRACT[ev]
            params = inspect.signature(fn).parameters
            required = {k for k, p in params.items() if p.default is inspect.Parameter.empty}
            self.assertTrue(required <= passed, f"{f}:{line}: {ev} needs {required - passed}")
            self.assertTrue(passed <= set(params), f"{f}:{line}: {ev} does not take {passed - set(params)}")
            n += 1
        self.assertGreaterEqual(n, 20)

    def test_the_wanted_lmcache_events_exist(self):
        src = (ROOT / "third_party/lmcache/lmcache/v1/mp_observability/event.py").read_text()
        for name in __import__("kvtrace").LMC_WANTED:
            self.assertIn(f'"{name}"', src)


class OverlayTests(unittest.TestCase):
    def tree(self):
        """A stand-in repository (two copies, one hooked file) and a stand-in site-packages with the release."""
        root, site = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        files = {"vllm": {"vllm/__init__.py": "x = 1\n", "vllm/core.py": "def f():\n    return 1\n"},
                 "lmcache": {"lmcache/__init__.py": "", "lmcache/bus.py": "def g():\n    pass\n"}}
        import hashlib
        for copy, fs in files.items():
            man = []
            for rel, text in fs.items():
                p = root / "third_party" / copy / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
                w = site / rel
                w.parent.mkdir(parents=True, exist_ok=True)
                w.write_text(text)
                man.append(f"{hashlib.sha256(text.encode()).hexdigest()}  {rel}")
            (root / "third_party" / f"{copy}.sha256").write_text("\n".join(man) + "\n")
        (site / "vllm" / "_C.so").write_bytes(b"\x7fELF")
        (site / "lmcache" / "cuda_ops.cpython-312-x86_64-linux-gnu.so").write_bytes(b"\x7fELF wheel")
        (root / "kvtrace").mkdir()
        for f in (ROOT / "kvtrace").glob("*.py"):                           # the real module
            (root / "kvtrace" / f.name).write_text(f.read_text())
        core = root / "third_party/vllm/vllm/core.py"
        core.write_text("import kvtrace  # KVTRACE\ndef f():\n    if kvtrace.ON: kvtrace.hook(\"x\")  # KVTRACE\n"
                        "    return 1\n")
        return root, site

    def test_build_puts_our_compiled_ops_in_place_of_the_wheels(self):
        import hashlib
        root, site = self.tree()
        ext = Path(tempfile.mkdtemp())
        (ext / "lmcache").mkdir()
        ours = b"\x7fELF ours"
        (ext / "lmcache" / "cuda_ops.cpython-312-x86_64-linux-gnu.so").write_bytes(ours)
        (ext / "BUILD.json").write_text(json.dumps({"modules": [{
            "module": "lmcache.cuda_ops", "file": "cuda_ops.cpython-312-x86_64-linux-gnu.so",
            "sha256": hashlib.sha256(ours).hexdigest()}], "from": "x", "torch": "t", "nvcc": "n"}))
        out = Path(tempfile.mkdtemp()) / "overlay"
        info = lmc_overlay.build(site, out, root, log=lambda m: None, ext_dirs=[ext])
        so = out / "lmcache" / "cuda_ops.cpython-312-x86_64-linux-gnu.so"
        self.assertEqual(so.read_bytes(), ours)
        self.assertEqual((site / "lmcache" / "cuda_ops.cpython-312-x86_64-linux-gnu.so").read_bytes(),
                         b"\x7fELF wheel")                                       # the environment untouched
        self.assertEqual(info["built"][0]["replaced"], ["cuda_ops.cpython-312-x86_64-linux-gnu.so"])
        # verify --need-ops: a file that is not a loadable module stops it
        env = dict(os.environ, PYTHONPATH=str(out), KVTRACE_DIR=tempfile.mkdtemp())
        r = subprocess.run([sys.executable, "-P", str(ROOT / "scripts" / "lmc_overlay.py"), "verify", "--out",
                            str(out), "--need-ops"], cwd=tempfile.mkdtemp(), env=env, capture_output=True, text=True,
                           timeout=60)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("compiled CUDA ops do not load", r.stdout)

    def test_build_lays_our_hooked_files_over_the_wheel(self):
        root, site = self.tree()
        self.assertEqual(lmc_overlay.check(root)[0], [])
        out = Path(tempfile.mkdtemp()) / "overlay"
        info = lmc_overlay.build(site, out, root, log=lambda m: None)
        self.assertEqual(info["hooked_files"], ["vllm/core.py"])
        self.assertIn("KVTRACE", (out / "vllm/core.py").read_text())
        self.assertNotIn("KVTRACE", (site / "vllm/core.py").read_text())        # the environment untouched
        self.assertEqual(os.stat(out / "vllm/_C.so").st_ino, os.stat(site / "vllm/_C.so").st_ino)
        self.assertTrue((out / "kvtrace/__init__.py").is_file())

    def test_verify_sees_what_an_engine_imports(self):
        """Run as the setup runs it: -P from a folder of its own; and from the checkout (which has its own kvtrace/),
        where a plain `python -` put the checkout's kvtrace first (job 30688): -P keeps the overlay's."""
        root, site = self.tree()
        out = Path(tempfile.mkdtemp()) / "overlay"
        lmc_overlay.build(site, out, root, log=lambda m: None)
        for cwd in (Path(tempfile.mkdtemp()), ROOT):
            trace = Path(tempfile.mkdtemp())
            env = dict(os.environ, PYTHONPATH=str(out), KVTRACE_DIR=str(trace), KVTRACE_ROLE="check")
            r = subprocess.run([sys.executable, "-P", str(ROOT / "scripts" / "lmc_overlay.py"), "verify", "--out",
                                str(out)], cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            v = json.loads((out / "VERIFY.json").read_text())
            self.assertTrue(v["ok"], v)
            self.assertIn("does not load", v["lmcache_cuda_ops"])          # no compiled ops in the stand-in
        # the old way, from the checkout: the checkout's kvtrace shadows the overlay's
        r = subprocess.run([sys.executable, "-", str(out)], cwd=ROOT, input="import kvtrace; print(kvtrace.__file__)",
                           env=dict(os.environ, PYTHONPATH=str(out)), capture_output=True, text=True, timeout=60)
        self.assertNotIn(str(out), r.stdout)
        # a hooked module missing from the overlay is a problem
        (out / "vllm" / "core.py").write_text("def f():\n    return 1\n")
        r = subprocess.run([sys.executable, "-P", str(ROOT / "scripts" / "lmc_overlay.py"), "verify", "--out",
                            str(out)], cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(out),
                                                          KVTRACE_DIR=tempfile.mkdtemp()),
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("0 hook lines", r.stdout)

    def test_a_change_beyond_the_hooks_or_a_different_wheel_stops_it(self):
        root, site = self.tree()
        core = root / "third_party/vllm/vllm/core.py"
        core.write_text(core.read_text().replace("return 1", "return 2"))
        self.assertTrue(any("changed beyond" in p for p in lmc_overlay.check(root)[0]))
        root, site = self.tree()
        (root / "third_party/lmcache/lmcache/extra.py").write_text("y = 1\n")
        self.assertTrue(any("not in the import" in p for p in lmc_overlay.check(root)[0]))
        root, site = self.tree()
        (root / "third_party/lmcache/lmcache/bus.py").write_text("kvtrace = 1\ndef g():\n    pass\n")
        self.assertTrue(any("mentions kvtrace" in p or "changed" in p for p in lmc_overlay.check(root)[0]))
        root, site = self.tree()
        (site / "lmcache/bus.py").write_text("def g():\n    return 0\n")
        info = lmc_overlay.build(site, Path(tempfile.mkdtemp()) / "o", root, log=lambda m: None)
        self.assertEqual(info["differ"], ["lmcache/bus.py"])


class _Fake:
    """A fake prefill and decode engine (aiohttp) recording bodies, headers and arrival times."""

    def __init__(self):
        import asyncio
        import threading
        from aiohttp import web
        self.seen = {"p": [], "d": []}
        self.ports = {}
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()

        def handler(name):
            async def h(request):
                body = await request.json()
                self.seen[name].append((time.time(), dict(request.headers), body))
                if name == "p":
                    return web.json_response({"choices": [{"text": "x"}]})
                resp = web.StreamResponse(headers={"content-type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(b"data: {\"i\": 0}\n\ndata: [DONE]\n\n")
                return resp
            return h

        def run():
            asyncio.set_event_loop(self.loop)

            async def start():
                for name in ("p", "d"):
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
class ProxyStoreModeTests(unittest.TestCase):
    def test_store_mode(self):
        import socket
        import urllib.request
        eng = _Fake()
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        log = Path(tempfile.mkdtemp()) / "proxy.jsonl"
        proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "pd_proxy.py"), "--port", str(port),
                                 "--prefill", f"http://127.0.0.1:{eng.ports['p']}",
                                 "--decode", f"http://127.0.0.1:{eng.ports['d']}", "--mode", "store",
                                 "--store-wait-ms", "300", "--policy", "conversation", "--log", str(log)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: (proc.kill(), proc.wait(), proc.stderr.close()))
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        body = {"model": "m", "prompt": [1, 2, 3], "max_tokens": 9, "stream": True, "conversation_id": "c1",
                "t_tag": "lmc|0|0", "t_trace": True, "t_pieces": [["doc", "a", 0, 3]]}
        r = urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=30) as f:
            self.assertIn("[DONE]", f.read().decode())
        (tp, hp, p), (td, hd, d) = eng.seen["p"][0], eng.seen["d"][0]
        self.assertEqual((p["max_tokens"], p["stream"], d["max_tokens"], d["stream"]), (1, False, 9, True))
        for b in (p, d):                                      # nothing of ours reaches the engines
            self.assertFalse(any(k.startswith("t_") or k in ("kv_transfer_params", "conversation_id") for k in b), b)
        key = {k.lower(): v for k, v in hp.items()}["x-request-id"]
        self.assertEqual({k.lower(): v for k, v in hd.items()}["x-request-id"], key)
        self.assertRegex(key, r"^[0-9a-f]{32}$")
        self.assertGreaterEqual(td - tp, 0.29)                # the store's wait
        rec = json.loads(log.read_text().splitlines()[0])
        self.assertEqual((rec["key"], rec["tag"], rec["mode"]), (key, "lmc|0|0", "store"))


class MatrixTests(unittest.TestCase):
    def test_lmctrace_runs(self):
        m = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                           env=dict(os.environ, MATRIX="lmctrace", PRINT_RUNS="1"))
        self.assertEqual(m.returncode, 0, m.stderr)
        self.assertEqual(m.stdout.splitlines(), [
            "RUNNER=lmc_run.sh LMC_MODE=memory MTRAG_LIMIT=4 STORE_WAIT_MS=500",
            "RUNNER=lmc_run.sh LMC_MODE=disk MTRAG_LIMIT=4 STORE_WAIT_MS=500"])
        bad = subprocess.run(["bash", "scripts/pd_matrix.sh"], cwd=ROOT, capture_output=True, text=True,
                             env=dict(os.environ, MATRIX="lmctrace", PRINT_RUNS="1", KVPD_LMC_CONVS="0"))
        self.assertEqual(bad.returncode, 2)

    def test_disk_plan_counts_the_disk_tier(self):
        import pd_campaign
        runs = ["RUNNER=lmc_run.sh LMC_MODE=memory MTRAG_LIMIT=4 STORE_WAIT_MS=500",
                "RUNNER=lmc_run.sh LMC_MODE=disk MTRAG_LIMIT=4 STORE_WAIT_MS=500"]
        self.assertAlmostEqual(pd_campaign.extra_gb("lmctrace", runs), 1.5 * 4 + 2 * 0.25)
        self.assertAlmostEqual(pd_campaign.extra_gb("lmctrace", runs[:1]), 0.25)
        self.assertIn("lmctrace", pd_campaign.MATRICES)

    @staticmethod
    def _cu13(prefix, nvcc_release, runtime=None, compiler=True, crt_tag=""):
        """A stand-in of pip's nvidia/cu13 packages: a compiler (a fake nvcc printing its release), and/or the
        runtime headers (CUDART_VERSION) and library."""
        w = prefix / "nvidia" / "cu13"
        for d in ("bin", "nvvm/bin", "include/crt", "include/cccl", "lib"):
            (w / d).mkdir(parents=True, exist_ok=True)
        if compiler:
            (w / "bin" / "nvcc").write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {nvcc_release}, V0'\n")
            (w / "nvvm" / "bin" / "cicc").write_text("")
            for f in ("bin/nvcc", "nvvm/bin/cicc"):
                os.chmod(w / f, 0o755)
            (w / "include" / "crt" / "host_config.h").write_text(crt_tag)
        if runtime is not None:
            (w / "include" / "cuda_runtime_api.h").write_text(f"#define CUDART_VERSION {runtime}\n")
            (w / "include" / "cuda_runtime.h").write_text("")
            (w / "lib" / "libcudart.so.13").write_text("")
        return w

    def toolkit(self, site, home, compiler=None):
        env = dict(os.environ, KV_HOME=str(home), CUDA13_FROM_SITE=str(site), CUDA13_HOME=str(home / "cuda13"),
                   CUDA13_NO_RUNFILE="1")
        if compiler is not None:
            env["CUDA13_COMPILER"] = str(compiler)
        return subprocess.run(["bash", str(ROOT / "scripts" / "cuda13_toolkit.sh")], capture_output=True, text=True,
                              env=env, timeout=60)

    def test_a_compiler_of_another_cuda_version_than_the_headers_is_refused(self):
        """Job 30716: the environment's nvcc 13.5 with its runtime headers 13.0 (CCCL refuses the pair)."""
        site, home = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        self._cu13(site, "13.5", runtime=13000)
        r = self.toolkit(site, home)
        self.assertEqual(r.returncode, 1)
        self.assertIn("CUDA 13.5", r.stderr)
        self.assertIn("CUDA 13.0", r.stderr)

    def test_the_environments_headers_with_a_compiler_of_their_version(self):
        site, home, comp = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
        w = self._cu13(site, "13.5", runtime=13000, crt_tag="13.5")
        c = self._cu13(comp, "13.0", crt_tag="13.0")
        (c / "lib" / "libnvvm.so.4").write_text("")
        r = self.toolkit(site, home, compiler=comp)
        self.assertEqual(r.returncode, 0, r.stderr)
        tk = Path(r.stdout.strip().splitlines()[-1])
        self.assertEqual(tk, home / "build" / "cuda13_home")
        self.assertEqual(os.path.realpath(tk / "bin" / "nvcc"), str((c / "bin" / "nvcc").resolve()))
        self.assertEqual((tk / "include" / "crt" / "host_config.h").read_text(), "13.0")        # the compiler's own
        self.assertEqual(os.path.realpath(tk / "include" / "cuda_runtime_api.h"),
                         str((w / "include" / "cuda_runtime_api.h").resolve()))                 # the environment's
        self.assertEqual(os.path.realpath(tk / "lib64" / "libcudart.so"), str((w / "lib" / "libcudart.so.13").resolve()))
        self.assertTrue((tk / "lib" / "libnvvm.so").exists())
        for d in ("include/cuda_runtime.h", "include/cccl", "lib/libcudart.so"):            # the 13.0 profile's paths
            self.assertTrue((tk / "targets" / "x86_64-linux" / d).exists(), d)
        self.assertIn("compiler CUDA 13.0", r.stderr)
        self.assertFalse((home / "cuda13").exists())                     # nothing downloaded or installed

    def test_the_build_makes_all_four_of_the_wheels_modules(self):
        import lmc_build_ops
        src = ROOT / "third_party" / "lmcache"
        names = [n for n, _f, _i in lmc_build_ops.common_specs(src)]
        self.assertEqual(sorted(names + ["cuda_ops"]), ["cuda_ops", "lmcache_fs", "lmcache_native", "lmcache_redis"])
        self.assertEqual(len(lmc_build_ops.cuda_sources(src)), 13)
        for _n, files, _i in lmc_build_ops.common_specs(src):
            self.assertTrue(all((src / f).is_file() for f in files))

    def test_scripts_parse(self):
        for f in ("lmc_run.sh", "lmc_setup.sh", "pd_matrix.sh", "pd_card.sh", "cuda13_toolkit.sh"):
            r = subprocess.run(["bash", "-n", str(ROOT / "scripts" / f)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f + r.stderr)


if __name__ == "__main__":
    unittest.main()
