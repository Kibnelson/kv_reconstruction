"""CPU tests of the T-ideas harness (kv_compare/t_ideas.py, scripts/t_ideas.py) on a tiny random Llama.

    python -m unittest kv_compare.test_t_ideas -v
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    torch = None

TRAIN = [f"wtrain{i}" for i in range(6)]
TESTS = ["ds_wikitext2_r0", "ds_gsm8k_r0"]


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class TIdeasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kv_compare.capture import capture_ids, model_meta
        from kv_compare.core import float_to_words, words_to_float
        from kv_compare.test_kv_compare import doc_ids, tiny_model
        cls.tmp = tempfile.TemporaryDirectory()
        cls.run_dir = Path(cls.tmp.name) / "run"
        cls.root = cls.run_dir / "captures"
        cls.model = tiny_model()
        cls.tokens = 320
        reports = [capture_ids(cls.model, doc_ids(i, cls.tokens), cls.root, n, torch.device("cpu"))
                   for i, n in enumerate(TRAIN + ["wtables"] + TESTS)]
        (cls.root / "meta.json").write_text(json.dumps(dict(model_meta(cls.model, "tiny-random-llama", None),
                                                            tokens=cls.tokens, captures=reports)))
        # a shifted document: the values of ds_gsm8k_r0 doubled (a request unlike the calibration text)
        K = np.load(cls.root / "ds_gsm8k_r0_K_bits.npy")
        V = np.load(cls.root / "ds_gsm8k_r0_V_bits.npy")
        v2 = float_to_words(words_to_float(torch.as_tensor(V.astype(np.int32))) * 2).numpy().astype(np.uint16)
        np.save(cls.root / "shifted_K_bits.npy", K)
        np.save(cls.root / "shifted_V_bits.npy", v2)
        np.save(cls.root / "shifted_ids.npy", np.load(cls.root / "ds_gsm8k_r0_ids.npy"))
        from kv_compare import fused
        from kv_compare.predict import Study
        cls.study = Study(cls.root, "cpu", TRAIN, "wtables", TESTS + ["shifted"], log=lambda m: None)
        cls.consts = fused.prepare(cls.study, log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_t_costs_side_bits_and_adaptation(self):
        from kv_compare import t_ideas as ti
        st = self.study
        N, W, G = self.tokens, st.W, st.W // st.hd
        doc = st.tests[0]
        per, side = ti.t_costs(st, self.consts, doc)
        for v, bits in per.items():
            self.assertTrue(np.isfinite(bits.numpy()).all(), v)
            self.assertLess(float(bits.mean()), 16.5, v)
        self.assertAlmostEqual(side["T"], 0.0)
        self.assertAlmostEqual(side["bwd-pilot8"], 0.0)
        self.assertAlmostEqual(side["fwd-spread/ch"], 6 / N)
        self.assertAlmostEqual(side["fwd-bias+spread/ch"], 14 / N)
        self.assertAlmostEqual(side["fwd-spread/head"], 6 * G / (N * W))
        # on the doubled values the request's own spreads must win clearly
        per, side = ti.t_costs(st, self.consts, st.tests[2])
        t, fwd = float(per["T"].mean()), float(per["fwd-spread/ch"].mean()) + side["fwd-spread/ch"]
        self.assertLess(fwd, t - 0.1, (t, fwd))
        self.assertLess(float(per["bwd-pilot8"].mean()), t - 0.05)

    def test_float_one_step_matches_study(self):
        """The class machinery with one class reproduces kv_compare.predict's one-step costs exactly."""
        from kv_compare import t_ideas as ti
        from kv_compare.predict import Config
        st = self.study
        ref, _ = st.lossless(Config("onestep", ("P",), 1, "natural", True))
        st._f.clear()
        got, info = ti.float_costs(st, ti.ANCHORS[0], st.train, st.tables, st.tests[:2], log=lambda m: None)
        for d in st.tests[:2]:
            np.testing.assert_allclose(got[d.name].numpy(), ref[d.name].mean(1).numpy(), rtol=0, atol=1e-6)
            self.assertEqual(info[d.name]["side_bits_per_value"], 0.0)

    def test_oracle_bound(self):
        from kv_compare import t_ideas as ti
        st = self.study
        for d in st.tests[:2]:
            cal, _ = ti.float_costs(st, ti.ANCHORS[0], st.train, st.tables, [d], log=lambda m: None)
            inside, cross = ti.oracle_costs(st, d, block=64, log=lambda m: None)
            self.assertLessEqual(float(inside.mean()), float(cal[d.name].mean()) + 0.02)
            self.assertLessEqual(float(inside.mean()), float(cross.mean()) + 1e-9)     # in-sample is optimistic
            self.assertTrue(np.isfinite(cross.numpy()).all())

    def test_anchor_and_class_specs(self):
        from kv_compare import t_ideas as ti
        st = self.study
        d = st.tests[0]
        L, W, N = st.L, st.W, self.tokens
        for spec in ti.ANCHORS + ti.CLASSES:
            per, info = ti.float_costs(st, spec, st.train, st.tables, [d], log=lambda m: None)
            v = per[d.name].numpy()
            self.assertTrue(np.isfinite(v).all(), spec.name)
            self.assertLess(float(v.mean()), 16.5, spec.name)
            row = info[d.name]
            if spec.anchor == "first":
                self.assertEqual(row["waves"], 2)
                self.assertGreater(row["class1_fraction"], 0.0)
            if spec.anchor == "recent":
                self.assertGreaterEqual(row["waves"], 2)
            if spec.anchor.startswith("nn"):
                K = int(spec.anchor[2:])
                self.assertEqual(row["waves"], 2)
                self.assertAlmostEqual(row["class1_fraction"], (1 - math.ceil(N / K) / N) * (L - 1) / L, places=6)
            if spec.classes.startswith("best"):
                k = int(spec.classes[4:])
                self.assertAlmostEqual(row["side_bits_per_value"], math.log2(k) * (L - 1) / (L * W))
            else:
                self.assertEqual(row["side_bits_per_value"], 0.0)

    def test_waves(self):
        from kv_compare.t_ideas import waves
        self.assertEqual(waves(np.array([-1, -1, 0, 2, 3, -1])), 4)
        self.assertEqual(waves(np.array([-1, -1])), 1)

    def test_layer0_recomputed_bit_for_bit(self):
        from kv_compare import t_ideas as ti
        st = self.study
        d = st.tests[0]
        K, V = ti.layer0_words(self.model, d.ids.tolist(), torch.device("cpu"))
        np.testing.assert_array_equal(K, np.asarray(d.K[0]))
        np.testing.assert_array_equal(V, np.asarray(d.V[0]))
        self.assertEqual(len(self.model.model.layers), st.L)          # the model is restored
        r = ti.layer0_costs(st, self.model, [d], st.tables, torch.device("cpu"), t_layer0={d.name: 12.0})
        row = r["docs"][d.name]
        self.assertEqual(row["equal_fraction"], 1.0)
        self.assertLess(row["bits_per_layer0_value"], 0.01)
        self.assertAlmostEqual(row["saving_per_value_all_layers"], (12.0 - row["bits_per_layer0_value"]) / st.L)

    def test_allocation_matches_brute_force(self):
        from kv_compare.t_ideas import allocate
        rng = np.random.default_rng(3)
        L = 5
        rows = [{"mode": m, "layer_bits": (rng.uniform(1, 3, L) * 1e8 * f).tolist(),
                 "layer_ms": (rng.uniform(0.5, 1.5, L) * t).tolist()} for m, f, t in
                (("small-slow", 0.7, 2.0), ("mid", 0.85, 1.2), ("big-fast", 1.0, 0.6))]
        for g in (1, 10, 100, 400):
            a = allocate(rows, g, step_ms=0.001)
            best = math.inf
            for pick in itertools.product(range(3), repeat=L):
                bits = sum(rows[m]["layer_bits"][l] for l, m in enumerate(pick))
                ms = sum(rows[m]["layer_ms"][l] for l, m in enumerate(pick))
                best = min(best, max(bits / (g * 1e9) * 1e3, ms))
            self.assertAlmostEqual(a["pipelined"]["mix_ms"], best, delta=0.01)
            self.assertLessEqual(a["pipelined"]["mix_ms"], a["pipelined"]["single_ms"] + 1e-9)
            self.assertLessEqual(a["serial"]["mix_ms"], a["serial"]["single_ms"] + 1e-9)
            self.assertEqual(len(a["pipelined"]["mix_modes"]), L)
        self.assertEqual(allocate(rows, 1)["serial"]["best_single"], "small-slow")
        self.assertEqual(allocate(rows, 400)["serial"]["best_single"], "big-fast")

    def test_layer0_lean_and_shapes(self):
        """The lean recompute (no attention, no MLP) gives the captured layer 0 bit for bit in the capture's own
        shape, and every other prefill shape gives the right rows with a finite cost."""
        from kv_compare import t_ideas as ti
        st = self.study
        d, other = st.tests[0], st.tables.ids[:2048].tolist()
        cpu = torch.device("cpu")
        K, V = ti.layer0_kv(self.model, d.ids.tolist(), cpu)
        got = d.words(0, cpu)
        self.assertTrue(torch.equal(torch.cat([K, V], 1), got))
        K2, V2 = ti.layer0_words(self.model, d.ids.tolist(), cpu)
        self.assertTrue(np.array_equal(K.numpy().astype(np.uint16), K2) and np.array_equal(V.numpy().astype(np.uint16), V2))
        for v in ti.L0_VARIANTS:
            K, V = ti.layer0_kv(self.model, d.ids.tolist(), cpu, v, other)
            self.assertEqual(tuple(K.shape), (self.tokens, st.C), v)
            self.assertEqual(tuple(V.shape), (self.tokens, st.W - st.C), v)
            # values carry no position: a different pass shape may change rounding, not the rows' identity
            self.assertGreater(float((V == got[:, st.C:]).float().mean()), 0.5, v)
        r = ti.layer0_robust(st, self.model, [d], st.tables, cpu, t_layer0={d.name: 12.0},
                             timer=lambda fn: (fn(), 0.001)[1])
        row = r["docs"][d.name]
        self.assertEqual(row["same"]["keys_equal"], 1.0)
        self.assertEqual(row["same"]["values_equal"], 1.0)
        self.assertLess(row["same"]["bits_per_layer0_value"], 0.01)
        for v in ti.L0_VARIANTS:
            self.assertTrue(math.isfinite(row[v]["bits_per_layer0_value"]), v)
            self.assertLess(row[v]["bits_per_layer0_value"], 16.5, v)
        self.assertAlmostEqual(row["lean_ms"], 1.0)
        self.assertEqual(len(self.model.model.layers), st.L)

    def test_ulp_coder(self):
        from kv_compare.t_ideas import ulp_bits, ulp_table
        rng = np.random.default_rng(1)
        ref = torch.as_tensor(rng.integers(0, 65536, size=(400, 32)), dtype=torch.int32)
        ref[0, 0], ref[0, 1] = 0x0000, 0x8000
        same = ulp_table(ref, ref)
        self.assertLess(ulp_bits(ref, ref, same), 0.01)
        got = ref.clone()
        got[0, 0] = 0x8000                              # -0 for +0: equal order, different word: an escape
        self.assertGreater(ulp_bits(got, ref, same), ulp_bits(ref, ref, same) + 15 / ref.numel())
        # one BF16 step away on half the values: the cost is the empirical entropy of the step differences
        from kv_compare.t_ideas import _ordered
        mag = ref & 0x7FFF
        step = torch.where((torch.rand(ref.shape, generator=torch.Generator().manual_seed(0)) < 0.5) & (mag < 0x7F00)
                           & (mag > 0), 1, 0)
        got = ref + step
        d = (_ordered(got) - _ordered(ref)).reshape(-1)
        p = torch.bincount(d + 1).double() / d.numel()
        H = float(-(p[p > 0] * torch.log2(p[p > 0])).sum())
        self.assertAlmostEqual(ulp_bits(got, ref, ulp_table(got, ref)), H, delta=0.02)
        self.assertGreater(H, 1.0)

    def test_pick_sets(self):
        from kv_compare.t_ideas import cal_sets, pick_sets
        costs = {"a": {"x": [9, 1, 1, 9], "y": [9, 5, 5, 1]}, "b": {"x": [9, 2, 2, 1], "y": [9, 1, 1, 9]}}
        early = pick_sets(costs, ["a", "b"], [1, 2])
        self.assertEqual({d: s for d, (s, _) in early.items()}, {"x": "a", "y": "b"})
        whole = pick_sets(costs, ["a", "b"], slice(1, None), cost=lambda v: float(np.sum(v)), side_bits=lambda d: 1.0)
        self.assertEqual(whole["x"], ("b", 15.0))
        sets = cal_sets(3)
        self.assertEqual(sets["wikitext"][0], [f"wtrain{i}" for i in range(6)])
        self.assertEqual(len(sets["mixed"][0]), 6)
        self.assertEqual(len(sets["mixed-all"][0]), 15)
        self.assertEqual(sets["code"][1], "cal_code_3")
        self.assertEqual(sets["mixed"][1], "cal_mixed_3")

    def test_driver_calib_cpu(self):
        """The calib stage on the tiny run folder (calibration windows captured beforehand) and its report."""
        from kv_compare.capture import capture_ids
        from kv_compare.t_ideas import cal_sets
        from kv_compare.test_kv_compare import doc_ids
        names = sorted({n for tr, tb, _ in cal_sets(3).values() for n in tr + [tb]} - {f"wtrain{i}" for i in range(6)}
                       - {"wtables"})
        for i, n in enumerate(names):
            if not (self.root / f"{n}_K_bits.npy").is_file():
                capture_ids(self.model, doc_ids(40 + i, self.tokens), self.root, n, torch.device("cpu"))
        out = Path(self.tmp.name) / "out_calib"
        cmd = [sys.executable, str(ROOT / "scripts" / "t_ideas.py"), "--run", str(self.run_dir), "--output", str(out),
               "--device", "cpu", "--windows", "1", "--stages", "calib,report"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        res = json.loads((out / "t_ideas.json").read_text())
        sets = res["calib"]["sets"]
        self.assertEqual(set(sets), set(cal_sets(3)))
        for name, s in sets.items():
            self.assertEqual(set(s["docs"]), set(TESTS), name)
        s = (out / "summary.md").read_text()
        self.assertIn("## 6. Calibration breadth", s)
        self.assertIn("pick per request", s)
        self.assertIn("6 calibration breadth", s)

    def test_driver_cpu(self):
        """The driver's CPU stages and the report, on the tiny run folder."""
        out = Path(self.tmp.name) / "out"
        cmd = [sys.executable, str(ROOT / "scripts" / "t_ideas.py"), "--run", str(self.run_dir), "--output", str(out),
               "--device", "cpu", "--windows", "1", "--stages", "adapt,anchor,classes,gpu,report"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        s = (out / "summary.md").read_text()
        for head in ("## 1. Per-request adaptation", "## 4. Anchors", "## 5. Token classes", "## Verdicts"):
            self.assertIn(head, s)
        res = json.loads((out / "t_ideas.json").read_text())
        self.assertEqual(set(res["adapt"]["docs"]), set(TESTS))
        self.assertTrue(all("partial" not in res[k] for k in ("adapt", "anchor", "classes", "gpu")))
        r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)          # kept, not measured again
        self.assertEqual(r2.returncode, 0, r2.stderr[-3000:])
        self.assertIn("adapt: kept", r2.stdout)


MODES_CHECK = r"""
import os, sys, json, tempfile
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from kv_compare.capture import capture_ids, model_meta
from kv_compare.predict import Study
from kv_compare import fused, t_ideas as ti
from kv_compare.grouped import prepare_groups
from kv_compare.layerwise import prepare_rank
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, max_position_embeddings=2048, rope_theta=500000.0)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
names = [f"train{i}" for i in range(4)] + ["tables", "test"]
root = Path(tempfile.mkdtemp()) / "c"
reports = []
for i, n in enumerate(names):
    ids = [int(v) for v in np.random.default_rng(i).integers(0, 97, size=64)]
    reports.append(capture_ids(model, ids, root, n, torch.device("cpu")))
(root / "meta.json").write_text(json.dumps(dict(model_meta(model, "tiny", None), tokens=64, captures=reports)))
study = Study(root, "cpu", names[:4], "tables", ["test"], log=lambda m: None)
consts = fused.prepare(study, log=lambda m: None)
groups = prepare_groups(study, consts, [2, 4], log=lambda m: None)
rank = prepare_rank(study, consts, 32, log=lambda m: None)
skip = lambda fn, dev, **kw: 0.0           # no timing under the interpreter: one encode and one decode per mode
rows = ti.mode_rows(study, consts, rank, groups, study.tests[0], ["dense", "4g", "rank256"], torch.device("cpu"),
                    skip, log=print, multiple=64)
per, _ = ti.t_costs(study, consts, study.tests[0], variants=("T",))
n = study.L * 64 * study.W
for r in rows:
    assert "error" not in r, r
    assert r["bit_exact"], r["mode"]
    total = sum(r["layer_bits"]) / n
    assert 0 < r["bits_per_value"] - total < 0.5, (r["mode"], r["bits_per_value"], total)
    print("ok", r["mode"], r["bits_per_value"])
d = next(r for r in rows if r["mode"] == "dense")
ref = float(per["T"].mean())
assert abs(d["exact_model_bits_per_value"] - ref) < 2e-3, (d["exact_model_bits_per_value"], ref)
print("ok reference", d["exact_model_bits_per_value"], ref)
"""


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class TIdeasKernelTests(unittest.TestCase):
    def test_modes_and_reference_interpreted(self):
        """Every mode's codec as the gpu stage builds it restores the KV bit for bit (Triton's CPU interpreter), the
        per-layer bits add up to the stream, and T's torch reference cost equals the cost from the kernel's own
        features and priors."""
        import importlib.util
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is not installed")
        r = subprocess.run([sys.executable, "-c", MODES_CHECK, str(ROOT)], capture_output=True, text=True, timeout=5400)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:] + r.stdout[-2000:])
        self.assertEqual(r.stdout.count("ok"), 4, r.stdout)


if __name__ == "__main__":
    unittest.main()
