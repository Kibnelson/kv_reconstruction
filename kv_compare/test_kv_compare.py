"""CPU tests on a tiny random Llama (no downloads): capture, RoPE geometry, lossless and lossy codecs, quality.

    python -m unittest discover -s kv_compare -p 'test_*.py' -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover
    torch = None

NAMES = [f"train{i}" for i in range(6)] + ["tables", "test_text", "test_code"]


def tiny_model():
    import os
    torch.manual_seed(int(os.environ.get("KV_COMPARE_TEST_SEED", "0")))
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
                      rope_theta=500000.0,
                      rope_scaling={"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                    "high_freq_factor": 4.0, "original_max_position_embeddings": 64})
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    return model


def doc_ids(seed, n, vocab=97):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, vocab, size=40)
    ids = np.concatenate([base[rng.integers(0, 40, size=n // 2)], rng.integers(0, vocab, size=n - n // 2)])
    return [int(v) for v in ids]


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class TinyLlamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from kv_compare.capture import capture_ids, model_meta
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "captures"
        cls.model = tiny_model()
        cls.device = torch.device("cpu")
        meta = model_meta(cls.model, "tiny-random-llama", None)
        reports = []
        cls.tokens = 320
        for i, name in enumerate(NAMES):
            reports.append(capture_ids(cls.model, doc_ids(i, cls.tokens), cls.root, name, cls.device,
                                       queries=name.startswith("train")))
        (cls.root / "meta.json").write_text(json.dumps(dict(meta, tokens=cls.tokens, captures=reports)))
        from kv_compare.predict import Study
        cls.study = Study(cls.root, "cpu", NAMES[:6], "tables", ["test_text", "test_code"], log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_capture_shapes_and_meta(self):
        meta = json.loads((self.root / "meta.json").read_text())
        self.assertEqual((meta["kv_heads"], meta["head_dim"], meta["query_heads"]), (2, 16, 4))
        K = np.load(self.root / "test_text_K_bits.npy")
        self.assertEqual(K.shape, (4, self.tokens, 32))
        self.assertEqual(np.load(self.root / "train0_Q_bits.npy").shape, (4, self.tokens, 64))
        self.assertFalse((self.root / "test_text_Q_bits.npy").exists())

    def test_rope_inverse_matches_pre_rope_keys(self):
        """De-rotating the cached keys must give the k_proj outputs (up to BF16 rounding)."""
        from kv_compare.core import words_to_float
        captured = {}
        layer = 2
        def keep(module, inputs, output):
            captured["k"] = output.detach()[0].float()     # observe only: returning None keeps the output
        hook = self.model.model.layers[layer].self_attn.k_proj.register_forward_hook(keep)
        ids = doc_ids(0, self.tokens)
        with torch.inference_mode():
            self.model(torch.tensor([ids]), use_cache=True)
        hook.remove()
        doc = self.study.train[0]
        x = words_to_float(doc.words(layer, self.study.device))
        f = self.study.rope.derotate_row(x, self.study.positions(x.shape[0]))[:, : self.study.C]
        ref = captured["k"].double()
        err = (f - ref).abs().max() / ref.abs().max()
        self.assertLess(float(err), 0.03)
        back = self.study.rope.rotate_row(self.study.rope.derotate_row(x, self.study.positions(x.shape[0])), self.study.positions(x.shape[0]))
        self.assertLess(float((back - x).abs().max()), 1e-9)

    def test_pivot_order_and_groups_keep_rope_pairs(self):
        from kv_compare.predict import CONFIGS
        F = self.study.fit(3, CONFIGS["fast"])
        perm = F.perm.cpu().numpy()
        self.assertEqual(sorted(perm.tolist()), list(range(self.study.W)))
        starts = F.starts.cpu().numpy()
        pos = np.empty_like(perm); pos[perm] = np.arange(len(perm))
        partner = self.study.partner.cpu().numpy()
        for c in range(self.study.C):
            self.assertEqual(starts[pos[c]], starts[pos[partner[c]]])
        self.assertTrue(np.all(starts <= np.arange(len(starts))))

    def test_lossless_costs(self):
        from kv_compare.baselines import exans_bits, splitzip_codebook, splitzip_token_bits
        from kv_compare.predict import CONFIGS
        inside = splitzip_codebook(self.study.train, self.study.L)
        doc = self.study.tests[0]
        sz = splitzip_token_bits(doc, self.study.L, inside)
        self.assertTrue(np.all(sz >= 12))
        ex = exans_bits(doc, range(self.study.L), 0, 256)
        self.assertTrue(8 < ex < 16)
        for name in ("prev", "block", "nohint", "fast"):
            per_layer, _ = self.study.lossless(CONFIGS[name])
            v = per_layer["test_text"].numpy()
            self.assertTrue(np.isfinite(v).all(), name)
            self.assertLess(float(v.mean()), 16.5, name)

    def test_frontier_matches_named_configs(self):
        """Variants fitted from the shared covariance reproduce the lossless stage's costs; reduced rank and
        8-bit formats run and cost more bits than (or the same as) the full-precision fit."""
        from kv_compare.frontier import Frontier, Variant, summarize
        from kv_compare.predict import CONFIGS
        same = {"prev": Variant(("P",), 1, "natural", adaptive=False), "block": Variant(("P",), 16, "natural", adaptive=False),
                "fast": Variant(("L0", "P2", "P", "ATT"), 8)}
        extra = [Variant(("L0", "P2", "P"), 8, rank=16), Variant(("L0", "P2", "P"), 8, rank=16, coef_bits=8, feat_bits=8),
                 Variant(("L0", "P2", "P"), 8)]
        vs, bits, macs = Frontier(self.study, list(same.values()) + extra, log=lambda m: None).run()
        for name, v in same.items():
            ref, _ = self.study.lossless(CONFIGS[name])
            for d in ref:
                self.assertAlmostEqual(float(bits[v][d].double().mean()), float(ref[d].mean()), places=4, msg=name)
        mean = {v: float(sum(b.double().mean() for b in bits[v].values())) for v in vs}
        self.assertGreater(mean[extra[0]], mean[extra[2]])
        self.assertGreaterEqual(mean[extra[1]] + 1e-3, mean[extra[0]])
        self.assertLess(macs[extra[0]], macs[extra[2]])
        rows = summarize(vs, bits, macs, [(0, 160), (160, 320)])
        self.assertEqual(len(rows), len(vs))

    def test_lossy_and_quality(self):
        from kv_compare import baselines
        from kv_compare.core import words_to_float
        from kv_compare.predict import CONFIGS
        from kv_compare.quality import continuation_nll, matrices_from_doc, matrices_from_values
        fits = {}
        cfg = CONFIGS["fast"]
        self.study.lossless(cfg, fits)
        doc = self.study.tests[0]
        P, M = 192, 96
        self.study._f.clear(); self.study._qmap.clear()
        rec, rate = self.study.lossy(cfg, fits, doc, P, 1.0)
        self.assertEqual(len(self.study._f), 0)      # the closed loop must not accumulate calibration features
        exact = torch.stack([words_to_float(doc.words(l, self.study.device, P)) for l in range(self.study.L)])
        self.assertEqual(tuple(rec.shape), tuple(exact.shape))
        self.assertTrue(0 < rate < 16)
        # error bound: half a step (step = 1 x the channel's spread) plus BF16 rounding of the reconstruction,
        # which is at most half a BF16 step of the pre-rounding value, |x| + step/2 in magnitude
        spread = torch.stack([fits[l].spread for l in range(self.study.L)])[:, None, :]
        bound = 0.5 * spread + (exact.abs() + 0.5 * spread) * 2.0 ** -8 + 1e-12
        self.assertTrue(bool(((rec.double() - exact).abs() <= bound * 1.0001).all()))
        aqua = baselines.aqua_fit(self.study, self.study.L)
        for out, r in (baselines.aqua_style(self.study, aqua, doc, P, 3), baselines.kivi_style(self.study, doc, P, 3),
                       baselines.fp8_cast(self.study, doc, P)):
            self.assertEqual(tuple(out.shape), tuple(exact.shape))
            self.assertTrue(torch.isfinite(out).all())
        base = continuation_nll(self.model, doc.ids, P, M, self.device)
        same = continuation_nll(self.model, doc.ids, P, M, self.device, matrices_from_doc(doc, P))
        self.assertEqual(base, same)
        lossy = continuation_nll(self.model, doc.ids, P, M, self.device, matrices_from_values(rec, self.study.C))
        self.assertTrue(math.isfinite(lossy))


CODER_CHECK = r"""
import os, sys
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, sys.argv[1])
import torch
from kv_compare.core import exact_bits, float_to_words
from kv_compare.gpu_coder import SliceCoder
torch.manual_seed(0)
n = 700
x = torch.randn(n, dtype=torch.float64) * torch.exp(torch.randn(n))
mu = (x + torch.randn(n, dtype=torch.float64) * 0.05 * x.abs().clamp_min(1e-3)).float()
s = (0.05 * x.abs().clamp_min(1e-3) * torch.exp(0.3 * torch.randn(n))).float()
w = float_to_words(x)
w[:5] = torch.tensor([0x0000, 0x8000, 0x7F80, 0x7FC0, 0xFF80])
mu[10:20] = 0.0
s[10:20] = 1e-12
w = w.to(torch.int16)
c = SliceCoder("cpu", values_per_stream=16, block=16)
buf, starts, m = c.encode(w, mu, s)
dec = c.decode(buf, starts, mu, s, m)
assert torch.equal(dec, w), "round trip"
ideal = c.ideal_bits(w, mu, s) / n
model = float(exact_bits(w.to(torch.int32) & 0xFFFF, mu.double(), s.double()).mean())
assert abs(ideal - model) < 0.1, (ideal, model)
print("ok", ideal, model)
"""


class CoderTests(unittest.TestCase):
    def test_value_slice_coder_round_trip_interpreted(self):
        """The Triton coder (run by Triton's CPU interpreter) restores every word, including zeros, Inf, NaN and
        tiny spreads, and its symbol cost stays close to the exact model."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None:
            self.skipTest("Triton is not installed")
        r = subprocess.run([sys.executable, "-c", CODER_CHECK, str(ROOT)], capture_output=True, text=True, timeout=900)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertIn("ok", r.stdout)


class PackageTests(unittest.TestCase):
    def test_package_copies_small_files_and_refuses_overwrite(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import package_llama_compare as pk
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            (run / "captures").mkdir(parents=True)
            (run / "captures" / "meta.json").write_text("{}")
            (run / "captures" / "x_K_bits.npy").write_bytes(b"not copied")
            (run / "run_args.json").write_text("{}")
            (run / "environment.json").write_text("{}")
            (run / "lossless.json").write_text(json.dumps({"layers": 2, "methods": {"ExANS-style": {}}}))
            (run / "summary.md").write_text("| ExANS-style | 10.7 |\n")
            dest = Path(tmp) / "out"
            self.assertEqual(pk.main([str(run), "--destination", str(dest)]), 0)
            self.assertTrue((dest / "capture_meta.json").is_file())
            self.assertFalse(any(p.suffix == ".npy" for p in dest.iterdir()))
            manifest = json.loads((dest / "manifest.json").read_text())
            self.assertIn("lossless.json", manifest["files"])
            with self.assertRaises(SystemExit):
                pk.main([str(run), "--destination", str(dest)])


if __name__ == "__main__":
    unittest.main()
