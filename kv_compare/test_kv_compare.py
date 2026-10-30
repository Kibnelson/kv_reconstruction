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

    def test_higgs_and_hadamard(self):
        """Our Walsh-Hadamard transform is the Sylvester matrix product; HIGGS error falls with the grid size and
        an all-zero row stays zero."""
        from scipy.linalg import hadamard as sylvester
        from kv_compare import baselines
        g = torch.Generator().manual_seed(0)
        x = torch.randn(7, 64, generator=g, dtype=torch.float64)
        H = torch.as_tensor(sylvester(64), dtype=torch.float64)
        self.assertTrue(torch.allclose(baselines.hadamard(x), x @ H, atol=1e-10))
        r = torch.randn(300, 1024, generator=g, dtype=torch.float64)
        r[5] = 0
        errs = []
        for n in (16, 64, 256):
            y = baselines.higgs_qdq(r, n)
            self.assertTrue(bool((y[5] == 0).all()))
            errs.append(float((y - r).norm() / r.norm()))
        self.assertTrue(errs[0] > errs[1] > errs[2] > 0)
        self.assertLess(errs[0], 0.4)

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
        rs, rate_s = self.study.lossy(cfg, fits, doc, P, 1.0, prefix=4, token_step=True)
        ex4 = torch.stack([words_to_float(doc.words(l, self.study.device, 4)) for l in range(self.study.L)])
        self.assertTrue(torch.equal(rs[:, :4].double(), ex4.float().double()))
        self.assertTrue(0 < rate_s < 16)
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
        aqua3 = baselines.aqua_fit(self.study, self.study.L, bits=3)     # fitted on its own 3-bit reconstructions
        a2 = baselines.aqua_official_fit(self.study, self.study.L, 2)
        a4 = baselines.aqua_official_fit(self.study, self.study.L, 4)
        outs = {}
        for name, (out, r) in (("aqua-style", baselines.aqua_style(self.study, aqua, doc, P, 3)),
                               ("aqua-style-closed", baselines.aqua_style(self.study, aqua3, doc, P, 3)),
                               ("aqua 2", baselines.aqua_official(self.study, a2, doc, P, 2)),
                               ("aqua 4", baselines.aqua_official(self.study, a4, doc, P, 4)),
                               ("kivi-style", baselines.kivi_style(self.study, doc, P, 3)),
                               ("kivi 2", baselines.kivi_official(self.study, doc, P, 2)),
                               ("kivi 4", baselines.kivi_official(self.study, doc, P, 4)),
                               ("fp8", baselines.fp8_cast(self.study, doc, P))):
            self.assertEqual(tuple(out.shape), tuple(exact.shape), name)
            self.assertTrue(torch.isfinite(out).all(), name)
            outs[name] = (out, r)
        err = lambda name: float((outs[name][0].double() - exact).norm() / exact.norm())
        self.assertLess(err("aqua 4"), err("aqua 2"))
        self.assertLess(err("kivi 4"), err("kivi 2"))
        # AQUA-KV keeps the first tokens exact; KIVI keeps the last 128 values tokens (and keys beyond whole 128s)
        p = baselines.AQUA_PREFIX
        self.assertTrue(torch.equal(outs["aqua 2"][0][:, :p].double(), exact[:, :p]))
        C = self.study.C
        self.assertTrue(torch.equal(outs["kivi 2"][0][:, -128:, C:].double(), exact[:, -128:, C:]))
        nk = P - P % 128
        self.assertTrue(torch.equal(outs["kivi 2"][0][:, nk:, :C].double(), exact[:, nk:, :C]))
        # rates: KIVI b + 1 per quantized value (two BF16 numbers per 32), 16 elsewhere
        nv = P - 128
        kivi2 = ((nk * C * 3 + (P - nk) * C * 16) + (nv * C * 3 + (P - nv) * C * 16)) / (P * 2 * C)
        self.assertAlmostEqual(outs["kivi 2"][1], kivi2)
        self.assertGreater(outs["aqua 2"][1], 2.0)
        self.assertLess(outs["aqua 2"][1], outs["aqua 4"][1])
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
import numpy as np
from kv_compare.gpu_coder import slice_tables, stage1_reference
t = slice_tables("cpu")
cum, off, K, slot = t["cum"].numpy().astype(np.int64), t["off"].numpy(), t["K"].numpy(), np.arange(65536)
for c in range(10):      # the search-free stage 1 finds every slot's symbol
    assert np.array_equal(stage1_reference(t, c, slot), np.searchsorted(cum[off[c]: off[c] + K[c] + 2], slot, side="right") - 1)
for mode in ("exact", "uniform"):
    c = SliceCoder("cpu", values_per_stream=16, block=16, mode=mode)
    buf, starts, m = c.encode(w, mu, s)
    dec = c.decode(buf, starts, mu, s, m)
    assert torch.equal(dec, w), ("round trip", mode)
    c.search = True      # the same streams decode with the earlier stage-1 search
    assert torch.equal(c.decode(buf, starts, mu, s, m), w), ("round trip with search", mode)
    c.search = False
    ideal = c.ideal_bits(w, mu, s) / n
    model = float(exact_bits(w.to(torch.int32) & 0xFFFF, mu.double(), s.double()).mean())
    assert abs(ideal - model) < 0.15, (mode, ideal, model)
    print("ok", mode, ideal, model)
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

    def test_fused_codec_round_trip_interpreted(self):
        """The fused one-step codec (prediction, RoPE, spread, coding and next-layer features in one kernel per
        direction) restores a tiny Llama's whole KV bit for bit, special words included, in both modes and with
        either stage-1 decoder; its features and priors match a PyTorch reference."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", FUSED_CHECK, str(ROOT)], capture_output=True, text=True, timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 2, r.stdout)

    def test_layerwise_codec_round_trip_interpreted(self):
        """The per-layer one-step codec (int8 product, then one coding kernel per layer; streams kept open across
        layers) restores the same KV bit for bit, with one and two token chunks, both modes, both stage-1 decoders
        and the reduced-rank product (fast-link mode), its requantization in a separate kernel or in the first
        product's epilogue."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", LAYERWISE_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=1800)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 4, r.stdout)


FUSED_CHECK = r"""
import os, sys, json, tempfile
os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from kv_compare.capture import capture_ids, model_meta
from kv_compare.predict import Study
from kv_compare.core import words_to_float
from kv_compare import fused
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, max_position_embeddings=2048, rope_theta=500000.0)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
names = [f"train{i}" for i in range(4)] + ["tables", "test"]
root = Path(tempfile.mkdtemp()) / "c"
reports = []
for i, n in enumerate(names):
    ids = [int(v) for v in np.random.default_rng(i).integers(0, 97, size=64)]
    reports.append(capture_ids(model, ids, root, n, torch.device("cpu"), queries=n.startswith("train")))
(root / "meta.json").write_text(json.dumps(dict(model_meta(model, "tiny", None), tokens=64, captures=reports)))
study = Study(root, "cpu", names[:4], "tables", ["test"], log=lambda m: None)
consts = fused.prepare(study, log=lambda m: None)
doc = study.tests[0]
L, T, W = study.L, doc.tokens, study.W
w32 = torch.stack([doc.words(l, "cpu") for l in range(L)])
w32[1, 3, :5] = torch.tensor([0x0000, 0x8000, 0x7F80, 0x7FC0, 0xFF80])      # zeros, Inf, NaN, -Inf
words = (w32 - 65536 * (w32 >= 32768).to(torch.int32)).to(torch.int16)
for mode, rows, search in (("uniform", 4, False), ("exact", 2, True)):
    codec = fused.FusedOneStep(consts, study.C, study.hd, tokens_per_program=16, rows=rows, kchunk=32, warps=4,
                               mode=mode, search=search)
    xall, prior = codec.model(words)
    pos, pr = torch.arange(T), torch.ones(T)
    for l in range(L):                       # features and priors against the PyTorch reference
        x = words_to_float(w32[l])
        f = fused._feat_ref(x, study.rope, pos, consts["isx"][l])
        f = torch.where(torch.isnan(f), torch.zeros_like(f), f)
        assert float((f == xall[l].float()).float().mean()) > 0.999, ("features", l)
        assert float((prior[l] - pr).abs().max()) < 2e-3, ("prior", l)
        m = fused.reference_mu(consts, l, None if l == 0 else xall[l - 1].float(), pos, study.C, study.hd)
        q = torch.floor((x - m.double()).abs().float() * consts["inv"][l][None, :])
        pr = torch.where(torch.isnan(q), torch.full_like(q, 524288.0), q).clamp(max=524288.0).sum(1) / (W * fused.QS)
    buf, starts = codec.encode(words, xall, prior)
    dec = codec.decode(buf, starts, T)
    assert torch.equal(dec, words), ("round trip", mode, int((dec != words).sum()))
    print("ok", mode, codec.bits(starts, T) / (L * T * W))
"""


LAYERWISE_CHECK = FUSED_CHECK.split("for mode, rows, search in")[0] + r"""
from kv_compare import layerwise
xall, prior = fused.FusedOneStep(consts, study.C, study.hd, tokens_per_program=16, rows=4, kchunk=32, warps=4).model(words)
rc = layerwise.prepare_rank(study, consts, 32, log=lambda m: None)
tiles = (16, 16, 32, 4)
ck = layerwise.LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=16, rank=rc, requant="kernel")
ce = layerwise.LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=16, rank=rc, requant="epilogue",
                                rank_tiles=tiles)
for l in range(1, L):          # requantized in the product's epilogue: the same 8-bit intermediate and products
    sk, se = ck._scratch(T), ce._scratch(T)
    ak = ck._product(xall[l - 1], l, torch.empty(T, W, dtype=torch.int32), sk)
    ae = ce._product(xall[l - 1], l, torch.empty(T, W, dtype=torch.int32), se)
    assert se["z"] is None and torch.equal(sk["zq"], se["zq"]) and torch.equal(ak, ae), ("epilogue", l)
for mode, chunks, search, rank, rq in (("uniform", 1, True, None, "kernel"), ("exact", 2, False, None, "kernel"),
                                       ("uniform", 2, True, rc, "kernel"), ("uniform", 2, True, rc, "epilogue")):
    codec = layerwise.LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=16, mode=mode, search=search,
                                       chunks=chunks, rank=rank, requant=rq, rank_tiles=tiles)
    if rank is not None:       # its own forward pass: the same features, priors from the reduced-rank predictor
        xr, pr = codec.model(words)
        assert torch.equal(xr[:L - 1], xall[:L - 1]), "features"
        buf, starts = codec.encode(words, xr, pr)
    else:
        buf, starts = codec.encode(words, xall, prior)
    dec = codec.decode(buf, starts, T)
    assert torch.equal(dec, words), ("round trip", mode, chunks, int((dec != words).sum()))
    print("ok", mode, chunks, rq, codec.bits(starts, T) / (L * T * W))
"""


LOSSY_CHECK = FUSED_CHECK.split("for mode, rows, search in")[0] + r"""
from kv_compare import lossy_gpu as lg, layerwise
xs = torch.stack([words_to_float(w32[l]) for l in range(L)])
rc = layerwise.prepare_rank(study, consts, 32, log=lambda m: None)
gen = torch.Generator().manual_seed(5)
shaped = torch.exp2(torch.empty(tuple(consts["sp"].shape)).uniform_(-4, 4, generator=gen))   # 1/16 .. 16 per lane
plain_bits = {}
for kappa, chunks, rank, prefix, tstep, lanes, rdo, cen in (
        (1.0, 2, None, 0, False, None, 0.0, False), (0.25, 1, rc, 0, False, None, 0.0, False),
        (2.0, 2, None, 4, True, None, 0.0, False), (1.0, 2, None, 4, False, shaped, 0.0, False),
        (0.5, 1, rc, 0, True, shaped, 0.0, False), (1.0, 2, None, 0, False, None, 0.25, False),
        (1.0, 2, None, 0, False, None, 0.0, True), (2.0, 1, rc, 4, False, shaped, 0.12, True)):
    codec = lg.LossyLayerwise(consts, study.C, study.hd, kappa, tokens_per_program=16, chunks=chunks, rank=rank,
                              requant="epilogue" if rank else "kernel", rank_tiles=(16, 16, 32, 4), prefix=prefix,
                              token_step=tstep, lane_scale=lanes, rdo=rdo, centroid=cen)
    rec, xall, prior = codec.quantize(words)
    buf, starts, _ = codec.encode(words, rec, xall, prior)
    dec = codec.decode(buf, starts, T)
    assert torch.equal(dec, rec), ("round trip", kappa, int((dec != rec).sum()))
    r32 = rec.to(torch.int32) & 0xFFFF
    xr = torch.stack([words_to_float(r32[l]) for l in range(L)])
    step = (codec.lane_kappa().cpu() * codec.k["sp"].double().cpu())[:, None, :] * (4.0 if tstep else 1.0)  # mult <= 4
    half = 0.5 + (1.0 if rdo else 0.0) + (0.5 if cen else 0.0)       # rate-aware rounding: 1.5 steps; centroid: +0.5
    bound = half * step + (xs.abs() + half * step) * 2.0 ** -8 + 1e-30
    fin = torch.isfinite(xs)
    ok = ((xr - xs).abs() <= bound * 1.0001) | (r32 == w32)
    assert bool(ok[fin].all()), ("error bound", kappa, int((~ok[fin]).sum()))
    assert torch.equal(r32[1, 3, 2:5], w32[1, 3, 2:5]), "Inf and NaN travel exactly"
    assert torch.equal(r32[:, :prefix], w32[:, :prefix]), "the first tokens travel exactly"
    bits = codec.bits(starts, T) / (L * T * W)
    key = (kappa, chunks, rank is None, prefix, tstep, lanes is None)
    if not rdo and not cen:
        plain_bits[key] = bits
    elif rdo and not cen and key in plain_bits:
        assert bits <= plain_bits[key] + 1e-9, ("rate-aware rounding saves bits", bits, plain_bits[key])
    print("ok", kappa, chunks, rank is not None, prefix, tstep, lanes is not None, rdo, cen, bits)
"""


MIXED_CHECK = FUSED_CHECK.split("for mode, rows, search in")[0] + r"""
from kv_compare import mixed
xall, prior = fused.FusedOneStep(consts, study.C, study.hd, tokens_per_program=16, rows=4, kchunk=32, warps=4).model(words)
sz = mixed.TorchSplitZipStyle("cpu")
sz.calibrate(words.view(torch.bfloat16))
for ours in ([True, False, True], [False, True, True], [False, False, False], [True, True, True], [True, True, False]):
    codec = mixed.MixedOneStep(consts, study.C, study.hd, ours, sz, tokens_per_program=16, chunks=2)
    enc = codec.encode(words, xall, prior)
    dec = codec.decode_mixed(enc, T)
    assert torch.equal(dec, words), ("round trip", ours, int((dec != words).sum()))
    print("ok", ours, codec.total_bits(enc, T) / (L * T * W))
"""


SZ_CHECK = FUSED_CHECK.split("for mode, rows, search in")[0] + r"""
from kv_compare import szcodec, layerwise, mixed
rc = layerwise.prepare_rank(study, consts, 32, log=lambda m: None)
def w16(d):
    w = torch.stack([d.words(l, "cpu") for l in range(L)])
    return (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
cal = [w16(d) for d in study.train]
c0 = szcodec.SplitZipFormatCodec(consts, study.C, study.hd, tokens_per_program=16)
_, s0 = next(c0.symbols(words))          # layer 0: prior 1 and no product, so the prediction is the rotated bias
m0 = layerwise._mu_from_acc(torch.zeros(T, W, dtype=torch.int32), consts["sa"][0], consts["b"][0], consts["cos"],
                            consts["sin"], study.C, study.hd)
ref, _, _ = szcodec.ctx_ref(m0, consts["sp"][0][None, :].expand(T, W).contiguous())
assert float(((s0 >> 17).long() == ref).float().mean()) > 0.99, "contexts"
sz = mixed.TorchSplitZipStyle("cpu")
sz.calibrate(words.view(torch.bfloat16))
books = {}
for rank, product, BM, ours, layout, chunks in (
        (None, "acc", 16, None, "tile", 1), (rc, "fused", 16, None, "tile", 2), (rc, "acc", 32, None, "tile", 1),
        (rc, "fused", 16, [False, True, True], "tile", 1), (None, "acc", 16, [True, False, True], "tile", 1),
        (None, "acc", 16, None, "row", 1), (rc, "acc", 16, None, "row", 2), (rc, "acc", 16, [True, False, True], "row", 2),
        (rc, "fused", 16, None, "row", 2), (None, "acc", 16, None, "row1", 1), (rc, "fused", 16, None, "row1", 1),
        (rc, "fused", 16, [False, True, True], "row1", 2)):
    codec = szcodec.SplitZipFormatCodec(consts, study.C, study.hd, rank=rank, product=product, tokens_per_program=BM,
                                        rank_tiles=(16, 16, 32, 4), layout=layout, chunks=chunks)
    key = (rank is not None, layout == "row1")
    if key in books:
        codec.share_books(books[key])
    else:
        books[key] = codec.calibrate(cal)
    lb = []
    if ours is None:
        stream, starts = codec.encode(words, layer_bits=lb)
        dec, bits = codec.decode(stream, starts, T), codec.bits(stream, starts)
    else:
        stream, starts, enc = codec.encode(words, layer_bits=lb, ours=ours, other=sz)
        dec, bits = codec.decode(stream, starts, T, ours=ours, sz=enc, other=sz), codec.bits(stream, starts, enc, ours)
    assert torch.equal(dec, words), ("round trip", product, ours, layout, chunks, int((dec != words).sum()))
    assert bits == sum(lb), ("layer bits", bits, sum(lb))
    if ours is None:                         # the coding alone, from precomputed products, is the same decode
        assert torch.equal(codec.decode(stream, starts, T, accs=codec.products(words)), words), "coding alone"
    print("ok", rank is not None, product, BM, ours, layout, chunks, bits / (L * T * W))
"""


class SplitZipFormatTests(unittest.TestCase):
    def test_splitzip_format_codec_round_trip_interpreted(self):
        """Our prediction with SplitZip-style fixed-length codes (kv_compare.szcodec) restores the KV bit for bit:
        dense and reduced-rank products (the second one inside the decoding kernel), two tile sizes, and mixed
        per layer with a SplitZip-style codec; contexts as documented."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", SZ_CHECK, str(ROOT)], capture_output=True, text=True, timeout=2400)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 12, r.stdout)


class MixedTests(unittest.TestCase):
    def test_mixed_codec_round_trip_interpreted(self):
        """Our per-layer codec mixed with a SplitZip-style codec per layer restores the KV bit for bit for any mix
        (our layers after a SplitZip layer start from its features and prior 1)."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", MIXED_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=2400)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 5, r.stdout)


class LossyGpuTests(unittest.TestCase):
    def test_lossy_gpu_codec_round_trip_interpreted(self):
        """The closed-loop lossy per-layer codec decodes exactly the encoder's reconstruction, within half a step
        plus BF16 rounding of the input, with Inf and NaN kept exact; dense and reduced-rank products; one kappa
        and shaped steps (kappa per lane); rate-aware rounding (fewer bits, within 1.5 steps) and centre-of-mass
        reconstruction."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", LOSSY_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=3600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 8, r.stdout)


CONTEXT_CHECK = FUSED_CHECK.split("for mode, rows, search in")[0] + r"""
from kv_compare import layerwise, context_sweep as cs, mixed
from kv_compare.szcodec import SplitZipFormatCodec
rc = layerwise.prepare_rank(study, consts, 32, log=lambda m: None)
def w16(d):
    w = torch.stack([d.words(l, "cpu") for l in range(L)])
    return (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16)
books = SplitZipFormatCodec(consts, study.C, study.hd, rank=rc, product="acc", tokens_per_program=16, warps=1,
                            rank_tiles=(16, 16, 32, 4), layout="row", chunks=2).calibrate([w16(d) for d in study.train[:2]])
sz = mixed.TorchSplitZipStyle("cpu")
sz.calibrate(w16(study.train[0]).view(torch.bfloat16))
# two requests of 64 tokens coded together, positions restarting at 0 (the second request is train1's KV)
words2 = torch.cat([w16(study.tests[0]), w16(study.train[1])], 1)
pos = torch.arange(64).repeat(2)
from kv_compare.grouped import prepare_groups
gr = prepare_groups(study, consts, [2], log=lambda m: None)
rows = cs.measure_case(study, consts, rc, words2, pos, torch.device("cpu"), log=lambda m: None, books=books, sz=sz,
                       reps=1, max_stream_tokens=64, grouped=gr,   # two streams for every codec, one per request
                       baseline_stream_tokens=64)
for r in rows:
    assert "error" not in r, r
    assert r["bit_exact"], r
    assert r.get("streams", 2) == 2, r
    print("ok", r["method"], r["bits"])
assert len(rows) == 5 and rows[1]["method"] == "ours, per layer, keys then values", rows
assert cs.link_ms(8.0, 1e9, 5.0, 8) == 1000.0 and cs.link_ms(8.0, 1e6, 5.0, 8) == 5.0
"""


class ContextSweepTests(unittest.TestCase):
    def test_batched_requests_round_trip_interpreted(self):
        """The context sweep's cases: two requests coded as one stream with positions restarting at 0, every
        method (our per-layer codec dense and rank-reduced, our SplitZip format, a SplitZip stand-in) bit-exact."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", CONTEXT_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=3600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 5, r.stdout)


GROUPED_CHECK = FUSED_CHECK.split("for mode, rows, search in")[0] + r"""
from kv_compare import layerwise, grouped
base = layerwise.LayerwiseOneStep(consts, study.C, study.hd, tokens_per_program=16, chunks=2)
xb, pb = base.model(words)
buf, starts = base.encode(words, xb, pb)
one = base.bits(starts, T) / (L * T * W)
for groups, chunks in ((2, 2), (4, 1), (4, 2)):
    cg = grouped.prepare_groups(study, consts, groups, log=lambda m: None)
    codec = layerwise.LayerwiseOneStep(cg, study.C, study.hd, tokens_per_program=16, chunks=chunks)
    xall, prior = codec.model(words)
    # the encoder's features are every layer's exact features; its priors follow the grouped prediction
    pos = torch.arange(T)
    for l in range(L):
        x = words_to_float(w32[l])
        f = fused._feat_ref(x, study.rope, pos, consts["isx"][l])
        f = torch.where(torch.isnan(f), torch.zeros_like(f), f)
        assert float((f == xall[l].float()).float().mean()) > 0.999, ("features", groups, l)
    buf, starts = codec.encode(words, xall, prior)
    dec = codec.decode(buf, starts, T)
    assert torch.equal(dec, words), ("round trip", groups, chunks, int((dec != words).sum()))
    print("ok", groups, chunks, codec.bits(starts, T) / (L * T * W), "one step", one)
"""


class GroupedCodecTests(unittest.TestCase):
    def test_head_group_codec_round_trip_interpreted(self):
        """The head-group predictor in the per-layer codec (2 and 4 groups, 1 and 2 token chunks): the decoder,
        adding each group's within-layer product after the groups before it, reproduces every word, Inf and NaN
        included."""
        import importlib.util
        import subprocess
        if importlib.util.find_spec("triton") is None or torch is None:
            self.skipTest("Triton and PyTorch are required")
        r = subprocess.run([sys.executable, "-c", GROUPED_CHECK, str(ROOT)], capture_output=True, text=True,
                           timeout=3600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        self.assertEqual(r.stdout.count("ok"), 3, r.stdout)


class StreamingProtocolTests(unittest.TestCase):
    def test_streaming_cache_follows_aquakv_buffering(self):
        """AQUA-KV's perplexity protocol harness: compressing with the identity changes nothing; blocks start after
        the 4 sinks, the first holds the 156 buffered tokens, later ones 128 (32 tokens per pass); a compressed
        block is read by the following passes; KIVI-like granules round blocks down."""
        if torch is None:
            self.skipTest("PyTorch is required")
        from transformers import LlamaConfig, LlamaForCausalLM
        from kv_compare import streaming as sm
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
        model = LlamaForCausalLM(cfg).to(torch.float32).eval()
        ids = torch.randint(0, 97, (1, 512))

        class Spy:
            granule = 1

            def __init__(self, fn):
                self.fn, self.calls = fn, []

            def __call__(self, a, X):
                self.calls.append((a, X.shape[1]))
                return self.fn(X), 0.0
        exact = sm.perplexity(model, [ids], "cpu")
        ident = Spy(lambda X: X.clone())
        same = sm.perplexity(model, [ids], "cpu", lambda: sm.StreamingCache(ident, 2))
        self.assertEqual(same["nll_sums_f32"], exact["nll_sums_f32"])
        self.assertEqual(ident.calls, [(4, 156), (160, 128), (288, 128)])
        zero = Spy(torch.zeros_like)
        changed = sm.perplexity(model, [ids], "cpu", lambda: sm.StreamingCache(zero, 2))
        self.assertNotEqual(changed["nll_sums_f32"], exact["nll_sums_f32"])
        grain = Spy(lambda X: X.clone())
        grain.granule = 64
        sm.perplexity(model, [ids], "cpu", lambda: sm.StreamingCache(grain, 2, granule=64))
        self.assertTrue(all(m % 64 == 0 for _, m in grain.calls), grain.calls)


class SensitivityTests(unittest.TestCase):
    def test_kv_gradients_and_shaped_steps(self):
        """The cached-KV gradients match finite differences of the loss (keys after RoPE and values, each lane);
        the shaped steps give F x step^2 equal on every lane (global) or within each head, geometric mean 1,
        clamped."""
        if torch is None:
            self.skipTest("PyTorch is required")
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.cache_utils import DynamicCache
        from kv_compare import sensitivity as sv
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
        model = LlamaForCausalLM(cfg).double().eval()
        ids = torch.randint(3, 97, (40,)).tolist()
        g, _ = sv.kv_grads(model, ids, torch.device("cpu"))
        self.assertEqual(tuple(g.shape), (3, 40, 64))
        C = 32

        class Bump(DynamicCache):
            def __init__(self, at):
                super().__init__()
                self.at = at

            def update(self, k, v, layer_idx, kw=None):
                l, t, c, eps = self.at
                if layer_idx == l:
                    k, v = k.clone(), v.clone()
                    tgt, cc = (k, c) if c < C else (v, c - C)
                    tgt[0, cc // 16, t, cc % 16] += eps
                return super().update(k, v, layer_idx, kw)

        def total(at):
            x = torch.tensor([ids])
            with torch.no_grad():
                out = model(x, use_cache=True, past_key_values=Bump(at))
            return float(torch.nn.functional.cross_entropy(out.logits[0, :-1], x[0, 1:], reduction="sum"))
        for l, t, c in ((0, 5, 3), (1, 0, 20), (2, 17, 40), (1, 30, 63)):
            eps = 1e-3                                   # the model keeps some float32 (RoPE tables): no smaller
            fd = (total((l, t, c, eps)) - total((l, t, c, -eps))) / (2 * eps)
            self.assertLessEqual(abs(fd - float(g[l, t, c])), 1e-5 + 0.02 * abs(fd), (l, t, c))
        F, n = sv.fisher(model, [ids, ids[::-1]], torch.device("cpu"), skip=4)
        self.assertEqual((tuple(F.shape), n), ((3, 64), 72))
        sp = torch.rand(3, 64, dtype=torch.float64) + 0.1
        s = sv.lane_scales(F, sp, mode="global", hd=16, clamp=1e9)
        w = s * sp * F.sqrt()
        self.assertTrue(torch.allclose(w, w.flatten()[:1].expand_as(w), rtol=1e-9))
        self.assertLess(abs(float(torch.log(s).mean())), 1e-9)
        s = sv.lane_scales(F, sp, mode="head", hd=16, clamp=1e9)
        hw = (s * sp * F.sqrt()).reshape(3, 4, 16)
        self.assertTrue(torch.allclose(hw, hw[..., :1].expand_as(hw), rtol=1e-9))
        self.assertLess(float(torch.log(s.reshape(3, 4, 16)).mean(2).abs().max()), 1e-9)
        s = sv.lane_scales(F, sp, clamp=4.0)
        self.assertTrue(float(s.max()) <= 4.0 + 1e-12 and float(s.min()) >= 0.25 - 1e-12)
        self.assertEqual(sv.chunks_of([1, 5, 6, 7, 8, 9, 10], 4, 1), [[1, 5, 6, 7], [1, 8, 9, 10]])
        u = sv.uniform_scales(sp, 16, clamp=1e9)                  # one absolute step per group of 16 lanes
        st = (u * sp).reshape(3, 4, 16)
        self.assertTrue(torch.allclose(st, st[..., :1].expand_as(st), rtol=1e-9))
        self.assertLess(float(torch.log(u.reshape(3, 4, 16)).mean(2).abs().max()), 1e-9)
        u = sv.uniform_scales(sp, 32, clamp=2.0)
        self.assertTrue(float(u.max()) <= 2.0 + 1e-12 and float(u.min()) >= 0.5 - 1e-12)


@unittest.skipIf(torch is None, "PyTorch and Transformers are required")
class NewModelCaptureTests(unittest.TestCase):
    """What the A5000 runs need beyond Llama-3.2-3B: chunked prefill for 32K captures, Qwen3's queries after q_norm,
    tokenizers without BOS, meta.json from calcap, SplitZip's exponent coverage."""

    def tiny_qwen(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
                          rope_theta=1e6, tie_word_embeddings=False)
        return Qwen3ForCausalLM(cfg).to(torch.float32).eval()

    def test_chunked_prefill_matches_one_pass(self):
        """The same cache positions, computed in chunks: the words agree up to BF16 rounding of a different order."""
        from kv_compare.capture import capture_ids
        from kv_compare.core import words_to_float
        for model in (tiny_model(), self.tiny_qwen().to(torch.bfloat16)):
            with tempfile.TemporaryDirectory() as tmp:
                ids = doc_ids(3, 96)
                a = capture_ids(model, ids, tmp, "one", torch.device("cpu"), queries=True)
                b = capture_ids(model, ids, tmp, "chunks", torch.device("cpu"), queries=True, chunk_tokens=32)
                self.assertNotIn("prefill_chunk_tokens", a)
                self.assertEqual(b["prefill_chunk_tokens"], 32)
                for part in ("K", "V", "Q"):
                    x = np.load(Path(tmp) / f"one_{part}_bits.npy")
                    y = np.load(Path(tmp) / f"chunks_{part}_bits.npy")
                    self.assertEqual(x.shape, y.shape)
                    self.assertEqual(x.shape[1], 96)
                    xf = words_to_float(torch.from_numpy(x.astype(np.int32)))
                    yf = words_to_float(torch.from_numpy(y.astype(np.int32)))
                    rel = float((xf - yf).norm() / xf.norm())
                    self.assertLess(rel, 0.02, (part, rel))
                    self.assertTrue(np.array_equal(x[:, :32], y[:, :32]), part)   # the first chunk is one pass

    def test_qwen3_queries_are_taken_after_q_norm(self):
        from kv_compare.capture import capture_ids, query_module
        model = self.tiny_qwen().to(torch.bfloat16)
        attn = model.model.layers[0].self_attn
        self.assertIs(query_module(attn), attn.q_norm)
        llama_attn = tiny_model().model.layers[0].self_attn
        self.assertIs(query_module(llama_attn), llama_attn.q_proj)
        seen = {}

        def keep(module, inputs, output):
            seen["q"] = output.detach()[0].reshape(output.shape[1], -1)
        h = attn.q_norm.register_forward_hook(keep)
        with tempfile.TemporaryDirectory() as tmp:
            capture_ids(model, doc_ids(1, 40), tmp, "d", torch.device("cpu"), queries=True)
            h.remove()
            q = np.load(Path(tmp) / "d_Q_bits.npy")
            self.assertEqual(q.shape, (2, 40, 4 * 16))
            ref = seen["q"].contiguous().view(torch.int16).numpy().view(np.uint16)
            self.assertTrue(np.array_equal(q[0], ref))

    def test_bos_prefix_and_meta(self):
        from kv_compare.capture import bos_prefix, ensure_meta

        class Tok:
            def __init__(self, b):
                self.bos_token_id = b
        self.assertEqual(bos_prefix(Tok(128000)), [128000])
        self.assertEqual(bos_prefix(Tok(None)), [])
        with tempfile.TemporaryDirectory() as tmp:
            m = ensure_meta(tmp, self.tiny_qwen(), "tiny-qwen3", None, 64, [{"name": "x"}])
            self.assertEqual((m["layers"], m["kv_heads"], m["head_dim"], m["query_heads"]), (2, 2, 16, 4))
            again = ensure_meta(tmp, tiny_model(), "other", None, 1, [])
            self.assertEqual(again["model"], "tiny-qwen3")          # kept as it was

    def test_exponent_coverage(self):
        from kv_compare.context_sweep import exponent_stats
        # exponents 120..135 cover 90% of the values, 100..109 the rest; the book from other words sees only 100..115
        e = torch.cat([torch.arange(120, 136).repeat(90), torch.arange(100, 110).repeat(16)])
        sign = torch.randint(0, 2, e.shape)
        words = ((sign << 15) | (e << 7) | torch.randint(0, 128, e.shape)).to(torch.int32)
        words = (words - 65536 * (words >= 32768).to(torch.int32)).to(torch.int16)
        book = ((torch.arange(100, 116).repeat(4) << 7)).to(torch.int16)
        st = exponent_stats(words, book)
        self.assertAlmostEqual(st["top16_coverage_self"], 1440 / 1600)
        self.assertAlmostEqual(st["top16_coverage_calibrated"], 160 / 1600)
        self.assertEqual(st["distinct_exponents"], 26)
        self.assertEqual(st["values"], 1600)
        self.assertGreater(st["exponent_entropy_bits"], 4.0)


class FetchTextsTests(unittest.TestCase):
    def test_renderings(self):
        """scripts/fetch_eval_texts.py: the plain-text renderings of SplitZip's datasets (no download)."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import fetch_eval_texts as ft
        self.assertEqual(ft.render("wikitext2", [{"text": " = A = \n"}, {"text": ""}, {"text": " b \n"}]),
                         " = A = \n\n\n\n\n b \n")
        self.assertEqual(ft.render("humaneval", [{"prompt": "def f():\n", "canonical_solution": "    return 1\n"}] * 2),
                         "def f():\n    return 1\n\n\ndef f():\n    return 1\n")
        self.assertEqual(ft.render("gsm8k", [{"question": "q?", "answer": "a #### 4"}]), "Question: q?\nAnswer: a #### 4")
        rows = [{"subject": s, "question": f"{s} {i}", "choices": ["w", "x", "y", "z"], "answer": i % 4}
                for s in ("b_sub", "a_sub") for i in range(2)]
        m = ft.render("mmlu", rows)
        order = [line for line in m.split("\n") if line.endswith((" 0", " 1"))]
        self.assertEqual(order, ["a_sub 0", "b_sub 0", "a_sub 1", "b_sub 1"])      # round-robin over subjects
        self.assertIn("about a sub.\n\na_sub 0\nA. w\nB. x\nC. y\nD. z\nAnswer: A", m)
        self.assertIn("Answer: B", m)


@unittest.skipIf(torch is None, "PyTorch is required")
class MoreBaselinesTests(unittest.TestCase):
    """kv_compare/more_baselines.py without the authors' code (that runs on the GPU machines): the parameter rule,
    the computed bounds, and the bookkeeping around a codec object."""

    def test_huffman_and_bounds(self):
        from kv_compare.more_baselines import exponent_bounds, huffman_lengths
        lens = huffman_lengths([5, 9, 12, 13, 16, 45])         # the textbook example
        self.assertEqual([lens[i] for i in range(6)], [4, 4, 3, 3, 3, 1])
        self.assertEqual(huffman_lengths([0, 7, 0]), {1: 1})
        # words whose exponents take two values equally often: entropy 1 bit, Huffman 1 bit
        e = torch.tensor([120, 121] * 64)
        w = ((e << 7) | 3).to(torch.int16).view(1, 16, 8)          # L=1, T=16, W=8 (C=4)
        b = exponent_bounds(w, 4, [(0, 16)])
        self.assertAlmostEqual(b["entropy"], 9.0)
        self.assertAlmostEqual(b["huffman"], 9.0)

    def test_adacode_rule(self):
        from kv_compare.more_baselines import adacode_best_bits
        def hist(d):
            h = [0] * 256
            for k, v in d.items():
                h[k] = v
            return h
        # peaked: 2 bits (top 3 coded, 100 outliers: 2000 + 800) beats 3 bits (3000 + 8); the window ends at 122
        self.assertEqual(adacode_best_bits(hist({118: 10, 119: 40, 120: 200, 121: 400, 122: 300, 123: 40, 124: 9,
                                                 125: 1})), (122, 2))
        # flatter: 3 bits (3000 + 80) beats 2 bits (2000 + 3600) and 4 bits (4000); the 7 coded end at 124
        self.assertEqual(adacode_best_bits(hist({118: 100, 119: 150, 120: 200, 121: 200, 122: 150, 123: 100,
                                                 124: 90, 125: 10})), (124, 3))

    def test_codec_bookkeeping_with_a_stand_in(self):
        from kv_compare import more_baselines as mb

        class Fake:                                   # AdaCode's CompressorBitMap interface, storing a copy
            def __init__(self, right=129, nbits=3):
                self.nbits = nbits
            def compress(self, x):
                self.x = x.clone()
            def uncompress(self):
                return self.x
            def get_memory(self):
                return {"compressed_memory": self.x.numel() * (8 + self.nbits) // 8}
        ada = mb.AdaCode.__new__(mb.AdaCode)
        ada.Compressor = Fake
        g = torch.Generator().manual_seed(0)
        words = (torch.randn(2, 32, 16, generator=g) * 0.1).to(torch.bfloat16).view(torch.int16)
        ada.calibrate(words, 8)
        self.assertEqual(len(ada.params), 2)
        t = lambda fn: (fn(), 0.001)[1]
        r = ada.measure(words, [(0, 16), (16, 32)], 8, t)
        self.assertTrue(r["bit_exact"])
        self.assertEqual(r["streams"], 2)
        self.assertAlmostEqual(r["bits"], 8 + r["code_bits"][0], places=6)

    def test_measure_case_rows_and_report(self):
        import tempfile
        from kv_compare import context_sweep as cs

        class FakeIbp:
            def measure(self, words, spans, device, timer):
                return {"bit_exact": True, "bits": 12.25, "encode_ms": 1.0, "decode_ms": 2.0, "raw_copy_ms": 1.5,
                        "streams": len(spans)}

        class Study:
            C, hd, W = 8, 4, 16
        words = (torch.randn(2, 32, 16) * 0.1).to(torch.bfloat16).view(torch.int16)
        consts = {"cos": torch.ones(32, 2), "sin": torch.zeros(32, 2)}
        st = Study()
        st.rope = type("R", (), {"tables": staticmethod(lambda p: (torch.ones(len(p), 2), torch.zeros(len(p), 2)))})()
        rows = cs.measure_case(st, consts, None, words, torch.arange(32), torch.device("cpu"), log=lambda m: None,
                               only={"IBP (authors' codec), host to GPU",
                                     "exponent coders' floor: 8 + exponent entropy",
                                     "exponent coders' floor: 8 + Huffman(exponent), DFloat11 / ZipNN format"},
                               extras={"ibp": FakeIbp()}, bounds=True, baseline_stream_tokens=16)
        names = [r["method"] for r in rows]
        self.assertEqual(sorted(names), sorted(cs.extra_method_names({"ibp": FakeIbp()}, True)))
        for r in rows:
            self.assertNotIn("error", r)
        sys.path.insert(0, str(ROOT / "scripts"))
        import bench_llama_compare as b
        out = Path(tempfile.mkdtemp())
        (out / "captures").mkdir()
        (out / "captures" / "meta.json").write_text(json.dumps({"model": "tiny"}))
        res = {"calibration": {"set": "wikitext", "documents": ["wtrain0"]},
               "cases": {"gsm8k": {"dataset": "GSM8K test", "tokens": 32, "batch": 1, "windows": ["ds_gsm8k_r0"],
                                   "values": words.numel(), "coverage": {}, "rows": rows}}, "links_gbps": [10]}
        (out / "datasets.json").write_text(json.dumps(res))
        text = "\n".join(b.dataset_lines(out, "datasets.json"))
        self.assertIn("(computed)", text)
        self.assertIn("[raw copy 1.50]", text)
        self.assertNotIn("NOT exact", text)


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
