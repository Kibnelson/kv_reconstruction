"""Tests of T2 (kv_compare/t2.py), kept apart from T's tests: every mode restores the KV bit for bit with the
receiver's layer 0, the streams are T's streams without layer 0, the per-request set and the fallback to T work.
The Triton kernels run in Triton's CPU interpreter (slow: a few minutes).

    python -m unittest kv_compare.test_t2 -v
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

T2_CHECK = r"""
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
from kv_compare.t2 import Layer0Recomputed, T2
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=3, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, max_position_embeddings=2048, rope_theta=500000.0)
model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
names = [f"train{i}" for i in range(4)] + ["tables", "test", "alt0", "alt1", "alttab"]
root = Path(tempfile.mkdtemp()) / "c"
reports = []
for i, n in enumerate(names):
    rng = np.random.default_rng(i)
    p = None if not n.startswith("alt") else rng.dirichlet(np.ones(97) * 0.2)     # another kind of text
    ids = [int(v) for v in rng.choice(97, size=64, p=p)]
    reports.append(capture_ids(model, ids, root, n, torch.device("cpu")))
(root / "meta.json").write_text(json.dumps(dict(model_meta(model, "tiny", None), tokens=64, captures=reports)))
cpu = torch.device("cpu")
study = Study(root, "cpu", names[:4], "tables", ["test"], log=lambda m: None)
alt = Study(root, "cpu", ["alt0", "alt1"], "alttab", ["test"], log=lambda m: None)
sets = {}
for name, st in (("main", study), ("alt", alt)):
    c = fused.prepare(st, log=lambda m: None)
    sets[name] = (c, prepare_rank(st, c, 32, log=lambda m: None), prepare_groups(st, c, [2, 4], log=lambda m: None))
L, W, C, hd, T = study.L, study.W, study.C, study.hd, 64
d = study.tests[0]
words = torch.stack([(lambda w: (w - 65536 * (w >= 32768).to(torch.int32)).to(torch.int16))(d.words(l, cpu))
                     for l in range(L)])
K, V = ti.layer0_kv(model, d.ids.tolist(), cpu)
l0 = torch.cat([K, V], 1)
l0 = (l0 - 65536 * (l0 >= 32768).to(torch.int32)).to(torch.int16)
assert torch.equal(l0, words[0]), "the lean recompute is not the captured layer 0"
consts, rank, groups = sets["main"]
for mode in ("dense", "4g", "rank256"):
    t = ti.mode_codec(consts, rank, groups, mode, C, hd, T)
    t2 = ti.mode_codec(consts, rank, groups, mode, C, hd, T, cls=Layer0Recomputed)
    xall, prior = t.model(words)
    lw, lw2 = torch.zeros(L, dtype=torch.int64), torch.zeros(L, dtype=torch.int64)
    buf, starts = t.encode(words, xall, prior, layer_words=lw)
    buf2, starts2 = t2.encode(words, xall, prior, layer_words=lw2)
    bits, bits2 = t.bits(starts, T), t2.bits(starts2, T)
    assert torch.equal(t.decode(buf, starts, T), words), mode
    dec = t2.decode(buf2, starts2, T, l0)
    assert torch.equal(dec, words), f"{mode}: T2 does not restore the KV"
    got = {}
    t2.decode(buf2, starts2, T, l0, sink=lambda l, w: got.__setitem__(l, w.clone()))
    assert sorted(got) == list(range(L)) and all(torch.equal(got[l], words[l]) for l in range(L)), f"{mode}: sink"
    assert lw2[0] == 0 and torch.equal(lw2[2:], lw[2:]), (mode, lw.tolist(), lw2.tolist())
    assert bits2 < bits, (mode, bits, bits2)
    print("ok", mode, bits, bits2, lw.tolist(), lw2.tolist())
# the wrapper: two sets, pick on layers 1..2, the fallback to T
builders = {n: (lambda n_layers, rec, s=s: ti.mode_codec(s[0], s[1], s[2], "dense", C, hd, T,
                                                          cls=Layer0Recomputed if rec else None, layers=n_layers))
            for n, s in sets.items()}
codec = T2(builders, pick_layers=2)
name, pbits = codec.pick(words)
assert name == min(pbits, key=pbits.get)
parts, header = codec.encode(words, True, name)
assert header == {"set": name, "layer0": "recomputed", "T": T}
assert torch.equal(codec.decode(parts, header, l0=l0), words)
fparts, fheader = codec.encode(words, False, name)
assert fheader["layer0"] == "coded"
assert torch.equal(codec.decode(fparts, fheader), words)
tt = ti.mode_codec(sets[name][0], None, None, "dense", C, hd, T)
xa, pa = tt.model(words)
assert codec.bits(fparts, fheader) == tt.bits(tt.encode(words, xa, pa)[1], T), "the fallback is not T's stream"
assert codec.bits(parts, header) < codec.bits(fparts, fheader)
print("ok wrapper", name, pbits)
"""


@unittest.skipIf(importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None,
                 "PyTorch and Triton are required")
class T2InterpretedTests(unittest.TestCase):
    def test_t2_round_trips(self):
        r = subprocess.run([sys.executable, "-c", T2_CHECK, str(ROOT)], capture_output=True, text=True, timeout=5400)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:] + r.stdout[-2000:])
        self.assertEqual(r.stdout.count("ok"), 4, r.stdout)


if __name__ == "__main__":
    unittest.main()
