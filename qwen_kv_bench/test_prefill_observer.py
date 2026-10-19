"""Live-observer protocol and equality to post-capture packet accounting."""
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import gpu_ops as g
from qwen_kv_bench.measurements import measure_native
from qwen_kv_bench.prefill_observer import PrefillRecipeObserver


class PrefillObserverTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.specs = [s for s in g.native_options() if s["method"] == "prefill"]
        rng = np.random.default_rng(348)
        self.matrices = [tuple(rng.integers(0, 65536, (19, 17), dtype=np.uint16)
                               for _ in range(2)) for _ in range(2)]

    def cache(self, end):
        return tuple(tuple(torch.from_numpy(w[:end].copy()).view(torch.bfloat16)
                           .reshape(1, end, 1, 17).permute(0, 2, 1, 3).contiguous()
                           for w in pair) for pair in self.matrices)

    def test_live_sample_matches_postcapture_exact_records_and_bytes(self):
        observer = PrefillRecipeObserver(self.specs, 19, 7, "cpu", max_records=8)
        observer(self.cache(5), 0, 5, 19)
        self.assertFalse(observer.report()["complete"])
        observer(self.cache(11), 5, 11, 19)
        observer(self.cache(19), 11, 19, 19)
        report = observer.report()
        self.assertTrue(report["complete"])
        self.assertEqual(report["chunks_observed"], 3)
        self.assertEqual(report["tokens_observed"], 19)
        self.assertEqual(len(report["results"]), 8)
        with tempfile.TemporaryDirectory() as directory:
            for spec, row in zip(self.specs, report["results"]):
                self.assertEqual(row["status"], "ok")
                m = row["measurement"]
                self.assertEqual(m["records_checked"], 8)
                self.assertFalse(m["source_KV_cpu_staging"])
                self.assertEqual(m["decoder_state_bytes"], 0)
                self.assertTrue(m["full_raw_cache_retained"])
                expected = measure_native(self.matrices, spec, g.make_state(), "cpu", directory,
                    repeats=1, tile_tokens=7, max_records=8, io_bytes_limit=0)
                for key in ("raw_payload_bytes", "serialized_bytes", "component_bytes",
                            "estimated_full_cache_serialized_bytes", "records_total"):
                    self.assertEqual(m[key], expected[key], (spec["name"], key))
        json.dumps(report)

    def test_full_coverage_and_invalid_chunk_or_recipe(self):
        observer = PrefillRecipeObserver(self.specs[:1], 19, 7, "cpu", max_records=None)
        observer(self.cache(19), 0, 19, 19)
        row = observer.report()["results"][0]
        self.assertEqual(row["measurement"]["records_checked"], 12)
        self.assertEqual(row["measurement"]["coverage"], "full")
        with self.assertRaises(ValueError):
            observer(self.cache(19), 0, 19, 19)
        with self.assertRaises(ValueError):
            PrefillRecipeObserver([], 19, 7, "cpu")
        with self.assertRaises(ValueError):
            PrefillRecipeObserver([dict(name="raw", method="raw")], 19, 7, "cpu")

    def test_corrupt_decode_is_a_failure_and_never_a_pass(self):
        observer = PrefillRecipeObserver(self.specs[:1], 19, 7, "cpu", max_records=2)
        decode = g.decode
        def corrupted(*args):
            words = decode(*args).to(torch.int32)
            words[0, 0] ^= 1
            return words.to(torch.uint16)
        with mock.patch.object(g, "decode", side_effect=corrupted):
            observer(self.cache(19), 0, 19, 19)
        row = observer.report()["results"][0]
        self.assertEqual(row["status"], "failed_correctness")
        self.assertNotIn("measurement", row)

    def test_tiny_qwen_live_recipes_and_every_variant_continuation(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from qwen_kv_bench.model_runtime import capture, generation_check
        from qwen_kv_bench.runner import _native_callback
        torch.manual_seed(531)
        config = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=96,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=64, attention_dropout=0.)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
        ids = list(range(1, 24))
        plain = capture(model, ids, "cpu", chunk_tokens=9)
        observer = PrefillRecipeObserver(self.specs, len(ids), 7, "cpu", max_records=None)
        live = capture(model, ids, "cpu", chunk_tokens=9, prefill_observer=observer)
        self.assertEqual(plain["first_token"], live["first_token"])
        for a, b in zip(plain["matrices"], live["matrices"]):
            for left, right in zip(a, b):
                np.testing.assert_array_equal(left, right)
        for row in live["prefill_recipes"]["results"]:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["measurement"]["coverage"], "full")
        for spec in self.specs:
            callback = _native_callback(live["matrices"], spec, g.make_state(), "cpu", 7)
            checked = generation_check(model, live["matrices"], callback, "cpu",
                first_token=live["first_token"], steps=2)
            self.assertTrue(checked["all_logits_bitwise_equal"], spec["name"])
            self.assertTrue(checked["all_tokens_equal"], spec["name"])
            self.assertTrue(checked["all_appended_KV_bitwise_equal"], spec["name"])
            self.assertTrue(all(s["logits_finite"] for s in checked["per_step"]))


if __name__ == "__main__":
    unittest.main()
