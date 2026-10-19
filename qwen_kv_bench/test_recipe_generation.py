"""Held-out tiny Qwen continuation through each new exact recipe family.

Random local BF16 weights establish integration and exactness, not pretrained
compression quality or accelerator performance. Dictionary fitting uses three
separate TRAIN captures; the continuation prefix is captured only after fitting.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_kv_bench import exponent_recipe, gpu_ops, residual_dictionary
from qwen_kv_bench.model_runtime import capture, generation_check
from qwen_kv_bench.recipe_sweep import insight_options
from qwen_kv_bench.runner import _native_callback


class RecipeGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(20261019)
        config = Qwen3Config(vocab_size=79, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=64, attention_dropout=0.)
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval().to(dtype=torch.bfloat16)
        cls.tile_tokens = 7
        # Distinct inputs, lengths and token hashes. No tokenizer or pretrained
        # loader is needed, so the complete fixture runs without network access.
        training_ids = [[1 + ((i * step + offset) % 77) for i in range(length)]
                        for step, offset, length in ((3, 2, 23), (5, 11, 24), (7, 29, 25))]
        train_caches = [capture(cls.model, ids, "cpu", chunk_tokens=8)["matrices"]
                        for ids in training_ids]
        training_metadata = [dict(split="train", name=f"tiny_qwen_train_{i}")
                             for i in range(3)]
        identity = dict(model="local-random-tiny-qwen3-bf16", config=config.to_dict())
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.dictionary_models = {}
        for count, layout in ((16, "channel"), (64, "time")):
            cls.dictionary_models[(count, layout)] = residual_dictionary.train(
                train_caches, Path(directory.name) / f"{count}_{layout}",
                tile_tokens=cls.tile_tokens, layout=layout, n_centroids=count,
                max_samples_per_key=128, iterations=2, seed=71,
                training_metadata=training_metadata, model_identity=identity)
        # Capture DEV only after the two dictionaries are frozen. Nineteen
        # tokens leave a five-token final tile, exercising partial record tails.
        cls.dev = capture(cls.model, [1 + ((i * 11 + 47) % 77) for i in range(19)],
                          "cpu", chunk_tokens=8)

    def assert_continuation(self, spec, state):
        self.assertEqual(self.dev["geometry"]["tokens"], 19)
        self.assertEqual(self.dev["geometry"]["head_dim"], 16)
        callback = _native_callback(self.dev["matrices"], spec, state,
                                    "cpu", self.tile_tokens)
        decoded = callback()
        self.assertEqual(len(decoded), 2)
        for original_pair, restored_pair in zip(self.dev["matrices"], decoded):
            for original, restored in zip(original_pair, restored_pair):
                np.testing.assert_array_equal(restored, original)
        checked = generation_check(self.model, self.dev["matrices"], callback,
            "cpu", first_token=self.dev["first_token"], steps=2)
        self.assertEqual(checked["steps"], 2)
        self.assertTrue(checked["all_logits_bitwise_equal"])
        self.assertTrue(checked["all_tokens_equal"])
        self.assertTrue(checked["all_appended_KV_bitwise_equal"])
        self.assertEqual(checked["max_abs_logit_difference"], 0.)
        for step in checked["per_step"]:
            self.assertTrue(step["logits_finite"])
            self.assertTrue(step["logits_bitwise_equal"])
            self.assertTrue(step["tokens_equal"])
            self.assertTrue(step["appended_full_cache_bitwise_equal"])

    def test_median_exponent_time_full_prefix_continuation(self):
        spec = next(option for option in exponent_recipe.options()
                    if option["options"]["predictor"] == "median"
                    and option["options"]["layout"] == "time")
        self.assert_continuation(spec, gpu_ops.make_state())

    def test_affine_time_layout_full_prefix_continuation(self):
        spec = next(option for option in gpu_ops.insight_layout_options()
                    if option["options"]["predictor"] == "affine"
                    and option["options"]["layout"] == "time"
                    and option["options"]["group_size"] == 8)
        self.assert_continuation(spec, gpu_ops.make_state())

    def assert_dictionary_continuation(self, count, layout):
        model = self.dictionary_models[(count, layout)]
        self.assertTrue(model.metadata["training_only"])
        self.assertEqual(len(model.metadata["training_cases"]), 3)
        self.assertTrue(all(case["split"] == "train"
                            for case in model.metadata["training_cases"]))
        self.assertEqual(len(set(model.metadata["training_hashes"])), 3)
        spec = next(option for option in insight_options()
                    if option["method"] == "prefill_dictionary"
                    and option["dictionary_size"] == count
                    and option["options"]["layout"] == layout)
        state = residual_dictionary.make_state(model)
        self.assertEqual(state.state_id, model.model_id)
        self.assert_continuation(spec, state)

    def test_dictionary16_channel_full_prefix_continuation(self):
        self.assert_dictionary_continuation(16, "channel")

    def test_dictionary64_time_full_prefix_continuation(self):
        self.assert_dictionary_continuation(64, "time")


if __name__ == "__main__":
    unittest.main()
