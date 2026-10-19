"""CPU-only checks with real tiny Qwen/GPT-like configs and random weights."""
import unittest
from unittest import mock
import hashlib
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_kv_bench import model_runtime as runtime


class ModelRuntimeTests(unittest.TestCase):
    def test_modern_layers_without_legacy_converter_are_exact(self):
        words = torch.arange(256, dtype=torch.int32).to(torch.uint16).reshape(1, 2, 1, 128)
        k = words.view(torch.bfloat16)
        v = words.to(torch.int32).flip(-1).to(torch.uint16).view(torch.bfloat16)
        modern = SimpleNamespace(layers=[SimpleNamespace(keys=k, values=v)])
        # This has the v5 storage interface and deliberately no converter.
        self.assertFalse(hasattr(modern, "to_legacy_cache"))
        self.assertEqual(runtime.cache_geometry(modern)["head_dim"], 128)
        pairs = runtime._pairs(modern)
        self.assertIs(pairs[0][0], k)
        self.assertIs(pairs[0][1], v)
        modern_words = runtime.cache_to_matrices(modern)
        legacy_words = runtime.cache_to_matrices(((k, v),))
        self.assertTrue(runtime._matrices_equal(modern_words, legacy_words))

    def test_legacy_cache_interfaces_remain_supported(self):
        k = torch.ones((1, 2, 3, 8), dtype=torch.bfloat16)
        v = k + 1
        expected = ((k, v),)
        candidates = [expected, [(k, v)], SimpleNamespace(to_legacy_cache=lambda: expected),
                      SimpleNamespace(key_cache=[k], value_cache=[v])]
        for cache in candidates:
            with self.subTest(cache_type=type(cache).__name__):
                result = runtime._pairs(cache)
                self.assertIs(result[0][0], k)
                self.assertIs(result[0][1], v)
        both = SimpleNamespace(layers=[SimpleNamespace(keys=k, values=v)],
                               to_legacy_cache=lambda: self.fail("Modern layers must be preferred"))
        self.assertIs(runtime._pairs(both)[0][0], k)

    def test_invalid_cache_layers_fail_with_class_and_version(self):
        k = torch.ones((1, 2, 3, 8), dtype=torch.bfloat16)
        candidates = [SimpleNamespace(layers=[SimpleNamespace(keys=k, values=None)]),
                      SimpleNamespace(layers=[SimpleNamespace(keys=None, values=None)]),
                      SimpleNamespace(layers=[SimpleNamespace(keys="wrong", values=k)]),
                      SimpleNamespace(), [(k,)]]
        for cache in candidates:
            with self.subTest(cache_type=type(cache).__name__):
                with self.assertRaisesRegex(TypeError, r"transformers="):
                    runtime._pairs(cache)
        with self.assertRaisesRegex(ValueError, "layer counts differ"):
            runtime._pairs(SimpleNamespace(key_cache=[k], value_cache=[]))

    def test_auto_does_not_silently_use_cpu(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False), \
             mock.patch.object(torch.xpu, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "explicitly"):
                runtime.select_device("auto")
        self.assertEqual(str(runtime.select_device("cpu")), "cpu")

    def test_doctor_cpu_without_network(self):
        info = runtime.doctor("cpu")
        self.assertEqual(info["selected_device"], "cpu")
        self.assertFalse(info["model_download_performed"])
        self.assertTrue(info["bf16_probe"]["success"])
        self.assertTrue(all(probe["success"] for probe in info["operation_probes"].values()))
        self.assertIsNone(info["memory"]["peak_allocated_bytes"])

    def test_auxiliary_probe_failure_does_not_veto_bf16(self):
        original_tensor = torch.tensor
        def reject_float64(*args, **kwargs):
            if kwargs.get("dtype") == torch.float64:
                raise RuntimeError("simulated backend without float64")
            return original_tensor(*args, **kwargs)
        with mock.patch.object(torch, "tensor", side_effect=reject_float64):
            info = runtime.doctor("cpu")
        self.assertTrue(info["bf16_probe"]["success"])
        self.assertFalse(info["operation_probes"]["float64_pointwise"]["success"])
        self.assertIn("simulated", info["operation_probes"]["float64_pointwise"]["error"])
        self.assertTrue(info["operation_probes"]["int64_bitwise"]["success"])

    def test_explicit_head_dim_128_not_hidden_div_queryheads(self):
        config = Qwen3Config(hidden_size=160, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=128, num_hidden_layers=2,
                             intermediate_size=192, vocab_size=31)
        rng = np.random.default_rng(6)
        matrices = [(rng.integers(0, 65536, size=(5, 256), dtype=np.uint16),
                     rng.integers(0, 65536, size=(5, 256), dtype=np.uint16)) for _ in range(2)]
        cache = runtime.cache_from_matrices(matrices, config, "cpu")
        self.assertEqual(runtime.cache_geometry(cache)["head_dim"], 128)
        recovered = runtime.cache_to_matrices(cache)
        self.assertTrue(runtime._matrices_equal(matrices, recovered))
        matrices[0][0][0, 0] ^= np.uint16(1)
        self.assertFalse(runtime._matrices_equal(matrices, runtime.cache_to_matrices(cache)))
        with self.assertRaisesRegex(ValueError, "head dimension"):
            runtime.cache_from_matrices([(np.zeros((5, 80), np.uint16), np.zeros((5, 80), np.uint16))
                                         for _ in range(2)], config)

    def test_tiny_random_qwen_chunked_capture_and_generation(self):
        # Functional fixture, not a pretrained-model compression experiment.
        torch.manual_seed(8)
        config = Qwen3Config(hidden_size=32, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, num_hidden_layers=2,
                             intermediate_size=48, vocab_size=47, max_position_embeddings=128)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        ids = [1, 7, 2, 9, 1, 4, 8]
        result = runtime.capture(model, ids, "cpu", chunk_tokens=3)
        self.assertEqual(result["geometry"]["tokens"], len(ids))
        self.assertEqual(result["geometry"]["head_dim"], 16)
        self.assertEqual(result["matrices"][0][0].shape, (7, 32))
        self.assertEqual(result["raw_payload_bytes"], 2 * 2 * 7 * 32 * 2)
        check = runtime.generation_check(model, result["matrices"],
                    lambda: [(k.copy(), v.copy()) for k, v in result["matrices"]],
                    "cpu", first_token=result["first_token"], steps=2)
        self.assertTrue(check["all_logits_bitwise_equal"])
        self.assertTrue(check["all_tokens_equal"])
        self.assertTrue(check["all_appended_KV_bitwise_equal"])
        self.assertEqual(check["recent_raw_cpu_bytes"], 2 * 2 * 2 * 32 * 2)

    def test_timing_and_unique_model_storage(self):
        value, timing = runtime.timed_call(lambda: 19, "cpu")
        self.assertEqual(value, 19)
        self.assertIsNone(timing["device_event_seconds"])
        self.assertGreaterEqual(timing["wall_seconds"], 0)
        layer = torch.nn.Linear(4, 3, bias=False).to(torch.bfloat16)
        self.assertEqual(runtime.model_storage_bytes(layer), 4 * 3 * 2)

    def test_loader_capability_failure_precedes_model_download(self):
        with mock.patch.object(runtime, "select_device", side_effect=RuntimeError("no accelerator")), \
             mock.patch("transformers.AutoConfig.from_pretrained") as config_loader:
            with self.assertRaisesRegex(RuntimeError, "no accelerator"):
                runtime.load_model()
            config_loader.assert_not_called()

    def test_local_config_identity_without_weight_download(self):
        config = Qwen3Config(hidden_size=16, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=16, num_hidden_layers=1,
                             intermediate_size=24, vocab_size=17)
        model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            expected_sha = hashlib.sha256((Path(directory) / "config.json").read_bytes()).hexdigest()
            with mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model), \
                 mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=object()):
                _, _, identity = runtime.load_model(directory, revision=None, device="cpu", local_files_only=True)
            self.assertEqual(identity["config_sha256"], expected_sha)
            self.assertTrue(identity["local_model"])
            self.assertIsNone(identity["resolved_commit"])
            self.assertFalse(identity["local_weights_content_hashed"])

    def test_tokenizer_validation_precedes_weights_and_propagates_failure(self):
        config = Qwen3Config(hidden_size=16, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=16, num_hidden_layers=1,
                             intermediate_size=24, vocab_size=17)
        model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        tokenizer = object()
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            for fail in (False, True):
                with self.subTest(invalid_inputs=fail):
                    stages = []
                    def validate(actual):
                        self.assertIs(actual, tokenizer)
                        stages.append("validate")
                        if fail:
                            raise ValueError("invalid DEV token budget")
                    def load_weights(*args, **kwargs):
                        stages.append("weights")
                        return model
                    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                         mock.patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=load_weights) as loader:
                        if fail:
                            with self.assertRaisesRegex(ValueError, "invalid DEV"):
                                runtime.load_model(directory, revision=None, device="cpu", local_files_only=True,
                                                   tokenizer_validator=validate)
                            loader.assert_not_called()
                            self.assertEqual(stages, ["validate"])
                        else:
                            result = runtime.load_model(directory, revision=None, device="cpu", local_files_only=True,
                                                        tokenizer_validator=validate)
                            self.assertIs(result[1], tokenizer)
                            self.assertEqual(stages, ["validate", "weights"])


if __name__ == "__main__":
    unittest.main()
