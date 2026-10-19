"""Live-prefill observer ordering and read-only correctness, without downloads."""
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_kv_bench import model_runtime as runtime


class _RecordingModel:
    """Deterministic full-prefix cache fixture with visible chunk boundaries."""

    def __init__(self, events):
        self.config = SimpleNamespace(num_hidden_layers=1)
        self.events = events
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask, past_key_values,
                 use_cache, logits_to_keep):
        self.calls += 1
        previous = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        end = previous + input_ids.shape[1]
        self.events.append(("model", previous, end))
        assert attention_mask.shape == (1, end)
        assert use_cache and logits_to_keep == 1
        rows = input_ids.to(torch.bfloat16).unsqueeze(1).unsqueeze(-1).repeat(1, 1, 1, 2)
        if past_key_values is not None:
            rows = torch.cat((past_key_values[0][0], rows), dim=2)
        cache = ((rows, rows + 1),)
        return SimpleNamespace(past_key_values=cache,
                               logits=torch.tensor([[[0, 1, 2]]], dtype=torch.bfloat16))


class PrefillCaptureTests(unittest.TestCase):
    def test_every_chunk_is_observed_on_full_prefix_before_next_chunk_or_export(self):
        events = []
        model = _RecordingModel(events)
        ids = [1, 7, 2, 9, 1, 4, 8]
        observations = []
        original_export = runtime.cache_to_matrices

        class Observer:
            def __call__(self, cache, start, end, total):
                events.append(("observer", start, end))
                self_test.assertTrue(torch.is_inference_mode_enabled())
                self_test.assertEqual(total, len(ids))
                self_test.assertEqual(runtime.cache_geometry(cache)["tokens"], end)
                keys = runtime._pairs(cache)[0][0]
                self_test.assertEqual(keys.device, torch.device("cpu"))
                self_test.assertEqual(keys[0, 0, :, 0].tolist(), ids[:end])
                observations.append((start, end, total))

            def report(self):
                events.append(("report",))
                return {"chunks_observed": len(observations), "complete_raw_cache_retained": True}

        self_test = self

        def export(cache):
            events.append(("export",))
            self.assertEqual(observations, [(0, 3, 7), (3, 6, 7), (6, 7, 7)])
            return original_export(cache)

        with mock.patch.object(runtime, "cache_to_matrices", side_effect=export):
            result = runtime.capture(model, ids, "cpu", chunk_tokens=3,
                                     prefill_observer=Observer())
        self.assertEqual(events, [("model", 0, 3), ("observer", 0, 3),
                                  ("model", 3, 6), ("observer", 3, 6),
                                  ("model", 6, 7), ("observer", 6, 7),
                                  ("export",), ("report",)])
        self.assertEqual(result["prefill_recipes"]["chunks_observed"], 3)
        self.assertTrue(result["timings"]["prefill"]["observer_included"])
        self.assertIn("included in prefill", result["prefill_observer_scope"])
        self.assertIn("raw device KV remains resident", result["prefill_observer_scope"])

    def test_callable_observer_does_not_require_report(self):
        model = _RecordingModel([])
        calls = []
        result = runtime.capture(model, [1, 2, 3], "cpu", chunk_tokens=2,
            prefill_observer=lambda cache, start, end, total: calls.append((start, end, total)))
        self.assertEqual(calls, [(0, 2, 3), (2, 3, 3)])
        self.assertNotIn("prefill_recipes", result)
        self.assertTrue(result["timings"]["prefill"]["observer_included"])

    def test_observer_failure_propagates_before_next_chunk_or_export(self):
        events = []
        model = _RecordingModel(events)

        def observer(cache, start, end, total):
            events.append(("observer", start, end))
            if start == 2:
                raise RuntimeError("recipe construction failed")

        with mock.patch.object(runtime, "cache_to_matrices") as exporter:
            with self.assertRaisesRegex(RuntimeError, "recipe construction failed"):
                runtime.capture(model, [1, 2, 3, 4, 5], "cpu", chunk_tokens=2,
                                prefill_observer=observer)
            exporter.assert_not_called()
        self.assertEqual(model.calls, 2)
        self.assertEqual(events, [("model", 0, 2), ("observer", 0, 2),
                                  ("model", 2, 4), ("observer", 2, 4)])

    def test_default_capture_has_no_observer_report(self):
        result = runtime.capture(_RecordingModel([]), [1, 2, 3], "cpu", chunk_tokens=2)
        self.assertFalse(result["timings"]["prefill"]["observer_included"])
        self.assertNotIn("prefill_recipes", result)
        self.assertNotIn("prefill_observer_scope", result)

    def test_read_only_observer_preserves_tiny_qwen_tokens_and_every_kv_bit(self):
        torch.manual_seed(83)
        config = Qwen3Config(hidden_size=32, num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, num_hidden_layers=2,
                             intermediate_size=48, vocab_size=47, max_position_embeddings=128)
        config._attn_implementation = "sdpa"
        model = Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        ids = [1, 7, 2, 9, 1, 4, 8]
        reference = runtime.capture(model, ids, "cpu", chunk_tokens=3)
        observed = []

        def read_only(cache, start, end, total):
            pairs = runtime._pairs(cache)
            self.assertEqual(len(pairs), config.num_hidden_layers)
            self.assertEqual(total, len(ids))
            self.assertTrue(all(tensor.shape[2] == end for pair in pairs for tensor in pair))
            # Device-local reads, including an out-of-place computation, must
            # leave both the prompt cache and the continuation input unchanged.
            observed.append((start, end, sum(float(k.float().square().sum()) for k, _ in pairs)))

        result = runtime.capture(model, ids, "cpu", chunk_tokens=3, prefill_observer=read_only)
        self.assertEqual([(start, end) for start, end, _ in observed], [(0, 3), (3, 6), (6, 7)])
        self.assertEqual(result["first_token"], reference["first_token"])
        self.assertEqual(result["geometry"], reference["geometry"])
        self.assertEqual(result["raw_payload_bytes"], reference["raw_payload_bytes"])
        self.assertTrue(runtime._matrices_equal(result["matrices"], reference["matrices"]))

    def test_observer_and_report_contract_fail_explicitly(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            runtime.capture(_RecordingModel([]), [1], "cpu", prefill_observer=object())

        class InvalidReport:
            def __call__(self, *args):
                pass

            def report(self):
                return ["not a report dict"]

        with self.assertRaisesRegex(TypeError, "must return a dict"):
            runtime.capture(_RecordingModel([]), [1], "cpu", prefill_observer=InvalidReport())


if __name__ == "__main__":
    unittest.main()
