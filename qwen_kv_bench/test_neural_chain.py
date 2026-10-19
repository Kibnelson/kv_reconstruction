"""Protocol tests for actual indexed cross-layer pages, without model downloads."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from qwen_kv_bench import neural_chain as chain, producer_codec


class _Predictor:
    device = torch.device("cpu")
    model_id = "synthetic-fixed-network"
    state_id = "synthetic-fixed-network/runtime"
    serialized_bytes = 64
    physical_nbytes = 40
    execution_contract = {"test_only": True}

    def __init__(self):
        self.offset = 0.
        self.calls = []

    def predict(self, layer, previous, start):
        self.calls.append((layer, start, previous.detach().cpu().numpy().copy()))
        mu = previous.to(torch.int32).to(torch.float32) / 65536 + layer + start + self.offset
        return mu, torch.zeros_like(mu)


class _LiteralEntropy:
    """Only a chain test double; actual entropy coding has separate tests."""
    @staticmethod
    def encode(words, mu, log_scale):
        return words.astype("<u2").tobytes(), dict(shape=list(words.shape),
            ideal_continuous_bits=16 * words.size, integer_cdf_bits=16 * words.size)

    @staticmethod
    def decode(payload, description, mu, log_scale):
        return np.frombuffer(payload, "<u2").copy().reshape(description["shape"])


class NeuralChainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.predictor = _Predictor()
        spec = next(row for row in producer_codec.recipes() if row["name"] == "producer_kv_fixed127")
        self.anchor = producer_codec.make_state(spec, {}, {}, "cpu")
        rng = np.random.default_rng(891)
        self.matrices = [(rng.integers(0, 65536, (9, 3), dtype=np.uint16),
                          rng.integers(0, 65536, (9, 3), dtype=np.uint16)) for _ in range(3)]

    def encode(self, **kwargs):
        return chain.encode_cache(self.matrices, self.predictor, self.anchor,
            tile_tokens=4, entropy_codec=_LiteralEntropy, **kwargs)

    def decode(self, cache, **kwargs):
        return chain.decode_cache(cache, self.predictor, self.anchor, layers=3,
            tokens=9, channels=3, entropy_codec=_LiteralEntropy, **kwargs)

    def test_full_wire_only_roundtrip_and_state_accounting(self):
        expected = [(k.copy(), v.copy()) for k, v in self.matrices]
        cache, measured = self.encode(max_pairs=None)
        wire = cache.to_bytes()
        self.assertEqual(sum(measured["component_bytes"].values()), len(wire))
        self.assertEqual(measured["decoder_state_bytes"], 64)
        self.assertEqual(measured["records_measured"], 9)
        self.assertEqual(measured["coverage"], "full")
        self.assertTrue(measured["container_roundtrip_verified"])
        # Every nonanchor encoder prediction and verification sees a previous
        # page that was actually decoded. Calls occur twice, not teacher forcing.
        self.assertEqual(len(self.predictor.calls), 12)
        for layer, start, previous in self.predictor.calls:
            np.testing.assert_array_equal(previous, np.concatenate(expected[layer - 1], axis=1)[start:start + 4])
        self.predictor.calls.clear()
        for pair in self.matrices:
            for value in pair:
                value.fill(0)
        self.matrices.clear()
        reopened = chain.NeuralCache.from_bytes(wire)
        # Addressing, rather than insertion order, determines decode order.
        reopened = chain.NeuralCache(tuple(reversed(reopened.pages)), 3, 9, 3, 4)
        decoded, timing = self.decode(reopened, return_metrics=True)
        for original, restored in zip(expected, decoded):
            for a, b in zip(original, restored):
                np.testing.assert_array_equal(a, b)
        self.assertFalse(timing["source_arrays_used"])
        self.assertEqual(timing["maximum_dependency_depth"], 2)
        self.assertLessEqual(timing["peak_raw_dependency_context_bytes"], 2 * 4 * 6 * 2)

    def test_aligned_sample_includes_all_layers_and_tail_accounting(self):
        cache, metric = self.encode(max_pairs=7)
        self.assertEqual(metric["selected_starts"], [0, 8])
        self.assertEqual(metric["records_measured"], 6)
        self.assertEqual(metric["raw_payload_bytes"], 3 * (4 + 1) * 3 * 4)
        self.assertTrue(metric["size_is_estimated"])
        with self.assertRaisesRegex(ValueError, "complete token-tile"):
            self.decode(cache)
        records = self.decode(cache, require_full=False)
        self.assertEqual([(r["start"], r["layer"]) for r in records],
                         [(s, layer) for s in (0, 8) for layer in range(3)])
        self.assertEqual([r["tokens"] for r in records], [4, 4, 4, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "every layer"):
            self.encode(max_pairs=2)

    def test_corruption_identity_prediction_and_dependency_fail_closed(self):
        cache, _ = self.encode(max_pairs=None)
        wire = bytearray(cache.to_bytes()); wire[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            chain.NeuralCache.from_bytes(wire)
        wire = bytearray(cache.to_bytes()); wire[32] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            chain.NeuralCache.from_bytes(wire)
        self.predictor.offset = .125
        with self.assertRaisesRegex(ValueError, "prediction changed"):
            self.decode(cache)
        self.predictor.offset = 0.
        self.predictor.state_id = "changed-model"
        with self.assertRaisesRegex(ValueError, "state identity"):
            self.decode(cache)
        self.predictor.state_id = "synthetic-fixed-network/runtime"
        pages = list(cache.pages)
        meta, payload, _ = chain._read_page(pages[1])
        meta["previous_page_sha256"] = "0" * 64
        pages[1] = chain._page(meta, payload)
        with self.assertRaisesRegex(ValueError, "prerequisite"):
            self.decode(chain.NeuralCache(tuple(pages), 3, 9, 3, 4))
        with self.assertRaisesRegex(ValueError, "preceding layers"):
            chain.NeuralCache(cache.pages[1:], 3, 9, 3, 4)

    @unittest.skipUnless(importlib.util.find_spec("qwen_kv_bench.logistic_reference"),
                         "actual reference entropy module not available")
    def test_actual_entropy_pages_restore_special_bf16_words(self):
        # Network parameters are finite even when source words are not.
        values = np.array([0, 0x8000, 1, 0x7f7f, 0x7f80, 0x7fc1], np.uint16).reshape(3, 2)
        matrices = [(values.copy(), values[::-1].copy()) for _ in range(2)]
        cache, metric = chain.encode_cache(matrices, self.predictor, self.anchor,
            tile_tokens=2, max_pairs=None)
        decoded = chain.decode_cache(cache.to_bytes(), self.predictor, self.anchor,
            layers=2, tokens=3, channels=2)
        for a, b in zip(matrices, decoded):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
        self.assertTrue(metric["exact_verified"])
        self.assertTrue(all(row["integer_cdf_bits"] is not None
                            for row in metric["record_diagnostics"] if row["layer"]))

    @unittest.skipUnless(importlib.util.find_spec("qwen_kv_bench.logistic_reference")
                         and importlib.util.find_spec("qwen_kv_bench.neural_probability"),
                         "real neural/entropy modules not available")
    def test_real_train_frozen_models_actual_entropy_and_source_free_decode(self):
        from qwen_kv_bench import neural_probability
        rng = np.random.default_rng(982)

        def words(tokens):
            values = rng.normal(size=(tokens, 2)).astype(np.float32)
            return (values.view(np.uint32) >> 16).astype(np.uint16)

        def matrices(tokens):
            return [(words(tokens), words(tokens)) for _ in range(2)]

        training = [matrices(6), matrices(7), matrices(8)]
        with tempfile.TemporaryDirectory() as directory:
            models = neural_probability.fit(training, directory,
                model_identity={"test": "synthetic finite BF16 source, not model quality"},
                training_metadata=[dict(split="train", name=str(i)) for i in range(3)],
                head_dim=2, rope_inv_freq=np.ones(1, np.float32), device="cpu",
                width=8, steps=2, batch_tokens=4, modes=("per_layer", "shared"))
            frozen = {str(path): Path(path).read_bytes() for model in models.values()
                      for path in model.artifact_paths}
            # Evaluation is generated only after all fits/artifacts are frozen.
            source = matrices(3)
            expected = [(k.copy(), v.copy()) for k, v in source]
            for mode, model in models.items():
                prepared = model.prepare("cpu")
                cache, measured = chain.encode_cache(source, prepared, self.anchor,
                    tile_tokens=2, max_pairs=None, head_dim=2)
                wire = cache.to_bytes()
                self.assertGreater(measured["decoder_state_bytes"], 0)
                # The decoder receives no TRAIN or TEST source matrices.
                decoded = chain.decode_cache(wire, prepared, self.anchor,
                    layers=2, tokens=3, channels=2)
                for a, b in zip(expected, decoded):
                    for x, y in zip(a, b):
                        np.testing.assert_array_equal(x, y)
            for path, data in frozen.items():
                self.assertEqual(Path(path).read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
