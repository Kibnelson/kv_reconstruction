"""Learned BF16 sidecode rate training, TRAIN isolation and frozen-state replay."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from qwen_kv_bench import neural_probability as neural, producer_models as producer
from qwen_kv_bench import side_probability as side, side_codec as codec


def words(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def records(seed=29, tokens=24):
    rng = np.random.default_rng(seed)
    groups = []
    key_weights = rng.standard_normal((2, 8)).astype(np.float32)
    value_weights = rng.standard_normal((2, 8)).astype(np.float32)
    for capture in range(3):
        rows = []
        for layer in range(2):
            hidden = rng.standard_normal((tokens, 12)).astype(np.float32)
            key = hidden[:, :2] @ key_weights + .01 * rng.standard_normal((tokens, 8)).astype(np.float32)
            value = hidden[:, :2] @ value_weights + .01 * rng.standard_normal((tokens, 8)).astype(np.float32)
            rows.append(dict(layer=layer, start=0, tokens=tokens, head_dim=4,
                             hidden=words(hidden), k=words(key), v=words(value)))
        groups.append(rows)
    return groups


def fit(directory, captures=None, **kwargs):
    return side.fit(records() if captures is None else captures, directory,
        training_metadata=[dict(split="train", name=f"train_{i}") for i in range(3)],
        model_identity=dict(model="synthetic-side", config=dict(num_hidden_layers=2, hidden_size=12,
            num_key_value_heads=2, head_dim=4)),
        inverse=np.array([1., .1], dtype=np.float32), rope_scaling=1., head_dim=4,
        rank=2, width=16, batch_tokens=12, device="cpu", seed=71, **kwargs)


class SideProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        cls.captures = records()
        cls.source_hashes = [producer._source_hash(rows) for rows in cls.captures]
        cls.models = fit(Path(cls.directory.name) / "models", cls.captures, steps=20)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_complete_artifacts_reload_and_exact_packet_only_predictions(self):
        for mode, model in self.models.items():
            self.assertEqual(model.state_bytes, sum(Path(p).stat().st_size for p in model.artifact_paths))
            self.assertGreater(model.state_bytes, model.total_float32_values * 4)
            self.assertEqual(model.model_id, hashlib.sha256(Path(model.path).read_bytes()).hexdigest())
            self.assertEqual(model.metadata["training_hashes"], self.source_hashes)
            restored = side.load(model.path, expected_model_identity=model.metadata["model_identity"],
                expected_training_hashes=self.source_hashes, expected_model_id=model.model_id, expected_mode=mode)
            original, reloaded = model.prepare("cpu"), restored.prepare("cpu")
            record = copy.deepcopy(self.captures[0][1])
            feature_words = original.encode_features(record)
            self.assertEqual(feature_words.shape, (24, 0 if mode == "none" else 2))
            self.assertEqual(feature_words.dtype, np.uint16)
            expected = original.predict(1, feature_words, 11)
            target = side.joint(record)
            wire, stats = codec.encode_page(record, original, feature_words=feature_words)
            self.assertEqual(stats["packet_bytes"], stats["feature_bytes"] +
                stats["entropy_payload_bytes"] + stats["framing_bytes"])
            record.clear()
            # Restoration does not need encoder inputs or encoder parameters.
            for name in ("encoder_W", "encoder_b", "source_mu", "source_sd"):
                reloaded.params.pop(name, None)
            actual = reloaded.predict(1, feature_words.copy(), 11)
            for a, b in zip(expected, actual):
                self.assertEqual(a.dtype, np.float32)
                self.assertEqual(a.shape, (24, 16))
                np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(codec.decode_page(wire, reloaded), target)
            self.assertEqual(original.physical_nbytes, model.total_float32_values * 4)
            self.assertEqual(original.state_bytes, max(model.state_bytes, original.physical_nbytes))
        self.assertEqual(self.source_hashes, [producer._source_hash(rows) for rows in self.captures])

    def test_same_sampling_and_budgets_but_all_state_charged(self):
        logs = [model.metadata["training"] for model in self.models.values()]
        for log in logs:
            self.assertEqual(log["total_optimizer_updates"], 20)
            self.assertEqual(sum(log["updates_by_layer"].values()), 20)
        schedules = [[(r["update"], r["sampled_layer"], r["record"], r["offset"], r["start"], r["tokens"])
            for r in log["checkpoints"]] for log in logs]
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(schedules[1], schedules[2])
        self.assertEqual(schedules[2], schedules[3])
        kv, hidden = self.models["kv"], self.models["hidden"]
        # Different source dimensions cause a real, charged encoder-state difference.
        self.assertEqual(kv.params["encoder_W"].shape, (16, 2))
        self.assertEqual(hidden.params["encoder_W"].shape, (12, 2))
        self.assertEqual(kv.parameter_count - hidden.parameter_count, 8)
        self.assertEqual(self.models["none"].metadata["geometry"]["rank"], 0)
        self.assertNotIn("encoder_W", self.models["none"].params)

    def test_canonical_control_uses_only_final_kv_and_known_rope(self):
        record = copy.deepcopy(self.captures[0][1])
        record["start"] = 37
        inverse = np.array([1., .1], np.float32)
        canonical = side.canonical_source(record, inverse, 1., 4)
        self.assertEqual(canonical.dtype, np.uint16)
        self.assertEqual(canonical.shape, (24, 16))
        np.testing.assert_array_equal(canonical[:, 8:], record["v"])
        self.assertTrue(np.any(canonical[:, :8] != record["k"]))
        prepared = self.models["kv_canonical"].prepare("cpu")
        expected = prepared.encode_features(record)
        record["hidden"] = words(np.ones((24, 12), np.float32) * 1000)
        record["k_pre_rope"] = words(np.ones((24, 8), np.float32) * -1000)
        np.testing.assert_array_equal(prepared.encode_features(record), expected)
        # At position0 and unit scaling the inverse rotation is identity.
        first = copy.deepcopy(record)
        first["start"], first["tokens"] = 0, 1
        first["k"], first["v"] = record["k"][:1], record["v"][:1]
        np.testing.assert_array_equal(side.canonical_source(first, inverse, 1., 4), side.joint(first))

    def test_learned_encoder_and_decoder_reduce_fixed_training_rate_proxy(self):
        source = self.captures
        with tempfile.TemporaryDirectory() as directory:
            initial = fit(Path(directory) / "initial", source, steps=1, modes=("hidden",))["hidden"]
            trained = fit(Path(directory) / "trained", source, steps=180, modes=("hidden",))["hidden"]
            losses = []
            for model in (initial, trained):
                prepared = model.prepare("cpu")
                total = []
                for rows in source:
                    for record in rows:
                        features = prepared.encode_features(record)
                        mean, logs = prepared.predict(record["layer"], features, record["start"])
                        bits = side.joint(record)
                        total.append(float(neural._loss(torch.tensor(bits.copy()), torch.tensor(neural._floats(bits)),
                            torch.tensor(mean), torch.tensor(logs))))
                losses.append(float(np.mean(total)))
            self.assertLess(losses[1], losses[0] - .1)
            self.assertGreater(np.linalg.norm(initial.params["encoder_W"] - trained.params["encoder_W"]), .01)

    def test_feature_rounding_is_exact_and_passes_encoder_gradient(self):
        source = torch.tensor([[.123456, -1.234567]], requires_grad=False)
        params = dict(source_mu=torch.zeros((1, 2)), source_sd=torch.ones((1, 2)),
                      encoder_W=torch.eye(2, requires_grad=True), encoder_b=torch.zeros(2, requires_grad=True))
        value = side._encode(params, source, 0, straight_through=True)
        np.testing.assert_array_equal(value.detach().numpy(), source.to(torch.bfloat16).float().numpy())
        value.sum().backward()
        self.assertGreater(float(params["encoder_W"].grad.abs().sum()), 0)

    def test_train_labels_geometry_and_bad_inference_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(training_metadata=[dict(split="train")] * 2 + [dict(split="test")],
                model_identity="test", inverse=np.array([1., .1], np.float32), rope_scaling=1., head_dim=4,
                width=8, steps=1, batch_tokens=4)
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                side.fit(self.captures, Path(directory) / "labels", **kwargs)
            bad = copy.deepcopy(self.captures)
            bad[0].append(copy.deepcopy(bad[0][0]))
            kwargs["training_metadata"] = [dict(split="train")] * 3
            with self.assertRaisesRegex(ValueError, "overlap"):
                side.fit(bad, Path(directory) / "overlap", **kwargs)
        prepared = self.models["hidden"].prepare("cpu")
        good = prepared.encode_features(self.captures[0][0])
        for layer, features, start in ((True, good, 0), (2, good, 0), (0, good, -1),
            (0, good.astype(np.int16), 0), (0, good[:, :1], 0)):
            with self.assertRaises(ValueError):
                prepared.predict(layer, features, start)

    def test_state_and_runtime_corruption_rejected(self):
        original = self.models["hidden"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            side.save(side.Model(original.mode, {k: a.copy() for k, a in original.params.items()},
                copy.deepcopy(original.metadata)), path)
            shard = Path(directory) / "side_hidden.npkv"
            data = bytearray(shard.read_bytes())
            data[-1] ^= 1
            shard.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "SHA"):
                side.load(path)
        prepared = original.prepare("cpu")
        prepared.execution_contract["rule"] = "changed"
        with self.assertRaisesRegex(ValueError, "contract"):
            prepared.predict(0, np.zeros((1, 2), np.uint16))


if __name__ == "__main__":
    unittest.main()
