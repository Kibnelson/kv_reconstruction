"""TRAIN isolation, packet-context inference and complete sharded neural state."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from qwen_kv_bench import neural_probability as n
from qwen_kv_bench import producer_models as producer


def words(value):
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def captures():
    rng = np.random.default_rng(29)
    result = []
    for index in range(3):
        rows = []
        k = rng.standard_normal((17, 8)).astype(np.float32)
        v = rng.standard_normal((17, 8)).astype(np.float32)
        for layer in range(3):
            k = .7 * k + .2 * v + rng.standard_normal(k.shape).astype(np.float32) * .1
            v = .6 * v - .15 * k + rng.standard_normal(v.shape).astype(np.float32) * .1
            rows.append((words(k), words(v)))
        result.append(rows)
    return result


def fit(path, source=None, **kwargs):
    return n.fit(captures() if source is None else source, path,
        model_identity=dict(model="tiny-neural", config=dict(num_hidden_layers=3, head_dim=4, num_key_value_heads=2)),
        training_metadata=[dict(split="train", name=f"train{i}") for i in range(3)],
        head_dim=4, rope_inv_freq=np.array([1., .1], dtype=np.float32), rope_scaling=1.,
        device="cpu", width=16, steps=2, batch_tokens=7, seed=71, **kwargs)


class NeuralProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.captures = captures()
        cls.hashes = [producer._source_hash(rows) for rows in cls.captures]
        cls.models = fit(cls.directory.name, cls.captures)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_sharded_complete_canonical_artifacts_and_predict_after_reload(self):
        previous = np.concatenate(self.captures[0][0], axis=1)
        for mode, model in self.models.items():
            data = Path(model.path).read_bytes()
            self.assertEqual(model.model_id, hashlib.sha256(data).hexdigest())
            descriptor = json.loads(data)
            self.assertEqual(model.state_bytes, sum(Path(path).stat().st_size for path in model.artifact_paths))
            self.assertGreater(model.state_bytes, model.total_float32_values * 4)
            self.assertTrue(all(Path(path).stat().st_size < 99 * 1024 ** 2 for path in model.artifact_paths))
            self.assertEqual(descriptor["total_serialized_bytes"], model.state_bytes)
            self.assertTrue(all(not array.flags.writeable for group in model.params.values() for array in group.values()))
            self.assertEqual(model.metadata["training_hashes"], self.hashes)
            loaded = n.load(model.path, expected_model_id=model.model_id, expected_training_hashes=self.hashes,
                expected_model_identity=model.metadata["model_identity"], expected_mode=mode)
            first, second = model.prepare("cpu"), loaded.prepare("cpu")
            one, two = first.predict(1, previous, 4), second.predict(1, previous, 4)
            for a, b in zip(one, two):
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(a.dtype, torch.float32)
                self.assertEqual(tuple(a.shape), (17, 16))
            self.assertEqual(first.physical_nbytes, model.total_float32_values * 4)
            self.assertEqual(first.state_bytes, max(model.state_bytes, first.physical_nbytes))
            self.assertNotEqual(first.state_id, first.model_id)
        self.assertEqual(self.hashes, [producer._source_hash(rows) for rows in self.captures])

    def test_training_budget_and_state_are_not_treated_as_equal_between_modes(self):
        per_layer, shared = self.models["per_layer"], self.models["shared"]
        self.assertEqual(per_layer.parameter_count, 2 * (16 * 16 + 16 + 16 * 16 + 16 + 16 * 32 + 32))
        self.assertEqual(shared.parameter_count, 18 * 16 + 16 + 16 * 16 + 16 + 16 * 32 + 32)
        self.assertEqual(per_layer.metadata["training"]["total_optimizer_updates"], 4)
        self.assertEqual(shared.metadata["training"]["total_optimizer_updates"], 2)
        self.assertEqual(sum(shared.metadata["training"]["networks"]["shared"]["updates_by_layer"].values()), 2)
        self.assertEqual(per_layer.macs_per_token(), 16 * 16 + 16 * 16 + 16 * 32)
        self.assertEqual(per_layer.macs_per_value, per_layer.macs_per_token() / 16)
        for mode, model in self.models.items():
            for layer in (1, 2):
                scale = model.params[f"norm_{layer:02d}"]["k_scale"].reshape(2, 4)
                np.testing.assert_array_equal(scale[:, :2], scale[:, 2:])
            for log in model.metadata["training"]["networks"].values():
                self.assertTrue(np.isfinite(log["initial_batch_loss"]))
                self.assertTrue(np.isfinite(log["final_batch_loss"]))

    def test_inference_uses_only_previous_layer_words_and_frozen_state(self):
        source = copy.deepcopy(self.captures)
        previous = np.concatenate(source[0][0], axis=1)
        prepared = self.models["per_layer"].prepare("cpu")
        expected = prepared.predict(1, previous, 0)
        source.clear()
        again = prepared.predict(1, torch.from_numpy(previous.copy()).view(torch.int16), 0)
        self.assertTrue(torch.equal(expected[0], again[0]))
        self.assertTrue(torch.equal(expected[1], again[1]))
        for layer in (0, 3, True):
            with self.assertRaises(ValueError):
                prepared.predict(layer, previous)
        with self.assertRaises(ValueError):
            prepared.predict(1, previous[:, :-1])
        bad = previous.copy()
        bad[0, 0] = 0x7f80
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            prepared.predict(1, bad)
        changed = dict(prepared.execution_contract, float32_matmul_precision="changed")
        with mock.patch.object(n, "_runtime", return_value=changed):
            with self.assertRaisesRegex(ValueError, "contract changed"):
                prepared.predict(1, previous)

    def test_prediction_uses_explicit_split_half_rope(self):
        channels, width = 4, 2
        parameters = {"W0": torch.zeros((8, width)), "b0": torch.zeros(width),
            "W1": torch.zeros((width, width)), "b1": torch.zeros(width),
            "W2": torch.zeros((width, 16)), "b2": torch.zeros(16)}
        parameters["b2"][:4] = torch.tensor([1., 2., 3., 4.])
        norm = dict(prev_mu=torch.zeros(8), prev_sd=torch.ones(8), k_scale=torch.ones(4),
            v_mu=torch.zeros(4), v_sd=torch.ones(4))
        inverse = torch.tensor([1., .25])
        mu, log_scale = n._forward(parameters, norm, inverse, 1., 4, torch.zeros((1, 8)), 1)
        angle = torch.tensor([1., .25, 1., .25])
        expected = torch.tensor([1., 2., 3., 4.]) * torch.cos(angle) + torch.tensor([-3., -4., 1., 2.]) * torch.sin(angle)
        torch.testing.assert_close(mu[0, :4], expected, rtol=0, atol=0)
        torch.testing.assert_close(log_scale, -torch.ones_like(log_scale), rtol=0, atol=0)

    def test_corrupt_missing_and_misbinding_artifacts_rejected(self):
        model = self.models["shared"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            n.save(model, path)
            descriptor = json.loads(path.read_bytes())
            shard = Path(directory) / descriptor["shards"][-1]["file"]
            original = shard.read_bytes()
            damaged = bytearray(original)
            damaged[-1] ^= 1
            shard.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                n.load(path)
            shard.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                n.load(path, expected_model_id="0" * 64)
            with self.assertRaisesRegex(ValueError, "TRAIN source"):
                n.load(path, expected_training_hashes=["x"] * 3)
            descriptor["metadata"]["geometry"]["head_dim"] = 8
            path.write_bytes(n._canonical(descriptor))
            with self.assertRaisesRegex(ValueError, "CRC"):
                n.load(path)

    def test_train_only_and_full_capture_mutation_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                n.fit(captures(), directory, model_identity="tiny", training_metadata=[dict(split="test")] * 3,
                    head_dim=4, rope_inv_freq=np.array([1., .1], np.float32), width=16, steps=1)
            data = captures()
            original = n._normalizers
            calls = [0]
            def mutate(*args, **kwargs):
                result = original(*args, **kwargs)
                if calls[0] == 0:
                    data[0][0][0][0, 0] ^= 1
                calls[0] += 1
                return result
            with mock.patch.object(n, "_normalizers", side_effect=mutate):
                with self.assertRaisesRegex(AssertionError, "changed during fitting"):
                    fit(directory, data, modes=("shared",))

    def test_deterministic_training_and_safe_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            second = fit(directory)
            for mode in self.models:
                self.assertEqual(self.models[mode].model_id, second[mode].model_id)

    def test_frozen_identity_properties_do_not_reserialize_large_weights(self):
        model = n.load(self.models["per_layer"].path)
        expected = model.model_id
        with mock.patch.object(n, "_shard_bytes", side_effect=AssertionError("unexpected payload serialization")):
            self.assertEqual(model.model_id, expected)
            self.assertGreater(model.state_bytes, 0)
            self.assertGreater(len(model.artifact_paths), 1)
        model.metadata["geometry"]["width"] += 1
        with self.assertRaisesRegex(ValueError, "frozen neural artifact"):
            _ = model.model_id


if __name__ == "__main__":
    unittest.main()
