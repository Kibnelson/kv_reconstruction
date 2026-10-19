"""Exact wire-cost TRAIN controls; these tests make no held-out savings claim."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from qwen_kv_bench import frontier_models as f
from qwen_kv_bench import producer_models as p


def _bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


class FrontierModelTests(unittest.TestCase):
    def sources(self):
        rng = np.random.default_rng(91)
        records = []
        for capture in range(3):
            sets = []
            for layer in (0, 1):
                # Median is103, but actual2/3/9wirecost prefers101 here.
                pattern = np.array([100] * 4 + [103] * 3 + [200] * 3, dtype=np.uint16)
                exponents = np.repeat(pattern[:, None], 4, axis=1)
                record = dict(layer=layer, start=0, tokens=10, head_dim=2,
                    hidden=_bf16(rng.standard_normal((10, 4)).astype(np.float32)),
                    k=(exponents << 7) | np.uint16(17),
                    v=((exponents // 2) << 7) | np.uint16(31),
                    k_pre_rope=((exponents + 20) << 7) | np.uint16(8),
                    k_proj=((exponents - 20) << 7) | np.uint16(77),
                    unused_source=np.array([capture, layer], dtype=np.int32))
                sets.append(record)
            records.append(sets)
        return records

    def fit(self, directory, sources=None):
        return f.fit(self.sources() if sources is None else sources, directory, rank=2,
            training_metadata=[dict(split="train", name=f"train_{i}") for i in range(3)],
            model_identity=dict(model="tiny-frontier", config=dict(head_dim=2, hidden_size=4, num_key_value_heads=2)))

    def test_exhaustive_cost_matches_independent_zigzag_oracle_with_boundaries(self):
        rng = np.random.default_rng(5)
        for exponents in (rng.integers(0, 256, (23, 7), dtype=np.int16),
                np.array([[0, 255, 10], [0, 255, 0], [255, 0, 5], [1, 254, 4]], dtype=np.int16),
                np.repeat(np.array([[127]], dtype=np.int16), 6, axis=0)):
            chosen, median, actual, median_actual = f._fit_bases(exponents)
            reference_median = np.floor(np.median(exponents, axis=0)).astype(np.uint8)
            np.testing.assert_array_equal(median, reference_median)
            for column in range(exponents.shape[1]):
                costs = []
                for base in range(256):
                    error = exponents[:, column].astype(np.int64) - base
                    zz = np.where(error >= 0, error * 2, -error * 2 - 1)
                    costs.append(int((2 + 3 * (zz >= 3) + 9 * (zz >= 10)).sum()))
                best = min(range(256), key=lambda base: (costs[base], abs(base - int(median[column])), base))
                self.assertEqual(int(chosen[column]), best)
                self.assertEqual(int(actual[column]), costs[best])
                self.assertEqual(int(median_actual[column]), costs[int(median[column])])
                self.assertLessEqual(int(actual[column]), int(median_actual[column]))

    def test_ties_follow_train_median_then_lower_and_optimum_can_beat_median(self):
        values = np.array([100] * 4 + [103] * 3 + [200] * 3, dtype=np.int16)[:, None]
        chosen, median, cost, median_cost = f._fit_bases(values)
        self.assertEqual(int(median[0]), 103)
        self.assertEqual(int(chosen[0]), 101)
        self.assertLess(int(cost[0]), int(median_cost[0]))
        # Even-count median follows floor(mean of the two central values).
        _, median, _, _ = f._fit_bases(np.array([[0], [1]], dtype=np.int16))
        self.assertEqual(int(median[0]), 0)

    def test_source_specific_bases_old_median_unchanged_full_state_charge(self):
        sources = self.sources()
        with tempfile.TemporaryDirectory() as directory:
            models = self.fit(directory, sources)
            self.assertEqual(set(models), {"frozen", "hidden", "hidden_abs", "cost_final", "cost_pre_rope", "cost_kproj"})
            for name, field in f._SOURCES.items():
                model = models[name]
                self.assertEqual(model.kind, "frozen")
                self.assertEqual(model.metadata["model_variant"], name)
                self.assertEqual(model.config["source_k"], field)
                self.assertEqual(model.config["source_v"], "v")
                self.assertEqual(model.config["candidate_bases"], 256)
                self.assertEqual(model.config["cost_objective"], "2+3*(zigzag(error)>=3)+9*(zigzag(error)>=10)")
                for layer in (0, 1):
                    rows = [record for group in sources for record in group if record["layer"] == layer]
                    expected, _, _, _ = f._fit_bases(f._exponents(rows, field))
                    np.testing.assert_array_equal(model.layer_params(layer)["base_K"], expected)
                    np.testing.assert_array_equal(model.layer_params(layer)["base_V"], models["cost_final"].layer_params(layer)["base_V"])
                    self.assertEqual(set(model.layer_params(layer)), {"base_K", "base_V"})
                data = Path(model.path).read_bytes()
                self.assertEqual(model.state_bytes, len(data))
                self.assertEqual(model.model_id, hashlib.sha256(data).hexdigest())
                self.assertGreater(model.state_bytes, sum(array.nbytes for params in model.params.values() for array in params.values()))
                self.assertTrue(all(not array.flags.writeable for params in model.params.values() for array in params.values()))
                self.assertEqual(f.load(model.path, expected_model_id=model.model_id).model_id, model.model_id)
            np.testing.assert_array_equal(models["frozen"].layer_params(0)["base_K"], np.full(4, 103, dtype=np.uint8))
            np.testing.assert_array_equal(models["cost_final"].layer_params(0)["base_K"], np.full(4, 101, dtype=np.uint8))
            self.assertNotIn("cost_objective", models["frozen"].config)
            self.assertNotEqual(models["cost_final"].model_id, models["cost_pre_rope"].model_id)

    def test_deterministic_reload_and_canonical_provenance(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = self.fit(a), self.fit(b)
            for name in first:
                self.assertEqual(Path(first[name].path).read_bytes(), Path(second[name].path).read_bytes())
                self.assertEqual(first[name].model_id, second[name].model_id)
                self.assertEqual(first[name].metadata["training_hashes"], first["frozen"].metadata["training_hashes"])
                loaded = f.load(first[name].path, expected_training_hashes=first[name].metadata["training_hashes"],
                    expected_model_identity=first[name].metadata["model_identity"], expected_kind=first[name].kind)
                self.assertEqual(loaded.model_id, first[name].model_id)
            changed = p.Model("frozen", first["cost_final"].params, copy.deepcopy(first["cost_final"].metadata))
            changed.metadata["source_k"] = "k_pre_rope"
            self.assertNotEqual(changed.model_id, first["cost_final"].model_id)
            corrupted = bytearray(Path(first["cost_final"].path).read_bytes())
            corrupted[-1] ^= 1
            path = Path(a) / "tampered.bin"
            path.write_bytes(corrupted)
            with self.assertRaises(ValueError):
                f.load(path)

    def test_train_only_required_source_geometry_and_full_source_mutation_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("dev", "test", None):
                with self.assertRaises(ValueError):
                    f.fit(self.sources(), directory, rank=2,
                        training_metadata=[dict(split="train"), dict(split=split), dict(split="train")], model_identity="tiny")
            with self.assertRaises(ValueError):
                self.fit(directory, self.sources()[:2])
            for field in ("k_pre_rope", "k_proj"):
                records = self.sources()
                del records[0][0][field]
                with self.assertRaises(ValueError):
                    self.fit(directory, records)
                records = self.sources()
                records[0][0][field] = records[0][0][field][:, :-1]
                with self.assertRaises(ValueError):
                    self.fit(directory, records)
            records = self.sources()
            original = f._fit_bases
            calls = [0]
            def mutate_after_producer_fit(*args, **kwargs):
                result = original(*args, **kwargs)
                if calls[0] == 0:
                    records[0][0]["unused_source"][0] += 1
                calls[0] += 1
                return result
            with mock.patch.object(f, "_fit_bases", side_effect=mutate_after_producer_fit):
                with self.assertRaises(AssertionError):
                    self.fit(directory, records)

    def test_cost_fit_rejects_noninteger_and_out_of_range_exponents(self):
        for value in (np.zeros((3, 4), dtype=np.float32), np.array([[-1]], dtype=np.int16),
                np.array([[256]], dtype=np.int16), np.empty((0, 4), dtype=np.int16)):
            with self.assertRaises(ValueError):
                f._fit_bases(value)


if __name__ == "__main__":
    unittest.main()
