"""Producer metadata predictors: fitting/provenance/arithmetic tests, no speed claim."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from qwen_kv_bench import producer_models as p


def bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


class ProducerModelTests(unittest.TestCase):
    def sources(self, *, zeros=False, ties=False):
        rng = np.random.default_rng(19)
        sets = []
        for capture in range(3):
            records = []
            for layer in (0, 1):
                for start in (0, 6):
                    hidden = rng.standard_normal((6, 8)).astype(np.float32)
                    if zeros:
                        hidden[:, 5:] = 0
                    if ties:
                        hidden[:, :] = 0
                        hidden[:, 0] = np.arange(6, dtype=np.float32) - 3
                        hidden[:, 1] = -hidden[:, 0]
                    quantized_hidden = (bf16(hidden).astype(np.uint32) << 16).view(np.float32)
                    exponent = np.clip(124 + np.rint(quantized_hidden[:, :4] * 3), 0, 255).astype(np.uint16)
                    k = ((exponent << 7) | rng.integers(0, 128, (6, 4), dtype=np.uint16))
                    v = (((255 - exponent) << 7) | rng.integers(0, 128, (6, 4), dtype=np.uint16))
                    records.append(dict(layer=layer, start=start, tokens=6, head_dim=2,
                        k=k, v=v, hidden=bf16(hidden), unused_source_field=dict(capture=capture),
                        unused_source_array=np.array([capture, layer, start], dtype=np.int32)))
            sets.append(records)
        return sets

    def identity(self):
        return dict(model="tiny-producer", config=dict(head_dim=2, hidden_size=8, num_key_value_heads=2))

    def fit(self, directory, sources=None, rank=3, **kwargs):
        return p.fit(sources if sources is not None else self.sources(), directory, rank=rank,
            training_metadata=[dict(split="train", name=f"train_{index}") for index in range(3)],
            model_identity=self.identity(), **kwargs)

    def test_deterministic_serialization_freeze_identity_and_complete_charge(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = self.fit(a), self.fit(b)
            self.assertEqual(set(first), {"frozen", "hidden", "hidden_abs"})
            for kind in first:
                m, n = first[kind], second[kind]
                self.assertEqual(m.kind, kind)
                self.assertEqual(m.model_id, n.model_id)
                self.assertEqual(Path(m.path).read_bytes(), Path(n.path).read_bytes())
                self.assertEqual(m.model_id, hashlib.sha256(Path(m.path).read_bytes()).hexdigest())
                self.assertEqual(m.state_bytes, Path(m.path).stat().st_size)
                arrays = sum(x.nbytes for q in m.params.values() for x in q.values())
                self.assertGreater(m.state_bytes, arrays)
                self.assertTrue(all(not x.flags.writeable for q in m.params.values() for x in q.values()))
                self.assertEqual(set(m.params["layer_00"]), {"base_K", "base_V"} if kind == "frozen" else
                    {"base_K", "base_V", "indices", "scales", "coeff_K", "coeff_V"})
                restored = p.load(m.path, expected_model_identity=self.identity(),
                    expected_training_hashes=m.metadata["training_hashes"], expected_kind=kind, expected_model_id=m.model_id)
                self.assertEqual(restored.model_id, m.model_id)
                self.assertEqual(m.config["seed"], 71)
                with self.assertRaises(ValueError):
                    p.load(m.path, expected_model_id="wrong")
                with self.assertRaises(ValueError):
                    p.load(m.path, expected_model_identity="wrong")
                with self.assertRaises(ValueError):
                    p.load(m.path, expected_training_hashes=["wrong"] * 3)
                with self.assertRaises(ValueError):
                    p.load(m.path, expected_kind="wrong")

    def test_source_identity_normalizes_standard_config_integer_json_keys(self):
        identity = self.identity()
        identity["config"]["id2label"] = {0: "LABEL_0", 1: "LABEL_1"}
        with tempfile.TemporaryDirectory() as directory:
            models = p.fit(self.sources(), directory, rank=3,
                training_metadata=[dict(split="train")] * 3, model_identity=identity)
            for model in models.values():
                self.assertEqual(model.metadata["model_identity"]["config"]["id2label"],
                    {"0": "LABEL_0", "1": "LABEL_1"})
                self.assertEqual(p.load(model.path, expected_model_identity=identity).model_id, model.model_id)

    def test_train_only_explicit_three_captures_and_architecture_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            for split in ("dev", "test", None):
                meta = [dict(split="train") for _ in range(3)]
                meta[1]["split"] = split
                with self.assertRaises(ValueError):
                    p.fit(self.sources(), directory, rank=3, training_metadata=meta, model_identity=self.identity())
            sources = self.sources()
            sources[0][0]["split"] = "test"
            with self.assertRaises(ValueError):
                self.fit(directory, sources)
            with self.assertRaises(ValueError):
                self.fit(directory, self.sources()[:2])
            with self.assertRaises(ValueError):
                self.fit(directory, rank=9)
            for identity in ({}, None, dict(head_dim=4), dict(hidden_size=9), dict(num_key_value_heads=3)):
                with self.assertRaises(ValueError):
                    p.fit(self.sources(), directory, rank=3, training_metadata=[dict(split="train")] * 3,
                        model_identity=identity)
            mismatched = self.sources()
            mismatched[1] = [record for record in mismatched[1] if record["layer"] == 0]
            with self.assertRaises(ValueError):
                self.fit(directory, mismatched)

    def test_medians_variance_ties_and_zero_feature_scales(self):
        sources = self.sources(zeros=True, ties=True)
        with tempfile.TemporaryDirectory() as directory:
            models = self.fit(directory, sources, rank=8)
            frozen, hidden = models["frozen"], models["hidden"]
            for layer in (0, 1):
                key = f"layer_{layer:02d}"
                np.testing.assert_array_equal(hidden.params[key]["indices"], np.arange(8, dtype=np.int32))
                self.assertTrue(np.all(hidden.params[key]["scales"][2:] == 1))
                for kind in ("K", "V"):
                    values = np.concatenate([((record[kind.lower()].astype(np.uint32) >> 7) & 255)
                        for records in sources for record in records if record["layer"] == layer])
                    expected = np.floor(np.median(values, axis=0)).astype(np.uint8)
                    np.testing.assert_array_equal(frozen.base[key][kind], expected)
                    prediction = frozen.predict_exponents(layer, kind, 11)
                    self.assertEqual(prediction.dtype, np.int16)
                    np.testing.assert_array_equal(prediction, np.broadcast_to(expected, (11, 4)))
            self.assertEqual(frozen.features(sources[0][0]).shape, (6, 0))
            self.assertEqual(frozen.features(dict(layer=0, tokens=3)).shape, (3, 0))

    def test_features_clipping_and_integer_decoder_oracle(self):
        sources = self.sources()
        with tempfile.TemporaryDirectory() as directory:
            hidden = self.fit(directory, sources)["hidden"]
            record = sources[0][0]
            params = hidden.params["layer_00"]
            value = (record["hidden"].astype(np.uint32) << 16).view(np.float32)
            expected_features = np.clip(np.rint(value[:, params["indices"]] / params["scales"]), -127, 127).astype(np.int8)
            features = hidden.features(record)
            self.assertEqual(features.dtype, np.int8)
            np.testing.assert_array_equal(features, expected_features)
            extreme = dict(record, hidden=bf16(np.full((6, 8), 1e8, dtype=np.float32)))
            np.testing.assert_array_equal(hidden.features(extreme), np.full((6, 3), 127, dtype=np.int8))
            for kind in ("K", "V"):
                expected = np.empty((6, 4), dtype=np.int16)
                # Independent Python integer accumulation; no float decoder.
                for row in range(6):
                    for column in range(4):
                        dot = sum(int(features[row, index]) * int(params["coeff_" + kind][index, column])
                                  for index in range(3))
                        expected[row, column] = min(255, max(0, int(params["base_" + kind][column]) + (dot + 128) // 256))
                got = hidden.predict_exponents(0, kind, 6, features)
                self.assertEqual(got.dtype, np.int16)
                np.testing.assert_array_equal(got, expected)
            with self.assertRaises(ValueError):
                hidden.predict_exponents(0, "K", 6, features.astype(np.float32))
            with self.assertRaises(ValueError):
                hidden.predict_exponents(0, "K", 6, features[:2])
            with self.assertRaises(ValueError):
                hidden.features(dict(record, hidden=bf16(np.zeros((6, 9)))))

    def test_abs_ablation_reuses_signed_features_with_own_integer_coefficients(self):
        sources = self.sources()
        with tempfile.TemporaryDirectory() as directory:
            models = self.fit(directory, sources)
            signed, absolute = models["hidden"], models["hidden_abs"]
            self.assertEqual(signed.config["feature_transform"], "signed")
            self.assertEqual(absolute.config["feature_transform"], "abs")
            self.assertNotEqual(signed.model_id, absolute.model_id)
            self.assertNotEqual(signed.path, absolute.path)
            for record in sources[0]:
                layer = record["layer"]
                key = f"layer_{layer:02d}"
                np.testing.assert_array_equal(signed.params[key]["indices"], absolute.params[key]["indices"])
                np.testing.assert_array_equal(signed.params[key]["scales"], absolute.params[key]["scales"])
                features = signed.features(record)
                np.testing.assert_array_equal(features, absolute.features(record))
                for kind in ("K", "V"):
                    params = absolute.params[key]
                    predicted = np.empty((record["tokens"], 4), dtype=np.int16)
                    for row in range(record["tokens"]):
                        for column in range(4):
                            dot = sum(abs(int(features[row, index])) * int(params["coeff_" + kind][index, column])
                                      for index in range(3))
                            predicted[row, column] = min(255, max(0, int(params["base_" + kind][column]) + (dot + 128) // 256))
                    np.testing.assert_array_equal(absolute.predict_exponents(layer, kind, record["tokens"], features), predicted)

    def test_source_all_fields_hash_before_after_and_crc_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = self.sources()
            hashes = [p._source_hash(records) for records in sources]
            models = self.fit(directory, sources)
            self.assertEqual(hashes, [p._source_hash(records) for records in sources])
            original = p._fit_layer_hidden
            calls = [0]
            def mutate_unused(*args, **kwargs):
                result = original(*args, **kwargs)
                if calls[0] == 0:
                    sources[0][0]["unused_source_array"][0] += 1
                calls[0] += 1
                return result
            with tempfile.TemporaryDirectory() as other, mock.patch.object(p, "_fit_layer_hidden", side_effect=mutate_unused):
                with self.assertRaises(AssertionError):
                    self.fit(other, sources)
            for kind, model in models.items():
                damaged = bytearray(Path(model.path).read_bytes())
                damaged[-1] ^= 1
                path = Path(directory) / f"bad_{kind}.bin"
                path.write_bytes(damaged)
                with self.assertRaises(ValueError):
                    p.load(path)
                changed = p.Model(kind, model.params, copy.deepcopy(model.metadata))
                changed.metadata["config"]["seed"] += 1
                self.assertNotEqual(changed.model_id, model.model_id)
                changed.metadata["training_record_sets"][1]["split"] = "dev"
                with self.assertRaises(ValueError):
                    p.model_bytes(changed)

    def test_parameter_validation_finite_hidden_and_missing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            for field, value in (("tokens", 0), ("head_dim", 3), ("start", -1)):
                records = self.sources()
                records[0][0][field] = value
                with self.assertRaises(ValueError):
                    self.fit(directory, records)
            records = self.sources()
            records[0][0]["hidden"][0, 0] = 0x7f80
            with self.assertRaises(ValueError):
                self.fit(directory, records)
            records = self.sources()
            records[0].append(records[0][0])
            with self.assertRaises(ValueError):
                self.fit(directory, records)


if __name__ == "__main__":
    unittest.main()
