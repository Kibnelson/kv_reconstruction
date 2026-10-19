import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from qwen_kv_bench.calibration import train_calibration, load_calibration
from kv_research import learned_codec
from kv_coset_fields import learned_lower


class CalibrationTests(unittest.TestCase):
    @staticmethod
    def cases(layers=2, columns=16, tokens=24):
        rng = np.random.default_rng(72)
        return [{f"layer_{layer:02d}_{kind}":
                 (rng.integers(15200, 15900, (tokens, columns), np.uint16)
                  | (rng.integers(0, 2, (tokens, columns), np.uint16) << 15))
                 for layer in range(layers) for kind in ("K", "V")}
                for _ in range(3)]

    def test_requested_models_are_native_bound_counted_and_exact(self):
        caches = self.cases(); identity = {"model_id": "unit-Qwen", "revision": "synthetic"}
        requirements = {"magnitude_sizes": [16, 64], "fixed_sizes": [64],
                        "lower_sizes": [16], "conditional_lower_sizes": [16]}
        metadata = [{"name": f"train_{i}", "split": "train", "input_token_ids": list(range(24))} for i in range(3)]
        with tempfile.TemporaryDirectory() as folder:
            artifacts = train_calibration(caches, folder, requirements, model_identity=identity,
                max_samples_per_key=64, iterations=1, training_metadata=metadata)
            self.assertEqual(set(artifacts.models), {"learned16", "learned64", "fixed64", "lower16", "lower16_conditional"})
            self.assertEqual(artifacts.metadata["schema"]["layers"], 2)
            self.assertEqual(artifacts.metadata["schema"]["columns"], 16)
            restored = load_calibration(folder, expected_model_identity=identity,
                expected_training_hashes=artifacts.metadata["training_hashes"], require_names=artifacts.models)
            for name, model in restored.models.items():
                self.assertEqual(restored.state_bytes[name], Path(restored.paths[name]).stat().st_size)
                self.assertEqual(model.config["source_model_identity"], identity)
                self.assertEqual(model.config["calibration_namespace"], "qwen_kv_bench")
                self.assertEqual(set(model.codebooks), set(caches[0]))
                entry = restored.metadata["models"][name]
                self.assertLessEqual(entry["parameter_array_bytes"], entry["serialized_bytes"])
                self.assertEqual(entry["trained"], name != "fixed64")
                words = caches[0]["layer_00_K"]
                if name.startswith("lower"):
                    lower = (((words >> 8) & 128) | (words & 127)).astype(np.uint8)
                    exponent = ((words >> 7) & 255).astype(np.uint8)
                    encoded = learned_lower.encode_lower(lower, exponent, model, "layer_00_K")
                    recovered = learned_lower.decode_lower(encoded, exponent, model, "layer_00_K")
                    self.assertTrue(np.array_equal(lower, recovered))
                else:
                    encoded = learned_codec.encode_matrix(words, model, "layer_00_K")
                    self.assertTrue(np.array_equal(words, learned_codec.decode_matrix(encoded, model)))

    def test_training_only_identity_and_layout_contracts(self):
        caches = self.cases()
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                train_calibration(caches, folder, model_identity="Qwen", training_metadata=[{}, {"split": "dev"}, {}])
            with self.assertRaisesRegex(ValueError, "model_identity"):
                train_calibration(caches, folder)
            with self.assertRaisesRegex(ValueError, "three"):
                train_calibration(caches[:2], folder, model_identity="Qwen")
            bad = self.cases(); bad[2]["layer_00_K"] = bad[2]["layer_00_K"][:, :8]
            with self.assertRaises(ValueError):
                train_calibration(bad, folder, model_identity="Qwen")
            with self.assertRaisesRegex(ValueError, "sample budget"):
                train_calibration(caches, folder, model_identity="Qwen", max_samples_per_key=32)
            with self.assertRaisesRegex(ValueError, "head dimensions"):
                train_calibration(caches, folder, model_identity={"head_dim": 128, "num_key_value_heads": 8})

    def test_actual_36_layer_1024_column_schema_without_head_inference(self):
        caches = self.cases(layers=36, columns=1024, tokens=8)
        with tempfile.TemporaryDirectory() as folder:
            artifacts = train_calibration(caches, folder, {"fixed_sizes": [64]}, model_identity={"head_dim": 128, "num_key_value_heads": 8})
            self.assertEqual(artifacts.metadata["schema"]["layers"], 36)
            self.assertEqual(artifacts.metadata["schema"]["columns"], 1024)
            self.assertEqual(len(artifacts.models["fixed64"].codebooks), 72)

    def test_model_hash_reproducibility_and_loader_rejections(self):
        caches = self.cases(); identity = {"model_id": "Qwen-unit"}
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            options = dict(requirements={"magnitude_sizes": [16]}, model_identity=identity,
                           max_samples_per_key=64, iterations=2)
            a = train_calibration(caches, first, **options)
            b = train_calibration(caches, second, **options)
            self.assertEqual(a.models["learned16"].model_id, b.models["learned16"].model_id)
            with self.assertRaisesRegex(ValueError, "source model"):
                load_calibration(first, expected_model_identity={"model_id": "another"})
            with self.assertRaisesRegex(ValueError, "training hashes"):
                load_calibration(first, expected_training_hashes=["bad"] * 3)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_calibration(first, require_names=["learned64"])
            with self.assertRaises(FileExistsError):
                train_calibration(caches, first, **options)
            path = Path(a.paths["learned16"]); path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "size/hash"):
                load_calibration(first)
            report_path = Path(b.report_path); report = json.loads(report_path.read_text())
            report["calibration_namespace"] = "old-smol-artifact"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Qwen"):
                load_calibration(second)


if __name__ == "__main__":
    unittest.main()
