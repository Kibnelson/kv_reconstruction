import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from qwen_kv_bench.reporting import write_json, checkpoint, bundle


class ReportingTests(unittest.TestCase):
    @staticmethod
    def report():
        def measured(execution, saving, sampled=False):
            return dict(execution=execution, roundtrip_exact=True, coverage="sampled" if sampled else "full",
                        size_is_estimated=sampled, records_measured=2, records_total=20 if sampled else 2,
                        raw_payload_bytes=1024, serialized_bytes=800, decoder_state_bytes=16,
                        savings_including_state_percent=saving,
                        estimated_full_savings_including_state_percent=saving + 1 if sampled else None,
                        sample_decode=dict(median_seconds=np.float64(.002), MiB_per_second=1.5),
                        transfers=dict(buffers=dict(pinned=dict(wall=dict(raw_h2d=dict(median_seconds=.0001))))))
        return dict(status="running", environment=dict(selected_device="cpu"), model_identity=dict(model_id="unit-model"),
            cases=[dict(name="dev_case", split="dev", tokens=512, fixture={"name": "development"}, results=[
                dict(name="ref", track="cpu_reference", status="ok", measurement=measured("cpu_reference", 99))]),
                dict(name="test_case", split="test", tokens=2048, fixture={"name": "fresh-test"}, results=[
                dict(name="native", track="native_packet", status="ok", measurement=measured("tensor_native_cpu_functional", 10, True)),
                dict(name="unsupported", track="native_packet", status="unsupported", error="unsupported,\nbackend")])],
            generation_checks=[], warnings=[])

    def test_strict_numpy_path_json_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"array": np.array([1, 2]), "scalar": np.int64(8), "path": Path("a/b"),
                              "bad": [np.float32(np.nan), float("inf")]})
            data = json.loads(path.read_text())
            self.assertEqual(data, {"array": [1, 2], "scalar": 8, "path": "a/b", "bad": [None, None]})
            original = path.read_bytes()
            with self.assertRaises(TypeError):
                write_json(path, object())
            self.assertEqual(path.read_bytes(), original)
            with mock.patch("qwen_kv_bench.reporting.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    write_json(path, {"replacement": True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["result.json"])

    def test_checkpoint_flattening_and_split_honesty(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, self.report())
            output = Path(directory)
            with (output / "results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["split"] for row in rows], ["dev", "test", "test"])
            self.assertEqual(rows[0]["measurement.transfers.buffers.pinned.wall.raw_h2d.median_seconds"], "0.0001")
            self.assertEqual(rows[1]["execution"], "tensor_native_cpu_functional")
            self.assertEqual(rows[2]["error"], "unsupported,\nbackend")
            text = (output / "summary.md").read_text()
            self.assertIn("Development — selection", text)
            self.assertIn("TEST — frozen comparison", text)
            self.assertIn("11.000% estimated", text)
            self.assertIn("CPU functionality check", text)
            self.assertIn("does not create a GPU decoder", text)
            self.assertNotIn("54.500", text)  # No development+test average.

    def test_bundle_allowlist_real_state_and_exclusions(self):
        from kv_research.learned_codec import LearnedModel, model_bytes
        from kv_binary.binary_codec import make_state, state_bytes
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"; output.mkdir()
            calibration = output / "calibration"; calibration.mkdir()
            model = LearnedModel({"layer_00_K": np.zeros((16, 8), np.uint16)},
                                 {"n_centroids": 16, "calibration_namespace": "qwen_kv_bench"}, {})
            raw = model_bytes(model); (calibration / "learned16.bin").write_bytes(raw)
            write_json(calibration / "calibration_manifest.json", dict(format="qwen_kv_calibration_v1",
                calibration_namespace="qwen_kv_bench", models={"learned16": dict(file="learned16.bin", family="magnitude",
                    serialized_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())}))
            state = state_bytes(make_state(model, dimension=8, transform="identity"))
            state_path = output / "binary_states" / ("identity_d8_" + model.model_id[:16] + ".bin")
            state_path.parent.mkdir(); state_path.write_bytes(state)
            report = self.report()
            report["cases"][0]["results"][0]["measurement"]["decoder_state_info"] = dict(transform_path=str(state_path),
                transform_sha256=hashlib.sha256(state).hexdigest(), transform_bytes=len(state))
            checkpoint(output, report)
            write_json(output / "source_manifest.json", {"sources_sha256": {"module.py": "abc"}})
            write_json(output / "frozen_manifest.json", {"sha256": "123"})
            (output / "README.md").write_text("Read summary.md")
            for name in ("cache.npz", "cache.bin", ".env", "arbitrary.log", "weights/model.safetensors", "spools/raw.bin", "_source/module.py"):
                path = output / name; path.parent.mkdir(exist_ok=True); path.write_bytes(b"SHOULD NOT BE SHARED")
            archive = bundle(output)
            self.assertEqual(archive.parent, output.parent)
            with zipfile.ZipFile(archive) as zipped:
                names = set(zipped.namelist())
                self.assertIn("calibration/learned16.bin", names)
                self.assertIn(str(state_path.relative_to(output)), names)
                self.assertIn("frozen_manifest.json", names)
                self.assertIn("source_manifest.json", names)
                self.assertTrue(all(name not in names for name in ("cache.npz", "cache.bin", ".env", "arbitrary.log", "weights/model.safetensors", "spools/raw.bin", "_source/module.py")))
                manifest = json.loads(zipped.read("bundle_manifest.json"))
                for entry in manifest["files"]:
                    payload = zipped.read(entry["file"])
                    self.assertEqual(len(payload), entry["bytes"])
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), entry["sha256"])
                self.assertIsNone(zipped.testzip())

    def test_partial_failure_bundle_and_oversized_source_omission(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed-run"
            checkpoint(output, dict(status="failed", fatal_error="No accelerator", cases=[], environment={}, warnings=[]))
            (output / "source_manifest.json").write_bytes(b"x" * ((1 << 20) + 1))
            (output / "unknown_state.bin").write_bytes(b"LKVW" + b"x" * 100)
            archive = bundle(output)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIn("results.json", zipped.namelist())
                self.assertNotIn("source_manifest.json", zipped.namelist())
                self.assertNotIn("unknown_state.bin", zipped.namelist())
                manifest = json.loads(zipped.read("bundle_manifest.json"))
                self.assertTrue(any(row["file"] == "source_manifest.json" for row in manifest["omitted"]))


if __name__ == "__main__":
    unittest.main()
