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

    def test_incremental_summary_is_case_local_and_oracle_is_not_execution(self):
        report = self.report()
        case = report["cases"][1]
        def comparison(candidate, gain, family="matched_layout"):
            return dict(baseline="baseline", candidate=candidate, comparison=family,
                full_cache_savings_gain_percentage_points=gain, full_cache_size_is_estimated=True,
                record_wins=3, record_ties=1, record_losses=2)
        case["recipe_analysis"] = dict(format="qwen_recipe_analysis_v1",
            comparisons=[comparison("less-effective", .25), comparison("temporal", 1.5),
                comparison("residual-dictionary", -.75, "add_frozen_residual_dictionary")],
            skipped_comparisons=[dict(reason="mismatched records")],
            stateless_size_oracle=dict(status="ok", full_cache_savings_percent=31.25,
                full_cache_savings_gain_over_best_fixed_percentage_points=.5, full_cache_size_is_estimated=True))
        case["capture"] = dict(prefill_recipes=dict(enabled=True))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, report)
            text = (Path(directory) / "summary.md").read_text()
            dev, test = text.split("## TEST — frozen comparison", 1)
            self.assertNotIn("Incremental representation diagnostics", dev)
            self.assertIn("Incremental representation diagnostics", test)
            self.assertIn("1.500 estimated", test)
            self.assertIn("-0.750 estimated", test)
            self.assertNotIn("less-effective", text)
            self.assertIn("3 / 1 / 2", test)
            self.assertIn("SIZE-ONLY stateless portfolio oracle", text)
            self.assertIn("upper bound", text)
            self.assertIn("not an implemented selector", text)
            self.assertIn("1 unmatched comparison(s)", text)
            self.assertIn("frozen TRAIN decoder state", text)
            self.assertIn("same correction histogram", text)
            self.assertIn("31.250% estimated", text)
            self.assertIn("0.500 percentage points", text)

    def test_incremental_summary_reports_unavailable_oracle(self):
        report = self.report()
        report["cases"][1]["recipe_analysis"] = dict(comparisons=[],
            stateless_size_oracle=dict(status="unavailable", reason="record identities differ"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint(directory, report)
            text = (Path(directory) / "summary.md").read_text()
            self.assertIn("stateless size oracle unavailable: record identities differ", text)

    def test_bundle_verified_residual_model_and_recipe_analysis(self):
        from qwen_kv_bench import residual_dictionary as residual
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"; output.mkdir()
            calibration = output / "calibration"
            rng = np.random.default_rng(81)
            caches = [{f"layer_00_{kind}": rng.integers(0, 65536, size=(8, 8), dtype=np.uint16)
                       for kind in ("K", "V")} for _ in range(3)]
            model = residual.train(caches, calibration, tile_tokens=8, layout="time", n_centroids=16,
                iterations=1, max_samples_per_key=24, model_identity="test-source")
            path = Path(model.path)
            write_json(calibration / "calibration_manifest.json", dict(format="qwen_kv_calibration_v1",
                calibration_namespace="qwen_kv_bench", models={path.stem: dict(file=path.name,
                    family="prefill_residual", serialized_bytes=model.state_bytes, sha256=model.model_id)}))
            checkpoint(output, self.report())
            write_json(output / "recipe_analysis.json", {"format": "qwen_recipe_analysis_v1"})
            archive = bundle(output)
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(zipped.read("calibration/" + path.name), path.read_bytes())
                self.assertIn("recipe_analysis.json", zipped.namelist())
                manifest = json.loads(zipped.read("bundle_manifest.json"))
                entry = next(row for row in manifest["files"] if row["file"] == "calibration/" + path.name)
                self.assertEqual(entry["sha256"], model.model_id)
                self.assertEqual(entry["bytes"], model.state_bytes)

    def test_bundle_rejects_unverified_or_misnamed_residual_state(self):
        cases = [
            ("recipe_residual16_channel_t8.bin", "prefill_residual", b"LKVWbad", "valid", "state size/hash/format check failed"),
            ("recipe_residual16_channel_t8.bin", "magnitude", b"RDKVvalid", "valid", "filename and family disagree"),
            ("learned16.bin", "prefill_residual", b"RDKVvalid", "valid", "filename and family disagree"),
            ("recipe_residual16_channel_t8.bin", "prefill_residual", b"RDKVvalid", "missing", "hash or size missing or invalid"),
            ("recipe_residual16_channel_t8.bin", "prefill_residual", b"RDKVvalid", "wrong", "state size/hash/format check failed"),
            ("recipe_residual4_channel_t8.bin", "prefill_residual", b"RDKVvalid", "valid", "not an approved calibration artifact"),
        ]
        for filename, family, data, verification, reason in cases:
            with self.subTest(filename=filename, family=family, verification=verification), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"; output.mkdir()
                calibration = output / "calibration"; calibration.mkdir()
                (calibration / filename).write_bytes(data)
                entry = dict(file=filename, family=family, serialized_bytes=len(data))
                if verification != "missing":
                    entry["sha256"] = hashlib.sha256(data).hexdigest() if verification == "valid" else "0" * 64
                write_json(calibration / "calibration_manifest.json", dict(format="qwen_kv_calibration_v1",
                    calibration_namespace="qwen_kv_bench", models={"candidate": entry}))
                checkpoint(output, self.report())
                with zipfile.ZipFile(bundle(output)) as zipped:
                    self.assertNotIn("calibration/" + filename, zipped.namelist())
                    manifest = json.loads(zipped.read("bundle_manifest.json"))
                    self.assertTrue(any(reason in row["reason"] for row in manifest["omitted"]))


if __name__ == "__main__":
    unittest.main()
