"""No-download integration of frontier codecs, real producer hooks and reports.

Random tiny BF16 weights verify the serving-code boundaries and exactness. No
pretrained compression-rate, accelerator or 40-percent result is inferred.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from qwen_kv_bench import producer_study


class FrontierStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.output = Path(cls.directory.name) / "frontier"
        cls.capture_order = []
        original = producer_study._capture
        def observe_capture(*args, **kwargs):
            fixture = args[2]
            split = fixture["split"]
            cls.capture_order.append(split)
            if split == "test" and not (cls.output / "frozen_manifest.json").is_file():
                raise AssertionError("Frontier TEST capture started before freezing candidates/state")
            return original(*args, **kwargs)
        # Eight-token records, a tail-crossing prefill chunk, a sampled TEST
        # cache, and an independently full-coverage DEV continuation prefix.
        argv = ["--tiny", "--producer-preset", "frontier", "--device", "cpu",
            "--lengths", "32", "--cases", "1", "--train-tokens", "24",
            "--dev-tokens", "16", "--chunk-tokens", "7", "--tile-tokens", "8",
            "--max-tiles", "6", "--rank", "2", "--repeats", "1",
            "--sample-pairs", "2", "--generation-steps", "1", "--threads", "2",
            "--output", str(cls.output)]
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(producer_study, "_capture", side_effect=observe_capture):
            cls.exit_code = producer_study.main(argv)
        cls.report = json.loads((cls.output / "results.json").read_text())

    def test_all_frontier_candidates_exact_inline_and_replay(self):
        from qwen_kv_bench import frontier_codec
        self.assertEqual(self.exit_code, 0, self.report.get("fatal_error"))
        self.assertEqual(self.report["status"], "completed")
        expected = {spec["name"] for spec in frontier_codec.recipes()}
        self.assertEqual(len(expected), 16)
        for case in self.report["cases"]:
            self.assertEqual({row["name"] for row in case["results"]}, expected)
            timing = case["capture"]["timings"]["prefill"]
            self.assertTrue(timing["producer_hooks_included"])
            self.assertTrue(timing["inline_producer_recipes_included"])
            self.assertTrue(case["capture"]["producer_audit"]["on_tile_enabled"])
            for row in case["results"]:
                self.assertEqual(row["status"], "ok", row.get("error"))
                m = row["measurement"]
                self.assertTrue(m["exact_verified"])
                self.assertTrue(m["roundtrip_exact"])
                self.assertTrue(m["inline_record_payload_matches_replay"])
                self.assertEqual(sum(m["component_bytes"].values()), m["serialized_bytes"])
                self.assertGreaterEqual(m["decoder_state_bytes"], m["native_state_physical_bytes"])
                self.assertGreaterEqual(m["decoder_state_bytes"], m["decoder_state_info"]["serialized_bytes"])

    def test_full_dev_continuation_covers_frontier_families(self):
        from qwen_kv_bench import frontier_codec
        checks = self.report["generation_checks"]
        self.assertEqual({row["name"] for row in checks}, set(frontier_codec.generation_representatives()))
        self.assertEqual(len(checks), 5)
        for row in checks:
            self.assertEqual(row["status"], "ok", row.get("error"))
            self.assertEqual(row["context_tokens"], 16)
            for key in ("all_logits_bitwise_equal", "all_tokens_equal", "all_appended_KV_bitwise_equal"):
                self.assertTrue(row["measurement"][key])
            self.assertEqual(row["measurement"]["max_abs_logit_difference"], 0.)
        dev = next(case for case in self.report["cases"] if case["split"] == "dev")
        self.assertTrue(all(row["measurement"]["coverage"] == "full" for row in dev["results"]))
        self.assertEqual(self.capture_order, ["train", "train", "train", "dev", "test"])
        frozen = self.report["freeze"]
        self.assertEqual(producer_study._digest(frozen["manifest"]), frozen["sha256"])
        self.assertEqual(frozen["manifest"]["configuration"]["producer_preset"], "frontier")

    def test_diagnostics_and_actual_target_ledger_are_separate(self):
        for case in self.report["cases"]:
            self.assertIn("frontier_diagnostics", case)
            self.assertGreaterEqual(case["frontier_diagnostics_cpu_seconds"], 0.)
            json.dumps(case["frontier_diagnostics"], allow_nan=False)
            diagnostic = case["frontier_diagnostics"]
            self.assertFalse(diagnostic["raw_arrays_included"])
            self.assertFalse(diagnostic["fitting_performed"])
            self.assertEqual(set(diagnostic["domains"]), {"final_K", "final_V", "pre_rope_K", "kproj_K"})
            self.assertEqual(diagnostic["target_ledger"]["max_total_bits_per_BF16_value"], 9.6)
            for row in case["results"]:
                m = row["measurement"]
                full = (m["estimated_full_cache_serialized_bytes"] if m["size_is_estimated"] else m["serialized_bytes"]) + m["decoder_state_bytes"]
                bits = 16 * full / m["full_raw_payload_bytes"]
                self.assertAlmostEqual(m["effective_bits_per_BF16_word_including_state"], bits)
                target = m["target_40_percent"]
                self.assertEqual(target["target_bits_per_word"], 9.6)
                self.assertEqual(target["met"], bits <= 9.6)
                self.assertEqual(target["size_is_estimated"], m["size_is_estimated"])
                self.assertIn("entropy diagnostics cannot satisfy", target["scope"])
                self.assertTrue(all("recipe_diagnostics" in trace for trace in m["record_diagnostics"]))
        test_case = next(case for case in self.report["cases"] if case["split"] == "test")
        self.assertTrue(all(row["measurement"]["size_is_estimated"] for row in test_case["results"]))
        summary = json.loads((self.output / "producer_summary.json").read_text())
        self.assertEqual(len(summary["cases"]), 2)
        self.assertIn("gain_vs_frozen_percentage_points", summary["cases"][0]["methods"][0])
        self.assertIn("40% target", (self.output / "summary.md").read_text())

    def test_bundle_contains_only_reports_and_verified_calibration_state(self):
        archive = self.output.with_name(self.output.name + "_results.zip")
        with zipfile.ZipFile(archive) as zipped:
            self.assertIsNone(zipped.testzip())
            manifest = json.loads(zipped.read("bundle_manifest.json"))
            self.assertFalse(manifest["raw_KV_or_hidden_features_included"])
            for info in manifest["files"]:
                data = zipped.read(info["file"])
                self.assertEqual(len(data), info["bytes"])
                self.assertEqual(producer_study.hashlib.sha256(data).hexdigest(), info["sha256"])
                self.assertFalse(info["file"].endswith((".npz", ".npy", ".safetensors")))
            self.assertIn("producer_summary.json", zipped.namelist())

    def test_baseline_remains_default_and_dependencies_can_be_injected(self):
        self.assertEqual(producer_study._parser().parse_args([]).producer_preset, "baseline")
        sentinel = object()
        observer = producer_study._Inline([], {}, "cpu", codec=sentinel)
        self.assertIs(observer.codec, sentinel)
        self.assertIs(producer_study._codec_or_default(sentinel), sentinel)


if __name__ == "__main__":
    unittest.main()
