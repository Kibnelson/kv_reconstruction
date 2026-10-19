"""Actual tiny forward-to-report integration, including unequal sampled tails."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from qwen_kv_bench import producer_study


class ConditionalStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.output = Path(cls.directory.name) / "conditional"
        original = producer_study._capture
        cls.order = []
        def observe(*args, **kwargs):
            split = args[2]["split"]
            cls.order.append(split)
            if split == "test" and not (cls.output / "frozen_manifest.json").is_file():
                raise AssertionError("Conditional TEST started before frozen protocol/state")
            return original(*args, **kwargs)
        argv = ["--tiny", "--producer-preset", "conditional", "--device", "cpu",
            "--lengths", "33", "--cases", "1", "--train-tokens", "24",
            "--dev-tokens", "17", "--chunk-tokens", "7", "--tile-tokens", "8",
            "--max-tiles", "6", "--rank", "2", "--repeats", "1",
            "--sample-pairs", "2", "--generation-steps", "1", "--threads", "2",
            "--output", str(cls.output)]
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(producer_study, "_capture", side_effect=observe):
            cls.status = producer_study.main(argv)
        cls.report = json.loads((cls.output / "results.json").read_text())
        cls.decision = json.loads((cls.output / "decision_summary.json").read_text())

    def test_every_idea_exact_in_inline_replay_and_full_dev_generation(self):
        from qwen_kv_bench import conditional_codec
        self.assertEqual(self.status, 0, self.report.get("fatal_traceback"))
        self.assertEqual(self.report["status"], "completed")
        expected = {row["name"] for row in conditional_codec.recipes()}
        self.assertEqual(len(expected), 13)
        for case in self.report["cases"]:
            self.assertEqual({r["name"] for r in case["results"]}, expected)
            for row in case["results"]:
                self.assertEqual(row["status"], "ok", row.get("error_traceback"))
                m = row["measurement"]
                self.assertTrue(m["roundtrip_exact"])
                self.assertTrue(m["inline_record_payload_matches_replay"])
                self.assertTrue(all("recipe_diagnostics" in t for t in m["record_diagnostics"]))
                self.assertEqual(sum(m["component_bytes"].values()), m["serialized_bytes"])
                self.assertGreaterEqual(m["decoder_state_bytes"], m["native_state_physical_bytes"])
                self.assertIn("isolated_workspace", m)
                self.assertIsNone(m["isolated_workspace"]["decode"]["peak_extra_allocated_bytes"])
        self.assertEqual({r["name"] for r in self.report["generation_checks"]}, expected)
        self.assertTrue(all(r["status"] == "ok" for r in self.report["generation_checks"]))
        self.assertTrue(all(step["logits_finite"] for r in self.report["generation_checks"] for step in r["measurement"]["per_step"]))

    def test_decision_ledgers_match_estimates_and_do_not_hide_losing_candidates(self):
        self.assertEqual(self.decision["status"], "completed")
        self.assertEqual(len(self.decision["test_case_averages"]), 13)
        for case in self.decision["cases"]:
            self.assertTrue(case["all_successful_methods_use_same_ordered_records"])
            for m in case["methods"]:
                self.assertAlmostEqual(sum(m["component_bits_per_word"].values()), m["effective_bits_per_word"])
                self.assertIn("conditional_fast_pre_rope", m["matched_control_deltas"])
                self.assertIn("conditional_raw_pair", m["matched_control_deltas"])
        test = next(c for c in self.report["cases"] if c["split"] == "test")
        self.assertTrue(all(r["measurement"]["size_is_estimated"] for r in test["results"]))
        self.assertTrue(any(t["tokens"] == 1 for t in test["results"][0]["measurement"]["record_diagnostics"]))

    def test_protocol_freeze_and_descriptive_metadata_precede_test(self):
        self.assertEqual(self.order, ["train", "train", "train", "dev", "test"])
        freeze = self.report["freeze"]
        self.assertEqual(producer_study._digest(freeze["manifest"]), freeze["sha256"])
        self.assertEqual(freeze["manifest"]["experiment_protocol"], self.report["experiment_protocol"])
        for case in self.report["cases"]:
            d = case["conditional_diagnostics"]
            self.assertFalse(d["fitting_performed"])
            self.assertFalse(d["raw_arrays_included"])
            self.assertIn("frontier_diagnostics", case)
        self.assertEqual(set(self.decision["experiment_protocol"]["ideas"]), {"1", "2", "3", "4"})

    def test_bundle_preserves_decision_report_and_verified_artifacts(self):
        with zipfile.ZipFile(self.output.with_name(self.output.name + "_results.zip")) as zipped:
            self.assertIsNone(zipped.testzip())
            manifest = json.loads(zipped.read("bundle_manifest.json"))
            self.assertFalse(manifest["raw_KV_or_hidden_features_included"])
            self.assertIn("decision_summary.json", zipped.namelist())
            self.assertEqual(json.loads(zipped.read("decision_summary.json")), self.decision)
            for item in manifest["files"]:
                data = zipped.read(item["file"])
                self.assertEqual(len(data), item["bytes"])
                self.assertEqual(producer_study.hashlib.sha256(data).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
