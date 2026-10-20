"""Prespecified arm matching and actual forward/packet/diagnostic integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from qwen_kv_bench import coset_training_study as study


class CosetTrainingStudyTests(unittest.TestCase):
    def test_reject_conflicting_budgets_and_preserve_existing_output(self):
        for argv in (["--steps", "10"], ["--short-steps", "0"],
                     ["--short-steps", "4", "--long-steps", "2"]):
            with self.assertRaises(SystemExit):
                study.main(argv)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"; output.mkdir()
            (output / "user.txt").write_text("keep")
            with self.assertRaises(SystemExit):
                study.main(["--tiny", "--output", str(output)])
            self.assertEqual((output / "user.txt").read_text(), "keep")

    def test_four_tiny_arms_have_exact_prefixes_fixed_probes_and_fresh_test(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(study.study, "_note"):
            output = Path(temporary) / "run"
            code = study.main(["--tiny", "--short-steps", "1", "--long-steps", "2",
                "--dev-tokens", "64", "--modes", "none", "hidden", "--residue-bits", "6",
                "--generation-steps", "1", "--repeats", "1", "--threads", "2", "--max-pairs", "4",
                "--output", str(output)])
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(code, 0, report.get("traceback"))
            self.assertEqual(len(report["arms"]), 4)
            self.assertEqual(report["plan"], json.loads((output / "frozen_manifest.json").read_text()))
            children = {a["name"]: json.loads((output / a["directory"] / "results.json").read_text()) for a in report["arms"]}
            self.assertEqual(len({r["training_captures"][0]["producer_inputs_sha256"] for r in children.values()}), 1)
            probe_hashes = set()
            for name, child in children.items():
                self.assertTrue(child["config"]["fresh_test_fixtures"])
                self.assertEqual(child["cases"][0]["name"], study.FRESH_TEST[0]["name"])
                self.assertTrue(all(c["status"] == "ok" for c in child["generation_checks"]))
                diag = json.loads((output / name / "training_diagnostics.json").read_text())
                probe_hashes.add(diag["probe_manifest"]["sha256"])
                for mode, value in diag["modes"].items():
                    self.assertEqual(value["checkpoints"][0]["update"], 0)
                    self.assertEqual(value["checkpoints"][-1]["update"], child["config"]["steps"])
                    self.assertEqual({c["metrics"]["prior_logits_sha256"] for c in value["checkpoints"]},
                                     {value["checkpoints"][0]["metrics"]["prior_logits_sha256"]})
            self.assertEqual(len(probe_hashes), 1)
            for sampler in ("random", "balanced"):
                short, long = (children[sampler + "_" + label] for label in ("short", "long"))
                for mode in short["models"]:
                    a, b = (r["models"][mode]["metadata"]["training"] for r in (short, long))
                    self.assertEqual(a["initial_parameter_sha256"], b["initial_parameter_sha256"])
                    checkpoint = next(c for c in b["checkpoints"] if c["update"] == 1)
                    self.assertEqual(a["final_parameter_sha256"], checkpoint["parameter_sha256"])
                    self.assertEqual(a["sampling_sha256"], checkpoint["sampling_prefix_sha256"])


if __name__ == "__main__":
    unittest.main()
