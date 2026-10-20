"""Evidence/integrity tests for the preserved team-source audit."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("team_source_audit", ROOT/"scripts/review_team_transfer_source.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
SOURCE = ROOT/"experiments/team_transfer_original"


class TeamTransferSourceAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.review(SOURCE)

    def test_historical_evidence_is_not_claimed_as_new_model_reproduction(self):
        report = self.report
        self.assertTrue(report["all_passed"])
        h = report["historical_evidence"]
        self.assertFalse(h["model_compression_independently_reproduced"])
        self.assertFalse(report["provenance"]["archive_hash_recomputed"])
        self.assertEqual(h["total_values"], 11_796_480)
        self.assertEqual(h["chunk_count"], 4)
        self.assertAlmostEqual(h["weighted_average_logged_bpv"]["transfer"], 8.360)
        self.assertAlmostEqual(h["weighted_average_logged_bpv"]["pool"], 11.388)
        independent = [c for c in report["checks"] if c["check"].startswith("new test:")]
        self.assertEqual(len(independent), 4)
        self.assertTrue(all(c["passed"] for c in independent))
        mapping = next(c for c in independent if "symbol mappings" in c["check"])
        self.assertEqual(mapping["tested_words"], 65_536)

    def test_layer_zero_and_actual_padded_state_are_charged(self):
        c = self.report["coefficient_accounting"]
        self.assertEqual(c["layer_zero_active_coefficients"], 67_584)
        self.assertEqual(c["active_coefficients"], 6_303_744)
        self.assertEqual(c["hypothetical_compact_int32_coefficient_bytes"], 25_214_976)
        self.assertEqual(c["actual_padded_int64_coefficient_bytes"], 69_599_232)
        self.assertEqual(c["useful_macs_per_value_all_layers"], 547.2)
        self.assertEqual(c["useful_macs_per_value_predicted_layers"], 560.)
        a = self.report["amortization"]
        self.assertAlmostEqual(a["actual_padded_int64_coefficients"]["state_inclusive_bpv"], 14.26)
        self.assertAlmostEqual(a["hypothetical_compact_int32_coefficients"]["state_inclusive_bpv"], 10.4975)
        self.assertGreater(a["actual_padded_int64_coefficients"]["minimum_aggregate_tokens_for_target"],
                           a["hypothetical_compact_int32_coefficients"]["minimum_aggregate_tokens_for_target"])

    def test_amortization_handles_no_possible_target_and_changing_geometry(self):
        self.assertIsNone(audit.amortization(100, 12., 2, 4)["minimum_aggregate_tokens_for_target"])
        # Brute force a small geometry so this check is independent of the formula.
        row = audit.amortization(16, 8., 2, 4)
        smallest = next(t for t in range(1, 100) if 8+16*8/(t*2*4) <= 9.6)
        self.assertEqual(row["minimum_aggregate_tokens_for_target"], smallest)
        with self.assertRaises(ValueError):
            audit.coefficient_accounting(2, 10, 4)

    def copy_source(self, root):
        destination = Path(root)/"source"
        shutil.copytree(SOURCE, destination)
        return destination

    def test_source_mutation_fails_identity_even_when_logic_still_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.copy_source(temp)
            path = source/"combined.py"
            path.write_text(path.read_text()+"\n# changed source\n")
            report = audit.review(source)
            self.assertFalse(report["all_passed"])
            self.assertIn("source identity: combined.py", [x["check"] for x in report["checks"] if not x["passed"]])

    def test_changed_manifest_cannot_relabel_another_codec_as_the_audited_version(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.copy_source(temp)
            path = source/"combined.py"
            data = path.read_bytes()+b"\n# changed source and updated manifest\n"
            path.write_bytes(data)
            manifest_path = source/"source_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["combined.py"] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            manifest_path.write_text(json.dumps(manifest))
            report = audit.review(source)
            self.assertFalse(report["all_passed"])
            self.assertIn("audited original combined.py version", [x["check"] for x in report["checks"] if not x["passed"]])

    def test_wrong_log_arithmetic_is_detected(self):
        text = (SOURCE/"combined_results.txt").read_text()
        text = text.replace("2,949,120 values", "2,949,121 values", 1)
        with tempfile.TemporaryDirectory() as temp:
            source = self.copy_source(temp)
            (source/"combined_results.txt").write_text(text)
            report = audit.review(source)
            failures = [x["check"] for x in report["checks"] if not x["passed"]]
            self.assertTrue(any(x.startswith("logged value geometry:") for x in failures))
            self.assertIn("source identity: combined_results.txt", failures)

    def test_false_historical_roundtrip_is_preserved_and_fails_audit(self):
        text = (SOURCE/"combined_results.txt").read_text().replace("decoded words == original: True", "decoded words == original: False", 1)
        rows = audit.parse_logs(text)
        self.assertFalse(rows[0]["historical_roundtrip_assertions"]["decoded"])
        self.assertTrue(rows[0]["historical_roundtrip_assertions"]["encoder"])

    def test_only_selected_pure_helpers_execute(self):
        source = (SOURCE/"combined.py").read_text()
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp)/"must-not-exist"
            helpers = audit.extract_helpers(f"open({str(marker)!r}, 'w').write('bad')\n"+source)
            self.assertFalse(marker.exists())
            self.assertIn("group_num", helpers)
        for old, new in (
            ("def ordered(w):", "def ordered(w):\n    open('bad', 'w')"),
            ("def ordered(w):", "def ordered(w):\n    np.save('bad', w)"),
            ("def ordered(w):", "def ordered(w):\n    import os"),
        ):
            with self.subTest(new=new):
                with self.assertRaises(ValueError):
                    audit.extract_helpers(source.replace(old, new, 1))

    def test_manifest_cannot_escape_source_directory(self):
        for name in ("../outside", "/outside", "dir\\outside"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                audit.safe_relative(name)

    def test_missing_or_duplicate_metrics_are_rejected(self):
        text = (SOURCE/"combined_results.txt").read_text()
        with self.assertRaisesRegex(ValueError, "No end-to-end"):
            audit.parse_logs("no measurements")
        text = text.replace("  transfer stream (real bytes)", "  ignored stream (real bytes)", 1)
        with self.assertRaisesRegex(ValueError, "Incomplete chunk"):
            audit.parse_logs(text)

    def test_cli_writes_honest_scope_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"audit"
            with mock.patch("sys.stdout", new_callable=io.StringIO) as captured:
                code = audit.main([str(SOURCE), "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertFalse(json.loads(captured.getvalue())["original_compression_reproduced"])
            saved = json.loads((output/"summary.json").read_text())
            self.assertEqual(saved["format"], audit.FORMAT)
            self.assertIn("not original-data compression reproduction", saved["scope"])
            before = hashlib.sha256((output/"summary.json").read_bytes()).hexdigest()
            with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as stopped:
                audit.main([str(SOURCE), "--output", str(output)])
            self.assertEqual(stopped.exception.code, 2)
            self.assertEqual(before, hashlib.sha256((output/"summary.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
