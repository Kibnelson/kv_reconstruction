"""Compact transfer packaging preserves evidence without publishing arrays."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transfer_packager", ROOT / "scripts/package_transfer_results.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class PackageTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run, self.review, self.destination = self.root / "run", self.root / "review", self.root / "publish"
        self.run.mkdir(); self.review.mkdir(); (self.run / "captures").mkdir()
        records = []
        for index, split in enumerate(packager.SPLITS):
            ids = np.array([index + 1, index + 2, index + 3], dtype=np.int64)
            path = self.run / "captures" / (split + "_case_3.npz")
            np.savez(path, ids=ids, positions=np.arange(3), Hk=1, hd=2, base=100000.,
                     H=np.zeros((1, 3, 4), np.float32), K_bits=np.zeros((1, 3, 2), np.uint16), V_bits=np.zeros((1, 3, 2), np.uint16))
            record = dict(split=split, name=split + "_case", kind="text", tokens=3, capture_file=str(path),
                          ids_sha256=packager.sha_array(ids), kv_sha256="a" * 64, hidden_sha256="b" * 64,
                          model_fingerprint="c" * 64)
            path.with_suffix(".json").write_text(json.dumps(record))
            records.append(record)
        self.report = dict(format="transfer_extension_study_v1", status="completed", exact_original_reproduction=False,
                           provenance=dict(model_identity=dict(model="fixture-model", revision="fixture-revision", captures=records)))
        self.save_report()
        for name in ("summary.md", "results.csv", "source_manifest.json", "capture_manifest.json", "run.log"):
            (self.run / name).write_text("{}\n" if name.endswith("json") else "fixture\n")
        (self.review / "summary.md").write_text("review summary\n")
        (self.run / "states").mkdir(); (self.run / "states" / "large.npz").write_bytes(b"omitted")
        (self.run / "packet_checks").mkdir(); (self.run / "packet_checks" / "decoded.npy").write_bytes(b"omitted")

    def save_report(self):
        (self.run / "results.json").write_text(json.dumps(self.report))
        self.insights = dict(format="transfer_extension_review_v1", all_passed=True, passed=1, total=1,
                             source_results_sha256=packager.sha_file(self.run / "results.json"),
                             checks=[dict(check="fixture executed evidence", passed=True)])
        self.save_review()

    def save_review(self):
        (self.review / "insights.json").write_text(json.dumps(self.insights))

    def package(self):
        return packager.package_transfer(self.run, self.review, self.destination)

    def test_package_retains_input_identities_and_omits_raw_arrays(self):
        before = packager.sha_file(self.run / "results.json")
        result = self.package()
        self.assertTrue(result["review_all_passed"])
        self.assertEqual(result["input_capture_counts"], dict(train=1, dev=1, test=1))
        self.assertEqual(before, packager.sha_file(self.run / "results.json"))
        self.assertFalse((self.destination / "states").exists())
        self.assertFalse((self.destination / "captures").exists())
        self.assertFalse((self.destination / "packet_checks").exists())
        data = json.loads((self.destination / "input_manifests/train_manifest.json").read_text())
        case = data["cases"][0]
        self.assertEqual(case["input_token_ids"], [1, 2, 3])
        self.assertEqual(case["positions"], [0, 1, 2])
        original = self.run / "captures" / case["original_npz_file"]
        self.assertEqual(case["original_npz_sha256"], packager.sha_file(original))
        self.assertEqual(data["model_identity"]["revision"], "fixture-revision")
        ledger = json.loads((self.destination / "package_manifest.json").read_text())
        for row in ledger["files"]:
            path = self.destination / row["file"]
            self.assertEqual(row["bytes"], path.stat().st_size)
            self.assertEqual(row["sha256"], packager.sha_file(path))
        self.assertIn("cannot rerun", (self.destination / "PACKAGING_SCOPE.md").read_text())

    def test_missing_or_failed_review_is_rejected_without_output(self):
        (self.review / "insights.json").unlink()
        with self.assertRaisesRegex(ValueError, "passing insights"):
            self.package()
        self.assertFalse(self.destination.exists())
        self.insights["all_passed"] = False
        self.save_review()
        with self.assertRaisesRegex(ValueError, "every recorded check"):
            self.package()

    def test_review_must_match_source_and_completed_run(self):
        self.report["status"] = "running"
        self.save_report()
        with self.assertRaisesRegex(ValueError, "completed"):
            self.package()
        self.report["status"] = "completed"
        self.save_report()
        (self.run / "results.json").write_text(json.dumps(self.report) + "\n")
        with self.assertRaisesRegex(ValueError, "different results"):
            self.package()

    def test_token_or_sidecar_identity_mismatch_rejected(self):
        path = self.run / "captures/train_case_3.npz"
        np.savez(path, ids=np.array([20, 21, 22]), positions=np.arange(3))
        with self.assertRaisesRegex(ValueError, "recorded token identity"):
            self.package()
        self.assertFalse(self.destination.exists())

    def test_fresh_destination_and_no_unexpected_review_trees(self):
        self.destination.mkdir(); (self.destination / "keep.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            self.package()
        self.assertEqual((self.destination / "keep.txt").read_text(), "keep")
        (self.destination / "keep.txt").unlink()
        (self.review / "unexpected").mkdir()
        with self.assertRaisesRegex(ValueError, "flat small metadata"):
            self.package()

    def test_limits_apply_to_published_files_not_omitted_arrays(self):
        with (self.run / "states/large.npz").open("wb") as stream:
            stream.truncate(packager.MAX_FILE_BYTES + 1)
        self.package()
        self.assertFalse(list(self.destination.rglob("*.npz")))

    def test_oversize_published_file_and_unexpected_tree_rejected(self):
        with (self.run / "run.log").open("wb") as stream:
            stream.truncate(packager.MAX_FILE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "50 MiB file limit"):
            self.package()
        self.assertFalse(self.destination.exists())
        (self.run / "run.log").write_text("restored\n")
        (self.run / "unrecognized_tree").mkdir()
        with self.assertRaisesRegex(ValueError, "Unexpected run artifact tree"):
            self.package()


if __name__ == "__main__":
    unittest.main()
