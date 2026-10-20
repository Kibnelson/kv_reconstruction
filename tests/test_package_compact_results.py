"""Compact-result allowlisting, complete hashes and safe fresh publication."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_compact_results as packager


class CompactPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.run.mkdir()
        (self.run / "results.json").write_text(json.dumps(dict(format="qwen_compact_study_v1", status="completed")))
        (self.run / "summary.md").write_text("# Completed\n")
        (self.run / "source_manifest.json").write_text(json.dumps(dict(sources_sha256="a" * 64)))
        (self.run / "calibration" / "hidden").mkdir(parents=True)
        (self.run / "calibration" / "hidden" / "parameters.bin").write_bytes(b"frozen-codec-parameters")
        (self.run / "calibration" / "hidden" / "model.json").write_text('{"training_only":true}')
        (self.run / "hints").mkdir()
        (self.run / "hints" / "test.npz").write_bytes(b"small-retained-hints")
        (self.run / "packets").mkdir()
        (self.run / "packets" / "test.bin").write_bytes(b"large-packet-must-not-be-copied")
        (self.run / "raw_capture.npz").write_bytes(b"raw")
        (self.run / "model.safetensors").write_bytes(b"model")
        (self.run / "hints" / "unexpected.bin").write_bytes(b"not-a-hint")
        self.target = self.root / "package"

    def test_hashes_byte_identity_and_packet_omission(self):
        before = {p.relative_to(self.run).as_posix(): p.read_bytes()
                  for p in self.run.rglob("*") if p.is_file()}
        result = packager.package_compact(self.run, self.target)
        self.assertEqual(result["run_status"], "completed")
        manifest = json.loads((self.target / "package_manifest.json").read_text())
        expected = {"results.json", "summary.md", "source_manifest.json",
                    "calibration/hidden/model.json", "calibration/hidden/parameters.bin", "hints/test.npz"}
        self.assertEqual({item["file"] for item in manifest["files"]}, expected)
        for item in manifest["files"]:
            data = (self.target / item["file"]).read_bytes()
            self.assertEqual(data, before[item["file"]])
            self.assertEqual(item["bytes"], len(data))
            self.assertEqual(item["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(manifest["source_identity"]["sources_sha256"], "a" * 64)
        self.assertFalse((self.target / "packets").exists())
        self.assertFalse((self.target / "model.safetensors").exists())
        archive = self.target.with_name("package.zip")
        with zipfile.ZipFile(archive) as stream:
            self.assertIsNone(stream.testzip())
            self.assertEqual(set(stream.namelist()), expected | {"package_manifest.json"})
            for item in manifest["files"]:
                self.assertEqual(stream.read(item["file"]), before[item["file"]])
        self.assertEqual(result["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())
        self.assertEqual(before, {p.relative_to(self.run).as_posix(): p.read_bytes()
                                for p in self.run.rglob("*") if p.is_file()})

    def test_partial_failed_run_and_deterministic_zip(self):
        minimal = self.root / "failed"
        minimal.mkdir()
        (minimal / "results.json").write_text('{"status":"failed","error":"test failure"}')
        first = packager.package_compact(minimal, self.root / "failed_package_a")
        second = packager.package_compact(minimal, self.root / "failed_package_b")
        self.assertEqual(first["run_status"], "failed")
        self.assertEqual(first["files"], 1)
        self.assertEqual(first["archive_sha256"], second["archive_sha256"])

    def test_refuses_existing_directory_or_archive_without_changes(self):
        self.target.mkdir()
        (self.target / "keep").write_bytes(b"keep")
        with self.assertRaisesRegex(FileExistsError, "fresh or empty"):
            packager.package_compact(self.run, self.target)
        self.assertEqual((self.target / "keep").read_bytes(), b"keep")
        self.assertFalse(self.target.with_name("package.zip").exists())
        other = self.root / "other"
        (self.root / "other.zip").write_bytes(b"keep zip")
        with self.assertRaisesRegex(FileExistsError, "Archive already exists"):
            packager.package_compact(self.run, other)
        self.assertFalse(other.exists())
        self.assertEqual((self.root / "other.zip").read_bytes(), b"keep zip")

    def test_refuses_symlink_report_artifact_and_parent(self):
        outside = self.root / "outside.json"
        outside.write_text('{"secret":true}')
        link = self.run / "decision_summary.json"
        link.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            packager.package_compact(self.run, self.target)
        self.assertFalse(self.target.exists())
        link.unlink()
        artifact = self.run / "hints" / "external.npz"
        artifact.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            packager.package_compact(self.run, self.target)
        artifact.unlink()
        parent = self.root / "linked"
        parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink parent"):
            packager.package_compact(parent / "run", self.target)

    def test_refuses_traversal_and_inside_source_destination(self):
        with self.assertRaisesRegex(ValueError, "normalized path"):
            packager.package_compact(self.run, self.root / "unused" / ".." / "package")
        with self.assertRaisesRegex(ValueError, "outside"):
            packager.package_compact(self.run, self.run / "package")
        self.assertFalse((self.run / "package").exists())

    def test_refuses_oversize_and_active_run_without_partial_output(self):
        with patch.object(packager, "MAX_TOTAL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "90 MiB"):
                packager.package_compact(self.run, self.target)
        with patch.object(packager, "MAX_JSON_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "JSON exceeds"):
                packager.package_compact(self.run, self.target)
        with patch.object(packager, "MAX_FILE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "per-file"):
                packager.package_compact(self.run, self.target)
        (self.run / "results.json").write_text('{"status":"running"}')
        with self.assertRaisesRegex(ValueError, "wait for the active study"):
            packager.package_compact(self.run, self.target)
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.with_name("package.zip").exists())


if __name__ == "__main__":
    unittest.main()
