"""Stdlib package integrity, input preservation and publication guards."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("results_packager", Path(__file__).resolve().parents[1] / "scripts" / "package_results.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


class PackageResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"; self.run.mkdir()
        self.raw = (json.dumps(dict(status="completed", repeated=["KV metadata"] * 2000)) + "\n").encode()
        (self.run / "results.json").write_bytes(self.raw)
        for name in packager.REPORTS:
            (self.run / name).write_text("{}\n" if name.endswith(".json") else "report\n")
        calibration = self.run / "calibration"; calibration.mkdir()
        (calibration / "decoder.bin").write_bytes(b"decoder state\0")
        (calibration / "manifest.json").write_text('{"kind":"decoder"}\n')
        (self.run / "raw_KV.npz").write_bytes(b"PRIVATE RAW KV")
        (self.run / "hidden.npy").write_bytes(b"PRIVATE HIDDEN")
        (self.run / "model.safetensors").write_bytes(b"WEIGHTS")

    def snapshot(self):
        return {str(path.relative_to(self.run)): path.read_bytes() for path in self.run.rglob("*") if path.is_file()}

    def test_roundtrip_manifest_states_allowlist_and_deterministic_gzip(self):
        before = self.snapshot()
        first, second = self.root / "first", self.root / "second"
        manifest = packager.package(self.run, first)
        packager.package(self.run, second)
        self.assertEqual(gzip.decompress((first / "results.json.gz").read_bytes()), self.raw)
        self.assertEqual((first / "results.json.gz").read_bytes(), (second / "results.json.gz").read_bytes())
        self.assertEqual(self.snapshot(), before)
        self.assertFalse((first / "results.json").exists())
        self.assertFalse((first / "raw_KV.npz").exists())
        self.assertFalse((first / "hidden.npy").exists())
        self.assertFalse((first / "model.safetensors").exists())
        self.assertEqual(manifest["uncompressed_results"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertTrue(manifest["uncompressed_results"]["gzip_roundtrip_verified"])
        self.assertIn("calibration/decoder.bin", {entry["file"] for entry in manifest["files"]})
        for entry in manifest["files"]:
            data = (first / entry["file"]).read_bytes()
            self.assertEqual(len(data), entry["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), entry["sha256"])
        self.assertEqual(json.loads((first / "package_manifest.json").read_text()), manifest)

    def test_destination_guard_and_empty_destination(self):
        destination = self.root / "package"; destination.mkdir()
        (destination / "keep.txt").write_bytes(b"USER FILE")
        with self.assertRaises(FileExistsError):
            packager.package(self.run, destination)
        self.assertEqual((destination / "keep.txt").read_bytes(), b"USER FILE")
        (destination / "keep.txt").unlink()
        (self.run / "producer_summary.json").unlink()  # Optional for ordinary KV runs.
        packager.package(self.run, destination)
        self.assertTrue((destination / "results.json.gz").is_file())
        self.assertFalse((destination / "producer_summary.json").exists())
        with self.assertRaises(ValueError):
            packager.package(self.run, self.run / "nested")

    def test_symlinks_unexpected_calibration_and_oversize_rejected_without_partial_output(self):
        decoder = self.run / "calibration" / "decoder.bin"
        original = decoder.read_bytes()
        decoder.unlink(); decoder.symlink_to(self.run / "results.json")
        with self.assertRaises(ValueError):
            packager.package(self.run, self.root / "symlink")
        self.assertFalse((self.root / "symlink").exists())
        decoder.unlink(); decoder.write_bytes(original)
        raw = self.run / "calibration" / "raw.npy"; raw.write_bytes(b"RAW")
        with self.assertRaises(ValueError):
            packager.package(self.run, self.root / "unexpected")
        raw.unlink()
        # The raw JSON may exceed the output cap, but copied state may not.
        with mock.patch.object(packager, "MAX_FILE_BYTES", 8):
            with self.assertRaisesRegex(ValueError, "99 MiB"):
                packager.package(self.run, self.root / "oversize")
        self.assertFalse((self.root / "oversize").exists())
        (self.run / "source_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing required"):
            packager.package(self.run, self.root / "missing_report")

    def test_failed_gzip_verification_keeps_empty_destination_and_inputs(self):
        before = self.snapshot()
        destination = self.root / "package"; destination.mkdir()
        with mock.patch.object(packager.gzip, "open", side_effect=OSError("corrupt gzip")):
            with self.assertRaisesRegex(OSError, "corrupt gzip"):
                packager.package(self.run, destination)
        self.assertEqual(list(destination.iterdir()), [])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(list(self.root.glob(".results-package-*")), [])


if __name__ == "__main__":
    unittest.main()
