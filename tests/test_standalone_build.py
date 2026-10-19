"""Portable build, source accounting, and launcher integrity checks (stdlib only)."""
from __future__ import annotations

import base64
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("standalone_builder", ROOT / "scripts" / "build_standalone.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class StandaloneBuildTests(unittest.TestCase):
    def setUp(self):
        self.private = tempfile.TemporaryDirectory()
        self.root = Path(self.private.name)
        for name in builder.NAMESPACES:
            directory = self.root / name
            directory.mkdir()
            (directory / "example.py").write_text("VALUE = 71\n", encoding="utf-8")
        (self.root / "qwen_kv_bench" / "README.md").write_text("A portable benchmark.\n", encoding="utf-8")
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "standalone_launcher.py.in").write_bytes(
            (ROOT / "scripts" / "standalone_launcher.py.in").read_bytes())

    def tearDown(self):
        self.private.cleanup()

    def launcher(self):
        source = builder.build(self.root)
        namespace = {"__name__": "standalone_test", "__file__": str(self.root / "portable.py")}
        exec(compile(source, str(self.root / "portable.py"), "exec"), namespace)
        return namespace

    def test_deterministic_manifest_and_source_selection(self):
        first = builder.build(self.root)
        ignored = self.root / "kv_binary" / "__pycache__"
        ignored.mkdir()
        (ignored / "ignored.py").write_text("THIS_IS_NOT_SOURCE = True\n")
        (self.root / "kv_binary" / "weights.safetensors").write_bytes(b"not included")
        self.assertEqual(first, builder.build(self.root))
        launcher = self.launcher()
        with zipfile.ZipFile(io.BytesIO(launcher["_payload"]())) as archive:
            manifest = json.loads(archive.read(builder.MANIFEST))["files"]
            self.assertEqual(len(manifest), len(builder.NAMESPACES) + 1)
            self.assertEqual(set(archive.namelist()), set(manifest) | {builder.MANIFEST})
            for name, entry in manifest.items():
                raw = archive.read(name)
                self.assertEqual(entry, {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            self.assertEqual({entry.date_time for entry in archive.infolist()}, {(1980, 1, 1, 0, 0, 0)})

    def test_check_detects_stale_source_without_overwriting(self):
        output = self.root / "portable.py"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(builder.main(["--root", str(self.root), "--output", str(output)]), 0)
            self.assertEqual(builder.main(["--root", str(self.root), "--output", str(output), "--check"]), 0)
        previous = output.read_bytes()
        (self.root / "kv_binary" / "example.py").write_text("VALUE = 72\n")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(builder.main(["--root", str(self.root), "--output", str(output), "--check"]), 1)
        self.assertEqual(output.read_bytes(), previous)
        self.assertNotEqual(builder.build(self.root), previous)

    def test_payload_tampering_is_rejected(self):
        launcher = self.launcher()
        launcher["_PAYLOAD_B64"] = base64.b64encode(b"corrupted source").decode("ascii")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            launcher["_payload"]()

    def test_rejects_unsafe_archive_names_and_symlinks(self):
        launcher = self.launcher()
        for name in ("../outside.py", "/absolute.py", "a/../b.py", "a\\b.py", "C:drive.py", "a//b.py"):
            with self.subTest(name=name), io.BytesIO() as stream:
                with zipfile.ZipFile(stream, "w") as archive:
                    archive.writestr(name, b"x")
                launcher["_MEMBER_COUNT"] = 1
                with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        launcher["_members"](archive)
        with io.BytesIO() as stream:
            with zipfile.ZipFile(stream, "w") as archive:
                info = zipfile.ZipInfo("linked.py")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, b"../outside.py")
            launcher["_MEMBER_COUNT"] = 1
            with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    launcher["_members"](archive)

    def test_extract_requires_empty_destination(self):
        launcher = self.launcher()
        output = self.root / "extracted"
        self.assertEqual(launcher["_extract"](output), output.resolve())
        self.assertEqual((output / "kv_binary" / "example.py").read_text(), "VALUE = 71\n")
        with self.assertRaisesRegex(ValueError, "empty directory"):
            launcher["_extract"](output)
        self.assertEqual((output / "kv_binary" / "example.py").read_text(), "VALUE = 71\n")

    def test_source_symlink_is_rejected(self):
        link = self.root / "kv_binary" / "linked.py"
        link.symlink_to(self.root / "kv_binary" / "example.py")
        with self.assertRaisesRegex(ValueError, "symlink"):
            builder.build(self.root)

    def test_stdlib_instructions_do_not_import_ml_dependencies(self):
        target = self.root / "portable.py"
        target.write_bytes(builder.build(self.root))
        result = subprocess.run([sys.executable, "-I", "-S", str(target), "--instructions"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "A portable benchmark.\n\n")

    def test_repository_standalone_matches_reviewable_source(self):
        self.assertEqual((ROOT / "qwen_kv_gpu_bench.py").read_bytes(), builder.build(ROOT),
                         "Run python scripts/build_standalone.py after source/template edits")


if __name__ == "__main__":
    unittest.main()
