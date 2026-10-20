"""All-arm package integrity and rejection before publication."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import package_coset_training_results as packager


class TrainingPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.run = self.root / "run"; self.run.mkdir()
        arms = []
        for sampling in ("random", "balanced"):
            for budget in ("short", "long"):
                name = sampling + "_" + budget
                child = self.run / name; child.mkdir()
                raw = json.dumps(dict(status="completed", repeated=[name]*100)).encode()
                diag = b'{"fixed":"dev"}\n'
                (child / "results.json").write_bytes(raw)
                (child / "training_diagnostics.json").write_bytes(diag)
                for filename in ("results.csv", "summary.md", "source_manifest.json", "frozen_manifest.json"):
                    (child / filename).write_text("{}\n")
                arms.append(dict(name=name, directory=name, status="completed",
                    report_sha256=hashlib.sha256(raw).hexdigest(), diagnostics_sha256=hashlib.sha256(diag).hexdigest()))
        self.report = dict(format="qwen_prefill_coset_training_study_v1", status="completed", arms=arms)
        (self.run / "results.json").write_text(json.dumps(self.report))
        for filename in ("results.csv", "summary.md", "source_manifest.json", "frozen_manifest.json"):
            (self.run / filename).write_text("{}\n")

    def test_all_arms_retain_diagnostics_and_byte_identical_reports(self):
        package = self.root / "package"
        before = {str(p.relative_to(self.run)):p.read_bytes() for p in self.run.rglob("*") if p.is_file()}
        packager.package_training(self.run, package)
        self.assertEqual(json.loads((package / "training_package_manifest.json").read_text())["format"],
                         "qwen_coset_training_package_v1")
        for arm in self.report["arms"]:
            name = arm["directory"]
            self.assertEqual(gzip.decompress((package / name / "results.json.gz").read_bytes()), before[name+"/results.json"])
            self.assertEqual((package / name / "training_diagnostics.json").read_bytes(), before[name+"/training_diagnostics.json"])
            self.assertFalse((package / name / "results.json").exists())
        self.assertEqual(before, {str(p.relative_to(self.run)):p.read_bytes() for p in self.run.rglob("*") if p.is_file()})

    def test_corrupt_arm_does_not_leave_partial_package(self):
        (self.run / "random_long" / "training_diagnostics.json").write_text("corrupt")
        package = self.root / "package"
        with self.assertRaisesRegex(ValueError, "diagnostics hash"):
            packager.package_training(self.run, package)
        self.assertFalse(package.exists())


if __name__ == "__main__":
    unittest.main()
