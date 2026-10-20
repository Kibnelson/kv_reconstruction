#!/usr/bin/env python3
"""Package every arm of a completed coset training study with verified gzip.

The input run is read-only. Each child package includes fixed DEV diagnostics
and fitted codec artifacts, but omits raw captures and packet blobs. Every file
must fit the existing 99 MiB package limit; no oversize JSON enters git.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

try:
    from .package_results import package, _file_info
except ImportError:
    from package_results import package, _file_info


def package_training(run, destination):
    source = Path(run).expanduser().resolve()
    target = Path(destination).expanduser().absolute()
    if Path(run).is_symlink() or not source.is_dir():
        raise ValueError("Run must be an existing real directory")
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise FileExistsError("Destination must be fresh or empty")
    if target.resolve().is_relative_to(source):
        raise ValueError("Destination must be outside the input run")
    report = json.loads((source / "results.json").read_bytes())
    if report.get("format") != "qwen_prefill_coset_training_study_v1" or report.get("status") != "completed":
        raise ValueError("Only completed four-arm training studies can be packaged")
    expected = {a + "_" + b for a in ("random", "balanced") for b in ("short", "long")}
    arms = report.get("arms", [])
    if len(arms) != 4 or {a["name"] for a in arms} != expected:
        raise ValueError("Require all four prespecified arms")
    for arm in arms:
        name = arm["directory"]
        if name != arm["name"] or name not in expected or arm["status"] != "completed":
            raise ValueError("Invalid arm path/status")
        if _file_info(source / name / "results.json")["sha256"] != arm["report_sha256"]:
            raise ValueError("Child result hash mismatch: " + name)
        if _file_info(source / name / "training_diagnostics.json")["sha256"] != arm["diagnostics_sha256"]:
            raise ValueError("Child diagnostics hash mismatch: " + name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".coset-training-package-", dir=target.parent) as temporary:
        staging = Path(temporary) / "package"
        root_manifest = package(source, staging)
        children = []
        for arm in arms:
            manifest = package(source / arm["directory"], staging / arm["directory"])
            children.append(dict(directory=arm["directory"],
                package_manifest=_file_info(staging / arm["directory"] / "package_manifest.json"),
                uncompressed_results=manifest["uncompressed_results"]))
        manifest = dict(format="qwen_coset_training_package_v1", arms=children,
                        root_package_manifest=_file_info(staging / "package_manifest.json"),
                        scope="Each root/arm package manifest binds its own delivered files; original runs are unmodified. No raw captures or packet files.")
        (staging / "training_package_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
            raise FileExistsError("Destination changed while packaging")
        os.replace(staging, target)
    return dict(destination=str(target), arms=len(children), root_files=len(root_manifest["files"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = package_training(args.run, args.destination)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
