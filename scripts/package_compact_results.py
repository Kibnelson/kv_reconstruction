#!/usr/bin/env python3
"""Share compact-prefill study reports, calibration and small hint arrays.

The run is read-only. This copies an explicit report allowlist plus .json/.bin
calibration and .npz hint files, omitting packets, raw captures and model weights.
It writes a fresh directory and a deterministic sibling .zip. Both successful
and failed studies can be packaged; a failed study need not have every report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import zipfile


REPORTS = (
    "results.json", "results.csv", "summary.md", "decision_summary.json",
    "frozen_manifest.json", "source_manifest.json", "producer_summary.json",
)
MAX_FILE_BYTES = 99 * 1024 ** 2
MAX_JSON_BYTES = 90 * 1024 ** 2
MAX_TOTAL_BYTES = 90 * 1024 ** 2
MAX_MANIFEST_BYTES = 1024 ** 2
MAX_FILES = 10000


def _canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _path(value, label):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError(label + " must not contain '..'; provide its normalized path")
    path = path.absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(label + " must not be a symlink or use a symlink parent")
    return path


def _fresh(target, archive):
    if target.is_symlink() or (target.exists() and
            (not target.is_dir() or any(target.iterdir()))):
        raise FileExistsError("Destination must be fresh or empty; choose a new --destination")
    if archive.is_symlink() or archive.exists():
        raise FileExistsError("Archive already exists; choose a new --destination: " + str(archive))


def _size(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Only regular artifact files are allowed: " + str(path))
    if value.st_size > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 99 MiB per-file limit: " + str(path))
    if path.suffix == ".json" and value.st_size > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds the 90 MiB review limit; reduce the report before packaging: " + str(path))
    return value.st_size


def _files(source):
    selected = []
    for name in REPORTS:
        path = source / name
        if path.is_symlink():
            raise ValueError("Report must not be a symlink: " + name)
        if path.exists():
            _size(path)
            selected.append(path)
    for directory, suffixes in (("calibration", {".bin", ".json"}), ("hints", {".npz"})):
        base = source / directory
        if base.is_symlink():
            raise ValueError("Artifact directory must not be a symlink: " + directory)
        if not base.exists():
            continue
        if not base.is_dir():
            raise ValueError("Artifact directory is not a directory: " + directory)
        pending = [base]
        while pending:
            current = pending.pop()
            for path in sorted(current.iterdir()):
                if path.is_symlink():
                    raise ValueError("Artifact must not be a symlink: " + str(path.relative_to(source)))
                if path.is_dir():
                    pending.append(path)
                elif path.suffix in suffixes:
                    _size(path)
                    selected.append(path)
                if len(selected) + len(pending) > MAX_FILES:
                    raise ValueError("Artifact count exceeds the package safety limit")
    selected.sort(key=lambda path: path.relative_to(source).as_posix())
    if len(selected) > MAX_FILES:
        raise ValueError("Artifact count exceeds the package safety limit")
    if sum(_size(path) for path in selected) > MAX_TOTAL_BYTES:
        raise ValueError("Selected artifact bytes exceed the 90 MiB package limit; use a smaller study/report")
    return selected


def _json_file(path):
    _size(path)
    with path.open("rb") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + path.name)
    return value


def _copy(source, destination):
    before = source.stat()
    digest = hashlib.sha256()
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while block := incoming.read(1024 ** 2):
            digest.update(block)
            total += len(block)
            if total > MAX_FILE_BYTES:
                raise ValueError("Artifact grew beyond the per-file limit during packaging")
            outgoing.write(block)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino) or total != before.st_size:
        raise ValueError("Run changed during packaging; wait for completion: " + str(source))
    return dict(bytes=total, sha256=digest.hexdigest())


def package_compact(run, destination):
    source, target = _path(run, "Run"), _path(destination, "Destination")
    archive = target.with_name(target.name + ".zip")
    if not source.is_dir():
        raise ValueError("Run must be an existing real directory")
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Destination must be separate from, and outside, the input run")
    _fresh(target, archive)
    selected = _files(source)
    report_path = source / "results.json"
    if report_path not in selected:
        raise ValueError("Run must contain results.json, including for a failed study")
    report = _json_file(report_path)
    status = report.get("status")
    if status not in ("completed", "failed"):
        raise ValueError("Run status must be completed or failed; wait for the active study to finish")
    source_manifest = _json_file(source / "source_manifest.json") if source / "source_manifest.json" in selected else {}
    source_identity = {key: value for key, value in source_manifest.items()
                       if key in ("format", "sources_sha256", "payload_sha256", "git_commit")
                       and isinstance(value, str) and len(value) <= 1024}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".compact-package-", dir=target.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        delivered = []
        for path in selected:
            relative = path.relative_to(source)
            delivered.append(dict(file=relative.as_posix(), **_copy(path, staging / relative)))
        by_file = {item["file"]: item for item in delivered}
        source_identity["report_sha256"] = by_file["results.json"]["sha256"]
        if "source_manifest.json" in by_file:
            source_identity["source_manifest_sha256"] = by_file["source_manifest.json"]["sha256"]
        manifest = dict(format="qwen_compact_results_package_v1", run_status=status,
            report_format=report.get("format") if isinstance(report.get("format"), str) else None,
            source_identity=source_identity, files=delivered,
            total_artifact_bytes=sum(item["bytes"] for item in delivered),
            omitted=["packets/", "raw captures", "original model weights", "files outside the explicit report/calibration/hint allowlist"],
            scope="Byte-identical artifacts from a completed or failed run; no new benchmark. Manifest hashes cover delivered artifacts, excluding the manifest itself.")
        data = _canonical(manifest)
        if len(data) > MAX_MANIFEST_BYTES or manifest["total_artifact_bytes"] + len(data) > MAX_TOTAL_BYTES:
            raise ValueError("Package plus manifest exceeds its review-size limit")
        (staging / "package_manifest.json").write_bytes(data)
        staged_zip = Path(temporary) / "package.zip"
        with zipfile.ZipFile(staged_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as stream:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    entry = zipfile.ZipInfo(path.relative_to(staging).as_posix(), (1980, 1, 1, 0, 0, 0))
                    entry.create_system = 3
                    entry.external_attr = 0o100644 << 16
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    stream.writestr(entry, path.read_bytes(), compresslevel=9)
        if staged_zip.stat().st_size > MAX_TOTAL_BYTES:
            raise ValueError("ZIP exceeds the 90 MiB package limit")
        with zipfile.ZipFile(staged_zip) as stream:
            if stream.testzip() is not None:
                raise ValueError("Package ZIP integrity check failed")
            for item in delivered:
                if hashlib.sha256(stream.read(item["file"])).hexdigest() != item["sha256"]:
                    raise ValueError("Package ZIP artifact hash differs: " + item["file"])
        archive_sha256 = hashlib.sha256(staged_zip.read_bytes()).hexdigest()
        _fresh(target, archive)
        os.replace(staging, target)
        try:
            os.replace(staged_zip, archive)
        except OSError:
            os.replace(target, staging)
            raise
    return dict(destination=str(target), archive=str(archive), run_status=status,
        files=len(delivered), artifact_bytes=manifest["total_artifact_bytes"],
        archive_bytes=archive.stat().st_size, archive_sha256=archive_sha256)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = package_compact(args.run, args.destination)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
