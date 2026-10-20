#!/usr/bin/env python3
"""Create a GitHub-size-safe results package without changing a benchmark run.

Only named reports and regular calibration .bin/.json files are copied.
results.json is streamed into deterministic gzip and verified by a complete
decompression/CRC/hash pass; the raw JSON is never copied. This checks byte
integrity, not JSON validity, benchmark correctness or model quality.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile


REPORTS = ("results.csv", "summary.md", "producer_summary.json",
           "decision_summary.json", "source_manifest.json", "frozen_manifest.json")
REQUIRED_REPORTS = ("results.csv", "summary.md", "source_manifest.json", "frozen_manifest.json")
MAX_FILE_BYTES = 99 << 20
CHUNK_BYTES = 1 << 20


def _stream_info(stream):
    checksum = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK_BYTES):
        checksum.update(chunk)
        size += len(chunk)
    return dict(bytes=size, sha256=checksum.hexdigest())


def _file_info(path):
    with path.open("rb") as stream:
        return _stream_info(stream)


def _regular(path, *, limit=True):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Only regular files are allowed; symlinks are rejected: {path}")
    if limit and path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds the 99 MiB package limit: {path}")


def _copy_checked(source, destination):
    _regular(source)
    before = _file_info(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if _file_info(destination) != before or _file_info(source) != before:
        raise ValueError(f"Input changed or copy verification failed: {source}")
    return before


def package(run, destination):
    """Publish two or more verified reports in a fresh/empty destination.

    All work happens in a sibling temporary directory. Failure leaves an
    existing empty destination empty, and never replaces user files.
    """
    run = Path(run).expanduser()
    target = Path(destination).expanduser().absolute()
    if run.is_symlink() or not run.is_dir():
        raise ValueError(f"Run must be an existing real directory: {run}")
    source = run.resolve()
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise FileExistsError(f"Destination must be fresh or empty: {target}")
    if target.resolve().is_relative_to(source):
        raise ValueError("Destination must be outside the input run; inputs are read-only")
    original = source / "results.json"
    if not original.exists() and not original.is_symlink():
        raise ValueError(f"Missing benchmark results: {original}")
    _regular(original, limit=False)  # Oversize raw JSON is precisely what gzip solves.
    to_copy = []
    for name in REPORTS:
        path = source / name
        if path.exists() or path.is_symlink():
            _regular(path)
            to_copy.append((path, name))
        elif name in REQUIRED_REPORTS:
            raise ValueError(f"Missing required benchmark report: {path}")
    calibration = source / "calibration"
    if calibration.is_symlink():
        raise ValueError("Calibration directory symlinks are rejected")
    if calibration.exists():
        if not calibration.is_dir():
            raise ValueError("Calibration must be a directory")
        for path in sorted(calibration.iterdir()):
            if path.suffix not in (".bin", ".json"):
                raise ValueError(f"Only calibration .bin/.json files are allowed: {path}")
            _regular(path)
            to_copy.append((path, "calibration/" + path.name))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".results-package-", dir=target.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        raw_before = _file_info(original)
        compressed = staging / "results.json.gz"
        with original.open("rb") as incoming, compressed.open("wb") as outgoing:
            with gzip.GzipFile(filename="", fileobj=outgoing, mode="wb", compresslevel=9, mtime=0) as zipped:
                shutil.copyfileobj(incoming, zipped, length=CHUNK_BYTES)
        _regular(compressed)
        with gzip.open(compressed, "rb") as restored:
            roundtrip = _stream_info(restored)  # Reading to EOF checks gzip CRC/length.
        if roundtrip != raw_before or _file_info(original) != raw_before:
            raise ValueError("Source results changed or gzip decompression/hash verification failed")
        files = [dict(file="results.json.gz", **_file_info(compressed))]
        for path, relative in to_copy:
            info = _copy_checked(path, staging / relative)
            files.append(dict(file=relative, **info))
        manifest = dict(format="qwen_results_package_v1", source_directory_name=source.name,
            uncompressed_results=dict(name="results.json", **raw_before,
                compressed_as="results.json.gz", gzip_roundtrip_verified=True),
            files=sorted(files, key=lambda entry: entry["file"]),
            raw_results_json_included=False, raw_KV_arrays_included=False,
            manifest_scope="All delivered files except package_manifest.json itself",
            scope="Byte-preserving gzip plus allowlisted reports and calibration state only. Calibration may contain fitted codec parameters. No raw results.json, KV arrays, hidden captures, pretrained LLM weights or arbitrary run files are exported. No JSON parse or benchmark correctness claim.")
        manifest_path = staging / "package_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _regular(manifest_path)
        # os.replace accepts an empty destination directory on supported POSIX
        # hosts. A concurrent new user file makes replacement fail safely.
        if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
            raise FileExistsError(f"Destination changed while packaging: {target}")
        os.replace(staging, target)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Original run directory containing results.json")
    parser.add_argument("--destination", type=Path, required=True, help="Fresh or empty package directory outside RUN")
    args = parser.parse_args(argv)
    try:
        manifest = package(args.run, args.destination)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(dict(destination=str(args.destination), files=len(manifest["files"]),
        original_bytes=manifest["uncompressed_results"]["bytes"],
        compressed_bytes=next(item["bytes"] for item in manifest["files"] if item["file"] == "results.json.gz")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
