#!/usr/bin/env python3
"""Publish small reviewed transfer results and exact input-token manifests.

The full run is read-only. Frozen state, packets, decoded arrays and hidden/KV
captures are deliberately omitted. The delivered review records checks already
executed against those full artifacts; the compact bundle cannot rerun them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

import numpy as np


REPORTS = ("results.json", "results.csv", "summary.md", "source_manifest.json",
           "capture_manifest.json", "model_download_manifest.json", "run.log")
REVIEW_SUFFIXES = {".json", ".md", ".csv", ".txt", ".log"}
RUN_DIRECTORIES = {"captures", "states", "packet_checks"}
MAX_FILE_BYTES = 50 * 1024 ** 2
MAX_TOTAL_BYTES = 90 * 1024 ** 2
SPLITS = ("train", "dev", "test")


def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 ** 2):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Run artifact changed while packaging: " + str(path))
    return digest.hexdigest()


def sha_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + compact(value.shape))
    digest.update(value.tobytes())
    return digest.hexdigest()


def real_path(value, label):
    path = Path(value).expanduser()
    if ".." in path.parts:
        raise ValueError(label + " must use a normalized path without '..'")
    path = path.absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(label + " must not be a symlink or have a symlink parent")
    return path


def regular_file(path, *, size_limit=MAX_FILE_BYTES):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Only regular artifact files are accepted: " + str(path))
    if size_limit is not None and info.st_size > size_limit:
        raise ValueError("Artifact exceeds the 50 MiB file limit: " + str(path))
    return info.st_size


def json_object(path):
    regular_file(path)
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object: " + str(path))
    return result


def fresh(destination):
    if destination.is_symlink() or (destination.exists() and
            (not destination.is_dir() or any(destination.iterdir()))):
        raise FileExistsError("Destination must be fresh or empty; choose a new --destination")


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_review(source, review):
    report = json_object(source / "results.json")
    if report.get("format") != "transfer_extension_study_v1" or report.get("status") != "completed":
        raise ValueError("Input must contain a completed transfer_extension_study_v1 run")
    insights_path = review / "insights.json"
    if not insights_path.is_file():
        raise ValueError("Review must contain passing insights.json from review_transfer_extensions.py")
    insights = json_object(insights_path)
    checks = insights.get("checks")
    if (insights.get("format") != "transfer_extension_review_v1" or insights.get("all_passed") is not True or
            not isinstance(checks, list) or not checks or
            any(not isinstance(item, dict) or item.get("passed") is not True for item in checks) or
            insights.get("passed") != len(checks) or insights.get("total") != len(checks)):
        raise ValueError("Review must report that every recorded check passed")
    source_hash = sha_file(source / "results.json")
    if insights.get("source_results_sha256") != source_hash:
        raise ValueError("Review belongs to a different results.json; rerun the full-artifact review")
    return report, insights


def capture_input_manifests(source, report):
    """Retain exact IDs and provenance without copying the activation arrays."""
    identity = report.get("provenance", {}).get("model_identity", {})
    captures = identity.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("Run lacks capture identities needed to retain exact input-token manifests")
    model = {key: value for key, value in identity.items() if key != "captures"}
    outputs = {split: {"format": "transfer_capture_input_manifest_v1", "split": split,
                       "model_identity": model, "cases": []} for split in SPLITS}
    seen = set()
    capture_root = source / "captures"
    if not capture_root.is_dir() or capture_root.is_symlink():
        raise ValueError("Original captures are required to extract exact IDs before compact packaging")
    for record in captures:
        if not isinstance(record, dict) or record.get("split") not in SPLITS:
            raise ValueError("Capture record must name train, dev or test split")
        split = record["split"]
        name = record.get("name")
        reported = record.get("capture_file")
        if not isinstance(name, str) or not name or not isinstance(reported, str):
            raise ValueError("Capture identity lacks name/path")
        filename = Path(reported).name
        if not filename.endswith(".npz") or filename in (".npz", "..npz"):
            raise ValueError("Capture identity must name an NPZ file")
        path = capture_root / filename
        regular_file(path, size_limit=None)  # Hash, but never copy, large raw arrays.
        sidecar_path = path.with_suffix(".json")
        sidecar = json_object(sidecar_path)
        key = (split, name)
        if key in seen:
            raise ValueError("Duplicate capture identity in input manifests")
        seen.add(key)
        for field in ("name", "tokens", "ids_sha256", "kv_sha256", "hidden_sha256", "model_fingerprint"):
            if field not in record or sidecar.get(field) != record[field]:
                raise ValueError("Capture sidecar/report identity mismatch: " + field)
        for field in ("ids_sha256", "kv_sha256", "hidden_sha256", "model_fingerprint"):
            if not valid_digest(record[field]):
                raise ValueError("Invalid capture SHA256 identity: " + field)
        before = path.stat()
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["ids"])
            if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0):
                raise ValueError("Capture IDs must be a one-dimensional nonnegative integer array")
            if len(ids) != record["tokens"] or sha_array(ids) != record["ids_sha256"]:
                raise ValueError("Capture IDs do not match the recorded token identity")
            positions = np.asarray(archive["positions"]) if "positions" in archive.files else np.arange(len(ids), dtype=np.int64)
            if positions.shape != ids.shape or not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0):
                raise ValueError("Capture positions must match the token IDs")
            fields = {field: np.asarray(archive[field]).item() for field in ("Hk", "hd", "base") if field in archive.files}
        original_hash = sha_file(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError("Capture changed while its input manifest was extracted")
        outputs[split]["cases"].append({
            "name": name, "fixture": name, "kind": record.get("kind"), "tokens": int(len(ids)),
            "input_token_ids": ids.astype(np.int64).tolist(), "positions": positions.astype(np.int64).tolist(),
            "ids_dtype": str(ids.dtype), "ids_sha256": record["ids_sha256"],
            "kv_sha256": record["kv_sha256"], "hidden_sha256": record["hidden_sha256"],
            "model_fingerprint": record["model_fingerprint"], "capture_geometry": fields,
            "original_npz_file": filename, "original_npz_bytes": after.st_size,
            "original_npz_sha256": original_hash, "capture_sidecar_sha256": sha_file(sidecar_path),
            "capture_sidecar": {key: value for key, value in sidecar.items() if key not in ("capture_file",)},
            "scope": "Exact token IDs and identities retained; original H/KV arrays are omitted.",
        })
    if any(not outputs[split]["cases"] for split in SPLITS):
        raise ValueError("Exact input manifests require at least one TRAIN, DEV and TEST capture")
    return outputs


def package_transfer(run, review_directory, destination):
    source = real_path(run, "Run")
    review = real_path(review_directory, "Review")
    target = real_path(destination, "Destination")
    if not source.is_dir() or not review.is_dir():
        raise ValueError("Run and review must be existing directories")
    for incoming in (source, review):
        if target.is_relative_to(incoming) or incoming.is_relative_to(target):
            raise ValueError("Destination must be separate from the input run and review")
    fresh(target)
    # Only the explicitly known large-artifact trees may be present in a run.
    allowed_directories = set(RUN_DIRECTORIES)
    if review.is_relative_to(source):
        allowed_directories.add(review.relative_to(source).parts[0])
    for path in source.iterdir():
        if path.is_symlink():
            raise ValueError("Run must not contain symlink entries")
        if path.is_dir() and path.name not in allowed_directories:
            raise ValueError("Unexpected run artifact tree: " + path.name)
    reports = []
    for name in REPORTS:
        path = source / name
        if path.exists():
            regular_file(path)
            reports.append((path, Path(name)))
    review_files = []
    for path in sorted(review.iterdir()):
        if path.is_dir() or path.is_symlink() or path.suffix not in REVIEW_SUFFIXES:
            raise ValueError("Review must contain only flat small metadata files")
        regular_file(path)
        review_files.append((path, Path("review") / path.name))
    report, insights = validate_review(source, review)
    manifests = capture_input_manifests(source, report)
    generated = {Path("input_manifests") / (split + "_manifest.json"): canonical(value)
                 for split, value in manifests.items()}
    generated[Path("PACKAGING_SCOPE.md")] = (
        "# Compact transfer results\n\n"
        "This package contains byte-identical reports and a passing review that was executed "
        "against the full original run. It retains exact input token IDs, capture hashes, "
        "model identity and source hashes.\n\n"
        "Frozen predictor states, encoded packets, decoded arrays and original hidden/KV "
        "captures are omitted. The compact package cannot rerun the full-artifact review "
        "without those original artifacts. Review assertions are historical executed evidence, "
        "not new checks performed by this packager. Capture NPZ hashes identify the omitted files; "
        "IDs are independently checked during extraction.\n"
    ).encode()
    selected = reports + review_files
    total = sum(regular_file(path) for path, _ in selected) + sum(len(value) for value in generated.values())
    if total > MAX_TOTAL_BYTES or any(len(value) > MAX_FILE_BYTES for value in generated.values()):
        raise ValueError("Compact package exceeds its 90 MiB total or 50 MiB per-file limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transfer-package-", dir=target.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        ledger = []
        for path, relative in selected:
            before = path.stat()
            destination_path = staging / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination_path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError("Run report changed while packaging")
            digest = sha_file(destination_path)
            if digest != sha_file(path):
                raise ValueError("Copied artifact changed during packaging")
            ledger.append({"file": relative.as_posix(), "bytes": destination_path.stat().st_size, "sha256": digest})
        for relative, data in generated.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            ledger.append({"file": relative.as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        ledger.sort(key=lambda item: item["file"])
        delivered = {item["file"]: item for item in ledger}
        if delivered["results.json"]["sha256"] != insights["source_results_sha256"]:
            raise ValueError("Results changed after review validation; rerun packaging")
        staged_review = json.loads((staging / "review/insights.json").read_bytes())
        if staged_review != insights:
            raise ValueError("Review changed during packaging")
        manifest = {
            "format": "transfer_results_package_v1", "run_status": "completed",
            "source_results_sha256": delivered["results.json"]["sha256"],
            "review_all_passed": True, "review_passed": insights["passed"], "review_total": insights["total"],
            "review_insights_sha256": delivered["review/insights.json"]["sha256"],
            "exact_original_reproduction": report.get("exact_original_reproduction", False),
            "input_capture_counts": {split: len(value["cases"]) for split, value in manifests.items()},
            "files": ledger, "total_artifact_bytes": sum(item["bytes"] for item in ledger),
            "omitted": ["states/", "packet_checks/", "captures/*.npz", "original model weights", "unlisted run files"],
            "scope": "Recorded full-artifact review evidence; compact package alone cannot rerun that review. Manifest hashes cover delivered files, excluding this manifest itself.",
        }
        manifest_bytes = canonical(manifest)
        if len(manifest_bytes) > MAX_FILE_BYTES or len(manifest_bytes) + manifest["total_artifact_bytes"] > MAX_TOTAL_BYTES:
            raise ValueError("Package and ledger exceed the compact size limit")
        (staging / "package_manifest.json").write_bytes(manifest_bytes)
        fresh(target)
        os.replace(staging, target)
    return {"destination": str(target), "files": len(ledger) + 1,
            "artifact_bytes": manifest["total_artifact_bytes"], "manifest_bytes": len(manifest_bytes),
            "input_capture_counts": manifest["input_capture_counts"],
            "source_results_sha256": manifest["source_results_sha256"], "review_all_passed": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = package_transfer(args.input, args.review, args.destination)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
