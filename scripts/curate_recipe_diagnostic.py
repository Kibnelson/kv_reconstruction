#!/usr/bin/env python3
"""Preserve a completed recipe diagnostic in a compact, auditable directory.

The complete results JSON is compressed losslessly; no methods, failed rows,
per-record traces or diagnostic fields are selected away.  Only explicitly
named reports are copied, so this utility never exports source KV arrays.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


REPORTS = (
    "summary.md", "results.csv", "source_manifest.json",
    "frozen_manifest.json", "recipe_analysis.json",
)
_CHUNK_BYTES = 1 << 20


def _stream_info(stream):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def _file_info(path):
    with path.open("rb") as stream:
        return _stream_info(stream)


def curate(input_dir: Path, output: Path):
    """Create a fresh output directory and return its integrity manifest."""
    source = Path(input_dir).resolve()
    target = Path(output).absolute()
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output must be a fresh directory: {target}")
    required = ("results.json", *REPORTS)
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f"Missing diagnostic report: {source / name}")
    with (source / "results.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError("Only completed diagnostics may be curated (results.json status must be 'completed')")
    # Bind the actual complete JSON bytes, including their original whitespace.
    original = _file_info(source / "results.json")
    del result
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recipe-curation-", dir=target.parent) as temporary:
        staging = Path(temporary)
        for name in REPORTS:
            before = _file_info(source / name)
            shutil.copyfile(source / name, staging / name)
            if _file_info(staging / name) != before or _file_info(source / name) != before:
                raise ValueError(f"Diagnostic report changed while copying: {name}")
        with (source / "results.json").open("rb") as incoming, \
                (staging / "results.json.gz").open("wb") as outgoing:
            # Empty embedded filename and mtime=0 keep the gzip header stable
            # across output paths and reruns with the same Python/zlib build.
            with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing,
                               compresslevel=9, mtime=0) as compressed:
                shutil.copyfileobj(incoming, compressed, length=_CHUNK_BYTES)
        with gzip.open(staging / "results.json.gz", "rb") as restored:
            roundtrip = _stream_info(restored)
        if roundtrip != original or _file_info(source / "results.json") != original:
            raise ValueError("Full results JSON changed or failed lossless gzip verification")
        readme = """# Recipe diagnostic reports

These files preserve a completed recipe diagnostic. The complete `results.json`,
including every method and per-record trace, is stored losslessly in
`results.json.gz`; no result filtering or tuning is performed by this utility.

Read `summary.md` and `recipe_analysis.json` for findings. Consult the reports'
environment, configuration and scope fields to distinguish CPU functional
measurements from GPU measurements. These diagnostics alone do not establish
long-context, concurrent-serving performance or compact-resident attention.

Restore the original JSON bytes with:

```bash
gzip -dc results.json.gz > results.json
```

`artifact_manifest.json` records SHA256 and byte counts for every other delivered
file, plus the uncompressed results SHA256 and byte count. Gzip decompression was
verified against the complete original JSON before this folder was created.

Raw KV arrays and calibration model binaries are not included. Full calibration
provenance, artifact identities and accounting remain in the original reports;
the source and frozen manifests are copied unchanged.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")
        files = {path.name: _file_info(path) for path in sorted(staging.iterdir())}
        manifest = {
            "format": "qwen_recipe_diagnostic_artifacts_v1",
            "source_directory_name": source.name,
            "source_status": "completed",
            "results_preserved_without_filtering": True,
            "raw_kv_included": False,
            "calibration_model_binaries_included": False,
            "calibration_provenance": "Complete original results.json.gz and unchanged frozen/source manifests",
            "uncompressed_results": {
                "name": "results.json", **original,
                "compressed_as": "results.json.gz", "gzip_roundtrip_verified": True,
            },
            "files": files,
            "manifest_scope": "Every delivered file except artifact_manifest.json itself",
        }
        (staging / "artifact_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # mkdir is exclusive even if another process creates the target after
        # preflight.  All expensive validation finishes before publishing files.
        target.mkdir()
        for path in staging.iterdir():
            path.replace(target / path.name)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Completed diagnostic directory containing the original reports")
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh directory to receive the compact report artifacts")
    args = parser.parse_args(argv)
    try:
        manifest = curate(args.input_dir, args.output)
    except (OSError, ValueError) as error:
        print(f"Recipe diagnostic curation: {error}", file=sys.stderr)
        return 2
    original = manifest["uncompressed_results"]
    print(f"Curated {len(manifest['files']) + 1} files into {args.output}")
    print(f"Complete results.json: {original['bytes']:,} bytes; SHA256 {original['sha256']}")
    print("Verified lossless deterministic gzip; no result fields or rows removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
