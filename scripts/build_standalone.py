#!/usr/bin/env python3
"""Build the portable benchmark deterministically from reviewable repository source.

Run ``python scripts/build_standalone.py`` after editing benchmark code, or add
``--check`` to detect a stale checked-in launcher without changing any files.
Only the Python standard library is needed to build or check the launcher.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import textwrap
import zipfile

NAMESPACES = (
    "kv_binary", "kv_combinations", "kv_coset_fields", "kv_research",
    "kv_residual", "kv_smoke", "kv_stacked", "qwen_kv_bench",
)
REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = "_qwen_source_manifest.json"


def collect_sources(root: Path) -> dict[str, bytes]:
    """Select source, tests, optional C++ source, and the embedded usage guide."""
    root = Path(root)
    files: dict[str, bytes] = {}
    for namespace in NAMESPACES:
        directory = root / namespace
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Missing source directory (or symlink): {directory}")
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix not in (".py", ".cpp"):
                continue
            if path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                         if parent != root and parent.is_relative_to(root)):
                raise ValueError(f"Source must not be a symlink: {path}")
            if path.is_file():
                files[relative.as_posix()] = path.read_bytes()
    guide = root / "qwen_kv_bench" / "README.md"
    if guide.is_symlink() or not guide.is_file():
        raise ValueError(f"Missing embedded usage guide: {guide}")
    files["qwen_kv_bench/README.md"] = guide.read_bytes()
    return dict(sorted(files.items()))


def make_payload(files: dict[str, bytes]) -> tuple[bytes, int]:
    """Use fixed ordering, permissions, timestamps, and a content manifest."""
    files = dict(files)
    if MANIFEST in files:
        raise ValueError("The source manifest must be generated, not supplied")
    manifest = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in sorted(files.items())}
    files[MANIFEST] = (json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n").encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data, compresslevel=9)
    return stream.getvalue(), len(files)


def build(root: Path = REPOSITORY, template: Path | None = None) -> bytes:
    root = Path(root)
    template = template or root / "scripts" / "standalone_launcher.py.in"
    payload, count = make_payload(collect_sources(root))
    encoded = "\n".join("    " + repr(line) for line in
                        textwrap.wrap(base64.b64encode(payload).decode("ascii"), 88))
    replacements = {
        "@@PAYLOAD_SHA256@@": repr(hashlib.sha256(payload).hexdigest()),
        "@@MEMBER_COUNT@@": str(count),
        "@@PAYLOAD_B64@@": "(\n" + encoded + "\n)",
    }
    rendered = Path(template).read_text(encoding="utf-8")
    for marker, replacement in replacements.items():
        if rendered.count(marker) != 1:
            raise ValueError(f"Launcher template must contain exactly one {marker}")
        rendered = rendered.replace(marker, replacement)
    ast.parse(rendered)
    return rendered.encode("utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the launcher is stale; do not write")
    parser.add_argument("--root", type=Path, default=REPOSITORY, help="Repository root")
    parser.add_argument("--output", type=Path, help="Output path; defaults to ROOT/qwen_kv_gpu_bench.py")
    args = parser.parse_args(argv)
    target = args.output or args.root / "qwen_kv_gpu_bench.py"
    try:
        data = build(args.root)
        if args.check:
            if not target.is_file() or target.read_bytes() != data:
                print(f"Stale standalone script: {target}. Run python scripts/build_standalone.py", file=sys.stderr)
                return 1
            print(f"Standalone script matches source: {target}")
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"Built {target} ({len(data):,} bytes, SHA256 {hashlib.sha256(data).hexdigest()})")
        return 0
    except (OSError, ValueError, SyntaxError) as exc:
        print(f"Standalone build: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
