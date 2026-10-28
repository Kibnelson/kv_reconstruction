#!/usr/bin/env python3
"""Package small causal-comparison evidence. Packets, KV and fitted tensors stay local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

FILES = ["results.json", "summary.md", "input_manifest.json", "source_manifest.json",
         "independent_replay.json", "independent_replay.log", "console.log"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path); p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args(argv)
    run, dest = args.run.resolve(), args.destination
    if dest.exists():
        raise ValueError("fresh report destination required")
    result = json.loads((run / "results.json").read_text())
    if result.get("codec_completed"):
        checks = {c["id"]: c for c in json.loads((run / "independent_replay.json").read_text())["checks"]}
        if set(checks) != {r["id"] for r in result["rows"]}:
            raise ValueError("independent replay coverage differs")
        for r in result["rows"]:
            if r["status"] != "verified" or not checks[r["id"]]["verified"]:
                raise ValueError("unverified row")
            if r["decoded_word_sha256"] != checks[r["id"]]["word_sha256"]:
                raise ValueError("receiver word digest differs")
            if r["mode"] == "lossless" and not r["original_exact"]:
                raise ValueError("lossless result is not exact")
            for name, field in (("packet", "packet_sha256"), ("decoded", "decoded_file_sha256")):
                path = run / r[name]
                if not path.resolve().is_relative_to(run) or sha(path) != r[field]:
                    raise ValueError("saved packet/receiver evidence changed")
            if abs(r["complete_bpv"] - 8 * r["packet_bytes"] / r["values"]) > 1e-10:
                raise ValueError("packet accounting differs")
    if result["status"] == "completed" and not (result.get("codec_completed") and result.get("quality_completed")):
        raise ValueError("completion requires codec and quality evidence")
    dest.mkdir(parents=True)
    manifest = dict(status=result["status"], gpu_measured=result.get("gpu_measured", False), files={},
        scope="small evidence only; no raw/compressed KV, model weights or fitted tensor state")
    for name in FILES:
        src = run / name
        if not src.is_file():
            continue
        if src.is_symlink() or not src.resolve().is_relative_to(run):
            raise ValueError("report symlink forbidden")
        shutil.copyfile(src, dest / name)
        manifest["files"][name] = dict(bytes=src.stat().st_size, sha256=sha(src))
    (dest / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
