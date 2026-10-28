#!/usr/bin/env python3
"""Package small, verified baseline/tuning evidence; keep tensors and packets local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kv_compare.lite_scoreboard import aggregate, network_model, validate_comparison


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checked(run, name):
    p = run / name
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(run.resolve()):
        raise ValueError("missing/outside-run evidence: " + str(name))
    return p


def verify(run, result):
    if result["status"] == "completed":
        if not result.get("gpu_measured"):
            raise ValueError("CPU evidence cannot be a completed GPU benchmark")
        if not result.get("selected_jobs") or set(result["jobs"]) != set(result["selected_jobs"]):
            raise ValueError("completed method coverage differs")
        expected_rows = [row for job in result["jobs"].values() if job.get("status") == "completed" for row in job["rows"]]
        if result["rows"] != expected_rows:
            raise ValueError("scoreboard differs from completed worker rows")
        recipe = result["recipe"]
        if hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest() != result["identity"]:
            raise ValueError("recipe identity differs")
        validate_comparison(result["rows"])
        if aggregate(result["rows"]) != result["aggregates"] or network_model(result["aggregates"]) != result["network_model"]:
            raise ValueError("derived scoreboard differs")
    for method, job in result.get("jobs", {}).items():
        if job.get("status") != "completed":
            if result["status"] == "completed":
                raise ValueError("unfinished method")
            continue
        for name, wanted in job["artifacts"].items():
            if sha(checked(run, name)) != wanted:
                raise ValueError("completed worker evidence changed")
        data = json.loads(checked(run, method + "/worker_results.json").read_text())
        proof_path = checked(run, method + "/receiver.json")
        proof = json.loads(proof_path.read_text())
        checks = {e["id"]: e for e in proof["checks"]}
        if (not data.get("gpu_measured") or not proof.get("gpu_measured") or
                not proof.get("no_source_kv_or_predictions") or data["rows"] != job["rows"] or
                sha(proof_path) != data["receiver_sha256"] or
                set(checks) != {r["id"] for r in job["rows"]} or len(checks) != len(job["rows"])):
            raise ValueError("independent CUDA receiver coverage differs")
        for row in job["rows"]:
            packet = Path(row["packet"])
            if not packet.resolve().is_relative_to(run.resolve()) or packet.is_symlink() or sha(packet) != row["packet_sha256"]:
                raise ValueError("completed packet changed")
            c = checks[row["id"]]
            if (not c["verified"] or c["packet_sha256"] != row["packet_sha256"] or
                    c["word_sha256"] != row["decoded_word_sha256"] or c["shape"] != row["shape"]):
                raise ValueError("receiver words differ")
            if row["mode"] == "lossless" and row["original_word_sha256"] != row["decoded_word_sha256"]:
                raise ValueError("lossless original-word mismatch")
            if abs(row["complete_bpv"] - 8 * row["packet_bytes"] / row["values"]) > 1e-10:
                raise ValueError("complete packet accounting differs")
            if method.startswith("lite_b64") and not row["packet_matches_original"]:
                raise ValueError("original Lite control packet differs")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path); p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args(argv); run = args.run.resolve(); dest = args.destination
    if dest.exists():
        raise ValueError("fresh report destination required")
    result = json.loads(checked(run, "results.json").read_text()); verify(run, result)
    names = [name for name in ("results.json", "summary.md", "source_manifest.json", "console.log") if (run / name).exists()]
    for method in result.get("jobs", {}):
        names += [method + "/" + name for name in ("worker_results.json", "receiver.json", "receiver.log", "worker.log")
                  if (run / method / name).exists()]
    sources = {name: checked(run, name) for name in names}
    dest.mkdir(parents=True)
    manifest = dict(status=result["status"], gpu_measured=result.get("gpu_measured", False), files={},
        scope="small evidence only; no KV tensors, packets, fitted state or model weights")
    for name, src in sources.items():
        target = dest / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, target)
        manifest["files"][name] = dict(bytes=src.stat().st_size, sha256=sha(src))
    (dest / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest / "summary.md"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
