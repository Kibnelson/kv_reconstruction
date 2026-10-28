#!/usr/bin/env python3
"""Publish small Lite optimization evidence; keep KV, state and traces local."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

FILES = ("results.json", "summary.md", "source_manifest.json", "base_results.json",
         "independent_replay.json", "receiver.log", "console.log")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked(root, name):
    path = root / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("missing/outside-run evidence: " + str(name))
    return path


def validate(run, result):
    if result["status"] != "completed":
        return
    if not result.get("gpu_measured"):
        raise ValueError("CPU evidence cannot be a completed GPU optimization")
    recipe = result["recipe"]
    if hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest() != result["identity"]:
        raise ValueError("recipe identity differs")
    if sha(checked(run, "base_results.json")) != recipe["base_results_sha256"]:
        raise ValueError("original result evidence changed")
    base = json.loads((run / "base_results.json").read_text())
    original = {r["id"]: r for r in base["rows"] if r["policy"] == "lite"}
    proof_path = checked(run, "independent_replay.json")
    if sha(proof_path) != result["receiver_sha256"]:
        raise ValueError("receiver proof changed")
    proof = json.loads(proof_path.read_text())
    checks = {c["id"]: c for c in proof["checks"]}
    ids = result["selected_ids"]
    if (not proof.get("gpu_measured") or not proof.get("no_source_kv_or_prediction_arrays") or
            len(checks) != len(proof["checks"]) or len(ids) != len(set(ids)) or not ids or
            set(checks) != set(ids) or {r["id"] for r in result["rows"]} != set(ids) or
            len(result["rows"]) != len(ids)):
        raise ValueError("independent GPU receiver coverage differs")
    for row in result["rows"]:
        old, received = original[row["id"]], checks[row["id"]]
        if (not received["verified"] or row["status"] != "matched" or not row["packet_matches_original"] or
                received["packet_sha256"] != row["packet_sha256"] or
                received["word_sha256"] != row["decoded_word_sha256"] or received["shape"] != row["shape"]):
            raise ValueError("unverified optimization row")
        for key in ("packet_sha256", "decoded_word_sha256", "original_word_sha256", "state_sha256",
                    "packet_bytes", "values", "kappa", "document", "original_exact"):
            if row[key] != old[key]:
                raise ValueError("optimization differs from original: " + key)
        if (not row["kappa"] and row["decoded_word_sha256"] != row["original_word_sha256"] or
                abs(row["complete_bpv"] - 8 * row["packet_bytes"] / row["values"]) > 1e-10):
            raise ValueError("lossless words or packet accounting differ")
        path = checked(Path(recipe["base_run"]), old["packet"])
        if sha(path) != row["packet_sha256"]:
            raise ValueError("original packet changed")
        for direction in ("encode", "decode"):
            for arm in ("reference", "host_rope"):
                t = row[direction][arm]
                if len(t["samples"]) != recipe["repeats"] or len(t["warmup_samples"]) != recipe["warmup"]:
                    raise ValueError("ordinary sample coverage incomplete")
    if recipe["profile"]:
        expected = {r["id"] for r in result["rows"] if r["document"] == "test_text" and r["kappa"] == 0}
        if set(result["profiles"]) != expected:
            raise ValueError("diagnostic coverage incomplete")
        for arms in result["profiles"].values():
            if set(arms) != {"reference", "host_rope"}:
                raise ValueError("diagnostic arm missing")
            for profile in arms.values():
                if set(profile) != {"encode", "decode"}:
                    raise ValueError("diagnostic direction missing")
                for d in profile.values():
                    if sha(checked(run, d["trace"])) != d["trace_sha256"]:
                        raise ValueError("diagnostic trace changed")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args(argv)
    run, dest = args.run.resolve(), args.destination
    if dest.exists():
        raise ValueError("fresh report destination required")
    result = json.loads(checked(run, "results.json").read_text())
    validate(run, result)
    dest.mkdir(parents=True)
    manifest = dict(status=result["status"], gpu_measured=result.get("gpu_measured", False), files={},
        scope="small evidence only; no KV packets, captures, fitted tensors or full traces")
    for name in FILES:
        if not (run / name).exists():
            continue
        source = checked(run, name)
        shutil.copyfile(source, dest / name)
        manifest["files"][name] = dict(bytes=source.stat().st_size, sha256=sha(source))
    (dest / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(dest / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
