#!/usr/bin/env python3
"""Publish only checked small baseline evidence; tensor payloads stay on the GPU host."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.package_llama_causal import sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path); p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args(argv)
    run, dest = args.run.resolve(), args.destination
    if dest.exists():
        raise ValueError("fresh destination required")
    result = json.loads((run/"results.json").read_text())
    base = run/"base_results.json"
    if base.exists():
        if sha(base) != result["base"]["results_sha256"]:
            raise ValueError("base report changed")
        if result["reused_rows"] != json.loads(base.read_text())["rows"]:
            raise ValueError("reused rows were modified")
    allowed = ["results.json", "summary.md", "source_manifest.json", "base_results.json",
               "base_independent_replay.json", "console.log"]
    for method, worker in result.get("workers",{}).items():
        if method not in ("raw_bf16", "fp8_e4m3fn_saturating", "splitzip_official_framed"):
            raise ValueError("unexpected baseline worker")
        proof = json.loads((run/method/"receiver.json").read_text())
        if not proof["gpu_measured"] or not proof["no_source_kv_or_predictions"]:
            raise ValueError("actual independent GPU receiver proof required")
        checks = {c["id"]: c for c in proof["checks"]}
        for row in [r for r in result["rows"] if r["method"] == method]:
            c = checks[row["id"]]
            if not c["verified"] or row["status"] != "verified" or c["word_sha256"] != row["decoded_word_sha256"]:
                raise ValueError("unverified baseline row")
            if row["mode"] == "lossless" and not row["original_exact"]:
                raise ValueError("lossless baseline changed original words")
            for field, digest in (("packet","packet_sha256"),("decoded","decoded_file_sha256")):
                path = Path(row[field]).resolve()
                if not path.is_relative_to(run) or sha(path) != row[digest]:
                    raise ValueError("baseline packet/receiver evidence changed")
            if c["decoded_file_sha256"] != row["decoded_file_sha256"]:
                raise ValueError("receiver file proof differs")
            if abs(row["complete_bpv"]-8*row["packet_bytes"]/row["values"])>1e-10:
                raise ValueError("complete packet accounting differs")
        allowed += [method+".log", method+"/worker_results.json", method+"/receiver.json", method+"/receiver.log"]
        allowed += [str(f.relative_to(run)) for f in (run/method).glob("*_native.json")]
        allowed += [str(f.relative_to(run)) for f in (run/method).glob("*_native.log")]
    for method in result.get("baseline_status",{}):
        if method in ("raw_bf16", "fp8_e4m3fn_saturating", "splitzip_official_framed"):
            allowed.append(method+".log")
    if result["status"] == "completed" and (not result.get("quality_completed") or len(result.get("workers",{}))!=3):
        raise ValueError("completed comparison lacks baseline/quality evidence")
    dest.mkdir(parents=True)
    manifest = dict(status=result["status"], gpu_measured=result.get("gpu_measured",False), files={},
                    scope="small evidence; no KV packets, raw arrays, model weights or fitted tensors")
    for name in sorted(set(allowed)):
        source = run/name
        if not source.is_file():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(run):
            raise ValueError("report symlinks forbidden")
        target = dest/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source,target)
        manifest["files"][name] = dict(bytes=source.stat().st_size, sha256=sha(source))
    (dest/"package_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(dest/"summary.md")
    return 0


if __name__ == "__main__":
    # Direct invocation also needs the repository root on sys.path.
    sys.exit(main())
