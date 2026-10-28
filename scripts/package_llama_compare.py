#!/usr/bin/env python3
"""Copy the small results of a bench_llama_compare.py run into benchmark_results/ for committing.

    python scripts/package_llama_compare.py runs/llama32_3b_compare \\
        --destination benchmark_results/llama32_3b_compare_a4500_$(date +%Y%m%d)

Copies summary.md, lossless.json, lossy.json, frontier.json, gpu_speed.json, pool.json, run_args.json, environment.json,
run.log and the capture metadata
(captures/meta.json -> capture_meta.json). Captured KV (*.npy) stays local. Checks that the JSON reports parse and
agree with summary.md's method list, and writes manifest.json with each file's size and SHA256. Prints the git
commands that commit the results to the current branch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

FILES = {"summary.md": "summary.md", "lossless.json": "lossless.json", "lossy.json": "lossy.json",
         "run_args.json": "run_args.json", "environment.json": "environment.json", "run.log": "run.log",
         "console.log": "console.log", "frontier.json": "frontier.json", "gpu_speed.json": "gpu_speed.json",
         "pool.json": "pool.json", "cheap_products.json": "cheap_products.json", "format.json": "format.json",
         "check_aqua.json": "check_aqua.json", "check_kivi.json": "check_kivi.json", "exans.json": "exans.json",
         "exans/build_fix.diff": "exans_build_fix.diff", "exans/official_test.log": "exans_official_test.log",
         "time_aqua.json": "time_aqua.json", "time_kivi.json": "time_kivi.json",
         "lossy_eval.json": "lossy_eval.json", "captures/eval_meta.json": "eval_capture_meta.json",
         "szcodec.json": "szcodec.json", "captures/meta.json": "capture_meta.json"}
REQUIRED = ("summary.md", "run_args.json", "environment.json", "captures/meta.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(run):
    for name in REQUIRED:
        if not (run / name).is_file():
            raise SystemExit(f"missing {run / name}: run the report stage first")
    summary = (run / "summary.md").read_text()
    checks = []
    if (run / "lossless.json").is_file():
        res = json.loads((run / "lossless.json").read_text())
        for name in res["methods"]:
            if f"| {name} |" not in summary:
                raise SystemExit(f"summary.md lacks lossless method {name!r}: rerun --stages report")
        checks.append(f"lossless: {len(res['methods'])} methods, {res['layers']} layers")
    if (run / "lossy.json").is_file():
        res = json.loads((run / "lossy.json").read_text())
        for doc, rows in res["documents"].items():
            if rows["exact KV"]["ppl_change_pct"] != 0.0:
                raise SystemExit(f"{doc}: lossy rows are not measured against the exact KV")
        checks.append(f"lossy: {len(next(iter(res['documents'].values())))} rows x {len(res['documents'])} documents")
    if (run / "lossy_eval.json").is_file():
        res = json.loads((run / "lossy_eval.json").read_text())
        for doc, rows in res["documents"].items():
            if rows["exact KV"]["ppl_change_pct"] != 0.0:
                raise SystemExit(f"{doc}: lossy_eval rows are not measured against the exact KV")
        if res["documents"] and "## Lossy quality on WikiText-2" not in summary:
            raise SystemExit("summary.md lacks the WikiText-2 table: rerun --stages report")
        checks.append(f"lossy_eval: {len(res['documents'])} WikiText-2 windows")
    if (run / "szcodec.json").is_file():
        res = json.loads((run / "szcodec.json").read_text())
        if "## Our prediction in SplitZip's format: the real codec" not in summary:
            raise SystemExit("summary.md lacks the SplitZip-format codec section: rerun --stages report")
        rows = [r for r in res["rows"] if r["part"].startswith("szcodec")]
        checks.append(f"szcodec: {len(rows)} rows, {sum(1 for r in rows if r.get('bit_exact'))} bit-exact")
    if (run / "frontier.json").is_file():
        res = json.loads((run / "frontier.json").read_text())
        checks.append(f"frontier: {len(res['variants'])} variants, {res['layers']} layers")
    if (run / "gpu_speed.json").is_file():
        res = json.loads((run / "gpu_speed.json").read_text())
        checks.append(f"gpu_speed: {len(res.get('rows', []))} timings on {res.get('gpu')}")
    if (run / "pool.json").is_file():
        res = json.loads((run / "pool.json").read_text())
        checks.append(f"pool: {len(res.get('codec_rows', []))} codec timings, {res.get('tokens')} tokens")
    if not checks:
        raise SystemExit("no lossless.json, lossy.json, frontier.json, gpu_speed.json, pool.json or szcodec.json in the run")
    return checks


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args(argv)
    run, dest = args.run, args.destination
    if dest.exists():
        raise SystemExit(f"{dest} already exists; choose a new name")
    checks = check(run)
    dest.mkdir(parents=True)
    manifest = {"source_run": str(run), "checks": checks, "files": {}}
    for src, name in FILES.items():
        if (run / src).is_file():
            shutil.copyfile(run / src, dest / name)
            manifest["files"][name] = {"bytes": (dest / name).stat().st_size, "sha256": sha256(dest / name)}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("\n".join(checks))
    print(f"\nPackaged {len(manifest['files'])} files in {dest}. Commit them to this branch:\n")
    print(f"  git add {dest}")
    print(f"  git commit -m \"Add Llama-3.2-3B codec comparison results ({dest.name})\"")
    print("  git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
