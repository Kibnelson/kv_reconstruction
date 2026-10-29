#!/usr/bin/env bash
# Fresh-build and run the released author correctness test, then publish receipts.
set -euo pipefail
cd "$(dirname "$0")/.."
branch="experiment/llama_causal_packet_compare"
test "$(git branch --show-current)" = "$branch" || { echo "Switch to $branch first."; exit 2; }
test "$#" -eq 0 || { echo "This wrapper takes no arguments."; exit 2; }
run_dir="runs/exans_official_preflight_a4500_$(date +%Y%m%d_%H%M%S)_$$"
code=0
python3 -u scripts/run_exans_official_preflight.py --output "$run_dir" || code=$?
test -f "$run_dir/results.json" || { echo "No report produced. Preserve the terminal output."; exit 2; }
cat "$run_dir/summary.md"
report_dir="benchmark_results/$(basename "$run_dir")"
mkdir "$report_dir"
# Keep binaries/build files locally. Publish only explicitly produced receipts/logs.
python3 - "$run_dir" "$report_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

source, target = map(Path, sys.argv[1:])
manifest = {}
paths = list(source.glob('*.json')) + list(source.glob('*.md')) + list(source.glob('*.log'))
paths += list((source / 'logs').glob('*.log'))
for path in sorted(paths):
    if not path.is_file():
        continue
    relative = path.relative_to(source)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    manifest[str(relative)] = dict(bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
(target / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
git add -f -- "$report_dir"
git commit --only -m "Add released ExANS A4500 correctness preflight (exit $code)" -- "$report_dir"
git -c rebase.autoStash=true pull --rebase origin "$branch"
git push origin "HEAD:$branch"
echo "Published: $report_dir/summary.md"
exit "$code"
