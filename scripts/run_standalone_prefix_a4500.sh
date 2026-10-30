#!/usr/bin/env bash
# Measure same-packet warp-prefix compaction against packed-inverse decoding.
set -euo pipefail
cd "$(dirname "$0")/.."
branch="experiment/standalone_prefix_v1"
test "$(git branch --show-current)" = "$branch" || { echo "Switch to $branch first."; exit 2; }
test "$#" -eq 0 || { echo "This wrapper takes no arguments."; exit 2; }
unset TRITON_INTERPRET
run_dir="runs/standalone_prefix_a4500_$(date +%Y%m%d_%H%M%S)_$$"
code=0
python3 -u scripts/bench_standalone_prefix.py --output "$run_dir" || code=$?
test -f "$run_dir/results.json" || { echo "No report produced. Preserve the terminal output."; exit 2; }
cat "$run_dir/summary.md"
report_dir="benchmark_results/$(basename "$run_dir")"
mkdir "$report_dir"
python3 - "$run_dir" "$report_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import shutil
import sys

source, target = map(Path, sys.argv[1:])
manifest = {}
for path in sorted(source.rglob('*')):
    if not path.is_file() or path.suffix not in ('.json', '.md', '.log') or path.name == 'full_trace.json':
        continue
    relative = path.relative_to(source)
    destination = target / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    manifest[str(relative)] = dict(bytes=destination.stat().st_size,
        sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
(target / 'package_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
PY
git add -f -- "$report_dir"
git commit --only -m "Add standalone prefix A4500 results (exit $code)" -- "$report_dir"
git -c rebase.autoStash=true pull --rebase origin "$branch"
git push origin "HEAD:$branch"
echo "Published: $report_dir/summary.md"
echo "Encoded packets and full profiler traces remain in $run_dir"
exit "$code"
