#!/usr/bin/env bash
# Publish prerequisite inventory only; no CUDA benchmark/training is started.
set -euo pipefail
cd "$(dirname "$0")/.."
branch="experiment/llama_causal_packet_compare"
test "$(git branch --show-current)" = "$branch" || { echo "Switch to $branch first."; exit 2; }
test "$#" -eq 0 || { echo "This inventory wrapper takes no arguments."; exit 2; }
run_dir="runs/nuc_baseline_readiness_$(date +%Y%m%d_%H%M%S)_$$"
python3 scripts/check_nuc_baseline_readiness.py --output "$run_dir"
cat "$run_dir/summary.md"
report_dir="benchmark_results/$(basename "$run_dir")"
mkdir "$report_dir"
cp -- "$run_dir/results.json" "$run_dir/summary.md" "$report_dir/"
git add -f -- "$report_dir"
git commit --only -m "Add NUC official-baseline readiness inventory" -- "$report_dir"
git -c rebase.autoStash=true pull --rebase origin "$branch"
git push origin "HEAD:$branch"
echo "Published inventory: $report_dir/summary.md"
