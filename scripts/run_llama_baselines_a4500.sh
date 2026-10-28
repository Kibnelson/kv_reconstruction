#!/usr/bin/env bash
# Reuses verified Block32/Lite results. No model download, fitting or baseline source edits.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
unset TRITON_INTERPRET
export CUBLAS_WORKSPACE_CONFIG=:4096:8
branch="experiment/llama_causal_packet_compare"
test "$(git branch --show-current)" = "$branch" || {
  echo "Switch to $branch before publishing this comparison."
  exit 2
}
base_run="runs/llama_causal_a4500_20261027_184615_1307629"
if (( $# )); then base_run="$1"; shift; fi
python -c 'import torch,triton; assert torch.cuda.is_available(), "Actual CUDA required"; print(torch.cuda.get_device_name(0))'
run_dir="runs/llama_baselines_a4500_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir"
echo "Reuse completed causal run: $base_run"
echo "New baseline output: $run_dir"
if python -u scripts/bench_llama_baselines.py --base-run "$base_run" --output "$run_dir" "$@" \
    2>&1 | tee "$run_dir/console.log"; then
  run_status=0
else
  run_status=$?
fi
report_dir="benchmark_results/$(basename "$run_dir")"
python scripts/package_llama_baselines.py "$run_dir" --destination "$report_dir"
git add -f -- "$report_dir"
git commit --only -m "Add matched Llama baseline controls (exit $run_status)" -- "$report_dir"
git -c rebase.autoStash=true pull --rebase origin "$branch"
git push origin "HEAD:$branch"
echo "Published: $report_dir/summary.md"
exit "$run_status"
