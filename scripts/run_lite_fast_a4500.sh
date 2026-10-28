#!/usr/bin/env bash
# Reuse completed Lite captures/fits; publish only small optimization evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
unset TRITON_INTERPRET
export CUBLAS_WORKSPACE_CONFIG=:4096:8
branch="experiment/llama_causal_packet_compare"
test "$(git branch --show-current)" = "$branch" || { echo "Switch to $branch first."; exit 2; }
python -c 'import torch,triton; assert torch.cuda.is_available(), "Actual CUDA required"; print(torch.cuda.get_device_name(0))'
python -m unittest tests.test_lite_fast.CudaTests -v
base_run="runs/llama_causal_a4500_20261027_184615_1307629"
if [[ "$#" -ge 2 && "$1" == "--resume" ]]; then
  run_dir="$2"; shift 2
  test -f "$run_dir/results.json"
else
  if [[ "$#" -ge 1 && "$1" != --* ]]; then base_run="$1"; shift; fi
  run_dir="runs/lite_fast_a4500_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$run_dir"
fi
echo "Existing Lite run: $base_run"
echo "Optimization output / resume directory: $run_dir"
if python -u scripts/bench_lite_fast.py --base-run "$base_run" --output "$run_dir" --profile "$@" \
    2>&1 | tee -a "$run_dir/console.log"; then
  run_status=0
else
  run_status=$?
fi
report_dir="benchmark_results/$(basename "$run_dir")_report_$(date +%Y%m%d_%H%M%S)"
python scripts/package_lite_fast.py "$run_dir" --destination "$report_dir"
git add -f -- "$report_dir"
git commit --only -m "Add paired Lite host-RoPE A4500 comparison (exit $run_status)" -- "$report_dir"
git -c rebase.autoStash=true pull --rebase origin "$branch"
git push origin "HEAD:$branch"
echo "Published: $report_dir/summary.md"
echo "Full profiler traces remain in $run_dir/traces/"
exit "$run_status"
