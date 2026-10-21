#!/usr/bin/env bash
# Run and package every available component; optional publication of reports.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

publish=0
if [[ "${1:-}" == "--publish-results" ]]; then
  publish=1
  shift
  branch="$(git branch --show-current)"
  if [[ "$branch" != "experiment/gpu_kv_path" ]]; then
    printf 'Run on experiment/gpu_kv_path before publishing results.\n' >&2
    exit 1
  fi
fi

source .venv/bin/activate
unset TRITON_INTERPRET
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python scripts/build_standalone.py
python scripts/build_standalone.py --check
if (( publish )) && ! git diff --quiet -- qwen_kv_gpu_bench.py; then
  git add qwen_kv_gpu_bench.py
  git commit --only -m "Regenerate standalone runner for GPU KV path" -- qwen_kv_gpu_bench.py
  git push origin HEAD:experiment/gpu_kv_path
fi

run_dir="runs/gpu_path_a4500_$(date +%Y%m%d_%H%M%S)_$$"
args=(--output "$run_dir" --attention-limit 0 --transfer-limit 0
  --warmup 1 --repeats 3 --attention-repeats 20)
previous="runs/pool_p1_a4500_20261021_090226"
if [[ -f "$previous/results.json" ]]; then
  args+=(--previous-p1 "$previous")
fi
captures="runs/block32_probability_qwen4_20261021_065346"
if [[ -f "$captures/capture_manifest.json" && -f "$captures/calibration/base.npz" && -f "$captures/calibration/B_decoder.npz" ]]; then
  args+=(--captures-from "$captures" --arms A,B)
else
  printf 'Original Qwen captures/calibration unavailable; frozen-fixture tests will run.\n'
fi
if python -c 'import constriction' >/dev/null 2>&1; then
  args+=(--cpu-t0-control)
else
  printf 'constriction unavailable; optional CPU T0 comparison is disabled. GPU T1 tests remain enabled.\n'
fi

status=0
python -u scripts/bench_gpu_path.py "${args[@]}" "$@" || status=$?
cat "$run_dir/summary.md"
report_dir="benchmark_results/$(basename "$run_dir")"
python scripts/package_gpu_path.py "$run_dir" --destination "$report_dir" --allow-failed
printf 'Packaged report: %s\n' "$report_dir"

if (( publish )); then
  git add qwen_kv_gpu_bench.py
  git add -f "$report_dir" "$report_dir.zip"
  # --only preserves unrelated staged changes outside these generated paths.
  git commit --only -m "Add A4500 GPU path batch (exit $status)" -- qwen_kv_gpu_bench.py "$report_dir" "$report_dir.zip"
  git push origin HEAD:experiment/gpu_kv_path
fi
exit "$status"
