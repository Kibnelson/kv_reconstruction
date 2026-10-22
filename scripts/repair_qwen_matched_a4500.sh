#!/usr/bin/env bash
# Remeasure saved targets and frozen mean predictors; never run capture or fit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source_dir="${1:?Usage: bash scripts/repair_qwen_matched_a4500.sh runs/ORIGINAL_RUN [--publish-results]}"
shift
test -f "$source_dir/capture_manifest.json"
test -f "$source_dir/calibration/wikitext2/fit_manifest.json"
test -f "$source_dir/calibration/c4/fit_manifest.json"
publish=0
if [[ "${1:-}" == --publish-results ]]; then publish=1; shift; fi
if (( $# )); then printf 'Unexpected argument: %s\n' "$1" >&2; exit 2; fi
if (( publish )) && [[ "$(git branch --show-current)" != experiment/qwen_matched_gpu ]]; then
  printf 'Publishing requires experiment/qwen_matched_gpu.\n' >&2; exit 2
fi
source .venv/bin/activate
unset TRITON_INTERPRET
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -c 'import torch,triton; assert torch.cuda.is_available(), "Actual CUDA required"; print(torch.cuda.get_device_name(0))'
run_dir="runs/qwen_matched_fixed_width_a4500_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir"
python scripts/build_standalone.py
python scripts/build_standalone.py --check
python scripts/setup_qwen_matched_baselines.py --install-nvcomp
cp runs/_matched_baselines/setup.json "$run_dir/baseline_setup.json"
status=0
python -m unittest tests.test_qwen_matched_gpu.CudaTests tests.test_qwen_matched_gpu.CudaNativeTests -v \
  2>&1 | tee "$run_dir/cuda_tests.log" || status=$?
if (( status )); then
  printf 'CUDA preflight failed. Original captures and fits are preserved. Log: %s/cuda_tests.log\n' "$run_dir" >&2
  exit "$status"
fi
methods=raw_bf16,block32_A,previous32_A,aqua_style32_A,strong_layers32_A,strong_layers64_A,splitzip_native,nvcomp_bitcomp,nvcomp_cascaded
python -u scripts/bench_qwen_matched_gpu.py --phase bench --captures-from "$source_dir" \
  --output "$run_dir" --methods "$methods" || status=$?
cat "$run_dir/summary.md"
report_dir="benchmark_results/$(basename "$run_dir")"
python scripts/package_qwen_matched_gpu.py "$run_dir" --destination "$report_dir"
if (( publish )); then
  git add qwen_kv_gpu_bench.py
  git add -f "$report_dir" "$report_dir.zip"
  git commit --only -m "Add matched fixed-width Qwen A4500 comparison (exit $status)" -- qwen_kv_gpu_bench.py "$report_dir" "$report_dir.zip"
  git push origin HEAD:experiment/qwen_matched_gpu
fi
exit "$status"
