#!/usr/bin/env bash
# One capture/fit/GPU comparison batch. Optional report publication is explicit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
publish=0
preset=pilot
run_dir="runs/qwen_matched_a4500_$(date +%Y%m%d_%H%M%S)_$$"
extra=()
while (( $# )); do
  case "$1" in
    --publish-results) publish=1; shift ;;
    --full) preset=full; shift ;;
    --resume) run_dir="$2"; shift 2 ;;
    --output|--preset) printf 'Use --resume DIR or --full with this wrapper.\n' >&2; exit 2 ;;
    *) extra+=("$1"); shift ;;
  esac
done
if (( publish )) && [[ "$(git branch --show-current)" != experiment/qwen_matched_gpu ]]; then
  printf 'Publishing requires branch experiment/qwen_matched_gpu.\n' >&2
  exit 2
fi
source .venv/bin/activate
unset TRITON_INTERPRET
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -c 'import torch,triton; assert torch.cuda.is_available(), "Actual CUDA required"; print(torch.cuda.get_device_name(0))'
if ! python -c 'import datasets' >/dev/null 2>&1; then
  python -m pip install 'datasets==4.2.0'
fi
if ! python -c 'import constriction' >/dev/null 2>&1; then
  python -m pip install 'constriction==0.5.0'
fi
mkdir -p "$run_dir"
python scripts/build_standalone.py
python scripts/build_standalone.py --check
python scripts/setup_qwen_matched_baselines.py --install-nvcomp
cp runs/_matched_baselines/setup.json "$run_dir/baseline_setup.json"

status=0
python -m unittest tests.test_transfer_t1.CudaTests tests.test_qwen_matched_gpu.CudaTests -v \
  2>&1 | tee "$run_dir/cuda_tests.log" || status=$?
if (( status == 0 )); then
  python -u scripts/bench_qwen_matched_gpu.py --output "$run_dir" --preset "$preset" "${extra[@]}" || status=$?
else
  printf 'CUDA correctness failed; no compression benchmark will run.\n' >&2
fi
if [[ -f "$run_dir/summary.md" ]]; then
  cat "$run_dir/summary.md"
  report_dir="benchmark_results/$(basename "$run_dir")_report_$(date +%Y%m%d_%H%M%S)"
  python scripts/package_qwen_matched_gpu.py "$run_dir" --destination "$report_dir"
  if (( publish )); then
    git add qwen_kv_gpu_bench.py
    git add -f "$report_dir" "$report_dir.zip"
    git commit --only -m "Add matched Qwen A4500 comparison (exit $status)" -- qwen_kv_gpu_bench.py "$report_dir" "$report_dir.zip"
    git push origin HEAD:experiment/qwen_matched_gpu
  fi
  printf '\nResume captures and calibration with:\n'
  printf 'bash scripts/run_qwen_matched_a4500.sh --resume %q' "$run_dir"
  if [[ "$preset" == full ]]; then printf ' --full'; fi
  printf '\n'
else
  printf 'Preserved test output: %s/cuda_tests.log\n' "$run_dir" >&2
fi
exit "$status"
