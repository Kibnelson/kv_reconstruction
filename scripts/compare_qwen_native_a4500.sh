#!/usr/bin/env bash
# Four-family comparison using the existing, hashed Qwen captures.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source_dir="${1:?Usage: bash scripts/compare_qwen_native_a4500.sh runs/ORIGINAL_RUN [--publish-results]}"
shift
publish=0
if [[ "${1:-}" == --publish-results ]]; then publish=1; shift; fi
if (( $# )); then printf 'Unexpected argument: %s\n' "$1" >&2; exit 2; fi
test -f "$source_dir/capture_manifest.json"
test -f "$source_dir/calibration/wikitext2/fit_manifest.json"
test -f "$source_dir/calibration/c4/fit_manifest.json"
if (( publish )) && [[ "$(git branch --show-current)" != experiment/qwen_matched_gpu ]]; then
  printf 'Publishing requires experiment/qwen_matched_gpu.\n' >&2; exit 2
fi
source .venv/bin/activate
unset TRITON_INTERPRET
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST=8.6
python -c 'import torch,triton; assert torch.cuda.is_available(), "Actual CUDA required"; print(torch.cuda.get_device_name(0))'
run_dir="runs/qwen_native_compare_a4500_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$run_dir"
python scripts/build_standalone.py
python scripts/build_standalone.py --check
status=0
python scripts/setup_qwen_cuda_toolchain.py 2>&1 | tee "$run_dir/toolchain_setup.log" || status=$?
if (( status )); then printf 'CUDA toolchain setup stopped; see %s/toolchain_setup.log\n' "$run_dir" >&2; exit "$status"; fi
source runs/_matched_baselines/cuda_env.sh
cp runs/_matched_baselines/cuda12_8/toolchain.json "$run_dir/toolchain.json"
python scripts/setup_qwen_matched_baselines.py --install-nvcomp 2>&1 | tee "$run_dir/base_setup.log" || status=$?
python scripts/setup_qwen_native_baselines.py --install 2>&1 | tee "$run_dir/native_setup.log" || status=$?
python - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
root=Path('runs/_matched_baselines');out=Path(sys.argv[1])
report={n:json.loads((root/n).read_text()) for n in ['setup.json','native_setup.json'] if (root/n).exists()}
(out/'baseline_setup.json').write_text(json.dumps(report,indent=2)+'\n')
PY
if (( status )); then
  printf 'Baseline setup stopped; preserved logs in %s. The comparison needs nvcc and the pinned dependencies.\n' "$run_dir" >&2
  exit "$status"
fi
python -m unittest tests.test_qwen_matched_gpu.CudaTests tests.test_qwen_native_baselines.CudaTests -v \
  2>&1 | tee "$run_dir/cuda_tests.log" || status=$?
if (( status )); then printf 'GPU preflight stopped; see %s/cuda_tests.log\n' "$run_dir" >&2; exit "$status"; fi
methods=raw_bf16,block32_A,block32_B,aqua_style32_A,splitzip_native,exans_native,aquakv_quanto2,aquakv_quanto4
python -u scripts/bench_qwen_matched_gpu.py --phase bench --captures-from "$source_dir" \
  --output "$run_dir" --methods "$methods" --request-tokens 32,128,488 --require-all || status=$?
cat "$run_dir/summary.md"
report_dir="benchmark_results/$(basename "$run_dir")_report_$(date +%Y%m%d_%H%M%S)"
python scripts/package_qwen_matched_gpu.py "$run_dir" --destination "$report_dir"
if (( publish )); then
  git add qwen_kv_gpu_bench.py
  git add -f "$report_dir" "$report_dir.zip"
  git commit -m "Add Qwen AQUA-KV Quanto, SplitZip, ExANS and exact-codec A4500 comparison (exit $status)"
  git pull --rebase origin experiment/qwen_matched_gpu
  git push origin HEAD:experiment/qwen_matched_gpu
fi
printf 'Report: %s\n' "$report_dir/summary.md"
exit "$status"
