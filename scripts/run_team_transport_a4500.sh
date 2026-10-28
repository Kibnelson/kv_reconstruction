#!/usr/bin/env bash
# Existing compact packets only: paired public byte I/O and separate profiling.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
unset TRITON_INTERPRET
export CUBLAS_WORKSPACE_CONFIG=:4096:8
branch="experiment/llama_causal_packet_compare"
test "$(git branch --show-current)" = "$branch" || { echo "Switch to $branch first."; exit 2; }
python -c 'import torch,triton; assert torch.cuda.is_available(), "Actual CUDA required"; print(torch.cuda.get_device_name(0))'
if [[ "$#" -ge 2 && "$1" == "--resume" ]]; then
  run_dir="$2"; shift 2
  test -f "$run_dir/results.json"
else
  run_dir="runs/team_transport_a4500_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$run_dir"
fi
echo "Output / resume directory: $run_dir"
if python -m unittest tests.test_team_transport.CudaTests -v 2>&1 | tee -a "$run_dir/cuda_tests.log"; then
  if python -u scripts/bench_team_transport.py --output "$run_dir" "$@" 2>&1 | tee -a "$run_dir/console.log"; then
    run_status=0
  else
    run_status=$?
  fi
else
  run_status=$?
  python - "$run_dir" "$run_status" <<'PY'
import json
from pathlib import Path
import sys
from scripts.bench_team_transport import aggregate,link_models,report
from scripts.bench_lite_baselines import write
folder=Path(sys.argv[1]);path=folder/'results.json'
result=json.loads(path.read_text()) if path.exists() else dict(format='team_transport_v1',gpu_measured=False,jobs={},profiles={},rows=[])
result.update(status='failed',error='Actual-CUDA transport correctness gate failed; benchmark stages were not started',cuda_test_exit=int(sys.argv[2]))
result['aggregates']=aggregate(result['rows']);result['link_models']=link_models(result['aggregates'])
write(path,result);report(result,folder)
PY
fi
report_dir="benchmark_results/$(basename "$run_dir")_report_$(date +%Y%m%d_%H%M%S)"
python scripts/package_team_transport.py "$run_dir" --destination "$report_dir"
git add -f -- "$report_dir"
git commit --only -m "Add frozen compact packet transport A4500 profile (exit $run_status)" -- "$report_dir"
git -c rebase.autoStash=true pull --rebase origin "$branch"
git push origin "HEAD:$branch"
echo "Published: $report_dir/summary.md"
echo "Full profiler traces remain in $run_dir/profiles/"
exit "$run_status"
