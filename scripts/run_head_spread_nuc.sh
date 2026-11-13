#!/usr/bin/env bash
# Read existing T captures, write a new experiment run, never merge/push results.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
EXPERIMENT_BRANCH=experiment/t-head-aware-spread-v1
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
HEAD_SPREAD_VENV=${HEAD_SPREAD_VENV:-$KV_HOME/venv}
CAPTURES=${CAPTURES:-$KV_HOME/runs/llama32_3b/captures}

if [ "${TRITON_INTERPRET:-0}" != 0 ]; then
    echo "STOP: unset TRITON_INTERPRET; this run requires actual compiled CUDA kernels." >&2
    exit 2
fi
[ -x "$HEAD_SPREAD_VENV/bin/python" ] || {
    echo "STOP: no $HEAD_SPREAD_VENV; see docs/t_head_spread_experiment.md for an isolated environment." >&2; exit 2; }
# Activation changes this shell only. No packages are installed or upgraded.
source "$HEAD_SPREAD_VENV/bin/activate"
python - <<'PY'
import torch,triton
assert torch.__version__ == '2.11.0+cu128', 'Use the pinned T environment; the experiment does not modify it.'
assert triton.__version__ == '3.6.0', 'Use Triton 3.6.0 as in the pinned T environment.'
assert torch.cuda.is_available(), 'CUDA is unavailable; no CPU fallback is permitted.'
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'Triton:',triton.__version__)
PY
python - <<'PY'
from scripts.bench_head_spread import check_base
check_base()
print('Pinned original T sources are unchanged.')
PY

head_spread_output=${HEAD_SPREAD_OUTPUT:-$KV_HOME/runs/t_head_spread_v1_$(date -u +%Y%m%dT%H%M%SZ)}
args=(--captures "$CAPTURES" --device cuda --output "$head_spread_output")
if [ "${SMOKE:-0}" = 1 ]; then
    args+=(--max-layers 4 --tokens 512 --train-tokens 512 --dev-tokens 512 --warmup 1 --repeats 3
           --test-docs ds_wikitext2_r0,ds_wikitext2_r1)
fi
if [ -n "${TEST_DOCS:-}" ]; then args+=(--test-docs "$TEST_DOCS"); fi
# The driver first runs and records real-device edge checks, including every
# BF16 pattern and original-stream parity in both prediction schedules.
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u scripts/bench_head_spread.py "${args[@]}"
cat "$head_spread_output/summary.md"
echo "Reports: $head_spread_output/share.zip"
echo "Nothing was merged, committed or pushed. Keep this experiment on $EXPERIMENT_BRANCH pending NUC review."
