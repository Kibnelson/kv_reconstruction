#!/usr/bin/env bash
# Read-only frozen captures/state; builds third-party code only in a new run.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
T_BASELINES_VENV=${T_BASELINES_VENV:-$KV_HOME/venv}
REFERENCE_RUN=${REFERENCE_RUN:-$KV_HOME/runs/t_head_spread_v1_20261113T151348Z}
if [ "${TRITON_INTERPRET:-0}" != 0 ]; then
    echo 'STOP: unset TRITON_INTERPRET; compiled CUDA is required.' >&2
    exit 2
fi
[ -x "$T_BASELINES_VENV/bin/python" ] || { echo "STOP: missing $T_BASELINES_VENV" >&2; exit 2; }
source "$T_BASELINES_VENV/bin/activate"
python - <<'PY'
import torch,triton
assert torch.__version__ == '2.11.0+cu128', 'Use the existing pinned T environment.'
assert triton.__version__ == '3.6.0', 'Use the existing Triton 3.6.0.'
assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback.'
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'Triton:',triton.__version__)
PY
if [ -z "${CUDA_HOME:-}" ] && [ -f "$KV_HOME/cuda_home" ]; then
    CUDA_HOME=$(cat "$KV_HOME/cuda_home")
fi
t_compare_output=${T_BASELINES_OUTPUT:-$KV_HOME/runs/t_native_baselines_v1_$(date -u +%Y%m%dT%H%M%SZ)}
args=(--reference-run "$REFERENCE_RUN" --output "$t_compare_output"
      --splitzip-source "${SPLITZIP_SOURCE:-$KV_HOME/sources/splitzip}"
      --exans-source "${EXANS_SOURCE:-$KV_HOME/sources/exans}"
      --cuda-home "${CUDA_HOME:-}")
if [ "${SMOKE:-0}" = 1 ]; then
    args+=(--cases ds_wikitext2_r0 --rounds 1 --repeats 3 --warmup 1)
fi
if [ -n "${CAPTURES:-}" ]; then args+=(--captures "$CAPTURES"); fi
if [ -n "${METHODS:-}" ]; then args+=(--methods "$METHODS"); fi
set +e
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u scripts/bench_t_baselines.py "${args[@]}"
t_compare_status=$?
set -e
if [ -f "$t_compare_output/summary.md" ]; then cat "$t_compare_output/summary.md"; fi
echo "Reports: $t_compare_output/share.zip"
echo 'Comparison stays on the experiment branch. No automatic merge or push.'
exit "$t_compare_status"
