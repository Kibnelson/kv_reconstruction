#!/usr/bin/env bash
# Compare frozen encoders/decoders. Reuse the completed NUC states and captures.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
T_SPEED_VENV=${T_SPEED_VENV:-${HEAD_SPREAD_VENV:-$KV_HOME/venv}}
REFERENCE_RUN=${REFERENCE_RUN:-$KV_HOME/runs/t_head_spread_v1_20261113T151348Z}
if [ "${TRITON_INTERPRET:-0}" != 0 ]; then
    echo "STOP: unset TRITON_INTERPRET; this benchmark requires actual CUDA." >&2
    exit 2
fi
[ -x "$T_SPEED_VENV/bin/python" ] || { echo "STOP: missing pinned environment $T_SPEED_VENV" >&2; exit 2; }
source "$T_SPEED_VENV/bin/activate"
python - <<'PY'
import torch,triton
assert torch.__version__ == '2.11.0+cu128', 'Use the existing pinned T environment.'
assert triton.__version__ == '3.6.0', 'Use the existing Triton 3.6.0.'
assert torch.cuda.is_available(), 'CUDA is unavailable; no CPU fallback.'
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'Triton:',triton.__version__)
PY
t_speed_output=${T_SPEED_OUTPUT:-$KV_HOME/runs/t_frozen_speed_v1_$(date -u +%Y%m%dT%H%M%SZ)}
args=(--reference-run "$REFERENCE_RUN" --output "$t_speed_output" --device cuda)
if [ "${SMOKE:-0}" = 1 ]; then
    args+=(--cases ds_wikitext2_r0 --rounds 1 --repeats 3 --warmup 1)
fi
if [ -n "${CAPTURES:-}" ]; then args+=(--captures "$CAPTURES"); fi
if [ -n "${VARIANTS:-}" ]; then args+=(--variants "$VARIANTS"); fi
if [ -n "${EXECUTIONS:-}" ]; then args+=(--executions "$EXECUTIONS"); fi
set +e
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u scripts/bench_t_speed.py "${args[@]}"
t_speed_status=$?
set -e
if [ -f "$t_speed_output/summary.md" ]; then cat "$t_speed_output/summary.md"; fi
if [ -f "$t_speed_output/share.zip" ]; then echo "Reports: $t_speed_output/share.zip"; fi
echo "Comparison stays on the experiment branch. No automatic merge, commit or push."
exit "$t_speed_status"
