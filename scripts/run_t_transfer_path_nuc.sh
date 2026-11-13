#!/usr/bin/env bash
# Compare complete checked local paths using existing frozen states and codecs.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
T_PATH_VENV=${T_PATH_VENV:-$KV_HOME/venv}
REFERENCE_RUN=${REFERENCE_RUN:-$KV_HOME/runs/t_native_baselines_v1_20261113T183344Z}
if [ "${TRITON_INTERPRET:-0}" != 0 ]; then
    echo 'STOP: unset TRITON_INTERPRET; actual CUDA is required.' >&2
    exit 2
fi
[ -x "$T_PATH_VENV/bin/python" ] || { echo "STOP: missing $T_PATH_VENV" >&2; exit 2; }
source "$T_PATH_VENV/bin/activate"
python - <<'PY'
import torch,triton
assert torch.__version__=='2.11.0+cu128', 'Use the existing pinned T environment.'
assert triton.__version__=='3.6.0', 'Use the existing Triton 3.6.0.'
assert torch.cuda.is_available(), 'CUDA unavailable; no CPU fallback.'
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,'Triton:',triton.__version__)
PY
t_path_output=${T_PATH_OUTPUT:-$KV_HOME/runs/t_checked_transfer_v1_$(date -u +%Y%m%dT%H%M%SZ)}
args=(--reference-run "$REFERENCE_RUN" --output "$t_path_output")
if [ "${SMOKE:-0}" = 1 ]; then args+=(--cases ds_wikitext2_r0 --rounds 1 --repeats 2 --warmup 1); fi
if [ -n "${METHODS:-}" ]; then args+=(--methods "$METHODS"); fi
if [ -n "${CAPTURES:-}" ]; then args+=(--captures "$CAPTURES"); fi
set +e
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u scripts/bench_t_transfer_path.py "${args[@]}"
t_path_status=$?
set -e
if [ -f "$t_path_output/summary.md" ]; then cat "$t_path_output/summary.md"; fi
echo "Reports: $t_path_output/share.zip"
echo 'Local GPU-host-GPU path measured; no real network or automatic merge/push.'
exit "$t_path_status"
