#!/usr/bin/env bash
# Own login/tmux, own cached minimal environment, one short four-GPU allocation.
set -Eeuo pipefail
umask 077
test -z "${SLURM_JOB_ID:-}"
export USER=$(id -un)
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
test -d "$repo/.git" && test ! -L "$repo/.git"
test ! -s "$repo/.git/objects/info/alternates"
test -z "$(git -C "$repo" status --porcelain --untracked-files=normal)"
for task_key in ${!KVPD_@}; do unset "$task_key"; done
export T_PD_DIAG_PIN=$(git -C "$repo" rev-parse HEAD)
test "$(git -C "$repo" ls-remote origin refs/heads/experiment/t-pd-matched-campaign-v1 | awk '{print $1}')" = "$T_PD_DIAG_PIN"
export T_PD_DIAG_REPO=$repo T_PD_DIAG_NODE=${T_PD_NODE:-gpu-grad-01}
export T_PD_DIAG_ROOT=${T_PD_ROOT:-/tmp/$USER/kvpd-t-matched-$USER}
export T_PD_DIAG_MIB=${T_PD_DIAG_MIB:-64}
export T_PD_DIAG_PAIRS=${T_PD_DIAG_PAIRS:-0-1,1-0,1-2,2-1}
[[ "$T_PD_DIAG_MIB" =~ ^[1-9][0-9]*$ ]] && (( T_PD_DIAG_MIB <= 896 ))
[[ "$T_PD_DIAG_ROOT" = /* && "$T_PD_DIAG_ROOT" != / && "$T_PD_DIAG_ROOT" != /tmp/"$USER" ]]
export T_PD_DIAG_TAG="links_$(date -u +%Y%m%dT%H%M%SZ)_$$"
export T_PD_DIAG_OUT="${T_PD_STATE_HOME:-$HOME/.kvpd_t_matched}/diagnostics/$T_PD_DIAG_TAG"
mkdir -m 700 -p "$T_PD_DIAG_OUT"
for task_key in T_PD_DIAG_PIN T_PD_DIAG_ROOT T_PD_DIAG_NODE T_PD_DIAG_MIB T_PD_DIAG_PAIRS; do
    printf 'export %s=%q\n' "$task_key" "${!task_key}" >> "$T_PD_DIAG_OUT/state.sh"
done
exec > >(tee -a "$T_PD_DIAG_OUT/login.log") 2>&1
printf 'PIN=%s\nDIAG_TAG=%s\nOUTPUT=%s\n' "$T_PD_DIAG_PIN" "$T_PD_DIAG_TAG" "$T_PD_DIAG_OUT"
if type module >/dev/null 2>&1; then module load slurm; fi
srun -p "${T_PD_PARTITION:-all}" -A "${T_PD_ACCOUNT:-oana2026}" -q "${T_PD_QOS:-research12h}" \
    -N 1 -w "$T_PD_DIAG_NODE" --gpus=4 -c 4 --mem=16G -t 00:45:00 \
    --chdir=/tmp --export=ALL --propagate=NONE bash -s <<'NODE'
set -Eeuo pipefail
umask 077
ulimit -c 0
test "$(hostname -s)" = "$T_PD_DIAG_NODE"
private_dir() {
    [[ -e "$1" || -L "$1" ]] || install -d -m 700 -- "$1"
    [[ -d "$1" && ! -L "$1" && -O "$1" ]] && test "$(readlink -f -- "$1")" = "$1"
    (( (8#$(stat -c '%a' -- "$1") & 0022) == 0 ))
}
private_dir "$T_PD_DIAG_ROOT"
private_dir "$T_PD_DIAG_ROOT/kvpd-diagnostics"
runtime=$T_PD_DIAG_ROOT/kvpd-diagnostics/$T_PD_DIAG_TAG
test ! -e "$runtime" && test ! -L "$runtime"
mkdir -m 700 "$runtime"
mkdir -m 700 "$runtime/code" "$runtime/tmp" "$runtime/cache"
git -C "$T_PD_DIAG_REPO" archive "$T_PD_DIAG_PIN" -- \
    scripts/pd_gpu_diagnostics.py scripts/pd_link_probe.py scripts/pd_env.sh | tar -x -C "$runtime/code"
for task_key in ${!UV_@} ${!PIP_@}; do unset "$task_key"; done
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE PYTHONSTARTUP VIRTUAL_ENV CONDA_PREFIX CUDA_HOME CUDA_PATH
for task_key in PATH LD_LIBRARY_PATH; do
    IFS=: read -ra parts <<< "${!task_key:-}"
    clean=()
    for part in "${parts[@]}"; do
        case "$part" in /usr/local/pkgs/cuda*|/usr/local/cuda*) ;; *) clean+=("$part") ;; esac
    done
    printf -v "$task_key" '%s' "$(IFS=:; echo "${clean[*]}")"
    export "$task_key"
done
export KV_UV=$T_PD_DIAG_ROOT/kvpd-cache/bin/uv KV_UV_PYTHON=3.12.14
test -x "$KV_UV"
export UV_CACHE_DIR=$T_PD_DIAG_ROOT/kvpd-cache/uv UV_LINK_MODE=hardlink
export UV_PYTHON_INSTALL_DIR=$T_PD_DIAG_ROOT/kvpd-cache/python UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1
export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1 TRITON_INTERPRET=0
export TMPDIR=$runtime/tmp XDG_CACHE_HOME=$runtime/cache PYTHONPYCACHEPREFIX=$runtime/cache/pycache
source "$runtime/code/scripts/pd_env.sh"
venv_create "$runtime/venv"
py=$runtime/venv/bin/python
pip_in "$py" -q 'torch==2.11.0+cu128' 'numpy==2.3.5' \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
pip_freeze "$py" > "$T_PD_DIAG_OUT/pip_freeze.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$T_PD_DIAG_OUT/gpus.txt"
nvidia-smi topo -m > "$T_PD_DIAG_OUT/topology.txt"
printf 'DIAG_JOB=%s\nDIAG_RUNTIME=%s\n' "$SLURM_JOB_ID" "$runtime"
"$py" -u "$runtime/code/scripts/pd_gpu_diagnostics.py" --output "$T_PD_DIAG_OUT" \
    --pin "$T_PD_DIAG_PIN" --mib "$T_PD_DIAG_MIB" --pairs "$T_PD_DIAG_PAIRS"
NODE
