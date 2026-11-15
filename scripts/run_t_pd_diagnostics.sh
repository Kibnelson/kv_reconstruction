#!/usr/bin/env bash
# Own login/tmux, own cached minimal environment, one short four-GPU allocation.
set -Eeuo pipefail
umask 077
diag_stage=login_preflight
trap 'diag_rc=$?; printf "DIAG_LOGIN_STOP stage=%s line=%s exit=%s\n" "$diag_stage" "$LINENO" "$diag_rc" >&2; exit "$diag_rc"' ERR
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
diag_stage=allocation
if type module >/dev/null 2>&1; then module load slurm; fi
srun -p "${T_PD_PARTITION:-all}" -A "${T_PD_ACCOUNT:-oana2026}" -q "${T_PD_QOS:-research12h}" \
    -N 1 -w "$T_PD_DIAG_NODE" --gpus=4 -c 4 --mem=16G -t 00:45:00 \
    --chdir=/tmp --export=ALL --propagate=NONE bash -s <<'NODE'
set -Eeuo pipefail
umask 077
diag_stage=node_preflight
trap 'diag_rc=$?; printf "DIAG_NODE_STOP stage=%s line=%s exit=%s job=%s\n" "$diag_stage" "$LINENO" "$diag_rc" "${SLURM_JOB_ID:-unknown}" >&2; printf "stage=%s exit=%s job=%s\n" "$diag_stage" "$diag_rc" "${SLURM_JOB_ID:-unknown}" > "$T_PD_DIAG_OUT/node_status.txt" || true; exit "$diag_rc"' ERR
printf 'DIAG_JOB=%s\nDIAG_NODE=%s\nDIAG_ROOT=%s\n' "${SLURM_JOB_ID:-unknown}" "$(hostname -s)" "$T_PD_DIAG_ROOT"
printf 'job=%s\nnode=%s\npin=%s\nroot=%s\n' "${SLURM_JOB_ID:-unknown}" "$(hostname -s)" "$T_PD_DIAG_PIN" "$T_PD_DIAG_ROOT" > "$T_PD_DIAG_OUT/node_job.txt"
ulimit -c 0
test "$(hostname -s)" = "$T_PD_DIAG_NODE"
private_dir() {
    [[ -e "$1" || -L "$1" ]] || install -d -m 700 -- "$1"
    if [[ ! -d "$1" || -L "$1" || ! -O "$1" ]] || [ "$(readlink -f -- "$1")" != "$1" ]; then
        printf 'STOP: diagnostic directory is missing, linked or not owned by this user: %s\n' "$1" >&2
        return 2
    fi
    local mode
    mode=$(stat -c '%a' -- "$1")
    if (( (8#$mode & 0022) != 0 )); then
        printf 'STOP: diagnostic directory is writable by others: %s (mode=%s)\n' "$1" "$mode" >&2
        return 2
    fi
}
diag_stage=private_root
printf 'DIAG_STAGE=%s\n' "$diag_stage"
private_dir "$T_PD_DIAG_ROOT"
private_dir "$T_PD_DIAG_ROOT/kvpd-diagnostics"
runtime=$T_PD_DIAG_ROOT/kvpd-diagnostics/$T_PD_DIAG_TAG
test ! -e "$runtime" && test ! -L "$runtime"
mkdir -m 700 "$runtime"
mkdir -m 700 "$runtime/code" "$runtime/tmp" "$runtime/cache"
printf 'DIAG_RUNTIME=%s\n' "$runtime"
diag_stage=source_archive
printf 'DIAG_STAGE=%s\n' "$diag_stage"
git -C "$T_PD_DIAG_REPO" archive "$T_PD_DIAG_PIN" -- \
    scripts/pd_gpu_diagnostics.py scripts/pd_link_probe.py scripts/pd_env.sh | tar -x -C "$runtime/code"
diag_stage=environment_cleanup
printf 'DIAG_STAGE=%s\n' "$diag_stage"
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
diag_stage=cached_uv
printf 'DIAG_STAGE=%s\n' "$diag_stage"
if [ ! -x "$KV_UV" ]; then
    printf 'STOP: our cached uv executable is unavailable: %s\n' "$KV_UV" >&2
    false
fi
"$KV_UV" --version
export UV_CACHE_DIR=$T_PD_DIAG_ROOT/kvpd-cache/uv UV_LINK_MODE=hardlink
export UV_PYTHON_INSTALL_DIR=$T_PD_DIAG_ROOT/kvpd-cache/python UV_PYTHON_PREFERENCE=only-managed UV_NO_CONFIG=1
export PIP_CONFIG_FILE=/dev/null PYTHONNOUSERSITE=1 PYTHONFAULTHANDLER=1 TRITON_INTERPRET=0
export TMPDIR=$runtime/tmp XDG_CACHE_HOME=$runtime/cache PYTHONPYCACHEPREFIX=$runtime/cache/pycache
source "$runtime/code/scripts/pd_env.sh"
diag_stage=create_environment
printf 'DIAG_STAGE=%s\n' "$diag_stage"
"$KV_UV" venv --python "$KV_UV_PYTHON" "$runtime/venv"
py=$runtime/venv/bin/python
diag_stage=install_packages
printf 'DIAG_STAGE=%s\n' "$diag_stage"
pip_in "$py" 'torch==2.11.0+cu128' 'numpy==2.3.5' \
    --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
diag_stage=collect_environment
printf 'DIAG_STAGE=%s\n' "$diag_stage"
pip_freeze "$py" > "$T_PD_DIAG_OUT/pip_freeze.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$T_PD_DIAG_OUT/gpus.txt"
nvidia-smi topo -m > "$T_PD_DIAG_OUT/topology.txt"
diag_stage=gpu_copy_diagnostics
printf 'DIAG_STAGE=%s\n' "$diag_stage"
"$py" -u "$runtime/code/scripts/pd_gpu_diagnostics.py" --output "$T_PD_DIAG_OUT" \
    --pin "$T_PD_DIAG_PIN" --mib "$T_PD_DIAG_MIB" --pairs "$T_PD_DIAG_PAIRS"
printf 'stage=complete exit=0 job=%s\n' "$SLURM_JOB_ID" > "$T_PD_DIAG_OUT/node_status.txt"
NODE
