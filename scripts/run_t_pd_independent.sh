#!/usr/bin/env bash
# Run in our own SSH login/tmux, from a standalone clone of our experiment branch.
# No model, Python environment, checkout or campaign cache from pd-baselines is used.
set -Eeuo pipefail
umask 077
stage=preflight
state_dir=
trap 'code=$?; if [ -n "$state_dir" ]; then printf "stopped stage=%s exit=%s\n" "$stage" "$code" > "$state_dir/status.txt"; fi; printf "T PD STOPPED: stage=%s line=%s exit=%s; state and logs are kept\n" "$stage" "$LINENO" "$code" >&2; exit "$code"' ERR

test -z "${SLURM_JOB_ID:-}"
export USER=$(id -un)
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
if [ ! -d "$repo/.git" ] || [ -L "$repo/.git" ] || [ -s "$repo/.git/objects/info/alternates" ]; then
    echo 'STOP: use a standalone git clone for this campaign.' >&2
    exit 2
fi
test "$(git rev-parse --show-toplevel)" = "$repo"
test -z "$(git status --porcelain --untracked-files=normal)"
for key in user.name user.email; do
    if [ -z "$(git config "$key" || true)" ]; then
        printf 'STOP: configure your identity in this clone: git config %s "your value"\n' "$key" >&2
        exit 2
    fi
done
matrix=${1:-smoke}
case "$matrix" in smoke|ring1p3d|ring2p2d|direct|tier|t2eager) ;; *) echo 'STOP: unknown matrix' >&2; exit 2 ;; esac
test "$#" -le 1

# Explicit T_PD_* settings belong to this launcher. Discard inherited campaign
# hooks/paths and benchmark overrides, including those from another terminal.
for key in ${!KVPD_@}; do unset "$key"; done
unset RUNS RATES LINKS NUM_PROMPTS PROMPT_TOKENS OUTPUT_LEN TTFT_SLO_MS
unset TIER_CLIENTS TIER_CONVERSATIONS TIER_MAX_TURNS TIER_GB TIER_CHUNK GPU_BLOCKS CHECK_LENGTHS
unset MAXLEN MAXBT P_MEM D_MEM ENGINE_EAGER PREFILL_GPUS DECODE_GPUS TRANSPORT CONNECTOR STAGES
unset TIER_CODECS CODECS MODEL MODEL_ID MODEL_PATH REVISION SMOKE T2_SETS
unset RUN_TIMEOUT_S VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS SELFTEST_CODECS SELFTEST_NVCOMP
unset PRINT_RUNS KEEP_ENGINES NO_PUSH SHARE_GPU LMCACHE NIXL KEEP_CAL_CAPTURES
export TRITON_INTERPRET=0 PD_MATRIX_ROUNDS=${T_PD_ROUNDS:-1}
export KVPD_BRANCH=experiment/t-pd-matched-campaign-v1
export KVPD_ROOT=${T_PD_ROOT:-/tmp/kvpd-t-matched-$USER}
export KVPD_NODE=${T_PD_NODE:-gpu-grad-01}
export KVPD_ACCOUNT=${T_PD_ACCOUNT:-oana2026}
export KVPD_PARTITION=${T_PD_PARTITION:-all}
export KVPD_QOS=${T_PD_QOS:-research12h}
export KVPD_SPECS=1 KVPD_MATRIX=$matrix
if [ -n "${T_PD_UV:-}" ]; then
    export KVPD_UV=$T_PD_UV
elif command -v uv >/dev/null 2>&1; then
    export KVPD_UV=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
    export KVPD_UV=$HOME/.local/bin/uv
else
    echo 'STOP: install the uv CLI in your own account, or set T_PD_UV to its executable.' >&2
    exit 2
fi
test -x "$KVPD_UV"
if [ -n "${T_PD_TIME:-}" ]; then export KVPD_TIME=$T_PD_TIME; fi
[[ "$KVPD_ROOT" = /* && "$KVPD_ROOT" != /tmp/"$USER" && "$KVPD_ROOT" != / ]] || {
    echo 'STOP: T_PD_ROOT must be a separate absolute campaign directory.' >&2; exit 2;
}
[[ "$PD_MATRIX_ROUNDS" =~ ^[1-9]$ ]]
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin "refs/heads/$KVPD_BRANCH" | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_TAG="pd_${matrix}_tmatched_$(date -u +%Y%m%dT%H%M%SZ)_$$"

state_dir="${T_PD_STATE_HOME:-$HOME/.kvpd_t_matched}/campaigns/$KVPD_TAG"
mkdir -p "$state_dir"
chmod 700 "$state_dir"
state=$state_dir/state.sh
for key in KVPD_BRANCH KVPD_PIN KVPD_TAG KVPD_ROOT KVPD_NODE KVPD_ACCOUNT KVPD_PARTITION KVPD_QOS KVPD_MATRIX PD_MATRIX_ROUNDS KVPD_SPECS; do
    printf 'export %s=%q\n' "$key" "${!key}" >> "$state"
done
printf 'cd -- %q\n' "$repo" >> "$state"
exec > >(tee -a "$state_dir/login.log") 2>&1
printf 'PIN=%s\nTAG=%s\nROOT=%s\nSTATE=%s\nLOG=%s\n' \
    "$KVPD_PIN" "$KVPD_TAG" "$KVPD_ROOT" "$state" "$state_dir/login.log"
for stage in create run cleanup verify-cleanup review publish; do
    printf 'running stage=%s\n' "$stage" > "$state_dir/status.txt"
    bash scripts/pd_card.sh "$stage" "$KVPD_PIN" "$KVPD_TAG"
done
printf 'closeout-complete\n' > "$state_dir/status.txt"
echo 'Campaign closeout complete. Review matched_comparison.json for GPU results and coverage.'
