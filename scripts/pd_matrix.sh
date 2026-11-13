#!/usr/bin/env bash
# Every baseline in real vLLM PD (docs/pd_baselines.md): a list of scripts/run_pd_vllm.sh runs, one after another in
# the same allocation (engines started and stopped per run), then one combined report
# (scripts/pd_matrix_report.py), committed and pushed once at the end.
#
#   MATRIX=smoke bash scripts/pd_matrix.sh        (inside the 4-GPU allocation; scripts/pd_matrix.sbatch runs it)
#
# MATRIX picks a list (each fits one 12-hour allocation; times are estimates until the smoke has measured them):
#   smoke       every layout, path and connector once, with few prompts (about 2 hours with model loads)
#   ring1p3d    1 prefill + 3 decode through host memory at 3/10/25/100 Gb/s: the in-process baselines and ours,
#               then nvCOMP on its own engines (a native crash costs that run only)
#   ring2p2d    the same with 2 prefill + 2 decode
#   direct      GPU to GPU (the p2p transport, unpaced) for 1+3 and 2+2, every codec; vLLM's NixlConnector for both
#   tier        4 colocated engines with the CPU tier: every codec, nvCOMP apart, and LMCache's CPU offload
#   t2eager     T and T2 in eager engines (T2's layer-0 recompute is bit-exact in eager mode), host ring and direct
# or RUNS="VAR=value VAR=value; VAR=value ...": one run per ';' (the variables of scripts/run_pd_vllm.sh).
# Extra variables in the environment (SMOKE, RATES, NUM_PROMPTS, PREFILL_GPUS, ...) apply to every run.
set -Eeuo pipefail
trap 'echo "PD MATRIX STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "$(dirname "$0")/.."
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch -q origin "$BRANCH"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || {
    echo "STOP: not GitHub's $BRANCH; run: git pull --ff-only origin $BRANCH"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
MAIN=raw,splitzip,exans,adacode,exactkv,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-dense,t2-rank256
NV=raw,nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp
TIERMAIN=raw,splitzip,exans,adacode,exactkv,ibp,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-rank256
T2SET=raw,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-dense,t2-rank256
MATRIX=${MATRIX:-smoke}
if [ -z "${RUNS:-}" ]; then
    case "$MATRIX" in
        smoke) RUNS="SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,splitzip,exans,adacode,exactkv,ours-hbh,t2-hbh;
                     SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=$NV;
                     SMOKE=1 LAYOUT=2p2d TRANSPORT=p2p CODECS=raw,ours-hbh,t2-hbh;
                     SMOKE=1 LAYOUT=2p2d CONNECTOR=nixl;
                     SMOKE=1 LAYOUT=colo4 TIER_CODECS=raw,exans,ibp,ours-hbh,t2-hbh;
                     SMOKE=1 LAYOUT=colo4 CONNECTOR=lmcache;
                     SMOKE=1 LAYOUT=1p3d TRANSPORT=ring ENGINE_EAGER=1 CODECS=raw,t2-hbh,t2-rank256" ;;
        ring1p3d) RUNS="LAYOUT=1p3d TRANSPORT=ring CODECS=$MAIN; LAYOUT=1p3d TRANSPORT=ring CODECS=$NV" ;;
        ring2p2d) RUNS="LAYOUT=2p2d TRANSPORT=ring CODECS=$MAIN; LAYOUT=2p2d TRANSPORT=ring CODECS=$NV" ;;
        direct) RUNS="LAYOUT=1p3d TRANSPORT=p2p CODECS=$MAIN; LAYOUT=1p3d TRANSPORT=p2p CODECS=$NV;
                      LAYOUT=2p2d TRANSPORT=p2p CODECS=$MAIN; LAYOUT=2p2d TRANSPORT=p2p CODECS=$NV;
                      LAYOUT=1p3d CONNECTOR=nixl; LAYOUT=2p2d CONNECTOR=nixl" ;;
        tier) RUNS="LAYOUT=colo4 TIER_CODECS=$TIERMAIN; LAYOUT=colo4 TIER_CODECS=$NV; LAYOUT=colo4 CONNECTOR=lmcache" ;;
        t2eager) RUNS="LAYOUT=1p3d TRANSPORT=ring ENGINE_EAGER=1 CODECS=$T2SET LINKS=10,100;
                       LAYOUT=1p3d TRANSPORT=p2p ENGINE_EAGER=1 CODECS=$T2SET;
                       LAYOUT=colo4 ENGINE_EAGER=1 TIER_CODECS=raw,ours-hbh,ours-rank256,t2-hbh,t2-rank256" ;;
        *) echo "STOP: MATRIX is smoke, ring1p3d, ring2p2d, direct, tier or t2eager (or set RUNS)"; exit 1 ;;
    esac
fi
START=$(git rev-parse HEAD)
STAMP=$(date +%Y%m%d_%H%M)
IFS=';' read -r -a LIST <<< "$(echo "$RUNS" | tr '\n\t' '  ')"   # read stops at a newline: one line first
n=0 failed=()
for run in "${LIST[@]}"; do
    run=$(echo "$run" | tr -s ' \n\t' ' ' | sed 's/^ //; s/ $//')
    [ -n "$run" ] || continue
    n=$((n + 1))
    echo
    echo "==== run $n of ${#LIST[@]}: $run"
    # shellcheck disable=SC2086
    if ! env $run PD_MATRIX=1 NO_PUSH=1 bash scripts/run_pd_vllm.sh; then
        failed+=("$run")
        echo "==== run $n did not finish (its results, if any, are committed as _partial); going on"
    fi
done
NEW=$(git diff --name-only "$START" HEAD -- benchmark_results | cut -d/ -f2 | sort -u | paste -sd, -)
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
DEST=benchmark_results/pd_matrix_${MATRIX}_${GPU_TAG}_${STAMP}
mkdir -p "$DEST"
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
"$KV_HOME/venv_vllm/bin/python" scripts/pd_matrix_report.py --runs "$NEW" --root benchmark_results \
    --title "PD matrix $MATRIX on $(hostname -s)" --failed "$(printf '%s;' "${failed[@]}")" --output "$DEST/summary.md" || \
    echo "WARNING: the combined report failed; every run's own summary.md is in its folder"
git add "$DEST" 2>/dev/null || true
git commit -q -m "PD matrix $MATRIX: $n runs (${#failed[@]} not finished)" || true
if [ "${NO_PUSH:-0}" = 1 ]; then echo "NO_PUSH=1: committed locally"; exit 0; fi
for i in 1 2 3; do
    git fetch -q origin "$BRANCH"
    if git rebase -q "origin/$BRANCH" && git push -q origin "HEAD:$BRANCH"; then
        echo "PUSHED $DEST (and the $n runs' folders) to $BRANCH"
        exit 0
    fi
    git rebase --abort 2>/dev/null || true
    sleep 10
done
echo "STOP: push failed; everything is committed locally (git push origin HEAD:$BRANCH later)"
exit 1
