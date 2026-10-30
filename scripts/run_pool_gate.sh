#!/usr/bin/env bash
# The P gate (scripts/bench_pool_gate.py) on a run folder of scripts/run_a5000.sh, then its small results pushed to
# this branch as benchmark_results/<model>_poolgate_<gpu>_<date>. On the NUC:
#
#   ALLOW_NO_SLURM=1 KV_HOME=$HOME/kv_a5000 MODEL=llama32_3b bash scripts/run_pool_gate.sh
#
# Uses the run's captures (MODEL's folder under $KV_HOME/runs, made by run_a5000.sh: calcap, datasets, context) and
# the environment of scripts/a5000_setup.sh. Options: STAGES (default attn,e2e,install,report), EXTRA (more
# arguments for bench_pool_gate.py, e.g. EXTRA="--rerun attn,e2e"), FLASHINFER=1 (also FlashInfer, exactkv's own
# dense baseline), NO_PUSH=1. Finished parts are kept: a rerun continues.
set -Eeuo pipefail
trap 'echo "POOL GATE STOPPED at line $LINENO (exit $?)" >&2' ERR
BRANCH=llama32-3b-lossy-lossless-compare
cd "$(dirname "$0")/.."
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp
MODEL=${MODEL:-llama32_3b}
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (or set ALLOW_NO_SLURM=1 on the NUC)"; exit 1
fi
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "STOP: not on $BRANCH"; exit 1; }
git fetch -q origin "$BRANCH"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || {
    echo "STOP: not GitHub's $BRANCH; run: git pull --ff-only origin $BRANCH"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
# shellcheck disable=SC1091
source "$KV_HOME/venv/bin/activate"
TAG=$(echo "${MODEL#/}" | tr '/.' '__' | tr 'A-Z' 'a-z')
RUN=${RUN:-$KV_RUNS/$TAG}
if [ ! -s "$RUN/captures/meta.json" ] && [ -s "$KV_RUNS/smoke_$TAG/captures/meta.json" ]; then
    # before the full run: the smoke test's captures (calibration plus one 8K GSM8K window); 32K is skipped
    echo "NOTE: no full run in $RUN yet; using the smoke test's captures in $KV_RUNS/smoke_$TAG"
    RUN=$KV_RUNS/smoke_$TAG
fi
[ -s "$RUN/captures/meta.json" ] || { echo "STOP: no captures in $RUN (run scripts/run_a5000.sh for $MODEL first)"; exit 1; }
OUT=$RUN/pool_gate
mkdir -p "$OUT"
# FLASHINFER=1: exactkv's own dense baseline (FlashInfer 0.6.18, the version its README reports), installed into
# the environment; it compiles its kernels on first use with the CUDA toolkit of a5000_setup.sh. Optional.
if [ "${FLASHINFER:-0}" = 1 ] && ! python -c "import flashinfer" 2>/dev/null; then
    python -m pip install -q "flashinfer-python==0.6.18" || echo "NOTE: FlashInfer did not install; its rows are left out"
fi
if [ -s "$KV_HOME/cuda_home" ]; then
    CUDA_HOME=$(cat "$KV_HOME/cuda_home"); export CUDA_HOME; export PATH=$CUDA_HOME/bin:$PATH
fi
STAGES=${STAGES:-attn,e2e,install,report}
echo "== P gate: $RUN, stages $STAGES, commit $(git log --oneline -1)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
set +e
# shellcheck disable=SC2086
python -u scripts/bench_pool_gate.py --run "$RUN" --output "$OUT" --stages "$STAGES" \
    --exactkv-source "$KV_HOME/sources/exactkv" --splitzip-source "$KV_HOME/sources/splitzip" \
    --eval-text "$KV_RUNS/dataset_texts/wikitext2.txt" ${EXTRA:-} 2>&1 | tee -a "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" != 0 ]; then
    echo "== failed (exit $STATUS): the report of what finished"
    python -u scripts/bench_pool_gate.py --run "$RUN" --output "$OUT" --stages report 2>&1 | tail -n 5 || true
fi
[ -s "$OUT/summary.md" ] || { echo "STOP: nothing to package (see $OUT/console.log)"; exit 1; }
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
NAME=${TAG}_poolgate_${GPU_TAG}_$(date +%Y%m%d_%H%M)$([ "$STATUS" = 0 ] || echo _partial)
DEST=benchmark_results/$NAME
mkdir -p "$DEST"
cp "$OUT/results.json" "$OUT/summary.md" "$DEST/"
cp "$OUT/console.log" "$DEST/console.log"
git add "$DEST"
git commit -q -m "Add the P gate on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1) ($NAME)"
if [ "${NO_PUSH:-0}" = 1 ]; then echo "NO_PUSH=1: committed locally"; exit "$STATUS"; fi
git fetch -q origin "$BRANCH"
git rebase -q "origin/$BRANCH" || { git rebase --abort; echo "STOP: rebase failed; committed locally"; exit 1; }
git push -q origin "HEAD:$BRANCH"
echo "PUSHED $DEST to $BRANCH"
exit "$STATUS"
