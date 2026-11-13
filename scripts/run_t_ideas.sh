#!/usr/bin/env bash
# Tests of five ideas for T (docs/t_ideas.md) on the captures of a finished scripts/run_a5000.sh run folder, on
# this machine's GPU. Packages the small results into benchmark_results/ and pushes them to this branch only.
#
#   MODEL=llama32_3b bash scripts/run_t_ideas.sh            (ALLOW_NO_SLURM=1 on the NUC)
#
# Options (environment): KV_HOME (default ~/kv_a5000), KV_RUNS (default $KV_HOME/runs), MODEL (the run folder's
# tag, default llama32_3b), STAGES (default capture,adapt,layer0,anchor,classes,gpu,report), EXTRA (more
# arguments for scripts/t_ideas.py), NO_PUSH=1 (commit locally only), SMOKE=1 (4 layers, one WikiText-2 window, a
# separate output folder, nothing committed: checks the whole path in a few minutes).
set -Eeuo pipefail
trap 'echo "RUN STOPPED at line $LINENO (exit $?)" >&2' ERR
BRANCH=t-ideas
cd "$(dirname "$0")/.."
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp
mkdir -p "$TMPDIR"
MODEL=${MODEL:-llama32_3b}
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (SLURM_JOB_ID is empty). Request the GPU first, or ALLOW_NO_SLURM=1 on the NUC."
    exit 1
fi
cur=$(git rev-parse --abbrev-ref HEAD)
[ "$cur" = "$BRANCH" ] || { echo "STOP: this checkout is on '$cur', not '$BRANCH'"; exit 1; }
git fetch -q origin "$BRANCH"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "STOP: $(git log --oneline -1) is not GitHub's $BRANCH: git pull --ff-only origin $BRANCH, then start again."
    exit 1
fi
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
[ -x "$KV_HOME/venv/bin/python" ] || { echo "STOP: no $KV_HOME/venv: run scripts/a5000_setup.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$KV_HOME/venv/bin/activate"
WANT_TORCH=$(sed -n 's/^torch==//p' requirements-cuda.txt)
HAVE_TORCH=$(python -c "import torch; print(torch.__version__)")
[ "$HAVE_TORCH" = "$WANT_TORCH" ] || {
    echo "STOP: torch $HAVE_TORCH in $KV_HOME/venv, not $WANT_TORCH: KV_HOME=$KV_HOME bash scripts/a5000_setup.sh"; exit 1; }
TAG=$(echo "$MODEL" | tr '/.' '__' | tr 'A-Z' 'a-z')
RUN=$KV_RUNS/$TAG
for f in captures/meta.json captures/wtrain0_K_bits.npy captures/wtrain5_K_bits.npy captures/wtables_K_bits.npy; do
    [ -e "$RUN/$f" ] || { echo "STOP: $RUN/$f missing: run MODEL=$MODEL bash scripts/run_a5000.sh (calcap) first"; exit 1; }
done
TEXTS=$KV_RUNS/dataset_texts
if [ ! -s "$TEXTS/manifest.json" ]; then
    echo "== dataset texts into $TEXTS"
    python scripts/fetch_eval_texts.py --output "$TEXTS"
fi
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr 'A-Z' 'a-z' | sed 's/nvidia//; s/geforce//; s/[^a-z0-9]//g')
STAMP=$(date +%Y%m%d_%H%M)
STAGES=${STAGES:-capture,adapt,layer0,anchor,classes,gpu,report}
OUT=$RUN/t_ideas
ARGS=(--run "$RUN" --output "$OUT" --texts "$TEXTS" --stages "$STAGES" --device cuda)
if [ "${SMOKE:-0}" = 1 ]; then
    OUT=$RUN/t_ideas_smoke
    rm -rf "$OUT"
    ARGS=(--run "$RUN" --output "$OUT" --texts "$TEXTS" --device cuda --stages adapt,anchor,classes,gpu,report
          --datasets wikitext2 --windows 1 --float-docs 1 --max-layers 4 --modes dense,rank256)
fi
echo "== commit $(git log --oneline -1); run folder $RUN; output $OUT; GPU $GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
# shellcheck disable=SC2086
python scripts/t_ideas.py "${ARGS[@]}" ${EXTRA:-}
[ -s "$OUT/summary.md" ] || { echo "STOP: no $OUT/summary.md"; exit 1; }
if [ "${SMOKE:-0}" = 1 ]; then
    echo "SMOKE OK: $OUT/summary.md (nothing committed)"
    exit 0
fi
DEST=benchmark_results/${TAG}_tideas_${GPU}_${STAMP}
mkdir -p "$DEST"
cp "$OUT/t_ideas.json" "$OUT/summary.md" "$DEST/"
gzip -c "$OUT/console.log" > "$DEST/console.log.gz"
python -m pip freeze > "$DEST/pip_freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$DEST/gpu.csv"
git add "$DEST"
git commit -q -m "T ideas measured: $TAG on $GPU ($STAMP)" -m "scripts/run_t_ideas.sh, stages $STAGES."
if [ "${NO_PUSH:-0}" = 1 ]; then
    echo "COMMITTED $DEST (not pushed: NO_PUSH=1)"
    exit 0
fi
for i in 1 2 3; do
    if git push -q origin "$BRANCH"; then
        echo "PUSHED $DEST"
        exit 0
    fi
    git pull -q --rebase origin "$BRANCH" || true
    sleep 5
done
echo "STOP: push failed; the results are committed locally in $DEST"
exit 1
