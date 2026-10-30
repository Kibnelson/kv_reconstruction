#!/usr/bin/env bash
# The lossless comparison for one model on one RTX A5000, run INSIDE the one-GPU Slurm allocation after
# scripts/a5000_setup.sh (docs/a5000_runbook.md). Measures, packages the small results into benchmark_results/ and
# pushes them to this branch - nowhere else.
#
#   MODEL=llama3_8b bash scripts/run_a5000.sh
#
# MODEL: llama32_3b (ours so far, the NUC's pin), llama3_8b (SplitZip's Table 1 and 4), qwen3_8b (the Qwen3
# family of SplitZip's Qwen3-32B, which does not fit 24 GB), qwen3_4b (Qwen3-4B-Instruct-2507 at the pin of the
# Phase 2 runs), any Hugging Face model id (Llama / Qwen architecture, head_dim 128), or an absolute path to a
# downloaded model folder.
# Stages (STAGES, default calcap,datasets,context,fp8pred,groupspeed,report), calibrated on WikiText-2 train (SplitZip's
# protocol; TRAIN_SET to change):
#   calcap      KV of 7 x 8,192 WikiText-2 train tokens (the calibration documents) and the model's geometry
#   datasets    SplitZip's five datasets (WikiText-2 test, HumanEval, GSM8K, MMLU, PTB), 4 x 8,192 tokens each:
#               every codec bit-exact and timed, our exact costs, SplitZip's exponent coverage (its Table 4)
#   context     WikiText-2 test at 512 ... 32,768 tokens, batch 1 and 16 (SplitZip's sweep), every codec
#   fp8pred     a whole FP8 copy of the model as the predictor (speculative KV coding's idea), exact cost
# Every codec: ours, SplitZip, ExANS, AdaCode, exactkv, nvCOMP, IBP (the authors' code), and the exponent bounds.
#   groupspeed  our head-group decoders against launch shapes, as CUDA graphs
#   report      summary.md
# plus the link probe (this GPU's host legs) first. A stage that finished is kept: rerunning continues.
# Other options (environment): KV_HOME (default ~/kv_a5000), KV_RUNS (default $KV_HOME/runs; the captures, up to
# ~60 GB for an 8B model), EXTRA (more arguments for bench_llama_compare.py), CLEAN_CAPTURES=1 (delete the run's
# captured KV after a successful push), NO_PUSH=1 (commit locally only), SMOKE=1 (a short check of the whole path
# on one dataset window, in a separate run folder; nothing is committed).
set -Eeuo pipefail
trap 'echo "RUN STOPPED at line $LINENO (exit $?)" >&2' ERR
BRANCH=llama32-3b-lossy-lossless-compare
cd "$(dirname "$0")/.."
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp
mkdir -p "$TMPDIR"
MODEL=${MODEL:-}
[ -n "$MODEL" ] || { echo "STOP: set MODEL (llama32_3b, llama3_8b, qwen3_8b, qwen3_4b or a Hugging Face id)"; exit 1; }

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (SLURM_JOB_ID is empty). Request the GPU first (runbook step 1)."
    exit 1
fi
cur=$(git rev-parse --abbrev-ref HEAD)
[ "$cur" = "$BRANCH" ] || { echo "STOP: this checkout is on '$cur', not '$BRANCH'"; exit 1; }
git fetch -q origin "$BRANCH"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "STOP: $(git log --oneline -1) is not GitHub's $BRANCH ($(git log --oneline -1 "origin/$BRANCH"))."
    echo "      Run:  git pull --ff-only origin $BRANCH   and start this script again."
    exit 1
fi
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
if [ "${SMOKE:-0}" != 1 ] && [ "${NO_PUSH:-0}" != 1 ] && ! git config user.email >/dev/null; then
    echo "STOP: git has no identity here to commit the results: git config --global user.name ...; user.email ..."
    exit 1
fi
[ -x "$KV_HOME/venv/bin/python" ] || { echo "STOP: no $KV_HOME/venv: run scripts/a5000_setup.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$KV_HOME/venv/bin/activate"
# every run on the pinned torch (requirements-cuda.txt): an install into this environment changes it (FlashInfer's
# brought torch 2.15 + CUDA 13.2 into the NUC's mid-run); scripts/a5000_setup.sh puts the pin back
WANT_TORCH=$(sed -n 's/^torch==//p' requirements-cuda.txt)
HAVE_TORCH=$(python -c "import torch; print(torch.__version__)")
[ "$HAVE_TORCH" = "$WANT_TORCH" ] || {
    echo "STOP: torch $HAVE_TORCH in $KV_HOME/venv, not $WANT_TORCH (requirements-cuda.txt); put it back:"
    echo "  KV_HOME=$KV_HOME bash scripts/a5000_setup.sh   (ALLOW_NO_SLURM=1 on the NUC)"; exit 1; }

case "$MODEL" in
    /*) [ -d "$MODEL" ] || { echo "STOP: no model folder $MODEL"; exit 1; }; IDS=$MODEL; PIN="" ;;   # a local copy
    llama32_3b) IDS="unsloth/Llama-3.2-3B"; PIN=d4446454d87d51aa42e1fb174f25acc5f8762331 ;;
    llama3_8b) IDS="meta-llama/Meta-Llama-3-8B NousResearch/Meta-Llama-3-8B unsloth/llama-3-8b"; PIN="" ;;
    qwen3_8b) IDS="Qwen/Qwen3-8B"; PIN="" ;;
    qwen3_4b) IDS="Qwen/Qwen3-4B-Instruct-2507"; PIN=cdbee75f17c01a7cc42f958dc650907174af0554 ;;
    */*) IDS=$MODEL; PIN=${REVISION:-} ;;
    *) echo "STOP: unknown MODEL '$MODEL'"; exit 1 ;;
esac
TAG=$(echo "${MODEL#/}" | tr '/.' '__' | tr 'A-Z' 'a-z')
if [ -d "$MODEL" ]; then TAG=local_$(basename "$MODEL" | tr '.' '_' | tr 'A-Z' 'a-z'); fi
RUN=$KV_RUNS/$TAG
if [ "${SMOKE:-0}" = 1 ]; then
    RUN=$KV_RUNS/smoke_$TAG
    STAGES=calcap,datasets,fp8pred,report
    EXTRA="--datasets gsm8k --dataset-docs 1 --no-dataset-exact ${EXTRA:-}"
fi
TEXTS=$KV_RUNS/dataset_texts
mkdir -p "$RUN"
echo "== job ${SLURM_JOB_ID:-none} on $(hostname); time left $(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || echo '?')"
echo "== commit $(git log --oneline -1); model $MODEL; run folder $RUN"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
nvidia-smi -q > "$RUN/nvidia_smi.txt" 2>&1 || true
python - "$RUN/slurm.json" <<'EOF'
import json, os, socket, sys
keys = sorted(k for k in os.environ if k.startswith("SLURM_") or k == "CUDA_VISIBLE_DEVICES")
json.dump({"host": socket.gethostname(), **{k: os.environ[k] for k in keys}}, open(sys.argv[1], "w"), indent=1)
EOF
python -m pip freeze > "$RUN/pip_freeze.txt"
df -h "$RUN" | tail -n 1

echo "== model: the first of [$IDS] this node can download (revision ${PIN:-latest, recorded})"
if [ -d "$MODEL" ]; then
    MODEL_ID=$MODEL REVISION=none
    printf '{"model": "%s", "revision": null, "local_folder": true}\n' "$MODEL" > "$RUN/model_resolved.json"
else
read -r MODEL_ID REVISION < <(python - "$RUN/model_resolved.json" "$PIN" $IDS <<'EOF'
import json, sys
from huggingface_hub import HfApi, hf_hub_download
out, pin, ids = sys.argv[1], sys.argv[2] or None, sys.argv[3:]
tried = []
for mid in ids:
    try:
        sha = HfApi().model_info(mid, revision=pin).sha
        hf_hub_download(mid, "config.json", revision=sha)          # a gated model without access fails here
        json.dump({"model": mid, "revision": sha, "pinned": pin, "tried": tried}, open(out, "w"), indent=1)
        print(mid, sha)
        sys.exit(0)
    except Exception as exc:
        tried.append({"model": mid, "error": repr(exc)[:300]})
        print(f"  {mid}: not available ({repr(exc)[:160]})", file=sys.stderr)
sys.exit("STOP: none of the models could be downloaded (gated: export HF_TOKEN with access, or huggingface-cli login)")
EOF
)
fi
[ -n "${MODEL_ID:-}" ] && [ -n "${REVISION:-}" ] || { echo "STOP: no model resolved"; exit 1; }
echo "   using $MODEL_ID at $REVISION"

if [ ! -s "$TEXTS/manifest.json" ] || ! python - "$TEXTS/manifest.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
sys.exit(0 if {"wikitext2", "wikitext2_train", "humaneval", "gsm8k", "mmlu", "ptb"} <= set(m) else 1)
EOF
then
    echo "== dataset texts into $TEXTS"
    python scripts/fetch_eval_texts.py --output "$TEXTS"
fi
mkdir -p "$RUN/dataset_texts"
cp "$TEXTS/manifest.json" "$RUN/dataset_texts/manifest.json"

echo "== link probe (this GPU's host legs)"
python scripts/pd_link_probe.py --output "$RUN/pd_link_probe.json" 2>&1 | tee -a "$RUN/console.log" \
    || echo "link probe failed (see above); continuing"

STAGES=${STAGES:-calcap,datasets,context,fp8pred,groupspeed,report}
TRAIN_SET=${TRAIN_SET:-wikitext}
ARGS=(--device cuda --model "$MODEL_ID" --revision "$REVISION" --output "$RUN"
      --train-set "$TRAIN_SET" --train-text "$TEXTS/wikitext2_train.txt" --eval-text "$TEXTS/wikitext2.txt"
      --dataset-texts "$TEXTS" --splitzip-source "$KV_HOME/sources/splitzip" --exans-source "$KV_HOME/sources/exans"
      --baseline-stream-tokens 8192 --baseline-sources "$KV_HOME/sources")
[ -s "$KV_HOME/cuda_home" ] && ARGS+=(--cuda-home "$(cat "$KV_HOME/cuda_home")")
echo "== bench_llama_compare.py, stages $STAGES (log: $RUN/console.log)"
set +e
for ATTEMPT in 1 2 3 4 5 6 7 8; do
    # shellcheck disable=SC2086
    python -u scripts/bench_llama_compare.py "${ARGS[@]}" --stages "$STAGES" ${EXTRA:-} 2>&1 | tee -a "$RUN/console.log"
    STATUS=${PIPESTATUS[0]}
    echo "== bench_llama_compare.py exit $STATUS (attempt $ATTEMPT)" | tee -a "$RUN/console.log"
    # a codec's native code took the process down: the marker names it; the next attempt records it and goes on
    if [ "$STATUS" != 0 ] && [ -s "$RUN/crash_marker.json" ]; then
        echo "== crashed in $(cat "$RUN/crash_marker.json"); resuming without that codec" | tee -a "$RUN/console.log"
        continue
    fi
    break
done
set -e
if [ "${SMOKE:-0}" = 1 ]; then
    echo "== smoke run finished (exit $STATUS); nothing committed. The table to check:"
    sed -n "/## Lossless on SplitZip's datasets/,/Transfer time per link/p" "$RUN/summary.md" 2>/dev/null | head -n 40
    MISSING=""
    for m in "SplitZip (authors' codec)" "ExANS (authors' codec)" "ours, per layer, head by head" \
             "AdaCode (authors' codec)" "exactkv (authors' codec)" "nvCOMP ANS"; do
        grep -F "| $m |" "$RUN/summary.md" 2>/dev/null | grep -qv "not measured" || MISSING="$MISSING [$m]"
    done
    grep -F "| IBP (authors' codec), host to GPU |" "$RUN/summary.md" 2>/dev/null | grep -qv "not measured" \
        || echo "note: IBP not measured (optional: its CUDA build, $KV_HOME/ibp_build.log)"
    if [ "$STATUS" = 0 ] && [ -z "$MISSING" ]; then echo "SMOKE OK: every codec measured"
    else echo "SMOKE: not complete (exit $STATUS; missing:${MISSING:- none}); see $RUN/run.log"; fi
    exit "$STATUS"
fi
SUFFIX=""
if [ "$STATUS" != 0 ]; then
    echo "== the run failed (exit $STATUS): writing the report of what finished, then packaging it as partial"
    SUFFIX=_partial
    # shellcheck disable=SC2086
    python -u scripts/bench_llama_compare.py "${ARGS[@]}" --stages report ${EXTRA:-} 2>&1 | tee -a "$RUN/console.log" || true
fi

# the GPU in the name: rtxa5000 on the cluster, rtxa4500 for a dress rehearsal on the NUC
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_TAG=$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | sed 's/nvidia//; s/geforce//' | tr -cd 'a-z0-9')
NAME=${TAG}_lossless_${GPU_TAG:-gpu}_$(date +%Y%m%d_%H%M)$SUFFIX
if ! python scripts/package_llama_compare.py "$RUN" --destination "benchmark_results/$NAME"; then
    echo "STOP: nothing to package yet (see $RUN/console.log and $RUN/run.log)"
    exit 1
fi
git add "benchmark_results/$NAME"
git commit -q -m "Add $MODEL_ID lossless results on $GPU_NAME ($NAME, stages $STAGES)"
if [ "${NO_PUSH:-0}" = 1 ]; then
    echo "NO_PUSH=1: committed locally as $(git log --oneline -1)"
    exit "$STATUS"
fi
git fetch -q origin "$BRANCH"
if ! git rebase -q "origin/$BRANCH"; then
    git rebase --abort
    echo "STOP: the results are committed locally but could not be put on top of origin/$BRANCH; run"
    echo "      git pull --rebase origin $BRANCH && git push origin HEAD:$BRANCH"
    exit 1
fi
if ! git push -q origin "HEAD:$BRANCH"; then
    tar -czf "$KV_RUNS/$NAME.tar.gz" -C benchmark_results "$NAME"
    echo "STOP: push refused (credentials?). The results are committed locally and saved as $KV_RUNS/$NAME.tar.gz"
    exit 1
fi
echo "PUSHED benchmark_results/$NAME to $BRANCH ($(git log --oneline -1))"
if [ "${CLEAN_CAPTURES:-0}" = 1 ] && [ "$STATUS" = 0 ]; then
    rm -rf "$RUN/captures"
    echo "CLEAN_CAPTURES=1: deleted $RUN/captures"
fi
exit "$STATUS"
