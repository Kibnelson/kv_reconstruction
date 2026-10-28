#!/usr/bin/env bash
# Run benchmark stages on this branch and push the packaged results to this branch - nowhere else.
#
#   cd ~/kv_llama32 && git checkout llama32-3b-lossy-lossless-compare \
#       && git pull --ff-only origin llama32-3b-lossy-lossless-compare && bash scripts/run_on_nuc.sh
#
# Stops before running anything if this folder is not on the branch, is not up to date with GitHub, has
# uncommitted changes, or the Python environment is not the one of the earlier runs (torch 2.11 / CUDA 12.8).
# Options (environment): STAGES (default gpu_speed,report), RUN (the run folder with the captures),
# VENV (default ~/kv_reconstruction/.venv), EXTRA (more arguments for bench_llama_compare.py).
set -euo pipefail
BRANCH=llama32-3b-lossy-lossless-compare
cd "$(dirname "$0")/.."

cur=$(git rev-parse --abbrev-ref HEAD)
if [ "$cur" != "$BRANCH" ]; then
    echo "STOP: this folder is on branch '$cur', not '$BRANCH'."
    echo "      Use the separate folder:  cd ~/kv_llama32 && git checkout $BRANCH"
    exit 1
fi
git fetch -q origin "$BRANCH"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "STOP: $(git log --oneline -1) is not GitHub's $BRANCH ($(git log --oneline -1 "origin/$BRANCH"))."
    echo "      Run:  git pull --ff-only origin $BRANCH   and start this script again."
    exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "STOP: uncommitted changes in tracked files (git status)."
    exit 1
fi

VENV=${VENV:-$HOME/kv_reconstruction/.venv}
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
python - <<'EOF'
import sys, torch
v = torch.__version__
print(f"python {sys.version.split()[0]}, torch {v}, CUDA {torch.version.cuda}")
if not v.startswith("2.11"):
    sys.exit(f"STOP: torch {v}; the earlier runs used 2.11.0+cu128 (source ~/kv_reconstruction/.venv/bin/activate)")
EOF

RUN=${RUN:-$HOME/kv_reconstruction/runs/llama32_3b_compare}
STAGES=${STAGES:-gpu_speed,report}
echo "running on branch $cur, commit $(git log --oneline -1); stages $STAGES; output $RUN"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# CHECKS=1: compare our AQUA-KV and KIVI re-implementations with the authors' own functions first (the report
# includes the results). AQUA-KV runs in the environment the earlier official run built (torch 2.5.1 with
# fast-hadamard-transform); KIVI's functions need only torch and Triton (this environment).
if [ "${CHECKS:-0}" = 1 ]; then
    OFF=$HOME/kv_reconstruction/runs/_llama_official
    AQUA_PY=$OFF/envs/aqua/bin/python
    AQUA_SRC=$OFF/sources/aqua
    KIVI_SRC=$OFF/sources/kivi
    if [ ! -d "$KIVI_SRC/.git" ]; then
        git clone -q https://github.com/jy-yuan/KIVI.git "$KIVI_SRC" && git -C "$KIVI_SRC" checkout -q 876b4d2
    fi
    if [ -x "$AQUA_PY" ] && [ -d "$AQUA_SRC" ]; then
        "$AQUA_PY" scripts/check_baselines_official.py aqua --source "$AQUA_SRC" --output "$RUN/check_aqua.json" \
            2>&1 | tail -n 40 | tee -a "$RUN/console.log" || echo "AQUA-KV check failed (see above)"
    else
        echo "AQUA-KV check skipped: no $AQUA_PY or $AQUA_SRC"
    fi
    python scripts/check_baselines_official.py kivi --source "$KIVI_SRC" --output "$RUN/check_kivi.json" \
        2>&1 | tail -n 30 | tee -a "$RUN/console.log" || echo "KIVI check failed (see above)"
fi
python -u scripts/bench_llama_compare.py --device cuda --output "$RUN" --stages "$STAGES" ${EXTRA:-} 2>&1 \
    | tee -a "$RUN/console.log"

NAME=llama32_3b_compare_a4500_$(date +%Y%m%d_%H%M)
python scripts/package_llama_compare.py "$RUN" --destination "benchmark_results/$NAME"
git add "benchmark_results/$NAME"
git commit -m "Add Llama-3.2-3B results ($NAME, stages $STAGES)"
git push origin "HEAD:$BRANCH"
echo "pushed benchmark_results/$NAME to $BRANCH"
