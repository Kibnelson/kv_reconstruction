#!/usr/bin/env bash
# One-time (idempotent) setup of the vLLM PD runs (docs/pd_protocol.md), after scripts/a5000_setup.sh, INSIDE the
# allocation (A5000: the 4-GPU one; NUC: ALLOW_NO_SLURM=1). Adds under KV_HOME:
#
#   $KV_HOME/venv_vllm         vLLM 0.32.0 (its own torch 2.13 + CUDA 13; the codec environment keeps its pins)
#   $KV_HOME/sources/vllm      the vLLM source at v0.32.0 (benchmarks/multi_turn)
#   $RUN/captures              the WikiText-2 train calibration KV (bench_llama_compare.py --stages calcap)
#   $RUN/pd_codecs.pt          the transfer codecs' constants (scripts/pd_export_codecs.py, codec environment)
#   $RUN/pd_selftest.json      every codec round-tripped through the paced transport in the vLLM environment
#
# Options: KV_HOME, KV_RUNS, MODEL (llama32_3b), PYTHON (python3.12).
set -Eeuo pipefail
trap 'echo "PD SETUP STOPPED at line $LINENO (exit $?); a rerun continues" >&2' ERR
cd "$(dirname "$0")/.."
REPO=$PWD
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp
mkdir -p "$TMPDIR" "$KV_HOME/sources"
MODEL=${MODEL:-llama32_3b}
case "$MODEL" in
    llama32_3b) MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-3B}; REVISION=${REVISION:-d4446454d87d51aa42e1fb174f25acc5f8762331} ;;
    *) MODEL_ID=${MODEL_ID:?set MODEL_ID (and REVISION) for $MODEL}; REVISION=${REVISION:-} ;;
esac
TAG=$(echo "${MODEL#/}" | tr '/.' '__' | tr 'A-Z' 'a-z')
RUN=$KV_RUNS/$TAG
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (or set ALLOW_NO_SLURM=1 on the NUC)"; exit 1
fi
[ -x "$KV_HOME/venv/bin/python" ] || { echo "STOP: no $KV_HOME/venv: run scripts/a5000_setup.sh first"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)
if [ "$DRIVER" -lt 580 ]; then
    echo "STOP: driver $DRIVER; vLLM 0.32.0's wheel runs on CUDA 13 (driver 580 or newer)"; exit 1
fi

echo "== vLLM environment $KV_HOME/venv_vllm"
PY=${PYTHON:-$(command -v python3.12 || command -v python3)}
[ -x "$KV_HOME/venv_vllm/bin/python" ] || "$PY" -m venv "$KV_HOME/venv_vllm"
VPY=$KV_HOME/venv_vllm/bin/python
if ! "$VPY" -c "import vllm, sys; sys.exit(0 if vllm.__version__ == '0.32.0' else 1)" 2>/dev/null; then
    "$VPY" -m pip install -q --upgrade pip
    "$VPY" -m pip install -q "vllm==0.32.0" "pandas>=2.0" "xlsxwriter>=3.2.1" "aiohttp>=3.10"
fi
"$VPY" - <<'EOF'
import torch, triton, vllm, transformers
print(f"   vllm {vllm.__version__}, torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, "
      f"transformers {transformers.__version__}")
assert torch.cuda.is_available(), "the vLLM environment's torch sees no GPU"
EOF

echo "== vLLM source (benchmarks/multi_turn)"
if [ ! -d "$KV_HOME/sources/vllm/.git" ]; then
    git clone -q --depth 1 --branch v0.32.0 https://github.com/vllm-project/vllm.git "$KV_HOME/sources/vllm"
fi
for s in splitzip adacode; do
    [ -d "$KV_HOME/sources/$s" ] || echo "NOTE: no $KV_HOME/sources/$s (scripts/a5000_setup.sh clones it); $s is left out"
done

echo "== calibration captures in $RUN"
TEXTS=$KV_RUNS/dataset_texts
if [ ! -s "$TEXTS/manifest.json" ]; then
    "$KV_HOME/venv/bin/python" scripts/fetch_eval_texts.py --output "$TEXTS"
fi
if [ ! -s "$RUN/captures/wtrain0_K_bits.npy" ] || [ ! -s "$RUN/captures/meta.json" ]; then
    CUDA_HOME_ARG=()
    [ -s "$KV_HOME/cuda_home" ] && CUDA_HOME_ARG=(--cuda-home "$(cat "$KV_HOME/cuda_home")")
    "$KV_HOME/venv/bin/python" -u scripts/bench_llama_compare.py --device cuda --model "$MODEL_ID" \
        ${REVISION:+--revision "$REVISION"} --output "$RUN" --train-set wikitext \
        --train-text "$TEXTS/wikitext2_train.txt" --eval-text "$TEXTS/wikitext2.txt" --stages calcap "${CUDA_HOME_ARG[@]}"
fi

echo "== codec constants $RUN/pd_codecs.pt (codec environment)"
if [ ! -s "$RUN/pd_codecs.pt" ]; then
    SZ=()
    [ -d "$KV_HOME/sources/splitzip" ] && SZ=(--splitzip-source "$KV_HOME/sources/splitzip")
    "$KV_HOME/venv/bin/python" -u scripts/pd_export_codecs.py --run "$RUN" --output "$RUN/pd_codecs.pt" "${SZ[@]}"
fi

echo "== self-test in the vLLM environment (every codec through the paced transport)"
DOC=ds_wikitext2_r0
[ -s "$RUN/captures/${DOC}_K_bits.npy" ] || DOC=wtrain0
PYTHONPATH=$REPO "$VPY" -m kv_compare.pd.selftest --codecs "$RUN/pd_codecs.pt" --captures "$RUN/captures" \
    --doc "$DOC" --tokens 8192,3000 --splitzip-source "$KV_HOME/sources/splitzip" \
    --adacode-source "$KV_HOME/sources/adacode" --output "$RUN/pd_selftest.json"
echo "PD SETUP DONE"
