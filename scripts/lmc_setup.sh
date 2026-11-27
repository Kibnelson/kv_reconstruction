#!/usr/bin/env bash
# Setup of the LMCache lifecycle runs (docs/lmcache_lifecycle.md), INSIDE the allocation; idempotent, and all a
# MATRIX=lmctrace campaign needs (scripts/pd_campaign.py runs it in place of the codec setup). Adds under KV_HOME:
#
#   $KV_HOME/venv_lmcache      vLLM 0.32.0 and LMCache 0.5.7.post1: the release wheels, for their compiled parts
#   $KV_HOME/build/lmc_ops     LMCache's compiled modules (cuda_ops, lmcache_native, _fs, _redis: all four the wheel
#                              has, so they share their types, job 30722) built from our copy's csrc/ against
#                              this torch (scripts/lmc_build_ops.py), with the environment's own CUDA 13 compiler (its
#                              pip packages, scripts/cuda13_toolkit.sh): the wheel's ops do not load beside torch
#                              2.13, and LMCache's torch baseline crashed its server in both of job 30694's runs
#   $KV_HOME/lmc_overlay       those wheels' packages with our copies' Python over them (third_party/, the trace
#                              hooks; scripts/lmc_overlay.py checks every other file is the wheel's, byte for byte),
#                              and our build of lmcache.cuda_ops in place of the wheel's
#   $KV_RUNS/mtrag/conversations.json   IBM's MTRAG conversations (pinned commit, SHA-256 checked), and its
#                              generation tasks RAG.jsonl and reference.jsonl (the same commit)
#   the model, unsloth/Llama-3.2-3B at its pinned revision, in HF_HOME (and Qwen/Qwen3-4B-Instruct-2507 for the KV
#                              pool experiment, KVPD_LMC_PRESET=pool or LMC_SETUP_MODELS=qwen3_4b)
#   NIXL 1.5.2 in that environment (nixl-cu13 + nixl), for LMC_TRANSFER=nixl
#
# then checks that the overlay is what the engines will import (each hooked module from the overlay, with its hooks,
# and kvtrace writing a record, and LMCache's compiled ops loading: ours). Options: KV_HOME, KV_RUNS, KV_HF_HOME, KV_UV
# (scripts/pd_env.sh), PYTHON, CUDA13_HOME (the toolkit; scripts/pd_card.sh's cache), LMC_TORCH_BASELINE=1 (no build:
# the wheel's ops, or LMCache's torch baseline, as job 30694 ran).
set -Eeuo pipefail
trap 'echo "LMC SETUP STOPPED at line $LINENO (exit $?); a rerun continues" >&2' ERR
cd "$(dirname "$0")/.."
REPO=$PWD
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=${KV_HF_HOME:-$KV_HOME/hf}
export TMPDIR=${TMPDIR:-$KV_HOME/tmp} XDG_CACHE_HOME=${XDG_CACHE_HOME:-$KV_HOME/cache}
mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$KV_RUNS"
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (or set ALLOW_NO_SLURM=1)"; exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p | cut -d. -f1)
[ "$DRIVER" -ge 580 ] || { echo "STOP: driver $DRIVER; vLLM 0.32.0's wheel runs on CUDA 13 (driver 580 or newer)"; exit 1; }

echo "== our copies: only the trace hooks differ from the import"
python3 -I scripts/lmc_overlay.py check

echo "== LMCache environment $KV_HOME/venv_lmcache (the release wheels)"
[ -n "${KV_UV:-}" ] || PY=${PYTHON:-$(command -v python3.12 || command -v python3)}
# shellcheck source=scripts/pd_env.sh
source "$REPO/scripts/pd_env.sh"
venv_create "$KV_HOME/venv_lmcache"
LPY=$KV_HOME/venv_lmcache/bin/python
if ! "$LPY" -c "import vllm, lmcache, sys; sys.exit(0 if (vllm.__version__, lmcache.__version__) == ('0.32.0', '0.5.7.post1') else 1)" 2>/dev/null; then
    pip_in "$LPY" -q "vllm==0.32.0" "lmcache==0.5.7.post1" "aiohttp>=3.10" ninja
fi
"$LPY" -c "import vllm, lmcache, torch; print(f'   vllm {vllm.__version__}, lmcache {lmcache.__version__}, torch {torch.__version__} (CUDA {torch.version.cuda})')"
# NIXL, for LMC_TRANSFER=nixl (vLLM's NixlConnector): the version our other vLLM 0.32 runs use (scripts/pd_setup.sh).
# NIXL 1.5.2 is two wheels: nixl-cu13 (the library) and nixl (the module vLLM imports, which picks nixl_cuXX by
# torch's CUDA); the nixl wheel requires nixl-cu12 too, and --no-deps keeps CUDA 12's copy out
if ! "$LPY" -c "import torch; from nixl._api import nixl_agent" 2>/dev/null; then
    pip_in "$LPY" -q "nixl-cu13==1.5.2" && pip_in "$LPY" -q --no-deps "nixl==1.5.2" || true
fi
"$LPY" -c "import torch; from nixl._api import nixl_agent; print('   NIXL 1.5.2 (nixl_cu13)')" 2>&1 | tail -n 3 || \
    echo "WARNING: NIXL does not import (above): runs with LMC_TRANSFER=nixl will stop"
pip_freeze "$LPY" > "$KV_HOME/pip_freeze_lmcache.txt" || true

EXT=() NEED_OPS=()
if [ "${LMC_TORCH_BASELINE:-0}" != 1 ]; then
    echo "== LMCache's compiled modules, built from our copy (third_party/lmcache/csrc)"
    SITE=$("$LPY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    # the environment's CUDA 13 headers and libraries (its pip packages), with a compiler of their CUDA version from pip
    # into a folder of its own (job 30716: the environment's own nvcc is 13.5, its runtime headers 13.0, and CCCL
    # refuses the pair); the runfile only where a toolkit is cached already (the disk plan has no room for it)
    RT=$(sed -n 's/^#define[[:space:]]*CUDART_VERSION[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$SITE/nvidia/cu13/include/cuda_runtime_api.h" | head -n 1)
    RT_MM=$((RT / 1000)).$(((RT % 1000) / 10))
    RT_PKG=$("$LPY" -I -c 'import importlib.metadata as m; print(m.version("nvidia-cuda-runtime"))')
    CU_C=$KV_HOME/build/cu13_compiler
    rm -rf "$CU_C"
    pip_in "$LPY" -q --no-deps --target "$CU_C" "nvidia-cuda-nvcc==$RT_MM.*" "nvidia-nvvm==$RT_MM.*" \
        "nvidia-cuda-crt==$RT_MM.*" "nvidia-cuda-cccl==$RT_MM.*" "nvidia-cuda-runtime==$RT_PKG"
    echo "   the CUDA $RT_MM compiler (pip, $(du -sh "$CU_C" | cut -f1)) for the environment's CUDA $RT_MM headers (nvidia-cuda-runtime $RT_PKG)"
    TK13=$(CUDA13_FROM_SITE=$SITE CUDA13_COMPILER=$CU_C CUDA13_NO_RUNFILE=1 CUDA13_HOME=${CUDA13_HOME:-$KV_HOME/cuda13} \
        KV_HOME=$KV_HOME bash scripts/cuda13_toolkit.sh | tail -n 1)
    pip_in "$LPY" -q setuptools wheel ninja         # a build without isolation: the environment's own, with its torch
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd';' -)
    B=$KV_HOME/build/lmc_ops
    rm -rf "$B" && mkdir -p "$B"
    if ! ( cd "$B" && CUDA_HOME=$TK13 PATH="$TK13/bin:$KV_HOME/venv_lmcache/bin:$PATH" TORCH_CUDA_ARCH_LIST=$ARCH \
            MAX_JOBS=${MAX_JOBS:-8} "$LPY" "$REPO/scripts/lmc_build_ops.py" --src "$REPO/third_party/lmcache" \
            --out "$B/out" ) > "$KV_HOME/lmc_ops_build.log" 2>&1; then
        echo "STOP: LMCache's compiled modules did not build; the end of $KV_HOME/lmc_ops_build.log:"
        tail -n 40 "$KV_HOME/lmc_ops_build.log"
        exit 1
    fi
    grep -E "^   (building|built)" "$KV_HOME/lmc_ops_build.log" || true
    cp "$B/out/BUILD.json" "$KV_HOME/lmc_ops_build.json"
    EXT=(--ext "$B/out") NEED_OPS=(--need-ops)
fi

echo "== overlay $KV_HOME/lmc_overlay: the wheels' packages, our hooked Python over them"
SITE=$("$LPY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
"$LPY" -I scripts/lmc_overlay.py build --site "$SITE" --out "$KV_HOME/lmc_overlay" "${EXT[@]}"
# what the engines will import, checked as they import it: -P and a folder of its own, so neither the script's
# folder nor the checkout (which has its own kvtrace/) comes before the overlay on the path (job 30688 stopped here)
CHECK_DIR=$(mktemp -d "$TMPDIR/kvtrace_check.XXXXXX")
( cd "$CHECK_DIR" && PYTHONPATH="$KV_HOME/lmc_overlay" KVTRACE_DIR="$CHECK_DIR/trace" KVTRACE_ROLE=check \
    "$LPY" -P "$REPO/scripts/lmc_overlay.py" verify --out "$KV_HOME/lmc_overlay" "${NEED_OPS[@]}" ) || VERIFIED=$?
cp "$KV_HOME/lmc_overlay/VERIFY.json" "$KV_HOME/lmc_verify.json" 2>/dev/null || true
[ "${VERIFIED:-0}" = 0 ] || { echo "STOP: the overlay is not what the engines must import (above)"; exit 1; }
rm -rf -- "$CHECK_DIR"

# IBM's MTRAG (github.com/IBM/mt-rag-benchmark, Apache-2.0): 110 human multi-turn RAG conversations, and the
# generation tasks made from them (each turn with its conversation's id: RAG.jsonl has the retriever's top 5 passages
# for each turn, reference.jsonl the passages marked relevant). Files at a pinned commit, each checked by its
# SHA-256; data only (read with python -I). conversations.json is the file scripts/pd_setup.sh fetches.
echo "== MTRAG"
MTRAG_COMMIT=e0970de1b1207c2a9506adf238e6e1a22037e38a
MTRAG_DIR=$KV_RUNS/mtrag
mkdir -p "$MTRAG_DIR"
mtrag_fetch() {   # path in the repository, its SHA-256, the file name here
    local src=$1 sha=$2 dst=$MTRAG_DIR/$3
    if [ -s "$dst" ] && echo "$sha  $dst" | sha256sum -c --quiet - 2>/dev/null; then
        echo "MTRAG: $dst (as fetched before)"
        return 0
    fi
    local tmp
    tmp=$(mktemp -d "$TMPDIR/mtrag.XXXXXX")
    if curl -fsSL --retry 3 -o "$tmp/f" "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/$MTRAG_COMMIT/$src" ||
       { git -C "$tmp" init -q && git -C "$tmp" remote add origin https://github.com/IBM/mt-rag-benchmark.git &&
         git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$MTRAG_COMMIT" &&
         git -C "$tmp" show "$MTRAG_COMMIT:$src" > "$tmp/f"; }; then
        echo "$sha  $tmp/f" | sha256sum -c --quiet - || {
            echo "STOP: MTRAG's $src does not match its SHA-256"; rm -rf -- "$tmp"; exit 1; }
        mv "$tmp/f" "$dst"
        echo "MTRAG: $dst (commit ${MTRAG_COMMIT:0:12}, SHA-256 checked)"
    else
        echo "STOP: MTRAG's $src could not be fetched"; rm -rf -- "$tmp"; exit 1
    fi
    rm -rf -- "$tmp"
}
mtrag_fetch mtrag-human/conversations/conversations.json \
    1a30baf5e025b4915e87556c64de37d7fe96e98f9f4ea9f3cf9322199e538f89 conversations.json
mtrag_fetch mtrag-human/generation_tasks/RAG.jsonl \
    5d5201da9fabd072fd8f6b8d051bfaafa7ef031e76722a4920c66e94cede1873 RAG.jsonl
mtrag_fetch mtrag-human/generation_tasks/reference.jsonl \
    b3dde4076f9f692a99513e424a0a39f57591dbe45600a4c1017aaa7a7edcfaac reference.jsonl

echo "== the model(s)"
# Llama-3.2-3B always; Qwen3-4B-Instruct-2507 too for the KV pool experiment (KVPD_LMC_PRESET=pool, or
# LMC_SETUP_MODELS=qwen3_4b), its fixture then checked against its tokenizer (scripts/lmc_pool.py check)
MODELS=llama32_3b
case ",${LMC_SETUP_MODELS:-}," in *,qwen3_4b,*) MODELS+=" qwen3_4b" ;; esac
[ "${KVPD_LMC_PRESET:-}" = pool ] && [[ " $MODELS " != *" qwen3_4b "* ]] && MODELS+=" qwen3_4b"
for m in $MODELS; do
    case "$m" in
        llama32_3b) repo=unsloth/Llama-3.2-3B rev=d4446454d87d51aa42e1fb174f25acc5f8762331 ;;
        qwen3_4b) repo=Qwen/Qwen3-4B-Instruct-2507 rev=cdbee75f17c01a7cc42f958dc650907174af0554 ;;
    esac
    path=$("$LPY" -I -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$repo" "$rev")
    echo "   $m: $repo at ${rev:0:12}: $path"
    if [ "$m" = qwen3_4b ]; then
        "$LPY" -I scripts/lmc_pool.py check --fixture workloads/kv-pool-concurrency-chat-v1.json --tokenizer "$path" \
            --output "$KV_HOME/pool_fixture_check.json" || {
            echo "STOP: the pool fixture does not match $repo's tokenizer ($KV_HOME/pool_fixture_check.json)"; exit 1; }
    fi
done
echo "LMC SETUP DONE"
