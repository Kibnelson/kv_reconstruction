#!/usr/bin/env bash
# One-time (idempotent) setup of the vLLM PD runs (docs/pd_protocol.md), after scripts/a5000_setup.sh, INSIDE the
# allocation (A5000: the 4-GPU one; NUC: ALLOW_NO_SLURM=1). Adds under KV_HOME:
#
#   $KV_HOME/venv_vllm         vLLM 0.32.0 (its own torch 2.13 + CUDA 13; the codec environment keeps its pins)
#   $KV_HOME/sources/vllm      the vLLM source at v0.32.0 (benchmarks/multi_turn)
#   $RUN/captures              the WikiText-2 train calibration KV (bench_llama_compare.py --stages calcap)
#   $RUN/pd_codecs.pt          the transfer codecs' constants (scripts/pd_export_codecs.py, codec environment), with
#                              T2's constant sets (T2_SETS; their calibration windows captured first, ~16 GB)
#   $RUN/exans/                the authors' ExANS library built with the CUDA 12.8 toolkit, and exans_info.json
#   $KV_HOME/venv_lmcache      vLLM 0.32.0 with LMCache (its own environment: LMCache pins other versions)
#   $RUN/pd_selftest.json      every codec round-tripped through the paced transport in the vLLM environment
#
# Options: KV_HOME, KV_RUNS, MODEL (llama32_3b), PYTHON (python3.12), T2_SETS (wikitext,code,math,exam,news; empty
# for none), LMCACHE=0 (no LMCache environment), NIXL=0 (no NIXL in the vLLM environment), IBP=0 (no IBP build in the
# vLLM environment; CUDA13_HOME: a CUDA 13 toolkit to build it with), KEEP_CAL_CAPTURES=1 (keep the T2 calibration
# captures after the export; default: deleted).
set -Eeuo pipefail
trap 'echo "PD SETUP STOPPED at line $LINENO (exit $?); a rerun continues" >&2' ERR
cd "$(dirname "$0")/.."
REPO=$PWD
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp
mkdir -p "$TMPDIR" "$KV_HOME/sources"
# every cache under KV_HOME, not the home directory (its quota ran out installing vLLM on gpu-grad-01): pip, vLLM and
# FlashInfer follow XDG_CACHE_HOME; Triton has its own variable
export XDG_CACHE_HOME=$KV_HOME/cache TRITON_CACHE_DIR=$KV_HOME/cache/triton
mkdir -p "$TRITON_CACHE_DIR"
case "$KV_HOME/" in
    "$HOME"/*) echo "NOTE: KV_HOME ($KV_HOME) is in your home directory. On a cluster with a home quota (gpu-grad-01:"
               echo "      'Disk quota exceeded' installing vLLM), set KV_HOME to a place with about 70 GB first." ;;
esac
command -v quota >/dev/null && quota -s 2>/dev/null | tail -n +2 | sed 's/^/   quota: /' || true
AVAIL_GB=$(df -Pk "$KV_HOME" | awk 'NR==2 {print int($4 / 1048576)}')
[ "$AVAIL_GB" -ge 25 ] || echo "WARNING: $AVAIL_GB GB free under $KV_HOME (the vLLM environment takes about 12 GB, the" \
    "calibration captures about 7 GB; a quota can be smaller than what df shows)"
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
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n 1p | cut -d. -f1)
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
# nvCOMP's CUDA 13 build (the baseline codecs nvcomp-<alg>) beside vLLM's torch, and NIXL (vLLM's NixlConnector)
"$VPY" -c "from nvidia import nvcomp" 2>/dev/null || "$VPY" -m pip install -q "nvidia-nvcomp-cu13==6.0.0.12" || \
    echo "WARNING: nvCOMP did not install in the vLLM environment; the nvcomp codecs are left out"
if [ "${NIXL:-1}" = 1 ] && ! "$VPY" -c "import nixl" 2>/dev/null; then
    "$VPY" -m pip install -q "nixl-cu13==1.5.2" || "$VPY" -m pip install -q "nixl==1.5.2" || \
        echo "WARNING: NIXL did not install; the NIXL reference runs are left out"
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
for s in splitzip adacode exactkv; do
    [ -d "$KV_HOME/sources/$s" ] || echo "NOTE: no $KV_HOME/sources/$s (scripts/a5000_setup.sh clones it); $s is left out"
done

if [ "${LMCACHE:-1}" = 1 ]; then
    echo "== LMCache environment $KV_HOME/venv_lmcache (the CPU-tier reference)"
    [ -x "$KV_HOME/venv_lmcache/bin/python" ] || "$PY" -m venv "$KV_HOME/venv_lmcache"
    LPY=$KV_HOME/venv_lmcache/bin/python
    if ! "$LPY" -c "import vllm, lmcache, sys; sys.exit(0 if vllm.__version__ == '0.32.0' else 1)" 2>/dev/null; then
        "$LPY" -m pip install -q --upgrade pip
        "$LPY" -m pip install -q "vllm==0.32.0" "lmcache==0.5.7.post1" "aiohttp>=3.10" || \
            echo "WARNING: vLLM 0.32.0 and LMCache 0.5.7.post1 did not install together; the LMCache reference is left out"
    fi
    "$LPY" -c "import vllm, lmcache; print(f'   vllm {vllm.__version__}, lmcache {lmcache.__version__}')" || \
        echo "WARNING: LMCache does not import; the LMCache reference is left out"
fi

if [ "${IBP:-1}" = 1 ]; then
    # IBP decompresses from host memory on the GPU: the CPU tier's baseline, built against vLLM's CUDA 13 torch with a
    # CUDA 13 toolkit (toolkit only, no driver; about 5 GB). Build fix, recorded in ibp_build_fix.diff: CUDA 13 has no
    # compute_70, so its setup.py's sm_70 target becomes the A5000's sm_86 (its sm_80 target stays).
    if "$VPY" -c "import torch, ibp" 2>/dev/null; then
        echo "== IBP in the vLLM environment: already built"
    elif [ ! -d "$KV_HOME/sources/ibp" ]; then
        echo "NOTE: no $KV_HOME/sources/ibp (scripts/a5000_setup.sh clones it); IBP is left out"
    else
        echo "== IBP in the vLLM environment ($KV_HOME/ibp_vllm_build.log)"
        TK13=${CUDA13_HOME:-$KV_HOME/cuda13}
        if [ ! -x "$TK13/bin/nvcc" ]; then
            URL=${CUDA13_RUNFILE_URL:-https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run}
            echo "   the CUDA 13 toolkit from $URL into $TK13"
            curl -fL --retry 3 -o "$KV_HOME/tmp/cuda13.run" "$URL" && \
                sh "$KV_HOME/tmp/cuda13.run" --silent --toolkit --installpath="$TK13" --no-opengl-libs --no-man-page \
                    --no-drm --override || echo "   WARNING: the CUDA 13 toolkit did not install"
            rm -f "$KV_HOME/tmp/cuda13.run"
        fi
        B=$KV_HOME/build/ibp_vllm
        rm -rf "$B" && mkdir -p "$(dirname "$B")" && cp -r "$KV_HOME/sources/ibp" "$B"
        sed -i 's/arch=compute_70,code=sm_70/arch=compute_86,code=sm_86/' "$B/setup.py"
        (cd "$KV_HOME/sources/ibp" && git diff --no-index -- setup.py "$B/setup.py") > "$B/ibp_build_fix.diff" || true
        if [ -x "$TK13/bin/nvcc" ] && (cd "$B" && CUDA_HOME=$TK13 CONDA_PREFIX=$KV_HOME/venv_vllm PATH=$TK13/bin:$PATH \
                "$VPY" -m pip install -v . --no-build-isolation) > "$KV_HOME/ibp_vllm_build.log" 2>&1 && \
                "$VPY" -c "import torch, ibp"; then
            echo "   IBP built"
        else
            echo "   WARNING: IBP did not build (tail of $KV_HOME/ibp_vllm_build.log below); the tier runs leave it out"
            tail -n 15 "$KV_HOME/ibp_vllm_build.log" 2>/dev/null || true
        fi
    fi
fi

if [ -s "$KV_HOME/cuda_home" ] && [ ! -s "$RUN/exans/exans_info.json" ]; then
    echo "== ExANS (the authors' library, CUDA 12.8 toolkit, codec environment)"
    mkdir -p "$RUN/exans"
    PYTHONPATH=$REPO "$KV_HOME/venv/bin/python" - "$RUN/exans" "$KV_HOME/sources/exans" "$(cat "$KV_HOME/cuda_home")" <<'EOF' \
        || echo "WARNING: ExANS did not build; it is left out"
import json, sys
from pathlib import Path
from kv_compare import exans_official as ex
work, src, cuda = sys.argv[1:4]
info = ex.build(work, src if Path(src).is_dir() else None, cuda)
Path(work, "exans_info.json").write_text(json.dumps(info, indent=1))
print("   ExANS built:", info["library"])
EOF
elif [ ! -s "$KV_HOME/cuda_home" ]; then
    echo "NOTE: no CUDA toolkit ($KV_HOME/cuda_home: scripts/a5000_setup.sh without SKIP_EXANS=1); ExANS is left out"
fi

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

T2_SETS=${T2_SETS-wikitext,code,math,exam,news}
if [ -n "$T2_SETS" ] && [ ! -s "$TEXTS/cal_manifest.json" ]; then
    "$KV_HOME/venv/bin/python" scripts/fetch_calib_texts.py --output "$TEXTS"
fi
echo "== codec constants $RUN/pd_codecs.pt (codec environment)"
HAVE_T2=none
if [ -s "$RUN/pd_codecs.pt" ]; then
    HAVE_T2=$("$KV_HOME/venv/bin/python" - "$RUN/pd_codecs.pt" <<'EOF' 2>/dev/null || echo none
import sys, torch
b = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(",".join(b.get("t2", {}).get("order", [])) or "none")
EOF
)
fi
if [ ! -s "$RUN/pd_codecs.pt" ] || { [ -n "$T2_SETS" ] && [ "$HAVE_T2" != "$T2_SETS" ]; }; then
    SZ=()
    [ -d "$KV_HOME/sources/splitzip" ] && SZ=(--splitzip-source "$KV_HOME/sources/splitzip")
    T2=()
    [ -n "$T2_SETS" ] && T2=(--t2-sets "$T2_SETS" --texts "$TEXTS")
    "$KV_HOME/venv/bin/python" -u scripts/pd_export_codecs.py --run "$RUN" --output "$RUN/pd_codecs.pt" "${SZ[@]}" "${T2[@]}"
    if [ -n "$T2_SETS" ] && [ "${KEEP_CAL_CAPTURES:-0}" != 1 ]; then
        rm -f "$RUN"/captures/cal_*                  # the T2 calibration windows: in pd_codecs.pt now
    fi
fi

echo "== self-test in the vLLM environment (every codec through the paced transport)"
DOC=ds_wikitext2_r0
[ -s "$RUN/captures/${DOC}_K_bits.npy" ] || DOC=wtrain0
EXANS_INFO=()
[ -s "$RUN/exans/exans_info.json" ] && EXANS_INFO=(--exans-info "$RUN/exans/exans_info.json")
# two processes: ExANS links nvCOMP's CUDA 12 library and the nvcomp codecs load its CUDA 13 one (the matrices keep
# them on different engines too). A codec that fails here is reported; the engines' warm-up leaves it out.
SELFTEST_OK=1
for part in main nvcomp; do
    if [ "$part" = main ]; then
        NAMES=${SELFTEST_CODECS:-raw,ours-dense,ours-kv,ours-4g,ours-hbh,ours-rank256,t2-dense,t2-hbh,t2-rank256,splitzip,adacode,exactkv,exans,ibp}
        OUTF=$RUN/pd_selftest.json
    else
        NAMES=${SELFTEST_NVCOMP:-nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp}
        OUTF=$RUN/pd_selftest_nvcomp.json
    fi
    PYTHONPATH=$REPO "$VPY" -m kv_compare.pd.selftest --codecs "$RUN/pd_codecs.pt" --captures "$RUN/captures" \
        --doc "$DOC" --tokens 8192,3000 --splitzip-source "$KV_HOME/sources/splitzip" \
        --adacode-source "$KV_HOME/sources/adacode" --exactkv-source "$KV_HOME/sources/exactkv" "${EXANS_INFO[@]}" \
        --names "$NAMES" --output "$OUTF" || SELFTEST_OK=0
done
[ "$SELFTEST_OK" = 1 ] || echo "WARNING: a codec failed the self-test (above, and $RUN/pd_selftest*.json); the runs leave it out"
echo "PD SETUP DONE"
