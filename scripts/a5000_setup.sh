#!/usr/bin/env bash
# One-time (idempotent) setup for the lossless comparison on one RTX A5000, run INSIDE the one-GPU Slurm allocation
# (docs/a5000_runbook.md). Builds everything under KV_HOME (default ~/kv_a5000), which persists across allocations:
#
#   $KV_HOME/venv            Python environment: torch 2.11.0+cu128 (the NUC's), transformers 4.57.6, pyarrow,
#                            cmake and nvCOMP for the ExANS build
#   $KV_HOME/hf              Hugging Face cache (models, datasets)
#   $KV_HOME/sources         the authors' SplitZip and ExANS (openlake) checkouts at the pinned commits
#   $KV_HOME/cuda12_8        a CUDA 12.8 toolkit (nvcc) for building ExANS, unless one is on PATH or CUDA_HOME is set
#
# Options (environment): KV_HOME, PYTHON (default python3.12, else python3), SKIP_EXANS=1 (no toolkit, ExANS left
# out of the comparison), CUDA_RUNFILE_URL (another CUDA 12.x toolkit installer).
set -Eeuo pipefail
trap 'echo "SETUP STOPPED at line $LINENO (exit $?); nothing is half-installed that a rerun does not fix" >&2' ERR
cd "$(dirname "$0")/.."
REPO=$PWD
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
mkdir -p "$KV_HOME"/{hf,sources,tmp}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (SLURM_JOB_ID is empty). Request the GPU first (runbook step 1)."
    exit 1
fi
echo "== allocation: job ${SLURM_JOB_ID:-none} on $(hostname), CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$NGPU" -ge 1 ] || { echo "STOP: no GPU visible in this allocation"; exit 1; }
[ "$NGPU" -eq 1 ] || echo "NOTE: $NGPU GPUs visible; the runs use the first one (cuda:0)"

echo "== network from this node (the runs download models, datasets and the baselines' sources)"
for url in https://pypi.org/simple/ https://download.pytorch.org/whl/cu128/ https://huggingface.co https://github.com \
           https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt; do
    if curl -sS -o /dev/null --max-time 20 -I "$url"; then echo "  ok  $url"; else
        echo "STOP: $url is not reachable from $(hostname). Ask whether compute nodes need a proxy (and export"
        echo "      HTTPS_PROXY before this script), or run the downloads once where they work and point KV_HOME there."
        exit 1
    fi
done
command -v gcc >/dev/null || { echo "STOP: no gcc on this node (Triton compiles a small launcher): module load gcc"; exit 1; }
command -v git >/dev/null || { echo "STOP: no git on this node"; exit 1; }
command -v g++ >/dev/null || [ "${SKIP_EXANS:-0}" = 1 ] || { echo "STOP: no g++ (the ExANS build): module load gcc, or SKIP_EXANS=1"; exit 1; }

echo "== disk under $KV_HOME"
df -h "$KV_HOME" | tail -n 1
AVAIL_GB=$(df -Pk "$KV_HOME" | awk 'NR==2 {print int($4 / 1048576)}')
[ "$AVAIL_GB" -ge 60 ] || echo "WARNING: only $AVAIL_GB GB free under $KV_HOME (venv ~8 GB, CUDA toolkit ~8 GB, each model 7-17 GB)"

PY=${PYTHON:-}
if [ -z "$PY" ]; then
    PY=$(command -v python3.12 || command -v python3 || true)
fi
[ -n "$PY" ] || { echo "STOP: no python3 (module load python, or set PYTHON=/path/to/python3.12)"; exit 1; }
"$PY" - <<'EOF'
import sys
v = sys.version_info
print(f"== python {sys.version.split()[0]} ({sys.executable})")
if v < (3, 10):
    sys.exit("STOP: Python 3.10 or newer needed (the NUC runs used 3.12): module load python/3.12 or set PYTHON")
EOF
if [ ! -x "$KV_HOME/venv/bin/python" ]; then
    "$PY" -m venv "$KV_HOME/venv"
fi
# shellcheck disable=SC1091
source "$KV_HOME/venv/bin/activate"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements-cuda.txt
# nvCOMP 5.x: ExANS (openlake de6c38b) calls nvcompBatchedANS*GetTempSizeAsync, which nvCOMP 6.0 renamed; the
# Python package pins the native library to its own version, so both stay on 5.3.0.16
python -m pip install -q pyarrow cmake ninja packaging nvidia-libnvcomp-cu12==5.3.0.16 nvidia-nvcomp-cu12==5.3.0.16
# Triton compiles only functions defined in a file: the check runs from one
cat > "$KV_HOME/tmp/triton_check.py" <<'EOF'
import torch, triton, transformers
print(f"== torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, transformers {transformers.__version__}")
assert torch.cuda.is_available(), "torch sees no GPU"
print("   GPU:", torch.cuda.get_device_name(0))
import triton.language as tl

@triton.jit
def _add(x, y, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(y + i, tl.load(x + i, mask=i < n) + 1, mask=i < n)
a = torch.arange(1000, device="cuda", dtype=torch.float32)
b = torch.empty_like(a)
_add[(4,)](a, b, 1000, B=256)
assert torch.equal(b, a + 1), "Triton kernel gave a wrong result"
print("   a Triton kernel compiled and ran")
EOF
python "$KV_HOME/tmp/triton_check.py"
python -m pip freeze > "$KV_HOME/pip_freeze.txt"

echo "== the authors' codecs at the pinned commits"
read -r SZ_REPO SZ_COMMIT EX_REPO EX_COMMIT < <(python - <<'EOF'
from kv_compare.mixed import SPLITZIP_REPO, SPLITZIP_COMMIT
from kv_compare.exans_official import EXANS_REPO, EXANS_COMMIT
print(SPLITZIP_REPO, SPLITZIP_COMMIT, EXANS_REPO, EXANS_COMMIT)
EOF
)
SZ=$KV_HOME/sources/splitzip
[ -d "$SZ/.git" ] || git clone -q "$SZ_REPO" "$SZ"
git -C "$SZ" cat-file -e "$SZ_COMMIT" 2>/dev/null || git -C "$SZ" fetch -q origin "$SZ_COMMIT"
git -C "$SZ" checkout -q "$SZ_COMMIT"
echo "   SplitZip $(git -C "$SZ" rev-parse --short HEAD) in $SZ"
EX=$KV_HOME/sources/exans
[ -d "$EX/.git" ] || git clone -q --filter=blob:none "$EX_REPO" "$EX"
git -C "$EX" cat-file -e "$EX_COMMIT" 2>/dev/null || git -C "$EX" fetch -q origin "$EX_COMMIT"
git -C "$EX" checkout -q "$EX_COMMIT"           # at the commit, the runs build from it directly (no worktree)
echo "   ExANS (openlake) $(git -C "$EX" rev-parse --short HEAD) in $EX; the runs build it from source"

echo "== the other bit-exact codecs (docs/lossless_related_work.md) at their pinned commits"
read -r ADA_REPO ADA_COMMIT XKV_REPO XKV_COMMIT IBP_REPO IBP_COMMIT < <(python - <<'EOF'
from kv_compare.more_baselines import (ADACODE_REPO, ADACODE_COMMIT, EXACTKV_REPO, EXACTKV_COMMIT, IBP_REPO,
                                       IBP_COMMIT)
print(ADACODE_REPO, ADACODE_COMMIT, EXACTKV_REPO, EXACTKV_COMMIT, IBP_REPO, IBP_COMMIT)
EOF
)
for spec in "adacode $ADA_REPO $ADA_COMMIT" "exactkv $XKV_REPO $XKV_COMMIT" "ibp $IBP_REPO $IBP_COMMIT"; do
    read -r name url commit <<< "$spec"
    D=$KV_HOME/sources/$name
    [ -d "$D/.git" ] || git clone -q "$url" "$D"
    git -C "$D" cat-file -e "$commit" 2>/dev/null || git -C "$D" fetch -q origin "$commit"
    git -C "$D" checkout -q "$commit"
    echo "   $name $(git -C "$D" rev-parse --short HEAD) in $D"
done
python - <<'EOF'
from nvidia import nvcomp
print("   nvCOMP (Python)", getattr(nvcomp, "__version__", "?"))
EOF

if [ "${SKIP_EXANS:-0}" = 1 ]; then
    echo "== SKIP_EXANS=1: no CUDA toolkit; ExANS is left out of the comparison"
else
    echo "== CUDA toolkit with nvcc (ExANS is built from source)"
    TK=""
    # a CUDA 12 nvcc: the nvCOMP wheel (cu12) and torch's CUDA 12.8 runtime match it; a CUDA 13 module would not
    is12() { "$1/bin/nvcc" --version 2>/dev/null | grep "release 12\." >/dev/null; }
    PATH_TK=""
    command -v nvcc >/dev/null && PATH_TK=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
    if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ] && is12 "$CUDA_HOME"; then TK=$CUDA_HOME
    elif [ -x "$KV_HOME/cuda12_8/bin/nvcc" ]; then TK=$KV_HOME/cuda12_8
    elif [ -n "$PATH_TK" ] && is12 "$PATH_TK"; then TK=$PATH_TK
    else
        URL=${CUDA_RUNFILE_URL:-https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run}
        echo "   no nvcc found: installing the toolkit only (no driver) from $URL into $KV_HOME/cuda12_8 (about 5 GB)"
        curl -fL --retry 3 -o "$KV_HOME/tmp/cuda.run" "$URL"
        # no root needed for the toolkit alone; the installer unpacks under TMPDIR ($KV_HOME/tmp)
        sh "$KV_HOME/tmp/cuda.run" --silent --toolkit --installpath="$KV_HOME/cuda12_8" --no-opengl-libs \
            --no-man-page --no-drm --override
        rm -f "$KV_HOME/tmp/cuda.run"
        TK=$KV_HOME/cuda12_8
    fi
    "$TK/bin/nvcc" --version | tail -n 2
    echo "$TK" > "$KV_HOME/cuda_home"
    echo "   CUDA_HOME for the runs: $TK"
fi
# IBP is a CUDA extension: built with the same toolkit (its setup.py reads CONDA_PREFIX for headers; the venv
# stands in). Optional: SKIP_IBP=1, and a failed build only leaves IBP out of the comparison.
if [ "${SKIP_IBP:-0}" != 1 ] && [ -s "$KV_HOME/cuda_home" ]; then
    if python -c "import torch, ibp" 2>/dev/null; then
        echo "== IBP: already built"
    else
        echo "== IBP: building the CUDA extension ($KV_HOME/ibp_build.log, a few minutes)"
        if (cd "$KV_HOME/sources/ibp" && CUDA_HOME=$(cat "$KV_HOME/cuda_home") CONDA_PREFIX=$VIRTUAL_ENV \
                PATH=$(cat "$KV_HOME/cuda_home")/bin:$PATH python -m pip install -v . --no-build-isolation) \
                > "$KV_HOME/ibp_build.log" 2>&1 && python -c "import torch, ibp"; then
            echo "   IBP built"
        else
            echo "   WARNING: IBP did not build (tail of $KV_HOME/ibp_build.log below); the runs leave it out"
            tail -n 15 "$KV_HOME/ibp_build.log"
        fi
    fi
fi
echo "SETUP DONE: KV_HOME=$KV_HOME (python $KV_HOME/venv/bin/python)"
