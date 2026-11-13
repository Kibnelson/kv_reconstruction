#!/usr/bin/env bash
# Environment-only stage for an isolated RTX P/D smoke. No model or engine launch.
set -Eeuo pipefail
trap 'printf "PD ENV STOPPED at line %s (exit %s)\n" "$LINENO" "$?" >&2' ERR
cd "$(dirname "$0")/.."
PD_REPO=$PWD
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
PD_VENV=${PD_VENV:-$KV_HOME/venv_vllm}
if [ "${TRITON_INTERPRET:-0}" != 0 ]; then
    echo 'STOP: unset TRITON_INTERPRET; the readiness check compiles a CUDA kernel.' >&2
    exit 2
fi
unset PYTHONPATH PYTHONHOME
PD_BOOTSTRAP_PYTHON=${PD_BOOTSTRAP_PYTHON:-$(command -v python3.12 || command -v python3)}
PD_VENV=$("$PD_BOOTSTRAP_PYTHON" - "$KV_HOME" "$PD_VENV" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:2] != (3,12):
    sys.exit('STOP: select Python 3.12 with PD_BOOTSTRAP_PYTHON.')
root, target = (Path(x).expanduser().resolve() for x in sys.argv[1:])
codec = (root/'venv').resolve()
if not root.is_dir():
    sys.exit('STOP: KV_HOME must be the existing codec data directory.')
if root not in target.parents or target==codec or codec in target.parents or target in codec.parents:
    sys.exit('STOP: PD_VENV must be below KV_HOME and separate from the codec venv.')
if target.exists() and not (target/'pyvenv.cfg').is_file():
    sys.exit('STOP: existing PD_VENV is not a virtual environment; choose a fresh path.')
print(target)
PY
)
[ -x "$KV_HOME/venv/bin/python" ] || { echo 'STOP: existing codec venv is missing.' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'STOP: nvidia-smi is unavailable.' >&2; exit 2; }
PD_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)
[ "$PD_DRIVER" -ge 580 ] || { echo 'STOP: this CUDA 13 environment needs an R580 or newer driver.' >&2; exit 2; }
"$PD_BOOTSTRAP_PYTHON" - "$KV_HOME" "${PD_MIN_FREE_GIB:-20}" <<'PY'
import shutil,sys
free = shutil.disk_usage(sys.argv[1]).free / 2**30
minimum = float(sys.argv[2])
print(f'Disk space under KV_HOME: {free:.1f} GiB free; setup headroom: {minimum:g} GiB.')
if free < minimum:
    sys.exit('STOP: insufficient selected disk headroom for the separate vLLM environment.')
PY
PD_ENV_OUTPUT=${PD_ENV_OUTPUT:-$KV_HOME/runs/pd_rtx_env_v1_$(date -u +%Y%m%dT%H%M%SZ)_$$}
mkdir -p "$(dirname "$PD_ENV_OUTPUT")"
mkdir "$PD_ENV_OUTPUT"
exec > >(tee "$PD_ENV_OUTPUT/setup.log") 2>&1
export HF_HOME=$KV_HOME/hf
export TMPDIR=$KV_HOME/tmp/pd_rtx_env XDG_CACHE_HOME=$KV_HOME/cache/pd_rtx_env
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv
nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv > "$PD_ENV_OUTPUT/gpus_before.csv"
pd_codec_snapshot() {
    "$KV_HOME/venv/bin/python" - <<'PY'
from importlib import metadata
import json,sys
d = {'python': sys.version.split()[0], 'prefix': sys.prefix}
for name in ('torch','triton','transformers','vllm'):
    try: d[name] = metadata.version(name)
    except metadata.PackageNotFoundError: d[name] = None
print(json.dumps(d,sort_keys=True))
PY
}
PD_CODEC_BEFORE=$(pd_codec_snapshot)
printf '%s\n' "$PD_CODEC_BEFORE" > "$PD_ENV_OUTPUT/codec_env_before.json"
if [ ! -x "$PD_VENV/bin/python" ]; then
    "$PD_BOOTSTRAP_PYTHON" -m venv "$PD_VENV"
fi
PD_PY=$PD_VENV/bin/python
"$PD_PY" - "$PD_VENV" <<'PY'
from pathlib import Path
import sys
if sys.version_info[:2] != (3,12) or sys.prefix==sys.base_prefix or Path(sys.prefix).resolve()!=Path(sys.argv[1]).resolve():
    sys.exit('STOP: the selected Python is not the intended isolated Python 3.12 venv.')
PY
echo "Installing vLLM 0.32.0 into $PD_VENV"
"$PD_PY" -m pip --isolated install --no-cache-dir --upgrade pip
"$PD_PY" -m pip --isolated install --no-cache-dir 'vllm==0.32.0' 'aiohttp>=3.10'
"$PD_PY" -m pip check
"$PD_PY" -m pip freeze > "$PD_ENV_OUTPUT/pip_freeze.txt"
"$PD_PY" scripts/check_pd_rtx_env.py --output "$PD_ENV_OUTPUT/environment.json"
PD_CODEC_AFTER=$(pd_codec_snapshot)
printf '%s\n' "$PD_CODEC_AFTER" > "$PD_ENV_OUTPUT/codec_env_after.json"
[ "$PD_CODEC_BEFORE" = "$PD_CODEC_AFTER" ] || { echo 'STOP: codec environment fingerprint changed.' >&2; exit 2; }
printf 'PD ENV READY: %s\n' "$PD_ENV_OUTPUT/environment.json"
printf 'Codec environment fingerprint is unchanged. Separate P/D engines have not been started.\n'
