# Sourced by scripts/a5000_setup.sh and scripts/pd_setup.sh: how environments are made and packages installed.
#
# Default (the NUC, a plain batch job): python -m venv and pip, as before.
# KV_UV=/path/to/uv (scripts/pd_card.sh sets it on gpu-grad-01): uv. With UV_CACHE_DIR on the same file system as the
# environments and UV_LINK_MODE=hardlink, every environment's files are hard links into the one cache, so a fresh
# environment per job costs almost no disk (the node's /tmp has less room than three copies of torch and vLLM).
#
#   venv_create DIR          an environment at DIR (kept if it is there)
#   pip_in PY ARGS...        install into the environment of interpreter PY
#   pip_out PY NAMES...      uninstall (missing packages are no error)
#   pip_freeze PY            the installed versions
if [ -n "${KV_UV:-}" ]; then
    [ -x "$KV_UV" ] || { echo "STOP: KV_UV=$KV_UV is not an executable uv"; exit 1; }
    # pip's choice among indexes (the best version across PyPI and PyTorch's index), no user configuration
    export UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY:-unsafe-best-match} UV_NO_CONFIG=1
    venv_create() {
        [ -x "$1/bin/python" ] && return 0
        "$KV_UV" venv -q --python "${KV_UV_PYTHON:-3.12}" "$1"
    }
    pip_in() { local py=$1; shift; "$KV_UV" pip install --python "$py" "$@"; }
    pip_out() { local py=$1; shift; "$KV_UV" pip uninstall -q --python "$py" "$@" || true; }
    pip_freeze() { "$KV_UV" pip freeze --python "$1" 2>/dev/null; }
else
    venv_create() {
        [ -x "$1/bin/python" ] && return 0
        "${PY:?}" -m venv "$1"
        "$1/bin/python" -m pip install -q --upgrade pip
    }
    pip_in() { local py=$1; shift; "$py" -m pip install "$@"; }
    pip_out() { local py=$1; shift; "$py" -m pip uninstall -q -y "$@" 2>/dev/null || true; }
    pip_freeze() { "$1" -m pip freeze; }
fi
