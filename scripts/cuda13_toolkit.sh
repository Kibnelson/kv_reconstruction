#!/usr/bin/env bash
# A CUDA 13 toolkit (nvcc and headers only: no driver, no root) in CUDA13_HOME, installed once and kept; for builds
# against vLLM 0.32's torch (CUDA 13.0), which the cluster's CUDA modules (11.8, 12.6) cannot do. The same install as
# scripts/pd_setup.sh's for IBP (jobs 30293 and 30304: the runfile's last step needs root, so its files are
# extracted when it leaves no nvcc), then pruned to what a torch extension's build uses (about 4 GB).
#
#   CUDA13_HOME=/path KV_HOME=/path bash scripts/cuda13_toolkit.sh     prints the toolkit's path on its last line
#
# First choice, CUDA13_FROM_SITE=<site-packages>: the CUDA 13 headers and libraries an environment already has from
# pip (vLLM 0.32's: the runtime torch was built with, the libraries' headers, all in nvidia/cu13/), with a compiler of
# the same CUDA version: CUDA13_COMPILER=<folder> (pip packages nvidia-cuda-nvcc, -nvvm, -crt, -cccl of that version,
# installed with --target; default: the environment's own). Job 30716: the environment's own nvcc is 13.5 and its
# runtime headers 13.0, and CCCL refuses that pair ("CUDA compiler and CUDA toolkit headers are incompatible"), so
# the two versions are checked first. The toolkit is a folder of links (KV_HOME/build/cuda13_home): the compiler's
# bin/ and nvvm/, the environment's include/ and lib/ with the compiler's own headers (crt/, cccl/) over them,
# lib64/, targets/x86_64-linux/ (where the 13.0 compiler's profile looks), an unversioned libcudart.so. Nothing in
# the environment changes. Only without such a compiler, a runfile toolkit already in CUDA13_HOME (or, unless
# CUDA13_NO_RUNFILE=1, the runfile: about 4 GB kept, 11 GB while it installs).
set -Eeuo pipefail
KV_HOME=${KV_HOME:?}
TK13=${CUDA13_HOME:-$KV_HOME/cuda13}
TMPD=${TMPDIR:-$KV_HOME/tmp}
mkdir -p "$TMPD"
W=${CUDA13_FROM_SITE:+$CUDA13_FROM_SITE/nvidia/cu13}
C=${CUDA13_COMPILER:+$CUDA13_COMPILER/nvidia/cu13}
[ -n "$C" ] || C=$W
if [ -n "$W" ] && [ -x "$C/bin/nvcc" ] && [ -x "$C/nvvm/bin/cicc" ] && [ -f "$W/include/cuda_runtime_api.h" ] && \
        ls "$W"/lib/libcudart.so.* >/dev/null 2>&1; then
    RT=$(sed -n 's/^#define[[:space:]]*CUDART_VERSION[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$W/include/cuda_runtime_api.h" | head -n 1)
    RT_MM=$((RT / 1000)).$(((RT % 1000) / 10))
    NV_MM=$("$C/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
    if [ "$NV_MM" != "$RT_MM" ]; then
        echo "STOP: the compiler in $C is CUDA $NV_MM and the runtime headers in $W are CUDA $RT_MM; CCCL refuses" \
             "that pair (job 30716). Install the compiler of CUDA $RT_MM (CUDA13_COMPILER)." >&2
        exit 1
    fi
    SHIM=${CUDA13_SHIM:-$KV_HOME/build/cuda13_home}
    rm -rf "$SHIM" && mkdir -p "$SHIM/include" "$SHIM/lib" "$SHIM/targets/x86_64-linux"
    ln -s "$C/bin" "$SHIM/bin"
    ln -s "$C/nvvm" "$SHIM/nvvm"
    for e in "$W"/include/*; do ln -s "$e" "$SHIM/include/"; done
    [ "$C" = "$W" ] || for e in "$C"/include/*; do ln -sfn "$e" "$SHIM/include/${e##*/}"; done   # the compiler's own
    for e in "$W"/lib/*; do ln -s "$e" "$SHIM/lib/"; done
    [ "$C" = "$W" ] || for e in "$C"/lib/*; do ln -sfn "$e" "$SHIM/lib/${e##*/}"; done
    for f in "$SHIM"/lib/lib*.so.*; do               # libcudart.so.13 -> libcudart.so, for -lcudart
        stem=${f##*/}; stem=${stem%%.so.*}.so
        [ -e "$SHIM/lib/$stem" ] || ln -s "$(readlink -f "$f")" "$SHIM/lib/$stem"
    done
    ln -s lib "$SHIM/lib64"
    for d in include lib lib64; do ln -s "../../${d/lib64/lib}" "$SHIM/targets/x86_64-linux/$d"; done
    for need in include/crt/host_config.h include/cccl include/cuda_runtime.h; do
        [ -e "$SHIM/$need" ] || { echo "STOP: no $need in the CUDA 13 packages ($C, $W)" >&2; exit 1; }
    done
    echo "   CUDA 13 toolkit: compiler CUDA $NV_MM ($C: $("$SHIM/bin/nvcc" --version | tail -n 1)), runtime headers" \
         "CUDA $RT_MM ($W), as $SHIM" >&2
    echo "$SHIM"
    exit 0
fi
[ -z "$W" ] || echo "   no complete CUDA 13 compiler in $W" >&2
ok() { [ -x "$TK13/bin/nvcc" ] && [ -f "$TK13/include/cuda_runtime.h" ]; }
if ok; then
    echo "   CUDA 13 toolkit: $TK13 ($("$TK13/bin/nvcc" --version | tail -n 1))" >&2
    echo "$TK13"
    exit 0
fi
if [ "${CUDA13_NO_RUNFILE:-0}" = 1 ]; then     # the caller's disk plan has no room for it
    echo "STOP: no CUDA 13 compiler in the environment nor in $TK13, and CUDA13_NO_RUNFILE=1" >&2
    exit 1
fi
URL=${CUDA13_RUNFILE_URL:-https://developer.download.nvidia.com/compute/cuda/13.2.0/local_installers/cuda_13.2.0_595.45.04_linux.run}
echo "   the CUDA 13 toolkit from $URL into $TK13" >&2
rm -rf "$TK13"
mkdir -p "$(dirname "$TK13")"
links() {
    local name sub
    for name in include:include lib64:lib; do
        sub=${name##*:}; name=${name%%:*}
        [ -e "$TK13/$name" ] || [ ! -d "$TK13/targets/x86_64-linux/$sub" ] || ln -s "targets/x86_64-linux/$sub" "$TK13/$name"
    done
}
RUN=$TMPD/cuda13.run
if curl -fL --retry 3 -o "$RUN" "$URL"; then
    sh "$RUN" --silent --toolkit --installpath="$TK13" --no-opengl-libs --no-man-page --no-drm --override \
        --tmpdir="$TMPD" >&2 || echo "   the installer exited $? (a step it lacks root for); judged by its nvcc" >&2
    links
    if [ ! -x "$TK13/bin/nvcc" ]; then
        X=$TMPD/cuda13_extract
        rm -rf "$X" "$TK13" && mkdir -p "$X" "$TK13"
        echo "   no bin/nvcc: extracting the runfile's files instead" >&2
        sh "$RUN" --silent --extract="$X" --tmpdir="$TMPD" >&2 || echo "   the extraction exited $?" >&2
        for c in "$X"/*/; do
            [ -d "$c" ] && cp -alf "$c". "$TK13"/ 2>/dev/null || cp -af "$c". "$TK13"/ || true
        done
        rm -rf "$X"
        links
    fi
fi
if [ -f /tmp/cuda-installer.log ] && [ -O /tmp/cuda-installer.log ]; then   # it logs there whatever --tmpdir says
    cp /tmp/cuda-installer.log "$KV_HOME/cuda13_installer.log" && rm -f /tmp/cuda-installer.log
fi
rm -rf "$RUN" "$TMPD"/cuda-installer* "$TMPD"/cuda_* 2>/dev/null || true
if ! ok; then
    echo "STOP: the CUDA 13 toolkit did not install (log: $KV_HOME/cuda13_installer.log)" >&2
    rm -rf "$TK13"
    exit 1
fi
rm -rf "$TK13"/nsight-compute* "$TK13"/nsight-systems* "$TK13"/nsightee_plugins "$TK13"/libnvvp \
    "$TK13"/extras/demo_suite "$TK13"/share/doc "$TK13"/gds* "$TK13"/tools
find "$TK13"/ -name 'lib*_static*.a' ! -name 'libcudart_static.a' -delete 2>/dev/null || true
echo "   CUDA 13 toolkit: $TK13 ($("$TK13/bin/nvcc" --version | tail -n 1); $(du -sh "$TK13" | cut -f1))" >&2
echo "$TK13"
