#!/usr/bin/env bash
# One LMCache lifecycle run (docs/lmcache_lifecycle.md), INSIDE the 4-GPU allocation, after scripts/lmc_setup.sh:
# ONE LMCache server shared by 1 prefill engine (P) and 3 decode engines (D1-D3), vLLM 0.32.0 with LMCache 0.5.7.post1
# from our copies (third_party/, laid over the release wheels), every process traced by the hooks (kvtrace). MTRAG's
# first conversations go through the proxy in store mode: P prefills each turn (one token) and its connector stores
# the prompt's chunks in the server; then the turn's D engine (by conversation) looks the prompt up and pulls what
# it does not hold from the server. Then scripts/lmc_lifecycle.py follows every block and chunk through that.
#
#   LMC_MODE=memory bash scripts/lmc_run.sh    the chunks live in the server's CPU memory (L1, LRU)
#   LMC_MODE=disk   bash scripts/lmc_run.sh    each chunk is written to disk (L2: files, O_DIRECT) and dropped from
#                                              memory once there (store policy skip_l1), so every pull reads the disk
#   LMC_MODE=tiered bash scripts/lmc_run.sh    written through to disk and kept in a small L1 (LRU): pulls from either
#
# Options (environment): LMC_L1_GB (memory 16, disk 4, tiered 2), LMC_CHUNK (256 tokens), LMC_ODIRECT (1),
# LMC_WORKERS (4), LMC_SAVE_DECODE (0: only prompt chunks are stored), LMC_D_ROLE (kv_consumer: D only pulls;
# kv_both: D stores what it computes too), MTRAG_LIMIT (4 conversations), MTRAG_LAYOUT (interleaved: each turn's
# prompt extends the previous one), MTRAG_CONCURRENCY (4), MAX_TOKENS (128), MAXLEN (32768), MAXBT (8192),
# STORE_WAIT_MS (500: the proxy's wait between P's answer and D's request, for the store to land; 0 shows the race),
# GPU_MEM (0.80), GPU_BLOCKS (empty: vLLM's; a number makes every engine's GPU cache that small, so its own blocks
# are evicted and come back from LMCache), D_PREFIX_CACHE (1; 0: D keeps no prefix cache, every turn pulls all),
# ENGINE_EAGER (0), KEEP_L2 (0: the disk tier's files are listed, then deleted), RUN_TIMEOUT_S (3600), PORT_BASE,
# MODEL_PATH, RESULTS_ROOT (the results folder's parent), PD_CARD=1 (scripts/pd_card.sh: no branch checks).
set -Eeuo pipefail
trap 'echo "LMC RUN STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "$(dirname "$0")/.."
REPO=$PWD
RESULTS_ROOT=${RESULTS_ROOT:-benchmark_results}
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=${KV_HF_HOME:-$KV_HOME/hf} TMPDIR=${TMPDIR:-$KV_HOME/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$KV_HOME/cache} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$KV_HOME/cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KV_HOME/cache/inductor} CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$KV_HOME/cache/nv}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (or set ALLOW_NO_SLURM=1)"; exit 1
fi
if [ "${PD_CARD:-0}" != 1 ] && [ "${PD_MATRIX:-0}" != 1 ]; then
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
fi

MODE=${LMC_MODE:-memory}
case "$MODE" in
    memory) L1_GB=${LMC_L1_GB:-16}; EVICT=LRU;  STORE_POLICY=default ;;
    disk)   L1_GB=${LMC_L1_GB:-4};  EVICT=noop; STORE_POLICY=skip_l1 ;;
    tiered) L1_GB=${LMC_L1_GB:-2};  EVICT=LRU;  STORE_POLICY=default ;;
    *) echo "STOP: LMC_MODE is memory, disk or tiered"; exit 1 ;;
esac
CHUNK=${LMC_CHUNK:-256}
ODIRECT=$([ "${LMC_ODIRECT:-1}" = 1 ] && echo true || echo false)
SAVE_DECODE=$([ "${LMC_SAVE_DECODE:-0}" = 1 ] && echo true || echo false)
D_ROLE=${LMC_D_ROLE:-kv_consumer}
case "$D_ROLE" in kv_consumer|kv_both) ;; *) echo "STOP: LMC_D_ROLE is kv_consumer or kv_both"; exit 1 ;; esac
LIMIT=${MTRAG_LIMIT:-4}
LAYOUT=${MTRAG_LAYOUT:-interleaved}
CONC=${MTRAG_CONCURRENCY:-4}
MAX_TOKENS=${MAX_TOKENS:-128}
MAXLEN=${MAXLEN:-32768}
MAXBT=${MAXBT:-8192}
STORE_WAIT_MS=${STORE_WAIT_MS:-500}
GPU_MEM=${GPU_MEM:-0.80}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-3600}
for v in LIMIT CONC MAX_TOKENS MAXLEN MAXBT CHUNK RUN_TIMEOUT_S; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "STOP: $v=${!v} is not a whole number"; exit 1; }
done
[[ "$L1_GB" =~ ^[0-9]+([.][0-9]+)?$ && "$STORE_WAIT_MS" =~ ^[0-9]+([.][0-9]+)?$ && "$GPU_MEM" =~ ^0?[.][0-9]+$ ]] || {
    echo "STOP: LMC_L1_GB, STORE_WAIT_MS and GPU_MEM are numbers (GPU_MEM a fraction)"; exit 1; }

ENV_DIR=$KV_HOME/venv_lmcache
LPY=$ENV_DIR/bin/python
OVERLAY=$KV_HOME/lmc_overlay
[ -x "$LPY" ] && [ -s "$OVERLAY/OVERLAY.json" ] || { echo "STOP: no $ENV_DIR or $OVERLAY: run scripts/lmc_setup.sh first"; exit 1; }
MTRAG_FILE=${MTRAG_FILE:-$KV_RUNS/mtrag/conversations.json}
[ -s "$MTRAG_FILE" ] || { echo "STOP: no $MTRAG_FILE: run scripts/lmc_setup.sh first"; exit 1; }
# the overlay must be this checkout's hooks (a rerun after a change to them rebuilds it)
python3 -I scripts/lmc_overlay.py check >/dev/null
for f in $("$LPY" -I -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["hooked_files"]))' "$OVERLAY/OVERLAY.json"); do
    copy=${f%%/*}
    cmp -s "third_party/$copy/$f" "$OVERLAY/$f" || { echo "STOP: $OVERLAY/$f is not this checkout's: run scripts/lmc_setup.sh"; exit 1; }
done
for f in kvtrace/*.py; do cmp -s "$f" "$OVERLAY/$f" || { echo "STOP: $OVERLAY/$f is not this checkout's: run scripts/lmc_setup.sh"; exit 1; }; done

# the job's GPUs, in Slurm's order (as scripts/run_pd_vllm.sh): P on the first, D1-D3 on the next three; the LMCache
# server sees all four (it copies to and from each engine's GPU memory over CUDA IPC)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t DEVS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
for i in "${!DEVS[@]}"; do
    d=$(echo "${DEVS[$i]}" | tr -d ' ')
    if ! [[ "$d" =~ ^[0-9]+$ ]]; then
        idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$d" '$2 == u {print $1}')
        [[ "$idx" =~ ^[0-9]+$ ]] || { echo "STOP: GPU $d of CUDA_VISIBLE_DEVICES is not one nvidia-smi lists"; exit 1; }
        d=$idx
    fi
    DEVS[$i]=$d
done
[ "${#DEVS[@]}" -ge 4 ] || { echo "STOP: ${#DEVS[@]} GPU(s) visible; 1 prefill + 3 decode engines need 4"; exit 1; }
DEVS=("${DEVS[@]:0:4}")
ALL_GPUS=$(IFS=,; echo "${DEVS[*]}")

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$KV_RUNS/lmc/${STAMP}_$MODE
TRACE=$OUT/kvtrace
L2DIR=$OUT/l2
mkdir -p "$OUT" "$TRACE" "$OUT/server"
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
RES=$RESULTS_ROOT/llama32_3b_lmc_${MODE}_${GPU_TAG}_${STAMP}
echo "== LMCache lifecycle run $OUT: mode $MODE (L1 $L1_GB GB, eviction $EVICT, store policy $STORE_POLICY), chunk $CHUNK," \
     "P on GPU ${DEVS[0]}, D1-D3 on ${DEVS[*]:1}, server on $ALL_GPUS; MTRAG $LIMIT conversations $LAYOUT; commit $(git log --oneline -1 2>/dev/null || echo '?')"

PORT_BASE=${PORT_BASE:-$((20000 + ${SLURM_JOB_ID:-0} % 80 * 500))}
PROXY_PORT=$PORT_BASE
LMC_PORT=$((PORT_BASE + 50)) LMC_HTTP=$((PORT_BASE + 51)) LMC_PROM=$((PORT_BASE + 52))
P_PORT=$((PORT_BASE + 101))
D_PORTS=($((PORT_BASE + 201)) $((PORT_BASE + 202)) $((PORT_BASE + 203)))
"$LPY" -I - "$PROXY_PORT" "$LMC_PORT" "$LMC_HTTP" "$LMC_PROM" "$P_PORT" "${D_PORTS[@]}" <<'EOF'
import socket, sys, time


def busy():
    out = []
    for port in map(int, sys.argv[1:]):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            out.append(port)
        finally:
            s.close()
    return out


for _ in range(30):
    b = busy()
    if not b:
        break
    time.sleep(2)
if b:
    sys.exit(f"STOP: port(s) {b} already in use on this node (another job?): start again with PORT_BASE=<another>")
print(f"   ports {min(map(int, sys.argv[1:]))}-{max(map(int, sys.argv[1:]))} free")
EOF
MP=${MODEL_PATH:-$("$LPY" -I -c "from huggingface_hub import snapshot_download; print(snapshot_download('unsloth/Llama-3.2-3B', revision='d4446454d87d51aa42e1fb174f25acc5f8762331'))")}
echo "   model folder $MP"

PIDS=()
stop_all() {   # SIGINT first (each process flushes its trace on exit), then what is left
    local p
    for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done
    for _ in $(seq 1 30); do
        local alive=0
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done
        [ "$alive" = 0 ] && break
        sleep 1
    done
    for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
    sleep 3
    for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    PIDS=()
}
trap stop_all EXIT
TRACE_ENV=(PYTHONPATH="$OVERLAY" KVTRACE_DIR="$TRACE" PYTHONHASHSEED=0 PATH="$ENV_DIR/bin:$PATH")

echo "== LMCache server (role LMC): zmq $LMC_PORT, HTTP $LMC_HTTP"
L2ARGS=()
if [ "$MODE" != memory ]; then
    mkdir -p "$L2DIR"
    L2ARGS=(--l2-adapter "{\"type\":\"fs\",\"base_path\":\"$L2DIR\",\"use_odirect\":$ODIRECT}" --l2-store-policy "$STORE_POLICY")
fi
# the server and the engines run in the run's folder: nothing there can shadow the overlay on their path
( cd "$OUT" && exec env CUDA_VISIBLE_DEVICES="$ALL_GPUS" "${TRACE_ENV[@]}" KVTRACE_ROLE=LMC \
    nohup "$ENV_DIR/bin/lmcache" server --host 127.0.0.1 --port "$LMC_PORT" --chunk-size "$CHUNK" \
    --http-host 127.0.0.1 --http-port "$LMC_HTTP" --prometheus-port "$LMC_PROM" --l1-size-gb "$L1_GB" \
    --l1-init-size-gb "$L1_GB" --eviction-policy "$EVICT" --max-workers "${LMC_WORKERS:-4}" "${L2ARGS[@]}" \
    > "$OUT/lmcache_server.log" 2>&1 ) &
PIDS+=($!)
for _ in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$LMC_HTTP/healthcheck" >/dev/null && break
    kill -0 "${PIDS[0]}" 2>/dev/null || break
    sleep 2
done
curl -sf "http://127.0.0.1:$LMC_HTTP/healthcheck" >/dev/null || {
    echo "STOP: the LMCache server did not come up:"; tail -n 60 "$OUT/lmcache_server.log"; exit 1; }
curl -sf "http://127.0.0.1:$LMC_HTTP/config" > "$OUT/server/config.json" || true

ENGINE_ARGS=()
[ "${ENGINE_EAGER:-0}" = 1 ] && ENGINE_ARGS+=(--enforce-eager)
[ -n "${GPU_BLOCKS:-}" ] && ENGINE_ARGS+=(--num-gpu-blocks-override "$GPU_BLOCKS")
serve() {   # role gpu port kv_role [extra engine args...]
    local role=$1 gpu=$2 port=$3 kv_role=$4
    shift 4
    local cfg="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"$kv_role\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$LMC_PORT,\"lmcache.mp.save_decode_cache\":$SAVE_DECODE}}"
    # VLLM_USE_FLASHINFER_SAMPLER=0: no sampler kernels compiled at start (scripts/run_pd_vllm.sh)
    ( cd "$OUT" && exec env CUDA_VISIBLE_DEVICES="$gpu" "${TRACE_ENV[@]}" KVTRACE_ROLE="$role" VLLM_SERVER_DEV_MODE=1 \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        nohup "$ENV_DIR/bin/vllm" serve "$MP" --host 127.0.0.1 --port "$port" --served-model-name llama \
        --block-size 16 --gpu-memory-utilization "$GPU_MEM" --max-model-len "$MAXLEN" --max-num-batched-tokens "$MAXBT" \
        --kv-transfer-config "$cfg" "${ENGINE_ARGS[@]}" "$@" > "$OUT/engine_$role.log" 2>&1 ) &
    PIDS+=($!)
    echo "   $role on GPU $gpu, port $port, $kv_role (log $OUT/engine_$role.log)"
}
echo "== engines"
serve P "${DEVS[0]}" "$P_PORT" kv_both
D_EXTRA=()
[ "${D_PREFIX_CACHE:-1}" = 0 ] && D_EXTRA=(--no-enable-prefix-caching)
for i in 1 2 3; do serve "D$i" "${DEVS[$i]}" "${D_PORTS[$((i - 1))]}" "$D_ROLE" "${D_EXTRA[@]}"; done
UP=1
for port in "$P_PORT" "${D_PORTS[@]}"; do
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
        dead=0
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || dead=1; done
        [ "$dead" = 1 ] && break
        sleep 5
    done
    curl -sf "http://127.0.0.1:$port/health" >/dev/null || { UP=0; break; }
done
STATUS=0
SERVER_ALIVE=
if [ "$UP" = 1 ]; then
    grep -h "GPU KV cache size\|Resolved LMCache MP geometry" "$OUT"/engine_*.log | sed 's/^/   /' || true
    echo "== proxy (store mode, wait $STORE_WAIT_MS ms) and MTRAG"
    DARGS=()
    for port in "${D_PORTS[@]}"; do DARGS+=(--decode "http://127.0.0.1:$port"); done
    nohup "$LPY" scripts/pd_proxy.py --port "$PROXY_PORT" --prefill "http://127.0.0.1:$P_PORT" "${DARGS[@]}" \
        --policy conversation --mode store --store-wait-ms "$STORE_WAIT_MS" --log "$OUT/proxy.jsonl" \
        > "$OUT/proxy.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$PROXY_PORT/health" >/dev/null && break; sleep 1; done
    set +e
    timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT_S" "$LPY" -I -u scripts/pd_mtrag.py \
        --conversations "$MTRAG_FILE" --proxy "http://127.0.0.1:$PROXY_PORT" --tokenize "http://127.0.0.1:$P_PORT" \
        --model llama --layout "$LAYOUT" --policy conversation --run "lmc_$MODE" --concurrency "$CONC" \
        --max-tokens "$MAX_TOKENS" --max-len "$MAXLEN" --limit "$LIMIT" --trace \
        --output "$OUT/mtrag.jsonl" --summary "$OUT/mtrag_summary.json" 2>&1 | tee "$OUT/mtrag.log"
    STATUS=${PIPESTATUS[0]}
    set -e
    sleep "${KVTRACE_SETTLE_S:-3}"                # the last stores land and every process flushes its trace
    # the server lasted the run? (job 30694: it crashed a minute in, and the engines went on without it)
    SERVER_ALIVE=1
    { kill -0 "${PIDS[0]}" 2>/dev/null && curl -sf "http://127.0.0.1:$LMC_HTTP/healthcheck" >/dev/null; } || SERVER_ALIVE=0
    if [ "$SERVER_ALIVE" = 0 ]; then
        echo "STOP: the LMCache server is gone; from its log and the engines':"
        grep -n -m 3 "Fatal Python error\|Segmentation fault\|Traceback" "$OUT/lmcache_server.log" || true
        grep -h "degraded mode" "$OUT"/engine_*.log | head -n 4 || true
        STATUS=1
    fi
    for q in status config "cache/objects?tier=l1&page_size=1000" "cache/objects?tier=l2&page_size=1000" metrics; do
        name=$(echo "$q" | sed 's/[?&=]/_/g; s#/#_#g')
        curl -sf "http://127.0.0.1:$LMC_HTTP/$q" > "$OUT/server/$name.txt" 2>/dev/null || rm -f "$OUT/server/$name.txt"
    done
else
    STATUS=1
    echo "STOP: the engines did not all come up; the first error of each log:"
    for f in "$OUT"/engine_*.log; do
        echo "---- $f"; grep -n -E "^(ERROR|CRITICAL)|Error:|Exception:|out of memory" "$f" | head -n 12 || true
    done
fi
stop_all
trap - EXIT

cp "$KV_HOME/lmc_verify.json" "$OUT/server/lmc_verify.json" 2>/dev/null || true
"$LPY" -I - "$OUT/env.json" "$MODE" "$L1_GB" "$EVICT" "$STORE_POLICY" "$CHUNK" "$ODIRECT" "$SAVE_DECODE" "$D_ROLE" \
    "$LIMIT" "$LAYOUT" "$CONC" "$MAX_TOKENS" "$STORE_WAIT_MS" "$GPU_MEM" "${GPU_BLOCKS:-}" "${D_PREFIX_CACHE:-1}" \
    "$OVERLAY/OVERLAY.json" "$ALL_GPUS" "$(git rev-parse HEAD 2>/dev/null || echo '?')" "$SERVER_ALIVE" <<'EOF'
import json, subprocess, sys
a = sys.argv[1:]
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
json.dump({"mode": a[1], "l1_gb": float(a[2]), "eviction": a[3], "store_policy": a[4], "chunk": int(a[5]),
           "odirect": a[6] == "true", "save_decode": a[7] == "true", "d_role": a[8], "mtrag_limit": int(a[9]),
           "layout": a[10], "concurrency": int(a[11]), "max_tokens": int(a[12]), "store_wait_ms": float(a[13]),
           "gpu_mem": float(a[14]), "gpu_blocks": a[15] or None, "d_prefix_cache": a[16] == "1",
           "overlay": json.load(open(a[17])), "gpus": a[18], "gpu_names": sorted(set(gpus.split("\n")) - {""}),
           "commit": a[19], "vllm": "0.32.0", "lmcache": "0.5.7.post1", "engines": {"P": "kv_both", "D": a[8]},
           "server_alive_at_end": {"1": True, "0": False}.get(a[20])},
          open(a[0], "w"), indent=1)
EOF
if [ "$MODE" != memory ] && [ -d "$L2DIR" ]; then
    find "$L2DIR" -type f -printf '%s\t%P\n' | sort -k2 > "$OUT/server/l2_files.tsv" || true
    echo "   disk tier: $(wc -l < "$OUT/server/l2_files.tsv") files, $(du -sh "$L2DIR" | cut -f1)"
    [ "${KEEP_L2:-0}" = 1 ] || rm -rf -- "$L2DIR"
fi

echo "== lifecycle report"
mkdir -p "$RES"
if ! "$LPY" -I scripts/lmc_lifecycle.py report --run "$OUT" --output "$RES"; then
    echo "WARNING: the lifecycle report failed (the raw traces are packaged all the same)"
    [ "$STATUS" != 0 ] || STATUS=2
fi
# the evidence beside the report: the raw traces, the run's records, the logs' tails
tar -C "$OUT" -czf "$RES/kvtrace.tar.gz" kvtrace
for f in proxy.jsonl mtrag.jsonl; do [ -s "$OUT/$f" ] && gzip -c "$OUT/$f" > "$RES/$f.gz"; done
cp "$OUT/env.json" "$RES/"
[ -s "$OUT/mtrag_summary.json" ] && cp "$OUT/mtrag_summary.json" "$RES/"
mkdir -p "$RES/server" "$RES/logs"
cp "$OUT"/server/* "$RES/server/" 2>/dev/null || true
for f in "$OUT"/engine_*.log "$OUT/lmcache_server.log" "$OUT/proxy.log" "$OUT/mtrag.log"; do
    [ -f "$f" ] || continue
    tail -n 400 "$f" > "$RES/logs/$(basename "$f" .log)_tail.log"
    gzip -c "$f" > "$RES/logs/$(basename "$f").gz"        # whole (job 30694: the crash was above the tail)
done
du -sh "$RES" | sed 's/^/   results /'
[ "$STATUS" = 0 ] || { echo "LMC RUN $MODE: not finished (status $STATUS); what there is is in $RES"; exit 1; }
echo "LMC RUN $MODE DONE: $RES"
