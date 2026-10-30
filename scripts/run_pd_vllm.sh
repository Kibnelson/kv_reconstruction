#!/usr/bin/env bash
# vLLM prefill/decode disaggregation with our transfer protocol (docs/pd_protocol.md), INSIDE the allocation, after
# scripts/pd_setup.sh: one prefill engine (GPU 0) and NUM_DECODE decode engines (GPUs 1..N), the push proxy, then
# scripts/pd_bench.py (correctness, the link x codec x rate sweep, the colocated baseline, the multi-turn reuse
# log), packaged and pushed to this branch as benchmark_results/<model>_pdvllm_<gpu>_<date>.
#
#   A5000 (4 GPUs):  MODEL=llama32_3b bash scripts/run_pd_vllm.sh
#   NUC (1 GPU):     ALLOW_NO_SLURM=1 SHARE_GPU=1 NUM_DECODE=1 SMOKE=1 bash scripts/run_pd_vllm.sh
#
# Options (environment): NUM_DECODE (3), SHARE_GPU=1 (every engine on GPU 0, small memory shares: the NUC smoke),
# SMOKE=1 (few codecs, links, rates and prompts), STAGES (pd_bench stages), CODECS, LINKS, RATES, NUM_PROMPTS,
# PROMPT_TOKENS (2048,8192), OUTPUT_LEN (128), TTFT_SLO_MS (2000), PORT_BASE (18000), NO_PUSH=1, KEEP_ENGINES=1
# (leave the engines running after the benchmark, for manual checks).
set -Eeuo pipefail
trap 'echo "PD RUN STOPPED at line $LINENO (exit $?)" >&2' ERR
BRANCH=pd-vllm-protocol
cd "$(dirname "$0")/.."
REPO=$PWD
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=$KV_HOME/hf TMPDIR=$KV_HOME/tmp
MODEL=${MODEL:-llama32_3b}
case "$MODEL" in
    llama32_3b) MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-3B}; REVISION=${REVISION:-d4446454d87d51aa42e1fb174f25acc5f8762331} ;;
    *) MODEL_ID=${MODEL_ID:?set MODEL_ID (and REVISION) for $MODEL}; REVISION=${REVISION:-} ;;
esac
TAG=$(echo "${MODEL#/}" | tr '/.' '__' | tr 'A-Z' 'a-z')
RUN=$KV_RUNS/$TAG
NUM_DECODE=${NUM_DECODE:-3}
PORT_BASE=${PORT_BASE:-18000}
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (or set ALLOW_NO_SLURM=1 on the NUC)"; exit 1
fi
[ "$(git rev-parse --abbrev-ref HEAD)" = "$BRANCH" ] || { echo "STOP: not on $BRANCH"; exit 1; }
git fetch -q origin "$BRANCH"
[ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || {
    echo "STOP: not GitHub's $BRANCH; run: git pull --ff-only origin $BRANCH"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
VPY=$KV_HOME/venv_vllm/bin/python
[ -x "$VPY" ] || { echo "STOP: no $KV_HOME/venv_vllm: run scripts/pd_setup.sh first"; exit 1; }
[ -s "$RUN/pd_codecs.pt" ] || { echo "STOP: no $RUN/pd_codecs.pt: run scripts/pd_setup.sh first"; exit 1; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "${SHARE_GPU:-0}" != 1 ] && [ "$NGPU" -lt $((NUM_DECODE + 1)) ]; then
    echo "STOP: $NGPU GPU(s) visible; 1 prefill + $NUM_DECODE decode engines need $((NUM_DECODE + 1)) (or SHARE_GPU=1)"
    exit 1
fi
STAMP=$(date +%Y%m%d_%H%M)
OUT=$RUN/pd/$STAMP
LOGS=$OUT/connector
mkdir -p "$OUT" "$LOGS"
echo "== PD run $OUT: 1 prefill + $NUM_DECODE decode engine(s), commit $(git log --oneline -1)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader | tee "$OUT/nvidia_smi.txt"

REVPY=None; [ -n "$REVISION" ] && REVPY="'$REVISION'"
MP=$("$VPY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_ID', revision=$REVPY))")
echo "   model folder $MP"

if [ "${SMOKE:-0}" = 1 ]; then
    CODECS=${CODECS:-raw,splitzip,ours-hbh,ours-rank256}
    LINKS=${LINKS:-10,100}
    RATES=${RATES:-0.5,inf}
    NUM_PROMPTS=${NUM_PROMPTS:-8}
    PROMPT_TOKENS=${PROMPT_TOKENS:-2048}
    STAGES=${STAGES:-check,sweep,report}
fi
CODECS=${CODECS:-raw,splitzip,adacode,ours-hbh,ours-kv,ours-dense,ours-rank256}
LINKS=${LINKS:-3,5,10,25,40,100,200,400}
RATES=${RATES:-0.25,0.5,1,2,inf}
NUM_PROMPTS=${NUM_PROMPTS:-64}
PROMPT_TOKENS=${PROMPT_TOKENS:-2048,8192}
OUTPUT_LEN=${OUTPUT_LEN:-128}
STAGES=${STAGES:-check,sweep,colocated,multiturn,report}

# memory: the engines take a share of each GPU; the rest holds our transfers (gather, coding, received blobs)
# (one shared GPU: two 3B engines = 12.9 GB of weights; eager mode, short contexts, small transfer budgets)
if [ "${SHARE_GPU:-0}" = 1 ]; then
    P_MEM=${P_MEM:-0.44}; D_MEM=${D_MEM:-0.44}; MAXLEN=${MAXLEN:-4096}; MAXBT=${MAXBT:-2048}
    SEND_BUDGET=$((256 << 20)); RECV_BUDGET=$((256 << 20)); ENGINE_ARGS=(--enforce-eager)
else
    P_MEM=${P_MEM:-0.78}; D_MEM=${D_MEM:-0.82}; MAXLEN=${MAXLEN:-16384}; MAXBT=${MAXBT:-16384}
    SEND_BUDGET=$((1 << 30)); RECV_BUDGET=$((1536 << 20)); ENGINE_ARGS=()
fi
CHECK_LENGTHS=${CHECK_LENGTHS:-1000,3000,5000,8192}
[ "${SHARE_GPU:-0}" = 1 ] && CHECK_LENGTHS=${CHECK_LENGTHS_SHARED:-500,1000,1777,3000}
PROMPTS=$RUN/pd/prompts_${PROMPT_TOKENS//,/_}_check_${CHECK_LENGTHS//,/_}
if [ ! -s "$PROMPTS/check.jsonl" ]; then
    "$VPY" scripts/pd_prompts.py --texts "$KV_RUNS/dataset_texts" --model "$MP" --tokens "$PROMPT_TOKENS" \
        --per-length "$NUM_PROMPTS" --output-len "$OUTPUT_LEN" --check-lengths "$CHECK_LENGTHS" --out "$PROMPTS"
fi
LINKFILE=$OUT/link.json
echo '{"gbps": 100}' > "$LINKFILE"
PIDS=()
cleanup() {
    if [ "${KEEP_ENGINES:-0}" != 1 ]; then
        for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
        sleep 3
        for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    fi
}
trap cleanup EXIT

serve() {   # name gpu port mem role extra_json
    local name=$1 gpu=$2 port=$3 mem=$4 role=$5 extra=$6
    local cfg="{\"kv_connector\":\"TKVConnector\",\"kv_connector_module_path\":\"kv_compare.pd.connector\",\"kv_role\":\"$role\",\"kv_connector_extra_config\":$extra}"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO nohup "$KV_HOME/venv_vllm/bin/vllm" serve "$MP" --port "$port" \
        --served-model-name llama --block-size 16 --gpu-memory-utilization "$mem" --max-model-len "$MAXLEN" \
        --max-num-batched-tokens "$MAXBT" --chat-template "$REPO/scripts/pd_chat_template.jinja" \
        --kv-transfer-config "$cfg" "${ENGINE_ARGS[@]}" > "$OUT/engine_$name.log" 2>&1 &
    PIDS+=($!)
    echo "   $name on GPU $gpu, port $port (log $OUT/engine_$name.log)"
}
COMMON="\"codecs_file\":\"$RUN/pd_codecs.pt\",\"splitzip_source\":\"$KV_HOME/sources/splitzip\",\"adacode_source\":\"$KV_HOME/sources/adacode\",\"log_dir\":\"$LOGS\""
P_PORT=$((PORT_BASE + 100))
serve prefill 0 "$P_PORT" "$P_MEM" kv_producer \
    "{$COMMON,\"transport\":\"${TRANSPORT:-ring}\",\"link_control\":\"$LINKFILE\",\"gpu_send_budget\":$SEND_BUDGET}"
[ "${SHARE_GPU:-0}" = 1 ] && sleep "${STAGGER_S:-60}"     # the second engine profiles memory after the first
DECODE_ARGS=()
for i in $(seq 1 "$NUM_DECODE"); do
    gpu=$i; [ "${SHARE_GPU:-0}" = 1 ] && gpu=0
    port=$((PORT_BASE + 200 + i)); kv=$((PORT_BASE + 300 + i))
    serve "decode$i" "$gpu" "$port" "$D_MEM" kv_consumer \
        "{$COMMON,\"listen\":\"127.0.0.1:$kv\",\"gpu_recv_budget\":$RECV_BUDGET,\"recv_timeout_s\":600}"
    DECODE_ARGS+=(--decode "http://127.0.0.1:$port=127.0.0.1:$kv")
    [ "${SHARE_GPU:-0}" = 1 ] && sleep "${STAGGER_S:-60}"
done
echo "== waiting for the engines"
for port in "$P_PORT" $(for i in $(seq 1 "$NUM_DECODE"); do echo $((PORT_BASE + 200 + i)); done); do
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || { echo "STOP: an engine exited; tail of its logs:";
            tail -n 40 "$OUT"/engine_*.log; exit 1; }; done
        sleep 5
    done
    curl -sf "http://127.0.0.1:$port/health" >/dev/null || { echo "STOP: engine on port $port not up in 15 min"; exit 1; }
done
grep -h "TKVConnector worker ready\|GPU KV cache size" "$OUT"/engine_*.log | sed 's/^/   /' || true
PROXY=http://127.0.0.1:$PORT_BASE
nohup "$VPY" scripts/pd_proxy.py --port "$PORT_BASE" --prefill "http://127.0.0.1:$P_PORT" "${DECODE_ARGS[@]}" \
    --log "$OUT/proxy.jsonl" > "$OUT/proxy.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do curl -sf "$PROXY/health" >/dev/null && break; sleep 1; done
"$VPY" - "$OUT/env.json" "$MODEL_ID" "$NUM_DECODE" <<'EOF'
import json, subprocess, sys, torch, vllm
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
json.dump({"model": sys.argv[2], "decodes": int(sys.argv[3]), "vllm": vllm.__version__, "torch": torch.__version__,
           "gpus": ", ".join(sorted(set(gpus.split("\n")) - {""}))}, open(sys.argv[1], "w"))
EOF
set +e
PYTHONPATH=$REPO "$VPY" -u scripts/pd_bench.py --proxy "$PROXY" --model "$MP" --out "$OUT" --prompts "$PROMPTS" \
    --texts "$KV_RUNS/dataset_texts" --log-dir "$LOGS" --link-control "$LINKFILE" --codecs "$CODECS" --links "$LINKS" \
    --rates "$RATES" --num-prompts "$NUM_PROMPTS" --output-len "$OUTPUT_LEN" --ttft-slo-ms "${TTFT_SLO_MS:-2000}" \
    --stages "$STAGES" --vllm-source "$KV_HOME/sources/vllm" --env "$OUT/env.json" 2>&1 | tee "$OUT/bench.log"
STATUS=${PIPESTATUS[0]}
set -e
cleanup
trap - EXIT

GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
NAME=${TAG}_pdvllm_${GPU_TAG}x$((NUM_DECODE + 1))_${STAMP}$([ "${SMOKE:-0}" = 1 ] && echo _smoke)$([ "$STATUS" = 0 ] || echo _partial)
[ "${SHARE_GPU:-0}" = 1 ] && NAME=${NAME/x$((NUM_DECODE + 1))/x1shared}
DEST=benchmark_results/$NAME
mkdir -p "$DEST/runs" "$DEST/connector"
for f in summary.md results.json opportunity.json env.json nvidia_smi.txt bench.log proxy.log link.json; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$DEST/"
done
cp "$OUT"/runs/*.json "$DEST/runs/" 2>/dev/null || true
for f in "$OUT"/engine_*.log; do tail -n 400 "$f" > "$DEST/$(basename "$f")"; done
gzip -c "$OUT/proxy.jsonl" > "$DEST/proxy.jsonl.gz" 2>/dev/null || true
for f in "$LOGS"/*.jsonl; do gzip -c "$f" > "$DEST/connector/$(basename "$f").gz"; done
cp "$RUN/pd_selftest.json" "$DEST/" 2>/dev/null || true
git add "$DEST"
git commit -q -m "Add vLLM PD results with our transfer protocol ($NAME)"
if [ "${NO_PUSH:-0}" = 1 ]; then echo "NO_PUSH=1: committed locally"; exit "$STATUS"; fi
git fetch -q origin "$BRANCH"
git rebase -q "origin/$BRANCH" || { git rebase --abort; echo "STOP: rebase failed; committed locally"; exit 1; }
git push -q origin "HEAD:$BRANCH"
echo "PUSHED $DEST to $BRANCH"
exit "$STATUS"
