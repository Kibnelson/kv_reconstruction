#!/usr/bin/env bash
# vLLM prefill/decode disaggregation and CPU-tier runs (docs/pd_protocol.md, docs/pd_baselines.md), INSIDE the
# allocation, after scripts/pd_setup.sh: the engines of one layout, the proxy, then scripts/pd_bench.py, packaged and
# pushed to this branch as benchmark_results/<model>_pd_<layout>_<connector>[_<transport>][_eager]_<gpu>_<date>.
#
#   A5000 (4 GPUs):  MODEL=llama32_3b bash scripts/run_pd_vllm.sh                 (1 prefill + 3 decode, host ring)
#                    LAYOUT=2p2d TRANSPORT=p2p bash scripts/run_pd_vllm.sh         (2 + 2, GPU to GPU)
#                    LAYOUT=colo4 STAGES=tier,report bash scripts/run_pd_vllm.sh   (4 engines with the CPU tier)
#                    CONNECTOR=nixl LAYOUT=2p2d bash scripts/run_pd_vllm.sh        (vLLM's NixlConnector)
#                    CONNECTOR=lmcache LAYOUT=colo4 bash scripts/run_pd_vllm.sh    (LMCache's CPU offload)
#   NUC (1 GPU):     ALLOW_NO_SLURM=1 SHARE_GPU=1 LAYOUT=1p1d SMOKE=1 bash scripts/run_pd_vllm.sh
#
# Options (environment):
#   LAYOUT       1p3d (default), 2p2d, 1p1d, colo4 (NUM_COLOCATED engines, default 4, each prefilling its own
#                requests, with a CPU tier); or NUM_PREFILL and NUM_DECODE
#   PREFILL_GPUS, DECODE_GPUS   positions in the job's GPU list (default: prefill first; scripts/node_specs.py says
#                which placement the PCIe layout favours)
#   CONNECTOR    tkv (ours, default), nixl (vLLM's NixlConnector, no codec), lmcache (LMCache's CPU offload)
#   TRANSPORT    ring (pinned host memory, paced links; default), p2p (GPU to GPU, CUDA IPC), tcp
#   ENGINE_EAGER=1   --enforce-eager (T2's layer-0 recompute matches the cache bit for bit in eager mode)
#   TIER_GB (8), TIER_CHUNK (256), GPU_BLOCKS (the engines' GPU KV blocks for the tier runs; default 2048, so the
#   conversations outgrow the GPU), TIER_CODECS, TIER_CLIENTS (8,16), TIER_CONVERSATIONS (64)
#   SHARE_GPU=1 (every engine on GPU 0, small memory shares: the NUC smoke), SMOKE=1 (few codecs, links, rates and
#   prompts), STAGES, CODECS, LINKS, RATES, NUM_PROMPTS, PROMPT_TOKENS (2048,8192), OUTPUT_LEN (128), TTFT_SLO_MS
#   (2000), PORT_BASE (from the Slurm job id: a shared node runs other people's jobs), NO_PUSH=1, KEEP_ENGINES=1
#   (leave the engines running after the benchmark, for manual checks), NIXL_ALL_GPUS=1 (NIXL engines see every
#   job GPU, as the p2p transport needs; default: one GPU each, as vLLM's own NIXL tests run), MODEL_PATH (a local
#   model folder instead of the Hub download)
#   RESULTS_ROOT (where the results folder goes; default benchmark_results in this checkout), PD_CARD=1 (run by
#   scripts/pd_card.sh from a detached node-local checkout: no branch check, no commit; the card publishes)
# On the cluster, scripts/pd_card.sh runs a list of these (scripts/pd_matrix.sh) in its run stage.
set -Eeuo pipefail
trap 'echo "PD RUN STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "$(dirname "$0")/.."
REPO=$PWD
PD_CARD=${PD_CARD:-0}
if [ "$PD_CARD" != 1 ]; then
    BRANCH=${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}
    case "$BRANCH" in pd-vllm-protocol|pd-baselines|pd-*) ;; *) echo "STOP: on branch $BRANCH, not a PD branch"; exit 1 ;; esac
fi
RESULTS_ROOT=${RESULTS_ROOT:-benchmark_results}
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
KV_RUNS=${KV_RUNS:-$KV_HOME/runs}
export HF_HOME=${KV_HF_HOME:-$KV_HOME/hf} TMPDIR=$KV_HOME/tmp
# every cache under KV_HOME, not the home directory (its quota ran out installing vLLM on gpu-grad-01): pip, vLLM's
# compile cache and FlashInfer follow XDG_CACHE_HOME; Triton, TorchInductor and CUDA's JIT cache have their own
export XDG_CACHE_HOME=$KV_HOME/cache TRITON_CACHE_DIR=$KV_HOME/cache/triton
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$KV_HOME/cache/inductor} CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$KV_HOME/cache/nv}
# CUDA numbers the GPUs as nvidia-smi does (by PCI bus), so engine i's index is the card nvidia-smi calls i
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR"
MODEL=${MODEL:-llama32_3b}
case "$MODEL" in
    llama32_3b) MODEL_ID=${MODEL_ID:-unsloth/Llama-3.2-3B}; REVISION=${REVISION:-d4446454d87d51aa42e1fb174f25acc5f8762331} ;;
    *) MODEL_ID=${MODEL_ID:?set MODEL_ID (and REVISION) for $MODEL}; REVISION=${REVISION:-} ;;
esac
TAG=$(echo "${MODEL#/}" | tr '/.' '__' | tr 'A-Z' 'a-z')
RUN=$KV_RUNS/$TAG
LAYOUT=${LAYOUT:-1p3d}
CONNECTOR=${CONNECTOR:-tkv}
TRANSPORT=${TRANSPORT:-ring}
case "$LAYOUT" in
    1p3d) NUM_PREFILL=1; NUM_DECODE=3 ;;
    2p2d) NUM_PREFILL=2; NUM_DECODE=2 ;;
    1p1d) NUM_PREFILL=1; NUM_DECODE=1 ;;
    colo*) NUM_PREFILL=0; NUM_DECODE=${NUM_COLOCATED:-4} ;;
    *) NUM_PREFILL=${NUM_PREFILL:?LAYOUT $LAYOUT: set NUM_PREFILL and NUM_DECODE}; NUM_DECODE=${NUM_DECODE:?} ;;
esac
COLO=0; [ "$NUM_PREFILL" = 0 ] && COLO=1
case "$CONNECTOR" in tkv|nixl|lmcache) ;; *) echo "STOP: CONNECTOR is tkv, nixl or lmcache"; exit 1 ;; esac
[ "$CONNECTOR" = lmcache ] && [ "$COLO" != 1 ] && { echo "STOP: the LMCache reference runs colocated (LAYOUT=colo4)"; exit 1; }
[ "$CONNECTOR" = nixl ] && [ "$COLO" = 1 ] && { echo "STOP: NIXL needs prefill and decode engines"; exit 1; }
# ports from the job id: PORT_BASE .. PORT_BASE+499 are used, and other jobs on the node may run engines too
PORT_BASE=${PORT_BASE:-$((20000 + ${SLURM_JOB_ID:-0} % 80 * 500))}
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_NO_SLURM:-0}" != 1 ]; then
    echo "STOP: not inside a Slurm allocation (or set ALLOW_NO_SLURM=1 on the NUC)"; exit 1
fi
if [ "${PD_MATRIX:-0}" != 1 ] && [ "$PD_CARD" != 1 ]; then   # scripts/pd_matrix.sh checks once and pushes at the end
    git fetch -q origin "$BRANCH"
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || {
        echo "STOP: not GitHub's $BRANCH; run: git pull --ff-only origin $BRANCH"; exit 1; }
fi
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
ENV_DIR=$KV_HOME/venv_vllm
[ "$CONNECTOR" = lmcache ] && ENV_DIR=$KV_HOME/venv_lmcache
VPY=$ENV_DIR/bin/python
[ -x "$VPY" ] || { echo "STOP: no $ENV_DIR: run scripts/pd_setup.sh first"; exit 1; }
[ "$CONNECTOR" != tkv ] || [ -s "$RUN/pd_codecs.pt" ] || { echo "STOP: no $RUN/pd_codecs.pt: run scripts/pd_setup.sh first"; exit 1; }
if [ "$CONNECTOR" = nixl ] && ! "$VPY" -c "import torch; from nixl._api import nixl_agent" 2>/dev/null; then
    echo "STOP: NIXL does not import in $ENV_DIR (scripts/pd_setup.sh):"
    "$VPY" -c "import torch; from nixl._api import nixl_agent" 2>&1 | tail -n 3 || true
    exit 1
fi
# the GPUs this job may use, in Slurm's order: CUDA_VISIBLE_DEVICES as Slurm set it (0..N-1 where the node confines
# jobs to their devices, as gpu-grad-01 does; the physical ids elsewhere), else every GPU nvidia-smi lists. Engine i
# gets the i-th of them, never a bare index that could be another job's card.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -r -a DEVS <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t DEVS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
# a GPU named by UUID (some Slurm setups do): its index, since vLLM reads CUDA_VISIBLE_DEVICES as numbers
for i in "${!DEVS[@]}"; do
    d=$(echo "${DEVS[$i]}" | tr -d ' ')
    if ! [[ "$d" =~ ^[0-9]+$ ]]; then
        idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | awk -F, -v u="$d" '$2 == u {print $1}')
        [[ "$idx" =~ ^[0-9]+$ ]] || { echo "STOP: GPU $d of CUDA_VISIBLE_DEVICES is not one nvidia-smi lists"; exit 1; }
        d=$idx
    fi
    DEVS[$i]=$d
done
NGPU=${#DEVS[@]}
NENG=$((NUM_PREFILL + NUM_DECODE))
seq_list() { local a=$1 n=$2 out=(); for ((i = 0; i < n; i++)); do out+=($((a + i))); done; (IFS=,; echo "${out[*]}"); }
IFS=, read -r -a PPOS <<< "${PREFILL_GPUS:-$(seq_list 0 "$NUM_PREFILL")}"
IFS=, read -r -a DPOS <<< "${DECODE_GPUS:-$(seq_list "$NUM_PREFILL" "$NUM_DECODE")}"
[ "$NUM_PREFILL" = 0 ] && PPOS=()
if [ "${SHARE_GPU:-0}" = 1 ]; then
    PPOS=(); for ((i = 0; i < NUM_PREFILL; i++)); do PPOS+=(0); done
    DPOS=(); for ((i = 0; i < NUM_DECODE; i++)); do DPOS+=(0); done
elif [ "$NGPU" -lt "$NENG" ]; then
    echo "STOP: $NGPU GPU(s) visible; $NUM_PREFILL prefill + $NUM_DECODE decode engines need $NENG (or SHARE_GPU=1)"
    exit 1
fi
EAGER=${ENGINE_EAGER:-0}
RUN_TAG=${LAYOUT}_${CONNECTOR}
if [ "$CONNECTOR" = tkv ] && [ "$COLO" = 0 ]; then RUN_TAG=${RUN_TAG}_$TRANSPORT; fi
if [ "$EAGER" = 1 ]; then RUN_TAG=${RUN_TAG}_eager; fi
STAMP=$(date +%Y%m%d_%H%M)
OUT=$RUN/pd/${STAMP}_$RUN_TAG
LOGS=$OUT/connector
mkdir -p "$OUT" "$LOGS"
echo "== PD run $OUT: $LAYOUT ($NUM_PREFILL prefill on GPUs ${PPOS[*]:-none}, $NUM_DECODE $([ "$COLO" = 1 ] && echo colocated || echo decode) on ${DPOS[*]}), connector $CONNECTOR, transport $TRANSPORT, eager $EAGER; job GPUs ${DEVS[*]}; commit $(git log --oneline -1)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv,noheader | tee "$OUT/nvidia_smi.txt"
# each job GPU's UUID as nvidia-smi and as torch (in the engines' setting) see it: engine i is on the card intended
"$VPY" - "$OUT/gpu_map.json" "${DEVS[@]}" <<'EOF'
import json, os, subprocess, sys
out, devs = sys.argv[1], sys.argv[2:]
rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True,
                      text=True).stdout
smi = {r.split(",")[0].strip(): r.split(",")[1].strip() for r in rows.splitlines() if r.count(",") == 1}
code = "import torch; print(torch.cuda.get_device_properties(0).uuid)"
hexd = lambda u: "".join(c for c in (u or "").lower().replace("gpu-", "") if c in "0123456789abcdef")
got, bad = {}, []
for d in devs:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=d, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    t = r.stdout.strip() if r.returncode == 0 else None
    got[d] = {"nvidia_smi": smi.get(d), "torch": t}
    # compared only when both are 32 hex digits (torch prints the UUID without nvidia-smi's GPU- prefix)
    if len(hexd(t)) == 32 and len(hexd(smi.get(d))) == 32 and hexd(t) != hexd(smi[d]):
        bad.append(d)
json.dump(got, open(out, "w"), indent=1)
if bad:
    sys.exit(f"STOP: torch and nvidia-smi disagree on the UUID of GPU(s) {bad}: {got}")
print("   GPUs " + ", ".join(f"{d}={v['nvidia_smi']}" + ("" if v["torch"] else " (torch check skipped)")
                            for d, v in got.items()))
EOF

REVPY=None; [ -n "$REVISION" ] && REVPY="'$REVISION'"
MP=${MODEL_PATH:-$("$VPY" -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_ID', revision=$REVPY))")}
echo "   model folder $MP"

# ---------------------------------------------------------------------------------------------- what to run
# a run that hangs (a transfer that never arrives) ends here instead of taking the allocation: seconds for the bench
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-$([ "${SMOKE:-0}" = 1 ] && echo 1200 || echo 10800)}
if [ "${SMOKE:-0}" = 1 ]; then
    CODECS=${CODECS:-raw,splitzip,ours-hbh,t2-hbh}
    LINKS=${LINKS:-$([ "$TRANSPORT" = p2p ] && echo native || echo 10)}
    RATES=${RATES:-inf}
    NUM_PROMPTS=${NUM_PROMPTS:-8}
    PROMPT_TOKENS=${PROMPT_TOKENS:-2048}
    TIER_CODECS=${TIER_CODECS:-raw,ours-hbh}
    TIER_CLIENTS=${TIER_CLIENTS:-4}
    TIER_CONVERSATIONS=${TIER_CONVERSATIONS:-8}
    TIER_MAX_TURNS=${TIER_MAX_TURNS:-4}
fi
case "$CONNECTOR" in
    nixl) CODECS=nixl; DEFAULT_STAGES=check,sweep,report; LINKS=native ;;
    lmcache) TIER_CODECS=lmcache; DEFAULT_STAGES=tier,report ;;
    *) if [ "$COLO" = 1 ]; then DEFAULT_STAGES=tier,report; else DEFAULT_STAGES=check,sweep,report; fi ;;
esac
CODECS=${CODECS:-raw,splitzip,exans,adacode,exactkv,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-dense,t2-rank256}
LINKS=${LINKS:-$([ "$TRANSPORT" = p2p ] && echo native || echo 3,10,25,100)}
RATES=${RATES:-0.5,1,2,inf}
NUM_PROMPTS=${NUM_PROMPTS:-48}
PROMPT_TOKENS=${PROMPT_TOKENS:-2048,8192}
OUTPUT_LEN=${OUTPUT_LEN:-128}
STAGES=${STAGES:-$DEFAULT_STAGES}
TIER_CODECS=${TIER_CODECS:-raw,splitzip,exans,adacode,exactkv,ibp,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-rank256}
TIER_CLIENTS=${TIER_CLIENTS:-8,16}
TIER_CONVERSATIONS=${TIER_CONVERSATIONS:-64}
TIER_MAX_TURNS=${TIER_MAX_TURNS:-12}
TIER_GB=${TIER_GB:-8}
TIER_CHUNK=${TIER_CHUNK:-256}
LOAD_CODECS=$([ "$COLO" = 1 ] && echo "$TIER_CODECS" || echo "$CODECS")
LOAD_CODECS=$(echo "$LOAD_CODECS" | tr ',' '\n' | { grep -v -x -e nixl -e lmcache -e '' || true; } | paste -sd, -)

# memory: the engines take a share of each GPU; the rest holds our transfers (gather, coding, received blobs) and
# the codecs' constants (T2 keeps every constant set on the GPU: about 1.3 GB)
if [ "${SHARE_GPU:-0}" = 1 ]; then
    P_MEM=${P_MEM:-0.44}; D_MEM=${D_MEM:-0.44}; MAXLEN=${MAXLEN:-4096}; MAXBT=${MAXBT:-2048}
    SEND_BUDGET=$((256 << 20)); RECV_BUDGET=$((256 << 20)); ENGINE_ARGS=(--enforce-eager)
else
    P_MEM=${P_MEM:-0.72}; D_MEM=${D_MEM:-0.72}; MAXLEN=${MAXLEN:-16384}; MAXBT=${MAXBT:-16384}
    SEND_BUDGET=$((1 << 30)); RECV_BUDGET=$((1536 << 20)); ENGINE_ARGS=()
    [ "$EAGER" = 1 ] && ENGINE_ARGS+=(--enforce-eager)
fi
[ "$COLO" = 1 ] && ENGINE_ARGS+=(--num-gpu-blocks-override "${GPU_BLOCKS:-2048}" --distributed-executor-backend uni)

CHECK_LENGTHS=${CHECK_LENGTHS:-1000,3000,5000,8192}
[ "${SHARE_GPU:-0}" = 1 ] && CHECK_LENGTHS=${CHECK_LENGTHS_SHARED:-500,1000,1777,3000}
PROMPTS=$RUN/pd/prompts_${PROMPT_TOKENS//,/_}_check_${CHECK_LENGTHS//,/_}
if [ ! -s "$PROMPTS/check.jsonl" ]; then
    "$VPY" scripts/pd_prompts.py --texts "$KV_RUNS/dataset_texts" --model "$MP" --tokens "$PROMPT_TOKENS" \
        --per-length "$NUM_PROMPTS" --output-len "$OUTPUT_LEN" --check-lengths "$CHECK_LENGTHS" --out "$PROMPTS"
fi
LINKFILE=$OUT/link.json
echo '{"gbps": 100}' > "$LINKFILE"
TIERFILE=$OUT/tier_control.json
echo '{"generation": 0, "codec": "raw"}' > "$TIERFILE"

# ports: proxy PORT_BASE; prefill i +100+i; engine i (decode or colocated) +200+i; its KV listener +300+i; NIXL
# side channels +400+i (prefill) and +450+i (decode)
PORTS=("$PORT_BASE")
for ((i = 1; i <= NUM_PREFILL; i++)); do PORTS+=($((PORT_BASE + 100 + i)) $((PORT_BASE + 400 + i))); done
for ((i = 1; i <= NUM_DECODE; i++)); do PORTS+=($((PORT_BASE + 200 + i)) $((PORT_BASE + 300 + i)) $((PORT_BASE + 450 + i))); done
"$VPY" - "${PORTS[@]}" <<'EOF'
import socket, sys
busy = []
for port in map(int, sys.argv[1:]):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # as the servers bind: closed connections of the
    try:                                                         # previous run (TIME_WAIT) do not hold the port
        s.bind(("127.0.0.1", port))
    except OSError:
        busy.append(port)
    finally:
        s.close()
if busy:
    sys.exit(f"STOP: port(s) {busy} already in use on this node (another job?): start again with PORT_BASE=<another>")
print(f"   ports {min(map(int, sys.argv[1:]))}-{max(map(int, sys.argv[1:]))} free")
EOF

PIDS=()
cleanup() {
    if [ "${KEEP_ENGINES:-0}" != 1 ]; then
        for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
        sleep 3
        for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null || true; done
    fi
}
trap cleanup EXIT

visible() {   # the CUDA_VISIBLE_DEVICES of an engine on job GPU position $1
    local own=${DEVS[$1]} all=0
    { [ "$CONNECTOR" = tkv ] && [ "$TRANSPORT" = p2p ]; } && all=1
    { [ "$CONNECTOR" = nixl ] && [ "${NIXL_ALL_GPUS:-0}" = 1 ]; } && all=1
    if [ "$all" = 1 ] && [ "${SHARE_GPU:-0}" != 1 ]; then
        local out=("$own")
        for d in "${DEVS[@]}"; do [ "$d" != "$own" ] && out+=("$d"); done
        (IFS=,; echo "${out[*]}")
    else
        echo "$own"
    fi
}

serve() {   # name position port mem kv_transfer_config [VAR=value ...]
    local name=$1 pos=$2 port=$3 mem=$4 cfg=$5
    shift 5
    # VLLM_USE_FLASHINFER_SAMPLER=0: compiled engines warm up top-k/top-p sampling, which with FlashInfer compiles
    # its kernels at start (ninja and a CUDA 13 nvcc); there is no such toolkit on gpu-grad-01, and every compiled
    # engine of job 30170 stopped there (FileNotFoundError: 'ninja'). vLLM's PyTorch sampler needs nothing compiled;
    # the sampler is the same for every codec. The environment's bin/ comes first on PATH (its ninja and tools).
    env CUDA_VISIBLE_DEVICES="$(visible "$pos")" PYTHONPATH="$REPO" VLLM_SERVER_DEV_MODE=1 \
        PATH="$ENV_DIR/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}" "$@" \
        nohup "$ENV_DIR/bin/vllm" serve "$MP" --host 127.0.0.1 --port "$port" \
        --served-model-name llama --block-size 16 --gpu-memory-utilization "$mem" --max-model-len "$MAXLEN" \
        --max-num-batched-tokens "$MAXBT" --chat-template "$REPO/scripts/pd_chat_template.jinja" \
        --kv-transfer-config "$cfg" "${ENGINE_ARGS[@]}" > "$OUT/engine_$name.log" 2>&1 &
    PIDS+=($!)
    echo "   $name on GPU ${DEVS[$pos]} (visible $(visible "$pos")), port $port (log $OUT/engine_$name.log)"
}

json_list() { echo "$1" | tr ',' '\n' | { grep -v '^$' || true; } | sed 's/.*/"&"/' | paste -sd, -; }
EXANS_INFO=$RUN/exans/exans_info.json
COMMON="\"codecs_file\":\"$RUN/pd_codecs.pt\",\"codecs\":[$(json_list "$LOAD_CODECS")],\"splitzip_source\":\"$KV_HOME/sources/splitzip\",\"adacode_source\":\"$KV_HOME/sources/adacode\",\"exactkv_source\":\"$KV_HOME/sources/exactkv\",\"log_dir\":\"$LOGS\""
[ -s "$EXANS_INFO" ] && COMMON="$COMMON,\"exans_info\":\"$EXANS_INFO\""
P_URLS=() DECODE_ARGS=()
STAGGER=$([ "${SHARE_GPU:-0}" = 1 ] && echo "${STAGGER_S:-60}" || echo 0)   # engines on one GPU profile in turn
for ((i = 1; i <= NUM_PREFILL; i++)); do
    port=$((PORT_BASE + 100 + i))
    case "$CONNECTOR" in
        tkv) serve "prefill$i" "${PPOS[$((i - 1))]}" "$port" "$P_MEM" \
                 "{\"kv_connector\":\"TKVConnector\",\"kv_connector_module_path\":\"kv_compare.pd.connector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{$COMMON,\"transport\":\"$TRANSPORT\",\"link_control\":\"$LINKFILE\",\"gpu_send_budget\":$SEND_BUDGET}}" ;;
        nixl) serve "prefill$i" "${PPOS[$((i - 1))]}" "$port" "$P_MEM" '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                  VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=$((PORT_BASE + 400 + i)) UCX_NET_DEVICES=all ;;
    esac
    P_URLS+=(--prefill "http://127.0.0.1:$port")
    [ "$STAGGER" != 0 ] && sleep "$STAGGER"
done
for ((i = 1; i <= NUM_DECODE; i++)); do
    port=$((PORT_BASE + 200 + i)); kv=$((PORT_BASE + 300 + i)); pos=${DPOS[$((i - 1))]}
    if [ "$COLO" = 1 ]; then
        case "$CONNECTOR" in
            tkv) serve "engine$i" "$pos" "$port" "$D_MEM" \
                     "{\"kv_connector\":\"TKVConnector\",\"kv_connector_module_path\":\"kv_compare.pd.connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$COMMON,\"tier\":{\"budget_gb\":$TIER_GB,\"chunk\":$TIER_CHUNK,\"control\":\"$TIERFILE\",\"codec\":\"raw\"}}}" ;;
            lmcache) serve "engine$i" "$pos" "$port" "$D_MEM" '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
                         LMCACHE_CHUNK_SIZE="$TIER_CHUNK" LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE="$TIER_GB" ;;
        esac
    else
        case "$CONNECTOR" in
            tkv) serve "decode$i" "$pos" "$port" "$D_MEM" \
                     "{\"kv_connector\":\"TKVConnector\",\"kv_connector_module_path\":\"kv_compare.pd.connector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{$COMMON,\"listen\":\"127.0.0.1:$kv\",\"gpu_recv_budget\":$RECV_BUDGET,\"recv_timeout_s\":600}}" ;;
            nixl) serve "decode$i" "$pos" "$port" "$D_MEM" '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
                      VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=$((PORT_BASE + 450 + i)) UCX_NET_DEVICES=all ;;
        esac
    fi
    DECODE_ARGS+=(--decode "http://127.0.0.1:$port=127.0.0.1:$kv")
    [ "$STAGGER" != 0 ] && sleep "$STAGGER"
done
echo "== waiting for the engines"
ENGINE_PORTS=()
for ((i = 1; i <= NUM_PREFILL; i++)); do ENGINE_PORTS+=($((PORT_BASE + 100 + i))); done
for ((i = 1; i <= NUM_DECODE; i++)); do ENGINE_PORTS+=($((PORT_BASE + 200 + i))); done
root_cause() {   # the first error of an engine log: vLLM's last traceback only says "See root cause above"
    local f=$1 first
    echo "---- $f"
    grep -n -E "^(ERROR|CRITICAL)|Error:|Exception:|Fatal Python error|Segmentation fault|out of memory" "$f" | head -n 12 || true
    first=$(grep -n -m 1 "Traceback (most recent call last)" "$f" | cut -d: -f1 || true)
    if [ -n "$first" ]; then echo "  (first traceback, line $first)"; sed -n "${first},$((first + 45))p" "$f"; fi
}
ENGINES_UP=1
for port in "${ENGINE_PORTS[@]}"; do
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
        dead=0
        for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || dead=1; done
        [ "$dead" = 1 ] && break
        sleep 5
    done
    if ! curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
        ENGINES_UP=0
        echo "STOP: the engines did not all come up (port $port); the first error of each log:"
        for f in "$OUT"/engine_*.log; do root_cause "$f"; done
        break
    fi
done
if [ "$ENGINES_UP" = 1 ]; then
grep -h "TKVConnector worker ready\|GPU KV cache size\|not available\|failed its warm-up" "$OUT"/engine_*.log | sed 's/^/   /' || true
PROXY=http://127.0.0.1:$PORT_BASE
nohup "$VPY" scripts/pd_proxy.py --port "$PORT_BASE" "${P_URLS[@]}" "${DECODE_ARGS[@]}" \
    --log "$OUT/proxy.jsonl" > "$OUT/proxy.log" 2>&1 &
PIDS+=($!)
for _ in $(seq 1 30); do curl -sf "$PROXY/health" >/dev/null && break; sleep 1; done
"$VPY" - "$OUT/env.json" "$MODEL_ID" "$NUM_PREFILL" "$NUM_DECODE" "$LAYOUT" "$CONNECTOR" "$TRANSPORT" "$EAGER" <<'EOF'
import json, subprocess, sys, torch, vllm
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout
extra = {}
for m in ("lmcache", "nixl"):
    try:
        extra[m] = getattr(__import__(m), "__version__", "?")
    except Exception:
        pass
json.dump({"model": sys.argv[2], "prefills": int(sys.argv[3]), "decodes": int(sys.argv[4]), "layout": sys.argv[5],
           "connector": sys.argv[6], "transport": sys.argv[7], "eager": sys.argv[8] == "1", "vllm": vllm.__version__,
           "torch": torch.__version__, "gpus": ", ".join(sorted(set(gpus.split("\n")) - {""})), **extra},
          open(sys.argv[1], "w"))
EOF
set +e
PYTHONPATH=$REPO timeout --signal=TERM --kill-after=60 "$RUN_TIMEOUT_S" \
    "$VPY" -u scripts/pd_bench.py --proxy "$PROXY" --model "$MP" --out "$OUT" --prompts "$PROMPTS" \
    --texts "$KV_RUNS/dataset_texts" --log-dir "$LOGS" --link-control "$LINKFILE" --codecs "$CODECS" --links "$LINKS" \
    --rates "$RATES" --num-prompts "$NUM_PROMPTS" --output-len "$OUTPUT_LEN" --ttft-slo-ms "${TTFT_SLO_MS:-2000}" \
    --stages "$STAGES" --vllm-source "$KV_HOME/sources/vllm" --env "$OUT/env.json" \
    --tier-codecs "$TIER_CODECS" --tier-clients "$TIER_CLIENTS" --tier-conversations "$TIER_CONVERSATIONS" \
    --tier-max-turns "$TIER_MAX_TURNS" --tier-control "$TIERFILE" 2>&1 | tee "$OUT/bench.log"
STATUS=${PIPESTATUS[0]}
set -e
if [ "$STATUS" = 124 ]; then echo "STOP: the bench ran past RUN_TIMEOUT_S=$RUN_TIMEOUT_S s; packaged as _partial"; fi
else
    STATUS=1                       # packaged below all the same: the engine logs are the evidence
fi
cleanup
trap - EXIT

GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
NAME=${TAG}_pd_${RUN_TAG}_${GPU_TAG}x$([ "${SHARE_GPU:-0}" = 1 ] && echo 1shared || echo "$NENG")_${STAMP}
if [ "${SMOKE:-0}" = 1 ]; then NAME=${NAME}_smoke; fi
if [ "$STATUS" != 0 ]; then NAME=${NAME}_partial; fi
DEST=$RESULTS_ROOT/$NAME
mkdir -p "$DEST/runs" "$DEST/connector"
for f in summary.md results.json opportunity.json env.json nvidia_smi.txt gpu_map.json bench.log proxy.log link.json; do
    [ -f "$OUT/$f" ] && cp "$OUT/$f" "$DEST/"
done
cp "$OUT"/runs/*.json "$DEST/runs/" 2>/dev/null || true
for f in "$OUT"/engine_*.log; do      # whole when small (a failed start), else its start and its end
    if [ "$(stat -c %s "$f")" -lt 4000000 ]; then cp "$f" "$DEST/"
    else { head -n 300 "$f"; echo "[... $(wc -l < "$f") lines in all ...]"; tail -n 400 "$f"; } > "$DEST/$(basename "$f")"; fi
done
gzip -c "$OUT/proxy.jsonl" > "$DEST/proxy.jsonl.gz" 2>/dev/null || true
for f in "$LOGS"/*.jsonl; do [ -f "$f" ] && gzip -c "$f" > "$DEST/connector/$(basename "$f").gz"; done
cp "$RUN"/pd_selftest*.json "$DEST/" 2>/dev/null || true
if [ "$PD_CARD" = 1 ]; then echo "RESULTS $DEST"; exit "$STATUS"; fi
git add -f "$DEST"                 # -f: .gitignore drops runs/ folders, and the runs/*.json are the point
git commit -q -m "Add vLLM PD results ($NAME)"
if [ "${NO_PUSH:-0}" = 1 ]; then echo "NO_PUSH=1: committed locally"; exit "$STATUS"; fi
git fetch -q origin "$BRANCH"
git rebase -q "origin/$BRANCH" || { git rebase --abort; echo "STOP: rebase failed; committed locally"; exit 1; }
git push -q origin "HEAD:$BRANCH"
echo "PUSHED $DEST to $BRANCH"
exit "$STATUS"
