#!/usr/bin/env bash
# Every baseline in real vLLM PD (docs/pd_baselines.md): a list of scripts/run_pd_vllm.sh runs, one after another in
# the same allocation (engines started and stopped per run), then one combined report
# (scripts/pd_matrix_report.py), committed and pushed once at the end.
#
#   MATRIX=smoke bash scripts/pd_matrix.sh        (inside the 4-GPU allocation; scripts/pd_matrix.sbatch runs it)
#
# MATRIX picks a list (each fits one 12-hour allocation; times are estimates until the smoke has measured them):
#   smoke       every layout, path and connector once, with few prompts (about 2 hours with model loads)
#   ring1p3d    1 prefill + 3 decode through host memory at 3/10/25/100 Gb/s: the in-process baselines and ours,
#               then nvCOMP on its own engines (a native crash costs that run only)
#   ring2p2d    the same with 2 prefill + 2 decode
#   direct      GPU to GPU (the p2p transport, unpaced) for 1+3 and 2+2, every codec; vLLM's NixlConnector for both
#   tier        4 colocated engines with the CPU tier: every codec, nvCOMP apart, and LMCache's CPU offload
#   t2eager     T and T2 in eager engines (T2's layer-0 recompute is bit-exact in eager mode), host ring and direct
#   links       the link speed from 10 to 400 Gb/s in steps of 25, 16 requests at once, 1 prefill + 3 decode: through
#               host memory and GPU to GPU (paced), the in-process baselines and ours; where each codec stops paying
#   workloads   every codec on the baselines' own published workloads (SplitZip's four sweeps, IBP's, exactkv's) and
#               on ours (rates 0.5-2 per second and all at once), 1 prefill + 3 decode at 10 and 100 Gb/s and GPU to
#               GPU, NIXL and nvCOMP too; AQUA-KV (lossy) beside them for speed; exactkv in runs of its own, last
#               (its install killed every decode engine in job 30334)
#   wlong       the biggest published workloads (32K-token prompts, 16 at once of 8K and 16K, 128 and 256 at once of
#               1,024), host memory at 10 and 100 Gb/s and GPU to GPU: a campaign of their own (exactkv apart, last)
#   wlours      the workloads campaign again for T and T2 after job 30371 (decoder in one chunk, requests over 4K
#               in pieces, one transfer over the GPU budget kept on the GPU), with raw, SplitZip and AdaCode
#               beside them: host memory at 10 and 100 Gb/s and GPU to GPU
#   capacity    how many requests each codec carries: 96 prompts of 2,048 tokens at 1-8 per second, the highest rate
#               where 90% meet a 1 s TTFT target; host memory at 10, 25, 50 and 100 Gb/s and GPU to GPU
#   quick       a check of the speed work before xfer (under an hour with setup, whose profile times T and T2 against
#               their calls as of job 30436): raw, T, T2 and SplitZip on 2K and 16K prompts one at a time and 16 2K
#               prompts at once, host memory at 10 and 100 Gb/s
#   trace       the reuse opportunity on IBM's MTRAG multi-turn RAG conversations (all 110, 8 at a time): raw
#               transfers, prompt layouts docs first / history first / interleaved, routing by conversation and round
#               robin, every page hashed (TRACE_PAGES) and related to earlier KV of its text (how many bits a delta
#               against it costs, per layer); trace_report.md says what was already where and how close
#               (scripts/pd_trace_report.py, docs/pd_trace.md)
#   capture     the raw pages themselves, for offline delta analysis: the trace's run on MTRAG's first 6
#               conversations, docs first and history first, routing by conversation; P stores every block it sends,
#               D its own copy of held blocks that differ (16 GB at most, kept in /tmp/$USER/kvpd-captures/<tag>);
#               then scripts/pd_delta_analysis.py over them (delta_report.md; docs/pd_delta.md)
#   diffgraph   no engines: our KV Diff Graph against EvidenceStore's on the newest kept capture (one GPU, after
#               setup's codec constants): T2 per request / segment / page, deltas by T's coder from the base, their
#               zlib graph, every page decoded and checked, rebuild times (scripts/pd_diffgraph.py)
#   tree        no engines: the conversations' block tree on the newest kept capture, kept by a rule (everything, the
#               newest path per conversation, with or without its documents, hindsight, nothing) and the rest
#               recomputed: hits, prefill left, the store at its peak raw and coded with T2 and checked; T2's padding
#               against none and a tail coded apart (scripts/pd_tree.py)
#   xfer        where the transfer dominates, and where it does not (SplitZip's regime and its caveats): prompts of
#               8K-64K tokens one at a time at 10-100 Gb/s per prefill GPU and GPU to GPU; three prefill GPUs behind
#               one link of 10-100 Gb/s (3.3-33 per GPU) with many requests at once; exactkv apart, last
# or RUNS="VAR=value VAR=value; VAR=value ...": one run per ';' (the variables of scripts/run_pd_vllm.sh).
# Extra variables in the environment (SMOKE, RATES, NUM_PROMPTS, PREFILL_GPUS, ...) apply to every run.
# PRINT_RUNS=1 prints the list and stops. PD_CARD=1 (scripts/pd_card.sh's run stage): no branch check and no commit;
# the run folders and the combined report go to RESULTS_ROOT with matrix.json (the runs and which did not finish).
set -Eeuo pipefail
trap 'echo "PD MATRIX STOPPED at line $LINENO (exit $?)" >&2' ERR
cd "$(dirname "$0")/.."
PD_CARD=${PD_CARD:-0}
RESULTS_ROOT=${RESULTS_ROOT:-benchmark_results}
# ours first, then the other baselines, exactkv last: in job 30334 its install killed every decode engine (an
# illegal memory access), and a run stops at a dead engine, so what comes after a crash is lost
MAIN=raw,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-dense,t2-rank256,splitzip,exans,adacode,exactkv
NV=raw,nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp
TIERMAIN=raw,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-rank256,splitzip,exans,adacode,ibp,exactkv
T2SET=raw,ours-hbh,ours-dense,ours-rank256,t2-hbh,t2-dense,t2-rank256
BASE=raw,ours-hbh,t2-hbh,t2-rank256,splitzip,exans,adacode
LINKSET=$BASE,exactkv
OURSRE=raw,ours-hbh,t2-hbh,t2-rank256,splitzip,adacode   # wlours: ours and the strongest two baselines
CAPSET=raw,ours-hbh,t2-rank256,splitzip,adacode,exans,aquakv-2,exactkv  # capacity (AQUA-KV lossy, for speed)
CAPRUN="LAYOUT=1p3d PROMPT_TOKENS=2048 RATES=1,2,3,4,5,6,8 NUM_PROMPTS=96 TTFT_SLO_MS=1000 STAGES=check,sweep,report RUN_TIMEOUT_S=25200"
XKV=raw,exactkv                           # exactkv in runs of its own (raw beside it: the same engines' reference)
AQUA=aquakv-2,aquakv-3                    # AQUA-KV, lossy: a speed and size reference beside the bit-exact codecs
SPEEDS=10,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400
LINKRUN="RATES=inf NUM_PROMPTS=16 PROMPT_TOKENS=2048 CODECS=$BASE,aquakv-2,exactkv"
# the published workloads, T:B:O:R:family (prompt tokens, requests at once, output tokens, rounds); docs/pd_baselines.md
WL=""
for t in 512 1024 2048 4096 8192; do WL+="$t:1:128:8:sz1,"; done
WL+="16384:1:128:4:sz1,"
for t in 128 256 512 1024 2048 4096; do WL+="$t:16:128:2:sz16,"; done
for b in 1:8 2:4 4:2 8:2 16:2 32:2 64:2; do WL+="1024:${b%%:*}:128:${b##*:}:sz1024,"; done
WL+="1920:20:128:2:ibp,4096:16:256:2:exactkv"
# the big ones in runs of their own, last: a request's KV near what an engine leaves free (32K tokens: 3.7 GB, coded
# beside itself), or more at once than the decode engines hold (they queue); an engine lost there costs nothing else
WLONG="32768:1:128:4:sz1,32768:1:128:4:sz32k,32768:2:128:2:sz32k,32768:4:128:2:sz32k,8192:16:128:2:sz16,"
WLONG+="16384:16:128:2:sz16,1024:128:128:2:sz1024,1024:256:128:2:sz1024"
WLRUN="LAYOUT=1p3d PROMPT_TOKENS=2048 RATES=0.5,1,2,inf NUM_PROMPTS=32 STAGES=check,sweep,shapes,report SHAPES=$WL RUN_TIMEOUT_S=25200"
# xfer: our modes for slow and fast links beside the baselines. 64K-token prompts need more of each GPU for vLLM's
# KV cache (a 64K request is 7.3 GB of KV): 0.68 of the prefill GPUs, 0.66 of the decode ones; requests over 4K go in
# pieces (PIPELINE_TOKENS), so the sender holds one piece's words and coding scratch at a time
XFER=raw,ours-hbh,ours-rank256,t2-hbh,t2-rank256,splitzip,adacode,exans
QUICK="LAYOUT=1p3d STAGES=check,shapes,report SHAPES=2048:1:128:4:sz1,16384:1:128:2:sz1,2048:16:128:1:sz16 RUN_TIMEOUT_S=5400"
# trace: interleaved prompts reach about 17K tokens by a conversation's last turn
TRACERUN="LAYOUT=1p3d STAGES=check,mtrag,report TRACE_PAGES=1 MAXLEN=32768 RUN_TIMEOUT_S=10800"
CAPTURERUN="LAYOUT=1p3d STAGES=check,mtrag,delta,report TRACE_PAGES=1 MAXLEN=32768 RUN_TIMEOUT_S=10800 CAPTURE_GB=16
        MTRAG_LIMIT=6 MTRAG_CAPTURE_CONVS=6 MTRAG_LAYOUTS=docs_first,history_first MTRAG_POLICIES=conversation"
XLONG="8192:1:128:4:long,16384:1:128:4:long,32768:1:128:2:long,65536:1:128:2:long"
XLRUN="LAYOUT=1p3d STAGES=check,shapes,report SHAPES=$XLONG P_MEM=0.68 D_MEM=0.66 RUN_TIMEOUT_S=16200"
XSHARE="2048:24:128:2:share,8192:12:128:2:share,16384:6:128:2:share"
XSRUN="LAYOUT=3p1d SHARED_LINK=1 PROMPT_TOKENS=2048 RATES=2,4,8,inf NUM_PROMPTS=48 STAGES=check,sweep,shapes,report SHAPES=$XSHARE RUN_TIMEOUT_S=18000"
WLLONG="LAYOUT=1p3d STAGES=check,shapes,report SHAPES=$WLONG RUN_TIMEOUT_S=10800"
MATRIX=${MATRIX:-smoke}
if [ -z "${RUNS:-}" ]; then
    case "$MATRIX" in
        smoke) RUNS="SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,ours-hbh,t2-hbh,splitzip,exans,adacode,aquakv-2,exactkv;
                     SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=$NV;
                     SMOKE=1 LAYOUT=2p2d TRANSPORT=p2p CODECS=raw,ours-hbh,t2-hbh;
                     SMOKE=1 LAYOUT=2p2d CONNECTOR=nixl NIXL_ALL_GPUS=1;
                     SMOKE=1 LAYOUT=colo4 TIER_CODECS=raw,exans,ibp,ours-hbh,t2-hbh,t2-rank256;
                     SMOKE=1 LAYOUT=colo4 CONNECTOR=lmcache;
                     SMOKE=1 LAYOUT=1p3d TRANSPORT=ring ENGINE_EAGER=1 CODECS=raw,t2-hbh,t2-rank256" ;;
        ring1p3d) RUNS="LAYOUT=1p3d TRANSPORT=ring CODECS=$MAIN; LAYOUT=1p3d TRANSPORT=ring CODECS=$NV" ;;
        ring2p2d) RUNS="LAYOUT=2p2d TRANSPORT=ring CODECS=$MAIN; LAYOUT=2p2d TRANSPORT=ring CODECS=$NV" ;;
        direct) RUNS="LAYOUT=1p3d TRANSPORT=p2p CODECS=$MAIN; LAYOUT=1p3d TRANSPORT=p2p CODECS=$NV;
                      LAYOUT=2p2d TRANSPORT=p2p CODECS=$MAIN; LAYOUT=2p2d TRANSPORT=p2p CODECS=$NV;
                      LAYOUT=1p3d CONNECTOR=nixl NIXL_ALL_GPUS=1; LAYOUT=2p2d CONNECTOR=nixl NIXL_ALL_GPUS=1;
                      LAYOUT=2p2d CONNECTOR=nixl" ;;
        tier) RUNS="LAYOUT=colo4 TIER_CODECS=$TIERMAIN; LAYOUT=colo4 TIER_CODECS=$NV; LAYOUT=colo4 CONNECTOR=lmcache" ;;
        t2eager) RUNS="LAYOUT=1p3d TRANSPORT=ring ENGINE_EAGER=1 CODECS=$T2SET LINKS=10,100;
                       LAYOUT=1p3d TRANSPORT=p2p ENGINE_EAGER=1 CODECS=$T2SET;
                       LAYOUT=colo4 ENGINE_EAGER=1 TIER_CODECS=raw,ours-hbh,ours-rank256,t2-hbh,t2-rank256" ;;
        links) RUNS="LAYOUT=1p3d TRANSPORT=ring LINKS=$SPEEDS $LINKRUN;
                     LAYOUT=1p3d TRANSPORT=p2p LINKS=$SPEEDS,native $LINKRUN" ;;
        workloads) RUNS="TRANSPORT=ring LINKS=10,100 CODECS=$BASE,$AQUA $WLRUN;
                         TRANSPORT=p2p LINKS=native CODECS=$BASE,$AQUA $WLRUN;
                         CONNECTOR=nixl NIXL_ALL_GPUS=1 $WLRUN;
                         TRANSPORT=ring LINKS=10,100 CODECS=$NV $WLRUN;
                         TRANSPORT=ring LINKS=10,100 CODECS=$XKV $WLRUN;
                         TRANSPORT=p2p LINKS=native CODECS=$XKV $WLRUN" ;;
        capacity) RUNS="TRANSPORT=ring LINKS=10,25,50,100 CODECS=$CAPSET $CAPRUN;
                        TRANSPORT=p2p LINKS=native CODECS=$CAPSET $CAPRUN" ;;
        wlours) RUNS="TRANSPORT=ring LINKS=10,100 CODECS=$OURSRE $WLRUN;
                      TRANSPORT=p2p LINKS=native CODECS=$OURSRE $WLRUN" ;;
        quick) RUNS="TRANSPORT=ring LINKS=10,100 CODECS=raw,ours-hbh,t2-hbh,t2-rank256,splitzip $QUICK" ;;
        trace) RUNS="TRANSPORT=ring LINKS=100 CODECS=raw $TRACERUN" ;;
        capture) RUNS="TRANSPORT=ring LINKS=100 CODECS=raw $CAPTURERUN" ;;
        diffgraph) RUNS="TRANSPORT=ring LINKS=100 CODECS=raw,t2-hbh,t2-rank256 LAYOUT=1p3d STAGES=diffgraph RUN_TIMEOUT_S=10800" ;;
        tree) RUNS="TRANSPORT=ring LINKS=100 CODECS=raw,t2-hbh,t2-rank256 LAYOUT=1p3d STAGES=tree RUN_TIMEOUT_S=10800" ;;
        xfer) RUNS="TRANSPORT=ring LINKS=10,25,50,100 CODECS=$XFER $XLRUN;
                    TRANSPORT=ring LINKS=10,25,50,100 CODECS=$XFER $XSRUN;
                    TRANSPORT=p2p LINKS=native CODECS=$XFER $XLRUN;
                    TRANSPORT=ring LINKS=10,25 CODECS=$XKV $XLRUN" ;;
        wlong) RUNS="TRANSPORT=ring LINKS=10,100 CODECS=$BASE,aquakv-2 $WLLONG;
                     TRANSPORT=p2p LINKS=native CODECS=$BASE,aquakv-2 $WLLONG;
                     TRANSPORT=ring LINKS=10,100 CODECS=$XKV $WLLONG;
                     TRANSPORT=p2p LINKS=native CODECS=$XKV $WLLONG" ;;
        *) echo "STOP: MATRIX is smoke, ring1p3d, ring2p2d, direct, tier, t2eager, links, workloads, wlong, wlours, capacity, quick, trace, capture, diffgraph, tree or xfer" \
                "(or set RUNS)"
           exit 1 ;;
    esac
fi
IFS=';' read -r -a LIST <<< "$(echo "$RUNS" | tr '\n\t' '  ')"   # read stops at a newline: one line first
for i in "${!LIST[@]}"; do LIST[$i]=$(echo "${LIST[$i]}" | tr -s ' \n\t' ' ' | sed 's/^ //; s/ $//'); done
if [ "${PRINT_RUNS:-0}" = 1 ]; then
    for run in "${LIST[@]}"; do if [ -n "$run" ]; then echo "$run"; fi; done
    exit 0
fi
if [ "$PD_CARD" != 1 ]; then
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git fetch -q origin "$BRANCH"
    [ "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")" ] || {
        echo "STOP: not GitHub's $BRANCH; run: git pull --ff-only origin $BRANCH"; exit 1; }
fi
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "STOP: uncommitted changes in tracked files"; exit 1; }
mkdir -p "$RESULTS_ROOT"
BEFORE=$(ls -1 "$RESULTS_ROOT")
START=$(git rev-parse HEAD)
STAMP=$(date +%Y%m%d_%H%M)
n=0 failed=() done_runs=()
for run in "${LIST[@]}"; do
    [ -n "$run" ] || continue
    n=$((n + 1))
    echo
    echo "==== run $n of ${#LIST[@]}: $run"
    # shellcheck disable=SC2086
    if ! env $run PD_MATRIX=1 NO_PUSH=1 bash scripts/run_pd_vllm.sh; then
        failed+=("$run")
        echo "==== run $n did not finish (its results, if any, are kept as _partial); going on"
    fi
    done_runs+=("$run")
done
if [ "$PD_CARD" = 1 ]; then
    NEW=$(comm -13 <(echo "$BEFORE" | sort) <(ls -1 "$RESULTS_ROOT" | sort) | paste -sd, -)
else
    NEW=$(git diff --name-only "$START" HEAD -- benchmark_results | cut -d/ -f2 | sort -u | paste -sd, -)
fi
GPU_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n 1p | tr 'A-Z' 'a-z' | sed 's/nvidia//' | tr -cd 'a-z0-9')
DEST=$RESULTS_ROOT/pd_matrix_${MATRIX}_${GPU_TAG}_${STAMP}
mkdir -p "$DEST"
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
"$KV_HOME/venv_vllm/bin/python" scripts/pd_matrix_report.py --runs "$NEW" --root "$RESULTS_ROOT" \
    --title "PD matrix $MATRIX on $(hostname -s)" --failed "$(printf '%s;' "${failed[@]}")" --output "$DEST/summary.md" || \
    echo "WARNING: the combined report failed; every run's own summary.md is in its folder"
"$KV_HOME/venv_vllm/bin/python" - "$DEST/matrix.json" "$MATRIX" "$NEW" "${#failed[@]}" "${done_runs[@]}" \
    "${failed[@]}" <<'EOF' || true
import json, sys
out, matrix, new, nfail, rest = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:]
runs, failed = rest[:len(rest) - nfail], rest[len(rest) - nfail:]
json.dump({"matrix": matrix, "runs": runs, "failed": failed, "folders": [f for f in new.split(",") if f]},
          open(out, "w"), indent=1)
EOF
if [ "$PD_CARD" = 1 ]; then
    echo "PD MATRIX $MATRIX: $n runs (${#failed[@]} not finished); results in $RESULTS_ROOT"
    [ "${#failed[@]}" = 0 ] || exit 3
    exit 0
fi
git add "$DEST" 2>/dev/null || true
git commit -q -m "PD matrix $MATRIX: $n runs (${#failed[@]} not finished)" || true
if [ "${NO_PUSH:-0}" = 1 ]; then echo "NO_PUSH=1: committed locally"; exit 0; fi
for i in 1 2 3; do
    git fetch -q origin "$BRANCH"
    if git rebase -q "origin/$BRANCH" && git push -q origin "HEAD:$BRANCH"; then
        echo "PUSHED $DEST (and the $n runs' folders) to $BRANCH"
        exit 0
    fi
    git rebase --abort 2>/dev/null || true
    sleep 10
done
echo "STOP: push failed; everything is committed locally (git push origin HEAD:$BRANCH later)"
exit 1
