#!/usr/bin/env bash
# Run benchmark stages on this branch and push the packaged results to this branch - nowhere else.
#
#   cd ~/kv_llama32 && git checkout llama32-3b-lossy-lossless-compare \
#       && git pull --ff-only origin llama32-3b-lossy-lossless-compare && bash scripts/run_on_nuc.sh
#
# Stops before running anything if this folder is not on the branch, is not up to date with GitHub, has
# uncommitted changes, or the Python environment is not the one of the earlier runs (torch 2.11 / CUDA 12.8).
# Options (environment): STAGES (default gpu_speed,report), RUN (the run folder with the captures),
# VENV (default ~/kv_reconstruction/.venv), EXTRA (more arguments for bench_llama_compare.py).
set -euo pipefail
BRANCH=llama32-3b-lossy-lossless-compare
cd "$(dirname "$0")/.."

cur=$(git rev-parse --abbrev-ref HEAD)
if [ "$cur" != "$BRANCH" ]; then
    echo "STOP: this folder is on branch '$cur', not '$BRANCH'."
    echo "      Use the separate folder:  cd ~/kv_llama32 && git checkout $BRANCH"
    exit 1
fi
git fetch -q origin "$BRANCH"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$BRANCH")" ]; then
    echo "STOP: $(git log --oneline -1) is not GitHub's $BRANCH ($(git log --oneline -1 "origin/$BRANCH"))."
    echo "      Run:  git pull --ff-only origin $BRANCH   and start this script again."
    exit 1
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "STOP: uncommitted changes in tracked files (git status)."
    exit 1
fi

VENV=${VENV:-$HOME/kv_reconstruction/.venv}
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi
python - <<'EOF'
import sys, torch
v = torch.__version__
print(f"python {sys.version.split()[0]}, torch {v}, CUDA {torch.version.cuda}")
if not v.startswith("2.11"):
    sys.exit(f"STOP: torch {v}; the earlier runs used 2.11.0+cu128 (source ~/kv_reconstruction/.venv/bin/activate)")
EOF

RUN=${RUN:-$HOME/kv_reconstruction/runs/llama32_3b_compare}
# LOCK=1: measure everything once on this commit (every stage but the captures, both lossy document sets, the
# authors' baseline checks and timings), for the locked scoreboard. About 1.5 hours.
if [ "${LOCK:-0}" = 1 ]; then
    CHECKS=1
    SPLITZIP_OFFICIAL=1
    STAGES=${STAGES:-lossless,frontier,lossy,gpu_speed,pool,szcodec,exans,aquappl,szprotocol,report}
    EXTRA="--lossy-docs test,eval --lossy-variants base,sinks,both --kappas 0.5,1,2,4 ${EXTRA:-}"
fi
STAGES=${STAGES:-gpu_speed,report}
echo "running on branch $cur, commit $(git log --oneline -1); stages $STAGES; output $RUN"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# CHECKS=1: compare our AQUA-KV and KIVI re-implementations with the authors' own functions, and time their
# decoders with the authors' code, first (the report includes the results). AQUA-KV runs in the environment the earlier official run built (torch 2.5.1 with
# fast-hadamard-transform); KIVI's functions need only torch and Triton (this environment).
if [ "${CHECKS:-0}" = 1 ]; then
    OFF=$HOME/kv_reconstruction/runs/_llama_official
    AQUA_PY=$OFF/envs/aqua/bin/python
    AQUA_SRC=$OFF/sources/aqua
    KIVI_SRC=$OFF/sources/kivi
    if [ ! -d "$KIVI_SRC/.git" ]; then
        git clone -q https://github.com/jy-yuan/KIVI.git "$KIVI_SRC" && git -C "$KIVI_SRC" checkout -q 876b4d2
    fi
    if [ -x "$AQUA_PY" ] && [ -d "$AQUA_SRC" ]; then
        "$AQUA_PY" scripts/check_baselines_official.py aqua --source "$AQUA_SRC" --output "$RUN/check_aqua.json" \
            2>&1 | tail -n 40 | tee -a "$RUN/console.log" || echo "AQUA-KV check failed (see above)"
    else
        echo "AQUA-KV check skipped: no $AQUA_PY or $AQUA_SRC"
    fi
    python scripts/check_baselines_official.py kivi --source "$KIVI_SRC" --output "$RUN/check_kivi.json" \
        2>&1 | tail -n 30 | tee -a "$RUN/console.log" || echo "KIVI check failed (see above)"
    # decode / encode time of one document with the authors' code (for the transfer and pool tables)
    if [ -x "$AQUA_PY" ] && [ -d "$AQUA_SRC" ]; then
        "$AQUA_PY" scripts/time_baseline_decoders.py aqua --source "$AQUA_SRC" --output "$RUN/time_aqua.json" \
            2>&1 | tail -n 10 | tee -a "$RUN/console.log" || echo "AQUA-KV timing failed (see above)"
    fi
    python scripts/time_baseline_decoders.py kivi --source "$KIVI_SRC" --output "$RUN/time_kivi.json" \
        2>&1 | tail -n 10 | tee -a "$RUN/console.log" || echo "KIVI timing failed (see above)"
fi
# stage evalcap: the WikiText-2 raw test split as one text file (joined with blank lines, as AQUA-KV's evaluation
# reads it), made once with the AQUA-KV environment's datasets library (or this one's, if it has it).
if [[ ",$STAGES," == *",evalcap,"* || ",$STAGES," == *",aquappl,"* || ",$STAGES," == *",szprotocol,"* \
      || ",$STAGES," == *",context,"* ]]; then
    WT=$RUN/wikitext2_test.txt
    if [ ! -s "$WT" ] || [ ! -s "$RUN/wikitext2_test_rows.json" ]; then
        PY=python
        [ -x "$HOME/kv_reconstruction/runs/_llama_official/envs/aqua/bin/python" ] && \
            PY=$HOME/kv_reconstruction/runs/_llama_official/envs/aqua/bin/python
        "$PY" - "$WT" <<'EOF'
import sys
from datasets import load_dataset
try:
    d = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
except Exception:
    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
open(sys.argv[1], "w", encoding="utf-8").write("\n\n".join(d["text"]))
import json, os
open(os.path.join(os.path.dirname(sys.argv[1]), "wikitext2_test_rows.json"), "w").write(json.dumps(d["text"]))
print(f"wrote {sys.argv[1]} and the test rows: {len(d)} lines")
EOF
    fi
    EXTRA="--eval-text $WT ${EXTRA:-}"
fi
# AQUA_OFFICIAL=1: AQUA-KV's own scripts, as its README runs them for unsloth/Llama-3.2-3B, in its environment:
# WikiText-2 perplexity without quantization, predictors trained on 256 x 8192 RedPajama tokens (2-bit HIGGS grid),
# then the perplexity with them (prefix 4, recent buffer 128). The predictors are kept for later runs.
if [ "${AQUA_OFFICIAL:-0}" = 1 ]; then
    OFF=$HOME/kv_reconstruction/runs/_llama_official
    AQUA_PY=$OFF/envs/aqua/bin/python
    AQUA_SRC=$OFF/sources/aqua
    D=$RUN/aqua_official
    mkdir -p "$D"
    export HF_DATASETS_TRUST_REMOTE_CODE=1
    # their training loads the slow tokenizer, which needs protobuf (not in their requirements.txt): tokenizer
    # support only, the authors' torch / transformers / datasets versions stay as they are
    "$AQUA_PY" -c "import google.protobuf, sentencepiece" 2>/dev/null \
        || "$AQUA_PY" -m pip install -q protobuf sentencepiece tiktoken blobfile
    COMMON="--model_name unsloth/Llama-3.2-3B --model_seqlen 8192 --edenn_d 2 --edenn_n 16 --hadamard_groupsize 1024"
    # their training keeps every sample's activations in RAM (offloaded): inputs and outputs of a layer (3072
    # channels) and two layers' keys and values (1024 each), BF16, about 168 MB per 8192-token sample (43 GB for
    # the paper's 256). With less free RAM, fewer samples (a multiple of 32, at least 64), recorded in the results.
    NS=${AQUA_NSAMPLES:-256}
    AVAIL_MB=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    FIT=$(( (AVAIL_MB - 8192) * 100 / 115 / 168 / 32 * 32 ))
    if [ -z "${AQUA_NSAMPLES:-}" ] && [ "$FIT" -lt "$NS" ]; then
        NS=$(( FIT < 64 ? 64 : FIT ))
        echo "AQUA-KV official: $AVAIL_MB MB of RAM available, training on $NS calibration samples instead of 256"
    fi
    echo "AQUA-KV official: perplexity without quantization ($D/no_quant.log)"
    (cd "$AQUA_SRC" && "$AQUA_PY" evaluate_perplexity.py $COMMON --no_quant) > "$D/no_quant.log" 2>&1 \
        || echo "AQUA-KV no-quant evaluation failed (see $D/no_quant.log)"
    # the calibration samples, drawn by the authors' sampler (togethercomputer/RedPajama-Data-1T-Sample is gone from
    # the Hub; scripts/aqua_calibration_data.py redirects only the download, to a Parquet mirror with the same rows)
    [ -s "$D/predictors_2bit.pt" ] || rm -f "$D/train_2bit.log" "$D/ppl_2bit.log"     # no stale logs in the results
    if [ ! -s "$D/redpajama_calibration.pt" ]; then
        echo "AQUA-KV official: drawing the RedPajama calibration samples ($D/calibration.log)"
        "$AQUA_PY" scripts/aqua_calibration_data.py --source "$AQUA_SRC" --model unsloth/Llama-3.2-3B \
            --nsamples 256 --output "$D/redpajama_calibration.pt" > "$D/calibration.log" 2>&1 \
            || echo "AQUA-KV calibration data failed (see $D/calibration.log)"
    fi
    if [ ! -s "$D/predictors_2bit.pt" ] && [ -s "$D/redpajama_calibration.pt" ]; then
        echo "AQUA-KV official: training predictors on $NS samples ($D/train_2bit.log; this can take an hour or more)"
        (cd "$AQUA_SRC" && "$AQUA_PY" train_predictors.py $COMMON --predictors_output_path "$D/predictors_2bit.pt" \
            --dataset "$D/redpajama_calibration.pt" --total_nsamples $NS --valid_nsamples 32 \
            --offload_activations) > "$D/train_2bit.log" 2>&1 && echo "$NS" > "$D/train_2bit.nsamples" \
            || echo "AQUA-KV training failed (see $D/train_2bit.log)"
    fi
    if [ -s "$D/predictors_2bit.pt" ]; then
        echo "AQUA-KV official: perplexity with the predictors ($D/ppl_2bit.log)"
        (cd "$AQUA_SRC" && "$AQUA_PY" evaluate_perplexity.py $COMMON --predictors_input_path "$D/predictors_2bit.pt" \
            --prefix_size 4 --recent_buffer_size 128) > "$D/ppl_2bit.log" 2>&1 \
            || echo "AQUA-KV evaluation failed (see $D/ppl_2bit.log)"
    fi
    python - "$D" "$RUN/aqua_official.json" <<'EOF'
import json, re, sys
from pathlib import Path
d, res = Path(sys.argv[1]), {}
for name, label in (("no_quant.log", "no quantization"), ("ppl_2bit.log", "HIGGS 2-bit with trained predictors")):
    p = d / name
    m = re.findall(r"WikiText-2 perplexity: ([0-9.]+)", p.read_text(errors="replace")) if p.is_file() else []
    ns = d / "train_2bit.nsamples"
    if m and "2-bit" in label and ns.is_file():
        label += f" ({ns.read_text().strip()} calibration samples; paper 256)"
    if m:
        res[label] = float(m[-1])
Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
print("AQUA-KV official:", res)
EOF
fi
# SPLITZIP_OFFICIAL=1: SplitZip's own benchmark (bench_codec_throughput.py) unchanged, its paper protocol, with
# Llama-3.2-3B in place of Qwen3-32B: code book from WikiText-2 train KV, KV of WikiText-2 test prompts.
if [ "${SPLITZIP_OFFICIAL:-0}" = 1 ]; then
    OFF=$HOME/kv_reconstruction/runs/_llama_official
    SZ_SRC=$OFF/sources/splitzip
    SZ_PY=python
    [ -x "$OFF/envs/splitzip/bin/python" ] && SZ_PY=$OFF/envs/splitzip/bin/python
    echo "SplitZip official protocol ($RUN/splitzip_protocol_official.log)"
    (cd "$SZ_SRC" && "$SZ_PY" bench_codec_throughput.py --device cuda:0 --model unsloth/Llama-3.2-3B --rows 65536 \
        --hidden-dim 4096 --chunk-size 1024 --warmup 10 --iters 50 --repeats 5 \
        --output "$RUN/splitzip_protocol_official.json") > "$RUN/splitzip_protocol_official.log" 2>&1 \
        || echo "SplitZip's benchmark failed (see $RUN/splitzip_protocol_official.log)"
fi
python -u scripts/bench_llama_compare.py --device cuda --output "$RUN" --stages "$STAGES" ${EXTRA:-} 2>&1 \
    | tee -a "$RUN/console.log"

NAME=llama32_3b_compare_a4500_$(date +%Y%m%d_%H%M)
[ "${LOCK:-0}" = 1 ] && NAME=${NAME}_lock
python scripts/package_llama_compare.py "$RUN" --destination "benchmark_results/$NAME"
git add "benchmark_results/$NAME"
git commit -m "Add Llama-3.2-3B results ($NAME, stages $STAGES)"
# code may have been pushed to the branch during the run: the results commit only adds its own folder, so put it
# on top of the branch before pushing (stops, leaving everything committed locally, if that does not apply cleanly)
git fetch -q origin "$BRANCH"
if ! git rebase -q "origin/$BRANCH"; then
    git rebase --abort
    echo "STOP: the results are committed locally but could not be put on top of origin/$BRANCH; run"
    echo "      git pull --rebase origin $BRANCH && git push origin HEAD:$BRANCH"
    exit 1
fi
git push origin "HEAD:$BRANCH"
echo "pushed benchmark_results/$NAME to $BRANCH"
