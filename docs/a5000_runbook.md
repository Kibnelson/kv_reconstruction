# A5000 runbook: lossless KV transfer, our codec against SplitZip and ExANS on their models and datasets

## Scope

This runbook measures **lossless** KV compression for prefill/decode transfer. It compares our predictive codec
with the authors' own SplitZip and ExANS codecs. Every method is checked bit-exact, on the models and datasets
the baselines report. The runs use **one RTX A5000 (24 GB)**, requested from Slurm. Every command after the
request runs **inside the allocation**. Results are pushed to the branch `llama32-3b-lossy-lossless-compare`
under `benchmark_results/`. Nothing goes anywhere else.

| | SplitZip (paper) | ExANS (blog) | This runbook, one A5000 |
| --- | --- | --- | --- |
| Models | Qwen3-30B-A3B, Qwen3-32B, Qwen3-Next-80B-A3B, Llama-3.1-70B, **Llama-3-8B**, Phi-2 (Table 1); Qwen3-32B, Qwen3-Coder-30B, **Llama-3-8B** (Table 4) | none named ("a real KV snapshot", 256K context) | **Llama-3-8B** (SplitZip's own), **Qwen3-8B** (the Qwen3 family; Qwen3-32B needs 64 GB), Llama-3.2-3B (our runs so far), optionally Qwen3-4B-Instruct-2507 (the Phase 2 model) |
| Calibration | code book from WikiText-2 train | not stated | WikiText-2 train for SplitZip's code book **and** our predictor (`--train-set wikitext`) |
| Evaluation data | WikiText-2 test, HumanEval, GSM8K, MMLU, PTB (Table 4; top-16 coverage only) | not named | the same five; every codec's bits, encode and decode time; SplitZip's coverage, which checks against their Table 4 |
| Context | 512 to 131K tokens, batch 1 and 16 | 256K | 512 to 32,768 tokens, batch 1 and 16 (WikiText-2 test) |
| Hardware | H200, RoCE 4x200G | H100 | RTX A5000, PCIe; the link probe measures this GPU's host legs |

Not covered on one 24 GB GPU: the 30B-80B models, Phi-2 (head size 80 with partial RoPE, which our kernels do not
handle yet), contexts beyond 32K (a 131K cache of an 8B model is 16 GiB alone), and GPU-to-GPU pairs (one GPU;
see section 7).

### What one run does (`scripts/run_a5000.sh`, per model)

1. Checks that it is inside a Slurm allocation, the checkout is clean and equal to GitHub's branch, and the
   environment exists.
2. Resolves the model to an exact Hugging Face commit and records it (`model_resolved.json`). Llama-3-8B tries
   `meta-llama/Meta-Llama-3-8B` (gated: needs `HF_TOKEN` with access), then the ungated copies
   `NousResearch/Meta-Llama-3-8B` and `unsloth/llama-3-8b`.
3. Downloads the dataset texts once (`scripts/fetch_eval_texts.py`, from the Hub's Parquet files; PTB from the raw
   file the Hub's loader uses) and records their sources and SHA256.
4. Runs the link probe: device to pinned and pageable host memory and back.
5. Runs these stages of `scripts/bench_llama_compare.py`:
   - `calcap`: KV of 7 x 8,192 WikiText-2 train tokens. Six are calibration documents with queries, one is the
     scales document. It also records the model's geometry.
   - `datasets`: for each of the five datasets, 4 windows of 8,192 consecutive tokens, coded as one case with one
     stream per window. Every codec is decoded bit-exact and timed. It also records our exact costs, SplitZip's
     top-8 and top-16 exponent coverage with the WikiText-2 train code book (A->B) and the dataset's own (B->B),
     and the exponent entropy.
   - `context`: WikiText-2 test at 512, 1K, 2K, 4K, 8K, 16K and 32K tokens with batch 1, and at 512, 1K and 2K
     with batch 16.
   - `groupspeed`: launch shapes of our head-group decoders, as CUDA graphs.
   - `report`: writes `summary.md`.
6. Packages the small files (JSON, logs, `summary.md`; no KV) into `benchmark_results/<model>_lossless_a5000_<date>`.
   It commits them and pushes to the branch.

Methods in every case:
- SplitZip, the authors' codec at `42c7431`.
- ExANS, the authors' codec at `de6c38b`, built from source with one build fix.
- Our per-layer rANS codec in four predictor modes: one step (dense), keys then values, 4 head groups, and head by
  head.
- Our codec with the rank-256 product.
- Our prediction in SplitZip's fixed-length format.
- The exact cost of our strong predictor (no GPU decoder yet).

Our decoders are also timed as one CUDA graph replay. SplitZip and ExANS are timed as their authors' code launches
them. Transfer times per link speed are the longer of the wire time and the decode.

Measured on the A4500 for Llama-3.2-3B, each stage takes minutes: `context` about 25 minutes including our exact
costs, `groupspeed` about 3. Our best estimate for a whole model is 1-1.5 hours for 3B and 2-3 hours for 8B, plus
the first download. All three models fit in one 12-hour allocation. If one ends early, rerunning the same block
continues where it stopped.

## 0. Before you start

- Slurm access to the A5000 node with the partition, account and QoS the Phase 2 cards use (`all` / `oana2026` /
  `research12h` on `gpu-grad-01`). If your names differ, change them in section 1.
- Push access to `Kibnelson/kv_reconstruction` from the compute node. Use the same way you push
  `context_pilot_paper` (SSH key or HTTPS token); block 2 picks the URL form from that checkout if it exists.
- Optional: `export HF_TOKEN=...` before the run blocks, for the original `meta-llama/Meta-Llama-3-8B`. Without it,
  the ungated copy is used and recorded.
- Disk under `~/kv_a5000` (or `KV_HOME`):
  - about 30 GB for the environment, the CUDA toolkit and the sources;
  - 6 / 16 / 16 GB for the 3B / Llama-3-8B / Qwen3-8B weights;
  - up to about 60 GB of captured KV per 8B model.

  Use `KV_RUNS=/path/with/space` for the captures, and `CLEAN_CAPTURES=1` to delete each model's captures after
  its results are pushed.
- Block 2 sets nothing persistent. To use another `KV_HOME` or `KV_RUNS`, `export` it inside the allocation before
  every block.

## 1. Request one GPU (the only command on the login host)

```bash
srun --partition=all --account=oana2026 --qos=research12h --nodelist=gpu-grad-01 \
     --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 --job-name=kv-lossless --pty bash -l
```

You are now in a shell on the node; `echo $SLURM_JOB_ID` prints the job. Every block below starts by checking
this, and stops if it is pasted on the login host. A dropped SSH connection ends an interactive allocation. Start
the request from `tmux` or `screen` on the login host if they are available, or use the batch form in section 6.
If the cluster names GPUs differently, use `--gpus=1` instead of `--gres=gpu:1`.

## 2. Inside the allocation: clone or update the repository

```bash
bash <<'A5000'
set -Eeuo pipefail
trap 'printf "A5000 CLONE STOPPED: line=%s exit=%s\n" "$LINENO" "$?" >&2' ERR
test -n "${SLURM_JOB_ID:-}"
KV_HOME=${KV_HOME:-$HOME/kv_a5000}
BRANCH=llama32-3b-lossy-lossless-compare
mkdir -p "$KV_HOME"
cd "$KV_HOME"
URL=https://github.com/Kibnelson/kv_reconstruction.git
if git -C "$HOME/context_pilot_paper" remote get-url origin 2>/dev/null | grep -q '^git@'; then
  URL=git@github.com:Kibnelson/kv_reconstruction.git
fi
if [ ! -d kv_reconstruction/.git ]; then
  git clone -b "$BRANCH" "$URL" kv_reconstruction
fi
cd kv_reconstruction
git checkout -q "$BRANCH"
git pull -q --ff-only origin "$BRANCH"
test -z "$(git status --porcelain --untracked-files=no)"
printf 'A5000 CHECKOUT: %s at %s (%s)\n' "$PWD" "$(git log --oneline -1)" "$(git remote get-url origin)"
A5000
```

Success marker: `A5000 CHECKOUT: ...`.

## 3. Environment, once per `KV_HOME` (a rerun is safe)

```bash
bash <<'A5000'
set -Eeuo pipefail
trap 'printf "A5000 SETUP STOPPED: line=%s exit=%s\n" "$LINENO" "$?" >&2' ERR
test -n "${SLURM_JOB_ID:-}"
cd "${KV_HOME:-$HOME/kv_a5000}/kv_reconstruction"
bash scripts/a5000_setup.sh
A5000
```

`scripts/a5000_setup.sh` does the following:
- Checks the GPU, the network from this node (PyPI, PyTorch, Hugging Face, GitHub), gcc, git and disk space.
- Builds `~/kv_a5000/venv` with the NUC's versions (torch 2.11.0+cu128, transformers 4.57.6), plus pyarrow, cmake
  and nvCOMP.
- Checks that a Triton kernel compiles and runs.
- Clones SplitZip and ExANS at the pinned commits.
- Finds a CUDA 12 `nvcc` for the ExANS build. If none is on the node, it installs the CUDA 12.8 toolkit (no driver)
  into `~/kv_a5000/cuda12_8`.

The first time takes 15-30 minutes.

Success marker: `SETUP DONE`.

If something is missing on the node:
- No Python 3.10+: `module load python/3.12` (or set `PYTHON=`).
- No gcc: `module load gcc`.
- No network from the compute node: export the proxy, or ask the admins.
- To leave ExANS out (no toolkit): `export SKIP_EXANS=1`.

## 4. Smoke test (about 10-15 minutes, nothing committed)

This tests the whole path on Llama-3.2-3B:
- the calibration captures;
- one GSM8K window coded by every codec, including the authors' SplitZip and ExANS, bit-exact;
- the report.

```bash
bash <<'A5000'
set -Eeuo pipefail
trap 'printf "A5000 SMOKE STOPPED: line=%s exit=%s\n" "$LINENO" "$?" >&2' ERR
test -n "${SLURM_JOB_ID:-}"
cd "${KV_HOME:-$HOME/kv_a5000}/kv_reconstruction"
git pull -q --ff-only origin llama32-3b-lossy-lossless-compare
SMOKE=1 MODEL=llama32_3b bash scripts/run_a5000.sh
A5000
```

Success marker: `SMOKE OK: every codec measured`. The bits table printed just above it should show about 9 bits
for our dense mode and about 12 for SplitZip, as on the NUC.

If the smoke test does not pass, stop here and send the last 60 lines of `~/kv_a5000/runs/smoke_llama32_3b/run.log`.

## 5. The measurements: one block, three models

Each model is measured, packaged and pushed before the next starts. A failure in one model is pushed as
`..._partial` and the loop goes on to the next.

```bash
bash <<'A5000'
set -Eeuo pipefail
test -n "${SLURM_JOB_ID:-}"
cd "${KV_HOME:-$HOME/kv_a5000}/kv_reconstruction"
for M in llama32_3b llama3_8b qwen3_8b; do
  git pull -q --ff-only origin llama32-3b-lossy-lossless-compare
  if MODEL=$M bash scripts/run_a5000.sh; then
    printf 'A5000 MODEL DONE: %s\n' "$M"
  else
    printf 'A5000 MODEL FAILED: %s (exit %s); results so far pushed as partial if possible\n' "$M" "$?"
  fi
done
printf 'A5000 ALL DONE at %s\n' "$(git log --oneline -1)"
A5000
```

Success markers, per model:
- `PUSHED benchmark_results/<model>_lossless_a5000_<date> to llama32-3b-lossy-lossless-compare`
- `A5000 MODEL DONE: <model>`

At the end: `A5000 ALL DONE`. Tell me when they are pushed. I pull, check them and write the comparison.

Optionally, add the Phase 2 model: replace the list with `qwen3_4b`, or append it.

### If the allocation ends, or a model fails

Request a new allocation (section 1) and paste the same block. Stages and cases that finished are kept in
`~/kv_a5000/runs/<model>/` and are not measured again. A model already pushed in full only gets a new report
commit.

If you see `STOP: ... is not GitHub's ...`, a result commit could not be pushed earlier. In that case:
- `git status` and `git log --oneline -3` in the checkout show it.
- `git pull --rebase origin llama32-3b-lossy-lossless-compare && git push origin HEAD:llama32-3b-lossy-lossless-compare`
  fixes it.
- If the push is refused (credentials), the script has saved `~/kv_a5000/runs/<name>.tar.gz`. Copy it to the NUC
  and commit it there.

## 6. Batch form (no interactive session; the job runs every command itself)

The same work as section 5, submitted from the login host as one job. Use it once sections 2-4 have run in an
interactive allocation. The job updates the checkout, reruns the idempotent setup check and measures. The output
goes to `kv-lossless-<job>.log` in the directory you submit from.

```bash
sbatch --partition=all --account=oana2026 --qos=research12h --nodelist=gpu-grad-01 \
       --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=12:00:00 \
       --job-name=kv-lossless --output=%x-%j.log --wrap '
set -Eeuo pipefail
cd "${KV_HOME:-$HOME/kv_a5000}/kv_reconstruction"
git pull -q --ff-only origin llama32-3b-lossy-lossless-compare
bash scripts/a5000_setup.sh
for M in llama32_3b llama3_8b qwen3_8b; do
  git pull -q --ff-only origin llama32-3b-lossy-lossless-compare
  MODEL=$M bash scripts/run_a5000.sh || echo "A5000 MODEL FAILED: $M"
done'
```

## 7. Optional: two GPUs for the GPU-to-GPU paths

The probe can also measure GPU pairs: peer-to-peer access, a direct copy, a copy through host memory, and the
same pipelined. This is the link prefill/decode traffic takes inside one machine, and it needs two GPUs in the
allocation. Request `--gres=gpu:2` instead of `--gres=gpu:1` in section 1, then:

```bash
bash <<'A5000'
set -Eeuo pipefail
test -n "${SLURM_JOB_ID:-}"
cd "${KV_HOME:-$HOME/kv_a5000}/kv_reconstruction"
source "${KV_HOME:-$HOME/kv_a5000}/venv/bin/activate"
python scripts/pd_link_probe.py --output "${KV_HOME:-$HOME/kv_a5000}/runs/pd_link_probe_2gpu.json"
A5000
```

Send me the printed table. It is small; I commit it with the analysis.

## 8. How the comparison is made (for review)

**Same data for every method.** Each case is the exact BF16 KV a Hugging Face prefill stores: keys after RoPE, in
cache order. All codecs code those same words, and every decode is compared bit for bit with the original. An
inexact entry is marked "NOT exact" in the report.

**Calibration.** As in SplitZip's protocol, WikiText-2 train calibrates both:
- SplitZip's top-16 exponent code book, from the first calibration document;
- our predictor: 6 x 8,192 tokens for the fit, plus one scales document.

ExANS adapts per block and has no calibration step. No evaluation text is used for calibration.

**Windows.**
- Windows are consecutive text. For HumanEval, GSM8K and MMLU, the windows join many items, so they read like long
  few-shot prompts. MMLU is taken round-robin over its subjects.
- Each window starts with BOS when the model has one. Qwen3 has none.
- Inputs longer than 8,192 tokens (the 16K and 32K context cases) are prefilled in 8,192-token chunks into one
  cache, as a serving engine's chunked prefill does. This keeps a 32K 8B prefill within 24 GB. The cache holds the
  same positions; only the rounding order inside the model differs. Shorter inputs are one forward pass, as on
  the NUC.

**Timing.**
- Times are CUDA-event medians after warm-up, on this GPU.
- Our encode is the sender's whole work from the words, including the forward pass for the features and priors.
- Our decoders are timed twice: plainly (Python launches each kernel) and as one CUDA graph replay. The link tables
  use the faster exact one.
- SplitZip and ExANS run as their authors' code launches them.
- ExANS codes whole blocks of 499,712 values, so a remainder is left out of its count. Its 32K case may run out of
  memory, as on the A4500; it is reported as "not measured".

**Link model.** Transfer time = max(wire time, decode time) on an ideal link, with decoding overlapped with
arrival. 1 Gb/s to 400 Gb/s are shown. The link probe adds the measured PCIe legs of this machine.

**Checks against the papers.**
- SplitZip's Table 4 lists top-16 coverage A->B for Llama-3-8B: 99.85 / 99.84 / 99.78 / 99.77 / 99.73% on
  WikiText-2 / HumanEval / GSM8K / MMLU / PTB. Our dataset table prints the same statistic beside these values.
  A match means our captures and calibration agree with theirs.
- SplitZip's realized compression ratio is 16 / (its bits per value) in our tables. This is comparable with their
  Table 1 for Llama-3-8B.
- ExANS publishes no model or data, so its numbers here are first measurements on public data.

**Records in each result folder.**
- `environment.json`, `pip_freeze.txt`, `nvidia_smi.txt`, `slurm.json`;
- `model_resolved.json` (exact model commit);
- `dataset_texts_manifest.json` (sources and SHA256);
- `capture_meta.json` (the model's geometry and RoPE);
- `run.log`, `console.log`;
- the JSON of each stage, and `summary.md`.
