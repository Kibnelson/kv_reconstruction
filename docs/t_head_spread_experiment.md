# T head-aware spread: an isolated NUC experiment

Branch: `experiment/t-head-aware-spread-v1`.
Base: `llama32-3b-lossy-lossless-compare` at
`d95fd10a1fae447d47c4ac6abaf77b90af1b7952`.

This branch adds experiment files only. Original T files, existing captures,
old run directories, and the existing Python environment are not modified.
The run script does not install packages, commit, push or merge anything.
Any adoption is a separate decision after the actual NUC results are reviewed.

## The one idea under test

T already computes integer prediction-surprise sums per token and head. It
currently reduces them to one previous-layer difficulty value per token. Test
whether keeping head-specific information improves the next layer's probability
width, at the same mean predictor and original-BF16 exactness.

- `legacy_global`: original T, entirely unchanged.
- `matched_global`: experiment kernels with the original global prior. Complete
  packets must be byte-identical to `legacy_global`, not merely decode correctly.
- `head_selected`: global/same-head blend. DEV selects alpha from
  `{0,0.25,0.5,0.75,1}`; zero is the explicit original-global fallback.
- `learned_selected`: a small TRAIN-only ridge map from previous-head surprise
  to current-head surprise, blended with global. Same DEV alpha selection.
- Shifted-head controls use the same selected alpha/state budget but the wrong
  head alignment. They are diagnostics, not configurations selected on TEST.

All exact BF16 symbols are still coded by T's real GPU rANS implementation.
No token-specific hint travels. The learned map adds shared float32 weights and
bias; its memory, serialized state and execution cost are recorded.
The encoder and decoder use the same float32 prior helper with FP fusion
disabled. This tests same-device exactness, not cross-GPU portability.

The copied experiment coding kernels change only where the spread prior comes
from. The base kernels and their files remain intact. The dispatch validates the
baseline layer view before choosing that layer's frozen head map.

## Calibration and decision

Reuse the existing model captures; no model download or prefill is needed.
The defaults are the existing Llama-3.2-3B run:

- TRAIN: `wtrain0` through `wtrain5`, used for the original predictor and the
  proposed head map.
- Nominal per-channel spread: first half of `wtables`.
- DEV selection: a disjoint window in the second half of `wtables`.
- TEST: the four `ds_wikitext2_r0` through `r3` windows, 8,192 tokens each.
- Predictors: dense and head by head, all model layers by default.

The first-half nominal-spread calibration means this is a newly matched
comparison, not a promise to reproduce the earlier 8.279 bpv number exactly.
Every variant of a given predictor uses identical means, coefficients,
nominal spreads, geometry, source words and position tables. Its policy is
frozen before TEST. No best-on-TEST deployment policy is constructed.

`candidate_for_review` requires an actual CUDA run, full-model coverage, exact
parent/packet-only checks, fewer complete bytes on DEV and held-out TEST, and
decode time no more than the default 1.05 times the original baseline. The
threshold is an explicit experiment budget, not a universal serving criterion.
All costs and per-case results remain visible even when the gate fails.
An alpha-zero choice is reported as no useful head-aware improvement.
The report also charges one extra serialized policy state against the observed
packet gain and reports how many average-size transfers amortize it. This is
separate from the warm, already-resident packet-only rate.

## Run on the NUC in a separate worktree

Keep the existing checkout where it is. Fetch this experiment and give it a
new directory:

```bash
cd ~/kv_llama32
git fetch origin experiment/t-head-aware-spread-v1
git worktree add -b test/t-head-aware-spread-nuc ../kv_t_head_spread_v1 FETCH_HEAD
cd ../kv_t_head_spread_v1
unset TRITON_INTERPRET
KV_HOME=~/kv_a5000 SMOKE=1 bash scripts/run_head_spread_nuc.sh
```

The new directory is on local branch `test/t-head-aware-spread-nuc`; your
existing checkout and its branch stay where they were.

The smoke checks all proposed policies but only four layers and shorter token
windows. It cannot pass the full-model adoption gate. After that passes, run:

```bash
KV_HOME=~/kv_a5000 bash scripts/run_head_spread_nuc.sh
```

The full run uses the completed four-window WikiText-2 capture set. Calibration
and every GPU encoder/decoder are real; no third-party codecs or nvCOMP are
needed. Logs show each fitting, DEV, TEST and independent decoder stage.
Dense fitting can take time. The captures directory is only read, and all new
files go to a timestamped `$KV_HOME/runs/t_head_spread_v1_*` directory.

Optional overrides:

```bash
CAPTURES=/absolute/path/to/captures HEAD_SPREAD_VENV=/absolute/path/to/venv \
  KV_HOME=~/kv_a5000 bash scripts/run_head_spread_nuc.sh
```

`TEST_DOCS` accepts a comma-separated list of already captured held-out windows
for subsequent HumanEval/GSM8K confirmation. Select that list before running;
missing captures fail explicitly. The script never silently substitutes inputs.
`HEAD_SPREAD_OUTPUT` selects a fresh output directory when desired.

The script requires torch `2.11.0+cu128`, Triton `3.6.0`, an actual CUDA GPU,
and no interpreter. It does not repair the old environment. If a separate one
is needed, create it without changing the existing T environment:

```bash
python3 -m venv ~/kv_a5000/venv_head_spread_v1
~/kv_a5000/venv_head_spread_v1/bin/python -m pip install \
  numpy==2.3.5 triton==3.6.0 torch==2.11.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
HEAD_SPREAD_VENV=~/kv_a5000/venv_head_spread_v1 \
  KV_HOME=~/kv_a5000 SMOKE=1 bash scripts/run_head_spread_nuc.sh
```

## What to send back

Share the new run's `share.zip`, or its `summary.md` and `results.json`.
The archive contains only small reports and the frozen DEV selection, not
original KV, model weights, expanded stream arenas or large state files.
Actual CUDA preflight results, including both schedules over all BF16 patterns,
are retained in `device_checks.json` and included in that archive.
Packets are temporary by default and removed only after their fresh-process
check passes; `--keep-packets` keeps them for a later artifact inspection.
Frozen state files remain in the new run for reproduction.

For each TEST packet the parent checks every word, then a fresh process loads
only the packet and its frozen state. It has no source-KV path as an input.
The compact wire pays a fixed header, one 32-bit directory item per rANS
segment, all stream states, alignment and used stream tails. The unused
fixed-capacity arena is not transmitted. Decoder arena expansion and packing
are separate from warm codec-only timing. Shared state, decoder scratch and
dense output are also separate fields; this is not an admission-capacity claim.

## Local validation scope

Software checks use CPU Torch and the explicit Triton interpreter. They verify
the same algorithm, original-stream parity, wire round trips and fit invariants.
They cannot establish CUDA compilation, GPU speed or compression improvement
on pretrained Llama KV. The NUC script runs compiled-device checks, including
all 65,536 BF16 word patterns, before starting the model experiment.

With Torch and Triton installed, reproduce the software suite using:

```bash
python -m unittest -v kv_compare.test_head_spread
```

An optional compiler check targets the A4500's `sm_86` without launching a GPU
kernel. Its JSON explicitly records `gpu_executed: false`:

```bash
unset TRITON_INTERPRET
python scripts/check_head_spread_compile.py --output /tmp/t_head_spread_compile.json
```

The offline compiler check establishes CUDA compilation only. Device execution,
actual compression gain and performance still require the NUC run above.
Recorded local results are in
[`benchmark_results/t_head_spread_software_20261112`](../benchmark_results/t_head_spread_software_20261112/README.md).
