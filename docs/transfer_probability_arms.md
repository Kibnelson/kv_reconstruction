# Four-arm exact Block32 probability experiment

This study asks whether the information left uncertain by Block32 can be
described more cheaply using decoded context and a small paid producer hint.
It keeps the original mean predictor frozen. It changes the probability width,
while continuing to code and recover the original BF16 words exactly.

| Arm | Probability width | Transmitted hint |
|---|---|---|
| A | Existing per-channel calibrated width | None |
| B | A plus an adjustment from decoder-known context | None |
| C | B plus a supervised produced-KV scale-innovation hint | Fixed 2 bits/token/layer by default |
| D | B plus a supervised actual-prefill-H scale-innovation hint | Same bit budget and decoder structure as C |

All four use equal fixed binary framing. A must reproduce the original
`transfer_reference` entropy stream byte for byte. Its legacy JSON-framed
packet size is also recorded; changing framing is not credited to prediction.
The new A rate need not reproduce 9.430599: the model, inputs and deliberately
separated fitting split can differ from that earlier SmolLM experiment.

## What is fitted and when

Three TRAIN captures are used by default. The first two fit the frozen Block32
mean predictor. The third, disjoint capture fits B and the C/D extractors.
The first half of DEV calibrates the base channel scales. The second half
selects adaptation strength from 0, 1/2 and 1 using actual complete packets.
No TEST values or TEST rate diagnostics choose a gate, code width or model.

B fits a small regression of residual scale on causal integer features:
previous-layer K/V exponent summaries, sign balance, and error-bucket summaries
from already recovered current-layer groups. The current group's unknown words
never enter its free context. Width adjustments are quantized to eighth-octaves,
bounded between 1/4 and 4 times the original scale.

C and D fit the same-capacity scalar readout to the scale innovation remaining
after B. Both use signed and absolute source features, allowing the scale hint
to respond to magnitude as well as sign. A fixed random sketch bounds fitting
cost; whitening, sketch and readout are folded into one deployed integer
projection. This is a deliberately small
supervised linear encoder, not an end-to-end neural information bottleneck.
TRAIN quantiles define its discrete IDs. A per-group table converts an ID to a
width adjustment. Equal hint budgets do not imply equal encoder state, because
KV and H have different input widths; all state is reported.

The objective used for the regression is a residual-scale surrogate. Actual DEV
packet lengths select strength. Poor surrogate/generalization behavior is a
possible measured outcome, not evidence that all upstream information is useless.
The original raw-hidden PCA hint and old syndrome decoder are separate controls
documented in `transfer_metadata_coset_validation.md`.

Actual hidden input is captured by an observational `self_attn` pre-hook during
the unchanged model forward. The decoder receives only the packed IDs and its
own decoded context. Capturing hidden states is diagnostic CPU-copy plumbing;
the run does not measure the marginal overhead of a fused deployed hook.

## GPU contract

`--device` selects model capture. `--parameter-device` selects CPU or CUDA for
integer mean prediction, scale-offset prediction and hint projection.
CUDA requests fail if CUDA is unavailable; no CPU fallback is labeled a GPU pass.

This is a **hybrid packet decoder**: the CPU entropy-decodes original words,
builds the next causal context from those recovered words, uploads parameter
inputs to CUDA, receives means/scale offsets and continues the packet decode.
It tests the real dependency order. It is stronger than the earlier known-KV
predictor replay, but the entropy coder and interval probabilities remain CPU.
Per-group CPU/device staging, guards and synchronization are included in timings.
No GPU entropy throughput, cross-machine CDF identity, compressed attention,
pool-capacity improvement or serving speedup follows from this experiment.

Each TEST packet is also decoded in a new process with three data inputs only:
the baseline state, arm decoder state and packet. Original KV, hidden states and
the hint encoder are not decoder inputs. CPU/device mean-and-offset digests,
hint IDs, encoded packet bytes and final original words must all agree.

## A4500 run

Use the existing working environment and install the transfer dependencies.
No CUDA/PyTorch reinstall is needed.

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -r requirements-transfer.txt
```

A bounded real-hook software check needs no pretrained model download:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/bench_transfer_probability.py --tiny \
  --device cuda:0 --parameter-device cuda:0 \
  --warmup 1 --repeats 3 --threads 2 \
  --output runs/block32_probability_tiny_cuda
```

It uses a random two-layer Qwen3 with actual BF16 prefill hooks. Its rates are
software-check results only. It cannot select a method for pretrained Qwen.

The initial pretrained comparison covers a **four-layer prefix**, eight sampled
pages per arm across text/code, and four token positions in each 2K context:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python scripts/bench_transfer_probability.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --device cuda:0 --parameter-device cuda:0 --max-layers 4 \
  --train-tokens 512 --capture-tokens 2048 \
  --page-tokens 32 --page-starts=0,512,1024,-1 \
  --hint-bits 2 --sketch-width 16 --reference-tokens 8192 \
  --warmup 1 --repeats 3 --threads 2 \
  --output runs/block32_probability_qwen4

cat runs/block32_probability_qwen4/summary.md
```

The model forward still executes all layers; fitting and codec measurements use
only the selected prefix. Do not describe this as a full-model compression rate.
Use a fresh output name for a repetition. A subsequent full-model experiment
can omit `--max-layers`, but dense baseline fitting/state grow quadratically
with KV width. CPU entropy parity and independent-process checks intentionally
make this reference expensive. Progress is printed per page and arm.

`--hint-bits 4` is a separate equal-budget C/D experiment. Keep two bits fixed
for the first comparison, rather than choosing a width after examining TEST.
The existing synthetic fixture suite is reused: treat these as exploratory
holdouts, and use fresh prompts for a final claim after studying these patterns.

The four-layer selection uses a prespecified 8K decoder-state amortization
scenario, in addition to reporting warm packet bytes. CPU decoder arrays,
encoder-only arrays, physical CUDA tensors and whole-process peaks are separate.
The scenario is not observed GPU admission capacity.

## Insights retained for the next design decision

| Artifact | What to inspect |
|---|---|
| `summary.md`, `results.json` | Complete packet rates, equal framing, raw fallbacks, exactness, timing samples and state |
| `insights.json` | B versus A, C/D versus B, main-stream benefit versus paid hint cost, source comparison, occupancy |
| `layer_diagnostics.csv` | Layer and K/V ideal gains after allocated hint cost; diagnostic only |
| `diagnostics.json.gz` | Group gains, scale histograms, early/later groups, residual tails and hard-bit screens |
| `calibration/fit_diagnostics.json` | TRAIN response spread, feature ranges and complete layer/group fitting coverage |
| `hints_*.npz` | Small actual categorical code arrays, for later fixed-model sensitivity analysis |
| `input_ids.json`, `capture_manifest.json` | Exact tokens, capture/hidden hashes, model identity, hook provenance and split identity |

The selected-model DEV zero/shuffle test and a forced full-strength hint test
are both saved. This distinguishes a selected zero-strength branch from a hint
that actually changes useful probabilities. Forced full-strength TEST group
costs are also retained as counterfactual ideal diagnostics; they cannot change
the frozen candidate or be called measured new packet savings.

B also has a fixed-model attribution diagnostic: replace its context with
TRAIN mean features or shuffle the token-aligned context inside the page.
If these cost almost the same as actual context, the apparent benefit may be
mostly static width adjustment. This is an ideal-cost counterfactual, not a
separately fitted best static-width codec, so it cannot establish that stronger
claim on its own.

Use the evidence to distinguish: missing residual-scale structure; useful free
context; a useful hint that costs too much; an inactive/collapsed hint; gains
limited to early groups, K/V or particular layers; and outliers beyond the
allowed scale range. Future selection policies must be fitted on TRAIN/DEV.

The 31-bit single-error screen uses bit-6 errors in a declared group-local
token-major order. It does not implement a syndrome stream, reproduce the old
page/head coset layout or establish any new coset saving. It is a feasibility
diagnostic before considering another correction format.

## Reuse and share

Keep the full run on the NUC, including captures, base state and packets.
`--capture-only` stops after verified capture. A new output directory can reuse
the original self-contained captures with `--captures-from ORIGINAL_RUN`;
TRAIN/DEV fitting and frozen selection are still rerun. Reuse validates file
hashes and hidden geometry. A portable KV-only replay fixture has no H and is
insufficient for the actual-prefill arm.

Package only the small evidence and adaptation artifacts:

```bash
report_dir="benchmark_results/block32_probability_qwen4_$(date +%Y%m%d_%H%M%S)"
python scripts/package_transfer_probability.py runs/block32_probability_qwen4 \
  --destination "$report_dir"
git add -f "$report_dir" "$report_dir.zip"
git commit -m "Add A4500 four-arm Block32 probability results"
git pull --rebase origin main
git push origin main
```

The packager reconciles actual packet hashes, byte ledgers, recorded parameter
parity and independent-process evidence before export. It omits raw activations,
the large baseline state and complete packets, so it is an evidence package,
not independently replayable full data. Save full artifacts for later audits.
