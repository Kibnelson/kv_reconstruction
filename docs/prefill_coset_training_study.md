# Fixed-capacity training study for producer-informed exact cosets

The first A4500 coset run restored original BF16 words exactly but retained no
short syndrome groups. Its hidden-code error rate did not beat constant zero,
and 41.59% of layer/channel pairs received no optimizer minibatch. Full TRAIN
captures still supplied priors and normalization. The next question is whether
that weak signal survives a controlled improvement in fitting.

This study holds the representation, rank and decoder width fixed. It changes
two training choices and reports every arm:

| Arm | Channel sampling | Optimizer updates per model |
|---|---|---:|
| `random_short` | Original independent random minibatches | 1,000 |
| `balanced_short` | Shuffled per-layer channel cycles | 1,000 |
| `random_long` | Original independent random minibatches | 8,000 |
| `balanced_long` | Shuffled per-layer channel cycles | 8,000 |

Default models are no-code, produced-KV code and actual-hidden code, with
rank 8 and width 32. The field baseline and matched error-position control
remain. `--modes none kv kv_canonical hidden hidden_norm` includes the two
additional sources at the same rank/width; all their state is charged.

## What makes the comparison controlled

Both samplers use identical source records, token windows and residue-width
draws. The balanced sampler uses an independent seeded channel permutation for
each layer. Each complete channel cycle visits every channel once; wrapping
avoids duplicate channels in a minibatch. At 1,000 updates, this still need not
cover every channel: 36 layers and 64 of 2,048 channels per update require
1,152 round-robin updates for one complete pass.

Within each representation, all arms use the same seed, TRAIN captures and initial numerical parameters.
They share an 8,000-update cosine learning-rate horizon. Thus a short run must
match its long run's parameters and sampling digest at update 1,000. This
horizon differs from the previous standalone 1,000-update run; its weights are
not a historical reproduction. Source and numerical hashes make the new
matched comparison explicit.

The four-arm plan is written before any arm runs. DEV diagnostics are
read-only: they do not change gradients, stop training or select a checkpoint.
The final requested TRAIN update is always used. Actual complete DEV packet
bytes subsequently choose a residue width before TEST, as in the earlier study.
Every arm is reported; TEST selects nothing.

Final comparisons use three new prespecified synthetic holdouts: canal-gate
operations, interval-merging code and museum inventory. `--cases 2` selects
text and code. They avoid reusing the already-inspected pottery TEST fixture,
but remain synthetic coverage rather than a representative workload benchmark.
TRAIN data is unchanged so this experiment isolates fitting; greater training
diversity remains a separate possible experiment.

## Metadata retained for the next decision

Each arm contains `training_diagnostics.json`, with a fixed DEV probe at
initialization and updates 250, 1,000, 4,000 and 8,000 when within its budget.
The default probe uses all channels and layers, selecting 32 contiguous head
and 32 contiguous tail tokens per layer. Groups never cross record boundaries.
Hashes bind the exact addresses, producer sources, targets and masked known
fields. No raw captures are exported.

At each checkpoint it records:

- BCE in bits, hard errors, original/predicted ones and Brier scores, separated
  by residue width and omitted bitplane, with K/V and layer summaries.
- Fixed-bin calibration counts and probability/truth sums.
- Frozen TRAIN-prior and constant-zero controls. The trainable predictor prior
  is never relabelled as the frozen reference.
- Error histograms for 31-position groups, recoverable fractions, tails and
  candidate byte ledgers before headers/features/state. These are diagnostics,
  not actual encoded packet savings.
- Retained-code moments, distinct values, clamping and hashes. At initialization
  and each arm's final update, zero/shuffle ablations report changed predictions,
  loss deltas and error deltas.
- Numerical parameter hashes, sampling-prefix hashes and optimizer exposure
  summaries. Complete final per-layer/channel update and word counts remain in
  this external report; deployment manifests retain compact summaries/hashes.
- Diagnostic wall time and selected-device memory observations. CUDA allocator
  peaks are cumulative without resets. They are not isolated kernel or serving
  measurements; fit wall time includes diagnostic work and identifies it.

The existing packet reports still retain all selected-width and common-width
comparisons, real field/feature/escape/framing bytes, complete fitted state,
page identities, shuffled independent restoration, DEV metadata ablations,
full-prefix finite/bitwise continuation, timings and source/runtime provenance.
Progress diagnostics are saved after each probe, even before fitting finishes.

## Run on the A4500

Use the existing CUDA virtual environment and pinned Qwen checkpoint:

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main

python qwen_kv_gpu_bench.py --prefill-coset-training-study \
  --device cuda --modes none kv hidden \
  --rank 8 --width 32 --short-steps 1000 --long-steps 8000 \
  --probe-updates 250 1000 4000 8000 --diagnostic-tokens 64 \
  --train-tokens 512 --dev-tokens 128 \
  --batch-tokens 32 --channels-per-step 64 \
  --lengths 2048 8192 --cases 2 \
  --chunk-tokens 512 --tile-tokens 64 --max-pairs 72 \
  --residue-bits 0 2 4 6 --generation-steps 2 --repeats 3 \
  --output runs/qwen_coset_training_2x2
```

The root summary combines all four child runs. Keep the output directory fresh.
This is a longer diagnostic experiment; neural fitting, DEV probes and packet
coding run separately in each arm. The CPU packet coder remains a reference.

Audit and package all four arms together:

```bash
python scripts/review_coset_training_results.py \
  runs/qwen_coset_training_2x2 --source-root . \
  --output runs/qwen_coset_training_2x2_review

python scripts/package_coset_training_results.py \
  runs/qwen_coset_training_2x2 \
  --destination benchmark_results/qwen_coset_training_2x2_a4500

git add -f benchmark_results/qwen_coset_training_2x2_a4500
git commit -m "Add A4500 coset training coverage and duration results"
git push origin main
```

Packaging verifies every child's report and diagnostic identity, gzip roundtrips,
and fitted calibration files. Raw large JSON never enters the package; each
delivered file must fit the 99 MiB limit. The original run is unchanged. The
package omits packet blobs and raw captures, so a later independent review
rehashes delivered state and audits recorded packet evidence without replaying
omitted GPU data.

## How to interpret the outcome

Compare balanced versus random at the same budget to measure channel coverage.
Compare long versus short for each sampler to measure additional training.
Inspect fixed DEV BCE and hard decisions separately: useful probabilities need
not immediately improve a thresholded single-error decoder. Ablation harm and
an advantage over no-code/equal-budget KV controls would provide evidence that
the upstream code contributes information. Code activation alone is insufficient.

Check actual final packet and complete-state deltas before claiming a storage
gain. If prediction improves but groups still contain many errors, that points
to a different correction/entropy coder. If balanced/long training improves
neither probability rate nor decisions, improving the extractor/objective or
TRAIN diversity is more justified than simply strengthening parity. Neither
outcome establishes that all producer-informed designs work or fail.

Local tiny-model tests verify software behavior, exact packets and matching
training prefixes. They do not establish pretrained Qwen savings, convergence,
40% reduction, migration throughput or high-concurrency serving benefits.

## Delivered local validation

The Qwen suite runs 331 tests successfully, with 11 CUDA-only skips. The 36
supporting standalone, packaging, analyzer and audit tests pass. The rebuilt
standalone completes four arms, three model sources and two fresh tiny TEST
fixtures, with all 12 full-DEV continuation checks exact. Every omitted plane
in that random tiny run remains raw, so its continuations do not exercise
retained syndrome recovery. Separate controlled syndrome tests remain in the
suite.

The [packaged tiny evidence](../benchmark_results/prefill_coset_training_tiny_20261020.zip)
passes all 5,295 training-study audit checks, including delivered numerical
state, matched prefixes and fixed probes. The [test summary](../benchmark_results/prefill_coset_training_tiny_review_20261020/test_summary.json),
[analysis](../benchmark_results/prefill_coset_training_tiny_review_20261020/analysis.json)
and [validation](../benchmark_results/prefill_coset_training_tiny_review_20261020/validation.json)
retain the identities and scope. To inspect the bundle:

```bash
unzip benchmark_results/prefill_coset_training_tiny_20261020.zip -d runs
python scripts/review_coset_training_results.py \
  runs/prefill_coset_training_tiny_20261020 --source-root . \
  --output runs/prefill_coset_training_tiny_review
```
