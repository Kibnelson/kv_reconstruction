# Actual prefill metadata for exact KV coset reconstruction

The [next controlled training study](prefill_coset_training_study.md) tests
channel coverage and training duration separately, with fixed DEV probes and
fresh TEST fixtures, while holding the representation and network size fixed.

This experiment tests the missing coupling: a small retained code derived from
actual producer activations guides reconstruction of original BF16 KV bits from
short coset indices. The coding machinery is established; the research question
is whether the particular upstream signal improves the complete size versus
restoration-work tradeoff over matched KV-only controls.

The [first A4500 review](qwen_prefill_coset_gpu_review.md) is now available:
all five fitted models retain raw mantissa planes, with no short syndrome
groups or producer-code payload gain. Hidden code estimates 28.91% net
reduction versus 30.50% for the field baseline. The logs identify sparse
optimizer channel coverage and a large mismatch with single-error recovery;
the next diagnostic is signal quality and training coverage at fixed size.

## Representation and exactness

The targets are the model's original final K and V words. A page contains one
layer and token tile. Exact exponent and sign streams, low mantissa residue
bits, a retained BF16 feature code, correction streams, headers and checksums
are all real serialized fields. Exponent centers are fitted on TRAIN only.
The initial field coder is a separately named CPU zlib/raw reference, not the
earlier tensor Huffman codec or a port of ExANS.

The decoder predicts only omitted mantissa bits, using its retained feature
code and the page's exact sign, exponent and residue bits. It never receives
the omitted target bits, original hidden state or another layer's KV. Unknown
bits are masked at the predictor boundary as well as by the caller.

For one omitted bitplane, group 31 consecutive token positions within a
channel. Let A have the 31 distinct nonzero five-bit columns. Transmit the
five-bit syndrome A*q. The decoder predicts p and uses A*q XOR A*p to locate
one erroneous bit. The encoder permits this mode only after verifying at most
one prediction error and checking exact recovery. Otherwise it stores the
31 original bits with an explicit raw-mode flag. Incomplete groups use exact
raw tails. A complete raw-plane choice prevents group descriptors from forcing
an unnecessarily expensive representation.

The matched error-position representation transmits zero or the one wrong
position using the same five bits. It should have the same correction size.
This control prevents attributing a mere change of notation to a coding gain.
All feature bytes remain charged even when every omitted plane selects raw.

## Matched models

| Mode | Encoder information | Decoder information |
|---|---|---|
| `none` | No prompt-specific feature | Frozen state and exact page fields |
| `kv` | Same-page produced final K/V | Retained feature code and exact page fields |
| `kv_canonical` | Final K approximately inverse-rotated using known RoPE, plus V | Same retained-code contract; no actual pre-RoPE hook access |
| `hidden` | Actual normalized hidden activations observed inside attention | Same retained-code contract |
| `hidden_norm` | Actual hidden activations plus normalization scales recomputed from hooked K-projection inputs | Same feature rank and byte budget; scales must influence the transmitted code, not appear as free decoder inputs |

The normalization-scale variant recomputes statistics from captured inputs.
It does not yet reuse an internal fused RMSNorm statistic. Its computation is
part of feature production, and its extra encoder parameters count.

The predictors share width, feature-bit budget where applicable, deterministic
sampling schedule and TRAIN data. Actual parameter counts differ and are
reported. All modes receive the same known channel-specific RoPE phase for K;
V receives a neutral phase. Training uses omitted-bit cross-entropy as a surrogate, not a claim
of joint optimal packet-rate training. DEV actual complete packet sizes select
residue width, then the selection freezes before TEST.
TEST also compares a common residue width selected by the no-code model on
DEV, separating metadata quality from differences in width selection.

## Diagnostics for the next decision

The run records source/checkpoint identities and environment; TRAIN provenance,
actual sampled word examples and updates per layer; fitted codec state and
hashes; DEV residue-width grids and selection; and full-prefix generation
checks. TEST contains measured page bytes and labelled full-cache estimates.

Per-page, bitplane and K/V diagnostics distinguish bit prediction errors,
error counts per 31-bit group, syndrome-eligible groups, retained syndrome
groups, raw escapes, raw-plane choices and tails. Byte ledgers distinguish
features, exact fields, corrections, padding, descriptors, container indices
and state. Position/layer breakdowns expose drift and poorly predicted layers.
DEV code-use ablations, when reported, are diagnostics rather than new fitted
models or TEST-selected policies.
Count-only screens of alternative channel grouping and 15/63-position groups
help assess future layouts or correction strength. These alternatives are not
encoded results and do not change TEST policy.

Interpret outcomes in this order:

1. Does actual hidden information improve correction payload over equal-budget
   KV-derived and no-code models?
2. Does that gain exceed retained feature bytes, extra state and framing?
3. Are high error rates concentrated in particular bitplanes, layers, heads or
   token positions? This decides whether to change the feature, training data,
   residue width or correction strength.
4. Does the observed restoration work justify a GPU implementation? Syndrome
   and error-position equivalence is expected; a faster implementation must be
   measured rather than inferred from XOR arithmetic.

## Run

The self-contained launcher includes this separate parser:

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main

python qwen_kv_gpu_bench.py --prefill-coset-study --device cuda \
  --steps 1000 --rank 8 --width 32 \
  --train-tokens 512 --dev-tokens 128 --chunk-tokens 512 \
  --lengths 2048 --cases 1 --tile-tokens 64 --max-pairs 72 \
  --residue-bits 0 2 4 6 --repeats 3 --generation-steps 2 \
  --output runs/qwen_prefill_coset_smoke
```

Keep the working CUDA environment; no dependency upgrade is required by this
study. `--prefill-coset-study --help` lists its options. A local functional run
uses `--prefill-coset-study --tiny --device cpu` with a fresh output directory;
the tiny random Qwen model downloads no weights and establishes no pretrained
Qwen compression result.

## Share complete review metadata

```bash
python scripts/package_results.py runs/qwen_prefill_coset_smoke \
  --destination benchmark_results/qwen_prefill_coset_smoke_a4500

git add -f benchmark_results/qwen_prefill_coset_smoke_a4500
git commit -m "Add A4500 prefill metadata and exact coset results"
git push origin main
```

The package preserves the full report as verified gzip, compact reports,
provenance and actual small fitted codec artifacts under `calibration/`.
It omits pretrained LLM weights, raw activation captures and packet files.
The original run retains packet files for local restoration checks.

To audit either the original run or the packaged results without loading the
model:

```bash
python scripts/review_syndrome_results.py \
  benchmark_results/qwen_prefill_coset_smoke_a4500 \
  --output runs/qwen_prefill_coset_review
```

## Limits

The initial local release completed the 319-test CPU suite (11 CUDA-only
checks skipped), 19 standalone/package/audit tests, and a fresh standalone
tiny-Qwen run with all five full-DEV continuations exact. Its packaged evidence
passes 652 stdlib audit checks. The ordinary tiny run retained no short
syndrome groups; separate controlled tests exercise actual syndrome recovery
and generation. These are functionality checks, not metadata savings evidence.
See [the local run](../benchmark_results/prefill_coset_tiny_cpu_20261020/summary.md)
and [audit](../benchmark_results/prefill_coset_tiny_review_20261020/validation.json).

Feature extraction and bit prediction use the selected Torch device. The
initial bitpacking, syndrome and zlib implementation is a CPU reference;
its timing is not GPU codec throughput. Instrumented prefill includes all
enabled encoders and diagnostic copies. Raw KV remains resident, and there is
no isolated production-hook overhead, compact-resident attention, enforced
materialization budget, SSD migration or concurrent-serving result.

Previous Qwen experiments did not reach 40% state-inclusive reduction. This
experiment tests a specific hypothesis and makes no savings, speed or priority
promise. [Neural binning](https://arxiv.org/abs/2310.16961),
[distributed source coding](https://arxiv.org/abs/2106.02797), and
[NestQuant's KV lattice cosets](https://arxiv.org/html/2502.09720v3) are relevant
precedents; the proposed contribution must be the particular producer signal
and its measured exact reconstruction tradeoff.
