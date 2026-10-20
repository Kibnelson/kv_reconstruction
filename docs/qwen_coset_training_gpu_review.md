# A4500 review: coverage and longer coset training

The three reported failures were an audit bug: cross-arm model identity included
`load_seconds`. This was the **only** differing field in the complete compared
signatures. The corrected package audit passes **12,803/12,803** checks.
The GPU benchmark and original failed evidence are preserved.

The storage result is negative: **all four training arms produce identical
packet lengths for every matched method and TEST case**. More coverage and
eight times as many updates improve some probability estimates, but retain no
short syndrome groups. Actual-hidden code still loses after its side costs.

Input results: [`58c4fe2`](https://github.com/Kibnelson/kv_reconstruction/commit/58c4fe2ac4b6ae4f3d84f1ce2ae58fe5995a912f).
Benchmark implementation: [`ed00571`](https://github.com/Kibnelson/kv_reconstruction/commit/ed00571ae0da0b37aab254d4b479085a46142325).

## Audit correction and evidence

The failed labels compare `random_long`, `balanced_short` and `balanced_long`
against `random_short` under "matched sources, probes, TEST addresses and
architecture". Their model-load observations were approximately 1.75–1.88
seconds. Removing only that timing makes the complete signatures equal:
revision/configuration, TRAIN/DEV inputs, producer and KV hashes, fixed probes,
TEST addresses and architecture all match.

The fix excludes only top-level `model_identity.load_seconds` from cross-arm
comparison, validates it as finite and nonnegative, and records all four values.
Full within-arm model identity and artifact binding remain checked. Regression
tests still reject changed revision/configuration identities. All 16 audit
tests pass; 25 relevant audit/packaging tests pass together. CLI summaries now
print exact failed check names as well as counts.

All 184 benchmark source files match. All six short/long numerical prefixes
match, and all 12 delivered final numerical states match their hashes. Each
packaged child passes 9,293 checks. The original run's child count was 9,306:
packet files were present there and package-manifest checks differed. Root
package checks also change totals versus an audit of the original run. These
count differences do not discard failures. No GPU rerun or independent replay
of omitted packet files was performed.

## Scope

The four arms cross random/balanced channel sampling with 1,000/8,000 updates.
They keep rank 8, width 32, three TRAIN fixtures of 512 tokens, the same seed
and an 8,000-update learning-rate horizon. Short models exactly match their
long counterparts at update 1,000. DEV/TEST do not select a checkpoint.

Fixed DEV diagnostics select 64 tokens per layer from a complete 128-token
prefix, with all 36 layers and all 2,048 joint KV channels. TEST uses two fresh
synthetic fixtures, text and code, at 2K and 8K. Each case sizes 72 paired pages,
or 18 MiB: **6.25% of the 2K cache and 1.5625% of the 8K cache**, at its head
and tail. Whole-cache sizes extrapolate the sample and charge full state once.
Four arms reuse four case/length combinations; these are not 16 independent
workloads.

All measured pages restore exactly. All 12 full-DEV two-step continuations
pass finite, bit-identical tokens, logits and appended KV. Every learned DEV
grid and TEST packet retains zero short groups, so these Qwen continuations
exercise raw-plane restoration rather than successful retained syndrome recovery.

## Actual storage

Balanced-long averages text and code as follows. Other arms have identical
packet lengths; their small net differences reflect about 1.3 KB of training-manifest
state.

| Representation | Net 2K reduction | Net 8K reduction |
|---|---:|---:|
| Independent field baseline | **30.4564%** | **30.4948%** |
| No-code predictor | 29.5355% | 30.2546% |
| Produced-KV code | 28.9274% | 29.8093% |
| Actual-hidden code | 28.8716% | 29.7942% |

Every model selects six raw residue bits on DEV. The remaining mantissa plane
also stays raw. In every TEST case, hidden adds **74,088 sample bytes** versus
no-code: 73,728 feature bytes and 360 framing bytes. Balanced-long hidden adds
**819,573 bytes of complete state**. Its net loss is 0.663923 percentage points
at 2K and 0.460380 at 8K. No correction saving offsets these costs.

Syndrome and equivalent error-position formats retain equal complete sizes.
They do not provide stackable savings.

## Coverage and hard predictions

| Arm | Slots without optimizer minibatches | Hidden DEV bit-6 BCE | Hidden DEV bit-6 error |
|---|---:|---:|---:|
| Random, 1K | 30,664 / 73,728 | 0.955907 | 41.7993% |
| Balanced, 1K | 9,728 / 73,728 | 0.955703 | 41.6648% |
| Random, 8K | 54 / 73,728 | 0.952723 | 41.6061% |
| Balanced, 8K | **0 / 73,728** | **0.952712** | **41.6058%** |

Priors and normalizers use full TRAIN captures in every arm; these coverage
counts refer to optimizer minibatches. Balanced-long covers every channel
overall. Random residue-width draws still leave 9,421 channel/layer slots
(12.78%) without examples for omitted planes 0/1; planes 2/3 leave 585, and
4/5 leave eight. Bit 6 has full optimizer coverage. Channel balancing does not
balance every channel/residue stratum.

Balanced-long hidden supports single-error recovery for only **32/147,456 DEV
groups (0.0217%)**. Its TEST eligibility is 78–98 out of 294,912 groups. The
largest hidden result in any arm is 102 groups (0.0346%). Histograms still
peak at 13 errors per 31-bit group. Five short bits plus a mode flag require
more than 3.846% eligible groups just to beat raw group bits, before headers
and features. Complete-plane fallback correctly rejects these candidates.

Long hidden predictions beat constant-zero hard error slightly on two TEST
cases and lose on two. On fixed DEV, constant zero's 41.5778% still beats
hidden's 41.6058%. Constant zero is a diagnostic, not a new encoded packet.

## Useful probability information remains below its cost

At balanced-long bit 6, hidden BCE is **0.952712 bits/value**, versus
**0.954210** without metadata. Zeroing hidden codes worsens BCE by 0.004247;
shuffling worsens it by 0.002936. The fitted extractor supplies weak useful
information. Better confidence can coexist with almost unchanged thresholded
decisions, which the current single-error decoder needs.

The eight BF16 features cost **128 bits per token per layer**, or **0.0625 bits
per original KV value**. Their bit-6 probability advantage over no-code is only
0.001498 bits/value before code, state and framing.

The apparently larger all-seven-plane result needs another correction:

- At `b=0`, balanced-long hidden has a 0.020619 bits/value total factorized-BCE
  advantage over no-code.
- **85.88%** of that difference repairs planes 0–3 whose hidden BCE still
  exceeds uniform one-bit storage: 1.023196, 1.005597, 1.003412 and 1.000534.
- A hypothetical per-plane choice of raw bits or the measured probability
  estimate reduces the hidden advantage to **0.002912 bits/value**. The largest
  such final-checkpoint advantage across all arms/widths is about **0.003011
  bits/value**, against 0.0625 bits/value of features.

This screen uses aggregate DEV losses with free fallback decisions. It neither
constructs an entropy coder nor charges its flags or measures new TEST packets.
It is not an information-theoretic bound on KV or on possible joint codes.
At unchanged predictions, even the best screened advantage permits only about
**6.17 side bits per token per layer**, before extra state/framing, versus the
current 128. Shrinking the code would change the predictions and must be tested.

## Next decision

Sparse optimizer exposure no longer sufficiently explains the selected bit-6
failure. More coverage and training produced a weak signal without a retained
rate gain. Lower-plane stratum coverage remains a concrete limitation; the
experiment does not establish convergence.

The next feasibility test should optimize information delivered per charged
side bit. A tiny discrete or quantized prefill code, a rate-aware objective and
calibrated/raw-fallback controls would test that directly. Require held-out
benefit to exceed code, framing and state costs before building a stronger soft
coset decoder or GPU packet kernel. A different extractor or joint dependency
model remains open; this scalar predictor and single-error experiment does not
test every such structure.

This is not yet a route to 40%: the best screened metadata-specific gain above
is only about **0.019 percentage points** before costs. Even deleting one whole
mantissa bit for free raises the field baseline only to about 36.71% at 2K and
36.74% at 8K. A material improvement needs substantially stronger conditional
structure and/or better coding of other fields.

Fit timings include normalizer, hashing and host overhead; fixed DEV observers
add substantial measured time. Neither fitting nor resident-state raw-plane
restores establish GPU codec throughput, SSD transfer, admission capacity or
serving latency gains.

## Reproduce the corrected audit

Pull the fix, reuse the completed GPU run and choose a fresh output directory:

```bash
git pull --ff-only origin main
python scripts/review_coset_training_results.py \
  runs/qwen_coset_training_2x2 --source-root . \
  --output runs/qwen_coset_training_2x2_review_fixed
```

The package audit instead takes
`benchmark_results/qwen_coset_training_2x2_a4500` as input. Original-run and
package totals differ because delivered manifests/packet files differ.

The [compact summary](../benchmark_results/qwen_coset_training_review_20261020/summary.json)
contains reproduced rates and costs. The [review evidence bundle](../benchmark_results/qwen_coset_training_review_20261020.zip)
contains the corrected full audit, original-failure reproduction, field-level
identity diagnosis, fixed DEV trajectories, numerical state/prefix checks,
and actual TEST component/error analysis. Its
[manifest](../benchmark_results/qwen_coset_training_review_20261020/review_bundle_manifest.json)
binds every bundled file and the tested audit code. The original uploaded
results and failed audit remain at their original repository paths.
