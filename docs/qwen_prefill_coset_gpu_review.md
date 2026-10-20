# A4500 review: producer metadata plus exact cosets

The direct coupling is now tested, but **this fitted representation adds no
correction-payload saving**. Every DEV and TEST mantissa plane selects raw
storage; no short syndrome group is retained. The independent field baseline
leads at **30.4979% estimated net reduction**, versus **28.9149%** for the
hidden-derived code. This result applies to the current rank-8, width-32 models
and single-error code, rather than all possible producer-informed coset designs.

Results: [`b429488`](https://github.com/Kibnelson/kv_reconstruction/commit/b429488119ef20266a7b2a3cf3c3fd67b664e9a3).
Benchmark implementation: [`014e5b2`](https://github.com/Kibnelson/kv_reconstruction/commit/014e5b24765efe03ced426be8d6842656a58e79a).

## Evidence and scope

All **4,052 independent audit checks pass**, including the 180 source hashes,
full gzip identity, delivered codec-state manifests/shards, provenance,
DEV selection, packet components, group diagnostics and state accounting.
All five delivered fitted models also load through the canonical model loader
and reconcile their complete state sizes. Unlike the previous probability
package, the actual small learned codec artifacts are present and independently
rehashable. Packet files and raw captures are omitted; the review does not rerun
the GPU or independently replay those omitted packets.

The 18,068,925-byte gzip restores the complete 454,675,495-byte report with
SHA256 `5e5fc9065165ee58e5d4843e9f1791adfb0d1ec922e831a8c898ea2722f199f6`.
The run uses pinned Qwen3-4B-Instruct-2507 on RTX A4500, Torch 2.11.0+cu128,
Transformers 4.57.6 and NumPy 2.3.5.

TRAIN has three separate 512-token synthetic fixtures. DEV covers the complete
128-token prefix. TEST contains one 2,048-token fixture, all 36 layers, and
64-token tiles starting at 0 and 1,984: **72 pages and 18 MiB**, or **6.25%** of
the full 288 MiB cache. Whole-context storage figures extrapolate this sample,
then charge complete state once. There is no new 8K or broad-workload result.

Every reported TEST row succeeds. All five DEV continuations pass two steps
with finite, bit-identical logits, tokens and appended KV. The reports explicitly
set `syndrome_recovery_exercised=false`: these Qwen checks validate raw-plane
restoration, not a successful retained syndrome path. Separate local controlled
tests exercise actual syndrome recovery and generation.

## Matched storage results

| Representation | Measured packet reduction | Estimated net 2K reduction | Omitted bit-6 error |
|---|---:|---:|---:|
| Frozen exponent control | 29.3620% | 29.2152% | — |
| Independent field baseline | **30.5223%** | **30.4979%** | — |
| No-code bit predictor | 30.5090% | 29.5788% | 41.6595% |
| Produced-KV code | 30.1180% | 28.9707% | 41.6983% |
| Canonicalized-KV code | 30.1142% | 28.9668% | 41.6663% |
| Actual-hidden code | 30.1164% | 28.9149% | 41.6672% |
| Hidden plus normalization-source code | 30.1145% | 28.9121% | 41.6496% |

Syndrome and matched error-position packets have identical complete sizes for
every model. All models select six residue bits on DEV; common-width and
individually selected-width comparisons therefore coincide.

The normalization variant recomputes per-head scales from actual hooked
pre-normalization K projections, incorporates them into the same eight-word
code, and charges its extra encoder state. It does not directly reuse an
internal RMSNorm accumulator or receive free scales at decode time.

## Why the coset correction does not save bits

With six raw mantissa residue bits, only the highest mantissa bit remains to
predict. The hidden model can correct at most one error in **62 of 294,912**
31-bit groups: **0.0210%**. No-code has 86 eligible groups. Hidden error counts
peak near **13 errors per group**, outside the single-error regime.

Ignoring tails, headers and features, a group consumes five bits when eligible
and 31 otherwise, plus a one-bit mode flag. If p is its eligible fraction,
average cost is `32 - 26*p` bits. Beating 31 raw bits needs **p > 1/26 = 3.846%**.
Hidden's observed fraction is about 183 times below that threshold. Whole-plane
raw fallback correctly rejects the attempted correction representation.

The best hidden layer/head aggregate, layer 5/K2, has 9/512 eligible groups
(1.758%), still below this basic threshold. This is descriptive TEST screening,
not a trained selection policy. Count-only 15/63-position and channel-group
diagnostics also expand the correction payload before extra headers/state.
Those alternatives were not encoded, and no stronger code was constructed.

There is an even simpler signal check: predicting bit 6 as constant zero would
make **41.5661%** errors, slightly fewer than every learned model on aggregate
TEST. This follows from original-one counts; it is a diagnostic, not an
implemented predictor, packet or generation result. Roughly 58% accuracy alone
does not establish useful upstream information.

## The metadata changes predictions, but not retained bytes

On DEV, zeroing hidden codes changes **5.0964%** of omitted-bit predictions;
shuffling changes **5.9368%**. Both change complete packet bytes by **zero**,
because all planes remain raw. The code is active in the network, but those
changed decisions do not produce a retained coding advantage.

Hidden loses **0.663917 percentage points** to no-code:

| Additional cost | Sample/full-cache bytes | Net loss |
|---|---:|---:|
| Retained features | 73,728 per 18 MiB sample | 0.390625 pp |
| Framing | 360 per sample | 0.001907 pp |
| Additional complete state | 819,555 once per full 288 MiB cache | 0.271385 pp |
| **Total** | | **0.663917 pp** |

Exponent, sign and mantissa payloads are identical across all fitted models.
No correction saving offsets their side costs. The field baseline also avoids
the unused bit-predictor state, explaining its lead over no-code.

Selecting six residues is itself a layout effect: every DEV width retains all
seven mantissa bits through residues plus raw planes. Moving from zero to six
residues removes **15,696 bytes of framing for every model**. It is not evidence
that the predictor successfully reconstructs the other bit.

## What the training metadata reveals

All modes use the identical verified sampling schedule and 1,000 optimizer
updates. Each layer receives **27–28 updates**, with 64 selected channels and
32 token rows per update. The full schedule was reconstructed and its SHA
matched before deriving coverage.

Across 73,728 layer/channel pairs, **30,664 (41.59%) never appear in an optimizer
minibatch**; 27,126 appear once. Priors and normalizers still use the full TRAIN
captures, and shared parameters can generalize, so these pairs are not unseen
by the experiment overall. The 2,048,000 sampled word examples equal 1.808% of
available TRAIN words, with repeats possible.

Training takes 21.83 seconds of the 421.67-second run. Checkpoint losses use
changing minibatches, layers and residue widths. They are not a fixed DEV
convergence curve. This limited optimization exposure prevents a conclusion
that the extractor has converged, but does not turn the observed failure into
evidence that more training will succeed.

## Next decision

The next inexpensive diagnostic should keep rank and width fixed and test
optimizer channel coverage and training duration separately. Record fixed DEV
bitwise cross-entropy, hard error, calibration and group recovery throughout
training, with constant-zero/no-code and equal-budget KV-derived controls.
Charge all metadata and state as before. Use DEV for choices and reserve fresh
held-out prompts for a final claim after this iterative analysis.

Proceed to stronger coset decoding or a GPU packet kernel only if the producer
code first demonstrates a useful held-out predictive/rate advantage. The
current error regime is far from single-error recovery. More network width,
more parity strength or faster XOR kernels alone are not supported as the next
fix by these data.

There is also a target ceiling: even removing the one selected omitted bit
completely for free would raise the field baseline only to **36.7479% net
reduction**, with all other costs held fixed. Reaching 40% needs useful coding
of more than one plane and/or better exponent coding. This is arithmetic for
the current layout, not a fundamental lower bound.

## Restoration and comparisons with earlier work

Median functional restore over the same 18 MiB sample is about 0.271 seconds
for the field baseline and 0.916 seconds for hidden; prediction accounts for
about 0.626 seconds in its first repeat. These are resident-state CPU-packet
paths with device neural work. All learned planes are raw, so these timings do
not demonstrate the speed of useful syndrome reconstruction.

The input and full KV hashes match the previous independent probability smoke
exactly, despite recorded runtime differences. Its no-code arithmetic model
stored fewer bytes (33.19% packet / 32.20% net estimate) but used a much slower
CPU arithmetic reference. This is a size-versus-implementation-cost comparison,
not a matched proof of codec or serving speedup. The older 33.39% Huffman/sign
mean uses another sample protocol and should not be pooled into this table.

No aggregate GPU admission budget, compact-resident attention, SSD migration,
high-concurrency TTFT or TPOT improvement was measured.

## Reproduce the audit and insights

```bash
python scripts/review_syndrome_results.py \
  benchmark_results/qwen_prefill_coset_smoke_a4500 \
  --output runs/qwen_prefill_coset_review

python scripts/analyze_syndrome_insights.py \
  benchmark_results/qwen_prefill_coset_smoke_a4500 \
  --output runs/qwen_prefill_coset_insights.json
```

Add `--source-root` pointing at the recorded benchmark source to verify all
source hashes. Machine-readable [audit](../benchmark_results/qwen_prefill_coset_review_20261020/validation.json),
[metadata insights](../benchmark_results/qwen_prefill_coset_review_20261020/metadata_insights.json),
and [error diagnostics](../benchmark_results/qwen_prefill_coset_review_20261020/error_insights.json)
retain the arithmetic, coverage checks and scope qualifications.
