# A4500 review: independent KV probability pages

The first Qwen test confirms exact independent restoration, including codes
created during actual prefill. It does **not** show a savings advantage from
the retained hidden-derived code. The no-code probability control leads this
smoke. Independence is also achieved without a retained code, so this result
does not establish a unique benefit from producer metadata.

Results commit: [`e0a51c5`](https://github.com/Kibnelson/kv_reconstruction/commit/e0a51c57c1dac77a26c18feb97369bc9c3a41891).
Benchmark source: [`7fe3a86`](https://github.com/Kibnelson/kv_reconstruction/commit/7fe3a8698de355081adfbc6b500bb751347d5c9a).
All **251 review checks pass**, covering package integrity, gzip identity,
174 source hashes, frozen identities, reported exactness and byte arithmetic.

## Current comparison

The run uses Qwen3-4B-Instruct-2507 at the pinned revision, BF16, on an RTX
A4500. All local decoders have width 128 and receive 200 shared optimizer
updates with 64-token batches. The three code-bearing variants use rank 8.
The chain control also uses width 128 and 200 total updates, but has a different
architecture and more parameters.

| Method | Measured packet savings | Estimated net 2K savings | Charged state | Functional decode time |
|---|---:|---:|---:|---:|
| Frozen exponent | 29.36% | 29.22% | 0.423 MiB | 0.166 s |
| No-code probability model | **33.19%** | **32.20%** | 2.869 MiB | 59.11 s |
| KV-derived code | 32.64% | 31.43% | 3.498 MiB | 58.11 s |
| Inverse-RoPE KV-derived code | 32.71% | 31.50% | 3.498 MiB | 58.16 s |
| Actual-hidden-derived code | 32.70% | 31.43% | 3.654 MiB | 57.89 s |
| Current previous-layer chain | 29.92% | 28.24% | 4.846 MiB | 63.38 s |

There is one 2,048-token TEST fixture. The measured sample contains all 36
layers at token starts 0 and 1,984: 72 pages, 128 positions and 18 MiB of raw
KV. That is **6.25%** of the full 288 MiB cache. Full-cache and concurrency
figures extrapolate that sample ratio and charge complete state; they remain
estimates. No new 8K result is present.

The no-code model takes only layer ID. Its network therefore produces static
per-layer/channel distribution parameters, with the K mean rotated by token
position. It has no prompt-specific retained representation.

## Why the side code loses

For the 18 MiB sample, the hidden packet is 93,177 bytes larger than no-code:

| Component | Extra sample bytes | Contribution to net loss |
|---|---:|---:|
| Retained BF16 code | 73,728 | 0.390625 percentage points |
| Exact entropy payload | 19,102 | 0.101206 points |
| Page framing | 347 | 0.001838 points |
| Additional shared state | 823,702, charged once to the full cache | 0.272758 points |
| Total | Sample bytes plus full-cache state | **0.766428 points** |

The hidden code loses even before feature/state overhead: its modeled ideal
rate is **10.61644 bits/value**, versus **10.60024** for no-code. The code adds
another **0.0625 bits/value**. KV and canonicalized-KV ideal rates are 10.62534
and 10.61424 respectively.

All three code-bearing variants lose complete packet bytes to no-code on every
layer aggregate. Hidden improves ideal probability rate on only layers 4, 8
and 24, by less than its side-bit cost. Selecting layers cannot rescue these
particular fitted models. Any future selection policy must be chosen on DEV;
a TEST-selected oracle is only a diagnostic.

Removing all current no-code headers and state would still leave its modeled
ideal rate about **1.00 bit/value above** the 9.6-bit target for 40% reduction.
This is a limitation of the current probability model, rather than a proved
information-theoretic lower bound.

## Exactness, dependencies and timing

Every reported TEST row succeeds. The four local variants log shuffled
restoration from their own serialized page and shared frozen state, with zero
previous-page prerequisites. The chain still has depth 35. All five learned
methods pass complete **128-token DEV**, two-step continuation with finite,
bit-identical logits, tokens and appended KV. These are DEV continuation checks,
rather than full 2K TEST generation checks.

Read-only hooks produce the retained codes synchronously inside attention.
Those BF16 words are passed verbatim to later exact entropy coding. Diagnostic
copies, re-staging and four simultaneous encoders are included in instrumented
prefill; no isolated marginal producer-code overhead or fused implementation
is measured.

CPU arithmetic decoding consumes **99.80–99.81%** of local restoration time.
Neural prediction takes about 0.08 seconds per sample. The 58–59 second figures
and frozen control's 0.166 seconds are one-pass functional path measurements,
with their different documented implementations. They establish no migration
or serving speedup. The whole run takes about 30 minutes; fitting takes only
10.6 seconds. Returning every sampled raw output also requires memory, despite
the absence of cross-page dependencies.

## Comparison with the previously trained chain

The earlier position-study pottery 2K prompt and complete KV hashes match this
run exactly. Restricting its existing records to the same 72 pages gives:

| Model | Width / total updates | Per-page savings, outer container excluded |
|---|---|---:|
| Earlier shared chain | 512 / 7,000 | **33.66522%** |
| Current no-code | 128 / 200 | 33.19582% |
| Current hidden code | 128 / 200 | 32.70215% |

This is a comparison of logged bytes on identical KV, without new model
execution. The earlier model has 14.156 MiB of charged state. Current no-code
beats the short-trained chain in this smoke; it does not beat the established
larger chain's packet rate. The earlier 33.39% mean estimated net result for
Huffman/sign coding also uses another sampling protocol and is not a paired
ranking against this run.

## Next controlled experiments

1. **Isolate training duration.** Keep width 128, rank 8, data and batch size
   fixed, and increase 200 to 7,000 total updates. Current local models see only
   2–10 updates per layer, or 128–640 token examples per layer. All four use the
   same sampled schedule. The smoke cannot establish converged performance.
2. **If more training is insufficient, test initialization.** Warm-start
   conditioned models from the trained no-code distribution, or add a residual
   probability branch whose contribution starts at zero. This is a proposed
   ablation, not an implemented result. It tests whether learning the code
   disrupts a useful unconditional baseline.
3. **Test a smaller inference representation for no-code.** Its network could
   be folded into static distribution tables: roughly 589,824 bytes of numeric
   parameters, plus identity/framing, versus 2,641,152 numeric bytes today.
   Removing network work alone will not fix the CPU entropy bottleneck.
   Predictor equality and/or re-encoding must be validated; reusing old stream
   hashes across different floating-point evaluation orders is unsafe.
4. **Separate rate screening from slow entropy implementation.** Longer-term,
   shortlist models using exact interval costs on DEV, then run actual packets
   and full continuation on frozen candidates. Fast CPU/device entropy work is
   a separate systems experiment. Ideal rates never replace actual packet
   accounting in a final claim.

Do not increase rank or width in the same first experiment: that would obscure
the effect of training duration and raise side/state cost. Broader 2K/8K and
multi-fixture validation should follow, with fresh held-out data for final
claims after iterative tuning. The original prefill-code contribution remains
a hypothesis until it improves a measured rate/restoration tradeoff over the
KV-derived and no-code controls.

## Evidence and reproduction

The package contains reports and anchor state, but omits neural parameter
shards, raw KV/hidden captures and compressed page files. This review verifies
the supplied reports, source identities and accounting; it cannot independently
replay the learned GPU caches or rehash omitted artifacts. Logged exactness
checks and declared artifact hashes must be interpreted within that boundary.
Complete serialized state, including encoder parameters and roughly 366 kB of
provenance per local model, is conservatively charged. A lean inference artifact
could reduce that charge, but has not been measured.

```bash
python scripts/review_side_results.py \
  benchmark_results/qwen_prefill_side_smoke_a4500 \
  --previous benchmark_results/qwen_neural_shared_positions_a4500 \
  --output benchmark_results/qwen_prefill_side_review_20261020
```

The [analysis](../benchmark_results/qwen_prefill_side_review_20261020/analysis.json),
[validation](../benchmark_results/qwen_prefill_side_review_20261020/validation.json)
and [review provenance](../benchmark_results/qwen_prefill_side_review_20261020/review_manifest.json)
preserve exact values and comparison scopes. This audit imports no ML packages
and does not rerun the model or benchmark.
