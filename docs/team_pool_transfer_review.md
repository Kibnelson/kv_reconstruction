# Review of the team's pool and transfer proposal

Reviewed 2026-10-20 against `Lossless_KV_Compression_Pool_and_Transfer_Formats.md`
and our published source at `154093c5116479dedce8a39db3c36b7b92f7f3cd`.
The proposal's reproduction ZIPs were not attached. This review verifies
mechanism overlap, cited primary sources and arithmetic; it does not reproduce
the team's compression rates or 11.8M-value round trip.

**Decision: keep both as candidate options, with the transfer predictor as a
separate codec experiment and the pool layout as a GPU attention experiment.**
Neither is an established replacement for our measured Qwen codecs yet.

## What the proposal contributes relative to our implementation

| Mechanism | Already present in our code | Specific additional option |
|---|---|---|
| Per-channel exponent calibration | `producer_models.fit` fits TRAIN median bases; `frontier_models._fit_bases` optimizes bases for tiered wire cost | Freeze a different bit width for each channel and arrange a fixed-stride dense row |
| Ordered escapes without stored positions | `exponent_recipe._segment_offsets` and `_segment_ranks` implement compact exception streams with paid segment indexes | Single-width channel codes, a sentinel and pool-oriented escape reservation/rank lookup |
| Closed-form linear fitting | `conditional_models._regression` solves ridge regression for retained-feature/anchor predictors | Fit the full previous-layer plus earlier-current-channel conditional predictor for a probability coder |
| Exact previous-layer probability coding | `neural_probability._forward`, `neural_chain` and `logistic_reference` already code original BF16 symbols using previous-layer KV and logistic bin probabilities | Linear rather than neural means; earlier 32-channel blocks; factorized symbol coding; a specified integer probability path |
| Current K helps V | Our compact K-first variant uses exact K through coarse/head statistics | Dense conditional regression on earlier current-layer channel blocks, rather than those small summaries |
| Four-bit token/head offset | We tested a related offset at head/page granularity | Per-token/head is finer granularity and is not the identical tested option |
| Attention reads compressed resident KV | Our experiments restore dense BF16 | A fused attention load path reading compressed exponent storage without a full dense-cache expansion |
| Actual upstream prefill information | Our hook studies retain information from actual hidden activations | The team proposal currently does not do this: its predictor sees produced KV and offline constants |

Executing the team encoder during prefill does not itself demonstrate access
to additional upstream information. It is a complementary produced-KV-based
codec. Its new path also does not establish that our 31-bit coset experiment
has improved; it uses another correction/probability representation.

## Novelty: the closest overlaps are more specific than the draft suggests

**Pool ingredients have low novelty.** SplitZip already evaluates per-channel
calibration and a top-15 sentinel design without explicit escape positions.
The latter saves some bytes but decodes 3.5 times slower in its implementation.
The proposed median-offset, variable-channel-width layout differs from its
top-k codebook. A useful contribution would be a concrete resident attention
layout that handles ranks/escapes cheaply, with measured capacity and latency.
See [SplitZip §§4.3.3–4.3.4](https://arxiv.org/html/2605.01708v2).

**The transfer combination is worth checking, but its novelty is not confirmed.**
AQUA-KV already uses closed-form linear predictors for keys from previous-layer
keys and values from previous-layer values plus current keys. It also examines
reduced-rank predictors. Its residuals are quantized, whereas this proposal
would entropy-code original BF16 words exactly. See
[AQUA-KV §3.2, Algorithm 1 and §4.1](https://arxiv.org/html/2501.19392v2).

Predictor-conditioned exact BF16 coding is present in
[Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/), whose
demonstrated predictor reruns an FP8 model. The team's linear predictor avoids
that transformer pass. [AlignPred](https://arxiv.org/html/2610.17970v1), submitted
13 October 2026, already combines same-token head/layer prediction with entropy
coding, but allows bounded reconstruction error. The narrower candidate is
the particular exact, block-autoregressive, integer-CDF implementation with
useful measured cost. A search finding no identical conjunction does not prove
a first-of-its-kind claim.

Using distinct transfer and resident formats is sensible system design, not
by itself a research contribution. Their reductions apply to different tiers
and must not be added together.

## Pool issue: a fixed dense stride does not give free escape access

For a non-escaped value, its dense exponent bits have an address determined
by token row, channel width and a precomputed channel offset. An escaped
value instead needs the ordinal of its sentinel:

`escape_index(i) = count(sentinels before i in this block)`.

A total escape count cannot answer that lookup. The decoder needs a scan,
rank checkpoints, explicit positions, or a cooperative tile restore. This is
precisely the index/work tradeoff our existing tiered codec exposes. It does
not invalidate the format, but it invalidates assuming arbitrary access costs
only an add. Width/offset lookup, packed bit extraction, inverse zigzag and
word assembly also cost work and memory accesses.

For example, a 16-bit prefix count every 64 values costs 0.25 bits/value.
Adding that to 11.39 would give 11.64 bits/value, or 27.25% reduction, before
alignment. This is illustrative arithmetic, not a measured codec result.
The best implementation may instead share rank work across an attention tile.

Fixed escape reservations do not finish the allocator design. A smaller
compressed slot cannot contain a full raw block on overflow. Specify the raw
overflow region, mode, pointer/block table and reclamation rules, or another
concrete allocation scheme. Count calibration, rank indexes, reservations,
alignment, unused space, output blocks and workspace in peak GPU memory.

The proposed admission arithmetic is consistent if each 1,616-block request
contains 1,600 compressed-prefix blocks and 16 raw-output blocks:

`spare = 2910 - 2 * (1600 * bpv / 16 + 16)`.

This gives approximately 604 raw-block equivalents at 11.37 bpv and 548 at
11.65. The 604 figure uses the whole-document 11.37 result, rather than the
11.39 chunk average. A third request requires at most 9.54 bpv under the stated
assumptions. These are byte-capacity estimates, not observed admissions by an
implemented allocator. The reported pool does not meet that third-request
threshold, irrespective of how small the transfer stream is.

## Transfer issue: dense predictor state grows quadratically with KV width

The headline 8.36 bits/value measures the transfer stream. That is a legitimate
wire metric when constants are already installed. It is not a state-inclusive
single-cache storage metric like many of our Qwen tables.

Let `d` be combined K/V width, `b` the channel block width and `L` the layers.
A dense predictor using all `d` previous-layer inputs and earlier current
blocks contains approximately

`(L - 1) * [d*d + d*(d-b)/2]`

coefficients, before biases, scales, RoPE/CDF tables and framing.

| Geometry, block width 32 | Coefficients | INT32 coefficient bytes |
|---|---:|---:|
| SmolLM2: 30 layers, d=384 | 6,236,160 | 24,944,640 B, about 25 MB |
| Our Qwen3-4B: 36 layers, d=2048 | 219,054,080 | 876,216,320 B, about 836 MiB |

This reproduces the approximate SmolLM coefficient count and projects the
unstructured dense construction to Qwen. It is not a measurement of a Qwen
implementation; structured, shared or reduced-rank predictors can change it.
At Qwen width, 32-channel blocks also mean 64 sequential channel steps per
predicted layer, rather than 12 on SmolLM.

An 8K SmolLM cache has 94,371,840 values. Combining an 8.36-bpv stream with
24,944,640 INT32 state bytes once gives about 34.53% net storage reduction,
before other state. This does not invalidate the 47.8% wire reduction. State
can amortize across compatible caches. Its physical GPU residence can also
be reduced by staging coefficients, but that introduces additional transfers
and work which need timing. A percentage of model parameters is not a byte
comparison against 4-bit model weights.

For Qwen, an unregularized dense fit is also underdetermined with our current
three 512-token training captures: the last block can use 4,064 context
coordinates, versus 1,536 training samples per layer. Use ridge/pseudoinverse
or structure and retain rank/conditioning diagnostics. The team's six 8K
calibration documents provide 32 times our current training-token budget;
match calibration data before attributing gains to the algorithm alone.

## Exactness and serving claims that still need evidence

- A covariance predictor is a best linear estimate; it is the conditional mean
  under additional distributional assumptions, not for arbitrary KV data.
- A predictor input grid of `2^-16` cannot exactly represent every BF16 input.
  That is acceptable if prediction alone is approximate and the coder still
  represents every original uint16 target symbol. Define overflow/clamping,
  integer sum bounds, rounding, signed shifts and all table identities.
- Positive CDF counts and bit-identical probabilities must cover signed zero,
  subnormals, both signs, infinities and NaN payloads. Remaining IEEE operations
  require explicit operation-order/FMA/subnormal rules. Integer design is a
  portability strategy; same-machine round trips are not cross-backend proof.
- Splitting a symbol into a 512-way sign/exponent choice and a 128-way mantissa
  choice does not by itself improve ideal joint rate. It changes coder/table
  implementation. Count both stages, stream states, headers and padding.
- Layers depend on previous layers of the same token chunk. Bounded previous/
  current-layer buffers are plausible, but this is not independent restoration
  of arbitrary layer pages. Indexed chunks and paid restart anchors may be
  needed. Final pool capacity plus scratch still governs admission.
- NumPy/GGUF perplexity and RoPE sanity checks do not establish bit parity with
  native HF/vLLM prefill. Exact reconstruction preserves the captured words;
  a new fused attention kernel still needs output and continuation validation.
- Integer MAC counts and a fraction of prefill FLOPs are not GPU latency
  measurements. Channel-stage barriers, range coding, INT64 execution and
  staging can dominate. Our existing neural reference already found CPU
  arithmetic decoding to be 99.4% of measured restoration time.

## The experiments to add, in order

1. **Reproduce the team's artifacts** when its two ZIPs are available. Verify
   complete stream bytes, no source-data reads during decode, calibration/test
   split, state precision, and all original BF16 word comparisons.
2. **Port the transfer predictor to our native-Qwen capture protocol.** Compare
   previous-layer only, previous-layer plus K-to-V, and 32-channel autoregression.
   Include dense and structured/reduced-rank state. Use the same calibration
   budget and exact coder for matched ablations; retain existing neural and
   independent-field controls. Report wire bytes separately from state-inclusive
   cache and shared-state estimates.
3. **Add actual-prefill information as a separate extension.** Condition this
   stronger predictor's uncertainty/distribution on a tiny actual-hidden or
   normalization-derived code. Compare the same predictor with no code,
   produced-KV code and upstream code, charging all side bits and weights.
   This is an untested hybrid, not evidence that upstream information helps.
4. **Evaluate the pool as its own GPU experiment.** Compare cooperative rank,
   a rank directory and explicit sparse positions under a fused attention load
   path. Measure attention/continuation correctness, actual allocated bytes,
   peak scratch, overflow rates and latency at long context and concurrent
   batches. A whole-cache BF16 expansion tests a different memory contract.

The transfer branch is a useful new control and potential cold-storage/wire
option. The pool branch could address resident capacity only after its access
and allocator costs are implemented. Neither the reported 47.8% nor the
combination currently demonstrates that a third large request can be admitted.

The [arithmetic record](../benchmark_results/team_pool_transfer_review_20261020/analysis.json)
stores input-document identity, assumptions and derived counts. No benchmark
source or existing result was changed by this review.
