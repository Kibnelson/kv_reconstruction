# Qwen GPU findings and the next research direction

Reviewed input commit: `ea7733a1380dce55240cf5fd6e4ac521fd6ac7c9`.
The [original A4500 reports](../benchmark_results/qwen_insights_a4500_20261019_095113/summary.md) contain the source/freeze manifests and every candidate, including negative results.

## What the new run establishes

All **420 measurements passed exact sampled reconstruction**. The run covers three synthetic TEST fixtures at 2,048 and 8,192 tokens, with 144 records per case. This is 6.25% and 1.5625% of each full cache respectively, so full-cache savings remain estimates. Each repeated TEST timing uses the same 32 records / 4 MiB of original BF16 data. Three selected methods also pass two continued generation steps on the complete 512-token DEV prefix, with finite, identical logits, tokens and appended KV.

All 128 recorded source hashes, the freeze/configuration, calibration identity and TRAIN hashes verify. All 175 direct-prefill measurements match their post-capture counterparts in sizes and components. Raw KV remains resident during prefill observation. The run does not measure SSD I/O, concurrent serving, compressed-resident attention, or reduced serving latency.

The uploaded reports contain calibration provenance but omit the fitted model binaries. Their recorded identities and accounting agree; this review cannot independently hash those omitted binary files.

| Method | Estimated full-cache savings, including state | GPU decode, ms per raw MiB |
|---|---:|---:|
| Existing exponent group packing | 24.17–24.21% | 2.00–2.06 |
| Fixed exponent 127 + tier coder | 21.27–21.63% | about 6.9–7.0 |
| **Prefill median exponent + tier coder** | **29.55–29.75%** | **6.99–7.03** |
| Midpoint predictor, time layout, groups of 64 | 26.25–26.60% | 4.67–4.74 |
| Best added residual dictionary, 16 entries/time | 23.64–23.87% | 6.17–6.30 |
| CPU Huffman + lower-byte Deflate | 32.59–32.86% | CPU: 209–216 |

The median exponent recipe improves on its fixed-127 control by **7.92–8.37 percentage points**. However, early Value layers account for much of the gain, while some late Value records prefer the fixed control. A fixed-127 baseline does not separate stable layer scaling from prompt-specific information. A stronger frozen TRAIN-derived per-layer/K/V/channel control is essential before attributing the gain to context adaptation.

## The remaining gap is predominantly metadata

The median representation stores approximately these bits per original BF16 value:

| Component | Bits/value |
|---|---:|
| Original sign and seven fraction bits | 8.000 |
| Encoded exponent corrections | 2.804–2.836 |
| Per-channel median anchors | 0.125 |
| Two segment-offset arrays | 0.25098 |
| JSON record header | 0.05920 |

The CPU codec's exponent payload already costs a similar **2.798–2.805 bits/value**. Its lower-byte Deflate saves only about **0.03–0.07 bits/value**. Most of its advantage comes from smaller metadata and indexing, rather than a dramatically better exponent model. Adding a heavier entropy coder is therefore not the first priority.

Three improvements have different evidence levels:

1. **Compact start indices:** each 65,536-value record currently stores two 257-entry uint32 arrays, or 2,056 bytes. Store only the 256 starts as uint16; existing terminal-count metadata retains the totals. The maximum start is 65,280, even when the terminal count is 65,536. This reduces the two arrays to 1,024 bytes: **1,032 bytes / 0.78735 percentage points** of raw size. `scripts/probe_compact_exponent_index.py` verifies the transformation and exact BF16 restoration, including all-escape and overflow-boundary cases. This is an index proof, not an integrated production format or a GPU speed result. Any new version/descriptor bytes must still be charged.
2. **Compact record descriptors:** replacing the current JSON header with a hypothetical 32-byte binary header adds about **0.346 percentage points**. Together with compact indices, the byte projection is **30.68–30.88%** savings on these sampled records. A complete binary schema, validation and timings are still required; this is not a measured new codec.
3. **Encode anchor information more economically:** current anchors alone cost 0.78125 percentage points of raw size. That is a gross upper bound on their removal, not a promised gain. Reconstructing anchors from a small shared profile plus exact deltas could recover part of it, after charging the profile, deltas, indexes and first-use transfer.

## Which combinations to stop pursuing

The whole-symbol residual dictionaries reduce correction bits by less than their ID costs. Across the six TEST cases, 16/time saves only 0.476–0.483 correction bits/value while IDs cost 0.5; 64/time saves 0.718–0.732 while IDs cost 0.75. Headers and state widen the loss. Even deleting all dictionary IDs, decoder state and excess headers would leave the best such recipe below the actual median exponent method.

The stateless per-record oracle adds only **0.001–0.011 percentage points**. It selects median exponent on 833/864 TEST records and fixed127 exponent on 31/864. No dictionary or whole-symbol recipe enters the unrestricted per-record minimum. The measured candidates therefore provide very little evidence for a costly adaptive selector.

Larger time groups help the midpoint recipe because they amortize width metadata: saving 0.4375 width bits/value costs only 0.191–0.217 extra correction bits. This adds 1.38–1.54 percentage points and wins every measured record. This is a useful packing insight, not evidence that the predictor became more accurate.

## The stronger mathematical representation to investigate

Move the next learned component to **the small recipe**, instead of adding a dictionary ID to every eight KV values. For each exponent:

\[
e_{l,b,t,c}=a_{l,c}+d_{l,b,c}+r_{l,b,t,c}.
\]

Here `a` is a compact shared channel profile, `d` exactly reconstructs the local tile anchor from that profile, and `r` is the existing exact per-value exponent correction. The original sign/fraction byte is retained. Integer addition reconstructs the exponent; no floating-point projection or approximate KV reconstruction is required.

Test frozen TRAIN profiles first. Then test a profile derived from the request's prefill, stored once and referenced by independently restorable pages. Each page must depend only on the small profile bank and its own bytes, never on another dense reconstructed KV page. All profile state and identifiers must be charged. A request-derived bank must be immutable or explicitly versioned so later updates cannot change old pages.

This separates a large **statistical domain** from a small **restoration unit**. It gives the original partial-capacity motivation a precise target: minimize total bytes while bounding the dense output tile, scratch space and shared-state dependencies. It is a hypothesis, not an established novelty claim.

## Where the novelty boundary actually lies

The [ExANS implementation report](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache) already describes exact BF16 exponent centering, cache-derived center tables, independent block records, entropy coding and GPU reconstruction. Its authors report 33–34% savings on their H100 workload; those results are not comparable directly with our A4500/model/protocol. Prefill metadata and indexed reconstruction alone cannot support our novelty claim.

[SplitZip](https://arxiv.org/html/2605.01708v2) already has exact exponent coding with sparse escapes and chunk-local GPU reconstruction. Its metadata ablation explicitly shows that fewer metadata bytes can slow decoding, so index changes must be timed as well as counted. [PackKV](https://arxiv.org/html/2512.24449v2) already combines independent compressed KV blocks with fused consumption, but begins with lossy quantization. [AlignPred](https://arxiv.org/html/2610.17970v1) combines alignment, prediction and error-bounded residual coding for persistent caches. [ZipServ](https://arxiv.org/html/2603.17435v1) demonstrates exact fused tile reconstruction for model weights, not KV attention. [VeriCache](https://arxiv.org/html/2605.17613v1) already reserves bandwidth and HBM, but reloads full KV for exact verification.

A defensible research hypothesis is **exact compact-resident KV attention with a hard materialization-memory bound**, using prefill-built statistical state while requiring only a small dense tile at a time. The contribution must be a demonstrated improvement in the savings/working-set/latency tradeoff, compared with those mechanisms. Shared metadata, fusion or memory credits individually are established ideas. The focused literature search does not prove the complete proposed combination is novel.

## What must be measured next

1. Integrate compact starts and a complete binary descriptor; retain exact random reads and compare decode costs.
2. Compare identical coding/index layouts using fixed127, frozen layer/K/V centers, frozen channel profiles, local tile medians and shared prefill profiles with exact anchor deltas. Fit learned choices on TRAIN, use DEV for selection and freeze before TEST.
3. Collect compact TRAIN/DEV/TEST diagnostics: exponent-error histograms, anchor-delta distributions and constant/biased sign-fraction bitplanes. These statistics are absent from the uploaded reports. Test actual coded bytes, with fallback and indexes, rather than interpreting a histogram entropy as a codec result.
4. Add realistic long-context prompts and full-cache correctness at feasible lengths. These three synthetic fixtures establish consistency within this experiment, not generality across workloads.
5. Implement attention-side consumption before claiming partial-capacity admission. Bound compact residency, state, staging, reconstructed tiles, codec scratch and attention workspace across concurrent requests. Processing can overlap page arrival, but exact attention must eventually consume every required context page before completing its result.

The current eager decoder is not fast enough to establish a transfer-latency win: median packet transfer plus reconstruction costs about **8.20–8.28 ms/MiB**, versus **0.77–0.83 ms/MiB** for the raw packet path. Bulk pinned raw copies cost only **0.046 ms/MiB** under a different buffer protocol. A fused decoder must be evaluated on the same hardware and transfer protocol. Exact KV operands also do not automatically guarantee identical logits if a new attention kernel changes reduction order; validate the two properties separately.

At the current correction rate, `8 + 2.8` payload bits already imply about 32.5% savings before metadata. Reaching 40% would require approximately 1.2 further bits/value from the unchanged lower byte or additional conditional exponent structure, even before metadata. This is a useful experimental target, not an information-theoretic impossibility or a promised result.
