# Four conditional KV experiments: exact rates and decision metadata

The [A4500 frontier review](qwen_frontier_gpu_review.md) measured 30.76% mean estimated savings, including complete state. The remaining gap to 40% is 1.48 bits per original BF16 value. This preset tests four ways to change that payload. It includes all outcomes and explicit controls; no new Qwen GPU savings are claimed before the next target-machine run.

## Fixed comparisons

Run `--producer-study --producer-preset conditional`. There are **13 fixed candidates**:

| Track | Candidate | What is retained and reconstructed |
|---|---|---|
| Previous control | `producer_frozen_exponent` | Existing final-K/V frozen exponent representation |
| Previous control | `frontier_cost_final_raw8` | Cost-trained final-K/V bases and compact indices |
| Fast source control | `conditional_fast_pre_rope` | Pre-RoPE K and V, frozen cost bases, compact indices; no head-shift or plane search |
| Idea 1 | `conditional_huffman_exponent` | Actual frozen Huffman stream for exponent residual symbols; raw sign/mantissa bytes |
| Ideas 1 + 2 | `conditional_huffman_exponent_sign` | Huffman exponents plus separately coded sign transitions and exact seven-bit mantissas |
| Idea 2 ablation | `conditional_tier_sign` | Existing exponent tiers plus the same separate sign coding |
| Raw transfer control | `conditional_raw_pair` | Actual final K/V words, no model/replay state; decoder clones output tensors |
| Correction control | `conditional_raw_fullword` | Zero prediction with exact full-word corrections to pre-RoPE K/V |
| Prediction control | `conditional_mean_predictor` | TRAIN means with exact corrections; no retained features |
| Idea 3 | `conditional_anchor_r8`, `conditional_anchor_r32` | Small subsets of actual joint K/V coordinates, shared prediction weights, exact corrections to remaining coordinates |
| Idea 4 | `conditional_latent_r8`, `conditional_latent_r32` | Small BF16 features derived from actual normalized hidden states during prefill, shared K/V prediction weights, exact corrections |

The new rank variants request 8 or 32 features, capped by actual TRAIN geometry. These are fixed candidates, rather than ranks selected using TEST. `--rank` still controls the earlier producer models fitted for compatibility; it does not change this new rank grid.

Every new source representation uses actual **normalized K before RoPE plus produced V**. The decoder restores that source and replays RoPE. The encoder compares the replay to actual final K; exact matches need no correction payload, otherwise it stores full-word coset corrections. The raw transfer control directly stores final K/V and needs no replay. The exact-correction guarantee applies to the original BF16 words, including their signs and mantissas.

## How the four ideas are implemented

**1. Exponent entropy coding.** The source-specific frozen TRAIN channel bases remain unchanged. Instead of reserving two primary bits and adding tier escapes, code the complete zigzag exponent residual with a canonical Huffman book. K and V have independent global residual books. Every alphabet symbol has positive fitting mass; additive smoothing increases only as needed to bound code lengths to 15 bits. Actual smoothing, code lengths and TRAIN coding costs are reported. This is a measured bitstream, rather than a Shannon-entropy estimate. It is a first pooled probability model, not a complete conditional neural entropy model.

**2. Sign coding.** Store first-row sign seeds for each independent tile. XOR subsequent signs with the preceding row in the same channel; pack eight transition bits into symbols and Huffman-code those symbols with separate K/V TRAIN books. The decoder uses prefix parity to recreate signs. Seven mantissa bits are retained exactly. Seeds, padding, checksums and descriptors count. Transitions reset at every record boundary, including sampled gaps. The tier/sign candidate separates this idea's effect from exponent Huffman coding.

**3. KV anchors.** Fit a joint target from actual pre-RoPE K and V. Greedy pivoting selects informative TRAIN coordinates. A small ridge predictor reconstructs the whole joint target from these anchor words. The packet retains the anchor bits once; their coordinates are omitted from correction streams. Other coordinates receive exact full-word corrections. Anchors and dependencies stay inside an independent record; there is no cross-record reconstruction chain.

**4. Producer features.** During prefill, take the actual normalized hidden states already observed by the hooks, center them and project into a small TRAIN-fitted supervised subspace. Quantize the resulting features to BF16 and retain them in the packet. The decoder uses only these retained features and shared weights to predict both K and V. It receives neither the original hidden tensor nor the model's full projections. The supervised fit uses a small randomized cross-covariance sketch, avoiding a full hidden-by-KV singular-value decomposition.

Anchor/latent weights start with real-value ridge MSE. A TRAIN-only gate evaluates shrinkage factors `[0, 0.5, 0.75, 1]` using BF16 correction-byte proxies and chooses the least costly per layer. Proxies count feature words, separate K/V record boundaries, five-bit correction widths, byte padding and omitted anchor coordinates. All candidates and the selected factor are logged. This is a NumPy TRAIN proxy; actual GPU rounding, complete packets and state determine final measured rates. Headers and same-shaped model state are not part of the shrinkage objective.

For mean/anchor/latent recipes, the encoder compares the **complete packet bytes** of prediction-plus-corrections, frozen exponent tiers, and raw source words. It retains the smallest packet, including headers and replay corrections. Full predictor state remains charged even if every selected record falls back. Logs retain the rejected candidate sizes and prediction quality, so fallback cannot hide an unsuccessful predictor. Exponent/sign candidates emit their actual requested streams directly, including expansions; they are not silently replaced with a better control.

## State, numerical replay and timing

Canonical artifacts include arrays, probability/lookup tables, metadata, CRC and SHA identities. TRAIN inputs and supplied provenance are hashed before and after fitting. The source manifest, configuration, model identities, all candidates and decoder artifacts are frozen before TEST capture. No diagnostic or TEST result refits a predictor or a table.

Headline state charges conservatively use the larger of **complete serialized artifacts** and **all resident numerical tensors**. This includes encoder-only latent projection/centering weights, even though they are not needed after features have been retained. State reports separately expose those bytes and required decoder numerical bytes. Thus the experiment can identify a later state-separation opportunity without pretending that a minimal decoder-only artifact has already been implemented.

Float prediction and RoPE are bound to the recorded eager backend, software/build/device identity, CPU thread count where relevant, deterministic settings, autocast settings and float32/TF32 controls. Encode/decode reject a changed contract. Corrections computed against one floating-point predictor must not be reused with a different predictor implementation. Cross-hardware replay equivalence is not established.

The Huffman payload path runs on the selected Torch device. The experimental decoder peeks codewords at bit positions and uses parallel pointer doubling to locate symbol boundaries. It uses scratch proportional to encoded bit count and logarithmically many pointer passes. No CPU entropy decoder or per-word host transfer is hidden in GPU decode. Scalar validation synchronizations and actual integrity checks remain timed. GPU speed and workspace must be measured, rather than inferred from the smaller stream.

All methods use identical selected records and repeated-timing ordinals. The default timing sample is 16 pairs / 4 MiB raw, with a warmup and three repeats. State preparation, input staging, encode, decode and pageable packet transfer plus decode are reported separately. Encode includes bounded diagnostic aggregate calculations; their cost is explicitly recorded. Per-record allocation peaks separately measure encode and decode; decode's extra allocation includes materialized outputs and excludes already resident input/state. CPU allocator peaks are not inferred.

By default, hooks encode/check every candidate before the attention callback returns. They first export selected producer features to CPU and stage them back to the device. Repeated codec timings are post-capture replays. Instrumented prefill includes all candidates, copies and checks, while full raw KV remains resident. This is not a fused prefill encoder or compact-resident attention, and demonstrates no concurrency/capacity, TTFT or TPOT gain.

## Logs for the next decision

- `results.json` contains complete outcomes, actual field bytes, sampled addresses, inline/replay checks, allocation peaks, state identities and generation results. Failures retain tracebacks.
- `decision_summary.json` retains every method, matched-control deltas, component-bit ledgers, per-layer rates, aggregated scalar/histogram diagnostics and separate TEST means. It makes no TEST-fitted selection policy.
- Candidate diagnostics retain feature/anchor costs, exact-word/sign/exponent/bit agreement, residual-width histograms, complete candidate bytes and selected/fallback modes. Entropy diagnostics retain actual symbol and code-length distributions.
- TRAIN artifacts record symbol counts, smoothing, code lengths, requested/effective ranks, predictor state, all shrinkage proxy costs and their fit objectives.
- Per-head `conditional_diagnostics` records source-specific tier payload costs, lower-byte distribution statistics, record-local sign changes and observed matched-channel K/V correlations. These are descriptive, with sparse-sample warnings; their costs omit headers/state and cannot claim a codec gain.
- Full existing frontier information diagnostics remain available, including reliable versus sparse conditional estimates and the independently framed CPU lower-byte controls. CPU controls are not GPU codecs.
- Every registered candidate gets complete DEV continuation checks by default: finite, bit-identical logits, generated tokens and appended KV. An explicit `--generation-steps 0` is a skip, not a pass.

The next review can therefore distinguish a bad probability model, insufficient predictor rank, expensive feature/state payload, wide/outlier corrections, a fallback-heavy representation, and an overly expensive device decoder.

## Authoring validation

The [validation record](../benchmark_results/conditional_validation_20261019.json) reports 236 tests with 11 hardware-dependent skips on Torch 2.11 / Transformers 4.57.6, 46 conditional tests with two CUDA skips on Torch 2.14 / Transformers 5.20.1, and 12 portable-build/packaging tests. The [standalone tiny-model run](../benchmark_results/conditional_tiny_20261019/summary.md) completes all 39 method rows and all 13 two-step continuation checks. Source, freeze, packet-component and complete-state identities were checked; its full reports and model/state artifacts are retained as verified gzip and binaries. These are random-model CPU functional checks, not pretrained Qwen compression results or GPU execution evidence.

## Run on the A4500

First pull and run one short TEST case. Each output directory must be fresh:

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main

python qwen_kv_gpu_bench.py --producer-study --producer-preset conditional \
  --device cuda --lengths 2048 --cases 1 \
  --train-tokens 512 --dev-tokens 512 --chunk-tokens 512 \
  --tile-tokens 64 --max-tiles 72 --sample-pairs 16 \
  --repeats 3 --generation-steps 2 \
  --output runs/qwen_conditional_smoke
```

After a completed smoke run, the matched full comparison uses three topics and both lengths:

```bash
python qwen_kv_gpu_bench.py --producer-study --producer-preset conditional \
  --device cuda --lengths 2048 8192 --cases 3 \
  --train-tokens 512 --dev-tokens 512 --chunk-tokens 512 \
  --tile-tokens 64 --max-tiles 72 --sample-pairs 16 \
  --repeats 3 --generation-steps 2 \
  --output runs/qwen_conditional_4ideas
```

TEST samples 72 pairs, so full-cache rates are estimates. DEV uses full coverage independently. Increasing `--max-tiles` can broaden TEST coverage, at additional runtime. GPU execution and pretrained Qwen savings must be established on the target machine; local random tiny-model checks establish functionality only.

Package the chosen completed run without adding oversized raw JSON to Git:

```bash
python scripts/package_results.py runs/qwen_conditional_4ideas \
  --destination benchmark_results/qwen_conditional_4ideas_a4500
git add -f benchmark_results/qwen_conditional_4ideas_a4500
git commit -m "Add A4500 four-idea conditional KV results"
git push origin main
```

For the smoke run, substitute `qwen_conditional_smoke` in both paths. The helper preserves the full report as verified gzip and includes `decision_summary.json` plus calibration/state artifacts. Model weights and raw hidden/KV captures are excluded.

For a no-download functional run:

```bash
python qwen_kv_gpu_bench.py --producer-study --producer-preset conditional \
  --tiny --device cpu --lengths 64 96 --cases 1 \
  --generation-steps 2 --output runs/conditional_tiny
```
