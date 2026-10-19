# A4500 review: exact neural probability coding

**Both neural modes now reconstruct real Qwen KV exactly and pass complete 128-token DEV continuation checks. Neither establishes 40% reduction or a restoration speedup.** The shared model is the useful neural candidate to investigate further: it uses much less state than the per-layer collection and produces a slightly smaller stream in this smoke. Its estimated single-context saving still loses to its matched frozen control after state is counted.

The follow-up [shared context-position study](qwen_neural_positions_review.md) is now reviewed: four positions per context remove the prefix-only limitation, and the 8K state-inclusive estimate beats its matched control by 2.69 percentage points. The measured rates below remain the original smoke's results.

Results: [package at `460c236`](https://github.com/Kibnelson/kv_reconstruction/tree/460c2366ddf383d060417f32eb14da434b99e45a/benchmark_results/qwen_neural_probability_smoke_v2_a4500). The [reproducible audit](../scripts/review_neural_results.py), [validation](../benchmark_results/qwen_neural_review_20261019/validation.json) and [analysis](../benchmark_results/qwen_neural_review_20261019/analysis.json) record the checks and arithmetic. This review audits submitted artifacts and matching source; it is not an independent Qwen GPU rerun.

## What the smoke actually measures

The run uses pinned Qwen3-4B-Instruct-2507, BF16, 36 layers, eight KV heads and head dimension 128 on RTX A4500. The environment reports Torch 2.14.0/CUDA 13.0, Transformers 5.17.0 and NumPy 2.5.2. Three TRAIN fixtures provide 512 tokens each. The 35 per-layer networks receive 200 updates each; the shared network receives 7,000 updates, matching the aggregate update count. Both have hidden width 512.

The short-DEV fix works: all five requested TRAIN/DEV/TEST prompt budgets pass tokenizer preflight before weights and fitting. DEV covers the entire 128-token prefix. Each neural mode checks 288/288 layer/tile pairs, including actual neural pages rather than a fallback representation. Both continued generation checks pass two steps with finite, bit-identical logits, identical tokens and identical appended KV. The maximum recorded logit difference is zero.

**TEST sizing covers only positions 0–15 of one 2,048-token fixture.** It checks one aligned 16-token chain across all 36 layers: 36 of 4,608 possible pairs, or 0.78125% coverage. All fixtures share a system/chat prefix, so these positions are especially weak evidence of held-out document-body behavior. We have not independently tokenized the prefix here to establish the exact number of overlapping tokens, nor compared TRAIN and TEST KV words directly. The sample cannot establish whole-context compression or generalization to different bodies and positions.

## Matched new-study results

The actual sampled raw payload is 2,359,296 bytes, or 2.25 MiB. The full raw 2K cache is 288 MiB. Full-cache estimates multiply sampled container density by 128 and add complete state once; no full 2K compressed container was measured.

| Method | Actual sampled container | Sample reduction | Charged state | Estimated net reduction at 2K | Sample decode wall time |
|---|---:|---:|---:|---:|---:|
| Matched frozen exponent control | 1,744,259 B | 26.0687% | 0.0815 MiB | **26.0404%** | **80.067 ms** |
| Per-layer neural | 1,656,754 B | 29.7776% | 456.8317 MiB | **-128.8445%** | 7,642.371 ms |
| Shared neural | 1,642,375 B | **30.3871%** | **14.1560 MiB** | **25.4718%** | 7,466.118 ms |

The shared model improves sampled payload reduction by **4.3184 percentage points** over the control, but loses **0.5686 points** after its state is charged to the estimated full 2K cache. Its sampled stream is 14,379 bytes smaller than the per-layer stream, while its complete charged state is about 32.3 times smaller. That is a result on this small sample, not evidence that sharing will always predict better.

The per-layer estimate means total retained bytes are approximately 2.29 times raw KV, rather than a saving. Sharing its weights across many contexts could amortize that cost, but this implementation has not measured concurrent serving. Under the same unvalidated sample density, shared neural would beat this control at two compatible 2K contexts; the per-layer collection would need 43. Neither can reach 40% simply through amortization: their measured payload asymptotes are only 30.39% and 29.78%.

As a separate DEV diagnostic, the complete 128-token containers save 31.7253% per-layer and 32.0073% shared before state. Those are actual full-DEV packet rates; they are not fresh TEST results and must not replace the TEST score.

## Why the current decoder is slow

For the shared model's independent 2.25 MiB sample decode:

| Stage | Recorded time |
|---|---:|
| CPU arithmetic decoding | **7,419.686 ms — 99.378%** |
| GPU neural inference | 21.418 ms |
| Previous KV staging to GPU | 3.077 ms |
| Probability parameters returned to CPU | 3.619 ms |
| Layer-zero anchor restoration | 2.421 ms |
| Remaining validation/dispatch | approximately 15.895 ms |

Shrinking or speeding up the network alone would barely change this reference path's total latency. The independent neural decode is about 93 times slower than the new study's frozen control under this functional protocol. These timings are not the earlier conditional study's repeated device-resident microbenchmarks.

One prediction pass stages 2.1875 MiB of prerequisite KV to the GPU and returns 8.75 MiB of FP32 means/scales to CPU. The returned probability parameters cost four times the corresponding raw BF16 words. This is an exact representation experiment, not a working migration optimization.

The run lasts 678 seconds, with 50.9 seconds of fitting. Repeated full DEV restorations deliberately dominate the correctness workload. The roughly 59-second DEV restore times are not measured production TPOT.

## What the bit ledger tells us

| Shared-model quantity | Bits per value | Scope |
|---|---:|---|
| Ideal integrated probabilities | 10.770500 | Conditional layers 1–35 only |
| Integer-CDF probabilities | 10.770499 | Same symbols; rounding can slightly favor this sample |
| Actual arithmetic streams | 10.772586 | Same layers, including stream framing |
| Complete sampled container | 11.138068 | All layers, anchor and outer/page metadata |
| Estimated full 2K total with state | 11.924514 | Prefix-density extrapolation with full state |
| Target for 40% total reduction | **9.600000** | All retained bytes |

The arithmetic stream costs only about **0.0021 bits per conditional value** beyond its integer probability cost. A better entropy coder is needed for speed, but coding inefficiency is not hiding several percentage points of size savings here. The probability model itself must improve.

All non-payload headers account for approximately 0.349 bits per sampled value, or 2.18 percentage points of raw size. Larger tiles and compact descriptors can reduce that charge. Even eliminating it completely would not reach 40% on this sample. The current density has a 1.538-bit gap before state and a 2.325-bit gap after state to the 9.6-bit target.

## Comparison with the existing conditional approaches

The earlier study averages six TEST cases at 2K/8K, uses 64-token tiles and samples different layer/tile addresses. The neural smoke uses one fixture and only a 16-token prefix. The two frozen controls use different sampled addresses, tile sizes and framing. Treat the following as an inventory of results, not a paired ranking or an additive savings table.

| Earlier conditional candidate | Mean estimated reduction including state | What was retained |
|---|---:|---|
| Raw pair | -0.165% | Raw words and framing |
| Frozen exponent | 29.843% | Final KV exponent tiers |
| Cost-trained final KV | 30.380% | Final KV with fitted bases |
| Fast pre-RoPE | 30.942% | Pre-RoPE tiers and replay |
| Exponent Huffman | **32.904%** | Actual device entropy stream |
| Exponent Huffman + sign | **33.390%** | Actual combined exponent/sign streams |
| Tier exponent + sign | 31.330% | Actual tier/sign combination |
| Zero full-word correction control | -0.845% | Exact full-word corrections |
| Mean predictor | 30.844% | Tier fallback on all measured pages |
| KV anchors r8 | 30.304% | Tier fallback on all measured pages |
| KV anchors r32 | 28.839% | Tier fallback on all measured pages |
| Prefill latent r8 | 29.658% | Tier fallback on all measured pages |
| Prefill latent r32 | 26.362% | Tier fallback on all measured pages |

The new neural test supplies something those failed point-prediction variants did not: actual learned probability pages reconstruct real Qwen values exactly. It does not yet show a better total representation. Combining its percentage with the exponent/sign percentage would double count the same source information; a real combined stream and matched ablation are required.

## Verification limits

All seven Git blobs and six package-manifest entries match their identities. The 196,895-byte gzip restores the expected 2,187,145-byte report. All 168 recorded Python source hashes and the embedded launcher payload match the reviewed implementation. Source, frozen configuration, TRAIN provenance, component sums, complete-state arithmetic and reported exactness reconcile.

The anchor artifact is delivered and its identity is verified. The 110 declared neural artifact files are omitted from the reports package; their reported sizes, identities, metadata and total charges reconcile, but their actual contents were not independently verified here. The run's source reloads saved models before use. Complete cache containers are serialized and reopened **in memory**; the benchmark does not demonstrate a file-backed, separate-process replay from the delivered bundle.

The 128 KiB two-tile dependency metric excludes returned cache tensors, staged inputs, probability arrays, coder workspace and model state. Dependency depth reaches 35. Measured GPU peaks above pre-prepare allocation are approximately 458.1 MiB per-layer and 15.5 MiB shared, including state preparation and functional work; these are not isolated decoder workspace measurements or evidence of admission under a GPU budget.

The nested model metadata still contains a lower-level note saying training budgets are unequal. The actual runner budgets and update logs correctly show 7,000 updates per mode; that stale descriptive note does not change the arithmetic.

## Next controlled experiment

First test **shared-only**, keeping the 200-step-per-layer aggregate budget while improving position coverage. This isolates the sampling question before adding network capacity or longer training:

```bash
python qwen_kv_gpu_bench.py --neural-probability-study \
  --device cuda --modes shared --width 512 --steps 200 \
  --train-tokens 512 --dev-tokens 128 \
  --lengths 2048 8192 --cases 3 \
  --tile-tokens 64 --max-pairs 144 \
  --output runs/qwen_neural_shared_positions
```

This measures four all-layer tile chains per TEST case, spanning first, interior and last positions; complete DEV correctness remains separate. It still estimates full-cache size and uses the slow CPU reference. Compare its own matched frozen control first. A strict comparison with the 13 conditional candidates needs identical positions, source tensors and framing costs.

After representative rates are known, the useful incremental options are:

| Experiment | Why this data motivates it | Cost that must be charged |
|---|---|---|
| **Actual hidden-derived features condition probabilities** | Previous point predictors failed because exact full-word corrections stayed wide. A retained feature can still help predict uncertainty without predicting the exact value. Test the existing prefill hidden-feature idea as an entropy conditioner. | Eight BF16 features per token/layer cost 0.0625 bits per KV value at this geometry, before headers and additional state. Conditional bit savings must exceed the complete extra cost. |
| **Page location/scale calibration** | A fixed shared network may miss page-specific distribution changes. Freeze it and transmit quantized adjustments to its predicted means/scales. | Two 8-bit coefficients per head, separately for K and V, cost 32 bytes per layer/page: 0.001953 bits/value at 64 tokens, before framing/state/search. This is a produced-KV-only metadata control, not evidence that upstream hidden access helps. |
| **Predict V after decoding current K** | Current networks explicitly do not condition V on current-layer K. Its exact decoded values may provide extra conditional information. | Extra network/state/MACs and a second prediction stage; no additional retained per-token feature payload is required. |
| **Parallel entropy streams and compact descriptors** | CPU arithmetic dominates time, while per-page descriptors cost about 2.10 percentage points at 16 tokens. | Substream headers, coder tables, discrete-probability state and peak workspace. Larger pages trade lower framing cost against coarser materialization and latency. |

Keep architecture, coefficient grids and feature selection on TRAIN/DEV, then freeze them before a fresh TEST comparison. Each change needs the same-coder ablation and complete state ledger. A native parallel entropy path and reproducible discrete-probability contract are separate speed work. None of these proposed changes has an additional measured saving or established novelty yet.
