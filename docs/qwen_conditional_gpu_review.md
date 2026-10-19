# A4500 review: the four conditional KV experiments

The best storage result in this run is **33.3898% mean estimated reduction**, including complete state, from exponent Huffman coding plus separate sign coding. Exponent-only Huffman reaches **32.9038%** with about half the decoding cost. The KV-anchor and prefill-feature point predictors always fall back to the tier codec. No candidate reaches 40%, and this eager implementation does not demonstrate faster migration or capacity-bounded serving.

Results: [pushed package at `2fc0c9f`](https://github.com/Kibnelson/kv_reconstruction/tree/2fc0c9f20c3607fad89a94567ae5c96de51f8a23/benchmark_results/qwen_conditional_4ideas_a4500). The [machine review](../benchmark_results/qwen_conditional_review_20261019/analysis.json) and [validation](../benchmark_results/qwen_conditional_review_20261019/validation.json) retain the arithmetic and protocol evidence.

## What was tested and verified

The run uses pinned Qwen3-4B-Instruct-2507, BF16 KV, 36 layers, eight KV heads and head dimension 128. Each 64-token layer tile contains a paired K/V payload of 256 KiB. The three synthetic TEST fixtures run at 2,048 and 8,192 tokens. Results from different lengths of one fixture remain correlated.

All 32 files in the share manifest were downloaded and checked against their sizes and SHA256 identities. The 11.34 MB gzip restored the complete 248,524,767-byte JSON with its expected hash. All 158 benchmark-source hashes match the embedded source in published release `efc74fe7477bcf524752d06d3aafe537a026f867`; its payload SHA256 is `73627c04d9abac02e887cf69fdb384c59d5acce5e202e3aa693303945c80a959`.

The recorded correctness evidence passes independent checks:

- All 91 method/case rows succeeded: 13 candidates on DEV and six TEST cases.
- 9,360 primary pair checks and 9,360 inside-forward pair checks were exact. Inline packet bytes/components agree with post-capture replay.
- All 13 complete 512-token DEV continuation checks passed. Their 26 generation steps have finite, bit-identical logits, equal next tokens and identical appended KV.
- All 12 calibration models pass canonical format, CRC/SHA, source-identity and TRAIN-provenance checks. All 25 frozen artifact files match their identities.
- Packet components, outer framing, complete state charges, ordered sampling and repeated timing sample counts reconcile.

These are checks of the submitted logs and artifacts, not an independent GPU rerun. In particular, all five learned methods also select fallback on every DEV tile: their Qwen continuation checks validate fallback restoration, rather than a retained learned representation.

**TEST sizes are estimates.** Each candidate checks 72 of 1,152 pairs at 2K, and 72 of 4,608 at 8K. Repeated timing uses the same 16 pairs, 4 MiB of raw KV, for all candidates, with three repetitions. The complete raw cache remains resident during instrumented prefill capture.

This run reports **Torch 2.14.0/CUDA 13.0, Transformers 5.17.0 and NumPy 2.5.2**. Earlier frontier results used a different runtime. Use the controls below for matched claims; the previous 30.76% result is historical context rather than an isolated algorithm comparison.

## Matched storage and restoration results

The table gives arithmetic means across the six TEST cases. Savings charge full decoder state once per context; full-cache bytes are extrapolated from sampled records. Timings are synchronized microbenchmarks in milliseconds per raw MiB.

| Candidate | Reduction including state | Device-resident decode | Packet transfer + decode |
|---|---:|---:|---:|
| Raw pair | -0.165% | 0.226 | 0.668 |
| Frozen exponent control | 29.843% | 7.363 | 8.375 |
| Cost-trained final-KV control | 30.380% | 8.172 | 9.303 |
| Fast pre-RoPE tier control | 30.942% | 8.916 | 9.991 |
| Exponent Huffman | **32.904%** | **10.655** | **11.521** |
| Exponent Huffman + separate sign | **33.390%** | 21.804 | 22.963 |
| Tier exponent + separate sign | 31.330% | 20.154 | 21.553 |
| Full-word zero-prediction control | -0.845% | 4.783 | 5.544 |
| Mean predictor with fallback | 30.844% | 8.974 | 10.047 |
| KV anchors, rank 8, with fallback | 30.304% | 8.788 | 9.839 |
| KV anchors, rank 32, with fallback | 28.839% | 8.827 | 9.882 |
| Prefill latent, rank 8, with fallback | 29.658% | 8.851 | 9.917 |
| Prefill latent, rank 32, with fallback | 26.362% | 8.935 | 10.024 |

Exponent Huffman adds **1.9616 percentage points** over the matched fast tier control. Adding sign coding is an actual tested combination: it adds another **0.4859 points**, while increasing decode cost by approximately **2.05x** relative to exponent-only Huffman. The best candidate ranges from 33.2482% to 33.5552% across TEST cases.

The sign-saving increment is not additive with the separate tier-plus-sign result. The measurements above already include their complete combination, with its own headers, seeds and state.

## Why the metadata predictors did not win

Each learned mean, anchor and latent method chooses tier fallback on **432/432 TEST records**, plus **288/288 DEV records**. No learned packet is retained. Their numerical payload components match the fast tier control; additional headers and unused-but-charged predictor state reduce final savings. Their roughly 8.8 ms/MiB decode times consequently measure fallback decoding, not cheap neural or latent reconstruction.

The predictors nevertheless capture some structure:

| Predictor | TEST sign matches | Exact word matches | Correction groups requiring 16 bits | Charged state, decimal MB |
|---|---:|---:|---:|---:|
| Mean | 66.39% | 0.12% | 99.993% | 0.706 |
| KV anchors r8 | 69.34% | 0.57% | 99.973% | 3.317 |
| KV anchors r32 | 72.37% | 1.78% | 99.968% | 10.391 |
| Prefill latent r8 | 71.98% | 0.21% | 99.952% | 6.435 |
| Prefill latent r32 | 75.53% | 0.26% | 99.914% | 22.361 |

The rejected learned packets cost about **16.13 bits/value** for mean/anchors, **16.19** for latent r8 and **16.37** for latent r32, compared with roughly **11.04** for the tier packet before its complete state charge. There is no hidden winning layer: per-layer predicted candidate sizes are approximately 44–49% larger than tier packets.

The correction format groups 64 full BF16 word differences under a common width. Large bit-pattern differences, including sign errors, can make an entire group require 16 bits. Better real-value or sign predictions therefore need not reduce this exact correction stream. This width saturation is present on TRAIN as well, so state pooling or more data alone does not address the demonstrated format failure.

The TRAIN byte proxy also gives a weak optimization signal: all four shrinkage choices tie exactly in 32/36 anchor-r8 layers and 29/36 latent-r8 layers. Selecting alpha 1 in these cases reflects the tie-break, not a measured coding improvement. Higher rank improves predictions but increases state, and the rank-32 predictors show worse TRAIN-to-TEST sign generalization.

Only **384 selected training token rows per layer** enter these fits: two 64-token tiles from each of three TRAIN captures. Increasing `--train-tokens` alone while retaining the same bounded selected-tile budget does not necessarily increase the fitting examples. Future data-budget experiments must record and vary the actual number of fitted rows.

These results do not prove that prefill metadata is uninformative. They show that these small point predictors and this grouped full-word correction format fail to turn its information into savings.

## What still stands between this result and 40%

The best candidate's mean ledger is:

| Component | Bits per original BF16 value |
|---|---:|
| Raw seven mantissa bits | **7.000000** |
| Exponent Huffman payloads | 2.631087 |
| Sign-transition Huffman payloads | 0.845285 |
| Sign seeds | 0.015625 |
| Headers, checksums, complete state and framing | 0.165639 |
| **Total** | **10.657637** |
| **40% target** | **9.600000** |

We still need **1.057637 bits/value**, equal to 6.6102 percentage points of raw storage or 9.924% of the current stored size. Mantissas contribute 65.68% of the current size. Holding everything else fixed, reaching the target would require reducing their average rate from seven to about **5.942 bits**. Alternatively, exponent coding alone would need to improve from 2.631 to 1.573 bits, about a 40.2% reduction in that component.

Even removing every counted header, checksum, state and framing byte would only reach about **34.425%** in this fixed layout. Smaller headers remain useful, but cannot by themselves close this gap.

Mantissas plus the current exponent payload already total **9.631087 bits/value**, exceeding the 9.6-bit target even if signs and all other costs were free. Some improvement to mantissa or exponent information is therefore necessary; sign/header polishing alone cannot reach 40% in this layout.

Per-head lower-byte marginal entropy is high: approximately 7.955 bits for pre-RoPE K and 7.914 for V. These finite-sample descriptive marginals are neither achievable-code guarantees nor lower bounds on conditional coding. They suggest looking for useful conditioning information, rather than assuming another independent lower-byte coder will recover a full bit.

An ideal TEST-only per-head choice among final, pre-RoPE and projection sources changes the measured tier payload by only about **0.027 percentage points** beyond always using pre-RoPE, before selector/replay costs. This is a diagnostic oracle, not a new codec; its gain cannot be added to the Huffman result.

## Restoration cost matters for the original capacity goal

The current eager Huffman decoder uses work proportional to compressed bit positions and pointer-doubling passes. Isolated decode allocations for one **0.25 MiB raw pair** are:

| Candidate | Peak extra allocation including output |
|---|---:|
| Raw pair | 0.250 MiB |
| Frozen exponent | 5.254 MiB |
| Fast pre-RoPE tier | 4.767 MiB |
| Exponent Huffman | 13.536 MiB |
| Exponent + sign Huffman | 15.579 MiB |

These exclude already resident state, input packets and staged producer tensors. The best storage codec currently needs about 62 times the raw pair size in extra allocation. That is directly relevant when little GPU capacity remains. Independent indexed tiles alone do not establish capacity-bounded materialization.

The observed pageable packet transfer-plus-decode path is slower than raw in this microbenchmark. Using the actual matched 16-pair timing samples, saved transfer bytes offset extra decoding only below about **31.75 MiB/s** for exponent Huffman versus raw. The sign increment versus exponent-only breaks even below about **0.388 MiB/s**. These are optimistic linear-link calculations with resident state: encoding, cold-state transfer, allocation differences, overlap, SSD behavior and serving effects are excluded. They are not a prediction of end-to-end serving performance.

Storage improvement and migration latency improvement therefore require separate acceptance criteria. No partial-capacity admission, high-concurrency scheduler, SSD cold-read benefit, compact-resident attention, TTFT or TPOT gain has been measured.

## Next experiments supported by the evidence

1. **Change the correction/statistical model before making point predictors larger.** Compare sign-separated, ordered-magnitude corrections with conditional probability coding. Charge the transmitted features, complete model state and every correction. Measure a forced learned DEV path as well as adaptive fallback so real-Qwen correctness of the learned branch is explicit.
2. **Run the already published fifth probability experiment.** It uses decoded previous-layer KV and a distribution over BF16 intervals, avoiding the maximum full-word correction width as its coding objective. Its shared/per-layer comparison also exposes state cost. It is a produced-KV experiment; it does not establish a prefill-hidden-metadata contribution, and its first entropy coder runs on CPU. Results remain pending.
3. **For speed and small materialization buffers, replace the eager bit-position decoder.** Test bounded independently indexed substreams or a fused decoder, reporting all restart offsets/padding and actual peak workspace. Keep raw and tier controls; a smaller payload with much larger restoration cost does not satisfy the original delay objective.
4. **Separate actual training data budget from TEST sampling.** Increase diverse fitted rows with TRAIN-only model selection and fresh held-out evaluation. More or longer documents are useful only if their additional rows enter the fit. The current fallback failure means this should accompany a better correction/probability format.

The confirmed increment is entropy coding over exact pre-RoPE reconstruction. The prefill-feature hypothesis remains open, with a concrete representation failure to fix. Neither collecting metadata during prefill nor these standard coding ingredients by themselves establish novelty.

## Reproduce the artifact audit

The audit is read-only and needs the source that actually produced the run. The current runner has since gained the separate neural dispatch, so checking historical hashes against current source should fail. From a repository with the release in its history:

```bash
study_source_dir=$(mktemp -d)
git show efc74fe7477bcf524752d06d3aafe537a026f867:qwen_kv_gpu_bench.py > "$study_source_dir/runner.py"
python "$study_source_dir/runner.py" --extract-source "$study_source_dir/source"
python scripts/review_conditional_results.py \
  benchmark_results/qwen_conditional_4ideas_a4500 \
  --source-root "$study_source_dir/source" --verify-models \
  --output runs/qwen_conditional_review
```

Omit `--verify-models` for a standard-library audit of file/source identities, gzip integrity, recorded provenance, byte accounting, sampling and continuation checks. The optional flag additionally loads all 12 canonical calibration models with their NumPy/Torch dependencies. Neither mode reruns the GPU or requires the pretrained checkpoint.
