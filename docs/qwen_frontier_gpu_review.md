# A4500 frontier review: 30.76% savings; the remaining gap is in the payload

Result commit: [`c287ce67842deb4bca1905377714ac71f498343f`](https://github.com/Kibnelson/kv_reconstruction/commit/c287ce67842deb4bca1905377714ac71f498343f). Benchmark implementation: `20ae4a2c9799eab57f505a5b86781aa6e1574420`. The later packaging fix preserves the complete report in `results.json.gz`.

The run completed successfully on the RTX A4500 using the pinned Qwen3-4B-Instruct-2507 checkpoint. The best tested representation is **source-specific exponent coding of actual pre-RoPE K, compact indices, verified exact replay, and raw lower bytes**. It saves an estimated **30.7566%**, including complete decoder state, averaged over six TEST cases. This improves the matched frozen control by **0.9135 percentage points**. **The 40% target was not reached.**

The meaningful producer result is that encoding K before RoPE improves savings by **0.3738 pp** against the corresponding final-K packet. Compact indices contribute more than the new predictor objective. The attempted lower-bit encodings all fall back, and head-shift search costs far more time than its small size benefit justifies.

## Evidence and scope

- All 18 downloaded files match their pinned Git blob hashes. The compressed report is 4,802,867 bytes; its complete decompressed contents are 108,815,114 bytes and pass gzip CRC verification.
- All 146 recorded benchmark source hashes match. Source/freeze digests, configuration, candidates, calibration identities and all 12 calibration/state artifact hashes agree.
- All **112 method rows** pass: 16 recipes on one DEV and six TEST cases. The audit recomputes packet components, outer framing, full state charges, effective bits and full-cache estimates without discrepancies.
- Each execution track reports **11,520 exact paired-packet checks**: inside attention forward and after capture. Matching addresses, component sizes and state identities agree. Hooks are removed and forward-stage audits pass.
- Five complete 512-token DEV continuations pass two decoding steps with identical finite logits, tokens and appended KV. These are the five fixed representatives listed in the validation report, rather than separate generation tests for every recipe.

TRAIN consists of three separate 512-token fixtures. TEST uses three synthetic topics at 2,048 and 8,192 tokens; length variants are correlated. TEST checks 72 paired tiles out of 1,152 at 2K (**6.25%**) and 4,608 at 8K (**1.5625%**). Full-cache savings are extrapolated from those samples; decoder state is charged once per complete cache. All methods use the same ordered pairs within this run. This is an audit of the submitted evidence, with no independent GPU rerun or checkpoint-weight attestation.

## Size and measured codec costs

Unweighted means over the six TEST cases follow. Times are milliseconds per **raw MiB**, rather than per encoded MiB. Repeated decode/restore timings use 16 pairs totaling 4 MiB raw, one warmup and three repetitions, with prepared decoder state. Encode time covers the larger sampled set and includes the search work.

| Representation | Estimated savings | GPU decode | Packet transfer + decode | Encode |
|---|---:|---:|---:|---:|
| Existing tile medians | 29.388% | 7.874 | 9.051 | 8.550 |
| Existing frozen exponents | 29.843% | **7.207** | **8.208** | **7.223** |
| Existing pre-RoPE coset | 29.207% | 9.732 | 11.005 | 12.054 |
| Cost-trained final K, compact indices | 30.380% | 7.938 | 9.069 | 18.489 |
| Above + head-shift search | 30.383% | 8.001 | 9.151 | 42.800 |
| Pre-RoPE K + shifts + raw lower bytes | **30.757%** | 8.763 | 9.925 | 44.090 |
| Raw K projection + normalization/RoPE | 30.726% | 8.916 | 10.056 | 44.287 |

The nine lower-bit variants save 30.373–30.750% on average, depending on source and attempted predictor. Every one retains the same lower-byte payload as its raw-byte counterpart and adds descriptor bytes. The machine-readable analysis contains all 16 means, per-case values and ranges. The best recipe ranges from **30.7295% to 30.7787%** across TEST cases.

These eager implementations demonstrate codec costs. The experiment has no matched raw H2D transfer baseline, SSD path, cold-state restore, compact attention kernel, concurrent-serving run or TTFT/TPOT comparison. During capture all 16 codecs are exercised and raw KV remains resident, so capture memory and prefill duration cannot establish serving capacity or a single-codec prefill penalty. Generation validation includes GPU/CPU staging for correctness.

## Which changes actually helped

The matched comparison separates stored bytes, including the current packet headers and state:

| Change from the frozen exponent control | Savings contribution |
|---|---:|
| Compact stream indices | +0.787354 pp |
| Net exponent payload improvement from cost-trained bases | +0.043483 pp |
| Larger headers | −0.290728 pp |
| Additional amortized state | −0.003658 pp |
| **Net: new final-K packet** | **+0.536450 pp** |
| Add head-shift search | +0.003202 pp |
| Switch to source-specific pre-RoPE K | +0.373845 pp |
| **Total improvement** | **+0.913497 pp** |

For final K → pre-RoPE K, the K exponent payload improves by **0.3789 pp**, partly offset by headers/state. V payloads are identical. The source improvement appears in all 36 layers on the pooled sampled records. That supports capturing the actual producer source and fitting its own coding model. It does not establish that all producer metadata helps: the previous hidden-feature predictors lost against the frozen control.

The new replay paths select `verified_zero` on every selected pair: **432/432 per source across TEST**, with no correction-width table or correction payload. The encoder verifies target equality before selecting this mode; a correction fallback remains available. Replay state records the execution contract. These results apply to the tested model, hardware and stack, rather than proving identical floating-point replay on arbitrary devices.

Coset coding remains part of the exact-correction mechanism. Here there are no nonzero replay corrections to compress, so these data demonstrate no additional coset payload saving. Its percentage cannot be added to exponent or source-coding savings.

## What did not help

**Head adjustments:** final K applies a shift vector in 2/432 pairs, pre-RoPE K and projection K in 0/432, and V in 38/432. These count applied head-shift vectors, rather than individual nonzero heads. Searching raises final-K encode cost from 18.489 to 42.800 ms/raw MiB for just **0.0032 pp**. The observed extra GPU allocation peak over the combined encode/decode operation rises from about 10.05 to 67.76 MiB per 0.25 MiB raw pair. This peak is not isolated decoder workspace.

**Sparse bitplanes:** every new primary stream retains fixed two-bit coding; every attempted lower stream retains raw eight-bit coding. Raw, first-token XOR and previous-token XOR variants produce no payload reduction. Their attempted encoders add roughly another 9.6 ms/raw MiB. Descriptor spelling alone makes their packets slightly larger. This rejects the tested sparse-plane scheme on these data; it does not prove that entropy coding cannot exploit moderate bias. Its minority-position format requires fewer than about 12% minority bits in a 256-bit group to beat raw storage.

**Mixing current recipes:** an offline enumeration checks all 127 nonempty subsets of the seven raw/control candidates. It counts existing headers, a mode byte when mixing, and the complete activated state union. In every TEST case it selects only the pre-RoPE raw-byte recipe: **zero additional savings**. This is a TEST-informed size calculation, with no trained selection policy or adaptive latency claim. Removing duplicate subarrays from distinct state artifacts would be another format change.

## Why metadata cleanup cannot close the 40% gap

The best representation uses **11.07894 bits per original BF16 word**. A 40% reduction requires **9.6 bits**, leaving **1.47894 bits/word**, or **9.24338 percentage points**, to remove. That is a further 13.35% reduction of the currently encoded size.

| Best representation component | Bits per BF16 word |
|---|---:|
| Sign + seven mantissa bits, stored raw | 8.000000 |
| Primary exponent symbols | 2.000000 |
| Secondary exponent escapes | 0.745185 |
| Rare exponent escapes | 0.103846 |
| Stream indices | 0.125000 |
| Packet headers | 0.096685 |
| Complete decoder state | 0.007691 |
| Outer framing + head shifts | 0.000533 |
| **Total** | **11.078940** |

Even deleting every index, header, state and framing byte leaves approximately **10.849 bits**, or **32.19%** savings. Within this accounting, another 1.249 payload bits are still needed. Retaining raw lower bytes and fixed primary symbols alone costs ten bits. The format can change those streams, but all observed Qwen packets keep that ten-bit floor. This is a limitation of the observed representations, rather than a universal information-theoretic bound.

## What the information diagnostics support

The sign/mantissa byte has near-eight-bit marginal entropy: approximately **7.93–7.97 bits** across the four captured domains. Conditioning on layer and exponent leaves roughly **7.83–7.93 bits**; simple previous-byte conditioning also leaves little headroom. K sign bits show temporal structure, while mantissa-bit XOR rates remain close to 50% and exact lower-byte repeats are below 1%.

The much lower empirical estimates obtained by conditioning on layer, channel and exponent are **not credible compression targets**. Each TEST case has only 128 selected observations per layer/channel, a median of eight observations per occupied context, many singletons and almost no reliable context mass. The empirical estimate rises from roughly 4.3–4.5 bits on TEST to 5.8–6.0 on the larger DEV sample, illustrating sample bias. A first-order Miller–Madow adjustment does not resolve unseen outcomes. These are descriptive estimates, with explicit reliability flags, rather than achievable code lengths or proven lower bounds.

The actual CPU zlib lower-byte controls round-trip exactly and charge seeds, framing and checksums. K previous-byte XOR produces roughly **7.87–7.92 bits**, and V raw-channel coding roughly **7.95–7.98 bits**. Substituting those sizes into the best GPU packet arithmetically suggests about **31.18%** mean savings. That combination is **not implemented**, and CPU timing does not predict GPU performance. It would close only a small part of the 1.48-bit gap. With all other fields held fixed, the lower-byte stream would need about **6.52 bits/value** to reach 40%.

## Strongest next experiments

1. **Recover speed with the demonstrated representation.** Keep source-specific pre-RoPE coding, compact indices and verified exact replay. Disable rejected head-shift, primary-plane and lower-plane searches. Compare against the frozen control, a matched raw transfer and the existing full-search recipe. Measure isolated decode workspace and end-to-end prefill/restore. The expected packet cost is close to the current winner, with the small V shift benefit removed; its speed must be measured.
2. **Gate the next savings idea on held-out actual bits.** Fit a pooled conditional bit or residual model on TRAIN, using only decoder-available context and explicitly retained producer features. Compare zero-feature, KV-only and producer-feature versions with the same entropy backend and packet schema. Store the state and exact correction stream; charge features, anchors, escape tables, headers and state before selecting on DEV. Use pooled parameters/backoff to avoid sparse per-channel probability estimates. Freeze before a fresh TEST run.
3. **Only port a successful rate model to GPU.** Require actual held-out complete-record savings beyond the current winner before investing in a GPU entropy kernel. Evaluate whether its probability construction, state and feature costs leave enough of the 1.48-bit gap to justify it. An entropy diagnostic or improved prediction error alone does not meet this gate.

The producer-aware hypothesis remains testable: information available before normalization/RoPE might help predict a cheaper exact representation or its bit probabilities. This run supports the pre-RoPE source choice. It does not yet support a claim of 40% savings, serving speedup or research novelty. A richer conditional model is a research direction with a clear failure criterion, rather than an established solution.

## Reproduce this review

```bash
python scripts/review_frontier_results.py \
  benchmark_results/qwen_frontier_40_a4500 \
  --output runs/frontier_review_check
```

Use a fresh output directory. The standard-library script reads the gzip report directly, verifies recorded artifacts and arithmetic, and produces [analysis.json](../benchmark_results/qwen_frontier_review_20261019/analysis.json) and [validation.json](../benchmark_results/qwen_frontier_review_20261019/validation.json). The [review manifest](../benchmark_results/qwen_frontier_review_20261019/review_manifest.json) pins downloaded files and review artifacts. It does not rerun GPU kernels. See the [study protocol](frontier_40_study.md) for the benchmark command and sharing instructions.
