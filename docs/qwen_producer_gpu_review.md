# A4500 producer-hook study: exact replay works; the static control wins

Reviewed result commit: [`53afe7004d89ff7d9461421b465947c9c5677c23`](https://github.com/Kibnelson/kv_reconstruction/commit/53afe7004d89ff7d9461421b465947c9c5677c23). Benchmark code: `763c9fb78bf045c5ebcc3e34b660d98dafff2b95`.

The study completed successfully on the RTX A4500 with the pinned Qwen3-4B-Instruct-2507 checkpoint. Producer hooks observed real forward intermediates, all nine recipes reconstructed exactly, and their inside-forward packets matched post-capture replay. The strongest current size and restore-time result is the **frozen per-layer/channel exponent control**. The tested hidden-feature and upstream-K recipes have not improved the net result over that control.

The useful new producer result is more specific: **both K replay paths required zero correction payload on every measured DEV and TEST tile**. That supports the feasibility of replaying these operations on the tested stack, while exposing metadata costs that currently hide the benefit.

## What was verified

- All 15 downloaded repository files matched their pinned Git blob hashes; 38,196,471 bytes were retrieved.
- All 136 recorded source-file hashes matched the reviewed implementation. Source/freeze digests, configuration, candidates and model identities agree.
- All three TRAIN-only predictor binaries passed canonical model loading, CRC, SHA, kind and model/TRAIN identity checks. All nine unique calibration/state artifacts have the reported file sizes and hashes.
- All 63 method rows passed: nine methods on one DEV case and six TEST cases. Byte components, outer framing, state charges and full-cache extrapolations recompute without discrepancies.
- Both execution tracks checked 6,480 paired packets: inside attention forward and post-capture replay. Corresponding addresses, component sizes and state identities agree.
- Three complete 512-token DEV continuations passed two decode steps with identical finite logits, tokens and appended KV.

TRAIN uses three separate synthetic fixtures, followed by DEV and frozen TEST settings. TEST covers three synthetic topics at 2,048 and 8,192 tokens. Length variants of a topic are correlated; this is not broad workload or task-quality evidence.

## Current size and speed results

The table gives unweighted means over the six TEST cases. Savings include complete decoder state once per full cache. TEST sizes are **estimates**: 72 paired tiles out of 1,152 at 2K (6.25%), and 72 out of 4,608 at 8K (1.5625%). Every method uses the same selected pairs.

Timings use the same 16-pair / 4 MiB raw sample, one warmup and three repetitions. Units are milliseconds per **raw** MiB.

| Recipe | Estimated savings | Device-resident decode | Pageable packet transfer + decode |
|---|---:|---:|---:|
| Frozen channel exponents | **29.843%** | **7.012** | **8.018** |
| Current-tile KV medians | 29.388% | 7.589 | 8.742 |
| Fixed exponent 127 | 21.225% | 7.515 | 8.594 |
| Signed hidden features | 28.779% | 8.466 | 9.521 |
| Absolute hidden features | 28.734% | 8.264 | 9.310 |
| Pre-RoPE + residual | 29.205% | 9.332 | 10.609 |
| Pre-RoPE + coset | 29.207% | 9.466 | 10.743 |
| K projection + normalization/RoPE + residual | 29.176% | 9.568 | 10.849 |
| K projection + normalization/RoPE + coset | 29.178% | 9.740 | 11.020 |

Frozen savings range from 29.766% to 29.916%. It is the observed winner in all six cases. Its decoder state is 213,519 bytes, compared with 1,432,460 for signed hidden features and 1,432,250 for absolute features. Pre-RoPE replay state is 520 bytes; projection replay state is 12,793 bytes.

The existing packet implementations have different headers and execution paths. A smaller or faster row does not isolate predictor quality. Fixed method order and three repetitions also limit performance claims.

## Why the frozen control wins

Relative to tile medians, the frozen representation gains **0.45560 percentage points** on average:

| Change | Contribution to savings gain |
|---|---:|
| Remove per-tile channel anchors | +0.78125 pp |
| Smaller current-format headers | +0.30428 pp |
| Larger exponent correction payloads | −0.58574 pp |
| Full frozen state charge | −0.04419 pp |

Its advantage comes from amortizing metadata and using a smaller header, despite a worse correction distribution. The header difference is a format effect. A fair predictor ablation should normalize descriptors and compare the actual correction costs separately.

This suggests a hierarchy: retain frozen per-channel bases, then encode only a small local adjustment. It does not justify replacing tile adaptation with a large learned feature model.

## Why the hidden-feature variants lose

Signed hidden features lose **1.06410 percentage points** against the frozen control. The contributions are:

| Additional cost | Effect on savings |
|---|---:|
| Exponent correction payload | −0.60170 pp |
| Eight int8 features per token/layer | −0.19531 pp |
| Additional headers | −0.01481 pp |
| Additional amortized state | −0.25227 pp |

The predictor reduces rare escapes but increases the more common secondary escapes enough to lose overall. It would still lose if features and state were free. On pooled sampled packet payloads it improves only layer 0, losing in the other 35 layers.

The current TRAIN objective is ridge squared error over exponent predictions. The actual coding cost is discontinuous. For zigzag residual \(z\), the tier payload cost is

\[
c(z)=2+3\,\mathbf{1}[z\ge3]+9\,\mathbf{1}[z\ge10].
\]

Reducing prediction error can move more values across a tier boundary. The next predictor should minimize measured coded bits, include a frozen-predictor fallback, and justify each retained feature. A lower-rank model or faster kernel alone would not repair the observed correction-payload loss.

## What the replay paths establish

Both pre-RoPE replay and projection → normalization → RoPE replay have **zero K correction payload in all 432 selected TEST pairs**: 28,311,552 original K words per representation. DEV also has zero correction payload on every measured pair. This is evidence for exact producer-path replay on the sampled model, stack and hardware.

Nevertheless, each pair stores a 640-byte all-zero correction-width table. Across raw K/V, this costs **0.24414 percentage points**. The compressed pre-RoPE K source has a smaller exponent payload than final K, gaining about 0.16284 pp, but extra headers and width tables outweigh it.

A future verified zero-correction mode could remove those width tables. The encoder must check exact equality and retain a correction fallback. Charging one mode byte per pair while holding all other bytes fixed gives a **projected** gain of 0.24376 pp. The projected pre-RoPE mean would be about 29.451%, still below the current frozen control. No new-format savings or speed is measured by this arithmetic.

The stronger next combination is **source-specific frozen exponent coding of pre-RoPE K + verified zero-replay mode**, compared against the same optimized coding of final K. Existing frozen K bases were fitted to final K; they should not be silently reused as if calibrated for an earlier source. TRAIN must fit both sources, and full state must count.

Replay compatibility remains bounded by identical encoder/decoder predictions. Different devices, kernels or software versions may alter floating-point replay. Stored corrections and explicit state identity remain necessary safeguards.

## What cosets and mixing add here

Residual and coset variants have identical numeric payload sizes and widths in this run. The coset packet is six bytes smaller per pair entirely because of JSON header spelling: a 0.00229 pp difference. These data provide no additional coset coding gain.

The review also enumerates all 511 nonempty subsets of the existing nine candidates. This offline size calculation retains existing packet headers, charges the complete activated state union, and adds a selector byte per pair when multiple methods are allowed. It uses TEST outcomes and does not implement a held-out selection policy or measure selector/encoder costs.

The best subset chooses only frozen exponents for all three short-context cases. At 8K it adds KV medians for one, two or four sampled pairs, improving savings by only **0.00158, 0.00318 or 0.00701 pp**. Combining the existing recipes offers negligible headroom under this model. Larger gains require better representations, coding or metadata design.

## Speed interpretation

Frozen reconstruction has the lowest observed decode and packet-restore cost among these nine eager implementations. This study has no matched raw H2D baseline and no SSD path, so it cannot establish an end-to-end transfer or serving speedup.

Input staging costs approximately 0.43 ms/raw MiB for median/frozen, 0.66–0.67 for hidden features, and 0.55–0.56 for replay. Hidden recipes currently stage the complete 2,560-dimensional hidden tensor before selecting the retained features. A production encoder should operate where those activations already exist on the GPU.

Instrumented TEST prefill takes about 3.40 seconds at 2K and 5.34–5.41 seconds at 8K. Every callback runs all nine encoders, reconstruction and validation, with diagnostic GPU→CPU→GPU roundtrips. TEST always selects 72 pairs, so inline codec work remains roughly constant across context lengths. These durations do not measure deployment overhead for one codec. DEV has complete 288-pair coverage and takes 11.79 seconds.

The generic `prefill.observer_included: false` field refers to the separate older KV observer argument. Producer hooks are active inside the timed model call; their dedicated audit records the inline work. Read that field together with `producer_audit` and `inline_recipe_metrics`. A subsequent runner should make the producer inclusion explicit in the timing descriptor.

Generation restoration uses GPU→CPU→GPU correctness plumbing. Full raw KV remains resident during capture. There is no measured compact-resident attention, materialization-budget enforcement, high-concurrency capacity, TTFT or TPOT gain. Prior-run raw-transfer timings use a different protocol and cannot supply a matched baseline for this study.

The old individual-KV sampling and new paired sampling also select different addresses. Only 129 of 144 old sampled logical records overlap at 2K, and 83 at 8K. Cross-run ratio changes are not clean causal comparisons; use the nine same-record candidates here.

## Prioritized next experiments

1. **Frozen base + small prefill adjustment.** Test a per-head exponent shift or sparse channel-anchor changes, chosen by actual coded-bit cost with a frozen fallback. Sixteen int8 shifts for eight K and eight V heads would cost 16 bytes per pair, versus 2,048 bytes for all channel anchors, before framing. Whether those shifts recover useful prediction accuracy is untested. Compare shifts derived from produced KV against actual producer-stage information to attribute the hook's contribution.
2. **Verified replay + source-specific coding.** Add a zero-correction flag with fallback, fit TRAIN bases separately for pre-RoPE K, and normalize headers across final-K and replay paths. Test combined source, state, descriptor and correction bytes before claiming a gain.
3. **Train for coding cost.** Optimize tier/entropy cost, retain only features whose benefit pays their full metadata/state cost, and freeze the policy before TEST. Include a static fallback and record which choices generalize.
4. **Improve packaging and execution separately.** Validate compact stream indices and binary descriptors; batch contiguous pinned packets; reconstruct into preallocated GPU tiles. Compare plain prefill, hooks alone and one encoder using matched repeated runs, and compare raw transfer with packet transfer plus reconstruction under the same memory/allocation protocol. Fusion needs a later implementation and GPU measurement.

The validated foundation is exact reconstruction plus real producer access. The next contribution should demonstrate a repeatable net advantage over the strong static/KV-only controls. These results alone do not establish research novelty.

## Reproduce the review

```bash
python scripts/review_producer_results.py \
  benchmark_results/qwen_producer_hooks_a4500 \
  --output runs/producer_review
```

The standard-library script validates source/artifact identities and byte contracts, recomputes matched increments and timing summaries, and labels projections and offline selection separately. It reads inputs without rerunning the model. `--source-root` should refer to the reviewed benchmark source when using a later checkout.

Machine-readable results: [analysis](../benchmark_results/qwen_producer_review_20261019/analysis.json), [validation](../benchmark_results/qwen_producer_review_20261019/validation.json), and [review manifest](../benchmark_results/qwen_producer_review_20261019/review_manifest.json).
