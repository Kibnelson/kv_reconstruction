# Optimize complete packet bytes and GPU latency together

The goal is lower transfer-plus-decode time at each target link, while recovering
every original BF16 word exactly. Keep useful size/latency tradeoffs instead of
choosing one decoder solely by its kernel time. Smaller packets also improve
DRAM/SSD cache density independently of a transfer latency win.

For a precompressed packet of S bytes, GPU decode latency D milliseconds, and
effective link rate R Gb/s, the serial model is:

    total_ms = D + 8 * S / (R * 1,000,000)

A smaller candidate wins when its additional decode time is less than the wire
time saved. Fresh-encoding experiments must additionally charge encoding time.
Ideal overlap, max(wire, decode), remains a separate bound; networking and
storage performance are not measured by this model.

## Concrete byte budgets

The 8K Llama-3.2-3B case contains 469,762,048 BF16 values. Reducing the complete
packet rate by 0.5 bits/value removes exactly 28 MiB. The maximum extra decode
time that saving can pay for is:

| Effective link | Wire time saved / extra decode budget |
|---:|---:|
| 1 Gb/s | 234.881 ms |
| 100 Gb/s | 2.349 ms |
| 200 Gb/s | 1.174 ms |
| 300 Gb/s | 0.783 ms |
| 400 Gb/s | 0.587 ms |

These are hypothetical rate reductions, not achieved new compression results.

The completed lookup/direct experiment already demonstrates a tradeoff:

| 8K direct/search configuration | Complete bpv | Packet MiB | GPU decode ms |
|---|---:|---:|---:|
| Dense | 8.920970 | 499.574 | 28.151 |
| Rank256 | 9.306323 | 521.154 | 21.709 |

Dense saves 21.580 MiB and costs 6.442 ms more decoding. Under the serial model,
Dense wins below approximately 28.1 Gb/s; Rank256 wins above that crossover.
Source: `benchmark_results/team_lookup_a4500_20261028_141701_1145261_report_20261028_143153/`.
These are means of per-document medians on two TEST documents, not a deployed
routing policy or measured network result.

## Byte experiments to prioritize

1. **Optimize probability calibration for actual packet cost.** Residual spreads
   are already calibrated on the separate `study.tables` document in both
   `team_onestep/fused.py::prepare` and `layerwise.py::prepare_rank`. The proposed
   improvement is the objective: compare actual finite-precision packet bytes
   against the current mean-absolute-residual scale estimate. Record escape
   frequencies, width classes, per-layer stream bytes and runtime. Parameter
   selection must use held-out data, leaving TEST untouched.

   Scale and adaptive-prior normalization are coupled. Scaling both `sp` and
   `inv` inversely largely cancels through the preceding layer's error-based
   multiplier. A pilot must distinguish coupled updates from fixed-`inv`
   probability-scale updates and replay the full decoder consistently.

2. **Refit for the predictor's actual INT8 representation.** Test regression on
   actual quantized features, coefficient-aware refinement, and feature-scale
   clipping chosen for held-out coded bytes. Keeping matrix dimensions and rank
   fixed can improve predictions without increasing the GPU operation count.
   Predictor approximation may change; the original BF16 output must stay exact.

3. **Use more decoder-available information selectively.** Compare a small
   number of current-layer channel groups before returning to many sequential
   groups. Another diagnostic is head-specific uncertainty: the decoder already
   computes per-head error sums but currently averages them into one token-wide
   scale. Test whether retaining some of that information predicts coding
   uncertainty better. Neither idea is an achieved rate improvement yet; both
   require new coherent encoder/decoder probabilities and measured packet bytes.

Paid prefill hints and coset residues remain possible later arms. Their complete
side-information cost must be charged. A representation change alone does not
establish an information gain, and theoretical entropy is not an actual packet.

## Measurement and iteration

- Keep the pending register-reuse runs frozen: they isolate GPU speed using
  identical packets. A byte experiment changes state or packets and gets its own
  rate/latency comparison against that speed control.
- Record complete packet bytes, encode/decode latency, exact word equality,
  shared-state bytes, and the five link models at 1/100/200/300/400 Gb/s. Charge
  headers, directories, escapes, padding, checksums and any per-request hints.
- Report fixed shared calibration separately; charge its transfer when testing
  cold setup. Compare identical original words and preserve native baseline
  implementations and accounting.
- Retain nondominated candidates and show the best measured tradeoff per link.
  Do not assume one configuration is best for every packet length and link.
- Combine the successful byte changes with the fastest validated kernel only
  after each change has its own control. Record actual composed measurements;
  do not add isolated speedups or theoretical bit savings together.
