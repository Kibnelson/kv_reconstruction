# Standalone lossless encoder with rolling prediction products

All 28 layers use our Dense or Rank256 predictive/rANS codec. SplitZip and
ExANS remain independent unchanged controls. This experiment changes encoder
execution only: coefficients, probabilities, complete packets, binary CDF
search and the original decoder stay fixed. No lossy reconstruction or hybrid
fallback is introduced.

## Evidence from the completed compute experiment

The preceding A4500 run is commit `710e027`, report
`benchmark_results/standalone_compute_a4500_20261029_151844_3927903`.
All 183 manifested files, 56 exact rows, 28 fresh receivers, 20 candidate
packet comparisons and two actual CUDA test methods passed. Measurement
sources, fixed predictors and native controls remained unchanged. Our
candidate packets match their original complete serialized bytes. Fixed
ExANS may produce different encoded bytes on reencoding; its unchanged
worker verifies the new packet, original words, geometry and accounting.

Reusing Dense encoder preparation is the clear gain from that experiment:

| Tokens | Basis | Original encode ms | Cached model encode ms | Reduction |
|---:|---|---:|---:|---:|
| 512 | isolated | 23.001 | 11.451 | 50.2% |
| 512 | batched | 22.341 | 10.369 | 53.6% |
| 8192 | isolated | 83.478 | 72.346 | 13.3% |
| 8192 | batched | 83.736 | 76.648 | 8.5% |

The separate decoder GEMM experiments were less conclusive. Rank256 isolated
decode improved from 21.578 to 21.239/21.105 ms, but batched improvements
were about 0.4%, with mixed or regressive 512-token results. Dense results
were also inconsistent. Neither decoder override is carried into this
encoder experiment. A full inverse CDF LUT had already regressed decoding
and is also excluded.

The profiles verified zero repeated table construction in the cached Dense
arm. Substantial GPU work remains: the fused Dense forward-model kernel
accounts for roughly 36–38 ms of accumulated GPU work per profiled request,
and the entropy encoder roughly 19–20 ms. These are nonadditive diagnostic
measurements, not a decomposition of full API latency. Directory export's
CPU wait includes preceding GPU work and must not be counted again as an
independent transfer bottleneck.

For context, the completed run's 8K isolated two-document means are:

| Method | Complete MiB | Encode ms | Decode ms |
|---|---:|---:|---:|
| Original Dense | 499.574 | 83.478 | 28.630 |
| Cached Dense | 499.574 | 72.346 | 28.981 |
| Original Rank256 | 521.154 | 41.959 | 21.578 |
| SplitZip, before control | 674.807 | 10.216 | 3.385 |
| Compact ExANS carrier | 609.129 | 209.858 | 23.556 |

Dense uses about 26.0% fewer transmitted bytes than SplitZip; Rank256 about
22.8% fewer. SplitZip still has substantially lower codec latency. The main
opportunity here is to reduce the fresh-encoding cost without spending the
byte advantage. A change confined to encoding cannot improve the
precompressed wire-plus-decode case.

## Why reverse encoding can reuse predictions

The encoder has every original layer available. A layer's probability spread
uses the preceding layer's fixed-point surprise, which depends on that
preceding layer's exact words and predicted mean. Computing that surprise
does not require decoding or the preceding layer's own probability spread.

The current Rank256 encoder computes 26 prediction products in its forward
model pass and recomputes 27 products while encoding layers in reverse.
The proposed schedule computes each of the 27 nonzero layer products once:

1. Generate the same INT8 features from original words.
2. Keep the current layer's product and compute the preceding layer's product
   in a second temporary buffer.
3. Use the unchanged surprise and prior kernels on the preceding layer to
   obtain the current layer's spread.
4. Encode the current layer with the unchanged entropy kernel, then swap the
   two product buffers and continue backward.
5. Use a zero product for layer zero, an all-ones first-layer prior, and the
   original final-state flush.

Rank256 therefore needs 27 full prediction products instead of 53. Dense can
use the same schedule to replace its fused forward-model pass with the
existing feature, surprise and prior kernels, reusing the encoding products.
This is an execution hypothesis; neither speedup has been measured yet.

The original INT8/INT32 products, FP32 operations, RoPE, fixed-point surprise
sums, rANS stream order, segment alignment and complete framing are retained.
No prediction arrays or source KV are sent to the receiver.

Two rolling product buffers avoid retaining every layer's INT32 products.
The additional accumulator versus a single-accumulator encoder is 64 MiB
at 8K tokens or 4 MiB at 512 tokens. It is temporary per-call scratch, not
persistent model state. Full API allocation peaks must still be reported:
other temporary arrays have different lifetimes, so the net peak change
cannot be inferred from this accumulator alone.

## Matched evaluation and run

Measure both Dense and Rank256 at 512 and 8192 tokens using the original
Llama-3.2-3B BF16 words and frozen predictor state. Compare original encoding,
the proven Dense cached preparation, and the new rolling schedule. Keep
the original compact-search graph decoder at 512 and direct decoder at 8K.
Require unchanged complete packet bytes, independent receiver reconstruction,
alternating inputs, retained independent outputs and packet immutability.
Six new jobs (rolling Dense, rolling Rank256 and cached Dense at both lengths)
plus 16 frozen controls and two compact ExANS controls produce 24 workers,
48 rows and 12 candidate/document packet comparisons.

The runner preserves author timing with 10 warmups, 50 calls and five repeats,
and separate isolated timing with two warmups and five samples. CUDA tests
gate measurement. Separate warmed profiles verify actual product counts and
the absence of a forward-model call or repeated table construction. Cold
setup, export, upload and offline fitting remain outside warm codec timing.
Native allocations outside Torch remain unmeasured.

Report calculated 1, 5, 10, 25, 50, 75, 100, 200, 300 and 400 Gb/s scenarios,
separating precompressed wire plus decode from fresh encode plus wire plus
decode. These are not measured NIC, SSD or serving results. The two repeatedly
reused development documents do not establish independent generalization.
This is transfer/storage T with full BF16 restoration, not compressed
resident attention P.

On the A4500 NUC, switch to `experiment/standalone_rolling_v1` and run
`bash scripts/run_standalone_rolling_a4500.sh` inside `~/kv_reconstruction`.
Reports publish automatically; failures retain their original exit code.
Encoded packets and full profiler traces stay in the run directory.

## Follow-on questions after measurement

If product reuse helps, a separate experiment can generate each layer's
features immediately before its product. Reusing one INT8 feature matrix
would reduce this experiment's 432 MiB of feature scratch to 16 MiB at 8K,
while retaining the same 27 feature launches and prediction products. This
has not been implemented or measured here.

Decoder latency remains a separate target. With the current Rank256 bytes,
matching the completed run's SplitZip precompressed cost at calculated
100 Gb/s would require about 16.27 ms of decoding. The small GEMM experiment
reached 21.11 ms isolated, with much smaller batched gains. Encoder gains
must not be presented as achieving that decoder target.
