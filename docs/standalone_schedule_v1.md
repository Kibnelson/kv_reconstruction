# Standalone lossless feature reuse and decoder scheduling

All 28 layers use our Dense or Rank256 predictor/rANS codec. This experiment
retains the successful rolling encoder, original coefficients, probability
arithmetic, binary CDF search and complete serialized packets. SplitZip and
ExANS are independent unchanged controls. Lossy methods and hybrid fallback
are outside this experiment.

## Completed rolling results

The preceding A4500 run is commit `eadd22e`, report
`benchmark_results/standalone_rolling_a4500_20261029_191839_664765`.
All 153 manifest entries, 48 exact rows, 24 fresh receivers, 12 own
candidate/reference packet comparisons and two actual CUDA test methods
passed. Measurement sources, shared predictors and native author libraries
remained unchanged. Every own candidate packet preserves its original SHA.
The unchanged fixed ExANS worker permits native reencode byte differences,
while checking new packet reconstruction, geometry and complete accounting.

The 8K isolated two-document means are:

| Method | Complete MiB | Encode ms | Decode ms |
|---|---:|---:|---:|
| Original Dense | 499.574 | 83.270 | 28.895 |
| Cached Dense setup | 499.574 | 73.247 | 29.099 |
| Rolling Dense | 499.574 | 42.304 | 28.849 |
| Original Rank256 | 521.154 | 41.986 | 21.649 |
| Rolling Rank256 | 521.154 | 34.648 | 21.563 |
| SplitZip, before control | 674.807 | 10.325 | 3.370 |
| Compact ExANS carrier | 609.129 | 209.806 | 23.554 |

Rolling Dense cuts encoding by 49.2% versus original and 42.2% versus cached
setup. Rolling Rank256 cuts encoding by 17.5%. Both documents and the batched
timings support the gain. At 512 tokens, Dense improves from 23.201 to
5.604 ms and Rank256 from 7.558 to 5.675 ms. Decoding remains effectively
unchanged.

The observed mean incremental encoder Torch peaks fall from 2742.830 to
2357.125 MiB for Dense, and 2764.107 to 2359.125 MiB for Rank256. Decoder
incremental peaks remain 896 MiB at 8K. Shared model state is additional;
native allocations outside Torch are unmeasured.

Profiles verify actual work removal: Rank encoding uses 27 prediction
products instead of 53, while decoding remains 54 products per request.
Dense no longer calls its fused forward model. Rolling entropy encoding
still accounts for roughly 17–18 ms of accumulated GPU work per request;
feature and surprise passes roughly 4.5 and 5 ms. The decoder entropy core
remains roughly 21–22 ms. These separate diagnostic totals are nonadditive.
The CPU time attributed to compact segment export includes waiting for
preceding GPU work and must not be counted again as independent copy cost.

At calculated 25 Gb/s, fresh rolling Rank transfer costs 231.081 ms against
SplitZip's 240.122 ms. At 50 Gb/s, precompressed rolling Rank costs
108.998 ms against 116.584 ms, but fresh encoding still makes it slower:
143.646 ms versus 126.908 ms. Higher-bandwidth precompressed performance
requires decoder improvements; encoder gains alone cannot deliver them.

## Independent feature-buffer arm

The rolling encoder currently creates all 27 consumed feature matrices
before reverse coding. Each matrix is used once for its layer's product.
Generate that layer's features immediately before the product instead,
reusing one INT8 `[tokens, width]` matrix on the same caller stream.

The feature kernel, source words, product arithmetic, priors and rANS order
stay unchanged. The 27 feature launches and 27 products remain. Feature
scratch falls from 432 to 16 MiB at 8K, saving 416 MiB of explicit temporary
storage. At 512 tokens it falls from 27 to 1 MiB. Actual API peaks and any
speed improvement still require measurement. No extra persistent tensors
or transmitted metadata are introduced.

The existing allocation accounting explains why the full API peak need not
fall by 416 MiB. At 8K, Dense's native encoding peak is approximately
1794.063 MiB of arena, 432 MiB of features, 128 MiB of accumulators and
3.063 MiB of other scratch: the observed 2357.125 MiB. Features are released
before packet compaction, but compaction retains both the arena and the
roughly 500 MiB Dense packet. That gives an inferred floor near 2294 MiB;
Rank's larger packet gives a floor near 2315 MiB. Thus the expected memory
bottleneck can move from prediction scratch to final packet construction.
These floors are inferences from existing geometry and lifetimes, not new
feature-reuse measurements.

Test Dense and Rank256 at 512 and 8192 tokens. This arm retains the original
two decoder chunks and does not combine with a scheduling change.

## Independent decoder scheduling arms

Keep the rolling encoder unchanged and partition 8K decoding into one, four
or eight chunks instead of two. Each chunk contains whole token-block/head
stream segments, so it changes execution rather than the format. It does
not change token dependencies or rANS states. No previous Layerwise
1/4/8-chunk sweep was found in this checkout; earlier Lite stream experiments
changed a different codec.

A shallow decoder-controller copy shares immutable predictor and coder
tensors. Its `decode` and `buffers` methods are installed before preparation;
both must use the same chunk count. The sender controller and on-wire
`chunks=2` identity remain unchanged. Execution receipts explicitly report
the decoder override. Preserve caller-stream ordering, chunk joins, and
borrowed input/output lifetime registration.

Require whole token-block partitioning and valid Rank requantization tiles.
At 8K, one/four/eight chunks produce 8192/2048/1024 rows, all divisible by
the Rank first-product tile of 128. This sweep is restricted to 8K: eight
chunks would leave only 64 rows at 512 tokens, which is unsupported for Rank.
Unsupported geometry is rejected, not repaired with another product backend.

Changing chunks partitions the same aggregate accumulator and Rank scratch
storage. More streams may require additional native cuBLAS/runtime workspace,
which Torch peaks do not fully observe. More chunks can expose concurrency
but also add launches and coefficient traffic. These are scheduling
hypotheses, not expected speedups or measured occupancy claims.

## Matched experiment and run

Use current rolling Dense/Rank256 references at both lengths. Replace the
older double-pass own references; retain the same 12 before/after raw,
SplitZip and fixed ExANS control jobs plus two compact ExANS jobs.
Four reference jobs and ten candidate jobs yield 28 total workers and
56 rows. All 28 own rows must match original frozen packets; 20 candidate
document groups are paired with the current rolling references.

Two actual CUDA test methods gate timing. Require original complete packet
identity, exact BF16 word reconstruction and fresh receivers with only
state and compressed packets. Alternating inputs, independently retained
outputs, caller-stream switches and early packet release exercise lifetime
handling. Keep graph execution at 512 and direct borrowed-input/fresh-output
decoding at 8K.

Full encode/decode API timing retains the author timing body with 10 warmups,
50 calls and five repeats, plus isolated two-warmup/five-sample measurements.
Cold setup, fitting, export and upload are separate. After ordinary timing,
paired warm profiles verify actual encoder and decoder product counts,
feature launches, decoder kernels, and absence of table/model rebuilds.
For three full cycles, products should equal
`3 * (27 + 27 * decoder_chunks)` and decoder kernels
`3 * 28 * decoder_chunks`. Rank requantization follows the product count.

Report all fixed arms at calculated 1, 5, 10, 25, 50, 75, 100, 200, 300 and
400 Gb/s, separating precompressed wire plus decode from fresh encode plus
wire plus decode. Do not select a deployment policy from the reused TEST
documents. The same two development documents support engineering checks,
not independent generalization. This is transfer/storage T with full BF16
restoration, not measured NIC/SSD, serving throughput or resident attention P.

On the NUC, switch to `experiment/standalone_schedule_v1` and run
`bash scripts/run_standalone_schedule_a4500.sh` inside `~/kv_reconstruction`.
The wrapper publishes success or failure reports and retains the original
exit code. Encoded packets and full traces remain in the run directory.
