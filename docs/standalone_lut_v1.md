# Standalone lossless codec: exhaustive inverse lookup

The main experiment now compares our standalone all-layer codec with unchanged
lossless baselines. SplitZip is not a component of either candidate. Hybrid
experiments remain separate and must be labelled Ours + SplitZip.

This iteration preserves the existing Dense and Rank256 predictors, probability
tables, original BF16 words and complete serialized packets. It changes how
the GPU decoder finds a symbol in a fixed cumulative distribution (CDF).

## Starting evidence

The trusted matched result is commit
`95889040b86862ac79d4b188c6441d456267f28e`, report
`benchmark_results/lossless_decode_a4500_20261029_125616_867717`.
These standalone control rows use all 28 Llama layers, original BF16, and the
same two 8K development documents. Isolated per-request timings are separate
from the report's batched-call timings.

| Method | Complete MiB | Encode ms | Decode ms |
|---|---:|---:|---:|
| Raw BF16, after round | 896.000 | 4.982 | 3.423 |
| Official SplitZip, checked frame | 674.807 | 10.288 | 3.368 |
| Compact carrier around released ExANS | 609.129 | 210.076 | 23.555 |
| Ours alone, Rank256 | 521.154 | 42.007 | 21.597 |
| Ours alone, Dense | 499.574 | 83.788 | 28.899 |

Baseline before/after rounds are preserved. The link model uses the faster
complete round for each objective; it does not combine incompatible calls.
Fixed ExANS remains an additional control beside its compact carrier.

Original standalone profiles are in
`benchmark_results/team_lookup_a4500_20261028_141701_1145261_report_20261028_143153/profiles`.
Across three 8K text decode calls, entropy-layer kernels account for about
20.444 ms of accumulated GPU work per Rank256 request and 21.393 ms per Dense
request. Rank products add about 4.932 ms cuBLAS and 3.420 ms requantized first
products; Dense products add about 15.218 ms. Streams overlap, so these costs
are nonadditive and are not a wall-time decomposition.

The earlier central/tail lookup was slightly slower than binary search.
Register partner reuse and more decoder warps also failed to improve those
experiments. This table is a different lookup design, not a rerun of that
central/tail implementation. CUDA graph pointer indirection, exact GEMM
scheduling and token chunking remain separate future experiments.

## Exact change

The current decoder maps a 16-bit rANS slot to a probability symbol using a
coarse table followed by binary search. That search performs repeated CDF
loads and block-wide convergence checks inside the token loop.

The candidate precomputes the complete inverse for all ten fixed probability
classes and all 65,536 slots. One UINT16 load returns the same symbol:

`symbol = inverse[class, slot]`

The shared table occupies exactly 1,310,720 bytes (1.25 MiB). It is built once
per installed codec and is additional GPU state, not additional packet bytes.
All subsequent CDF frequencies, rANS updates, BF16 interval/word recovery,
escape handling, prediction, RoPE, features and uncertainty calculations keep
their original arithmetic. Sender encoding is unchanged.

An exhaustive CPU proof compares all 655,360 entries with independent
searchsorted, a simulation of the original packed coarse binary search, and
the original central/tail reference. The proof binds the fixed table hashes
to the codec's native CDF receipt. Owned lookup storage is version-guarded.
The decoder dispatch is changed only on the candidate instance before graph
capture; vendor snapshots and baseline implementations remain unchanged.

This is an execution optimization of our conditional lossless codec. Table
lookup itself is not a novelty claim. Cache pressure may erase or reverse the
speed benefit; no speedup is assumed before the actual A4500 measurement.

## Frozen comparison

Both candidates are standalone: Dense and Rank256 at 512 and 8192 tokens.
At 512 both reference and candidate retain the existing checked graph path;
at 8192 both retain borrowed-input direct decoding with a fresh output.
The full lookup replaces only CDF inversion in either path. No layer mask,
fallback SplitZip source/codebook, mixed codec or alternative predictor is
accepted in a candidate receiver specification.

The runner uses 22 workers and 44 rows: sixteen unchanged frozen jobs, four
candidate jobs and two unchanged compact ExANS jobs. The unchanged jobs
include standalone Dense/Rank references plus raw BF16, official framed
SplitZip and fixed ExANS before/after controls. Each job restores both text
and code packets and launches a fresh receiver process.

Candidate encoding must match the original complete packet SHA, byte count,
header, geometry and state exactly before timing. Receivers receive only
fixed state and encoded packet assertions, alternate 0/1/0, retain earlier
outputs and verify original word hashes and packet immutability. The latest
214-file report and its recursive frozen-source/runtime/calibration chain are
verified; no pending hybrid report is required.

Two mandatory actual-CUDA test methods run before benchmarking. They check
the complete inverse table, original native streams, all 65,536 BF16 word
patterns, and frozen production Dense/Rank graph/direct paths with retained
outputs and changing caller streams. CPU/interpreter checks establish
software semantics only and cannot substitute for those GPU gates.

Full encode and decode calls retain the unchanged author timing body:
10 warmups, 50 calls, five repeats. Separate isolated measurements use two
warmups and five samples, with synchronized wall time, CUDA intervals and
Torch memory peaks. Shared state, new LUT bytes, output/scratch and native
allocation limitations are separately disclosed. Cold installation/capture
and file export are outside warm codec timing.

After ordinary measurements, a fresh reference instance and the warm candidate
collect separate warmed CPU/CUDA profiles for each 8K text arm. Their kernel work and
stage diagnostics remain nonadditive; full traces stay on the NUC.

## Objective and run

Prioritize lower and middle link rates while keeping the smaller standalone
packets. Report all 1, 5, 10, 25, 50, 75, 100, 200, 300 and 400 Gb/s scenarios.
Keep precompressed `wire + decode` separate from fresh
`encode + wire + decode`; a decoder gain does not remove sender encoding cost.
Do not choose a deployment policy from these reused TEST documents.

On the NUC, switch to `experiment/standalone_full_lut_v1`, then run
`bash scripts/run_standalone_lut_a4500.sh` in `~/kv_reconstruction`.
The wrapper publishes the new report and preserves failure exit codes.

This experiment uses pinned Llama-3.2-3B BF16 and the A4500. Two reused
development documents do not establish independent generalization. Link
costs remain calculated, not NIC/SSD measurements. All methods reconstruct a
full BF16 output: this is transfer/storage T, not compressed-resident attention
P or serving throughput. AQUA-KV and other lossy methods are outside this
lossless comparison.
