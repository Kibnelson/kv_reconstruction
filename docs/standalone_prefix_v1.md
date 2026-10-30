# Standalone lossless decoder refill experiment

The next target is faster GPU reconstruction of the same compact transfer
packets. Every one of the 28 layers uses our own predictive/rANS codec.
SplitZip and ExANS remain independent, unchanged baseline controls.

## Starting point and target

The completed A4500 comparison is commit `d5aee68`, report
`benchmark_results/standalone_inverse_a4500_20261030_060024_1787751`.
It verified 52 exact rows, 26 fresh receivers and three actual CUDA gates.
Packed inversion improved 8K isolated decode by about 2% without changing
packet bytes. Masked binary search did not provide a consistent gain.

| 8K result, mean over two engineering documents | Dense packed | Rank256 packed | SplitZip | Compact ExANS |
|---|---:|---:|---:|---:|
| Complete packet MiB | 499.574 | 521.154 | 674.807 | 609.129 |
| Warm isolated encode ms | 42.788 | 34.613 | 10.302 | 209.828 |
| Warm isolated decode ms | 28.159 | 21.184 | 3.372 | 23.593 |

SplitZip encode/decode cells above use the faster before/after observation
for each operation. Link comparisons instead retain one complete baseline
round per objective, as in the preceding experiment.

With an already compressed Rank256 packet, decoding must fall to roughly
16.3 ms to match SplitZip at a calculated 100 Gb/s. At 400 Gb/s the budget
is roughly 6.6 ms. These are targets derived from bytes and measured local
latency, not promised performance or measured network paths. The new report
recomputes both budgets from its own baseline observations and packet sizes.
Fresh encoding plus transfer plus decode is a separate objective.

## What the entropy-decoding kernel does

The predictor provides a probability distribution. The entropy decoder uses
that distribution and an integer rANS state to recover the original words:

1. Identify the value's probability interval from its current state. The
   retained packed inverse resolves this using compact exact tables.
2. Recover the original BF16 word inside that interval, or its raw escape.
3. Update the integer state. If the state becomes too small, fetch the next
   16-bit word from the compressed stream.

The prediction may be imperfect. Reconstruction remains exact because the
stream identifies the original BF16 word, rather than a rounded prediction.

A group of 128 logical channels shares a stream. Only some channels need a
refill on a given step. They must read successive words in channel order.
For example, starting at stream position 100:

| Channel | Needs refill | Stream position read |
|---|---|---|
| 0 | yes | 100 |
| 1 | no | none |
| 2 | yes | 101 |
| 3 | yes | 102 |

The next shared position is 103. Computing these addresses is a prefix count:
each requesting channel counts the earlier requesting channels. The original
helper computes a cumulative sum and a total over the 128 flags. It runs
twice per decoded value step, because the format has two entropy symbols.
It is part of the hot decoding loop, not a one-time setup cost.

## Candidate

Keep packed inversion, the rolling single-feature encoder, two decoder
chunks, fitted coefficients, probability tables, rANS arithmetic and every
complete packet byte. Change only the refill prefix-count implementation.

The candidate uses warp votes and population counts for four ordered groups
of 32 flags. A warp vote produces a 32-bit mask; counting set bits before a
lane gives its local prefix. Add the preceding groups' counts to preserve
the original 128-channel stream order. No new predictor or probability
tables are introduced.

Logical tensor positions must match the physical warp operations. Triton's
inline-assembly grouping is not a general layout guarantee. Compiler checks
and actual CUDA mapping probes are therefore required; CPU arithmetic checks
alone cannot validate this optimization. Unsupported compilation or mapping
must fail rather than silently switch to an unreported implementation.
Each production specialization is compiled and inspected before its first
execution. Warm calls retain the lightweight verified-variant cache lookup
inside measured API latency; they must not compile, inspect layouts or run
mapping probes. An unseen variant during CUDA graph capture is an error.

The latest Rank profile attributed about 20.6 ms of accumulated GPU work per
cycle to the entropy decoder. Profile sums overlap other measurements and
are nonadditive. They identify the broad target, not how much a prefix change
will save. Reaching 16.3 ms requires about a 23% end-to-end improvement;
prefix counting may account for too little work to achieve it by itself.

## Evaluation and interpretation

Use the same pinned Llama-3.2-3B BF16 captures and shared state. Compare
existing packed decoding against the candidate for Dense and Rank256 at
512 and 8192 tokens. Keep before/after raw, SplitZip and fixed ExANS controls
and the compact ExANS controls. This gives 22 workers, 44 document rows,
22 fresh receivers, 16 own frozen-packet rows and eight candidate/reference
document groups.

Actual CUDA gates must cover structured/random refill masks, returned read
positions and totals, original state updates, word/escape cases, and the
production graph/direct paths. Every own encoded packet must retain its
original frozen SHA. Fresh receivers get the actual encoder packets and
fixed state, without source KV or precomputed predictions. Preserve
alternating packets, independent returned outputs and packet immutability.

Retain the author timing body: 10 warmups, 50 calls and five repeats.
Separately measure isolated calls with two warmups and five samples. The
512-token path retains graph replay and owned output cloning; the 8K path
retains borrowed compact input and fresh full BF16 output. Record cold setup,
compilation, shared state and measured allocation peaks separately.

Profiles follow ordinary timing. They must prove dispatch of the new decoder
and unchanged prediction/encoding work. Report both timing scopes even if
they disagree. A local prefix microprobe or instruction count cannot replace
whole decode latency. A candidate is useful only if exactness, packet size
and memory accounting hold and the matched measurements support its gain.

Calculated links remain 1, 5, 10, 25, 50, 75, 100, 200, 300 and 400 Gb/s,
with precompressed and fresh encoding objectives separate. Two reused
engineering documents do not establish generalization. These are transfer
format T experiments with full BF16 reconstruction, not measured NIC/SSD,
serving throughput or compressed-resident attention P.

## NUC run

Switch to `experiment/standalone_prefix_v1` inside `~/kv_reconstruction`,
then run `bash scripts/run_standalone_prefix_a4500.sh`. The wrapper publishes
the report, preserves a failure's original exit code and leaves encoded
packets and full profiler traces in the run directory. Local validation is
CPU correctness and offline compiler evidence only; actual CUDA correctness
and performance remain for the A4500 run.
