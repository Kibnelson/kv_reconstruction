# Standalone lossless entropy-decoder core experiment

All 28 layers use our predictive/rANS codec. Keep the successful rolling
encoder, single feature buffer and two decoder chunks. This experiment
changes only how the decoder finds a symbol in the same fixed stage-one
probability tables. Prediction, rANS state updates, BF16 word reconstruction,
complete packet bytes and the encoder remain unchanged. SplitZip and ExANS
are independent unchanged controls.

## Completed scheduling result

The preceding A4500 run is commit `b958150`, report
`benchmark_results/standalone_schedule_a4500_20261029_195745_1506118`.
All 189 manifest entries, 56 exact rows, 28 fresh receivers, 20 paired packet
groups and two actual CUDA gates passed. All 28 own rows retained their
original frozen packet SHA. Measurement sources, shared state and native
author libraries remained unchanged. The fixed ExANS control may reencode
to different bytes; its unchanged worker checks the new packet, exact words,
geometry and accounting.

Retain one reusable feature matrix for its measured memory benefit:

| 8K isolated API result | Rolling reference | Single feature |
|---|---:|---:|
| Dense encode ms | 42.730 | 42.727 |
| Rank256 encode ms | 34.675 | 34.662 |
| Dense encoder incremental peak MiB | 2357.125 | 2293.955 |
| Rank256 encoder incremental peak MiB | 2359.125 | 2315.232 |

Feature storage itself fell by 416 MiB, but the observed full API reductions
are 63.170 MiB Dense and 43.893 MiB Rank256. Packet construction now sets the
peak through simultaneous arena and compact-output storage. There is no
meaningful 8K encode speed gain. Small 512-token encode gains are consistent;
the apparent isolated Rank decode gain is not an encoder effect and is not
supported by batched decoding.

Keep two decoder chunks. Other chunk counts regress or give inconsistent
results. At 8K, Dense isolated decode is 29.048 ms for two chunks versus
30.384/29.640/29.457 ms for one/four/eight. Rank256 is 21.612 ms versus
22.565/22.117/26.286 ms. More launches do not shorten the measured critical
path. No hardware-counter evidence establishes a cache or occupancy cause.

Current single-feature packets remain 499.574 MiB Dense and 521.154 MiB
Rank256. Its 8K decode means are 28.712 and 21.620 ms. SplitZip is about
674.807 MiB and 3.38 ms; compact ExANS 609.129 MiB and 23.596 ms. We preserve
the compression advantage but still need a substantial decoder gain against
SplitZip at faster links.

## Why change the inverse algorithm

The decoder's stage-one symbol is determined by a 16-bit rANS slot and one
of ten fixed CDF classes. The current coarse table narrows a symbol range,
then a program-wide binary-search loop runs until every channel finishes.
A rare channel requiring more comparisons can keep already-resolved channels
executing table reads and reductions. The entropy decoder remains roughly
21–22 ms of accumulated GPU work per profiled request; separate profile
totals are nonadditive and do not establish a promised speedup.

Previous full inverse lookup added 1.25 MiB and slowed decoding. The earlier
central/tail lookup also failed to improve the binary reference. Test two
different, independent changes rather than promoting those failed variants.

### Mask resolved search lanes

Keep the existing search and tables, but mask the cumulative-table load for
channels whose interval is already resolved. Their symbol stays fixed while
unresolved channels continue. This adds no device tables. The packed zero
returned for masked lanes reproduces their original comparison result, since
their lower bound already contains the slot. Exhaustive proof must cover
mixed resolved and unresolved channels, not just one symbol at a time.

### Packed cutoff plus sparse exceptions

Partition each class's 65,536 slots into 1024 buckets of 64 slots. Most
buckets contain only one or two symbols. Store a starting symbol and one
local cutoff in a single 32-bit entry, so a comparison resolves the symbol
without a search or another cumulative-table load. A cutoff of 64 handles
single-symbol buckets.

Only 36 buckets across all ten classes contain more than two symbols. Their
entries point to exact 64-slot UINT16 exception tables. The layout is:

| Additional table | Shape/type | Tensor bytes |
|---|---|---:|
| Packed coarse entries | 10 x 1024 INT32 | 40,960 |
| Sparse exception symbols | 2304 UINT16 | 4,608 |
| Total | | 45,568 |

The original 57,688-byte coder tables remain installed and unchanged. Packed
mode therefore reports 103,256 total coder-table bytes, with 45,568 additional
persistent tensor bytes. Masked mode reports the original 57,688 bytes and
zero additional tables. Predictor state is additional and unchanged.

This differs from the prior central/tail lookup: the usual path gets its
base and cutoff from one entry, without two class-boundary metadata reads or
a second CDF boundary read. It differs from the full inverse table by storing
direct slot symbols only for the 36 complex buckets. It remains a performance
hypothesis; small tables do not guarantee faster GPU execution.

Both inverses must reproduce all 655,360 class/slot combinations of the
original discrete CDF exactly. The original CDF tensors and derived table
hashes bind this proof to the installed native state. No probabilities are
refitted and no KV values are approximated.

## Matched evaluation and run

Four current single-feature/two-chunk references and eight candidates cover
Dense/Rank256 at 512 and 8192 tokens. Retain the 12 before/after raw,
SplitZip and fixed ExANS control jobs plus two compact ExANS jobs: 26 workers,
52 rows and 26 fresh receivers. All 24 own rows must preserve frozen packet
SHA; 16 candidate/document groups compare against the current references.

Three actual CUDA test methods gate timing: both inverses for every fixed
CDF slot, BF16-pattern/symbol coverage, and production Dense/Rank512/8K
graph/direct packet reconstruction with caller-stream and lifetime checks.
Fresh receivers receive only fixed state, actual encoded packets and the
declared inverse mode. Preserve alternating inputs, retained independent
outputs and packet immutability.

The frozen author timing body uses 10 warmups, 50 calls and five repeats;
isolated API timing uses two warmups and five samples. Report actual Torch
allocation peaks and separate cold setup/table installation. Native allocator
peaks outside Torch remain unmeasured. After ordinary timing, four paired
8K profiles verify 168 replacement decoder kernels versus zero original
decoder kernels over three cycles, with the opposite counts in references.
Prediction products, feature passes and the encoder remain unchanged.

Report calculated 1–400 Gb/s scenarios for fixed arms, separating
precompressed wire plus decode from fresh encode plus wire plus decode.
No deployment selection is fitted on these reused development documents.
These are transfer/storage T experiments with full BF16 restoration, not
measured NIC/SSD, serving throughput or compressed-resident attention P.

On the NUC, switch to `experiment/standalone_inverse_v1` and run
`bash scripts/run_standalone_inverse_a4500.sh` inside `~/kv_reconstruction`.
Reports publish automatically, retaining a failure's original exit code.
Encoded packets and full profiler traces remain in the run directory.
