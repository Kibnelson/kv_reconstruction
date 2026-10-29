# Lossless critical-first fallback/rANS overlap

This experiment changes GPU execution only. It preserves the complete v2
packets, original BF16 words, calibrated predictors and original TABLES masks.
The team branch and every existing source file remain unchanged. SplitZip and
ExANS controls retain their pinned author implementation and timing bodies;
partitioned mapped kernels are explicitly a local integration variant.

## What the completed v2 experiment established

Source/result commit: `95889040b86862ac79d4b188c6441d456267f28e`.
Report: `benchmark_results/lossless_decode_a4500_20261029_125616_867717`.
All 214 packaged files match their receipts, 68 rows restore original words,
34 independent receivers pass, and all three actual CUDA gates pass. Eight
document/length/mask groups preserve identical packet hashes and rANS fields
across all four execution arms and the prior bundled format.

These are 8K isolated per-request results, averaged over the two development
documents. They are not the separate batched-call measurements.

| Method | Complete MiB | Encode ms | Decode ms |
|---|---:|---:|---:|
| SplitZip unchanged framed control, faster complete round | 674.807 | 10.288 | 3.368 |
| Compact ExANS carrier around unchanged native codec | 609.129 | 210.076 | 23.555 |
| Mapped + selective k2 | 663.993 | 12.583 | 5.321 |
| Mapped + selective k14 | 595.837 | 26.549 | 13.228 |
| Existing Rank256 | 521.154 | 42.007 | 21.597 |
| Existing Dense | 499.574 | 83.788 | 28.899 |

Mapped I/O reduced k2 decode from 8.888 to 5.321 ms and k14 from 15.254 to
13.228 ms. Incremental Torch decode peaks, including the full output, fell
from 1728/1344 MiB respectively to 896 MiB. Context skipping alone saved only
about 1–1.5%. Complete bytes did not change.

The new mixes still sit above the Dense/Rank/SplitZip serial cost frontier.
For precompressed 8K transfer, k2 is 61.021 ms versus SplitZip 59.975 ms at
100 Gb/s; at 400 Gb/s it is 19.246 versus 17.519 ms. K14 uses 2.18% fewer
bytes than compact ExANS and decodes 1.78 times as fast, but that does not
make it the best choice among all controls.

## Dependency being changed

The v2 controller decodes all fallback layers before starting rANS. Only the
fallback layer immediately preceding a coded layer is a required input to
rANS; the other fallback layers write independent final-output rows.

| Frozen mask | Critical fallback layers | Critical output at 8K | Bulk output at 8K |
|---|---|---:|---:|
| k2 | 0, 26 | 64 MiB | 768 MiB |
| k14 | 0, 3, 15 | 96 MiB | 352 MiB |

The candidate decodes critical dense values and their escapes on the caller
stream. A dedicated stream then decodes bulk fallback rows while rANS chunk
streams process coded layers. The caller joins both before returning its
fresh output. Cross-stream calls serialize reuse of decoder workspaces.

Dense grids visit only selected logical bundle ordinals. Escape repair scans
the existing list once per nonempty partition and predicates stores by
partition; it does not assume sorted escapes. Owned ordinal tensors and a
membership table are prepared once and charged as shared state. No BF16
staging buffer or transmitted metadata is added. Kernel attribution and the
SplitZip MIT license remain with the source.

The k2 profile contains about 2.678 ms of mapped dense fallback work plus
0.127 ms of escape repair per request, before rANS begins. Moving bulk work
off that dependency path is a hypothesis: memory bandwidth and SM contention
can reduce or erase the benefit. These summed GPU times are not additive
wall-time components or a prediction of speedup.

## Frozen experiment

Two modes use original k2/k14 masks at 512 and 8192 tokens on the same two
documents. Reference is the unchanged v2 mapped + selective codec. Candidate
changes only the partitioned fallback controller. Both must match the prior
complete packet SHA, bytes, original-word SHA and compact rANS fields.

There are eight new jobs, sixteen frozen control jobs and two compact ExANS
jobs: 26 workers, 52 rows and 26 fresh receivers. Controls include raw BF16,
unchanged framed SplitZip, fixed ExANS, compact ExANS, Dense and Rank256.
Receivers receive only packet/state inputs, alternate packets 0/1/0, retain
previous outputs and verify original words and input immutability.

Two mandatory real-CUDA test methods run first. Semantic tests cover all
65,536 BF16 words, partition boundaries, escapes, empty partitions, nonzero
prediction and adjacent coded layers. Production CUDA tests also exercise
borrowed packet lifetimes, delayed bulk work, back-to-back caller streams and
512-to-8192 shape changes. Local CPU/interpreter checks cannot establish GPU
correctness or performance.

Timing retains the author's whole-call body: 10 warmups, 50 calls, five
repeats. Separate isolated timing uses two warmups and five samples. Four
8K jobs collect warmed, separate profiles. Candidate diagnostics record
critical, bulk and rANS-controller event intervals, including waits, apart
from performance measurements; interval intersection does not establish
simultaneous kernel residency. Complete encode/decode, memory and calculated
1–400 Gb/s serial precompressed/fresh costs remain separately reported.

At unchanged bytes, the k2 decode budgets to match SplitZip are 4.275 ms at
100 Gb/s and 3.595 ms at 400 Gb/s. The fresh-transfer comparison also charges
encoding and is stricter. No expected win is baked into validation or policy.

## Run and scope

On the NUC, switch to `experiment/lossless_overlap_v3` and run
`bash scripts/run_lossless_overlap_a4500.sh` from `~/kv_reconstruction`.
The wrapper verifies receipts, runs CUDA gates and benchmarks, packages logs
and reports, and publishes only the new result directory to this branch.
Packets and full traces stay in the run directory. Failure is published with
its original nonzero exit code and must not be reported as a completed run.

Model: pinned `unsloth/Llama-3.2-3B` BF16, all 28 layers, A4500. These two
documents have been reused for engineering and do not establish independent
generalization. Link costs are calculated scenarios, not NIC, SSD or serving
measurements. This is GPU transfer format T, with a full reconstructed BF16
output; compressed resident attention P is outside this experiment. Lossy
methods, including AQUA-KV, remain outside this lossless comparison.
