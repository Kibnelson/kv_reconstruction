# Review of the 29 October milestone against the verified transfer track

This is a source/results review, not a new GPU experiment. It compares the
[milestone at 8a40dd9](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/docs/milestone_report_2026-10-29.md)
and its referenced result bundle `llama32_3b_compare_a4500_20261029_0518` with
our [frozen transfer result at d0972c3](https://github.com/Kibnelson/kv_reconstruction/blob/d0972c3/benchmark_results/frozen_transfer_a4500_20261028_215633_2611827/summary.md).
The milestone report and its branch have not been edited by this review.

The branches have the same fast GPU codec foundation. The milestone adds useful
predictor, mixed-format and lossy-quality experiments. The transfer track adds
complete checked packets, full GPU encode/restore API measurements, explicit
shared-state accounting and fresh-process receiver proofs. Combining the work
requires aligning metrics, not choosing one branch's headlines over the other.

## 1. The core is the same

The milestone branch's `core.py`, `predict.py`, `fused.py`, `layerwise.py` and
`gpu_coder.py` are byte-identical to the corresponding files in our
`kv_compare/team_onestep/` snapshot, pinned at `a7778073902bc2bb8bb0433ff7cd2e10bd79f5c3`.
Both use the pinned original BF16 Llama-3.2-3B and the PEP/CPython captures.

Compare the same first PEP document and TB32/two-chunk configurations:

| Mode | Milestone native stream bpv | Our complete packet bpv | Milestone native decode | Our full restore API |
|---|---:|---:|---:|---:|
| Dense | 9.025188 | 9.025758 | 31.51 ms | 29.15 ms |
| Rank256 | 9.404913 | 9.405483 | 22.76 ms | 21.64 ms |

The size gaps amount to only 33,471 and 33,474 bytes, consistent with small
directory/envelope accounting differences. This review proves core source
identity, not cross-branch coefficient or payload digest identity. Our headline 8.92097/9.30632
bpv averages two TEST documents; comparing that average with a selected
single-document milestone configuration would invent a compression improvement.
The execution paths also differ: native preallocated CUDA-event intervals versus
complete compact-packet restore with an independently owned output.

## 2. The 400-Gb/s claim uses ideal overlap

[scoreboard.py](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/scripts/scoreboard.py#L112-L128)
uses `max(wire, decode)`. Neither branch has demonstrated the network/decoder
pipeline required to attain that bound. Our serial scenario uses `wire + decode`
and reports ideal overlap separately. Fresh generation additionally needs encode;
both models exclude host export/hash/upload unless those stages are added.

For the milestone's selected 400-Gb/s mix, its own measured native rows give:

| Model | Selected 14-layer mix | SplitZip |
|---|---:|---:|
| Wire alone | 13.03 ms | 14.15 ms |
| Decode alone | 12.62 ms | 3.04 ms |
| Ideal `max(wire, decode)` | **13.03 ms** | 14.15 ms |
| Serial `wire + decode` | 25.65 ms | **17.19 ms** |

Reselecting across **all** measured bit-exact modes under the serial objective
does not establish a win either. The best mode containing at least one of our
layers is the two-layer rank256 fixed-width mix:

| Link | Best nonzero-ours mix, serial | Native SplitZip, serial |
|---|---:|---:|
| 100 Gb/s | 63.40 ms | **59.64 ms** |
| 200 Gb/s | 35.44 ms | **31.34 ms** |
| 400 Gb/s | 21.46 ms | **17.19 ms** |

These remain calculations from the team's native rows, not measurements of
network transfer or our complete T wrapper. The milestone should say
"an advantage under an ideal-overlap model," rather than presenting the 400-Gb/s
result as an established transfer win.

Layer masks are chosen from TEST per-layer sizes and the scoreboard minimizes
over TEST configurations per link. Falling back to whole-document SplitZip
makes the candidate envelope "never slower" by construction. Our own earlier
execution choices were also informed by profiling this frozen pair. Both tracks
need validation-selected policies and new held-out documents before claiming
generalization; a one-commit lock does not remove selection on TEST.

## 3. Encoder times cover different work

In [gpu_speed.py](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/kv_compare/gpu_speed.py#L558-L590),
features and priors are prepared before the timed `encode`, and the encoder
reuses an output arena. The log explicitly says "after the forward pass."

The selected Rank256 row has 24.41 ms coding and a separately measured 16.92 ms
model pass. That explains why our complete encode is about 42.14 ms. Adding the
separately synchronized components is only an estimate, not a replacement for a
direct full-call measurement. The reported 24 ms does not establish a twofold
encode improvement over our full API. Likewise, the fused exact row has about
54.82 ms encoding plus 41.64 ms preparation.

Keep prediction, prepared-input coding, full GPU encode, export/checksum/upload,
native decode and complete restore in separately named fields. CUDA-event
intervals may include launch gaps; they are not simply sums of kernel durations.

## 4. ExANS: native used bytes, reserved bytes and complete T

The milestone's approximately 608 MiB is a valid **native used-record byte**
metric. [exans_official.py](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/kv_compare/exans_official.py#L139-L215)
reads `actual_bytes` from native headers. It nevertheless reserves records of
999,408 bytes each: 940 records occupy approximately **895.92 MiB**. The code
does not compact/export those used bytes and restore native slots from a wire
packet; its decode reads the slot array and writes a preallocated output.

Our approximately 691.05 MiB instead charges the complete fixed-record carrier
using 770,048-byte slots, padding, directory, framing and any raw fallback. Its
23.63-ms restore and the team's 17.5-ms native decode have different boundaries.
Neither number should replace the other without implementing the corresponding
carrier. The used-byte result suggests a worthwhile **compact ExANS carrier**
experiment that could strengthen our baseline.

The native measurement also floors to 940 whole blocks and omits **32,768 BF16
words / 64 KiB** from the 8K tensor. Exactness covers 469,729,280 words, not the
full 469,762,048. Transport the tail and require exact total coverage. All current
blocks fit; the script does not implement the raw-fallback reconstruction its
comment suggests. Lock promotion should fail on equality, magic, fit or coverage
errors; currently the scoreboard does not enforce all of those gates.

Use five explicit columns: native used bytes, reserved slots, complete T bytes,
native time and complete restore time. Fresh binary/library/source/environment
receipts from our native preflight should accompany the team's measurements.

## 5. Lossy work: useful new quality evidence, with different claims

The 35-window growing-cache WikiText test is a meaningful advance over the
earlier static-prefix continuation tests. Its raw PPL matches the author run at
6.98645. The local results recompute: our step 2 is about +0.54%, step 4 +11.25%,
and the adapted AQUA 2-bit row +1.51%. The sink, token-step and negative Fisher
experiments provide useful direction. Raw PPL agreement validates part of the
workload/evaluator, not every compressed-cache policy.

Required qualifications:

- **Official AQUA reproduction remains pending at this commit.** The local
  predictors use the team's TRAIN documents without the official holdout. The
  backup helper uses a mutable mirror revision and row-count checks, which do
  not establish ordered-row/token identity with the original data. Tokenizer
  fallback and possible RAM-driven sample reduction are explicit protocol
  variations. Pin the mirror, sample-token digests, tokenizer and actual
  256/32 training/holdout configuration before using a reproduction label.
  These receipts make a disclosed backup-data variation reproducible; they do
  not by themselves prove equivalence to the original ordered RedPajama corpus.
- **Rates are not measured physical window bytes.**
  [streaming.py](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/kv_compare/streaming.py#L184-L261)
  estimates packed HIGGS codes and 16-bit scales. The official
  [HIGGS implementation](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/aquakv/quantizers.py#L44-L68)
  stores uint8 indices and FP32 scales. Reconstructed cache tensors are retained
  for the quality test. Even the JSON's nominal whole-cache accounting, including
  exact sinks/recent values, gives about **126.39 MiB AQUA 2-bit, 163.43 MiB ours step 2,
  and 131.06 MiB ours step 4**, rather than 117/154/122. Label all these as logical
  equivalents until actual packed carriers/allocations are measured.
- **Received-stream quality is not yet checked by this path.** `OursGpu` encodes,
  discards the payload and feeds `rec`, the encoder reconstruction, back to the
  model. Add actual decode and block/start/per-lane parity checks, preferably a
  fresh receiver, before attaching quality to a transmitted format.
- **78/46-ms baseline speed figures are synthetic component tests.**
  [time_baseline_decoders.py](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/scripts/time_baseline_decoders.py#L1-L15)
  uses random AQUA predictors/Gaussian residuals and random KIVI tensors. It
  manually recreates the AQUA read path; KIVI restores BF16 rather than exercising
  its packed attention path. These are not the trained packets in the quality
  table or official serving measurements. Keep the microbenchmarks, label them,
  and measure the actual matched quality packets separately.
- **KIVI is an adapted common-cache experiment.** The generic flush policy is
  not its native rolling value-tail policy. Better PPL than the paper is not a
  fidelity check. Likewise, the report's "beats HIGGS at every rate" is contradicted
  by HIGGS 2-bit at 113 MiB/+8.08% versus ours at 122 MiB/+11.25%, or 129 MiB/+2.53%:
  neither tested point dominates HIGGS 2-bit in both dimensions.
- **Error placement is a hypothesis.** The similar KV error norms quoted are
  from the static transfer test, whereas the large PPL gap is from growing-cache
  evaluation. Measure in-protocol error before excluding error magnitude.

## 6. The pool table is a store projection, not attention-facing P

[stage_pool](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/scripts/bench_llama_compare.py#L1115-L1157)
measures prefill, raw host copies and free memory. Counts in
[scoreboard.py](https://github.com/Kibnelson/kv_reconstruction/blob/8a40dd9/scripts/scoreboard.py#L214-L217)
are `floor(free / inferred compressed bytes)`, not measured multi-prefix capacity.
Reserve coefficients once per model, a full 896-MiB BF16 restore output, workspace
and allocator space. ExANS's 608-MiB used-byte count cannot replace its current
approximately 896-MiB record allocation in a physical capacity calculation.

This can motivate a **dormant compressed T-prefix store**. It does not demonstrate
attention consuming compressed P, a reusable serving allocator or T-to-P
conversion. Our P1 GPU packing/selected-tile work is a separate earlier prototype;
our Llama compressed attention and T-to-P path are also unverified.

The quoted 117 MB is approximately the dense coefficient matrix alone
(117,440,512 bytes = 112 MiB). Our measured installed shared tensors are 117.31 MiB
Dense and 146.24 MiB Rank, plus coder tables. Charge cold installation separately,
and reserve warm shared state in pool budgets.

## 7. What to carry forward

1. Keep our checked T framing/receiver/accounting and common full-API comparison
   as the transport reference. Preserve native metrics as separate diagnostics.
2. Bring the team's stronger no-attention predictor and mixed fixed-width/rANS
   variants into that harness as new arms. Select masks/configurations on
   validation, freeze them, then evaluate fresh held-out documents.
3. Retain the growing-cache lossy evaluation and negative experiments. Fix
   actual-decode checks, whole-cache/physical accounting and baseline labels
   before locking the joint scoreboard.
4. Add compact ExANS as a stronger complete-T control, including tail, directory,
   compaction/scatter and full restore costs. Keep its native used-byte metric.
5. Keep P experiments distinct from dormant T storage and byte-capacity bounds.

The 7.87-bpv predictor is a promising **size arm**, not a demonstrated fast-link
win. At 441 MiB its wire cost is about 37 ms at 100 Gb/s. With the projected 55–70 ms
decode, ideal overlap is 55–70 ms versus SplitZip's 56.6 ms; serial cost is 92–107 ms
versus about 59.6 ms. A projected 1.5x benefit fits a wire-dominated slower link,
not the stated 100-Gb/s case.

Finally, a coder close to its current model's ideal cost leaves little packing
overhead, but does not prove a fundamental entropy floor. Better probability
models or conditioning, as well as stronger predictors, can reduce cross-entropy.
Locking consistent scope and provenance should precede new headline comparisons.
