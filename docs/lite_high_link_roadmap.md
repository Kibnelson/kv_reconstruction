# Lite: target a better latency/bytes frontier through 400 Gb/s

The objective is exact reconstruction with fewer transferred/stored bytes and
total latency at or below the strongest suitable baseline. The latest V64/B32
finish-fused graph path saves 43.19% and decodes a 56 MiB request in 88.224 ms.
Its modeled 1 Gb/s fetch beats ExANS and remains about 1.02 ms behind SplitZip.
At higher link rates, saved transmission time shrinks substantially.

The [finish-fusion findings](lite_finish_findings.md) explain the completed
paired GPU result and the next controlled allocation-fill experiment. Retimed
SplitZip is slightly faster in that run, so the budgets below use its latest
0.512 ms decode rather than the previous run's 0.555 ms.

## What the target requires

Serial precompressed fetch is `decode_ms + 8*packet_bytes/(Gbps*1e6)`.
For the current packet and retimed baseline, the maximum Lite decode time to
match SplitZip is approximately:

| Effective link | Required decode ms | Current decode ms |
|---:|---:|---:|
| 1 Gb/s | 87.204 | 88.224 |
| 5 Gb/s | 17.851 | 88.224 |
| 25 Gb/s | 3.980 | 88.224 |
| 100 Gb/s | 1.379 | 88.224 |
| 400 Gb/s | 0.729 | 88.224 |

At 400 Gb/s, the current Lite packet takes 0.667 ms to transmit. SplitZip's
complete modeled fetch takes 1.396 ms, leaving Lite 0.729 ms to decode. Matching
raw BF16 transfer+decode instead permits 0.857 ms. These are calculated targets,
not measured network results. They imply roughly 121x and 103x further decoder
improvement, respectively; the next fusion alone is not a 400 Gb/s solution.

A separate steady-throughput target is approximately 88 GB/s of reconstructed
BF16 output to consume this compressed representation arriving at 400 Gb/s.
Meeting that rate through batching does not prove the same latency for one
request. Reports must keep single-request latency, batch throughput and queueing
separate. Cold capture and fixed state are additional costs.

The A4500 has a PCIe 4.0 x16 interface. NVIDIA's PCIe reference lists about
31.5 GB/s theoretical bandwidth for that link, below the 50 GB/s represented by
400 Gb/s. Thus a single such PCIe ingress path cannot sustain full 400 Gb/s
compressed arrivals. The A4500 remains useful for compute experiments and
modeled link sweeps; actual 400 Gb/s validation requires a suitable end-to-end
hardware path. This inference concerns PCIe ingress, not the card's much faster
local GPU memory or an assumed NVLink path. Actual negotiated links and transfer
rates have not been measured here. The finish run records negotiated PCIe Gen4
x16 metadata; that is not a bandwidth measurement.

Primary interface references:

- [NVIDIA A4500 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx/nvidia-rtx-a4500-datasheet.pdf)
- [NVIDIA PCIe ceiling reference, section 6](https://github.com/NVIDIA/DeepStream/blob/main/skills/deepstream-profile-pipeline/references/hw-ceiling-formulas.md)

## Ranked experiments

| Arm | Change | What it targets | Status |
|---|---|---|---|
| A | Fuse BF16 conversion, inverse RoPE, feature writes and recovered-word writes | Repeated kernels, temporary buffers and memory traffic | GPU completed: decode 96.206 to 88.224 ms, identical packets |
| A2 | Remove protective allocation fills after a full-write audit | Empty-buffer initialization and graph nodes | Implemented experiment; GPU result pending |
| B | Use an exact integer-symbol/table decoder | Dynamic floating-point interval inversion and symbol searches | Proposed; first measure held-out rate/escapes |
| C | Fit smaller predictors and a few dependency waves | Dense products and 448 sequential groups | Proposed; new calibration and packet sizes required |
| D | Transmit small prefill hints as parallel prediction context | Replace expensive reconstruction dependencies with cheap shared context | Proposed; new hooked captures and full side-information accounting required |
| E | Batch independent packets and pipeline input/decode | Occupancy and steady throughput | Proposed; measure latency/queueing too |
| F | Direct staging, safe GPU validation and owned output slots | Endpoint costs that become large at sub-millisecond budgets | Proposed; retain validation and ownership guarantees |

### A: completed controlled fusion

`FinishEngine` keeps the existing predictor, probability model, entropy kernel,
channel order, fixed state and packet representation. One GPU kernel converts
the recovered BF16 words into values, performs the original float64 inverse
RoPE, stores current-group features, and optionally stores the original words
in the recovered output. FMA is disabled; signed zero is preserved.

The same float32 values feed the unchanged Torch adaptive-statistics expression.
Its division, sum reduction and update order remain intact. Calibration and
nonunit attention scaling use the inherited path. A syntax-tree comparison
locks the execution-loop change to this conversion/write region.

The completed GPU experiment compares the previous V64/B32 graph with the finish graph
using identical packet bytes. A sender check must reproduce those bytes exactly
outside timing. Fresh receiver processes only decode fixed state plus packets.
Paired calls alternate arms and different-length packets. Both fitted engines
and graph buffers are resident; memory results describe that paired setup.

Reports include retimed raw BF16, official SplitZip and official ExANS; targets
against both raw and SplitZip through 400 Gb/s; separate phase diagnostics; and
optional eager kernel profiles. The latter show kernel-count/cost changes, not
captured-graph latency. The 400 Gb/s rows expose the remaining gap honestly.

Before changing the format, A2 isolates the allocation-fill policy described in
the [write audit](lite_allocation_audit.md). Other same-format candidates are
caching immutable group spreads and replacing the coarse stage-1 lookup/search
with a full inverse CDF of the exact same frequencies. That lookup changes
memory footprint and execution, not packet probabilities; actual GPU gains
remain unmeasured. Neither execution tuning nor table lookup alone establishes
a new research contribution.

### B: an exact table-friendly conditional codec

The current value-slice decoder performs floating-point interval calculations,
two symbol stages and word searches. A candidate replacement predicts a BF16
word, maps it and the original word to monotone integer indices, and codes their
**exact integer difference**. A calibrated table handles common differences;
tails escape to the original raw word. Only the prediction is rounded. No
original data is quantized, and reconstruction must recover every input bit.

An illustrative alphabet has differences -127..127 plus one escape. A full
16-bit-slot-to-8-bit-symbol inverse table for 65 classes is about 4.06 MiB.
This illustrates a search-versus-table tradeoff, not a chosen optimal format.
Class definitions, escape rate, predictor error and table footprint must be
calibrated and measured. Reusing the existing lossy ResidualCoder would not
provide this exact method.

First measure symbol cost plus paid escapes, rANS states, directories and shared
tables on separate TRAIN/validation/TEST data. Then implement a new versioned
format and a complete causal GPU receiver. Marginal entropy or oracle-conditioned
bulk throughput does not establish complete packet size or speed.

### C: predict in parallel dependency waves

The current request has 28 layers x 16 groups = 448 causal stages. A candidate
wave schedule lets groups in one wave depend on previous-layer KV and already
decoded earlier waves, while allowing groups within a wave to execute together.
Compare previous-layer-only, 2-wave, 4-wave and current 16-step conditioning with
the same exact coder and full accounting. This measures how much extra entropy
is worth paying for a shorter critical path and larger GPU batches.

Also test reduced-rank predictors. The present dense shapes imply approximately
86 GMAC per request, or 172 GFLOP counting multiply and add separately. This is
a shape-based estimate, not measured hardware throughput. Earlier profiles
already spent about 7.2 ms in matrix kernels, well above the 400 Gb/s budget.
Changing just the final copy or entropy kernel therefore cannot settle that
target. Any new predictor needs its own fit, probability calibration and exact
packet evaluation.

Canonical fixed-point/integer prediction is another candidate here. Approximate
predictions can still support exact data reconstruction, but both endpoints must
generate identical coding probabilities. Changing GPU architecture or GEMM shape
can change floating-point rounding. Integer ranges, overflow and rounding rules
therefore need explicit tests; the current device proof uses the same A4500 and
software, not different GPU architectures.

### D: use prefill information to buy parallelism

A small hidden-state projection captured during prefill might provide prediction
context for many groups without first decoding their neighbors. That could
reduce both conditional entropy and the dependency chain, while a cheap predictor
uses the transmitted hints instead of rerunning the model. This is an untested
extension; current frozen KV fixtures contain no such hidden-state hints.

The experiment must charge hint generation, every hint byte, fitted state and
decode work. Capture hooks must preserve the common original KV supplied to all
methods. Compare hints against an equally budgeted KV-derived summary and the
no-hint control. Coset/low-bit side information is another possible exact symbol
representation; it needs full disambiguation/escape accounting and cannot be
credited with theoretical bits alone. These ingredients have prior art; the
candidate contribution is their demonstrated KV-specific combination and access
schedule, not the names of the ingredients.

### E/F: throughput and endpoint work

Batch independent packets without introducing dependencies between requests.
Preserve each request's probability arithmetic: larger GEMM shapes may change
floating-point rounding. Start with independent captures/streams or canonical
tiles, then measure queueing, per-request latency, byte throughput and GPU memory
at several concurrency levels. Amortizing capture requires reusable geometry
and position ranges.

At the sub-millisecond target, validation, staging and output ownership themselves
become material. Test safe GPU or vectorized validation, direct receive into an
input arena, and caller-owned output slots. Their costs stay in the comparison;
an overwritten static output or skipped directory check is not a speedup.
If attention will consume the separate resident pool format, also test direct
decoded-word-to-pool writes. That could avoid a full-cache intermediate copy,
but pool packing/transcoding work and exact reconstruction must be measured.
The current finish experiment returns BF16 and makes no pool-attention claim.

## Fair decision rule and next command

For each arm, preserve exact original BF16 words and report complete bytes,
shared state, cold/warm latency, memory and the relevant quality setting.
Keep lossless and lossy frontiers separate. A faster decoder with more bytes is
promoted only for links/workloads where total measured or modeled cost improves.
Large-link compute targets do not replace actual DRAM/SSD/network measurements.
No new first-of-kind or 400 Gb/s performance claim follows from this plan.

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_lite_finish_a4500.sh --profile
```

The default parent is `runs/lite_graph_a4500_20261028_092354_3260980`. Only arm A
is implemented in this command. It reuses fits, captures and native libraries;
no refitting or model downloads occur. Separate profiles follow ordinary timings.
Resume with `bash scripts/run_lite_finish_a4500.sh --resume RUN_DIR --profile`.
The small report is published automatically; full traces remain local.

The numerical targets are reproducible from
`benchmark_results/lite_high_link_study_20261028/targets.json` and the cited
published graph report. They are derived analysis, not new GPU measurements.
