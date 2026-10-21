# A4500: compact T1 confirmed, prediction dispatch comparison

Update: the 64/512/all comparison passed on A4500 in commit `e3e296d`.
Qwen B's paired all-column speedups are 1.583x encode and 1.612x decode,
with identical complete T1 bytes. See the [entropy comparison](gpu_path_entropy_a4500.md)
for measured memory costs, the choice of a 512-column control and the next
exact probability-scheduling experiment. The original plan below is historical.

The completed compact-envelope run is
`benchmark_results/gpu_path_a4500_20261021_110229_3928210`, published in
commit `9df187e`. All eight actual CUDA tests passed. Its 24 attention cases
and 20 transfer cases completed without failures. Every transfer was coded
without raw fallback, reconstructed the original BF16 words, restored P1
attention and passed fresh-process GPU decoding without original KV input.

## Measured size result

| Group | Complete T1 bpv | Entropy payload bpv | Directory + envelope bpv | BF16 reduction |
|---|---:|---:|---:|---:|
| Frozen SmolLM A | 9.441417 | 9.423600 | 0.017817 | 40.99% |
| Qwen four-layer A | 9.579681 | 9.554596 | 0.025085 | 40.13% |
| Qwen four-layer B | 8.853500 | 8.828415 | 0.025085 | 44.67% |

The predicted compact-envelope rates are now actual A4500 results. Removing
the repeated full identity saved exactly 5,564 bytes per Qwen packet and 283
per SmolLM packet. Entropy payloads stayed unchanged. Complete Qwen B packets
are 7.5804% smaller than A packets. Both arms have the same Block32 means;
B adapts widths using already-decoded context. The identity-framing saving
is not an entropy-model advance or a research novelty claim.

These rates concern exploratory 32-token pages from the first four Qwen
layers, not a full-model or full-context rate. Shared T1 GPU state remains
87.22 MiB for Qwen A and 87.86 MiB for B. P1 is a separate resident layout;
its GPU allocation savings are not added to T1 transfer savings.

## What the timing controls say

| Qwen selected-page control | Encoder parameter replay ms | Entropy encode from prepared parameters ms |
|---|---:|---:|
| A, prefix | 304.651 | 142.472 |
| A, last code page | 305.682 | 143.959 |
| B, prefix | 395.360 | 142.195 |
| B, last code page | 396.780 | 139.582 |

Parameter replay includes the mean predictor, canonical/RoPE features, width
context and error-bucket updates. Prepared entropy is an encoder control;
the actual decoder still predicts causally from packet-recovered words.
Different controls retain different tensors, so their peaks are not additive.

The recorded warm component means are 455.05/317.63 ms for Qwen A
encode/decode and 544.80/415.39 ms for B. The direct file-cycle means are
792.95 and 984.65 ms. These were collected in a process that also ran
earlier profiles. Differences from the previous run cannot be attributed to
compact framing, which is outside the GPU-body interval. Future parent
measurements never enable profiling.

The operator summary records 10,240 `aten::sum` calls in Qwen A prediction.
Each reduction follows a product of a 64-column slice and is followed by
accumulation. Thousands of separate Torch dispatches and intermediate tensors
therefore implement a much smaller number of mathematical block predictions.
The selected B profiles contain 12,096 sums because width-context and bucket
updates add work. This identifies a concrete implementation target; it does
not by itself establish the eventual speedup.

P1 attention continues to pass. Matched output differences are zero for
SmolLM and at most 2.3842e-7 for Qwen. Mean sequential layer-order wall times
are 3.81% and 3.37% above the matched dense control respectively. These small
seeded-query cases do not establish native generation or serving throughput.

## Profiler correction

The previous summarizer counted all CUDA events, including nested `T1/*`
user-annotation ranges. Those ranges are not kernels. As a result, its
`CUDA_device_event_total_ms` double-counts time and must not be interpreted
as GPU work or divided by wall time. The old event counts also include ranges.
Per-operator CPU counts and separately measured component controls remain
useful. Historical reports are retained unchanged.

The corrected summarizer excludes CUDA user annotations and named `T1/*`
ranges. It records their excluded count separately. Remaining device events
include kernels and copies; durations can overlap and are not wall latency.
A regression test includes a CUDA annotation as well as a CPU parent.

All profiling and selected diagnostic controls now run in the fresh
packet-only decoder child. That process first reconstructs words from T1,
measures normal encode/decode and encoder substeps, then profiles. It exits
afterward. The parent benchmark never enables the profiler. This prevents
profiler callbacks and collector state from carrying into later parent
measurements; it does not eliminate ordinary hardware/runtime variability.

The console warning about events being cleared per profiler cycle does not
indicate a failed test. Each profile here records one call in one cycle.

## Three exact implementation controls

The next batch changes the input reduction chunk of the existing integer
predictor. It does **not** change the 32-channel prediction group size.

| Input chunk | Mean reductions per Qwen chunk | Largest product temporary at 32 tokens, upper bound |
|---|---:|---:|
| 64 columns, current reference | 10,240 | 0.5 MiB |
| 512 columns | 1,392 | 4 MiB |
| All available columns (`0`) | 255 | Less than 32 MiB |

Counts come from the known four-layer/2,048-channel schedule. Temporary bounds
cover the `[tokens,32,input_columns]` INT64 product only. Selected features,
reductions, parameters, output, entropy scratch and shared state are additional;
actual total/per-call peaks are measured. These counts are not time forecasts.

The underlying operation is the same bounded INT64 dot product. Its existing
L1 bound prevents overflow of partial sums, so changing summation grouping
does not change the result. Clipping, fixed-point rounding, integer RoPE,
widths, probability tables, entropy ABI and state identity remain unchanged.
Local checks compare 769-column reductions with an independent NumPy integer
oracle, including keys/RoPE and values, and compare complete causal packets.
The new actual-CUDA test repeats those checks on the A4500.

The batch runs 64/512/all consecutively for each source page, sharing the
same frozen device state. Every variant must match the original CPU parameters
and the complete chunk64 packet hash, reconstruct words exactly, survive
file/restore/P1 attention and pass its independent packet-only child decode.
If the chunk64 reference fails, the alternatives cannot be reported as valid
comparisons. CPU T0 is measured once per source/arm at chunk64 because all
variants must have identical T1 bytes. A/B comparisons match the input chunk.

No variant has a measured A4500 speedup yet. All-column reduction deliberately
trades more transient memory for fewer dispatches; this is a comparison, not
a new default selected without measurements. Fusing integer prediction is a
subsequent option if Torch chunking still leaves excessive overhead. The
unchanged serial entropy kernel remains another optimization target.

## Run all available cases

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/gpu_kv_path
git pull --ff-only origin experiment/gpu_kv_path
bash scripts/run_gpu_path_a4500.sh --publish-results \
  --predictor-input-chunks 64,512,0 --diagnostics
)
```

With the original Qwen files present this tests 24 attention cases and 60
transfer implementation cases over the same 20 source/arm cases. Nine actual
CUDA unit tests are required, including the new reduction/packet-equivalence
test. Fresh-process decoding is required by diagnostics. The helper rebuilds
and publishes the standalone runner and allowlisted reports. Encoded KV and
captures remain local. New input-chunk GPU results are pending.

Hint/prefill-H/coset research, native generation, concurrency, real network
and cold SSD remain separate experiments. This step measures implementation
cost while holding the coded model and complete packet bytes fixed.
