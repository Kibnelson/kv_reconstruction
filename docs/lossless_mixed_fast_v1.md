# Lossless mixed execution experiment v1

This is a lossless-only follow-up to the completed A4500 combined-transfer run
at `14c5259`. Every reconstructed word must equal its original BF16 word,
including its sign, exponent and mantissa bits. Quantized predictor features
help assign coding probabilities; they do not replace or quantize the original
KV symbols in the packet.

The experiment lives on `experiment/lossless_mixed_fast_v1`. Existing source
tracks, the official baseline checkouts, fitted coefficients, TRAIN codebooks,
frozen packets and the completed combined report remain unchanged. New adapters
and tests use separate files. There is no model download, refit, or dependency
installation in the runner.

## Evidence motivating this change

The completed run's 177 packaged files, 56 exact result rows, 28 independent
receiver receipts, source receipts and policy digest have been checked. Its 8K
results average the same two original documents across all 28 Llama layers:

| Method | Complete MiB | Isolated encode ms | Isolated decode ms |
|---|---:|---:|---:|
| SplitZip, before control | 674.807 | 10.389 | 3.374 |
| Mixed k2 | 664.004 | 30.500 | 10.476 |
| Mixed k14 | 595.842 | 37.308 | 16.264 |
| Existing Rank256 | 521.154 | 42.136 | 21.631 |
| Existing Dense | 499.574 | 83.891 | 29.010 |
| Compact ExANS | 609.129 | 210.052 | 23.610 |

The k2 packet saves just 10.8 MiB versus SplitZip, yet it pays for the inherited
full-model preparation. That preparation performs 27 feature passes, 27
surprise/prior passes and 26 predictor products even when only layers 1 and 27
use rANS. Both selected layers follow SplitZip layers, so their probability
priors reset to one; neither needs a surprise pass.

The k14 profile records 14 native SplitZip encodes and 14 decodes per full
request. Each fallback decode is then copied to its final layer. Separate
diagnostic medians were 2.28 ms for checked unpack and 14.56 ms for native
decode. These samples are nonadditive, but show that unpack alone cannot explain
the total decode cost.

Metadata is a small byte target: mixed metadata/directories are about 40 KiB per
8K request, and SplitZip exponent escapes account for only about 2.77 MiB on
TABLES. Reducing required operations is the priority for this iteration.

## Four paired execution arms

| Arm | Predictor preparation | Fallback SplitZip |
|---|---|---|
| Reference | Original full preparation | Original separate call per layer |
| Selective | Required features and probability priors only | Separate calls |
| Bundle | Original full preparation | One increasing-layer bundle |
| Both | Selective preparation | One bundle |

Selective preparation uses the original feature, product, surprise and prior
kernels with the same arithmetic and coefficients. It computes feature layer
`l-1` when rANS layer `l` needs it. A prior for `l` needs surprise from `l-1`
only when both layers use rANS; otherwise the original reset-to-one rule applies.
Unused features are not consumed. This is an encoder change: it does not claim
to shorten precompressed decoding.
The native encoder still addresses a full layer-indexed feature array; selective
preparation skips writes and computation without reducing that array's reserved
extent or the shared predictor coefficients.

The bundled arm gathers fallback layers in increasing order and calls the
unchanged native SplitZip codec once. A checked directory binds their positions.
The receiver makes one SplitZip decode, scatters those exact words into a fresh
output tensor, and invokes the original mixed rANS decoder on that output.
Gathering, scatter, allocation, validation and packet framing are timed. Any
extra memory or copy cost must be visible rather than assumed away.

The bundled representation changes framing and fallback chunk addresses, but
not the rANS probabilities or words. A zero-rANS mask still delegates to the
original whole-request SplitZip adapter. A mask without fallback layers keeps
the existing rANS frame.

## Fixed workload and correctness

Use the original BF16 `unsloth/Llama-3.2-3B` snapshot at revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`, all 28 layers, eight KV heads and
128 channels per head. Reuse the same fitted state and TRAIN SplitZip LUTs.
Measure both original documents at 512 and 8,192 tokens, using k2 and k14 masks
frozen from the previous run's TABLES calibration. No new mask or parameter is
chosen using the new timings.

These documents have been used repeatedly during development. This experiment
tests correctness and implementation cost; it does not establish performance
generalization to independent documents or models. TABLES also fitted the
probability spreads and is not independent validation.

Mandatory actual-CUDA tests run before long measurements. Each measured packet
is exported, restored, and decoded in a fresh receiver given only the packet and
shared state. The receiver alternates documents 0/1/0, retains previous outputs,
and checks every original word digest, output ownership, input immutability and
runtime flags.

The selective arm must produce the exact reference packet. Both must produce
the exact bundled packet. All four arms must share the same rANS payload and
directory bytes for each document/mask/length. Those are required equalities,
not compression estimates. Bundled packet sizes are actual serialized sizes.

## Timing, controls and useful diagnostics

The unchanged raw, SplitZip and fixed ExANS controls bracket the experiment.
Existing Dense and Rank256 remain paired controls, and the completed compact
ExANS adapter is remeasured unchanged. The original workers and environments
are preserved. In particular, the SplitZip baseline keeps its original Torch
2.9/Triton 3.5 environment; mixed integration uses the original codec source in
our Torch 2.11/Triton 3.6 environment. This environment difference remains
disclosed in any comparison.

Ordinary full encode and decode calls use the unchanged author timing body:
10 warmups, 50 calls per sample and five repeats. Isolated calls use two warmups
and five samples. Cold setup, disk, host serialization, input upload and fitting
are separate. Torch output/scratch peaks are observed; native allocations
outside Torch remain unmeasured. Shared coefficients and LUTs are additional.

Separate representative profiles collect kernel counts/time, predictor pass
counts, native SplitZip calls, copies, allocation and synchronization operations.
Warm profiler registration before CPU profiling so import/registration time is
not mistaken for steady codec work. Profiled kernel totals across streams and
unpack/native component samples are not an additive wall-time decomposition.

Calculate serial wire plus full decode, and full encode plus wire plus decode,
at 1, 5, 10, 25, 50, 75, 100, 200, 300 and 400 Gb/s. Preserve every fixed arm;
do not select a TEST winner and present it as a frozen deployment policy. Link
tables are calculated scenarios, not measured NIC or SSD transfers. The current
Rank packet needs decode below roughly 16.3 ms at 100 Gb/s and 6.6 ms at
400 Gb/s to match the measured SplitZip serial cost. Encoding improvements alone
cannot satisfy those precompressed targets.

## Run on the NUC

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/lossless_mixed_fast_v1
git pull --ff-only origin experiment/lossless_mixed_fast_v1
bash scripts/run_lossless_mixed_fast_a4500.sh
```

The wrapper publishes small reports, source/equality receipts and diagnostic
summaries to this new branch, including failure logs if a stage stops. Encoded
packets and full profiler traces remain in the NUC run directory. Preparation
on the local CPU does not provide GPU performance results for these new arms.

All results here concern lossless transfer T. Resident attention P, T-to-P
conversion, lossy quality, network, SSD and serving throughput remain separate
experiments.
