# A4500 GPU path results and follow-up

The completed report is
`benchmark_results/gpu_path_a4500_20261021_103014_3238265`, published in commit
`1e8d5bd`. The recorded device is an NVIDIA RTX A4500, SM86, Torch
2.11.0+cu128 and Triton 3.6.0. There are no component failures.

## What passed

All eight actual CUDA unit tests passed: byte-oracle agreement, all 65,536
BF16 patterns through the entropy core, special-word parallel lanes, P1
bounds/capacity handling, compressed attention and external restore. The
model-page batch completed 24 attention cases and 20 transfer cases.
All 20 transfer cases used compressed streams rather than raw fallback,
recovered every original word, restored attention-readable P1 and passed
fresh-process GPU decoding without the original KV.

This is actual GPU probability construction and entropy decoding. CPU code
dispatches work, reads the final encoded allocation length, and frames,
checksums and transports compressed bytes. It performs no T1 entropy decode.

The measured cycle starts at P1, produces T1, moves compressed bytes outside
GPU, writes/fsyncs/reads a file, restores P1 and checks attention. The bridge
temporarily materializes one bounded 32-token chunk. Native model generation,
concurrent serving, real network and cold-SSD performance remain unmeasured.
Filesystem reads may hit the OS page cache.

## Rates and state

Rates below weight complete packet bytes by original word count. Shared state
is separate.

| Dataset/arm | Complete T1 bpv | BF16 reduction | Same-page CPU T0 bpv | T1 entropy payload bpv |
|---|---:|---:|---:|---:|
| Frozen SmolLM A | 9.447559 | 40.95% | 9.424197 | 9.423600 |
| Qwen four-layer A | 9.749481 | 39.07% | 9.555740 | 9.554596 |
| Qwen four-layer B | 9.023300 | 43.60% | 8.828674 | 8.828415 |

The earlier ~44.8% was CPU T0. This run independently demonstrates 43.6%
with complete GPU-produced T1 packets on the same exploratory Qwen pages.
Their difference is overwhelmingly framing: every Qwen packet has a 512-byte
lane directory and 5,872–5,875 envelope bytes, including a repeated full model
identity. The entropy payload is already close to the earlier CPU rate.

P1 is a different representation. The full-prefix arena plus directory is
13.05 bpv for SmolLM (18.44% saving) and 12.925293 bpv for Qwen (19.22%).
Allocating shared pool calibration to one such small chunk adds 0.5 bpv,
making those figures 13.55 and 13.425293 bpv. Calibration can be shared across
chunks. Encoder scan workspace and temporary/output allocations are additional.

Do not add T1 and P1 savings. T1 reduces external storage/movement; P1 permits
independent GPU head-tile reads. T1 predictor state is also separate: 26.60
MiB for SmolLM A, 87.22 MiB for Qwen A and 87.86 MiB for Qwen B on GPU.
Charge it when resident; serving capacity has not been established.

## The A/B insight

Both A and B use the same Block32 mean predictor with previous-layer and
earlier current-layer conditioning. B adapts logistic widths using decoded
context. This is **not** a previous-layer-only versus channel-group ablation,
and it is not an AQUA-KV comparison.

B saves 7.4484% of complete Qwen packet bytes relative to A.

| Sampled region | B complete bpv | B gain over A, bpv | Groups at minimum width multiplier |
|---|---:|---:|---:|
| First 32 tokens | 7.885757 | 1.436813 | 70.94% |
| Later sampled pages | 9.402481 | 0.489304 | 16.24% |

GPU means, inverse widths and offsets exactly match the original CPU reference
on all pages. Prefix gains and width-floor saturation therefore survive the
move to GPU. Narrower widths or additional free causal context merit a
TRAIN/VALID experiment and new holdout. Saturation alone does not establish
that narrower widths will help. Prefix pages are 25% of this sample but only
1.5625% of a complete 2K context; the sample mean is not a full-context rate.

Earlier hint arms selected zero hint strength and their extra bytes did not
help. They are not implemented by this GPU A/B baseline. Prefill-H/coset arms
must pay all metadata and production costs under the same GPU coder.

## Latency is the main current limitation

These are means of per-page warm synchronized wall medians. Chunks have 32
tokens; Qwen uses four layers and 524,288 original bytes.

| Group | KV to T1 ms | T1 to KV ms | P1 to T1 ms | T1 to P1 ms | Direct file-cycle ms |
|---|---:|---:|---:|---:|---:|
| SmolLM A | 355.61 | 242.11 | 355.89 | 240.89 | 645.07 |
| Qwen A | 389.88 | 256.13 | 389.36 | 256.92 | 693.55 |
| Qwen B | 461.40 | 328.96 | 462.73 | 329.34 | 811.60 |

B's 7.45% packet gain costs 18.35% more encode latency and 28.43% more decode
latency than A. Qwen B export/import each average about 0.4 ms; file/directory
fsync averages 18.83 ms and cached reads 0.063 ms. These do not account for
the hundreds of milliseconds inside the GPU-body codec.

Raw controls unpack/repack P1 in about 0.1–0.15 ms each. The report's serial,
no-overlap link diagnostic breaks even around 0.25–0.34 MB/s for Qwen B. It
excludes CPU staging differences and is not a network measurement. The codec
has not demonstrated faster KV movement.

CUDA event intervals can include host dispatch gaps; their similarity to
wall time does not isolate kernel execution. The code launches many Torch
integer operations per group and scalar per-token rANS programs. Profiling
must separate dispatch, prediction and entropy serialization before a rewrite.

## P1 attention

Sequential layer-order P1 wall time averages 3.53% above dense for SmolLM
and 2.66% above dense for Qwen. These small seeded-query cases use a matched
FP32 dense control, not optimized serving attention or end-to-end generation.

SmolLM matched outputs are identical; Qwen's maximum absolute difference is
2.3842e-7. Both agree with the independent float64 oracle within tolerance.
KV words remain bit-exact; attention outputs are not always bit-identical.
Raw-tail and selected-layer paths passed. All-layer batched timings remain a
separate diagnostic and cannot replace model layer-order timings.

Imported historical P0/P1 results show about 130x faster packing and 225x
faster full unpack for P1 on this fixture. These compare earlier untuned P0
code to P1; they are not transfer or serving speedups.

## Implemented next batch

New packets carry SHA256 of the complete canonical model identity. The
receiver must independently supply that same expected identity and saved
state. The full identity remains in the source/report manifest. Each packet
retains checksum, state digest, shape, position and lane directory. Updated
decoders also read legacy full-identity envelopes; old decoders need an update
for compact envelopes. Predictor math, entropy bytes and P1 layout are unchanged.

Re-serialization exactly reproduces all 20 old envelope lengths. Keeping the
measured payload/directory counts projects these rates:

| Group | Old measured complete bpv | Compact envelope projection bpv | Projected BF16 reduction |
|---|---:|---:|---:|
| SmolLM A | 9.447559 | 9.441417 | 40.99% |
| Qwen A | 9.749481 | 9.579681 | 40.13% |
| Qwen B | 9.023300 | 8.853500 | 44.67% |

This removes 5,564 repeated bytes per Qwen chunk and 283 per SmolLM chunk.
These are **CPU accounting projections**, not new GPU results or an entropy
model advance. The next run measures actual packets, compares both envelope
forms with identical streams, and repeats isolated decode/restored attention.

With `--diagnostics`, first/last selected pages for every dataset/arm also
measure encoder parameter replay and entropy encoding from those prepared
parameters. Those are encoder controls, not decoding. Separate actual
encode/decode profiler summaries count CUDA events and Torch calls and group
prediction/context work. Instrumented times are not normal latency samples;
missing CUDA activity is explicit. No tensor values, stacks or trace files
are published.

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/gpu_kv_path
git pull --ff-only origin experiment/gpu_kv_path
bash scripts/run_gpu_path_a4500.sh --publish-results --diagnostics
)
```

The command rebuilds the standalone runner on the A4500 host, tests all
available frozen/Qwen pages, compares compact/legacy accounting, profiles
selected calls and publishes only allowlisted reports. Execution of the
follow-up changes on A4500 remains pending.
