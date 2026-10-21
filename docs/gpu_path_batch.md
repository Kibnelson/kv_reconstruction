# GPU KV path: attention, T1 and external restore batch

Both full A4500 batches passed. Compact Qwen B T1 packets now measure
8.853500 bpv, a 44.67% BF16 reduction. See the
[dispatch comparison](gpu_path_dispatch_a4500.md) for these results, the
profiler annotation correction and the next 64/512/all-column timing batch.
The [earlier review](gpu_path_a4500_20261021_review.md) retains the original
framing and state-accounting comparison.

The user has approved batching related implementation and testing without a
separate approval after each step. This batch connects the component paths
below. It does not implement a native model cache backend or serving engine.

## Existing A4500 result

The user reported the completed `pool_p1_a4500_20261021_090226` run on the RTX
A4500. Four frozen model pages were exact at both tested arena budgets. With
the 80% arena, warm synchronized wall medians were 0.11854–0.12078 ms for
packing and 0.06498–0.06783 ms for full-sector unpack. The reserved packet
was 13.05 bits/value, excluding calibration and workspace. These are console
summary values; `--previous-p1` imports the full recorded samples, selected
tile reads and paired P0 controls, retaining their historical source hashes.

That is the **P1 pool** result. The earlier ~44% reduction is a **T0 transfer**
result from the Qwen four-layer workload, whose entropy coder was on CPU.
This batch implements a new GPU T1 stream and measures its rate afresh.

## Implemented paths

| Path | GPU work | CPU work/control |
|---|---|---|
| P1 attention | Load exact head tiles inside attention; FP32 online softmax; GQA; raw tail | Dispatch and status validation |
| KV to T1 | Frozen integer mean and A/B width context, BF16 probabilities, binary rANS, lane compaction | Dispatch loops and a final byte-count read for output allocation |
| T1 to KV | Causal parameters, probabilities and actual entropy decoding | Dispatch; diagnostic status/hash checks |
| P1 to T1 | Unpack one bounded chunk, then GPU T1 encode | Dispatch and compact allocation length |
| T1 to P1 | GPU decode, pool allocation and GPU pack | Dispatch and host metadata |
| DRAM/filesystem round trip | Compressed-byte D2H/H2D | Envelope/index/checksum, write/read, file and directory fsync |

CPU parameter oracles and the optional CPU T0 comparison are validation
controls outside the GPU codec intervals. No CPU entropy decoder is used by
the T1 path. Raw CPU fixtures are validation inputs, not a production external
cache. Stored/restored objects contain T bytes throughout.

## Attention component

`pool_attention.py` implements one query per layer/query-head. Dense and P1
controls run the same FP32 attention arithmetic. P1 reconstructs words inside
the attention kernel, without a full-cache BF16 output buffer. The raw-tail
control uses a sealed prefix and three additional valid tokens. GQA heads map
to their shared KV head without duplicating stored K/V.

The API selects the layer range explicitly. Primary timings enqueue one layer
at a time, matching decoder layer order. An all-layer independent-query batch
is recorded separately as a throughput diagnostic, not model decode latency.

The runner validates against an independent float64 dense oracle and the
matched dense GPU kernel. It releases the original dense GPU input before
timing packed attention. Queries are seeded component inputs, not captured
model queries. Output/logit/continuation equivalence for a native model is
still an integration requirement. These tests cannot establish serving VRAM
capacity or throughput.

The sector reader is shared between unpacking and fused attention. Its
descriptor bounds reject signed-overflow offsets as well as unpublished
sectors. Invalid attention input returns an explicit nonzero status.

## T1 feasibility ABI

T1 retains the frozen metadata-free Block32 mean predictor and A/B widths.
Keys use saved integer RoPE tables for canonical prediction features and
rotation of the prediction back to the original word coordinates.

Each original BF16 word is bijectively mapped to its ordered 16-bit rank.
Each rank bit is coded conditional on its higher bits. Probabilities integrate
the saved logistic CDF over the BF16 rounding interval and include the positive
uniform word prior. Interpolation/masses use integers; boundary standardization
uses float64 with fusion disabled. Conditional counts are rounded to 12-bit
frequencies in [1,4095]. This finite-precision factorization differs from the
T0 group/mantissa constriction stream and has a new format/state identity.

One rANS lane per token advances through all local layers and channel groups.
The encoder replays parameters from known source KV, then encodes groups in
reverse order. The decoder predicts forward using only words already recovered
from the packet. Encoder parameter replay is not presented as decompression.
The fresh-process control receives saved state, expected model identity and
the packet only, and returns a decoded-word hash.

Source feature clipping is a bounded, symmetric safeguard for arbitrary words,
including infinities and NaN payloads. Original words are still coded exactly.
On model pages the runner requires exact mean/scale/offset parity with the
original CPU reference, so clipping cannot silently change the claimed frozen
A/B comparison. State validation rejects noncausal columns and unsafe tables.

Lane restart states, the lane directory, envelope, model/state identity digests,
checksum, padding and raw fallback all count in complete external packet bytes.
The full expected model identity is canonicalized and hashed at both ends;
the packet carries its SHA256. Updated decoders also accept the earlier full
identity envelope. Full identities remain available in source/report manifests.
The uniform prior and positive conditional counts support every BF16 pattern.
An actual CUDA test exercises all 65,536 original words through entropy coding.

This first implementation is deliberately untuned: Torch integer predictor
operations and scalar per-token rANS programs may incur substantial dispatch,
coefficient-traffic and serialization costs. Encoder parameter replay and a
bounded worst-case byte arena are workspace, not packet payload. The runner
measures peaks and shared state separately. No compression or speed figure is
inherited from T0.

## External T/P bridge and controls

The bridge handles at most 32 tokens across the declared local layers. It
temporarily materializes that bounded chunk of original words before packing
P1. This is a composed bridge, not the planned fused converter using only two
layers of word scratch. Invalid T decode status prevents P1 publication.

The measured serial cycle starts at P1, encodes T1, exports compressed bytes,
writes and fsyncs the file and directory, reads the identical T object, imports
it and installs P1. The source pool is released after successful persistence,
before restore. Restored attention is checked against the dense oracle.
The cycle is timed directly rather than estimated by adding component medians.

P-to-raw and raw-to-P controls use the same allocation boundaries. Their
timings support a serial link break-even diagnostic using bytes saved and
extra GPU codec time. This excludes CPU staging/envelope differences and is
not a measured network result. Filesystem reads may use the OS page cache;
neither cold SSD nor transport throughput is claimed.

## One A4500 command

After fetching the published source branch:

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/gpu_kv_path
git pull --ff-only origin experiment/gpu_kv_path
bash scripts/run_gpu_path_a4500.sh --publish-results
)
```

The script rebuilds/checks the generated standalone artifact, runs the real
CUDA component tests without allowing skips, then tests all frozen pages.
If the original `runs/block32_probability_qwen4_20261021_065346` captures and
calibration remain present, it also tests all original Qwen pages with A and B
under the same new GPU coder. It does not load model weights or refit the
transfer predictor. Pool calibration is fitted on the original TRAIN split.

It imports the earlier P1/P0 details when that report remains present, and
adds the paired CPU T0 control when `constriction` is installed. Source/state
loading and initialization are recorded separately from warm medians. GPU
timings include synchronized wall time, CUDA stream intervals and per-call
Torch allocation/reservation samples. Intermediate component failures are
recorded; a failed batch exits nonzero.

Pool setup first requests the declared 80% arena. If that budget rejects a
sector, it discards that unpublished attempt and retries at the fixed full raw
capacity. The report charges the actual reservation and records the rejection
and cold retry time. A full raw-capacity reservation may provide no resident
memory saving; the benchmark never silently retains the 18.4% claim for it.
Warm bridge measurements use the actual provisioned capacity.

Reports include `summary.md`, `highlights.json`, `results.json`,
`source_manifest.json` and the CUDA test log. `highlights.json` contains tile
timings/P0 controls, memory peaks, A/B gains, shared state and link diagnostics.
The report packager copies only these named files. Raw captures, encoded KV
objects, identity validation inputs and decoder working files stay local.
`--publish-results` commits the generated runner and small report package to
`experiment/gpu_kv_path`, preserving unrelated staged paths. It leaves `main`
unchanged. Failed reports retain their failed status when packaged.

For a smaller diagnostic run, omit `--publish-results` and pass
`--attention-limit 1 --transfer-limit 1 --repeats 1`. A value of zero means all
source pages. The normal batch uses all available pages, three transfer samples
and twenty attention samples after warmup.

Add `--diagnostics` to measure source-derived encoder parameter replay and
entropy encoding from prepared parameters separately, and collect profiler
summaries for actual encode and causal decode. By default this samples the
first and last selected pages per dataset/arm. `--diagnostic-limit 0` profiles
all selected pages. Profiler times are separate from normal medians; missing
CUDA profiler activity is explicitly recorded. Reports contain operation names
and times, not tensor contents or raw trace files. Diagnostics execute only
in a fresh packet-only decoder child; parent timings never enable profiling.
CUDA annotation ranges are excluded from leaf activity totals. Old reports
whose totals include `T1/*` ranges must not be used as kernel-time estimates.

Add `--predictor-input-chunks 64,512,0` to compare the original 64-column
integer reduction with 512-column and all-column (`0`) reductions. The
prediction group remains 32 channels. Candidates share the frozen state and
must produce identical complete T1 packet bytes. Actual transient-memory
peaks are charged. See the dispatch guide for the full command.

## Local checks and remaining integration

The development workspace has CPU-compatible Torch and Triton but no CUDA.
CPU interpreter tests execute the actual kernel source and compare independent
word/byte/probability oracles, including a complete T/P/file/attention cycle and
safe rejection. Offline SM86 compilation is a separate check. Neither is an
A4500 execution or timing result.

```bash
TRITON_INTERPRET=1 python -m unittest \
  tests.test_pool_attention tests.test_transfer_t1 \
  tests.test_gpu_kv_bridge tests.test_gpu_path_runner -v
python scripts/check_gpu_path_kernels.py --output runs/gpu_path_sm86.json
python scripts/bench_gpu_path.py --validate-inputs-only \
  --output runs/gpu_path_inputs
```

Native cache-writer/prefill-to-decode integration, a reusable serving allocator,
concurrent ownership/cancellation, native logits/continuation, 2K/8K serving
capacity, actual network transport and cold SSD tests remain separate work.
The actual-H hint and selective coset research arms also remain experiments;
this A/B GPU baseline does not establish a new novelty or further-reduction
claim. The measurements will determine which bottleneck is worth changing.

Implementation context: [Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
and [byte-rANS equations](https://github.com/rygorous/ryg_rans/blob/master/rans_byte.h).
