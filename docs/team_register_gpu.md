# Reusing partner values in the exact GPU decoder

This experiment targets the entropy-and-feature decoder kernel after the
borrowed-input experiment reduced 8K Rank256 public decode latency to about
21.7 ms. It keeps the same compressed packets, predictor state, CDFs and
binary-search inverse. The candidate's performance remains unmeasured until
the A4500 run.

The original kernel separately loads and scales the RoPE partner accumulator,
although the complete scaled accumulator vector already contains that value.
It also stores decoded words, waits at a barrier and reloads partner words for
the next layer's features. The candidate obtains both partners by swapping the
two 64-channel halves of values already in registers. The BF16 output store is
retained. Predictor arithmetic, entropy arithmetic and probability tables stay
fixed; parity is required for decoded words and intermediate features.

| Arm | Partner access | GPU execution |
|---|---|---|
| `graph_reference` | Frozen native kernel | Compact staging, CUDA graph, owned output clone |
| `graph_register` | Static half swap | Compact staging, CUDA graph, owned output clone |
| `direct_reference` | Frozen native kernel | Borrowed input, fresh owned output, eager execution |
| `direct_register` | Static half swap | Borrowed input, fresh owned output, eager execution |

The frozen native source is not edited. A new kernel preserves its signature
and replaces only the partner-access operations. Instance-local dispatch
selects the reference or candidate kernel. A run uses one predeclared warp
count for both kernels: `--decoder-warps 1` (default) or `--decoder-warps 2`.
Changing this option starts a separate controlled run, with its own before/after
baselines and report. Native baseline configuration stays unchanged. Source
receipts identify the selected implementation and warp count for every worker,
receiver and profile; no automatic TEST-based warp selection is deployed.

## Matched workload and controls

All methods reconstruct the same original Llama-3.2-3B BF16 cache: all 28
layers, text and code TEST documents, with 512-token and 8,192-token requests
reported separately. The six-document TRAIN fit and existing packets are
reused. There is no new fitting, model download, capture or encoding.

The completed parent is
`runs/team_lookup_a4500_20261028_141701_1145261`. Its verified receipts bind
all 84 earlier source files, the original packets and shared state, and the
previous CUDA checks and profiler summaries. The original packet/state run
remains `runs/team_packet_a4500_20261028_122017_2834427`.

Raw BF16, SplitZip and ExANS run before and after the four-arm comparisons.
Official codec bodies and intended runtime settings are unchanged. SplitZip
retains its pinned Torch 2.9/Triton 3.5 environment; ours and ExANS retain the
existing Torch 2.11/Triton 3.6 environment. This is a matched GPU-resident
framed API comparison, rather than each paper's complete serving evaluation.

Two warmups and eight measured calls per document and arm use a predeclared
balanced rotation/reversal order and alternate documents. Primary latency is
the synchronized public decode call: validation, small metadata transfers,
normal staging, kernel launches and an independently owned BF16 result.
Bulk host hashing, serialization, uploads, storage reads and cold graph setup
happen outside warm timing. Direct mode borrows aligned GPU input and
allocates the returned output directly; it adds no hidden payload copy or
output clone. Sequential caller-stream changes order shared workspace reuse.

Actual CUDA correctness gates run first. A fresh receiver receives fixed
codec state and unchanged packet paths, with original-word digests used only
as assertions. It checks alternating packets, held-output ownership, unchanged
wire bytes, CDF proof receipts and implementation provenance. Graph arms
capture once; direct arms do not capture.

## Profiles and link models

Separate profiling processes run after ordinary timing. They capture CPU
operators, cProfile summaries and actual CUDA kernel events. Digest checks
stay outside both profilers. Summed kernel durations can overlap across
streams and are not wall-time percentages. The full Chrome traces remain on
the GPU host; small summaries and hashes are published.

Calculated serial `wire time + full GPU decode` is compared against both
SplitZip and ExANS at **1, 100, 200, 300 and 400 Gb/s**. The faster before/after
control is an explicit conservative target. Ideal `max(wire, decode)` overlap
bounds are recorded separately. These are link models, not measured networks.
No speed win is claimed before actual measurements.

Complete packet bytes include framing, directories and checksums. Shared
predictors, staging, decoder workspace and output allocations are additional.
Global allocation peaks from a process holding all four codecs are not a
standalone serving-capacity result. Native ExANS allocation peaks remain
unmeasured. This experiment does not measure AQUA-KV, compressed attention,
T-to-P conversion, SSD I/O or serving throughput.

## Run

On the A4500 host, on `experiment/llama_causal_packet_compare`:

```bash
bash scripts/run_team_register_a4500.sh
```

To measure both supported warp counts sequentially, with two independent
reports:

```bash
(
set -euo pipefail
for decoder_warps in 1 2; do
  bash scripts/run_team_register_a4500.sh --decoder-warps "$decoder_warps"
done
)
```

The wrapper archives CUDA correctness output, runs the comparison, packages
small evidence and publishes report files only. To resume, use
`bash scripts/run_team_register_a4500.sh --resume RUN_DIR --decoder-warps N`
with the original warp count and unchanged source/workload options. Packets,
fitted tensors and full profiler traces remain on the GPU host.
