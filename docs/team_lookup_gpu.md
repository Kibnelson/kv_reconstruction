# GPU decoder search and copy costs

This experiment aims to reduce full GPU-resident decode latency while keeping
the existing compressed packets and fitted predictors fixed. It compares four
decoder paths for each dense-uniform and rank-256-uniform configuration:

| Arm | Symbol lookup | GPU input and output |
|---|---|---|
| `compact_search` | Existing binary CDF search | Stable compact staging, graph replay, owned output clone |
| `compact_lookup` | Existing central/tail-table lookup | Stable compact staging, graph replay, owned output clone |
| `direct_search` | Existing binary CDF search | Borrow packet payload, allocate an owned output, eager execution |
| `direct_lookup` | Existing central/tail-table lookup | Borrow packet payload, allocate an owned output, eager execution |

The lookup option changes how the decoder inverts the same discrete CDF. It
does not change the probabilities, predictor arithmetic, BF16 words, packet
format, or original packet checksum. The packet's existing search option
records its original encoder configuration; the receiving implementation can
use an equivalent inverse. An exhaustive check compares every 16-bit slot in
each width class against the binary-search reference before GPU timing.

Direct decoding borrows the checked packet's GPU payload for the duration of
its work. It returns a fresh, independently owned BF16 tensor. Directory
validation, small directory uploads and pointer mapping, output allocation,
native kernel launches and synchronization required by the API are charged.
It avoids copying the full compressed payload into another GPU buffer and
avoids cloning a persistent output. Internal stream dependencies preserve the
input and output lifetimes. The public API remains sequential.
Direct inputs require 16-byte-aligned storage, as provided by the existing
packet importer. Sequential calls can switch CUDA caller streams; shared
workspace reuse is ordered before the next call writes to it.

## Workload and controls

The same original Llama-3.2-3B BF16 words are reconstructed for all 28 layers,
two TEST documents, and separate 512-token and 8,192-token cases. Existing
packets and the six-document TRAIN fit are reused. The experiment performs no
model download, fitting, raw KV capture or encoding.

SplitZip, ExANS and raw BF16 controls run before and after the paired tests.
Their method bodies, fixed state and intended runtime settings stay unchanged.
SplitZip retains its pinned Torch 2.9/Triton 3.5 environment; ours and ExANS use
the existing Torch 2.11/Triton 3.6 environment. This is a matched GPU-resident
framed API comparison in a local harness, rather than a reproduction of each
paper's serving evaluation.

The ordinary timing starts with compressed payloads already on GPU and ends
with an independently owned reconstructed output. Large host packet hashing,
serialization, host-to-device payload uploads, file reads and cold graph setup
are outside warm decode timing. Small CPU metadata checks remain part of the
public decode call. Default measurements use two warmups and eight measured
samples per document and arm, with balanced arm order and alternating
documents. Before/after native results remain separate to expose drift.

A fresh receiver receives only fixed codec state and packet paths, with
expected original-word digests used as assertions. It checks alternating
packets, held-output ownership, unchanged packet hashes and graph counts.
Graph arms capture once; direct arms do not capture. Actual CUDA correctness
tests are required before the benchmark runs. No CPU estimate is presented as
a GPU result.

## Profiles and speed targets

Profiles run in separate processes after ordinary measurements. They record
actual CUDA kernel events, CPU operators and function summaries; digest
assertions remain outside both profilers. Summed device durations can overlap
across streams and are not wall-time percentages. Full Chrome traces remain
on the GPU host; the published report includes their hashes and small
summaries. Diagnostic phase times are separate measurements, not additive
decompositions of ordinary latency.

The report compares calculated serial `wire time + GPU decode` at **1, 100,
200, 300 and 400 Gb/s** against both SplitZip and ExANS. It also records
idealized `max(wire time, decode)` overlap bounds separately. These are link
models, not measured network transfers. A candidate is useful only if its
measured latency improves under the relevant workload and link; exact
reconstruction is required for every candidate. No throughput win is assumed
before the A4500 results.

Complete packet bytes include framing, directories and checksums. Shared
predictors, CUDA staging, decoder scratch and output allocations are additional
and reported separately. Native ExANS allocation peaks remain unmeasured.
AQUA-KV is lossy and its official evaluation remains separate; no adapted AQUA
result is substituted in this exact-codec comparison. The experiment does not
measure SSD I/O, compressed attention, T-to-P conversion or serving throughput.

## Run

On the existing A4500 host, on `experiment/llama_causal_packet_compare`:

```bash
bash scripts/run_team_lookup_a4500.sh
```

The wrapper archives its actual-CUDA correctness gate, runs the comparison,
packages a small report and publishes report files only. Resume an interrupted
run with `bash scripts/run_team_lookup_a4500.sh --resume RUN_DIR`; its source,
parent receipts and workload options must remain unchanged. Original packets,
predictor tensors and full profiler traces stay on the GPU host.

The default parent is `runs/team_gpu_a4500_20261028_131628_4041134`. Its verified
receipts bind the original compact packets and fitted state from
`runs/team_packet_a4500_20261028_122017_2834427`.
