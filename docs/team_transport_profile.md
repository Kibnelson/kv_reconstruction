# Profile and reduce compact-packet export and import costs

The completed team packet comparison measures useful GPU decoding gains, but
its current Python byte path takes approximately 1.08–1.12 seconds to export
an 8K packet and 0.74–0.77 seconds to import and upload it. Those operations
were excluded from its modeled link results. This iteration profiles and
optimizes that path while retaining the exact serialized packet and decoder.

Reference evidence:
[matched A4500 report](../benchmark_results/team_packet_a4500_20261028_122017_2834427_report_20261028_122819/summary.md).

## Fixed inputs and comparison

Reuse the parent run's serialized packets and fitted state, with no fitting,
KV generation, model forward pass or entropy re-encoding. The predeclared
cases are dense uniform and rank-256 uniform, each at 512 and 8,192 tokens,
for both TEST documents and all 28 layers. The native kernels, prediction,
probabilities, compact geometry and source hashes remain fixed.

Compare the original `TeamPacket.export` and `TeamPacket.import_bytes` with
new functions in `kv_compare/team_transport.py`. An optimized export must
produce exactly the parent's bytes, including its canonical JSON header,
directory, alignment padding and SHA-256 checksum. An optimized import must
recover the same fields and independently owned payload. A fresh receiver
must reconstruct the original BF16 words from serialized bytes plus the same
fixed state, without source KV or encoder predictions.

The existing Lite, SplitZip and ExANS measurements remain historical controls
for their recorded GPU decode scope. They are not retimed in this iteration
and must not be ranked against a newly timed complete host byte path as if
their import/export costs had also been measured here.

## Profile questions

The reference export performs a device-to-host copy, a dtype copy, NumPy byte
materialization, frame concatenation, checksum calculation and a further
concatenation to append the digest. The reference import copies a large slice
to hash, copies the payload slice, copies the NumPy view and uploads it.

Measure those costs separately, including allocation, directory/header work,
validation, byte assembly and transfer synchronization. CPU function profiles
and CUDA kernel/copy intervals are diagnostic records. Ordinary paired full
operation timings run separately, without profiler overhead. Do not add
separately synchronized phase medians and call the sum an ordinary latency.

The optimization uses buffer views to avoid large intermediate byte slices,
hashes the same content incrementally, and assembles the final immutable
frame once. It retains the same SHA-256 protection and rejection of malformed
headers, lengths and directories. CPU payloads must remain owned and writable;
GPU uploads must complete while any borrowed host view is alive. A mutable
input must not become a hidden alias of a returned packet.

## Measurement scope

Report at least:

- Export from an already encoded GPU packet to immutable host bytes.
- Import from an existing host frame to an owned GPU packet.
- Import plus the unchanged graph decoder and an owned BF16 output.
- Export, import and decode as one measured serial round trip, beginning with
  an already encoded GPU packet.
- The unchanged GPU-resident decoder separately from these host operations.
- Cold initialization, graph setup, CPU/GPU memory and full phase profiles.

Alternate documents and rotate/reverse the reference/optimized timing order.
Verify wire hashes and recovered original-word hashes outside every timed
invocation. Retain per-sample timing distributions. A fresh receiver process
checks both import implementations, changing packet data and output ownership.

For effective links of 1, 100, 200, 300 and 400 Gb/s, distinguish a host-resident
precompressed fetch from exporting a GPU-resident compressed packet first.
Charge the measured import/upload and decode in either case, and charge
export where the modeled starting point requires it. These are serial
planning models; they do not measure a network, a storage device or an
arrival/decode pipeline. Any old encoder timing used for context remains
explicitly reused evidence.

## Running and deciding

```bash
bash scripts/run_team_transport_a4500.sh
```

The default parent is
`runs/team_packet_a4500_20261028_122017_2834427`. The wrapper checks actual CUDA,
runs transport correctness gates, performs the paired comparison and publishes
small receipts. Existing parent packets and state stay read-only. Full profile
traces stay on the GPU host; the report includes compact summaries. Resume
binds the source, parent packets, state, software, hardware and settings.

Use measured full-path improvements and the phase breakdown to choose the
next task. Direct decoding from compact segments, a different checksum,
changes to prediction/group count and a serving allocator are separate
experiments. This step should identify how much of the workstation transfer
cost can be removed without changing compression or reconstruction.

## Local evidence

The final suite ran 29 tests: 28 passed and the actual-CUDA gate was skipped.
It covers exact wire compatibility for all BF16 bit patterns, checksum and
directory rejection, unaligned payloads, mutable-input snapshots, returned
buffer ownership, caller lifetime, profiler scope and a real native decoder
round trip through the CPU Triton interpreter. All 66 inherited source hashes
remain unchanged. Shell syntax and Python compilation also pass.

A separate synthetic 32 MiB CPU bit-pattern fixture measured six alternating
samples per operation:

| CPU operation | Reference median | Optimized median |
|---|---:|---:|
| Export | 77.313 ms | 42.291 ms |
| Import | 59.083 ms | 34.908 ms |

The fixture's bytes remain identical, including the checksum. These numbers
are local CPU diagnostics, not A4500 transfer or GPU performance results.
The local receipt includes raw samples, phase records and CPU function
profiles. The actual GPU comparison remains required.
