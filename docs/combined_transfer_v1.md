# Combined lossless transfer experiment v1

This experiment combines the useful transfer components from the two tracks on
`experiment/combined_transfer_v1`. Neither `llama32-3b-lossy-lossless-compare`
nor `experiment/llama_causal_packet_compare` is changed. The base is our reviewed
commit `033c9359b095b096b33cb153b20a219ab41daff3`; the team's added mixed-layer
implementation is pinned to `8a40dd9` in a separate extension namespace. Existing
predictor kernels, fitted state, original words, and author source trees remain
unchanged. No model download, training, or dependency installation is performed.

The first experiment is **lossless T only**. Its question is whether combining
layer-wise coding and a complete compact native carrier improves transported
bytes and full GPU latency on the same A4500 workload. This is not a resident P
attention experiment or an official serving reproduction.

## Arms

| Arm | Implementation and purpose |
|---|---|
| Raw BF16 | Unchanged checked transfer control, before and after new arms |
| SplitZip | Unchanged official native codec inside our existing checked carrier and its pinned environment; before and after |
| Fixed ExANS | Existing unchanged native C ABI and fixed-record carrier; before and after |
| Dense and Rank256 | Existing frozen transfer implementation, fitted state, and packets; paired controls |
| Mixed Rank256 / SplitZip | The team's unchanged mixed GPU method body, with our complete checked packet, original predictor state, and original TRAIN SplitZip LUTs |
| Compact ExANS | Same native codec and fixed-slot configuration as the control; transmit actual record prefixes, directories, complete raw overflow blocks and physical-tail coverage |

The mixed arm tests 0, 2, 8, 14, and 28 rANS layers. Zero delegates to the existing
whole-request SplitZip adapter and must reproduce its exact serialized packet.
Partial masks use native SplitZip per selected fallback layer and direct compact
rANS decoding. The extra per-layer calls, validation, framing and allocation are
included in full-call latency. This is a new local transfer adapter, rather than
an unchanged author benchmark entrypoint.

The compact ExANS arm initially calls the fixed adapter, then compacts its
actually used prefixes. The receiver validates native headers and directories,
scatters prefixes into the native record slots, and calls the unchanged decoder.
Those intermediate copies, host checks and synchronization are charged. Short
physical tails are preserved, padded and charged. Overflow has explicit raw
fallback. Native fixed-slot reservations and padded output remain additional GPU
memory costs even though fewer bytes travel.

## Workload and selection

Use the frozen original BF16 captures from `unsloth/Llama-3.2-3B`, revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`: all 28 layers, eight KV heads,
128 channels per head. The fitted state and TRAIN SplitZip LUTs are reused.
The two existing TEST documents are measured at 512 and 8,192 tokens. Original
word digests, token/capture receipts and packet digests are checked.

Before any TEST execution, the runner loads the original TABLES calibration
document and verifies its original words and token identities. It measures
native per-layer rANS words and SplitZip bytes, ranks their size difference with
stable layer-number ties, and freezes the five masks. The same state and coding
options are used for that ranking and the mixed codec. Negative benefits remain
visible in diagnostics.

For each length, the runner then measures complete packets and full encode and
decode calls on TABLES for all five masks. It freezes a choice for each
1/100/200/300/400 Gb/s scenario and each objective:

- **Precompressed:** wire time plus full GPU decode.
- **Fresh transfer:** full GPU encode plus wire time plus full GPU decode.

Selection uses three calibration samples after one warmup. TABLES also fitted
the probability spreads: this is **calibration-selected**, not independent
validation. Every fixed-mask TEST result is retained, but TEST results do not
choose a mask or policy. A policy JSON and digest are saved before TEST workers
start. Independent documents will be needed to assess generalization after this
implementation experiment.

## Measurements and checks

Run actual CUDA smoke tests before the long comparison. They exercise native
mixed coding, packet restoration, direct compact decoding, and ExANS packing and
native round trips. A skipped CUDA test does not pass the gate. The runner
requires the frozen A4500/CUDA 12.8 setting and existing pinned environments.

Every new measured packet is serialized and restored, then decoded in a fresh
process given only compressed packets and shared state. That receiver alternates
the two documents in the order 0/1/0, retains earlier returned outputs, verifies
their ownership and original BF16 word digests, and checks compressed inputs
remain unchanged. Existing controls keep their original frozen worker and
receiver checks. Author, frozen-core and measurement source receipts are checked
before and after execution.

Both full encode and full decode use the same unchanged author timing body:
10 warmups, 50 calls and five repeats. Separate isolated timings use two warmups
and five samples. Packet export to CPU, disk I/O, upload, offline fitting and
cold initialization are separate from codec timing. Actual serialized file bytes
provide complete bpv. Shared state, native reservations, output and workspace
are reported separately where observed; native allocator peaks are unmeasured.

The mixed method uses unchanged author SplitZip code in our Torch 2.11 /
Triton 3.6 environment, while its control keeps the original Torch 2.9 /
Triton 3.5 environment. Record both software receipts. This compatibility
variation and the checked carriers must accompany any speed comparison.

Link tables calculate **serial** cost using `8 * packet_bytes / (Gb/s * 1e6)`
milliseconds. They show batched and isolated timings separately and use the
faster before/after baseline round as a conservative control. They do not claim
measured NIC bandwidth, overlap, SSD performance or serving throughput.

## Insights to retain

- Per-layer native bytes, rANS words, escapes, and mask transitions: identify
  where prediction saves enough bytes to justify a serial dependency or another
  fallback call.
- Complete wire bytes versus native used bytes, fixed reservations and framing:
  distinguish real compression from untransmitted capacity.
- Full-call CPU and CUDA profiles for representative 8K mixed and compact ExANS
  requests: allocation, copying, validation, synchronization, kernel counts and
  kernel time.
- Separate unpack/native-decode or expansion diagnostics where available. These
  are **nonadditive** samples, not replacements for complete API timings.
- Frozen policies and all fixed-mask TEST rows across the link scenarios: expose
  where fewer bytes offset extra GPU work and where the decoder remains limiting.

Small summaries and failure logs are published automatically on the new branch.
Large packets and full profiler traces stay in the NUC run directory. A failure
stops the run and preserves its report; it does not substitute CPU measurements.

## Run on the NUC

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/combined_transfer_v1
git pull --ff-only origin experiment/combined_transfer_v1
bash scripts/run_combined_transfer_a4500.sh
```

No GPU performance has been measured by the local software checks that prepare
this experiment. Existing results remain reference evidence, not results for
these newly combined arms.

## Separate next experiment: lossy quality and resident memory

After this lossless run, carry over the team's growing-cache quality protocol
with an actual serialized-packet decoder feeding the continuation. Keep global
initial sinks and the recent tail exact, charge every frame and protected value,
and test the same windows and state. Lossy quality, lossless transfer, logical
compressed storage, and resident attention memory require separate tables.
An attention-facing P format, T-to-P conversion, and serving allocator remain
their own experiments.
