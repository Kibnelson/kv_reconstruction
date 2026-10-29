# Frozen identical-KV lossless comparison

## What has been established

SplitZip's unchanged native entrypoint completed on the pinned BF16 Llama-3.2-3B
and A4500: 512 MiB of original WikiText KV became 385.46048 MiB, 12.045640 bpv,
24.71475% saved, with exact original-word reconstruction. Native mean encode and
decode were 3.80874 and 1.71505 ms. Its separate report is commit `10771af`.

The released ExANS synthetic test completed from a fresh SM86 build in
commit `7e65a98`. All eleven packaged artifact digests match. The eight checked
author source files stayed unchanged, all seven commands exited zero, and the
runtime-library paths were checked. The author test verifies a compressible
round trip and an overflowing fixture; it is silent on success. It provides
correctness evidence, with no new Llama compression or throughput measurement.
The published executable/library hashes are build receipts; binaries remain on
the NUC and have not been independently rehashed from GitHub.

## Next run

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_frozen_transfer_a4500.sh
```

This is a **matched transfer integration experiment**, using unchanged codec
implementations and existing fitted state. It is distinct from the author-native
WikiText procedure and from an official ExANS serving reproduction. No model
download, calibration fit, predictor/kernel edit, lossy quantizer or attention
integration is introduced.

| Fixed item | Choice |
|---|---|
| Reference | `810b516`, existing serialized packets and checked state |
| Model | Original BF16 `unsloth/Llama-3.2-3B`, revision `d4446454d87d51aa42e1fb174f25acc5f8762331` |
| TEST workload | The same PEP/CPython text/code documents, both 512 and 8192 tokens |
| Original geometry | All 28 layers, `[28, T, 2048]`, start 0 |
| Methods | Raw BF16, SplitZip framed native codec, ExANS framed C-ABI control, our Dense and Rank256 |
| Our decoder | Compact/search at 512; direct/search at 8192, fixed before this run |
| Calibration | Reuse existing TRAIN/TABLES fits and LUTs; no TEST refitting |
| Native software | Existing pinned environments, author sources and frozen ExANS library |

Each encoder receives original words decoded from the frozen raw-BF16 control
outside the timed interval. Word digests must match the frozen reference across
all methods. Complete layer/token/channel geometry is retained for prediction;
generic codecs see the same words flattened internally.

Every newly encoded packet is decoded and checked. Own/raw/SplitZip packets must
match their frozen serialized bytes. For ExANS, exact reconstruction, metadata,
actual size, fit/fallback counts and fixed geometry are mandatory; old/new wire
digests are also recorded. The author clears fixed-slot padding, but the released
nvCOMP API does not establish cross-run byte canonicality, so a digest difference
alone is not treated as an incorrect lossless reconstruction. The actual newly
encoded packet is used for decode timing and independent receiver validation.

The receiver is a new process with only packet/state/native-library references
and expected digests. It receives no original KV, prediction arrays or raw-input
references. Alternating packets and retained outputs check that returned values
do not alias mutable decoder scratch. Packet immutability is checked outside
ordinary timing intervals.

## Two timing views

The common batched convention uses the actual `bench` and `synchronize`
function bodies from the SHA-checked SplitZip author source: ten warmups, fifty
API calls bracketed by synchronization, five repeats. Only those two functions
are loaded, unchanged, into each worker's native Torch environment. This reuses
the author's **timing convention**, not its entire native calibration/workload
procedure. Samples, mean time and original-BF16-byte GB/s are retained.

A separate isolated-call view uses two warmups and five synchronized calls for
every method, retaining wall medians, CUDA event intervals and Torch allocation
samples. Batched per-call time measures amortized execution; isolated time is
the receiver latency view. A difference between these views is evidence about
execution overhead, not an optimization of the frozen codec.

Both encode and decode call the complete existing GPU-resident packet APIs.
Encode includes prediction, coding and compaction/framing; it receives no oracle
predictions. Decode includes required metadata handling, native work and returned
BF16 output. JIT/state loading, graph setup, input loading, serialized export,
checksumming and host upload remain outside those GPU API intervals. Shared
coefficients/tables and decoder workspace are additional to complete packet
bytes. Torch peaks do not measure every native nvCOMP allocation or serving
capacity.

Baseline workers bracket the own-method workers, before and after, using their
pinned interpreters. Both rounds stay visible. Link targets use the faster
before/after baseline as an explicitly conservative comparison, not a deployed
runtime selector. Both documents contribute equally; all per-document samples
remain available.

## ExANS scope

The frozen C-ABI adapter uses 999,424-byte raw blocks, 770,048-byte fixed records,
physical tail padding and explicitly charged raw overflow fallback. Its 8K
request submits 941 blocks together. Preserve this setting to compare with
`810b516`; do not silently substitute the official connector's maximum-64 batches
and stop-at-first-overflow policy. Batch-wide phase costs can change encoded
bytes when batching changes, so that would require its own new experiment.

The original frozen native library remains the active control. Its digest is
checked separately from the newer correctness-test build. Both use the reviewed
author source pin and disclosed compiler compatibility header. Neither this
adapter nor the successful synthetic preflight establishes compatible unchanged
Llama/vLLM serving geometry or reproduces the post's H100 headline.

## Results and next decisions

Report complete packet bytes/bpv, shared state, exactness and encode/decode for
both timing conventions. Compute 1/100/200/300/400 Gb/s scenarios separately for
precompressed `wire + decode` and fresh `encode + wire + decode`. `max(wire,
decode)` is an unmeasured ideal overlap bound. These scenarios measure no NIC,
SSD, T-to-P conversion, compressed attention or serving throughput.

Keep official-native reproduction receipts and matched-transfer tables distinct.
After this run, use the common measurements to identify the decode reduction or
byte reduction required to beat each baseline at each link speed. The full
WikiText quality experiment for official lossy AQUA and our lossy codecs remains
pending; resident P is another workstream. No lossy result belongs in this
lossless exact-original-word table.

See [the protocol/T/P plan](nuc_baseline_reproduction_plan.md) and
[official baseline progress](official_baseline_progress.md).
