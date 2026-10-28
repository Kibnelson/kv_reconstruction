# Team one-step codec: compact packets and independent receivers

This first comparison brings the team's measured one-step lossless decoder
into the existing transfer measurement scope. It preserves its predictor,
probability arithmetic, integer products and native kernels. The experiment
asks whether its speed survives real compact packet storage, reconstruction
of the decoder arena and independently owned outputs.

The source is pinned to
`a7778073902bc2bb8bb0433ff7cd2e10bd79f5c3` on
`llama32-3b-lossy-lossless-compare`. Five source modules are copied byte for
byte into `kv_compare/team_onestep/`; a manifest records their original paths
and SHA256 hashes. The inherited Lite, SplitZip and ExANS implementations
remain unchanged. This is a transfer-codec comparison, not a reproduction of
each paper's complete serving evaluation.

## Arms and inputs

The predeclared team arms are dense prediction with exact within-slice
probabilities, the same dense prediction with uniform within-slice
probabilities, and rank-256 prediction with uniform probabilities. The latter
two remain lossless: prediction approximations change coding cost, while the
stream still identifies every original BF16 word. No lossy residual
quantization is included.

The controls are the current lossless Lite graph decoder and the pinned raw
BF16, SplitZip and ExANS framed implementations. Native baseline workers keep
their original environments and runtime policies. ExANS retains its charged
fixed records, physical input padding and raw overflow fallback. Its policy
is not silently changed to the smaller used-record estimate from the other
report. AQUA-KV remains a separate lossy evaluation; no local AQUA adapter or
synthetic decoder timing is substituted in this exact-codec table.

Every arm receives identical original BF16 Llama-3.2-3B words, revision
`d4446454d87d51aa42e1fb174f25acc5f8762331`, all 28 layers. The default cases are
the first 512 and 8,192 tokens of both `test_text` and `test_code`, starting at
position zero. These documents are PEP 8 and CPython `argparse.py` from the
existing capture set. Full capture and token hashes are verified before
preparing the smaller case files. No model download or new KV generation is
required.

Calibration deliberately preserves the existing implementations:

| Implementation | Predictor training | Spread calibration |
|---|---|---|
| Team one-step | Six TRAIN documents, 8,192 tokens each | `tables`, 8,192 tokens |
| Existing Lite | Original four TRAIN prefixes, 1,024 tokens each; frozen fitted state | Original `tables` prefix, 1,024 tokens; frozen |

This difference is part of the result, not evidence for an isolated benefit of
one prediction structure. The dense exact/uniform pair shares one fitted
predictor and can isolate the probability-model change within the team
implementation. No configuration or layer mask is chosen using TEST cost.

## Packet and receiver scope

The team encoder reserves a worst-case arena and places each segment in its
own region. Only each segment's used suffix, including entropy states and
required alignment, belongs in the compact payload. A versioned packet
charges these suffixes, their directory, geometry/options, source and fixed
state identities, and checksum. A GPU expansion step restores their original
arena locations before invoking the unchanged native decoder. Unused arena
regions do not become transmitted bytes.

The experiment reports compact packet bytes and the much larger decoder
reservation separately. The reservation, shared constants, scratch and full
BF16 output remain real GPU memory costs; compact packet size is not a
resident-pool capacity measurement.

A fresh receiver process gets serialized packets, the fixed decoder state and
expected digests only. It does not get capture files, original KV arrays,
encoder features, predictions or per-request encoder priors. It must recover
every original BF16 word, handle alternating packets and preserve an earlier
returned output across later decodes. Software checks and GPU gates remain
separate; CPU evidence cannot be labeled a completed GPU comparison.

## Timing and link models

Warm complete decode includes packet metadata validation, staging or arena
expansion, decoder execution and an independently owned output. GPU kernel
intervals are separately labeled. Cold setup and JIT, serialized export and
host-to-device import, predictor fitting, shared state and allocation peaks
are additional records. The encoder's forward feature/prior computation is
included in its reported total encode cost. Diagnostic phase measurements
are not added to ordinary full-decode measurements.

For every case and each of 1, 100, 200, 300 and 400 Gb/s, show three different
planning quantities using the actual complete packet size `B`:

- Serial precompressed fetch: `8 B / link + decode`.
- Ideal fully overlapped pipeline estimate: `max(8 B / link, decode)`.
- Fresh encoding followed by serial fetch: `encode + 8 B / link + decode`.

These are calculated effective-link models. No network, SSD, transfer/decode
pipeline, T-to-P conversion, compressed attention or serving throughput is
measured. Serialized import/upload costs remain separate so that any future
transport model can charge the appropriate ingress path without counting it
twice.

## Run and decision

```bash
bash scripts/run_team_packet_a4500.sh
```

The wrapper runs actual CUDA correctness gates, executes the comparison and
publishes only small receipts. Captures, fitted state, compact packets and
large temporary arrays stay on the GPU host. The runner supports resuming
the same output directory with its recorded identities intact.

Use this result to choose the next narrow ablation. Fewer channel groups and
INT8 or reduced-rank prediction are not silently applied to existing Lite in
this step. A later 1/4/8/16-group experiment needs one common training protocol
to measure how much finer conditioning is worth its sequential cost. The
previous rank-delta rate probe remains a diagnostic, not a replacement for
the existing exact codec.

## Local validation

The 22-test suite passed 19 software checks and skipped three actual-CUDA
gates. It includes real native entropy encode/decode round trips in both
dense modes with nonzero INT8 coefficients under Triton's CPU interpreter,
all 65,536 BF16 bit patterns through framing, alignment and state identity,
corruption and bounds rejection, atomic state publication, receiver-source
exclusion and link-model accounting. The final runner recheck passed all ten
tests. Shell syntax and Python compilation also passed.

All five copied modules retain their pinned byte hashes, and all 53 inherited
source files retain the prior experiment's hashes. The local environment has
CPU-only Torch; no GPU timing, CUDA graph result or cross-GPU reproducibility
claim follows from these checks. The wrapper's three CUDA tests cover native
all-pattern decoding, alternating compact graph packets and output ownership,
and the rank-256 epilogue before starting the benchmark.
