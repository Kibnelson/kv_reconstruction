# Calibrating exact KV probabilities for packet bytes

The register-reuse experiment preserved every original BF16 word but did not
improve decode speed. Its one-warp Rank256 reference decoded an 8K request in
21.621 ms, versus 21.894 ms with register reuse; two warps was slower. This
experiment keeps the one-warp reference decoder and tests whether a change in
fixed probability calibration can reduce bytes while retaining the same kernel
structure, matrix sizes and launch count.
No reduction or speed improvement is claimed before the new GPU run.

The existing 8K Rank256 packet is 521.154 MiB, with only about 49,858 bytes of
framing, directories and checksum. That overhead is about 0.0091% of the packet.
Useful size improvements therefore need to reduce the coded payload. The
corresponding Dense packet is 499.574 MiB, but its 28.413 ms decode is slower.
The goal is to keep useful byte/time tradeoffs, rather than promote a candidate
solely for its packet size or decoder latency.

## What changes

The frozen encoder uses per-channel residual spread `sp` times a token multiplier
derived from the preceding layer's prediction error. This pilot changes `sp`
only. It keeps the predictor coefficients, INT8 feature scales, error
normalization `inv`, probability tables, stream layout and decoder kernels
unchanged. Keeping `inv` fixed preserves the adaptive priors; recomputing it
inversely from the new spread would partly cancel the intervention.

For each of the 16 key/value heads, one factor scales its 128 channel spreads
across all 28 layers. Dense and Rank256 receive independent factor vectors.
The predeclared grid is **0.5, 0.75, 1, 1.25 and 1.5**. The decoder reads the
resulting fixed arrays through the same code and uses the same matrix sizes,
rank and launch count. Probability changes can still change symbol, refill and
escape work and runtime, so decode time is measured again.

Calibration uses the existing full 8,192-token **TABLES** document, which already
supplied the original residual spreads. Each grid setting produces actual GPU
packets. The objective is each head's encoded segment lengths, including rANS
states and alignment, rather than a theoretical entropy estimate. Ties prefer
factor 1. The selected vector is re-encoded as one coherent state and its segment
lengths, relative starts and segment hashes must match the selected grid
observations. Every calibration decode must also recover the original BF16 words
exactly.

The per-head choice is possible because each head has separate streams and
neither the predictor nor fixed-`inv` priors depend on `sp`. It is fixed shared
calibration, with no per-request hint bits. State identity and all required
packet parts remain charged or reported through the checked packet format.
This is an engineering calibration experiment, not a new entropy-coding
novelty claim.

## Workload and fair controls

The model remains pinned `unsloth/Llama-3.2-3B`, original BF16, all 28 layers.
The six TRAIN documents and predictor fit are reused without modification.
TABLES alone selects probability factors. The chosen state is frozen before
TEST encoding and timing; text/code TEST results do not select
factors, fallbacks or routing.

The completed one-warp parent is
`runs/team_register_a4500_20261028_145318_1925338`. Its receipts bind the original
packets, fixed state, native baseline configuration and earlier source files.
The source captures remain in `runs/llama32_3b_compare`.

Fresh candidate packets are encoded for both TEST documents at **512 and
8,192 tokens**. Frozen original packets form the control. The measured encoder
latency is recorded separately from packet export and upload. The receiver
uses graph/reference mode for 512 and direct/reference mode for 8K; both have
one decoder warp, binary search, and independently owned BF16 outputs.

Raw BF16, official SplitZip and official ExANS run before and after the own-codec
comparisons. Their codec implementations, launch settings and pinned
environments stay unchanged. This remains a matched GPU-resident framed API
comparison; it does not reproduce every paper's complete serving evaluation.
AQUA-KV is lossy and remains a separate evaluation.

Each timed decode starts with compressed payload on GPU and includes validation,
normal metadata work, kernel execution and the owned output. Bulk host hashing,
serialization, uploads and cold setup are excluded from warm decode timing.
Actual CUDA gates precede the experiment. Independent packet-only receivers
check alternating inputs, retained-output ownership, unchanged packets and exact
original-word digests. Calibration/state/packet hashes bind the experiment and
its resume identity. This pilot adds no new profiler pass.

## Comparing bytes and time

All complete packet bytes are counted, including directories, padding and
checksums. Shared state, scratch and output allocations are additional; a smaller
transfer packet is not by itself a GPU-resident serving-capacity result.

The report evaluates calculated serial **wire time + measured full decode** at
**1, 100, 200, 300 and 400 Gb/s**, against both native baseline control rounds.
Ideal `max(wire, decode)` overlap is recorded separately as an unmeasured bound.
Fresh-encoding cost is reported for the own codecs and is not silently added to
only one side of a comparison against precompressed native packets. No actual
NIC, SSD, T-to-P conversion, compressed attention or serving throughput is measured.

The calibration grid contains the original factors, so calibration bytes cannot
increase after selection. That does not guarantee smaller TEST packets, nor
does unchanged kernel structure guarantee unchanged runtime. Negative results
are retained and published with the controls.

## Run

On the A4500 host:

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_team_probability_a4500.sh
```

To resume with the original source, calibration and workload options:

```bash
bash scripts/run_team_probability_a4500.sh --resume RUN_DIR
```

The wrapper packages and publishes small evidence files only. Packet binaries,
captured inputs and fitted tensor archives remain on the GPU host. The report
includes state manifests and binary hash receipts so subsequent experiments
can verify the exact packets and probability state.
