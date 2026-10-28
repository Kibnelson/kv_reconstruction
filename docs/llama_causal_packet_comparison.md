# Matched complete packets: Block32 and the team's fast predictor

This experiment connects the team's probability model and value-slice GPU coder
to an actual causal receiver. It compares predictor policies using one packet
format, identical original BF16 input words, TRAIN/scale partitions, arithmetic,
timing rules and accounting.

The default model remains unsloth/Llama-3.2-3B at
d4446454d87d51aa42e1fb174f25acc5f8762331, all 28 layers.
It reuses existing captures; it does not download another checkpoint or prefill
the documents again.

| Policy | Decoder context | Channel order / groups | Width |
|---|---|---|---|
| block32 | Same-token previous-layer KV | Adjacent RoPE pairs, groups of 32 channels | Fixed per channel |
| lite | Same-token previous-layer KV | Team's covariance pivot order, 16 groups | Adaptive from recovered residuals |
| strong_noatt, optional | Layer 0, layer l-2 and previous layer | Pivot order, 16 groups | Adaptive |

Block32 is a matched policy control implemented in this shared fitter/coder.
It is not the old frozen INT64 bitstream and must not replace the historical
8.857 complete-bpv result without a new measurement. Layer 0 is coded from its
available earlier channel groups in this experiment.

For each policy, the default modes are:

- Lossless: code each original BF16 word with the team's value-slice rANS model.
- Lossy step 0.5 and 1.0: quantize the residual with step kappa times the channel's
  calibration spread. Code integer residual symbols with 16-bit rANS frequency
  tables. Tail residuals escape to an exact original BF16 word.

Lossy tables use 65 fixed width classes from 0.25 to 4.0, spaced by 1/16 octave.
Their IDs are regenerated from recovered context. Each table assigns a positive
frequency to residuals -127 through 127 and an escape. Frequencies total 65536.
This discrete residual coder is a new implementation; it is not an official
AQUA quantizer. A lossy receiver must match the sender's intended reconstruction
word for word; only the lossless mode must match every original input word.

## Packet and receiver

1. Fixed fits, channel order and per-channel spreads are shared beforehand.
2. For each layer, project available earlier-layer KV once. Decode channel groups
   in order and update their de-rotated features and residual surprise.
3. Encode into a bounded staging arena, then compact all used stream segments on
   the GPU. One scalar size read allocates the final compact payload.
4. Serialize the compact payload, INT64 stream directory, checked geometry,
   mode/state/table identities and SHA256 integrity trailer.
5. A new receiver process loads only fitted state and packets. It regenerates
   predictions and spreads from recovered words. The verification request
   contains expected digests, not original KV or prediction/spread arrays.

Exceptional or extremely large BF16 input packets take an explicitly charged
raw fallback, preserving all 65536 word patterns. Lossy reconstruction is closed
loop: recovered words, including escaped originals, supply subsequent context.

The first implementation uses BF16 tensor-core products by default. FP32 is
available with --predictor-format fp32. Products have the same shapes and
arithmetic on encoder and receiver. CUDA deterministic algorithms and explicit
workspace settings are enabled. Cross-GPU portability is untested. The earlier
random INT8 product timings are not substituted for a measured causal INT8
predictor; this experiment does not implement that predictor yet.

## Fast first A4500 run

The default run fits four TRAIN documents, up to 1024 tokens each, and uses up to
1024 tokens of the separate scale document. It tests the first 512 tokens of
each TEST document and scores the following 127 target tokens from a 128-token
continuation. This is a small matched test, not the earlier full-calibration
37-variant frontier.

~~~bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_llama_causal_a4500.sh runs/llama32_3b_compare
)
~~~

The script reuses the original team captures, runs both policies and all three
modes, verifies every packet in a separate receiver process, measures continuation
quality and publishes a small report. A failed test is preserved and published
as failed. It never publishes captured KV, packets or fitted tensor files.

To use the exact frozen WikiText captures from the earlier official comparison:

~~~bash
bash scripts/run_llama_causal_a4500.sh \
  runs/llama32_official_a4500_20261022_125030
~~~

That path verifies the original capture file hashes and recovers token IDs from
the existing local WikiText cache. Text and complete token stream hashes must
match the old manifest. A missing cache is an error, with no replacement data.

Fitted layers are atomically saved and hashed in runs/_causal_fit_cache/. A second
packet length reuses matching fits. Interrupted fitting also resumes by layer.
The two predictor policies are fitted once and reused across all lossy steps.
The separate scale document calibrates spreads using the actual runtime BF16
or FP32 products. Completed measured packets are reused and independently
replayed on resume, without repeating their timing samples.
There is no attention predictor, query capture or 37-variant sweep by default.

To add the stronger context or increase the batch after the first result:

~~~bash
bash scripts/run_llama_causal_a4500.sh runs/llama32_3b_compare \
  --policies block32,lite,strong_noatt --tokens 2048
~~~

Resume a stopped run directly in its original directory, retaining the original
options. Completed packets and receiver outputs are checksum checked:

~~~bash
source .venv/bin/activate
unset TRITON_INTERPRET
python -u scripts/bench_llama_causal.py \
  --source-run runs/llama32_3b_compare --output runs/YOUR_CAUSAL_RUN
~~~

## Report interpretation

Complete bpv comes from actual serialized bytes, including directory, framing,
states in the entropy stream and checksum. Fixed fitted state is additional.
The JSON also records serialized fixed decoder-state bytes and the first
packet's rate if that complete fixed state had to accompany it.
Encoder and decoder wall samples include causal projection, group loops, RoPE,
adaptive widths, entropy operations, source/directory checks and compaction.
Torch allocation peaks include staging, recovered KV and temporary arrays.
Serialization/export, upload, fitting and model quality evaluation are separate.
Cold state initialization wall time is recorded separately. Resume identities
include framework versions, CUDA/GPU capability and arithmetic settings.
The implementation has CPU control and GPU prediction/entropy operations.

The quality reference restores the same captured exact KV. Restoring it twice
must agree; every lossless row must retain that NLL. The model's own prefill is
recorded separately because the shorter prompt can round differently. Lossy
quality is measured using the independent receiver's recovered KV.

This is a transfer-packet experiment, not compressed attention, T-to-P conversion,
a resident-pool memory result, SSD/network throughput or a serving benchmark.
It adds no stand-in AQUA/SplitZip/ExANS baselines. Existing official results remain
separate. The 49-51% probability costs and prior component GB/s are hypotheses
to test, not assumed rates or speeds of these packets.
