# Lite: speed, compression and the research contribution

The latest fused lossless codec saves **44.17%** of raw BF16 bytes and takes
**157.745 ms encode / 158.274 ms decode** on the A4500. Its B8 reference took
212.698 / 249.423 ms in the same experiment. Every packet byte and reconstructed
word matched; the improvement came from execution changes.

The new profile leaves entropy decoding as the largest GPU component:
448 calls consume 76.431 ms of 127.186 ms captured kernel time. These are
instrumented diagnostic times, separate from ordinary codec latency. There are
still 18,877 GPU kernels in the profiled fused decoder. Matrix multiplication
accounts for about 7.2 ms of device time.

## The next experiment

Each causal group has about 512 tokens x 128 channels = 65,536 values. The
current layout splits them into 512 independent rANS streams, each processing
128 values in sequence. With eight streams per program, that launches only
64 programs per group. Shorter streams increase the independent work available
to the GPU, while adding terminal states and directory entries.

| Values per stream | Lanes/program | Streams | Programs | State + directory bpv | Extra versus current |
|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 512 | 64 | 0.3125 | 0 |
| 64 | 8 | 1,024 | 128 | 0.6250 | 0.3125 |
| 32 | 8 | 2,048 | 256 | 1.2500 | 0.9375 |
| 64 | 32 | 1,024 | 32 | 0.53125 | 0.21875 |
| 32 | 32 | 2,048 | 64 | 1.0625 | 0.7500 |

These are exact state/directory costs for this divisible geometry, not full
packet rates or predicted speedups. They charge 32 bits for every lane's rANS
state and 64 bits per program offset. Inactive tail lanes also have states.
The new experiment measures actual framing, entropy bytes and complete packet
sizes. The predictor, calibration, adaptive probabilities and entropy arithmetic
remain unchanged; the new stream grouping changes the wire layout.

The control uses the successful fused implementation with 128 values/stream
and eight lanes/program. It runs before and after the four candidate layouts.
All candidates decode exactly the same original BF16 words. The control must
also reproduce the previous packet bytes. Fresh receivers get only fixed state
and newly serialized packets.

Raw BF16, pinned official SplitZip and pinned official ExANS are retimed through
the existing baseline harness and runtime environments. No model is rerun, no
fit is changed, and no baseline is replaced by a local approximation. This
focused experiment measures lossless mode; the previous lossy results remain
available and are not promoted using the lossless stream tuning.

## The decision rule

For each actual packet size S, compare serial precompressed fetch

    decode_ms + 8 * S_bytes / (effective_Gbps * 1e6)

at 1, 2.5, 5 and 10 Gb/s. Fresh transfer also pays encode. These calculations
exclude actual network transport, staging, SSD, queueing, serialization and
overlap; they are not serving measurements.

The present packet needs decode below about 18.8 ms to beat SplitZip at 5 Gb/s.
A larger packet has a smaller decode budget. The report recomputes this budget
for every candidate using retimed SplitZip and actual transmitted bytes.
Promote a candidate for an intended link only when its complete fetch cost is
lower. Keep size and timing results together even if a candidate is faster but
larger. The fixed shared predictor state remains additional to packet storage.

## What makes the approach a research candidate

The lens is **conditional entropy of the complete BF16 word**. A byte field
can have high marginal entropy while becoming more predictable once earlier
KV is known. Our decoder uses fixed fitted predictors conditioned on recovered
same-token previous-layer KV and finer earlier channel groups within the current
layer. The lossless coder identifies the original BF16 symbols exactly, even
when a prediction is wrong. It does not require an extra predictor-model forward
pass over the prompt.

This is a candidate contribution of the complete design, not a claim that
conditional coding, channel conditioning, linear regression or rANS is new.
Stream tuning and kernel fusion are performance engineering that preserve the
design; they are not independent novelty claims.

The closest distinctions remain specific:

| Prior method | Established overlap | Working distinction being tested here |
|---|---|---|
| AQUA-KV | Previous-layer key prediction; values predicted from previous-layer values and reconstructed current keys; quantized residuals | Exact original-BF16 coding together with finer earlier channel groups |
| SplitZip | Exact BF16 exponent coding with raw sign/mantissa and escapes | Conditional probability coding of the complete original word using reconstructed KV |
| ExANS | Exponent alignment and ANS coding, with raw sign/fraction bytes | Exploiting decoder-available conditional context across the full word |
| Speculative KV coding | Exact predictor-based KV entropy coding | Obtain prediction context from recovered KV without rerunning a predictor model on the prompt |

AQUA-KV already uses cross-layer and reconstructed K-to-V dependence. Those
ingredients alone cannot distinguish our method. Speculative KV coding already
establishes exact predictor-based KV coding. More systematic prior-art checking
and matched ablations remain necessary before claiming a first.

Primary method sources checked on 2026-10-28:

- [AQUA-KV, section 3.2 and Algorithm 1](https://arxiv.org/html/2501.19392v4)
- [SplitZip official implementation](https://github.com/Intelligent-Microsystems-Lab/SplitZip)
- [ExANS author description](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache)
- [Speculative KV coding author description](https://fergusfinn.com/blog/kv-entropy-coder/)

## Run on the A4500

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_lite_streams_a4500.sh
```

The default parent is
`runs/lite_execution_a4500_20261028_081432_1775498`. The run reuses the existing
model captures, fit, official codec environments and native ExANS library.
There are no downloads, refits or native builds. The wrapper runs actual-CUDA
parity checks and publishes the small report. Resume with
`bash scripts/run_lite_streams_a4500.sh --resume RUN_DIR` using identical options.

Optional separate profiles can be requested with `--profile`; full traces stay
local. GPU performance and large-request correctness are unmeasured until this
experiment completes. CPU interpreter parity tests use shorter dynamic loops
to keep software checks quick; the actual-CUDA preflight tests all five full
configurations above. CPU interpreter checks are software evidence only.
