# Lite: isolate dispatch cost without changing compression

The stream experiment accelerated the V64/B32 entropy decoder from 76.4 to
49.6 ms in the diagnostic trace, but full ordinary decode stayed near 156–159 ms.
Both layouts still launched 18,877 GPU kernels. This supports testing dispatch
overhead; it does not establish a precise CPU share of ordinary latency.

This experiment keeps both **V128/B8** (44.17% saved) and **V64/B32** (43.19%
saved). It changes only how the existing lossless decoder is submitted to the
GPU. Predictor coefficients, channel-group dependencies, floating-point math,
entropy kernels, calibration, packet bytes and reconstructed BF16 words remain
the reference. No model forward pass, fit, baseline rebuild or download occurs.

## Ordinary decode costs

| Step | Where it runs | Included in total graph decode? |
|---|---|---|
| Validate the actual received header and directory | CPU, including the existing directory readback | Yes |
| Copy the new payload prefix and directory into fixed buffers | GPU device-to-device copies | Yes |
| Replay the captured causal decoder | GPU, submitted through a CUDA graph | Yes |
| Clone the decoded output for independent ownership | GPU | Yes |
| Restore serialized packet bytes and upload them | CPU / GPU upload | Separate |
| Allocate, warm and capture a new graph | CPU / GPU | Separate cold setup record |

The graph captures decoder execution, with the existing per-group math and
dependencies. It does not capture packet validation, file access, hashing,
serialization or encoding. Validation runs on the **actual packet**, before
staging. Validating padded buffers would change the last-stream bounds check.

One fixed input arena uses the encoder's worst-case capacity for the selected
geometry. Different compressed lengths can share the same graph; only the valid
prefix is copied. This reserves more memory than the packet itself. The report
records staging bytes, captured output bytes, allocated/reserved Torch memory,
peaks, and synchronized cold warmup/capture/first-replay time. Torch accounting
does not cover every driver graph allocation; these are not total native memory
measurements.

The prototype has one active graph cache and one caller CUDA stream. Header
changes such as token positions invalidate the cache and pay another cold
setup. A cache replacement synchronizes before releasing the previous graph.
Each cache entry owns references to its cosine/sine tables, whose ordinary
single-entry RoPE cache can otherwise be replaced. A different caller stream
is rejected. This is not a concurrent serving allocator.

## Correctness and fair timing

Each layout loads two real packets for the original 28-layer, 512-token
Llama-3.2-3B requests. They have identical graph-bound metadata but different
payload lengths and data. Calls alternate packets and alternate eager/graph
order AB/BA. Both arms share a resident fitted engine and graph buffers during
the paired comparison; allocation baselines include that setup.

Recovered-word hashes are checked outside timed calls. A returned A output
must remain unchanged after decoding B, and both must match their original BF16
words. A/B alternation must reuse one capture. A fresh GPU process repeats the
packet-only check using fixed state, packet bytes and expected hashes; it has
no source KV or prediction arrays.

Actual-CUDA preflight tests cover changing payload lengths, independently owned
outputs, malformed-directory rejection before replay, start 0 → start 7 →
start 0 cache invalidation, raw BF16 fallback and stream guards. Capture failures
stop the arm; eager execution is never reported as graph execution.

Raw BF16, official SplitZip and official ExANS are retimed using their unchanged
workers and pinned environments. This is a **decoder-only** experiment. It
reports precompressed transfer plus decode at 1, 2.5, 5 and 10 Gb/s using actual
packet bytes; it does not combine new decode times with stale encoder times to
claim a fresh-transfer result. These link costs are calculated, not measured
network or serving latencies.

## Where time goes

After every ordinary timing job is complete, separate diagnostic processes
measure validation, input copies, graph replay and output cloning. Each phase
reports synchronized wall samples; GPU phases also report CUDA event intervals
and CPU submission duration. Added phase barriers change scheduling, so these
diagnostic medians must not be added into a purported ordinary latency breakdown.

Interpret the result as follows:

- If replay is much faster than ordinary eager decode, dispatch is a substantial
  opportunity. Compare both stream layouts again under that execution path.
- If graph replay itself remains expensive, inspect its GPU work next: entropy
  decoding and repeated prediction preparation, casts and statistics.
- If validation or copies take a material fraction of total latency, improve
  their representation/access path while continuing to charge their cost.
- If cold capture or reserved memory is too expensive for reuse, measure the
  request count needed to amortize setup before proposing deployment.

Reaching the target still requires measured total decode below each layout's
link-specific budget. CUDA graphs alone are not evidence that the budget is
achievable. The proposed conditional full-BF16 codec remains the research
candidate; graph submission is performance engineering.

## Run on the A4500

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_lite_graph_a4500.sh
```

The default parent is
`runs/lite_streams_a4500_20261028_085102_2556937`. The wrapper first runs actual
GPU graph checks, then timings and separate diagnostics, and publishes a small
report. Resume with `bash scripts/run_lite_graph_a4500.sh --resume RUN_DIR` using
the same source and options; completed jobs are retained after hash checks.

Local software tests check staging, changing lengths, decoder-only bookkeeping
and GPU/publication guards. Actual capture correctness and speed require the
A4500 run. CUDA graph mechanics follow the
[PyTorch 2.11 CUDA graph documentation](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
