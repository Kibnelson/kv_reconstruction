# Lite: optimize against complete baseline costs

The target is lower ready-to-use KV latency than the strongest appropriate
baseline at effective 1, 2.5, 5 and 10 Gb/s. A smaller packet or a faster isolated
kernel is insufficient. Raw BF16, pinned official SplitZip and pinned official
ExANS remain lossless controls. FP8 is a separate lossy speed/quality control.
Official AQUA is still blocked; an AQUA-style adapter is not substituted.

The completed host-RoPE change preserves the original packets and reduces the
lossless 56 MiB request from about 340/428 ms encode/decode to 229/307 ms. SplitZip
previously measured 0.874/0.521 ms and ExANS 15.455/3.848 ms on identical words.
These measurements came from separate runs. The next experiment retimes every
control and candidate rather than treating older baseline timings as current.

## Next controlled experiment: coding-lane grouping

The complete Lite decoder uses 448 sequential channel groups. A typical group
contains 512 tokens x 128 channels = 65,536 values. With 128 values per rANS
stream there are 512 independent streams. Grouping 64 streams per GPU program
launches only eight programs. Grouping 32/16/8 streams launches 16/32/64 programs.
This may expose more parallelism and reduce the cost of program-wide prefix
operations and worst-lane searches. Actual speedup must be measured.

Keep the predictor, fitted state, RoPE arithmetic, probability tables and the
existing entropy kernels unchanged. Tune only the existing `block` option.
The B64 controls must match historical packets exactly. B8/B16/B32 have a
different stream grouping and offset directory: their exact serialized sizes
are measured, rather than inheriting the historical 44.5% saving.

Every lossless output must match every original BF16 word. Nonzero-kappa modes
must match the previous lossy reconstruction word for word, so their existing
short continuation quality evidence remains applicable. Fresh receiver
processes receive only fixed state and packet bytes. No reference prediction
arrays are available to those receivers.

The team's newer 20261028 component report motivates investigating integration
overhead, but its random-input matrix-product timings and oracle-conditioned
entropy round trips are not complete causal packet decoding results. Its
ExANS-style and AQUA-KV-style controls do not replace the official baselines.

## A4500 command

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_lite_baselines_a4500.sh
```

This runs actual-GPU lane-layout checks, then raw BF16, official SplitZip, FP8,
official ExANS, and Lite B64/B8/B16/B32/B64. B64 is repeated at the end to expose
drift. Each method uses a fresh process; timing workers never run the profiler.
All use the same two 512-token requests across all 28 layers. SplitZip keeps its
released environment and original TRAIN-fitted LUTs. ExANS reuses the checked
native library, fixed slots and paid raw overflow/padding. No training, model
downloads, dependency installs or native rebuilds are performed.

Results include complete packet rates, encode/decode wall samples, CUDA
intervals, Torch peaks, independent replay, time ratios against each baseline,
and a serial network model for 0.1/0.25/0.5/1/2.5/5/10/25/50/100 Gb/s. The lower
points expose the crossover; the primary targets remain 1–10 Gb/s.
The model reports fresh encode+transfer+decode and precompressed transfer+decode
separately. These calculations are not measured network or serving results.
Packet serialization/upload and codec initialization are recorded separately.

Only small evidence is published. Completed methods can be reused after an
interruption with `bash scripts/run_lite_baselines_a4500.sh --resume RUN_DIR`.
Source, state, packet and receiver-proof hashes must still match. Large packets
remain under runs/. Do not compare the lossy rows to lossless rows without an
explicit quality constraint. No resident pool, compressed attention, physical
SSD or serving-concurrency result is implied.
