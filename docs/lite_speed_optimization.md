# Lite speed: remove RoPE table lookup readbacks

The first isolated optimization addresses the 1,792 CUDA scalar readbacks per
512-token, 28-layer request identified in the existing A4500 profile. Every
group looks up RoPE tables twice, and the reference builds its cache key by
converting the first/last CUDA positions to Python integers, even on cache hits.

`FastEngine` subclasses the unchanged causal engine. It binds the host-known
request start, token count and device while the inherited `_run` executes.
`RequestRope.tables` uses this key, keeping the same float64 positions, angle,
cos/sin, scale, single-entry cache policy and downstream arithmetic. Start or
length changes invalidate the cache. The bound key is cleared on success or
failure. The engine is not a concurrent serving object.

No measured core file, fitting recipe, predictor, probability table, entropy
kernel or packet format changes. Existing calibration, captures and packets are
read-only. The original class remains the control. The experiment requires the
same GPU model, root software and original source hashes as the completed run.

## Run on the A4500

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_lite_fast_a4500.sh \
  runs/llama_causal_a4500_20261027_184615_1307629
```

The wrapper first runs the actual-CUDA packet/word parity test. It then tests
lossless Lite and the existing kappa 0.5/1 lossy modes on the same
512-token text/code requests, all 28 original BF16 layers. No fitting, model
download, quality rerun, AQUA or ExANS rebuild is performed. Every output packet
must match the previously measured packet byte for byte, including lossy modes.
Decoded word hashes must also match; lossless modes compare all original words.
A separate receiver process loads fixed state and original packet bytes only.
For lossy modes identical decoded words preserve the existing quality evidence;
no new task-quality claim is made.

Ordinary samples alternate reference/optimized execution order each round
(AB/BA), with two warmup rounds and six measured rounds. Encode includes the
existing extreme-value test and final compaction synchronization. Decode still
includes complete directory validation. Both engines are resident during the
paired measurements; baseline/peak Torch allocation samples include both
engines, and each engine's persistent predictor bytes are reported separately.
Packet export/upload and cold initialization are separate from codec time.

After the corresponding ordinary text/lossless timings, each arm receives a
separate encode/decode diagnostic profile. These instrumented times must not
be used as the speedup. Stream synchronization counts and operator diagnostics
help decide which indexing/conversion/scatter/entropy operations to optimize
next. Inclusive `kv/` ranges are not standalone kernels or additive costs.

The wrapper publishes only small evidence. Full traces remain in the run
directory. If interrupted, reuse the printed output directory:

```bash
bash scripts/run_lite_fast_a4500.sh --resume runs/lite_fast_a4500_TIMESTAMP_PID
```

The resume recipe must match. Completed paired rows and profiles are retained;
packets, source code and trace hashes are checked again. The independent
receiver is replayed. Use a fresh output directory if changing timing options.

Local CPU interpreter tests establish software compatibility only. Actual
A4500 speedup and synchronization reduction remain unmeasured until this run.
There is no resident-pool, compressed-attention or serving-throughput result.
