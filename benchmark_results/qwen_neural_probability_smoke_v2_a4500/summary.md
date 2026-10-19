# Learned probability reference study

Status: **completed**

Exact BF16 previous-layer probability coding with real indexed packets, GPU/CPU neural prediction, and separately timed CPU arithmetic coding. TRAIN-only fits; held-out DEV continuation and sampled TEST bytes. Complete neural/anchor state is charged. This experiment uses produced KV rather than prefill hidden metadata. No pretrained-model quality, compact-resident attention, partial-resume scheduler, high-concurrency, latency or novelty claim is established.

| Case | Tokens | Method | Sample stream reduction | Net single-cache estimate | State MiB |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 2048 | frozen_exponent_control (ok) | 26.07% | 26.04% | 0.081 |
| test_pottery_kiln_diary | 2048 | neural_per_layer (ok) | 29.78% | -128.84% | 456.832 |
| test_pottery_kiln_diary | 2048 | neural_shared (ok) | 30.39% | 25.47% | 14.156 |

Payload rates use actual sampled packet bytes. Full-context and concurrent shared-state figures are estimates from the same sampled addresses, with full state charged. These are not measured concurrent serving results.

Neural inference runs on the selected device; arithmetic encode/decode runs on the CPU. Dependency decoding is sequential across layers for each token tile. Full original KV is still used for model capture; encoded-resident attention is not implemented.

All development generation checks:

- per_layer: ok
- shared: ok
