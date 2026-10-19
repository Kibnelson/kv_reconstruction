# Learned probability reference study

Status: **completed**

Exact BF16 previous-layer probability coding with real indexed packets, GPU/CPU neural prediction, and separately timed CPU arithmetic coding. TRAIN-only fits; held-out DEV continuation and sampled TEST bytes. Complete neural/anchor state is charged. This experiment uses produced KV rather than prefill hidden metadata. No pretrained-model quality, compact-resident attention, partial-resume scheduler, high-concurrency, latency or novelty claim is established.

| Case | Tokens | Method | Sample stream reduction | Net single-cache estimate | State MiB |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 2048 | frozen_exponent_control (ok) | 29.65% | 29.62% | 0.081 |
| test_pottery_kiln_diary | 2048 | neural_shared (ok) | 34.25% | 29.33% | 14.156 |
| test_python_dependency_walk | 2048 | frozen_exponent_control (ok) | 29.64% | 29.61% | 0.081 |
| test_python_dependency_walk | 2048 | neural_shared (ok) | 34.13% | 29.21% | 14.156 |
| test_museum_loan_register | 2048 | frozen_exponent_control (ok) | 29.64% | 29.61% | 0.081 |
| test_museum_loan_register | 2048 | neural_shared (ok) | 34.18% | 29.26% | 14.156 |
| test_pottery_kiln_diary | 8192 | frozen_exponent_control (ok) | 29.53% | 29.52% | 0.081 |
| test_pottery_kiln_diary | 8192 | neural_shared (ok) | 33.48% | 32.25% | 14.156 |
| test_python_dependency_walk | 8192 | frozen_exponent_control (ok) | 29.52% | 29.51% | 0.081 |
| test_python_dependency_walk | 8192 | neural_shared (ok) | 33.37% | 32.14% | 14.156 |
| test_museum_loan_register | 8192 | frozen_exponent_control (ok) | 29.51% | 29.50% | 0.081 |
| test_museum_loan_register | 8192 | neural_shared (ok) | 33.44% | 32.21% | 14.156 |

Payload rates use actual sampled packet bytes. Full-context and concurrent shared-state figures are estimates from the same sampled addresses, with full state charged. These are not measured concurrent serving results.

Neural inference runs on the selected device; arithmetic encode/decode runs on the CPU. Dependency decoding is sequential across layers for each token tile. Full original KV is still used for model capture; encoded-resident attention is not implemented.

All development generation checks:

- shared: ok
