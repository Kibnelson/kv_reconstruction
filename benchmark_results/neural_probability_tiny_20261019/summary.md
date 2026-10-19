# Learned probability reference study

Status: **completed**

Exact BF16 previous-layer probability coding with real indexed packets, GPU/CPU neural prediction, and separately timed CPU arithmetic coding. TRAIN-only fits; held-out DEV continuation and sampled TEST bytes. Complete neural/anchor state is charged. This experiment uses produced KV rather than prefill hidden metadata. No pretrained-model quality, compact-resident attention, partial-resume scheduler, high-concurrency, latency or novelty claim is established.

| Case | Tokens | Method | Sample stream reduction | Net single-cache estimate | State MiB |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 16 | frozen_exponent_control (ok) | -177.78% | -304.64% | 0.005 |
| test_pottery_kiln_diary | 16 | neural_per_layer (ok) | -239.48% | -958.84% | 0.028 |
| test_pottery_kiln_diary | 16 | neural_shared (ok) | -237.72% | -960.38% | 0.028 |
| test_python_dependency_walk | 16 | frozen_exponent_control (ok) | -177.91% | -304.76% | 0.005 |
| test_python_dependency_walk | 16 | neural_per_layer (ok) | -238.89% | -958.25% | 0.028 |
| test_python_dependency_walk | 16 | neural_shared (ok) | -237.40% | -960.06% | 0.028 |
| test_museum_loan_register | 16 | frozen_exponent_control (ok) | -177.83% | -304.69% | 0.005 |
| test_museum_loan_register | 16 | neural_per_layer (ok) | -238.21% | -957.57% | 0.028 |
| test_museum_loan_register | 16 | neural_shared (ok) | -235.69% | -958.35% | 0.028 |
| test_pottery_kiln_diary | 24 | frozen_exponent_control (ok) | -178.15% | -262.72% | 0.005 |
| test_pottery_kiln_diary | 24 | neural_per_layer (ok) | -239.65% | -719.22% | 0.028 |
| test_pottery_kiln_diary | 24 | neural_shared (ok) | -237.77% | -719.54% | 0.028 |
| test_python_dependency_walk | 24 | frozen_exponent_control (ok) | -178.08% | -262.65% | 0.005 |
| test_python_dependency_walk | 24 | neural_per_layer (ok) | -235.40% | -714.97% | 0.028 |
| test_python_dependency_walk | 24 | neural_shared (ok) | -237.87% | -719.64% | 0.028 |
| test_museum_loan_register | 24 | frozen_exponent_control (ok) | -177.91% | -262.48% | 0.005 |
| test_museum_loan_register | 24 | neural_per_layer (ok) | -237.48% | -717.05% | 0.028 |
| test_museum_loan_register | 24 | neural_shared (ok) | -236.40% | -718.17% | 0.028 |

Payload rates use actual sampled packet bytes. Full-context and concurrent shared-state figures are estimates from the same sampled addresses, with full state charged. These are not measured concurrent serving results.

Neural inference runs on the selected device; arithmetic encode/decode runs on the CPU. Dependency decoding is sequential across layers for each token tile. Full original KV is still used for model capture; encoded-resident attention is not implemented.

All development generation checks:

- per_layer: ok
- shared: ok
