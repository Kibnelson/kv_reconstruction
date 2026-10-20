# Compact prefill metadata and progressive exact KV study

Status: **completed**

Exact original-BF16 page-local progressive coding with zero-payload known-field summaries, 1/2/4-bit actual-prefill hints, page sharing and exact same-page K-to-V controls. Real complete CPU-reference packets and full state count. Full-cache/concurrency numbers are estimates, not measured serving or GPU codec gains.

Frozen DEV winner: **field_baseline**.

| Case | Tokens | Method | Packet reduction | Net full-cache estimate | Decode seconds |
|---|---:|---|---:|---:|---:|
| test_training_holdout_canal_gates | 2048 | frozen_exponent_control | 29.346% | 29.199% | 0.161 |
| test_training_holdout_canal_gates | 2048 | field_baseline | 30.484% | 30.459% | 0.265 |
| test_training_holdout_canal_gates | 2048 | lsb_local_none | 8.664% | 8.637% | 30.794 |
| test_training_holdout_canal_gates | 2048 | lsb_pool_none | 8.858% | 8.831% | 55.150 |
| test_training_holdout_canal_gates | 2048 | lsb_pool_kv2 | 8.838% | 8.808% | 46.527 |
| test_training_holdout_canal_gates | 2048 | lsb_pool_hidden1 | 8.865% | 8.834% | 55.865 |
| test_training_holdout_canal_gates | 2048 | lsb_pool_hidden2 | 8.792% | 8.761% | 53.005 |
| test_training_holdout_canal_gates | 2048 | lsb_pool_hidden4 | 8.841% | 8.810% | 54.955 |
| test_training_holdout_canal_gates | 2048 | lsb_page_hidden2 | 8.862% | 8.831% | 55.248 |
| test_training_holdout_canal_gates | 2048 | lsb_page_kv2 | 8.853% | 8.823% | 55.215 |
| test_training_holdout_canal_gates | 2048 | lsb_kfirst_none | 14.994% | 14.966% | 23.847 |
| test_training_holdout_canal_gates | 2048 | lsb_kfirst_kv2 | 14.991% | 14.960% | 18.984 |
| test_training_holdout_canal_gates | 2048 | lsb_kfirst_hidden2 | 15.007% | 14.976% | 23.723 |
| test_training_holdout_canal_gates | 2048 | msb_pool_none | 8.998% | 8.971% | 30.741 |
| test_training_holdout_canal_gates | 2048 | msb_pool_kv2 | 8.983% | 8.953% | 30.559 |
| test_training_holdout_canal_gates | 2048 | msb_pool_hidden2 | 8.982% | 8.951% | 39.199 |
| test_training_holdout_interval_merge | 2048 | frozen_exponent_control | 29.381% | 29.234% | 0.161 |
| test_training_holdout_interval_merge | 2048 | field_baseline | 30.478% | 30.454% | 0.267 |
| test_training_holdout_interval_merge | 2048 | lsb_local_none | 8.837% | 8.810% | 30.409 |
| test_training_holdout_interval_merge | 2048 | lsb_pool_none | 9.032% | 9.005% | 55.410 |
| test_training_holdout_interval_merge | 2048 | lsb_pool_kv2 | 9.012% | 8.982% | 46.015 |
| test_training_holdout_interval_merge | 2048 | lsb_pool_hidden1 | 9.039% | 9.008% | 55.298 |
| test_training_holdout_interval_merge | 2048 | lsb_pool_hidden2 | 8.969% | 8.938% | 51.810 |
| test_training_holdout_interval_merge | 2048 | lsb_pool_hidden4 | 9.016% | 8.985% | 55.428 |
| test_training_holdout_interval_merge | 2048 | lsb_page_hidden2 | 9.036% | 9.005% | 55.357 |
| test_training_holdout_interval_merge | 2048 | lsb_page_kv2 | 9.027% | 8.997% | 54.942 |
| test_training_holdout_interval_merge | 2048 | lsb_kfirst_none | 15.188% | 15.161% | 23.986 |
| test_training_holdout_interval_merge | 2048 | lsb_kfirst_kv2 | 15.185% | 15.154% | 19.434 |
| test_training_holdout_interval_merge | 2048 | lsb_kfirst_hidden2 | 15.202% | 15.172% | 23.250 |
| test_training_holdout_interval_merge | 2048 | msb_pool_none | 9.187% | 9.160% | 30.630 |
| test_training_holdout_interval_merge | 2048 | msb_pool_kv2 | 9.171% | 9.141% | 30.811 |
| test_training_holdout_interval_merge | 2048 | msb_pool_hidden2 | 9.171% | 9.140% | 38.381 |
| test_training_holdout_canal_gates | 8192 | frozen_exponent_control | 29.267% | 29.230% | 0.165 |
| test_training_holdout_canal_gates | 8192 | field_baseline | 30.506% | 30.500% | 0.265 |
| test_training_holdout_canal_gates | 8192 | lsb_local_none | 8.657% | 8.650% | 30.311 |
| test_training_holdout_canal_gates | 8192 | lsb_pool_none | 8.855% | 8.848% | 55.295 |
| test_training_holdout_canal_gates | 8192 | lsb_pool_kv2 | 8.836% | 8.829% | 48.689 |
| test_training_holdout_canal_gates | 8192 | lsb_pool_hidden1 | 8.860% | 8.852% | 55.341 |
| test_training_holdout_canal_gates | 8192 | lsb_pool_hidden2 | 8.785% | 8.777% | 53.802 |
| test_training_holdout_canal_gates | 8192 | lsb_pool_hidden4 | 8.840% | 8.832% | 55.453 |
| test_training_holdout_canal_gates | 8192 | lsb_page_hidden2 | 8.859% | 8.851% | 55.282 |
| test_training_holdout_canal_gates | 8192 | lsb_page_kv2 | 8.850% | 8.843% | 55.388 |
| test_training_holdout_canal_gates | 8192 | lsb_kfirst_none | 15.020% | 15.013% | 23.376 |
| test_training_holdout_canal_gates | 8192 | lsb_kfirst_kv2 | 15.018% | 15.011% | 18.633 |
| test_training_holdout_canal_gates | 8192 | lsb_kfirst_hidden2 | 15.034% | 15.027% | 24.314 |
| test_training_holdout_canal_gates | 8192 | msb_pool_none | 8.990% | 8.983% | 30.603 |
| test_training_holdout_canal_gates | 8192 | msb_pool_kv2 | 8.978% | 8.971% | 30.610 |
| test_training_holdout_canal_gates | 8192 | msb_pool_hidden2 | 8.980% | 8.972% | 40.853 |
| test_training_holdout_interval_merge | 8192 | frozen_exponent_control | 29.319% | 29.282% | 0.165 |
| test_training_holdout_interval_merge | 8192 | field_baseline | 30.496% | 30.490% | 0.264 |
| test_training_holdout_interval_merge | 8192 | lsb_local_none | 8.834% | 8.828% | 30.282 |
| test_training_holdout_interval_merge | 8192 | lsb_pool_none | 9.036% | 9.029% | 55.899 |
| test_training_holdout_interval_merge | 8192 | lsb_pool_kv2 | 9.016% | 9.008% | 47.846 |
| test_training_holdout_interval_merge | 8192 | lsb_pool_hidden1 | 9.040% | 9.032% | 55.374 |
| test_training_holdout_interval_merge | 8192 | lsb_pool_hidden2 | 8.969% | 8.961% | 52.111 |
| test_training_holdout_interval_merge | 8192 | lsb_pool_hidden4 | 9.020% | 9.012% | 55.725 |
| test_training_holdout_interval_merge | 8192 | lsb_page_hidden2 | 9.040% | 9.032% | 55.747 |
| test_training_holdout_interval_merge | 8192 | lsb_page_kv2 | 9.030% | 9.022% | 55.731 |
| test_training_holdout_interval_merge | 8192 | lsb_kfirst_none | 15.208% | 15.201% | 23.725 |
| test_training_holdout_interval_merge | 8192 | lsb_kfirst_kv2 | 15.206% | 15.198% | 19.589 |
| test_training_holdout_interval_merge | 8192 | lsb_kfirst_hidden2 | 15.223% | 15.215% | 22.856 |
| test_training_holdout_interval_merge | 8192 | msb_pool_none | 9.192% | 9.185% | 30.614 |
| test_training_holdout_interval_merge | 8192 | msb_pool_kv2 | 9.180% | 9.172% | 30.632 |
| test_training_holdout_interval_merge | 8192 | msb_pool_hidden2 | 9.182% | 9.174% | 40.516 |

All reported storage numbers use real complete packets. Models, lookup parameters, calibration, hint padding, raw fallbacks and all framing are charged. TEST cannot change DEV selection.

A producer-information gain requires reduced learned-stage payload against the same-layout no-hint/KV control. Descriptive header/name length, hint cost and full state deltas are reported separately.

The current reference performs entropy coding on CPU and probability inference on the selected device. All raw KV remains resident during capture. Timings include diagnostic staging; no fused GPU decoder, SSD transfer, concurrent admission or serving speedup is established.

Full DEV continuation checks: lsb_kfirst_hidden2: ok, lsb_pool_none: ok, lsb_pool_hidden2: ok, msb_pool_hidden2: ok, field_baseline: ok

Raw categorical hints are retained under hints/ for future TRAIN-only fitting and DEV/TEST sensitivity analysis. Raw KV and hidden captures are not exported.
