# Frozen lossless transfer comparison

**Status: completed.** GPU measured: **True**.

Matched complete lossless transfer adapters on frozen PEP/CPython KV. No source/model/predictor change. Native WikiText results are separate. ExANS retains the original 59/941-block C-ABI adapter, fixed slots, padding and raw fallback; this is not its maximum-64-block, stop-on-overflow vLLM serving procedure.

Timing basis: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.351 | 3.422 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.276 | 3.415 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.322 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.733 | 0.311 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.732 | 0.310 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.490 | 4.149 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.908 | 4.583 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.759 | 23.394 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 185.087 | 23.473 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.318 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.318 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.320 | 3.187 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.251 | 3.186 |
| 8192 | team_dense_uniform | paired | 8.92097 | 84.502 | 32.526 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.887 | 23.282 |

Timing basis: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.620 | 3.675 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.792 | 3.643 |
| 512 | raw_bf16 | after | 16.00009 | 0.411 | 0.336 |
| 512 | raw_bf16 | before | 16.00009 | 0.411 | 0.335 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.875 | 0.535 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.873 | 0.543 |
| 512 | team_dense_uniform | paired | 9.11376 | 23.111 | 4.203 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.627 | 4.807 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 194.892 | 23.631 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 194.019 | 23.711 |
| 8192 | raw_bf16 | after | 16.00001 | 4.982 | 3.433 |
| 8192 | raw_bf16 | before | 16.00001 | 4.978 | 3.427 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.414 | 3.379 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.409 | 3.383 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.701 | 29.249 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 42.136 | 21.581 |

Calculated **precompressed wire + full GPU decode**, milliseconds. Baseline cells use the faster before/after round as a predeclared conservative target; ours uses its paired round.

Timing basis: **batched**.

| Tokens | Gb/s | Raw | SplitZip | ExANS adapter | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 469.993 | 353.888 | 366.884 | 271.730 | 283.652 |
| 512 | 100 | 4.926 | 3.845 | 7.049 | 6.825 | 7.374 |
| 512 | 200 | 2.577 | 2.078 | 5.232 | 5.487 | 5.978 |
| 512 | 300 | 1.794 | 1.488 | 4.626 | 5.041 | 5.513 |
| 512 | 400 | 1.403 | 1.194 | 4.323 | 4.818 | 5.281 |
| 8192 | 1 | 7519.513 | 5663.878 | 5820.351 | 4223.260 | 4395.039 |
| 8192 | 100 | 78.479 | 59.793 | 81.364 | 74.434 | 66.999 |
| 8192 | 200 | 40.898 | 31.490 | 52.379 | 53.480 | 45.141 |
| 8192 | 300 | 28.372 | 22.055 | 42.717 | 46.495 | 37.854 |
| 8192 | 400 | 22.108 | 17.338 | 37.886 | 43.003 | 34.211 |

Timing basis: **isolated**.

| Tokens | Gb/s | Raw | SplitZip | ExANS adapter | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 470.099 | 354.114 | 367.112 | 271.784 | 283.877 |
| 512 | 100 | 5.032 | 4.071 | 7.277 | 6.879 | 7.598 |
| 512 | 200 | 2.683 | 2.303 | 5.460 | 5.541 | 6.203 |
| 512 | 300 | 1.900 | 1.714 | 4.854 | 5.095 | 5.738 |
| 512 | 400 | 1.509 | 1.419 | 4.551 | 4.872 | 5.505 |
| 8192 | 1 | 7519.623 | 5664.071 | 5820.587 | 4219.983 | 4393.338 |
| 8192 | 100 | 78.589 | 59.986 | 81.600 | 71.157 | 65.298 |
| 8192 | 200 | 41.008 | 31.682 | 52.615 | 50.203 | 43.440 |
| 8192 | 300 | 28.481 | 22.248 | 42.954 | 43.218 | 36.153 |
| 8192 | 400 | 22.218 | 17.531 | 38.123 | 39.726 | 32.510 |

Fresh **encode + wire + decode** and perfect-overlap lower bounds are recorded separately for both timing bases in `results.json`; overlap is not measured.

Batched timing uses the pinned author SplitZip bench loop: 10 warmups, 50 calls between synchronizations, five repeats. Isolated timing records per-call synchronized medians separately. Both are complete GPU adapter calls; source data/uploads, packet serialization, fitting and cold setup are excluded.

Link models at 1/100/200/300/400 Gb/s are calculated, not measured network transfers. Faster before/after baseline controls are conservative targets; every round remains visible. Do not interpret a timing-basis change from 810b516 as a codec optimization.

All results here are lossless. Official lossy AQUA quality, resident P attention and T-to-P conversion remain separate experiments.
