# Combined lossless transfer comparison

**Status: completed.** GPU measured: **True**.

Lossless complete GPU transfer adapters on frozen original Llama words and fitted state. Mixed masks and link policies use TABLES calibration only, not TEST. TABLES also calibrated spreads; it is not an independent validation document. Native source benchmarks and lossy quality remain separate. No P attention, T-to-P, NIC, or SSD measurement.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.180 | 3.498 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.377 | 3.413 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.293 | 3.418 |
| 512 | mixed_rank256_k0 | paired | 12.04281 | 0.734 | 0.354 |
| 512 | mixed_rank256_k14 | paired | 10.75255 | 6.644 | 5.030 |
| 512 | mixed_rank256_k2 | paired | 11.87517 | 6.051 | 3.602 |
| 512 | mixed_rank256_k28 | paired | 9.50487 | 7.289 | 6.382 |
| 512 | mixed_rank256_k8 | paired | 11.31117 | 6.319 | 4.374 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.322 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.735 | 0.311 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.734 | 0.311 |
| 512 | team_dense_uniform | paired | 9.11376 | 24.216 | 4.154 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.939 | 4.547 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.461 | 23.046 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.804 | 23.397 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 184.871 | 23.359 |
| 8192 | mixed_rank256_k0 | paired | 12.05013 | 10.293 | 3.223 |
| 8192 | mixed_rank256_k14 | paired | 10.64004 | 37.909 | 16.319 |
| 8192 | mixed_rank256_k2 | paired | 11.85722 | 30.515 | 10.141 |
| 8192 | mixed_rank256_k28 | paired | 9.30605 | 46.532 | 23.357 |
| 8192 | mixed_rank256_k8 | paired | 11.24262 | 34.300 | 13.299 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.318 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.317 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.310 | 3.188 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.229 | 3.186 |
| 8192 | team_dense_uniform | paired | 8.92097 | 84.230 | 32.480 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.807 | 23.288 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.451 | 3.772 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.313 | 3.651 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.498 | 3.680 |
| 512 | mixed_rank256_k0 | paired | 12.04281 | 0.884 | 0.602 |
| 512 | mixed_rank256_k14 | paired | 10.75255 | 7.369 | 5.811 |
| 512 | mixed_rank256_k2 | paired | 11.87517 | 6.953 | 4.280 |
| 512 | mixed_rank256_k28 | paired | 9.50487 | 7.875 | 7.256 |
| 512 | mixed_rank256_k8 | paired | 11.31117 | 7.199 | 5.144 |
| 512 | raw_bf16 | after | 16.00009 | 0.422 | 0.350 |
| 512 | raw_bf16 | before | 16.00009 | 0.404 | 0.335 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.870 | 0.537 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.876 | 0.533 |
| 512 | team_dense_uniform | paired | 9.11376 | 24.757 | 4.381 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.543 | 4.830 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 210.052 | 23.610 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 195.424 | 23.621 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 193.756 | 23.619 |
| 8192 | mixed_rank256_k0 | paired | 12.05013 | 10.407 | 3.445 |
| 8192 | mixed_rank256_k14 | paired | 10.64004 | 37.308 | 16.264 |
| 8192 | mixed_rank256_k2 | paired | 11.85722 | 30.500 | 10.476 |
| 8192 | mixed_rank256_k28 | paired | 9.30605 | 45.031 | 21.697 |
| 8192 | mixed_rank256_k8 | paired | 11.24262 | 33.930 | 13.489 |
| 8192 | raw_bf16 | after | 16.00001 | 4.982 | 3.435 |
| 8192 | raw_bf16 | before | 16.00001 | 4.976 | 3.423 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.416 | 3.381 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.389 | 3.374 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.891 | 29.010 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 42.136 | 21.631 |

Frozen mixed policy selected on TABLES before any TEST encode/decode.

| Tokens | Gb/s | Precompressed k | Fresh k |
|---:|---:|---:|---:|
| 512 | 1 | 28 | 28 |
| 512 | 100 | 0 | 0 |
| 512 | 200 | 0 | 0 |
| 512 | 300 | 0 | 0 |
| 512 | 400 | 0 | 0 |
| 8192 | 1 | 28 | 28 |
| 8192 | 100 | 0 | 0 |
| 8192 | 200 | 0 | 0 |
| 8192 | 300 | 0 | 0 |
| 8192 | 400 | 0 | 0 |

Calculated serial precompressed latency (batched), ms. Fixed baseline controls use the faster full before/after round.

| Tokens | Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 | Frozen mixed policy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 469.993 | 353.889 | 366.882 | 326.383 | 271.735 | 283.616 | 285.446 |
| 512 | 100 | 4.926 | 3.847 | 7.047 | 6.727 | 6.829 | 7.338 | 3.890 |
| 512 | 200 | 2.577 | 2.079 | 5.230 | 5.112 | 5.492 | 5.943 | 2.122 |
| 512 | 300 | 1.795 | 1.489 | 4.624 | 4.574 | 5.046 | 5.477 | 1.533 |
| 512 | 400 | 1.403 | 1.195 | 4.321 | 4.305 | 4.823 | 5.245 | 1.238 |
| 8192 | 1 | 7519.513 | 5663.878 | 5820.316 | 5132.793 | 4223.214 | 4395.045 | 4394.987 |
| 8192 | 100 | 78.479 | 59.793 | 81.329 | 74.144 | 74.388 | 67.005 | 59.830 |
| 8192 | 200 | 40.898 | 31.490 | 52.344 | 48.595 | 53.434 | 45.147 | 31.527 |
| 8192 | 300 | 28.371 | 22.055 | 42.682 | 40.079 | 46.449 | 37.860 | 22.092 |
| 8192 | 400 | 22.108 | 17.338 | 37.852 | 35.821 | 42.957 | 34.217 | 17.375 |

Calculated serial fresh latency (batched), ms. Fixed baseline controls use the faster full before/after round.

| Tokens | Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 | Frozen mixed policy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 470.316 | 354.624 | 380.180 | 340.564 | 295.951 | 290.556 | 292.735 |
| 512 | 100 | 5.249 | 4.581 | 20.345 | 20.907 | 31.046 | 14.277 | 4.625 |
| 512 | 200 | 2.900 | 2.813 | 18.528 | 19.293 | 29.708 | 12.882 | 2.857 |
| 512 | 300 | 2.117 | 2.224 | 17.922 | 18.754 | 29.262 | 12.417 | 2.267 |
| 512 | 400 | 1.725 | 1.929 | 17.619 | 18.485 | 29.039 | 12.184 | 1.973 |
| 8192 | 1 | 7524.402 | 5674.108 | 6005.186 | 5326.253 | 4307.443 | 4438.853 | 4441.519 |
| 8192 | 100 | 83.369 | 70.023 | 266.199 | 267.604 | 158.617 | 110.813 | 70.123 |
| 8192 | 200 | 45.788 | 41.719 | 237.215 | 242.056 | 137.664 | 88.954 | 41.820 |
| 8192 | 300 | 33.261 | 32.285 | 227.553 | 233.539 | 130.679 | 81.668 | 32.385 |
| 8192 | 400 | 26.998 | 27.567 | 222.722 | 229.281 | 127.187 | 78.025 | 27.668 |

Calculated serial precompressed latency (isolated), ms. Fixed baseline controls use the faster full before/after round.

| Tokens | Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 | Frozen mixed policy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 470.100 | 354.112 | 367.120 | 326.658 | 271.962 | 283.900 | 286.321 |
| 512 | 100 | 5.033 | 4.069 | 7.285 | 7.001 | 7.057 | 7.621 | 4.137 |
| 512 | 200 | 2.684 | 2.301 | 5.468 | 5.386 | 5.719 | 6.226 | 2.369 |
| 512 | 300 | 1.901 | 1.712 | 4.862 | 4.848 | 5.273 | 5.761 | 1.780 |
| 512 | 400 | 1.510 | 1.417 | 4.559 | 4.579 | 5.050 | 5.528 | 1.486 |
| 8192 | 1 | 7519.619 | 5664.067 | 5820.575 | 5133.357 | 4219.743 | 4393.388 | 4393.327 |
| 8192 | 100 | 78.585 | 59.981 | 81.588 | 74.708 | 70.917 | 65.348 | 60.052 |
| 8192 | 200 | 41.004 | 31.678 | 52.604 | 49.159 | 49.964 | 43.489 | 31.749 |
| 8192 | 300 | 28.477 | 22.243 | 42.942 | 40.643 | 42.979 | 36.203 | 22.314 |
| 8192 | 400 | 22.214 | 17.526 | 38.111 | 36.385 | 39.487 | 32.560 | 17.597 |

Calculated serial fresh latency (isolated), ms. Fixed baseline controls use the faster full before/after round.

| Tokens | Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 | Frozen mixed policy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 470.504 | 354.986 | 380.648 | 342.108 | 296.720 | 291.442 | 294.195 |
| 512 | 100 | 5.437 | 4.943 | 20.813 | 22.452 | 31.814 | 15.164 | 5.021 |
| 512 | 200 | 3.089 | 3.175 | 18.996 | 20.837 | 30.476 | 13.768 | 3.253 |
| 512 | 300 | 2.306 | 2.586 | 18.390 | 20.299 | 30.030 | 13.303 | 2.664 |
| 512 | 400 | 1.914 | 2.291 | 18.087 | 20.030 | 29.807 | 13.071 | 2.369 |
| 8192 | 1 | 7524.595 | 5674.456 | 6014.331 | 5343.409 | 4303.634 | 4435.524 | 4438.358 |
| 8192 | 100 | 83.562 | 70.371 | 275.344 | 284.760 | 154.808 | 107.484 | 70.459 |
| 8192 | 200 | 45.981 | 42.067 | 246.359 | 259.211 | 133.855 | 85.626 | 42.156 |
| 8192 | 300 | 33.454 | 32.633 | 236.698 | 250.695 | 126.870 | 78.339 | 32.721 |
| 8192 | 400 | 27.190 | 27.915 | 231.867 | 246.437 | 123.378 | 74.696 | 28.004 |

`results.json` retains every fixed-k TEST result, frozen-policy TEST costs, and calculated 1/100/200/300/400 Gb/s serial wire + decode and encode + wire + decode costs. No TEST argmin, network measurement, or overlap speedup claim is made.

Batched full GPU APIs use unchanged author timing bodies: 10 warmups × 50 calls × five repeats. Isolated calls use two warmups and five samples. Packet export, disk, upload, fitting, and cold setup are excluded. Complete bytes come from actual serialized files; shared predictors and native memory reservations are additional.

Before/after raw, native SplitZip, and fixed ExANS controls run through the unchanged frozen worker in their original environments. Dense and Rank256 remain paired controls. New mixed and compact ExANS arms are local framed adapters, not unchanged author serving evaluations. Diagnostic profiler/component samples are separate and nonadditive. All results are lossless T; lossy quality and resident P remain follow-on work.
