# Lossless critical-first fallback/rANS overlap comparison

**Status: completed.** GPU measured: **True**.

Lossless GPU transfer ablation: decode critical fallback predecessors first, then overlap remaining SplitZip fallback decode with rANS. The reference is unchanged mapped-I/O plus selective-context decoding. Identical original BF16 words, fitted state and frozen TABLES k2/k14 masks; both arms must preserve complete prior bundled packets. No refitting or TEST policy selection. These two documents are reused for development, not independent generalization evidence. Complete GPU calls and actual serialized bytes; serial link calculations are not NIC/SSD, resident P, lossy quality or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.251 | 3.493 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.361 | 3.418 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.312 | 3.412 |
| 512 | overlap_overlap_k14 | paired | 10.75101 | 4.007 | 3.671 |
| 512 | overlap_overlap_k2 | paired | 11.87216 | 1.240 | 1.146 |
| 512 | overlap_reference_k14 | paired | 10.75101 | 4.056 | 3.741 |
| 512 | overlap_reference_k2 | paired | 11.87216 | 1.246 | 1.079 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.733 | 0.309 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.732 | 0.308 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.426 | 4.148 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.944 | 4.526 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.710 | 23.029 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.510 | 23.367 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 185.102 | 23.460 |
| 8192 | overlap_overlap_k14 | paired | 10.63994 | 27.212 | 13.674 |
| 8192 | overlap_overlap_k2 | paired | 11.85702 | 12.380 | 5.036 |
| 8192 | overlap_reference_k14 | paired | 10.63994 | 27.159 | 13.624 |
| 8192 | overlap_reference_k2 | paired | 11.85702 | 12.385 | 5.026 |
| 8192 | raw_bf16 | after | 16.00001 | 4.891 | 3.317 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.318 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.285 | 3.188 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.215 | 3.187 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.859 | 32.272 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.530 | 23.068 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.678 | 3.785 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.944 | 3.632 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.053 | 3.672 |
| 512 | overlap_overlap_k14 | paired | 10.75101 | 4.503 | 4.580 |
| 512 | overlap_overlap_k2 | paired | 11.87216 | 1.666 | 1.618 |
| 512 | overlap_reference_k14 | paired | 10.75101 | 4.556 | 4.465 |
| 512 | overlap_reference_k2 | paired | 11.87216 | 1.705 | 1.567 |
| 512 | raw_bf16 | after | 16.00009 | 0.421 | 0.354 |
| 512 | raw_bf16 | before | 16.00009 | 0.418 | 0.346 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.868 | 0.525 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.871 | 0.523 |
| 512 | team_dense_uniform | paired | 9.11376 | 23.027 | 4.480 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.553 | 4.838 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.837 | 23.563 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 192.206 | 23.610 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 194.889 | 23.726 |
| 8192 | overlap_overlap_k14 | paired | 10.63994 | 26.622 | 13.265 |
| 8192 | overlap_overlap_k2 | paired | 11.85702 | 12.571 | 5.341 |
| 8192 | overlap_reference_k14 | paired | 10.63994 | 26.566 | 13.198 |
| 8192 | overlap_reference_k2 | paired | 11.85702 | 12.571 | 5.317 |
| 8192 | raw_bf16 | after | 16.00001 | 4.977 | 3.423 |
| 8192 | raw_bf16 | before | 16.00001 | 4.975 | 3.430 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.415 | 3.393 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.414 | 3.396 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.472 | 28.941 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 42.028 | 21.606 |

Paired changes relative to the unchanged mapped-I/O and selective-context execution. Speedup below 1 means slower.

| Tokens | k | Change | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---:|---|---|---:|---:|---:|
| 512 | 2 | overlap | batched | 0 | 1.005 | 0.941 |
| 512 | 2 | overlap | isolated | 0 | 1.023 | 0.969 |
| 512 | 14 | overlap | batched | 0 | 1.012 | 1.019 |
| 512 | 14 | overlap | isolated | 0 | 1.012 | 0.975 |
| 8192 | 2 | overlap | batched | 0 | 1.000 | 0.998 |
| 8192 | 2 | overlap | isolated | 0 | 1.000 | 0.995 |
| 8192 | 14 | overlap | batched | 0 | 0.998 | 0.996 |
| 8192 | 14 | overlap | isolated | 0 | 0.998 | 0.995 |

Link summary below uses **8K isolated** timings only. All lengths and both timing bases remain in `results.json`. Baseline controls use the faster complete before/after round for each objective. No TEST policy selection.

Calculated serial precompressed cost, **8K isolated**, ms.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7519.618 | 5664.085 | 5820.567 | 5133.310 | 4219.675 | 4393.363 |
| 5 | 1506.662 | 1135.531 | 1183.002 | 1045.512 | 867.088 | 895.957 |
| 10 | 755.043 | 569.462 | 603.306 | 534.538 | 448.015 | 458.782 |
| 25 | 304.071 | 229.820 | 255.489 | 227.953 | 196.571 | 196.476 |
| 50 | 153.747 | 116.606 | 139.549 | 125.758 | 112.756 | 109.041 |
| 75 | 103.639 | 78.868 | 100.903 | 91.693 | 84.818 | 79.896 |
| 100 | 78.585 | 59.999 | 81.580 | 74.661 | 70.849 | 65.323 |
| 200 | 41.004 | 31.696 | 52.595 | 49.112 | 49.895 | 43.465 |
| 300 | 28.477 | 22.262 | 42.933 | 40.596 | 42.911 | 36.178 |
| 400 | 22.214 | 17.544 | 38.103 | 36.338 | 39.418 | 32.535 |

Fixed k2 execution arms, serial precompressed, **8K isolated**, ms.

| Gb/s | SplitZip | Mapped reference | Critical-first overlap |
|---:|---:|---:|---:|
| 1 | 5664.085 | 5575.296 | 5575.320 |
| 5 | 1135.531 | 1119.312 | 1119.337 |
| 10 | 569.462 | 562.315 | 562.339 |
| 25 | 229.820 | 228.116 | 228.141 |
| 50 | 116.606 | 116.716 | 116.741 |
| 75 | 78.868 | 79.583 | 79.608 |
| 100 | 59.999 | 61.016 | 61.041 |
| 200 | 31.696 | 33.167 | 33.191 |
| 300 | 22.262 | 23.883 | 23.908 |
| 400 | 17.544 | 19.242 | 19.266 |

Fixed k14 execution arms, serial precompressed, **8K isolated**, ms.

| Gb/s | SplitZip | Mapped reference | Critical-first overlap |
|---:|---:|---:|---:|
| 1 | 5664.085 | 5011.440 | 5011.507 |
| 5 | 1135.531 | 1012.847 | 1012.914 |
| 10 | 569.462 | 513.023 | 513.089 |
| 25 | 229.820 | 213.128 | 213.195 |
| 50 | 116.606 | 113.163 | 113.230 |
| 75 | 78.868 | 79.842 | 79.908 |
| 100 | 59.999 | 63.181 | 63.248 |
| 200 | 31.696 | 38.190 | 38.256 |
| 300 | 22.262 | 29.859 | 29.926 |
| 400 | 17.544 | 25.694 | 25.761 |

Calculated serial fresh cost, **8K isolated**, ms.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7524.596 | 5674.500 | 6012.772 | 5343.147 | 4303.147 | 4435.391 |
| 5 | 1511.639 | 1145.946 | 1375.207 | 1255.350 | 950.560 | 937.985 |
| 10 | 760.020 | 579.877 | 795.511 | 744.375 | 531.487 | 500.809 |
| 25 | 309.048 | 240.235 | 447.694 | 437.790 | 280.043 | 238.504 |
| 50 | 158.724 | 127.021 | 331.755 | 335.595 | 196.228 | 151.068 |
| 75 | 108.616 | 89.283 | 293.109 | 301.531 | 168.290 | 121.923 |
| 100 | 83.562 | 70.414 | 273.785 | 284.498 | 154.321 | 107.351 |
| 200 | 45.981 | 42.111 | 244.801 | 258.949 | 133.367 | 85.492 |
| 300 | 33.454 | 32.676 | 235.139 | 250.433 | 126.382 | 78.206 |
| 400 | 27.191 | 27.959 | 230.308 | 246.175 | 122.890 | 74.563 |

Fixed k2 execution arms, serial fresh, **8K isolated**, ms.

| Gb/s | SplitZip | Mapped reference | Critical-first overlap |
|---:|---:|---:|---:|
| 1 | 5674.500 | 5587.866 | 5587.891 |
| 5 | 1145.946 | 1131.883 | 1131.908 |
| 10 | 579.877 | 574.885 | 574.910 |
| 25 | 240.235 | 240.686 | 240.711 |
| 50 | 127.021 | 129.287 | 129.312 |
| 75 | 89.283 | 92.154 | 92.179 |
| 100 | 70.414 | 73.587 | 73.612 |
| 200 | 42.111 | 45.737 | 45.762 |
| 300 | 32.676 | 36.454 | 36.479 |
| 400 | 27.959 | 31.812 | 31.837 |

Fixed k14 execution arms, serial fresh, **8K isolated**, ms.

| Gb/s | SplitZip | Mapped reference | Critical-first overlap |
|---:|---:|---:|---:|
| 1 | 5674.500 | 5038.007 | 5038.129 |
| 5 | 1145.946 | 1039.413 | 1039.536 |
| 10 | 579.877 | 539.589 | 539.711 |
| 25 | 240.235 | 239.695 | 239.817 |
| 50 | 127.021 | 139.730 | 139.852 |
| 75 | 89.283 | 106.408 | 106.530 |
| 100 | 70.414 | 89.747 | 89.870 |
| 200 | 42.111 | 64.756 | 64.878 |
| 300 | 32.676 | 56.426 | 56.548 |
| 400 | 27.959 | 52.261 | 52.383 |

Both arms must preserve the complete prior bundled packet, including identical compact rANS words and directories. Fresh receivers see only compressed packets and state, alternate 0/1/0, retain previous outputs, and verify original-word hashes.

Full GPU E/D calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls have two warmups and five samples. Export, disk, upload, fitting and cold setup are excluded. Complete bytes use actual serialized files. Shared state and native reservations are additional.

Original author sources, baseline environments, predictor state, TABLES masks and all prior experiment source files stay unchanged. Partitioned mapped decoding and its overlap schedule are local integration changes, not the official baseline implementation. New-arm CPU/CUDA profiles are warmed, separate diagnostics and nonadditive; the compact control retains its original diagnostic procedure. Full traces remain local. No measured network, resident P, lossy quality, or serving speed claim.
