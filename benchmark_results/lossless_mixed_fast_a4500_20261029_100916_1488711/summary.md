# Lossless mixed execution comparison

**Status: completed.** GPU measured: **True**.

Lossless GPU transfer execution ablation: identical original BF16 words, original fitted state, and previously frozen TABLES masks k2/k14. No refitting or TEST mask selection. The two TEST documents are reused for engineering iterations, not independent generalization evidence. Complete GPU calls and actual serialized bytes; calculated serial link scenarios, not NIC/SSD, lossy quality, compressed attention, or resident P memory measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.173 | 3.497 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.304 | 3.423 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.409 | 3.420 |
| 512 | fast_both_k14 | paired | 10.75101 | 4.102 | 3.581 |
| 512 | fast_both_k2 | paired | 11.87216 | 1.434 | 1.129 |
| 512 | fast_bundle_k14 | paired | 10.75101 | 5.623 | 3.614 |
| 512 | fast_bundle_k2 | paired | 11.87216 | 4.179 | 1.121 |
| 512 | fast_reference_k14 | paired | 10.75255 | 6.900 | 5.263 |
| 512 | fast_reference_k2 | paired | 11.87517 | 6.096 | 3.632 |
| 512 | fast_selective_k14 | paired | 10.75255 | 5.222 | 5.144 |
| 512 | fast_selective_k2 | paired | 11.87517 | 3.464 | 3.630 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.228 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.733 | 0.311 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.732 | 0.309 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.455 | 4.147 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.903 | 4.524 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.188 | 23.030 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.750 | 23.393 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 185.399 | 23.369 |
| 8192 | fast_both_k14 | paired | 10.63994 | 29.443 | 15.794 |
| 8192 | fast_both_k2 | paired | 11.85702 | 15.294 | 8.782 |
| 8192 | fast_bundle_k14 | paired | 10.63994 | 39.310 | 15.840 |
| 8192 | fast_bundle_k2 | paired | 11.85702 | 31.955 | 8.816 |
| 8192 | fast_reference_k14 | paired | 10.64004 | 37.703 | 16.227 |
| 8192 | fast_reference_k2 | paired | 11.85722 | 30.466 | 10.141 |
| 8192 | fast_selective_k14 | paired | 10.64004 | 27.878 | 16.103 |
| 8192 | fast_selective_k2 | paired | 11.85722 | 14.079 | 10.367 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.317 |
| 8192 | raw_bf16 | before | 16.00001 | 4.889 | 3.317 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.293 | 3.188 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.242 | 3.186 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.771 | 32.193 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.514 | 23.051 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.078 | 3.784 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.655 | 3.677 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.401 | 3.662 |
| 512 | fast_both_k14 | paired | 10.75101 | 4.595 | 4.390 |
| 512 | fast_both_k2 | paired | 11.87216 | 1.930 | 1.621 |
| 512 | fast_bundle_k14 | paired | 10.75101 | 6.189 | 4.342 |
| 512 | fast_bundle_k2 | paired | 11.87216 | 4.851 | 1.690 |
| 512 | fast_reference_k14 | paired | 10.75255 | 7.626 | 6.594 |
| 512 | fast_reference_k2 | paired | 11.87517 | 7.035 | 4.286 |
| 512 | fast_selective_k14 | paired | 10.75255 | 5.846 | 5.962 |
| 512 | fast_selective_k2 | paired | 11.87517 | 4.213 | 4.399 |
| 512 | raw_bf16 | after | 16.00009 | 0.416 | 0.349 |
| 512 | raw_bf16 | before | 16.00009 | 0.411 | 0.333 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.876 | 0.535 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.871 | 0.533 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.938 | 4.278 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.505 | 4.741 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 210.695 | 23.582 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 193.913 | 23.639 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 193.260 | 23.631 |
| 8192 | fast_both_k14 | paired | 10.63994 | 28.871 | 15.228 |
| 8192 | fast_both_k2 | paired | 11.85702 | 15.547 | 8.871 |
| 8192 | fast_bundle_k14 | paired | 10.63994 | 38.480 | 15.213 |
| 8192 | fast_bundle_k2 | paired | 11.85702 | 31.775 | 8.867 |
| 8192 | fast_reference_k14 | paired | 10.64004 | 37.202 | 16.270 |
| 8192 | fast_reference_k2 | paired | 11.85722 | 30.527 | 10.446 |
| 8192 | fast_selective_k14 | paired | 10.64004 | 27.423 | 16.017 |
| 8192 | fast_selective_k2 | paired | 11.85722 | 14.425 | 10.864 |
| 8192 | raw_bf16 | after | 16.00001 | 4.982 | 3.425 |
| 8192 | raw_bf16 | before | 16.00001 | 4.976 | 3.424 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.339 | 3.383 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.407 | 3.402 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.140 | 28.571 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 41.945 | 21.551 |

Paired changes relative to the unchanged mixed execution. Speedup below 1 means slower.

| Tokens | k | Change | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---:|---|---|---:|---:|---:|
| 512 | 2 | selective | batched | 0 | 1.760 | 1.001 |
| 512 | 2 | selective | isolated | 0 | 1.670 | 0.974 |
| 512 | 2 | bundle | batched | -11047 | 1.458 | 3.240 |
| 512 | 2 | bundle | isolated | -11047 | 1.450 | 2.536 |
| 512 | 2 | both | batched | -11047 | 4.252 | 3.218 |
| 512 | 2 | both | isolated | -11047 | 3.645 | 2.643 |
| 512 | 14 | selective | batched | 0 | 1.321 | 1.023 |
| 512 | 14 | selective | isolated | 0 | 1.305 | 1.106 |
| 512 | 14 | bundle | batched | -5652 | 1.227 | 1.456 |
| 512 | 14 | bundle | isolated | -5652 | 1.232 | 1.519 |
| 512 | 14 | both | batched | -5652 | 1.682 | 1.470 |
| 512 | 14 | both | isolated | -5652 | 1.660 | 1.502 |
| 8192 | 2 | selective | batched | 0 | 2.164 | 0.978 |
| 8192 | 2 | selective | isolated | 0 | 2.116 | 0.962 |
| 8192 | 2 | bundle | batched | -11448 | 0.953 | 1.150 |
| 8192 | 2 | bundle | isolated | -11448 | 0.961 | 1.178 |
| 8192 | 2 | both | batched | -11448 | 1.992 | 1.155 |
| 8192 | 2 | both | isolated | -11448 | 1.964 | 1.178 |
| 8192 | 14 | selective | batched | 0 | 1.352 | 1.008 |
| 8192 | 14 | selective | isolated | 0 | 1.357 | 1.016 |
| 8192 | 14 | bundle | batched | -5864 | 0.959 | 1.024 |
| 8192 | 14 | bundle | isolated | -5864 | 0.967 | 1.069 |
| 8192 | 14 | both | batched | -5864 | 1.281 | 1.027 |
| 8192 | 14 | both | isolated | -5864 | 1.289 | 1.068 |

Link summary below uses **8K isolated** timings only. All lengths and both timing bases remain in `results.json`. Baseline controls use the faster complete before/after round for each objective. No TEST policy selection.

Calculated serial precompressed cost, **8K isolated**, ms.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7519.620 | 5664.076 | 5820.587 | 5133.329 | 4219.304 | 4393.309 |
| 5 | 1506.663 | 1135.522 | 1183.022 | 1045.532 | 866.717 | 895.903 |
| 10 | 755.044 | 569.453 | 603.326 | 534.557 | 447.644 | 458.727 |
| 25 | 304.072 | 229.811 | 255.509 | 227.972 | 196.200 | 196.422 |
| 50 | 153.748 | 116.597 | 139.570 | 125.777 | 112.385 | 108.986 |
| 75 | 103.640 | 78.859 | 100.923 | 91.712 | 84.447 | 79.841 |
| 100 | 78.586 | 59.990 | 81.600 | 74.680 | 70.478 | 65.269 |
| 200 | 41.005 | 31.687 | 52.615 | 49.131 | 49.524 | 43.410 |
| 300 | 28.478 | 22.252 | 42.954 | 40.615 | 42.540 | 36.124 |
| 400 | 22.215 | 17.535 | 38.123 | 36.357 | 39.047 | 32.481 |

Fixed k2 execution arms, serial precompressed, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Selective | Bundle | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.076 | 5580.517 | 5580.935 | 5578.846 | 5578.850 |
| 5 | 1135.522 | 1124.460 | 1124.878 | 1122.863 | 1122.867 |
| 10 | 569.453 | 567.453 | 567.871 | 565.865 | 565.869 |
| 25 | 229.811 | 233.249 | 233.667 | 231.666 | 231.670 |
| 50 | 116.597 | 121.847 | 122.265 | 120.266 | 120.270 |
| 75 | 78.859 | 84.714 | 85.132 | 83.133 | 83.137 |
| 100 | 59.990 | 66.147 | 66.565 | 64.567 | 64.571 |
| 200 | 31.687 | 38.296 | 38.714 | 36.717 | 36.721 |
| 300 | 22.252 | 29.013 | 29.431 | 27.433 | 27.437 |
| 400 | 17.535 | 24.371 | 24.789 | 22.792 | 22.796 |

Fixed k14 execution arms, serial precompressed, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Selective | Bundle | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.076 | 5014.559 | 5014.306 | 5013.455 | 5013.470 |
| 5 | 1135.522 | 1015.928 | 1015.675 | 1014.861 | 1014.877 |
| 10 | 569.453 | 516.099 | 515.846 | 515.037 | 515.052 |
| 25 | 229.811 | 216.202 | 215.949 | 215.143 | 215.158 |
| 50 | 116.597 | 116.236 | 115.983 | 115.178 | 115.193 |
| 75 | 78.859 | 82.914 | 82.661 | 81.856 | 81.871 |
| 100 | 59.990 | 66.253 | 66.000 | 65.195 | 65.211 |
| 200 | 31.687 | 41.262 | 41.009 | 40.204 | 40.219 |
| 300 | 22.252 | 32.931 | 32.678 | 31.874 | 31.889 |
| 400 | 17.535 | 28.766 | 28.513 | 27.709 | 27.724 |

Calculated serial fresh cost, **8K isolated**, ms.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7524.595 | 5674.415 | 6013.847 | 5344.024 | 4302.444 | 4435.254 |
| 5 | 1511.639 | 1145.861 | 1376.281 | 1256.227 | 949.857 | 937.848 |
| 10 | 760.019 | 579.792 | 796.586 | 745.252 | 530.784 | 500.672 |
| 25 | 309.048 | 240.150 | 448.768 | 438.668 | 279.340 | 238.366 |
| 50 | 158.724 | 126.937 | 332.829 | 336.473 | 195.525 | 150.931 |
| 75 | 108.616 | 89.199 | 294.183 | 302.408 | 167.587 | 121.786 |
| 100 | 83.562 | 70.330 | 274.860 | 285.375 | 153.618 | 107.214 |
| 200 | 45.981 | 42.026 | 245.875 | 259.826 | 132.664 | 85.355 |
| 300 | 33.454 | 32.592 | 236.213 | 251.310 | 125.679 | 78.069 |
| 400 | 27.190 | 27.874 | 231.383 | 247.052 | 122.187 | 74.425 |

Fixed k2 execution arms, serial fresh, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Selective | Bundle | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.415 | 5611.044 | 5595.360 | 5610.621 | 5594.397 |
| 5 | 1145.861 | 1154.987 | 1139.303 | 1154.637 | 1138.414 |
| 10 | 579.792 | 597.980 | 582.296 | 597.639 | 581.416 |
| 25 | 240.150 | 263.776 | 248.092 | 263.441 | 247.217 |
| 50 | 126.937 | 152.374 | 136.691 | 152.041 | 135.818 |
| 75 | 89.199 | 115.241 | 99.557 | 114.908 | 98.684 |
| 100 | 70.330 | 96.674 | 80.990 | 96.341 | 80.118 |
| 200 | 42.026 | 68.823 | 53.140 | 68.491 | 52.268 |
| 300 | 32.592 | 59.540 | 43.856 | 59.208 | 42.985 |
| 400 | 27.874 | 54.898 | 39.214 | 54.566 | 38.343 |

Fixed k14 execution arms, serial fresh, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Selective | Bundle | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.415 | 5051.761 | 5041.730 | 5051.935 | 5042.341 |
| 5 | 1145.861 | 1053.130 | 1043.098 | 1053.341 | 1043.748 |
| 10 | 579.792 | 553.301 | 543.269 | 553.517 | 543.923 |
| 25 | 240.150 | 253.403 | 243.372 | 253.623 | 244.029 |
| 50 | 126.937 | 153.438 | 143.406 | 153.658 | 144.064 |
| 75 | 89.199 | 120.116 | 110.084 | 120.336 | 110.742 |
| 100 | 70.330 | 103.455 | 93.423 | 103.675 | 94.082 |
| 200 | 42.026 | 78.463 | 68.432 | 78.684 | 69.090 |
| 300 | 32.592 | 70.133 | 60.102 | 70.354 | 60.760 |
| 400 | 27.874 | 65.967 | 55.936 | 66.189 | 56.595 |

Selective preparation must preserve complete reference wire bytes; the combined change must preserve bundled wire bytes. All four arms must have identical compact rANS words and directories. Fresh receivers see only compressed packets and state, alternate 0/1/0, retain previous outputs, and verify original-word hashes.

Full GPU E/D calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls have two warmups and five samples. Export, disk, upload, fitting and cold setup are excluded. Complete bytes use actual serialized files. Shared state and native reservations are additional.

Original author sources, baseline environments, predictor state, TABLES masks and prior experiment source files stay unchanged. Bundling is a local integration arm using unchanged author SplitZip kernels in the main environment. New fast-arm CPU/CUDA profiles are warmed, separate diagnostics and nonadditive; the compact control retains its original diagnostic procedure. Full traces remain local. No measured network, resident P, lossy quality, or serving speed claim.
