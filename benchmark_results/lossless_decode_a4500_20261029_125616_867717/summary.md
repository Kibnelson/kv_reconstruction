# Lossless mapped-I/O and decoder context comparison

**Status: completed.** GPU measured: **True**.

Lossless GPU transfer ablation: mapped SplitZip source/destination access and selective decoder context. Identical original BF16 words, original fitted state and frozen TABLES k2/k14 masks; all four arms must preserve complete prior bundled packets. No refitting or TEST policy selection. These two documents are reused for development, not independent generalization evidence. Complete GPU calls and actual serialized bytes; serial link calculations are not NIC/SSD, resident P, lossy quality or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | decode_both_k14 | paired | 10.75101 | 4.026 | 3.661 |
| 512 | decode_both_k2 | paired | 11.87216 | 1.251 | 1.081 |
| 512 | decode_context_k14 | paired | 10.75101 | 4.154 | 3.672 |
| 512 | decode_context_k2 | paired | 11.87216 | 1.414 | 1.106 |
| 512 | decode_mapped_k14 | paired | 10.75101 | 4.092 | 3.802 |
| 512 | decode_mapped_k2 | paired | 11.87216 | 1.246 | 1.074 |
| 512 | decode_reference_k14 | paired | 10.75101 | 4.133 | 3.660 |
| 512 | decode_reference_k2 | paired | 11.87216 | 1.412 | 1.121 |
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.142 | 3.498 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.353 | 3.416 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.335 | 3.424 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.733 | 0.308 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.734 | 0.309 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.482 | 4.138 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.928 | 4.528 |
| 8192 | decode_both_k14 | paired | 10.63994 | 27.151 | 13.618 |
| 8192 | decode_both_k2 | paired | 11.85702 | 12.404 | 5.041 |
| 8192 | decode_context_k14 | paired | 10.63994 | 29.384 | 15.630 |
| 8192 | decode_context_k2 | paired | 11.85702 | 15.276 | 8.696 |
| 8192 | decode_mapped_k14 | paired | 10.63994 | 27.192 | 13.780 |
| 8192 | decode_mapped_k2 | paired | 11.85702 | 12.376 | 5.071 |
| 8192 | decode_reference_k14 | paired | 10.63994 | 29.447 | 15.790 |
| 8192 | decode_reference_k2 | paired | 11.85702 | 15.263 | 8.771 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.152 | 23.033 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.326 | 23.351 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 184.853 | 23.357 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.317 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.317 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.282 | 3.187 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.210 | 3.186 |
| 8192 | team_dense_uniform | paired | 8.92097 | 84.277 | 32.322 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.541 | 23.087 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | decode_both_k14 | paired | 10.75101 | 4.532 | 4.382 |
| 512 | decode_both_k2 | paired | 11.87216 | 1.692 | 1.562 |
| 512 | decode_context_k14 | paired | 10.75101 | 4.608 | 4.422 |
| 512 | decode_context_k2 | paired | 11.87216 | 1.856 | 1.574 |
| 512 | decode_mapped_k14 | paired | 10.75101 | 4.744 | 4.541 |
| 512 | decode_mapped_k2 | paired | 11.87216 | 1.686 | 1.543 |
| 512 | decode_reference_k14 | paired | 10.75101 | 4.594 | 4.408 |
| 512 | decode_reference_k2 | paired | 11.87216 | 1.837 | 1.570 |
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.238 | 3.765 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.464 | 3.671 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 15.187 | 3.684 |
| 512 | raw_bf16 | after | 16.00009 | 0.412 | 0.338 |
| 512 | raw_bf16 | before | 16.00009 | 0.413 | 0.340 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.875 | 0.531 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.876 | 0.547 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.955 | 4.317 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.559 | 5.006 |
| 8192 | decode_both_k14 | paired | 10.63994 | 26.549 | 13.228 |
| 8192 | decode_both_k2 | paired | 11.85702 | 12.583 | 5.321 |
| 8192 | decode_context_k14 | paired | 10.63994 | 28.811 | 15.098 |
| 8192 | decode_context_k2 | paired | 11.85702 | 15.524 | 8.758 |
| 8192 | decode_mapped_k14 | paired | 10.63994 | 26.583 | 13.292 |
| 8192 | decode_mapped_k2 | paired | 11.85702 | 12.566 | 5.330 |
| 8192 | decode_reference_k14 | paired | 10.63994 | 28.836 | 15.254 |
| 8192 | decode_reference_k2 | paired | 11.85702 | 15.541 | 8.888 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 210.076 | 23.555 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 194.770 | 23.592 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 194.723 | 23.635 |
| 8192 | raw_bf16 | after | 16.00001 | 4.982 | 3.423 |
| 8192 | raw_bf16 | before | 16.00001 | 4.978 | 3.441 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.407 | 3.378 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.288 | 3.368 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.788 | 28.899 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 42.007 | 21.597 |

Paired changes relative to the unchanged mixed execution. Speedup below 1 means slower.

| Tokens | k | Change | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---:|---|---|---:|---:|---:|
| 512 | 2 | mapped | batched | 0 | 1.133 | 1.043 |
| 512 | 2 | mapped | isolated | 0 | 1.089 | 1.017 |
| 512 | 2 | context | batched | 0 | 0.999 | 1.013 |
| 512 | 2 | context | isolated | 0 | 0.989 | 0.997 |
| 512 | 2 | both | batched | 0 | 1.128 | 1.037 |
| 512 | 2 | both | isolated | 0 | 1.085 | 1.005 |
| 512 | 14 | mapped | batched | 0 | 1.010 | 0.963 |
| 512 | 14 | mapped | isolated | 0 | 0.968 | 0.971 |
| 512 | 14 | context | batched | 0 | 0.995 | 0.997 |
| 512 | 14 | context | isolated | 0 | 0.997 | 0.997 |
| 512 | 14 | both | batched | 0 | 1.027 | 1.000 |
| 512 | 14 | both | isolated | 0 | 1.014 | 1.006 |
| 8192 | 2 | mapped | batched | 0 | 1.233 | 1.730 |
| 8192 | 2 | mapped | isolated | 0 | 1.237 | 1.668 |
| 8192 | 2 | context | batched | 0 | 0.999 | 1.009 |
| 8192 | 2 | context | isolated | 0 | 1.001 | 1.015 |
| 8192 | 2 | both | batched | 0 | 1.230 | 1.740 |
| 8192 | 2 | both | isolated | 0 | 1.235 | 1.670 |
| 8192 | 14 | mapped | batched | 0 | 1.083 | 1.146 |
| 8192 | 14 | mapped | isolated | 0 | 1.085 | 1.148 |
| 8192 | 14 | context | batched | 0 | 1.002 | 1.010 |
| 8192 | 14 | context | isolated | 0 | 1.001 | 1.010 |
| 8192 | 14 | both | batched | 0 | 1.085 | 1.159 |
| 8192 | 14 | both | isolated | 0 | 1.086 | 1.153 |

Link summary below uses **8K isolated** timings only. All lengths and both timing bases remain in `results.json`. Baseline controls use the faster complete before/after round for each objective. No TEST policy selection.

Calculated serial precompressed cost, **8K isolated**, ms.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7519.619 | 5664.060 | 5820.548 | 5133.301 | 4219.633 | 4393.354 |
| 5 | 1506.662 | 1135.506 | 1182.983 | 1045.504 | 867.046 | 895.949 |
| 10 | 755.043 | 569.437 | 603.287 | 534.529 | 447.973 | 458.773 |
| 25 | 304.071 | 229.795 | 255.470 | 227.944 | 196.529 | 196.467 |
| 50 | 153.747 | 116.582 | 139.531 | 125.749 | 112.714 | 109.032 |
| 75 | 103.639 | 78.844 | 100.884 | 91.685 | 84.776 | 79.887 |
| 100 | 78.585 | 59.975 | 81.561 | 74.652 | 70.807 | 65.315 |
| 200 | 41.004 | 31.671 | 52.576 | 49.103 | 49.853 | 43.456 |
| 300 | 28.477 | 22.237 | 42.915 | 40.587 | 42.868 | 36.170 |
| 400 | 22.214 | 17.519 | 38.084 | 36.329 | 39.376 | 32.526 |

Fixed k2 execution arms, serial precompressed, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Mapped I/O | Context | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.060 | 5578.867 | 5575.309 | 5578.737 | 5575.301 |
| 5 | 1135.506 | 1122.884 | 1119.326 | 1122.754 | 1119.317 |
| 10 | 569.437 | 565.886 | 562.328 | 565.756 | 562.319 |
| 25 | 229.795 | 231.688 | 228.129 | 231.557 | 228.121 |
| 50 | 116.582 | 120.288 | 116.730 | 120.158 | 116.721 |
| 75 | 78.844 | 83.155 | 79.596 | 83.024 | 79.588 |
| 100 | 59.975 | 64.588 | 61.030 | 64.458 | 61.021 |
| 200 | 31.671 | 36.738 | 33.180 | 36.608 | 33.171 |
| 300 | 22.237 | 27.455 | 23.897 | 27.325 | 23.888 |
| 400 | 17.519 | 22.813 | 19.255 | 22.683 | 19.246 |

Fixed k14 execution arms, serial precompressed, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Mapped I/O | Context | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.060 | 5013.496 | 5011.534 | 5013.340 | 5011.470 |
| 5 | 1135.506 | 1014.903 | 1012.941 | 1014.747 | 1012.876 |
| 10 | 569.437 | 515.079 | 513.116 | 514.922 | 513.052 |
| 25 | 229.795 | 215.184 | 213.222 | 215.028 | 213.157 |
| 50 | 116.582 | 115.219 | 113.257 | 115.063 | 113.192 |
| 75 | 78.844 | 81.898 | 79.935 | 81.741 | 79.871 |
| 100 | 59.975 | 65.237 | 63.275 | 65.081 | 63.210 |
| 200 | 31.671 | 40.246 | 38.283 | 40.089 | 38.219 |
| 300 | 22.237 | 31.915 | 29.953 | 31.759 | 29.888 |
| 400 | 17.519 | 27.750 | 25.788 | 27.594 | 25.723 |

Calculated serial fresh cost, **8K isolated**, ms.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense | Rank256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7524.600 | 5674.348 | 6015.314 | 5343.377 | 4303.421 | 4435.361 |
| 5 | 1511.644 | 1145.794 | 1377.749 | 1255.580 | 950.834 | 937.956 |
| 10 | 760.025 | 579.725 | 798.053 | 744.605 | 531.761 | 500.780 |
| 25 | 309.053 | 240.083 | 450.236 | 438.021 | 280.317 | 238.474 |
| 50 | 158.729 | 126.870 | 334.297 | 335.826 | 196.502 | 151.039 |
| 75 | 108.621 | 89.132 | 295.651 | 301.761 | 168.564 | 121.894 |
| 100 | 83.567 | 70.263 | 276.327 | 284.728 | 154.595 | 107.322 |
| 200 | 45.986 | 41.959 | 247.343 | 259.180 | 133.641 | 85.463 |
| 300 | 33.459 | 32.525 | 237.681 | 250.663 | 126.657 | 78.177 |
| 400 | 27.196 | 27.808 | 232.850 | 246.405 | 123.164 | 74.533 |

Fixed k2 execution arms, serial fresh, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Mapped I/O | Context | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.348 | 5594.408 | 5587.875 | 5594.261 | 5587.883 |
| 5 | 1145.794 | 1138.425 | 1131.892 | 1138.278 | 1131.900 |
| 10 | 579.725 | 581.427 | 574.894 | 581.280 | 574.902 |
| 25 | 240.083 | 247.228 | 240.695 | 247.081 | 240.703 |
| 50 | 126.870 | 135.829 | 129.296 | 135.682 | 129.304 |
| 75 | 89.132 | 98.695 | 92.162 | 98.548 | 92.170 |
| 100 | 70.263 | 80.129 | 73.596 | 79.982 | 73.604 |
| 200 | 41.959 | 52.279 | 45.746 | 52.132 | 45.754 |
| 300 | 32.525 | 42.996 | 36.463 | 42.849 | 36.471 |
| 400 | 27.808 | 38.354 | 31.821 | 38.207 | 31.829 |

Fixed k14 execution arms, serial fresh, **8K isolated**, ms.

| Gb/s | SplitZip | Reference | Mapped I/O | Context | Both |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.348 | 5042.332 | 5038.117 | 5042.151 | 5038.018 |
| 5 | 1145.794 | 1043.739 | 1039.524 | 1043.558 | 1039.425 |
| 10 | 579.725 | 543.915 | 539.699 | 543.734 | 539.601 |
| 25 | 240.083 | 244.020 | 239.805 | 243.839 | 239.706 |
| 50 | 126.870 | 144.055 | 139.840 | 143.874 | 139.741 |
| 75 | 89.132 | 110.734 | 106.518 | 110.553 | 106.420 |
| 100 | 70.263 | 94.073 | 89.858 | 93.892 | 89.759 |
| 200 | 41.959 | 69.082 | 64.866 | 68.901 | 64.768 |
| 300 | 32.525 | 60.751 | 56.536 | 60.570 | 56.437 |
| 400 | 27.808 | 56.586 | 52.371 | 56.405 | 52.272 |

All four arms must preserve the complete prior bundled packet, including identical compact rANS words and directories. Fresh receivers see only compressed packets and state, alternate 0/1/0, retain previous outputs, and verify original-word hashes.

Full GPU E/D calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls have two warmups and five samples. Export, disk, upload, fitting and cold setup are excluded. Complete bytes use actual serialized files. Shared state and native reservations are additional.

Original author sources, baseline environments, predictor state, TABLES masks and all prior experiment source files stay unchanged. Mapped I/O and selective decoding are new local integration kernels, not the official baseline implementation. New-arm CPU/CUDA profiles are warmed, separate diagnostics and nonadditive; the compact control retains its original diagnostic procedure. Full traces remain local. No measured network, resident P, lossy quality, or serving speed claim.
