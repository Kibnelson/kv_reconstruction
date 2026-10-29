# Standalone lossless decoder products and cached encoder model

**Status: completed.** GPU measured: **True**.

Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. Candidates use exact INT8 decoder products or reuse the Dense encoder model; complete original packets, fitted coefficients, prediction results, fixed binary-CDF inverse and BF16 words stay fixed. SplitZip and ExANS are separate unchanged controls, not candidate components. No full inverse table is installed. Existing compact-search graph execution at 512 tokens and direct-search execution at 8192 tokens are retained. These two documents are reused engineering evidence, not independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | compute_cached_model_dense_uniform | paired | 9.11376 | 10.369 | 4.134 |
| 512 | compute_gemm128_dense_uniform | paired | 9.11376 | 22.531 | 4.137 |
| 512 | compute_gemm128_rank256_uniform | paired | 9.50504 | 7.121 | 4.594 |
| 512 | compute_gemm64_dense_uniform | paired | 9.11376 | 24.139 | 4.187 |
| 512 | compute_gemm64_rank256_uniform | paired | 9.50504 | 6.987 | 4.728 |
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.166 | 3.500 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.298 | 3.419 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.387 | 3.414 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.732 | 0.309 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.732 | 0.307 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.341 | 4.154 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.932 | 4.542 |
| 8192 | compute_cached_model_dense_uniform | paired | 8.92097 | 76.648 | 32.169 |
| 8192 | compute_gemm128_dense_uniform | paired | 8.92097 | 85.225 | 32.856 |
| 8192 | compute_gemm128_rank256_uniform | paired | 9.30632 | 43.562 | 22.983 |
| 8192 | compute_gemm64_dense_uniform | paired | 8.92097 | 83.759 | 34.529 |
| 8192 | compute_gemm64_rank256_uniform | paired | 9.30632 | 43.556 | 22.990 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.399 | 23.015 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.913 | 23.360 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 185.308 | 23.362 |
| 8192 | raw_bf16 | after | 16.00001 | 4.891 | 3.317 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.319 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.297 | 3.188 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.229 | 3.186 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.736 | 32.185 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.540 | 23.086 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | compute_cached_model_dense_uniform | paired | 9.11376 | 11.451 | 4.154 |
| 512 | compute_gemm128_dense_uniform | paired | 9.11376 | 22.957 | 4.575 |
| 512 | compute_gemm128_rank256_uniform | paired | 9.50504 | 7.687 | 4.532 |
| 512 | compute_gemm64_dense_uniform | paired | 9.11376 | 24.672 | 4.464 |
| 512 | compute_gemm64_rank256_uniform | paired | 9.50504 | 7.616 | 5.295 |
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.170 | 3.772 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.303 | 3.671 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.342 | 3.633 |
| 512 | raw_bf16 | after | 16.00009 | 0.414 | 0.336 |
| 512 | raw_bf16 | before | 16.00009 | 0.416 | 0.345 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.870 | 0.527 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.865 | 0.534 |
| 512 | team_dense_uniform | paired | 9.11376 | 23.001 | 4.329 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.478 | 4.708 |
| 8192 | compute_cached_model_dense_uniform | paired | 8.92097 | 72.346 | 28.981 |
| 8192 | compute_gemm128_dense_uniform | paired | 8.92097 | 85.101 | 28.319 |
| 8192 | compute_gemm128_rank256_uniform | paired | 9.30632 | 41.997 | 21.239 |
| 8192 | compute_gemm64_dense_uniform | paired | 8.92097 | 83.356 | 30.236 |
| 8192 | compute_gemm64_rank256_uniform | paired | 9.30632 | 41.978 | 21.105 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.858 | 23.556 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 193.523 | 23.600 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 194.234 | 23.631 |
| 8192 | raw_bf16 | after | 16.00001 | 4.974 | 3.424 |
| 8192 | raw_bf16 | before | 16.00001 | 4.977 | 3.425 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.418 | 3.383 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.216 | 3.385 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.478 | 28.630 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 41.959 | 21.578 |

Each fixed compute arm versus the unchanged binary-search standalone reference. Speedup below 1 means slower.

| Tokens | Mode | Predictor | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---|---|---|---:|---:|---:|
| 512 | gemm128 | dense_uniform | batched | 0 | 0.992 | 1.004 |
| 512 | gemm128 | dense_uniform | isolated | 0 | 1.002 | 0.946 |
| 512 | gemm128 | rank256_uniform | batched | 0 | 0.973 | 0.989 |
| 512 | gemm128 | rank256_uniform | isolated | 0 | 0.973 | 1.039 |
| 512 | gemm64 | dense_uniform | batched | 0 | 0.925 | 0.992 |
| 512 | gemm64 | dense_uniform | isolated | 0 | 0.932 | 0.970 |
| 512 | gemm64 | rank256_uniform | batched | 0 | 0.992 | 0.961 |
| 512 | gemm64 | rank256_uniform | isolated | 0 | 0.982 | 0.889 |
| 512 | cached_model | dense_uniform | batched | 0 | 2.155 | 1.005 |
| 512 | cached_model | dense_uniform | isolated | 0 | 2.009 | 1.042 |
| 8192 | gemm128 | dense_uniform | batched | 0 | 0.983 | 0.980 |
| 8192 | gemm128 | dense_uniform | isolated | 0 | 0.981 | 1.011 |
| 8192 | gemm128 | rank256_uniform | batched | 0 | 1.000 | 1.004 |
| 8192 | gemm128 | rank256_uniform | isolated | 0 | 0.999 | 1.016 |
| 8192 | gemm64 | dense_uniform | batched | 0 | 1.000 | 0.932 |
| 8192 | gemm64 | dense_uniform | isolated | 0 | 1.001 | 0.947 |
| 8192 | gemm64 | rank256_uniform | batched | 0 | 1.000 | 1.004 |
| 8192 | gemm64 | rank256_uniform | isolated | 0 | 1.000 | 1.022 |
| 8192 | cached_model | dense_uniform | batched | 0 | 1.092 | 1.001 |
| 8192 | cached_model | dense_uniform | isolated | 0 | 1.154 | 0.988 |

The link tables use **8K isolated** API timing. Both lengths and timing bases remain in results.json. Baseline cells use the faster complete before/after round independently for each objective. All compute arms are fixed; no TEST-selected deployment policy or combined optimization is produced.

Calculated serial **precompressed**, milliseconds.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Rank old |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7519.620 | 5664.075 | 5820.557 | 5133.302 | 4219.363 | 4393.335 |
| 5 | 1506.663 | 1135.521 | 1182.992 | 1045.505 | 866.776 | 895.929 |
| 10 | 755.044 | 569.452 | 603.296 | 534.530 | 447.703 | 458.753 |
| 25 | 304.072 | 229.810 | 255.478 | 227.946 | 196.259 | 196.448 |
| 50 | 153.748 | 116.597 | 139.539 | 125.751 | 112.444 | 109.013 |
| 75 | 103.640 | 78.859 | 100.893 | 91.686 | 84.506 | 79.868 |
| 100 | 78.586 | 59.990 | 81.570 | 74.653 | 70.537 | 65.295 |
| 200 | 41.005 | 31.686 | 52.585 | 49.105 | 49.583 | 43.436 |
| 300 | 28.478 | 22.252 | 42.923 | 40.588 | 42.599 | 36.150 |
| 400 | 22.215 | 17.534 | 38.093 | 36.330 | 39.107 | 32.507 |

Fixed dense_uniform arms, serial **precompressed**, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Reference | gemm128 | gemm64 | cached_model |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5664.075 | 5133.302 | 4219.363 | 4219.053 | 4220.970 | 4219.714 |
| 5 | 1135.521 | 1045.505 | 866.776 | 866.466 | 868.383 | 867.128 |
| 10 | 569.452 | 534.530 | 447.703 | 447.393 | 449.310 | 448.054 |
| 25 | 229.810 | 227.946 | 196.259 | 195.949 | 197.866 | 196.610 |
| 50 | 116.597 | 125.751 | 112.444 | 112.134 | 114.051 | 112.796 |
| 75 | 78.859 | 91.686 | 84.506 | 84.196 | 86.113 | 84.857 |
| 100 | 59.990 | 74.653 | 70.537 | 70.227 | 72.144 | 70.888 |
| 200 | 31.686 | 49.105 | 49.583 | 49.273 | 51.190 | 49.935 |
| 300 | 22.252 | 40.588 | 42.599 | 42.289 | 44.206 | 42.950 |
| 400 | 17.534 | 36.330 | 39.107 | 38.796 | 40.713 | 39.458 |

Fixed rank256_uniform arms, serial **precompressed**, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Reference | gemm128 | gemm64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.075 | 5133.302 | 4393.335 | 4392.997 | 4392.862 |
| 5 | 1135.521 | 1045.505 | 895.929 | 895.591 | 895.456 |
| 10 | 569.452 | 534.530 | 458.753 | 458.415 | 458.280 |
| 25 | 229.810 | 227.946 | 196.448 | 196.110 | 195.975 |
| 50 | 116.597 | 125.751 | 109.013 | 108.674 | 108.540 |
| 75 | 78.859 | 91.686 | 79.868 | 79.529 | 79.395 |
| 100 | 59.990 | 74.653 | 65.295 | 64.957 | 64.822 |
| 200 | 31.686 | 49.105 | 43.436 | 43.098 | 42.963 |
| 300 | 22.252 | 40.588 | 36.150 | 35.812 | 35.677 |
| 400 | 17.534 | 36.330 | 32.507 | 32.169 | 32.034 |

Calculated serial **fresh**, milliseconds.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Rank old |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7524.593 | 5674.293 | 6014.080 | 5343.160 | 4302.841 | 4435.294 |
| 5 | 1511.637 | 1145.739 | 1376.514 | 1255.363 | 950.254 | 937.888 |
| 10 | 760.017 | 579.670 | 796.819 | 744.389 | 531.181 | 500.712 |
| 25 | 309.046 | 240.028 | 449.001 | 437.804 | 279.737 | 238.407 |
| 50 | 158.722 | 126.815 | 333.062 | 335.609 | 195.922 | 150.972 |
| 75 | 108.614 | 89.077 | 294.416 | 301.544 | 167.984 | 121.827 |
| 100 | 83.560 | 70.208 | 275.093 | 284.511 | 154.015 | 107.254 |
| 200 | 45.979 | 41.904 | 246.108 | 258.963 | 133.061 | 85.395 |
| 300 | 33.452 | 32.470 | 236.446 | 250.446 | 126.077 | 78.109 |
| 400 | 27.188 | 27.753 | 231.615 | 246.188 | 122.585 | 74.466 |

Fixed dense_uniform arms, serial **fresh**, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Reference | gemm128 | gemm64 | cached_model |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5674.293 | 5343.160 | 4302.841 | 4304.154 | 4304.326 | 4292.060 |
| 5 | 1145.739 | 1255.363 | 950.254 | 951.567 | 951.739 | 939.473 |
| 10 | 579.670 | 744.389 | 531.181 | 532.494 | 532.666 | 520.400 |
| 25 | 240.028 | 437.804 | 279.737 | 281.050 | 281.222 | 268.956 |
| 50 | 126.815 | 335.609 | 195.922 | 197.235 | 197.407 | 185.141 |
| 75 | 89.077 | 301.544 | 167.984 | 169.297 | 169.469 | 157.203 |
| 100 | 70.208 | 284.511 | 154.015 | 155.328 | 155.500 | 143.234 |
| 200 | 41.904 | 258.963 | 133.061 | 134.374 | 134.546 | 122.280 |
| 300 | 32.470 | 250.446 | 126.077 | 127.390 | 127.561 | 115.296 |
| 400 | 27.753 | 246.188 | 122.585 | 123.898 | 124.069 | 111.804 |

Fixed rank256_uniform arms, serial **fresh**, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Reference | gemm128 | gemm64 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.293 | 5343.160 | 4435.294 | 4434.994 | 4434.841 |
| 5 | 1145.739 | 1255.363 | 937.888 | 937.588 | 937.435 |
| 10 | 579.670 | 744.389 | 500.712 | 500.412 | 500.259 |
| 25 | 240.028 | 437.804 | 238.407 | 238.107 | 237.953 |
| 50 | 126.815 | 335.609 | 150.972 | 150.672 | 150.518 |
| 75 | 89.077 | 301.544 | 121.827 | 121.527 | 121.373 |
| 100 | 70.208 | 284.511 | 107.254 | 106.954 | 106.801 |
| 200 | 41.904 | 258.963 | 85.395 | 85.095 | 84.942 |
| 300 | 32.470 | 250.446 | 78.109 | 77.809 | 77.656 |
| 400 | 27.753 | 246.188 | 74.466 | 74.166 | 74.012 |

Every candidate uses our predictive/rANS coder for all 28 layers, with the original binary-CDF inverse and no full inverse lookup table. There is no SplitZip fallback, layer mask, AQUA adapter or lossy reconstruction. Predictor state, original coder tables, complete packets and BF16 words stay fixed; additional shared device tensor bytes are zero.

GEMM128 and GEMM64 replace decoder matrix products only with exact INT8/INT32 Triton products. The encoder keeps its original cuBLAS arithmetic and packet header. The independent Dense cached_model arm reuses its existing fused encoder model and native tables; it does not combine with a decoder GEMM change.

Whole encode/decode calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls use two warmups and five samples. Export, disk, upload, offline fitting and cold setup are outside warm measurements. Actual serialized encoder output is restored in a fresh process, alternating packets 0/1/0 with retained independently owned outputs.

The 512-token path retains compact staging, graph replay and owned output clone; the 8192-token path retains borrowed compact input and fresh output allocation. Candidate dispatch is installed before graph capture. Original author controls, native libraries, environments and all frozen sources remain unchanged.

Separate warmed profiles compare a fresh original standalone reference and each candidate after ordinary timing. Summed kernel work and stage timings are nonadditive; full traces stay local. Native allocator peaks outside Torch are unmeasured. Link scenarios are calculated, not measured NIC/SSD/serving or compressed-resident attention results.
