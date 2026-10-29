# Standalone lossless full-CDF inverse lookup

**Status: completed.** GPU measured: **True**.

Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. The candidate changes only fixed-CDF inversion; complete original packets, fitted coefficients, prediction arithmetic and BF16 words stay fixed. SplitZip and ExANS are separate unchanged controls, not candidate components. Existing compact-search graph execution at 512 tokens and direct-search execution at 8192 tokens are retained. These two documents are reused engineering evidence, not independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.163 | 3.492 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.357 | 3.430 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.388 | 3.412 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.734 | 0.312 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.733 | 0.311 |
| 512 | standalone_lut_dense_uniform | paired | 9.11376 | 22.600 | 4.243 |
| 512 | standalone_lut_rank256_uniform | paired | 9.50504 | 7.022 | 4.798 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.518 | 4.127 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.996 | 4.529 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.413 | 23.022 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.850 | 23.349 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 185.450 | 23.373 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.317 |
| 8192 | raw_bf16 | before | 16.00001 | 4.891 | 3.318 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.287 | 3.188 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.233 | 3.198 |
| 8192 | standalone_lut_dense_uniform | paired | 8.92097 | 83.683 | 35.861 |
| 8192 | standalone_lut_rank256_uniform | paired | 9.30632 | 43.598 | 26.088 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.647 | 32.218 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.556 | 23.080 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.168 | 3.770 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.318 | 3.690 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.383 | 3.654 |
| 512 | raw_bf16 | after | 16.00009 | 0.414 | 0.337 |
| 512 | raw_bf16 | before | 16.00009 | 0.408 | 0.344 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.880 | 0.531 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.876 | 0.545 |
| 512 | standalone_lut_dense_uniform | paired | 9.11376 | 23.148 | 4.331 |
| 512 | standalone_lut_rank256_uniform | paired | 9.50504 | 7.609 | 5.167 |
| 512 | team_dense_uniform | paired | 9.11376 | 23.210 | 4.481 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.590 | 4.604 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.958 | 23.554 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 195.409 | 23.617 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 192.030 | 23.624 |
| 8192 | raw_bf16 | after | 16.00001 | 4.981 | 3.439 |
| 8192 | raw_bf16 | before | 16.00001 | 4.978 | 3.427 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.418 | 3.379 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.411 | 3.375 |
| 8192 | standalone_lut_dense_uniform | paired | 8.92097 | 82.574 | 32.132 |
| 8192 | standalone_lut_rank256_uniform | paired | 9.30632 | 41.959 | 23.894 |
| 8192 | team_dense_uniform | paired | 8.92097 | 82.856 | 28.807 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 41.893 | 21.539 |

Fixed standalone candidate versus its unchanged standalone reference. Speedup below 1 means slower.

| Tokens | Predictor | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---|---|---:|---:|---:|
| 512 | dense_uniform | batched | 0 | 0.996 | 0.972 |
| 512 | dense_uniform | isolated | 0 | 1.003 | 1.035 |
| 512 | rank256_uniform | batched | 0 | 0.996 | 0.944 |
| 512 | rank256_uniform | isolated | 0 | 0.998 | 0.891 |
| 8192 | dense_uniform | batched | 0 | 1.000 | 0.898 |
| 8192 | dense_uniform | isolated | 0 | 1.003 | 0.897 |
| 8192 | rank256_uniform | batched | 0 | 0.999 | 0.885 |
| 8192 | rank256_uniform | isolated | 0 | 0.998 | 0.901 |

The link tables use **8K isolated** API timing. Both lengths and both timing bases remain in results.json. Baseline cells use the faster complete before/after round independently for each objective. Dense and Rank remain fixed arms; no TEST-selected deployment policy is produced.

Calculated serial **precompressed**, milliseconds.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Dense LUT | Rank old | Rank LUT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7519.622 | 5664.068 | 5820.574 | 5133.301 | 4219.541 | 4222.866 | 4393.297 | 4395.651 |
| 5 | 1506.666 | 1135.514 | 1183.009 | 1045.504 | 866.954 | 870.279 | 895.891 | 898.245 |
| 10 | 755.046 | 569.445 | 603.313 | 534.529 | 447.881 | 451.206 | 458.715 | 461.070 |
| 25 | 304.074 | 229.803 | 255.496 | 227.944 | 196.437 | 199.762 | 196.409 | 198.764 |
| 50 | 153.751 | 116.589 | 139.557 | 125.749 | 112.622 | 115.947 | 108.974 | 111.329 |
| 75 | 103.643 | 78.851 | 100.910 | 91.684 | 84.684 | 88.009 | 79.829 | 82.184 |
| 100 | 78.589 | 59.982 | 81.587 | 74.652 | 70.715 | 74.040 | 65.257 | 67.611 |
| 200 | 41.008 | 31.679 | 52.602 | 49.103 | 49.761 | 53.086 | 43.398 | 45.753 |
| 300 | 28.481 | 22.244 | 42.941 | 40.587 | 42.777 | 46.101 | 36.112 | 38.466 |
| 400 | 22.217 | 17.527 | 38.110 | 36.329 | 39.284 | 42.609 | 32.469 | 34.823 |

Calculated serial **fresh**, milliseconds.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Dense LUT | Rank old | Rank LUT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7524.599 | 5674.479 | 6012.610 | 5343.259 | 4302.397 | 4305.440 | 4435.190 | 4437.611 |
| 5 | 1511.643 | 1145.925 | 1375.045 | 1255.461 | 949.810 | 952.853 | 937.784 | 940.205 |
| 10 | 760.024 | 579.856 | 795.350 | 744.487 | 530.737 | 533.780 | 500.608 | 503.029 |
| 25 | 309.052 | 240.214 | 447.532 | 437.902 | 279.293 | 282.336 | 238.302 | 240.724 |
| 50 | 158.728 | 127.000 | 331.593 | 335.707 | 195.478 | 198.521 | 150.867 | 153.288 |
| 75 | 108.620 | 89.263 | 292.947 | 301.642 | 167.540 | 170.583 | 121.722 | 124.143 |
| 100 | 83.566 | 70.394 | 273.624 | 284.610 | 153.571 | 156.614 | 107.150 | 109.571 |
| 200 | 45.985 | 42.090 | 244.639 | 259.061 | 132.617 | 135.660 | 85.291 | 87.712 |
| 300 | 33.458 | 32.656 | 234.977 | 250.545 | 125.633 | 128.676 | 78.005 | 80.426 |
| 400 | 27.195 | 27.938 | 230.146 | 246.286 | 122.141 | 125.183 | 74.361 | 76.783 |

Every standalone candidate uses our predictive/rANS coder for all 28 layers. No SplitZip fallback, layer mask, AQUA adapter or lossy reconstruction is involved. The new inverse LUT is 1,310,720 bytes (1.25 MiB) of additional shared GPU state; packet bytes and shared predictor coefficients stay fixed.

Whole encode/decode calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls use two warmups and five samples. Export, disk, upload, offline fitting and cold setup are outside warm measurements. Actual serialized encoder output is restored in a fresh process, alternating packets 0/1/0 with retained independently owned outputs.

The 512-token path retains compact staging, graph replay and owned output clone; the 8192-token path retains borrowed compact input and fresh output allocation. Candidate dispatch is installed before graph capture. Original author controls, native libraries, environments and all frozen sources remain unchanged.

Separate warmed profiles compare current standalone reference and candidate. Summed kernel work and stage timings are nonadditive; full traces stay local. Native allocator peaks outside Torch are unmeasured. Link scenarios are calculated, not measured NIC/SSD/serving or compressed-resident attention results.
