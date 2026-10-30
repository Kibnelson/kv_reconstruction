# Standalone lossless rolling encoder

**Status: completed.** GPU measured: **True**.

Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. The rolling encoder computes each nonzero-layer prediction product once, retaining only two temporary INT32 matrices. Decoder, original complete packets, fitted coefficients, fixed binary-CDF inverse and BF16 words stay fixed. A paired cached-Dense control retains the preceding verified setup optimization. SplitZip and ExANS are separate unchanged controls, not candidate components. Compact-search graph execution at 512 tokens and direct-search execution at 8192 tokens are retained. These two documents are reused engineering evidence, not independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | cached_dense_control | paired | 9.11376 | 10.338 | 4.152 |
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.168 | 3.499 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.410 | 3.421 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.313 | 3.412 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | rolling_dense_uniform | paired | 9.11376 | 5.227 | 4.121 |
| 512 | rolling_rank256_uniform | paired | 9.50504 | 5.214 | 4.529 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.734 | 0.310 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.734 | 0.310 |
| 512 | team_dense_uniform | paired | 9.11376 | 22.496 | 4.140 |
| 512 | team_rank256_uniform | paired | 9.50504 | 6.955 | 4.568 |
| 8192 | cached_dense_control | paired | 8.92097 | 76.857 | 32.322 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.052 | 23.034 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.263 | 23.389 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 184.775 | 23.353 |
| 8192 | raw_bf16 | after | 16.00001 | 4.891 | 3.318 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.317 |
| 8192 | rolling_dense_uniform | paired | 8.92097 | 45.052 | 32.012 |
| 8192 | rolling_rank256_uniform | paired | 9.30632 | 36.104 | 23.034 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.284 | 3.187 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.236 | 3.186 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.877 | 32.332 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 43.567 | 23.101 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | cached_dense_control | paired | 9.11376 | 11.531 | 4.301 |
| 512 | exans_official_compact_framed | paired | 10.99741 | 16.048 | 3.767 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.852 | 3.651 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.493 | 3.674 |
| 512 | raw_bf16 | after | 16.00009 | 0.410 | 0.332 |
| 512 | raw_bf16 | before | 16.00009 | 0.412 | 0.339 |
| 512 | rolling_dense_uniform | paired | 9.11376 | 5.604 | 4.460 |
| 512 | rolling_rank256_uniform | paired | 9.50504 | 5.675 | 4.786 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.874 | 0.529 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.875 | 0.528 |
| 512 | team_dense_uniform | paired | 9.11376 | 23.201 | 4.504 |
| 512 | team_rank256_uniform | paired | 9.50504 | 7.558 | 4.655 |
| 8192 | cached_dense_control | paired | 8.92097 | 73.247 | 29.099 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.806 | 23.554 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 193.187 | 23.648 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 193.874 | 23.637 |
| 8192 | raw_bf16 | after | 16.00001 | 4.976 | 3.423 |
| 8192 | raw_bf16 | before | 16.00001 | 4.981 | 3.435 |
| 8192 | rolling_dense_uniform | paired | 8.92097 | 42.304 | 28.849 |
| 8192 | rolling_rank256_uniform | paired | 9.30632 | 34.648 | 21.563 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.357 | 3.400 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.325 | 3.370 |
| 8192 | team_dense_uniform | paired | 8.92097 | 83.270 | 28.895 |
| 8192 | team_rank256_uniform | paired | 9.30632 | 41.986 | 21.649 |

Each fixed encoder arm versus the unchanged binary-search standalone reference; rolling Dense is also paired against the verified cached-Dense control. Speedup below 1 means slower.

| Tokens | Method | Reference | Predictor | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---|---|---|---|---:|---:|---:|
| 512 | rolling_dense_uniform | team_dense_uniform | dense_uniform | batched | 0 | 4.304 | 1.005 |
| 512 | rolling_dense_uniform | team_dense_uniform | dense_uniform | isolated | 0 | 4.140 | 1.010 |
| 512 | rolling_dense_uniform | cached_dense_control | dense_uniform | batched | 0 | 1.978 | 1.008 |
| 512 | rolling_dense_uniform | cached_dense_control | dense_uniform | isolated | 0 | 2.058 | 0.964 |
| 512 | rolling_rank256_uniform | team_rank256_uniform | rank256_uniform | batched | 0 | 1.334 | 1.009 |
| 512 | rolling_rank256_uniform | team_rank256_uniform | rank256_uniform | isolated | 0 | 1.332 | 0.973 |
| 512 | cached_dense_control | team_dense_uniform | dense_uniform | batched | 0 | 2.176 | 0.997 |
| 512 | cached_dense_control | team_dense_uniform | dense_uniform | isolated | 0 | 2.012 | 1.047 |
| 8192 | rolling_dense_uniform | team_dense_uniform | dense_uniform | batched | 0 | 1.862 | 1.010 |
| 8192 | rolling_dense_uniform | team_dense_uniform | dense_uniform | isolated | 0 | 1.968 | 1.002 |
| 8192 | rolling_dense_uniform | cached_dense_control | dense_uniform | batched | 0 | 1.706 | 1.010 |
| 8192 | rolling_dense_uniform | cached_dense_control | dense_uniform | isolated | 0 | 1.731 | 1.009 |
| 8192 | rolling_rank256_uniform | team_rank256_uniform | rank256_uniform | batched | 0 | 1.207 | 1.003 |
| 8192 | rolling_rank256_uniform | team_rank256_uniform | rank256_uniform | isolated | 0 | 1.212 | 1.004 |
| 8192 | cached_dense_control | team_dense_uniform | dense_uniform | batched | 0 | 1.091 | 1.000 |
| 8192 | cached_dense_control | team_dense_uniform | dense_uniform | isolated | 0 | 1.137 | 0.993 |

The link tables use **8K isolated** API timing. Both lengths and timing bases remain in results.json. Baseline cells use the faster complete before/after round independently for each objective. All encoder arms are fixed; no TEST-selected deployment policy or combined optimization is produced.

Calculated serial **precompressed**, milliseconds.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Rank old |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7519.618 | 5664.062 | 5820.593 | 5133.301 | 4219.629 | 4393.407 |
| 5 | 1506.662 | 1135.508 | 1183.028 | 1045.503 | 867.042 | 896.001 |
| 10 | 755.042 | 569.439 | 603.332 | 534.529 | 447.969 | 458.825 |
| 25 | 304.071 | 229.798 | 255.515 | 227.944 | 196.525 | 196.520 |
| 50 | 153.747 | 116.584 | 139.576 | 125.749 | 112.710 | 109.084 |
| 75 | 103.639 | 78.846 | 100.929 | 91.684 | 84.772 | 79.939 |
| 100 | 78.585 | 59.977 | 81.606 | 74.652 | 70.803 | 65.367 |
| 200 | 41.004 | 31.673 | 52.621 | 49.103 | 49.849 | 43.508 |
| 300 | 28.477 | 22.239 | 42.960 | 40.587 | 42.864 | 36.222 |
| 400 | 22.213 | 17.522 | 38.129 | 36.328 | 39.372 | 32.579 |

Fixed encoder arms, serial **precompressed**, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Dense original | Rank original | Rolling Dense | Rolling Rank | Cached Dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5664.062 | 5133.301 | 4219.629 | 4393.407 | 4219.582 | 4393.320 | 4219.832 |
| 5 | 1135.508 | 1045.503 | 867.042 | 896.001 | 866.995 | 895.914 | 867.245 |
| 10 | 569.439 | 534.529 | 447.969 | 458.825 | 447.922 | 458.738 | 448.172 |
| 25 | 229.798 | 227.944 | 196.525 | 196.520 | 196.478 | 196.433 | 196.728 |
| 50 | 116.584 | 125.749 | 112.710 | 109.084 | 112.663 | 108.998 | 112.913 |
| 75 | 78.846 | 91.684 | 84.772 | 79.939 | 84.725 | 79.853 | 84.975 |
| 100 | 59.977 | 74.652 | 70.803 | 65.367 | 70.756 | 65.280 | 71.006 |
| 200 | 31.673 | 49.103 | 49.849 | 43.508 | 49.802 | 43.422 | 50.052 |
| 300 | 22.239 | 40.587 | 42.864 | 36.222 | 42.818 | 36.135 | 43.068 |
| 400 | 17.522 | 36.328 | 39.372 | 32.579 | 39.325 | 32.492 | 39.576 |

Calculated serial **fresh**, milliseconds.

| Gb/s | Raw | SplitZip | Fixed ExANS | Compact ExANS | Dense old | Rank old |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7524.595 | 5674.387 | 6013.791 | 5343.107 | 4302.899 | 4435.393 |
| 5 | 1511.638 | 1145.833 | 1376.226 | 1255.309 | 950.312 | 937.987 |
| 10 | 760.019 | 579.764 | 796.530 | 744.335 | 531.239 | 500.811 |
| 25 | 309.047 | 240.122 | 448.713 | 437.750 | 279.795 | 238.506 |
| 50 | 158.723 | 126.908 | 332.774 | 335.555 | 195.980 | 151.071 |
| 75 | 108.615 | 89.171 | 294.127 | 301.490 | 168.042 | 121.926 |
| 100 | 83.561 | 70.302 | 274.804 | 284.458 | 154.073 | 107.353 |
| 200 | 45.980 | 41.998 | 245.819 | 258.909 | 133.119 | 85.495 |
| 300 | 33.453 | 32.564 | 236.158 | 250.393 | 126.134 | 78.208 |
| 400 | 27.190 | 27.846 | 231.327 | 246.135 | 122.642 | 74.565 |

Fixed encoder arms, serial **fresh**, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Dense original | Rank original | Rolling Dense | Rolling Rank | Cached Dense |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5674.387 | 5343.107 | 4302.899 | 4435.393 | 4261.886 | 4427.968 | 4293.079 |
| 5 | 1145.833 | 1255.309 | 950.312 | 937.987 | 909.300 | 930.563 | 940.492 |
| 10 | 579.764 | 744.335 | 531.239 | 500.811 | 490.226 | 493.387 | 521.419 |
| 25 | 240.122 | 437.750 | 279.795 | 238.506 | 238.782 | 231.081 | 269.975 |
| 50 | 126.908 | 335.555 | 195.980 | 151.071 | 154.968 | 143.646 | 186.160 |
| 75 | 89.171 | 301.490 | 168.042 | 121.926 | 127.029 | 114.501 | 158.222 |
| 100 | 70.302 | 284.458 | 154.073 | 107.353 | 113.060 | 99.929 | 144.253 |
| 200 | 41.998 | 258.909 | 133.119 | 85.495 | 92.107 | 78.070 | 123.299 |
| 300 | 32.564 | 250.393 | 126.134 | 78.208 | 85.122 | 70.784 | 116.315 |
| 400 | 27.846 | 246.135 | 122.642 | 74.565 | 81.630 | 67.140 | 112.822 |

Every candidate uses our predictive/rANS coder for all 28 layers, with the original binary-CDF inverse and no full inverse lookup table. There is no SplitZip fallback, layer mask, AQUA adapter or lossy reconstruction. Predictor state, original coder tables, complete packets and BF16 words stay fixed; additional shared device tensor bytes are zero.

The rolling encoder first extracts exact source-word features, then visits layers in reverse, reusing each prediction product for the preceding-layer surprise and its own coding step. Only two INT32 product matrices are retained, not products for every layer. Their per-call scratch is included in incremental Torch allocation peaks. The original decoder and encoder product arithmetic are unchanged. The paired cached-Dense control retains the preceding verified encoder setup optimization.

Whole encode/decode calls use the unchanged author timing body: 10 warmups, 50 calls, five repeats. Separate isolated calls use two warmups and five samples. Export, disk, upload, offline fitting and cold setup are outside warm measurements. Actual serialized encoder output is restored in a fresh process, alternating packets 0/1/0 with retained independently owned outputs.

The 512-token path retains compact staging, graph replay and owned output clone; the 8192-token path retains borrowed compact input and fresh output allocation. Candidate dispatch is installed before graph capture. Original author controls, native libraries, environments and all frozen sources remain unchanged.

Separate warmed profiles compare a fresh original standalone reference and each candidate after ordinary timing. Summed kernel work and stage timings are nonadditive; full traces stay local. Native allocator peaks outside Torch are unmeasured. Link scenarios are calculated, not measured NIC/SSD/serving or compressed-resident attention results.
