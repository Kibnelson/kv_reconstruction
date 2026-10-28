# Frozen one-step packet transport overhead

Status: **completed**. GPU measured: **True**.

Existing compact packets and fitted state are reused unchanged. Every exported byte string must match the parent SHA256 and every decoded BF16 word must match the parent original-word digest.
All timings below are means of per-document synchronized wall medians. Reference and optimized arms use balanced rotation/reversal and alternate two documents. CUDA intervals around host code can include host gaps; public wall latency is the primary metric.

| Tokens | Method | I/O arm | Export ms | Import ms | Import + decode ms | Export + import + decode ms | GPU-resident decode ms |
|---:|---|---|---:|---:|---:|---:|---:|
| 512 | team_dense_uniform | optimized | 42.899 | 20.973 | 25.510 | 67.738 | 4.678 |
| 512 | team_dense_uniform | reference | 61.503 | 45.935 | 50.725 | 104.248 | 4.678 |
| 512 | team_rank256_uniform | optimized | 45.681 | 21.774 | 26.969 | 71.801 | 5.220 |
| 512 | team_rank256_uniform | reference | 77.372 | 53.920 | 59.353 | 134.243 | 5.220 |
| 8192 | team_dense_uniform | optimized | 645.522 | 306.316 | 341.305 | 984.172 | 35.720 |
| 8192 | team_dense_uniform | reference | 1087.524 | 761.139 | 795.737 | 1888.155 | 35.720 |
| 8192 | team_rank256_uniform | optimized | 665.637 | 319.606 | 348.597 | 1014.351 | 28.786 |
| 8192 | team_rank256_uniform | reference | 1130.977 | 796.485 | 821.212 | 1949.525 | 28.786 |

Calculated serial link models. Host fetch starts with an immutable compressed blob in host DRAM. GPU egress starts with an already-compressed GPU packet and charges the measured full export/import/decode sequence plus wire time. This step does not encode raw KV or measure a network.

| Tokens | Method | Gb/s | Host fetch reference ms | Host fetch optimized ms | GPU egress reference ms | GPU egress optimized ms |
|---:|---|---:|---:|---:|---:|---:|
| 512 | team_dense_uniform | 1 | 318.306 | 293.091 | 371.830 | 335.320 |
| 512 | team_dense_uniform | 100 | 53.401 | 28.186 | 106.924 | 70.414 |
| 512 | team_dense_uniform | 200 | 52.063 | 26.848 | 105.586 | 69.076 |
| 512 | team_dense_uniform | 300 | 51.617 | 26.402 | 105.140 | 68.630 |
| 512 | team_dense_uniform | 400 | 51.394 | 26.179 | 104.917 | 68.407 |
| 512 | team_rank256_uniform | 1 | 338.422 | 306.039 | 413.312 | 350.870 |
| 512 | team_rank256_uniform | 100 | 62.144 | 29.760 | 137.034 | 74.592 |
| 512 | team_rank256_uniform | 200 | 60.748 | 28.365 | 135.638 | 73.196 |
| 512 | team_rank256_uniform | 300 | 60.283 | 27.900 | 135.173 | 72.731 |
| 512 | team_rank256_uniform | 400 | 60.051 | 27.667 | 134.941 | 72.499 |
| 8192 | team_dense_uniform | 1 | 4986.470 | 4532.038 | 6078.889 | 5174.906 |
| 8192 | team_dense_uniform | 100 | 837.644 | 383.212 | 1930.063 | 1026.080 |
| 8192 | team_dense_uniform | 200 | 816.690 | 362.258 | 1909.109 | 1005.126 |
| 8192 | team_dense_uniform | 300 | 809.706 | 355.274 | 1902.125 | 998.141 |
| 8192 | team_dense_uniform | 400 | 806.213 | 351.781 | 1898.632 | 994.649 |
| 8192 | team_rank256_uniform | 1 | 5192.970 | 4720.355 | 6321.282 | 5386.108 |
| 8192 | team_rank256_uniform | 100 | 864.930 | 392.315 | 1993.242 | 1058.069 |
| 8192 | team_rank256_uniform | 200 | 843.071 | 370.456 | 1971.384 | 1036.210 |
| 8192 | team_rank256_uniform | 300 | 835.785 | 363.170 | 1964.097 | 1028.923 |
| 8192 | team_rank256_uniform | 400 | 832.142 | 359.527 | 1960.454 | 1025.280 |

Separate profiled 8K text phase diagnostics (wall ms). These instrumented calls run in different processes from ordinary paired timing; their phase totals are not additive ordinary latency. An em dash means the phase is absent, rather than a measured zero. CUDA copy intervals and top cProfile/Torch functions remain in profile.json; full Chrome traces stay local.

| Method | I/O arm | D2H | Dtype/layout copy | Payload tobytes | Frame assembly | Checksum | Checksum append |
|---|---|---:|---:|---:|---:|---:|---:|
| team_dense_uniform | reference | 182.815 | 71.187 | 172.028 | 168.480 | 272.697 | 169.799 |
| team_dense_uniform | optimized | 189.597 | 0.060 | — | 169.024 | 285.756 | — |
| team_rank256_uniform | reference | 190.202 | 74.701 | 179.904 | 177.052 | 284.169 | 178.686 |
| team_rank256_uniform | optimized | 191.783 | 0.033 | — | 177.694 | 284.155 | — |

| Method | I/O arm | Checksum input copy | Checksum | Payload bytes slice | Owned CPU copy | H2D | Validation |
|---|---|---:|---:|---:|---:|---:|---:|
| team_dense_uniform | reference | 172.069 | 272.531 | 171.425 | 69.987 | 35.826 | 0.382 |
| team_dense_uniform | optimized | — | 272.716 | — | — | 39.181 | 0.299 |
| team_rank256_uniform | reference | 180.512 | 284.014 | 182.656 | 73.762 | 36.823 | 0.361 |
| team_rank256_uniform | optimized | — | 284.353 | — | — | 37.257 | 0.299 |

Earlier GPU-resident decode anchors, reused without retiming. Their export/import paths were not measured here, so these are not matched end-to-end transport competitors.

| Tokens | Earlier method | Complete bpv | Earlier GPU-resident decode ms |
|---:|---|---:|---:|
| 512 | exans_official_fixed_framed | 12.37970 | 3.662 |
| 512 | lite_current_graph | 9.09012 | 83.233 |
| 512 | raw_bf16 | 16.00009 | 0.365 |
| 512 | splitzip_official_framed | 12.04281 | 0.527 |
| 8192 | exans_official_fixed_framed | 12.34020 | 23.614 |
| 8192 | lite_current_graph | 9.12640 | 299.527 |
| 8192 | raw_bf16 | 16.00001 | 3.430 |
| 8192 | splitzip_official_framed | 12.05013 | 3.374 |

Cold state/graph setup and file-read samples are separate. No pinned reusable workspace is introduced by these byte I/O arms. Native codec arithmetic, prediction, compressed attention, pool conversion and serving remain outside this change.
