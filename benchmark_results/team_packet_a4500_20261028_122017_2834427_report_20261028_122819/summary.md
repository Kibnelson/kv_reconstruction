# Matched one-step compact packets versus Lite and official controls

Status: **completed**. GPU measured: **True**.

Identical original BF16 TEST words for all methods; all 28 layers, start 0; 512 and 8,192 tokens reported separately.
Team arithmetic is unchanged. Team fit uses six 8,192-token TRAIN documents and 8,192 scale tokens; Lite reuses its four 1,024-token fit. This first comparison does not isolate calibration or conditioning.
Complete packets include framing, actual directories and checksums. Fixed predictor/table state is additional. ExANS keeps its original fixed-slot padding/overflow policy.
D is a synchronized wall median. Team/Lite include GPU-resident packet validation, staging/expansion, graph decode and owned output. Native controls retain their original framed decode path and runtime. Export/checksum parsing/host upload are separate.

| Tokens | Method | Complete bpv | Saved | Full encode ms | Full decode ms | Native graph CUDA ms | Exact |
|---:|---|---:|---:|---:|---:|---:|---|
| 512 | exans_official_fixed_framed | 12.37970 | 22.63% | 15.154 | 3.662 | unmeasured | True |
| 512 | lite_current_graph | 9.09012 | 43.19% | 120.840 | 83.233 | unmeasured | True |
| 512 | raw_bf16 | 16.00009 | -0.00% | 0.416 | 0.365 | unmeasured | True |
| 512 | splitzip_official_framed | 12.04281 | 24.73% | 0.883 | 0.527 | unmeasured | True |
| 512 | team_dense_exact | 9.04882 | 43.44% | 25.502 | 8.028 | 7.111 | True |
| 512 | team_dense_uniform | 9.11376 | 43.04% | 23.240 | 4.695 | 3.870 | True |
| 512 | team_rank256_uniform | 9.50504 | 40.59% | 7.563 | 5.754 | 4.897 | True |
| 8192 | exans_official_fixed_framed | 12.34020 | 22.87% | 196.382 | 23.614 | unmeasured | True |
| 8192 | lite_current_graph | 9.12640 | 42.96% | 320.014 | 299.527 | unmeasured | True |
| 8192 | raw_bf16 | 16.00001 | -0.00% | 4.976 | 3.430 | unmeasured | True |
| 8192 | splitzip_official_framed | 12.05013 | 24.69% | 10.420 | 3.374 | unmeasured | True |
| 8192 | team_dense_exact | 8.86381 | 44.60% | 86.977 | 47.330 | 39.379 | True |
| 8192 | team_dense_uniform | 8.92097 | 44.24% | 83.411 | 35.545 | 27.854 | True |
| 8192 | team_rank256_uniform | 9.30632 | 41.84% | 42.053 | 28.715 | 21.147 | True |

Mean per-document accounting in MiB. The compact packet is the actual exported transfer object. The native arena, full BF16 output and scratch are additional working memory; these are not a resident-pool capacity result.

| Tokens | Team arm | Complete packet | Used native stream | Native arena | Compact staging | Directory staging | BF16 output | Workspace tensors | Shared predictors | Coder tables |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | team_dense_exact | 31.671 | 31.668 | 112.129 | 31.949 | 0.003 | 56.000 | 6.220 | 117.312 | 0.055 |
| 512 | team_dense_uniform | 31.898 | 31.896 | 112.129 | 32.184 | 0.003 | 56.000 | 6.220 | 117.312 | 0.055 |
| 512 | team_rank256_uniform | 33.268 | 33.265 | 112.129 | 33.529 | 0.003 | 56.000 | 6.345 | 146.242 | 0.055 |
| 8192 | team_dense_exact | 496.373 | 496.341 | 1794.062 | 502.046 | 0.047 | 896.000 | 99.516 | 117.312 | 0.055 |
| 8192 | team_dense_uniform | 499.574 | 499.542 | 1794.062 | 505.395 | 0.047 | 896.000 | 99.516 | 117.312 | 0.055 |
| 8192 | team_rank256_uniform | 521.154 | 521.122 | 1794.062 | 526.659 | 0.047 | 896.000 | 101.516 | 146.242 | 0.055 |

Separate encode diagnostics (wall medians, ms); the full encode above includes both model generation and encoding/compaction. These separately synchronized components are not additive.

| Tokens | Team arm | Model only | Encoding + compaction with prepared model | Export to host bytes | Import + host-to-device upload |
|---:|---|---:|---:|---:|---:|
| 512 | team_dense_exact | 20.455 | 4.982 | 59.866 | 37.794 |
| 512 | team_dense_uniform | 18.831 | 4.538 | 71.044 | 45.235 |
| 512 | team_rank256_uniform | 3.760 | 4.474 | 74.758 | 51.259 |
| 8192 | team_dense_exact | 47.511 | 36.880 | 1077.258 | 737.820 |
| 8192 | team_dense_uniform | 45.986 | 33.289 | 1082.543 | 748.674 |
| 8192 | team_rank256_uniform | 16.620 | 25.454 | 1118.333 | 766.408 |

Calculated precompressed serial wire + full receiver latency (ms), for every requested link target. Complete compact packets are charged; no network measured.

| Tokens | Gb/s | Lite | Team dense uniform | Team rank 256 uniform | SplitZip | ExANS fixed records |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 350.120 | 272.276 | 284.824 | 354.105 | 367.132 |
| 512 | 100 | 85.901 | 7.371 | 8.545 | 4.062 | 7.297 |
| 512 | 200 | 84.567 | 6.033 | 7.150 | 2.295 | 5.480 |
| 512 | 300 | 84.122 | 5.587 | 6.685 | 1.705 | 4.874 |
| 512 | 400 | 83.900 | 5.364 | 6.452 | 1.411 | 4.571 |
| 8192 | 1 | 4586.766 | 4226.278 | 4400.472 | 5664.066 | 5820.570 |
| 8192 | 100 | 342.400 | 77.452 | 72.432 | 59.981 | 81.583 |
| 8192 | 200 | 320.963 | 56.498 | 50.573 | 31.677 | 52.599 |
| 8192 | 300 | 313.818 | 49.514 | 43.287 | 22.243 | 42.937 |
| 8192 | 400 | 310.245 | 46.022 | 39.644 | 17.526 | 38.106 |

Calculated link models; no network, arrival pipeline or SSD measured.

| Tokens | Gb/s | Best precompressed wire + D | Best ideal max(wire,D) | Best fresh E + wire + D |
|---:|---:|---|---|---|
| 512 | 1 | team_dense_uniform (272.276ms) | team_dense_exact (265.675ms) | team_rank256_uniform (292.386ms) |
| 512 | 100 | splitzip_official_framed (4.062ms) | splitzip_official_framed (3.536ms) | splitzip_official_framed (4.945ms) |
| 512 | 200 | splitzip_official_framed (2.295ms) | splitzip_official_framed (1.768ms) | raw_bf16 (3.130ms) |
| 512 | 300 | splitzip_official_framed (1.705ms) | splitzip_official_framed (1.179ms) | raw_bf16 (2.347ms) |
| 512 | 400 | splitzip_official_framed (1.411ms) | splitzip_official_framed (0.884ms) | raw_bf16 (1.956ms) |
| 8192 | 1 | team_dense_exact (4211.210ms) | team_dense_exact (4163.880ms) | team_dense_exact (4298.187ms) |
| 8192 | 100 | splitzip_official_framed (59.981ms) | team_dense_uniform (41.907ms) | splitzip_official_framed (70.401ms) |
| 8192 | 200 | splitzip_official_framed (31.677ms) | splitzip_official_framed (28.303ms) | splitzip_official_framed (42.098ms) |
| 8192 | 300 | splitzip_official_framed (22.243ms) | splitzip_official_framed (18.869ms) | splitzip_official_framed (32.663ms) |
| 8192 | 400 | splitzip_official_framed (17.526ms) | splitzip_official_framed (14.152ms) | raw_bf16 (27.196ms) |

No lossy quality, AQUA adaptation, attention-facing pool, T-to-P conversion or serving throughput is measured. Native allocation peaks outside Torch remain unmeasured. Ideal overlap is a lower-bound model, not a measured transport result.
