# Standalone lossless warp-prefix compaction

**Status: completed.** GPU measured: **True**.

Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. The current single-feature rolling encoder and two-chunk packed-inverse decoder are the reference. The candidate changes only exact renormalization-word prefix compaction to warp ballot/popcount. Original complete packets, fitted coefficients, probabilities, rANS state updates and BF16 words stay fixed. Both modes retain the same 45,568 shared inverse-table bytes; prefix compaction adds no device tables. SplitZip and ExANS remain separate unchanged controls, not candidate components. 512-token graph execution and 8K borrowed-input direct execution are retained. The two documents are reused engineering evidence, not independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.181 | 3.503 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.320 | 3.425 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.358 | 3.418 |
| 512 | prefix_ref_dense_uniform | paired | 9.11376 | 5.182 | 3.928 |
| 512 | prefix_ref_rank256_uniform | paired | 9.50504 | 5.120 | 3.910 |
| 512 | prefix_warp_dense_uniform | paired | 9.11376 | 5.174 | 4.075 |
| 512 | prefix_warp_rank256_uniform | paired | 9.50504 | 5.149 | 4.172 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.733 | 0.310 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.732 | 0.308 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.494 | 23.015 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.201 | 23.347 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 184.707 | 23.354 |
| 8192 | prefix_ref_dense_uniform | paired | 8.92097 | 45.109 | 31.660 |
| 8192 | prefix_ref_rank256_uniform | paired | 9.30632 | 36.098 | 22.693 |
| 8192 | prefix_warp_dense_uniform | paired | 8.92097 | 44.960 | 32.765 |
| 8192 | prefix_warp_rank256_uniform | paired | 9.30632 | 36.056 | 24.014 |
| 8192 | raw_bf16 | after | 16.00001 | 4.889 | 3.318 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.318 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.270 | 3.188 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.234 | 3.188 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.184 | 3.789 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.377 | 3.660 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.012 | 3.677 |
| 512 | prefix_ref_dense_uniform | paired | 9.11376 | 5.593 | 4.151 |
| 512 | prefix_ref_rank256_uniform | paired | 9.50504 | 5.606 | 4.012 |
| 512 | prefix_warp_dense_uniform | paired | 9.11376 | 5.626 | 4.291 |
| 512 | prefix_warp_rank256_uniform | paired | 9.50504 | 5.605 | 5.345 |
| 512 | raw_bf16 | after | 16.00009 | 0.418 | 0.353 |
| 512 | raw_bf16 | before | 16.00009 | 0.420 | 0.346 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.871 | 0.536 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.873 | 0.528 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.938 | 23.586 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 192.606 | 23.587 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 193.934 | 23.587 |
| 8192 | prefix_ref_dense_uniform | paired | 8.92097 | 42.540 | 28.161 |
| 8192 | prefix_ref_rank256_uniform | paired | 9.30632 | 34.571 | 21.204 |
| 8192 | prefix_warp_dense_uniform | paired | 8.92097 | 42.275 | 29.453 |
| 8192 | prefix_warp_rank256_uniform | paired | 9.30632 | 34.668 | 22.372 |
| 8192 | raw_bf16 | after | 16.00001 | 4.990 | 3.420 |
| 8192 | raw_bf16 | before | 16.00001 | 4.989 | 3.438 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.408 | 3.382 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.303 | 3.367 |

Incremental Torch API peaks: mean of each document's median over five isolated calls. Output and scratch are included; native allocations outside Torch are unmeasured.

| Tokens | Method | Round | Encode peak MiB | Decode peak MiB |
|---:|---|---|---:|---:|
| 512 | exans_official_compact_framed | paired | 143.564 | 56.234 |
| 512 | exans_official_fixed_framed | after | 144.236 | 56.235 |
| 512 | exans_official_fixed_framed | before | 144.236 | 56.235 |
| 512 | prefix_ref_dense_uniform | paired | 144.028 | 56.000 |
| 512 | prefix_ref_rank256_uniform | paired | 145.633 | 56.000 |
| 512 | prefix_warp_dense_uniform | paired | 144.028 | 56.000 |
| 512 | prefix_warp_rank256_uniform | paired | 145.633 | 56.000 |
| 512 | raw_bf16 | after | 56.000 | 56.000 |
| 512 | raw_bf16 | before | 56.000 | 56.000 |
| 512 | splitzip_official_framed | after | 84.628 | 56.513 |
| 512 | splitzip_official_framed | before | 84.628 | 56.513 |
| 8192 | exans_official_compact_framed | paired | 2279.953 | 896.891 |
| 8192 | exans_official_fixed_framed | after | 2280.902 | 896.898 |
| 8192 | exans_official_fixed_framed | before | 2280.902 | 896.898 |
| 8192 | prefix_ref_dense_uniform | paired | 2293.955 | 896.000 |
| 8192 | prefix_ref_rank256_uniform | paired | 2315.232 | 896.000 |
| 8192 | prefix_warp_dense_uniform | paired | 2293.955 | 896.000 |
| 8192 | prefix_warp_rank256_uniform | paired | 2315.232 | 896.000 |
| 8192 | raw_bf16 | after | 896.000 | 896.000 |
| 8192 | raw_bf16 | before | 896.000 | 896.000 |
| 8192 | splitzip_official_framed | after | 1354.864 | 905.623 |
| 8192 | splitzip_official_framed | before | 1354.864 | 905.623 |

Warp-prefix candidate versus the current packed-inverse reference. Speedup below 1 means slower.

| Tokens | Method | Reference | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---|---|---|---:|---:|---:|
| 512 | prefix_warp_dense_uniform | prefix_ref_dense_uniform | batched | 0 | 1.002 | 0.964 |
| 512 | prefix_warp_dense_uniform | prefix_ref_dense_uniform | isolated | 0 | 0.994 | 0.967 |
| 512 | prefix_warp_rank256_uniform | prefix_ref_rank256_uniform | batched | 0 | 0.994 | 0.937 |
| 512 | prefix_warp_rank256_uniform | prefix_ref_rank256_uniform | isolated | 0 | 1.000 | 0.751 |
| 8192 | prefix_warp_dense_uniform | prefix_ref_dense_uniform | batched | 0 | 1.003 | 0.966 |
| 8192 | prefix_warp_dense_uniform | prefix_ref_dense_uniform | isolated | 0 | 1.006 | 0.956 |
| 8192 | prefix_warp_rank256_uniform | prefix_ref_rank256_uniform | batched | 0 | 1.001 | 0.945 |
| 8192 | prefix_warp_rank256_uniform | prefix_ref_rank256_uniform | isolated | 0 | 0.997 | 0.948 |

Link tables use **8K isolated** API timing. Baseline cells use the faster complete before/after round independently for each objective. Each arm is fixed; no TEST-selected deployment policy or combined optimization is produced.

Calculated serial **precompressed**, dense_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Packed reference | Warp prefix |
|---:|---:|---:|---:|---:|
| 1 | 5664.059 | 5133.332 | 4218.894 | 4220.186 |
| 5 | 1135.505 | 1045.535 | 866.307 | 867.599 |
| 10 | 569.436 | 534.560 | 447.234 | 448.526 |
| 25 | 229.795 | 227.976 | 195.790 | 197.082 |
| 50 | 116.581 | 125.781 | 111.975 | 113.267 |
| 75 | 78.843 | 91.716 | 84.037 | 85.329 |
| 100 | 59.974 | 74.683 | 70.068 | 71.360 |
| 200 | 31.670 | 49.134 | 49.114 | 50.406 |
| 300 | 22.236 | 40.618 | 42.130 | 43.422 |
| 400 | 17.519 | 36.360 | 38.637 | 39.930 |

Calculated serial **precompressed**, rank256_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Packed reference | Warp prefix |
|---:|---:|---:|---:|---:|
| 1 | 5664.059 | 5133.332 | 4392.961 | 4394.129 |
| 5 | 1135.505 | 1045.535 | 895.555 | 896.723 |
| 10 | 569.436 | 534.560 | 458.380 | 459.548 |
| 25 | 229.795 | 227.976 | 196.074 | 197.242 |
| 50 | 116.581 | 125.781 | 108.639 | 109.807 |
| 75 | 78.843 | 91.716 | 79.494 | 80.662 |
| 100 | 59.974 | 74.683 | 64.921 | 66.089 |
| 200 | 31.670 | 49.134 | 43.063 | 44.231 |
| 300 | 22.236 | 40.618 | 35.776 | 36.944 |
| 400 | 17.519 | 36.360 | 32.133 | 33.301 |

Calculated serial **fresh**, dense_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Packed reference | Warp prefix |
|---:|---:|---:|---:|---:|
| 1 | 5674.362 | 5343.270 | 4261.434 | 4262.461 |
| 5 | 1145.808 | 1255.473 | 908.847 | 909.875 |
| 10 | 579.739 | 744.498 | 489.774 | 490.801 |
| 25 | 240.097 | 437.914 | 238.330 | 239.357 |
| 50 | 126.884 | 335.719 | 154.515 | 155.543 |
| 75 | 89.146 | 301.654 | 126.577 | 127.604 |
| 100 | 70.277 | 284.621 | 112.608 | 113.635 |
| 200 | 41.973 | 259.072 | 91.654 | 92.682 |
| 300 | 32.539 | 250.556 | 84.670 | 85.697 |
| 400 | 27.821 | 246.298 | 81.178 | 82.205 |

Calculated serial **fresh**, rank256_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Packed reference | Warp prefix |
|---:|---:|---:|---:|---:|
| 1 | 5674.362 | 5343.270 | 4427.532 | 4428.798 |
| 5 | 1145.808 | 1255.473 | 930.126 | 931.392 |
| 10 | 579.739 | 744.498 | 492.951 | 494.216 |
| 25 | 240.097 | 437.914 | 230.645 | 231.911 |
| 50 | 126.884 | 335.719 | 143.210 | 144.475 |
| 75 | 89.146 | 301.654 | 114.065 | 115.330 |
| 100 | 70.277 | 284.621 | 99.492 | 100.758 |
| 200 | 41.973 | 259.072 | 77.634 | 78.899 |
| 300 | 32.539 | 250.556 | 70.347 | 71.613 |
| 400 | 27.821 | 246.298 | 66.704 | 67.970 |

Measured-packet decoder budgets against SplitZip for precompressed transfer. These are calculated targets, not measured NIC performance or promised kernel gains.

| Method | Gb/s | Decode budget ms | Actual decode ms | Reduction needed ms | Target met |
|---|---:|---:|---:|---:|---|
| prefix_warp_dense_uniform | 100 | 18.067 | 29.453 | 11.386 | False |
| prefix_warp_dense_uniform | 400 | 7.042 | 29.453 | 22.411 | False |
| prefix_warp_rank256_uniform | 100 | 16.256 | 22.372 | 6.115 | False |
| prefix_warp_rank256_uniform | 400 | 6.589 | 22.372 | 15.782 | False |

All 28 layers use our predictive/rANS codec. Original complete packets, predictor state, probability tables and BF16 words stay fixed. Both reference and candidate retain the original 57,688-byte coder state plus 45,568 packed-inverse bytes, for 103,256 total coder bytes. Warp-prefix compaction adds no device tables. Predictor tensors are additional.

All own arms retain the measured single-feature rolling encoder and two decoder chunks. Only renormalization-word prefix compaction changes. Probability tables, packed CDF inversion, state arithmetic, word bounds and escapes remain unchanged. Actual CUDA mapping/probe and whole-packet gates must pass before any timings. Full per-call Torch peaks include output and scratch; shared state is additional. Native allocator peaks outside Torch remain unmeasured.

Whole encode/decode calls preserve author timing: 10 warmups, 50 calls and five repeats. Separate isolated calls use two warmups and five samples. Cold setup, export, disk, upload and fitting remain outside warm measurements. Actual serialized encoder packets are restored in fresh processes, alternating 0/1/0 while retaining independent returned outputs.

512-token decoding keeps compact staging, graph replay and an owned output clone. 8K decoding borrows aligned compact input and allocates fresh output. Sender geometry, encoder arithmetic, coefficient tensors and probability tables stay fixed. Source, environments and native libraries are checked before and after the run.

Separate warmed profiles bind actual packed/prefix decoder, predictor and feature-controller dispatch after ordinary timing. Summed work and stage samples are nonadditive. Two reused engineering documents and calculated link scenarios do not establish independent generalization, measured NIC/SSD performance, resident compressed attention or serving throughput.
