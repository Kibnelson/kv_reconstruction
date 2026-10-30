# Standalone lossless packed and masked CDF inversion

**Status: completed.** GPU measured: **True**.

Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. The current single-feature rolling encoder and two-chunk binary-search decoder are the reference. Separate fixed decoder arms use packed coarse cutoffs with sparse 64-slot exceptions, or mask CDF loads for converged binary-search lanes. Original complete packets, fitted coefficients, probabilities, rANS state updates and BF16 words stay fixed. Packed inversion adds 45,568 shared device bytes; masked inversion adds none. SplitZip and ExANS remain separate unchanged controls, not candidate components. 512-token graph execution and 8K borrowed-input direct execution are retained. The two documents are reused engineering evidence, not independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.135 | 3.489 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.328 | 3.413 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.380 | 3.423 |
| 512 | inverse_masked_dense_uniform | paired | 9.11376 | 5.170 | 4.053 |
| 512 | inverse_masked_rank256_uniform | paired | 9.50504 | 5.150 | 4.440 |
| 512 | inverse_packed_dense_uniform | paired | 9.11376 | 5.181 | 3.929 |
| 512 | inverse_packed_rank256_uniform | paired | 9.50504 | 5.109 | 3.892 |
| 512 | inverse_ref_dense_uniform | paired | 9.11376 | 5.178 | 4.128 |
| 512 | inverse_ref_rank256_uniform | paired | 9.50504 | 5.114 | 4.540 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.322 | 0.229 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.733 | 0.309 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.734 | 0.312 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.287 | 23.045 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.809 | 23.434 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 184.910 | 23.372 |
| 8192 | inverse_masked_dense_uniform | paired | 8.92097 | 45.070 | 32.035 |
| 8192 | inverse_masked_rank256_uniform | paired | 9.30632 | 36.092 | 23.080 |
| 8192 | inverse_packed_dense_uniform | paired | 8.92097 | 45.000 | 31.476 |
| 8192 | inverse_packed_rank256_uniform | paired | 9.30632 | 36.139 | 22.712 |
| 8192 | inverse_ref_dense_uniform | paired | 8.92097 | 45.162 | 32.151 |
| 8192 | inverse_ref_rank256_uniform | paired | 9.30632 | 36.133 | 23.026 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.320 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.317 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.276 | 3.186 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.216 | 3.188 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.206 | 3.761 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.363 | 3.660 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 14.001 | 3.711 |
| 512 | inverse_masked_dense_uniform | paired | 9.11376 | 5.582 | 4.388 |
| 512 | inverse_masked_rank256_uniform | paired | 9.50504 | 5.594 | 4.954 |
| 512 | inverse_packed_dense_uniform | paired | 9.11376 | 5.574 | 4.157 |
| 512 | inverse_packed_rank256_uniform | paired | 9.50504 | 5.550 | 4.399 |
| 512 | inverse_ref_dense_uniform | paired | 9.11376 | 5.573 | 4.390 |
| 512 | inverse_ref_rank256_uniform | paired | 9.50504 | 5.544 | 5.003 |
| 512 | raw_bf16 | after | 16.00009 | 0.408 | 0.337 |
| 512 | raw_bf16 | before | 16.00009 | 0.412 | 0.336 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.876 | 0.537 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.872 | 0.533 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.828 | 23.593 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 194.541 | 23.691 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 193.272 | 23.626 |
| 8192 | inverse_masked_dense_uniform | paired | 8.92097 | 42.425 | 29.081 |
| 8192 | inverse_masked_rank256_uniform | paired | 9.30632 | 34.568 | 21.603 |
| 8192 | inverse_packed_dense_uniform | paired | 8.92097 | 42.788 | 28.159 |
| 8192 | inverse_packed_rank256_uniform | paired | 9.30632 | 34.613 | 21.184 |
| 8192 | inverse_ref_dense_uniform | paired | 8.92097 | 42.835 | 28.692 |
| 8192 | inverse_ref_rank256_uniform | paired | 9.30632 | 34.685 | 21.597 |
| 8192 | raw_bf16 | after | 16.00001 | 4.997 | 3.434 |
| 8192 | raw_bf16 | before | 16.00001 | 4.980 | 3.435 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.414 | 3.372 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.302 | 3.373 |

Incremental Torch API peaks: mean of each document's median over five isolated calls. Output and scratch are included; native allocations outside Torch are unmeasured.

| Tokens | Method | Round | Encode peak MiB | Decode peak MiB |
|---:|---|---|---:|---:|
| 512 | exans_official_compact_framed | paired | 143.564 | 56.234 |
| 512 | exans_official_fixed_framed | after | 144.236 | 56.235 |
| 512 | exans_official_fixed_framed | before | 144.236 | 56.235 |
| 512 | inverse_masked_dense_uniform | paired | 144.028 | 56.000 |
| 512 | inverse_masked_rank256_uniform | paired | 145.633 | 56.000 |
| 512 | inverse_packed_dense_uniform | paired | 144.028 | 56.000 |
| 512 | inverse_packed_rank256_uniform | paired | 145.633 | 56.000 |
| 512 | inverse_ref_dense_uniform | paired | 144.028 | 56.000 |
| 512 | inverse_ref_rank256_uniform | paired | 145.633 | 56.000 |
| 512 | raw_bf16 | after | 56.000 | 56.000 |
| 512 | raw_bf16 | before | 56.000 | 56.000 |
| 512 | splitzip_official_framed | after | 84.628 | 56.513 |
| 512 | splitzip_official_framed | before | 84.628 | 56.513 |
| 8192 | exans_official_compact_framed | paired | 2279.953 | 896.891 |
| 8192 | exans_official_fixed_framed | after | 2280.902 | 896.898 |
| 8192 | exans_official_fixed_framed | before | 2280.902 | 896.898 |
| 8192 | inverse_masked_dense_uniform | paired | 2293.955 | 896.000 |
| 8192 | inverse_masked_rank256_uniform | paired | 2315.232 | 896.000 |
| 8192 | inverse_packed_dense_uniform | paired | 2293.955 | 896.000 |
| 8192 | inverse_packed_rank256_uniform | paired | 2315.232 | 896.000 |
| 8192 | inverse_ref_dense_uniform | paired | 2293.955 | 896.000 |
| 8192 | inverse_ref_rank256_uniform | paired | 2315.232 | 896.000 |
| 8192 | raw_bf16 | after | 896.000 | 896.000 |
| 8192 | raw_bf16 | before | 896.000 | 896.000 |
| 8192 | splitzip_official_framed | after | 1354.864 | 905.623 |
| 8192 | splitzip_official_framed | before | 1354.864 | 905.623 |

Each fixed inverse versus the current single-feature rolling reference. Speedup below 1 means slower.

| Tokens | Method | Reference | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---|---|---|---:|---:|---:|
| 512 | inverse_packed_dense_uniform | inverse_ref_dense_uniform | batched | 0 | 0.999 | 1.051 |
| 512 | inverse_packed_dense_uniform | inverse_ref_dense_uniform | isolated | 0 | 1.000 | 1.056 |
| 512 | inverse_packed_rank256_uniform | inverse_ref_rank256_uniform | batched | 0 | 1.001 | 1.166 |
| 512 | inverse_packed_rank256_uniform | inverse_ref_rank256_uniform | isolated | 0 | 0.999 | 1.137 |
| 512 | inverse_masked_dense_uniform | inverse_ref_dense_uniform | batched | 0 | 1.002 | 1.019 |
| 512 | inverse_masked_dense_uniform | inverse_ref_dense_uniform | isolated | 0 | 0.999 | 1.001 |
| 512 | inverse_masked_rank256_uniform | inverse_ref_rank256_uniform | batched | 0 | 0.993 | 1.022 |
| 512 | inverse_masked_rank256_uniform | inverse_ref_rank256_uniform | isolated | 0 | 0.991 | 1.010 |
| 8192 | inverse_packed_dense_uniform | inverse_ref_dense_uniform | batched | 0 | 1.004 | 1.021 |
| 8192 | inverse_packed_dense_uniform | inverse_ref_dense_uniform | isolated | 0 | 1.001 | 1.019 |
| 8192 | inverse_packed_rank256_uniform | inverse_ref_rank256_uniform | batched | 0 | 1.000 | 1.014 |
| 8192 | inverse_packed_rank256_uniform | inverse_ref_rank256_uniform | isolated | 0 | 1.002 | 1.020 |
| 8192 | inverse_masked_dense_uniform | inverse_ref_dense_uniform | batched | 0 | 1.002 | 1.004 |
| 8192 | inverse_masked_dense_uniform | inverse_ref_dense_uniform | isolated | 0 | 1.010 | 0.987 |
| 8192 | inverse_masked_rank256_uniform | inverse_ref_rank256_uniform | batched | 0 | 1.001 | 0.998 |
| 8192 | inverse_masked_rank256_uniform | inverse_ref_rank256_uniform | isolated | 0 | 1.003 | 1.000 |

Link tables use **8K isolated** API timing. Baseline cells use the faster complete before/after round independently for each objective. Each arm is fixed; no TEST-selected deployment policy or combined optimization is produced.

Calculated serial **precompressed**, dense_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Single-feature reference | Packed inverse | Masked search |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.064 | 5133.339 | 4219.425 | 4218.893 | 4219.814 |
| 5 | 1135.510 | 1045.542 | 866.839 | 866.306 | 867.227 |
| 10 | 569.441 | 534.567 | 447.765 | 447.233 | 448.154 |
| 25 | 229.799 | 227.983 | 196.321 | 195.789 | 196.710 |
| 50 | 116.586 | 125.788 | 112.507 | 111.974 | 112.895 |
| 75 | 78.848 | 91.723 | 84.568 | 84.036 | 84.957 |
| 100 | 59.979 | 74.690 | 70.599 | 70.067 | 70.988 |
| 200 | 31.675 | 49.141 | 49.646 | 49.113 | 50.034 |
| 300 | 22.241 | 40.625 | 42.661 | 42.129 | 43.050 |
| 400 | 17.524 | 36.367 | 39.169 | 38.636 | 39.557 |

Calculated serial **precompressed**, rank256_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Single-feature reference | Packed inverse | Masked search |
|---:|---:|---:|---:|---:|---:|
| 1 | 5664.064 | 5133.339 | 4393.355 | 4392.941 | 4393.360 |
| 5 | 1135.510 | 1045.542 | 895.949 | 895.535 | 895.954 |
| 10 | 569.441 | 534.567 | 458.773 | 458.359 | 458.779 |
| 25 | 229.799 | 227.983 | 196.467 | 196.054 | 196.473 |
| 50 | 116.586 | 125.788 | 109.032 | 108.619 | 109.038 |
| 75 | 78.848 | 91.723 | 79.887 | 79.474 | 79.893 |
| 100 | 59.979 | 74.690 | 65.315 | 64.901 | 65.320 |
| 200 | 31.675 | 49.141 | 43.456 | 43.042 | 43.462 |
| 300 | 22.241 | 40.625 | 36.170 | 35.756 | 36.175 |
| 400 | 17.524 | 36.367 | 32.527 | 32.113 | 32.532 |

Calculated serial **fresh**, dense_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Single-feature reference | Packed inverse | Masked search |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.368 | 5343.168 | 4262.260 | 4261.681 | 4262.239 |
| 5 | 1145.814 | 1255.370 | 909.674 | 909.095 | 909.652 |
| 10 | 579.745 | 744.396 | 490.600 | 490.021 | 490.579 |
| 25 | 240.103 | 437.811 | 239.156 | 238.577 | 239.135 |
| 50 | 126.889 | 335.616 | 155.342 | 154.763 | 155.320 |
| 75 | 89.152 | 301.551 | 127.404 | 126.824 | 127.382 |
| 100 | 70.283 | 284.519 | 113.434 | 112.855 | 113.413 |
| 200 | 41.979 | 258.970 | 92.481 | 91.902 | 92.459 |
| 300 | 32.545 | 250.454 | 85.496 | 84.917 | 85.475 |
| 400 | 27.827 | 246.196 | 82.004 | 81.425 | 81.982 |

Calculated serial **fresh**, rank256_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Single-feature reference | Packed inverse | Masked search |
|---:|---:|---:|---:|---:|---:|
| 1 | 5674.368 | 5343.168 | 4428.040 | 4427.554 | 4427.928 |
| 5 | 1145.814 | 1255.370 | 930.634 | 930.148 | 930.522 |
| 10 | 579.745 | 744.396 | 493.458 | 492.972 | 493.346 |
| 25 | 240.103 | 437.811 | 231.153 | 230.667 | 231.041 |
| 50 | 126.889 | 335.616 | 143.718 | 143.232 | 143.606 |
| 75 | 89.152 | 301.551 | 114.573 | 114.087 | 114.461 |
| 100 | 70.283 | 284.519 | 100.000 | 99.514 | 99.888 |
| 200 | 41.979 | 258.970 | 78.141 | 77.655 | 78.029 |
| 300 | 32.545 | 250.454 | 70.855 | 70.369 | 70.743 |
| 400 | 27.827 | 246.196 | 67.212 | 66.726 | 67.100 |

All 28 layers use our predictive/rANS codec. Original complete packets, predictor state, probability tables and BF16 words stay fixed. The original 57,688-byte coder state remains present. Packed inversion adds 40,960 bytes of coarse cutoffs and 4,608 bytes of sparse exceptions, for 45,568 additional shared device bytes and 103,256 total coder bytes. Masked search adds zero.

All own arms retain the measured single-feature rolling encoder and two decoder chunks. Only stage-one CDF inversion changes: packed inversion resolves ordinary buckets directly and uses sparse tables for complex buckets; masked search skips loads for already-converged lanes. All cumulative probabilities, state updates, word bounds and escapes remain unchanged. Full per-call Torch peaks include output and scratch; shared inverse tables are additional. Native allocator peaks outside Torch remain unmeasured.

Whole encode/decode calls preserve author timing: 10 warmups, 50 calls and five repeats. Separate isolated calls use two warmups and five samples. Cold setup, export, disk, upload and fitting remain outside warm measurements. Actual serialized encoder packets are restored in fresh processes, alternating 0/1/0 while retaining independent returned outputs.

512-token decoding keeps compact staging, graph replay and an owned output clone. 8K decoding borrows aligned compact input and allocates fresh output. Sender geometry, encoder arithmetic, coefficient tensors and probability tables stay fixed. Source, environments and native libraries are checked before and after the run.

Separate warmed profiles bind actual original/replacement decoder, predictor and feature-controller dispatch after ordinary timing. Summed work and stage samples are nonadditive. Two reused engineering documents and calculated link scenarios do not establish independent generalization, measured NIC/SSD performance, resident compressed attention or serving throughput.
