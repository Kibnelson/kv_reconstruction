# Standalone lossless feature reuse and decoder scheduling

**Status: completed.** GPU measured: **True**.

Standalone lossless GPU transfer codec: all 28 layers use our Dense or Rank256 predictor/rANS. Current rolling encoding is the reference. Separate fixed arms reuse one temporary INT8 feature matrix or partition 8K decoding into one, four or eight token chunks instead of two. They are not combined. Original complete packets, fitted coefficients, arithmetic, binary CDF search and BF16 words stay fixed. SplitZip and ExANS remain separate unchanged controls, not candidate components. 512-token graph execution and 8K borrowed-input direct execution are retained. The two documents are reused engineering evidence, not independent generalization. Link costs are calculated, not NIC/SSD, resident P or serving measurements.

Timing: **batched**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 14.162 | 3.497 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 13.384 | 3.423 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.366 | 3.414 |
| 512 | raw_bf16 | after | 16.00009 | 0.323 | 0.229 |
| 512 | raw_bf16 | before | 16.00009 | 0.323 | 0.229 |
| 512 | schedule_feature_dense_uniform | paired | 9.11376 | 5.169 | 4.136 |
| 512 | schedule_feature_rank256_uniform | paired | 9.50504 | 5.128 | 4.572 |
| 512 | schedule_ref_dense_uniform | paired | 9.11376 | 5.234 | 4.140 |
| 512 | schedule_ref_rank256_uniform | paired | 9.50504 | 5.211 | 4.559 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.734 | 0.309 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.731 | 0.307 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 193.153 | 23.019 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 185.693 | 23.377 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 185.346 | 23.358 |
| 8192 | raw_bf16 | after | 16.00001 | 4.890 | 3.318 |
| 8192 | raw_bf16 | before | 16.00001 | 4.890 | 3.317 |
| 8192 | schedule_chunks1_dense_uniform | paired | 8.92097 | 45.080 | 32.690 |
| 8192 | schedule_chunks1_rank256_uniform | paired | 9.30632 | 36.107 | 23.551 |
| 8192 | schedule_chunks4_dense_uniform | paired | 8.92097 | 44.954 | 32.052 |
| 8192 | schedule_chunks4_rank256_uniform | paired | 9.30632 | 36.115 | 23.557 |
| 8192 | schedule_chunks8_dense_uniform | paired | 8.92097 | 45.103 | 33.639 |
| 8192 | schedule_chunks8_rank256_uniform | paired | 9.30632 | 36.183 | 24.952 |
| 8192 | schedule_feature_dense_uniform | paired | 8.92097 | 45.041 | 32.028 |
| 8192 | schedule_feature_rank256_uniform | paired | 9.30632 | 36.127 | 23.071 |
| 8192 | schedule_ref_dense_uniform | paired | 8.92097 | 45.070 | 32.040 |
| 8192 | schedule_ref_rank256_uniform | paired | 9.30632 | 36.126 | 23.025 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.296 | 3.187 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.238 | 3.188 |

Timing: **isolated**.

| Tokens | Method | Round | Complete bpv | Encode ms | Decode ms |
|---:|---|---|---:|---:|---:|
| 512 | exans_official_compact_framed | paired | 10.99741 | 15.262 | 3.772 |
| 512 | exans_official_fixed_framed | after | 12.37970 | 14.706 | 3.643 |
| 512 | exans_official_fixed_framed | before | 12.37970 | 13.959 | 3.660 |
| 512 | raw_bf16 | after | 16.00009 | 0.415 | 0.343 |
| 512 | raw_bf16 | before | 16.00009 | 0.408 | 0.340 |
| 512 | schedule_feature_dense_uniform | paired | 9.11376 | 5.508 | 4.503 |
| 512 | schedule_feature_rank256_uniform | paired | 9.50504 | 5.566 | 4.573 |
| 512 | schedule_ref_dense_uniform | paired | 9.11376 | 5.659 | 4.467 |
| 512 | schedule_ref_rank256_uniform | paired | 9.50504 | 5.638 | 5.256 |
| 512 | splitzip_official_framed | after | 12.04281 | 0.871 | 0.524 |
| 512 | splitzip_official_framed | before | 12.04281 | 0.869 | 0.525 |
| 8192 | exans_official_compact_framed | paired | 10.87731 | 209.143 | 23.596 |
| 8192 | exans_official_fixed_framed | after | 12.34020 | 193.891 | 23.659 |
| 8192 | exans_official_fixed_framed | before | 12.34020 | 194.402 | 23.613 |
| 8192 | raw_bf16 | after | 16.00001 | 4.978 | 3.426 |
| 8192 | raw_bf16 | before | 16.00001 | 4.973 | 3.421 |
| 8192 | schedule_chunks1_dense_uniform | paired | 8.92097 | 42.941 | 30.384 |
| 8192 | schedule_chunks1_rank256_uniform | paired | 9.30632 | 34.653 | 22.565 |
| 8192 | schedule_chunks4_dense_uniform | paired | 8.92097 | 42.593 | 29.640 |
| 8192 | schedule_chunks4_rank256_uniform | paired | 9.30632 | 34.612 | 22.117 |
| 8192 | schedule_chunks8_dense_uniform | paired | 8.92097 | 42.707 | 29.457 |
| 8192 | schedule_chunks8_rank256_uniform | paired | 9.30632 | 34.599 | 26.286 |
| 8192 | schedule_feature_dense_uniform | paired | 8.92097 | 42.727 | 28.712 |
| 8192 | schedule_feature_rank256_uniform | paired | 9.30632 | 34.662 | 21.620 |
| 8192 | schedule_ref_dense_uniform | paired | 8.92097 | 42.730 | 29.048 |
| 8192 | schedule_ref_rank256_uniform | paired | 9.30632 | 34.675 | 21.612 |
| 8192 | splitzip_official_framed | after | 12.05013 | 10.371 | 3.376 |
| 8192 | splitzip_official_framed | before | 12.05013 | 10.323 | 3.390 |

Incremental Torch API peaks: mean of each document's median over five isolated calls. Output and scratch are included; native allocations outside Torch are unmeasured.

| Tokens | Method | Round | Encode peak MiB | Decode peak MiB |
|---:|---|---|---:|---:|
| 512 | exans_official_compact_framed | paired | 143.564 | 56.234 |
| 512 | exans_official_fixed_framed | after | 144.236 | 56.235 |
| 512 | exans_official_fixed_framed | before | 144.236 | 56.235 |
| 512 | raw_bf16 | after | 56.000 | 56.000 |
| 512 | raw_bf16 | before | 56.000 | 56.000 |
| 512 | schedule_feature_dense_uniform | paired | 144.028 | 56.000 |
| 512 | schedule_feature_rank256_uniform | paired | 145.633 | 56.000 |
| 512 | schedule_ref_dense_uniform | paired | 148.320 | 56.000 |
| 512 | schedule_ref_rank256_uniform | paired | 148.445 | 56.000 |
| 512 | splitzip_official_framed | after | 84.628 | 56.513 |
| 512 | splitzip_official_framed | before | 84.628 | 56.513 |
| 8192 | exans_official_compact_framed | paired | 2279.953 | 896.891 |
| 8192 | exans_official_fixed_framed | after | 2280.902 | 896.898 |
| 8192 | exans_official_fixed_framed | before | 2280.902 | 896.898 |
| 8192 | raw_bf16 | after | 896.000 | 896.000 |
| 8192 | raw_bf16 | before | 896.000 | 896.000 |
| 8192 | schedule_chunks1_dense_uniform | paired | 2357.125 | 896.000 |
| 8192 | schedule_chunks1_rank256_uniform | paired | 2359.125 | 896.000 |
| 8192 | schedule_chunks4_dense_uniform | paired | 2357.125 | 896.000 |
| 8192 | schedule_chunks4_rank256_uniform | paired | 2359.125 | 896.000 |
| 8192 | schedule_chunks8_dense_uniform | paired | 2357.125 | 896.000 |
| 8192 | schedule_chunks8_rank256_uniform | paired | 2359.125 | 896.000 |
| 8192 | schedule_feature_dense_uniform | paired | 2293.955 | 896.000 |
| 8192 | schedule_feature_rank256_uniform | paired | 2315.232 | 896.000 |
| 8192 | schedule_ref_dense_uniform | paired | 2357.125 | 896.000 |
| 8192 | schedule_ref_rank256_uniform | paired | 2359.125 | 896.000 |
| 8192 | splitzip_official_framed | after | 1354.864 | 905.623 |
| 8192 | splitzip_official_framed | before | 1354.864 | 905.623 |

Each fixed arm versus the current rolling reference. Speedup below 1 means slower.

| Tokens | Method | Reference | Basis | Wire delta bytes | Encode speedup | Decode speedup |
|---:|---|---|---|---:|---:|---:|
| 512 | schedule_feature_dense_uniform | schedule_ref_dense_uniform | batched | 0 | 1.012 | 1.001 |
| 512 | schedule_feature_dense_uniform | schedule_ref_dense_uniform | isolated | 0 | 1.027 | 0.992 |
| 512 | schedule_feature_rank256_uniform | schedule_ref_rank256_uniform | batched | 0 | 1.016 | 0.997 |
| 512 | schedule_feature_rank256_uniform | schedule_ref_rank256_uniform | isolated | 0 | 1.013 | 1.149 |
| 8192 | schedule_feature_dense_uniform | schedule_ref_dense_uniform | batched | 0 | 1.001 | 1.000 |
| 8192 | schedule_feature_dense_uniform | schedule_ref_dense_uniform | isolated | 0 | 1.000 | 1.012 |
| 8192 | schedule_feature_rank256_uniform | schedule_ref_rank256_uniform | batched | 0 | 1.000 | 0.998 |
| 8192 | schedule_feature_rank256_uniform | schedule_ref_rank256_uniform | isolated | 0 | 1.000 | 1.000 |
| 8192 | schedule_chunks1_dense_uniform | schedule_ref_dense_uniform | batched | 0 | 1.000 | 0.980 |
| 8192 | schedule_chunks1_dense_uniform | schedule_ref_dense_uniform | isolated | 0 | 0.995 | 0.956 |
| 8192 | schedule_chunks1_rank256_uniform | schedule_ref_rank256_uniform | batched | 0 | 1.001 | 0.978 |
| 8192 | schedule_chunks1_rank256_uniform | schedule_ref_rank256_uniform | isolated | 0 | 1.001 | 0.958 |
| 8192 | schedule_chunks4_dense_uniform | schedule_ref_dense_uniform | batched | 0 | 1.003 | 1.000 |
| 8192 | schedule_chunks4_dense_uniform | schedule_ref_dense_uniform | isolated | 0 | 1.003 | 0.980 |
| 8192 | schedule_chunks4_rank256_uniform | schedule_ref_rank256_uniform | batched | 0 | 1.000 | 0.977 |
| 8192 | schedule_chunks4_rank256_uniform | schedule_ref_rank256_uniform | isolated | 0 | 1.002 | 0.977 |
| 8192 | schedule_chunks8_dense_uniform | schedule_ref_dense_uniform | batched | 0 | 0.999 | 0.952 |
| 8192 | schedule_chunks8_dense_uniform | schedule_ref_dense_uniform | isolated | 0 | 1.001 | 0.986 |
| 8192 | schedule_chunks8_rank256_uniform | schedule_ref_rank256_uniform | batched | 0 | 0.998 | 0.923 |
| 8192 | schedule_chunks8_rank256_uniform | schedule_ref_rank256_uniform | isolated | 0 | 1.002 | 0.822 |

Link tables use **8K isolated** API timing. Baseline cells use the faster complete before/after round independently for each objective. Each arm is fixed; no TEST-selected deployment policy or combined optimization is produced.

Calculated serial **precompressed**, dense_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Rolling reference | Single feature | Chunks 1 | Chunks 4 | Chunks 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5664.068 | 5133.342 | 4219.781 | 4219.446 | 4221.117 | 4220.374 | 4220.190 |
| 5 | 1135.515 | 1045.545 | 867.194 | 866.859 | 868.530 | 867.787 | 867.604 |
| 10 | 569.446 | 534.570 | 448.121 | 447.786 | 449.457 | 448.714 | 448.530 |
| 25 | 229.804 | 227.986 | 196.677 | 196.342 | 198.013 | 197.270 | 197.086 |
| 50 | 116.590 | 125.791 | 112.862 | 112.527 | 114.198 | 113.455 | 113.272 |
| 75 | 78.852 | 91.726 | 84.924 | 84.589 | 86.260 | 85.517 | 85.333 |
| 100 | 59.983 | 74.693 | 70.955 | 70.620 | 72.291 | 71.548 | 71.364 |
| 200 | 31.680 | 49.145 | 50.001 | 49.666 | 51.337 | 50.594 | 50.411 |
| 300 | 22.245 | 40.628 | 43.017 | 42.681 | 44.353 | 43.609 | 43.426 |
| 400 | 17.528 | 36.370 | 39.525 | 39.189 | 40.861 | 40.117 | 39.934 |

Calculated serial **precompressed**, rank256_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Rolling reference | Single feature | Chunks 1 | Chunks 4 | Chunks 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5664.068 | 5133.342 | 4393.369 | 4393.378 | 4394.323 | 4393.874 | 4398.043 |
| 5 | 1135.515 | 1045.545 | 895.963 | 895.972 | 896.917 | 896.468 | 900.637 |
| 10 | 569.446 | 534.570 | 458.788 | 458.796 | 459.741 | 459.293 | 463.461 |
| 25 | 229.804 | 227.986 | 196.482 | 196.490 | 197.436 | 196.987 | 201.156 |
| 50 | 116.590 | 125.791 | 109.047 | 109.055 | 110.001 | 109.552 | 113.721 |
| 75 | 78.852 | 91.726 | 79.902 | 79.910 | 80.855 | 80.407 | 84.576 |
| 100 | 59.983 | 74.693 | 65.330 | 65.338 | 66.283 | 65.835 | 70.003 |
| 200 | 31.680 | 49.145 | 43.471 | 43.479 | 44.424 | 43.976 | 48.144 |
| 300 | 22.245 | 40.628 | 36.185 | 36.193 | 37.138 | 36.689 | 40.858 |
| 400 | 17.528 | 36.370 | 32.541 | 32.550 | 33.495 | 33.046 | 37.215 |

Calculated serial **fresh**, dense_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Rolling reference | Single feature | Chunks 1 | Chunks 4 | Chunks 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5674.405 | 5342.485 | 4262.511 | 4262.173 | 4264.058 | 4262.967 | 4262.897 |
| 5 | 1145.852 | 1254.688 | 909.924 | 909.586 | 911.472 | 910.380 | 910.310 |
| 10 | 579.782 | 743.714 | 490.851 | 490.513 | 492.398 | 491.307 | 491.237 |
| 25 | 240.141 | 437.129 | 239.407 | 239.069 | 240.954 | 239.863 | 239.793 |
| 50 | 126.927 | 334.934 | 155.592 | 155.254 | 157.140 | 156.048 | 155.978 |
| 75 | 89.189 | 300.869 | 127.654 | 127.316 | 129.202 | 128.110 | 128.040 |
| 100 | 70.320 | 283.836 | 113.685 | 113.347 | 115.232 | 114.141 | 114.071 |
| 200 | 42.017 | 258.288 | 92.731 | 92.393 | 94.279 | 93.187 | 93.117 |
| 300 | 32.582 | 249.771 | 85.747 | 85.408 | 87.294 | 86.203 | 86.133 |
| 400 | 27.865 | 245.513 | 82.254 | 81.916 | 83.802 | 82.711 | 82.641 |

Calculated serial **fresh**, rank256_uniform, milliseconds.

| Gb/s | SplitZip | Compact ExANS | Rolling reference | Single feature | Chunks 1 | Chunks 4 | Chunks 8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5674.405 | 5342.485 | 4428.044 | 4428.040 | 4428.976 | 4428.486 | 4432.642 |
| 5 | 1145.852 | 1254.688 | 930.638 | 930.634 | 931.570 | 931.080 | 935.236 |
| 10 | 579.782 | 743.714 | 493.463 | 493.458 | 494.394 | 493.904 | 498.060 |
| 25 | 240.141 | 437.129 | 231.157 | 231.152 | 232.089 | 231.599 | 235.755 |
| 50 | 126.927 | 334.934 | 143.722 | 143.717 | 144.654 | 144.164 | 148.319 |
| 75 | 89.189 | 300.869 | 114.577 | 114.572 | 115.509 | 115.019 | 119.174 |
| 100 | 70.320 | 283.836 | 100.004 | 100.000 | 100.936 | 100.446 | 104.602 |
| 200 | 42.017 | 258.288 | 78.146 | 78.141 | 79.077 | 78.587 | 82.743 |
| 300 | 32.582 | 249.771 | 70.859 | 70.855 | 71.791 | 71.301 | 75.457 |
| 400 | 27.865 | 245.513 | 67.216 | 67.212 | 68.148 | 67.658 | 71.814 |

All 28 layers use our predictive/rANS codec. Complete original packets, predictor state, 57,688-byte coder tables and BF16 words stay fixed. The packet header records the unchanged sender configuration with two chunks; decoder chunk overrides are explicit execution receipts. Additional shared device tensor bytes are zero.

The single-feature arm generates the next required source-word features immediately before each prediction product, reusing one temporary INT8 matrix. Decoder chunk arms retain the measured rolling encoder and change only the token-row partition. These changes are not combined. Full per-call Torch allocation peaks include output and scratch; stream/native allocator peaks outside Torch remain unmeasured.

Whole encode/decode calls preserve author timing: 10 warmups, 50 calls and five repeats. Separate isolated calls use two warmups and five samples. Cold setup, export, disk, upload and fitting remain outside warm measurements. Actual serialized encoder packets are restored in fresh processes, alternating 0/1/0 while retaining independent returned outputs.

512-token decoding keeps compact staging, graph replay and an owned output clone. 8K decoding borrows aligned compact input and allocates fresh output. Chunk arms are measured only at 8K; sender geometry and encoder arithmetic stay fixed. Source, environments and native libraries are checked before and after the run.

Separate warmed profiles bind actual product, decoder kernel and feature-controller counts to each declared schedule. Summed kernel work and stage samples are nonadditive. Two reused engineering documents and calculated link scenarios do not establish independent generalization, measured NIC/SSD performance, resident compressed attention or serving throughput.
