# Qwen KV benchmark results

Run status: **completed**. Model: `Qwen/Qwen3-4B-Instruct-2507`.
Selected device: `cuda:0`. Method results: 26 ok.

CPU-reference codecs retain their original CPU wire formats. Copying their encoded bytes to a device is staging; it does not create a GPU decoder. Tensor-native packets are separately named formats. `tensor_native_accelerator` reports actual device tensor execution; `tensor_native_cpu_functional` is a CPU functionality check and supplies no GPU performance evidence.

Full decoder state is included in the reported savings. Native packet state charges actual distinct tensor storage; original CPU formats charge their serialized decoder files. Those are different accounting contracts. Sampled full-cache estimates are labeled explicitly; they are not full-cache measurements. Development and TEST are separated below and never averaged.

## Development — selection and diagnostics only

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| dev_orchard_sensor_log | 512 | raw | cpu_reference | ok | 8/2304 | -0.064% estimated | 0.024 |
| dev_orchard_sensor_log | 512 | exponent_huffman_lower_deflate | cpu_reference | ok | 8/2304 | 32.360% estimated | 53.483 |
| dev_orchard_sensor_log | 512 | learned_entropy_l1 | cpu_reference | ok | 8/2304 | 30.551% estimated | 198.379 |
| dev_orchard_sensor_log | 512 | binary_identity_d64_bf16 | cpu_reference | ok | 8/2304 | 26.348% estimated | 203.066 |
| dev_orchard_sensor_log | 512 | binary_hadamard_d64_bf16 | cpu_reference | ok | 8/2304 | 26.384% estimated | 204.104 |
| dev_orchard_sensor_log | 512 | native_raw_words | tensor_native_accelerator | ok | 8/2304 | -0.949% estimated | 0.293 |
| dev_orchard_sensor_log | 512 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 8/2304 | 23.196% estimated | 2.073 |
| dev_orchard_sensor_log | 512 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 8/2304 | 23.288% estimated | 4.211 |
| dev_orchard_sensor_log | 512 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 8/2304 | 6.833% estimated | 6.954 |
| dev_orchard_sensor_log | 512 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.994% estimated | 7.861 |
| dev_orchard_sensor_log | 512 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.143% estimated | 11.861 |
| dev_orchard_sensor_log | 512 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.781% estimated | 11.859 |
| dev_orchard_sensor_log | 512 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.976% estimated | 11.879 |

## TEST — frozen comparison

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| test_pottery_kiln_diary_2048 | 2048 | raw | cpu_reference | ok | 72/9216 | -0.064% estimated | 0.253 |
| test_pottery_kiln_diary_2048 | 2048 | exponent_huffman_lower_deflate | cpu_reference | ok | 72/9216 | 32.345% estimated | 213.902 |
| test_pottery_kiln_diary_2048 | 2048 | learned_entropy_l1 | cpu_reference | ok | 72/9216 | 30.579% estimated | 794.465 |
| test_pottery_kiln_diary_2048 | 2048 | binary_identity_d64_bf16 | cpu_reference | ok | 72/9216 | 26.329% estimated | 817.983 |
| test_pottery_kiln_diary_2048 | 2048 | binary_hadamard_d64_bf16 | cpu_reference | ok | 72/9216 | 26.474% estimated | 821.088 |
| test_pottery_kiln_diary_2048 | 2048 | native_raw_words | tensor_native_accelerator | ok | 72/9216 | -0.949% estimated | 1.027 |
| test_pottery_kiln_diary_2048 | 2048 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 72/9216 | 23.109% estimated | 8.007 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 72/9216 | 23.328% estimated | 16.122 |
| test_pottery_kiln_diary_2048 | 2048 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 72/9216 | 6.972% estimated | 27.805 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 72/9216 | 20.119% estimated | 31.099 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 72/9216 | 19.284% estimated | 47.084 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 72/9216 | 19.894% estimated | 47.354 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 72/9216 | 20.166% estimated | 47.048 |

## Model continuation checks

These checks use the DEV prefix and development-selected candidates. Full-prefix reconstruction for a model check does not establish compact-resident attention.

| Candidate | Track | Status | Exact tokens | Exact logits | Exact appended KV |
|---|---|---|---|---|---|
| exponent_huffman_lower_deflate | cpu_reference | ok | True | True | True |
| native_learned_residual_bitpack | native_packet | ok | True | True | True |

## Measurement scope

CPU encode/decode totals cover selected records. Repeated decode, transfer and buffered I/O microbenchmarks use a bounded representative sample. Transfer fields separate pageable/pinned raw copies, encoded-byte staging and CPU decode/host pack followed by a bulk copy. Native packet copies and decoding have separate fields in results.json and results.csv.

Device-event durations can include host launch gaps. Native peak allocator deltas exclude the pre-existing model/state baseline and are not total process or reserved-memory peaks. Buffered reads may hit the page cache; write+fsync/read tests do not establish physical SSD bandwidth. This run does not demonstrate attention integration, concurrency, admission capacity, TPOT or TTFT gains.

results.json contains all captured metadata, exactness flags, components, transfer timings and errors. results.csv contains one row per method measurement and flat numeric fields; repeated sample timing vectors remain in JSON. Nonfinite JSON numeric values are written as null.
