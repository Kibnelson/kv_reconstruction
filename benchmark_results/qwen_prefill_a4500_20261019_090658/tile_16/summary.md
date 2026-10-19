# Qwen KV benchmark results

Run status: **completed**. Model: `Qwen/Qwen3-4B-Instruct-2507`.
Selected device: `cuda:0`. Method results: 54 ok.

CPU-reference codecs retain their original CPU wire formats. Copying their encoded bytes to a device is staging; it does not create a GPU decoder. Tensor-native packets are separately named formats. `tensor_native_accelerator` reports actual device tensor execution; `tensor_native_cpu_functional` is a CPU functionality check and supplies no GPU performance evidence.

Full decoder state is included in the reported savings. Native packet state charges actual distinct tensor storage; original CPU formats charge their serialized decoder files. Those are different accounting contracts. Sampled full-cache estimates are labeled explicitly; they are not full-cache measurements. Development and TEST are separated below and never averaged.

## Development — selection and diagnostics only

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| dev_orchard_sensor_log | 512 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 8/2304 | -5.368% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 8/2304 | -5.374% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 8/2304 | 18.243% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 8/2304 | 18.237% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 8/2304 | 19.652% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 8/2304 | 19.646% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 8/2304 | 20.987% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 8/2304 | 20.980% estimated | — |
| dev_orchard_sensor_log | 512 | raw | cpu_reference | ok | 8/2304 | -0.064% estimated | 0.022 |
| dev_orchard_sensor_log | 512 | exponent_huffman_lower_deflate | cpu_reference | ok | 8/2304 | 32.360% estimated | 53.822 |
| dev_orchard_sensor_log | 512 | learned_entropy_l1 | cpu_reference | ok | 8/2304 | 30.551% estimated | 197.664 |
| dev_orchard_sensor_log | 512 | native_raw_words | tensor_native_accelerator | ok | 8/2304 | -0.949% estimated | 0.287 |
| dev_orchard_sensor_log | 512 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 8/2304 | 23.196% estimated | 2.057 |
| dev_orchard_sensor_log | 512 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 8/2304 | 23.288% estimated | 4.078 |
| dev_orchard_sensor_log | 512 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 8/2304 | 6.833% estimated | 7.073 |
| dev_orchard_sensor_log | 512 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.994% estimated | 7.838 |
| dev_orchard_sensor_log | 512 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.143% estimated | 11.782 |
| dev_orchard_sensor_log | 512 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.781% estimated | 11.846 |
| dev_orchard_sensor_log | 512 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/2304 | 19.976% estimated | 11.926 |
| dev_orchard_sensor_log | 512 | native_prefill_zero_coset | tensor_native_accelerator | ok | 8/2304 | -5.368% estimated | 5.218 |
| dev_orchard_sensor_log | 512 | native_prefill_zero_residual | tensor_native_accelerator | ok | 8/2304 | -5.374% estimated | 4.040 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 8/2304 | 18.243% estimated | 5.447 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 8/2304 | 18.237% estimated | 4.180 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_coset | tensor_native_accelerator | ok | 8/2304 | 19.652% estimated | 5.586 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_residual | tensor_native_accelerator | ok | 8/2304 | 19.646% estimated | 4.473 |
| dev_orchard_sensor_log | 512 | native_prefill_e8_coset | tensor_native_accelerator | ok | 8/2304 | 20.987% estimated | 9.100 |
| dev_orchard_sensor_log | 512 | native_prefill_e8_residual | tensor_native_accelerator | ok | 8/2304 | 20.980% estimated | 7.939 |

## TEST — frozen comparison

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/9216 | -5.368% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/9216 | -5.374% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/9216 | 18.013% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/9216 | 18.007% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/9216 | 19.555% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/9216 | 19.549% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/9216 | 20.720% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/9216 | 20.714% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | raw | cpu_reference | ok | 144/9216 | -0.064% estimated | 0.128 |
| test_pottery_kiln_diary_2048 | 2048 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/9216 | 32.337% estimated | 210.537 |
| test_pottery_kiln_diary_2048 | 2048 | learned_entropy_l1 | cpu_reference | ok | 144/9216 | 30.576% estimated | 790.631 |
| test_pottery_kiln_diary_2048 | 2048 | native_raw_words | tensor_native_accelerator | ok | 144/9216 | -0.949% estimated | 1.031 |
| test_pottery_kiln_diary_2048 | 2048 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/9216 | 23.067% estimated | 7.945 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 23.313% estimated | 16.151 |
| test_pottery_kiln_diary_2048 | 2048 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/9216 | 6.934% estimated | 27.703 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 20.046% estimated | 31.562 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 19.252% estimated | 47.323 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 19.901% estimated | 47.031 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 20.204% estimated | 47.063 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/9216 | -5.368% estimated | 20.354 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/9216 | -5.374% estimated | 15.654 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/9216 | 18.013% estimated | 21.204 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/9216 | 18.007% estimated | 16.455 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/9216 | 19.555% estimated | 22.490 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/9216 | 19.549% estimated | 17.595 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/9216 | 20.720% estimated | 36.730 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/9216 | 20.714% estimated | 31.882 |

## Recipes constructed during prefill

Rows with track `prefill_observer` build recipes directly from returned device KV between prefill chunks, before CPU export. Their encode/decode totals cover selected records and one exact verification; they are not repeated restore latency samples. Post-capture `native_packet` rows measure the same method separately. Instrumented capture prefill time includes all observer work. Full raw KV remains resident and compact packets are discarded after verification; this does not establish reduced model-cache memory or compact attention.

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
