# Qwen KV benchmark results

Run status: **completed**. Model: `Qwen/Qwen3-4B-Instruct-2507`.
Selected device: `cuda:0`. Method results: 54 ok.

CPU-reference codecs retain their original CPU wire formats. Copying their encoded bytes to a device is staging; it does not create a GPU decoder. Tensor-native packets are separately named formats. `tensor_native_accelerator` reports actual device tensor execution; `tensor_native_cpu_functional` is a CPU functionality check and supplies no GPU performance evidence.

Full decoder state is included in the reported savings. Native packet state charges actual distinct tensor storage; original CPU formats charge their serialized decoder files. Those are different accounting contracts. Sampled full-cache estimates are labeled explicitly; they are not full-cache measurements. Development and TEST are separated below and never averaged.

## Development — selection and diagnostics only

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| dev_orchard_sensor_log | 512 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 8/576 | -3.687% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 8/576 | -3.688% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 8/576 | 20.171% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 8/576 | 20.169% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 8/576 | 24.152% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 8/576 | 24.150% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 8/576 | 20.710% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 8/576 | 20.709% estimated | — |
| dev_orchard_sensor_log | 512 | raw | cpu_reference | ok | 8/576 | -0.016% estimated | 0.057 |
| dev_orchard_sensor_log | 512 | exponent_huffman_lower_deflate | cpu_reference | ok | 8/576 | 32.653% estimated | 212.059 |
| dev_orchard_sensor_log | 512 | learned_entropy_l1 | cpu_reference | ok | 8/576 | 30.695% estimated | 782.493 |
| dev_orchard_sensor_log | 512 | native_raw_words | tensor_native_accelerator | ok | 8/576 | -0.238% estimated | 0.291 |
| dev_orchard_sensor_log | 512 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 8/576 | 24.267% estimated | 2.054 |
| dev_orchard_sensor_log | 512 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 8/576 | 24.477% estimated | 4.897 |
| dev_orchard_sensor_log | 512 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 8/576 | 8.623% estimated | 7.856 |
| dev_orchard_sensor_log | 512 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 8/576 | 21.827% estimated | 8.808 |
| dev_orchard_sensor_log | 512 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/576 | 20.607% estimated | 12.374 |
| dev_orchard_sensor_log | 512 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/576 | 21.302% estimated | 12.373 |
| dev_orchard_sensor_log | 512 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/576 | 21.560% estimated | 12.378 |
| dev_orchard_sensor_log | 512 | native_prefill_zero_coset | tensor_native_accelerator | ok | 8/576 | -3.687% estimated | 6.289 |
| dev_orchard_sensor_log | 512 | native_prefill_zero_residual | tensor_native_accelerator | ok | 8/576 | -3.688% estimated | 4.936 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 8/576 | 20.171% estimated | 6.242 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 8/576 | 20.169% estimated | 4.747 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_coset | tensor_native_accelerator | ok | 8/576 | 24.152% estimated | 6.287 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_residual | tensor_native_accelerator | ok | 8/576 | 24.150% estimated | 4.958 |
| dev_orchard_sensor_log | 512 | native_prefill_e8_coset | tensor_native_accelerator | ok | 8/576 | 20.710% estimated | 9.930 |
| dev_orchard_sensor_log | 512 | native_prefill_e8_residual | tensor_native_accelerator | ok | 8/576 | 20.709% estimated | 8.608 |

## TEST — frozen comparison

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/2304 | -3.687% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/2304 | -3.688% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/2304 | 20.527% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/2304 | 20.526% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/2304 | 24.534% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/2304 | 24.533% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/2304 | 21.062% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/2304 | 21.060% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | raw | cpu_reference | ok | 144/2304 | -0.016% estimated | 0.261 |
| test_pottery_kiln_diary_2048 | 2048 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/2304 | 32.594% estimated | 839.811 |
| test_pottery_kiln_diary_2048 | 2048 | learned_entropy_l1 | cpu_reference | ok | 144/2304 | 30.885% estimated | 3124.482 |
| test_pottery_kiln_diary_2048 | 2048 | native_raw_words | tensor_native_accelerator | ok | 144/2304 | -0.238% estimated | 1.027 |
| test_pottery_kiln_diary_2048 | 2048 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.208% estimated | 8.291 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.669% estimated | 19.675 |
| test_pottery_kiln_diary_2048 | 2048 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/2304 | 8.763% estimated | 31.497 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 22.052% estimated | 35.312 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 20.892% estimated | 50.037 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.507% estimated | 49.624 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.792% estimated | 49.661 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/2304 | -3.687% estimated | 25.307 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/2304 | -3.688% estimated | 19.535 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/2304 | 20.527% estimated | 24.685 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/2304 | 20.526% estimated | 19.040 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/2304 | 24.534% estimated | 25.512 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/2304 | 24.533% estimated | 19.861 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/2304 | 21.062% estimated | 40.072 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/2304 | 21.060% estimated | 34.661 |

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
