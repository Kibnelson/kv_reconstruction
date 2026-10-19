# Qwen KV benchmark results

Run status: **completed**. Model: `Qwen/Qwen3-4B-Instruct-2507`.
Selected device: `cuda:0`. Method results: 420 ok.

CPU-reference codecs retain their original CPU wire formats. Copying their encoded bytes to a device is staging; it does not create a GPU decoder. Tensor-native packets are separately named formats. `tensor_native_accelerator` reports actual device tensor execution; `tensor_native_cpu_functional` is a CPU functionality check and supplies no GPU performance evidence.

Full decoder state is included in the reported savings. Native packet rows identify whether their state charge covers actual tensor storage or complete serialized recipe state; original CPU formats charge their serialized decoder files. Those accounting contracts differ and are recorded explicitly. Sampled full-cache estimates are labeled; they are not full-cache measurements. Development and TEST are separated below and never averaged.

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
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 8/576 | 24.707% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 8/576 | 9.444% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 8/576 | 26.003% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 8/576 | 23.888% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 8/576 | 21.056% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 8/576 | 20.891% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 8/576 | -39.409% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 8/576 | 25.988% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 8/576 | 25.990% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 8/576 | 29.243% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 8/576 | 29.246% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 8/576 | 22.545% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 8/576 | 22.547% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 8/576 | 22.959% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 8/576 | 22.837% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 8/576 | 23.455% estimated | — |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 8/576 | 23.170% estimated | — |
| dev_orchard_sensor_log | 512 | raw | cpu_reference | ok | 8/576 | -0.016% estimated | 0.058 |
| dev_orchard_sensor_log | 512 | exponent_huffman_lower_deflate | cpu_reference | ok | 8/576 | 32.653% estimated | 212.011 |
| dev_orchard_sensor_log | 512 | native_raw_words | tensor_native_accelerator | ok | 8/576 | -0.238% estimated | 0.286 |
| dev_orchard_sensor_log | 512 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 8/576 | 24.267% estimated | 2.049 |
| dev_orchard_sensor_log | 512 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 8/576 | 24.477% estimated | 4.839 |
| dev_orchard_sensor_log | 512 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 8/576 | 8.623% estimated | 7.815 |
| dev_orchard_sensor_log | 512 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 8/576 | 21.827% estimated | 8.685 |
| dev_orchard_sensor_log | 512 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/576 | 20.607% estimated | 12.305 |
| dev_orchard_sensor_log | 512 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/576 | 21.302% estimated | 12.272 |
| dev_orchard_sensor_log | 512 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 8/576 | 21.560% estimated | 12.276 |
| dev_orchard_sensor_log | 512 | native_prefill_zero_coset | tensor_native_accelerator | ok | 8/576 | -3.687% estimated | 6.271 |
| dev_orchard_sensor_log | 512 | native_prefill_zero_residual | tensor_native_accelerator | ok | 8/576 | -3.688% estimated | 4.992 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 8/576 | 20.171% estimated | 6.126 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 8/576 | 20.169% estimated | 4.819 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_coset | tensor_native_accelerator | ok | 8/576 | 24.152% estimated | 6.319 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_residual | tensor_native_accelerator | ok | 8/576 | 24.150% estimated | 4.981 |
| dev_orchard_sensor_log | 512 | native_prefill_e8_coset | tensor_native_accelerator | ok | 8/576 | 20.710% estimated | 9.918 |
| dev_orchard_sensor_log | 512 | native_prefill_e8_residual | tensor_native_accelerator | ok | 8/576 | 20.709% estimated | 8.565 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 8/576 | 24.707% estimated | 4.706 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 8/576 | 9.444% estimated | 4.892 |
| dev_orchard_sensor_log | 512 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 8/576 | 26.003% estimated | 4.676 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 8/576 | 23.888% estimated | 5.083 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 8/576 | 21.056% estimated | 5.053 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 8/576 | 20.891% estimated | 5.092 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 8/576 | -39.409% estimated | 7.365 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 8/576 | 25.988% estimated | 6.875 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 8/576 | 25.990% estimated | 6.927 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 8/576 | 29.243% estimated | 6.966 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 8/576 | 29.246% estimated | 7.036 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 8/576 | 22.545% estimated | 7.322 |
| dev_orchard_sensor_log | 512 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 8/576 | 22.547% estimated | 7.358 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 8/576 | 22.959% estimated | 6.195 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 8/576 | 22.837% estimated | 6.193 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 8/576 | 23.455% estimated | 6.150 |
| dev_orchard_sensor_log | 512 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 8/576 | 23.170% estimated | 6.147 |

### Incremental representation diagnostics

Each row shows the largest observed savings change within one comparison family for that case. This is a diagnostic summary of measured candidates, not a policy selected for deployment. All matched comparisons, record components, state charges, and normalized timings remain in results.json. Positive changes mean fewer stored bytes; DEV and TEST remain separate.

| Case | Comparison | Baseline | Candidate | Saving change, percentage points | Record wins / ties / losses |
|---|---|---|---|---:|---:|
| dev_orchard_sensor_log | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.433 estimated | 1 / 0 / 7 |
| dev_orchard_sensor_log | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | -0.121 estimated | 4 / 0 / 4 |
| dev_orchard_sensor_log | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 4.979 estimated | 8 / 0 / 0 |
| dev_orchard_sensor_log | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 8 / 0 / 0 |
| dev_orchard_sensor_log | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.296 estimated | 8 / 0 / 0 |
| dev_orchard_sensor_log | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 16.560 estimated | 8 / 0 / 0 |
| dev_orchard_sensor_log | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 68.653 estimated | 8 / 0 / 0 |

Case `dev_orchard_sensor_log`: the **SIZE-ONLY stateless portfolio oracle** gives 29.296% estimated savings, 0.051 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

Record wins compare actual inner packet bytes and do not assign cache-wide state to individual records. Exponent-tier channel/time layouts preserve the same correction histogram and tier bit totals; small header differences do not establish an entropy improvement. Cross-family comparisons change multiple mechanisms.

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
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 24.845% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 9.559% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 26.249% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 24.208% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 21.271% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 20.932% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | -39.411% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 21.626% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 21.628% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 29.547% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 29.550% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 24.924% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 24.926% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.080% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.172% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.641% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.602% estimated | — |
| test_pottery_kiln_diary_2048 | 2048 | raw | cpu_reference | ok | 144/2304 | -0.016% estimated | 0.232 |
| test_pottery_kiln_diary_2048 | 2048 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/2304 | 32.594% estimated | 865.035 |
| test_pottery_kiln_diary_2048 | 2048 | native_raw_words | tensor_native_accelerator | ok | 144/2304 | -0.238% estimated | 1.011 |
| test_pottery_kiln_diary_2048 | 2048 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.208% estimated | 8.097 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.669% estimated | 19.360 |
| test_pottery_kiln_diary_2048 | 2048 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/2304 | 8.763% estimated | 31.237 |
| test_pottery_kiln_diary_2048 | 2048 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 22.052% estimated | 34.847 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 20.892% estimated | 50.556 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.507% estimated | 50.219 |
| test_pottery_kiln_diary_2048 | 2048 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.792% estimated | 50.272 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/2304 | -3.687% estimated | 24.976 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/2304 | -3.688% estimated | 19.537 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/2304 | 20.527% estimated | 24.363 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/2304 | 20.526% estimated | 18.968 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/2304 | 24.534% estimated | 25.306 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/2304 | 24.533% estimated | 19.832 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/2304 | 21.062% estimated | 39.286 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/2304 | 21.060% estimated | 34.225 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 24.845% estimated | 18.845 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 144/2304 | 9.559% estimated | 19.385 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 144/2304 | 26.249% estimated | 18.672 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 24.208% estimated | 20.450 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 144/2304 | 21.271% estimated | 20.022 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 144/2304 | 20.932% estimated | 20.451 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 144/2304 | -39.411% estimated | 29.275 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 21.626% estimated | 27.597 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 144/2304 | 21.628% estimated | 27.821 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 29.547% estimated | 27.713 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 144/2304 | 29.550% estimated | 27.958 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 24.924% estimated | 28.475 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 144/2304 | 24.926% estimated | 28.881 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.080% estimated | 24.625 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.172% estimated | 24.884 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.641% estimated | 24.675 |
| test_pottery_kiln_diary_2048 | 2048 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.602% estimated | 24.858 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/2304 | -3.687% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/2304 | -3.688% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/2304 | 20.563% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/2304 | 20.561% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/2304 | 24.559% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/2304 | 24.557% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/2304 | 21.097% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/2304 | 21.095% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 24.871% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 9.769% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 26.284% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 24.244% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 21.392% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 21.005% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | -39.411% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 21.417% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 21.419% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 29.552% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 29.554% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 24.883% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 24.885% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.109% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.205% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.650% estimated | — |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.600% estimated | — |
| test_python_dependency_walk_2048 | 2048 | raw | cpu_reference | ok | 144/2304 | -0.016% estimated | 0.239 |
| test_python_dependency_walk_2048 | 2048 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/2304 | 32.754% estimated | 839.844 |
| test_python_dependency_walk_2048 | 2048 | native_raw_words | tensor_native_accelerator | ok | 144/2304 | -0.238% estimated | 0.991 |
| test_python_dependency_walk_2048 | 2048 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.207% estimated | 8.249 |
| test_python_dependency_walk_2048 | 2048 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.683% estimated | 19.371 |
| test_python_dependency_walk_2048 | 2048 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/2304 | 8.777% estimated | 31.254 |
| test_python_dependency_walk_2048 | 2048 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 22.080% estimated | 35.406 |
| test_python_dependency_walk_2048 | 2048 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 20.886% estimated | 49.334 |
| test_python_dependency_walk_2048 | 2048 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.527% estimated | 49.916 |
| test_python_dependency_walk_2048 | 2048 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.803% estimated | 50.425 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/2304 | -3.687% estimated | 25.141 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/2304 | -3.688% estimated | 19.800 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/2304 | 20.563% estimated | 24.473 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/2304 | 20.561% estimated | 18.843 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/2304 | 24.559% estimated | 25.074 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/2304 | 24.557% estimated | 19.647 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/2304 | 21.097% estimated | 39.623 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/2304 | 21.095% estimated | 34.362 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 24.871% estimated | 18.766 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 144/2304 | 9.769% estimated | 19.513 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 144/2304 | 26.284% estimated | 18.833 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 24.244% estimated | 20.156 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 144/2304 | 21.392% estimated | 19.988 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 144/2304 | 21.005% estimated | 20.424 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 144/2304 | -39.411% estimated | 29.382 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 21.417% estimated | 27.729 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 144/2304 | 21.419% estimated | 27.620 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 29.552% estimated | 27.733 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 144/2304 | 29.554% estimated | 28.083 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 24.883% estimated | 29.011 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 144/2304 | 24.885% estimated | 29.519 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.109% estimated | 24.654 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.205% estimated | 24.658 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.650% estimated | 24.735 |
| test_python_dependency_walk_2048 | 2048 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.600% estimated | 24.714 |
| test_museum_loan_register_2048 | 2048 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/2304 | -3.687% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/2304 | -3.688% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/2304 | 20.613% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/2304 | 20.612% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/2304 | 24.684% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/2304 | 24.683% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/2304 | 21.142% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/2304 | 21.140% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 24.978% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 9.748% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 26.515% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 24.328% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 21.476% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 144/2304 | 21.184% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | -39.411% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 21.486% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 21.489% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 29.658% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 29.660% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 144/2304 | 25.009% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 144/2304 | 25.011% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.223% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.307% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.723% estimated | — |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 144/2304 | 23.664% estimated | — |
| test_museum_loan_register_2048 | 2048 | raw | cpu_reference | ok | 144/2304 | -0.016% estimated | 0.264 |
| test_museum_loan_register_2048 | 2048 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/2304 | 32.802% estimated | 841.950 |
| test_museum_loan_register_2048 | 2048 | native_raw_words | tensor_native_accelerator | ok | 144/2304 | -0.238% estimated | 1.000 |
| test_museum_loan_register_2048 | 2048 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.197% estimated | 8.133 |
| test_museum_loan_register_2048 | 2048 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 24.668% estimated | 19.294 |
| test_museum_loan_register_2048 | 2048 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/2304 | 8.771% estimated | 31.356 |
| test_museum_loan_register_2048 | 2048 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/2304 | 22.054% estimated | 34.816 |
| test_museum_loan_register_2048 | 2048 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 20.878% estimated | 50.335 |
| test_museum_loan_register_2048 | 2048 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.502% estimated | 50.095 |
| test_museum_loan_register_2048 | 2048 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/2304 | 21.773% estimated | 49.794 |
| test_museum_loan_register_2048 | 2048 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/2304 | -3.687% estimated | 24.998 |
| test_museum_loan_register_2048 | 2048 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/2304 | -3.688% estimated | 19.537 |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/2304 | 20.613% estimated | 24.456 |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/2304 | 20.612% estimated | 18.853 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/2304 | 24.684% estimated | 25.232 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/2304 | 24.683% estimated | 19.827 |
| test_museum_loan_register_2048 | 2048 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/2304 | 21.142% estimated | 39.483 |
| test_museum_loan_register_2048 | 2048 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/2304 | 21.140% estimated | 34.309 |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 24.978% estimated | 19.167 |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 144/2304 | 9.748% estimated | 19.513 |
| test_museum_loan_register_2048 | 2048 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 144/2304 | 26.515% estimated | 18.893 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 24.328% estimated | 20.206 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 144/2304 | 21.476% estimated | 20.048 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 144/2304 | 21.184% estimated | 20.462 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 144/2304 | -39.411% estimated | 29.358 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 21.486% estimated | 27.500 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 144/2304 | 21.489% estimated | 27.706 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 29.658% estimated | 28.008 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 144/2304 | 29.660% estimated | 27.976 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 144/2304 | 25.009% estimated | 29.317 |
| test_museum_loan_register_2048 | 2048 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 144/2304 | 25.011% estimated | 29.091 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.223% estimated | 24.619 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.307% estimated | 24.928 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.723% estimated | 25.192 |
| test_museum_loan_register_2048 | 2048 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 144/2304 | 23.664% estimated | 24.915 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/9216 | -3.687% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/9216 | -3.688% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/9216 | 20.368% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/9216 | 20.366% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/9216 | 24.598% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/9216 | 24.597% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/9216 | 20.891% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/9216 | 20.889% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 24.922% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 9.235% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 26.300% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 24.354% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 21.344% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 21.011% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | -39.409% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 21.498% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 21.501% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 29.641% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 29.643% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 25.022% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 25.025% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.155% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.277% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.798% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.759% estimated | — |
| test_pottery_kiln_diary_8192 | 8192 | raw | cpu_reference | ok | 144/9216 | -0.016% estimated | 0.238 |
| test_pottery_kiln_diary_8192 | 8192 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/9216 | 32.597% estimated | 838.727 |
| test_pottery_kiln_diary_8192 | 8192 | native_raw_words | tensor_native_accelerator | ok | 144/9216 | -0.238% estimated | 1.020 |
| test_pottery_kiln_diary_8192 | 8192 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/9216 | 24.180% estimated | 8.206 |
| test_pottery_kiln_diary_8192 | 8192 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 24.654% estimated | 19.449 |
| test_pottery_kiln_diary_8192 | 8192 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/9216 | 8.682% estimated | 31.459 |
| test_pottery_kiln_diary_8192 | 8192 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 22.034% estimated | 35.109 |
| test_pottery_kiln_diary_8192 | 8192 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 20.870% estimated | 50.138 |
| test_pottery_kiln_diary_8192 | 8192 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 21.495% estimated | 49.644 |
| test_pottery_kiln_diary_8192 | 8192 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 21.800% estimated | 50.066 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/9216 | -3.687% estimated | 24.916 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/9216 | -3.688% estimated | 19.569 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/9216 | 20.368% estimated | 24.522 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/9216 | 20.366% estimated | 19.143 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/9216 | 24.598% estimated | 25.151 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/9216 | 24.597% estimated | 19.815 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/9216 | 20.891% estimated | 39.328 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/9216 | 20.889% estimated | 34.295 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 24.922% estimated | 19.030 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 144/9216 | 9.235% estimated | 19.379 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 144/9216 | 26.300% estimated | 18.898 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 24.354% estimated | 20.267 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 144/9216 | 21.344% estimated | 19.947 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 144/9216 | 21.011% estimated | 20.534 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 144/9216 | -39.409% estimated | 29.306 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 21.498% estimated | 27.541 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 144/9216 | 21.501% estimated | 27.610 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 29.641% estimated | 27.822 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 144/9216 | 29.643% estimated | 27.987 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 25.022% estimated | 28.422 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 144/9216 | 25.025% estimated | 28.704 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.155% estimated | 25.816 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.277% estimated | 24.618 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.798% estimated | 24.721 |
| test_pottery_kiln_diary_8192 | 8192 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.759% estimated | 25.124 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/9216 | -3.687% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/9216 | -3.688% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/9216 | 20.416% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/9216 | 20.415% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/9216 | 24.612% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/9216 | 24.610% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/9216 | 20.940% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/9216 | 20.938% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 24.924% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 9.159% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 26.336% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 24.349% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 21.324% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 21.057% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | -39.410% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 21.268% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 21.270% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 29.625% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 29.627% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 24.997% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 25.000% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.177% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.282% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.760% estimated | — |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.729% estimated | — |
| test_python_dependency_walk_8192 | 8192 | raw | cpu_reference | ok | 144/9216 | -0.016% estimated | 0.248 |
| test_python_dependency_walk_8192 | 8192 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/9216 | 32.804% estimated | 836.264 |
| test_python_dependency_walk_8192 | 8192 | native_raw_words | tensor_native_accelerator | ok | 144/9216 | -0.238% estimated | 1.001 |
| test_python_dependency_walk_8192 | 8192 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/9216 | 24.189% estimated | 8.006 |
| test_python_dependency_walk_8192 | 8192 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 24.660% estimated | 19.438 |
| test_python_dependency_walk_8192 | 8192 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/9216 | 8.702% estimated | 31.421 |
| test_python_dependency_walk_8192 | 8192 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 22.029% estimated | 35.104 |
| test_python_dependency_walk_8192 | 8192 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 20.848% estimated | 49.646 |
| test_python_dependency_walk_8192 | 8192 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 21.509% estimated | 49.845 |
| test_python_dependency_walk_8192 | 8192 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 21.801% estimated | 50.440 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/9216 | -3.687% estimated | 25.218 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/9216 | -3.688% estimated | 19.495 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/9216 | 20.416% estimated | 24.557 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/9216 | 20.415% estimated | 19.063 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/9216 | 24.612% estimated | 25.481 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/9216 | 24.610% estimated | 19.854 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/9216 | 20.940% estimated | 39.939 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/9216 | 20.938% estimated | 34.439 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 24.924% estimated | 19.069 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 144/9216 | 9.159% estimated | 19.550 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 144/9216 | 26.336% estimated | 18.956 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 24.349% estimated | 20.272 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 144/9216 | 21.324% estimated | 19.832 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 144/9216 | 21.057% estimated | 20.431 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 144/9216 | -39.410% estimated | 29.399 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 21.268% estimated | 27.397 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 144/9216 | 21.270% estimated | 27.796 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 29.625% estimated | 27.682 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 144/9216 | 29.627% estimated | 28.103 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 24.997% estimated | 28.715 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 144/9216 | 25.000% estimated | 29.523 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.177% estimated | 24.756 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.282% estimated | 24.898 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.760% estimated | 25.121 |
| test_python_dependency_walk_8192 | 8192 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.729% estimated | 25.308 |
| test_museum_loan_register_8192 | 8192 | native_prefill_zero_coset | prefill_observer_accelerator | ok | 144/9216 | -3.687% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_zero_residual | prefill_observer_accelerator | ok | 144/9216 | -3.688% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_coset | prefill_observer_accelerator | ok | 144/9216 | 20.404% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_residual | prefill_observer_accelerator | ok | 144/9216 | 20.402% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_coset | prefill_observer_accelerator | ok | 144/9216 | 24.732% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_residual | prefill_observer_accelerator | ok | 144/9216 | 24.730% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_e8_coset | prefill_observer_accelerator | ok | 144/9216 | 20.924% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_e8_residual | prefill_observer_accelerator | ok | 144/9216 | 20.922% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 25.070% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_channel_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 8.973% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_time_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 26.603% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 24.467% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_channel_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 21.389% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_time_g64_residual | prefill_observer_accelerator | ok | 144/9216 | 21.258% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_zero_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | -39.409% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_fixed127_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 21.375% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_fixed127_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 21.377% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_median_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 29.748% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_median_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 29.750% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_affine_channel_tiered | prefill_observer_accelerator | ok | 144/9216 | 25.175% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_affine_time_tiered | prefill_observer_accelerator | ok | 144/9216 | 25.178% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary16_channel_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.286% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary64_channel_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.387% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary16_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.866% estimated | — |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary64_time_g8_residual | prefill_observer_accelerator | ok | 144/9216 | 23.811% estimated | — |
| test_museum_loan_register_8192 | 8192 | raw | cpu_reference | ok | 144/9216 | -0.016% estimated | 0.279 |
| test_museum_loan_register_8192 | 8192 | exponent_huffman_lower_deflate | cpu_reference | ok | 144/9216 | 32.856% estimated | 839.342 |
| test_museum_loan_register_8192 | 8192 | native_raw_words | tensor_native_accelerator | ok | 144/9216 | -0.238% estimated | 0.991 |
| test_museum_loan_register_8192 | 8192 | native_exponent_group_bitpack | tensor_native_accelerator | ok | 144/9216 | 24.167% estimated | 8.068 |
| test_museum_loan_register_8192 | 8192 | native_learned_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 24.668% estimated | 19.347 |
| test_museum_loan_register_8192 | 8192 | native_e8_symbol_bitpack | tensor_native_accelerator | ok | 144/9216 | 8.662% estimated | 31.150 |
| test_museum_loan_register_8192 | 8192 | native_learned_e8_residual_bitpack | tensor_native_accelerator | ok | 144/9216 | 22.039% estimated | 34.919 |
| test_museum_loan_register_8192 | 8192 | native_binary_identity_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 20.884% estimated | 49.494 |
| test_museum_loan_register_8192 | 8192 | native_binary_dense_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 21.522% estimated | 50.130 |
| test_museum_loan_register_8192 | 8192 | native_binary_hadamard_d64_bf16_bitpack | tensor_native_accelerator | ok | 144/9216 | 21.807% estimated | 50.604 |
| test_museum_loan_register_8192 | 8192 | native_prefill_zero_coset | tensor_native_accelerator | ok | 144/9216 | -3.687% estimated | 25.057 |
| test_museum_loan_register_8192 | 8192 | native_prefill_zero_residual | tensor_native_accelerator | ok | 144/9216 | -3.688% estimated | 19.614 |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_coset | tensor_native_accelerator | ok | 144/9216 | 20.404% estimated | 24.388 |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_residual | tensor_native_accelerator | ok | 144/9216 | 20.402% estimated | 18.896 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_coset | tensor_native_accelerator | ok | 144/9216 | 24.732% estimated | 25.320 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_residual | tensor_native_accelerator | ok | 144/9216 | 24.730% estimated | 19.837 |
| test_museum_loan_register_8192 | 8192 | native_prefill_e8_coset | tensor_native_accelerator | ok | 144/9216 | 20.924% estimated | 39.720 |
| test_museum_loan_register_8192 | 8192 | native_prefill_e8_residual | tensor_native_accelerator | ok | 144/9216 | 20.922% estimated | 34.282 |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 25.070% estimated | 19.116 |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_channel_g64_residual | tensor_native_accelerator | ok | 144/9216 | 8.973% estimated | 19.461 |
| test_museum_loan_register_8192 | 8192 | native_prefill_midpoint_time_g64_residual | tensor_native_accelerator | ok | 144/9216 | 26.603% estimated | 18.926 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 24.467% estimated | 20.267 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_channel_g64_residual | tensor_native_accelerator | ok | 144/9216 | 21.389% estimated | 19.897 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_time_g64_residual | tensor_native_accelerator | ok | 144/9216 | 21.258% estimated | 20.481 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_zero_channel_tiered | tensor_native_accelerator | ok | 144/9216 | -39.409% estimated | 29.176 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_fixed127_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 21.375% estimated | 27.528 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_fixed127_time_tiered | tensor_native_accelerator | ok | 144/9216 | 21.377% estimated | 28.091 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_median_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 29.748% estimated | 27.777 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_median_time_tiered | tensor_native_accelerator | ok | 144/9216 | 29.750% estimated | 28.093 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_affine_channel_tiered | tensor_native_accelerator | ok | 144/9216 | 25.175% estimated | 28.955 |
| test_museum_loan_register_8192 | 8192 | native_prefill_exponent_affine_time_tiered | tensor_native_accelerator | ok | 144/9216 | 25.178% estimated | 29.246 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary16_channel_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.286% estimated | 25.045 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary64_channel_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.387% estimated | 24.995 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary16_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.866% estimated | 25.100 |
| test_museum_loan_register_8192 | 8192 | native_prefill_affine_dictionary64_time_g8_residual | tensor_native_accelerator | ok | 144/9216 | 23.811% estimated | 24.795 |

### Incremental representation diagnostics

Each row shows the largest observed savings change within one comparison family for that case. This is a diagnostic summary of measured candidates, not a policy selected for deployment. All matched comparisons, record components, state charges, and normalized timings remain in results.json. Positive changes mean fewer stored bytes; DEV and TEST remain separate.

| Case | Comparison | Baseline | Candidate | Saving change, percentage points | Record wins / ties / losses |
|---|---|---|---|---:|---:|
| test_pottery_kiln_diary_2048 | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.567 estimated | 2 / 0 / 142 |
| test_pottery_kiln_diary_2048 | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | 0.093 estimated | 90 / 1 / 53 |
| test_pottery_kiln_diary_2048 | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 5.342 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_2048 | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_2048 | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.404 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_2048 | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 16.690 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_2048 | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 68.958 estimated | 144 / 0 / 0 |

Case `test_pottery_kiln_diary_2048`: the **SIZE-ONLY stateless portfolio oracle** gives 29.557% estimated savings, 0.008 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

| test_python_dependency_walk_2048 | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.593 estimated | 0 / 0 / 144 |
| test_python_dependency_walk_2048 | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | 0.095 estimated | 89 / 1 / 54 |
| test_python_dependency_walk_2048 | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 5.347 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_2048 | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_2048 | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.413 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_2048 | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 16.516 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_2048 | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 68.963 estimated | 144 / 0 / 0 |

Case `test_python_dependency_walk_2048`: the **SIZE-ONLY stateless portfolio oracle** gives 29.565% estimated savings, 0.011 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

| test_museum_loan_register_2048 | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.605 estimated | 1 / 0 / 143 |
| test_museum_loan_register_2048 | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | 0.084 estimated | 93 / 0 / 51 |
| test_museum_loan_register_2048 | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 5.463 estimated | 144 / 0 / 0 |
| test_museum_loan_register_2048 | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 144 / 0 / 0 |
| test_museum_loan_register_2048 | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.537 estimated | 144 / 0 / 0 |
| test_museum_loan_register_2048 | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 16.767 estimated | 144 / 0 / 0 |
| test_museum_loan_register_2048 | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 69.069 estimated | 144 / 0 / 0 |

Case `test_museum_loan_register_2048`: the **SIZE-ONLY stateless portfolio oracle** gives 29.662% estimated savings, 0.002 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

| test_pottery_kiln_diary_8192 | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.556 estimated | 3 / 0 / 141 |
| test_pottery_kiln_diary_8192 | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | 0.122 estimated | 90 / 0 / 54 |
| test_pottery_kiln_diary_8192 | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 5.462 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_8192 | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_8192 | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.378 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_8192 | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 17.065 estimated | 144 / 0 / 0 |
| test_pottery_kiln_diary_8192 | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 69.050 estimated | 144 / 0 / 0 |

Case `test_pottery_kiln_diary_8192`: the **SIZE-ONLY stateless portfolio oracle** gives 29.646% estimated savings, 0.003 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

| test_python_dependency_walk_8192 | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.589 estimated | 0 / 0 / 144 |
| test_python_dependency_walk_8192 | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | 0.105 estimated | 90 / 0 / 54 |
| test_python_dependency_walk_8192 | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 5.438 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_8192 | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_8192 | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.412 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_8192 | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 17.177 estimated | 144 / 0 / 0 |
| test_python_dependency_walk_8192 | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 69.035 estimated | 144 / 0 / 0 |

Case `test_python_dependency_walk_8192`: the **SIZE-ONLY stateless portfolio oracle** gives 29.632% estimated savings, 0.004 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

| test_museum_loan_register_8192 | Add residual dictionary | native_prefill_affine_time_g8_residual | native_prefill_affine_dictionary16_time_g8_residual | -0.601 estimated | 1 / 0 / 143 |
| test_museum_loan_register_8192 | dictionary_capacity_comparison | native_prefill_affine_dictionary16_channel_g8_residual | native_prefill_affine_dictionary64_channel_g8_residual | 0.101 estimated | 93 / 0 / 51 |
| test_museum_loan_register_8192 | Exponent baseline; multiple mechanisms | native_exponent_group_bitpack | native_prefill_exponent_median_time_tiered | 5.584 estimated | 144 / 0 / 0 |
| test_museum_loan_register_8192 | Correction coding | native_prefill_zero_residual | native_prefill_zero_coset | 0.002 estimated | 144 / 0 / 0 |
| test_museum_loan_register_8192 | Correction group size | native_prefill_midpoint_time_g8_residual | native_prefill_midpoint_time_g64_residual | 1.533 estimated | 144 / 0 / 0 |
| test_museum_loan_register_8192 | Correction layout | native_prefill_midpoint_channel_g64_residual | native_prefill_midpoint_time_g64_residual | 17.630 estimated | 144 / 0 / 0 |
| test_museum_loan_register_8192 | Predictor | native_prefill_exponent_zero_channel_tiered | native_prefill_exponent_median_channel_tiered | 69.157 estimated | 144 / 0 / 0 |

Case `test_museum_loan_register_8192`: the **SIZE-ONLY stateless portfolio oracle** gives 29.751% estimated savings, 0.001 percentage points beyond its best fixed stateless candidate. This is an upper bound over separately encoded packet choices under the declared byte contract, including one mode byte per record and calculated outer framing. It is not an implemented selector, repacked cache, or latency result; stateful candidates are excluded.

Record wins compare actual inner packet bytes and do not assign cache-wide state to individual records. Exponent-tier channel/time layouts preserve the same correction histogram and tier bit totals; small header differences do not establish an entropy improvement. Cross-family comparisons change multiple mechanisms.

## Recipes constructed during prefill

Rows with track `prefill_observer` build recipes directly from returned device KV between prefill chunks, before CPU export. Their encode/decode totals cover selected records and one exact verification; they are not repeated restore latency samples. Post-capture `native_packet` rows measure the same method separately. Tile-only recipes need no external dictionary; residual-dictionary recipes also require the frozen TRAIN decoder state, charged once under the row's stated contract. Instrumented capture prefill time includes all observer work. Full raw KV remains resident and compact packets are discarded after verification; this does not establish reduced model-cache memory or compact attention.

## Model continuation checks

These checks use the DEV prefix and development-selected candidates. Full-prefix reconstruction for a model check does not establish compact-resident attention.

| Candidate | Track | Status | Exact tokens | Exact logits | Exact appended KV |
|---|---|---|---|---|---|
| native_prefill_affine_time_g8_residual | native_packet | ok | True | True | True |
| native_prefill_exponent_median_time_tiered | native_packet | ok | True | True | True |
| native_prefill_affine_dictionary16_channel_g8_residual | native_packet | ok | True | True | True |

## Measurement scope

CPU encode/decode totals cover selected records. Repeated decode, transfer and buffered I/O microbenchmarks use a bounded representative sample. Transfer fields separate pageable/pinned raw copies, encoded-byte staging and CPU decode/host pack followed by a bulk copy. Native packet copies and decoding have separate fields in results.json and results.csv.

Device-event durations can include host launch gaps. Native peak allocator deltas exclude the pre-existing model/state baseline and are not total process or reserved-memory peaks. Buffered reads may hit the page cache; write+fsync/read tests do not establish physical SSD bandwidth. This run does not demonstrate attention integration, concurrency, admission capacity, TPOT or TTFT gains.

results.json contains all captured metadata, exactness flags, components, transfer timings and errors. results.csv contains one row per method measurement and flat numeric fields; repeated sample timing vectors remain in JSON. Nonfinite JSON numeric values are written as null.
