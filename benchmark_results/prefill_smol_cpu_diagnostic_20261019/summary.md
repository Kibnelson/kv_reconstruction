# Qwen KV benchmark results

Run status: **completed**. Model: `HuggingFaceTB/SmolLM2-135M`.
Selected device: `cpu`. Method results: 24 ok.

CPU-reference codecs retain their original CPU wire formats. Copying their encoded bytes to a device is staging; it does not create a GPU decoder. Tensor-native packets are separately named formats. `tensor_native_accelerator` reports actual device tensor execution; `tensor_native_cpu_functional` is a CPU functionality check and supplies no GPU performance evidence.

Full decoder state is included in the reported savings. Native packet state charges actual distinct tensor storage; original CPU formats charge their serialized decoder files. Those are different accounting contracts. Sampled full-cache estimates are labeled explicitly; they are not full-cache measurements. Development and TEST are separated below and never averaged.

## Other cases — split diagnostic_reused_sample

| Case | Tokens | Method | Execution | Status | Records | Full-cache saving incl. state | Sample decode, ms |
|---|---:|---|---|---|---:|---:|---:|
| test_antarctic_story_512_tile16 | 512 | raw | cpu_reference | ok | 120/1920 | -0.342% estimated | — |
| test_antarctic_story_512_tile16 | 512 | exponent_huffman_lower_deflate | cpu_reference | ok | 120/1920 | 31.573% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_raw_words | tensor_native_cpu_functional | ok | 120/1920 | -5.811% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_exponent_group_bitpack | tensor_native_cpu_functional | ok | 120/1920 | 16.047% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_zero_coset | tensor_native_cpu_functional | ok | 120/1920 | -14.991% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_zero_residual | tensor_native_cpu_functional | ok | 120/1920 | -15.023% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_midpoint_coset | tensor_native_cpu_functional | ok | 120/1920 | 8.343% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_midpoint_residual | tensor_native_cpu_functional | ok | 120/1920 | 8.311% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_affine_coset | tensor_native_cpu_functional | ok | 120/1920 | 9.640% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_affine_residual | tensor_native_cpu_functional | ok | 120/1920 | 9.608% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_e8_coset | tensor_native_cpu_functional | ok | 120/1920 | 7.839% estimated | — |
| test_antarctic_story_512_tile16 | 512 | native_prefill_e8_residual | tensor_native_cpu_functional | ok | 120/1920 | 7.806% estimated | — |
| test_antarctic_story_512_tile64 | 512 | raw | cpu_reference | ok | 120/480 | -0.086% estimated | — |
| test_antarctic_story_512_tile64 | 512 | exponent_huffman_lower_deflate | cpu_reference | ok | 120/480 | 32.544% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_raw_words | tensor_native_cpu_functional | ok | 120/480 | -1.457% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_exponent_group_bitpack | tensor_native_cpu_functional | ok | 120/480 | 22.592% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_zero_coset | tensor_native_cpu_functional | ok | 120/480 | -6.108% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_zero_residual | tensor_native_cpu_functional | ok | 120/480 | -6.116% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_midpoint_coset | tensor_native_cpu_functional | ok | 120/480 | 19.169% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_midpoint_residual | tensor_native_cpu_functional | ok | 120/480 | 19.160% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_affine_coset | tensor_native_cpu_functional | ok | 120/480 | 22.052% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_affine_residual | tensor_native_cpu_functional | ok | 120/480 | 22.044% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_e8_coset | tensor_native_cpu_functional | ok | 120/480 | 18.955% estimated | — |
| test_antarctic_story_512_tile64 | 512 | native_prefill_e8_residual | tensor_native_cpu_functional | ok | 120/480 | 18.947% estimated | — |

## Model continuation checks

No model continuation check is recorded yet. This report does not infer one from cache roundtrips.

## Measurement scope

CPU encode/decode totals cover selected records. Repeated decode, transfer and buffered I/O microbenchmarks use a bounded representative sample. Transfer fields separate pageable/pinned raw copies, encoded-byte staging and CPU decode/host pack followed by a bulk copy. Native packet copies and decoding have separate fields in results.json and results.csv.

Device-event durations can include host launch gaps. Native peak allocator deltas exclude the pre-existing model/state baseline and are not total process or reserved-memory peaks. Buffered reads may hit the page cache; write+fsync/read tests do not establish physical SSD bandwidth. This run does not demonstrate attention integration, concurrency, admission capacity, TPOT or TTFT gains.

results.json contains all captured metadata, exactness flags, components, transfer timings and errors. results.csv contains one row per method measurement and flat numeric fields; repeated sample timing vectors remain in JSON. Nonfinite JSON numeric values are written as null.
