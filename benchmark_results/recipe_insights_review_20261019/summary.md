# Prefill recipe increments: same-capture diagnostic

Same-capture diagnostic only: all slices originate in one previously captured SmolLM request, historically labeled TEST. Token ranges are disjoint, but share a causal context and are not independent prompts or held-out model validation. Synthetic TRAIN/DEV/TEST labels apply only to this diagnostic. Only the three TRAIN ranges fit fresh dictionaries; DEV is reserved and unused; all selected candidates/settings are frozen before fitting, with TEST used only for exact record reconstruction and size analysis. Native tensor execution on CPU establishes functionality and measured byte sizes, never GPU throughput. No fresh forward pass, live-prefill timing, attention, concurrency, capacity gain or full cache container is measured. Raw KV is never saved in the output. Stateful recipes charge their full decoder state once per candidate cache. The portfolio analysis is a size-only oracle over already encoded stateless packets; it is not an implemented online prefill selector.

Status: **completed**. Device: `cpu`. Source SHA256: `349b763486ecf5b4942e04d3fea2c6c59e082a04350026c3e6b49962979dda2e`.

TRAIN slices: `[0,64)`, `[64,128)`, `[128,192)`. Reserved DEV: `[192,256)`. Measured TEST: `[256,512)`. Ranges denote source token positions; TEST record offsets are local to its slice.

## 64-token tiles

Savings include actual packet fields, JSON headers, calculated outer framing/index bytes, and complete state once. Full-cache savings are extrapolated when coverage is sampled.

| Representation | Full-cache savings | Coverage | Decode ms/MiB | Exact |
|---|---:|---|---:|---|
| native_raw_words | -1.262% | 240/240 | 0.724 | yes |
| native_exponent_group_bitpack | 22.786% | 240/240 | 52.834 | yes |
| native_learned_residual_bitpack | 20.783% | 240/240 | 82.870 | yes |
| native_e8_symbol_bitpack | 6.690% | 240/240 | 169.976 | yes |
| native_learned_e8_residual_bitpack | 17.153% | 240/240 | 130.224 | yes |
| native_binary_identity_d64_bf16_bitpack | 16.050% | 240/240 | 156.027 | yes |
| native_binary_dense_d64_bf16_bitpack | 16.711% | 240/240 | 138.237 | yes |
| native_binary_hadamard_d64_bf16_bitpack | 16.868% | 240/240 | 160.336 | yes |
| native_prefill_zero_coset | -6.108% | 240/240 | 90.138 | yes |
| native_prefill_zero_residual | -6.116% | 240/240 | 91.834 | yes |
| native_prefill_midpoint_coset | 19.142% | 240/240 | 75.277 | yes |
| native_prefill_midpoint_residual | 19.134% | 240/240 | 88.555 | yes |
| native_prefill_affine_coset | 22.242% | 240/240 | 99.451 | yes |
| native_prefill_affine_residual | 22.234% | 240/240 | 91.753 | yes |
| native_prefill_e8_coset | 18.920% | 240/240 | 111.556 | yes |
| native_prefill_e8_residual | 18.912% | 240/240 | 81.486 | yes |
| native_prefill_midpoint_time_g8_residual | 22.879% | 240/240 | 81.899 | yes |
| native_prefill_midpoint_channel_g64_residual | 12.443% | 240/240 | 83.237 | yes |
| native_prefill_midpoint_time_g64_residual | 24.274% | 240/240 | 74.566 | yes |
| native_prefill_affine_time_g8_residual | 22.160% | 240/240 | 65.837 | yes |
| native_prefill_affine_channel_g64_residual | 19.126% | 240/240 | 63.940 | yes |
| native_prefill_affine_time_g64_residual | 18.986% | 240/240 | 120.175 | yes |
| native_prefill_exponent_zero_channel_tiered | -40.996% | 240/240 | 109.131 | yes |
| native_prefill_exponent_fixed127_channel_tiered | 22.761% | 240/240 | 82.364 | yes |
| native_prefill_exponent_fixed127_time_tiered | 22.774% | 240/240 | 67.030 | yes |
| native_prefill_exponent_median_channel_tiered | 28.022% | 240/240 | 55.961 | yes |
| native_prefill_exponent_median_time_tiered | 28.034% | 240/240 | 61.406 | yes |
| native_prefill_exponent_affine_channel_tiered | 23.578% | 240/240 | 78.462 | yes |
| native_prefill_exponent_affine_time_tiered | 23.591% | 240/240 | 98.933 | yes |
| native_prefill_affine_dictionary16_channel_g8_residual | 17.980% | 240/240 | 76.434 | yes |
| native_prefill_affine_dictionary64_channel_g8_residual | 16.186% | 240/240 | 85.177 | yes |
| native_prefill_affine_dictionary16_time_g8_residual | 18.593% | 240/240 | 82.903 | yes |
| native_prefill_affine_dictionary64_time_g8_residual | 16.816% | 240/240 | 72.007 | yes |

Size-only stateless portfolio oracle: **28.030%** full-cache savings; -0.004 percentage points over the best fixed stateless candidate after its extra mode-byte charge. No selector implementation or selection latency is measured.

Matched incremental comparisons: 70; see `recipe_analysis.json` for paired record identities, component byte deltas, K/V breakdowns and state charges.

CPU decode timings are reference implementation costs and do not predict optimized CUDA kernel speed.
