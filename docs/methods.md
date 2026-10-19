# Method inventory

The CPU and native tracks have separate formats and timing boundaries. All retained corrections target exact BF16 word recovery.

## CPU reference methods

| Name | Family | Fixed options |
|---|---|---|
| `raw` | raw | `{"shuffle": false}` |
| `raw_paged` | simple | `{"method": "raw"}` |
| `raw_deflate` | wrap | `{"parent": "raw"}` |
| `byte_shuffle` | raw | `{"shuffle": true}` |
| `byte_shuffle_deflate` | wrap | `{"parent": "byte_shuffle"}` |
| `exponent` | simple | `{"method": "exponent"}` |
| `zero_coset` | simple | `{"method": "zero_coset"}` |
| `lattice2_coset` | simple | `{"method": "lattice2_coset"}` |
| `lattice4_coset` | simple | `{"method": "lattice4_coset"}` |
| `lattice2_sign_coset` | simple | `{"method": "lattice2_sign_coset"}` |
| `exponent_deflate` | wrap | `{"parent": "exponent"}` |
| `field_packed` | field | `{"strategy": "packed"}` |
| `field_packed_deflate` | field | `{"strategy": "packed_deflate"}` |
| `field_deflate` | field | `{"strategy": "deflate"}` |
| `field_best` | field | `{"strategy": "best"}` |
| `field_best_deflate` | wrap | `{"parent": "field_best"}` |
| `exponent_huffman` | huffman | `{}` |
| `exponent_huffman_deflate` | wrap | `{"parent": "exponent_huffman"}` |
| `exponent_huffman_lower_deflate` | lower_control | `{"strategy": "deflate"}` |
| `exponent_huffman_lower_split_deflate` | lower_control | `{"strategy": "split_deflate"}` |
| `exponent_huffman_lower_huffman` | lower_control | `{"strategy": "huffman"}` |
| `temporal_previous_token` | temporal | `{"predictor": "previous_token"}` |
| `temporal_previous_channel` | temporal | `{"predictor": "previous_channel"}` |
| `temporal_zero` | temporal | `{"predictor": "zero"}` |
| `temporal_midpoint` | temporal | `{"predictor": "midpoint"}` |
| `temporal_previous_token_scaled` | temporal | `{"predictor": "previous_token_scaled"}` |
| `lower16` | lower_learned | `{"force": false, "model": "lower16"}` |
| `lower16_forced` | lower_learned | `{"force": true, "model": "lower16"}` |
| `lower16_conditional` | lower_learned | `{"force": false, "model": "lower16_conditional"}` |
| `lower16_conditional_forced` | lower_learned | `{"force": true, "model": "lower16_conditional"}` |
| `learned16` | dictionary | `{"model": "learned16"}` |
| `learned64` | dictionary | `{"model": "learned64"}` |
| `learned256` | dictionary | `{"model": "learned256"}` |
| `fixed64` | dictionary | `{"model": "fixed64"}` |
| `learned64_deflate` | wrap | `{"parent": "learned64"}` |
| `hybrid_group` | hybrid | `{"model": "learned64"}` |
| `hybrid_tile` | hybrid_tile | `{"model": "learned64"}` |
| `e8_float` | original_e8 | `{"method": "e8_float"}` |
| `e8_symbol` | original_e8 | `{"method": "e8_symbol"}` |
| `direct_entropy` | direct | `{}` |
| `learned_coset_radius` | residual | `{"assignment": "radius", "correction": "coset", "model": "learned64"}` |
| `learned_entropy_radius` | residual | `{"assignment": "radius", "correction": "best_entropy", "model": "learned64"}` |
| `learned_best_radius` | residual | `{"assignment": "radius", "correction": "best", "model": "learned64"}` |
| `learned_entropy_radius_deflate` | wrap | `{"parent": "learned_entropy_radius"}` |
| `learned_coset_l1` | residual | `{"assignment": "l1", "correction": "coset", "model": "learned64"}` |
| `learned_entropy_l1` | residual | `{"assignment": "l1", "correction": "best_entropy", "model": "learned64"}` |
| `learned_best_l1` | residual | `{"assignment": "l1", "correction": "best", "model": "learned64"}` |
| `learned_entropy_l1_deflate` | wrap | `{"parent": "learned_entropy_l1"}` |
| `learned_e8_coset` | residual_e8 | `{"correction": "coset", "family": "e8", "model": "learned64", "step_log2": 10}` |
| `learned_e8_coset_deflate` | wrap | `{"parent": "learned_e8_coset"}` |
| `learned_e8_entropy` | residual_e8 | `{"correction": "best", "family": "e8", "model": "learned64", "step_log2": 10}` |
| `learned_e8_entropy_deflate` | wrap | `{"parent": "learned_e8_entropy"}` |
| `learned_scalar_coset` | residual_e8 | `{"correction": "coset", "family": "scalar", "model": "learned64", "step_log2": 10}` |
| `learned_scalar_coset_deflate` | wrap | `{"parent": "learned_scalar_coset"}` |
| `learned_scalar_entropy` | residual_e8 | `{"correction": "best", "family": "scalar", "model": "learned64", "step_log2": 10}` |
| `learned_scalar_entropy_deflate` | wrap | `{"parent": "learned_scalar_entropy"}` |
| `binary_identity_d8_f32` | binary | `{"amplitude_precision": "f32", "dimension": 8, "model": "learned64", "transform": "identity"}` |
| `binary_identity_d8_f32_deflate` | wrap | `{"parent": "binary_identity_d8_f32"}` |
| `binary_identity_d8_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 8, "model": "learned64", "transform": "identity"}` |
| `binary_identity_d8_bf16_deflate` | wrap | `{"parent": "binary_identity_d8_bf16"}` |
| `binary_dense_d8_f32` | binary | `{"amplitude_precision": "f32", "dimension": 8, "model": "learned64", "transform": "dense"}` |
| `binary_dense_d8_f32_deflate` | wrap | `{"parent": "binary_dense_d8_f32"}` |
| `binary_dense_d8_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 8, "model": "learned64", "transform": "dense"}` |
| `binary_dense_d8_bf16_deflate` | wrap | `{"parent": "binary_dense_d8_bf16"}` |
| `binary_hadamard_d8_f32` | binary | `{"amplitude_precision": "f32", "dimension": 8, "model": "learned64", "transform": "hadamard"}` |
| `binary_hadamard_d8_f32_deflate` | wrap | `{"parent": "binary_hadamard_d8_f32"}` |
| `binary_hadamard_d8_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 8, "model": "learned64", "transform": "hadamard"}` |
| `binary_hadamard_d8_bf16_deflate` | wrap | `{"parent": "binary_hadamard_d8_bf16"}` |
| `binary_identity_d32_f32` | binary | `{"amplitude_precision": "f32", "dimension": 32, "model": "learned64", "transform": "identity"}` |
| `binary_identity_d32_f32_deflate` | wrap | `{"parent": "binary_identity_d32_f32"}` |
| `binary_identity_d32_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 32, "model": "learned64", "transform": "identity"}` |
| `binary_identity_d32_bf16_deflate` | wrap | `{"parent": "binary_identity_d32_bf16"}` |
| `binary_dense_d32_f32` | binary | `{"amplitude_precision": "f32", "dimension": 32, "model": "learned64", "transform": "dense"}` |
| `binary_dense_d32_f32_deflate` | wrap | `{"parent": "binary_dense_d32_f32"}` |
| `binary_dense_d32_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 32, "model": "learned64", "transform": "dense"}` |
| `binary_dense_d32_bf16_deflate` | wrap | `{"parent": "binary_dense_d32_bf16"}` |
| `binary_hadamard_d32_f32` | binary | `{"amplitude_precision": "f32", "dimension": 32, "model": "learned64", "transform": "hadamard"}` |
| `binary_hadamard_d32_f32_deflate` | wrap | `{"parent": "binary_hadamard_d32_f32"}` |
| `binary_hadamard_d32_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 32, "model": "learned64", "transform": "hadamard"}` |
| `binary_hadamard_d32_bf16_deflate` | wrap | `{"parent": "binary_hadamard_d32_bf16"}` |
| `binary_identity_d64_f32` | binary | `{"amplitude_precision": "f32", "dimension": 64, "model": "learned64", "transform": "identity"}` |
| `binary_identity_d64_f32_deflate` | wrap | `{"parent": "binary_identity_d64_f32"}` |
| `binary_identity_d64_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 64, "model": "learned64", "transform": "identity"}` |
| `binary_identity_d64_bf16_deflate` | wrap | `{"parent": "binary_identity_d64_bf16"}` |
| `binary_dense_d64_f32` | binary | `{"amplitude_precision": "f32", "dimension": 64, "model": "learned64", "transform": "dense"}` |
| `binary_dense_d64_f32_deflate` | wrap | `{"parent": "binary_dense_d64_f32"}` |
| `binary_dense_d64_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 64, "model": "learned64", "transform": "dense"}` |
| `binary_dense_d64_bf16_deflate` | wrap | `{"parent": "binary_dense_d64_bf16"}` |
| `binary_hadamard_d64_f32` | binary | `{"amplitude_precision": "f32", "dimension": 64, "model": "learned64", "transform": "hadamard"}` |
| `binary_hadamard_d64_f32_deflate` | wrap | `{"parent": "binary_hadamard_d64_f32"}` |
| `binary_hadamard_d64_bf16` | binary | `{"amplitude_precision": "bf16", "dimension": 64, "model": "learned64", "transform": "hadamard"}` |
| `binary_hadamard_d64_bf16_deflate` | wrap | `{"parent": "binary_hadamard_d64_bf16"}` |
| `learned_raw_radius` | residual | `{"assignment": "radius", "correction": "raw", "model": "learned64"}` |
| `learned_rice_radius` | residual | `{"assignment": "radius", "correction": "rice", "model": "learned64"}` |
| `learned_zigzag_deflate_radius` | residual | `{"assignment": "radius", "correction": "zigzag_deflate", "model": "learned64"}` |
| `learned_centered_deflate_radius` | residual | `{"assignment": "radius", "correction": "centered_deflate", "model": "learned64"}` |
| `learned_centered_packed_radius` | residual | `{"assignment": "radius", "correction": "centered_packed", "model": "learned64"}` |
| `learned_zigzag_packed_radius` | residual | `{"assignment": "radius", "correction": "zigzag_packed", "model": "learned64"}` |
| `learned_raw_l1` | residual | `{"assignment": "l1", "correction": "raw", "model": "learned64"}` |
| `learned_rice_l1` | residual | `{"assignment": "l1", "correction": "rice", "model": "learned64"}` |
| `learned_zigzag_deflate_l1` | residual | `{"assignment": "l1", "correction": "zigzag_deflate", "model": "learned64"}` |
| `learned_centered_deflate_l1` | residual | `{"assignment": "l1", "correction": "centered_deflate", "model": "learned64"}` |
| `learned_centered_packed_l1` | residual | `{"assignment": "l1", "correction": "centered_packed", "model": "learned64"}` |
| `learned_zigzag_packed_l1` | residual | `{"assignment": "l1", "correction": "zigzag_packed", "model": "learned64"}` |

## Native tensor packets

These use grouped bitpacking, a validated host launch descriptor, and device tensor operations. They are eager reference implementations. Unsupported device operations are reported individually.

| Name | Family | Options |
|---|---|---|
| `native_raw_words` | raw | `{}` |
| `native_exponent_group_bitpack` | exponent | `{}` |
| `native_learned_residual_bitpack` | learned | `{"assignment": "radius"}` |
| `native_e8_symbol_bitpack` | e8_symbol | `{"step_log2": 6}` |
| `native_learned_e8_residual_bitpack` | learned_e8 | `{"step_log2": 10}` |
| `native_binary_identity_d8_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 8, "transform": "identity"}` |
| `native_binary_identity_d8_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 8, "transform": "identity"}` |
| `native_binary_dense_d8_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 8, "transform": "dense"}` |
| `native_binary_dense_d8_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 8, "transform": "dense"}` |
| `native_binary_hadamard_d8_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 8, "transform": "hadamard"}` |
| `native_binary_hadamard_d8_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 8, "transform": "hadamard"}` |
| `native_binary_identity_d32_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 32, "transform": "identity"}` |
| `native_binary_identity_d32_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 32, "transform": "identity"}` |
| `native_binary_dense_d32_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 32, "transform": "dense"}` |
| `native_binary_dense_d32_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 32, "transform": "dense"}` |
| `native_binary_hadamard_d32_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 32, "transform": "hadamard"}` |
| `native_binary_hadamard_d32_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 32, "transform": "hadamard"}` |
| `native_binary_identity_d64_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 64, "transform": "identity"}` |
| `native_binary_identity_d64_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 64, "transform": "identity"}` |
| `native_binary_dense_d64_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 64, "transform": "dense"}` |
| `native_binary_dense_d64_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 64, "transform": "dense"}` |
| `native_binary_hadamard_d64_f32_bitpack` | binary | `{"amplitude_precision": "f32", "assignment": "radius", "dimension": 64, "transform": "hadamard"}` |
| `native_binary_hadamard_d64_bf16_bitpack` | binary | `{"amplitude_precision": "bf16", "assignment": "radius", "dimension": 64, "transform": "hadamard"}` |
| `native_prefill_zero_coset` | prefill | `{"correction": "coset", "group_size": 8, "predictor": "zero", "step_log2": 6}` |
| `native_prefill_zero_residual` | prefill | `{"correction": "residual", "group_size": 8, "predictor": "zero", "step_log2": 6}` |
| `native_prefill_midpoint_coset` | prefill | `{"correction": "coset", "group_size": 8, "predictor": "midpoint", "step_log2": 6}` |
| `native_prefill_midpoint_residual` | prefill | `{"correction": "residual", "group_size": 8, "predictor": "midpoint", "step_log2": 6}` |
| `native_prefill_affine_coset` | prefill | `{"correction": "coset", "group_size": 8, "predictor": "affine", "step_log2": 6}` |
| `native_prefill_affine_residual` | prefill | `{"correction": "residual", "group_size": 8, "predictor": "affine", "step_log2": 6}` |
| `native_prefill_e8_coset` | prefill | `{"correction": "coset", "group_size": 8, "predictor": "e8", "step_log2": 6}` |
| `native_prefill_e8_residual` | prefill | `{"correction": "residual", "group_size": 8, "predictor": "e8", "step_log2": 6}` |

The base inventory follows `candidate_specs("all", grid=True)` and
`native_options("all")`. The default all preset omits the extra CPU parameter
grid; main and smoke select smaller sets.

## Incremental prefill insight preset

`--native-preset insights` uses `recipe_sweep.insight_options()`: **33 methods**,
consisting of the 16-method native main set above and these 17 additions. It is
a targeted extension of main; the existing main/all sets are unchanged. Use
`--native-preset insights --list-methods` for the exact executable inventory.

| Additional name | Method | Change being tested |
|---|---|---|
| `native_prefill_midpoint_time_g8_residual` | `prefill` | Midpoint; errors grouped over time, eight values/group |
| `native_prefill_midpoint_channel_g64_residual` | `prefill` | Midpoint; errors grouped over channels, 64 values/group |
| `native_prefill_midpoint_time_g64_residual` | `prefill` | Midpoint; errors grouped over time, 64 values/group |
| `native_prefill_affine_time_g8_residual` | `prefill` | Affine; errors grouped over time, eight values/group |
| `native_prefill_affine_channel_g64_residual` | `prefill` | Affine; errors grouped over channels, 64 values/group |
| `native_prefill_affine_time_g64_residual` | `prefill` | Affine; errors grouped over time, 64 values/group |
| `native_prefill_exponent_zero_channel_tiered` | `prefill_exponent` | Zero exponent prediction; channel layout control |
| `native_prefill_exponent_fixed127_channel_tiered` | `prefill_exponent` | Source-independent exponent-bias prediction; channel control |
| `native_prefill_exponent_fixed127_time_tiered` | `prefill_exponent` | Source-independent exponent-bias prediction; time control |
| `native_prefill_exponent_median_channel_tiered` | `prefill_exponent` | Tile-derived per-channel median exponent; channel layout |
| `native_prefill_exponent_median_time_tiered` | `prefill_exponent` | Tile-derived per-channel median exponent; time layout |
| `native_prefill_exponent_affine_channel_tiered` | `prefill_exponent` | Tile-derived exponent endpoints; channel layout |
| `native_prefill_exponent_affine_time_tiered` | `prefill_exponent` | Tile-derived exponent endpoints; time layout |
| `native_prefill_affine_dictionary16_channel_g8_residual` | `prefill_dictionary` | Affine + 16-entry TRAIN residual dictionary; channel layout |
| `native_prefill_affine_dictionary64_channel_g8_residual` | `prefill_dictionary` | Affine + 64-entry TRAIN residual dictionary; channel layout |
| `native_prefill_affine_dictionary16_time_g8_residual` | `prefill_dictionary` | Affine + 16-entry TRAIN residual dictionary; time layout |
| `native_prefill_affine_dictionary64_time_g8_residual` | `prefill_dictionary` | Affine + 64-entry TRAIN residual dictionary; time layout |

Magnitude layout variants preserve their predictor/sign streams and change
correction grouping only. Their matched baseline is
`native_prefill_<midpoint|affine>_residual` with channel layout and group size
eight. Four-bit group widths, correction bits, tails, and headers all count.

Exponent tiers preserve one sign/fraction byte per value. Exact exponent errors
use 2-bit primary, 3-bit secondary and 9-bit rare streams, with uint32 offsets
per 256-value segment. Prefill-derived median/affine predictors must beat the
`fixed127` control after paying for their anchors. Channel/time layout preserves
the tier histograms and bit totals; header spelling is not an entropy gain.

The new dictionaries are fitted on affine **residuals**, separately for each
layout, using the same three TRAIN captures only. IDs cost 0.5 or 0.75 bits per
original value for 16 or 64 entries, followed by exact residual corrections.
The larger of the full serialized state and physical device tensor state is
charged once per cache; both are reported. Comparing dictionary capacities
changes IDs, fitted state and corrections together, rather than isolating one
coding change.

`--prefill-recipes` observes device KV before CPU export; it retains the full
raw transformer cache and includes all recipe/verification work in capture
prefill time. `--recipe-diagnostics` (automatic with insights) records actual
per-record sizes and timings. `recipe_analysis.json` provides matched increments
and a stateless **size-only oracle**, charging one mode byte per selected record
and full outer framing. That oracle is selection headroom, not an implemented
online selector or a latency gain. Full-cache extrapolations remain labelled
estimates when sampling. See [prefill_recipes.md](prefill_recipes.md) for commands,
continuation checks, accounting, and result sharing. No new GPU savings are
asserted by this inventory.
