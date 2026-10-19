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

The list is generated from `candidate_specs("all", grid=True)` and `native_options("all")`. The default all preset omits the extra CPU parameter grid; main and smoke select smaller sets.
