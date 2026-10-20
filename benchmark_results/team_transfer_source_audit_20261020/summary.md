# Team transfer source audit

Source/log audit and independent helper tests; not original-data compression reproduction.

New audit checks: **29/29 passed**. This does not independently confirm the original model's compression rates.

## Historical supplied logs

4 chunks; 11,796,480 values. Logs report all encoder, decoder and pool word comparisons true.

| Metric | Average logged bits/value | Reduction from BF16 |
|---|---:|---:|
| Transfer packed stream | 8.360 | 47.75% |
| Pool packed bytes | 11.388 | 28.82% |
| Analytical baseline | 12.239 | 23.51% |

The averages above use rounded historical log values. Source/data reproduction remains separate.

## Independent tests performed now

The audit executes only six AST-whitelisted NumPy expression helpers; it never imports or executes the model script. It verifies every BF16 word mapping, positive normalized group counts, all mantissa-group sums for five mean/scale pairs, interval monotonicity, source hashes and log arithmetic.

## Predictor state

Layer zero contributes 67,584 coefficients. Total active coefficients: **6,303,744**.

A hypothetical compact INT32 store costs 25,214,976 bytes. The actual original padded INT64 `A` arrays cost **69,599,232 bytes**, before biases, scales and tables.

Useful MACs/value: 547.2 across all layers; 560.0 in predicted layers.

| State charged once | 8K state-inclusive bpv | 8K reduction | Aggregate tokens for 40% |
|---|---:|---:|---:|
| hypothetical compact int32 coefficients | 10.497 | 34.39% | 14,122 |
| actual padded int64 coefficients | 14.260 | 10.88% | 38,979 |

Amortization assumes a constant observed 8.360-bpv stream rate and reusable installed coefficients. Other constants, framing and scratch are excluded; these are arithmetic scenarios rather than model measurements.

## Production limitations

- **original_model_data_missing (unverified):** Original GGUF and ACT captures were not supplied; historical compression and word comparisons have not been independently reproduced.
- **monolithic_entropy_stream (observed):** One RangeEncoder is used across tokens, layers and channel groups. Independent predictor tokens do not make the original entropy stream independently decodable.
- **external_framing (observed):** Shape, token positions, fitted constants and tables come from CLI/globals. Their delivery, identities, chunk offsets and checksums are not included in stream bytes.
- **full_dense_output (observed):** transfer allocates a full decoded output with zeros_like(ORIG); the script retains originals and additional encoder/decoded/unpacked copies. Bounded-scratch compressed residency is not demonstrated.
- **integer_portability (unverified):** The predictor uses integers, but standardization and constriction inputs use floating point. Constants/tables are regenerated; cross-machine parity is not tested.
- **extreme_feature_conversion (unverified):** fxv casts very large finite values, infinities and NaNs to int64 without specified fallback/clamping. Positive symbol counts alone do not validate this feature path.
- **gpu_performance (unmeasured):** Supplied execution is CPU only. MAC counts and payload ratios do not establish GPU throughput or serving benefit.
- **baseline_size_analytical (observed):** The original baseline GPU-format size is calculated by a formula; transfer/pool payloads are actually packed.
- **hook_coset_extensions (absent):** The original combined codec conditions on produced KV; it contains no actual upstream hidden metadata or transmitted coset residues.

## Next matched comparison

Use the same real KV captures, frozen exact coder and TRAIN/DEV/TEST split for previous-layer-only, AQUA-style K/V, and earlier-channel predictors; then add paid KV-derived metadata, paid true-hidden metadata, coset residues, and their combination. Freeze options on DEV. Report packet bytes, shared state, peak scratch, exact packet-only reconstruction and encode/decode timings separately.

Coset factorization is bijective and does not create additional ideal entropy savings by itself. Metadata must pay its own bytes; any improvement is a measured gain of the practical model/format.
