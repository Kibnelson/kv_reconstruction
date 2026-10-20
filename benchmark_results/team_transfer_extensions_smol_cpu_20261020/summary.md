# Transfer and paid prefill-metadata comparison

Independent controlled validation of the supplied transfer algorithm. Capture uses a real model; the codec is a NumPy/constriction CPU reference. These are not GPU codec timings and do not reproduce the team's checkpoint/documents.

Model: `../.smoke_model`. All 40 TEST packet round trips passed bit-exact comparison in a separate decoder process.

TRAIN alone fits predictors and metadata PCA/quantizers. First half of DEV fits scales; a disjoint DEV page chooses metadata rank and coset width. TEST reports the frozen choices.

| Variant | Packet bpv | Reduction | Decoder state MiB | Encoder projector MiB | Encode s | Decode s |
|---|---:|---:|---:|---:|---:|---:|
| mean | 10.2273 | 36.08% | 0.55 | 0.00 | 48.198 | 55.634 |
| previous | 9.8933 | 38.17% | 33.69 | 0.00 | 67.094 | 74.643 |
| aqua | 9.8408 | 38.50% | 25.69 | 0.00 | 61.925 | 67.728 |
| block32 | 9.4306 | 41.06% | 49.40 | 0.00 | 69.620 | 85.079 |
| block_kv_r4 | 9.5021 | 40.61% | 49.75 | 0.44 | 58.853 | 160.513 |
| block_hidden_r4 | 9.5078 | 40.58% | 49.75 | 0.66 | 53.061 | 112.684 |
| block_hidden_shuffled_r4 | 9.5142 | 40.54% | 49.75 | 0.66 | 250.976 | 75.823 |
| block32_coset_uniform_r2 | 9.4306 | 41.06% | 49.40 | 0.00 | 101.250 | 76.949 |
| block32_coset_model_r2 | 9.4309 | 41.06% | 49.40 | 0.00 | 58.787 | 72.521 |
| block_hidden_r4_coset_uniform_r2 | 9.5078 | 40.58% | 49.75 | 0.66 | 49.746 | 64.934 |

Packet rates include transmitted int8 metadata, framing and the entropy stream. Decoder resident state and encoder-only projector state are charged separately on their respective machines. Cold rates and amortized capacity figures are in results.json. Dense predictor arrays are physical bytes; a compact coefficient estimate is only a diagnostic.

`coset_model` factorizes the same joint symbol distribution and is a control; `coset_uniform` sends residues under a uniform model before conditional quotient coding. Coset savings must be compared on total packet bytes rather than multiplied into another reduction.

Actual hidden metadata is extracted from attention input hidden_states during prefill. The same-budget KV-derived control and fixed token-roll hidden control test whether hidden hooks add useful information.

The entropy stream is serial in this CPU reference. Parallel-token prediction does not establish a parallel GPU entropy decoder. Cross-machine probability/coder reproducibility and GPU speed remain untested.
