# Transfer extension review

**422/422 checks passed.**

All results below refer to independent controlled captures. They do not reproduce the team's original GGUF/document rates.

## Complete warm packet comparison

| Variant | Packet bpv | Reduction | Bpv saved vs block | Metadata bytes | Decoder MiB | Encoder projector MiB |
|---|---:|---:|---:|---:|---:|---:|
| aqua | 9.8408 | 38.50% | -0.4102 | 0 | 25.69 | 0.00 |
| block32 | 9.4306 | 41.06% | +0.0000 | 0 | 49.40 | 0.00 |
| block32_coset_model_r2 | 9.4309 | 41.06% | -0.0003 | 0 | 49.40 | 0.00 |
| block32_coset_uniform_r2 | 9.4306 | 41.06% | -0.0000 | 0 | 49.40 | 0.00 |
| block_hidden_r4 | 9.5078 | 40.58% | -0.0772 | 15,360 | 49.75 | 0.66 |
| block_hidden_r4_coset_uniform_r2 | 9.5078 | 40.58% | -0.0772 | 15,360 | 49.75 | 0.66 |
| block_hidden_shuffled_r4 | 9.5142 | 40.54% | -0.0836 | 15,360 | 49.75 | 0.66 |
| block_kv_r4 | 9.5021 | 40.61% | -0.0715 | 15,360 | 49.75 | 0.44 |
| mean | 10.2273 | 36.08% | -0.7967 | 0 | 0.55 | 0.00 |
| previous | 9.8933 | 38.17% | -0.4627 | 0 | 33.69 | 0.00 |

Packet deltas include framing, metadata and coded corrections. Serialized state and separate encoder/decoder resident amortization are retained in insights.json.

## Layer opportunities: diagnostic only

These are TEST ideal-bit screens after paying 8 × metadata_rank × token_count bits per layer. They are not measured layer packets or a deployed gate. Any gate must be fitted on DEV, serialized and tested on fresh held-out data.

**block_hidden_r4**: positive layers [0]; nonpositive layers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29].

| Layer | Net ideal bpv saved after metadata |
|---|---:|
| 0 | +0.0123 |
| 1 | -0.0733 |
| 2 | -0.0772 |
| 3 | -0.0795 |
| 4 | -0.0816 |
| 5 | -0.0823 |
| 6 | -0.0804 |
| 7 | -0.0808 |
| 8 | -0.0822 |
| 9 | -0.0814 |
| 10 | -0.0815 |
| 11 | -0.0775 |
| 12 | -0.0780 |
| 13 | -0.0814 |
| 14 | -0.0804 |
| 15 | -0.0809 |
| 16 | -0.0797 |
| 17 | -0.0802 |
| 18 | -0.0787 |
| 19 | -0.0789 |
| 20 | -0.0811 |
| 21 | -0.0792 |
| 22 | -0.0798 |
| 23 | -0.0818 |
| 24 | -0.0805 |
| 25 | -0.0799 |
| 26 | -0.0815 |
| 27 | -0.0813 |
| 28 | -0.0823 |
| 29 | -0.0818 |

**block_hidden_shuffled_r4**: positive layers []; nonpositive layers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29].

| Layer | Net ideal bpv saved after metadata |
|---|---:|
| 0 | -0.0833 |
| 1 | -0.0832 |
| 2 | -0.0820 |
| 3 | -0.0831 |
| 4 | -0.0839 |
| 5 | -0.0831 |
| 6 | -0.0840 |
| 7 | -0.0825 |
| 8 | -0.0842 |
| 9 | -0.0838 |
| 10 | -0.0835 |
| 11 | -0.0817 |
| 12 | -0.0835 |
| 13 | -0.0818 |
| 14 | -0.0828 |
| 15 | -0.0834 |
| 16 | -0.0838 |
| 17 | -0.0835 |
| 18 | -0.0833 |
| 19 | -0.0844 |
| 20 | -0.0823 |
| 21 | -0.0846 |
| 22 | -0.0836 |
| 23 | -0.0846 |
| 24 | -0.0838 |
| 25 | -0.0833 |
| 26 | -0.0842 |
| 27 | -0.0839 |
| 28 | -0.0849 |
| 29 | -0.0844 |

**block_kv_r4**: positive layers [0]; nonpositive layers [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29].

| Layer | Net ideal bpv saved after metadata |
|---|---:|
| 0 | +0.0759 |
| 1 | -0.0647 |
| 2 | -0.0716 |
| 3 | -0.0775 |
| 4 | -0.0736 |
| 5 | -0.0804 |
| 6 | -0.0782 |
| 7 | -0.0792 |
| 8 | -0.0801 |
| 9 | -0.0766 |
| 10 | -0.0781 |
| 11 | -0.0750 |
| 12 | -0.0749 |
| 13 | -0.0772 |
| 14 | -0.0781 |
| 15 | -0.0756 |
| 16 | -0.0742 |
| 17 | -0.0682 |
| 18 | -0.0720 |
| 19 | -0.0725 |
| 20 | -0.0748 |
| 21 | -0.0773 |
| 22 | -0.0806 |
| 23 | -0.0788 |
| 24 | -0.0815 |
| 25 | -0.0792 |
| 26 | -0.0813 |
| 27 | -0.0791 |
| 28 | -0.0789 |
| 29 | -0.0774 |

## Interpretation limits

- Historical team rates use a different checkpoint/corpus; this study cannot establish their exact numerical reproduction.
- Layer screening uses TEST ideal costs and pays raw metadata bytes; it excludes layer framing/state differences and cannot be presented as a deployed gated codec.
- Coset model is a same-distribution factorization control; complete packet bytes decide benefit.
- Codec timing is CPU reference timing; encoder and decoder state occupy different machines.
