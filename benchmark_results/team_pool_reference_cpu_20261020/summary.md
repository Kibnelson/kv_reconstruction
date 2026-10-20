# Original team pool functions: independent CPU validation

The original supplied packing/unpacking functions were executed on fresh real-model KV. This is not an exact reproduction of the team's Instruct Q4_1 checkpoint and documents.

Model: HuggingFaceTB/SmolLM2-135M base, revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2; native Hugging Face BF16 CPU capture. Keys were converted from split-half RoPE channel order to adjacent-pair order; values were unchanged.

All 1,474,560 original BF16 words round-tripped exactly across 4 pages and 240 blocks, using all 30 layers.

| Test | Start | Original packed pool, bpv | Reduction vs BF16 | Group-min formula, bpv | Exact |
| --- | ---: | ---: | ---: | ---: | --- |
| test_test_antarctic_story_512 | 0 | 11.6527 | 27.17% | 12.2609 | True |
| test_test_antarctic_story_512 | 480 | 11.3383 | 29.14% | 12.2547 | True |
| test_test_rust_log_parser_512 | 0 | 11.6032 | 27.48% | 12.2554 | True |
| test_test_rust_log_parser_512 | 480 | 11.2947 | 29.41% | 12.2467 | True |
| **Weighted aggregate** | | **11.4722** | **28.30%** | **12.2544** | **True** |

Actual packed bytes include one 2-byte escape count per 16-token block, raw sign/mantissa, packed exponent codes and raw escape exponents. Every actual block size matched its exact formula. The comparison baseline is a size formula, not an independently executed encoder.

Shared calibration arrays, block/page indexes, escape-rank lookup indexes, GPU alignment/allocation, fixed escape reservation and overflow allocation are excluded. No GPU kernel or attention throughput was measured. The results therefore establish the original variable-size payload representation's correctness and size on these captures, not its serving performance.

Three TRAIN documents fit references and widths; the two TEST documents are disjoint. Only the first and final 32-token pages of each 512-token TEST capture were measured. Results JSON retains source and capture SHA-256 identities and every block's byte accounting.
