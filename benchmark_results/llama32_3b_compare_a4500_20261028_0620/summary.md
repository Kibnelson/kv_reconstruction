# KV transfer codecs on unsloth/Llama-3.2-3B

Captures: 9 documents x 8192 tokens; 28 layers, 8 KV heads x 128; torch 2.11.0+cu128, transformers 4.57.6.

## Lossless (bits per value; BF16 = 16)

| Method | Four-chunk average | PEP 8 whole | argparse.py whole | Predictor multiply-adds per value |
| --- | --- | --- | --- | --- |
| SplitZip-style | 12.021 | 12.021 | 12.022 | 0 |
| ExANS-style | 10.722 | 10.707 | 10.711 | 0 |
| Our 44% version (layer l-1) | 8.962 | 8.951 | 8.716 | 2,048 |
| Our Block (+ earlier groups) | 8.235 | 8.228 | 7.911 | 3,008 |
| Our strongest no-hint | 7.922 | 7.917 | 7.624 | 9,153 |
| Our strong-and-fast | 7.821 | 7.831 | 7.541 | 13,379 |

## Bits against compute (37 predictor variants, 28 layers)

Lossless, exact costs like the table above. Multiply-adds per value include the shared projection of reduced-rank variants and, for the attention input, the query map and the attention itself. Integer formats: int8 x int16 needs 2 and int8 x int8 1 eight-bit tensor-core product per multiply-add, and the result is exact on any GPU. P = Pareto point among the floating-point variants (no cheaper variant has fewer bits).

| Variant | Multiply-adds per value | 8-bit products per value | Four-chunk average | PEP 8 / argparse.py whole | |
| --- | --- | --- | --- | --- | --- |
| L0+P2+P, 8 pivot steps, rank 64 | 1,152 | float | 8.762 | 8.749 / 8.495 | P |
| P, 16 pivot steps, rank 128 | 1,216 | float | 8.623 | 8.612 / 8.347 | P |
| L0+P2+P, 8 pivot steps, rank 128 | 1,408 | float | 8.691 | 8.679 / 8.419 |  |
| P, 16 pivot steps, rank 256 | 1,472 | float | 8.529 | 8.519 / 8.247 | P |
| L0+P2+P, 8 pivot steps, rank 256 | 1,920 | float | 8.586 | 8.577 / 8.309 |  |
| L0+P2+P, 8 pivot steps, rank 256, int8 coef x int16 feat | 1,920 | 3,840 | 8.604 | 8.594 / 8.332 |  |
| L0+P2+P, 8 pivot steps, rank 256, int8 coef x int8 feat | 1,920 | 1,920 | 8.610 | 8.599 / 8.340 |  |
| P, 16 pivot steps, rank 512 | 1,984 | float | 8.387 | 8.381 / 8.101 | P |
| P, 1 pivot step | 2,048 | float | 8.908 | 8.898 / 8.689 |  |
| P, 1 natural step, static spread | 2,048 | float | 8.962 | 8.951 / 8.716 |  |
| P, 4 pivot steps | 2,816 | float | 8.250 | 8.247 / 7.975 | P |
| P, 8 pivot steps | 2,944 | float | 8.109 | 8.108 / 7.828 | P |
| L0+P2+P, 8 pivot steps, rank 512 | 2,944 | float | 8.430 | 8.425 / 8.150 |  |
| P, 16 pivot steps | 3,008 | float | 8.038 | 8.037 / 7.755 | P |
| P, 16 pivot steps, int8 coef x int16 feat | 3,008 | 6,016 | 8.067 | 8.063 / 7.790 |  |
| P, 16 pivot steps, int8 coef x int8 feat | 3,008 | 3,008 | 8.078 | 8.074 / 7.806 |  |
| P, 16 natural steps, static spread | 3,008 | float | 8.235 | 8.228 / 7.911 |  |
| L0+P, 1 pivot step | 4,096 | float | 8.886 | 8.879 / 8.670 |  |
| L0+P2+P+ATT, 8 pivot steps, rank 64 | 4,516 | float | 8.755 | 8.742 / 8.489 |  |
| L0+P, 4 pivot steps | 4,864 | float | 8.226 | 8.226 / 7.955 |  |
| L0+P2+P+ATT, 8 pivot steps, rank 128 | 4,868 | float | 8.678 | 8.666 / 8.406 |  |
| L0+P, 8 pivot steps | 4,992 | float | 8.085 | 8.086 / 7.808 |  |
| L0+P, 16 pivot steps | 5,056 | float | 8.013 | 8.015 / 7.733 | P |
| L0+P2+P+ATT, 8 pivot steps, rank 256 | 5,572 | float | 8.564 | 8.556 / 8.286 |  |
| L0+P2+P, 1 pivot step | 6,144 | float | 8.726 | 8.726 / 8.520 |  |
| L0+P2+P, 4 pivot steps | 6,912 | float | 8.077 | 8.083 / 7.814 |  |
| L0+P2+P+ATT, 8 pivot steps, rank 512 | 6,980 | float | 8.395 | 8.392 / 8.113 |  |
| L0+P2+P, 8 pivot steps | 7,040 | float | 7.940 | 7.947 / 7.669 | P |
| L0+P2+P, 8 pivot steps, int8 coef x int16 feat | 7,040 | 14,080 | 7.969 | 7.973 / 7.707 |  |
| L0+P2+P, 8 pivot steps, int8 coef x int8 feat | 7,040 | 7,040 | 7.982 | 7.985 / 7.724 |  |
| L0+P2+P, 16 pivot steps | 7,104 | float | 7.870 | 7.877 / 7.597 | P |
| L0+P2+P+ATT, 1 pivot step | 12,484 | float | 8.581 | 8.586 / 8.353 |  |
| L0+P2+P+ATT, 4 pivot steps | 13,252 | float | 7.953 | 7.962 / 7.678 |  |
| L0+P2+P+ATT, 8 pivot steps | 13,380 | float | 7.821 | 7.831 / 7.541 | P |
| L0+P2+P+ATT, 8 pivot steps, int8 coef x int16 feat | 13,380 | 26,761 | 7.859 | 7.866 / 7.590 |  |
| L0+P2+P+ATT, 8 pivot steps, int8 coef x int8 feat | 13,380 | 13,380 | 7.874 | 7.879 / 7.610 |  |
| L0+P2+P+ATT, 16 pivot steps | 13,444 | float | 7.754 | 7.764 / 7.472 | P |

## GPU speed (NVIDIA RTX A4500)

Kernel time only (CUDA-event medians after warm-up), like the published SplitZip and ExANS GB/s; GB/s counts raw BF16 bytes (2 per value). Predictor: one layer of 8192 tokens, random data of the real shapes (the inputs product, then one product per group in sequence; the attention input adds the query map and the windowed attention, in BF16 for the integer formats). Not included: RoPE rotation and the spread arithmetic (element-wise).

| Predictor | Format | ms per layer | G values/s | GB/s | As a CUDA graph: ms, G values/s |
| --- | --- | --- | --- | --- | --- |
| P, 1 pivot step | fp32 | 4.50 | 3.7 | 7.5 | 4.54, 3.7 |
| P, 1 pivot step | bf16 | 0.87 | 19.3 | 38.7 | 0.86, 19.5 |
| P, 1 pivot step | int8x16 | 3.42 | 4.9 | 9.8 | 3.41, 4.9 |
| P, 1 pivot step | int8x8 | 0.53 | 31.5 | 63.0 | 0.52, 32.0 |
| P, 1 pivot step | int8x8s | not measured | | | |
| P, 16 pivot steps, rank 128 | fp32 | 3.31 | 5.1 | 10.1 | 3.48, 4.8 |
| P, 16 pivot steps, rank 128 | bf16 | 1.08 | 15.5 | 31.1 | 1.01, 16.6 |
| P, 16 pivot steps, rank 128 | int8x16 | 6.32 | 2.7 | 5.3 | 6.13, 2.7 |
| P, 16 pivot steps, rank 128 | int8x8 | 0.70 | 24.1 | 48.1 | 0.66, 25.5 |
| P, 4 pivot steps | fp32 | 6.77 | 2.5 | 5.0 | 6.75, 2.5 |
| P, 4 pivot steps | bf16 | 1.43 | 11.7 | 23.4 | 1.26, 13.3 |
| P, 4 pivot steps | int8x16 | 5.84 | 2.9 | 5.7 | 5.78, 2.9 |
| P, 4 pivot steps | int8x8 | 0.83 | 20.3 | 40.5 | 0.82, 20.6 |
| P, 4 pivot steps | int8x8s | not measured | | | |
| P, 8 pivot steps | fp32 | 7.31 | 2.3 | 4.6 | 7.25, 2.3 |
| P, 8 pivot steps | bf16 | 1.66 | 10.1 | 20.3 | 1.43, 11.7 |
| P, 8 pivot steps | int8x16 | 6.30 | 2.7 | 5.3 | 6.21, 2.7 |
| P, 8 pivot steps | int8x8 | 0.88 | 19.0 | 38.1 | 0.86, 19.5 |
| P, 8 pivot steps | int8x8s | not measured | | | |
| P, 16 pivot steps | fp32 | 7.75 | 2.2 | 4.3 | 7.75, 2.2 |
| P, 16 pivot steps | bf16 | 1.86 | 9.0 | 18.1 | 1.69, 9.9 |
| P, 16 pivot steps | int8x16 | 6.81 | 2.5 | 4.9 | 6.64, 2.5 |
| P, 16 pivot steps | int8x8 | 1.02 | 16.5 | 32.9 | 0.99, 17.0 |
| P, 16 pivot steps | int8x8s | not measured | | | |
| L0+P2+P, 16 pivot steps | fp32 | 16.83 | 1.0 | 2.0 | 16.97, 1.0 |
| L0+P2+P, 16 pivot steps | bf16 | 3.62 | 4.6 | 9.3 | 3.25, 5.2 |
| L0+P2+P, 16 pivot steps | int8x16 | 8.45 | 2.0 | 4.0 | 8.24, 2.0 |
| L0+P2+P, 16 pivot steps | int8x8 | 1.84 | 9.1 | 18.3 | 1.78, 9.4 |
| L0+P2+P, 16 pivot steps | int8x8s | not measured | | | |
| L0+P2+P+ATT, 16 pivot steps | fp32 | 62.25 | 0.3 | 0.5 |  |
| L0+P2+P+ATT, 16 pivot steps | bf16 | 33.85 | 0.5 | 1.0 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x16 | 38.76 | 0.4 | 0.9 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x8 | 31.99 | 0.5 | 1.0 |  |

Value-slice coder (Triton) on 67,108,864 real values (lite predictions, layers [4, 12, 20, 27]); the round trip must be bit-exact.

Stage 1: how the decoder finds the slice symbol (direct: a table and one comparison; search: a coarse table and a binary search); the streams are the same.

| Mode | Stage 1 | Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | direct | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 29.7 | 25.6 |
| exact | search | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 29.7 | 27.9 |
| exact | direct | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 20.9 | 15.3 |
| exact | search | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 20.9 | 16.2 |
| exact | direct | 768 x 128 (1) | True | 8.0942 | 8.0346 | 30.6 | 22.6 |
| exact | search | 768 x 128 (1) | True | 8.0942 | 8.0346 | 30.6 | 27.3 |
| exact | direct | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.8 | 13.9 |
| exact | search | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.8 | 17.9 |
| exact | direct | 512 x 128 (1) | True | 8.1100 | 8.0346 | 32.2 | 26.9 |
| exact | search | 512 x 128 (1) | True | 8.1100 | 8.0346 | 32.2 | 31.0 |
| uniform | direct | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 34.8 | 37.6 |
| uniform | search | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 34.8 | 45.1 |
| uniform | direct | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 22.4 | 25.3 |
| uniform | search | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 22.4 | 30.9 |
| uniform | direct | 768 x 128 (1) | True | 8.1283 | 8.0346 | 34.8 | 30.0 |
| uniform | search | 768 x 128 (1) | True | 8.1283 | 8.0346 | 34.8 | 38.7 |
| uniform | direct | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.8 | 18.4 |
| uniform | search | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.8 | 24.4 |
| uniform | direct | 512 x 128 (1) | True | 8.1441 | 8.0346 | 36.0 | 40.7 |
| uniform | search | 512 x 128 (1) | True | 8.1441 | 8.0346 | 36.0 | 50.0 |

Decode estimate per predictor (fastest exact integer format, else BF16, CUDA graph if faster; then the fastest coder, uniform mode; one after the other, not overlapped). Bits include that coder's measured overhead. Link it keeps up with = decoded values/s x bits per value.

| Predictor | Bits per value | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- |
| P, 1 pivot step (int8x8) | 9.017 | 14.0 | 28.1 | 127 Gb/s |
| P, 16 pivot steps, rank 128 (int8x8) | 8.733 | 12.6 | 25.2 | 110 Gb/s |
| P, 4 pivot steps (int8x8) | 8.359 | 11.3 | 22.6 | 94 Gb/s |
| P, 8 pivot steps (int8x8) | 8.219 | 11.0 | 21.9 | 90 Gb/s |
| P, 16 pivot steps (int8x8) | 8.148 | 10.1 | 20.2 | 82 Gb/s |
| L0+P2+P, 16 pivot steps (int8x8) | 7.979 | 6.8 | 13.7 | 55 Gb/s |
| L0+P2+P+ATT, 16 pivot steps (int8x8) | 7.863 | 0.5 | 1.0 | 4 Gb/s |

Overlap, exact coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.06 ms + decode 4.68 ms one after the other, 6.57 ms together, so 10.2 G values/s (20.4 GB/s), keeping up with a 91 Gb/s link.

Overlap, exact coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.88 ms + decode 4.77 ms one after the other, 7.92 ms together, so 8.5 G values/s (16.9 GB/s), keeping up with a 69 Gb/s link.

Overlap, exact coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.33 ms + decode 4.70 ms one after the other, 8.27 ms together, so 8.1 G values/s (16.2 GB/s), keeping up with a 65 Gb/s link.

Overlap, uniform coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.14 ms + decode 2.98 ms one after the other, 4.91 ms together, so 13.7 G values/s (27.4 GB/s), keeping up with a 122 Gb/s link.

Overlap, uniform coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.66 ms + decode 2.94 ms one after the other, 6.33 ms together, so 10.6 G values/s (21.2 GB/s), keeping up with a 86 Gb/s link.

Overlap, uniform coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.25 ms + decode 2.93 ms one after the other, 6.71 ms together, so 10.0 G values/s (20.0 GB/s), keeping up with a 80 Gb/s link.

Whole-document decode, fused one-step codec (Triton, measured): test_text, 28 layers x 8192 tokens = 470M values, encoded and decoded layer by layer as a receiver must: one kernel predicts each layer from the decoded previous layer (int8 tensor-core product, RoPE, spread), decodes it and makes the next layer's features. Times are the kernels end to end (CUDA events); the decode must be bit-exact. Bits are the stream's own (states included); the predictor's exact-model bits show the cost of its int8 coefficients and features.

| Tokens per program x rows (warps) | Mode | Stage 1 | Bit-exact | Stream bits/value | Predictor exact-model bits/value | Encoder forward + encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 8 (8) | uniform | direct | True | 9.0114 | 8.9108 | 39.5 + 56.4 | 61.5 | 7.6 | 15.3 | 69 Gb/s |
| 64 x 8 (8) | uniform | direct | True | 9.0045 | 8.9108 | 58.1 + 78.0 | 79.8 | 5.9 | 11.8 | 53 Gb/s |
| 32 x 4 (4) | uniform | direct | True | 9.0045 | 8.9108 | 36.7 + 65.5 | 58.2 | 8.1 | 16.2 | 73 Gb/s |
| 16 x 4 (4) | uniform | direct | True | 9.0114 | 8.9108 | 54.2 + 78.1 | 93.7 | 5.0 | 10.0 | 45 Gb/s |
| 64 x 16 (16) | uniform | direct | True | 9.0114 | 8.9108 | 96.3 + 119.1 | 111.4 | 4.2 | 8.4 | 38 Gb/s |
| 32 x 4 (4) | uniform | search | True | 9.0045 | 8.9108 | 35.9 + 64.3 | 63.3 | 7.4 | 14.8 | 67 Gb/s |
| 32 x 4 (4) | exact | direct | True | 8.9447 | 8.9108 | 38.5 + 55.6 | 100.0 | 4.7 | 9.4 | 42 Gb/s |

Whole-document decode, per-layer one-step codec (measured; same document, predictor and checks): for each layer a cuBLAS int8 product (the previous layer's 8-bit features times the 8-bit coefficients, exact int32), then one Triton kernel of one-warp programs (a token block x one head) that scales, rotates, codes and makes the next layer's features. Streams stay open across layers. Chunks: the tokens are split into that many parts, each decoded on its own CUDA stream, so one part's product runs while another part decodes. Encode time excludes the encoder's forward pass (as in the fused table).

| Tokens per program | Chunks | Mode | Stage 1 | Bit-exact | Stream bits/value | Encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 1 | uniform | search | True | 9.0045 | 36.6 | 36.6 | 12.8 | 25.6 | 115 Gb/s |
| 128 | 2 | uniform | search | True | 9.0045 | 37.0 | 34.2 | 13.7 | 27.5 | 124 Gb/s |
| 128 | 4 | uniform | search | True | 9.0045 | 37.1 | 33.0 | 14.3 | 28.5 | 128 Gb/s |
| 64 | 2 | uniform | search | True | 9.0114 | 34.6 | 32.5 | 14.5 | 28.9 | 130 Gb/s |
| 256 | 2 | uniform | search | True | 9.0010 | 39.3 | 38.4 | 12.2 | 24.5 | 110 Gb/s |
| 64 | 2 | uniform | direct | True | 9.0114 | 34.6 | 32.6 | 14.4 | 28.8 | 130 Gb/s |
| 64 | 2 | exact | search | True | 8.9516 | 37.6 | 44.6 | 10.5 | 21.0 | 94 Gb/s |

Reference: SplitZip's native codec on the same RTX A4500 (official run, branch experiment/llama32_official_gpu, benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517): 12.04 bits per value, encode 137 GB/s, decode 311 GB/s.


Our codec alone (exact coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.87 bits/value, 10.2 G values/s); P, 8 pivot steps (8.04 bits/value, 8.5 G values/s); P, 16 pivot steps (7.97 bits/value, 8.1 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 7516 ms | 5647 ms | 3745 ms (16 pivot steps) | 3745 ms (28 16 pivot steps) | 1.51x |
| 5 Gb/s | 1503 ms | 1129 ms | 749 ms (16 pivot steps) | 749 ms (28 16 pivot steps) | 1.51x |
| 10 Gb/s | 752 ms | 565 ms | 374 ms (16 pivot steps) | 374 ms (28 16 pivot steps) | 1.51x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.51x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.51x |
| 100 Gb/s | 75 ms | 56 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 1.23x |
| 200 Gb/s | 38 ms | 28 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 0.61x |
| 400 Gb/s | 19 ms | 14 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 0.31x |
| 800 Gb/s | 9 ms | 7 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 0.15x |

Our codec alone (uniform coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.90 bits/value, 13.7 G values/s); P, 8 pivot steps (8.08 bits/value, 10.6 G values/s); P, 16 pivot steps (8.01 bits/value, 10.0 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 7516 ms | 5647 ms | 3761 ms (16 pivot steps) | 3761 ms (28 16 pivot steps) | 1.50x |
| 5 Gb/s | 1503 ms | 1129 ms | 752 ms (16 pivot steps) | 752 ms (28 16 pivot steps) | 1.50x |
| 10 Gb/s | 752 ms | 565 ms | 376 ms (16 pivot steps) | 376 ms (28 16 pivot steps) | 1.50x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.50x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.50x |
| 100 Gb/s | 75 ms | 56 ms | 42 ms (1 pivot step) | 39 ms (14 1 pivot step, 14 8 pivot steps) | 1.44x |
| 200 Gb/s | 38 ms | 28 ms | 34 ms (1 pivot step) | 34 ms (28 1 pivot step) | 0.82x |
| 400 Gb/s | 19 ms | 14 ms | 34 ms (1 pivot step) | 34 ms (28 1 pivot step) | 0.41x |
| 800 Gb/s | 9 ms | 7 ms | 34 ms (1 pivot step) | 34 ms (28 1 pivot step) | 0.21x |

Our codec alone, measured end to end (per-layer one-step codec, whole test_text: 470M values, 9.011 bits/value, decoded layer by layer at 14.5 G values/s; 64 tokens per program, 2 chunk(s), uniform). SplitZip: 12.021 bits/value on the same document, its measured A4500 decode speed. Ideal link as above; the bands are typical deployments (see the doc).

| Link | Typical setting | Raw BF16 | SplitZip | Ours (measured) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | ordinary cloud, storage | 7516 ms | 5647 ms | 4233 ms | 1.33x |
| 5 Gb/s | ordinary cloud | 1503 ms | 1129 ms | 847 ms | 1.33x |
| 10 Gb/s | consumer, cloud Ethernet | 752 ms | 565 ms | 423 ms | 1.33x |
| 25 Gb/s | cloud Ethernet | 301 ms | 226 ms | 169 ms | 1.33x |
| 50 Gb/s | workstation | 150 ms | 113 ms | 85 ms | 1.33x |
| 100 Gb/s | workstation, cross-cluster | 75 ms | 56 ms | 42 ms | 1.33x |
| 200 Gb/s | datacenter RDMA, per GPU | 38 ms | 28 ms | 32 ms | 0.87x |
| 400 Gb/s | datacenter RDMA, per GPU | 19 ms | 14 ms | 32 ms | 0.43x |
| 800 Gb/s | next-generation RDMA | 9 ms | 7 ms | 32 ms | 0.22x |

## GPU-resident KV pool (NVIDIA RTX A4500, measured)

A pool keeps finished requests' KV so that a later request with the same prefix skips its prefill. Bringing back one cached prefix of 8192 tokens (test_text, 28 layers, 896 MiB of raw BF16 KV). Times are CUDA-event medians; our decode is bit-exact. Copies are host to GPU over this machine's PCIe link.

| Way to bring the prefix back | Time | Kept in | Memory held |
| --- | --- | --- | --- |
| Recompute: the model's prefill | 988 ms | nothing | 0 |
| Raw KV kept in GPU memory | 0 ms | GPU | 896 MiB |
| Raw KV in CPU memory, copied back (pinned / pageable) | 39 / 68 ms | CPU | 896 MiB |
| Ours (per layer, uniform, search) kept in GPU memory, decoded | 34 ms | GPU | 504 MiB |
| Ours (per layer, uniform) in CPU memory: copy, then decode | 22 + 34 ms | CPU | 504 MiB |
| Ours (per layer, uniform) kept in GPU memory, decoded | 34 ms | GPU | 504 MiB |
| Ours (per layer, uniform) in CPU memory: copy, then decode | 22 + 34 ms | CPU | 504 MiB |
| Ours (per layer, exact, search) kept in GPU memory, decoded | 48 ms | GPU | 501 MiB |
| Ours (per layer, exact) in CPU memory: copy, then decode | 22 + 48 ms | CPU | 501 MiB |
| Ours (fused, uniform) kept in GPU memory, decoded | 58 ms | GPU | 504 MiB |
| Ours (fused, uniform) in CPU memory: copy, then decode | 22 + 58 ms | CPU | 504 MiB |
| Ours (fused, uniform, search) kept in GPU memory, decoded | 63 ms | GPU | 504 MiB |
| Ours (fused, uniform) in CPU memory: copy, then decode | 22 + 63 ms | CPU | 504 MiB |
| Ours (fused, exact) kept in GPU memory, decoded | 100 ms | GPU | 501 MiB |
| Ours (fused, exact) in CPU memory: copy, then decode | 22 + 100 ms | CPU | 501 MiB |
| SplitZip kept in GPU memory, decoded (its published A4500 speed, not run here) | 3 ms | GPU | 673 MiB |

Adding the prefix to the pool (ours): the encoder's forward pass (features and spreads of the exact KV, the same for both implementations) plus the encode, against the prefill that produced the KV.

| Codec | Forward + encode ms | Share of the prefill |
| --- | --- | --- |
| fused, uniform | 35 + 64 | 10% |
| fused, uniform, search | 37 + 65 | 10% |
| fused, exact | 39 + 56 | 10% |
| per layer, uniform, search | 35 + 37 | 7% |
| per layer, uniform | 35 + 37 | 7% |
| per layer, exact, search | 35 + 40 | 8% |

Capacity: KV bytes per token = 28 layers x 2048 values x bits / 8. Pool = the 12.5 GiB free after loading the model (no room left for activations, so an upper bound).

| Format | Bits per value | KiB per token | Tokens per GiB | Tokens in the free memory | vs raw |
| --- | --- | --- | --- | --- | --- |
| raw BF16 | 16.00 | 112.0 | 9,362 | 116,691 | 1.00x |
| SplitZip | 12.02 | 84.1 | 12,461 | 155,319 | 1.33x |
| ours lossless (uniform) | 9.00 | 63.0 | 16,636 | 207,349 | 1.78x |
| ours lossless (exact) | 8.94 | 62.6 | 16,747 | 208,734 | 1.79x |
| Ours, step 0.25 (lossy: 0.5% KV error, perplexity +0.25%; GPU decoder not built yet) | 5.17 | 36.2 | 28,948 | 360,811 | 3.09x |
| Ours, step 0.5 (lossy: 0.9% KV error, perplexity +0.33%; GPU decoder not built yet) | 4.21 | 29.5 | 35,579 | 443,451 | 3.80x |
| Ours, step 1.0 (lossy: 1.9% KV error, perplexity +1.26%; GPU decoder not built yet) | 3.31 | 23.1 | 45,295 | 564,559 | 4.84x |
| Ours, step 2.0 (lossy: 3.7% KV error, perplexity +3.81%; GPU decoder not built yet) | 2.56 | 17.9 | 58,419 | 728,139 | 6.24x |
| Ours, step 4.0 (lossy: 7.5% KV error, perplexity +14.50%; GPU decoder not built yet) | 2.14 | 15.0 | 70,102 | 873,757 | 7.49x |

## Lossy (prompt 3072 tokens; quality = perplexity of the next 512 tokens with the transferred KV)

| Method | Bits per value | KV relative error | Perplexity change |
| --- | --- | --- | --- |
| exact KV | 16.00 / 16.00 | 0.00% / 0.00% | +0.00% / +0.00% |
| Ours, step 0.25 | 5.17 / 4.85 | 0.48% / 0.47% | +0.25% / -0.15% |
| Ours, step 0.5 | 4.21 / 3.91 | 0.95% / 0.93% | +0.33% / +0.53% |
| Ours, step 1.0 | 3.31 / 3.04 | 1.88% / 1.84% | +1.26% / +0.28% |
| Ours, step 2.0 | 2.56 / 2.38 | 3.75% / 3.67% | +3.81% / +0.59% |
| Ours, step 4.0 | 2.14 / 2.05 | 7.49% / 7.32% | +14.50% / +1.00% |
| AQUA-KV-style, 2-bit | 2.57 / 2.57 | 2034.01% / 1953.46% | +10208341.91% / +16246160.94% |
| AQUA-KV-style, 3-bit | 3.54 / 3.54 | 23.80% / 23.02% | +6251.08% / +4985.11% |
| AQUA-KV-style, 4-bit | 4.50 / 4.50 | 5.72% / 5.45% | +4.31% / +2.17% |
| KIVI-style, 2-bit | 2.38 / 2.38 | 29.54% / 28.47% | +22.56% / +15.82% |
| KIVI-style, 3-bit | 3.38 / 3.38 | 12.56% / 12.10% | +5.08% / +2.69% |
| KIVI-style, 4-bit | 4.38 / 4.38 | 5.86% / 5.64% | +0.33% / +0.45% |
| FP8 cast | 8.00 / 8.00 | 2.64% / 2.64% | +0.33% / -0.32% |

Values are test_text / test_code. Reference: the captured exact KV restored into the cache.
The model's own prefill of the prompt alone differs from it by +0.050% / +0.039% (GPU kernels round differently at another sequence length).
