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
| P, 1 pivot step | fp32 | 4.40 | 3.8 | 7.6 | 4.52, 3.7 |
| P, 1 pivot step | bf16 | 0.87 | 19.4 | 38.7 | 0.86, 19.5 |
| P, 1 pivot step | int8x16 | 3.42 | 4.9 | 9.8 | 3.41, 4.9 |
| P, 1 pivot step | int8x8 | 0.53 | 31.6 | 63.1 | 0.52, 32.0 |
| P, 1 pivot step | int8x8s | not measured | | | |
| P, 16 pivot steps, rank 128 | fp32 | 3.29 | 5.1 | 10.2 | 3.60, 4.7 |
| P, 16 pivot steps, rank 128 | bf16 | 1.08 | 15.6 | 31.1 | 0.95, 17.7 |
| P, 16 pivot steps, rank 128 | int8x16 | 6.32 | 2.7 | 5.3 | 6.14, 2.7 |
| P, 16 pivot steps, rank 128 | int8x8 | 0.70 | 24.0 | 48.0 | 0.65, 25.6 |
| P, 4 pivot steps | fp32 | 6.77 | 2.5 | 5.0 | 6.71, 2.5 |
| P, 4 pivot steps | bf16 | 1.42 | 11.8 | 23.7 | 1.41, 11.9 |
| P, 4 pivot steps | int8x16 | 5.82 | 2.9 | 5.8 | 5.78, 2.9 |
| P, 4 pivot steps | int8x8 | 0.83 | 20.3 | 40.6 | 0.81, 20.6 |
| P, 4 pivot steps | int8x8s | not measured | | | |
| P, 8 pivot steps | fp32 | 7.25 | 2.3 | 4.6 | 7.23, 2.3 |
| P, 8 pivot steps | bf16 | 1.64 | 10.2 | 20.5 | 1.47, 11.4 |
| P, 8 pivot steps | int8x16 | 6.31 | 2.7 | 5.3 | 6.22, 2.7 |
| P, 8 pivot steps | int8x8 | 0.88 | 19.1 | 38.2 | 0.86, 19.6 |
| P, 8 pivot steps | int8x8s | not measured | | | |
| P, 16 pivot steps | fp32 | 7.68 | 2.2 | 4.4 | 7.67, 2.2 |
| P, 16 pivot steps | bf16 | 1.84 | 9.1 | 18.3 | 1.64, 10.2 |
| P, 16 pivot steps | int8x16 | 6.80 | 2.5 | 4.9 | 6.64, 2.5 |
| P, 16 pivot steps | int8x8 | 1.01 | 16.6 | 33.1 | 0.98, 17.1 |
| P, 16 pivot steps | int8x8s | not measured | | | |
| L0+P2+P, 16 pivot steps | fp32 | 16.87 | 1.0 | 2.0 | 16.98, 1.0 |
| L0+P2+P, 16 pivot steps | bf16 | 3.59 | 4.7 | 9.3 | 3.31, 5.1 |
| L0+P2+P, 16 pivot steps | int8x16 | 8.41 | 2.0 | 4.0 | 8.24, 2.0 |
| L0+P2+P, 16 pivot steps | int8x8 | 1.82 | 9.2 | 18.4 | 1.78, 9.4 |
| L0+P2+P, 16 pivot steps | int8x8s | not measured | | | |
| L0+P2+P+ATT, 16 pivot steps | fp32 | 62.11 | 0.3 | 0.5 |  |
| L0+P2+P+ATT, 16 pivot steps | bf16 | 33.62 | 0.5 | 1.0 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x16 | 38.55 | 0.4 | 0.9 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x8 | 31.70 | 0.5 | 1.1 |  |

Value-slice coder (Triton) on 67,108,864 real values (lite predictions, layers [4, 12, 20, 27]); the round trip must be bit-exact.

Stage 1: how the decoder finds the slice symbol (direct: a table and one comparison; search: a coarse table and a binary search); the streams are the same.

| Mode | Stage 1 | Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | direct | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 29.5 | 24.1 |
| exact | search | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 29.5 | 26.3 |
| exact | direct | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 21.0 | 13.6 |
| exact | search | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 21.0 | 14.2 |
| exact | direct | 768 x 128 (1) | True | 8.0942 | 8.0346 | 32.4 | 22.0 |
| exact | search | 768 x 128 (1) | True | 8.0942 | 8.0346 | 32.4 | 26.8 |
| exact | direct | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.9 | 12.7 |
| exact | search | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.9 | 16.6 |
| exact | direct | 512 x 128 (1) | True | 8.1100 | 8.0346 | 32.8 | 26.9 |
| exact | search | 512 x 128 (1) | True | 8.1100 | 8.0346 | 32.8 | 30.9 |
| uniform | direct | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 33.9 | 38.2 |
| uniform | search | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 33.9 | 43.5 |
| uniform | direct | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 21.8 | 23.1 |
| uniform | search | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 21.8 | 28.2 |
| uniform | direct | 768 x 128 (1) | True | 8.1283 | 8.0346 | 34.5 | 30.0 |
| uniform | search | 768 x 128 (1) | True | 8.1283 | 8.0346 | 34.5 | 37.3 |
| uniform | direct | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.5 | 16.9 |
| uniform | search | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.5 | 22.7 |
| uniform | direct | 512 x 128 (1) | True | 8.1441 | 8.0346 | 37.7 | 41.2 |
| uniform | search | 512 x 128 (1) | True | 8.1441 | 8.0346 | 37.7 | 50.4 |

Decode estimate per predictor (fastest exact integer format, else BF16, CUDA graph if faster; then the fastest coder, uniform mode; one after the other, not overlapped). Bits include that coder's measured overhead. Link it keeps up with = decoded values/s x bits per value.

| Predictor | Bits per value | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- |
| P, 1 pivot step (int8x8) | 9.017 | 14.1 | 28.2 | 127 Gb/s |
| P, 16 pivot steps, rank 128 (int8x8) | 8.733 | 12.7 | 25.4 | 111 Gb/s |
| P, 4 pivot steps (int8x8) | 8.359 | 11.3 | 22.7 | 95 Gb/s |
| P, 8 pivot steps (int8x8) | 8.219 | 11.0 | 22.0 | 91 Gb/s |
| P, 16 pivot steps (int8x8) | 8.148 | 10.2 | 20.4 | 83 Gb/s |
| L0+P2+P, 16 pivot steps (int8x8) | 7.979 | 6.9 | 13.7 | 55 Gb/s |
| L0+P2+P+ATT, 16 pivot steps (int8x8) | 7.863 | 0.5 | 1.0 | 4 Gb/s |

Overlap, exact coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.04 ms + decode 4.71 ms one after the other, 6.53 ms together, so 10.3 G values/s (20.6 GB/s), keeping up with a 92 Gb/s link.

Overlap, exact coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.74 ms + decode 4.69 ms one after the other, 7.86 ms together, so 8.5 G values/s (17.1 GB/s), keeping up with a 69 Gb/s link.

Overlap, exact coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.22 ms + decode 4.58 ms one after the other, 8.11 ms together, so 8.3 G values/s (16.6 GB/s), keeping up with a 67 Gb/s link.

Overlap, uniform coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.11 ms + decode 2.94 ms one after the other, 4.88 ms together, so 13.7 G values/s (27.5 GB/s), keeping up with a 122 Gb/s link.

Overlap, uniform coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.64 ms + decode 2.88 ms one after the other, 6.36 ms together, so 10.5 G values/s (21.1 GB/s), keeping up with a 86 Gb/s link.

Overlap, uniform coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.27 ms + decode 2.88 ms one after the other, 6.61 ms together, so 10.1 G values/s (20.3 GB/s), keeping up with a 82 Gb/s link.

Whole-document decode, fused one-step codec (Triton, measured): test_text, 28 layers x 8192 tokens = 470M values, encoded and decoded layer by layer as a receiver must: one kernel predicts each layer from the decoded previous layer (int8 tensor-core product, RoPE, spread), decodes it and makes the next layer's features. Times are the kernels end to end (CUDA events); the decode must be bit-exact. Bits are the stream's own (states included); the predictor's exact-model bits show the cost of its int8 coefficients and features.

| Tokens per program x rows (warps) | Mode | Stage 1 | Bit-exact | Stream bits/value | Predictor exact-model bits/value | Encoder forward + encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 8 (8) | uniform | direct | True | 9.0114 | 8.9108 | 37.8 + 56.0 | 60.9 | 7.7 | 15.4 | 70 Gb/s |
| 64 x 8 (8) | uniform | direct | True | 9.0045 | 8.9108 | 58.0 + 77.5 | 79.2 | 5.9 | 11.9 | 53 Gb/s |
| 32 x 4 (4) | uniform | direct | True | 9.0045 | 8.9108 | 37.2 + 63.9 | 58.4 | 8.1 | 16.1 | 72 Gb/s |
| 16 x 4 (4) | uniform | direct | True | 9.0114 | 8.9108 | 53.3 + 74.6 | 92.8 | 5.1 | 10.1 | 46 Gb/s |
| 64 x 16 (16) | uniform | direct | True | 9.0114 | 8.9108 | 96.3 + 118.5 | 111.0 | 4.2 | 8.5 | 38 Gb/s |
| 32 x 4 (4) | uniform | search | True | 9.0045 | 8.9108 | 35.3 + 65.0 | 63.1 | 7.4 | 14.9 | 67 Gb/s |
| 32 x 4 (4) | exact | direct | True | 8.9447 | 8.9108 | 39.2 + 54.2 | 98.5 | 4.8 | 9.5 | 43 Gb/s |

Reference: SplitZip's native codec on the same RTX A4500 (official run, branch experiment/llama32_official_gpu, benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517): 12.04 bits per value, encode 137 GB/s, decode 311 GB/s.


Our codec alone (exact coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.87 bits/value, 10.3 G values/s); P, 8 pivot steps (8.04 bits/value, 8.5 G values/s); P, 16 pivot steps (7.97 bits/value, 8.3 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 10 Gb/s | 752 ms | 565 ms | 374 ms (16 pivot steps) | 374 ms (28 16 pivot steps) | 1.51x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.51x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.51x |
| 100 Gb/s | 75 ms | 56 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 1.24x |
| 200 Gb/s | 38 ms | 28 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 0.62x |
| 400 Gb/s | 19 ms | 14 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 0.31x |
| 700 Gb/s | 11 ms | 8 ms | 46 ms (1 pivot step) | 46 ms (28 1 pivot step) | 0.18x |

Our codec alone (uniform coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.90 bits/value, 13.7 G values/s); P, 8 pivot steps (8.08 bits/value, 10.5 G values/s); P, 16 pivot steps (8.01 bits/value, 10.1 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 10 Gb/s | 752 ms | 565 ms | 376 ms (16 pivot steps) | 376 ms (28 16 pivot steps) | 1.50x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.50x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.50x |
| 100 Gb/s | 75 ms | 56 ms | 42 ms (1 pivot step) | 39 ms (15 1 pivot step, 8 8 pivot steps, 5 16 pivot steps) | 1.44x |
| 200 Gb/s | 38 ms | 28 ms | 34 ms (1 pivot step) | 34 ms (28 1 pivot step) | 0.83x |
| 400 Gb/s | 19 ms | 14 ms | 34 ms (1 pivot step) | 34 ms (28 1 pivot step) | 0.41x |
| 700 Gb/s | 11 ms | 8 ms | 34 ms (1 pivot step) | 34 ms (28 1 pivot step) | 0.24x |

Our codec alone, measured end to end (fused one-step codec, whole test_text: 470M values, 9.004 bits/value, decoded layer by layer at 8.1 G values/s; 32 x 4 (4), uniform). SplitZip: 12.021 bits/value on the same document, its measured A4500 decode speed. Ideal link as above.

| Link | Raw BF16 | SplitZip | Ours (fused, measured) | Ours vs SplitZip |
| --- | --- | --- | --- | --- |
| 10 Gb/s | 752 ms | 565 ms | 423 ms | 1.33x |
| 25 Gb/s | 301 ms | 226 ms | 169 ms | 1.33x |
| 50 Gb/s | 150 ms | 113 ms | 85 ms | 1.33x |
| 100 Gb/s | 75 ms | 56 ms | 58 ms | 0.97x |
| 200 Gb/s | 38 ms | 28 ms | 58 ms | 0.48x |
| 400 Gb/s | 19 ms | 14 ms | 58 ms | 0.24x |
| 700 Gb/s | 11 ms | 8 ms | 58 ms | 0.14x |

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
