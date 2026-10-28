# KV transfer codecs on unsloth/Llama-3.2-3B

Captures: 9 documents x 8192 tokens; 28 layers, 8 KV heads x 128; torch 2.11.0+cu128, transformers 4.57.6.

## Lossless (bits per value; BF16 = 16)

| Method | Four-chunk average | PEP 8 whole | argparse.py whole | Predictor multiply-adds per value |
| --- | --- | --- | --- | --- |
| SplitZip-style | 12.021 | 12.021 | 12.022 | 0 |
| ExANS-style | 10.722 | 10.707 | 10.711 | 0 |
| ExANS, official codec (measured stream) | - | 10.864 | 10.874 | 0 |
| Our 44% version (layer l-1) | 8.962 | 8.951 | 8.716 | 2,048 |
| Our Block (+ earlier groups) | 8.235 | 8.228 | 7.911 | 3,008 |
| Our strongest no-hint | 7.922 | 7.917 | 7.624 | 9,153 |
| Our strong-and-fast | 7.821 | 7.831 | 7.541 | 13,379 |

ExANS official: de6c38b built with one fix (return fail_cuda(codec, status__, #member); -> return fail_cuda(&*codec, status__, #member);); the authors' test passed; 940 blocks of 499,712 values, all bit-exact: test_text 10.864 bits/value, decode 54 GB/s (encode 5 GB/s), test_code 10.874 bits/value, decode 54 GB/s (encode 5 GB/s) (wall clock of the codec calls, raw BF16 bytes).

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
| P, 1 pivot step | fp32 | 4.02 | 4.2 | 8.3 | 4.55, 3.7 |
| P, 1 pivot step | bf16 | 1.01 | 16.6 | 33.2 | 0.97, 17.2 |
| P, 1 pivot step | int8x16 | 3.46 | 4.8 | 9.7 | 3.41, 4.9 |
| P, 1 pivot step | int8x8 | 0.53 | 31.5 | 63.0 | 0.53, 31.5 |
| P, 1 pivot step | int8x8s | 0.70 | 24.1 | 48.2 | 0.39, 43.5 |
| P, 16 pivot steps, rank 128 | fp32 | 3.35 | 5.0 | 10.0 | 3.68, 4.6 |
| P, 16 pivot steps, rank 128 | bf16 | 1.12 | 14.9 | 29.9 | 1.03, 16.2 |
| P, 16 pivot steps, rank 128 | int8x16 | 6.32 | 2.7 | 5.3 | 6.14, 2.7 |
| P, 16 pivot steps, rank 128 | int8x8 | 0.69 | 24.2 | 48.3 | 0.66, 25.5 |
| P, 4 pivot steps | fp32 | 7.12 | 2.4 | 4.7 | 6.93, 2.4 |
| P, 4 pivot steps | bf16 | 1.46 | 11.5 | 23.0 | 1.41, 11.9 |
| P, 4 pivot steps | int8x16 | 5.86 | 2.9 | 5.7 | 5.78, 2.9 |
| P, 4 pivot steps | int8x8 | 0.82 | 20.5 | 41.0 | 0.82, 20.4 |
| P, 4 pivot steps | int8x8s | 0.99 | 16.9 | 33.7 | 0.67, 25.1 |
| P, 8 pivot steps | fp32 | 7.61 | 2.2 | 4.4 | 7.62, 2.2 |
| P, 8 pivot steps | bf16 | 1.70 | 9.9 | 19.8 | 1.61, 10.4 |
| P, 8 pivot steps | int8x16 | 6.34 | 2.6 | 5.3 | 6.22, 2.7 |
| P, 8 pivot steps | int8x8 | 0.89 | 18.9 | 37.8 | 0.87, 19.3 |
| P, 8 pivot steps | int8x8s | 1.04 | 16.1 | 32.1 | 0.72, 23.3 |
| P, 16 pivot steps | fp32 | 8.09 | 2.1 | 4.1 | 8.06, 2.1 |
| P, 16 pivot steps | bf16 | 1.93 | 8.7 | 17.4 | 1.78, 9.4 |
| P, 16 pivot steps | int8x16 | 6.85 | 2.5 | 4.9 | 6.66, 2.5 |
| P, 16 pivot steps | int8x8 | 1.02 | 16.5 | 33.0 | 0.98, 17.2 |
| P, 16 pivot steps | int8x8s | 1.23 | 13.7 | 27.3 | 0.85, 19.8 |
| L0+P2+P, 16 pivot steps | fp32 | 17.70 | 0.9 | 1.9 | 17.54, 1.0 |
| L0+P2+P, 16 pivot steps | bf16 | 3.72 | 4.5 | 9.0 | 3.46, 4.8 |
| L0+P2+P, 16 pivot steps | int8x16 | 8.62 | 1.9 | 3.9 | 8.33, 2.0 |
| L0+P2+P, 16 pivot steps | int8x8 | 1.88 | 8.9 | 17.9 | 1.78, 9.4 |
| L0+P2+P, 16 pivot steps | int8x8s | 1.75 | 9.6 | 19.1 | 1.31, 12.8 |
| L0+P2+P+ATT, 16 pivot steps | fp32 | 63.86 | 0.3 | 0.5 |  |
| L0+P2+P+ATT, 16 pivot steps | bf16 | 35.00 | 0.5 | 1.0 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x16 | 39.94 | 0.4 | 0.8 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x8 | 32.97 | 0.5 | 1.0 |  |

Value-slice coder (Triton) on 67,108,864 real values (lite predictions, layers [4, 12, 20, 27]); the round trip must be bit-exact.

Stage 1: how the decoder finds the slice symbol (direct: a table and one comparison; search: a coarse table and a binary search); the streams are the same.

| Mode | Stage 1 | Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | direct | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 28.6 | 24.8 |
| exact | search | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 28.6 | 27.4 |
| exact | direct | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 20.5 | 15.1 |
| exact | search | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 20.5 | 15.8 |
| exact | direct | 768 x 128 (1) | True | 8.0942 | 8.0346 | 29.2 | 21.7 |
| exact | search | 768 x 128 (1) | True | 8.0942 | 8.0346 | 29.2 | 26.8 |
| exact | direct | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.6 | 13.7 |
| exact | search | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.6 | 17.5 |
| exact | direct | 512 x 128 (1) | True | 8.1100 | 8.0346 | 32.0 | 26.7 |
| exact | search | 512 x 128 (1) | True | 8.1100 | 8.0346 | 32.0 | 30.3 |
| uniform | direct | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 32.6 | 38.1 |
| uniform | search | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 32.6 | 44.4 |
| uniform | direct | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 21.9 | 25.0 |
| uniform | search | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 21.9 | 30.7 |
| uniform | direct | 768 x 128 (1) | True | 8.1283 | 8.0346 | 35.3 | 29.8 |
| uniform | search | 768 x 128 (1) | True | 8.1283 | 8.0346 | 35.3 | 37.7 |
| uniform | direct | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.6 | 18.2 |
| uniform | search | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.6 | 24.2 |
| uniform | direct | 512 x 128 (1) | True | 8.1441 | 8.0346 | 36.1 | 39.7 |
| uniform | search | 512 x 128 (1) | True | 8.1441 | 8.0346 | 36.1 | 49.2 |

Decode estimate per predictor (fastest exact integer format, else BF16, CUDA graph if faster; then the fastest coder, uniform mode; one after the other, not overlapped). Bits include that coder's measured overhead. Link it keeps up with = decoded values/s x bits per value.

| Predictor | Bits per value | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- |
| P, 1 pivot step (int8x8) | 9.017 | 13.8 | 27.6 | 125 Gb/s |
| P, 16 pivot steps, rank 128 (int8x8) | 8.733 | 12.5 | 25.1 | 109 Gb/s |
| P, 4 pivot steps (int8x8) | 8.359 | 11.2 | 22.4 | 93 Gb/s |
| P, 8 pivot steps (int8x8) | 8.219 | 10.8 | 21.6 | 89 Gb/s |
| P, 16 pivot steps (int8x8) | 8.148 | 10.1 | 20.2 | 82 Gb/s |
| L0+P2+P, 16 pivot steps (int8x8) | 7.979 | 6.8 | 13.6 | 54 Gb/s |
| L0+P2+P+ATT, 16 pivot steps (int8x8) | 7.863 | 0.5 | 1.0 | 4 Gb/s |

Overlap, exact coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.05 ms + decode 4.96 ms one after the other, 6.65 ms together, so 10.1 G values/s (20.2 GB/s), keeping up with a 90 Gb/s link.

Overlap, exact coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.70 ms + decode 4.83 ms one after the other, 8.04 ms together, so 8.4 G values/s (16.7 GB/s), keeping up with a 68 Gb/s link.

Overlap, exact coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.42 ms + decode 4.79 ms one after the other, 8.33 ms together, so 8.1 G values/s (16.1 GB/s), keeping up with a 65 Gb/s link.

Overlap, uniform coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.16 ms + decode 2.98 ms one after the other, 5.00 ms together, so 13.4 G values/s (26.9 GB/s), keeping up with a 120 Gb/s link.

Overlap, uniform coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.73 ms + decode 2.98 ms one after the other, 6.45 ms together, so 10.4 G values/s (20.8 GB/s), keeping up with a 84 Gb/s link.

Overlap, uniform coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.43 ms + decode 2.97 ms one after the other, 6.87 ms together, so 9.8 G values/s (19.5 GB/s), keeping up with a 79 Gb/s link.

Whole-document decode, fused one-step codec (Triton, measured): test_text, 28 layers x 8192 tokens = 470M values, encoded and decoded layer by layer as a receiver must: one kernel predicts each layer from the decoded previous layer (int8 tensor-core product, RoPE, spread), decodes it and makes the next layer's features. Times are the kernels end to end (CUDA events); the decode must be bit-exact. Bits are the stream's own (states included); the predictor's exact-model bits show the cost of its int8 coefficients and features.

| Tokens per program x rows (warps) | Mode | Stage 1 | Bit-exact | Stream bits/value | Predictor exact-model bits/value | Encoder forward + encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 8 (8) | uniform | direct | True | 9.0114 | 8.9108 | 42.2 + 56.9 | 62.2 | 7.6 | 15.1 | 68 Gb/s |
| 64 x 8 (8) | uniform | direct | True | 9.0045 | 8.9108 | 58.1 + 76.5 | 79.6 | 5.9 | 11.8 | 53 Gb/s |
| 32 x 4 (4) | uniform | direct | True | 9.0045 | 8.9108 | 41.6 + 65.2 | 59.3 | 7.9 | 15.8 | 71 Gb/s |
| 16 x 4 (4) | uniform | direct | True | 9.0114 | 8.9108 | 54.1 + 76.7 | 96.3 | 4.9 | 9.8 | 44 Gb/s |
| 64 x 16 (16) | uniform | direct | True | 9.0114 | 8.9108 | 96.9 + 119.0 | 112.1 | 4.2 | 8.4 | 38 Gb/s |
| 32 x 4 (4) | uniform | search | True | 9.0045 | 8.9108 | 41.8 + 65.1 | 63.9 | 7.4 | 14.7 | 66 Gb/s |
| 32 x 4 (4) | exact | direct | True | 8.9447 | 8.9108 | 42.8 + 55.4 | 101.0 | 4.6 | 9.3 | 42 Gb/s |

Whole-document decode, per-layer one-step codec (measured; same document, predictor and checks): for each layer a cuBLAS int8 product (the previous layer's 8-bit features times the 8-bit coefficients, exact int32), then one Triton kernel of one-warp programs (a token block x one head) that scales, rotates, codes and makes the next layer's features. Streams stay open across layers. Chunks: the tokens are split into that many parts, each decoded on its own CUDA stream, so one part's product runs while another part decodes. Encode time excludes the encoder's forward pass (as in the fused table).

| Tokens per program | Chunks | Mode | Stage 1 | Variant | Bit-exact | Stream bits/value | Encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 1 | uniform | search | - | True | 9.0045 | 37.0 | 37.2 | 12.6 | 25.2 | 114 Gb/s |
| 128 | 2 | uniform | search | - | True | 9.0045 | 37.0 | 34.4 | 13.6 | 27.3 | 123 Gb/s |
| 128 | 4 | uniform | search | - | True | 9.0045 | 37.1 | 33.3 | 14.1 | 28.2 | 127 Gb/s |
| 64 | 2 | uniform | search | - | True | 9.0114 | 35.1 | 32.8 | 14.3 | 28.6 | 129 Gb/s |
| 256 | 2 | uniform | search | - | True | 9.0010 | 39.2 | 38.8 | 12.1 | 24.2 | 109 Gb/s |
| 32 | 2 | uniform | search | - | True | 9.0252 | 33.7 | 31.7 | 14.8 | 29.6 | 134 Gb/s |
| 32 | 2 | uniform | search | - | True | 9.0252 | 33.9 | 31.7 | 14.8 | 29.6 | 134 Gb/s |
| 32 | 2 | uniform | direct | - | True | 9.0252 | 34.0 | 32.0 | 14.7 | 29.3 | 132 Gb/s |
| 32 | 2 | uniform | search | CUDA graph | True | 9.0252 | 33.7 | 31.7 | 14.8 | 29.6 | 134 Gb/s |
| 32 | 2 | uniform | search | Triton product 128x128x64 (4 warps) | True | 9.0252 | 34.4 | 32.4 | 14.5 | 29.0 | 131 Gb/s |
| 32 | 2 | uniform | search | Triton product 128x128x64 (4 warps), CUDA graph | True | 9.0252 | 34.1 | 32.3 | 14.5 | 29.1 | 131 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x128x64 (4 warps) | True | 9.0252 | 35.6 | 33.8 | 13.9 | 27.8 | 125 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x128x64 (4 warps), CUDA graph | True | 9.0252 | 35.8 | 34.0 | 13.8 | 27.7 | 125 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x64x64 (4 warps) | True | 9.0252 | 39.6 | 37.5 | 12.5 | 25.1 | 113 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x64x64 (4 warps), CUDA graph | True | 9.0252 | 39.4 | 37.4 | 12.6 | 25.1 | 113 Gb/s |
| 32 | 4 | uniform | search | Triton product 64x128x64 (4 warps), CUDA graph | True | 9.0252 | 36.0 | 34.3 | 13.7 | 27.4 | 124 Gb/s |
| 32 | 4 | uniform | search | CUDA graph | True | 9.0252 | 34.2 | 31.7 | 14.8 | 29.6 | 134 Gb/s |
| 32 | 8 | uniform | search | Triton product 64x128x64 (4 warps), CUDA graph | True | 9.0252 | 36.1 | 33.8 | 13.9 | 27.8 | 125 Gb/s |
| 32 | 8 | uniform | search | CUDA graph | True | 9.0252 | 34.2 | 33.7 | 13.9 | 27.9 | 126 Gb/s |
| 32 | 2 | exact | search | - | True | 8.9654 | 37.0 | 44.0 | 10.7 | 21.3 | 96 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512) | True | 9.2214 | 29.8 | 28.3 | 16.6 | 33.3 | 153 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 64x128x64 (4 warps)) | True | 9.2214 | 28.5 | 26.9 | 17.5 | 34.9 | 161 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.1 | 25.5 | 18.4 | 36.9 | 170 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 64x64x64 (4 warps)) | True | 9.2214 | 29.4 | 28.3 | 16.6 | 33.2 | 153 Gb/s |
| 32 | 1 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.0 | 27.1 | 17.4 | 34.7 | 160 Gb/s |
| 32 | 4 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.2 | 26.9 | 17.5 | 35.0 | 161 Gb/s |
| 32 | 2 | uniform | search | CUDA graph, reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.3 | 25.6 | 18.4 | 36.8 | 169 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384) | True | 9.2960 | 28.2 | 26.3 | 17.9 | 35.8 | 166 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 64x128x64 (4 warps)) | True | 9.2960 | 26.7 | 25.5 | 18.4 | 36.8 | 171 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.4 | 24.9 | 18.9 | 37.8 | 176 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 64x64x64 (4 warps)) | True | 9.2960 | 27.6 | 26.1 | 18.0 | 36.0 | 167 Gb/s |
| 32 | 1 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.4 | 25.3 | 18.6 | 37.1 | 173 Gb/s |
| 32 | 4 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.4 | 25.3 | 18.6 | 37.1 | 173 Gb/s |
| 32 | 2 | uniform | search | CUDA graph, reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.4 | 25.3 | 18.6 | 37.2 | 173 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256) | True | 9.4049 | 25.3 | 24.4 | 19.3 | 38.6 | 181 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 64x128x64 (4 warps)) | True | 9.4049 | 24.4 | 23.6 | 19.9 | 39.7 | 187 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.3 | 22.8 | 20.6 | 41.2 | 194 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 64x64x64 (4 warps)) | True | 9.4049 | 25.1 | 24.5 | 19.2 | 38.4 | 181 Gb/s |
| 32 | 1 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.3 | 23.5 | 20.0 | 40.0 | 188 Gb/s |
| 32 | 4 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.4 | 23.4 | 20.1 | 40.2 | 189 Gb/s |
| 32 | 2 | uniform | search | CUDA graph, reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.3 | 22.7 | 20.7 | 41.3 | 194 Gb/s |

Where the decode time goes (32 tokens per program, 2 chunk(s)): the 27 products alone 14.9 ms, the 28 coding kernels alone (products precomputed) 15.8 ms, both as decoded 31.7 ms; overlap hides -1.0 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), Triton product 64x128x64 (4 warps)): the 27 products alone 18.2 ms, the 28 coding kernels alone (products precomputed) 16.4 ms, both as decoded 33.8 ms; overlap hides 0.8 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 512)): the 27 products alone 11.2 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 28.3 ms; overlap hides -1.4 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 512, requantized in a Triton product epilogue 64x128x64 (4 warps))): the 27 products alone 9.1 ms, the 28 coding kernels alone (products precomputed) 15.8 ms, both as decoded 26.9 ms; overlap hides -2.0 ms.

Where the decode time goes (32 tokens per program, 1 chunk(s), reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps))): the 27 products alone 10.3 ms, the 28 coding kernels alone (products precomputed) 16.8 ms, both as decoded 27.1 ms; overlap hides -0.0 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 384)): the 27 products alone 9.5 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 26.3 ms; overlap hides -1.1 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 384, requantized in a Triton product epilogue 64x128x64 (4 warps))): the 27 products alone 7.4 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 25.5 ms; overlap hides -2.5 ms.

Where the decode time goes (32 tokens per program, 1 chunk(s), reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps))): the 27 products alone 8.7 ms, the 28 coding kernels alone (products precomputed) 16.6 ms, both as decoded 25.3 ms; overlap hides 0.1 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 256)): the 27 products alone 7.5 ms, the 28 coding kernels alone (products precomputed) 15.5 ms, both as decoded 24.4 ms; overlap hides -1.3 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 256, requantized in a Triton product epilogue 64x128x64 (4 warps))): the 27 products alone 5.5 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 23.6 ms; overlap hides -2.5 ms.

Where the decode time goes (32 tokens per program, 1 chunk(s), reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps))): the 27 products alone 6.9 ms, the 28 coding kernels alone (products precomputed) 16.6 ms, both as decoded 23.5 ms; overlap hides 0.0 ms.

Reference: SplitZip's native codec on the same RTX A4500 (official run, branch experiment/llama32_official_gpu, benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517): 12.04 bits per value, encode 137 GB/s, decode 311 GB/s.


Our codec alone (exact coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.87 bits/value, 10.1 G values/s); P, 8 pivot steps (8.04 bits/value, 8.4 G values/s); P, 16 pivot steps (7.97 bits/value, 8.1 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 7516 ms | 5647 ms | 3745 ms (16 pivot steps) | 3745 ms (28 16 pivot steps) | 1.51x |
| 5 Gb/s | 1503 ms | 1129 ms | 749 ms (16 pivot steps) | 749 ms (28 16 pivot steps) | 1.51x |
| 10 Gb/s | 752 ms | 565 ms | 374 ms (16 pivot steps) | 374 ms (28 16 pivot steps) | 1.51x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.51x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.51x |
| 100 Gb/s | 75 ms | 56 ms | 47 ms (1 pivot step) | 47 ms (28 1 pivot step) | 1.21x |
| 200 Gb/s | 38 ms | 28 ms | 47 ms (1 pivot step) | 47 ms (28 1 pivot step) | 0.61x |
| 400 Gb/s | 19 ms | 14 ms | 47 ms (1 pivot step) | 47 ms (28 1 pivot step) | 0.30x |
| 800 Gb/s | 9 ms | 7 ms | 47 ms (1 pivot step) | 47 ms (28 1 pivot step) | 0.15x |

Our codec alone (uniform coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.90 bits/value, 13.4 G values/s); P, 8 pivot steps (8.08 bits/value, 10.4 G values/s); P, 16 pivot steps (8.01 bits/value, 9.8 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 7516 ms | 5647 ms | 3761 ms (16 pivot steps) | 3761 ms (28 16 pivot steps) | 1.50x |
| 5 Gb/s | 1503 ms | 1129 ms | 752 ms (16 pivot steps) | 752 ms (28 16 pivot steps) | 1.50x |
| 10 Gb/s | 752 ms | 565 ms | 376 ms (16 pivot steps) | 376 ms (28 16 pivot steps) | 1.50x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.50x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.50x |
| 100 Gb/s | 75 ms | 56 ms | 42 ms (1 pivot step) | 40 ms (16 1 pivot step, 11 8 pivot steps, 1 16 pivot steps) | 1.43x |
| 200 Gb/s | 38 ms | 28 ms | 35 ms (1 pivot step) | 35 ms (28 1 pivot step) | 0.81x |
| 400 Gb/s | 19 ms | 14 ms | 35 ms (1 pivot step) | 35 ms (28 1 pivot step) | 0.40x |
| 800 Gb/s | 9 ms | 7 ms | 35 ms (1 pivot step) | 35 ms (28 1 pivot step) | 0.20x |

Our codec alone, measured end to end (whole test_text: 470M values, decoded layer by layer, bit-exact). For each link the mode with the shortest transfer among: fused 32x8, uniform (9.011 bits/value, 7.6 G values/s); fused 64x8, uniform (9.004 bits/value, 5.9 G values/s); fused 32x4, uniform (9.004 bits/value, 7.9 G values/s); fused 16x4, uniform (9.011 bits/value, 4.9 G values/s); fused 64x16, uniform (9.011 bits/value, 4.2 G values/s); fused 32x4, exact (8.945 bits/value, 4.6 G values/s); per layer, dense product, uniform (9.025 bits/value, 14.8 G values/s); per layer, dense product, exact (8.965 bits/value, 10.7 G values/s); per layer, rank 512 product, uniform (9.221 bits/value, 18.4 G values/s); per layer, rank 384 product, uniform (9.296 bits/value, 18.9 G values/s); per layer, rank 256 product, uniform (9.405 bits/value, 20.7 G values/s). SplitZip: 12.021 bits/value on the same document, its measured A4500 decode speed. Ideal link as above; the bands are typical deployments (see the doc).

| Link | Typical setting | Raw BF16 | SplitZip | ExANS | Ours (measured) | Mode | Ours vs SplitZip | Ours vs ExANS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Gb/s | ordinary cloud, storage | 7516 ms | 5647 ms | 5103 ms | 4202 ms | fused 32x4, exact | 1.34x | 1.21x |
| 5 Gb/s | ordinary cloud | 1503 ms | 1129 ms | 1021 ms | 840 ms | fused 32x4, exact | 1.34x | 1.21x |
| 10 Gb/s | consumer, cloud Ethernet | 752 ms | 565 ms | 510 ms | 420 ms | fused 32x4, exact | 1.34x | 1.21x |
| 25 Gb/s | cloud Ethernet | 301 ms | 226 ms | 204 ms | 168 ms | fused 32x4, exact | 1.34x | 1.21x |
| 50 Gb/s | workstation | 150 ms | 113 ms | 102 ms | 84 ms | per layer, dense product, exact | 1.34x | 1.21x |
| 100 Gb/s | workstation, cross-cluster | 75 ms | 56 ms | 51 ms | 42 ms | per layer, dense product, uniform | 1.33x | 1.20x |
| 200 Gb/s | datacenter RDMA, per GPU | 38 ms | 28 ms | 26 ms | 23 ms | per layer, rank 256 product, uniform | 1.24x | 1.12x |
| 400 Gb/s | datacenter RDMA, per GPU | 19 ms | 14 ms | 18 ms | 23 ms | per layer, rank 256 product, uniform | 0.62x | 0.77x |
| 800 Gb/s | next-generation RDMA | 9 ms | 7 ms | 18 ms | 23 ms | per layer, rank 256 product, uniform | 0.31x | 0.77x |


Hybrid with SplitZip's format (not our codec alone; estimated from measured parts): each of the 28 layers of test_text coded either by our per-layer codec or by SplitZip's format, chosen per link to minimise the transfer. Our modes: dense, uniform (9.025 bits/value, 31.7 ms); dense, exact (8.965 bits/value, 44.0 ms); rank 512, uniform (9.221 bits/value, 25.5 ms); rank 384, uniform (9.296 bits/value, 24.9 ms); rank 256, uniform (9.405 bits/value, 22.7 ms). SplitZip: 12.021 bits/value, 311 GB/s decode, plus a features pass of 0.168 ms per SplitZip layer. Not modelled: a layer after a SplitZip layer has no surprise prior (slightly more bits), and SplitZip layers could also decode beside ours.

| Link | SplitZip | Ours alone (measured) | Hybrid | Layers ours (mode) | Hybrid bits/value | Hybrid vs SplitZip |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 5646.9 ms | 4211.6 ms | 4211.6 ms | 28 (dense, exact) | 8.965 | 1.34x |
| 5 Gb/s | 1129.4 ms | 842.3 ms | 842.3 ms | 28 (dense, exact) | 8.965 | 1.34x |
| 10 Gb/s | 564.7 ms | 421.2 ms | 421.2 ms | 28 (dense, exact) | 8.965 | 1.34x |
| 25 Gb/s | 225.9 ms | 168.5 ms | 168.5 ms | 28 (dense, exact) | 8.965 | 1.34x |
| 50 Gb/s | 112.9 ms | 84.2 ms | 84.2 ms | 28 (dense, exact) | 8.965 | 1.34x |
| 100 Gb/s | 56.5 ms | 42.4 ms | 42.4 ms | 28 (dense, uniform) | 9.025 | 1.33x |
| 200 Gb/s | 28.2 ms | 22.7 ms | 22.2 ms | 27 (rank 256, uniform) | 9.436 | 1.27x |
| 400 Gb/s | 14.1 ms | 22.7 ms | 13.1 ms | 9 (rank 256, uniform) | 11.116 | 1.08x |
| 800 Gb/s | 7.1 ms | 22.7 ms | 7.7 ms | 0 (dense, uniform) | 12.021 | 0.91x |

## GPU-resident KV pool (NVIDIA RTX A4500, measured)

A pool keeps finished requests' KV so that a later request with the same prefix skips its prefill. Bringing back one cached prefix of 8192 tokens (test_text, 28 layers, 896 MiB of raw BF16 KV). Times are CUDA-event medians; our decode is bit-exact. Copies are host to GPU over this machine's PCIe link.

| Way to bring the prefix back | Time | Kept in | Memory held |
| --- | --- | --- | --- |
| Recompute: the model's prefill | 988 ms | nothing | 0 |
| Raw KV kept in GPU memory | 0 ms | GPU | 896 MiB |
| Raw KV in CPU memory, copied back (pinned / pageable) | 39 / 68 ms | CPU | 896 MiB |
| Ours (per layer, uniform, search) kept in GPU memory, decoded | 34 ms | GPU | 504 MiB |
| Ours (per layer, uniform) in CPU memory: copy, then decode | 22 + 34 ms | CPU | 504 MiB |
| Ours (per layer, exact, search) kept in GPU memory, decoded | 48 ms | GPU | 501 MiB |
| Ours (per layer, exact) in CPU memory: copy, then decode | 22 + 48 ms | CPU | 501 MiB |
| Ours (fused, uniform) kept in GPU memory, decoded | 58 ms | GPU | 504 MiB |
| Ours (fused, uniform) in CPU memory: copy, then decode | 22 + 58 ms | CPU | 504 MiB |
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

## Our prediction inside SplitZip's format (bits only; scripts: kv_compare/szformat.py)

Fixed-length codes as in SplitZip (no entropy coder: every value can be placed independently), with the code book of each value chosen by our one-step prediction; a value's code length depends only on its context, which the decoder knows from the prediction. Code books from the calibration documents (train0, train1), scored on the test documents. The decoder still computes the prediction (the products, timed above) but no rANS. See kv_compare/szformat.py for the variants.

| Variant | dense (text / code) | rank 256 (text / code) | Mean vs SplitZip-style |
| --- | --- | --- | --- |
| SplitZip-style | 12.021 (12.021 / 12.022) | 12.021 (12.021 / 12.022) | +0.000 |
| exponent, context | 11.075 (11.117 / 11.033) | 11.216 (11.256 / 11.177) | -0.947 |
| sign + exponent, context | 10.579 (10.647 / 10.511) | 10.809 (10.872 / 10.747) | -1.442 |
| ordered residual | 10.850 (11.014 / 10.685) | 11.217 (11.344 / 11.089) | -1.172 |
| ordered residual, static spread | 11.025 (11.269 / 10.781) | 11.309 (11.479 / 11.139) | -0.997 |
| sign + magnitude | 10.175 (10.291 / 10.059) | 10.511 (10.606 / 10.416) | -1.846 |
| sign + magnitude, static spread | 10.416 (10.645 / 10.187) | 10.624 (10.770 / 10.479) | -1.605 |


## Lossless and lossy side by side (test_text: 470M values per link)

Time to move one document's KV over an ideal link, decoding overlapped with arrival: max(bits / link, decode time) with the decode measured on this GPU; a * marks a method whose decode was not measured (its time on the wire only, so a lower bound). Bits and quality of the lossy rows from the lossy stage (first 3072 tokens of each test document; quality = perplexity change of the next 512 tokens, test_text / test_code).

| Method | Bits per value | Perplexity change | 1 Gb/s (ms) | 10 Gb/s (ms) | 25 Gb/s (ms) | 100 Gb/s (ms) | 200 Gb/s (ms) | 400 Gb/s (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip (lossless) | 12.02 | 0 | 5,647 | 565 | 226 | 56 | 28 | 14 |
| Ours, lossless (best measured mode per link) | 8.94-9.40 | 0 | 4,202 | 420 | 168 | 42 | 23 | 23 |
| Ours, step 0.25 | 5.01 | +0.25% / -0.15% | 2,356* | 236* | 94* | 24* | 12* | 6* |
| Ours, step 0.5 | 4.06 | +0.33% / +0.53% | 1,906* | 191* | 76* | 19* | 10* | 5* |
| Ours, step 1.0 | 3.18 | +1.26% / +0.28% | 1,492* | 149* | 60* | 15* | 7* | 4* |
| Ours, step 2.0 | 2.47 | +3.81% / +0.59% | 1,161* | 116* | 46* | 12* | 6* | 3* |
| Ours, step 4.0 | 2.09 | +14.50% / +1.00% | 982* | 98* | 39* | 10* | 5* | 2* |
| FP8 cast | 8.00 | +0.33% / -0.32% | 3,758* | 376* | 150* | 38* | 19* | 9* |
| AQUA-KV, 2-bit | 2.11 | +1.46% / -0.35% | 989* | 99* | 40* | 10* | 5* | 2* |
| AQUA-KV, 3-bit | 3.07 | +0.06% / -0.23% | 1,441* | 144* | 58* | 14* | 7* | 4* |
| AQUA-KV, 4-bit | 4.03 | -0.08% / -0.10% | 1,894* | 189* | 76* | 19* | 9* | 5* |
| KIVI, 2-bit | 3.27 | +20.79% / +3.46% | 1,537* | 154* | 61* | 15* | 8* | 4* |
| KIVI, 4-bit | 5.23 | +0.28% / +0.23% | 2,456* | 246* | 98* | 25* | 12* | 6* |
| Ours, GPU codec (one-step, measured stream), step 0.25 | 5.00 | +0.86% / +0.02% | 2,351 | 235 | 94 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 0.5 | 4.02 | +0.93% / -0.21% | 1,890 | 189 | 76 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 1.0 | 3.10 | +6.77% / +1.16% | 1,455 | 146 | 58 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 2.0 | 2.34 | +20.39% / +7.01% | 1,098 | 110 | 44 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 4.0 | 1.90 | +776.12% / +416.80% | 892 | 89 | 36 | 27 | 27 | 27 |

## Cheaper one-step products (scripts/probe_cheap_products.py)

Lossless bits with a cheaper product in the one-step predictor (layers 1-27; layer 0 is predicted by its mean in every variant). Halving the product is for fast links, where decode speed matters more than bits.

| Product | Multiply-adds per token | Bits per value (text / code) | vs dense int8 |
| --- | --- | --- | --- |
| dense int8 x int8 | 4,194,304 | 8.863 / 8.651 | +0.000 |
| 2:4 sparse int8 x int8 | 2,097,152 | 9.003 / 8.793 | +0.141 |
| rank W/4 int8 | 2,097,152 | 9.058 / 8.845 | +0.195 |
| rank W/8 int8 | 1,048,576 | 9.241 / 9.044 | +0.386 |
| rank W/4 float | 2,097,152 | 9.055 / 8.841 | +0.191 |
| rank W/8 float | 1,048,576 | 9.239 / 9.042 | +0.384 |

## Lossy (prompt 3072 tokens; quality = perplexity of the next 512 tokens with the transferred KV)

| Method | Bits per value | KV relative error | Perplexity change |
| --- | --- | --- | --- |
| exact KV | 16.00 / 16.00 | 0.00% / 0.00% | +0.00% / +0.00% |
| Ours, step 0.25 | 5.17 / 4.85 | 0.48% / 0.47% | +0.25% / -0.15% |
| Ours, step 0.5 | 4.21 / 3.91 | 0.95% / 0.93% | +0.33% / +0.53% |
| Ours, step 1.0 | 3.31 / 3.04 | 1.88% / 1.84% | +1.26% / +0.28% |
| Ours, step 2.0 | 2.56 / 2.38 | 3.75% / 3.67% | +3.81% / +0.59% |
| Ours, step 4.0 | 2.14 / 2.05 | 7.49% / 7.32% | +14.50% / +1.00% |
| FP8 cast | 8.00 / 8.00 | 2.64% / 2.64% | +0.33% / -0.32% |
| AQUA-KV, 2-bit | 2.11 / 2.11 | 10.00% / 8.83% | +1.46% / -0.35% |
| AQUA-KV, 3-bit | 3.07 / 3.07 | 5.33% / 4.72% | +0.06% / -0.23% |
| AQUA-KV, 4-bit | 4.03 / 4.03 | 2.96% / 2.65% | -0.08% / -0.10% |
| KIVI, 2-bit | 3.27 / 3.27 | 21.94% / 21.17% | +20.79% / +3.46% |
| KIVI, 4-bit | 5.23 / 5.23 | 4.37% / 4.22% | +0.28% / +0.23% |
| Ours, GPU codec (one-step, measured stream), step 0.25 | 5.12 / 4.89 | 0.82% / 0.80% | +0.86% / +0.02% |
| Ours, GPU codec (one-step, measured stream), step 0.5 | 4.14 / 3.91 | 1.63% / 1.60% | +0.93% / -0.21% |
| Ours, GPU codec (one-step, measured stream), step 1.0 | 3.20 / 2.99 | 3.25% / 3.18% | +6.77% / +1.16% |
| Ours, GPU codec (one-step, measured stream), step 2.0 | 2.42 / 2.26 | 6.49% / 6.35% | +20.39% / +7.01% |
| Ours, GPU codec (one-step, measured stream), step 4.0 | 1.94 / 1.86 | 12.96% / 12.67% | +776.12% / +416.80% |

Values are test_text / test_code. Reference: the captured exact KV restored into the cache.
AQUA-KV follows the authors' code (goodevening13/aquakv@e744183f2f3a4a90381f20a6975456871fd1ae29): HIGGS residuals on 2-D EDEN grids, predictors fitted on their own closed-loop reconstructions, layer 0 at 4 bits, the first 4 tokens in BF16; calibrated on this study's calibration documents (the authors use 256 x 8192 RedPajama tokens). KIVI follows the authors' code (jy-yuan/KIVI@876b4d2) with the paper's group size 32 and residual length 128 (recent tokens in BF16); its scales and minima are counted at 16 bits.
Check against the AQUA-KV code (e744183, torch 2.5.1+cu124): HIGGS values identical for 99.99% (2-bit), 99.99% (3-bit), 99.94% (4-bit) (relative error ours / theirs 0.3280 / 0.3280, 0.1720 / 0.1720, 0.0880 / 0.0880); regression weights within 0.0039.
Check against the KIVI code (876b4d2): largest difference bfloat16 2-bit: 0 RMS, bfloat16 4-bit: 0 RMS.
The model's own prefill of the prompt alone differs from it by +0.050% / +0.039% (GPU kernels round differently at another sequence length).
