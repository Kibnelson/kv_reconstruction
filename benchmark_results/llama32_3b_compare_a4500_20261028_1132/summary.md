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
| P, 1 pivot step | fp32 | 4.03 | 4.2 | 8.3 | 4.70, 3.6 |
| P, 1 pivot step | bf16 | 1.01 | 16.6 | 33.2 | 0.95, 17.6 |
| P, 1 pivot step | int8x16 | 3.49 | 4.8 | 9.6 | 3.41, 4.9 |
| P, 1 pivot step | int8x8 | 0.53 | 31.4 | 62.9 | 0.53, 31.8 |
| P, 1 pivot step | int8x8s | 0.70 | 23.8 | 47.7 | 0.39, 43.5 |
| P, 16 pivot steps, rank 128 | fp32 | 3.77 | 4.5 | 8.9 | 3.74, 4.5 |
| P, 16 pivot steps, rank 128 | bf16 | 1.11 | 15.1 | 30.1 | 1.03, 16.2 |
| P, 16 pivot steps, rank 128 | int8x16 | 6.32 | 2.7 | 5.3 | 6.15, 2.7 |
| P, 16 pivot steps, rank 128 | int8x8 | 0.70 | 24.1 | 48.2 | 0.66, 25.5 |
| P, 4 pivot steps | fp32 | 7.01 | 2.4 | 4.8 | 6.91, 2.4 |
| P, 4 pivot steps | bf16 | 1.47 | 11.4 | 22.8 | 1.36, 12.3 |
| P, 4 pivot steps | int8x16 | 5.85 | 2.9 | 5.7 | 5.78, 2.9 |
| P, 4 pivot steps | int8x8 | 0.83 | 20.3 | 40.5 | 0.82, 20.6 |
| P, 4 pivot steps | int8x8s | 1.01 | 16.7 | 33.3 | 0.68, 24.8 |
| P, 8 pivot steps | fp32 | 7.58 | 2.2 | 4.4 | 7.55, 2.2 |
| P, 8 pivot steps | bf16 | 1.70 | 9.8 | 19.7 | 1.55, 10.8 |
| P, 8 pivot steps | int8x16 | 6.32 | 2.7 | 5.3 | 6.22, 2.7 |
| P, 8 pivot steps | int8x8 | 0.89 | 18.9 | 37.8 | 0.87, 19.4 |
| P, 8 pivot steps | int8x8s | 1.06 | 15.9 | 31.8 | 0.72, 23.2 |
| P, 16 pivot steps | fp32 | 8.11 | 2.1 | 4.1 | 8.11, 2.1 |
| P, 16 pivot steps | bf16 | 1.89 | 8.9 | 17.8 | 1.67, 10.0 |
| P, 16 pivot steps | int8x16 | 6.83 | 2.5 | 4.9 | 6.65, 2.5 |
| P, 16 pivot steps | int8x8 | 1.02 | 16.4 | 32.9 | 0.99, 17.0 |
| P, 16 pivot steps | int8x8s | 1.20 | 14.0 | 27.9 | 0.85, 19.7 |
| L0+P2+P, 16 pivot steps | fp32 | 17.64 | 1.0 | 1.9 | 17.53, 1.0 |
| L0+P2+P, 16 pivot steps | bf16 | 3.72 | 4.5 | 9.0 | 3.34, 5.0 |
| L0+P2+P, 16 pivot steps | int8x16 | 8.63 | 1.9 | 3.9 | 8.32, 2.0 |
| L0+P2+P, 16 pivot steps | int8x8 | 1.89 | 8.9 | 17.7 | 1.79, 9.4 |
| L0+P2+P, 16 pivot steps | int8x8s | 1.70 | 9.9 | 19.8 | 1.31, 12.8 |
| L0+P2+P+ATT, 16 pivot steps | fp32 | 63.54 | 0.3 | 0.5 |  |
| L0+P2+P+ATT, 16 pivot steps | bf16 | 34.95 | 0.5 | 1.0 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x16 | 39.86 | 0.4 | 0.8 |  |
| L0+P2+P+ATT, 16 pivot steps | int8x8 | 32.99 | 0.5 | 1.0 |  |

Value-slice coder (Triton) on 67,108,864 real values (lite predictions, layers [4, 12, 20, 27]); the round trip must be bit-exact.

Stage 1: how the decoder finds the slice symbol (direct: a table and one comparison; search: a coarse table and a binary search); the streams are the same.

| Mode | Stage 1 | Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact | direct | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 28.4 | 24.8 |
| exact | search | 1024 x 128 (1) | True | 8.0862 | 8.0346 | 28.4 | 27.4 |
| exact | direct | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 20.3 | 15.1 |
| exact | search | 1024 x 256 (1) | True | 8.0861 | 8.0346 | 20.3 | 15.8 |
| exact | direct | 768 x 128 (1) | True | 8.0942 | 8.0346 | 30.2 | 21.6 |
| exact | search | 768 x 128 (1) | True | 8.0942 | 8.0346 | 30.2 | 26.7 |
| exact | direct | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.4 | 13.7 |
| exact | search | 768 x 256 (1) | True | 8.0941 | 8.0346 | 17.4 | 17.6 |
| exact | direct | 512 x 128 (1) | True | 8.1100 | 8.0346 | 31.3 | 26.0 |
| exact | search | 512 x 128 (1) | True | 8.1100 | 8.0346 | 31.3 | 31.0 |
| uniform | direct | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 32.4 | 38.1 |
| uniform | search | 1024 x 128 (1) | True | 8.1204 | 8.0346 | 32.4 | 44.3 |
| uniform | direct | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 21.9 | 25.1 |
| uniform | search | 1024 x 256 (1) | True | 8.1202 | 8.0346 | 21.9 | 30.5 |
| uniform | direct | 768 x 128 (1) | True | 8.1283 | 8.0346 | 35.6 | 29.5 |
| uniform | search | 768 x 128 (1) | True | 8.1283 | 8.0346 | 35.6 | 37.9 |
| uniform | direct | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.6 | 18.1 |
| uniform | search | 768 x 256 (1) | True | 8.1282 | 8.0346 | 18.6 | 24.1 |
| uniform | direct | 512 x 128 (1) | True | 8.1441 | 8.0346 | 35.7 | 39.7 |
| uniform | search | 512 x 128 (1) | True | 8.1441 | 8.0346 | 35.7 | 49.0 |

Decode estimate per predictor (fastest exact integer format, else BF16, CUDA graph if faster; then the fastest coder, uniform mode; one after the other, not overlapped). Bits include that coder's measured overhead. Link it keeps up with = decoded values/s x bits per value.

| Predictor | Bits per value | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- |
| P, 1 pivot step (int8x8) | 9.017 | 13.8 | 27.7 | 125 Gb/s |
| P, 16 pivot steps, rank 128 (int8x8) | 8.733 | 12.5 | 25.0 | 109 Gb/s |
| P, 4 pivot steps (int8x8) | 8.359 | 11.2 | 22.4 | 93 Gb/s |
| P, 8 pivot steps (int8x8) | 8.219 | 10.8 | 21.6 | 89 Gb/s |
| P, 16 pivot steps (int8x8) | 8.148 | 10.0 | 20.1 | 82 Gb/s |
| L0+P2+P, 16 pivot steps (int8x8) | 7.979 | 6.8 | 13.6 | 54 Gb/s |
| L0+P2+P+ATT, 16 pivot steps (int8x8) | 7.863 | 0.5 | 1.0 | 4 Gb/s |

Overlap, exact coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.06 ms + decode 4.87 ms one after the other, 6.67 ms together, so 10.1 G values/s (20.1 GB/s), keeping up with a 90 Gb/s link.

Overlap, exact coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.86 ms + decode 4.81 ms one after the other, 8.06 ms together, so 8.3 G values/s (16.7 GB/s), keeping up with a 68 Gb/s link.

Overlap, exact coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.42 ms + decode 4.64 ms one after the other, 8.40 ms together, so 8.0 G values/s (16.0 GB/s), keeping up with a 64 Gb/s link.

Overlap, uniform coder (P, 1 pivot step (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 2.14 ms + decode 2.98 ms one after the other, 5.02 ms together, so 13.4 G values/s (26.7 GB/s), keeping up with a 119 Gb/s link.

Overlap, uniform coder (P, 8 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 3.73 ms + decode 3.05 ms one after the other, 6.44 ms together, so 10.4 G values/s (20.9 GB/s), keeping up with a 85 Gb/s link.

Overlap, uniform coder (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.42 ms + decode 2.96 ms one after the other, 6.71 ms together, so 10.0 G values/s (20.0 GB/s), keeping up with a 80 Gb/s link.

Whole-document decode, fused one-step codec (Triton, measured): test_text, 28 layers x 8192 tokens = 470M values, encoded and decoded layer by layer as a receiver must: one kernel predicts each layer from the decoded previous layer (int8 tensor-core product, RoPE, spread), decodes it and makes the next layer's features. Times are the kernels end to end (CUDA events); the decode must be bit-exact. Bits are the stream's own (states included); the predictor's exact-model bits show the cost of its int8 coefficients and features.

| Tokens per program x rows (warps) | Mode | Stage 1 | Bit-exact | Stream bits/value | Predictor exact-model bits/value | Encoder forward + encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 x 8 (8) | uniform | direct | True | 9.0114 | 8.9108 | 44.6 + 57.1 | 62.3 | 7.5 | 15.1 | 68 Gb/s |
| 64 x 8 (8) | uniform | direct | True | 9.0045 | 8.9108 | 58.4 + 77.1 | 80.0 | 5.9 | 11.7 | 53 Gb/s |
| 32 x 4 (4) | uniform | direct | True | 9.0045 | 8.9108 | 41.1 + 64.8 | 58.6 | 8.0 | 16.0 | 72 Gb/s |
| 16 x 4 (4) | uniform | direct | True | 9.0114 | 8.9108 | 54.1 + 78.4 | 95.0 | 4.9 | 9.9 | 45 Gb/s |
| 64 x 16 (16) | uniform | direct | True | 9.0114 | 8.9108 | 97.1 + 119.3 | 111.7 | 4.2 | 8.4 | 38 Gb/s |
| 32 x 4 (4) | uniform | search | True | 9.0045 | 8.9108 | 41.2 + 65.2 | 64.1 | 7.3 | 14.7 | 66 Gb/s |
| 32 x 4 (4) | exact | direct | True | 8.9447 | 8.9108 | 41.6 + 54.8 | 101.2 | 4.6 | 9.3 | 42 Gb/s |

Whole-document decode, per-layer one-step codec (measured; same document, predictor and checks): for each layer a cuBLAS int8 product (the previous layer's 8-bit features times the 8-bit coefficients, exact int32), then one Triton kernel of one-warp programs (a token block x one head) that scales, rotates, codes and makes the next layer's features. Streams stay open across layers. Chunks: the tokens are split into that many parts, each decoded on its own CUDA stream, so one part's product runs while another part decodes. Encode time excludes the encoder's forward pass (as in the fused table).

| Tokens per program | Chunks | Mode | Stage 1 | Variant | Bit-exact | Stream bits/value | Encode ms | Decode ms | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 1 | uniform | search | - | True | 9.0045 | 36.6 | 37.2 | 12.6 | 25.3 | 114 Gb/s |
| 128 | 2 | uniform | search | - | True | 9.0045 | 37.1 | 34.6 | 13.6 | 27.1 | 122 Gb/s |
| 128 | 4 | uniform | search | - | True | 9.0045 | 37.1 | 33.3 | 14.1 | 28.2 | 127 Gb/s |
| 64 | 2 | uniform | search | - | True | 9.0114 | 34.7 | 32.7 | 14.4 | 28.7 | 129 Gb/s |
| 256 | 2 | uniform | search | - | True | 9.0010 | 39.0 | 38.7 | 12.1 | 24.3 | 109 Gb/s |
| 32 | 2 | uniform | search | - | True | 9.0252 | 33.9 | 31.6 | 14.9 | 29.7 | 134 Gb/s |
| 32 | 2 | uniform | search | - | True | 9.0252 | 33.4 | 31.6 | 14.9 | 29.7 | 134 Gb/s |
| 32 | 2 | uniform | direct | - | True | 9.0252 | 33.4 | 32.0 | 14.7 | 29.4 | 133 Gb/s |
| 32 | 2 | uniform | search | CUDA graph | True | 9.0252 | 33.7 | 31.5 | 14.9 | 29.8 | 135 Gb/s |
| 32 | 2 | uniform | search | Triton product 128x128x64 (4 warps) | True | 9.0252 | 33.8 | 32.1 | 14.6 | 29.2 | 132 Gb/s |
| 32 | 2 | uniform | search | Triton product 128x128x64 (4 warps), CUDA graph | True | 9.0252 | 33.8 | 32.1 | 14.6 | 29.3 | 132 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x128x64 (4 warps) | True | 9.0252 | 35.4 | 33.8 | 13.9 | 27.8 | 125 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x128x64 (4 warps), CUDA graph | True | 9.0252 | 35.6 | 33.8 | 13.9 | 27.8 | 125 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x64x64 (4 warps) | True | 9.0252 | 39.2 | 37.5 | 12.5 | 25.1 | 113 Gb/s |
| 32 | 2 | uniform | search | Triton product 64x64x64 (4 warps), CUDA graph | True | 9.0252 | 39.3 | 37.5 | 12.5 | 25.1 | 113 Gb/s |
| 32 | 4 | uniform | search | Triton product 64x128x64 (4 warps), CUDA graph | True | 9.0252 | 35.6 | 33.9 | 13.9 | 27.7 | 125 Gb/s |
| 32 | 4 | uniform | search | CUDA graph | True | 9.0252 | 33.8 | 31.8 | 14.8 | 29.5 | 133 Gb/s |
| 32 | 8 | uniform | search | Triton product 64x128x64 (4 warps), CUDA graph | True | 9.0252 | 35.8 | 33.8 | 13.9 | 27.8 | 126 Gb/s |
| 32 | 8 | uniform | search | CUDA graph | True | 9.0252 | 33.8 | 32.9 | 14.3 | 28.6 | 129 Gb/s |
| 32 | 2 | exact | search | - | True | 8.9654 | 36.6 | 43.7 | 10.8 | 21.5 | 96 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512) | True | 9.2214 | 29.6 | 28.2 | 16.7 | 33.4 | 154 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 64x128x64 (4 warps)) | True | 9.2214 | 28.6 | 26.8 | 17.5 | 35.0 | 161 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.0 | 25.5 | 18.5 | 36.9 | 170 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 64x64x64 (4 warps)) | True | 9.2214 | 29.4 | 28.2 | 16.6 | 33.3 | 153 Gb/s |
| 32 | 1 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 27.9 | 26.9 | 17.5 | 35.0 | 161 Gb/s |
| 32 | 4 | uniform | search | reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.0 | 26.6 | 17.7 | 35.4 | 163 Gb/s |
| 32 | 2 | uniform | search | CUDA graph, reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2214 | 28.0 | 25.5 | 18.5 | 36.9 | 170 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384) | True | 9.2960 | 28.2 | 26.3 | 17.9 | 35.8 | 166 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 64x128x64 (4 warps)) | True | 9.2960 | 26.6 | 25.6 | 18.3 | 36.7 | 170 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.3 | 25.1 | 18.7 | 37.5 | 174 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 64x64x64 (4 warps)) | True | 9.2960 | 27.4 | 25.9 | 18.2 | 36.3 | 169 Gb/s |
| 32 | 1 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.3 | 25.3 | 18.6 | 37.1 | 173 Gb/s |
| 32 | 4 | uniform | search | reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.3 | 25.2 | 18.7 | 37.3 | 174 Gb/s |
| 32 | 2 | uniform | search | CUDA graph, reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.2960 | 26.4 | 25.2 | 18.6 | 37.2 | 173 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256) | True | 9.4049 | 25.0 | 24.5 | 19.2 | 38.4 | 181 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 64x128x64 (4 warps)) | True | 9.4049 | 24.5 | 23.8 | 19.8 | 39.6 | 186 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.3 | 22.8 | 20.6 | 41.2 | 194 Gb/s |
| 32 | 2 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 64x64x64 (4 warps)) | True | 9.4049 | 24.9 | 24.6 | 19.1 | 38.2 | 180 Gb/s |
| 32 | 1 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.4 | 23.6 | 19.9 | 39.9 | 188 Gb/s |
| 32 | 4 | uniform | search | reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.4 | 23.3 | 20.2 | 40.3 | 190 Gb/s |
| 32 | 2 | uniform | search | CUDA graph, reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps)) | True | 9.4049 | 24.4 | 22.8 | 20.6 | 41.3 | 194 Gb/s |

Where the decode time goes (32 tokens per program, 2 chunk(s)): the 27 products alone 15.0 ms, the 28 coding kernels alone (products precomputed) 15.8 ms, both as decoded 31.6 ms; overlap hides -0.8 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), Triton product 64x128x64 (4 warps)): the 27 products alone 18.1 ms, the 28 coding kernels alone (products precomputed) 16.2 ms, both as decoded 33.8 ms; overlap hides 0.4 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 512)): the 27 products alone 11.2 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 28.2 ms; overlap hides -1.3 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 512, requantized in a Triton product epilogue 64x128x64 (4 warps))): the 27 products alone 9.2 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 26.8 ms; overlap hides -2.0 ms.

Where the decode time goes (32 tokens per program, 1 chunk(s), reduced-rank product (rank 512, requantized in a Triton product epilogue 128x128x64 (4 warps))): the 27 products alone 10.2 ms, the 28 coding kernels alone (products precomputed) 16.7 ms, both as decoded 26.9 ms; overlap hides 0.1 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 384)): the 27 products alone 9.6 ms, the 28 coding kernels alone (products precomputed) 15.5 ms, both as decoded 26.3 ms; overlap hides -1.2 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 384, requantized in a Triton product epilogue 64x128x64 (4 warps))): the 27 products alone 7.2 ms, the 28 coding kernels alone (products precomputed) 15.7 ms, both as decoded 25.6 ms; overlap hides -2.8 ms.

Where the decode time goes (32 tokens per program, 1 chunk(s), reduced-rank product (rank 384, requantized in a Triton product epilogue 128x128x64 (4 warps))): the 27 products alone 8.7 ms, the 28 coding kernels alone (products precomputed) 16.6 ms, both as decoded 25.3 ms; overlap hides 0.1 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 256)): the 27 products alone 7.5 ms, the 28 coding kernels alone (products precomputed) 15.5 ms, both as decoded 24.5 ms; overlap hides -1.4 ms.

Where the decode time goes (32 tokens per program, 2 chunk(s), reduced-rank product (rank 256, requantized in a Triton product epilogue 64x128x64 (4 warps))): the 27 products alone 5.6 ms, the 28 coding kernels alone (products precomputed) 15.6 ms, both as decoded 23.8 ms; overlap hides -2.5 ms.

Where the decode time goes (32 tokens per program, 1 chunk(s), reduced-rank product (rank 256, requantized in a Triton product epilogue 128x128x64 (4 warps))): the 27 products alone 6.9 ms, the 28 coding kernels alone (products precomputed) 16.6 ms, both as decoded 23.6 ms; overlap hides 0.0 ms.

Reference: SplitZip's native codec on the same RTX A4500 (official run, branch experiment/llama32_official_gpu, benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517): 12.04 bits per value, encode 137 GB/s, decode 311 GB/s.


Our codec alone (exact coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.87 bits/value, 10.1 G values/s); P, 8 pivot steps (8.04 bits/value, 8.3 G values/s); P, 16 pivot steps (7.97 bits/value, 8.0 G values/s). SplitZip and raw BF16 are references only.

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

Our codec alone (uniform coder; no SplitZip coding anywhere): transfer time of one document's KV (470M values), from measured overlapped decode speeds and an ideal link (decoding overlapped with arrival; no network measured). Our predictors: P, 1 pivot step (8.90 bits/value, 13.4 G values/s); P, 8 pivot steps (8.08 bits/value, 10.4 G values/s); P, 16 pivot steps (8.01 bits/value, 10.0 G values/s). SplitZip and raw BF16 are references only.

| Link | Raw BF16 | SplitZip | Ours, best single predictor | Ours, per-layer choice (layers) | Ours vs SplitZip |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 7516 ms | 5647 ms | 3761 ms (16 pivot steps) | 3761 ms (28 16 pivot steps) | 1.50x |
| 5 Gb/s | 1503 ms | 1129 ms | 752 ms (16 pivot steps) | 752 ms (28 16 pivot steps) | 1.50x |
| 10 Gb/s | 752 ms | 565 ms | 376 ms (16 pivot steps) | 376 ms (28 16 pivot steps) | 1.50x |
| 25 Gb/s | 301 ms | 226 ms | 150 ms (16 pivot steps) | 150 ms (28 16 pivot steps) | 1.50x |
| 50 Gb/s | 150 ms | 113 ms | 75 ms (16 pivot steps) | 75 ms (28 16 pivot steps) | 1.50x |
| 100 Gb/s | 75 ms | 56 ms | 42 ms (1 pivot step) | 40 ms (16 1 pivot step, 10 8 pivot steps, 2 16 pivot steps) | 1.43x |
| 200 Gb/s | 38 ms | 28 ms | 35 ms (1 pivot step) | 35 ms (28 1 pivot step) | 0.80x |
| 400 Gb/s | 19 ms | 14 ms | 35 ms (1 pivot step) | 35 ms (28 1 pivot step) | 0.40x |
| 800 Gb/s | 9 ms | 7 ms | 35 ms (1 pivot step) | 35 ms (28 1 pivot step) | 0.20x |

### Measured transfer, version 1: our codec alone (test_text: 470M values)

Each cell: time to move the document over an ideal link (decoding overlapped with arrival: the longer of the time on the wire and the decode) and the bytes sent. Our row is the measured mode with the shortest time for that link, among: fused 32x8, uniform (9.011 bits/value, 7.5 G values/s); fused 64x8, uniform (9.004 bits/value, 5.9 G values/s); fused 32x4, uniform (9.004 bits/value, 8.0 G values/s); fused 16x4, uniform (9.011 bits/value, 4.9 G values/s); fused 64x16, uniform (9.011 bits/value, 4.2 G values/s); fused 32x4, exact (8.945 bits/value, 4.6 G values/s); per layer, dense product, uniform (9.025 bits/value, 14.9 G values/s); per layer, dense product, exact (8.965 bits/value, 10.8 G values/s); per layer, rank 512 product, uniform (9.221 bits/value, 18.5 G values/s); per layer, rank 384 product, uniform (9.296 bits/value, 18.7 G values/s); per layer, rank 256 product, uniform (9.405 bits/value, 20.6 G values/s). SplitZip: the authors' codec (42c7431) run here: 12.049 bits/value, 309 GB/s decode. ExANS: the authors' codec run here, 10.864 bits/value, 54 GB/s decode.

| Link | Typical setting | SplitZip | ExANS | Ours | Mode | Ours: time vs SplitZip | Ours: bytes vs SplitZip | Ours: time vs ExANS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Gb/s | ordinary cloud, storage | 5660 ms, 675 MiB | 5103 ms, 608 MiB | 4202 ms, 501 MiB | fused 32x4, exact | 1.35x | 26% fewer | 1.21x |
| 5 Gb/s | ordinary cloud | 1132 ms, 675 MiB | 1021 ms, 608 MiB | 840 ms, 501 MiB | fused 32x4, exact | 1.35x | 26% fewer | 1.21x |
| 10 Gb/s | consumer, cloud Ethernet | 566 ms, 675 MiB | 510 ms, 608 MiB | 420 ms, 501 MiB | fused 32x4, exact | 1.35x | 26% fewer | 1.21x |
| 25 Gb/s | cloud Ethernet | 226 ms, 675 MiB | 204 ms, 608 MiB | 168 ms, 501 MiB | fused 32x4, exact | 1.35x | 26% fewer | 1.21x |
| 50 Gb/s | workstation | 113 ms, 675 MiB | 102 ms, 608 MiB | 84 ms, 502 MiB | per layer, dense product, exact | 1.34x | 26% fewer | 1.21x |
| 100 Gb/s | workstation, cross-cluster | 57 ms, 675 MiB | 51 ms, 608 MiB | 42 ms, 505 MiB | per layer, dense product, uniform | 1.33x | 25% fewer | 1.20x |
| 200 Gb/s | datacenter RDMA, per GPU | 28 ms, 675 MiB | 26 ms, 608 MiB | 23 ms, 527 MiB | per layer, rank 256 product, uniform | 1.24x | 22% fewer | 1.12x |
| 400 Gb/s | datacenter RDMA, per GPU | 14 ms, 675 MiB | 18 ms, 608 MiB | 23 ms, 527 MiB | per layer, rank 256 product, uniform | 0.62x | 22% fewer | 0.77x |
| 800 Gb/s | next-generation RDMA | 7 ms, 675 MiB | 18 ms, 608 MiB | 23 ms, 527 MiB | per layer, rank 256 product, uniform | 0.31x | 22% fewer | 0.77x |

Where our codec alone is slower (from 400 Gb/s), it is our decode that sets the time, not the link: at 400 Gb/s it takes 23 ms against SplitZip's 14 ms, but the link is busy for only 11 ms and carries 22% fewer bytes (527 against 675 MiB): a shared or metered link carries more of everything else.

### Measured transfer, version 2: our codec + SplitZip per layer

The layers where ours saves the most are coded by ours, the rest by the authors' SplitZip codec; every mix is a real stream, decoded bit-exact and timed. Measured mixes: dense 0/28 (12.05 bits, 6.5 ms), dense 2/28 (11.84 bits, 8.5 ms), dense 4/28 (11.60 bits, 10.4 ms), dense 6/28 (11.36 bits, 12.3 ms), dense 8/28 (11.12 bits, 14.0 ms), dense 10/28 (10.89 bits, 15.7 ms), dense 12/28 (10.66 bits, 17.2 ms), dense 14/28 (10.44 bits, 19.2 ms), dense 16/28 (10.23 bits, 21.2 ms), dense 20/28 (9.81 bits, 25.3 ms), dense 24/28 (9.40 bits, 28.7 ms), dense 28/28 (9.03 bits, 31.2 ms), rank 256 0/28 (12.05 bits, 6.5 ms), rank 256 2/28 (11.87 bits, 8.0 ms), rank 256 4/28 (11.67 bits, 9.2 ms), rank 256 6/28 (11.47 bits, 10.6 ms), rank 256 8/28 (11.28 bits, 12.0 ms), rank 256 10/28 (11.09 bits, 13.6 ms), rank 256 12/28 (10.89 bits, 14.3 ms), rank 256 14/28 (10.70 bits, 15.7 ms), rank 256 16/28 (10.51 bits, 16.8 ms), rank 256 20/28 (10.13 bits, 19.2 ms), rank 256 24/28 (9.75 bits, 21.0 ms), rank 256 28/28 (9.40 bits, 22.7 ms), rank 384 0/28 (12.05 bits, 6.5 ms), rank 384 2/28 (11.86 bits, 8.0 ms), rank 384 4/28 (11.65 bits, 9.3 ms), rank 384 6/28 (11.45 bits, 10.9 ms), rank 384 8/28 (11.24 bits, 12.1 ms), rank 384 10/28 (11.03 bits, 13.3 ms), rank 384 12/28 (10.83 bits, 14.7 ms), rank 384 14/28 (10.63 bits, 16.1 ms), rank 384 16/28 (10.43 bits, 17.5 ms), rank 384 20/28 (10.04 bits, 20.1 ms), rank 384 24/28 (9.65 bits, 22.3 ms), rank 384 28/28 (9.30 bits, 24.4 ms), rank 512 0/28 (12.05 bits, 6.5 ms), rank 512 2/28 (11.86 bits, 8.1 ms), rank 512 4/28 (11.64 bits, 9.5 ms), rank 512 6/28 (11.42 bits, 11.0 ms), rank 512 8/28 (11.21 bits, 12.5 ms), rank 512 10/28 (11.00 bits, 13.7 ms), rank 512 12/28 (10.79 bits, 15.0 ms), rank 512 14/28 (10.58 bits, 16.6 ms), rank 512 16/28 (10.38 bits, 17.7 ms), rank 512 20/28 (9.98 bits, 20.8 ms), rank 512 24/28 (9.58 bits, 23.2 ms), rank 512 28/28 (9.22 bits, 25.3 ms). With 0 layers of ours the mix is SplitZip itself, so the best mix is never slower than SplitZip.

| Link | SplitZip | Ours alone | Ours + SplitZip mix | Layers ours (mode) | Mix: time vs SplitZip | Mix: bytes vs SplitZip |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 5660 ms, 675 MiB | 4202 ms, 501 MiB | 4202 ms, 501 MiB | all (fused 32x4, exact) | 1.35x | 26% fewer |
| 5 Gb/s | 1132 ms, 675 MiB | 840 ms, 501 MiB | 840 ms, 501 MiB | all (fused 32x4, exact) | 1.35x | 26% fewer |
| 10 Gb/s | 566 ms, 675 MiB | 420 ms, 501 MiB | 420 ms, 501 MiB | all (fused 32x4, exact) | 1.35x | 26% fewer |
| 25 Gb/s | 226 ms, 675 MiB | 168 ms, 501 MiB | 168 ms, 501 MiB | all (fused 32x4, exact) | 1.35x | 26% fewer |
| 50 Gb/s | 113 ms, 675 MiB | 84 ms, 502 MiB | 84 ms, 502 MiB | all (per layer, dense product, exact) | 1.34x | 26% fewer |
| 100 Gb/s | 57 ms, 675 MiB | 42 ms, 505 MiB | 42 ms, 505 MiB | 28 of 28 (dense) | 1.33x | 25% fewer |
| 200 Gb/s | 28 ms, 675 MiB | 23 ms, 527 MiB | 23 ms, 541 MiB | 24 of 28 (rank 384) | 1.25x | 20% fewer |
| 400 Gb/s | 14 ms, 675 MiB | 23 ms, 527 MiB | 13 ms, 628 MiB | 8 of 28 (rank 512) | 1.07x | 7% fewer |
| 800 Gb/s | 7 ms, 675 MiB | 23 ms, 527 MiB | 7 ms, 675 MiB | 0 of 28 (dense) | 1.00x | 0% fewer |

## GPU-resident KV pool (NVIDIA RTX A4500)

A pool keeps finished requests' KV so that a later request with the same prefix skips its prefill. One cached prefix of 8192 tokens (test_text, 28 layers; 896 MiB as raw BF16). Recomputing it (the model's prefill, what a miss costs) takes 992 ms. From GPU memory a method pays its decode; from CPU memory, the host-to-GPU copy of its bytes (23.9 GB/s pinned, measured with the raw KV) and then the decode. Adding a prefix costs its encode. Prefixes that fit: the 12.4 GiB left after the model's weights (no room kept for activations, so an upper bound). Decode and encode times are measured on this GPU: ours and SplitZip in gpu_speed, ExANS in its stage, AQUA-KV and KIVI with the authors' own code (scripts/time_baseline_decoders.py); our lossy variants (sinks, token step) decode at the base variant's speed. Lossy quality = perplexity change (test_text / test_code).

| Method | Lossless / perplexity change | Bits per value | MiB per prefix | Prefixes in free GPU memory | Back from GPU memory | Back from CPU memory | Adding it |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recompute (the model's prefill) | lossless | - | 0 | - | 992 ms | 992 ms | 0 |
| Raw BF16 | lossless | 16.00 | 896 | 14.2 | 0 ms | 39 ms | 0 |
| SplitZip (authors' codec) | lossless | 12.05 | 675 | 18.9 | 3 ms | 33 ms | 7 ms |
| ExANS (authors' codec) | lossless | 10.86 | 608 | 20.9 | 18 ms | 44 ms | 176 ms |
| Ours lossless, fewest bits (fused, exact) | lossless | 8.94 | 501 | 25.4 | 101 ms | 123 ms | 96 ms |
| Ours lossless, fastest decode (per layer, rank 256, uniform) | lossless | 9.40 | 527 | 24.2 | 23 ms | 46 ms | 41 ms |
| Ours, step 0.25 | +0.25% / -0.15% | 5.01 | 281 | 45.4 | not timed | not timed | not timed |
| Ours, step 0.5 | +0.33% / +0.53% | 4.06 | 227 | 56.0 | not timed | not timed | not timed |
| Ours, step 1.0 | +1.26% / +0.28% | 3.18 | 178 | 71.6 | not timed | not timed | not timed |
| Ours, step 2.0 | +3.81% / +0.59% | 2.47 | 138 | 92.0 | not timed | not timed | not timed |
| Ours, step 4.0 | +14.50% / +1.00% | 2.09 | 117 | 108.7 | not timed | not timed | not timed |
| FP8 cast | +0.33% / -0.32% | 8.00 | 448 | 28.4 | not timed | not timed | not timed |
| AQUA-KV, 2-bit | +1.46% / -0.35% | 2.11 | 118 | 108.0 | 78 ms | 83 ms | 951 ms |
| AQUA-KV, 3-bit | +0.06% / -0.23% | 3.07 | 172 | 74.1 | 78 ms | 85 ms | 1,418 ms |
| AQUA-KV, 4-bit | -0.08% / -0.10% | 4.03 | 226 | 56.4 | 78 ms | 88 ms | 2,434 ms |
| KIVI, 2-bit | +20.79% / +3.46% | 3.27 | 183 | 69.5 | 46 ms | 54 ms | 4,328 ms |
| KIVI, 4-bit | +0.28% / +0.23% | 5.23 | 293 | 43.5 | 46 ms | 59 ms | 4,339 ms |
| Ours, GPU codec (one-step, measured stream), step 0.25 | +0.86% / +0.02% | 5.00 | 280 | 45.4 | 28 ms | 40 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 0.5 | +0.93% / -0.21% | 4.02 | 225 | 56.5 | 28 ms | 38 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 1.0 | +6.77% / +1.16% | 3.10 | 173 | 73.4 | 27 ms | 35 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 2.0 | +20.39% / +7.01% | 2.34 | 131 | 97.3 | 27 ms | 33 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 4.0 | +776.12% / +416.80% | 1.90 | 106 | 119.7 | 27 ms | 31 ms | 55 ms |
| Ours, step 0.25, sinks | +0.29% / -0.03% | 5.03 | 282 | 45.2 | not timed | not timed | not timed |
| Ours, step 0.5, sinks | +0.26% / +0.07% | 4.07 | 228 | 55.8 | not timed | not timed | not timed |
| Ours, step 1.0, sinks | +0.22% / -0.06% | 3.19 | 179 | 71.3 | not timed | not timed | not timed |
| Ours, step 2.0, sinks | -0.01% / +0.39% | 2.48 | 139 | 91.6 | not timed | not timed | not timed |
| Ours, step 4.0, sinks | +0.09% / -0.25% | 2.10 | 118 | 108.4 | not timed | not timed | not timed |
| Ours, step 0.25, token step | +0.06% / -0.06% | 4.89 | 274 | 46.5 | not timed | not timed | not timed |
| Ours, step 0.5, token step | +0.26% / -0.07% | 3.90 | 219 | 58.3 | not timed | not timed | not timed |
| Ours, step 1.0, token step | +0.31% / +0.07% | 2.96 | 166 | 76.9 | not timed | not timed | not timed |
| Ours, step 2.0, token step | +4.71% / +2.12% | 2.15 | 120 | 105.9 | not timed | not timed | not timed |
| Ours, step 4.0, token step | +2049.00% / +1270.18% | 1.75 | 98 | 129.8 | not timed | not timed | not timed |
| Ours, step 0.25, sinks + token step | +0.28% / +0.16% | 4.90 | 275 | 46.4 | not timed | not timed | not timed |
| Ours, step 0.5, sinks + token step | +0.18% / -0.04% | 3.92 | 219 | 58.0 | not timed | not timed | not timed |
| Ours, step 1.0, sinks + token step | -0.02% / -0.00% | 2.98 | 167 | 76.4 | not timed | not timed | not timed |
| Ours, step 2.0, sinks + token step | +0.03% / +0.14% | 2.17 | 121 | 105.0 | not timed | not timed | not timed |
| Ours, step 4.0, sinks + token step | +1.77% / -0.72% | 1.77 | 99 | 128.4 | not timed | not timed | not timed |
| Ours, GPU codec (one-step, measured stream), step 0.25, sinks | +0.03% / -0.04% | 5.02 | 281 | 45.3 | 28 ms | 40 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 0.5, sinks | +0.01% / +0.19% | 4.04 | 226 | 56.3 | 28 ms | 38 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 1.0, sinks | +0.12% / -0.11% | 3.11 | 174 | 73.0 | 27 ms | 35 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 2.0, sinks | +0.52% / +0.23% | 2.35 | 132 | 96.6 | 27 ms | 33 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 4.0, sinks | +0.98% / +0.68% | 1.92 | 107 | 118.6 | 27 ms | 31 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 0.25, token step | +0.91% / +0.23% | 4.92 | 276 | 46.2 | 28 ms | 40 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 0.5, token step | +3.96% / -0.52% | 3.93 | 220 | 57.9 | 28 ms | 37 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 1.0, token step | +21.73% / +3.51% | 2.98 | 167 | 76.3 | 27 ms | 35 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 2.0, token step | +845.21% / +247.99% | 2.26 | 126 | 100.7 | 27 ms | 33 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 4.0, token step | +9034.06% / +8759.12% | 2.51 | 140 | 90.7 | 27 ms | 33 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 0.25, sinks + token step | +0.41% / -0.13% | 4.94 | 276 | 46.1 | 28 ms | 40 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 0.5, sinks + token step | -0.06% / +0.05% | 3.95 | 221 | 57.6 | 28 ms | 37 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 1.0, sinks + token step | +0.28% / +0.25% | 3.00 | 168 | 75.9 | 27 ms | 35 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 2.0, sinks + token step | -0.58% / -1.13% | 2.28 | 127 | 99.9 | 27 ms | 33 ms | 55 ms |
| Ours, GPU codec (one-step, measured stream), step 4.0, sinks + token step | +2.53% / +2.11% | 2.52 | 141 | 90.1 | 27 ms | 33 ms | 55 ms |

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

Time to move one document's KV over an ideal link, decoding overlapped with arrival: max(bits / link, decode time) with the decode measured on this GPU (AQUA-KV and KIVI with the authors' code); a * marks a method whose decode was not measured (its time on the wire only, so a lower bound). MiB = bytes sent for the document. Bits and quality of the lossy rows from the lossy stage (first 3072 tokens of each test document; quality = perplexity change of the next 512 tokens, test_text / test_code).

| Method | Bits per value | MiB | Perplexity change | 1 Gb/s (ms) | 10 Gb/s (ms) | 25 Gb/s (ms) | 100 Gb/s (ms) | 200 Gb/s (ms) | 400 Gb/s (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SplitZip (lossless) | 12.05 | 675 | 0 | 5,660 | 566 | 226 | 57 | 28 | 14 |
| Ours, lossless (best measured mode per link) | 8.94-9.40 | 501-527 | 0 | 4,202 | 420 | 168 | 42 | 23 | 23 |
| Ours, step 0.25 | 5.01 | 281 | +0.25% / -0.15% | 2,356* | 236* | 94* | 24* | 12* | 6* |
| Ours, step 0.5 | 4.06 | 227 | +0.33% / +0.53% | 1,906* | 191* | 76* | 19* | 10* | 5* |
| Ours, step 1.0 | 3.18 | 178 | +1.26% / +0.28% | 1,492* | 149* | 60* | 15* | 7* | 4* |
| Ours, step 2.0 | 2.47 | 138 | +3.81% / +0.59% | 1,161* | 116* | 46* | 12* | 6* | 3* |
| Ours, step 4.0 | 2.09 | 117 | +14.50% / +1.00% | 982* | 98* | 39* | 10* | 5* | 2* |
| FP8 cast | 8.00 | 448 | +0.33% / -0.32% | 3,758* | 376* | 150* | 38* | 19* | 9* |
| AQUA-KV, 2-bit | 2.11 | 118 | +1.46% / -0.35% | 989 | 99 | 78 | 78 | 78 | 78 |
| AQUA-KV, 3-bit | 3.07 | 172 | +0.06% / -0.23% | 1,441 | 144 | 78 | 78 | 78 | 78 |
| AQUA-KV, 4-bit | 4.03 | 226 | -0.08% / -0.10% | 1,894 | 189 | 78 | 78 | 78 | 78 |
| KIVI, 2-bit | 3.27 | 183 | +20.79% / +3.46% | 1,537 | 154 | 61 | 46 | 46 | 46 |
| KIVI, 4-bit | 5.23 | 293 | +0.28% / +0.23% | 2,456 | 246 | 98 | 46 | 46 | 46 |
| Ours, GPU codec (one-step, measured stream), step 0.25 | 5.00 | 280 | +0.86% / +0.02% | 2,351 | 235 | 94 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 0.5 | 4.02 | 225 | +0.93% / -0.21% | 1,890 | 189 | 76 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 1.0 | 3.10 | 173 | +6.77% / +1.16% | 1,455 | 146 | 58 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 2.0 | 2.34 | 131 | +20.39% / +7.01% | 1,098 | 110 | 44 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 4.0 | 1.90 | 106 | +776.12% / +416.80% | 892 | 89 | 36 | 27 | 27 | 27 |
| Ours, step 0.25, sinks | 5.03 | 282 | +0.29% / -0.03% | 2,362* | 236* | 94* | 24* | 12* | 6* |
| Ours, step 0.5, sinks | 4.07 | 228 | +0.26% / +0.07% | 1,914* | 191* | 77* | 19* | 10* | 5* |
| Ours, step 1.0, sinks | 3.19 | 179 | +0.22% / -0.06% | 1,498* | 150* | 60* | 15* | 7* | 4* |
| Ours, step 2.0, sinks | 2.48 | 139 | -0.01% / +0.39% | 1,166* | 117* | 47* | 12* | 6* | 3* |
| Ours, step 4.0, sinks | 2.10 | 118 | +0.09% / -0.25% | 986* | 99* | 39* | 10* | 5* | 2* |
| Ours, step 0.25, token step | 4.89 | 274 | +0.06% / -0.06% | 2,296* | 230* | 92* | 23* | 11* | 6* |
| Ours, step 0.5, token step | 3.90 | 219 | +0.26% / -0.07% | 1,834* | 183* | 73* | 18* | 9* | 5* |
| Ours, step 1.0, token step | 2.96 | 166 | +0.31% / +0.07% | 1,390* | 139* | 56* | 14* | 7* | 3* |
| Ours, step 2.0, token step | 2.15 | 120 | +4.71% / +2.12% | 1,009* | 101* | 40* | 10* | 5* | 3* |
| Ours, step 4.0, token step | 1.75 | 98 | +2049.00% / +1270.18% | 823* | 82* | 33* | 8* | 4* | 2* |
| Ours, step 0.25, sinks + token step | 4.90 | 275 | +0.28% / +0.16% | 2,303* | 230* | 92* | 23* | 12* | 6* |
| Ours, step 0.5, sinks + token step | 3.92 | 219 | +0.18% / -0.04% | 1,841* | 184* | 74* | 18* | 9* | 5* |
| Ours, step 1.0, sinks + token step | 2.98 | 167 | -0.02% / -0.00% | 1,398* | 140* | 56* | 14* | 7* | 3* |
| Ours, step 2.0, sinks + token step | 2.17 | 121 | +0.03% / +0.14% | 1,018* | 102* | 41* | 10* | 5* | 3* |
| Ours, step 4.0, sinks + token step | 1.77 | 99 | +1.77% / -0.72% | 832* | 83* | 33* | 8* | 4* | 2* |
| Ours, GPU codec (one-step, measured stream), step 0.25, sinks | 5.02 | 281 | +0.03% / -0.04% | 2,357 | 236 | 94 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 0.5, sinks | 4.04 | 226 | +0.01% / +0.19% | 1,897 | 190 | 76 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 1.0, sinks | 3.11 | 174 | +0.12% / -0.11% | 1,463 | 146 | 59 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 2.0, sinks | 2.35 | 132 | +0.52% / +0.23% | 1,106 | 111 | 44 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 4.0, sinks | 1.92 | 107 | +0.98% / +0.68% | 901 | 90 | 36 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 0.25, token step | 4.92 | 276 | +0.91% / +0.23% | 2,312 | 231 | 92 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 0.5, token step | 3.93 | 220 | +3.96% / -0.52% | 1,847 | 185 | 74 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 1.0, token step | 2.98 | 167 | +21.73% / +3.51% | 1,400 | 140 | 56 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 2.0, token step | 2.26 | 126 | +845.21% / +247.99% | 1,061 | 106 | 42 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 4.0, token step | 2.51 | 140 | +9034.06% / +8759.12% | 1,177 | 118 | 47 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 0.25, sinks + token step | 4.94 | 276 | +0.41% / -0.13% | 2,319 | 232 | 93 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 0.5, sinks + token step | 3.95 | 221 | -0.06% / +0.05% | 1,854 | 185 | 74 | 28 | 28 | 28 |
| Ours, GPU codec (one-step, measured stream), step 1.0, sinks + token step | 3.00 | 168 | +0.28% / +0.25% | 1,408 | 141 | 56 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 2.0, sinks + token step | 2.28 | 127 | -0.58% / -1.13% | 1,069 | 107 | 43 | 27 | 27 | 27 |
| Ours, GPU codec (one-step, measured stream), step 4.0, sinks + token step | 2.52 | 141 | +2.53% / +2.11% | 1,186 | 119 | 47 | 27 | 27 | 27 |

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
| Ours, step 0.25, sinks | 5.19 / 4.87 | 0.48% / 0.47% | +0.29% / -0.03% |
| Ours, step 0.5, sinks | 4.23 / 3.92 | 0.95% / 0.93% | +0.26% / +0.07% |
| Ours, step 1.0, sinks | 3.32 / 3.06 | 1.88% / 1.84% | +0.22% / -0.06% |
| Ours, step 2.0, sinks | 2.58 / 2.39 | 3.75% / 3.67% | -0.01% / +0.39% |
| Ours, step 4.0, sinks | 2.14 / 2.05 | 7.48% / 7.32% | +0.09% / -0.25% |
| Ours, step 0.25, token step | 4.90 / 4.87 | 0.57% / 0.48% | +0.06% / -0.06% |
| Ours, step 0.5, token step | 3.92 / 3.89 | 1.14% / 0.96% | +0.26% / -0.07% |
| Ours, step 1.0, token step | 2.97 / 2.95 | 2.38% / 2.02% | +0.31% / +0.07% |
| Ours, step 2.0, token step | 2.16 / 2.14 | 5.73% / 5.06% | +4.71% / +2.12% |
| Ours, step 4.0, token step | 1.75 / 1.75 | 18.07% / 16.76% | +2049.00% / +1270.18% |
| Ours, step 0.25, sinks + token step | 4.92 / 4.89 | 0.57% / 0.48% | +0.28% / +0.16% |
| Ours, step 0.5, sinks + token step | 3.94 / 3.90 | 1.14% / 0.96% | +0.18% / -0.04% |
| Ours, step 1.0, sinks + token step | 2.99 / 2.96 | 2.37% / 2.02% | -0.02% / -0.00% |
| Ours, step 2.0, sinks + token step | 2.17 / 2.16 | 5.70% / 5.02% | +0.03% / +0.14% |
| Ours, step 4.0, sinks + token step | 1.77 / 1.77 | 16.65% / 15.24% | +1.77% / -0.72% |
| Ours, GPU codec (one-step, measured stream), step 0.25, sinks | 5.13 / 4.90 | 0.82% / 0.80% | +0.03% / -0.04% |
| Ours, GPU codec (one-step, measured stream), step 0.5, sinks | 4.15 / 3.92 | 1.63% / 1.59% | +0.01% / +0.19% |
| Ours, GPU codec (one-step, measured stream), step 1.0, sinks | 3.22 / 3.01 | 3.24% / 3.17% | +0.12% / -0.11% |
| Ours, GPU codec (one-step, measured stream), step 2.0, sinks | 2.43 / 2.27 | 6.48% / 6.34% | +0.52% / +0.23% |
| Ours, GPU codec (one-step, measured stream), step 4.0, sinks | 1.96 / 1.88 | 12.95% / 12.66% | +0.98% / +0.68% |
| Ours, GPU codec (one-step, measured stream), step 0.25, token step | 4.95 / 4.89 | 0.92% / 0.81% | +0.91% / +0.23% |
| Ours, GPU codec (one-step, measured stream), step 0.5, token step | 3.96 / 3.90 | 1.85% / 1.63% | +3.96% / -0.52% |
| Ours, GPU codec (one-step, measured stream), step 1.0, token step | 3.01 / 2.96 | 3.84% / 3.42% | +21.73% / +3.51% |
| Ours, GPU codec (one-step, measured stream), step 2.0, token step | 2.26 / 2.26 | 8.83% / 7.97% | +845.21% / +247.99% |
| Ours, GPU codec (one-step, measured stream), step 4.0, token step | 2.51 / 2.51 | 17.51% / 15.66% | +9034.06% / +8759.12% |
| Ours, GPU codec (one-step, measured stream), step 0.25, sinks + token step | 4.96 / 4.91 | 0.92% / 0.81% | +0.41% / -0.13% |
| Ours, GPU codec (one-step, measured stream), step 0.5, sinks + token step | 3.97 / 3.92 | 1.85% / 1.63% | -0.06% / +0.05% |
| Ours, GPU codec (one-step, measured stream), step 1.0, sinks + token step | 3.02 / 2.97 | 3.84% / 3.42% | +0.28% / +0.25% |
| Ours, GPU codec (one-step, measured stream), step 2.0, sinks + token step | 2.28 / 2.27 | 8.82% / 7.96% | -0.58% / -1.13% |
| Ours, GPU codec (one-step, measured stream), step 4.0, sinks + token step | 2.53 / 2.52 | 17.50% / 15.64% | +2.53% / +2.11% |

Values are test_text / test_code. Reference: the captured exact KV restored into the cache.
AQUA-KV follows the authors' code (goodevening13/aquakv@e744183f2f3a4a90381f20a6975456871fd1ae29): HIGGS residuals on 2-D EDEN grids, predictors fitted on their own closed-loop reconstructions, layer 0 at 4 bits, the first 4 tokens in BF16; calibrated on this study's calibration documents (the authors use 256 x 8192 RedPajama tokens). KIVI follows the authors' code (jy-yuan/KIVI@876b4d2) with the paper's group size 32 and residual length 128 (recent tokens in BF16); its scales and minima are counted at 16 bits.
Check against the AQUA-KV code (e744183, torch 2.5.1+cu124): HIGGS values identical for 99.99% (2-bit), 99.99% (3-bit), 99.94% (4-bit) (relative error ours / theirs 0.3280 / 0.3280, 0.1720 / 0.1720, 0.0880 / 0.0880); regression weights within 0.0039.
Check against the KIVI code (876b4d2): largest difference bfloat16 2-bit: 0 RMS, bfloat16 4-bit: 0 RMS.
The model's own prefill of the prompt alone differs from it by +0.050% / +0.039% (GPU kernels round differently at another sequence length).
