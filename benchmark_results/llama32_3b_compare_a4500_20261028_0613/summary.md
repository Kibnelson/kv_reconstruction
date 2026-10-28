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

| Predictor | Format | ms per layer | G values/s | GB/s |
| --- | --- | --- | --- | --- |
| P, 1 pivot step | fp32 | 4.05 | 4.1 | 8.3 |
| P, 1 pivot step | bf16 | 0.87 | 19.3 | 38.7 |
| P, 1 pivot step | int8x16 | 3.42 | 4.9 | 9.8 |
| P, 1 pivot step | int8x8 | 0.53 | 31.5 | 63.1 |
| P, 16 pivot steps, rank 128 | fp32 | 3.39 | 4.9 | 9.9 |
| P, 16 pivot steps, rank 128 | bf16 | 1.10 | 15.3 | 30.5 |
| P, 16 pivot steps, rank 128 | int8x16 | 6.34 | 2.6 | 5.3 |
| P, 16 pivot steps, rank 128 | int8x8 | 0.70 | 24.0 | 47.9 |
| P, 16 pivot steps | fp32 | 7.72 | 2.2 | 4.3 |
| P, 16 pivot steps | bf16 | 1.83 | 9.2 | 18.3 |
| P, 16 pivot steps | int8x16 | 6.82 | 2.5 | 4.9 |
| P, 16 pivot steps | int8x8 | 1.02 | 16.5 | 32.9 |
| L0+P2+P, 16 pivot steps | fp32 | 16.84 | 1.0 | 2.0 |
| L0+P2+P, 16 pivot steps | bf16 | 3.64 | 4.6 | 9.2 |
| L0+P2+P, 16 pivot steps | int8x16 | 8.42 | 2.0 | 4.0 |
| L0+P2+P, 16 pivot steps | int8x8 | 1.83 | 9.2 | 18.4 |
| L0+P2+P+ATT, 16 pivot steps | fp32 | 62.23 | 0.3 | 0.5 |
| L0+P2+P+ATT, 16 pivot steps | bf16 | 33.69 | 0.5 | 1.0 |
| L0+P2+P+ATT, 16 pivot steps | int8x16 | 38.69 | 0.4 | 0.9 |
| L0+P2+P+ATT, 16 pivot steps | int8x8 | 31.87 | 0.5 | 1.1 |

Value-slice coder (Triton) on 67,108,864 real values (lite predictions, layers [4, 12, 20, 27]); the round trip must be bit-exact.

| Values per stream x lanes per program (warps) | Bit-exact | Stream bits/value | Exact model bits/value | Encode GB/s | Decode GB/s |
| --- | --- | --- | --- | --- | --- |
| 1024 x 32 (1) | True | 8.0869 | 8.0346 | 27.1 | 24.9 |
| 1024 x 64 (1) | True | 8.0865 | 8.0346 | 26.3 | 26.0 |
| 1024 x 128 (1) | True | 8.0862 | 8.0346 | 28.5 | 24.9 |
| 1024 x 64 (2) | True | 8.0865 | 8.0346 | 29.1 | 23.0 |
| 2048 x 64 (1) | True | 8.0745 | 8.0346 | 24.3 | 24.0 |

Decode estimate per predictor (fastest exact integer format, else BF16, then the fastest coder; run one after the other, not overlapped). Link it keeps up with = decoded values/s x bits per value.

| Predictor | Bits per value | Decode G values/s | Decode GB/s | Keeps up with a link of |
| --- | --- | --- | --- | --- |
| P, 1 pivot step (int8x8) | 8.908 | 9.2 | 18.4 | 82 Gb/s |
| P, 16 pivot steps, rank 128 (int8x8) | 8.623 | 8.4 | 16.8 | 73 Gb/s |
| P, 16 pivot steps (int8x8) | 8.038 | 7.3 | 14.5 | 58 Gb/s |
| L0+P2+P, 16 pivot steps (int8x8) | 7.870 | 5.4 | 10.8 | 42 Gb/s |
| L0+P2+P+ATT, 16 pivot steps (int8x8) | 7.754 | 0.5 | 1.0 | 4 Gb/s |

Overlap (P, 16 pivot steps (int8x8) on one CUDA stream, the coder decoding on another, as a pipelined decoder would on different token blocks): predictor 4.16 ms + decode 5.41 ms one after the other, 9.08 ms together, so 7.4 G values/s (14.8 GB/s), keeping up with a 59 Gb/s link.

Reference: SplitZip's native codec on the same RTX A4500 (official run, branch experiment/llama32_official_gpu, benchmark_results/llama32_official_a4500_20261022_125030_report_20261022_134517): 12.04 bits per value, encode 137 GB/s, decode 311 GB/s.

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
