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
