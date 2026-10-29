## Scoreboard: our best against the baselines

Run /home/nelson/kv_reconstruction/runs/llama32_3b_compare, commit f2f7e7b, NVIDIA RTX A4500.

**Lossless, bits per value** (BF16 = 16; the exact cost of a predictor is what its coder would need; a stream is a real encoding decoded bit-exact on the GPU):

| Entry | Mode | Bits per value | Decode (one document) |
| --- | --- | --- | --- |
| SplitZip-style (exact cost) |  | 12.021 | - |
| ExANS-style (exact cost) |  | 10.722 | - |
| fewest bits, any predictor (exact cost) | L0+P2+P+ATT, 16 pivot steps | 7.754 | - |
| fewest bits without the attention input (exact cost) | L0+P2+P, 16 pivot steps | 7.870 | - |
| fewest bits, real GPU stream (bit-exact) | rANS, fused one-kernel codec | 8.945 | 101.2 ms |
| fewest bits, real GPU stream, decode within 2x our fastest | rANS, per layer, dense | 9.004 | 33.3 ms |
| fastest decode of ours alone (bit-exact) | SplitZip format, rank 256, row | 10.199 | 17.2 ms |
| SplitZip (authors' codec) |  | 12.049 | 3.0 ms |
| ExANS (authors' codec) |  | 10.864 | 17.5 ms |

**Transfer of one document (ideal link, decode overlapped with arrival)**: SplitZip, ExANS, our fastest mode, and our fewest bytes within +25% of the fastest time:

| Link | SplitZip | ExANS | Ours, fastest | vs SplitZip | Ours, fewest bytes within +25% |
| --- | --- | --- | --- | --- | --- |
| 1 Gb/s | 5660.0 ms, 675 MiB | 5103.3 ms, 608 MiB | 4201.9 ms, 501 MiB (rANS, fused one-kernel codec) | 1.35x | 4201.9 ms, 501 MiB (rANS, fused one-kernel codec) |
| 10 Gb/s | 566.0 ms, 675 MiB | 510.3 ms, 608 MiB | 420.2 ms, 501 MiB (rANS, fused one-kernel codec) | 1.35x | 420.2 ms, 501 MiB (rANS, fused one-kernel codec) |
| 50 Gb/s | 113.2 ms, 675 MiB | 102.1 ms, 608 MiB | 84.2 ms, 502 MiB (rANS, per layer, dense) | 1.34x | 101.2 ms, 501 MiB (rANS, fused one-kernel codec) |
| 100 Gb/s | 56.6 ms, 675 MiB | 51.0 ms, 608 MiB | 42.3 ms, 504 MiB (rANS, per layer, dense) | 1.34x | 43.7 ms, 502 MiB (rANS, per layer, dense) |
| 200 Gb/s | 28.3 ms, 675 MiB | 25.5 ms, 608 MiB | 22.7 ms, 541 MiB (rANS + SplitZip, rank 384, 24 layers ours) | 1.25x | 28.2 ms, 516 MiB (rANS, per layer, rank 512) |
| 400 Gb/s | 14.1 ms, 675 MiB | 17.5 ms, 608 MiB | 13.0 ms, 621 MiB (SplitZip format + SplitZip, rank 256, 14 layers ours) | 1.09x | 16.2 ms, 585 MiB (SplitZip format + SplitZip, rank 256, 24 layers ours) |
| 800 Gb/s | 7.1 ms, 675 MiB | 17.5 ms, 608 MiB | 7.1 ms, 675 MiB (SplitZip itself (no layer of ours helps)) | 1.00x | 8.8 ms, 658 MiB (SplitZip format + SplitZip, rank 512, 4 layers ours) |

**Lossy, at the baselines' quality** (lossy_eval.json (WikiText-2 windows), 8 windows; ours = fewest bits with a pooled perplexity change at most the baseline's plus one standard error):

| Baseline | Its bits, change | Ours, CPU coder (strong predictor) | Ours, GPU codec | Decode: baseline / ours GPU |
| --- | --- | --- | --- | --- |
| AQUA-KV, 2-bit | 2.11 bits, +0.92% ± 0.21 | 1.79 bits, +0.80% (step 4.0, sinks + token step) | 2.18 bits, +1.11% (step 4.0, sinks) | 78 ms / 27 ms |
| AQUA-KV, 3-bit | 3.07 bits, +0.04% ± 0.07 | 2.21 bits, +0.04% (step 2.0, sinks + token step) | 2.78 bits, +0.04% (step 2.0, sinks) | 78 ms / 27 ms |
| AQUA-KV, 4-bit | 4.03 bits, -0.04% ± 0.06 | 3.02 bits, -0.07% (step 2.0, sinks) | 4.07 bits, +0.01% (step 0.5, sinks + token step) | 78 ms / 27 ms |
| KIVI, 4-bit | 5.23 bits, +0.08% ± 0.09 | 2.21 bits, +0.04% (step 2.0, sinks + token step) | 2.78 bits, +0.04% (step 2.0, sinks) | 46 ms / 27 ms |

**Lossy in AQUA-KV's own protocol** (WikiText-2 perplexity with a streaming compressed cache, 35 windows of 8192 tokens; perplexity with the authors' BF16 loss, exact cache 6.986, paper 6.98; the change uses a float32 loss; ours = fewest bits per compressed value with a change at most the baseline's plus one standard error):

| Baseline | Its bits, perplexity, change | Ours, GPU codec |
| --- | --- | --- |
| higgs2 | 2.016 bits, 113 MiB per window, 7.554, +8.08% ± 0.27 | 2.185 bits, 122 MiB per window, 7.427, +6.46% (ours4h) |
| kivi2g128 | 2.250 bits, 126 MiB per window, 7.784, +11.41% ± 0.72 | 2.172 bits, 122 MiB per window, 7.771, +11.25% (ours4) |
| kivi2g32 | 3.000 bits, 168 MiB per window, 7.274, +4.12% ± 0.23 | 2.296 bits, 129 MiB per window, 7.160, +2.53% (ours2t) |
| kivi4g32 | 5.000 bits, 280 MiB per window, 6.988, +0.08% ± 0.01 | 3.606 bits, 202 MiB per window, 6.983, -0.03% (ours1) |
| aqua2 | 2.087 bits, 117 MiB per window, 7.093, +1.51% ± 0.13 | 2.757 bits, 154 MiB per window, 7.022, +0.54% (ours2) |
| aqua3 | 3.051 bits, 171 MiB per window, 6.991, +0.08% ± 0.05 | 3.606 bits, 202 MiB per window, 6.983, -0.03% (ours1) |
| aqua4 | 4.016 bits, 225 MiB per window, 6.983, -0.09% ± 0.03 | none at this quality |
| AQUA-KV, the authors' own scripts: no quantization | perplexity 6.986 | - |

**Pool: one cached 8K prefix** (recomputing it: 992 ms):

| Method | Memory | Prefixes that fit | Bring back |
| --- | --- | --- | --- |
| raw BF16 | 896 MiB | 14 | a copy |
| SplitZip | 675 MiB | 18 | 3 ms |
| ExANS | 608 MiB | 20 | 18 ms |
| ours lossless (rANS, fused one-kernel codec) | 501 MiB | 25 | 101 ms |
| ours lossless (rANS, per layer, dense) | 504 MiB | 25 | 33 ms |
| ours lossless (SplitZip format, rank 256, row) | 571 MiB | 22 | 17 ms |
| AQUA-KV, 2-bit | 118 MiB | 108 | 78 ms |
| ours lossy, GPU codec, at AQUA-KV, 2-bit quality | 122 MiB | 104 | 27 ms |
| AQUA-KV, 3-bit | 172 MiB | 74 | 78 ms |
| ours lossy, GPU codec, at AQUA-KV, 3-bit quality | 156 MiB | 81 | 27 ms |
| AQUA-KV, 4-bit | 226 MiB | 56 | 78 ms |
| ours lossy, GPU codec, at AQUA-KV, 4-bit quality | 228 MiB | 55 | 27 ms |

