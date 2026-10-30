# Lossless KV compression: the literature, checked against the sources (29 October 2026)

Each entry below was read at its source (paper, PDF, README or blog). Bits per value are 16 / ratio, or
16 x (compressed / original), computed here when the source gives only a ratio. "Bit-exact" means the original BF16
words come back. A method whose outputs are unchanged while the stored tensors change is not bit-exact.

## Bit-exact codecs for BF16 KV

| Method | Source | What it codes | Reported | Bits / value | Models, data | Speed, hardware | Code | In our comparison |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **SplitZip** | arXiv 2605.01708 (v3) | top-16 exponent code book (4 bits) + escapes; sign and mantissa raw | 1.324x (Qwen3-32B KV) | 12.08 | Qwen3-32B KV, WikiText-2; coverage on HumanEval, GSM8K, MMLU, PTB (Table 4); Table 1 also Llama-3-8B, Phi-2, ... | 613 / 2182 GB/s enc / dec on H200; RoCE 4x200G | yes | yes, the authors' codec (12.05 bits on our Llama-3.2-3B KV) |
| **ExANS** | OpenLake blog (no paper) | exponent recentred per block, rANS; independent blocks | 1.515x, 622 GB/s decode (position-aligned, H100) | 10.56 | "a real KV snapshot", 256K; no model or data named | H100 | yes (openlake) | yes, the authors' codec (10.87 bits on ours) |
| **AdaCode** | EMNLP 2026 main, paper 269 | short fixed-length exponent code (3 bits on most models) + explicit outliers (3-8 %); online length adjuster | 71.78 % of the original size (the paper defines it as compressed / original) | 11.48 | Qwen3-4B, Mistral-7B-Instruct-v0.3, Qwen1.5-MoE-A2.7B, Huihui-MoE-1.2B; LongBench, LiveCodeBench, MBPP, AIME; calibrated on C4 | one A800; Triton; memory and throughput per GB, **no transfer test**; only lossless baseline NeuZip | yes: github.com/lichangshiwhu/Adacode | added: the authors' codec (`kv_compare/more_baselines.py`), parameters calibrated on WikiText-2 train and from the coded KV |
| **exactkv** | GitHub project (Apache-2.0) | per 16-token page and channel, a base exponent + 4-bit offsets; raw rows for outliers | 12.6 bits (12.7 with the raw-row reserve); a 3-bit variant 11.7 | 12.7 | Qwen2.5-7B, Qwen3-8B, Llama-3.1-8B | RTX 4090; decoded inside decode attention (1.2x FlashInfer) | yes | added: the authors' page encoder and decoder |
| **IBP** (Invariant Bit Packing) | EuroSys '26 (extended: arXiv 2605.30728) | drops bits invariant across a tensor group, for CPU-to-GPU transfers | BF16 KV 23.43 % smaller; FP16 KV (OPT, the LLM experiment) 4.49 % | 12.25 | OPT 2.7-30B KV (FP16) in InfiniGen; Gemma-7B weights | A100, PCIe Gen 4; throughput relative to raw transfer | yes: github.com/AKKamath/InvariantBitPacking | added (optional build): its PCIe fetch-and-decompress, beside a raw copy |
| **compressed-kv-attention** | GitHub project | 4-bit (sometimes 2-bit) exponent residuals, decoded inside CPU attention | 0.754 of BF16 | 12.06 | Qwen2.5-0.5B, one document | CPU; 10 % slower than raw | yes | no (CPU, one small model) |
| ZipNN, DFloat11, Falcon, ZipServ, nvCOMP | SplitZip v3, Table 2: measured on KV, except ZipNN (taken from ZipNN's own paper) | Huffman / ANS / LZ-style codecs | ZipNN 1.515x, DFloat11 1.423x, Falcon 1.428x, ZipServ 1.236x, nvCOMP LZ4 1.019x, Cascaded 1.000x, Bitcomp 1.056x | 10.56, 11.24, 11.20, 12.94, 15.7, 16.0, 15.2 | Qwen3-32B KV | H200; ZipNN 1.2 / 1.7 GB/s (CPU), DFloat11 encode ~0.004 GB/s | yes (pip / nvCOMP) | added: nvCOMP LZ4, Cascaded, Bitcomp, ANS, Zstd (GPU); for DFloat11 / ZipNN, the computed floor 8 + Huffman(exponent) |
| Lossless compression of NN components | arXiv 2508.19263 | exponent and mantissa entropy-coded apart (extends ZipNN to FP8 / FP4) | "up to 62 %" for BF16 (mostly weights); for KV only exponent ratios (< 0.20) and an unattributed 20-30 % saving | not determinable | no model named for KV | none reported | none | cannot run |
| **Speculative KV coding** | blog note (Fergus Finn), "early research note" | **a predictor + arithmetic coding**: the FP8 copy of the target model gives each value's mean; spread per channel; Gaussian mixture | BF16 cache: 5.92-6.74 bits (2.37-2.70x); its "~4x" is against an **FP8** cache (2.05-2.59 bits per FP8 value) | 5.92-6.74 | Qwen3 0.6B-32B, C4 | none; the predictor (a whole FP8 forward pass on both sides) is not counted; no round-trip check | none | its predictor emulated: stage `fp8pred` (FP8 weights and inputs, our coder) |

## Not bit-exact, although called lossless

- **KVCodec** (arXiv 2602.09725, no venue on arXiv). It quantizes the KV to integers first, with CacheGen's
  quantization, then codes them with H.265 `lossless=1` (NVENC). Only the video stage is lossless; the BF16 values
  do not come back. In that mode its ratio is comparable to CacheGen and ShadowServe; its headline 11.9x is lossy.
  Models: LWM-7B, Yi-34B, Llama3-70B.
- **VeriCache** (arXiv 2605.17613): drafts with a lossy cache and verifies against the full cache, kept off the
  GPU. The outputs are exact; the cache it reads is not.
- **KeepKV** (arXiv 2504.09936): merges KV entries; "single-step lossless" refers to attention output, with
  error bounds over several steps.
- **CLLA, "Lossless KV Cache Compression to 2 %"** (arXiv 2410.15252): a new attention architecture with
  comparable accuracy, not compression of an existing model's cache.

## Lossy, for background

- **kvtc** (ICLR 2026): PCA, dynamic-programming bit allocation, quantization, then DEFLATE (nvCOMP). 16x (about
  20x after DEFLATE) within one score point on Llama 3.1 8B, Mistral NeMo 12B, MN-Minitron 8B.
- **PackKV** (arXiv 2512.24449): lossy by its own title. It quantizes, then repacks and bit-packs the integers
  losslessly; 15.3x K and 18.7x V against KIVI at matched accuracy.
- **GEAR**: quantization with low-rank and sparse error correction.

## Corrections to the list we started from

- KVCodec belongs with the not-bit-exact methods: it quantizes before the lossless video stage. Its venue
  (SIGCOMM 2026 in that list) is not on its arXiv page.
- Speculative KV coding's "~4x" is over an FP8 cache. Against BF16 it is 2.4-2.7x (5.9-6.7 bits). Its predictor is
  the whole FP8 model, run on both sides and not counted.
- DFloat11, ZipServ, Falcon and nvCOMP are not only weight codecs here: SplitZip v3 measures them on KV (Table 2),
  so they are KV baselines in its own evaluation. Its ZipNN row is quoted from ZipNN's paper.
- 2508.19263 gives no total ratio for KV and names no model for it. "BF16 to 62 %" is a general, mostly-weights
  claim.
- AdaCode's 71.78 % is the remaining size (about 11.5 bits per value), measured for memory capacity on one GPU,
  not for transfer.
- PackKV is lossy (a lossy step, then lossless packing).

## Where we stand

| Method | Bits per value | Basis |
| --- | --- | --- |
| nvCOMP (LZ4, Cascaded, Bitcomp) | 15.2-16.0 | SplitZip Table 2, Qwen3-32B |
| ZipServ | 12.9 | SplitZip Table 2 |
| exactkv | 12.7 | its README |
| IBP | 12.3 | its paper, BF16 KV |
| SplitZip | 12.05 | the authors' codec on our KV (Llama-3.2-3B, WikiText-2, 8K) |
| AdaCode | 11.5 | its paper, average over its models |
| DFloat11 / Falcon | 11.2 | SplitZip Table 2 |
| ExANS | 10.87 | the authors' codec on our KV |
| ZipNN | 10.56 | SplitZip Table 2, quoted from ZipNN's paper (not measured on KV) |
| **ours, one step** | **9.03** | a real stream on our KV, decoded bit-exact on the GPU |
| **ours, head by head** | **8.32-8.41** | a real stream on our KV, decoded bit-exact on the GPU |
| ours, strong predictor (exact cost) | 7.83 | exact cost; no GPU decoder yet |
| speculative KV coding | 5.9-6.7 | its note: Qwen3, C4; a whole FP8 model as the predictor, not counted |

Every published exponent codec stays between 10.5 and 13 bits, because sign and mantissa (8 bits) pass through
raw and only the exponent is coded. Ours predicts the whole value, so the mantissa is coded too. The only other
predictive scheme is the speculative KV coding note. It reaches 5.9-6.7 bits by running the whole model in FP8 on
both sides. That means a full FP8 prefill of the prompt at the receiver, which prefill/decode disaggregation
exists to avoid, and it is not timed, verified or released. Our predictor costs one int8 product per layer, and our decoder
runs on the GPU and is checked bit-exact. To be fair to it, stage `fp8pred` reports its predictor's exact cost on our
KV: the model fake-quantized to FP8 prefills the prompt, and our coder codes the residual. That needs no code from
the note.

## Sources

- [SplitZip, arXiv 2605.01708 v3](https://arxiv.org/html/2605.01708v3)
- [ExANS, OpenLake](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache)
- [AdaCode, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.269/), [PDF](https://aclanthology.org/2026.emnlp-main.269.pdf), [code](https://github.com/lichangshiwhu/Adacode)
- [exactkv](https://github.com/zyfyz666/exactkv)
- [IBP, arXiv 2605.30728 (extended EuroSys '26)](https://arxiv.org/pdf/2605.30728), [code](https://github.com/AKKamath/InvariantBitPacking)
- [compressed-kv-attention](https://github.com/cvs0/compressed-kv-attention)
- [Lossless Compression of Neural Network Components, arXiv 2508.19263](https://arxiv.org/abs/2508.19263)
- [Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/)
- [KVCodec, arXiv 2602.09725](https://arxiv.org/abs/2602.09725)
- [VeriCache, arXiv 2605.17613](https://arxiv.org/abs/2605.17613)
- [KeepKV, arXiv 2504.09936](https://arxiv.org/abs/2504.09936)
- [CLLA, arXiv 2410.15252](https://arxiv.org/abs/2410.15252)
- [kvtc, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/3fb6f10bd2784f6cfb6a6ed6280df40c-Paper-Conference.pdf)
- [PackKV, arXiv 2512.24449](https://arxiv.org/html/2512.24449v2)
