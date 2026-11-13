# T ideas: measured

Run folder `/home/nelson/kv_a5000/runs/llama32_3b`, calibration wikitext, commit `98638b13ecd175b017766ae8c40a4cf869524a7c`, NVIDIA RTX A4500. Bits per value are exact costs (-log2 P of every BF16 word); side bits are added where a variant sends something. Decision rules: docs/t_ideas.md.

## 1. Per-request adaptation (T's int8 predictor)

| Variant | WikiText-2 | HumanEval | GSM8K | MMLU | PTB | side bits |
| --- | --- | --- | --- | --- | --- | --- |
| T as it is (calibration constants) | 8.839 | 9.418 | 9.154 | 9.176 | 9.114 | 0.0000 |
| forward: the request's own spread per channel and layer (6 bits each, sent) | 8.834 | 9.385 | 9.138 | 9.163 | 9.100 | 0.0007 |
| forward: the request's own bias (8 bits) and spread (6 bits) per channel and layer, sent | 8.823 | 9.301 | 9.077 | 9.132 | 9.035 | 0.0017 |
| forward: one spread scale per head and layer (6 bits each, sent) | 8.839 | 9.401 | 9.149 | 9.171 | 9.112 | 0.0000 |
| backward: bias and spread per channel measured on every 8th token, decoded first (nothing sent) | 8.823 | 9.314 | 9.085 | 9.137 | 9.044 | 0.0000 |
| oracle, cross-fitted: one-step predictor fitted on the other half of the window (not sendable) | 8.922 | 8.767 | 8.729 | 9.017 | 8.884 | 0.0000 |
| oracle, in-sample: fitted on the whole window (optimistic bound) | 8.203 | 7.996 | 7.975 | 8.168 | 8.171 | 0.0000 |
| float one-step (T before 8-bit rounding) | 8.825 | 9.412 | 9.145 | 9.167 | 9.105 | 0 |

| | WikiText-2 | HumanEval | GSM8K | MMLU | PTB |
| --- | --- | --- | --- | --- | --- |
| T minus cross-fitted oracle (room) | -0.082 | 0.651 | 0.425 | 0.159 | 0.230 |
| best sendable saving | 0.017 | 0.117 | 0.077 | 0.044 | 0.079 |
| best sendable variant | fwd-bias+spread/ch | fwd-bias+spread/ch | fwd-bias+spread/ch | fwd-bias+spread/ch | fwd-bias+spread/ch |

## 2. Layer 0 recomputed on the decode side

| Window | words recomputed bit for bit | bits per layer-0 value | T's layer 0 | saving per value (all layers) | recompute time |
| --- | --- | --- | --- | --- | --- |
| ds_wikitext2_r0 | 100.000% | 0.0000 | 9.967 | 0.3559 | 60.6 ms |
| ds_humaneval_r0 | 100.000% | 0.0000 | 10.148 | 0.3624 | 54.1 ms |
| ds_gsm8k_r0 | 100.000% | 0.0000 | 10.050 | 0.3589 | 53.9 ms |
| ds_mmlu_r0 | 100.000% | 0.0000 | 10.101 | 0.3607 | 53.8 ms |
| ds_ptb_r0 | 100.000% | 0.0000 | 9.919 | 0.3543 | 53.7 ms |

## 2c. Layer 0: the lean recompute under other prefill shapes

The receiver computes layer 0 with the embedding, the first layer's input norm, the K and V projections and RoPE only. Worst window of 5 (one per dataset); the cost codes each captured word against the recomputed one (difference in BF16 steps, table from the scales document under the same shape).

| Prefill shape | keys bit for bit | values bit for bit | largest difference | bits per layer-0 value | saving per value (all layers) |
| --- | --- | --- | --- | --- | --- |
| same: one pass of the request, as the capture ran it | 100.000% | 100.000% | 0 steps | 0.0000 | 0.3543 |
| chunks of 2048: the request in prefill chunks of 2,048 tokens | 82.836% | 66.771% | 30976 steps | 1.5999 | 0.3012 |
| chunks of 512: the request in prefill chunks of 512 tokens | 76.522% | 53.870% | 30976 steps | 2.1359 | 0.2826 |
| behind 1 token: one token of another request ahead of it in the same pass (vLLM flattens a batch) | 100.000% | 100.000% | 0 steps | 0.0000 | 0.3543 |
| behind 2048 tokens: 2,048 tokens of another request ahead of it in the same pass | 100.000% | 100.000% | 0 steps | 0.0000 | 0.3543 |

Time per window: lean recompute 9.67-9.76 ms, the whole first layer (attention and MLP too) 52.81-54.76 ms; T's dense decode of the cache takes 31.2 ms.

## 2d. Layer 0 when the receiver knows the sender's prefill steps

The sender computes layer 0 in vLLM-like steps (its own tokens with another request's tokens in the same matrix product, at most 2,048 rows); the receiver recomputes it as said. Worst window of 5. Bits: the better of the step coder and the value coder (a flag, then a logistic around the recomputed value), tables from the scales document under the same pair.

| Sender's steps | receiver | keys bit for bit | values bit for bit | step coder | value coder | saving per value (all layers) |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 own | one pass | 82.836% | 66.771% | 1.5999 | 1.7222 | 0.3012 |
| 2048 own | same chunks | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 2048 own | padded to the step | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 2048 own | padded, own rows first | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 1024 own after 1024 other | one pass | 82.836% | 66.771% | 1.5999 | 1.7222 | 0.3012 |
| 1024 own after 1024 other | same chunks | 72.041% | 52.817% | 2.2473 | 2.2923 | 0.2789 |
| 1024 own after 1024 other | padded to the step | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 1024 own after 1024 other | padded, own rows first | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 1536 own before 512 other | one pass | 82.409% | 66.206% | 1.6270 | 1.7476 | 0.3003 |
| 1536 own before 512 other | same chunks | 98.139% | 96.969% | 0.2486 | 0.2527 | 0.3461 |
| 1536 own before 512 other | padded to the step | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 1536 own before 512 other | padded, own rows first | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 512 own after 1536 other | one pass | 82.836% | 66.771% | 1.5999 | 1.7222 | 0.3012 |
| 512 own after 1536 other | same chunks | 71.977% | 49.537% | 2.3605 | 2.3615 | 0.2748 |
| 512 own after 1536 other | padded to the step | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |
| 512 own after 1536 other | padded, own rows first | 100.000% | 100.000% | 0.0000 | 0.0000 | 0.3543 |

## 4. Anchors (float one-step predictor)

| Variant | ds_wikitext2_r0 | ds_humaneval_r0 | ds_gsm8k_r0 | ds_mmlu_r0 | ds_ptb_r0 | saving vs one-step | waves | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one-step (float) | 8.849 | 9.427 | 9.148 | 9.187 | 9.105 | 0.000 | 1 | layer l-1 only: T's predictor before 8-bit rounding |
| anchor: most recent occurrence | 8.734 | 9.327 | 9.025 | 9.090 | 9.000 | 0.108 | 382/383/433/528/643 | + layer l at the token's most recent earlier occurrence (waves = longest repeat chain) |
| anchor: first occurrence | 8.857 | 9.459 | 9.162 | 9.217 | 9.119 | -0.019 | 2 | + layer l at the first occurrence; 2 waves |
| anchor: nearest key frame, K=8 | 8.891 | 9.413 | 9.148 | 9.200 | 9.122 | -0.012 | 2 | key frames t % 8 == 0 first; others + layer l of the key frame nearest in layer l-1; 2 waves |
| anchor: nearest key frame, K=4 | 8.857 | 9.372 | 9.111 | 9.163 | 9.088 | 0.025 | 2 | as K=8 with key frames t % 4 == 0 |

## 5. Token classes (float one-step predictor)

| Variant | ds_wikitext2_r0 | ds_humaneval_r0 | ds_gsm8k_r0 | ds_mmlu_r0 | ds_ptb_r0 | saving vs one-step | waves | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classes: sink tokens | 8.848 | 9.430 | 9.148 | 9.188 | 9.105 | -0.000 | 1 | tokens 0-3 their own predictor (decoder knows) |
| classes: repeated tokens | 8.857 | 9.456 | 9.169 | 9.207 | 9.128 | -0.020 | 1 | first occurrences and repeats separate predictors, no anchor input (decoder knows) |
| classes: k-means of layer l-1, k=4 | 8.835 | 9.612 | 9.256 | 9.283 | 9.171 | -0.088 | 1 | decoder assigns each token by its decoded layer l-1 (no side bits) |
| classes: encoder picks 1 of 2 | 8.828 | 9.461 | 9.166 | 9.208 | 9.116 | -0.012 | 1 | 2 predictors (k-means start, one refit); 1 bit per token and layer sent |
| classes: encoder picks 1 of 4 | 8.824 | 9.532 | 9.213 | 9.245 | 9.136 | -0.047 | 1 | 4 predictors; 2 bits per token and layer sent |

## 3. Decoder profile and per-layer modes (NVIDIA RTX A4500, ds_wikitext2_r0)

| Mode | bits/value | decode | products alone | coding alone | coding floor | coding / floor |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 8.9795 | 31.2 ms | 14.9 ms | 15.9 ms | 6.7 ms | 2.37 |
| kv | 8.6432 | 42.4 ms | 15.6 ms | 28.9 ms | 6.7 ms | 4.33 |
| 4g | 8.4213 | 52.8 ms | 15.3 ms | 40.0 ms | 6.7 ms | 6.01 |
| hbh | 8.2941 | 108.1 ms | 15.6 ms | 103.8 ms | 6.6 ms | 15.62 |
| rank256 | 9.3456 | 23.1 ms | 5.7 ms | 15.8 ms | 6.8 ms | 2.35 |

Check: T's exact-model bits on ds_wikitext2_r0 from the kernel's own features and priors 8.8627, from the torch reference of the adapt stage 8.8627 (the cut to a multiple of 256 tokens aside).

**Per-layer mode allocation** (offline, from the per-layer bits and times above; pipelined: max(link, decode) for the whole cache)

| Link | best single mode | per-layer mix | gain | modes used |
| --- | --- | --- | --- | --- |
| 1 Gb/s | hbh 3896.00 ms | 3887.82 ms | 0.2% | 4g x1, hbh x27 |
| 3 Gb/s | hbh 1298.67 ms | 1295.94 ms | 0.2% | 4g x1, hbh x27 |
| 10 Gb/s | hbh 389.60 ms | 388.78 ms | 0.2% | 4g x1, hbh x27 |
| 25 Gb/s | hbh 155.84 ms | 155.51 ms | 0.2% | 4g x1, hbh x27 |
| 100 Gb/s | dense 42.18 ms | 40.74 ms | 3.4% | 4g x5, kv x11, dense x12 |
| 400 Gb/s | rank256 23.64 ms | 23.64 ms | -0.0% | rank256 x28 |

## 6. Calibration breadth: T's constants from other text

Every row is T (its int8 predictor) with layer 0 recomputed by the receiver (layer 0 at 0.0000 bits per value), in bits per value, mean of the windows of each dataset. Calibration texts are disjoint from the test windows (scripts/fetch_calib_texts.py). A pick sends the chosen set's number (log2 of the number of sets per request, counted).

**T with layer 0 recomputed**

| Constants | WikiText-2 | HumanEval | GSM8K | MMLU | PTB | mean |
| --- | --- | --- | --- | --- | --- | --- |
| wikitext: T's constants: WikiText-2 train | 8.483 | 9.055 | 8.795 | 8.815 | 8.760 | 8.782 |
| mixed: 6 documents: 2 WikiText-2 train, 1 of each domain | 8.620 | 8.585 | 8.284 | 8.427 | 8.472 | 8.478 |
| mixed-all: 15 documents: 3 of every source | 8.590 | 8.536 | 8.232 | 8.374 | 8.429 | 8.432 |
| code: MBPP, all splits (Python problems and solutions; HumanEval's kind of text) | 9.259 | 8.531 | 8.966 | 9.103 | 9.264 | 9.024 |
| math: GSM8K train | 9.029 | 8.993 | 8.068 | 8.913 | 9.025 | 8.805 |
| exam: MMLU dev and validation (MMLU's prompt format) | 8.848 | 8.888 | 8.657 | 8.293 | 8.856 | 8.708 |
| news: Penn Treebank train | 8.900 | 9.217 | 8.946 | 8.943 | 8.329 | 8.867 |
| pick per request: WikiText or a domain set, chosen on layers 1-4 | 8.483 | 8.531 | 8.068 | 8.293 | 8.329 | 8.341 |
| (sets chosen) | wikitext/wikitext | code/code | math/math | exam/exam | news/news | |
| pick per request: WikiText or a domain set, chosen on the whole cache | 8.483 | 8.531 | 8.068 | 8.293 | 8.329 | 8.341 |
| pick per request: any set, chosen on layers 1-4 | 8.483 | 8.526 | 8.068 | 8.293 | 8.329 | 8.340 |
| room: one-step predictor fitted on the other half of the window (not sendable) | 8.565 | 8.407 | 8.371 | 8.657 | 8.530 | 8.506 |
| today: T as it is, layer 0 coded | 8.839 | 9.418 | 9.154 | 9.176 | 9.114 | 9.140 |

**... and the request's own bias and spread (sent, counted)**

| Constants | WikiText-2 | HumanEval | GSM8K | MMLU | PTB | mean |
| --- | --- | --- | --- | --- | --- | --- |
| wikitext: T's constants: WikiText-2 train | 8.466 | 8.941 | 8.719 | 8.773 | 8.682 | 8.716 |
| mixed: 6 documents: 2 WikiText-2 train, 1 of each domain | 8.597 | 8.563 | 8.280 | 8.425 | 8.464 | 8.466 |
| mixed-all: 15 documents: 3 of every source | 8.571 | 8.515 | 8.228 | 8.373 | 8.423 | 8.422 |
| code: MBPP, all splits (Python problems and solutions; HumanEval's kind of text) | 9.109 | 8.505 | 8.859 | 9.003 | 9.094 | 8.914 |
| math: GSM8K train | 8.933 | 8.901 | 8.067 | 8.860 | 8.907 | 8.734 |
| exam: MMLU dev and validation (MMLU's prompt format) | 8.791 | 8.826 | 8.621 | 8.293 | 8.770 | 8.660 |
| news: Penn Treebank train | 8.824 | 9.064 | 8.842 | 8.875 | 8.325 | 8.786 |
| pick per request: WikiText or a domain set, chosen on layers 1-4 | 8.466 | 8.505 | 8.067 | 8.293 | 8.325 | 8.331 |
| (sets chosen) | wikitext/wikitext | code/code | math/math | exam/exam | news/news | |
| pick per request: WikiText or a domain set, chosen on the whole cache | 8.466 | 8.505 | 8.067 | 8.293 | 8.325 | 8.331 |
| pick per request: any set, chosen on layers 1-4 | 8.466 | 8.505 | 8.067 | 8.293 | 8.325 | 8.331 |

## 6b. Calibration with the test's own source left out

Each dataset is coded with constants that never saw its kind of text: mixed-all refitted without its source (WikiText-2 for WikiText-2, MBPP for HumanEval, GSM8K train for GSM8K, MMLU dev for MMLU, PTB train for PTB; its scales document leaves the source out too), and the per-request pick without the source's own set. Layer 0 recomputed; bits per value.

| Constants | WikiText-2 | HumanEval | GSM8K | MMLU | PTB | mean |
| --- | --- | --- | --- | --- | --- | --- |
| T's constants (WikiText-2 train) | 8.483 | 9.055 | 8.795 | 8.815 | 8.760 | 8.782 |
| its own source's set (in-domain, from section 6) | 8.483 | 8.531 | 8.068 | 8.293 | 8.329 | 8.341 |
| mixed-all, which contains its source (section 6) | 8.590 | 8.536 | 8.232 | 8.374 | 8.429 | 8.432 |
| mixed-all without its source | 8.758 | 8.835 | 8.594 | 8.695 | 8.754 | 8.727 |
| pick per request, its own source's set left out (chosen on layers 1-4) | 8.758 | 8.835 | 8.594 | 8.695 | 8.760 | 8.729 |
| (sets chosen) | mixed without it/mixed without it | mixed without it/mixed without it | mixed without it/mixed without it | mixed without it/mixed without it | wikitext/wikitext | |
| pick per request with every set (section 6) | 8.483 | 8.526 | 8.068 | 8.293 | 8.329 | 8.340 |

## 7. T2: T with layer 0 recomputed and a constant set per request (kv_compare/t2.py)

T2 is a separate codec on T's kernels; T is unchanged. Dense mode, one window per dataset, cut to a multiple of 256 tokens. Bits per value of the real streams (data, states, starts). T: WikiText-2 constants, layer 0 coded. Decode: the codec's GPU time; T2 adds the receiver's layer-0 recompute.

| Window | set picked | layer 0 | bit-exact | T | T2, T's constants | T2 | saving | decode T | decode T2 | recompute |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ds_wikitext2_r0 | wikitext | recomputed | yes | 8.9795 | 8.6174 | 8.6174 | 4.0% | 30.1 ms | 30.0 ms | 9.9 ms |
| ds_humaneval_r0 | code | recomputed | yes | 9.5719 | 9.2022 | 8.6288 | 9.9% | 30.3 ms | 30.2 ms | 9.8 ms |
| ds_gsm8k_r0 | math | recomputed | yes | 9.2844 | 8.9189 | 8.1637 | 12.1% | 30.3 ms | 30.1 ms | 9.8 ms |
| ds_mmlu_r0 | exam | recomputed | yes | 9.3252 | 8.9576 | 8.3820 | 10.1% | 30.5 ms | 30.2 ms | 9.8 ms |
| ds_ptb_r0 | news | recomputed | yes | 9.2385 | 8.8786 | 8.4384 | 8.7% | 30.3 ms | 30.3 ms | 9.8 ms |

The sender's pick (GPU time per request; the decision against the pick on layers 1-4 of all tokens):

| Pick | time | same set as the full pick |
| --- | --- | --- |
| layers 1-4, all tokens (codecs kept) | 103.5 ms | 5 of 5 windows |
| layers 1-2, all tokens | 81.3 ms | 5 of 5 windows |
| layers 1-4, first 1024 tokens | 63.8 ms | 5 of 5 windows |
| layers 1-2, first 1024 tokens | 58.7 ms | 5 of 5 windows |

Every mode on ds_humaneval_r0 (T2 with its picked set code):

| Mode | bit-exact | T | T2 | saving | decode T | decode T2 |
| --- | --- | --- | --- | --- | --- | --- |
| dense | yes | 9.5719 | 8.6288 | 9.9% | 31.4 ms | 31.6 ms |
| kv | yes | 9.3029 | 8.2849 | 10.9% | 42.5 ms | 42.8 ms |
| 4g | yes | 9.1144 | 8.0773 | 11.4% | 52.7 ms | 52.9 ms |
| hbh | yes | 9.0072 | 7.9599 | 11.6% | 113.0 ms | 109.4 ms |
| rank256 | yes | 9.8239 | 8.9657 | 8.7% | 23.5 ms | 23.2 ms |

## 8. T2 against T and the baselines (same windows, same GPU, same run)

One 8,192-token window per dataset. Bit-exact round trips only (a failed or crashed codec is listed below). T2: layer 0 recomputed by the receiver, its constant set picked per request by the cheap pick (T's int8 product on 1,024 sampled tokens of layers 1-2: 48.9 ms per request on average, the same set as coding layers 1-4 with every set in 5 of 5 windows). T: WikiText-2 constants, layer 0 coded. The baselines are measured by the benchmark's own code (bench_llama_compare's datasets stage).

**Bits per value** (lower is better; ratio = 16 / bits)

| Method | WikiText-2 | HumanEval | GSM8K | MMLU | PTB | mean | ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T2, head by head | 8.004 | 7.960 | 7.477 | 7.772 | 7.804 | 7.803 | 2.05x |
| T2, 4 head groups | 8.112 | 8.077 | 7.597 | 7.879 | 7.916 | 7.916 | 2.02x |
| T2, keys then values | 8.304 | 8.285 | 7.810 | 8.072 | 8.114 | 8.117 | 1.97x |
| T2, dense | 8.617 | 8.629 | 8.164 | 8.382 | 8.438 | 8.446 | 1.89x |
| T, head by head | 8.294 | 9.007 | 8.645 | 8.701 | 8.603 | 8.650 | 1.85x |
| T, 4 head groups | 8.421 | 9.114 | 8.765 | 8.818 | 8.721 | 8.768 | 1.82x |
| T2, rank 256 | 8.984 | 8.966 | 8.546 | 8.742 | 8.814 | 8.810 | 1.82x |
| T, keys then values | 8.643 | 9.303 | 8.975 | 9.024 | 8.929 | 8.975 | 1.78x |
| T, dense | 8.979 | 9.572 | 9.284 | 9.325 | 9.238 | 9.280 | 1.72x |
| T, rank 256 | 9.346 | 9.824 | 9.597 | 9.625 | 9.547 | 9.588 | 1.67x |
| T, SplitZip format, rank 256 | 10.128 | 10.587 | 10.372 | 10.394 | 10.323 | 10.361 | 1.54x |
| exponent coders' floor: 8 + exponent entropy (computed bound) | 10.691 | 10.687 | 10.689 | 10.693 | 10.683 | 10.689 | 1.50x |
| exponent coders' floor: 8 + Huffman(exponent), DFloat11 / ZipNN format (computed bound) | 10.730 | 10.726 | 10.728 | 10.732 | 10.722 | 10.727 | 1.49x |
| ExANS (authors' codec) | 10.872 | 10.874 | 10.881 | 10.879 | 10.863 | 10.874 | 1.47x |
| AdaCode, code from the coded KV (authors' codec) | 11.369 | 11.365 | 11.367 | 11.370 | 11.364 | 11.367 | 1.41x |
| AdaCode (authors' codec) | 11.369 | 11.367 | 11.367 | 11.370 | 11.364 | 11.367 | 1.41x |
| SplitZip (authors' codec) | 12.050 | 12.056 | 12.056 | 12.054 | 12.047 | 12.053 | 1.33x |
| exactkv (authors' codec) | 12.575 | 12.574 | 12.575 | 12.576 | 12.574 | 12.575 | 1.27x |
| raw BF16 (no codec) | 16.000 | 16.000 | 16.000 | 16.000 | 16.000 | 16.000 | 1.00x |

**Time per 8K request** (mean over the windows; T2's encode includes its pick, its decode the receiver's layer-0 recompute; IBP decodes from host memory)

| Method | encode | decode | 1 Gb/s | 10 Gb/s | 25 Gb/s | 100 Gb/s | 400 Gb/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T2, head by head | 154.2 ms | 111.0 ms | 0.27 | 2.73 | 6.82 | 9.01 | 9.01 |
| T2, 4 head groups | 154.2 ms | 63.1 ms | 0.27 | 2.69 | 6.72 | 15.9 | 15.9 |
| T2, keys then values | 153.3 ms | 52.5 ms | 0.26 | 2.62 | 6.56 | 19.0 | 19.0 |
| T2, dense | 130.2 ms | 41.2 ms | 0.25 | 2.52 | 6.30 | 24.3 | 24.3 |
| T, head by head | 107.4 ms | 104.2 ms | 0.25 | 2.46 | 6.15 | 9.59 | 9.59 |
| T, 4 head groups | 107.2 ms | 53.3 ms | 0.24 | 2.43 | 6.07 | 18.8 | 18.8 |
| T2, rank 256 | 89.9 ms | 33.0 ms | 0.24 | 2.42 | 6.04 | 24.2 | 30.3 |
| T, keys then values | 106.6 ms | 42.6 ms | 0.24 | 2.37 | 5.93 | 23.5 | 23.5 |
| T, dense | 74.0 ms | 31.6 ms | 0.23 | 2.29 | 5.73 | 22.9 | 31.6 |
| T, rank 256 | 41.7 ms | 23.1 ms | 0.22 | 2.22 | 5.55 | 22.2 | 43.2 |
| T, SplitZip format, rank 256 | 1002.3 ms | 17.9 ms | 0.21 | 2.05 | 5.14 | 20.5 | 56.0 |
| ExANS (authors' codec) | 176.0 ms | 17.6 ms | 0.20 | 1.96 | 4.89 | 19.6 | 56.9 |
| AdaCode, code from the coded KV (authors' codec) | 21.6 ms | 3.1 ms | 0.19 | 1.87 | 4.68 | 18.7 | 74.9 |
| AdaCode (authors' codec) | 21.4 ms | 3.0 ms | 0.19 | 1.87 | 4.68 | 18.7 | 74.9 |
| SplitZip (authors' codec) | 6.8 ms | 3.0 ms | 0.18 | 1.77 | 4.42 | 17.7 | 70.6 |
| exactkv (authors' codec) | 8.7 ms | 21.2 ms | 0.17 | 1.69 | 4.23 | 16.9 | 47.1 |
| raw BF16 (no codec) | 0.0 ms | 0.0 ms | 0.13 | 1.33 | 3.33 | 13.3 | 53.2 |

Requests per second one link carries, decoding overlapped with arrival (max of wire time and decode time, as the benchmark computes it); the sender's encode runs on the prefill GPU.

| Link | most requests per second |
| --- | --- |
| 1 Gb/s | T2, head by head (0.3/s) |
| 10 Gb/s | T2, head by head (2.7/s) |
| 25 Gb/s | T2, head by head (6.8/s) |
| 100 Gb/s | T2, dense (24.3/s) |
| 400 Gb/s | AdaCode, code from the coded KV (authors' codec) (74.9/s) |

## Verdicts

| Idea | worth building? | evidence |
| --- | --- | --- |
| 1 per-request adaptation | yes | best sendable saving 0.117 bits/value (rule: >= 1% of T, 0.091) |
| 2 layer 0 on the decode side | yes | saving 0.354 bits/value, 100.00% bit-identical (rule: >= 0.1 bits/value and >= 99%) |
| 2c layer 0 holds under other prefill shapes | no | worst saving 0.283 bits/value over the shapes, lean recompute up to 9.8 ms (rule: >= 0.3 bits/value in every shape and <= 10 ms) |
| 2d layer 0 exact when the receiver pads to the sender's steps | yes | worst share bit for bit with padding 100.000% (rule: 100% in every schedule) |
| 4 anchors | no | best saving 0.025 bits/value (rule: >= 0.1; anchors: 2 waves) |
| 5 token classes | no | best saving 0.000 bits/value (rule: >= 0.05; anchors: 2 waves) |
| 2b per-layer mode allocation | no | best gain 3.4% at 3-100 Gb/s (rule: >= 5%) |
| 3 leaner coding kernel (table-free decode) | yes | coding / memory floor 2.35-15.62 (rule: >= 2) |
| 6 calibration breadth | yes | best practical option (pick per request) saves 0.442 bits/value over the five datasets, WikiText +0.000 (rule: >= 1% of T, 0.088) |
| 6b calibration breadth on unseen text | no | mixed-all without its source saves 0.054 bits/value over the datasets (rule: >= 1% of T, 0.088) |
| 7 T2 works | yes | every round trip bit-exact and smaller than T: yes; saving 4.0-12.1% of T's bytes (dense) |
| 8 T2 against the baselines | yes | best T2 (T2, head by head) 7.803 bits/value against the best baseline (ExANS (authors' codec)) 10.874; T2 lower on every dataset: yes |
