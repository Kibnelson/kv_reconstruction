# T ideas: measured

Run folder `/home/nelson/kv_a5000/runs/llama32_3b`, calibration wikitext, commit `02fe4d59af8052b07b997935731692dba1128f3b`, NVIDIA RTX A4500. Bits per value are exact costs (-log2 P of every BF16 word); side bits are added where a variant sends something. Decision rules: docs/t_ideas.md.

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

## Verdicts

| Idea | worth building? | evidence |
| --- | --- | --- |
| 1 per-request adaptation | yes | best sendable saving 0.117 bits/value (rule: >= 1% of T, 0.091) |
| 2 layer 0 on the decode side | yes | saving 0.354 bits/value, 100.00% bit-identical (rule: >= 0.1 bits/value and >= 99%) |
| 4 anchors | no | best saving 0.025 bits/value (rule: >= 0.1; anchors: 2 waves) |
| 5 token classes | no | best saving 0.000 bits/value (rule: >= 0.05; anchors: 2 waves) |
| 2b per-layer mode allocation | no | best gain 3.4% at 3-100 Gb/s (rule: >= 5%) |
| 3 leaner coding kernel (table-free decode) | yes | coding / memory floor 2.35-15.62 (rule: >= 2) |
