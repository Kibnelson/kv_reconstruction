# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |

## MTRAG (multi-turn RAG conversations)

| Codec | Layout | Routing | Traced | Requests (failed) | TTFT p50 (ms) | Prompt tokens mean / max | Transfers | GB sent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | interleaved | conversation | yes | 30 (0) | 817 | 4265 / 9765 | 30 | 14.67 |

## The reuse opportunity (traced PD run)

Per prompt layout and routing. Moved again: tokens P sent that the chosen D already held. Recomputed: tokens P prefilled without its cache; answers: tokens of earlier answers (made on D). Same bytes: pages P sent again that D held and whose bytes D's equal. Affinity: later turns that went to the D of their previous turn.

| Layout / routing | Requests | Moved again (share of sent, GB) | Recomputed tokens | Earlier answers recomputed | Same bytes: answers / docs / rest | Affinity |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved / conversation | 30 | 79% (11.7 GB) | 127,953 | 12,544 of 12,544 | 9% / 100% / 44% | 100% |

## Every page, by what already existed (traced variants)

1 prefix: an earlier request had its token prefix (prefix caches share it). 2 same bytes: not 1, but an earlier page at its layer has its bytes. 3 same text: not 1 or 2, but its 16 tokens were in an earlier prompt (a differential store's candidates). 4 new. Store: what holding every page once would take, keyed by prefix (LMCache), by content, and by content without layer 0 (recomputable from tokens).

| Layout / routing | Pages | 1 prefix | 2 same bytes | 3 same text | 4 new | Store by prefix / content / content w/o layer 0 (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved / conversation | 224,364 | 76.8% | - | 6.6% | 16.7% | 3.4 / 3.4 / 3.3 |

Same bytes (2) by layer, and same text (3) by kind of piece:

- interleaved / conversation: same bytes at none; same text in {'answer': 2044, 'doc': 12712}

## How close related KV is (the differential opportunity)

Related: prompt tokens whose 16-token window appeared earlier in the run, not as a shared prefix (so a prefix cache misses them), at the same place after another prefix or moved; within the conversation or across. Bits per value of coding each against its first occurrence: xor; diff, the difference of the words as ordered integers; rot, keys turned to the new position first; best, the cheapest per layer; alone, the same tokens without a base (sign and exponent entropy + 7 bits, the exponent coders' floor). Raw BF16 is 16. Measured: a sample of the related tokens, chosen by content (1 in the sample's number), each against its first occurrence when it was still held.

| Layout / routing | Prompt tokens | Related (share) | Docs / answers / questions / other | Same place / moved | Within / across conversations | Median shift |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved / conversation | 127,953 | 58,148 (45.4%) | 45,158 / 12,937 / 11 / 42 | 0 / 58,148 | 57,776 / 372 | 1,359 |

Measured related tokens: interleaved / conversation: 5,203 (1 in 8), 0 requests not measured

| Layout / routing | Kind | Measured values | Equal | K: xor / diff / rot / alone | V: xor / diff / alone | Best vs alone (saving) |
| --- | --- | --- | --- | --- | --- | --- |
| interleaved / conversation | every kind | 298,360,832 | 2.6% | 12.92 / 12.40 / 10.15 / 10.74 | 11.19 / 11.03 / 10.66 | 10.59 vs 10.70 (1.0%) |
| interleaved / conversation | answer | 30,851,072 | 2.7% | 12.77 / 12.23 / 10.10 / 10.73 | 11.15 / 10.99 / 10.66 | 10.54 vs 10.70 (1.5%) |
| interleaved / conversation | doc | 267,337,728 | 2.6% | 12.94 / 12.42 / 10.16 / 10.74 | 11.19 / 11.04 / 10.67 | 10.60 vs 10.70 (1.0%) |
| interleaved / conversation | other | 172,032 | 1.7% | 12.30 / 11.93 / 9.13 / 10.74 | 10.37 / 10.05 / 10.62 | 9.59 vs 10.68 (10.2%) |

Best bits per value by layer (keys, values), against alone:

- interleaved / conversation, K: 0: 1.60/10.63, 1: 6.61/10.64, 2: 7.46/10.72, 3: 8.74/10.73, 4: 9.54/10.71, 5: 9.67/10.74, 6: 10.13/10.74, 7: 10.24/10.73, 8: 10.64/10.74, 9: 10.97/10.71, 10: 11.15/10.73, 11: 11.13/10.75, 12: 11.36/10.76, 13: 11.33/10.78, 14: 11.14/10.72, 15: 11.19/10.75, 16: 10.85/10.74, 17: 10.86/10.73, 18: 10.83/10.76, 19: 11.09/10.76, 20: 10.92/10.77, 21: 10.95/10.80, 22: 10.98/10.81, 23: 10.95/10.74, 24: 10.96/10.77, 25: 11.13/10.75, 26: 10.99/10.74, 27: 10.88/10.69
- interleaved / conversation, V: 0: 0.03/10.88, 1: 7.33/10.71, 2: 7.91/10.67, 3: 9.49/10.64, 4: 10.33/10.63, 5: 10.69/10.63, 6: 11.18/10.64, 7: 11.40/10.64, 8: 11.74/10.67, 9: 12.04/10.65, 10: 12.25/10.66, 11: 12.32/10.65, 12: 12.27/10.66, 13: 12.25/10.67, 14: 12.14/10.65, 15: 12.23/10.67, 16: 11.73/10.66, 17: 11.72/10.63, 18: 11.76/10.65, 19: 12.07/10.68, 20: 11.83/10.65, 21: 11.85/10.67, 22: 11.92/10.66, 23: 12.19/10.66, 24: 11.93/10.65, 25: 12.14/10.63, 26: 12.10/10.67, 27: 12.04/10.69

Sizes of the related tokens' KV: raw, coded alone, coded as deltas (GB); and D's pages that P recomputed (it sent them again; D held them):

- interleaved / conversation: 6.67 raw, 4.46 alone, 4.41 as deltas; held pages 177,856, of them 21,701 differ (best 0.86 vs alone 10.70 bits per value, 91.0% of the words equal)


## Delta analysis of the stored pages

28 layers, 8 KV heads of 128; RoPE on; 3.4 GB of P's blocks, 0.38 GB of D's.

Bits per value (raw BF16: 16). alone: sign+exponent entropy + 7 mantissa bits (no base). prev: given the same token's previous layer (a stand-in for our codec's context). xor, diff: against the base, bit length coded. planes0 / planes: a bit-plane context coder without / with the base. base: sign+exponent given the base's, mantissa given the base's when they match. both: base and previous layer together. Keys: the base turned by RoPE to the new place. Cross-validated entropies.

## mtrag_raw_interleaved_conversation

30 requests, 127,953 prompt tokens; prefix any earlier request shares: 98,400 (77%); related beyond it: 10,383 (8%), by kind {'other': 78, 'doc': 8203, 'question': 99, 'answer': 2003}.

Lossless, against the first copy (10,383 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.74 | 10.73 | 11.24 | 10.40 | 10.93 | 9.73 | 9.31 | 9.32 | 2% | 43% |
| V mean | 10.66 | 10.66 | 11.53 | 11.37 | 10.75 | 10.10 | 9.85 | 9.85 | 2% | 32% |
| K 0 | 10.63 | - | 2.03 | 1.60 | 10.82 | 1.88 | 1.61 | - | 69% | 99% |
| K 1 | 10.64 | 10.62 | 7.63 | 6.90 | 10.84 | 7.12 | 6.62 | 6.62 | 5% | 84% |
| K 2 | 10.72 | 10.70 | 8.50 | 7.80 | 10.93 | 7.97 | 7.51 | 7.51 | 4% | 75% |
| K 3 | 10.74 | 10.73 | 9.44 | 8.79 | 10.96 | 8.74 | 8.27 | 8.28 | 3% | 64% |
| K 4 | 10.71 | 10.69 | 10.13 | 9.42 | 10.93 | 9.18 | 8.71 | 8.72 | 2% | 56% |
| K 8 | 10.73 | 10.72 | 11.43 | 10.57 | 10.92 | 9.88 | 9.47 | 9.48 | 2% | 40% |
| K 12 | 10.75 | 10.75 | 12.24 | 11.30 | 10.93 | 10.25 | 9.88 | 9.89 | 1% | 31% |
| K 16 | 10.73 | 10.73 | 11.66 | 10.77 | 10.93 | 9.98 | 9.57 | 9.58 | 2% | 38% |
| K 20 | 10.77 | 10.76 | 11.63 | 10.81 | 10.96 | 10.02 | 9.61 | 9.62 | 2% | 37% |
| K 24 | 10.76 | 10.76 | 11.75 | 10.84 | 10.95 | 10.05 | 9.63 | 9.64 | 2% | 37% |
| K 27 | 10.69 | 10.69 | 11.55 | 10.70 | 10.90 | 9.90 | 9.50 | 9.50 | 2% | 39% |
| V 0 | 10.90 | - | 0.10 | 0.09 | 11.00 | 0.10 | 0.09 | - | 99% | 100% |
| V 1 | 10.71 | 10.70 | 8.24 | 7.69 | 10.88 | 7.86 | 7.42 | 7.42 | 4% | 76% |
| V 2 | 10.67 | 10.68 | 8.79 | 8.28 | 10.76 | 8.31 | 7.91 | 7.92 | 3% | 70% |
| V 3 | 10.63 | 10.63 | 9.91 | 9.51 | 10.72 | 9.21 | 8.85 | 8.85 | 2% | 55% |
| V 4 | 10.63 | 10.63 | 10.55 | 10.23 | 10.71 | 9.64 | 9.30 | 9.30 | 2% | 46% |
| V 8 | 10.67 | 10.68 | 11.82 | 11.68 | 10.76 | 10.33 | 10.09 | 10.10 | 1% | 28% |
| V 12 | 10.66 | 10.67 | 12.28 | 12.21 | 10.75 | 10.48 | 10.28 | 10.28 | 1% | 22% |
| V 16 | 10.66 | 10.66 | 11.78 | 11.65 | 10.76 | 10.30 | 10.04 | 10.04 | 1% | 28% |
| V 20 | 10.65 | 10.65 | 11.82 | 11.70 | 10.74 | 10.30 | 10.05 | 10.06 | 1% | 27% |
| V 24 | 10.66 | 10.66 | 11.85 | 11.76 | 10.73 | 10.31 | 10.07 | 10.08 | 1% | 27% |
| V 27 | 10.69 | 10.69 | 11.95 | 11.86 | 10.76 | 10.38 | 10.17 | 10.18 | 1% | 26% |

Lossless, against the latest copy (10,089 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.74 | 10.73 | 9.60 | 8.90 | 10.93 | 8.76 | 8.35 | 8.35 | 4% | 63% |
| V mean | 10.66 | 10.66 | 10.23 | 9.91 | 10.75 | 9.38 | 9.08 | 9.08 | 2% | 51% |
| K 0 | 10.63 | - | 2.09 | 1.64 | 10.82 | 1.93 | 1.64 | - | 68% | 99% |
| K 1 | 10.64 | 10.62 | 6.40 | 5.75 | 10.84 | 6.10 | 5.62 | 5.62 | 12% | 90% |
| K 2 | 10.72 | 10.70 | 7.39 | 6.75 | 10.93 | 7.06 | 6.60 | 6.61 | 7% | 84% |
| K 3 | 10.74 | 10.73 | 8.04 | 7.43 | 10.96 | 7.65 | 7.20 | 7.20 | 6% | 79% |
| K 4 | 10.71 | 10.69 | 8.44 | 7.82 | 10.93 | 7.96 | 7.52 | 7.52 | 5% | 75% |
| K 8 | 10.74 | 10.72 | 9.85 | 9.13 | 10.92 | 8.97 | 8.57 | 8.58 | 3% | 60% |
| K 12 | 10.76 | 10.75 | 10.60 | 9.84 | 10.93 | 9.43 | 9.06 | 9.06 | 2% | 52% |
| K 16 | 10.73 | 10.73 | 10.02 | 9.30 | 10.93 | 9.08 | 8.68 | 8.68 | 3% | 58% |
| K 20 | 10.77 | 10.77 | 9.92 | 9.24 | 10.96 | 9.05 | 8.65 | 8.65 | 3% | 59% |
| K 24 | 10.76 | 10.76 | 9.91 | 9.18 | 10.95 | 9.01 | 8.61 | 8.61 | 3% | 60% |
| K 27 | 10.69 | 10.69 | 9.62 | 8.94 | 10.90 | 8.79 | 8.39 | 8.39 | 3% | 63% |
| V 0 | 10.90 | - | 0.11 | 0.10 | 10.99 | 0.10 | 0.10 | - | 99% | 100% |
| V 1 | 10.71 | 10.70 | 7.55 | 6.99 | 10.88 | 7.25 | 6.81 | 6.82 | 7% | 82% |
| V 2 | 10.67 | 10.68 | 8.07 | 7.52 | 10.76 | 7.71 | 7.30 | 7.30 | 5% | 78% |
| V 3 | 10.63 | 10.64 | 8.73 | 8.24 | 10.72 | 8.29 | 7.91 | 7.92 | 4% | 71% |
| V 4 | 10.63 | 10.63 | 9.26 | 8.81 | 10.71 | 8.71 | 8.35 | 8.36 | 3% | 64% |
| V 8 | 10.67 | 10.67 | 10.62 | 10.33 | 10.76 | 9.69 | 9.40 | 9.40 | 2% | 46% |
| V 12 | 10.67 | 10.67 | 11.08 | 10.86 | 10.75 | 9.94 | 9.68 | 9.68 | 2% | 40% |
| V 16 | 10.66 | 10.66 | 10.48 | 10.18 | 10.76 | 9.59 | 9.28 | 9.29 | 2% | 48% |
| V 20 | 10.65 | 10.65 | 10.40 | 10.09 | 10.74 | 9.53 | 9.23 | 9.24 | 2% | 49% |
| V 24 | 10.66 | 10.66 | 10.33 | 10.04 | 10.73 | 9.49 | 9.19 | 9.20 | 2% | 50% |
| V 27 | 10.69 | 10.69 | 10.31 | 10.02 | 10.76 | 9.49 | 9.20 | 9.21 | 2% | 50% |

Relative deviation |target - base| / |target| by place in the related run (tokens since it started), layers 1 / 4 / 12 / 27:

- K 0-16: 0.06 / 0.17 / 0.46 / 0.27 (2,120 tokens)
- K 16-64: 0.05 / 0.22 / 0.54 / 0.37 (2,441 tokens)
- K 64-256: 0.06 / 0.22 / 0.55 / 0.39 (3,845 tokens)
- K 256-: 0.06 / 0.15 / 0.49 / 0.32 (1,977 tokens)
- V 0-16: 0.13 / 0.36 / 0.76 / 0.54 (2,120 tokens)
- V 16-64: 0.09 / 0.41 / 0.83 / 0.76 (2,441 tokens)
- V 64-256: 0.07 / 0.42 / 0.86 / 0.79 (3,845 tokens)
- V 256-: 0.08 / 0.32 / 0.79 / 0.69 (1,977 tokens)

Whole related chunks (39, up to 512 tokens each), per layer (mean): relative size of the difference; the energy rank 4 / 16 / 64 holds; tokens for 50% / 90% of it; the top 15% tokens' share; overlap of the top 15% with layer 2's; the largest head's share:

| | rel | rank 4 / 16 / 64 | tokens for 50% / 90% | top 15% share | overlap with layer 2 | top head |
| --- | --- | --- | --- | --- | --- | --- |
| K 1 | 0.05 | 73% / 88% / 98% | 24% / 75% | 40% | 64% | 22% |
| K 2 | 0.10 | 72% / 87% / 97% | 21% / 73% | 44% | 100% | 22% |
| K 4 | 0.22 | 72% / 85% / 97% | 27% / 72% | 34% | 32% | 23% |
| K 8 | 0.39 | 56% / 76% / 94% | 27% / 76% | 36% | 24% | 20% |
| K 12 | 0.54 | 56% / 77% / 94% | 32% / 79% | 30% | 19% | 16% |
| K 16 | 0.41 | 50% / 71% / 92% | 34% / 80% | 28% | 21% | 15% |
| K 20 | 0.40 | 51% / 72% / 92% | 36% / 81% | 26% | 19% | 17% |
| K 24 | 0.41 | 54% / 72% / 92% | 37% / 81% | 25% | 18% | 18% |
| K 27 | 0.38 | 47% / 67% / 90% | 36% / 79% | 25% | 15% | 17% |
| V 1 | 0.09 | 55% / 80% / 96% | 13% / 64% | 55% | 63% | 26% |
| V 2 | 0.15 | 59% / 81% / 96% | 16% / 69% | 50% | 100% | 34% |
| V 4 | 0.44 | 66% / 82% / 96% | 25% / 70% | 36% | 34% | 21% |
| V 8 | 0.77 | 46% / 68% / 92% | 27% / 76% | 34% | 22% | 26% |
| V 12 | 0.86 | 34% / 60% / 89% | 31% / 79% | 30% | 20% | 17% |
| V 16 | 0.74 | 36% / 59% / 88% | 35% / 80% | 26% | 14% | 25% |
| V 20 | 0.82 | 35% / 57% / 86% | 35% / 79% | 27% | 11% | 22% |
| V 24 | 0.78 | 32% / 54% / 85% | 35% / 79% | 26% | 9% | 17% |
| V 27 | 0.77 | 32% / 55% / 86% | 36% / 79% | 25% | 8% | 21% |

## D's held copies against P's recomputation

784 blocks whose bytes differ (same text, same place), mean over layers 1-27:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K | 10.73 | 10.72 | 4.46 | 3.88 | 10.94 | 4.28 | 3.82 | 3.83 | 31% |
| V | 10.65 | 10.65 | 5.47 | 4.85 | 10.74 | 5.26 | 4.78 | 4.79 | 18% |




# The conversations' block tree, kept by a rule

Pages of 16 tokens x 28 layers (1,835,008 B raw). The requests replayed in arrival order; a page is identified by its bytes (the capture's rows). Hits: blocks found in the store from the request's first block on (what a prefix cache reuses); the rest are computed. Peak: the store at its largest, coded with T2 per run of contiguous kept pages (layer 0 recomputed), decoded back and checked byte for byte. Bits: per value (16 raw).

Prefill in the capture's run: 0.035 ms per token at the median over 30 requests (arrival to the end of P's prefill, queueing included; P had no prefix cache), so 0.6 ms to recompute a page.

Recompute is exact: D compared the pages it held with P's recompute of the same prompt prefix: interleaved / conversation: 177,856 pages of one layer (6,352 blocks), 21701 differing.

## mtrag_raw_interleaved_conversation

30 requests, 1,835 distinct pages. P computed 6,150 blocks again under a token prefix computed before: 6,150 with the same bytes, 0 not. Blocks with an earlier block's bytes under another token prefix in the run's records: 0.

| Rule | Hits (blocks reused) | Prefill left (tokens) | Prefill left (s) | Peak pages | Peak raw GB | Peak t2-hbh GB (bits) | Peak t2-rank256 GB (bits) | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Everything (every distinct page) | 6,150 of 7,985 | 29,553 of 127,953 | 1.0 | 1,835 | 3.37 | 2.34 (11.14) | 2.54 (12.08) | 30 of 30 runs; 30 of 30 runs |
| Newest path per conversation | 6,150 of 7,985 | 29,553 of 127,953 | 1.0 | 1,835 | 3.37 | 2.34 (11.14) | 2.54 (12.08) | 30 of 30 runs; 30 of 30 runs |
| Newest path, no document pages | 56 of 7,985 | 127,057 of 127,953 | 4.4 | 235 | 0.43 | 0.79 (29.21) | 0.82 (30.38) | 30 of 30 runs; 30 of 30 runs |
| Hindsight (kept until its last use) | 6,150 of 7,985 | 29,553 of 127,953 | 1.0 | 1,371 | 2.52 | 1.72 (10.94) | 1.87 (11.88) | 22 of 22 runs; 22 of 22 runs |
| Nothing kept | 0 of 7,985 | 127,953 of 127,953 | 4.4 | 0 | 0.00 | - | - | - |

Padding, on the first requests of 256 tokens or more (bits per real value; the padding's bytes counted):

| Codec | Requests | As now: zeros to 256 | No padding (the 256-multiple below) | Tail apart, zeros to 32 | Tail apart, copies to 32 | Copies of the last token to 256 | Padding tokens now / split | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| t2-hbh | 24 | 9.10 | 8.30 | 8.35 | 8.33 | 8.80 | 3,104 / 192 | 24 of 24 |
| t2-rank256 | 24 | 10.02 | 9.21 | 9.26 | 9.23 | 9.66 | 3,104 / 192 | 24 of 24 |

## Times (CUDA-synchronised, one GPU)

| Step | n | ms per page | mean ms per call |
| --- | ---: | ---: | ---: |
| t2-hbh kept decode | 60 | 1.526 | 52.65 |
| t2-hbh kept encode | 60 | 0.611 | 21.07 |
| t2-hbh probe decode | 118 | - | 51.62 |
| t2-hbh probe encode | 118 | - | 24.07 |
| t2-rank256 kept decode | 60 | 0.309 | 10.66 |
| t2-rank256 kept encode | 60 | 0.305 | 10.52 |
| t2-rank256 probe decode | 118 | - | 8.24 |
| t2-rank256 probe encode | 118 | - | 19.01 |
| recompute (the run's prefill) | - | 0.554 | - |

# Diff edges: pages coded against their earlier versions (plain), then with T's prediction

Pages of 16 tokens x 28 layers. Every value coded with the value-slice coder (exact mode) against a prediction made from the page's earlier versions by plain arithmetic (keys turned by RoPE to the page's place), decoded and checked. Weights and spreads fitted on the even conversations; held out: the odd ones. Bits per value (raw 16), all layers coded. Agreement bucket: the median over the page's tokens of how many tokens up to them its text has agreed with its latest version's.

## mtrag_raw_interleaved_conversation

7,985 blocks in 4 conversations: 6,150 references (bytes P stored before), 1,835 distinct pages: 1,440 twins with D's bytes, 208 twins with other bytes, 49 moved (1 / 2 / 3+ earlier versions: 8 / 0 / 41), 138 new.

Held out: 7 moved pages, bits per value by agreement bucket (pages)

| Edge | All | 0-15 (1) | 16-63 (3) | 64-255 (3) | 256-1023 (0) | 1024+ (0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Its first version (star) | 9.16 | 9.30 | 9.24 | 9.04 | - | - |
| Its latest version (chain) | 10.29 | 9.28 | 9.63 | 11.28 | - | - |
| Latest, scale + offset per channel | 10.24 | 9.25 | 9.58 | 11.22 | - | - |
| Second order: latest + l (latest - previous) | 10.32 | 9.22 | 9.58 | 11.43 | - | - |
| a latest + b previous + c first | 10.32 | 9.22 | 9.58 | 11.42 | - | - |
| Composite: a latest + b drift + c first + d, per channel | 10.44 | 9.28 | 9.66 | 11.59 | - | - |
| T alone, its own spread (as now) | 8.22 | 8.23 | 8.21 | 8.23 | - | - |
| T alone, fitted spread | 8.23 | 8.24 | 8.22 | 8.23 | - | - |
| Blend: T + w (latest - T) | 8.89 | 8.17 | 8.24 | 9.79 | - | - |
| Best plain edge per page (or raw) | 9.13 | 9.22 | 9.20 | 9.04 | - | - |
| Best edge per page (or raw) | 8.17 | 8.17 | 8.11 | 8.23 | - | - |
| Rule by bucket, from training (t_blend / composite / composite / composite / latest) | 10.28 | 8.17 | 9.66 | 11.59 | - | - |

Values equal to the latest version's (V; keys are turned): 4.3%. Best edge per page: t_fit 3, t_blend 3, t 1.

Held out, by earlier versions (pages): diff of a diff needs two

| Edge | 1 (7) | 2 (0) | 3+ (0) |
| --- | ---: | ---: | ---: |
| Its first version (star) | 9.16 | - | - |
| Its latest version (chain) | 10.29 | - | - |
| Latest, scale + offset per channel | 10.24 | - | - |
| Second order: latest + l (latest - previous) | 10.32 | - | - |
| a latest + b previous + c first | 10.32 | - | - |
| Composite: a latest + b drift + c first + d, per channel | 10.44 | - | - |
| T alone, its own spread (as now) | 8.22 | - | - |
| T alone, fitted spread | 8.23 | - | - |
| Blend: T + w (latest - T) | 8.89 | - | - |

Held out, by what the page holds (pages)

| Edge | answer (1) | doc (6) |
| --- | ---: | ---: |
| Its first version (star) | 8.73 | 9.23 |
| Its latest version (chain) | 9.08 | 10.49 |
| Latest, scale + offset per channel | 9.09 | 10.43 |
| Second order: latest + l (latest - previous) | 9.17 | 10.51 |
| a latest + b previous + c first | 9.16 | 10.51 |
| Composite: a latest + b drift + c first + d, per channel | 9.27 | 10.63 |
| T alone, its own spread (as now) | 8.35 | 8.20 |
| T alone, fitted spread | 8.34 | 8.21 |
| Blend: T + w (latest - T) | 9.92 | 8.72 |

Held out, by layer group and K/V

| Edge | K 0-0 | V 0-0 | K 1-4 | V 1-4 | K 5-13 | V 5-13 | K 14-27 | V 14-27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Its first version (star) | 2.47 | 0.28 | 7.12 | 7.80 | 9.35 | 10.15 | 9.38 | 10.27 |
| Its latest version (chain) | 2.34 | 0.28 | 7.32 | 8.10 | 10.58 | 11.18 | 10.85 | 11.72 |
| Latest, scale + offset per channel | 2.35 | 0.28 | 7.28 | 8.07 | 10.56 | 11.13 | 10.83 | 11.60 |
| Second order: latest + l (latest - previous) | 2.35 | 0.28 | 7.40 | 8.13 | 10.64 | 11.20 | 10.89 | 11.73 |
| a latest + b previous + c first | 2.34 | 0.28 | 7.42 | 8.14 | 10.64 | 11.20 | 10.88 | 11.73 |
| Composite: a latest + b drift + c first + d, per channel | 2.30 | 0.28 | 7.43 | 8.18 | 10.75 | 11.33 | 11.04 | 11.87 |
| T alone, its own spread (as now) | 2.34 | 0.28 | 7.84 | 8.97 | 8.22 | 9.02 | 7.79 | 9.03 |
| T alone, fitted spread | 2.34 | 0.28 | 7.84 | 8.99 | 8.22 | 9.03 | 7.79 | 9.02 |
| Blend: T + w (latest - T) | 2.34 | 0.28 | 9.66 | 10.60 | 8.98 | 9.53 | 8.16 | 9.53 |
| Twin: P's page against D's copy | 3.02 | 5.00 | 3.10 | 3.73 | 4.21 | 4.97 | 4.17 | 4.88 |

Held out: 104 twins: 4.37 bits per value; 26.2% of their words equal to D's.

Training (fitted on these): 42 moved pages, bits per value by agreement bucket (pages)

| Edge | All | 0-15 (2) | 16-63 (9) | 64-255 (21) | 256-1023 (10) | 1024+ (0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Its first version (star) | 9.27 | 9.58 | 9.35 | 9.29 | 9.09 | - |
| Its latest version (chain) | 6.79 | 8.62 | 6.86 | 6.76 | 6.41 | - |
| Latest, scale + offset per channel | 6.77 | 8.58 | 6.86 | 6.74 | 6.39 | - |
| Second order: latest + l (latest - previous) | 6.73 | 8.50 | 6.83 | 6.71 | 6.33 | - |
| a latest + b previous + c first | 6.73 | 8.51 | 6.83 | 6.71 | 6.33 | - |
| Composite: a latest + b drift + c first + d, per channel | 6.64 | 8.45 | 6.74 | 6.65 | 6.15 | - |
| T alone, its own spread (as now) | 8.22 | 8.31 | 8.16 | 8.21 | 8.26 | - |
| T alone, fitted spread | 8.18 | 8.30 | 8.13 | 8.17 | 8.21 | - |
| Blend: T + w (latest - T) | 6.79 | 7.94 | 6.86 | 6.81 | 6.44 | - |
| Best plain edge per page (or raw) | 6.63 | 8.41 | 6.74 | 6.65 | 6.15 | - |
| Best edge per page (or raw) | 6.60 | 7.87 | 6.73 | 6.64 | 6.15 | - |
| Rule by bucket, from training (t_blend / composite / composite / composite / latest) | 6.61 | 7.94 | 6.74 | 6.65 | 6.15 | - |

Training (fitted on these): 104 twins: 3.86 bits per value; 28.0% of their words equal to D's.

The graph (MB): plain edges, then with T's

| | Held out | Training | All |
| --- | ---: | ---: | ---: |
| Raw, every block | 6,580.3 | 8,072.2 | 14,652.5 |
| Raw, distinct pages (references free) | 1,471.7 | 1,895.6 | 3,367.2 |
| Twins (D's copy) as diffs, or raw | 52.1 | 46.1 | 98.2 |
| Moved pages, best plain edge or raw | 7.3 | 31.9 | 39.3 |
| Moved pages, best edge with T's, or raw | 6.6 | 31.8 | 38.4 |
| Moved pages, latest edge or raw | 8.3 | 32.7 | 41.0 |
| New pages raw | 122.9 | 130.3 | 253.2 |
| **The plain graph** | 182.4 | 208.3 | 390.7 |
| The plain graph, latest edge only | 183.3 | 209.0 | 392.3 |
| **The graph with T's edges** | 181.6 | 208.1 | 389.8 |
| Plain graph under raw distinct pages | 87.6% | 89.0% | 88.4% |
| With T's edges, under raw distinct pages | 87.7% | 89.0% | 88.4% |

Chains and stars (moved pages; a twin is depth 1)

| Coded against | depth: pages | pages to decode, mean / p90 / max |
| --- | --- | --- |
| latest | 1: 49 | 1.0 / 1 / 1 |
| first | 1: 49 | 1.0 / 1 / 1 |

Rebuild: turning a version 2.94 ms per page, decoding 6.821 ms per page (batches of 32); recomputing: 0.55 ms per page (P's prefill in this run, 30 requests).

Fitted weights, pages with two versions or more, by agreement bucket (mean over layers): second order's l: K -0.54 / -0.17 / -0.25 / -0.23 / 0.00; V -0.54 / -0.27 / -0.29 / -0.30 / 0.00.
a latest + b previous + c first, (a, b, c): K (0.51, 0.49, 0.00) / (0.85, 0.14, 0.01) / (0.76, 0.24, 0.01) / (0.78, 0.22, -0.01) / (1.00, 0.00, 0.00); V (0.43, 0.52, 0.02) / (0.72, 0.26, 0.02) / (0.70, 0.29, 0.01) / (0.70, 0.30, -0.01) / (1.00, 0.00, 0.00).
Composite (a, b, c, d), mean over channels too: K (0.97, -0.52, 0.01, 0.00) / (0.98, -0.26, 0.01, 0.00) / (0.98, -0.29, 0.01, -0.00) / (1.00, -0.27, -0.00, -0.00) / (1.00, 0.00, 0.00, 0.00); V (0.94, -0.49, 0.02, -0.00) / (0.97, -0.27, 0.02, 0.00) / (0.97, -0.29, 0.01, 0.00) / (0.99, -0.30, -0.00, -0.00) / (1.00, 0.00, 0.00, 0.00).

Latest's scale and offset (mean over layers and channels): K 0.982, -0.0009; V 0.965, -0.0001.

T's edges: t2-hbh, set wikitext (smallest on the first training pages); layers 1.. with T's coder (uniform mode), layer 0 as the latest edge codes it.
Blend w (0: T alone, 1: the latest version), pages with two versions or more, by agreement bucket: K 1-4 0.96 / 0.96 / 0.99 / 0.99 / -; V 1-4 0.93 / 0.95 / 0.98 / 0.99 / -; K 5-13 0.57 / 0.83 / 0.81 / 0.89 / -; V 5-13 0.49 / 0.75 / 0.72 / 0.83 / -; K 14-27 0.43 / 0.72 / 0.64 / 0.83 / -; V 14-27 0.57 / 0.78 / 0.77 / 0.90 / -.
With one version: K 1-4 0.48 / 0.50 / - / - / -; V 1-4 0.40 / 0.42 / - / - / -; K 5-13 0.11 / 0.09 / - / - / -; V 5-13 0.10 / 0.10 / - / - / -; K 14-27 0.05 / 0.11 / - / - / -; V 14-27 0.07 / 0.13 / - / - / -.

Spread multipliers chosen (by class: buckets with one version, then two or more): first 1/1/1/1/1/1/1/1/1/1; latest 1/1/1/1/1/1/1/1/1/1; latest_affine 1/1/1/1/1/1/1/1/1/1; second_order 1/1/1/1/1/1/1/1/1/1; multi_ref 1/1/1/1/1/1/1/1/1/1; composite 1/1/1/1/1/1/1/1/1/1; twin 1/1/1/1/1/1/1/1/1/1; t_fit 1/1/1/1/1/1/1/1/1/1; t_blend 1/1/1/1/1/1/1/1/1/1.

Coded a page and a slice at a time (the tables) against whole batches of 32 pages, all layers (bits per value; the difference is the coder's streams' overhead): latest 7.29 vs 7.27; second_order 7.24 vs 7.22; composite 7.18 vs 7.16; twin 4.11 vs 4.09.

Decoded exactly: latest 394 of 394, second_order 394 of 394, composite 394 of 394, first 392 of 392, latest_affine 392 of 392, multi_ref 392 of 392, t 294 of 294, t_fit 294 of 294, t_blend 294 of 294, twin 1,671 of 1,671 (ALL).

## Times (CUDA-synchronised, one GPU)

| Step | calls | ms per page | mean ms per call |
| --- | ---: | ---: | ---: |
| batch composite decode | 2 | 0.20 | 4.89 |
| batch composite encode | 2 | 0.13 | 3.06 |
| batch latest decode | 2 | 6.82 | 167.10 |
| batch latest encode | 2 | 5.22 | 127.88 |
| batch second_order decode | 2 | 0.20 | 4.89 |
| batch second_order encode | 2 | 0.12 | 3.05 |
| batch twin decode | 7 | 0.14 | 4.22 |
| batch twin encode | 7 | 0.11 | 3.16 |
| predict composite | 2 | 0.24 | 5.89 |
| predict first | 2 | 0.07 | 1.75 |
| predict latest | 2 | 0.07 | 1.61 |
| predict latest_affine | 2 | 0.11 | 2.59 |
| predict multi_ref | 2 | 0.15 | 3.65 |
| predict second_order | 2 | 0.12 | 2.99 |
| predict t | 18 | 1.43 | 30.56 |
| turn versions | 18 | 2.94 | 188.65 |

Sample for analysis: `diffedge_sample.npz` (17.2 MB): 49 moved pages with their latest, previous and first versions, 64 twins with D's copy; layers 0, 1, 3, 4, 12, 13, 26, 27, head 0's keys and values; per-page sizes in `diffedge_pages.json`.
