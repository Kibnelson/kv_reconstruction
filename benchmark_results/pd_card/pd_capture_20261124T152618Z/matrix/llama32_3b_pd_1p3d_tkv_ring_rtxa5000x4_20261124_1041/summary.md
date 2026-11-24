# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 3/4 | 16.000 |

## MTRAG (multi-turn RAG conversations)

| Codec | Layout | Routing | Traced | Requests (failed) | TTFT p50 (ms) | Prompt tokens mean / max | Transfers | GB sent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | docs_first | conversation | yes | 30 (0) | 616 | 1346 / 2424 | 30 | 4.63 |
| raw | history_first | conversation | yes | 30 (0) | 552 | 1354 / 2424 | 30 | 4.66 |

## The reuse opportunity (traced PD run)

Per prompt layout and routing. Moved again: tokens P sent that the chosen D already held. Recomputed: tokens P prefilled without its cache; answers: tokens of earlier answers (made on D). Same bytes: pages P sent again that D held and whose bytes D's equal. Affinity: later turns that went to the D of their previous turn.

| Layout / routing | Requests | Moved again (share of sent, GB) | Recomputed tokens | Earlier answers recomputed | Same bytes: answers / docs / rest | Affinity |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 30 | 12% (0.6 GB) | 40,390 | 12,304 of 12,304 | - / 100% / 100% | 100% |
| history_first / conversation | 30 | 26% (1.2 GB) | 40,630 | 12,544 of 12,544 | 100% / - / 100% | 100% |

## Every page, by what already existed (traced variants)

1 prefix: an earlier request had its token prefix (prefix caches share it). 2 same bytes: not 1, but an earlier page at its layer has its bytes. 3 same text: not 1 or 2, but its 16 tokens were in an earlier prompt (a differential store's candidates). 4 new. Store: what holding every page once would take, keyed by prefix (LMCache), by content, and by content without layer 0 (recomputable from tokens).

| Layout / routing | Pages | 1 prefix | 2 same bytes | 3 same text | 4 new | Store by prefix / content / content w/o layer 0 (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 71,148 | 12.5% | 0.1% | 32.8% | 54.7% | 4.1 / 4.1 / 3.9 |
| history_first / conversation | 71,512 | 26.2% | 0.0% | 20.2% | 53.6% | 3.5 / 3.5 / 3.3 |

Same bytes (2) by layer, and same text (3) by kind of piece:

- docs_first / conversation: same bytes at layer 0: 43; same text in {'doc': 5096, 'answer': 16706, 'other': 1535}
- history_first / conversation: same bytes at layer 0: 8; same text in {'answer': 1848, 'doc': 12592}

## How close related KV is (the differential opportunity)

Related: prompt tokens whose 16-token window appeared earlier in the run, not as a shared prefix (so a prefix cache misses them), at the same place after another prefix or moved; within the conversation or across. Bits per value of coding each against its first occurrence: xor; diff, the difference of the words as ordered integers; rot, keys turned to the new position first; best, the cheapest per layer; alone, the same tokens without a base (sign and exponent entropy + 7 bits, the exponent coders' floor). Raw BF16 is 16. Measured: a sample of the related tokens, chosen by content (1 in the sample's number), each against its first occurrence when it was still held.

| Layout / routing | Prompt tokens | Related (share) | Docs / answers / questions / other | Same place / moved | Within / across conversations | Median shift |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 40,390 | 22,357 (55.4%) | 6,254 / 14,238 / 1,434 / 431 | 139 / 22,218 | 22,295 / 62 | 195 |
| history_first / conversation | 40,630 | 17,257 (42.5%) | 10,014 / 6,985 / 205 / 53 | 0 / 17,257 | 17,195 / 62 | 370 |

Measured related tokens: docs_first / conversation: 1,848 (1 in 8), 0 requests not measured; history_first / conversation: 1,501 (1 in 8), 0 requests not measured

| Layout / routing | Kind | Measured values | Equal | K: xor / diff / rot / alone | V: xor / diff / alone | Best vs alone (saving) |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | every kind | 105,971,712 | 4.0% | 11.60 / 11.13 / 8.15 / 10.73 | 9.31 / 8.93 / 10.65 | 8.54 vs 10.69 (20.1%) |
| docs_first / conversation | answer | 74,145,792 | 4.0% | 11.65 / 11.17 / 8.23 / 10.73 | 9.39 / 9.03 / 10.65 | 8.63 vs 10.69 (19.3%) |
| docs_first / conversation | doc | 26,320,896 | 4.1% | 11.52 / 11.07 / 7.95 / 10.73 | 9.06 / 8.65 / 10.65 | 8.30 vs 10.69 (22.3%) |
| docs_first / conversation | other | 1,204,224 | 3.8% | 11.21 / 10.80 / 7.85 / 10.73 | 9.19 / 8.77 / 10.65 | 8.31 vs 10.69 (22.3%) |
| docs_first / conversation | question | 4,300,800 | 3.7% | 11.45 / 10.96 / 8.08 / 10.71 | 9.37 / 8.98 / 10.65 | 8.53 vs 10.68 (20.2%) |
| history_first / conversation | every kind | 86,073,344 | 3.5% | 11.90 / 11.45 / 8.55 / 10.73 | 9.60 / 9.25 / 10.65 | 8.90 vs 10.69 (16.7%) |
| history_first / conversation | answer | 34,406,400 | 3.1% | 11.96 / 11.45 / 9.12 / 10.73 | 10.18 / 9.89 / 10.66 | 9.51 vs 10.69 (11.1%) |
| history_first / conversation | doc | 50,749,440 | 3.7% | 11.85 / 11.44 / 8.11 / 10.73 | 9.15 / 8.74 / 10.65 | 8.43 vs 10.69 (21.2%) |
| history_first / conversation | other | 573,440 | 2.8% | 12.09 / 11.61 / 9.28 / 10.72 | 10.50 / 10.23 / 10.63 | 9.76 vs 10.68 (8.6%) |
| history_first / conversation | question | 344,064 | 2.8% | 12.26 / 11.75 / 9.55 / 10.70 | 10.54 / 10.28 / 10.62 | 9.91 vs 10.66 (7.1%) |

Best bits per value by layer (keys, values), against alone:

- docs_first / conversation, K: 0: 1.51/10.62, 1: 5.84/10.63, 2: 6.83/10.72, 3: 7.51/10.73, 4: 7.71/10.72, 5: 7.86/10.73, 6: 8.15/10.74, 7: 8.28/10.73, 8: 8.62/10.75, 9: 8.88/10.73, 10: 8.98/10.74, 11: 8.93/10.75, 12: 9.23/10.75, 13: 9.15/10.78, 14: 8.88/10.70, 15: 8.88/10.72, 16: 8.62/10.72, 17: 8.57/10.72, 18: 8.55/10.74, 19: 8.68/10.76, 20: 8.58/10.76, 21: 8.57/10.78, 22: 8.56/10.79, 23: 8.58/10.72, 24: 8.56/10.75, 25: 8.67/10.76, 26: 8.66/10.71, 27: 8.47/10.71
- docs_first / conversation, V: 0: 0.00/10.90, 1: 7.16/10.70, 2: 7.65/10.68, 3: 8.23/10.63, 4: 8.65/10.62, 5: 8.83/10.63, 6: 9.17/10.63, 7: 9.32/10.63, 8: 9.66/10.65, 9: 9.85/10.65, 10: 9.96/10.65, 11: 10.15/10.63, 12: 10.04/10.64, 13: 9.93/10.64, 14: 9.78/10.63, 15: 9.68/10.64, 16: 9.24/10.64, 17: 9.24/10.63, 18: 9.22/10.64, 19: 9.46/10.64, 20: 9.22/10.65, 21: 9.22/10.65, 22: 9.26/10.64, 23: 9.48/10.64, 24: 9.31/10.64, 25: 9.43/10.63, 26: 9.54/10.66, 27: 9.53/10.67
- history_first / conversation, K: 0: 1.55/10.62, 1: 6.13/10.64, 2: 7.11/10.72, 3: 7.83/10.73, 4: 8.06/10.71, 5: 8.25/10.74, 6: 8.55/10.74, 7: 8.64/10.72, 8: 9.02/10.74, 9: 9.30/10.72, 10: 9.39/10.74, 11: 9.33/10.74, 12: 9.63/10.75, 13: 9.51/10.78, 14: 9.29/10.71, 15: 9.29/10.73, 16: 8.99/10.72, 17: 9.00/10.72, 18: 8.98/10.74, 19: 9.10/10.76, 20: 8.97/10.76, 21: 9.02/10.78, 22: 9.04/10.79, 23: 9.05/10.72, 24: 9.07/10.75, 25: 9.18/10.74, 26: 9.19/10.72, 27: 9.05/10.71
- history_first / conversation, V: 0: 0.31/10.90, 1: 7.13/10.71, 2: 7.76/10.68, 3: 8.41/10.63, 4: 8.86/10.63, 5: 9.05/10.63, 6: 9.46/10.63, 7: 9.59/10.64, 8: 9.91/10.65, 9: 10.15/10.65, 10: 10.28/10.65, 11: 10.44/10.63, 12: 10.33/10.64, 13: 10.23/10.64, 14: 10.09/10.63, 15: 10.03/10.64, 16: 9.55/10.64, 17: 9.56/10.63, 18: 9.54/10.64, 19: 9.80/10.64, 20: 9.55/10.64, 21: 9.63/10.65, 22: 9.68/10.64, 23: 9.91/10.63, 24: 9.78/10.64, 25: 9.93/10.62, 26: 10.05/10.66, 27: 10.02/10.66

Sizes of the related tokens' KV: raw, coded alone, coded as deltas (GB); and D's pages that P recomputed (it sent them again; D held them):

- docs_first / conversation: 2.56 raw, 1.71 alone, 1.37 as deltas; held pages 8,764, of them 0 differ
- history_first / conversation: 1.98 raw, 1.32 alone, 1.10 as deltas; held pages 18,592, of them 0 differ


## Delta analysis of the stored pages

28 layers, 8 KV heads of 128; RoPE on; 7.2 GB of P's blocks, 0.00 GB of D's.

Bits per value (raw BF16: 16). alone: sign+exponent entropy + 7 mantissa bits (no base). prev: given the same token's previous layer (a stand-in for our codec's context). xor, diff: against the base, bit length coded. planes0 / planes: a bit-plane context coder without / with the base. base: sign+exponent given the base's, mantissa given the base's when they match. both: base and previous layer together. Keys: the base turned by RoPE to the new place. Cross-validated entropies.

## mtrag_raw_docs_first_conversation

30 requests, 40,390 prompt tokens; prefix any earlier request shares: 5,072 (13%); related beyond it: 18,672 (46%), by kind {'other': 706, 'doc': 3459, 'answer': 13073, 'question': 1434}.

Lossless, against the first copy (12,000 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.74 | 10.73 | 9.40 | 8.74 | 10.94 | 8.65 | 8.25 | 8.26 | 3% | 65% |
| V mean | 10.65 | 10.65 | 9.98 | 9.64 | 10.74 | 9.21 | 8.90 | 8.91 | 2% | 55% |
| K 0 | 10.63 | - | 1.87 | 1.49 | 10.81 | 1.73 | 1.49 | - | 72% | 100% |
| K 1 | 10.64 | 10.61 | 6.66 | 6.03 | 10.83 | 6.39 | 5.92 | 5.93 | 11% | 89% |
| K 2 | 10.72 | 10.70 | 7.74 | 7.11 | 10.93 | 7.39 | 6.94 | 6.95 | 6% | 81% |
| K 3 | 10.74 | 10.73 | 8.41 | 7.80 | 10.96 | 7.96 | 7.52 | 7.52 | 4% | 75% |
| K 4 | 10.72 | 10.69 | 8.66 | 8.04 | 10.93 | 8.14 | 7.71 | 7.72 | 4% | 73% |
| K 8 | 10.75 | 10.74 | 9.68 | 9.01 | 10.94 | 8.87 | 8.49 | 8.50 | 3% | 62% |
| K 12 | 10.75 | 10.74 | 10.34 | 9.61 | 10.94 | 9.29 | 8.93 | 8.94 | 2% | 55% |
| K 16 | 10.72 | 10.72 | 9.65 | 8.97 | 10.93 | 8.82 | 8.43 | 8.43 | 3% | 63% |
| K 20 | 10.76 | 10.76 | 9.54 | 8.90 | 10.96 | 8.78 | 8.39 | 8.39 | 3% | 63% |
| K 24 | 10.75 | 10.75 | 9.57 | 8.88 | 10.96 | 8.79 | 8.39 | 8.40 | 3% | 64% |
| K 27 | 10.71 | 10.70 | 9.42 | 8.77 | 10.93 | 8.68 | 8.28 | 8.28 | 3% | 65% |
| V 0 | 10.91 | - | 0.00 | 0.00 | 11.01 | 0.00 | 0.00 | - | 100% | 100% |
| V 1 | 10.70 | 10.70 | 7.94 | 7.38 | 10.88 | 7.60 | 7.16 | 7.16 | 5% | 79% |
| V 2 | 10.68 | 10.68 | 8.48 | 7.95 | 10.76 | 8.06 | 7.66 | 7.66 | 4% | 74% |
| V 3 | 10.63 | 10.64 | 9.02 | 8.55 | 10.72 | 8.52 | 8.15 | 8.15 | 3% | 67% |
| V 4 | 10.63 | 10.63 | 9.46 | 9.04 | 10.72 | 8.86 | 8.51 | 8.52 | 2% | 62% |
| V 8 | 10.65 | 10.65 | 10.39 | 10.08 | 10.74 | 9.51 | 9.23 | 9.23 | 1% | 50% |
| V 12 | 10.64 | 10.64 | 10.73 | 10.47 | 10.73 | 9.71 | 9.45 | 9.46 | 1% | 46% |
| V 16 | 10.64 | 10.64 | 9.98 | 9.64 | 10.74 | 9.22 | 8.91 | 8.92 | 2% | 55% |
| V 20 | 10.65 | 10.65 | 9.92 | 9.58 | 10.75 | 9.18 | 8.87 | 8.87 | 2% | 56% |
| V 24 | 10.65 | 10.65 | 9.98 | 9.66 | 10.72 | 9.23 | 8.92 | 8.93 | 2% | 55% |
| V 27 | 10.67 | 10.67 | 10.16 | 9.85 | 10.75 | 9.37 | 9.08 | 9.08 | 2% | 53% |

Lossless, against the latest copy (12,000 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.74 | 10.73 | 9.23 | 8.57 | 10.94 | 8.51 | 8.12 | 8.12 | 4% | 67% |
| V mean | 10.65 | 10.65 | 9.85 | 9.50 | 10.74 | 9.11 | 8.80 | 8.81 | 2% | 57% |
| K 0 | 10.62 | - | 1.83 | 1.46 | 10.81 | 1.70 | 1.46 | - | 73% | 100% |
| K 1 | 10.64 | 10.61 | 6.50 | 5.87 | 10.83 | 6.24 | 5.77 | 5.78 | 12% | 90% |
| K 2 | 10.72 | 10.70 | 7.52 | 6.89 | 10.93 | 7.19 | 6.74 | 6.74 | 7% | 83% |
| K 3 | 10.74 | 10.73 | 8.19 | 7.59 | 10.96 | 7.78 | 7.33 | 7.34 | 5% | 77% |
| K 4 | 10.72 | 10.69 | 8.46 | 7.84 | 10.93 | 7.98 | 7.55 | 7.56 | 5% | 75% |
| K 8 | 10.75 | 10.74 | 9.47 | 8.81 | 10.94 | 8.72 | 8.33 | 8.34 | 3% | 64% |
| K 12 | 10.75 | 10.75 | 10.14 | 9.42 | 10.94 | 9.16 | 8.80 | 8.81 | 3% | 58% |
| K 16 | 10.72 | 10.72 | 9.47 | 8.81 | 10.93 | 8.70 | 8.31 | 8.31 | 3% | 65% |
| K 20 | 10.76 | 10.76 | 9.39 | 8.76 | 10.95 | 8.66 | 8.27 | 8.28 | 4% | 65% |
| K 24 | 10.76 | 10.75 | 9.45 | 8.77 | 10.96 | 8.69 | 8.30 | 8.30 | 3% | 65% |
| K 27 | 10.71 | 10.70 | 9.31 | 8.67 | 10.94 | 8.59 | 8.20 | 8.20 | 4% | 66% |
| V 0 | 10.91 | - | 0.00 | 0.00 | 11.00 | 0.00 | 0.00 | - | 100% | 100% |
| V 1 | 10.70 | 10.70 | 7.79 | 7.23 | 10.88 | 7.47 | 7.03 | 7.04 | 6% | 80% |
| V 2 | 10.68 | 10.68 | 8.29 | 7.76 | 10.76 | 7.90 | 7.49 | 7.50 | 4% | 76% |
| V 3 | 10.63 | 10.64 | 8.84 | 8.36 | 10.72 | 8.37 | 7.99 | 8.00 | 3% | 69% |
| V 4 | 10.63 | 10.63 | 9.27 | 8.83 | 10.72 | 8.72 | 8.36 | 8.37 | 3% | 64% |
| V 8 | 10.65 | 10.65 | 10.25 | 9.93 | 10.74 | 9.42 | 9.12 | 9.13 | 2% | 52% |
| V 12 | 10.64 | 10.64 | 10.58 | 10.30 | 10.73 | 9.62 | 9.35 | 9.36 | 2% | 48% |
| V 16 | 10.64 | 10.64 | 9.87 | 9.52 | 10.74 | 9.14 | 8.82 | 8.83 | 2% | 57% |
| V 20 | 10.65 | 10.65 | 9.83 | 9.48 | 10.74 | 9.11 | 8.79 | 8.80 | 2% | 57% |
| V 24 | 10.65 | 10.65 | 9.88 | 9.56 | 10.72 | 9.16 | 8.86 | 8.86 | 2% | 56% |
| V 27 | 10.67 | 10.67 | 10.06 | 9.74 | 10.75 | 9.29 | 9.00 | 9.00 | 2% | 54% |

Relative deviation |target - base| / |target| by place in the related run (tokens since it started), layers 1 / 4 / 12 / 27:

- K 0-16: 0.04 / 0.14 / 0.39 / 0.21 (3,586 tokens)
- K 16-64: 0.03 / 0.10 / 0.30 / 0.15 (4,297 tokens)
- K 64-256: 0.03 / 0.07 / 0.20 / 0.11 (3,831 tokens)
- K 256-: 0.01 / 0.03 / 0.12 / 0.05 (286 tokens)
- V 0-16: 0.10 / 0.32 / 0.63 / 0.44 (3,586 tokens)
- V 16-64: 0.07 / 0.21 / 0.46 / 0.33 (4,297 tokens)
- V 64-256: 0.06 / 0.15 / 0.29 / 0.23 (3,831 tokens)
- V 256-: 0.02 / 0.07 / 0.19 / 0.13 (286 tokens)

Whole related chunks (48, up to 512 tokens each), per layer (mean): relative size of the difference; the energy rank 4 / 16 / 64 holds; tokens for 50% / 90% of it; the top 15% tokens' share; overlap of the top 15% with layer 2's; the largest head's share:

| | rel | rank 4 / 16 / 64 | tokens for 50% / 90% | top 15% share | overlap with layer 2 | top head |
| --- | --- | --- | --- | --- | --- | --- |
| K 1 | 0.04 | 60% / 80% / 96% | 20% / 70% | 46% | 61% | 21% |
| K 2 | 0.07 | 63% / 80% / 95% | 15% / 64% | 54% | 100% | 22% |
| K 4 | 0.10 | 58% / 79% / 96% | 18% / 66% | 48% | 58% | 20% |
| K 8 | 0.17 | 56% / 80% / 97% | 16% / 66% | 51% | 45% | 19% |
| K 12 | 0.27 | 61% / 83% / 98% | 20% / 70% | 44% | 38% | 17% |
| K 16 | 0.18 | 56% / 79% / 97% | 19% / 68% | 47% | 37% | 18% |
| K 20 | 0.16 | 57% / 80% / 97% | 20% / 68% | 46% | 35% | 19% |
| K 24 | 0.16 | 54% / 78% / 96% | 21% / 68% | 45% | 36% | 20% |
| K 27 | 0.14 | 49% / 74% / 95% | 21% / 69% | 43% | 36% | 18% |
| V 1 | 0.08 | 56% / 81% / 97% | 15% / 65% | 53% | 61% | 23% |
| V 2 | 0.12 | 56% / 79% / 96% | 12% / 63% | 56% | 100% | 35% |
| V 4 | 0.21 | 48% / 74% / 95% | 15% / 63% | 52% | 60% | 20% |
| V 8 | 0.36 | 44% / 73% / 96% | 15% / 62% | 52% | 47% | 23% |
| V 12 | 0.42 | 41% / 70% / 95% | 16% / 64% | 50% | 40% | 17% |
| V 16 | 0.28 | 39% / 68% / 94% | 16% / 63% | 51% | 39% | 26% |
| V 20 | 0.28 | 38% / 66% / 93% | 16% / 62% | 52% | 37% | 22% |
| V 24 | 0.27 | 37% / 65% / 93% | 17% / 62% | 50% | 36% | 18% |
| V 27 | 0.29 | 40% / 68% / 94% | 20% / 68% | 45% | 34% | 23% |

## mtrag_raw_history_first_conversation

30 requests, 40,630 prompt tokens; prefix any earlier request shares: 10,688 (26%); related beyond it: 10,259 (25%), by kind {'other': 151, 'doc': 8178, 'answer': 1859, 'question': 71}.

Lossless, against the first copy (10,259 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.73 | 10.72 | 9.36 | 8.70 | 10.94 | 8.65 | 8.24 | 8.24 | 4% | 65% |
| V mean | 10.64 | 10.64 | 9.83 | 9.47 | 10.73 | 9.12 | 8.80 | 8.81 | 3% | 57% |
| K 0 | 10.63 | - | 1.91 | 1.52 | 10.82 | 1.77 | 1.52 | - | 71% | 100% |
| K 1 | 10.64 | 10.61 | 6.88 | 6.20 | 10.83 | 6.50 | 6.02 | 6.03 | 9% | 88% |
| K 2 | 10.72 | 10.70 | 7.80 | 7.15 | 10.93 | 7.41 | 6.95 | 6.95 | 6% | 81% |
| K 3 | 10.73 | 10.73 | 8.37 | 7.77 | 10.95 | 7.92 | 7.48 | 7.49 | 5% | 76% |
| K 4 | 10.71 | 10.69 | 8.56 | 7.94 | 10.92 | 8.07 | 7.63 | 7.64 | 5% | 74% |
| K 8 | 10.74 | 10.73 | 9.56 | 8.91 | 10.93 | 8.82 | 8.42 | 8.43 | 3% | 63% |
| K 12 | 10.75 | 10.75 | 10.23 | 9.52 | 10.95 | 9.26 | 8.89 | 8.89 | 3% | 56% |
| K 16 | 10.72 | 10.72 | 9.55 | 8.87 | 10.93 | 8.80 | 8.40 | 8.40 | 3% | 64% |
| K 20 | 10.76 | 10.76 | 9.49 | 8.85 | 10.95 | 8.79 | 8.38 | 8.39 | 3% | 64% |
| K 24 | 10.75 | 10.75 | 9.62 | 8.95 | 10.96 | 8.87 | 8.46 | 8.47 | 3% | 63% |
| K 27 | 10.71 | 10.70 | 9.53 | 8.88 | 10.94 | 8.79 | 8.38 | 8.39 | 3% | 63% |
| V 0 | 10.90 | - | 0.22 | 0.20 | 11.00 | 0.20 | 0.19 | - | 98% | 100% |
| V 1 | 10.71 | 10.71 | 7.77 | 7.21 | 10.89 | 7.43 | 7.00 | 7.00 | 6% | 80% |
| V 2 | 10.68 | 10.68 | 8.34 | 7.81 | 10.76 | 7.93 | 7.53 | 7.54 | 5% | 75% |
| V 3 | 10.63 | 10.63 | 8.85 | 8.37 | 10.72 | 8.38 | 8.01 | 8.01 | 4% | 69% |
| V 4 | 10.63 | 10.63 | 9.23 | 8.78 | 10.71 | 8.68 | 8.32 | 8.33 | 3% | 65% |
| V 8 | 10.65 | 10.65 | 10.14 | 9.80 | 10.73 | 9.36 | 9.05 | 9.06 | 2% | 53% |
| V 12 | 10.64 | 10.64 | 10.50 | 10.21 | 10.73 | 9.60 | 9.31 | 9.32 | 2% | 48% |
| V 16 | 10.64 | 10.64 | 9.79 | 9.42 | 10.74 | 9.12 | 8.79 | 8.80 | 3% | 57% |
| V 20 | 10.64 | 10.64 | 9.77 | 9.41 | 10.73 | 9.10 | 8.78 | 8.79 | 3% | 58% |
| V 24 | 10.64 | 10.64 | 9.98 | 9.65 | 10.71 | 9.25 | 8.93 | 8.94 | 3% | 55% |
| V 27 | 10.66 | 10.66 | 10.18 | 9.87 | 10.74 | 9.40 | 9.10 | 9.11 | 2% | 52% |

Lossless, against the latest copy (9,898 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.73 | 10.72 | 8.71 | 8.08 | 10.94 | 8.14 | 7.75 | 7.75 | 6% | 71% |
| V mean | 10.64 | 10.64 | 9.34 | 8.96 | 10.73 | 8.75 | 8.43 | 8.44 | 4% | 63% |
| K 0 | 10.63 | - | 1.90 | 1.51 | 10.82 | 1.77 | 1.52 | - | 72% | 100% |
| K 1 | 10.64 | 10.61 | 6.04 | 5.41 | 10.83 | 5.78 | 5.31 | 5.31 | 15% | 92% |
| K 2 | 10.72 | 10.70 | 7.04 | 6.42 | 10.93 | 6.75 | 6.30 | 6.31 | 9% | 86% |
| K 3 | 10.73 | 10.73 | 7.64 | 7.06 | 10.95 | 7.31 | 6.88 | 6.88 | 8% | 81% |
| K 4 | 10.71 | 10.69 | 7.87 | 7.28 | 10.92 | 7.50 | 7.08 | 7.08 | 8% | 79% |
| K 8 | 10.74 | 10.73 | 8.88 | 8.26 | 10.94 | 8.30 | 7.92 | 7.92 | 5% | 70% |
| K 12 | 10.75 | 10.75 | 9.56 | 8.89 | 10.94 | 8.78 | 8.42 | 8.43 | 4% | 64% |
| K 16 | 10.72 | 10.72 | 8.95 | 8.31 | 10.93 | 8.33 | 7.94 | 7.95 | 5% | 70% |
| K 20 | 10.76 | 10.76 | 8.90 | 8.29 | 10.95 | 8.32 | 7.93 | 7.94 | 5% | 70% |
| K 24 | 10.75 | 10.75 | 9.00 | 8.36 | 10.96 | 8.39 | 8.00 | 8.00 | 5% | 69% |
| K 27 | 10.71 | 10.70 | 8.88 | 8.26 | 10.93 | 8.29 | 7.90 | 7.91 | 5% | 70% |
| V 0 | 10.91 | - | 0.20 | 0.18 | 11.00 | 0.19 | 0.18 | - | 98% | 100% |
| V 1 | 10.71 | 10.71 | 7.23 | 6.67 | 10.89 | 6.97 | 6.53 | 6.54 | 8% | 84% |
| V 2 | 10.68 | 10.68 | 7.76 | 7.22 | 10.76 | 7.44 | 7.03 | 7.04 | 6% | 80% |
| V 3 | 10.63 | 10.63 | 8.30 | 7.80 | 10.72 | 7.93 | 7.55 | 7.56 | 5% | 75% |
| V 4 | 10.63 | 10.63 | 8.71 | 8.24 | 10.71 | 8.27 | 7.91 | 7.91 | 4% | 71% |
| V 8 | 10.65 | 10.65 | 9.64 | 9.27 | 10.73 | 8.99 | 8.69 | 8.70 | 3% | 60% |
| V 12 | 10.64 | 10.64 | 10.03 | 9.70 | 10.73 | 9.26 | 8.98 | 8.99 | 3% | 55% |
| V 16 | 10.64 | 10.64 | 9.34 | 8.97 | 10.74 | 8.77 | 8.45 | 8.46 | 4% | 63% |
| V 20 | 10.64 | 10.64 | 9.33 | 8.96 | 10.73 | 8.77 | 8.44 | 8.45 | 4% | 63% |
| V 24 | 10.64 | 10.64 | 9.47 | 9.12 | 10.71 | 8.87 | 8.56 | 8.57 | 4% | 61% |
| V 27 | 10.66 | 10.66 | 9.64 | 9.30 | 10.74 | 9.00 | 8.70 | 8.71 | 3% | 59% |

Relative deviation |target - base| / |target| by place in the related run (tokens since it started), layers 1 / 4 / 12 / 27:

- K 0-16: 0.05 / 0.14 / 0.35 / 0.20 (2,048 tokens)
- K 16-64: 0.04 / 0.13 / 0.32 / 0.19 (2,380 tokens)
- K 64-256: 0.03 / 0.08 / 0.22 / 0.14 (3,854 tokens)
- K 256-: 0.03 / 0.04 / 0.16 / 0.09 (1,977 tokens)
- V 0-16: 0.10 / 0.28 / 0.56 / 0.41 (2,048 tokens)
- V 16-64: 0.08 / 0.24 / 0.48 / 0.39 (2,380 tokens)
- V 64-256: 0.05 / 0.16 / 0.33 / 0.28 (3,854 tokens)
- V 256-: 0.04 / 0.09 / 0.22 / 0.19 (1,977 tokens)

Whole related chunks (40, up to 512 tokens each), per layer (mean): relative size of the difference; the energy rank 4 / 16 / 64 holds; tokens for 50% / 90% of it; the top 15% tokens' share; overlap of the top 15% with layer 2's; the largest head's share:

| | rel | rank 4 / 16 / 64 | tokens for 50% / 90% | top 15% share | overlap with layer 2 | top head |
| --- | --- | --- | --- | --- | --- | --- |
| K 1 | 0.04 | 68% / 85% / 97% | 17% / 66% | 49% | 66% | 23% |
| K 2 | 0.08 | 68% / 84% / 96% | 14% / 62% | 55% | 100% | 21% |
| K 4 | 0.12 | 62% / 81% / 96% | 16% / 59% | 52% | 66% | 20% |
| K 8 | 0.21 | 57% / 79% / 96% | 15% / 59% | 54% | 54% | 18% |
| K 12 | 0.30 | 57% / 80% / 96% | 18% / 64% | 48% | 48% | 16% |
| K 16 | 0.21 | 52% / 76% / 95% | 17% / 62% | 51% | 47% | 16% |
| K 20 | 0.20 | 54% / 77% / 95% | 17% / 62% | 50% | 48% | 17% |
| K 24 | 0.20 | 53% / 76% / 94% | 18% / 63% | 48% | 47% | 17% |
| K 27 | 0.18 | 48% / 72% / 93% | 20% / 66% | 45% | 43% | 17% |
| V 1 | 0.08 | 56% / 81% / 96% | 10% / 56% | 60% | 65% | 24% |
| V 2 | 0.13 | 55% / 80% / 96% | 11% / 57% | 59% | 100% | 35% |
| V 4 | 0.24 | 47% / 74% / 94% | 14% / 56% | 55% | 64% | 20% |
| V 8 | 0.41 | 42% / 70% / 94% | 14% / 56% | 55% | 53% | 23% |
| V 12 | 0.47 | 36% / 65% / 92% | 16% / 58% | 53% | 46% | 18% |
| V 16 | 0.34 | 37% / 64% / 92% | 15% / 56% | 55% | 45% | 23% |
| V 20 | 0.35 | 39% / 65% / 92% | 14% / 54% | 57% | 44% | 22% |
| V 24 | 0.36 | 40% / 65% / 92% | 17% / 59% | 50% | 40% | 22% |
| V 27 | 0.37 | 37% / 63% / 91% | 19% / 66% | 46% | 40% | 21% |




# The conversations' block tree, kept by a rule

Pages of 16 tokens x 28 layers (1,835,008 B raw). The requests replayed in arrival order; a page is identified by its bytes (the capture's rows). Hits: blocks found in the store from the request's first block on (what a prefix cache reuses); the rest are computed. Peak: the store at its largest, coded with T2 per run of contiguous kept pages (layer 0 recomputed), decoded back and checked byte for byte. Bits: per value (16 raw).

Prefill in the capture's run: 0.087 ms per token at the median over 60 requests (arrival to the end of P's prefill, queueing included; P had no prefix cache), so 1.4 ms to recompute a page.

Recompute is exact: D compared the pages it held with P's recompute of the same prompt prefix: docs_first / conversation: 8,764 pages of one layer (313 blocks), 0 differing; history_first / conversation: 18,592 pages of one layer (664 blocks), 0 differing.

## mtrag_raw_docs_first_conversation

30 requests, 2,195 distinct pages. P computed 317 blocks again under a token prefix computed before: 317 with the same bytes, 0 not. Blocks with an earlier block's bytes under another token prefix in the run's records: 0.

| Rule | Hits (blocks reused) | Prefill left (tokens) | Prefill left (s) | Peak pages | Peak raw GB | Peak t2-hbh GB (bits) | Peak t2-rank256 GB (bits) | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Everything (every distinct page) | 317 of 2,512 | 35,318 of 40,390 | 3.1 | 2,195 | 4.03 | 2.67 (10.61) | 2.91 (11.54) | 30 of 30 runs; 30 of 30 runs |
| Newest path per conversation | 282 of 2,512 | 35,878 of 40,390 | 3.1 | 417 | 0.77 | 0.58 (12.13) | 0.63 (13.07) | 8 of 8 runs; 8 of 8 runs |
| Newest path, no document pages | 41 of 2,512 | 39,734 of 40,390 | 3.4 | 229 | 0.42 | 0.46 (17.61) | 0.49 (18.64) | 9 of 9 runs; 9 of 9 runs |
| Hindsight (kept until its last use) | 317 of 2,512 | 35,318 of 40,390 | 3.1 | 74 | 0.14 | 0.16 (18.66) | 0.17 (19.67) | 4 of 4 runs; 4 of 4 runs |
| Nothing kept | 0 of 2,512 | 40,390 of 40,390 | 3.5 | 0 | 0.00 | - | - | - |

Padding, on the first requests of 256 tokens or more (bits per real value; the padding's bytes counted):

| Codec | Requests | As now: zeros to 256 | No padding (the 256-multiple below) | Tail apart, zeros to 32 | Tail apart, copies to 32 | Copies of the last token to 256 | Padding tokens now / split | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| t2-hbh | 24 | 10.32 | 8.32 | 8.46 | 8.40 | 9.53 | 2,976 / 224 | 24 of 24 |
| t2-rank256 | 24 | 11.25 | 9.22 | 9.37 | 9.29 | 10.33 | 2,976 / 224 | 24 of 24 |

## mtrag_raw_history_first_conversation

30 requests, 1,858 distinct pages. P computed 668 blocks again under a token prefix computed before: 668 with the same bytes, 0 not. Blocks with an earlier block's bytes under another token prefix in the run's records: 0.

| Rule | Hits (blocks reused) | Prefill left (tokens) | Prefill left (s) | Peak pages | Peak raw GB | Peak t2-hbh GB (bits) | Peak t2-rank256 GB (bits) | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Everything (every distinct page) | 668 of 2,526 | 29,942 of 40,630 | 2.6 | 1,858 | 3.41 | 2.36 (11.06) | 2.55 (11.98) | 30 of 30 runs; 30 of 30 runs |
| Newest path per conversation | 668 of 2,526 | 29,942 of 40,630 | 2.6 | 421 | 0.77 | 0.69 (14.19) | 0.73 (15.15) | 17 of 17 runs; 17 of 17 runs |
| Newest path, no document pages | 668 of 2,526 | 29,942 of 40,630 | 2.6 | 234 | 0.43 | 0.68 (25.27) | 0.71 (26.38) | 27 of 27 runs; 27 of 27 runs |
| Hindsight (kept until its last use) | 668 of 2,526 | 29,942 of 40,630 | 2.6 | 168 | 0.31 | 0.51 (26.48) | 0.53 (27.59) | 20 of 20 runs; 20 of 20 runs |
| Nothing kept | 0 of 2,526 | 40,630 of 40,630 | 3.5 | 0 | 0.00 | - | - | - |

Padding, on the first requests of 256 tokens or more (bits per real value; the padding's bytes counted):

| Codec | Requests | As now: zeros to 256 | No padding (the 256-multiple below) | Tail apart, zeros to 32 | Tail apart, copies to 32 | Copies of the last token to 256 | Padding tokens now / split | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| t2-hbh | 24 | 10.30 | 8.31 | 8.47 | 8.40 | 9.52 | 2,896 / 240 | 24 of 24 |
| t2-rank256 | 24 | 11.22 | 9.21 | 9.38 | 9.30 | 10.31 | 2,896 / 240 | 24 of 24 |

## Times (CUDA-synchronised, one GPU)

| Step | n | ms per page | mean ms per call |
| --- | ---: | ---: | ---: |
| t2-hbh kept decode | 96 | 0.991 | 45.55 |
| t2-hbh kept encode | 96 | 0.428 | 19.68 |
| t2-hbh probe decode | 236 | - | 32.93 |
| t2-hbh probe encode | 236 | - | 13.64 |
| t2-rank256 kept decode | 96 | 0.189 | 8.66 |
| t2-rank256 kept encode | 96 | 0.262 | 12.02 |
| t2-rank256 probe decode | 236 | - | 4.45 |
| t2-rank256 probe encode | 236 | - | 11.83 |
| recompute (the run's prefill) | - | 1.384 | - |

# T plus the base against T alone, on pages with an earlier copy

Layers 1-27 of every distinct page with a base (layer 0 recomputed, as T2 does), coded with T's value-slice coder against each prediction, decoded and checked. Fitted on the even conversations. Bits per value (16 raw) over all 28 layers, layer 0 at 0 bits (as pd_diffgraph.py's rows), held out (the odd conversations) and on the training pages; the K and V columns over their own values. Base: the latest earlier copy of the 16 tokens (keys turned), else the latest earlier turn's page at the same place.

## t2-hbh mtrag_raw_docs_first_conversation

1,842 pages with a base (position 1,010, text 832); held out 962, training 880. T's constants: set wikitext (the smallest on the first training pages).

| Prediction | Held out | Training | K 1-4 | K 5-13 | K 14-27 | V 1-4 | V 5-13 | V 14-27 | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T, its own spread (as now) | 8.32 | 8.23 | 8.00 | 8.30 | 8.11 | 9.10 | 9.04 | 9.12 | 43 of 43 calls |
| T, fitted spread | 8.31 | 8.22 | 7.98 | 8.30 | 8.10 | 9.10 | 9.04 | 9.11 | 43 of 43 calls |
| Base (the diff graph's delta) | 9.68 | 8.93 | 8.89 | 9.79 | 9.77 | 9.85 | 10.53 | 10.52 | 43 of 43 calls |
| Base, spread with T's prior | 9.70 | 8.95 | 8.98 | 9.80 | 9.79 | 9.95 | 10.54 | 10.54 | 43 of 43 calls |
| Blend: T + w (base - T) | 8.18 | 7.79 | 7.56 | 8.20 | 8.08 | 8.54 | 8.94 | 9.02 | 43 of 43 calls |

Blend against T as now, held out: +1.6% (smaller).

Blend weights w (0: T alone, 1: the base alone), least squares per layer group: K 1-4 0.08, V 1-4 0.08, K 5-13 0.06, V 5-13 0.06, K 14-27 0.05, V 14-27 0.07. By agreement bucket (tokens the text has agreed with the base's: 0-15 / 16-63 / 64-255 / 256+ / the same place, other text), all layers: K 0.19 / 0.22 / 0.46 / 0.77 / 0.01; V 0.23 / 0.26 / 0.49 / 0.81 / 0.01.

Check, 6 whole requests (4,864 tokens, no padding): the codec's own size 8.35 bits per value (layer 0 recomputed), T's prediction through this coder 8.34 (the same layers); exact 6 of 6.

## t2-hbh mtrag_raw_history_first_conversation

1,417 pages with a base (position 904, text 513); held out 595, training 822. T's constants: set wikitext (the smallest on the first training pages).

| Prediction | Held out | Training | K 1-4 | K 5-13 | K 14-27 | V 1-4 | V 5-13 | V 14-27 | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T, its own spread (as now) | 8.31 | 8.25 | 8.00 | 8.27 | 8.13 | 9.12 | 9.02 | 9.12 | 40 of 40 calls |
| T, fitted spread | 8.31 | 8.24 | 7.99 | 8.27 | 8.12 | 9.12 | 9.01 | 9.11 | 40 of 40 calls |
| Base (the diff graph's delta) | 10.36 | 8.75 | 9.80 | 10.50 | 10.42 | 10.82 | 11.19 | 11.18 | 40 of 40 calls |
| Base, spread with T's prior | 10.39 | 8.78 | 9.90 | 10.51 | 10.44 | 10.93 | 11.21 | 11.21 | 40 of 40 calls |
| Blend: T + w (base - T) | 8.41 | 7.67 | 7.96 | 8.42 | 8.27 | 8.99 | 9.13 | 9.24 | 40 of 40 calls |

Blend against T as now, held out: -1.2% (larger).

Blend weights w (0: T alone, 1: the base alone), least squares per layer group: K 1-4 0.09, V 1-4 0.08, K 5-13 0.06, V 5-13 0.06, K 14-27 0.05, V 14-27 0.08. By agreement bucket (tokens the text has agreed with the base's: 0-15 / 16-63 / 64-255 / 256+ / the same place, other text), all layers: K 0.23 / 0.35 / 0.57 / 0.84 / 0.01; V 0.26 / 0.39 / 0.60 / 0.84 / 0.02.

Check, 6 whole requests (4,352 tokens, no padding): the codec's own size 8.33 bits per value (layer 0 recomputed), T's prediction through this coder 8.33 (the same layers); exact 6 of 6.

## t2-rank256 mtrag_raw_docs_first_conversation

1,842 pages with a base (position 1,010, text 832); held out 962, training 880. T's constants: set wikitext (the smallest on the first training pages).

| Prediction | Held out | Training | K 1-4 | K 5-13 | K 14-27 | V 1-4 | V 5-13 | V 14-27 | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T, its own spread (as now) | 9.23 | 9.17 | 8.84 | 8.90 | 8.76 | 10.32 | 10.25 | 10.37 | 43 of 43 calls |
| T, fitted spread | 9.22 | 9.16 | 8.81 | 8.90 | 8.75 | 10.31 | 10.24 | 10.36 | 43 of 43 calls |
| Base (the diff graph's delta) | 9.68 | 8.93 | 8.89 | 9.79 | 9.77 | 9.85 | 10.53 | 10.52 | 43 of 43 calls |
| Base, spread with T's prior | 9.68 | 8.93 | 8.91 | 9.79 | 9.78 | 9.87 | 10.53 | 10.53 | 43 of 43 calls |
| Blend: T + w (base - T) | 8.86 | 8.32 | 8.04 | 8.68 | 8.63 | 9.27 | 9.88 | 9.96 | 43 of 43 calls |

Blend against T as now, held out: +3.9% (smaller).

Blend weights w (0: T alone, 1: the base alone), least squares per layer group: K 1-4 0.19, V 1-4 0.31, K 5-13 0.11, V 5-13 0.25, K 14-27 0.10, V 14-27 0.32. By agreement bucket (tokens the text has agreed with the base's: 0-15 / 16-63 / 64-255 / 256+ / the same place, other text), all layers: K 0.35 / 0.39 / 0.67 / 0.89 / 0.02; V 0.67 / 0.70 / 0.87 / 0.97 / 0.06.

Check, 6 whole requests (4,864 tokens, no padding): the codec's own size 9.24 bits per value (layer 0 recomputed), T's prediction through this coder 9.26 (the same layers); exact 6 of 6.

## t2-rank256 mtrag_raw_history_first_conversation

1,417 pages with a base (position 904, text 513); held out 595, training 822. T's constants: set wikitext (the smallest on the first training pages).

| Prediction | Held out | Training | K 1-4 | K 5-13 | K 14-27 | V 1-4 | V 5-13 | V 14-27 | Decoded exactly |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T, its own spread (as now) | 9.22 | 9.18 | 8.83 | 8.87 | 8.77 | 10.32 | 10.23 | 10.36 | 40 of 40 calls |
| T, fitted spread | 9.21 | 9.17 | 8.81 | 8.87 | 8.76 | 10.32 | 10.22 | 10.35 | 40 of 40 calls |
| Base (the diff graph's delta) | 10.36 | 8.75 | 9.80 | 10.50 | 10.42 | 10.82 | 11.19 | 11.18 | 40 of 40 calls |
| Base, spread with T's prior | 10.37 | 8.76 | 9.82 | 10.51 | 10.43 | 10.84 | 11.20 | 11.19 | 40 of 40 calls |
| Blend: T + w (base - T) | 9.26 | 8.15 | 8.64 | 8.99 | 8.90 | 10.01 | 10.29 | 10.40 | 40 of 40 calls |

Blend against T as now, held out: -0.4% (larger).

Blend weights w (0: T alone, 1: the base alone), least squares per layer group: K 1-4 0.20, V 1-4 0.32, K 5-13 0.12, V 5-13 0.27, K 14-27 0.10, V 14-27 0.33. By agreement bucket (tokens the text has agreed with the base's: 0-15 / 16-63 / 64-255 / 256+ / the same place, other text), all layers: K 0.42 / 0.55 / 0.75 / 0.93 / 0.02; V 0.70 / 0.81 / 0.90 / 0.97 / 0.06.

Check, 6 whole requests (4,352 tokens, no padding): the codec's own size 9.23 bits per value (layer 0 recomputed), T's prediction through this coder 9.25 (the same layers); exact 6 of 6.

## Times (CUDA-synchronised, one GPU; batches of up to 64 pages, one call per layer group and K/V)

| Step | calls | mean ms per call | ms per page | total s |
| --- | ---: | ---: | ---: | ---: |
| base decode | 996 | 3.47 | 0.531 | 3.5 |
| base encode | 996 | 3.09 | 0.472 | 3.1 |
| base_prior decode | 996 | 3.47 | 0.530 | 3.5 |
| base_prior encode | 996 | 3.09 | 0.472 | 3.1 |
| blend decode | 996 | 3.47 | 0.531 | 3.5 |
| blend encode | 996 | 3.09 | 0.472 | 3.1 |
| check T here encode | 48 | 3.47 | - | 0.2 |
| check codec decode | 24 | 218.56 | - | 5.2 |
| check codec encode | 24 | 17.04 | - | 0.4 |
| set choice decode | 240 | 4.54 | - | 1.1 |
| set choice encode | 240 | 4.15 | - | 1.0 |
| t_fit decode | 996 | 3.48 | 0.532 | 3.5 |
| t_fit encode | 996 | 3.09 | 0.472 | 3.1 |
| t_own decode | 996 | 3.72 | 0.569 | 3.7 |
| t_own encode | 996 | 3.35 | 0.512 | 3.3 |
