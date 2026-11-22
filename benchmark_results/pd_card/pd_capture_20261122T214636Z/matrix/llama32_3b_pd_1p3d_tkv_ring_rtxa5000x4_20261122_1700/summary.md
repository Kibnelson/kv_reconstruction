# vLLM PD with our transfer protocol

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130; 1 prefill engine, 3 decode engine(s); prompts prompts_mixed.jsonl, 128 output tokens; TTFT SLO 2000.0 ms (goodput).

## Correctness (every transfer checked bit-exact on the decode GPU)

| Codec | Transfers bit-exact | Outputs equal to local prefill | Bits per value |
| --- | --- | --- | --- |
| raw | 4/4 | 4/4 | 16.000 |

## MTRAG (multi-turn RAG conversations)

| Codec | Layout | Routing | Traced | Requests (failed) | TTFT p50 (ms) | Prompt tokens mean / max | Transfers | GB sent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | docs_first | conversation | yes | 47 (0) | 579 | 1237 / 2424 | 47 | 6.67 |
| raw | history_first | conversation | yes | 47 (0) | 483 | 1250 / 2424 | 47 | 6.74 |

## The reuse opportunity (traced PD run)

Per prompt layout and routing. Moved again: tokens P sent that the chosen D already held. Recomputed: tokens P prefilled without its cache; answers: tokens of earlier answers (made on D). Same bytes: pages P sent again that D held and whose bytes D's equal. Affinity: later turns that went to the D of their previous turn.

| Layout / routing | Requests | Moved again (share of sent, GB) | Recomputed tokens | Earlier answers recomputed | Same bytes: answers / docs / rest | Affinity |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 47 | 9% (0.6 GB) | 58,152 | 20,030 of 20,030 | - / 100% / 100% | 100% |
| history_first / conversation | 47 | 30% (2.1 GB) | 58,746 | 20,624 of 20,624 | 100% / - / 100% | 100% |

## Every page, by what already existed (traced variants)

1 prefix: an earlier request had its token prefix (prefix caches share it). 2 same bytes: not 1, but an earlier page at its layer has its bytes. 3 same text: not 1 or 2, but its 16 tokens were in an earlier prompt (a differential store's candidates). 4 new. Store: what holding every page once would take, keyed by prefix (LMCache), by content, and by content without layer 0 (recomputable from tokens).

| Layout / routing | Pages | 1 prefix | 2 same bytes | 3 same text | 4 new | Store by prefix / content / content w/o layer 0 (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 102,480 | 9.5% | 0.1% | 37.2% | 53.3% | 6.1 / 6.1 / 5.9 |
| history_first / conversation | 103,432 | 30.4% | 0.0% | 17.1% | 52.5% | 4.7 / 4.7 / 4.5 |

Same bytes (2) by layer, and same text (3) by kind of piece:

- docs_first / conversation: same bytes at layer 0: 57; same text in {'doc': 6132, 'answer': 29094, 'other': 2909}
- history_first / conversation: same bytes at layer 0: 8; same text in {'answer': 3640, 'doc': 13964, 'other': 84}

## How close related KV is (the differential opportunity)

Related: prompt tokens whose 16-token window appeared earlier in the run, not as a shared prefix (so a prefix cache misses them), at the same place after another prefix or moved; within the conversation or across. Bits per value of coding each against its first occurrence: xor; diff, the difference of the words as ordered integers; rot, keys turned to the new position first; best, the cheapest per layer; alone, the same tokens without a base (sign and exponent entropy + 7 bits, the exponent coders' floor). Raw BF16 is 16. Measured: a sample of the related tokens, chosen by content (1 in the sample's number), each against its first occurrence when it was still held.

| Layout / routing | Prompt tokens | Related (share) | Docs / answers / questions / other | Same place / moved | Within / across conversations | Median shift |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 58,152 | 47,797 (82.2%) | 6,955 / 37,289 / 2,803 / 750 | 211 / 47,586 | 47,735 / 62 | 240 |
| history_first / conversation | 58,746 | 29,481 (50.2%) | 10,922 / 17,490 / 854 / 215 | 0 / 29,481 | 29,419 / 62 | 216 |

Measured related tokens: docs_first / conversation: 3,063 (1 in 8), 0 requests not measured; history_first / conversation: 2,719 (1 in 8), 0 requests not measured

| Layout / routing | Kind | Measured values | Equal | K: xor / diff / rot / alone | V: xor / diff / alone | Best vs alone (saving) |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | every kind | 175,644,672 | 3.6% | 11.92 / 11.45 / 8.67 / 10.73 | 9.80 / 9.47 / 10.66 | 9.07 vs 10.69 (15.2%) |
| docs_first / conversation | answer | 131,891,200 | 3.5% | 12.03 / 11.55 / 8.87 / 10.73 | 9.99 / 9.69 / 10.66 | 9.28 vs 10.69 (13.2%) |
| docs_first / conversation | doc | 31,195,136 | 3.9% | 11.59 / 11.14 / 8.03 / 10.73 | 9.12 / 8.72 / 10.65 | 8.38 vs 10.69 (21.6%) |
| docs_first / conversation | other | 3,325,952 | 3.3% | 11.76 / 11.34 / 8.64 / 10.74 | 9.89 / 9.54 / 10.66 | 9.09 vs 10.70 (15.0%) |
| docs_first / conversation | question | 9,232,384 | 3.9% | 11.43 / 10.97 / 7.74 / 10.71 | 9.06 / 8.63 / 10.65 | 8.18 vs 10.68 (23.4%) |
| history_first / conversation | every kind | 155,918,336 | 3.2% | 12.13 / 11.65 / 9.15 / 10.73 | 10.19 / 9.91 / 10.65 | 9.53 vs 10.69 (10.9%) |
| history_first / conversation | answer | 93,356,032 | 3.0% | 12.29 / 11.75 / 9.69 / 10.73 | 10.71 / 10.49 / 10.66 | 10.09 vs 10.70 (5.7%) |
| history_first / conversation | doc | 57,171,968 | 3.6% | 11.83 / 11.41 / 8.10 / 10.73 | 9.17 / 8.76 / 10.65 | 8.43 vs 10.69 (21.2%) |
| history_first / conversation | other | 2,236,416 | 2.6% | 12.58 / 12.11 / 10.16 / 10.74 | 11.24 / 11.07 / 10.65 | 10.62 vs 10.69 (0.7%) |
| history_first / conversation | question | 3,153,920 | 2.7% | 12.15 / 11.67 / 8.99 / 10.70 | 10.18 / 9.86 / 10.64 | 9.42 vs 10.67 (11.7%) |

Best bits per value by layer (keys, values), against alone:

- docs_first / conversation, K: 0: 1.51/10.62, 1: 6.16/10.63, 2: 7.18/10.72, 3: 7.96/10.74, 4: 8.25/10.72, 5: 8.42/10.74, 6: 8.73/10.74, 7: 8.88/10.73, 8: 9.19/10.75, 9: 9.45/10.72, 10: 9.59/10.74, 11: 9.51/10.75, 12: 9.82/10.74, 13: 9.71/10.77, 14: 9.46/10.71, 15: 9.44/10.72, 16: 9.17/10.72, 17: 9.12/10.72, 18: 9.11/10.74, 19: 9.23/10.76, 20: 9.10/10.76, 21: 9.08/10.77, 22: 9.08/10.78, 23: 9.10/10.72, 24: 9.08/10.75, 25: 9.20/10.75, 26: 9.17/10.71, 27: 8.98/10.71
- docs_first / conversation, V: 0: 0.00/10.90, 1: 7.39/10.69, 2: 7.94/10.67, 3: 8.65/10.63, 4: 9.22/10.63, 5: 9.41/10.63, 6: 9.76/10.63, 7: 9.93/10.63, 8: 10.25/10.65, 9: 10.45/10.65, 10: 10.57/10.64, 11: 10.75/10.63, 12: 10.65/10.63, 13: 10.53/10.64, 14: 10.37/10.64, 15: 10.30/10.64, 16: 9.87/10.64, 17: 9.81/10.63, 18: 9.80/10.65, 19: 10.05/10.66, 20: 9.81/10.65, 21: 9.80/10.65, 22: 9.84/10.64, 23: 10.10/10.65, 24: 9.87/10.66, 25: 10.01/10.64, 26: 10.07/10.66, 27: 10.07/10.67
- history_first / conversation, K: 0: 1.58/10.62, 1: 6.27/10.63, 2: 7.36/10.72, 3: 8.33/10.74, 4: 8.76/10.71, 5: 8.90/10.74, 6: 9.25/10.74, 7: 9.37/10.73, 8: 9.73/10.75, 9: 10.03/10.73, 10: 10.12/10.74, 11: 10.07/10.75, 12: 10.31/10.75, 13: 10.20/10.78, 14: 9.96/10.71, 15: 9.97/10.73, 16: 9.67/10.72, 17: 9.66/10.72, 18: 9.62/10.75, 19: 9.76/10.76, 20: 9.60/10.76, 21: 9.60/10.77, 22: 9.62/10.78, 23: 9.65/10.72, 24: 9.65/10.75, 25: 9.78/10.75, 26: 9.75/10.71, 27: 9.59/10.71
- history_first / conversation, V: 0: 0.50/10.88, 1: 7.28/10.70, 2: 8.05/10.67, 3: 8.92/10.63, 4: 9.57/10.63, 5: 9.80/10.63, 6: 10.20/10.63, 7: 10.35/10.64, 8: 10.70/10.65, 9: 10.92/10.65, 10: 11.05/10.65, 11: 11.17/10.63, 12: 11.08/10.64, 13: 10.98/10.64, 14: 10.82/10.64, 15: 10.78/10.64, 16: 10.32/10.64, 17: 10.29/10.63, 18: 10.26/10.65, 19: 10.53/10.65, 20: 10.29/10.65, 21: 10.32/10.65, 22: 10.37/10.64, 23: 10.62/10.64, 24: 10.44/10.65, 25: 10.57/10.63, 26: 10.63/10.66, 27: 10.63/10.67

Sizes of the related tokens' KV: raw, coded alone, coded as deltas (GB); and D's pages that P recomputed (it sent them again; D held them):

- docs_first / conversation: 5.48 raw, 3.66 alone, 3.11 as deltas; held pages 9,548, of them 0 differ
- history_first / conversation: 3.38 raw, 2.26 alone, 2.01 as deltas; held pages 31,304, of them 0 differ


## Delta analysis of the stored pages

28 layers, 8 KV heads of 128; RoPE on; 10.3 GB of P's blocks, 0.00 GB of D's.

Bits per value (raw BF16: 16). alone: sign+exponent entropy + 7 mantissa bits (no base). prev: given the same token's previous layer (a stand-in for our codec's context). xor, diff: against the base, bit length coded. planes0 / planes: a bit-plane context coder without / with the base. base: sign+exponent given the base's, mantissa given the base's when they match. both: base and previous layer together. Keys: the base turned by RoPE to the new place. Cross-validated entropies.

## mtrag_raw_docs_first_conversation

47 requests, 58,152 prompt tokens; prefix any earlier request shares: 5,552 (10%); related beyond it: 43,071 (74%), by kind {'other': 1433, 'doc': 4221, 'answer': 34614, 'question': 2803}.

Lossless, against the first copy (12,000 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.73 | 10.72 | 10.18 | 9.46 | 10.93 | 9.17 | 8.77 | 8.78 | 2% | 56% |
| V mean | 10.65 | 10.65 | 10.77 | 10.52 | 10.75 | 9.72 | 9.44 | 9.45 | 2% | 44% |
| K 0 | 10.61 | - | 1.88 | 1.49 | 10.79 | 1.74 | 1.49 | - | 72% | 100% |
| K 1 | 10.63 | 10.60 | 7.60 | 6.95 | 10.83 | 7.17 | 6.73 | 6.73 | 8% | 83% |
| K 2 | 10.73 | 10.70 | 8.70 | 8.02 | 10.94 | 8.13 | 7.71 | 7.71 | 4% | 73% |
| K 3 | 10.75 | 10.74 | 9.12 | 8.50 | 10.98 | 8.51 | 8.07 | 8.07 | 3% | 68% |
| K 4 | 10.72 | 10.69 | 9.34 | 8.71 | 10.94 | 8.65 | 8.22 | 8.23 | 3% | 65% |
| K 8 | 10.74 | 10.73 | 10.49 | 9.75 | 10.93 | 9.39 | 9.02 | 9.02 | 2% | 53% |
| K 12 | 10.74 | 10.74 | 11.12 | 10.36 | 10.93 | 9.76 | 9.42 | 9.42 | 2% | 45% |
| K 16 | 10.71 | 10.71 | 10.39 | 9.67 | 10.92 | 9.30 | 8.91 | 8.92 | 2% | 54% |
| K 20 | 10.76 | 10.75 | 10.29 | 9.59 | 10.95 | 9.28 | 8.88 | 8.88 | 2% | 55% |
| K 24 | 10.74 | 10.74 | 10.29 | 9.53 | 10.93 | 9.26 | 8.87 | 8.87 | 2% | 55% |
| K 27 | 10.71 | 10.70 | 10.13 | 9.44 | 10.93 | 9.16 | 8.76 | 8.76 | 2% | 57% |
| V 0 | 10.86 | - | 0.00 | 0.00 | 10.96 | 0.00 | 0.00 | - | 100% | 100% |
| V 1 | 10.68 | 10.68 | 8.85 | 8.37 | 10.86 | 8.36 | 7.98 | 7.97 | 4% | 69% |
| V 2 | 10.67 | 10.67 | 9.34 | 8.91 | 10.76 | 8.74 | 8.39 | 8.39 | 3% | 63% |
| V 3 | 10.62 | 10.62 | 9.73 | 9.33 | 10.71 | 9.05 | 8.71 | 8.71 | 2% | 58% |
| V 4 | 10.63 | 10.63 | 10.19 | 9.84 | 10.72 | 9.37 | 9.06 | 9.06 | 2% | 52% |
| V 8 | 10.65 | 10.65 | 11.29 | 11.09 | 10.74 | 10.02 | 9.79 | 9.80 | 1% | 38% |
| V 12 | 10.64 | 10.64 | 11.56 | 11.40 | 10.73 | 10.15 | 9.94 | 9.95 | 1% | 35% |
| V 16 | 10.65 | 10.65 | 10.81 | 10.56 | 10.75 | 9.76 | 9.47 | 9.48 | 2% | 44% |
| V 20 | 10.66 | 10.66 | 10.66 | 10.40 | 10.75 | 9.69 | 9.39 | 9.40 | 2% | 45% |
| V 24 | 10.68 | 10.68 | 10.66 | 10.42 | 10.75 | 9.69 | 9.40 | 9.40 | 1% | 45% |
| V 27 | 10.68 | 10.68 | 10.81 | 10.59 | 10.76 | 9.79 | 9.53 | 9.53 | 1% | 43% |

Lossless, against the latest copy (12,000 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.73 | 10.72 | 9.70 | 9.01 | 10.93 | 8.82 | 8.42 | 8.43 | 3% | 62% |
| V mean | 10.65 | 10.65 | 10.43 | 10.13 | 10.75 | 9.48 | 9.18 | 9.19 | 2% | 49% |
| K 0 | 10.60 | - | 1.87 | 1.49 | 10.79 | 1.73 | 1.48 | - | 72% | 100% |
| K 1 | 10.63 | 10.60 | 6.74 | 6.09 | 10.83 | 6.41 | 5.94 | 5.94 | 10% | 88% |
| K 2 | 10.73 | 10.70 | 7.75 | 7.11 | 10.94 | 7.35 | 6.90 | 6.91 | 6% | 82% |
| K 3 | 10.75 | 10.74 | 8.30 | 7.68 | 10.97 | 7.84 | 7.38 | 7.39 | 5% | 76% |
| K 4 | 10.72 | 10.69 | 8.58 | 7.96 | 10.93 | 8.07 | 7.62 | 7.63 | 4% | 74% |
| K 8 | 10.74 | 10.73 | 10.07 | 9.35 | 10.93 | 9.11 | 8.73 | 8.73 | 3% | 58% |
| K 12 | 10.74 | 10.74 | 10.80 | 10.05 | 10.93 | 9.55 | 9.21 | 9.22 | 2% | 50% |
| K 16 | 10.71 | 10.71 | 10.07 | 9.36 | 10.92 | 9.08 | 8.69 | 8.70 | 3% | 58% |
| K 20 | 10.76 | 10.75 | 9.93 | 9.24 | 10.95 | 9.02 | 8.62 | 8.62 | 3% | 59% |
| K 24 | 10.74 | 10.74 | 9.88 | 9.16 | 10.93 | 8.99 | 8.60 | 8.60 | 3% | 60% |
| K 27 | 10.71 | 10.70 | 9.68 | 9.01 | 10.93 | 8.85 | 8.44 | 8.45 | 3% | 62% |
| V 0 | 10.86 | - | 0.00 | 0.00 | 10.96 | 0.00 | 0.00 | - | 100% | 100% |
| V 1 | 10.68 | 10.68 | 8.20 | 7.66 | 10.86 | 7.79 | 7.36 | 7.36 | 5% | 76% |
| V 2 | 10.67 | 10.66 | 8.58 | 8.07 | 10.76 | 8.12 | 7.72 | 7.73 | 4% | 72% |
| V 3 | 10.62 | 10.62 | 9.01 | 8.53 | 10.70 | 8.48 | 8.10 | 8.10 | 3% | 67% |
| V 4 | 10.63 | 10.63 | 9.49 | 9.06 | 10.72 | 8.86 | 8.50 | 8.51 | 2% | 61% |
| V 8 | 10.65 | 10.65 | 11.06 | 10.83 | 10.74 | 9.89 | 9.64 | 9.65 | 2% | 41% |
| V 12 | 10.64 | 10.64 | 11.39 | 11.22 | 10.73 | 10.06 | 9.84 | 9.85 | 1% | 37% |
| V 16 | 10.65 | 10.65 | 10.61 | 10.33 | 10.75 | 9.63 | 9.32 | 9.33 | 2% | 46% |
| V 20 | 10.66 | 10.66 | 10.39 | 10.09 | 10.75 | 9.51 | 9.18 | 9.19 | 2% | 49% |
| V 24 | 10.68 | 10.68 | 10.35 | 10.06 | 10.75 | 9.48 | 9.17 | 9.18 | 2% | 49% |
| V 27 | 10.67 | 10.67 | 10.44 | 10.16 | 10.76 | 9.55 | 9.25 | 9.26 | 2% | 48% |

Relative deviation |target - base| / |target| by place in the related run (tokens since it started), layers 1 / 4 / 12 / 27:

- K 0-16: 0.09 / 0.17 / 0.48 / 0.24 (6,250 tokens)
- K 16-64: 0.03 / 0.12 / 0.36 / 0.18 (3,919 tokens)
- K 64-256: 0.03 / 0.08 / 0.23 / 0.12 (1,713 tokens)
- K 256-: 0.01 / 0.03 / 0.12 / 0.05 (118 tokens)
- V 0-16: 0.26 / 0.44 / 0.82 / 0.53 (6,250 tokens)
- V 16-64: 0.07 / 0.25 / 0.55 / 0.39 (3,919 tokens)
- V 64-256: 0.06 / 0.16 / 0.33 / 0.25 (1,713 tokens)
- V 256-: 0.02 / 0.07 / 0.19 / 0.13 (118 tokens)

Whole related chunks (48, up to 512 tokens each), per layer (mean): relative size of the difference; the energy rank 4 / 16 / 64 holds; tokens for 50% / 90% of it; the top 15% tokens' share; overlap of the top 15% with layer 2's; the largest head's share:

| | rel | rank 4 / 16 / 64 | tokens for 50% / 90% | top 15% share | overlap with layer 2 | top head |
| --- | --- | --- | --- | --- | --- | --- |
| K 1 | 0.04 | 64% / 84% / 97% | 24% / 74% | 41% | 63% | 22% |
| K 2 | 0.08 | 64% / 83% / 97% | 20% / 71% | 46% | 100% | 21% |
| K 4 | 0.13 | 61% / 81% / 97% | 22% / 71% | 42% | 58% | 21% |
| K 8 | 0.22 | 58% / 82% / 98% | 20% / 70% | 45% | 43% | 18% |
| K 12 | 0.33 | 63% / 84% / 98% | 24% / 73% | 40% | 34% | 16% |
| K 16 | 0.22 | 58% / 81% / 98% | 23% / 73% | 41% | 37% | 17% |
| K 20 | 0.20 | 59% / 81% / 98% | 25% / 74% | 38% | 39% | 18% |
| K 24 | 0.20 | 58% / 80% / 97% | 26% / 74% | 37% | 38% | 19% |
| K 27 | 0.18 | 53% / 76% / 97% | 27% / 75% | 35% | 37% | 18% |
| V 1 | 0.09 | 56% / 82% / 98% | 16% / 68% | 50% | 62% | 24% |
| V 2 | 0.13 | 55% / 80% / 97% | 16% / 67% | 51% | 100% | 34% |
| V 4 | 0.27 | 48% / 75% / 96% | 19% / 67% | 46% | 58% | 20% |
| V 8 | 0.45 | 45% / 73% / 97% | 19% / 67% | 46% | 44% | 22% |
| V 12 | 0.51 | 41% / 71% / 97% | 19% / 68% | 45% | 37% | 17% |
| V 16 | 0.36 | 40% / 69% / 96% | 20% / 69% | 45% | 36% | 25% |
| V 20 | 0.37 | 40% / 68% / 95% | 20% / 69% | 45% | 36% | 24% |
| V 24 | 0.36 | 39% / 67% / 95% | 22% / 70% | 43% | 35% | 19% |
| V 27 | 0.38 | 42% / 69% / 96% | 25% / 74% | 38% | 34% | 23% |

## mtrag_raw_history_first_conversation

47 requests, 58,746 prompt tokens; prefix any earlier request shares: 17,984 (31%); related beyond it: 13,203 (22%), by kind {'other': 307, 'doc': 9092, 'answer': 3561, 'question': 243}.

Lossless, against the first copy (12,000 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.73 | 10.72 | 9.60 | 8.94 | 10.94 | 8.82 | 8.43 | 8.43 | 4% | 63% |
| V mean | 10.64 | 10.65 | 10.06 | 9.74 | 10.74 | 9.29 | 8.99 | 8.99 | 3% | 54% |
| K 0 | 10.63 | - | 1.93 | 1.54 | 10.81 | 1.80 | 1.54 | - | 71% | 99% |
| K 1 | 10.64 | 10.61 | 6.90 | 6.23 | 10.83 | 6.56 | 6.08 | 6.09 | 9% | 88% |
| K 2 | 10.72 | 10.70 | 7.89 | 7.25 | 10.93 | 7.53 | 7.07 | 7.08 | 6% | 80% |
| K 3 | 10.73 | 10.73 | 8.60 | 7.99 | 10.96 | 8.13 | 7.70 | 7.70 | 5% | 73% |
| K 4 | 10.71 | 10.69 | 8.88 | 8.25 | 10.92 | 8.33 | 7.91 | 7.92 | 5% | 70% |
| K 8 | 10.75 | 10.73 | 9.85 | 9.18 | 10.94 | 9.02 | 8.64 | 8.65 | 3% | 60% |
| K 12 | 10.75 | 10.75 | 10.49 | 9.78 | 10.94 | 9.43 | 9.07 | 9.07 | 3% | 53% |
| K 16 | 10.72 | 10.72 | 9.81 | 9.12 | 10.93 | 8.97 | 8.58 | 8.59 | 3% | 61% |
| K 20 | 10.76 | 10.75 | 9.73 | 9.08 | 10.95 | 8.95 | 8.56 | 8.57 | 3% | 61% |
| K 24 | 10.75 | 10.75 | 9.85 | 9.16 | 10.96 | 9.03 | 8.63 | 8.63 | 3% | 60% |
| K 27 | 10.71 | 10.70 | 9.75 | 9.09 | 10.93 | 8.95 | 8.55 | 8.55 | 3% | 61% |
| V 0 | 10.91 | - | 0.43 | 0.38 | 11.00 | 0.39 | 0.36 | - | 96% | 100% |
| V 1 | 10.70 | 10.70 | 7.81 | 7.26 | 10.88 | 7.51 | 7.08 | 7.08 | 6% | 80% |
| V 2 | 10.68 | 10.68 | 8.45 | 7.93 | 10.76 | 8.06 | 7.67 | 7.67 | 4% | 74% |
| V 3 | 10.63 | 10.63 | 9.05 | 8.59 | 10.72 | 8.56 | 8.20 | 8.21 | 4% | 67% |
| V 4 | 10.63 | 10.63 | 9.51 | 9.10 | 10.71 | 8.92 | 8.58 | 8.59 | 3% | 61% |
| V 8 | 10.65 | 10.65 | 10.40 | 10.10 | 10.74 | 9.54 | 9.26 | 9.27 | 2% | 50% |
| V 12 | 10.64 | 10.64 | 10.75 | 10.50 | 10.73 | 9.75 | 9.49 | 9.50 | 2% | 45% |
| V 16 | 10.64 | 10.64 | 10.04 | 9.72 | 10.74 | 9.30 | 8.99 | 9.00 | 3% | 54% |
| V 20 | 10.64 | 10.65 | 10.02 | 9.70 | 10.74 | 9.28 | 8.97 | 8.98 | 3% | 54% |
| V 24 | 10.65 | 10.65 | 10.19 | 9.90 | 10.72 | 9.40 | 9.10 | 9.10 | 2% | 52% |
| V 27 | 10.66 | 10.66 | 10.38 | 10.10 | 10.74 | 9.54 | 9.26 | 9.27 | 2% | 49% |

Lossless, against the latest copy (12,000 tokens sampled), mean over layers 1-27 and per layer:

| | alone | prev | xor | diff | planes0 | planes | base | both | equal | same sign+exp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K mean | 10.73 | 10.72 | 8.85 | 8.22 | 10.94 | 8.23 | 7.84 | 7.84 | 6% | 70% |
| V mean | 10.64 | 10.65 | 9.48 | 9.11 | 10.74 | 8.83 | 8.52 | 8.52 | 4% | 61% |
| K 0 | 10.62 | - | 1.95 | 1.55 | 10.81 | 1.80 | 1.55 | - | 71% | 99% |
| K 1 | 10.64 | 10.61 | 6.10 | 5.47 | 10.83 | 5.85 | 5.38 | 5.39 | 15% | 91% |
| K 2 | 10.72 | 10.70 | 7.12 | 6.50 | 10.93 | 6.83 | 6.38 | 6.39 | 9% | 86% |
| K 3 | 10.73 | 10.73 | 7.75 | 7.17 | 10.96 | 7.40 | 6.97 | 6.97 | 8% | 80% |
| K 4 | 10.71 | 10.69 | 8.03 | 7.43 | 10.92 | 7.61 | 7.19 | 7.20 | 7% | 78% |
| K 8 | 10.74 | 10.73 | 9.07 | 8.44 | 10.94 | 8.42 | 8.04 | 8.04 | 5% | 68% |
| K 12 | 10.75 | 10.75 | 9.75 | 9.07 | 10.94 | 8.89 | 8.53 | 8.54 | 4% | 61% |
| K 16 | 10.72 | 10.72 | 9.11 | 8.47 | 10.93 | 8.43 | 8.04 | 8.05 | 5% | 68% |
| K 20 | 10.76 | 10.75 | 9.03 | 8.41 | 10.95 | 8.40 | 8.01 | 8.01 | 5% | 68% |
| K 24 | 10.75 | 10.75 | 9.10 | 8.46 | 10.96 | 8.45 | 8.05 | 8.06 | 5% | 68% |
| K 27 | 10.71 | 10.70 | 8.95 | 8.33 | 10.94 | 8.33 | 7.93 | 7.94 | 5% | 70% |
| V 0 | 10.90 | - | 0.49 | 0.43 | 11.00 | 0.47 | 0.43 | - | 95% | 100% |
| V 1 | 10.70 | 10.70 | 7.28 | 6.73 | 10.88 | 7.01 | 6.58 | 6.58 | 9% | 84% |
| V 2 | 10.68 | 10.68 | 7.83 | 7.30 | 10.76 | 7.50 | 7.10 | 7.10 | 6% | 80% |
| V 3 | 10.63 | 10.63 | 8.39 | 7.90 | 10.72 | 7.99 | 7.61 | 7.62 | 5% | 74% |
| V 4 | 10.63 | 10.63 | 8.84 | 8.39 | 10.71 | 8.36 | 8.01 | 8.02 | 4% | 69% |
| V 8 | 10.65 | 10.65 | 9.82 | 9.47 | 10.74 | 9.10 | 8.80 | 8.81 | 3% | 57% |
| V 12 | 10.64 | 10.64 | 10.22 | 9.92 | 10.73 | 9.37 | 9.10 | 9.10 | 3% | 52% |
| V 16 | 10.64 | 10.64 | 9.51 | 9.15 | 10.74 | 8.87 | 8.56 | 8.56 | 4% | 61% |
| V 20 | 10.64 | 10.65 | 9.48 | 9.11 | 10.74 | 8.85 | 8.53 | 8.54 | 4% | 61% |
| V 24 | 10.65 | 10.65 | 9.57 | 9.23 | 10.72 | 8.93 | 8.62 | 8.62 | 4% | 60% |
| V 27 | 10.66 | 10.66 | 9.71 | 9.37 | 10.74 | 9.04 | 8.74 | 8.74 | 3% | 58% |

Relative deviation |target - base| / |target| by place in the related run (tokens since it started), layers 1 / 4 / 12 / 27:

- K 0-16: 0.05 / 0.17 / 0.39 / 0.22 (2,846 tokens)
- K 16-64: 0.04 / 0.14 / 0.35 / 0.21 (2,982 tokens)
- K 64-256: 0.03 / 0.10 / 0.25 / 0.15 (4,360 tokens)
- K 256-: 0.03 / 0.04 / 0.16 / 0.09 (1,812 tokens)
- V 0-16: 0.10 / 0.35 / 0.63 / 0.46 (2,846 tokens)
- V 16-64: 0.08 / 0.28 / 0.54 / 0.43 (2,982 tokens)
- V 64-256: 0.05 / 0.20 / 0.37 / 0.31 (4,360 tokens)
- V 256-: 0.04 / 0.09 / 0.21 / 0.19 (1,812 tokens)

Whole related chunks (48, up to 512 tokens each), per layer (mean): relative size of the difference; the energy rank 4 / 16 / 64 holds; tokens for 50% / 90% of it; the top 15% tokens' share; overlap of the top 15% with layer 2's; the largest head's share:

| | rel | rank 4 / 16 / 64 | tokens for 50% / 90% | top 15% share | overlap with layer 2 | top head |
| --- | --- | --- | --- | --- | --- | --- |
| K 1 | 0.04 | 67% / 85% / 97% | 16% / 65% | 50% | 64% | 23% |
| K 2 | 0.08 | 67% / 84% / 96% | 14% / 61% | 55% | 100% | 21% |
| K 4 | 0.14 | 62% / 81% / 96% | 17% / 60% | 51% | 62% | 21% |
| K 8 | 0.23 | 55% / 78% / 96% | 16% / 59% | 53% | 51% | 18% |
| K 12 | 0.33 | 55% / 78% / 96% | 19% / 65% | 47% | 47% | 16% |
| K 16 | 0.23 | 50% / 74% / 95% | 18% / 63% | 50% | 47% | 16% |
| K 20 | 0.21 | 52% / 76% / 95% | 19% / 64% | 49% | 47% | 17% |
| K 24 | 0.22 | 52% / 75% / 94% | 20% / 64% | 47% | 46% | 17% |
| K 27 | 0.20 | 48% / 71% / 93% | 21% / 67% | 44% | 42% | 17% |
| V 1 | 0.08 | 56% / 81% / 97% | 10% / 54% | 61% | 63% | 24% |
| V 2 | 0.14 | 56% / 81% / 96% | 11% / 55% | 60% | 100% | 36% |
| V 4 | 0.28 | 49% / 74% / 94% | 15% / 57% | 54% | 61% | 20% |
| V 8 | 0.46 | 41% / 69% / 94% | 15% / 56% | 54% | 52% | 22% |
| V 12 | 0.52 | 34% / 63% / 92% | 17% / 59% | 51% | 46% | 18% |
| V 16 | 0.38 | 35% / 62% / 91% | 16% / 58% | 53% | 43% | 23% |
| V 20 | 0.39 | 37% / 63% / 91% | 16% / 57% | 54% | 43% | 22% |
| V 24 | 0.39 | 38% / 64% / 91% | 18% / 61% | 48% | 41% | 21% |
| V 27 | 0.41 | 36% / 62% / 91% | 21% / 67% | 44% | 38% | 21% |



