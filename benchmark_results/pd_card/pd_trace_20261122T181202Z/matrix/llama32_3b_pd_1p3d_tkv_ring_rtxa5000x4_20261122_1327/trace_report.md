# The reuse opportunity (traced PD run)

Per prompt layout and routing. Moved again: tokens P sent that the chosen D already held. Recomputed: tokens P prefilled without its cache; answers: tokens of earlier answers (made on D). Same bytes: pages P sent again that D held and whose bytes D's equal. Affinity: later turns that went to the D of their previous turn.

| Layout / routing | Requests | Moved again (share of sent, GB) | Recomputed tokens | Earlier answers recomputed | Same bytes: answers / docs / rest | Affinity |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 842 | 18% (28.6 GB) | 1,378,840 | 361,757 of 361,757 | 61% / 100% / 97% | 100% |
| docs_first / round_robin | 842 | 11% (16.9 GB) | 1,381,449 | 364,366 of 364,366 | 71% / 100% / 98% | 23% |
| history_first / conversation | 842 | 24% (37.7 GB) | 1,386,102 | 369,019 of 369,019 | 100% / 100% / 100% | 100% |
| history_first / round_robin | 842 | 14% (22.3 GB) | 1,383,606 | 366,523 of 366,523 | 100% / 0% / 98% | 23% |
| interleaved / conversation | 842 | 79% (406.0 GB) | 4,495,041 | 367,629 of 367,629 | 8% / 86% / 42% | 100% |
| interleaved / round_robin | 842 | 49% (251.6 GB) | 4,495,051 | 367,639 of 367,639 | 45% / 89% / 70% | 30% |

## Every page, by what already existed (traced variants)

1 prefix: an earlier request had its token prefix (prefix caches share it). 2 same bytes: not 1, but an earlier page at its layer has its bytes. 3 same text: not 1 or 2, but its 16 tokens were in an earlier prompt (a differential store's candidates). 4 new. Store: what holding every page once would take, keyed by prefix (LMCache), by content, and by content without layer 0 (recomputable from tokens).

| Layout / routing | Pages | 1 prefix | 2 same bytes | 3 same text | 4 new | Store by prefix / content / content w/o layer 0 (GB) |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 2,424,044 | 18.1% | 0.0% | 35.0% | 46.9% | 130.1 / 130.3 / 125.6 |
| history_first / conversation | 2,436,700 | 23.8% | 0.0% | 28.8% | 47.4% | 121.6 / 121.8 / 117.4 |
| interleaved / conversation | 7,877,156 | 76.7% | 0.0% | 9.1% | 14.2% | 120.4 / 140.8 / 136.5 |

Same bytes (2) by layer, and same text (3) by kind of piece:

- docs_first / conversation: same bytes at layer 0: 243, layer 1: 1, layer 2: 1, layer 3: 1, layer 4: 1, layer 5: 1, layer 6: 1, layer 7: 1, layer 8: 1, layer 9: 1, layer 10: 1, layer 11: 1, layer 12: 1, layer 13: 1, layer 14: 1, layer 15: 1, layer 16: 1, layer 17: 1, layer 18: 1, layer 19: 1, layer 20: 1, layer 21: 1, layer 22: 1, layer 23: 1, layer 24: 1, layer 25: 1, layer 26: 1, layer 27: 1; same text in {'answer': 523688, 'doc': 263571, 'other': 61958}
- history_first / conversation: same bytes at layer 0: 59, layer 1: 1, layer 2: 1, layer 3: 1, layer 4: 1, layer 5: 1, layer 6: 1, layer 7: 1, layer 8: 1, layer 9: 1, layer 10: 1, layer 11: 1, layer 12: 1, layer 13: 1, layer 14: 1, layer 15: 1, layer 16: 1, layer 17: 1, layer 18: 1, layer 19: 1, layer 20: 1, layer 21: 1, layer 22: 1, layer 23: 1, layer 24: 1, layer 25: 1, layer 26: 1, layer 27: 1; same text in {'answer': 60522, 'doc': 637061, 'other': 3668}
- interleaved / conversation: same bytes at layer 0: 6, layer 1: 1, layer 2: 1, layer 3: 1, layer 4: 1, layer 5: 1, layer 6: 1, layer 7: 1, layer 8: 1, layer 9: 1, layer 10: 1, layer 11: 1, layer 12: 1, layer 13: 1, layer 14: 1, layer 15: 1, layer 16: 1, layer 17: 1, layer 18: 1, layer 19: 1, layer 20: 1, layer 21: 1, layer 22: 1, layer 23: 1, layer 24: 1, layer 25: 1, layer 26: 1, layer 27: 1; same text in {'answer': 80304, 'doc': 635851, 'other': 1428}

## How close related KV is (the differential opportunity)

Related: prompt tokens whose 16-token window appeared earlier in the run, not as a shared prefix (so a prefix cache misses them), at the same place after another prefix or moved; within the conversation or across. Bits per value of coding each against its first occurrence: xor; diff, the difference of the words as ordered integers; rot, keys turned to the new position first; best, the cheapest per layer; alone, the same tokens without a base (sign and exponent entropy + 7 bits, the exponent coders' floor). Raw BF16 is 16. Measured: a sample of the related tokens, chosen by content (1 in the sample's number), each against its first occurrence when it was still held.

| Layout / routing | Prompt tokens | Related (share) | Docs / answers / questions / other | Same place / moved | Within / across conversations | Median shift |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | 1,378,840 | 985,103 (71.4%) | 375,734 / 532,632 / 61,120 / 15,617 | 3,926 / 981,177 | 908,056 / 77,047 | 419 |
| history_first / conversation | 1,386,102 | 866,000 (62.5%) | 490,374 / 349,243 / 21,466 / 4,917 | 164 / 865,836 | 783,011 / 82,989 | 317 |
| interleaved / conversation | 4,495,041 | 2,287,206 (50.9%) | 1,917,721 / 365,366 / 1,135 / 2,984 | 126 / 2,287,080 | 2,019,621 / 267,585 | 2,265 |

Measured related tokens: docs_first / conversation: 33,197 (1 in 8), 358 requests not measured; history_first / conversation: 35,287 (1 in 8), 292 requests not measured; interleaved / conversation: 57,273 (1 in 8), 393 requests not measured

| Layout / routing | Kind | Measured values | Equal | K: xor / diff / rot / alone | V: xor / diff / alone | Best vs alone (saving) |
| --- | --- | --- | --- | --- | --- | --- |
| docs_first / conversation | every kind | 1,903,648,768 | 3.5% | 12.02 / 11.57 / 8.69 / 10.73 | 9.79 / 9.46 / 10.66 | 9.08 vs 10.69 (15.1%) |
| docs_first / conversation | answer | 999,849,984 | 3.3% | 12.23 / 11.75 / 9.11 / 10.73 | 10.23 / 9.96 / 10.66 | 9.54 vs 10.70 (10.8%) |
| docs_first / conversation | doc | 799,432,704 | 3.7% | 11.81 / 11.38 / 8.21 / 10.73 | 9.25 / 8.85 / 10.65 | 8.53 vs 10.69 (20.2%) |
| docs_first / conversation | other | 19,267,584 | 3.6% | 11.51 / 11.08 / 7.94 / 10.73 | 9.33 / 8.92 / 10.67 | 8.43 vs 10.70 (21.2%) |
| docs_first / conversation | question | 85,098,496 | 3.6% | 11.65 / 11.20 / 8.04 / 10.72 | 9.31 / 8.90 / 10.66 | 8.47 vs 10.69 (20.7%) |
| history_first / conversation | every kind | 2,023,497,728 | 3.5% | 12.04 / 11.58 / 8.78 / 10.73 | 9.85 / 9.54 / 10.66 | 9.16 vs 10.69 (14.3%) |
| history_first / conversation | answer | 692,543,488 | 3.0% | 12.37 / 11.86 / 9.77 / 10.73 | 10.81 / 10.60 / 10.66 | 10.19 vs 10.70 (4.8%) |
| history_first / conversation | doc | 1,292,247,040 | 3.7% | 11.83 / 11.40 / 8.15 / 10.73 | 9.25 / 8.86 / 10.65 | 8.50 vs 10.69 (20.4%) |
| history_first / conversation | other | 10,838,016 | 2.5% | 12.57 / 12.12 / 9.91 / 10.73 | 11.01 / 10.81 / 10.66 | 10.36 vs 10.70 (3.2%) |
| history_first / conversation | question | 27,869,184 | 2.7% | 12.38 / 11.93 / 9.37 / 10.72 | 10.41 / 10.14 / 10.65 | 9.75 vs 10.69 (8.7%) |
| interleaved / conversation | every kind | 3,284,262,912 | 2.6% | 12.95 / 12.42 / 10.22 / 10.74 | 11.26 / 11.11 / 10.67 | 10.67 vs 10.70 (0.3%) |
| interleaved / conversation | answer | 554,287,104 | 2.7% | 12.81 / 12.26 / 10.19 / 10.74 | 11.26 / 11.11 / 10.66 | 10.65 vs 10.70 (0.5%) |
| interleaved / conversation | doc | 2,728,083,456 | 2.6% | 12.97 / 12.45 / 10.23 / 10.74 | 11.26 / 11.11 / 10.67 | 10.67 vs 10.70 (0.3%) |
| interleaved / conversation | other | 1,318,912 | 2.2% | 12.90 / 12.44 / 10.41 / 10.73 | 11.50 / 11.36 / 10.65 | 10.89 vs 10.69 (-1.8%) |
| interleaved / conversation | question | 573,440 | 2.4% | 12.77 / 12.28 / 9.71 / 10.70 | 10.94 / 10.74 / 10.67 | 10.23 vs 10.69 (4.3%) |

Best bits per value by layer (keys, values), against alone:

- docs_first / conversation, K: 0: 1.57/10.62, 1: 6.05/10.64, 2: 7.09/10.72, 3: 7.95/10.74, 4: 8.24/10.72, 5: 8.41/10.74, 6: 8.75/10.74, 7: 8.87/10.73, 8: 9.20/10.74, 9: 9.49/10.73, 10: 9.61/10.74, 11: 9.55/10.75, 12: 9.86/10.75, 13: 9.75/10.77, 14: 9.52/10.71, 15: 9.51/10.73, 16: 9.23/10.72, 17: 9.17/10.72, 18: 9.16/10.74, 19: 9.27/10.76, 20: 9.14/10.76, 21: 9.13/10.77, 22: 9.12/10.78, 23: 9.15/10.72, 24: 9.11/10.75, 25: 9.22/10.75, 26: 9.21/10.71, 27: 8.98/10.71
- docs_first / conversation, V: 0: 0.26/10.91, 1: 7.25/10.70, 2: 7.86/10.68, 3: 8.62/10.63, 4: 9.18/10.63, 5: 9.38/10.64, 6: 9.73/10.63, 7: 9.89/10.64, 8: 10.20/10.65, 9: 10.41/10.65, 10: 10.55/10.64, 11: 10.72/10.63, 12: 10.65/10.64, 13: 10.54/10.64, 14: 10.39/10.64, 15: 10.31/10.64, 16: 9.86/10.64, 17: 9.82/10.64, 18: 9.80/10.65, 19: 10.05/10.65, 20: 9.79/10.65, 21: 9.80/10.66, 22: 9.83/10.65, 23: 10.07/10.64, 24: 9.85/10.66, 25: 10.00/10.64, 26: 10.06/10.67, 27: 10.02/10.67
- history_first / conversation, K: 0: 1.57/10.62, 1: 6.20/10.64, 2: 7.21/10.72, 3: 8.07/10.74, 4: 8.36/10.71, 5: 8.53/10.74, 6: 8.86/10.74, 7: 8.97/10.73, 8: 9.31/10.74, 9: 9.58/10.72, 10: 9.69/10.73, 11: 9.64/10.75, 12: 9.92/10.75, 13: 9.82/10.77, 14: 9.59/10.71, 15: 9.58/10.73, 16: 9.30/10.72, 17: 9.25/10.72, 18: 9.24/10.74, 19: 9.36/10.76, 20: 9.22/10.75, 21: 9.21/10.77, 22: 9.22/10.78, 23: 9.25/10.72, 24: 9.22/10.75, 25: 9.33/10.75, 26: 9.32/10.71, 27: 9.11/10.71
- history_first / conversation, V: 0: 0.30/10.90, 1: 7.33/10.69, 2: 7.96/10.68, 3: 8.71/10.63, 4: 9.24/10.63, 5: 9.46/10.64, 6: 9.81/10.63, 7: 9.96/10.64, 8: 10.28/10.65, 9: 10.49/10.65, 10: 10.61/10.65, 11: 10.78/10.63, 12: 10.70/10.64, 13: 10.60/10.64, 14: 10.45/10.64, 15: 10.37/10.64, 16: 9.92/10.64, 17: 9.89/10.64, 18: 9.87/10.65, 19: 10.13/10.65, 20: 9.87/10.65, 21: 9.89/10.66, 22: 9.93/10.65, 23: 10.17/10.64, 24: 9.96/10.66, 25: 10.12/10.63, 26: 10.19/10.67, 27: 10.14/10.67
- interleaved / conversation, K: 0: 1.67/10.62, 1: 6.41/10.65, 2: 7.36/10.72, 3: 8.79/10.74, 4: 9.66/10.71, 5: 9.77/10.74, 6: 10.31/10.75, 7: 10.42/10.74, 8: 10.76/10.74, 9: 11.09/10.72, 10: 11.26/10.73, 11: 11.27/10.75, 12: 11.43/10.76, 13: 11.39/10.77, 14: 11.22/10.72, 15: 11.27/10.75, 16: 10.93/10.73, 17: 10.93/10.73, 18: 10.92/10.75, 19: 11.15/10.75, 20: 10.98/10.77, 21: 10.98/10.79, 22: 11.02/10.80, 23: 11.02/10.73, 24: 11.00/10.76, 25: 11.19/10.75, 26: 11.06/10.73, 27: 10.98/10.69
- interleaved / conversation, V: 0: 0.32/10.89, 1: 7.20/10.70, 2: 7.88/10.67, 3: 9.51/10.63, 4: 10.45/10.63, 5: 10.85/10.63, 6: 11.35/10.64, 7: 11.57/10.65, 8: 11.88/10.67, 9: 12.17/10.65, 10: 12.37/10.66, 11: 12.42/10.65, 12: 12.34/10.66, 13: 12.31/10.67, 14: 12.20/10.66, 15: 12.28/10.67, 16: 11.81/10.67, 17: 11.79/10.63, 18: 11.81/10.65, 19: 12.14/10.68, 20: 11.89/10.65, 21: 11.90/10.66, 22: 11.97/10.66, 23: 12.24/10.66, 24: 11.95/10.66, 25: 12.20/10.64, 26: 12.14/10.67, 27: 12.11/10.70

Sizes of the related tokens' KV: raw, coded alone, coded as deltas (GB); and D's pages that P recomputed (it sent them again; D held them):

- docs_first / conversation: 112.98 raw, 75.51 alone, 64.08 as deltas; held pages 435,764, of them 9,045 differ (best 0.94 vs alone 10.70 bits per value, 90.0% of the words equal)
- history_first / conversation: 99.32 raw, 66.37 alone, 56.87 as deltas; held pages 574,840, of them 440 differ (best 0.25 vs alone 10.70 bits per value, 97.8% of the words equal)
- interleaved / conversation: 262.32 raw, 175.44 alone, 174.87 as deltas; held pages 6,195,672, of them 1,369,400 differ (best 1.43 vs alone 10.70 bits per value, 83.3% of the words equal)
