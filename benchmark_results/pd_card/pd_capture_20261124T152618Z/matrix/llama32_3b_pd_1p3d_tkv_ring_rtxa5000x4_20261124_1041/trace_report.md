# The reuse opportunity (traced PD run)

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
