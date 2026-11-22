# The reuse opportunity (traced PD run)

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
