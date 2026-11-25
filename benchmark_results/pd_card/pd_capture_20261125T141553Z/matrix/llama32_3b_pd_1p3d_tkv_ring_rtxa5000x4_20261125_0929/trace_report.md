# The reuse opportunity (traced PD run)

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
