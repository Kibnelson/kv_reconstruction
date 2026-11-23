# KV Diff Graph against EvidenceStore, on the kept capture

Pages of 16 tokens x 28 layers (1,835,008 B raw). Ours (t2-*): layer 0 recomputed from the token ids (taken as exact offline; T2 checks it per request in the engine); per page, T pads a page to 32 tokens (its programs' size), and that padding is in the sizes; the rest with T2 or as a delta coded by T's value-slice coder against the base (keys turned by RoPE). Every representation decoded and compared byte for byte. Theirs: zlib level 1 (scripts/pd_page_compare.py).

## mtrag_raw_docs_first_conversation

47 requests; 3,614 complete blocks, of them 347 identical to a stored one (references) and 3,267 distinct pages (5,995.0 MB raw).

Every row covers the distinct pages, except T2 per request: it codes each request's whole prompt, the pages identical to stored ones included (no page is shared between requests), against that prompt's own raw size. Bits: per value (16 raw).

| Representation | MB | Bits | Saving vs raw | Decoded exactly |
| --- | ---: | ---: | ---: | --- |
| Raw | 5,995.0 | 16.00 | - | - |
| Theirs: best independent (zlib, exponent / sign-fraction streams) | 4,326.4 | 11.55 | 27.8% | - |
| Theirs: graph (position, text) | 4,220.6 | 11.26 | 29.6% | - |
| Theirs: graph + keys turned | 4,153.4 | 11.09 | 30.7% | - |
| Ours: t2-hbh per request (whole prompts: 3,614 blocks, repeats included) | 4,258.8 | 10.27 | 35.8% | 47 of 47 |
| Ours: t2-hbh per segment | 3,950.6 | 10.54 | 34.1% | 47 of 47 |
| Ours: t2-hbh per page | 10,648.2 | 28.42 | -77.6% | 3267 of 3267 |
| Ours: graph (t2-hbh page or delta): 1508 deltas on 1159 bases | 7,304.4 | 19.49 | -21.8% (beyond t2-hbh per page: 31.4%) | deltas 3977 of 3977 |
| Ours: the same, odd conversations only (spread fitted on the even ones) | 3,898.6 | - | beyond t2-hbh per page: 30.5% | - |
| Ours: t2-rank256 per request (whole prompts: 3,614 blocks, repeats included) | 4,642.0 | 11.20 | 30.0% | 47 of 47 |
| Ours: t2-rank256 per segment | 4,297.4 | 11.47 | 28.3% | 47 of 47 |
| Ours: t2-rank256 per page | 11,071.2 | 29.55 | -84.7% | 3267 of 3267 |
| Ours: graph (t2-rank256 page or delta): 1497 deltas on 1158 bases | 7,556.5 | 20.17 | -26.0% (beyond t2-rank256 per page: 31.7%) | deltas 3977 of 3977 |
| Ours: the same, odd conversations only (spread fitted on the even ones) | 4,026.7 | - | beyond t2-rank256 per page: 30.9% | - |
| Ours: delta against the base alone, on the 2,798 pages that have one (layer 0 recomputed) | 2,952.4 | 9.20 | 42.5% | 3977 of 3977 |
| Ours: the same, odd conversations only (1,494 pages) | 1,603.0 | 9.36 | 41.5% | - |

## mtrag_raw_history_first_conversation

47 requests; 3,650 complete blocks, of them 1,372 identical to a stored one (references) and 2,278 distinct pages (4,180.1 MB raw).

Every row covers the distinct pages, except T2 per request: it codes each request's whole prompt, the pages identical to stored ones included (no page is shared between requests), against that prompt's own raw size. Bits: per value (16 raw).

| Representation | MB | Bits | Saving vs raw | Decoded exactly |
| --- | ---: | ---: | ---: | --- |
| Raw | 4,180.1 | 16.00 | - | - |
| Theirs: best independent (zlib, exponent / sign-fraction streams) | 3,018.9 | 11.56 | 27.8% | - |
| Theirs: graph (position, text) | 2,943.9 | 11.27 | 29.6% | - |
| Theirs: graph + keys turned | 2,897.5 | 11.09 | 30.7% | - |
| Ours: t2-hbh per request (whole prompts: 3,650 blocks, repeats included) | 4,318.5 | 10.32 | 35.5% | 47 of 47 |
| Ours: t2-hbh per segment | 2,966.4 | 11.35 | 29.0% | 41 of 41 |
| Ours: t2-hbh per page | 7,423.2 | 28.41 | -77.6% | 2278 of 2278 |
| Ours: graph (t2-hbh page or delta): 1056 deltas on 811 bases | 5,105.3 | 19.54 | -22.1% (beyond t2-hbh per page: 31.2%) | deltas 2494 of 2494 |
| Ours: the same, odd conversations only (spread fitted on the even ones) | 2,578.5 | - | beyond t2-hbh per page: 28.0% | - |
| Ours: t2-rank256 per request (whole prompts: 3,650 blocks, repeats included) | 4,705.1 | 11.24 | 29.8% | 47 of 47 |
| Ours: t2-rank256 per segment | 3,209.2 | 12.28 | 23.2% | 41 of 41 |
| Ours: t2-rank256 per page | 7,717.4 | 29.54 | -84.6% | 2278 of 2278 |
| Ours: graph (t2-rank256 page or delta): 1064 deltas on 811 bases | 5,245.0 | 20.08 | -25.5% (beyond t2-rank256 per page: 32.0%) | deltas 2494 of 2494 |
| Ours: the same, odd conversations only (spread fitted on the even ones) | 2,641.3 | - | beyond t2-rank256 per page: 29.0% | - |
| Ours: delta against the base alone, on the 1,883 pages that have one (layer 0 recomputed) | 2,034.8 | 9.42 | 41.1% | 2494 of 2494 |
| Ours: the same, odd conversations only (877 pages) | 1,014.7 | 10.09 | 37.0% | - |

## Times (CUDA-synchronised, one RTX A5000)

| Step | n | mean ms | p50 ms |
| --- | ---: | ---: | ---: |
| delta decode (base warm) | 6,471 | 7.02 | 7.68 |
| delta encode | 6,471 | 2.84 | 2.80 |
| raw page host to GPU | 400 | 0.08 | 0.08 |
| rebuild cold (base pages decoded with T2, then the delta) | 200 | 22.63 | 19.51 |
| t2-hbh per page decode | 5,545 | 19.87 | 19.82 |
| t2-hbh per page encode | 5,545 | 6.23 | 6.22 |
| t2-hbh per request decode | 94 | 45.55 | 25.19 |
| t2-hbh per request encode | 94 | 23.89 | 19.95 |
| t2-hbh per segment decode | 88 | 25.84 | 23.93 |
| t2-hbh per segment encode | 88 | 16.77 | 17.30 |
| t2-rank256 per page decode | 5,545 | 3.17 | 3.16 |
| t2-rank256 per page encode | 5,545 | 6.16 | 6.13 |
| t2-rank256 per request decode | 94 | 9.57 | 5.40 |
| t2-rank256 per request encode | 94 | 15.44 | 16.41 |
| t2-rank256 per segment decode | 88 | 4.95 | 4.73 |
| t2-rank256 per segment encode | 88 | 13.67 | 14.61 |

