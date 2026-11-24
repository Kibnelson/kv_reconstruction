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
