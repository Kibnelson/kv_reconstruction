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
