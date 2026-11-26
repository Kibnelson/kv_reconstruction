# PD matrix lmctrace on gpu-grad-01, job 30774 (completed), reviewed from the kept results


# LMCache lifecycle: the runs side by side

| | memory, NIXL, shared, carry-forward | memory, NIXL, shared, carry-forward |
| --- | ---: | ---: |
| LMCache server lasted the run | True | True |
| LMCache server errors | 0 | 0 |
| P to D | nixl | nixl |
| LMCache servers | shared | shared |
| prompt layout | interleaved | interleaved |
| passages | all | all |
| conversations | 1 | 1 |
| disk reads | retain | retain |
| chunks bound by the engines' stores | 12 | 12 |
| chunks stored by | {'P': 12} | {'P': 12} |
| chunks written to memory | 12 | 12 |
| stored MB | 352.300 | 352.300 |
| chunks to disk | 0 | 0 |
| chunks removed from memory | 0 | 0 |
| pulls | 0 | 0 |
| pulled from | {} | {} |
| never pulled | 12 | 12 |
| D tokens from own GPU | 1888 | 1888 |
| D tokens loaded from LMCache | 0 | 0 |
| D tokens read from P over NIXL | 7341 | 7341 |
| NIXL read ms (median) | 352.920 | 353.650 |
| NIXL read ms (p90) | 368.690 | 365.960 |
| NIXL read GB/s (median) | 0.590 | 0.594 |
| UCX transports | cma, cuda_copy, tcp | cma, cuda_copy, tcp |
| P held its blocks for D, ms (median) | 387.900 | 385.140 |
| D tokens computed | 0 | 0 |
| pull ms, memory (median) | – | – |
| pull ms, disk (median) | – | – |
| pull ms, disk (p90) | – | – |
| lookup and prefetch ms, memory (median) | – | – |
| lookup and prefetch ms, disk (median) | – | – |
| retrieve ms (median) | – | – |
| P store ms (median) | 69.770 | 69.940 |
| memory to disk ms (median) | – | – |
| D lookups before P's store ended | 0 | 0 |
| checks | 12 of 12; 0 of 0; 12 of 12 | 12 of 12; 0 of 0; 12 of 12 |

Runs: llama32_3b_lmc_memory_nixl_shared_carry_all_allgpus_rtxa5000_20261126_183639, llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_183446


# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 99536dce0cd9.

Trace: 1081 records from 5 processes (D1 310, D2 312, D3 157, LMC 222, P 80); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | – |

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 352.920 ms, p90 368.690 ms
- P held its blocks for D from its request's end to D's read done: median 387.900 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.590 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all)

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 71.670 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.770 ms, p90 75.410 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded (LMCache / NIXL) | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 0 | 0 | 515 | 0 | 0 | 0 / 0 | 515 | new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 0 | 515 | 0 | 512 | 0 / 515 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 0 | 1 | 1131 | 512 | 512 | 0 / 0 | 619 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D2 | 0 | 1 | 1131 | 0 | 1024 | 0 / 1131 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 0 | 2 | 1847 | 1120 | 1024 | 0 / 0 | 727 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 0 | 2 | 1847 | 0 | 1792 | 0 / 1847 | 0 | hit, not loaded 7 | – | – | 0 |
| P | 0 | 3 | 2531 | 1840 | 1792 | 0 / 0 | 691 | held (on GPU already) 7, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 3 | 2531 | 640 | 2304 | 0 / 1891 | 0 | held (on GPU already) 2, hit, not loaded 7 | – | – | 0 |
| P | 0 | 4 | 3205 | 2528 | 2304 | 0 / 0 | 677 | held (on GPU already) 9, new: stored by this request 3 | – | – | 768 |
| D2 | 0 | 4 | 3205 | 1248 | 3072 | 0 / 1957 | 0 | held (on GPU already) 4, hit, not loaded 8 | – | – | 0 |

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0024 | P looked it up (turn 0): new: stored by this request |
| 0.0047 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 57447d51) |
| 0.0047 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request 57447d51) |
| 0.0049 | P's connector planned the store |
| 0.0154 | P's worker sent it to the server |
| 0.0168 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0175 | server memory (L1) reserved |
| 0.0786 | eviction order refreshed (a touch: no I/O) |
| 0.0814 | written to memory (L1) |
| 0.1032 | D1 looked it up (turn 0): hit, not loaded |
| 1.7501 | eviction order refreshed (a touch: no I/O) |
| 1.7586 | P looked it up (turn 1): held (on GPU already) |
| 1.7588 | eviction order refreshed (a touch: no I/O) |
| 1.8174 | eviction order refreshed (a touch: no I/O) |
| 1.8436 | D2 looked it up (turn 1): hit, not loaded |
| 3.6251 | eviction order refreshed (a touch: no I/O) |
| 3.6382 | P looked it up (turn 2): held (on GPU already) |
| 3.6384 | eviction order refreshed (a touch: no I/O) |
| 3.7085 | eviction order refreshed (a touch: no I/O) |
| 3.7276 | D3 looked it up (turn 2): hit, not loaded |
| 5.6564 | eviction order refreshed (a touch: no I/O) |
| 5.6666 | P looked it up (turn 3): held (on GPU already) |
| 5.6667 | eviction order refreshed (a touch: no I/O) |
| 5.7380 | eviction order refreshed (a touch: no I/O) |
| 5.7489 | D1 looked it up (turn 3): held (on GPU already) |
| 5.7491 | eviction order refreshed (a touch: no I/O) |
| 7.5549 | eviction order refreshed (a touch: no I/O) |
| 7.5686 | P looked it up (turn 4): held (on GPU already) |
| 7.5688 | eviction order refreshed (a touch: no I/O) |
| 7.6426 | eviction order refreshed (a touch: no I/O) |
| 7.6555 | D2 looked it up (turn 4): held (on GPU already) |
| 7.6557 | eviction order refreshed (a touch: no I/O) |
| 9.4830 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0021 | the proxy received the turn |
| 0.0808 | P answered (stored) |
| 0.1006 | D1 took the request |
| 0.1032 | the server got its lookup |
| 0.1040 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1047 | its connector had the lookup's answer |
| 0.3809 | the first token reached the proxy |
| 1.7495 | finished |
| 1.7529 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.


# LMCache lifecycle: memory mode, NIXL, shared servers, carry-forward, all passages

1 prefill engine (P) and 3 decode engines (D1-D3), one LMCache server shared by all four; mode **memory** (memory 8.0 GB each, eviction LRU, store policy default, chunk 256 tokens); MTRAG's conversations 1534a095, interleaved (carry-forward), passages: every passage MTRAG has for the turn (reference.jsonl and RAG.jsonl, each once), one conversation at a time. P's KV reaches D over NIXL (vLLM's MultiConnector on both engines: NixlConnector, D reading P's GPU blocks, and LMCache's connector, kv_both, storing and loading beside it); the proxy sends D the turn with P's reply's kv_transfer_params as soon as P answers (as vllm-router does), D engines round-robin. Commit 99536dce0cd9.

Trace: 1081 records from 5 processes (D1 312, D2 310, D3 157, LMC 222, P 80); hook errors 0; unreadable lines 0.

## What happened

| | P | D (all three) |
| --- | ---: | ---: |
| requests | 5 | 5 |
| prompt tokens | 9229 | 9229 |
| from the engine's own GPU prefix cache | 6000 | 1888 |
| loaded from LMCache | 0 | 0 |
| read from P's GPU over NIXL | – | 7341 |
| computed | 3229 | 0 |
| stored to LMCache (tokens) | 3072 | – |

## NIXL

- D requests that read from P: 5, 7341 tokens; the read (D's worker posts it, to the KV in D's GPU blocks): median 353.650 ms, p90 365.960 ms
- P held its blocks for D from its request's end to D's read done: median 385.140 ms
- handshakes (D's worker meets P's agent, once a pair): 3; failed reads: 0
- the read's speed: median 0.594 GB/s (the KV it read over its time)
- UCX's transports (its protocol tables in the engines' logs): cma, cuda_copy, tcp — no cuda_ipc: the copy goes GPU to host, over TCP, host to GPU; every GPU visible to each engine: no (its own only); UCX_TLS unset (UCX's default, all)

## Chunks

- bound to their tokens by the engines' stores: 12 chunks (by {'P': 12}); written to memory 12 (352.3 MB); to disk 0; removed from memory 0
- pulled by D or P: 0 times, from –; never pulled: 12 (5 of them hold an earlier answer, which D generated itself and holds)
- reads from disk: 0; touches (eviction order refreshed, no I/O): 96
- every lookup's outcome per chunk: {'new: stored by this request': 12, 'hit, not loaded': 28, 'held (on GPU already)': 28}
- pulls per stored chunk: {0: 12}
- median time from a chunk's last block computed on P to the chunk in memory: 73.680 ms; memory to disk: – ms
- lookups of chunks never seen stored: 0

D's loads from LMCache: 0 requests (NIXL gives the load to D first; LMCache loads only what an engine's own GPU and NIXL do not have).

P's stores (planned to stored): median 69.940 ms, p90 73.710 ms.

## Checks

- chunk tokens match prompt: 12 of 12
- pulled blocks same hash as p: 0 of 0
- chunks with full chain: 12 of 12

## Requests

| engine | conv | turn | prompt | own GPU | LMCache hit | loaded (LMCache / NIXL) | computed | chunks (outcome) | from | pull ms | stored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| P | 0 | 0 | 515 | 0 | 0 | 0 / 0 | 515 | new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 0 | 515 | 0 | 512 | 0 / 515 | 0 | hit, not loaded 2 | – | – | 0 |
| P | 0 | 1 | 1131 | 512 | 512 | 0 / 0 | 619 | held (on GPU already) 2, new: stored by this request 2 | – | – | 512 |
| D2 | 0 | 1 | 1131 | 0 | 1024 | 0 / 1131 | 0 | hit, not loaded 4 | – | – | 0 |
| P | 0 | 2 | 1847 | 1120 | 1024 | 0 / 0 | 727 | held (on GPU already) 4, new: stored by this request 3 | – | – | 768 |
| D3 | 0 | 2 | 1847 | 0 | 1792 | 0 / 1847 | 0 | hit, not loaded 7 | – | – | 0 |
| P | 0 | 3 | 2531 | 1840 | 1792 | 0 / 0 | 691 | held (on GPU already) 7, new: stored by this request 2 | – | – | 512 |
| D1 | 0 | 3 | 2531 | 640 | 2304 | 0 / 1891 | 0 | held (on GPU already) 2, hit, not loaded 7 | – | – | 0 |
| P | 0 | 4 | 3205 | 2528 | 2304 | 0 / 0 | 677 | held (on GPU already) 9, new: stored by this request 3 | – | – | 768 |
| D2 | 0 | 4 | 3205 | 1248 | 3072 | 0 / 1957 | 0 | held (on GPU already) 4, hit, not loaded 8 | – | – | 0 |

## One chunk's life: 2fef896f56fe534c

Conversation 0, turn 0, tokens 0-256 (system, doc 825986711_374-788-0-414, doc 825986711_2085-2489-0-404, doc 825986711_1099-1546-0-447); 29360128 bytes. Seconds from the run's first record:

| s | what |
| ---: | --- |
| 0.0029 | P looked it up (turn 0): new: stored by this request |
| 0.0047 | P computed and cached block 0 (GPU block 1, hash 9f9c499d1c876b7e, request 8cba8514) |
| 0.0047 | P computed and cached block 15 (GPU block 16, hash edc1367af924a57d, request 8cba8514) |
| 0.0049 | P's connector planned the store |
| 0.0150 | P's worker sent it to the server |
| 0.0219 | the server bound the chunk to its tokens (mp.tokens) |
| 0.0223 | server memory (L1) reserved |
| 0.0757 | eviction order refreshed (a touch: no I/O) |
| 0.0816 | written to memory (L1) |
| 0.1332 | D1 looked it up (turn 0): hit, not loaded |
| 1.7674 | eviction order refreshed (a touch: no I/O) |
| 1.7770 | P looked it up (turn 1): held (on GPU already) |
| 1.7772 | eviction order refreshed (a touch: no I/O) |
| 1.8358 | eviction order refreshed (a touch: no I/O) |
| 1.8626 | D2 looked it up (turn 1): hit, not loaded |
| 3.6242 | eviction order refreshed (a touch: no I/O) |
| 3.6372 | P looked it up (turn 2): held (on GPU already) |
| 3.6374 | eviction order refreshed (a touch: no I/O) |
| 3.7074 | eviction order refreshed (a touch: no I/O) |
| 3.7345 | D3 looked it up (turn 2): hit, not loaded |
| 5.6590 | eviction order refreshed (a touch: no I/O) |
| 5.6691 | P looked it up (turn 3): held (on GPU already) |
| 5.6692 | eviction order refreshed (a touch: no I/O) |
| 5.7416 | eviction order refreshed (a touch: no I/O) |
| 5.7646 | D1 looked it up (turn 3): held (on GPU already) |
| 5.7647 | eviction order refreshed (a touch: no I/O) |
| 7.5789 | eviction order refreshed (a touch: no I/O) |
| 7.5912 | P looked it up (turn 4): held (on GPU already) |
| 7.5913 | eviction order refreshed (a touch: no I/O) |
| 7.6662 | eviction order refreshed (a touch: no I/O) |
| 7.6790 | D2 looked it up (turn 4): held (on GPU already) |
| 7.6790 | eviction order refreshed (a touch: no I/O) |
| 9.5024 | eviction order refreshed (a touch: no I/O) |

## One D request: conversation 0, turn 0 on D1

| s | what |
| ---: | --- |
| -0.0022 | the proxy received the turn |
| 0.0779 | P answered (stored) |
| 0.1304 | D1 took the request |
| 0.1332 | the server got its lookup |
| 0.1342 | the server found 2 of 2 chunks (memory 512, disk 0 tokens) |
| 0.1358 | its connector had the lookup's answer |
| 0.3988 | the first token reached the proxy |
| 1.7667 | finished |
| 1.7700 | its GPU blocks freed (they stay cached) |

Files: requests.jsonl.gz, chunks.jsonl.gz, blocks.jsonl.gz (one row each, every time; docs/lmcache_lifecycle.md says what each field is), results.json, kvtrace.tar.gz (the raw records).

One page per conversation, from its prompt's pieces to its chunks and blocks: conversations/index.md.
