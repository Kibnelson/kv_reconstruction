# PD matrix smoke on gpu-grad-01

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 5 runs: `llama32_3b_pd_1p3d_tkv_ring_eager_rtxa5000x4_20261115_1344_smoke`, `llama32_3b_pd_2p2d_nixl_rtxa5000x4_20261115_1339_smoke`, `llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_20261115_1337_smoke`, `llama32_3b_pd_colo4_lmcache_rtxa5000x4_20261115_1343_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261115_1340_smoke`.

Runs that did not finish: `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,splitzip,exans,adacode,exactkv,ours-hbh,t2-hbh`; `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory, eager | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| 1p3d, host memory, eager | t2-rank256 | 4/4 | 4/4 | yes | 9.578 |
| 2p2d, NIXL (vLLM) | nixl | n/a | 4/4 | yes | - |
| 2p2d, GPU to GPU | raw | 4/4 | 4/4 | yes | 16.000 |
| 2p2d, GPU to GPU | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 2p2d, GPU to GPU | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| colo4, LMCache (vLLM) tier | lmcache (c8) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | raw (c8) | 7/7 chunks | - | yes | 27.569 |
| colo4, host memory tier | exans (c8) | 3/3 chunks | - | yes | 20.351 |
| colo4, host memory tier | ours-hbh (c8) | 4/4 chunks | - | yes | 13.991 |
| colo4, host memory tier | t2-hbh (c8) | 9/9 chunks | - | yes | 9.944 |

## 1p3d, host memory, eager engines: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.46 (0.00); 3545 / 3968 | 11.9 |
| t2-hbh | 8.146 | 2.12 (1.06); 2009 / 2248 | 13.1 |
| t2-rank256 | 9.230 | 2.14 (1.33); 1973 / 2224 | 12.1 |

Best by goodput, then TTFT p50: rate inf: t2-rank256 (goodput 1.33/s, TTFT p50 1973 ms).

## 2p2d, GPU to GPU: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.69 (0.00); 3110 / 3204 | 12.1 |
| ours-hbh | 8.684 | 2.80 (2.80); 1319 / 1332 | 12.1 |
| t2-hbh | 8.146 | 2.82 (2.82); 1293 / 1297 | 12.1 |

Best by goodput, then TTFT p50: rate inf: t2-hbh (goodput 2.82/s, TTFT p50 1293 ms).

## 2p2d, NIXL (vLLM): link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| nixl | 16.000 | 2.12 (0.00); 2230 / 2232 | 12.1 |

Best by goodput, then TTFT p50: rate inf: nixl (goodput 0.00/s, TTFT p50 2230 ms).

## CPU tier, colo4: 8 clients

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Loads | Tokens loaded | Load ms (copy / decode) | Bits stored | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t2-hbh | 5.27 | 67 / 251 / 673 | 11.8 | 4 | 1936 | 355.5 (1.5 / 334.9) | 9.944 | 110 |
| exans | 6.00 | 70 / 185 / 272 | 12.0 | 2 | 416 | 47.9 (1.2 / 11.2) | 20.351 | 107 |
| ours-hbh | 5.58 | 78 / 203 / 401 | 11.9 | 2 | 608 | 309.3 (1.2 / 289.1) | 13.991 | 104 |
| LMCache CPU offload (raw) | 5.68 | 78 / 359 / 402 | 11.7 | 0 | 0 | - (- / -) | - | 0 |
| raw | 5.74 | 87 / 167 / 256 | 11.9 | 5 | 1040 | 43.0 (1.7 / 16.0) | 27.569 | 112 |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | t2-hbh | 10 | inf | 0 | 8 | {'news': 1, 'code': 2, 'math': 2, 'wikitext': 2, 'exam': 1} | recompute failed: RuntimeError('no model with .model.embed_tokens and .model.layers in this process; modules seen: ') |
| 1p3d, host memory, eager | t2-rank256 | 10 | inf | 0 | 8 | {'wikitext': 2, 'math': 2, 'news': 1, 'code': 2, 'exam': 1} | recompute failed: RuntimeError('no model with .model.embed_tokens and .model.layers in this process; modules seen: ') |
| 2p2d, GPU to GPU | t2-hbh | native | inf | 0 | 8 | {'news': 1, 'wikitext': 2, 'code': 2, 'math': 2, 'exam': 1} | recompute failed: RuntimeError('no model with .model.embed_tokens and .model.layers in this process; modules seen: ') |

## Codecs that did not load

- ibp: ModuleNotFoundError("No module named 'ibp'")

