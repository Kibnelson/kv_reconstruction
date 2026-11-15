# PD matrix smoke on gpu-grad-01, job 30210 (completed with failures), reviewed from the kept results

Model unsloth/Llama-3.2-3B; NVIDIA RTX A5000; vLLM 0.32.0, torch 2.13.0+cu130. 5 runs: `llama32_3b_pd_1p3d_tkv_ring_eager_rtxa5000x4_20261115_0835_smoke`, `llama32_3b_pd_2p2d_nixl_rtxa5000x4_20261115_0830_smoke`, `llama32_3b_pd_2p2d_tkv_p2p_rtxa5000x4_20261115_0828_smoke`, `llama32_3b_pd_colo4_lmcache_rtxa5000x4_20261115_0833_smoke`, `llama32_3b_pd_colo4_tkv_rtxa5000x4_20261115_0831_smoke`.

Runs that did not finish: `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,splitzip,exans,adacode,exactkv,ours-hbh,t2-hbh`; `SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,nvcomp-ans,nvcomp-zstd,nvcomp-bitcomp`.

## Correctness

| Run | Codec | Bit-exact | Outputs equal to local prefill | OK | Bits per value |
| --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | raw | 4/4 | 4/4 | yes | 16.000 |
| 1p3d, host memory, eager | t2-hbh | 4/4 | 4/4 | yes | 8.489 |
| 1p3d, host memory, eager | t2-rank256 | 4/4 | 4/4 | yes | 9.578 |
| 2p2d, NIXL (vLLM) | nixl | n/a | 3/4 | yes | - |
| 2p2d, GPU to GPU | raw | 4/4 | 4/4 | yes | 16.000 |
| 2p2d, GPU to GPU | ours-hbh | 4/4 | 4/4 | yes | 9.053 |
| 2p2d, GPU to GPU | t2-hbh | 4/4 | 4/4 | yes | 8.490 |
| colo4, LMCache (vLLM) tier | lmcache (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | raw (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | exans (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | ours-hbh (c4) | 0/0 chunks | - | yes | - |
| colo4, host memory tier | t2-hbh (c4) | 0/0 chunks | - | yes | - |

## 1p3d, host memory, eager engines: link 10 Gb/s

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.85 (0.69); 2395 / 2816 | 11.9 |
| t2-hbh | 8.146 | 3.06 (3.06); 877 / 1079 | 12.5 |
| t2-rank256 | 9.230 | 3.11 (3.11); 778 / 1051 | 12.1 |

Best by goodput, then TTFT p50: rate inf: t2-rank256 (goodput 3.11/s, TTFT p50 778 ms).

## 2p2d, GPU to GPU: link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| raw | 16.000 | 1.15 (0.14); 5363 / 5392 | 12.1 |
| ours-hbh | 8.684 | 3.27 (3.27); 888 / 912 | 12.1 |
| t2-hbh | 8.146 | 2.70 (2.70); 1364 / 1433 | 12.0 |

Best by goodput, then TTFT p50: rate inf: ours-hbh (goodput 3.27/s, TTFT p50 888 ms).

## 2p2d, NIXL (vLLM): link unpaced (the machine's own path)

Requests per second (goodput at the TTFT target) and TTFT p50 / p99 in ms, per request rate.

| Codec | Bits | rate inf | TPOT p50 (ms) |
| --- | --- | --- | --- |
| nixl | 16.000 | 2.01 (0.00); 2337 / 2443 | 12.1 |

Best by goodput, then TTFT p50: rate inf: nixl (goodput 0.00/s, TTFT p50 2337 ms).

## CPU tier, colo4: 4 clients

| Tier | Req/s | TTFT p50 / p90 / p99 (ms) | TPOT p50 (ms) | Loads | Tokens loaded | Load ms (copy / decode) | Bits stored | Chunks stored |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exans | 2.91 | 84 / 200 / 217 | 11.3 | 0 | 0 | - (- / -) | - | 44 |
| ours-hbh | 3.04 | 87 / 164 / 249 | 11.4 | 0 | 0 | - (- / -) | - | 45 |
| raw | 2.86 | 100 / 265 / 276 | 11.1 | 0 | 0 | - (- / -) | - | 44 |
| t2-hbh | 3.11 | 112 / 154 / 157 | 11.2 | 0 | 0 | - (- / -) | - | 43 |
| LMCache CPU offload (raw) | - | - / - / - | - | 0 | 0 | - (- / -) | - | 0 |

## T2: layer 0 recomputed by the receiver, and the constant sets picked

| Run | Codec | Link | Rate | Layer 0 recomputed | Layer 0 coded | Sets | Sender's check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1p3d, host memory, eager | t2-hbh | 10 | inf | 0 | 8 | {'news': 1, 'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1} | no steps for every token (a prefix-cache hit or a short prompt list) |
| 1p3d, host memory, eager | t2-rank256 | 10 | inf | 0 | 8 | {'news': 1, 'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1} | no steps for every token (a prefix-cache hit or a short prompt list) |
| 2p2d, GPU to GPU | t2-hbh | native | inf | 0 | 8 | {'code': 2, 'wikitext': 2, 'math': 2, 'exam': 1, 'news': 1} | no steps for every token (a prefix-cache hit or a short prompt list); recompute failed: RuntimeError('no model with .model.embed_tokens and .model.layers in thi |

## Codecs that did not load

- ibp: ModuleNotFoundError("No module named 'ibp'")

