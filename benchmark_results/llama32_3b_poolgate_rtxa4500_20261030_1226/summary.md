# The P gate on NVIDIA RTX A4500: unsloth/Llama-3.2-3B

P = exact KV kept compressed in decode-GPU memory, decoded in registers by attention. The pool is exactkv (the published P; its dense kernel performs the same arithmetic). See scripts/bench_pool_gate.py.

## Decode attention: pool vs dense on identical KV

Per decode step = the mean per layer x 28 layers. Bits per value of the pool include its directory and raw-row reserve (allocated) or only the rows used.

| Batch x context | Pool, per step | Dense, same arithmetic | PyTorch SDPA (dense) | Pool vs dense | Bit-identical | Bits allocated / used |
| --- | --- | --- | --- | --- | --- | --- |
| 1 x 2,048 | 1.45 ms | 3.87 ms | 1.19 ms | 2.67x | yes | 12.82 / 12.58 |
| 16 x 2,048 | 9.94 ms | 49.86 ms | 8.89 ms | 5.02x | yes | 12.82 / 12.58 |
| 32 x 2,048 | 18.75 ms | 98.70 ms | 14.47 ms | 5.26x | yes | 12.82 / 12.58 |
| 1 x 8,192 | 3.18 ms | 13.10 ms | 2.36 ms | 4.12x | yes | 12.72 / 12.58 |
| 4 x 8,192 | 9.90 ms | 49.80 ms | 7.08 ms | 5.03x | yes | 12.72 / 12.58 |
| 8 x 8,192 | 18.77 ms | 98.75 ms | 13.75 ms | 5.26x | yes | 12.72 / 12.58 |

## Generation: BF16 cache vs pool (same engine and weights)

Greedy check, batch 2, 1024-token prompts, 32 tokens: tokens equal True, logits bit-identical True.

| Prompt | Backend | Batch | KV per sequence | ms per step | Decode tokens/s |
| --- | --- | --- | --- | --- | --- |
| 2,048 + 128 | bf16 | 49 | 238 MiB | 154.04 | 318 |
| 2,048 + 128 | exactkv | 62 | 191 MiB | 70.01 | 886 |
| 2,048 + 128 | exactkv | 49 | 191 MiB | 48.29 | 1015 |
| 8,192 + 128 | bf16 | 12 | 910 MiB | 146.20 | 82 |
| 8,192 + 128 | exactkv | 15 | 724 MiB | 54.80 | 274 |
| 8,192 + 128 | exactkv | 12 | 724 MiB | 44.98 | 267 |

Each backend runs at the largest batch the same KV budget fits, then the pool at the BF16 batch (same work). The engine is exactkv's plain Python loop (no CUDA graphs) for both.

## T -> P installation: ds_gsm8k_r0 (8,192 tokens, 28 layers, 896 MiB in BF16)

| Path | Wire bits/value | Install time | Peak extra memory | of which transient | Pool | Pool exact |
| --- | --- | --- | --- | --- | --- | --- |
| ours, layer by layer into the pool (two-layer ring) | 9.28 | 52 ms | 777 MiB | 64 MiB | 712 MiB | True |
| ours, whole cache decoded, then the pool | 9.28 | 47 ms | 1609 MiB | 896 MiB | 712 MiB | True |
| SplitZip, whole cache decoded, then the pool | 12.06 | 58 ms | 1609 MiB | 896 MiB | 712 MiB | True |
| raw BF16 received whole, then the pool | 16.00 | 45 ms | 1609 MiB | 896 MiB | 712 MiB | True |

**Time until the pool is ready, per link** (the wire and the installation in sequence; a chunked protocol would overlap them):

| Link | ours (layer by layer) | ours | SplitZip | raw BF16 received whole |
| --- | --- | --- | --- | --- |
| 1 Gb/s | 4414 ms | 4408 ms | 5722 ms | 7562 ms |
| 10 Gb/s | 489 ms | 483 ms | 625 ms | 797 ms |
| 25 Gb/s | 227 ms | 221 ms | 285 ms | 346 ms |
| 50 Gb/s | 140 ms | 134 ms | 172 ms | 196 ms |
| 100 Gb/s | 96 ms | 91 ms | 115 ms | 121 ms |
| 200 Gb/s | 74 ms | 69 ms | 87 ms | 83 ms |
| 400 Gb/s | 63 ms | 58 ms | 73 ms | 64 ms |

