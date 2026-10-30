# The P gate on NVIDIA RTX A4500: unsloth/Llama-3.2-3B

P = exact KV kept compressed in decode-GPU memory, decoded in registers by attention. The pool is exactkv (the published P; its dense kernel performs the same arithmetic). See scripts/bench_pool_gate.py.

## Decode attention: the pool against production dense kernels, on identical KV

Per decode step = the mean per layer x 28 layers. The production dense kernels are PyTorch SDPA (contiguous [B, H, T, D], and on the engine's cache view) and FlashInfer (exactkv's own baseline; when installed). The pool's bit-identity is checked against exactkv's dense kernel with the same arithmetic, which is a reference, not a production kernel (it assembles BF16 from bytes). Bits per value of the pool: allocated (directory and raw-row reserve) / used.

| Batch x context | Pool | SDPA | SDPA, cache view | FlashInfer | Pool vs fastest dense | Same-arithmetic reference | Bit-identical | Bits allocated / used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 x 2,048 | 1.57 ms | 1.19 ms | 1.38 ms | 1.23 ms | 0.76x | 1.70 ms | yes | 12.82 / 12.58 |
| 16 x 2,048 | 10.89 ms | 9.12 ms | 9.42 ms | 7.68 ms | 0.71x | 10.23 ms | yes | 12.82 / 12.58 |
| 32 x 2,048 | 20.84 ms | 14.77 ms | 15.43 ms | 13.96 ms | 0.67x | 23.07 ms | yes | 12.82 / 12.58 |
| 1 x 8,192 | 3.54 ms | 2.41 ms | 2.63 ms | 2.44 ms | 0.68x | 3.81 ms | yes | 12.72 / 12.58 |
| 4 x 8,192 | 10.92 ms | 7.10 ms | 7.34 ms | 7.16 ms | 0.65x | 11.13 ms | yes | 12.72 / 12.58 |
| 8 x 8,192 | 20.82 ms | 13.77 ms | 13.98 ms | 13.42 ms | 0.64x | 24.87 ms | yes | 12.72 / 12.58 |

Pool vs fastest dense above 1 means the pool is faster.

## Generation: BF16 cache vs pool (same engine and weights)

Greedy check, batch 2, 1024-token prompts, 32 tokens: tokens equal True, logits bit-identical True.

| Prompt | Backend | Batch | KV per sequence | ms per step | Decode tokens/s |
| --- | --- | --- | --- | --- | --- |
| 2,048 + 128 | bf16 + SDPA | 49 | 238 MiB | 41.68 | 1176 |
| 2,048 + 128 | pool (exactkv) | 62 | 191 MiB | 70.55 | 879 |
| 2,048 + 128 | bf16 + FlashInfer | 49 | 238 MiB | 38.99 | 1257 |
| 2,048 + 128 | pool (exactkv) (at the dense batch) | 49 | 191 MiB | 46.14 | 1062 |
| 8,192 + 128 | bf16 + SDPA | 12 | 910 MiB | 41.55 | 289 |
| 8,192 + 128 | pool (exactkv) | 15 | 724 MiB | 57.86 | 259 |
| 8,192 + 128 | bf16 + FlashInfer | 12 | 910 MiB | 34.83 | 344 |
| 8,192 + 128 | pool (exactkv) (at the dense batch) | 12 | 724 MiB | 46.54 | 258 |

Each backend runs at the largest batch the same KV budget fits, then the pool at the dense batch (the same work). Same engine for all (exactkv's plain Python loop, no CUDA graphs); only the cache and the decode-attention kernel differ.

## T -> P installation: ds_gsm8k_r0 (8,192 tokens, 28 layers, 896 MiB in BF16)

| Path | Wire bits/value | Install time | Peak extra memory | of which transient | Pool | Pool exact |
| --- | --- | --- | --- | --- | --- | --- |
| ours, layer by layer into the pool (two-layer ring) | 9.28 | 47 ms | 777 MiB | 64 MiB | 712 MiB | True |
| ours, whole cache decoded, then the pool | 9.28 | 46 ms | 1609 MiB | 896 MiB | 712 MiB | True |
| SplitZip, whole cache decoded, then the pool | 12.06 | 54 ms | 1609 MiB | 896 MiB | 712 MiB | True |
| raw BF16 received whole, then the pool | 16.00 | 42 ms | 1609 MiB | 896 MiB | 712 MiB | True |

**Time until the pool is ready, per link** (the wire and the installation in sequence; a chunked protocol would overlap them):

| Link | ours (layer by layer) | ours | SplitZip | raw BF16 received whole |
| --- | --- | --- | --- | --- |
| 1 Gb/s | 4408 ms | 4407 ms | 5717 ms | 7559 ms |
| 10 Gb/s | 483 ms | 482 ms | 620 ms | 794 ms |
| 25 Gb/s | 221 ms | 220 ms | 280 ms | 343 ms |
| 50 Gb/s | 134 ms | 133 ms | 167 ms | 193 ms |
| 100 Gb/s | 90 ms | 89 ms | 110 ms | 117 ms |
| 200 Gb/s | 69 ms | 67 ms | 82 ms | 80 ms |
| 400 Gb/s | 58 ms | 57 ms | 68 ms | 61 ms |

