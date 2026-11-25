# Conversation 2: mt-rag-govt-elser-512-100-20240611 (disk mode)

7 turns on D2 (routed by conversation); prompts grow from 949 to 7,503 tokens (interleaved: each turn's prompt is the previous one, its answer, then this turn's documents and question). Chunks are 256 tokens (16 vLLM blocks of 16); a chunk's name is LMCache's hash of its tokens and everything before them, so the same prefix has the same chunks on every turn and engine.

Legend for the strips (one character a chunk): P: `=` on P's GPU from an earlier turn, `#` computed and stored now, `+` computed, not stored (a partial chunk). D: `=` on D's GPU from an earlier turn, `M` pulled from memory, `S` pulled from disk, `+` computed by D (a partial chunk), `?` missed.

| turn | prompt | chunks + tail | P: own GPU / computed / stored | D: own GPU / loaded / computed | pulled | pull ms | D first token ms |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| [0](#turn-0) | 949 | 3 + 181 | 0 / 949 / 3 chunks | 0 / 768 / 181 | 3 chunks (disk 3) | 51.6 | 86.5 |
| [1](#turn-1) | 2,023 | 7 + 231 | 944 / 1,079 / 4 chunks | 1,072 / 720 / 231 | 3 chunks (disk 3) | 77.3 | 114.1 |
| [2](#turn-2) | 3,246 | 12 + 174 | 2,016 / 1,230 / 5 chunks | 2,144 / 928 / 174 | 4 chunks (disk 4) | 98.7 | 134.6 |
| [3](#turn-3) | 4,343 | 16 + 247 | 3,232 / 1,111 / 4 chunks | 3,360 / 736 / 247 | 3 chunks (disk 3) | 77.4 | 120.3 |
| [4](#turn-4) | 5,393 | 21 + 17 | 4,336 / 1,057 / 5 chunks | 4,464 / 912 / 17 | 4 chunks (disk 4) | 96.7 | 128.7 |
| [5](#turn-5) | 6,497 | 25 + 97 | 5,392 / 1,105 / 4 chunks | 5,520 / 880 / 97 | 4 chunks (disk 4) | 98.3 | 133.2 |
| [6](#turn-6) | 7,503 | 29 + 79 | 6,496 / 1,007 / 4 chunks | 6,624 / 800 / 79 | 4 chunks (disk 4) | 43.5 | 89.1 |

### One chunk, block by block: chunk 0 of turn 0 (`85f8dd3d9f0a22f8`)

Tokens 0-256 (system: 28; doc 62964e3a387835: 228). P computed its blocks on turn 0 and stored them as one chunk; D2 pulled the chunk into these GPU blocks:

| block | tokens | P's GPU block | vLLM hash | D's GPU block | D wrote it | same hash on D |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | 0-16 | 1 | `9f9c499d1c876b7e` | 1 | yes | True |
| 1 | 16-32 | 2 | `76827a2f3154693e` | 2 | yes | True |
| 2 | 32-48 | 3 | `a35ee4baf572391e` | 3 | yes | True |
| 3 | 48-64 | 4 | `e6dbd6621e81463a` | 4 | yes | True |
| 4 | 64-80 | 5 | `27acba166ab4b848` | 5 | yes | True |
| 5 | 80-96 | 6 | `13e9a85d6c60e734` | 6 | yes | True |
| 6 | 96-112 | 7 | `10f8d51cb8b3d184` | 7 | yes | True |
| 7 | 112-128 | 8 | `ef73b0a6aa8db24b` | 8 | yes | True |
| 8 | 128-144 | 9 | `f5228c6b14aff6f0` | 9 | yes | True |
| 9 | 144-160 | 10 | `6b4708c4da6918be` | 10 | yes | True |
| 10 | 160-176 | 11 | `4e3bc5526b3d9372` | 11 | yes | True |
| 11 | 176-192 | 12 | `213157c34e5d730f` | 12 | yes | True |
| 12 | 192-208 | 13 | `09547c085c09ceb1` | 13 | yes | True |
| 13 | 208-224 | 14 | `16d8cfabed6e2758` | 14 | yes | True |
| 14 | 224-240 | 15 | `1e1596807cf66c1a` | 15 | yes | True |
| 15 | 240-256 | 16 | `f948ec801d8ffca5` | 16 | yes | True |

## Turn 0

949 prompt tokens: 3 whole chunks and 181 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 0 tokens on its GPU, computed 949 and stored 3 new chunks; D2 held 0, loaded 768 from LMCache and computed 181.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |

```
P  ###+
D  SSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16; 16/16 hashes = P's | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32; 16/16 hashes = P's | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48; 16/16 hashes = P's | `882398f250b36d4a` stored on turn 0, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 3 | 768-949 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2 | 48-59 | P 49-59; D 49-60 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 1

2,023 prompt tokens: 7 whole chunks and 231 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 944 tokens on its GPU, computed 1,079 and stored 4 new chunks; D2 held 1,072, loaded 720 from LMCache and computed 231.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |
| answer 0 | 949-1077 | 128 | new this turn |
| newline 0 | 1077-1078 | 1 | new this turn |
| doc b4521304b77788 | 1078-1550 | 472 | new this turn |
| doc b4521304b77788 | 1550-2013 | 463 | new this turn |
| question 1 | 2013-2021 | 8 | new this turn |
| `assistant:` 1 | 2021-2023 | 2 | new this turn |

```
P  ===####+
D  ====SSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16 | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32 | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48 | `882398f250b36d4a` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2; answer 0 (new): 75 | 48-63 | P 49-59, 216-220; D 49-64 | `6225245ac6d7c372` stored on turn 1, on disk, out of memory | computed (its first 176 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0 (new): 53; newline 0 (new): 1; doc b4521304b77788 (new): 202 | 64-79 | P 221-236; D 65-80; 16/16 hashes = P's | `21cd5119cfb8110c` stored on turn 1, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 48 tokens on D's GPU already) |
| 5 | 1280-1536 | doc b4521304b77788 (new): 256 | 80-95 | P 237-252; D 81-96; 16/16 hashes = P's | `89c31d073cb3dd45` stored on turn 1, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 6 | 1536-1792 | doc b4521304b77788 (new): 14; doc b4521304b77788 (new): 242 | 96-111 | P 253-268; D 97-112; 16/16 hashes = P's | `fe165302915a1f84` stored on turn 1, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 7 | 1792-2023 | doc b4521304b77788 (new): 221; question 1 (new): 8; `assistant:` 1 (new): 2 | 112-126 | P 269-282; D 113-127 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 2

3,246 prompt tokens: 12 whole chunks and 174 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 2,016 tokens on its GPU, computed 1,230 and stored 5 new chunks; D2 held 2,144, loaded 928 from LMCache and computed 174.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |
| answer 0 | 949-1077 | 128 |  |
| newline 0 | 1077-1078 | 1 |  |
| doc b4521304b77788 | 1078-1550 | 472 |  |
| doc b4521304b77788 | 1550-2013 | 463 |  |
| question 1 | 2013-2021 | 8 |  |
| `assistant:` 1 | 2021-2023 | 2 |  |
| answer 1 | 2023-2151 | 128 | new this turn |
| newline 1 | 2151-2152 | 1 | new this turn |
| doc b4521304b77788 | 2152-2624 | 472 | new this turn |
| doc 8b076ccff5b504 | 2624-3238 | 614 | new this turn |
| question 2 | 3238-3244 | 6 | new this turn |
| `assistant:` 2 | 3244-3246 | 2 | new this turn |

```
P  =======#####+
D  ========SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16 | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32 | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48 | `882398f250b36d4a` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2; answer 0: 75 | 48-63 | P 49-59, 216-220; D 49-64 | `6225245ac6d7c372` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 53; newline 0: 1; doc b4521304b77788: 202 | 64-79 | P 221-236; D 65-80 | `21cd5119cfb8110c` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc b4521304b77788: 256 | 80-95 | P 237-252; D 81-96 | `89c31d073cb3dd45` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc b4521304b77788: 14; doc b4521304b77788: 242 | 96-111 | P 253-268; D 97-112 | `fe165302915a1f84` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc b4521304b77788: 221; question 1: 8; `assistant:` 1: 2; answer 1 (new): 25 | 112-127 | P 269-282, 457-458; D 113-128 | `7ad72aaf7a9563b2` stored on turn 2, on disk, out of memory | computed (its first 224 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | answer 1 (new): 103; newline 1 (new): 1; doc b4521304b77788 (new): 152 | 128-143 | P 459-474; D 129-144; 16/16 hashes = P's | `59e65ae8809a4cb6` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 96 tokens on D's GPU already) |
| 9 | 2304-2560 | doc b4521304b77788 (new): 256 | 144-159 | P 475-490; D 145-160; 16/16 hashes = P's | `1a5c24426a423af1` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 10 | 2560-2816 | doc b4521304b77788 (new): 64; doc 8b076ccff5b504 (new): 192 | 160-175 | P 491-506; D 161-176; 16/16 hashes = P's | `383d5afb2961c615` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 11 | 2816-3072 | doc 8b076ccff5b504 (new): 256 | 176-191 | P 507-522; D 177-192; 16/16 hashes = P's | `b3fe075f4455a184` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 12 | 3072-3246 | doc 8b076ccff5b504 (new): 166; question 2 (new): 6; `assistant:` 2 (new): 2 | 192-202 | P 523-532; D 193-203 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 3

4,343 prompt tokens: 16 whole chunks and 247 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 3,232 tokens on its GPU, computed 1,111 and stored 4 new chunks; D2 held 3,360, loaded 736 from LMCache and computed 247.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |
| answer 0 | 949-1077 | 128 |  |
| newline 0 | 1077-1078 | 1 |  |
| doc b4521304b77788 | 1078-1550 | 472 |  |
| doc b4521304b77788 | 1550-2013 | 463 |  |
| question 1 | 2013-2021 | 8 |  |
| `assistant:` 1 | 2021-2023 | 2 |  |
| answer 1 | 2023-2151 | 128 |  |
| newline 1 | 2151-2152 | 1 |  |
| doc b4521304b77788 | 2152-2624 | 472 |  |
| doc 8b076ccff5b504 | 2624-3238 | 614 |  |
| question 2 | 3238-3244 | 6 |  |
| `assistant:` 2 | 3244-3246 | 2 |  |
| answer 2 | 3246-3374 | 128 | new this turn |
| newline 2 | 3374-3375 | 1 | new this turn |
| doc b4521304b77788 | 3375-3847 | 472 | new this turn |
| doc 62964e3a387835 | 3847-4331 | 484 | new this turn |
| question 3 | 4331-4341 | 10 | new this turn |
| `assistant:` 3 | 4341-4343 | 2 | new this turn |

```
P  ============####+
D  =============SSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16 | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32 | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48 | `882398f250b36d4a` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2; answer 0: 75 | 48-63 | P 49-59, 216-220; D 49-64 | `6225245ac6d7c372` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 53; newline 0: 1; doc b4521304b77788: 202 | 64-79 | P 221-236; D 65-80 | `21cd5119cfb8110c` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc b4521304b77788: 256 | 80-95 | P 237-252; D 81-96 | `89c31d073cb3dd45` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc b4521304b77788: 14; doc b4521304b77788: 242 | 96-111 | P 253-268; D 97-112 | `fe165302915a1f84` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc b4521304b77788: 221; question 1: 8; `assistant:` 1: 2; answer 1: 25 | 112-127 | P 269-282, 457-458; D 113-128 | `7ad72aaf7a9563b2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | answer 1: 103; newline 1: 1; doc b4521304b77788: 152 | 128-143 | P 459-474; D 129-144 | `59e65ae8809a4cb6` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 9 | 2304-2560 | doc b4521304b77788: 256 | 144-159 | P 475-490; D 145-160 | `1a5c24426a423af1` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc b4521304b77788: 64; doc 8b076ccff5b504: 192 | 160-175 | P 491-506; D 161-176 | `383d5afb2961c615` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8b076ccff5b504: 256 | 176-191 | P 507-522; D 177-192 | `b3fe075f4455a184` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 8b076ccff5b504: 166; question 2: 6; `assistant:` 2: 2; answer 2 (new): 82 | 192-207 | P 523-532, 764-769; D 193-208 | `5ba380551af48729` stored on turn 3, on disk, out of memory | computed (its first 160 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 13 | 3328-3584 | answer 2 (new): 46; newline 2 (new): 1; doc b4521304b77788 (new): 209 | 208-223 | P 770-785; D 209-224; 16/16 hashes = P's | `b8a0a9df1331b56e` stored on turn 3, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 32 tokens on D's GPU already) |
| 14 | 3584-3840 | doc b4521304b77788 (new): 256 | 224-239 | P 786-801; D 225-240; 16/16 hashes = P's | `d99bd2026626c9be` stored on turn 3, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 15 | 3840-4096 | doc b4521304b77788 (new): 7; doc 62964e3a387835 (new): 249 | 240-255 | P 802-817; D 241-256; 16/16 hashes = P's | `4494d5f18d24b867` stored on turn 3, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 16 | 4096-4343 | doc 62964e3a387835 (new): 235; question 3 (new): 10; `assistant:` 3 (new): 2 | 256-271 | P 818-832; D 257-272 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 4

5,393 prompt tokens: 21 whole chunks and 17 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 4,336 tokens on its GPU, computed 1,057 and stored 5 new chunks; D2 held 4,464, loaded 912 from LMCache and computed 17.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |
| answer 0 | 949-1077 | 128 |  |
| newline 0 | 1077-1078 | 1 |  |
| doc b4521304b77788 | 1078-1550 | 472 |  |
| doc b4521304b77788 | 1550-2013 | 463 |  |
| question 1 | 2013-2021 | 8 |  |
| `assistant:` 1 | 2021-2023 | 2 |  |
| answer 1 | 2023-2151 | 128 |  |
| newline 1 | 2151-2152 | 1 |  |
| doc b4521304b77788 | 2152-2624 | 472 |  |
| doc 8b076ccff5b504 | 2624-3238 | 614 |  |
| question 2 | 3238-3244 | 6 |  |
| `assistant:` 2 | 3244-3246 | 2 |  |
| answer 2 | 3246-3374 | 128 |  |
| newline 2 | 3374-3375 | 1 |  |
| doc b4521304b77788 | 3375-3847 | 472 |  |
| doc 62964e3a387835 | 3847-4331 | 484 |  |
| question 3 | 4331-4341 | 10 |  |
| `assistant:` 3 | 4341-4343 | 2 |  |
| answer 3 | 4343-4471 | 128 | new this turn |
| newline 3 | 4471-4472 | 1 | new this turn |
| doc 6eeb0327b99b9a | 4472-4965 | 493 | new this turn |
| doc 7d4f6f216a8dff | 4965-5387 | 422 | new this turn |
| question 4 | 5387-5391 | 4 | new this turn |
| `assistant:` 4 | 5391-5393 | 2 | new this turn |

```
P  ================#####+
D  =================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16 | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32 | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48 | `882398f250b36d4a` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2; answer 0: 75 | 48-63 | P 49-59, 216-220; D 49-64 | `6225245ac6d7c372` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 53; newline 0: 1; doc b4521304b77788: 202 | 64-79 | P 221-236; D 65-80 | `21cd5119cfb8110c` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc b4521304b77788: 256 | 80-95 | P 237-252; D 81-96 | `89c31d073cb3dd45` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc b4521304b77788: 14; doc b4521304b77788: 242 | 96-111 | P 253-268; D 97-112 | `fe165302915a1f84` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc b4521304b77788: 221; question 1: 8; `assistant:` 1: 2; answer 1: 25 | 112-127 | P 269-282, 457-458; D 113-128 | `7ad72aaf7a9563b2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | answer 1: 103; newline 1: 1; doc b4521304b77788: 152 | 128-143 | P 459-474; D 129-144 | `59e65ae8809a4cb6` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 9 | 2304-2560 | doc b4521304b77788: 256 | 144-159 | P 475-490; D 145-160 | `1a5c24426a423af1` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc b4521304b77788: 64; doc 8b076ccff5b504: 192 | 160-175 | P 491-506; D 161-176 | `383d5afb2961c615` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8b076ccff5b504: 256 | 176-191 | P 507-522; D 177-192 | `b3fe075f4455a184` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 8b076ccff5b504: 166; question 2: 6; `assistant:` 2: 2; answer 2: 82 | 192-207 | P 523-532, 764-769; D 193-208 | `5ba380551af48729` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 13 | 3328-3584 | answer 2: 46; newline 2: 1; doc b4521304b77788: 209 | 208-223 | P 770-785; D 209-224 | `b8a0a9df1331b56e` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | doc b4521304b77788: 256 | 224-239 | P 786-801; D 225-240 | `d99bd2026626c9be` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc b4521304b77788: 7; doc 62964e3a387835: 249 | 240-255 | P 802-817; D 241-256 | `4494d5f18d24b867` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 62964e3a387835: 235; question 3: 10; `assistant:` 3: 2; answer 3 (new): 9 | 256-271 | P 818-832, 1013; D 257-272 | `a5971893925ee45b` stored on turn 4, on disk, out of memory | computed (its first 240 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 17 | 4352-4608 | answer 3 (new): 119; newline 3 (new): 1; doc 6eeb0327b99b9a (new): 136 | 272-287 | P 1014-1029; D 273-288; 16/16 hashes = P's | `7a933aee77f29e38` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 112 tokens on D's GPU already) |
| 18 | 4608-4864 | doc 6eeb0327b99b9a (new): 256 | 288-303 | P 1030-1045; D 289-304; 16/16 hashes = P's | `e21b98aea830a2b7` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 19 | 4864-5120 | doc 6eeb0327b99b9a (new): 101; doc 7d4f6f216a8dff (new): 155 | 304-319 | P 1046-1061; D 305-320; 16/16 hashes = P's | `d19792f5b7c0cfc9` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 20 | 5120-5376 | doc 7d4f6f216a8dff (new): 256 | 320-335 | P 1062-1077; D 321-336; 16/16 hashes = P's | `d85c5943b188bcc6` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 21 | 5376-5393 | doc 7d4f6f216a8dff (new): 11; question 4 (new): 4; `assistant:` 4 (new): 2 | 336-337 | P 1078; D 337-338 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 5

6,497 prompt tokens: 25 whole chunks and 97 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 5,392 tokens on its GPU, computed 1,105 and stored 4 new chunks; D2 held 5,520, loaded 880 from LMCache and computed 97.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |
| answer 0 | 949-1077 | 128 |  |
| newline 0 | 1077-1078 | 1 |  |
| doc b4521304b77788 | 1078-1550 | 472 |  |
| doc b4521304b77788 | 1550-2013 | 463 |  |
| question 1 | 2013-2021 | 8 |  |
| `assistant:` 1 | 2021-2023 | 2 |  |
| answer 1 | 2023-2151 | 128 |  |
| newline 1 | 2151-2152 | 1 |  |
| doc b4521304b77788 | 2152-2624 | 472 |  |
| doc 8b076ccff5b504 | 2624-3238 | 614 |  |
| question 2 | 3238-3244 | 6 |  |
| `assistant:` 2 | 3244-3246 | 2 |  |
| answer 2 | 3246-3374 | 128 |  |
| newline 2 | 3374-3375 | 1 |  |
| doc b4521304b77788 | 3375-3847 | 472 |  |
| doc 62964e3a387835 | 3847-4331 | 484 |  |
| question 3 | 4331-4341 | 10 |  |
| `assistant:` 3 | 4341-4343 | 2 |  |
| answer 3 | 4343-4471 | 128 |  |
| newline 3 | 4471-4472 | 1 |  |
| doc 6eeb0327b99b9a | 4472-4965 | 493 |  |
| doc 7d4f6f216a8dff | 4965-5387 | 422 |  |
| question 4 | 5387-5391 | 4 |  |
| `assistant:` 4 | 5391-5393 | 2 |  |
| answer 4 | 5393-5521 | 128 | new this turn |
| newline 4 | 5521-5522 | 1 | new this turn |
| doc d7fa2708b82825 | 5522-6018 | 496 | new this turn |
| doc d7fa2708b82825 | 6018-6490 | 472 | new this turn |
| question 5 | 6490-6495 | 5 | new this turn |
| `assistant:` 5 | 6495-6497 | 2 | new this turn |

```
P  =====================####+
D  =====================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16 | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32 | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48 | `882398f250b36d4a` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2; answer 0: 75 | 48-63 | P 49-59, 216-220; D 49-64 | `6225245ac6d7c372` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 53; newline 0: 1; doc b4521304b77788: 202 | 64-79 | P 221-236; D 65-80 | `21cd5119cfb8110c` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc b4521304b77788: 256 | 80-95 | P 237-252; D 81-96 | `89c31d073cb3dd45` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc b4521304b77788: 14; doc b4521304b77788: 242 | 96-111 | P 253-268; D 97-112 | `fe165302915a1f84` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc b4521304b77788: 221; question 1: 8; `assistant:` 1: 2; answer 1: 25 | 112-127 | P 269-282, 457-458; D 113-128 | `7ad72aaf7a9563b2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | answer 1: 103; newline 1: 1; doc b4521304b77788: 152 | 128-143 | P 459-474; D 129-144 | `59e65ae8809a4cb6` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 9 | 2304-2560 | doc b4521304b77788: 256 | 144-159 | P 475-490; D 145-160 | `1a5c24426a423af1` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc b4521304b77788: 64; doc 8b076ccff5b504: 192 | 160-175 | P 491-506; D 161-176 | `383d5afb2961c615` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8b076ccff5b504: 256 | 176-191 | P 507-522; D 177-192 | `b3fe075f4455a184` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 8b076ccff5b504: 166; question 2: 6; `assistant:` 2: 2; answer 2: 82 | 192-207 | P 523-532, 764-769; D 193-208 | `5ba380551af48729` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 13 | 3328-3584 | answer 2: 46; newline 2: 1; doc b4521304b77788: 209 | 208-223 | P 770-785; D 209-224 | `b8a0a9df1331b56e` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | doc b4521304b77788: 256 | 224-239 | P 786-801; D 225-240 | `d99bd2026626c9be` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc b4521304b77788: 7; doc 62964e3a387835: 249 | 240-255 | P 802-817; D 241-256 | `4494d5f18d24b867` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 62964e3a387835: 235; question 3: 10; `assistant:` 3: 2; answer 3: 9 | 256-271 | P 818-832, 1013; D 257-272 | `a5971893925ee45b` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 17 | 4352-4608 | answer 3: 119; newline 3: 1; doc 6eeb0327b99b9a: 136 | 272-287 | P 1014-1029; D 273-288 | `7a933aee77f29e38` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | doc 6eeb0327b99b9a: 256 | 288-303 | P 1030-1045; D 289-304 | `e21b98aea830a2b7` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 19 | 4864-5120 | doc 6eeb0327b99b9a: 101; doc 7d4f6f216a8dff: 155 | 304-319 | P 1046-1061; D 305-320 | `d19792f5b7c0cfc9` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 7d4f6f216a8dff: 256 | 320-335 | P 1062-1077; D 321-336 | `d85c5943b188bcc6` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 7d4f6f216a8dff: 11; question 4: 4; `assistant:` 4: 2; answer 4 (new): 128; newline 4 (new): 1; doc d7fa2708b82825 (new): 110 | 336-351 | P 1078, 1303-1317; D 337-352; 16/16 hashes = P's | `d90f84be6aa74482` stored on turn 5, on disk, out of memory | computed (its first 16 tokens on P's GPU already), **stored now** | **pulled from disk** (its first 144 tokens on D's GPU already) |
| 22 | 5632-5888 | doc d7fa2708b82825 (new): 256 | 352-367 | P 1318-1333; D 353-368; 16/16 hashes = P's | `554b75a6a5ea5d03` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 23 | 5888-6144 | doc d7fa2708b82825 (new): 130; doc d7fa2708b82825 (new): 126 | 368-383 | P 1334-1349; D 369-384; 16/16 hashes = P's | `0a853a0ddd2cc0ee` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 24 | 6144-6400 | doc d7fa2708b82825 (new): 256 | 384-399 | P 1350-1365; D 385-400; 16/16 hashes = P's | `4bf14990159a5f0a` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 25 | 6400-6497 | doc d7fa2708b82825 (new): 90; question 5 (new): 5; `assistant:` 5 (new): 2 | 400-406 | P 1366-1371; D 401-407 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 6

7,503 prompt tokens: 29 whole chunks and 79 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 6,496 tokens on its GPU, computed 1,007 and stored 4 new chunks; D2 held 6,624, loaded 800 from LMCache and computed 79.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 62964e3a387835 | 28-478 | 450 |  |
| doc b4521304b77788 | 478-939 | 461 |  |
| question 0 | 939-947 | 8 |  |
| `assistant:` 0 | 947-949 | 2 |  |
| answer 0 | 949-1077 | 128 |  |
| newline 0 | 1077-1078 | 1 |  |
| doc b4521304b77788 | 1078-1550 | 472 |  |
| doc b4521304b77788 | 1550-2013 | 463 |  |
| question 1 | 2013-2021 | 8 |  |
| `assistant:` 1 | 2021-2023 | 2 |  |
| answer 1 | 2023-2151 | 128 |  |
| newline 1 | 2151-2152 | 1 |  |
| doc b4521304b77788 | 2152-2624 | 472 |  |
| doc 8b076ccff5b504 | 2624-3238 | 614 |  |
| question 2 | 3238-3244 | 6 |  |
| `assistant:` 2 | 3244-3246 | 2 |  |
| answer 2 | 3246-3374 | 128 |  |
| newline 2 | 3374-3375 | 1 |  |
| doc b4521304b77788 | 3375-3847 | 472 |  |
| doc 62964e3a387835 | 3847-4331 | 484 |  |
| question 3 | 4331-4341 | 10 |  |
| `assistant:` 3 | 4341-4343 | 2 |  |
| answer 3 | 4343-4471 | 128 |  |
| newline 3 | 4471-4472 | 1 |  |
| doc 6eeb0327b99b9a | 4472-4965 | 493 |  |
| doc 7d4f6f216a8dff | 4965-5387 | 422 |  |
| question 4 | 5387-5391 | 4 |  |
| `assistant:` 4 | 5391-5393 | 2 |  |
| answer 4 | 5393-5521 | 128 |  |
| newline 4 | 5521-5522 | 1 |  |
| doc d7fa2708b82825 | 5522-6018 | 496 |  |
| doc d7fa2708b82825 | 6018-6490 | 472 |  |
| question 5 | 6490-6495 | 5 |  |
| `assistant:` 5 | 6495-6497 | 2 |  |
| answer 5 | 6497-6625 | 128 | new this turn |
| newline 5 | 6625-6626 | 1 | new this turn |
| doc 5fefc2cc192686 | 6626-7039 | 413 | new this turn |
| doc 9a8c6d9e77c91e | 7039-7495 | 456 | new this turn |
| question 6 | 7495-7501 | 6 | new this turn |
| `assistant:` 6 | 7501-7503 | 2 | new this turn |

```
P  =========================####+
D  =========================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 62964e3a387835: 228 | 0-15 | P 1-16; D 1-16 | `85f8dd3d9f0a22f8` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 62964e3a387835: 222; doc b4521304b77788: 34 | 16-31 | P 17-32; D 17-32 | `8d2f3ac802435f4f` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc b4521304b77788: 256 | 32-47 | P 33-48; D 33-48 | `882398f250b36d4a` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc b4521304b77788: 171; question 0: 8; `assistant:` 0: 2; answer 0: 75 | 48-63 | P 49-59, 216-220; D 49-64 | `6225245ac6d7c372` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 53; newline 0: 1; doc b4521304b77788: 202 | 64-79 | P 221-236; D 65-80 | `21cd5119cfb8110c` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc b4521304b77788: 256 | 80-95 | P 237-252; D 81-96 | `89c31d073cb3dd45` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc b4521304b77788: 14; doc b4521304b77788: 242 | 96-111 | P 253-268; D 97-112 | `fe165302915a1f84` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc b4521304b77788: 221; question 1: 8; `assistant:` 1: 2; answer 1: 25 | 112-127 | P 269-282, 457-458; D 113-128 | `7ad72aaf7a9563b2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | answer 1: 103; newline 1: 1; doc b4521304b77788: 152 | 128-143 | P 459-474; D 129-144 | `59e65ae8809a4cb6` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 9 | 2304-2560 | doc b4521304b77788: 256 | 144-159 | P 475-490; D 145-160 | `1a5c24426a423af1` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc b4521304b77788: 64; doc 8b076ccff5b504: 192 | 160-175 | P 491-506; D 161-176 | `383d5afb2961c615` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8b076ccff5b504: 256 | 176-191 | P 507-522; D 177-192 | `b3fe075f4455a184` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 8b076ccff5b504: 166; question 2: 6; `assistant:` 2: 2; answer 2: 82 | 192-207 | P 523-532, 764-769; D 193-208 | `5ba380551af48729` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 13 | 3328-3584 | answer 2: 46; newline 2: 1; doc b4521304b77788: 209 | 208-223 | P 770-785; D 209-224 | `b8a0a9df1331b56e` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | doc b4521304b77788: 256 | 224-239 | P 786-801; D 225-240 | `d99bd2026626c9be` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc b4521304b77788: 7; doc 62964e3a387835: 249 | 240-255 | P 802-817; D 241-256 | `4494d5f18d24b867` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 62964e3a387835: 235; question 3: 10; `assistant:` 3: 2; answer 3: 9 | 256-271 | P 818-832, 1013; D 257-272 | `a5971893925ee45b` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 17 | 4352-4608 | answer 3: 119; newline 3: 1; doc 6eeb0327b99b9a: 136 | 272-287 | P 1014-1029; D 273-288 | `7a933aee77f29e38` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | doc 6eeb0327b99b9a: 256 | 288-303 | P 1030-1045; D 289-304 | `e21b98aea830a2b7` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 19 | 4864-5120 | doc 6eeb0327b99b9a: 101; doc 7d4f6f216a8dff: 155 | 304-319 | P 1046-1061; D 305-320 | `d19792f5b7c0cfc9` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 7d4f6f216a8dff: 256 | 320-335 | P 1062-1077; D 321-336 | `d85c5943b188bcc6` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 7d4f6f216a8dff: 11; question 4: 4; `assistant:` 4: 2; answer 4: 128; newline 4: 1; doc d7fa2708b82825: 110 | 336-351 | P 1078, 1303-1317; D 337-352 | `d90f84be6aa74482` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 22 | 5632-5888 | doc d7fa2708b82825: 256 | 352-367 | P 1318-1333; D 353-368 | `554b75a6a5ea5d03` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 23 | 5888-6144 | doc d7fa2708b82825: 130; doc d7fa2708b82825: 126 | 368-383 | P 1334-1349; D 369-384 | `0a853a0ddd2cc0ee` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 24 | 6144-6400 | doc d7fa2708b82825: 256 | 384-399 | P 1350-1365; D 385-400 | `4bf14990159a5f0a` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc d7fa2708b82825: 90; question 5: 5; `assistant:` 5: 2; answer 5 (new): 128; newline 5 (new): 1; doc 5fefc2cc192686 (new): 30 | 400-415 | P 1366-1371, 1502-1511; D 401-416; 16/16 hashes = P's | `4ae2dafc0e9e84d1` stored on turn 6, on disk, out of memory | computed (its first 96 tokens on P's GPU already), **stored now** | **pulled from disk** (its first 224 tokens on D's GPU already) |
| 26 | 6656-6912 | doc 5fefc2cc192686 (new): 256 | 416-431 | P 1512-1527; D 417-432; 16/16 hashes = P's | `481464c33618395b` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 27 | 6912-7168 | doc 5fefc2cc192686 (new): 127; doc 9a8c6d9e77c91e (new): 129 | 432-447 | P 1528-1543; D 433-448; 16/16 hashes = P's | `5022b363f69dc27b` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 28 | 7168-7424 | doc 9a8c6d9e77c91e (new): 256 | 448-463 | P 1544-1559; D 449-464; 16/16 hashes = P's | `11ededca43b8c20c` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 29 | 7424-7503 | doc 9a8c6d9e77c91e (new): 71; question 6 (new): 6; `assistant:` 6 (new): 2 | 464-468 | P 1560-1563; D 465-469 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |
