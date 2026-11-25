# Conversation 3: mt-rag-fiqa-beir-elser-512-100-20240501 (memory mode)

7 turns on D3 (routed by conversation); prompts grow from 337 to 3,102 tokens (interleaved: each turn's prompt is the previous one, its answer, then this turn's documents and question). Chunks are 256 tokens (16 vLLM blocks of 16); a chunk's name is LMCache's hash of its tokens and everything before them, so the same prefix has the same chunks on every turn and engine.

Legend for the strips (one character a chunk): P: `=` on P's GPU from an earlier turn, `#` computed and stored now, `+` computed, not stored (a partial chunk). D: `=` on D's GPU from an earlier turn, `M` pulled from memory, `S` pulled from disk, `+` computed by D (a partial chunk), `?` missed.

| turn | prompt | chunks + tail | P: own GPU / computed / stored | D: own GPU / loaded / computed | pulled | pull ms | D first token ms |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| [0](#turn-0) | 337 | 1 + 81 | 16 / 321 / 1 chunk | 0 / 256 / 81 | 1 chunks (memory 1) | 18.2 | 42.9 |
| [1](#turn-1) | 846 | 3 + 78 | 336 / 510 / 2 chunks | 464 / 304 / 78 | 2 chunks (memory 2) | 8.8 | 35.1 |
| [2](#turn-2) | 1,226 | 4 + 202 | 832 / 394 / 1 chunk | 960 / 64 / 202 | 1 chunks (memory 1) | 7.3 | 43.0 |
| [3](#turn-3) | 1,639 | 6 + 103 | 1,216 / 423 / 2 chunks | 1,344 / 192 / 103 | 1 chunks (memory 1) | 8.4 | 36.1 |
| [4](#turn-4) | 2,336 | 9 + 32 | 1,632 / 704 / 3 chunks | 1,760 / 544 / 32 | 3 chunks (memory 3) | 8.7 | 36.1 |
| [5](#turn-5) | 2,718 | 10 + 158 | 2,336 / 382 / 1 chunk | 2,448 / 112 / 158 | 1 chunks (memory 1) | 8.7 | 43.0 |
| [6](#turn-6) | 3,102 | 12 + 30 | 2,704 / 398 / 2 chunks | 2,832 / 240 / 30 | 1 chunks (memory 1) | 9.4 | 36.2 |

### One chunk, block by block: chunk 0 of turn 0 (`47dc4d47ce072861`)

Tokens 0-256 (system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52). P computed its blocks on turn 0 and stored them as one chunk; D3 pulled the chunk into these GPU blocks:

| block | tokens | P's GPU block | vLLM hash | D's GPU block | D wrote it | same hash on D |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | 0-16 | 1 | `9f9c499d1c876b7e` | 1 | yes | True |
| 1 | 16-32 | 117 | `f0aa562f7a5736de` | 2 | yes | True |
| 2 | 32-48 | 118 | `9df4288d52128763` | 3 | yes | True |
| 3 | 48-64 | 119 | `22346d454b037dde` | 4 | yes | True |
| 4 | 64-80 | 120 | `df34a823be5b1b7f` | 5 | yes | True |
| 5 | 80-96 | 121 | `96c1d75a7e880e94` | 6 | yes | True |
| 6 | 96-112 | 122 | `d908d6e813f0a6d4` | 7 | yes | True |
| 7 | 112-128 | 123 | `78a9824aea4d7338` | 8 | yes | True |
| 8 | 128-144 | 124 | `bae2e9c4c69e49f7` | 9 | yes | True |
| 9 | 144-160 | 125 | `ee282b7939ed7f2b` | 10 | yes | True |
| 10 | 160-176 | 126 | `aa096713a93fb29f` | 11 | yes | True |
| 11 | 176-192 | 127 | `68cbb151452211bd` | 12 | yes | True |
| 12 | 192-208 | 128 | `435b65f22ecdb757` | 13 | yes | True |
| 13 | 208-224 | 129 | `59ab3ffc2a0ed74d` | 14 | yes | True |
| 14 | 224-240 | 130 | `bb55580321998834` | 15 | yes | True |
| 15 | 240-256 | 131 | `e48f383f10716db6` | 16 | yes | True |

## Turn 0

337 prompt tokens: 1 whole chunk and 81 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 16 tokens on its GPU, computed 321 and stored 1 new chunk; D3 held 0, loaded 256 from LMCache and computed 81.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |

```
P  #+
D  M+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16; 16/16 hashes = P's | `47dc4d47ce072861` stored on turn 0 | computed (its first 16 tokens on P's GPU already), **stored now** | **pulled from memory** |
| 1 | 256-337 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2 | 16-21 | P 132-136; D 17-22 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 1

846 prompt tokens: 3 whole chunks and 78 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 336 tokens on its GPU, computed 510 and stored 2 new chunks; D3 held 464, loaded 304 from LMCache and computed 78.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |
| answer 0 | 337-465 | 128 | new this turn |
| newline 0 | 465-466 | 1 | new this turn |
| doc 416727-0-1356 | 466-775 | 309 | new this turn |
| doc 418610-0-264 | 775-837 | 62 | new this turn |
| question 1 | 837-844 | 7 | new this turn |
| `assistant:` 1 | 844-846 | 2 | new this turn |

```
P  =##+
D  =MM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16 | `47dc4d47ce072861` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2; answer 0 (new): 128; newline 0 (new): 1; doc 416727-0-1356 (new): 46 | 16-31 | P 132-136 … 189-195 (16 blocks); D 17-32; 16/16 hashes = P's | `c3de14602f3d4726` stored on turn 1 | computed (its first 80 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 208 tokens on D's GPU already) |
| 2 | 512-768 | doc 416727-0-1356 (new): 256 | 32-47 | P 196-211; D 33-48; 16/16 hashes = P's | `cd6084602efd4e78` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 3 | 768-846 | doc 416727-0-1356 (new): 7; doc 418610-0-264 (new): 62; question 1 (new): 7; `assistant:` 1 (new): 2 | 48-52 | P 212-215; D 49-53 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 2

1,226 prompt tokens: 4 whole chunks and 202 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 832 tokens on its GPU, computed 394 and stored 1 new chunk; D3 held 960, loaded 64 from LMCache and computed 202.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |
| answer 0 | 337-465 | 128 |  |
| newline 0 | 465-466 | 1 |  |
| doc 416727-0-1356 | 466-775 | 309 |  |
| doc 418610-0-264 | 775-837 | 62 |  |
| question 1 | 837-844 | 7 |  |
| `assistant:` 1 | 844-846 | 2 |  |
| answer 1 | 846-974 | 128 | new this turn |
| newline 1 | 974-975 | 1 | new this turn |
| doc 346760-0-445 | 975-1072 | 97 | new this turn |
| doc 192515-0-701 | 1072-1219 | 147 | new this turn |
| question 2 | 1219-1224 | 5 | new this turn |
| `assistant:` 2 | 1224-1226 | 2 | new this turn |

```
P  ===#+
D  ===M+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16 | `47dc4d47ce072861` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 416727-0-1356: 46 | 16-31 | P 132-136 … 189-195 (16 blocks); D 17-32 | `c3de14602f3d4726` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 2 | 512-768 | doc 416727-0-1356: 256 | 32-47 | P 196-211; D 33-48 | `cd6084602efd4e78` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 416727-0-1356: 7; doc 418610-0-264: 62; question 1: 7; `assistant:` 1: 2; answer 1 (new): 128; newline 1 (new): 1; doc 346760-0-445 (new): 49 | 48-63 | P 212-215 … 435-444 (16 blocks); D 49-64; 16/16 hashes = P's | `98e036362aa62d61` stored on turn 2 | computed (its first 64 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 192 tokens on D's GPU already) |
| 4 | 1024-1226 | doc 346760-0-445 (new): 48; doc 192515-0-701 (new): 147; question 2 (new): 5; `assistant:` 2 (new): 2 | 64-76 | P 445-456; D 65-77 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 3

1,639 prompt tokens: 6 whole chunks and 103 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 1,216 tokens on its GPU, computed 423 and stored 2 new chunks; D3 held 1,344, loaded 192 from LMCache and computed 103.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |
| answer 0 | 337-465 | 128 |  |
| newline 0 | 465-466 | 1 |  |
| doc 416727-0-1356 | 466-775 | 309 |  |
| doc 418610-0-264 | 775-837 | 62 |  |
| question 1 | 837-844 | 7 |  |
| `assistant:` 1 | 844-846 | 2 |  |
| answer 1 | 846-974 | 128 |  |
| newline 1 | 974-975 | 1 |  |
| doc 346760-0-445 | 975-1072 | 97 |  |
| doc 192515-0-701 | 1072-1219 | 147 |  |
| question 2 | 1219-1224 | 5 |  |
| `assistant:` 2 | 1224-1226 | 2 |  |
| answer 2 | 1226-1354 | 128 | new this turn |
| newline 2 | 1354-1355 | 1 | new this turn |
| doc 587508-0-651 | 1355-1492 | 137 | new this turn |
| doc 366249-0-656 | 1492-1630 | 138 | new this turn |
| question 3 | 1630-1637 | 7 | new this turn |
| `assistant:` 3 | 1637-1639 | 2 | new this turn |

```
P  ====##+
D  =====M+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16 | `47dc4d47ce072861` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 416727-0-1356: 46 | 16-31 | P 132-136 … 189-195 (16 blocks); D 17-32 | `c3de14602f3d4726` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 2 | 512-768 | doc 416727-0-1356: 256 | 32-47 | P 196-211; D 33-48 | `cd6084602efd4e78` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 416727-0-1356: 7; doc 418610-0-264: 62; question 1: 7; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 346760-0-445: 49 | 48-63 | P 212-215 … 435-444 (16 blocks); D 49-64 | `98e036362aa62d61` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 346760-0-445: 48; doc 192515-0-701: 147; question 2: 5; `assistant:` 2: 2; answer 2 (new): 54 | 64-79 | P 445-456 … 741 (16 blocks); D 65-80 | `b6960f33b0ee6772` stored on turn 3 | computed (its first 192 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | answer 2 (new): 74; newline 2 (new): 1; doc 587508-0-651 (new): 137; doc 366249-0-656 (new): 44 | 80-95 | P 742-757; D 81-96; 16/16 hashes = P's | `e6db7091ab30e9e9` stored on turn 3 | computed, **stored now** | **pulled from memory** (its first 64 tokens on D's GPU already) |
| 6 | 1536-1639 | doc 366249-0-656 (new): 94; question 3 (new): 7; `assistant:` 3 (new): 2 | 96-102 | P 758-763; D 97-103 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 4

2,336 prompt tokens: 9 whole chunks and 32 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 1,632 tokens on its GPU, computed 704 and stored 3 new chunks; D3 held 1,760, loaded 544 from LMCache and computed 32.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |
| answer 0 | 337-465 | 128 |  |
| newline 0 | 465-466 | 1 |  |
| doc 416727-0-1356 | 466-775 | 309 |  |
| doc 418610-0-264 | 775-837 | 62 |  |
| question 1 | 837-844 | 7 |  |
| `assistant:` 1 | 844-846 | 2 |  |
| answer 1 | 846-974 | 128 |  |
| newline 1 | 974-975 | 1 |  |
| doc 346760-0-445 | 975-1072 | 97 |  |
| doc 192515-0-701 | 1072-1219 | 147 |  |
| question 2 | 1219-1224 | 5 |  |
| `assistant:` 2 | 1224-1226 | 2 |  |
| answer 2 | 1226-1354 | 128 |  |
| newline 2 | 1354-1355 | 1 |  |
| doc 587508-0-651 | 1355-1492 | 137 |  |
| doc 366249-0-656 | 1492-1630 | 138 |  |
| question 3 | 1630-1637 | 7 |  |
| `assistant:` 3 | 1637-1639 | 2 |  |
| answer 3 | 1639-1767 | 128 | new this turn |
| newline 3 | 1767-1768 | 1 | new this turn |
| doc 554465-0-780 | 1768-1942 | 174 | new this turn |
| doc 477310-0-1809 | 1942-2327 | 385 | new this turn |
| question 4 | 2327-2334 | 7 | new this turn |
| `assistant:` 4 | 2334-2336 | 2 | new this turn |

```
P  ======###+
D  ======MMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16 | `47dc4d47ce072861` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 416727-0-1356: 46 | 16-31 | P 132-136 … 189-195 (16 blocks); D 17-32 | `c3de14602f3d4726` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 2 | 512-768 | doc 416727-0-1356: 256 | 32-47 | P 196-211; D 33-48 | `cd6084602efd4e78` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 416727-0-1356: 7; doc 418610-0-264: 62; question 1: 7; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 346760-0-445: 49 | 48-63 | P 212-215 … 435-444 (16 blocks); D 49-64 | `98e036362aa62d61` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 346760-0-445: 48; doc 192515-0-701: 147; question 2: 5; `assistant:` 2: 2; answer 2: 54 | 64-79 | P 445-456 … 741 (16 blocks); D 65-80 | `b6960f33b0ee6772` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | answer 2: 74; newline 2: 1; doc 587508-0-651: 137; doc 366249-0-656: 44 | 80-95 | P 742-757; D 81-96 | `e6db7091ab30e9e9` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 6 | 1536-1792 | doc 366249-0-656: 94; question 3: 7; `assistant:` 3: 2; answer 3 (new): 128; newline 3 (new): 1; doc 554465-0-780 (new): 24 | 96-111 | P 758-763, 969-978; D 97-112; 16/16 hashes = P's | `1e8ed7f848e45caf` stored on turn 4 | computed (its first 96 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 224 tokens on D's GPU already) |
| 7 | 1792-2048 | doc 554465-0-780 (new): 150; doc 477310-0-1809 (new): 106 | 112-127 | P 979-994; D 113-128; 16/16 hashes = P's | `5cc2c4dd6717f930` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 8 | 2048-2304 | doc 477310-0-1809 (new): 256 | 128-143 | P 995-1010; D 129-144; 16/16 hashes = P's | `712fee3710dd597a` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 9 | 2304-2336 | doc 477310-0-1809 (new): 23; question 4 (new): 7; `assistant:` 4 (new): 2 | 144-145 | P 1011-1012; D 145-146 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 5

2,718 prompt tokens: 10 whole chunks and 158 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 2,336 tokens on its GPU, computed 382 and stored 1 new chunk; D3 held 2,448, loaded 112 from LMCache and computed 158.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |
| answer 0 | 337-465 | 128 |  |
| newline 0 | 465-466 | 1 |  |
| doc 416727-0-1356 | 466-775 | 309 |  |
| doc 418610-0-264 | 775-837 | 62 |  |
| question 1 | 837-844 | 7 |  |
| `assistant:` 1 | 844-846 | 2 |  |
| answer 1 | 846-974 | 128 |  |
| newline 1 | 974-975 | 1 |  |
| doc 346760-0-445 | 975-1072 | 97 |  |
| doc 192515-0-701 | 1072-1219 | 147 |  |
| question 2 | 1219-1224 | 5 |  |
| `assistant:` 2 | 1224-1226 | 2 |  |
| answer 2 | 1226-1354 | 128 |  |
| newline 2 | 1354-1355 | 1 |  |
| doc 587508-0-651 | 1355-1492 | 137 |  |
| doc 366249-0-656 | 1492-1630 | 138 |  |
| question 3 | 1630-1637 | 7 |  |
| `assistant:` 3 | 1637-1639 | 2 |  |
| answer 3 | 1639-1767 | 128 |  |
| newline 3 | 1767-1768 | 1 |  |
| doc 554465-0-780 | 1768-1942 | 174 |  |
| doc 477310-0-1809 | 1942-2327 | 385 |  |
| question 4 | 2327-2334 | 7 |  |
| `assistant:` 4 | 2334-2336 | 2 |  |
| answer 4 | 2336-2464 | 128 | new this turn |
| newline 4 | 2464-2465 | 1 | new this turn |
| doc 597850-0-335 | 2465-2545 | 80 | new this turn |
| doc 467535-0-727 | 2545-2706 | 161 | new this turn |
| question 5 | 2706-2716 | 10 | new this turn |
| `assistant:` 5 | 2716-2718 | 2 | new this turn |

```
P  =========#+
D  =========M+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16 | `47dc4d47ce072861` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 416727-0-1356: 46 | 16-31 | P 132-136 … 189-195 (16 blocks); D 17-32 | `c3de14602f3d4726` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 2 | 512-768 | doc 416727-0-1356: 256 | 32-47 | P 196-211; D 33-48 | `cd6084602efd4e78` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 416727-0-1356: 7; doc 418610-0-264: 62; question 1: 7; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 346760-0-445: 49 | 48-63 | P 212-215 … 435-444 (16 blocks); D 49-64 | `98e036362aa62d61` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 346760-0-445: 48; doc 192515-0-701: 147; question 2: 5; `assistant:` 2: 2; answer 2: 54 | 64-79 | P 445-456 … 741 (16 blocks); D 65-80 | `b6960f33b0ee6772` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | answer 2: 74; newline 2: 1; doc 587508-0-651: 137; doc 366249-0-656: 44 | 80-95 | P 742-757; D 81-96 | `e6db7091ab30e9e9` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 6 | 1536-1792 | doc 366249-0-656: 94; question 3: 7; `assistant:` 3: 2; answer 3: 128; newline 3: 1; doc 554465-0-780: 24 | 96-111 | P 758-763, 969-978; D 97-112 | `1e8ed7f848e45caf` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 7 | 1792-2048 | doc 554465-0-780: 150; doc 477310-0-1809: 106 | 112-127 | P 979-994; D 113-128 | `5cc2c4dd6717f930` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 477310-0-1809: 256 | 128-143 | P 995-1010; D 129-144 | `712fee3710dd597a` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 477310-0-1809: 23; question 4: 7; `assistant:` 4: 2; answer 4 (new): 128; newline 4 (new): 1; doc 597850-0-335 (new): 80; doc 467535-0-727 (new): 15 | 144-159 | P 1011-1012, 1280-1293; D 145-160; 16/16 hashes = P's | `125142856548d9f2` stored on turn 5 | computed (its first 32 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 144 tokens on D's GPU already) |
| 10 | 2560-2718 | doc 467535-0-727 (new): 146; question 5 (new): 10; `assistant:` 5 (new): 2 | 160-169 | P 1294-1302; D 161-170 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 6

3,102 prompt tokens: 12 whole chunks and 30 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 2,704 tokens on its GPU, computed 398 and stored 2 new chunks; D3 held 2,832, loaded 240 from LMCache and computed 30.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 471123-0-798 | 28-204 | 176 |  |
| doc 414940-0-474 | 204-322 | 118 |  |
| question 0 | 322-335 | 13 |  |
| `assistant:` 0 | 335-337 | 2 |  |
| answer 0 | 337-465 | 128 |  |
| newline 0 | 465-466 | 1 |  |
| doc 416727-0-1356 | 466-775 | 309 |  |
| doc 418610-0-264 | 775-837 | 62 |  |
| question 1 | 837-844 | 7 |  |
| `assistant:` 1 | 844-846 | 2 |  |
| answer 1 | 846-974 | 128 |  |
| newline 1 | 974-975 | 1 |  |
| doc 346760-0-445 | 975-1072 | 97 |  |
| doc 192515-0-701 | 1072-1219 | 147 |  |
| question 2 | 1219-1224 | 5 |  |
| `assistant:` 2 | 1224-1226 | 2 |  |
| answer 2 | 1226-1354 | 128 |  |
| newline 2 | 1354-1355 | 1 |  |
| doc 587508-0-651 | 1355-1492 | 137 |  |
| doc 366249-0-656 | 1492-1630 | 138 |  |
| question 3 | 1630-1637 | 7 |  |
| `assistant:` 3 | 1637-1639 | 2 |  |
| answer 3 | 1639-1767 | 128 |  |
| newline 3 | 1767-1768 | 1 |  |
| doc 554465-0-780 | 1768-1942 | 174 |  |
| doc 477310-0-1809 | 1942-2327 | 385 |  |
| question 4 | 2327-2334 | 7 |  |
| `assistant:` 4 | 2334-2336 | 2 |  |
| answer 4 | 2336-2464 | 128 |  |
| newline 4 | 2464-2465 | 1 |  |
| doc 597850-0-335 | 2465-2545 | 80 |  |
| doc 467535-0-727 | 2545-2706 | 161 |  |
| question 5 | 2706-2716 | 10 |  |
| `assistant:` 5 | 2716-2718 | 2 |  |
| answer 5 | 2718-2846 | 128 | new this turn |
| newline 5 | 2846-2847 | 1 | new this turn |
| doc 241308-0-414 | 2847-2942 | 95 | new this turn |
| doc 558617-0-616 | 2942-3088 | 146 | new this turn |
| question 6 | 3088-3100 | 12 | new this turn |
| `assistant:` 6 | 3100-3102 | 2 | new this turn |

```
P  ==========##+
D  ===========M+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 471123-0-798: 176; doc 414940-0-474: 52 | 0-15 | P 1, 117-131; D 1-16 | `47dc4d47ce072861` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 414940-0-474: 66; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 416727-0-1356: 46 | 16-31 | P 132-136 … 189-195 (16 blocks); D 17-32 | `c3de14602f3d4726` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 2 | 512-768 | doc 416727-0-1356: 256 | 32-47 | P 196-211; D 33-48 | `cd6084602efd4e78` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 416727-0-1356: 7; doc 418610-0-264: 62; question 1: 7; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 346760-0-445: 49 | 48-63 | P 212-215 … 435-444 (16 blocks); D 49-64 | `98e036362aa62d61` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 346760-0-445: 48; doc 192515-0-701: 147; question 2: 5; `assistant:` 2: 2; answer 2: 54 | 64-79 | P 445-456 … 741 (16 blocks); D 65-80 | `b6960f33b0ee6772` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | answer 2: 74; newline 2: 1; doc 587508-0-651: 137; doc 366249-0-656: 44 | 80-95 | P 742-757; D 81-96 | `e6db7091ab30e9e9` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 6 | 1536-1792 | doc 366249-0-656: 94; question 3: 7; `assistant:` 3: 2; answer 3: 128; newline 3: 1; doc 554465-0-780: 24 | 96-111 | P 758-763, 969-978; D 97-112 | `1e8ed7f848e45caf` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 7 | 1792-2048 | doc 554465-0-780: 150; doc 477310-0-1809: 106 | 112-127 | P 979-994; D 113-128 | `5cc2c4dd6717f930` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 477310-0-1809: 256 | 128-143 | P 995-1010; D 129-144 | `712fee3710dd597a` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 477310-0-1809: 23; question 4: 7; `assistant:` 4: 2; answer 4: 128; newline 4: 1; doc 597850-0-335: 80; doc 467535-0-727: 15 | 144-159 | P 1011-1012, 1280-1293; D 145-160 | `125142856548d9f2` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 10 | 2560-2816 | doc 467535-0-727: 146; question 5: 10; `assistant:` 5: 2; answer 5 (new): 98 | 160-175 | P 1294-1302, 1478-1484; D 161-176 | `90a80fb01fe6b9c3` stored on turn 6 | computed (its first 144 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 11 | 2816-3072 | answer 5 (new): 30; newline 5 (new): 1; doc 241308-0-414 (new): 95; doc 558617-0-616 (new): 130 | 176-191 | P 1485-1500; D 177-192; 16/16 hashes = P's | `604ff888ee28bd9d` stored on turn 6 | computed, **stored now** | **pulled from memory** (its first 16 tokens on D's GPU already) |
| 12 | 3072-3102 | doc 558617-0-616 (new): 16; question 6 (new): 12; `assistant:` 6 (new): 2 | 192-193 | P 1501; D 193-194 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |
