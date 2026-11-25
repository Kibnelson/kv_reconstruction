# Conversation 1: mt-rag-govt-elser-512-100-20240611 (memory mode)

8 turns on D1 (routed by conversation); prompts grow from 925 to 9,765 tokens (interleaved: each turn's prompt is the previous one, its answer, then this turn's documents and question). Chunks are 256 tokens (16 vLLM blocks of 16); a chunk's name is LMCache's hash of its tokens and everything before them, so the same prefix has the same chunks on every turn and engine.

Legend for the strips (one character a chunk): P: `=` on P's GPU from an earlier turn, `#` computed and stored now, `+` computed, not stored (a partial chunk). D: `=` on D's GPU from an earlier turn, `M` pulled from memory, `S` pulled from disk, `+` computed by D (a partial chunk), `?` missed.

| turn | prompt | chunks + tail | P: own GPU / computed / stored | D: own GPU / loaded / computed | pulled | pull ms | D first token ms |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| [0](#turn-0) | 925 | 3 + 157 | 0 / 925 / 3 chunks | 0 / 768 / 157 | 3 chunks (memory 3) | 21.1 | 54.2 |
| [1](#turn-1) | 2,336 | 9 + 32 | 912 / 1,424 / 6 chunks | 1,040 / 1,264 / 32 | 5 chunks (memory 5) | 16.3 | 43.8 |
| [2](#turn-2) | 3,921 | 15 + 81 | 2,336 / 1,585 / 6 chunks | 2,448 / 1,392 / 81 | 6 chunks (memory 6) | 15.2 | 48.8 |
| [3](#turn-3) | 5,032 | 19 + 168 | 3,920 / 1,112 / 4 chunks | 4,048 / 816 / 168 | 4 chunks (memory 4) | 10.3 | 51.2 |
| [4](#turn-4) | 6,995 | 27 + 83 | 5,024 / 1,971 / 8 chunks | 5,152 / 1,760 / 83 | 7 chunks (memory 7) | 14.3 | 50.1 |
| [5](#turn-5) | 7,585 | 29 + 161 | 6,992 / 593 / 2 chunks | 7,120 / 304 / 161 | 2 chunks (memory 2) | 10.7 | 59.5 |
| [6](#turn-6) | 8,705 | 34 + 1 | 7,584 / 1,121 / 5 chunks | 7,712 / 992 / 1 | 4 chunks (memory 4) | 14.1 | 46.1 |
| [7](#turn-7) | 9,765 | 38 + 37 | 8,704 / 1,061 / 4 chunks | 8,832 / 896 / 37 | 4 chunks (memory 4) | 13.9 | 54.6 |

### One chunk, block by block: chunk 0 of turn 0 (`1dd495890c85343c`)

Tokens 0-256 (system: 28; doc c8db6e06ff4666: 228). P computed its blocks on turn 0 and stored them as one chunk; D1 pulled the chunk into these GPU blocks:

| block | tokens | P's GPU block | vLLM hash | D's GPU block | D wrote it | same hash on D |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | 0-16 | 1 | `9f9c499d1c876b7e` | 1 | yes | True |
| 1 | 16-32 | 2 | `76827a2f3154693e` | 2 | yes | True |
| 2 | 32-48 | 3 | `fb3b0d8a267bb4ed` | 3 | yes | True |
| 3 | 48-64 | 4 | `8d92f86df79acbc2` | 4 | yes | True |
| 4 | 64-80 | 5 | `a7500780232c0a2d` | 5 | yes | True |
| 5 | 80-96 | 6 | `c51d16a97f22208e` | 6 | yes | True |
| 6 | 96-112 | 7 | `cea451b28ed2eb08` | 7 | yes | True |
| 7 | 112-128 | 8 | `a038e48664bc438e` | 8 | yes | True |
| 8 | 128-144 | 9 | `957caa342b15b4e3` | 9 | yes | True |
| 9 | 144-160 | 10 | `3effd4342bb857a2` | 10 | yes | True |
| 10 | 160-176 | 11 | `883c60777d5e6d1d` | 11 | yes | True |
| 11 | 176-192 | 12 | `7ec2c6abd00cf383` | 12 | yes | True |
| 12 | 192-208 | 13 | `86a4aa2913df969a` | 13 | yes | True |
| 13 | 208-224 | 14 | `e9a9f2d05d921c73` | 14 | yes | True |
| 14 | 224-240 | 15 | `f90fdb8523fe5179` | 15 | yes | True |
| 15 | 240-256 | 16 | `549e5b84eeeceb71` | 16 | yes | True |

## Turn 0

925 prompt tokens: 3 whole chunks and 157 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 0 tokens on its GPU, computed 925 and stored 3 new chunks; D1 held 0, loaded 768 from LMCache and computed 157.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |

```
P  ###+
D  MMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16; 16/16 hashes = P's | `1dd495890c85343c` stored on turn 0 | computed, **stored now** | **pulled from memory** |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32; 16/16 hashes = P's | `931e9f699f8d5ce5` stored on turn 0 | computed, **stored now** | **pulled from memory** |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48; 16/16 hashes = P's | `e6a2246ebc7818db` stored on turn 0 | computed, **stored now** | **pulled from memory** |
| 3 | 768-925 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2 | 48-57 | P 49-57; D 97-106 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 1

2,336 prompt tokens: 9 whole chunks and 32 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 912 tokens on its GPU, computed 1,424 and stored 6 new chunks; D1 held 1,040, loaded 1,264 from LMCache and computed 32.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 | new this turn |
| newline 0 | 1053-1054 | 1 | new this turn |
| doc c8db6e06ff4666 | 1054-1515 | 461 | new this turn |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 | new this turn |
| doc 88aca7ee734372 | 1954-2326 | 372 | new this turn |
| question 1 | 2326-2334 | 8 | new this turn |
| `assistant:` 1 | 2334-2336 | 2 | new this turn |

```
P  ===######+
D  ====MMMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0 (new): 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | computed (its first 144 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0 (new): 29; newline 0 (new): 1; doc c8db6e06ff4666 (new): 226 | 64-79 | P 291-306; D 123, 125-139; 16/16 hashes = P's | `ef56131fbdafd648` stored on turn 1 | computed, **stored now** | **pulled from memory** (its first 16 tokens on D's GPU already) |
| 5 | 1280-1536 | doc c8db6e06ff4666 (new): 235; doc 7d4d64e7f6aff1 (new): 21 | 80-95 | P 307-322; D 140-155; 16/16 hashes = P's | `8c391d82c20faefd` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1 (new): 256 | 96-111 | P 323-338; D 156-171; 16/16 hashes = P's | `29b263badcec2b7e` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1 (new): 162; doc 88aca7ee734372 (new): 94 | 112-127 | P 339-354; D 172-187; 16/16 hashes = P's | `af1c4476de067ff9` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 8 | 2048-2304 | doc 88aca7ee734372 (new): 256 | 128-143 | P 355-370; D 188-203; 16/16 hashes = P's | `62545b3002b3e711` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 9 | 2304-2336 | doc 88aca7ee734372 (new): 22; question 1 (new): 8; `assistant:` 1 (new): 2 | 144-145 | P 371-372; D 257-258 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 2

3,921 prompt tokens: 15 whole chunks and 81 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 2,336 tokens on its GPU, computed 1,585 and stored 6 new chunks; D1 held 2,448, loaded 1,392 from LMCache and computed 81.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 |  |
| newline 0 | 1053-1054 | 1 |  |
| doc c8db6e06ff4666 | 1054-1515 | 461 |  |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 |  |
| doc 88aca7ee734372 | 1954-2326 | 372 |  |
| question 1 | 2326-2334 | 8 |  |
| `assistant:` 1 | 2334-2336 | 2 |  |
| answer 1 | 2336-2464 | 128 | new this turn |
| newline 1 | 2464-2465 | 1 | new this turn |
| doc 8bb77f30210c5f | 2465-2924 | 459 | new this turn |
| doc 7f851a7621a884 | 2924-3418 | 494 | new this turn |
| doc 8bb77f30210c5f | 3418-3910 | 492 | new this turn |
| question 2 | 3910-3919 | 9 | new this turn |
| `assistant:` 2 | 3919-3921 | 2 | new this turn |

```
P  =========######+
D  =========MMMMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0: 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 29; newline 0: 1; doc c8db6e06ff4666: 226 | 64-79 | P 291-306; D 123, 125-139 | `ef56131fbdafd648` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc c8db6e06ff4666: 235; doc 7d4d64e7f6aff1: 21 | 80-95 | P 307-322; D 140-155 | `8c391d82c20faefd` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1: 256 | 96-111 | P 323-338; D 156-171 | `29b263badcec2b7e` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1: 162; doc 88aca7ee734372: 94 | 112-127 | P 339-354; D 172-187 | `af1c4476de067ff9` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 88aca7ee734372: 256 | 128-143 | P 355-370; D 188-203 | `62545b3002b3e711` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 88aca7ee734372: 22; question 1: 8; `assistant:` 1: 2; answer 1 (new): 128; newline 1 (new): 1; doc 8bb77f30210c5f (new): 95 | 144-159 | P 371-372, 534-547; D 257-259 … 276-280 (16 blocks); 16/16 hashes = P's | `30b12248dc08664f` stored on turn 2 | computed (its first 32 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 144 tokens on D's GPU already) |
| 10 | 2560-2816 | doc 8bb77f30210c5f (new): 256 | 160-175 | P 548-563; D 281-296; 16/16 hashes = P's | `dda183e62d75bc8d` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 11 | 2816-3072 | doc 8bb77f30210c5f (new): 108; doc 7f851a7621a884 (new): 148 | 176-191 | P 564-579; D 297-312; 16/16 hashes = P's | `9627ab104bc4ba56` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 12 | 3072-3328 | doc 7f851a7621a884 (new): 256 | 192-207 | P 580-595; D 313-328; 16/16 hashes = P's | `b07bc63b3fc66ef0` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 13 | 3328-3584 | doc 7f851a7621a884 (new): 90; doc 8bb77f30210c5f (new): 166 | 208-223 | P 596-611; D 329-344; 16/16 hashes = P's | `f3016ff95618a267` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 14 | 3584-3840 | doc 8bb77f30210c5f (new): 256 | 224-239 | P 612-627; D 345-360; 16/16 hashes = P's | `d049aa24668c0744` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 15 | 3840-3921 | doc 8bb77f30210c5f (new): 70; question 2 (new): 9; `assistant:` 2 (new): 2 | 240-245 | P 628-632; D 361-366 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 3

5,032 prompt tokens: 19 whole chunks and 168 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 3,920 tokens on its GPU, computed 1,112 and stored 4 new chunks; D1 held 4,048, loaded 816 from LMCache and computed 168.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 |  |
| newline 0 | 1053-1054 | 1 |  |
| doc c8db6e06ff4666 | 1054-1515 | 461 |  |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 |  |
| doc 88aca7ee734372 | 1954-2326 | 372 |  |
| question 1 | 2326-2334 | 8 |  |
| `assistant:` 1 | 2334-2336 | 2 |  |
| answer 1 | 2336-2464 | 128 |  |
| newline 1 | 2464-2465 | 1 |  |
| doc 8bb77f30210c5f | 2465-2924 | 459 |  |
| doc 7f851a7621a884 | 2924-3418 | 494 |  |
| doc 8bb77f30210c5f | 3418-3910 | 492 |  |
| question 2 | 3910-3919 | 9 |  |
| `assistant:` 2 | 3919-3921 | 2 |  |
| answer 2 | 3921-4049 | 128 | new this turn |
| newline 2 | 4049-4050 | 1 | new this turn |
| doc 35b1d4a5ce1b03 | 4050-4565 | 515 | new this turn |
| doc 2806a7d8f0775d | 4565-5022 | 457 | new this turn |
| question 3 | 5022-5030 | 8 | new this turn |
| `assistant:` 3 | 5030-5032 | 2 | new this turn |

```
P  ===============####+
D  ===============MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0: 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 29; newline 0: 1; doc c8db6e06ff4666: 226 | 64-79 | P 291-306; D 123, 125-139 | `ef56131fbdafd648` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc c8db6e06ff4666: 235; doc 7d4d64e7f6aff1: 21 | 80-95 | P 307-322; D 140-155 | `8c391d82c20faefd` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1: 256 | 96-111 | P 323-338; D 156-171 | `29b263badcec2b7e` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1: 162; doc 88aca7ee734372: 94 | 112-127 | P 339-354; D 172-187 | `af1c4476de067ff9` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 88aca7ee734372: 256 | 128-143 | P 355-370; D 188-203 | `62545b3002b3e711` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 88aca7ee734372: 22; question 1: 8; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 8bb77f30210c5f: 95 | 144-159 | P 371-372, 534-547; D 257-259 … 276-280 (16 blocks) | `30b12248dc08664f` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 10 | 2560-2816 | doc 8bb77f30210c5f: 256 | 160-175 | P 548-563; D 281-296 | `dda183e62d75bc8d` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8bb77f30210c5f: 108; doc 7f851a7621a884: 148 | 176-191 | P 564-579; D 297-312 | `9627ab104bc4ba56` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 7f851a7621a884: 256 | 192-207 | P 580-595; D 313-328 | `b07bc63b3fc66ef0` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 7f851a7621a884: 90; doc 8bb77f30210c5f: 166 | 208-223 | P 596-611; D 329-344 | `f3016ff95618a267` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 14 | 3584-3840 | doc 8bb77f30210c5f: 256 | 224-239 | P 612-627; D 345-360 | `d049aa24668c0744` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc 8bb77f30210c5f: 70; question 2: 9; `assistant:` 2: 2; answer 2 (new): 128; newline 2 (new): 1; doc 35b1d4a5ce1b03 (new): 46 | 240-255 | P 628-632, 833-843; D 361-366 … 480-482 (16 blocks); 16/16 hashes = P's | `aea46b321df5ee34` stored on turn 3 | computed (its first 80 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 208 tokens on D's GPU already) |
| 16 | 4096-4352 | doc 35b1d4a5ce1b03 (new): 256 | 256-271 | P 844-859; D 483-498; 16/16 hashes = P's | `4b79490b2e8e77e2` stored on turn 3 | computed, **stored now** | **pulled from memory** |
| 17 | 4352-4608 | doc 35b1d4a5ce1b03 (new): 213; doc 2806a7d8f0775d (new): 43 | 272-287 | P 860-875; D 499-514; 16/16 hashes = P's | `4ffb45c105c97bf2` stored on turn 3 | computed, **stored now** | **pulled from memory** |
| 18 | 4608-4864 | doc 2806a7d8f0775d (new): 256 | 288-303 | P 876-891; D 515-530; 16/16 hashes = P's | `b9e95cd1762d03bb` stored on turn 3 | computed, **stored now** | **pulled from memory** |
| 19 | 4864-5032 | doc 2806a7d8f0775d (new): 158; question 3 (new): 8; `assistant:` 3 (new): 2 | 304-314 | P 892-901; D 531-541 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 4

6,995 prompt tokens: 27 whole chunks and 83 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 5,024 tokens on its GPU, computed 1,971 and stored 8 new chunks; D1 held 5,152, loaded 1,760 from LMCache and computed 83.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 |  |
| newline 0 | 1053-1054 | 1 |  |
| doc c8db6e06ff4666 | 1054-1515 | 461 |  |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 |  |
| doc 88aca7ee734372 | 1954-2326 | 372 |  |
| question 1 | 2326-2334 | 8 |  |
| `assistant:` 1 | 2334-2336 | 2 |  |
| answer 1 | 2336-2464 | 128 |  |
| newline 1 | 2464-2465 | 1 |  |
| doc 8bb77f30210c5f | 2465-2924 | 459 |  |
| doc 7f851a7621a884 | 2924-3418 | 494 |  |
| doc 8bb77f30210c5f | 3418-3910 | 492 |  |
| question 2 | 3910-3919 | 9 |  |
| `assistant:` 2 | 3919-3921 | 2 |  |
| answer 2 | 3921-4049 | 128 |  |
| newline 2 | 4049-4050 | 1 |  |
| doc 35b1d4a5ce1b03 | 4050-4565 | 515 |  |
| doc 2806a7d8f0775d | 4565-5022 | 457 |  |
| question 3 | 5022-5030 | 8 |  |
| `assistant:` 3 | 5030-5032 | 2 |  |
| answer 3 | 5032-5160 | 128 | new this turn |
| newline 3 | 5160-5161 | 1 | new this turn |
| doc 031693dc1490c3 | 5161-5652 | 491 | new this turn |
| doc a46d1e3a08f352 | 5652-6137 | 485 | new this turn |
| doc a46d1e3a08f352 | 6137-6633 | 496 | new this turn |
| doc 031693dc1490c3 | 6633-6987 | 354 | new this turn |
| question 4 | 6987-6993 | 6 | new this turn |
| `assistant:` 4 | 6993-6995 | 2 | new this turn |

```
P  ===================########+
D  ====================MMMMMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0: 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 29; newline 0: 1; doc c8db6e06ff4666: 226 | 64-79 | P 291-306; D 123, 125-139 | `ef56131fbdafd648` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc c8db6e06ff4666: 235; doc 7d4d64e7f6aff1: 21 | 80-95 | P 307-322; D 140-155 | `8c391d82c20faefd` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1: 256 | 96-111 | P 323-338; D 156-171 | `29b263badcec2b7e` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1: 162; doc 88aca7ee734372: 94 | 112-127 | P 339-354; D 172-187 | `af1c4476de067ff9` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 88aca7ee734372: 256 | 128-143 | P 355-370; D 188-203 | `62545b3002b3e711` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 88aca7ee734372: 22; question 1: 8; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 8bb77f30210c5f: 95 | 144-159 | P 371-372, 534-547; D 257-259 … 276-280 (16 blocks) | `30b12248dc08664f` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 10 | 2560-2816 | doc 8bb77f30210c5f: 256 | 160-175 | P 548-563; D 281-296 | `dda183e62d75bc8d` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8bb77f30210c5f: 108; doc 7f851a7621a884: 148 | 176-191 | P 564-579; D 297-312 | `9627ab104bc4ba56` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 7f851a7621a884: 256 | 192-207 | P 580-595; D 313-328 | `b07bc63b3fc66ef0` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 7f851a7621a884: 90; doc 8bb77f30210c5f: 166 | 208-223 | P 596-611; D 329-344 | `f3016ff95618a267` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 14 | 3584-3840 | doc 8bb77f30210c5f: 256 | 224-239 | P 612-627; D 345-360 | `d049aa24668c0744` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc 8bb77f30210c5f: 70; question 2: 9; `assistant:` 2: 2; answer 2: 128; newline 2: 1; doc 35b1d4a5ce1b03: 46 | 240-255 | P 628-632, 833-843; D 361-366 … 480-482 (16 blocks) | `aea46b321df5ee34` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 16 | 4096-4352 | doc 35b1d4a5ce1b03: 256 | 256-271 | P 844-859; D 483-498 | `4b79490b2e8e77e2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 35b1d4a5ce1b03: 213; doc 2806a7d8f0775d: 43 | 272-287 | P 860-875; D 499-514 | `4ffb45c105c97bf2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 18 | 4608-4864 | doc 2806a7d8f0775d: 256 | 288-303 | P 876-891; D 515-530 | `b9e95cd1762d03bb` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 19 | 4864-5120 | doc 2806a7d8f0775d: 158; question 3: 8; `assistant:` 3: 2; answer 3 (new): 88 | 304-319 | P 892-901, 1079-1084; D 531-542 … 609 (16 blocks) | `2cc0c05a2fb893ac` stored on turn 4 | computed (its first 160 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 20 | 5120-5376 | answer 3 (new): 40; newline 3 (new): 1; doc 031693dc1490c3 (new): 215 | 320-335 | P 1085-1100; D 611, 613, 616-629; 16/16 hashes = P's | `18c543a30447b7c9` stored on turn 4 | computed, **stored now** | **pulled from memory** (its first 32 tokens on D's GPU already) |
| 21 | 5376-5632 | doc 031693dc1490c3 (new): 256 | 336-351 | P 1101-1116; D 630-645; 16/16 hashes = P's | `8650ee985a5d877c` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 22 | 5632-5888 | doc 031693dc1490c3 (new): 20; doc a46d1e3a08f352 (new): 236 | 352-367 | P 1117-1132; D 646-661; 16/16 hashes = P's | `1a4cedb3fb65b249` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 23 | 5888-6144 | doc a46d1e3a08f352 (new): 249; doc a46d1e3a08f352 (new): 7 | 368-383 | P 1133-1148; D 662-677; 16/16 hashes = P's | `7eacadaf10b20818` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 24 | 6144-6400 | doc a46d1e3a08f352 (new): 256 | 384-399 | P 1149-1164; D 678-693; 16/16 hashes = P's | `c080a47ca3563089` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 25 | 6400-6656 | doc a46d1e3a08f352 (new): 233; doc 031693dc1490c3 (new): 23 | 400-415 | P 1165-1180; D 694-709; 16/16 hashes = P's | `59c7c924ce9a2e4a` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 26 | 6656-6912 | doc 031693dc1490c3 (new): 256 | 416-431 | P 1181-1196; D 710-725; 16/16 hashes = P's | `93287088554b79c4` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 27 | 6912-6995 | doc 031693dc1490c3 (new): 75; question 4 (new): 6; `assistant:` 4 (new): 2 | 432-437 | P 1197-1201; D 726-731 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 5

7,585 prompt tokens: 29 whole chunks and 161 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 6,992 tokens on its GPU, computed 593 and stored 2 new chunks; D1 held 7,120, loaded 304 from LMCache and computed 161.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 |  |
| newline 0 | 1053-1054 | 1 |  |
| doc c8db6e06ff4666 | 1054-1515 | 461 |  |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 |  |
| doc 88aca7ee734372 | 1954-2326 | 372 |  |
| question 1 | 2326-2334 | 8 |  |
| `assistant:` 1 | 2334-2336 | 2 |  |
| answer 1 | 2336-2464 | 128 |  |
| newline 1 | 2464-2465 | 1 |  |
| doc 8bb77f30210c5f | 2465-2924 | 459 |  |
| doc 7f851a7621a884 | 2924-3418 | 494 |  |
| doc 8bb77f30210c5f | 3418-3910 | 492 |  |
| question 2 | 3910-3919 | 9 |  |
| `assistant:` 2 | 3919-3921 | 2 |  |
| answer 2 | 3921-4049 | 128 |  |
| newline 2 | 4049-4050 | 1 |  |
| doc 35b1d4a5ce1b03 | 4050-4565 | 515 |  |
| doc 2806a7d8f0775d | 4565-5022 | 457 |  |
| question 3 | 5022-5030 | 8 |  |
| `assistant:` 3 | 5030-5032 | 2 |  |
| answer 3 | 5032-5160 | 128 |  |
| newline 3 | 5160-5161 | 1 |  |
| doc 031693dc1490c3 | 5161-5652 | 491 |  |
| doc a46d1e3a08f352 | 5652-6137 | 485 |  |
| doc a46d1e3a08f352 | 6137-6633 | 496 |  |
| doc 031693dc1490c3 | 6633-6987 | 354 |  |
| question 4 | 6987-6993 | 6 |  |
| `assistant:` 4 | 6993-6995 | 2 |  |
| answer 4 | 6995-7123 | 128 | new this turn |
| newline 4 | 7123-7124 | 1 | new this turn |
| doc 7d4d64e7f6aff1 | 7124-7566 | 442 | new this turn |
| question 5 | 7566-7583 | 17 | new this turn |
| `assistant:` 5 | 7583-7585 | 2 | new this turn |

```
P  ===========================##+
D  ===========================MM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0: 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 29; newline 0: 1; doc c8db6e06ff4666: 226 | 64-79 | P 291-306; D 123, 125-139 | `ef56131fbdafd648` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc c8db6e06ff4666: 235; doc 7d4d64e7f6aff1: 21 | 80-95 | P 307-322; D 140-155 | `8c391d82c20faefd` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1: 256 | 96-111 | P 323-338; D 156-171 | `29b263badcec2b7e` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1: 162; doc 88aca7ee734372: 94 | 112-127 | P 339-354; D 172-187 | `af1c4476de067ff9` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 88aca7ee734372: 256 | 128-143 | P 355-370; D 188-203 | `62545b3002b3e711` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 88aca7ee734372: 22; question 1: 8; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 8bb77f30210c5f: 95 | 144-159 | P 371-372, 534-547; D 257-259 … 276-280 (16 blocks) | `30b12248dc08664f` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 10 | 2560-2816 | doc 8bb77f30210c5f: 256 | 160-175 | P 548-563; D 281-296 | `dda183e62d75bc8d` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8bb77f30210c5f: 108; doc 7f851a7621a884: 148 | 176-191 | P 564-579; D 297-312 | `9627ab104bc4ba56` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 7f851a7621a884: 256 | 192-207 | P 580-595; D 313-328 | `b07bc63b3fc66ef0` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 7f851a7621a884: 90; doc 8bb77f30210c5f: 166 | 208-223 | P 596-611; D 329-344 | `f3016ff95618a267` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 14 | 3584-3840 | doc 8bb77f30210c5f: 256 | 224-239 | P 612-627; D 345-360 | `d049aa24668c0744` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc 8bb77f30210c5f: 70; question 2: 9; `assistant:` 2: 2; answer 2: 128; newline 2: 1; doc 35b1d4a5ce1b03: 46 | 240-255 | P 628-632, 833-843; D 361-366 … 480-482 (16 blocks) | `aea46b321df5ee34` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 16 | 4096-4352 | doc 35b1d4a5ce1b03: 256 | 256-271 | P 844-859; D 483-498 | `4b79490b2e8e77e2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 35b1d4a5ce1b03: 213; doc 2806a7d8f0775d: 43 | 272-287 | P 860-875; D 499-514 | `4ffb45c105c97bf2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 18 | 4608-4864 | doc 2806a7d8f0775d: 256 | 288-303 | P 876-891; D 515-530 | `b9e95cd1762d03bb` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 19 | 4864-5120 | doc 2806a7d8f0775d: 158; question 3: 8; `assistant:` 3: 2; answer 3: 88 | 304-319 | P 892-901, 1079-1084; D 531-542 … 609 (16 blocks) | `2cc0c05a2fb893ac` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 20 | 5120-5376 | answer 3: 40; newline 3: 1; doc 031693dc1490c3: 215 | 320-335 | P 1085-1100; D 611, 613, 616-629 | `18c543a30447b7c9` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 21 | 5376-5632 | doc 031693dc1490c3: 256 | 336-351 | P 1101-1116; D 630-645 | `8650ee985a5d877c` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 031693dc1490c3: 20; doc a46d1e3a08f352: 236 | 352-367 | P 1117-1132; D 646-661 | `1a4cedb3fb65b249` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 23 | 5888-6144 | doc a46d1e3a08f352: 249; doc a46d1e3a08f352: 7 | 368-383 | P 1133-1148; D 662-677 | `7eacadaf10b20818` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 24 | 6144-6400 | doc a46d1e3a08f352: 256 | 384-399 | P 1149-1164; D 678-693 | `c080a47ca3563089` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc a46d1e3a08f352: 233; doc 031693dc1490c3: 23 | 400-415 | P 1165-1180; D 694-709 | `59c7c924ce9a2e4a` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 031693dc1490c3: 256 | 416-431 | P 1181-1196; D 710-725 | `93287088554b79c4` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | doc 031693dc1490c3: 75; question 4: 6; `assistant:` 4: 2; answer 4 (new): 128; newline 4 (new): 1; doc 7d4d64e7f6aff1 (new): 44 | 432-447 | P 1197-1201, 1372-1382; D 726-732 … 818-819 (16 blocks); 16/16 hashes = P's | `aa5ec94636aa1f2c` stored on turn 5 | computed (its first 80 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 208 tokens on D's GPU already) |
| 28 | 7168-7424 | doc 7d4d64e7f6aff1 (new): 256 | 448-463 | P 1383-1398; D 820-835; 16/16 hashes = P's | `250f8fca7969c39b` stored on turn 5 | computed, **stored now** | **pulled from memory** |
| 29 | 7424-7585 | doc 7d4d64e7f6aff1 (new): 142; question 5 (new): 17; `assistant:` 5 (new): 2 | 464-474 | P 1399-1408; D 836-846 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 6

8,705 prompt tokens: 34 whole chunks and 1 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 7,584 tokens on its GPU, computed 1,121 and stored 5 new chunks; D1 held 7,712, loaded 992 from LMCache and computed 1.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 |  |
| newline 0 | 1053-1054 | 1 |  |
| doc c8db6e06ff4666 | 1054-1515 | 461 |  |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 |  |
| doc 88aca7ee734372 | 1954-2326 | 372 |  |
| question 1 | 2326-2334 | 8 |  |
| `assistant:` 1 | 2334-2336 | 2 |  |
| answer 1 | 2336-2464 | 128 |  |
| newline 1 | 2464-2465 | 1 |  |
| doc 8bb77f30210c5f | 2465-2924 | 459 |  |
| doc 7f851a7621a884 | 2924-3418 | 494 |  |
| doc 8bb77f30210c5f | 3418-3910 | 492 |  |
| question 2 | 3910-3919 | 9 |  |
| `assistant:` 2 | 3919-3921 | 2 |  |
| answer 2 | 3921-4049 | 128 |  |
| newline 2 | 4049-4050 | 1 |  |
| doc 35b1d4a5ce1b03 | 4050-4565 | 515 |  |
| doc 2806a7d8f0775d | 4565-5022 | 457 |  |
| question 3 | 5022-5030 | 8 |  |
| `assistant:` 3 | 5030-5032 | 2 |  |
| answer 3 | 5032-5160 | 128 |  |
| newline 3 | 5160-5161 | 1 |  |
| doc 031693dc1490c3 | 5161-5652 | 491 |  |
| doc a46d1e3a08f352 | 5652-6137 | 485 |  |
| doc a46d1e3a08f352 | 6137-6633 | 496 |  |
| doc 031693dc1490c3 | 6633-6987 | 354 |  |
| question 4 | 6987-6993 | 6 |  |
| `assistant:` 4 | 6993-6995 | 2 |  |
| answer 4 | 6995-7123 | 128 |  |
| newline 4 | 7123-7124 | 1 |  |
| doc 7d4d64e7f6aff1 | 7124-7566 | 442 |  |
| question 5 | 7566-7583 | 17 |  |
| `assistant:` 5 | 7583-7585 | 2 |  |
| answer 5 | 7585-7713 | 128 | new this turn |
| newline 5 | 7713-7714 | 1 | new this turn |
| doc 4d3eba5aeb31c3 | 7714-8184 | 470 | new this turn |
| doc 89c8998bbb932f | 8184-8692 | 508 | new this turn |
| question 6 | 8692-8703 | 11 | new this turn |
| `assistant:` 6 | 8703-8705 | 2 | new this turn |

```
P  =============================#####+
D  ==============================MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0: 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 29; newline 0: 1; doc c8db6e06ff4666: 226 | 64-79 | P 291-306; D 123, 125-139 | `ef56131fbdafd648` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc c8db6e06ff4666: 235; doc 7d4d64e7f6aff1: 21 | 80-95 | P 307-322; D 140-155 | `8c391d82c20faefd` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1: 256 | 96-111 | P 323-338; D 156-171 | `29b263badcec2b7e` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1: 162; doc 88aca7ee734372: 94 | 112-127 | P 339-354; D 172-187 | `af1c4476de067ff9` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 88aca7ee734372: 256 | 128-143 | P 355-370; D 188-203 | `62545b3002b3e711` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 88aca7ee734372: 22; question 1: 8; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 8bb77f30210c5f: 95 | 144-159 | P 371-372, 534-547; D 257-259 … 276-280 (16 blocks) | `30b12248dc08664f` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 10 | 2560-2816 | doc 8bb77f30210c5f: 256 | 160-175 | P 548-563; D 281-296 | `dda183e62d75bc8d` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8bb77f30210c5f: 108; doc 7f851a7621a884: 148 | 176-191 | P 564-579; D 297-312 | `9627ab104bc4ba56` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 7f851a7621a884: 256 | 192-207 | P 580-595; D 313-328 | `b07bc63b3fc66ef0` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 7f851a7621a884: 90; doc 8bb77f30210c5f: 166 | 208-223 | P 596-611; D 329-344 | `f3016ff95618a267` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 14 | 3584-3840 | doc 8bb77f30210c5f: 256 | 224-239 | P 612-627; D 345-360 | `d049aa24668c0744` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc 8bb77f30210c5f: 70; question 2: 9; `assistant:` 2: 2; answer 2: 128; newline 2: 1; doc 35b1d4a5ce1b03: 46 | 240-255 | P 628-632, 833-843; D 361-366 … 480-482 (16 blocks) | `aea46b321df5ee34` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 16 | 4096-4352 | doc 35b1d4a5ce1b03: 256 | 256-271 | P 844-859; D 483-498 | `4b79490b2e8e77e2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 35b1d4a5ce1b03: 213; doc 2806a7d8f0775d: 43 | 272-287 | P 860-875; D 499-514 | `4ffb45c105c97bf2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 18 | 4608-4864 | doc 2806a7d8f0775d: 256 | 288-303 | P 876-891; D 515-530 | `b9e95cd1762d03bb` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 19 | 4864-5120 | doc 2806a7d8f0775d: 158; question 3: 8; `assistant:` 3: 2; answer 3: 88 | 304-319 | P 892-901, 1079-1084; D 531-542 … 609 (16 blocks) | `2cc0c05a2fb893ac` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 20 | 5120-5376 | answer 3: 40; newline 3: 1; doc 031693dc1490c3: 215 | 320-335 | P 1085-1100; D 611, 613, 616-629 | `18c543a30447b7c9` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 21 | 5376-5632 | doc 031693dc1490c3: 256 | 336-351 | P 1101-1116; D 630-645 | `8650ee985a5d877c` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 031693dc1490c3: 20; doc a46d1e3a08f352: 236 | 352-367 | P 1117-1132; D 646-661 | `1a4cedb3fb65b249` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 23 | 5888-6144 | doc a46d1e3a08f352: 249; doc a46d1e3a08f352: 7 | 368-383 | P 1133-1148; D 662-677 | `7eacadaf10b20818` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 24 | 6144-6400 | doc a46d1e3a08f352: 256 | 384-399 | P 1149-1164; D 678-693 | `c080a47ca3563089` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc a46d1e3a08f352: 233; doc 031693dc1490c3: 23 | 400-415 | P 1165-1180; D 694-709 | `59c7c924ce9a2e4a` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 031693dc1490c3: 256 | 416-431 | P 1181-1196; D 710-725 | `93287088554b79c4` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | doc 031693dc1490c3: 75; question 4: 6; `assistant:` 4: 2; answer 4: 128; newline 4: 1; doc 7d4d64e7f6aff1: 44 | 432-447 | P 1197-1201, 1372-1382; D 726-732 … 818-819 (16 blocks) | `aa5ec94636aa1f2c` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 28 | 7168-7424 | doc 7d4d64e7f6aff1: 256 | 448-463 | P 1383-1398; D 820-835 | `250f8fca7969c39b` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 29 | 7424-7680 | doc 7d4d64e7f6aff1: 142; question 5: 17; `assistant:` 5: 2; answer 5 (new): 95 | 464-479 | P 1399-1408, 1564-1569; D 836-847 … 917 (16 blocks) | `209669262a3aa2f3` stored on turn 6 | computed (its first 160 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 30 | 7680-7936 | answer 5 (new): 33; newline 5 (new): 1; doc 4d3eba5aeb31c3 (new): 222 | 480-495 | P 1570-1585; D 919, 921, 923-936; 16/16 hashes = P's | `a1314184fdfde0ea` stored on turn 6 | computed, **stored now** | **pulled from memory** (its first 32 tokens on D's GPU already) |
| 31 | 7936-8192 | doc 4d3eba5aeb31c3 (new): 248; doc 89c8998bbb932f (new): 8 | 496-511 | P 1586-1601; D 937-952; 16/16 hashes = P's | `f4a6b2d9ab94b553` stored on turn 6 | computed, **stored now** | **pulled from memory** |
| 32 | 8192-8448 | doc 89c8998bbb932f (new): 256 | 512-527 | P 1602-1617; D 953-968; 16/16 hashes = P's | `68a7ccd00644150b` stored on turn 6 | computed, **stored now** | **pulled from memory** |
| 33 | 8448-8704 | doc 89c8998bbb932f (new): 244; question 6 (new): 11; `assistant:` 6 (new): 1 | 528-543 | P 1618-1633; D 969-984; 16/16 hashes = P's | `aadee84c47bd5f63` stored on turn 6 | computed, **stored now** | **pulled from memory** |
| 34 | 8704-8705 | `assistant:` 6 (new): 1 | 544 | P –; D 985 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 7

9,765 prompt tokens: 38 whole chunks and 37 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 8,704 tokens on its GPU, computed 1,061 and stored 4 new chunks; D1 held 8,832, loaded 896 from LMCache and computed 37.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc c8db6e06ff4666 | 28-468 | 440 |  |
| doc 7d4d64e7f6aff1 | 468-910 | 442 |  |
| question 0 | 910-923 | 13 |  |
| `assistant:` 0 | 923-925 | 2 |  |
| answer 0 | 925-1053 | 128 |  |
| newline 0 | 1053-1054 | 1 |  |
| doc c8db6e06ff4666 | 1054-1515 | 461 |  |
| doc 7d4d64e7f6aff1 | 1515-1954 | 439 |  |
| doc 88aca7ee734372 | 1954-2326 | 372 |  |
| question 1 | 2326-2334 | 8 |  |
| `assistant:` 1 | 2334-2336 | 2 |  |
| answer 1 | 2336-2464 | 128 |  |
| newline 1 | 2464-2465 | 1 |  |
| doc 8bb77f30210c5f | 2465-2924 | 459 |  |
| doc 7f851a7621a884 | 2924-3418 | 494 |  |
| doc 8bb77f30210c5f | 3418-3910 | 492 |  |
| question 2 | 3910-3919 | 9 |  |
| `assistant:` 2 | 3919-3921 | 2 |  |
| answer 2 | 3921-4049 | 128 |  |
| newline 2 | 4049-4050 | 1 |  |
| doc 35b1d4a5ce1b03 | 4050-4565 | 515 |  |
| doc 2806a7d8f0775d | 4565-5022 | 457 |  |
| question 3 | 5022-5030 | 8 |  |
| `assistant:` 3 | 5030-5032 | 2 |  |
| answer 3 | 5032-5160 | 128 |  |
| newline 3 | 5160-5161 | 1 |  |
| doc 031693dc1490c3 | 5161-5652 | 491 |  |
| doc a46d1e3a08f352 | 5652-6137 | 485 |  |
| doc a46d1e3a08f352 | 6137-6633 | 496 |  |
| doc 031693dc1490c3 | 6633-6987 | 354 |  |
| question 4 | 6987-6993 | 6 |  |
| `assistant:` 4 | 6993-6995 | 2 |  |
| answer 4 | 6995-7123 | 128 |  |
| newline 4 | 7123-7124 | 1 |  |
| doc 7d4d64e7f6aff1 | 7124-7566 | 442 |  |
| question 5 | 7566-7583 | 17 |  |
| `assistant:` 5 | 7583-7585 | 2 |  |
| answer 5 | 7585-7713 | 128 |  |
| newline 5 | 7713-7714 | 1 |  |
| doc 4d3eba5aeb31c3 | 7714-8184 | 470 |  |
| doc 89c8998bbb932f | 8184-8692 | 508 |  |
| question 6 | 8692-8703 | 11 |  |
| `assistant:` 6 | 8703-8705 | 2 |  |
| answer 6 | 8705-8833 | 128 | new this turn |
| newline 6 | 8833-8834 | 1 | new this turn |
| doc 28f6abec5bb810 | 8834-9306 | 472 | new this turn |
| doc 28f6abec5bb810 | 9306-9752 | 446 | new this turn |
| question 7 | 9752-9763 | 11 | new this turn |
| `assistant:` 7 | 9763-9765 | 2 | new this turn |

```
P  ==================================####+
D  ==================================MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc c8db6e06ff4666: 228 | 0-15 | P 1-16; D 1-16 | `1dd495890c85343c` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc c8db6e06ff4666: 212; doc 7d4d64e7f6aff1: 44 | 16-31 | P 17-32; D 17-32 | `931e9f699f8d5ce5` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 7d4d64e7f6aff1: 256 | 32-47 | P 33-48; D 33-48 | `e6a2246ebc7818db` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 7d4d64e7f6aff1: 142; question 0: 13; `assistant:` 0: 2; answer 0: 99 | 48-63 | P 49-57, 284-290; D 97-106 … 121 (16 blocks) | `db3378056f154f8b` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | answer 0: 29; newline 0: 1; doc c8db6e06ff4666: 226 | 64-79 | P 291-306; D 123, 125-139 | `ef56131fbdafd648` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 5 | 1280-1536 | doc c8db6e06ff4666: 235; doc 7d4d64e7f6aff1: 21 | 80-95 | P 307-322; D 140-155 | `8c391d82c20faefd` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 7d4d64e7f6aff1: 256 | 96-111 | P 323-338; D 156-171 | `29b263badcec2b7e` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | doc 7d4d64e7f6aff1: 162; doc 88aca7ee734372: 94 | 112-127 | P 339-354; D 172-187 | `af1c4476de067ff9` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 8 | 2048-2304 | doc 88aca7ee734372: 256 | 128-143 | P 355-370; D 188-203 | `62545b3002b3e711` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 88aca7ee734372: 22; question 1: 8; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 8bb77f30210c5f: 95 | 144-159 | P 371-372, 534-547; D 257-259 … 276-280 (16 blocks) | `30b12248dc08664f` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 10 | 2560-2816 | doc 8bb77f30210c5f: 256 | 160-175 | P 548-563; D 281-296 | `dda183e62d75bc8d` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 8bb77f30210c5f: 108; doc 7f851a7621a884: 148 | 176-191 | P 564-579; D 297-312 | `9627ab104bc4ba56` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 7f851a7621a884: 256 | 192-207 | P 580-595; D 313-328 | `b07bc63b3fc66ef0` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 7f851a7621a884: 90; doc 8bb77f30210c5f: 166 | 208-223 | P 596-611; D 329-344 | `f3016ff95618a267` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 14 | 3584-3840 | doc 8bb77f30210c5f: 256 | 224-239 | P 612-627; D 345-360 | `d049aa24668c0744` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 15 | 3840-4096 | doc 8bb77f30210c5f: 70; question 2: 9; `assistant:` 2: 2; answer 2: 128; newline 2: 1; doc 35b1d4a5ce1b03: 46 | 240-255 | P 628-632, 833-843; D 361-366 … 480-482 (16 blocks) | `aea46b321df5ee34` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 16 | 4096-4352 | doc 35b1d4a5ce1b03: 256 | 256-271 | P 844-859; D 483-498 | `4b79490b2e8e77e2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 35b1d4a5ce1b03: 213; doc 2806a7d8f0775d: 43 | 272-287 | P 860-875; D 499-514 | `4ffb45c105c97bf2` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 18 | 4608-4864 | doc 2806a7d8f0775d: 256 | 288-303 | P 876-891; D 515-530 | `b9e95cd1762d03bb` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 19 | 4864-5120 | doc 2806a7d8f0775d: 158; question 3: 8; `assistant:` 3: 2; answer 3: 88 | 304-319 | P 892-901, 1079-1084; D 531-542 … 609 (16 blocks) | `2cc0c05a2fb893ac` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 20 | 5120-5376 | answer 3: 40; newline 3: 1; doc 031693dc1490c3: 215 | 320-335 | P 1085-1100; D 611, 613, 616-629 | `18c543a30447b7c9` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 21 | 5376-5632 | doc 031693dc1490c3: 256 | 336-351 | P 1101-1116; D 630-645 | `8650ee985a5d877c` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 031693dc1490c3: 20; doc a46d1e3a08f352: 236 | 352-367 | P 1117-1132; D 646-661 | `1a4cedb3fb65b249` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 23 | 5888-6144 | doc a46d1e3a08f352: 249; doc a46d1e3a08f352: 7 | 368-383 | P 1133-1148; D 662-677 | `7eacadaf10b20818` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 24 | 6144-6400 | doc a46d1e3a08f352: 256 | 384-399 | P 1149-1164; D 678-693 | `c080a47ca3563089` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc a46d1e3a08f352: 233; doc 031693dc1490c3: 23 | 400-415 | P 1165-1180; D 694-709 | `59c7c924ce9a2e4a` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 031693dc1490c3: 256 | 416-431 | P 1181-1196; D 710-725 | `93287088554b79c4` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | doc 031693dc1490c3: 75; question 4: 6; `assistant:` 4: 2; answer 4: 128; newline 4: 1; doc 7d4d64e7f6aff1: 44 | 432-447 | P 1197-1201, 1372-1382; D 726-732 … 818-819 (16 blocks) | `aa5ec94636aa1f2c` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 28 | 7168-7424 | doc 7d4d64e7f6aff1: 256 | 448-463 | P 1383-1398; D 820-835 | `250f8fca7969c39b` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 29 | 7424-7680 | doc 7d4d64e7f6aff1: 142; question 5: 17; `assistant:` 5: 2; answer 5: 95 | 464-479 | P 1399-1408, 1564-1569; D 836-847 … 917 (16 blocks) | `209669262a3aa2f3` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 30 | 7680-7936 | answer 5: 33; newline 5: 1; doc 4d3eba5aeb31c3: 222 | 480-495 | P 1570-1585; D 919, 921, 923-936 | `a1314184fdfde0ea` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 31 | 7936-8192 | doc 4d3eba5aeb31c3: 248; doc 89c8998bbb932f: 8 | 496-511 | P 1586-1601; D 937-952 | `f4a6b2d9ab94b553` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 32 | 8192-8448 | doc 89c8998bbb932f: 256 | 512-527 | P 1602-1617; D 953-968 | `68a7ccd00644150b` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 33 | 8448-8704 | doc 89c8998bbb932f: 244; question 6: 11; `assistant:` 6: 1 | 528-543 | P 1618-1633; D 969-984 | `aadee84c47bd5f63` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 34 | 8704-8960 | `assistant:` 6: 1; answer 6 (new): 128; newline 6 (new): 1; doc 28f6abec5bb810 (new): 126 | 544-559 | P 1703-1718; D 985-986 … 1062-1069 (16 blocks); 16/16 hashes = P's | `a71974bb29fde67b` stored on turn 7 | computed, **stored now** | **pulled from memory** (its first 128 tokens on D's GPU already) |
| 35 | 8960-9216 | doc 28f6abec5bb810 (new): 256 | 560-575 | P 1719-1734; D 1070-1085; 16/16 hashes = P's | `877913ff52927cca` stored on turn 7 | computed, **stored now** | **pulled from memory** |
| 36 | 9216-9472 | doc 28f6abec5bb810 (new): 90; doc 28f6abec5bb810 (new): 166 | 576-591 | P 1735-1750; D 1086-1101; 16/16 hashes = P's | `b9b48f0c0169a93d` stored on turn 7 | computed, **stored now** | **pulled from memory** |
| 37 | 9472-9728 | doc 28f6abec5bb810 (new): 256 | 592-607 | P 1751-1766; D 1102-1117; 16/16 hashes = P's | `84e46a6bce6c942b` stored on turn 7 | computed, **stored now** | **pulled from memory** |
| 38 | 9728-9765 | doc 28f6abec5bb810 (new): 24; question 7 (new): 11; `assistant:` 7 (new): 2 | 608-610 | P 1767-1768; D 1118-1120 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |
