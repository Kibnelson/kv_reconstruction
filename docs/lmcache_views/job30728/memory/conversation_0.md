# Conversation 0: mt-rag-clapnq-elser-512-100-20240503 (memory mode)

8 turns on D1 (routed by conversation); prompts grow from 817 to 9,097 tokens (interleaved: each turn's prompt is the previous one, its answer, then this turn's documents and question). Chunks are 256 tokens (16 vLLM blocks of 16); a chunk's name is LMCache's hash of its tokens and everything before them, so the same prefix has the same chunks on every turn and engine.

Legend for the strips (one character a chunk): P: `=` on P's GPU from an earlier turn, `#` computed and stored now, `+` computed, not stored (a partial chunk). D: `=` on D's GPU from an earlier turn, `M` pulled from memory, `S` pulled from disk, `+` computed by D (a partial chunk), `?` missed.

| turn | prompt | chunks + tail | P: own GPU / computed / stored | D: own GPU / loaded / computed | pulled | pull ms | D first token ms |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| [0](#turn-0) | 817 | 3 + 49 | 16 / 801 / 3 chunks | 0 / 768 / 49 | 3 chunks (memory 3) | 24.3 | 64.5 |
| [1](#turn-1) | 1,799 | 7 + 7 | 816 / 983 / 4 chunks | 944 / 848 / 7 | 4 chunks (memory 4) | 18.6 | 66.2 |
| [2](#turn-2) | 3,494 | 13 + 166 | 1,792 / 1,702 / 6 chunks | 1,920 / 1,408 / 166 | 6 chunks (memory 6) | 38.8 | 109.3 |
| [3](#turn-3) | 4,561 | 17 + 209 | 3,488 / 1,073 / 4 chunks | 3,616 / 736 / 209 | 3 chunks (memory 3) | 35.7 | 116.9 |
| [4](#turn-4) | 5,813 | 22 + 181 | 4,560 / 1,253 / 5 chunks | 4,688 / 944 / 181 | 4 chunks (memory 4) | 37.7 | 123.6 |
| [5](#turn-5) | 6,925 | 27 + 13 | 5,808 / 1,117 / 5 chunks | 5,936 / 976 / 13 | 4 chunks (memory 4) | 37.9 | 109.7 |
| [6](#turn-6) | 8,025 | 31 + 89 | 6,912 / 1,113 / 4 chunks | 7,040 / 896 / 89 | 4 chunks (memory 4) | 34.9 | 117.4 |
| [7](#turn-7) | 9,097 | 35 + 137 | 8,016 / 1,081 / 4 chunks | 8,144 / 816 / 137 | 4 chunks (memory 4) | 37.6 | 128.4 |

### One chunk, block by block: chunk 0 of turn 0 (`f6bcdcf6a0cbe97e`)

Tokens 0-256 (system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91). P computed its blocks on turn 0 and stored them as one chunk; D1 pulled the chunk into these GPU blocks:

| block | tokens | P's GPU block | vLLM hash | D's GPU block | D wrote it | same hash on D |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | 0-16 | 1 | `9f9c499d1c876b7e` | 49 | yes | True |
| 1 | 16-32 | 138 | `add775ff3d671c91` | 50 | yes | True |
| 2 | 32-48 | 139 | `d06f2312c2b2c4b0` | 51 | yes | True |
| 3 | 48-64 | 140 | `2e56c372645fbf91` | 52 | yes | True |
| 4 | 64-80 | 141 | `198bbe7f6d9acb78` | 53 | yes | True |
| 5 | 80-96 | 142 | `4fbfc818bf13cd97` | 54 | yes | True |
| 6 | 96-112 | 143 | `b7f32df1e1f34de5` | 55 | yes | True |
| 7 | 112-128 | 144 | `06f6fc561bc2b48d` | 56 | yes | True |
| 8 | 128-144 | 145 | `a61bbe5e4de373c4` | 57 | yes | True |
| 9 | 144-160 | 146 | `934b10bb9418a828` | 58 | yes | True |
| 10 | 160-176 | 147 | `7dc84b44126672f7` | 59 | yes | True |
| 11 | 176-192 | 148 | `99be881fdc72b973` | 60 | yes | True |
| 12 | 192-208 | 149 | `bad3973680bb4bfd` | 61 | yes | True |
| 13 | 208-224 | 150 | `f31724477bfa42c3` | 62 | yes | True |
| 14 | 224-240 | 151 | `436ab34d3f2eb22a` | 63 | yes | True |
| 15 | 240-256 | 152 | `00b829e6b92a441c` | 64 | yes | True |

## Turn 0

817 prompt tokens: 3 whole chunks and 49 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 16 tokens on its GPU, computed 801 and stored 3 new chunks; D1 held 0, loaded 768 from LMCache and computed 49.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |

```
P  ###+
D  MMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 49-64; 16/16 hashes = P's | `f6bcdcf6a0cbe97e` stored on turn 0 | computed (its first 16 tokens on P's GPU already), **stored now** | **pulled from memory** |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80; 16/16 hashes = P's | `bc60126ba58d64c3` stored on turn 0 | computed, **stored now** | **pulled from memory** |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96; 16/16 hashes = P's | `f640fb7afed278d4` stored on turn 0 | computed, **stored now** | **pulled from memory** |
| 3 | 768-817 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2 | 48-51 | P 185-187; D 107-110 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 1

1,799 prompt tokens: 7 whole chunks and 7 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 816 tokens on its GPU, computed 983 and stored 4 new chunks; D1 held 944, loaded 848 from LMCache and computed 7.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 | new this turn |
| newline 0 | 945-946 | 1 | new this turn |
| doc 822086267_3295 | 946-1025 | 79 | new this turn |
| doc 822086267_3634 | 1025-1105 | 80 | new this turn |
| doc 822086267_7384 | 1105-1516 | 411 | new this turn |
| doc 822086267_6698 | 1516-1653 | 137 | new this turn |
| doc 822086267_453- | 1653-1786 | 133 | new this turn |
| question 1 | 1786-1797 | 11 | new this turn |
| `assistant:` 1 | 1797-1799 | 2 | new this turn |

```
P  ===####+
D  ===MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0 (new): 128; newline 0 (new): 1; doc 822086267_3295 (new): 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks); 16/16 hashes = P's | `50c0f6d5e58eb4cc` stored on turn 1 | computed (its first 48 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 176 tokens on D's GPU already) |
| 4 | 1024-1280 | doc 822086267_3295 (new): 1; doc 822086267_3634 (new): 80; doc 822086267_7384 (new): 175 | 64-79 | P 386-401; D 209-224; 16/16 hashes = P's | `fc65370ee173d3fb` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 5 | 1280-1536 | doc 822086267_7384 (new): 236; doc 822086267_6698 (new): 20 | 80-95 | P 402-417; D 225-240; 16/16 hashes = P's | `e3854004ac7cde07` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 6 | 1536-1792 | doc 822086267_6698 (new): 117; doc 822086267_453- (new): 133; question 1 (new): 6 | 96-111 | P 418-433; D 241-256; 16/16 hashes = P's | `b745229884366b35` stored on turn 1 | computed, **stored now** | **pulled from memory** |
| 7 | 1792-1799 | question 1 (new): 5; `assistant:` 1 (new): 2 | 112 | P –; D 260 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 2

3,494 prompt tokens: 13 whole chunks and 166 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 1,792 tokens on its GPU, computed 1,702 and stored 6 new chunks; D1 held 1,920, loaded 1,408 from LMCache and computed 166.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 |  |
| newline 0 | 945-946 | 1 |  |
| doc 822086267_3295 | 946-1025 | 79 |  |
| doc 822086267_3634 | 1025-1105 | 80 |  |
| doc 822086267_7384 | 1105-1516 | 411 |  |
| doc 822086267_6698 | 1516-1653 | 137 |  |
| doc 822086267_453- | 1653-1786 | 133 |  |
| question 1 | 1786-1797 | 11 |  |
| `assistant:` 1 | 1797-1799 | 2 |  |
| answer 1 | 1799-1927 | 128 | new this turn |
| newline 1 | 1927-1928 | 1 | new this turn |
| doc 822086267_6698 | 1928-2065 | 137 | new this turn |
| doc 822086267_7384 | 2065-2476 | 411 | new this turn |
| doc 822086267_453- | 2476-2609 | 133 | new this turn |
| doc 866343245_6575 | 2609-2714 | 105 | new this turn |
| doc 837407666_8367 | 2714-2903 | 189 | new this turn |
| doc 836236385_7052 | 2903-3311 | 408 | new this turn |
| doc 837407666_122- | 3311-3478 | 167 | new this turn |
| question 2 | 3478-3492 | 14 | new this turn |
| `assistant:` 2 | 3492-3494 | 2 | new this turn |

```
P  =======######+
D  =======MMMMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 386-401; D 209-224 | `fc65370ee173d3fb` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 402-417; D 225-240 | `e3854004ac7cde07` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 418-433; D 241-256 | `b745229884366b35` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1 (new): 128; newline 1 (new): 1; doc 822086267_6698 (new): 120 | 112-127 | P 634-649; D 260-261 … 367-374 (16 blocks); 16/16 hashes = P's | `c4ff1d48f0dc0d62` stored on turn 2 | computed, **stored now** | **pulled from memory** (its first 128 tokens on D's GPU already) |
| 8 | 2048-2304 | doc 822086267_6698 (new): 17; doc 822086267_7384 (new): 239 | 128-143 | P 650-665; D 375-390; 16/16 hashes = P's | `436a280e0d93efc2` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 9 | 2304-2560 | doc 822086267_7384 (new): 172; doc 822086267_453- (new): 84 | 144-159 | P 666-681; D 391-406; 16/16 hashes = P's | `a930ae09a0bfcac5` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 10 | 2560-2816 | doc 822086267_453- (new): 49; doc 866343245_6575 (new): 105; doc 837407666_8367 (new): 102 | 160-175 | P 682-697; D 407-422; 16/16 hashes = P's | `01a93d9215c9bf09` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 11 | 2816-3072 | doc 837407666_8367 (new): 87; doc 836236385_7052 (new): 169 | 176-191 | P 698-713; D 423-438; 16/16 hashes = P's | `789ae0a2d1db7245` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 12 | 3072-3328 | doc 836236385_7052 (new): 239; doc 837407666_122- (new): 17 | 192-207 | P 714-729; D 439-454; 16/16 hashes = P's | `ce31b70d1883b445` stored on turn 2 | computed, **stored now** | **pulled from memory** |
| 13 | 3328-3494 | doc 837407666_122- (new): 150; question 2 (new): 14; `assistant:` 2 (new): 2 | 208-218 | P 730-739; D 456-466 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 3

4,561 prompt tokens: 17 whole chunks and 209 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 3,488 tokens on its GPU, computed 1,073 and stored 4 new chunks; D1 held 3,616, loaded 736 from LMCache and computed 209.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 |  |
| newline 0 | 945-946 | 1 |  |
| doc 822086267_3295 | 946-1025 | 79 |  |
| doc 822086267_3634 | 1025-1105 | 80 |  |
| doc 822086267_7384 | 1105-1516 | 411 |  |
| doc 822086267_6698 | 1516-1653 | 137 |  |
| doc 822086267_453- | 1653-1786 | 133 |  |
| question 1 | 1786-1797 | 11 |  |
| `assistant:` 1 | 1797-1799 | 2 |  |
| answer 1 | 1799-1927 | 128 |  |
| newline 1 | 1927-1928 | 1 |  |
| doc 822086267_6698 | 1928-2065 | 137 |  |
| doc 822086267_7384 | 2065-2476 | 411 |  |
| doc 822086267_453- | 2476-2609 | 133 |  |
| doc 866343245_6575 | 2609-2714 | 105 |  |
| doc 837407666_8367 | 2714-2903 | 189 |  |
| doc 836236385_7052 | 2903-3311 | 408 |  |
| doc 837407666_122- | 3311-3478 | 167 |  |
| question 2 | 3478-3492 | 14 |  |
| `assistant:` 2 | 3492-3494 | 2 |  |
| answer 2 | 3494-3622 | 128 | new this turn |
| newline 2 | 3622-3623 | 1 | new this turn |
| doc 822086267_6698 | 3623-3760 | 137 | new this turn |
| doc 822086267_7384 | 3760-4171 | 411 | new this turn |
| doc 822086267_453- | 4171-4304 | 133 | new this turn |
| doc 832860310_2886 | 4304-4401 | 97 | new this turn |
| doc 837407666_1762 | 4401-4549 | 148 | new this turn |
| question 3 | 4549-4559 | 10 | new this turn |
| `assistant:` 3 | 4559-4561 | 2 | new this turn |

```
P  =============####+
D  ==============MMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 386-401; D 209-224 | `fc65370ee173d3fb` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 402-417; D 225-240 | `e3854004ac7cde07` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 418-433; D 241-256 | `b745229884366b35` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 634-649; D 260-261 … 367-374 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 650-665; D 375-390 | `436a280e0d93efc2` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 666-681; D 391-406 | `a930ae09a0bfcac5` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 682-697; D 407-422 | `01a93d9215c9bf09` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 698-713; D 423-438 | `789ae0a2d1db7245` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 714-729; D 439-454 | `ce31b70d1883b445` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2 (new): 90 | 208-223 | P 730-739, 902-907; D 456-467 … 475 (16 blocks) | `5948ffc3133d7d65` stored on turn 3 | computed (its first 160 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2 (new): 38; newline 2 (new): 1; doc 822086267_6698 (new): 137; doc 822086267_7384 (new): 80 | 224-239 | P 908-923; D 477, 479, 543-556; 16/16 hashes = P's | `a855a548a0b7c9c1` stored on turn 3 | computed, **stored now** | **pulled from memory** (its first 32 tokens on D's GPU already) |
| 15 | 3840-4096 | doc 822086267_7384 (new): 256 | 240-255 | P 924-939; D 557-572; 16/16 hashes = P's | `a4829622e99f6a85` stored on turn 3 | computed, **stored now** | **pulled from memory** |
| 16 | 4096-4352 | doc 822086267_7384 (new): 75; doc 822086267_453- (new): 133; doc 832860310_2886 (new): 48 | 256-271 | P 940-955; D 573-588; 16/16 hashes = P's | `a568e2db65f9df9e` stored on turn 3 | computed, **stored now** | **pulled from memory** |
| 17 | 4352-4561 | doc 832860310_2886 (new): 49; doc 837407666_1762 (new): 148; question 3 (new): 10; `assistant:` 3 (new): 2 | 272-285 | P 956-968; D 589-602 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 4

5,813 prompt tokens: 22 whole chunks and 181 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 4,560 tokens on its GPU, computed 1,253 and stored 5 new chunks; D1 held 4,688, loaded 944 from LMCache and computed 181.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 |  |
| newline 0 | 945-946 | 1 |  |
| doc 822086267_3295 | 946-1025 | 79 |  |
| doc 822086267_3634 | 1025-1105 | 80 |  |
| doc 822086267_7384 | 1105-1516 | 411 |  |
| doc 822086267_6698 | 1516-1653 | 137 |  |
| doc 822086267_453- | 1653-1786 | 133 |  |
| question 1 | 1786-1797 | 11 |  |
| `assistant:` 1 | 1797-1799 | 2 |  |
| answer 1 | 1799-1927 | 128 |  |
| newline 1 | 1927-1928 | 1 |  |
| doc 822086267_6698 | 1928-2065 | 137 |  |
| doc 822086267_7384 | 2065-2476 | 411 |  |
| doc 822086267_453- | 2476-2609 | 133 |  |
| doc 866343245_6575 | 2609-2714 | 105 |  |
| doc 837407666_8367 | 2714-2903 | 189 |  |
| doc 836236385_7052 | 2903-3311 | 408 |  |
| doc 837407666_122- | 3311-3478 | 167 |  |
| question 2 | 3478-3492 | 14 |  |
| `assistant:` 2 | 3492-3494 | 2 |  |
| answer 2 | 3494-3622 | 128 |  |
| newline 2 | 3622-3623 | 1 |  |
| doc 822086267_6698 | 3623-3760 | 137 |  |
| doc 822086267_7384 | 3760-4171 | 411 |  |
| doc 822086267_453- | 4171-4304 | 133 |  |
| doc 832860310_2886 | 4304-4401 | 97 |  |
| doc 837407666_1762 | 4401-4549 | 148 |  |
| question 3 | 4549-4559 | 10 |  |
| `assistant:` 3 | 4559-4561 | 2 |  |
| answer 3 | 4561-4689 | 128 | new this turn |
| newline 3 | 4689-4690 | 1 | new this turn |
| doc 822086267_6698 | 4690-4827 | 137 | new this turn |
| doc 822086267_7384 | 4827-5238 | 411 | new this turn |
| doc 822086267_453- | 5238-5371 | 133 | new this turn |
| doc 832860310_2886 | 5371-5468 | 97 | new this turn |
| doc 832860310_2802 | 5468-5634 | 166 | new this turn |
| doc 832860310_2539 | 5634-5800 | 166 | new this turn |
| question 4 | 5800-5811 | 11 | new this turn |
| `assistant:` 4 | 5811-5813 | 2 | new this turn |

```
P  =================#####+
D  ==================MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 386-401; D 209-224 | `fc65370ee173d3fb` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 402-417; D 225-240 | `e3854004ac7cde07` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 418-433; D 241-256 | `b745229884366b35` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 634-649; D 260-261 … 367-374 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 650-665; D 375-390 | `436a280e0d93efc2` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 666-681; D 391-406 | `a930ae09a0bfcac5` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 682-697; D 407-422 | `01a93d9215c9bf09` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 698-713; D 423-438 | `789ae0a2d1db7245` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 714-729; D 439-454 | `ce31b70d1883b445` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 730-739, 902-907; D 456-467 … 475 (16 blocks) | `5948ffc3133d7d65` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 908-923; D 477, 479, 543-556 | `a855a548a0b7c9c1` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 924-939; D 557-572 | `a4829622e99f6a85` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 940-955; D 573-588 | `a568e2db65f9df9e` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3 (new): 47 | 272-287 | P 956-968, 1202-1204; D 589-602, 604, 606 | `81f24ee9f0445b92` stored on turn 4 | computed (its first 208 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3 (new): 81; newline 3 (new): 1; doc 822086267_6698 (new): 137; doc 822086267_7384 (new): 37 | 288-303 | P 1205-1220; D 608 … 733-743 (16 blocks); 16/16 hashes = P's | `494b88f933dd4b6c` stored on turn 4 | computed, **stored now** | **pulled from memory** (its first 80 tokens on D's GPU already) |
| 19 | 4864-5120 | doc 822086267_7384 (new): 256 | 304-319 | P 1221-1236; D 744-759; 16/16 hashes = P's | `bc817e38622ef274` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 20 | 5120-5376 | doc 822086267_7384 (new): 118; doc 822086267_453- (new): 133; doc 832860310_2886 (new): 5 | 320-335 | P 1237-1252; D 760-775; 16/16 hashes = P's | `33bfd1edeabaeedb` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 21 | 5376-5632 | doc 832860310_2886 (new): 92; doc 832860310_2802 (new): 164 | 336-351 | P 1253-1268; D 776-791; 16/16 hashes = P's | `122f7da1727edcfb` stored on turn 4 | computed, **stored now** | **pulled from memory** |
| 22 | 5632-5813 | doc 832860310_2802 (new): 2; doc 832860310_2539 (new): 166; question 4 (new): 11; `assistant:` 4 (new): 2 | 352-363 | P 1269-1279; D 792-803 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 5

6,925 prompt tokens: 27 whole chunks and 13 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 5,808 tokens on its GPU, computed 1,117 and stored 5 new chunks; D1 held 5,936, loaded 976 from LMCache and computed 13.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 |  |
| newline 0 | 945-946 | 1 |  |
| doc 822086267_3295 | 946-1025 | 79 |  |
| doc 822086267_3634 | 1025-1105 | 80 |  |
| doc 822086267_7384 | 1105-1516 | 411 |  |
| doc 822086267_6698 | 1516-1653 | 137 |  |
| doc 822086267_453- | 1653-1786 | 133 |  |
| question 1 | 1786-1797 | 11 |  |
| `assistant:` 1 | 1797-1799 | 2 |  |
| answer 1 | 1799-1927 | 128 |  |
| newline 1 | 1927-1928 | 1 |  |
| doc 822086267_6698 | 1928-2065 | 137 |  |
| doc 822086267_7384 | 2065-2476 | 411 |  |
| doc 822086267_453- | 2476-2609 | 133 |  |
| doc 866343245_6575 | 2609-2714 | 105 |  |
| doc 837407666_8367 | 2714-2903 | 189 |  |
| doc 836236385_7052 | 2903-3311 | 408 |  |
| doc 837407666_122- | 3311-3478 | 167 |  |
| question 2 | 3478-3492 | 14 |  |
| `assistant:` 2 | 3492-3494 | 2 |  |
| answer 2 | 3494-3622 | 128 |  |
| newline 2 | 3622-3623 | 1 |  |
| doc 822086267_6698 | 3623-3760 | 137 |  |
| doc 822086267_7384 | 3760-4171 | 411 |  |
| doc 822086267_453- | 4171-4304 | 133 |  |
| doc 832860310_2886 | 4304-4401 | 97 |  |
| doc 837407666_1762 | 4401-4549 | 148 |  |
| question 3 | 4549-4559 | 10 |  |
| `assistant:` 3 | 4559-4561 | 2 |  |
| answer 3 | 4561-4689 | 128 |  |
| newline 3 | 4689-4690 | 1 |  |
| doc 822086267_6698 | 4690-4827 | 137 |  |
| doc 822086267_7384 | 4827-5238 | 411 |  |
| doc 822086267_453- | 5238-5371 | 133 |  |
| doc 832860310_2886 | 5371-5468 | 97 |  |
| doc 832860310_2802 | 5468-5634 | 166 |  |
| doc 832860310_2539 | 5634-5800 | 166 |  |
| question 4 | 5800-5811 | 11 |  |
| `assistant:` 4 | 5811-5813 | 2 |  |
| answer 4 | 5813-5941 | 128 | new this turn |
| newline 4 | 5941-5942 | 1 | new this turn |
| doc 822086267_6698 | 5942-6079 | 137 | new this turn |
| doc 822086267_7384 | 6079-6490 | 411 | new this turn |
| doc 822086267_453- | 6490-6623 | 133 | new this turn |
| doc 826807874_2330 | 6623-6665 | 42 | new this turn |
| doc 826807874_1205 | 6665-6911 | 246 | new this turn |
| question 5 | 6911-6923 | 12 | new this turn |
| `assistant:` 5 | 6923-6925 | 2 | new this turn |

```
P  ======================#####+
D  =======================MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 386-401; D 209-224 | `fc65370ee173d3fb` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 402-417; D 225-240 | `e3854004ac7cde07` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 418-433; D 241-256 | `b745229884366b35` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 634-649; D 260-261 … 367-374 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 650-665; D 375-390 | `436a280e0d93efc2` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 666-681; D 391-406 | `a930ae09a0bfcac5` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 682-697; D 407-422 | `01a93d9215c9bf09` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 698-713; D 423-438 | `789ae0a2d1db7245` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 714-729; D 439-454 | `ce31b70d1883b445` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 730-739, 902-907; D 456-467 … 475 (16 blocks) | `5948ffc3133d7d65` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 908-923; D 477, 479, 543-556 | `a855a548a0b7c9c1` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 924-939; D 557-572 | `a4829622e99f6a85` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 940-955; D 573-588 | `a568e2db65f9df9e` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3: 47 | 272-287 | P 956-968, 1202-1204; D 589-602, 604, 606 | `81f24ee9f0445b92` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3: 81; newline 3: 1; doc 822086267_6698: 137; doc 822086267_7384: 37 | 288-303 | P 1205-1220; D 608 … 733-743 (16 blocks) | `494b88f933dd4b6c` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 19 | 4864-5120 | doc 822086267_7384: 256 | 304-319 | P 1221-1236; D 744-759 | `bc817e38622ef274` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 822086267_7384: 118; doc 822086267_453-: 133; doc 832860310_2886: 5 | 320-335 | P 1237-1252; D 760-775 | `33bfd1edeabaeedb` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 832860310_2886: 92; doc 832860310_2802: 164 | 336-351 | P 1253-1268; D 776-791 | `122f7da1727edcfb` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 832860310_2802: 2; doc 832860310_2539: 166; question 4: 11; `assistant:` 4: 2; answer 4 (new): 75 | 352-367 | P 1269-1279, 1409-1413; D 792-803 … 811 (16 blocks) | `807dd430611412f7` stored on turn 5 | computed (its first 176 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 23 | 5888-6144 | answer 4 (new): 53; newline 4 (new): 1; doc 822086267_6698 (new): 137; doc 822086267_7384 (new): 65 | 368-383 | P 1414-1429; D 813 … 848-860 (16 blocks); 16/16 hashes = P's | `8cb171ff6a8e0a6f` stored on turn 5 | computed, **stored now** | **pulled from memory** (its first 48 tokens on D's GPU already) |
| 24 | 6144-6400 | doc 822086267_7384 (new): 256 | 384-399 | P 1430-1445; D 861-876; 16/16 hashes = P's | `fdd1868e3cf70cbf` stored on turn 5 | computed, **stored now** | **pulled from memory** |
| 25 | 6400-6656 | doc 822086267_7384 (new): 90; doc 822086267_453- (new): 133; doc 826807874_2330 (new): 33 | 400-415 | P 1446-1461; D 877-892; 16/16 hashes = P's | `8eb2d652cb50ea52` stored on turn 5 | computed, **stored now** | **pulled from memory** |
| 26 | 6656-6912 | doc 826807874_2330 (new): 9; doc 826807874_1205 (new): 246; question 5 (new): 1 | 416-431 | P 1462-1477; D 893-908; 16/16 hashes = P's | `83be56885e524f9f` stored on turn 5 | computed, **stored now** | **pulled from memory** |
| 27 | 6912-6925 | question 5 (new): 11; `assistant:` 5 (new): 2 | 432 | P –; D 909 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 6

8,025 prompt tokens: 31 whole chunks and 89 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 6,912 tokens on its GPU, computed 1,113 and stored 4 new chunks; D1 held 7,040, loaded 896 from LMCache and computed 89.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 |  |
| newline 0 | 945-946 | 1 |  |
| doc 822086267_3295 | 946-1025 | 79 |  |
| doc 822086267_3634 | 1025-1105 | 80 |  |
| doc 822086267_7384 | 1105-1516 | 411 |  |
| doc 822086267_6698 | 1516-1653 | 137 |  |
| doc 822086267_453- | 1653-1786 | 133 |  |
| question 1 | 1786-1797 | 11 |  |
| `assistant:` 1 | 1797-1799 | 2 |  |
| answer 1 | 1799-1927 | 128 |  |
| newline 1 | 1927-1928 | 1 |  |
| doc 822086267_6698 | 1928-2065 | 137 |  |
| doc 822086267_7384 | 2065-2476 | 411 |  |
| doc 822086267_453- | 2476-2609 | 133 |  |
| doc 866343245_6575 | 2609-2714 | 105 |  |
| doc 837407666_8367 | 2714-2903 | 189 |  |
| doc 836236385_7052 | 2903-3311 | 408 |  |
| doc 837407666_122- | 3311-3478 | 167 |  |
| question 2 | 3478-3492 | 14 |  |
| `assistant:` 2 | 3492-3494 | 2 |  |
| answer 2 | 3494-3622 | 128 |  |
| newline 2 | 3622-3623 | 1 |  |
| doc 822086267_6698 | 3623-3760 | 137 |  |
| doc 822086267_7384 | 3760-4171 | 411 |  |
| doc 822086267_453- | 4171-4304 | 133 |  |
| doc 832860310_2886 | 4304-4401 | 97 |  |
| doc 837407666_1762 | 4401-4549 | 148 |  |
| question 3 | 4549-4559 | 10 |  |
| `assistant:` 3 | 4559-4561 | 2 |  |
| answer 3 | 4561-4689 | 128 |  |
| newline 3 | 4689-4690 | 1 |  |
| doc 822086267_6698 | 4690-4827 | 137 |  |
| doc 822086267_7384 | 4827-5238 | 411 |  |
| doc 822086267_453- | 5238-5371 | 133 |  |
| doc 832860310_2886 | 5371-5468 | 97 |  |
| doc 832860310_2802 | 5468-5634 | 166 |  |
| doc 832860310_2539 | 5634-5800 | 166 |  |
| question 4 | 5800-5811 | 11 |  |
| `assistant:` 4 | 5811-5813 | 2 |  |
| answer 4 | 5813-5941 | 128 |  |
| newline 4 | 5941-5942 | 1 |  |
| doc 822086267_6698 | 5942-6079 | 137 |  |
| doc 822086267_7384 | 6079-6490 | 411 |  |
| doc 822086267_453- | 6490-6623 | 133 |  |
| doc 826807874_2330 | 6623-6665 | 42 |  |
| doc 826807874_1205 | 6665-6911 | 246 |  |
| question 5 | 6911-6923 | 12 |  |
| `assistant:` 5 | 6923-6925 | 2 |  |
| answer 5 | 6925-7053 | 128 | new this turn |
| newline 5 | 7053-7054 | 1 | new this turn |
| doc 822086267_6698 | 7054-7191 | 137 | new this turn |
| doc 822086267_7384 | 7191-7602 | 411 | new this turn |
| doc 822086267_453- | 7602-7735 | 133 | new this turn |
| doc 826807874_2550 | 7735-7822 | 87 | new this turn |
| doc 826807874_1485 | 7822-8008 | 186 | new this turn |
| question 6 | 8008-8023 | 15 | new this turn |
| `assistant:` 6 | 8023-8025 | 2 | new this turn |

```
P  ===========================####+
D  ===========================MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 386-401; D 209-224 | `fc65370ee173d3fb` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 402-417; D 225-240 | `e3854004ac7cde07` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 418-433; D 241-256 | `b745229884366b35` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 634-649; D 260-261 … 367-374 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 650-665; D 375-390 | `436a280e0d93efc2` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 666-681; D 391-406 | `a930ae09a0bfcac5` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 682-697; D 407-422 | `01a93d9215c9bf09` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 698-713; D 423-438 | `789ae0a2d1db7245` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 714-729; D 439-454 | `ce31b70d1883b445` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 730-739, 902-907; D 456-467 … 475 (16 blocks) | `5948ffc3133d7d65` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 908-923; D 477, 479, 543-556 | `a855a548a0b7c9c1` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 924-939; D 557-572 | `a4829622e99f6a85` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 940-955; D 573-588 | `a568e2db65f9df9e` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3: 47 | 272-287 | P 956-968, 1202-1204; D 589-602, 604, 606 | `81f24ee9f0445b92` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3: 81; newline 3: 1; doc 822086267_6698: 137; doc 822086267_7384: 37 | 288-303 | P 1205-1220; D 608 … 733-743 (16 blocks) | `494b88f933dd4b6c` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 19 | 4864-5120 | doc 822086267_7384: 256 | 304-319 | P 1221-1236; D 744-759 | `bc817e38622ef274` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 822086267_7384: 118; doc 822086267_453-: 133; doc 832860310_2886: 5 | 320-335 | P 1237-1252; D 760-775 | `33bfd1edeabaeedb` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 832860310_2886: 92; doc 832860310_2802: 164 | 336-351 | P 1253-1268; D 776-791 | `122f7da1727edcfb` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 832860310_2802: 2; doc 832860310_2539: 166; question 4: 11; `assistant:` 4: 2; answer 4: 75 | 352-367 | P 1269-1279, 1409-1413; D 792-803 … 811 (16 blocks) | `807dd430611412f7` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 23 | 5888-6144 | answer 4: 53; newline 4: 1; doc 822086267_6698: 137; doc 822086267_7384: 65 | 368-383 | P 1414-1429; D 813 … 848-860 (16 blocks) | `8cb171ff6a8e0a6f` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 24 | 6144-6400 | doc 822086267_7384: 256 | 384-399 | P 1430-1445; D 861-876 | `fdd1868e3cf70cbf` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc 822086267_7384: 90; doc 822086267_453-: 133; doc 826807874_2330: 33 | 400-415 | P 1446-1461; D 877-892 | `8eb2d652cb50ea52` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 826807874_2330: 9; doc 826807874_1205: 246; question 5: 1 | 416-431 | P 1462-1477; D 893-908 | `83be56885e524f9f` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | question 5: 11; `assistant:` 5: 2; answer 5 (new): 128; newline 5 (new): 1; doc 822086267_6698 (new): 114 | 432-447 | P 1634-1649; D 909-910 … 987-994 (16 blocks); 16/16 hashes = P's | `e8d161a8116f24a1` stored on turn 6 | computed, **stored now** | **pulled from memory** (its first 128 tokens on D's GPU already) |
| 28 | 7168-7424 | doc 822086267_6698 (new): 23; doc 822086267_7384 (new): 233 | 448-463 | P 1650-1665; D 995-1010; 16/16 hashes = P's | `b442f7847ac580d2` stored on turn 6 | computed, **stored now** | **pulled from memory** |
| 29 | 7424-7680 | doc 822086267_7384 (new): 178; doc 822086267_453- (new): 78 | 464-479 | P 1666-1681; D 1011-1026; 16/16 hashes = P's | `4aff7312036d0133` stored on turn 6 | computed, **stored now** | **pulled from memory** |
| 30 | 7680-7936 | doc 822086267_453- (new): 55; doc 826807874_2550 (new): 87; doc 826807874_1485 (new): 114 | 480-495 | P 1682-1697; D 1027-1042; 16/16 hashes = P's | `edf741ae8f59acea` stored on turn 6 | computed, **stored now** | **pulled from memory** |
| 31 | 7936-8025 | doc 826807874_1485 (new): 72; question 6 (new): 15; `assistant:` 6 (new): 2 | 496-501 | P 1698-1702; D 1043-1048 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

## Turn 7

9,097 prompt tokens: 35 whole chunks and 137 tokens over (the partial chunk, computed by both engines and stored only when a later turn completes it). P held 8,016 tokens on its GPU, computed 1,081 and stored 4 new chunks; D1 held 8,144, loaded 816 from LMCache and computed 137.

Pieces:

| piece | tokens | length | |
| --- | --- | ---: | --- |
| system | 0-28 | 28 |  |
| doc 822086267_6698 | 28-165 | 137 |  |
| doc 822086267_7384 | 165-576 | 411 |  |
| doc 822086267_6171 | 576-723 | 147 |  |
| doc 822086267_3295 | 723-802 | 79 |  |
| question 0 | 802-815 | 13 |  |
| `assistant:` 0 | 815-817 | 2 |  |
| answer 0 | 817-945 | 128 |  |
| newline 0 | 945-946 | 1 |  |
| doc 822086267_3295 | 946-1025 | 79 |  |
| doc 822086267_3634 | 1025-1105 | 80 |  |
| doc 822086267_7384 | 1105-1516 | 411 |  |
| doc 822086267_6698 | 1516-1653 | 137 |  |
| doc 822086267_453- | 1653-1786 | 133 |  |
| question 1 | 1786-1797 | 11 |  |
| `assistant:` 1 | 1797-1799 | 2 |  |
| answer 1 | 1799-1927 | 128 |  |
| newline 1 | 1927-1928 | 1 |  |
| doc 822086267_6698 | 1928-2065 | 137 |  |
| doc 822086267_7384 | 2065-2476 | 411 |  |
| doc 822086267_453- | 2476-2609 | 133 |  |
| doc 866343245_6575 | 2609-2714 | 105 |  |
| doc 837407666_8367 | 2714-2903 | 189 |  |
| doc 836236385_7052 | 2903-3311 | 408 |  |
| doc 837407666_122- | 3311-3478 | 167 |  |
| question 2 | 3478-3492 | 14 |  |
| `assistant:` 2 | 3492-3494 | 2 |  |
| answer 2 | 3494-3622 | 128 |  |
| newline 2 | 3622-3623 | 1 |  |
| doc 822086267_6698 | 3623-3760 | 137 |  |
| doc 822086267_7384 | 3760-4171 | 411 |  |
| doc 822086267_453- | 4171-4304 | 133 |  |
| doc 832860310_2886 | 4304-4401 | 97 |  |
| doc 837407666_1762 | 4401-4549 | 148 |  |
| question 3 | 4549-4559 | 10 |  |
| `assistant:` 3 | 4559-4561 | 2 |  |
| answer 3 | 4561-4689 | 128 |  |
| newline 3 | 4689-4690 | 1 |  |
| doc 822086267_6698 | 4690-4827 | 137 |  |
| doc 822086267_7384 | 4827-5238 | 411 |  |
| doc 822086267_453- | 5238-5371 | 133 |  |
| doc 832860310_2886 | 5371-5468 | 97 |  |
| doc 832860310_2802 | 5468-5634 | 166 |  |
| doc 832860310_2539 | 5634-5800 | 166 |  |
| question 4 | 5800-5811 | 11 |  |
| `assistant:` 4 | 5811-5813 | 2 |  |
| answer 4 | 5813-5941 | 128 |  |
| newline 4 | 5941-5942 | 1 |  |
| doc 822086267_6698 | 5942-6079 | 137 |  |
| doc 822086267_7384 | 6079-6490 | 411 |  |
| doc 822086267_453- | 6490-6623 | 133 |  |
| doc 826807874_2330 | 6623-6665 | 42 |  |
| doc 826807874_1205 | 6665-6911 | 246 |  |
| question 5 | 6911-6923 | 12 |  |
| `assistant:` 5 | 6923-6925 | 2 |  |
| answer 5 | 6925-7053 | 128 |  |
| newline 5 | 7053-7054 | 1 |  |
| doc 822086267_6698 | 7054-7191 | 137 |  |
| doc 822086267_7384 | 7191-7602 | 411 |  |
| doc 822086267_453- | 7602-7735 | 133 |  |
| doc 826807874_2550 | 7735-7822 | 87 |  |
| doc 826807874_1485 | 7822-8008 | 186 |  |
| question 6 | 8008-8023 | 15 |  |
| `assistant:` 6 | 8023-8025 | 2 |  |
| answer 6 | 8025-8153 | 128 | new this turn |
| newline 6 | 8153-8154 | 1 | new this turn |
| doc 822086267_6698 | 8154-8291 | 137 | new this turn |
| doc 822086267_453- | 8291-8424 | 133 | new this turn |
| doc 822086267_7384 | 8424-8835 | 411 | new this turn |
| doc 836318260_1173 | 8835-8930 | 95 | new this turn |
| doc 821803535_1441 | 8930-9086 | 156 | new this turn |
| question 7 | 9086-9095 | 9 | new this turn |
| `assistant:` 7 | 9095-9097 | 2 | new this turn |

```
P  ===============================####+
D  ===============================MMMM+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1, 50-64 | `f6bcdcf6a0cbe97e` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 65-80 | `bc60126ba58d64c3` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 81-96 | `f640fb7afed278d4` stored on turn 0 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 373-385; D 107-110 … 204-208 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 386-401; D 209-224 | `fc65370ee173d3fb` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 402-417; D 225-240 | `e3854004ac7cde07` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 418-433; D 241-256 | `b745229884366b35` stored on turn 1 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 634-649; D 260-261 … 367-374 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 650-665; D 375-390 | `436a280e0d93efc2` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 666-681; D 391-406 | `a930ae09a0bfcac5` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 682-697; D 407-422 | `01a93d9215c9bf09` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 698-713; D 423-438 | `789ae0a2d1db7245` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 714-729; D 439-454 | `ce31b70d1883b445` stored on turn 2 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 730-739, 902-907; D 456-467 … 475 (16 blocks) | `5948ffc3133d7d65` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 908-923; D 477, 479, 543-556 | `a855a548a0b7c9c1` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 924-939; D 557-572 | `a4829622e99f6a85` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 940-955; D 573-588 | `a568e2db65f9df9e` stored on turn 3 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3: 47 | 272-287 | P 956-968, 1202-1204; D 589-602, 604, 606 | `81f24ee9f0445b92` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3: 81; newline 3: 1; doc 822086267_6698: 137; doc 822086267_7384: 37 | 288-303 | P 1205-1220; D 608 … 733-743 (16 blocks) | `494b88f933dd4b6c` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 19 | 4864-5120 | doc 822086267_7384: 256 | 304-319 | P 1221-1236; D 744-759 | `bc817e38622ef274` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 822086267_7384: 118; doc 822086267_453-: 133; doc 832860310_2886: 5 | 320-335 | P 1237-1252; D 760-775 | `33bfd1edeabaeedb` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 832860310_2886: 92; doc 832860310_2802: 164 | 336-351 | P 1253-1268; D 776-791 | `122f7da1727edcfb` stored on turn 4 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 832860310_2802: 2; doc 832860310_2539: 166; question 4: 11; `assistant:` 4: 2; answer 4: 75 | 352-367 | P 1269-1279, 1409-1413; D 792-803 … 811 (16 blocks) | `807dd430611412f7` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 23 | 5888-6144 | answer 4: 53; newline 4: 1; doc 822086267_6698: 137; doc 822086267_7384: 65 | 368-383 | P 1414-1429; D 813 … 848-860 (16 blocks) | `8cb171ff6a8e0a6f` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 24 | 6144-6400 | doc 822086267_7384: 256 | 384-399 | P 1430-1445; D 861-876 | `fdd1868e3cf70cbf` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc 822086267_7384: 90; doc 822086267_453-: 133; doc 826807874_2330: 33 | 400-415 | P 1446-1461; D 877-892 | `8eb2d652cb50ea52` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 826807874_2330: 9; doc 826807874_1205: 246; question 5: 1 | 416-431 | P 1462-1477; D 893-908 | `83be56885e524f9f` stored on turn 5 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | question 5: 11; `assistant:` 5: 2; answer 5: 128; newline 5: 1; doc 822086267_6698: 114 | 432-447 | P 1634-1649; D 909-910 … 987-994 (16 blocks) | `e8d161a8116f24a1` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 28 | 7168-7424 | doc 822086267_6698: 23; doc 822086267_7384: 233 | 448-463 | P 1650-1665; D 995-1010 | `b442f7847ac580d2` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 29 | 7424-7680 | doc 822086267_7384: 178; doc 822086267_453-: 78 | 464-479 | P 1666-1681; D 1011-1026 | `4aff7312036d0133` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 30 | 7680-7936 | doc 822086267_453-: 55; doc 826807874_2550: 87; doc 826807874_1485: 114 | 480-495 | P 1682-1697; D 1027-1042 | `edf741ae8f59acea` stored on turn 6 | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 31 | 7936-8192 | doc 826807874_1485: 72; question 6: 15; `assistant:` 6: 2; answer 6 (new): 128; newline 6 (new): 1; doc 822086267_6698 (new): 38 | 496-511 | P 1698-1702, 1769-1779; D 1043-1048 … 1122-1124 (16 blocks); 16/16 hashes = P's | `c53c45e8ce76bb0b` stored on turn 7 | computed (its first 80 tokens on P's GPU already), **stored now** | **pulled from memory** (its first 208 tokens on D's GPU already) |
| 32 | 8192-8448 | doc 822086267_6698 (new): 99; doc 822086267_453- (new): 133; doc 822086267_7384 (new): 24 | 512-527 | P 1780-1795; D 1125-1140; 16/16 hashes = P's | `88820f1cefa5b15d` stored on turn 7 | computed, **stored now** | **pulled from memory** |
| 33 | 8448-8704 | doc 822086267_7384 (new): 256 | 528-543 | P 1796-1811; D 1141-1156; 16/16 hashes = P's | `9d58b599f3c38f3f` stored on turn 7 | computed, **stored now** | **pulled from memory** |
| 34 | 8704-8960 | doc 822086267_7384 (new): 131; doc 836318260_1173 (new): 95; doc 821803535_1441 (new): 30 | 544-559 | P 1812-1827; D 1157-1172; 16/16 hashes = P's | `0c76fcdd544ea64f` stored on turn 7 | computed, **stored now** | **pulled from memory** |
| 35 | 8960-9097 | doc 821803535_1441 (new): 126; question 7 (new): 9; `assistant:` 7 (new): 2 | 560-568 | P 1828-1835; D 1174-1182 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |
