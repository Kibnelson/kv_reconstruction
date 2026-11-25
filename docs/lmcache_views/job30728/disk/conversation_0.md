# Conversation 0: mt-rag-clapnq-elser-512-100-20240503 (disk mode)

8 turns on D1 (routed by conversation); prompts grow from 817 to 9,097 tokens (interleaved: each turn's prompt is the previous one, its answer, then this turn's documents and question). Chunks are 256 tokens (16 vLLM blocks of 16); a chunk's name is LMCache's hash of its tokens and everything before them, so the same prefix has the same chunks on every turn and engine.

Legend for the strips (one character a chunk): P: `=` on P's GPU from an earlier turn, `#` computed and stored now, `+` computed, not stored (a partial chunk). D: `=` on D's GPU from an earlier turn, `M` pulled from memory, `S` pulled from disk, `+` computed by D (a partial chunk), `?` missed.

| turn | prompt | chunks + tail | P: own GPU / computed / stored | D: own GPU / loaded / computed | pulled | pull ms | D first token ms |
| ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| [0](#turn-0) | 817 | 3 + 49 | 16 / 801 / 3 chunks | 0 / 768 / 49 | 3 chunks (disk 3) | 52.8 | 91.4 |
| [1](#turn-1) | 1,799 | 7 + 7 | 816 / 983 / 4 chunks | 944 / 848 / 7 | 4 chunks (disk 4) | 90.4 | 115.4 |
| [2](#turn-2) | 3,494 | 13 + 166 | 1,792 / 1,702 / 6 chunks | 1,920 / 1,408 / 166 | 6 chunks (disk 6) | 140.3 | 177.7 |
| [3](#turn-3) | 4,561 | 17 + 209 | 3,488 / 1,073 / 4 chunks | 3,616 / 736 / 209 | 3 chunks (disk 3) | 34.1 | 76.4 |
| [4](#turn-4) | 5,813 | 22 + 181 | 4,560 / 1,253 / 5 chunks | 4,688 / 944 / 181 | 4 chunks (disk 4) | 97.0 | 144.1 |
| [5](#turn-5) | 6,925 | 27 + 13 | 5,808 / 1,117 / 5 chunks | 5,936 / 976 / 13 | 4 chunks (disk 4) | 53.6 | 88.7 |
| [6](#turn-6) | 8,025 | 31 + 89 | 6,912 / 1,113 / 4 chunks | 7,040 / 896 / 89 | 4 chunks (disk 4) | 61.7 | 98.6 |
| [7](#turn-7) | 9,097 | 35 + 137 | 8,016 / 1,081 / 4 chunks | 8,144 / 816 / 137 | 4 chunks (disk 4) | 97.7 | 148.5 |

### One chunk, block by block: chunk 0 of turn 0 (`f6bcdcf6a0cbe97e`)

Tokens 0-256 (system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91). P computed its blocks on turn 0 and stored them as one chunk; D1 pulled the chunk into these GPU blocks:

| block | tokens | P's GPU block | vLLM hash | D's GPU block | D wrote it | same hash on D |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 0 | 0-16 | 1 | `9f9c499d1c876b7e` | 1 | yes | True |
| 1 | 16-32 | 138 | `add775ff3d671c91` | 2 | yes | True |
| 2 | 32-48 | 139 | `d06f2312c2b2c4b0` | 3 | yes | True |
| 3 | 48-64 | 140 | `2e56c372645fbf91` | 4 | yes | True |
| 4 | 64-80 | 141 | `198bbe7f6d9acb78` | 5 | yes | True |
| 5 | 80-96 | 142 | `4fbfc818bf13cd97` | 6 | yes | True |
| 6 | 96-112 | 143 | `b7f32df1e1f34de5` | 7 | yes | True |
| 7 | 112-128 | 144 | `06f6fc561bc2b48d` | 8 | yes | True |
| 8 | 128-144 | 145 | `a61bbe5e4de373c4` | 9 | yes | True |
| 9 | 144-160 | 146 | `934b10bb9418a828` | 10 | yes | True |
| 10 | 160-176 | 147 | `7dc84b44126672f7` | 11 | yes | True |
| 11 | 176-192 | 148 | `99be881fdc72b973` | 12 | yes | True |
| 12 | 192-208 | 149 | `bad3973680bb4bfd` | 13 | yes | True |
| 13 | 208-224 | 150 | `f31724477bfa42c3` | 14 | yes | True |
| 14 | 224-240 | 151 | `436ab34d3f2eb22a` | 15 | yes | True |
| 15 | 240-256 | 152 | `00b829e6b92a441c` | 16 | yes | True |

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
D  SSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16; 16/16 hashes = P's | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | computed (its first 16 tokens on P's GPU already), **stored now** | **pulled from disk** |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32; 16/16 hashes = P's | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48; 16/16 hashes = P's | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 3 | 768-817 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2 | 48-51 | P 185-187; D 97-100 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  ===SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0 (new): 128; newline 0 (new): 1; doc 822086267_3295 (new): 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks); 16/16 hashes = P's | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | computed (its first 48 tokens on P's GPU already), **stored now** | **pulled from disk** (its first 176 tokens on D's GPU already) |
| 4 | 1024-1280 | doc 822086267_3295 (new): 1; doc 822086267_3634 (new): 80; doc 822086267_7384 (new): 175 | 64-79 | P 296-311; D 130-145; 16/16 hashes = P's | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 5 | 1280-1536 | doc 822086267_7384 (new): 236; doc 822086267_6698 (new): 20 | 80-95 | P 312-327; D 146-161; 16/16 hashes = P's | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 6 | 1536-1792 | doc 822086267_6698 (new): 117; doc 822086267_453- (new): 133; question 1 (new): 6 | 96-111 | P 328-343; D 162-177; 16/16 hashes = P's | `b745229884366b35` stored on turn 1, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 7 | 1792-1799 | question 1 (new): 5; `assistant:` 1 (new): 2 | 112 | P –; D 178 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  =======SSSSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 296-311; D 130-145 | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 312-327; D 146-161 | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 328-343; D 162-177 | `b745229884366b35` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1 (new): 128; newline 1 (new): 1; doc 822086267_6698 (new): 120 | 112-127 | P 533-548; D 178-179 … 276-281 (16 blocks); 16/16 hashes = P's | `c4ff1d48f0dc0d62` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 128 tokens on D's GPU already) |
| 8 | 2048-2304 | doc 822086267_6698 (new): 17; doc 822086267_7384 (new): 239 | 128-143 | P 549-564; D 282-297; 16/16 hashes = P's | `436a280e0d93efc2` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 9 | 2304-2560 | doc 822086267_7384 (new): 172; doc 822086267_453- (new): 84 | 144-159 | P 565-580; D 298-313; 16/16 hashes = P's | `a930ae09a0bfcac5` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 10 | 2560-2816 | doc 822086267_453- (new): 49; doc 866343245_6575 (new): 105; doc 837407666_8367 (new): 102 | 160-175 | P 581-596; D 314-329; 16/16 hashes = P's | `01a93d9215c9bf09` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 11 | 2816-3072 | doc 837407666_8367 (new): 87; doc 836236385_7052 (new): 169 | 176-191 | P 597-612; D 330-345; 16/16 hashes = P's | `789ae0a2d1db7245` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 12 | 3072-3328 | doc 836236385_7052 (new): 239; doc 837407666_122- (new): 17 | 192-207 | P 613-628; D 346-361; 16/16 hashes = P's | `ce31b70d1883b445` stored on turn 2, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 13 | 3328-3494 | doc 837407666_122- (new): 150; question 2 (new): 14; `assistant:` 2 (new): 2 | 208-218 | P 629-638; D 362-372 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  ==============SSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 296-311; D 130-145 | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 312-327; D 146-161 | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 328-343; D 162-177 | `b745229884366b35` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 533-548; D 178-179 … 276-281 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 549-564; D 282-297 | `436a280e0d93efc2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 565-580; D 298-313 | `a930ae09a0bfcac5` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 581-596; D 314-329 | `01a93d9215c9bf09` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 597-612; D 330-345 | `789ae0a2d1db7245` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 613-628; D 346-361 | `ce31b70d1883b445` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2 (new): 90 | 208-223 | P 629-638, 833-838; D 362-372 … 473 (16 blocks) | `059383bf92113ddc` stored on turn 3, on disk, out of memory | computed (its first 160 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2 (new): 38; newline 2 (new): 1; doc 822086267_6698 (new): 137; doc 822086267_7384 (new): 80 | 224-239 | P 839-854; D 475 … 481-493 (16 blocks); 16/16 hashes = P's | `27f87b70d8b35919` stored on turn 3, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 32 tokens on D's GPU already) |
| 15 | 3840-4096 | doc 822086267_7384 (new): 256 | 240-255 | P 855-870; D 494-509; 16/16 hashes = P's | `02478bea713d9170` stored on turn 3, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 16 | 4096-4352 | doc 822086267_7384 (new): 75; doc 822086267_453- (new): 133; doc 832860310_2886 (new): 48 | 256-271 | P 871-886; D 510-525; 16/16 hashes = P's | `da46f933ba9e3ae4` stored on turn 3, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 17 | 4352-4561 | doc 832860310_2886 (new): 49; doc 837407666_1762 (new): 148; question 3 (new): 10; `assistant:` 3 (new): 2 | 272-285 | P 887-899; D 526-539 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  ==================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 296-311; D 130-145 | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 312-327; D 146-161 | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 328-343; D 162-177 | `b745229884366b35` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 533-548; D 178-179 … 276-281 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 549-564; D 282-297 | `436a280e0d93efc2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 565-580; D 298-313 | `a930ae09a0bfcac5` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 581-596; D 314-329 | `01a93d9215c9bf09` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 597-612; D 330-345 | `789ae0a2d1db7245` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 613-628; D 346-361 | `ce31b70d1883b445` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 629-638, 833-838; D 362-372 … 473 (16 blocks) | `059383bf92113ddc` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 839-854; D 475 … 481-493 (16 blocks) | `27f87b70d8b35919` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 855-870; D 494-509 | `02478bea713d9170` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 871-886; D 510-525 | `da46f933ba9e3ae4` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3 (new): 47 | 272-287 | P 887-899, 1079-1081; D 526-539, 591, 604 | `64f37155a8bc7bab` stored on turn 4, on disk, out of memory | computed (its first 208 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3 (new): 81; newline 3 (new): 1; doc 822086267_6698 (new): 137; doc 822086267_7384 (new): 37 | 288-303 | P 1082-1097; D 606 … 616-626 (16 blocks); 16/16 hashes = P's | `c0a8dec361f0d9a0` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 80 tokens on D's GPU already) |
| 19 | 4864-5120 | doc 822086267_7384 (new): 256 | 304-319 | P 1098-1113; D 627-642; 16/16 hashes = P's | `b7ffb402bbe34a5b` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 20 | 5120-5376 | doc 822086267_7384 (new): 118; doc 822086267_453- (new): 133; doc 832860310_2886 (new): 5 | 320-335 | P 1114-1129; D 643-658; 16/16 hashes = P's | `f6e9bb519aa3f466` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 21 | 5376-5632 | doc 832860310_2886 (new): 92; doc 832860310_2802 (new): 164 | 336-351 | P 1130-1145; D 659-674; 16/16 hashes = P's | `e068d540a64690ae` stored on turn 4, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 22 | 5632-5813 | doc 832860310_2802 (new): 2; doc 832860310_2539 (new): 166; question 4 (new): 11; `assistant:` 4 (new): 2 | 352-363 | P 1146-1156; D 675-686 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  =======================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 296-311; D 130-145 | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 312-327; D 146-161 | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 328-343; D 162-177 | `b745229884366b35` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 533-548; D 178-179 … 276-281 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 549-564; D 282-297 | `436a280e0d93efc2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 565-580; D 298-313 | `a930ae09a0bfcac5` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 581-596; D 314-329 | `01a93d9215c9bf09` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 597-612; D 330-345 | `789ae0a2d1db7245` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 613-628; D 346-361 | `ce31b70d1883b445` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 629-638, 833-838; D 362-372 … 473 (16 blocks) | `059383bf92113ddc` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 839-854; D 475 … 481-493 (16 blocks) | `27f87b70d8b35919` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 855-870; D 494-509 | `02478bea713d9170` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 871-886; D 510-525 | `da46f933ba9e3ae4` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3: 47 | 272-287 | P 887-899, 1079-1081; D 526-539, 591, 604 | `64f37155a8bc7bab` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3: 81; newline 3: 1; doc 822086267_6698: 137; doc 822086267_7384: 37 | 288-303 | P 1082-1097; D 606 … 616-626 (16 blocks) | `c0a8dec361f0d9a0` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 19 | 4864-5120 | doc 822086267_7384: 256 | 304-319 | P 1098-1113; D 627-642 | `b7ffb402bbe34a5b` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 822086267_7384: 118; doc 822086267_453-: 133; doc 832860310_2886: 5 | 320-335 | P 1114-1129; D 643-658 | `f6e9bb519aa3f466` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 832860310_2886: 92; doc 832860310_2802: 164 | 336-351 | P 1130-1145; D 659-674 | `e068d540a64690ae` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 832860310_2802: 2; doc 832860310_2539: 166; question 4: 11; `assistant:` 4: 2; answer 4 (new): 75 | 352-367 | P 1146-1156, 1372-1376; D 675-687 … 808 (16 blocks) | `ca4fd8103cd03d7e` stored on turn 5, on disk, out of memory | computed (its first 176 tokens on P's GPU already), **stored now** | on D's GPU (earlier turn, its own answer) |
| 23 | 5888-6144 | answer 4 (new): 53; newline 4 (new): 1; doc 822086267_6698 (new): 137; doc 822086267_7384 (new): 65 | 368-383 | P 1377-1392; D 810 … 817-829 (16 blocks); 16/16 hashes = P's | `7b0f4c93606a8d70` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 48 tokens on D's GPU already) |
| 24 | 6144-6400 | doc 822086267_7384 (new): 256 | 384-399 | P 1393-1408; D 830-845; 16/16 hashes = P's | `2665830c2f0dfda0` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 25 | 6400-6656 | doc 822086267_7384 (new): 90; doc 822086267_453- (new): 133; doc 826807874_2330 (new): 33 | 400-415 | P 1409-1424; D 846-861; 16/16 hashes = P's | `736f69d77f4f5b8e` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 26 | 6656-6912 | doc 826807874_2330 (new): 9; doc 826807874_1205 (new): 246; question 5 (new): 1 | 416-431 | P 1425-1440; D 862-877; 16/16 hashes = P's | `6aae6d67d588c4fd` stored on turn 5, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 27 | 6912-6925 | question 5 (new): 11; `assistant:` 5 (new): 2 | 432 | P –; D 878 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  ===========================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 296-311; D 130-145 | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 312-327; D 146-161 | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 328-343; D 162-177 | `b745229884366b35` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 533-548; D 178-179 … 276-281 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 549-564; D 282-297 | `436a280e0d93efc2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 565-580; D 298-313 | `a930ae09a0bfcac5` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 581-596; D 314-329 | `01a93d9215c9bf09` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 597-612; D 330-345 | `789ae0a2d1db7245` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 613-628; D 346-361 | `ce31b70d1883b445` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 629-638, 833-838; D 362-372 … 473 (16 blocks) | `059383bf92113ddc` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 839-854; D 475 … 481-493 (16 blocks) | `27f87b70d8b35919` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 855-870; D 494-509 | `02478bea713d9170` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 871-886; D 510-525 | `da46f933ba9e3ae4` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3: 47 | 272-287 | P 887-899, 1079-1081; D 526-539, 591, 604 | `64f37155a8bc7bab` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3: 81; newline 3: 1; doc 822086267_6698: 137; doc 822086267_7384: 37 | 288-303 | P 1082-1097; D 606 … 616-626 (16 blocks) | `c0a8dec361f0d9a0` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 19 | 4864-5120 | doc 822086267_7384: 256 | 304-319 | P 1098-1113; D 627-642 | `b7ffb402bbe34a5b` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 822086267_7384: 118; doc 822086267_453-: 133; doc 832860310_2886: 5 | 320-335 | P 1114-1129; D 643-658 | `f6e9bb519aa3f466` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 832860310_2886: 92; doc 832860310_2802: 164 | 336-351 | P 1130-1145; D 659-674 | `e068d540a64690ae` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 832860310_2802: 2; doc 832860310_2539: 166; question 4: 11; `assistant:` 4: 2; answer 4: 75 | 352-367 | P 1146-1156, 1372-1376; D 675-687 … 808 (16 blocks) | `ca4fd8103cd03d7e` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 23 | 5888-6144 | answer 4: 53; newline 4: 1; doc 822086267_6698: 137; doc 822086267_7384: 65 | 368-383 | P 1377-1392; D 810 … 817-829 (16 blocks) | `7b0f4c93606a8d70` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 24 | 6144-6400 | doc 822086267_7384: 256 | 384-399 | P 1393-1408; D 830-845 | `2665830c2f0dfda0` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc 822086267_7384: 90; doc 822086267_453-: 133; doc 826807874_2330: 33 | 400-415 | P 1409-1424; D 846-861 | `736f69d77f4f5b8e` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 826807874_2330: 9; doc 826807874_1205: 246; question 5: 1 | 416-431 | P 1425-1440; D 862-877 | `6aae6d67d588c4fd` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | question 5: 11; `assistant:` 5: 2; answer 5 (new): 128; newline 5 (new): 1; doc 822086267_6698 (new): 114 | 432-447 | P 1564-1579; D 878-880 … 923-930 (16 blocks); 16/16 hashes = P's | `4f92db674933b9da` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** (its first 128 tokens on D's GPU already) |
| 28 | 7168-7424 | doc 822086267_6698 (new): 23; doc 822086267_7384 (new): 233 | 448-463 | P 1580-1595; D 931-946; 16/16 hashes = P's | `ce31489a90c0a852` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 29 | 7424-7680 | doc 822086267_7384 (new): 178; doc 822086267_453- (new): 78 | 464-479 | P 1596-1611; D 947-962; 16/16 hashes = P's | `55085e0b4afa2c8e` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 30 | 7680-7936 | doc 822086267_453- (new): 55; doc 826807874_2550 (new): 87; doc 826807874_1485 (new): 114 | 480-495 | P 1612-1627; D 963-978; 16/16 hashes = P's | `5df8323c0e607a22` stored on turn 6, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 31 | 7936-8025 | doc 826807874_1485 (new): 72; question 6 (new): 15; `assistant:` 6 (new): 2 | 496-501 | P 1628-1632; D 979-984 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |

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
D  ===============================SSSS+
```

| chunk | tokens | pieces in it (tokens) | vLLM blocks | GPU blocks | in LMCache | P this turn | D this turn |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0-256 | system: 28; doc 822086267_6698: 137; doc 822086267_7384: 91 | 0-15 | P 1, 138-152; D 1-16 | `f6bcdcf6a0cbe97e` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 1 | 256-512 | doc 822086267_7384: 256 | 16-31 | P 153-168; D 17-32 | `bc60126ba58d64c3` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 2 | 512-768 | doc 822086267_7384: 64; doc 822086267_6171: 147; doc 822086267_3295: 45 | 32-47 | P 169-184; D 33-48 | `f640fb7afed278d4` stored on turn 0, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 3 | 768-1024 | doc 822086267_3295: 34; question 0: 13; `assistant:` 0: 2; answer 0: 128; newline 0: 1; doc 822086267_3295: 78 | 48-63 | P 185-187, 283-295; D 97-100 … 124-129 (16 blocks) | `50c0f6d5e58eb4cc` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 4 | 1024-1280 | doc 822086267_3295: 1; doc 822086267_3634: 80; doc 822086267_7384: 175 | 64-79 | P 296-311; D 130-145 | `fc65370ee173d3fb` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 5 | 1280-1536 | doc 822086267_7384: 236; doc 822086267_6698: 20 | 80-95 | P 312-327; D 146-161 | `e3854004ac7cde07` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 6 | 1536-1792 | doc 822086267_6698: 117; doc 822086267_453-: 133; question 1: 6 | 96-111 | P 328-343; D 162-177 | `b745229884366b35` stored on turn 1, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 7 | 1792-2048 | question 1: 5; `assistant:` 1: 2; answer 1: 128; newline 1: 1; doc 822086267_6698: 120 | 112-127 | P 533-548; D 178-179 … 276-281 (16 blocks) | `c4ff1d48f0dc0d62` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 8 | 2048-2304 | doc 822086267_6698: 17; doc 822086267_7384: 239 | 128-143 | P 549-564; D 282-297 | `436a280e0d93efc2` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 9 | 2304-2560 | doc 822086267_7384: 172; doc 822086267_453-: 84 | 144-159 | P 565-580; D 298-313 | `a930ae09a0bfcac5` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 10 | 2560-2816 | doc 822086267_453-: 49; doc 866343245_6575: 105; doc 837407666_8367: 102 | 160-175 | P 581-596; D 314-329 | `01a93d9215c9bf09` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 11 | 2816-3072 | doc 837407666_8367: 87; doc 836236385_7052: 169 | 176-191 | P 597-612; D 330-345 | `789ae0a2d1db7245` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 12 | 3072-3328 | doc 836236385_7052: 239; doc 837407666_122-: 17 | 192-207 | P 613-628; D 346-361 | `ce31b70d1883b445` stored on turn 2, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 13 | 3328-3584 | doc 837407666_122-: 150; question 2: 14; `assistant:` 2: 2; answer 2: 90 | 208-223 | P 629-638, 833-838; D 362-372 … 473 (16 blocks) | `059383bf92113ddc` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 14 | 3584-3840 | answer 2: 38; newline 2: 1; doc 822086267_6698: 137; doc 822086267_7384: 80 | 224-239 | P 839-854; D 475 … 481-493 (16 blocks) | `27f87b70d8b35919` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 15 | 3840-4096 | doc 822086267_7384: 256 | 240-255 | P 855-870; D 494-509 | `02478bea713d9170` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 16 | 4096-4352 | doc 822086267_7384: 75; doc 822086267_453-: 133; doc 832860310_2886: 48 | 256-271 | P 871-886; D 510-525 | `da46f933ba9e3ae4` stored on turn 3, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 17 | 4352-4608 | doc 832860310_2886: 49; doc 837407666_1762: 148; question 3: 10; `assistant:` 3: 2; answer 3: 47 | 272-287 | P 887-899, 1079-1081; D 526-539, 591, 604 | `64f37155a8bc7bab` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 18 | 4608-4864 | answer 3: 81; newline 3: 1; doc 822086267_6698: 137; doc 822086267_7384: 37 | 288-303 | P 1082-1097; D 606 … 616-626 (16 blocks) | `c0a8dec361f0d9a0` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 19 | 4864-5120 | doc 822086267_7384: 256 | 304-319 | P 1098-1113; D 627-642 | `b7ffb402bbe34a5b` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 20 | 5120-5376 | doc 822086267_7384: 118; doc 822086267_453-: 133; doc 832860310_2886: 5 | 320-335 | P 1114-1129; D 643-658 | `f6e9bb519aa3f466` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 21 | 5376-5632 | doc 832860310_2886: 92; doc 832860310_2802: 164 | 336-351 | P 1130-1145; D 659-674 | `e068d540a64690ae` stored on turn 4, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 22 | 5632-5888 | doc 832860310_2802: 2; doc 832860310_2539: 166; question 4: 11; `assistant:` 4: 2; answer 4: 75 | 352-367 | P 1146-1156, 1372-1376; D 675-687 … 808 (16 blocks) | `ca4fd8103cd03d7e` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 23 | 5888-6144 | answer 4: 53; newline 4: 1; doc 822086267_6698: 137; doc 822086267_7384: 65 | 368-383 | P 1377-1392; D 810 … 817-829 (16 blocks) | `7b0f4c93606a8d70` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 24 | 6144-6400 | doc 822086267_7384: 256 | 384-399 | P 1393-1408; D 830-845 | `2665830c2f0dfda0` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 25 | 6400-6656 | doc 822086267_7384: 90; doc 822086267_453-: 133; doc 826807874_2330: 33 | 400-415 | P 1409-1424; D 846-861 | `736f69d77f4f5b8e` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 26 | 6656-6912 | doc 826807874_2330: 9; doc 826807874_1205: 246; question 5: 1 | 416-431 | P 1425-1440; D 862-877 | `6aae6d67d588c4fd` stored on turn 5, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 27 | 6912-7168 | question 5: 11; `assistant:` 5: 2; answer 5: 128; newline 5: 1; doc 822086267_6698: 114 | 432-447 | P 1564-1579; D 878-880 … 923-930 (16 blocks) | `4f92db674933b9da` stored on turn 6, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn, its own answer) |
| 28 | 7168-7424 | doc 822086267_6698: 23; doc 822086267_7384: 233 | 448-463 | P 1580-1595; D 931-946 | `ce31489a90c0a852` stored on turn 6, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 29 | 7424-7680 | doc 822086267_7384: 178; doc 822086267_453-: 78 | 464-479 | P 1596-1611; D 947-962 | `55085e0b4afa2c8e` stored on turn 6, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 30 | 7680-7936 | doc 822086267_453-: 55; doc 826807874_2550: 87; doc 826807874_1485: 114 | 480-495 | P 1612-1627; D 963-978 | `5df8323c0e607a22` stored on turn 6, on disk, out of memory | on P's GPU (earlier turn) | on D's GPU (earlier turn) |
| 31 | 7936-8192 | doc 826807874_1485: 72; question 6: 15; `assistant:` 6: 2; answer 6 (new): 128; newline 6 (new): 1; doc 822086267_6698 (new): 38 | 496-511 | P 1628-1632, 1703-1713; D 979-986 … 1062-1064 (16 blocks); 16/16 hashes = P's | `e6f2be060b674d77` stored on turn 7, on disk, out of memory | computed (its first 80 tokens on P's GPU already), **stored now** | **pulled from disk** (its first 208 tokens on D's GPU already) |
| 32 | 8192-8448 | doc 822086267_6698 (new): 99; doc 822086267_453- (new): 133; doc 822086267_7384 (new): 24 | 512-527 | P 1714-1729; D 1065-1080; 16/16 hashes = P's | `cae9b5abbd04bd41` stored on turn 7, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 33 | 8448-8704 | doc 822086267_7384 (new): 256 | 528-543 | P 1730-1745; D 1081-1096; 16/16 hashes = P's | `c3851b3050956ecc` stored on turn 7, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 34 | 8704-8960 | doc 822086267_7384 (new): 131; doc 836318260_1173 (new): 95; doc 821803535_1441 (new): 30 | 544-559 | P 1746-1761; D 1097-1112; 16/16 hashes = P's | `73761cb1bf2c834a` stored on turn 7, on disk, out of memory | computed, **stored now** | **pulled from disk** |
| 35 | 8960-9097 | doc 821803535_1441 (new): 126; question 7 (new): 9; `assistant:` 7 (new): 2 | 560-568 | P 1762-1769; D 1113-1121 | – | computed (partial chunk: not stored) | computed by D (partial chunk) |
