# NIXL with per-engine LMCache · memory, chat RAG timeline

Conversation `cfbcfa3e382f12aed450b10f87a72aeb`, **memory, chat RAG**, 6 turns. P and each D have their own LMCache server; 2.0 GB of LMCache memory a server. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](../memory-per-engine-carry/05-cfbcfa3e.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/matrix/llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_231716) · Commit `268a9a96a324`, run `lmc_memory_nixl_per_engine_chat_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2) · [Turn 6](#turn-6-on-d3)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> Where do guinea pigs sleep in the wild?

**1,087 prompt tokens · 68 GPU blocks · 4 complete LMCache objects · 63 tail tokens.** [Same turn in carry-forward](../memory-per-engine-carry/05-cfbcfa3e.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.663–2.586 | Lookup · 4 objects found | 67 | 0 | 1 | 0 | 0 | 0 | 0 | 68 |
| P | 0.000–4.571 | Accept prompt · 67 local | **67** carried | 0 | **1** carried | 0 | 0 | 0 | 0 | 68 |
| P | 4.815–20.562 | Allocate slots and write 1 blocks | **68** · 67 local + 1 written | 0 | 1 | 0 | 0 | 0 | 0 | **69** |
| Proxy → D1 | 24.996–29.348 | Dispatch prompt and metadata · D accepts | 68 | 0 | 1 | 0 | 0 | 0 | 0 | 69 |
| D1 · LMCache | 30.947–32.047 | Lookup · 0 objects found | 68 | 0 | 1 | 0 | 0 | 0 | 0 | 69 |
| D1 | 33.800–34.125 | Plan fetch · 1 local, 67 NIXL | 68 | **67** · L1–67 | 1<br>67 slots reserved | 0 | 0 | 0 | 0 | 69 |
| NIXL | 234.244 | D reports prompt KV ready | 68 | **0** · 67 ready | **68** · 1 local + 67 NIXL | 0 | 0 | 0 | 0 | **136** |
| P | 242.554 | Release uncached partial prompt block | **67** · releases L67 | 0 | 68 | 0 | 0 | 0 | 0 | **135** |
| D1 GPU → LMCache | 255.376–271.688 | Offload 4 objects | 67 | 0 | 68 | 0 | 0 | **64** · L0–63 | 0 | **199** |
| D1 | 234.244–1,648.565 | Run logits and write answer KV | 67 | 0 | 68 | 0 | 0 | 64 | 0 | 199 |
| D1 | 1,652.152 | Release request | 67 | 0 | 68 | 0 | 0 | 64 | 0 | 199 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 207 / 1 | Local prefix | 642 / 2 | NIXL from P |
| 2–66 | `[32, 1072)` | 208–272 / 1 | Local prefix | 643–707 / 1 | NIXL from P |
| 67 | `[1072, 1087)` | 1516 / 2 | Model write | 708 / 1 | NIXL from P |

</details>

## Turn 2 on D2

> where does Domestic guinea pigs live and what does Domestic guinea pigs activity look like?

**1,256 prompt tokens · 79 GPU blocks · 4 complete LMCache objects · 232 tail tokens.** [Same turn in carry-forward](../memory-per-engine-carry/05-cfbcfa3e.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.575–3.443 | Lookup · 0 objects found | 11 | 0 | 10 | 0 | 0 | 0 | 0 | 21 |
| P | 0.000–4.290 | Accept prompt · 11 local | **11** carried | 0 | **10** carried | 0 | 0 | 0 | 0 | 21 |
| P | 4.521–92.981 | Allocate slots and write 68 blocks | **79** · 11 local + 68 written | 0 | 10 | 0 | 0 | 0 | 0 | **89** |
| P GPU → LMCache | 92.544–101.408 | Offload 4 objects | 79 | 0 | 10 | **64** · L0–63 | 0 | 0 | 0 | **153** |
| Proxy → D2 | 97.743–102.218 | Dispatch prompt and metadata · D accepts | 79 | 0 | 10 | 64 | 0 | 0 | 0 | 153 |
| D2 · LMCache | 104.543–105.840 | Lookup · 0 objects found | 79 | 0 | 10 | 64 | 0 | 0 | 0 | 153 |
| D2 | 107.595–107.971 | Plan fetch · 10 local, 69 NIXL | 79 | **69** · L10–78 | 10<br>69 slots reserved | 64 | 0 | 0 | 0 | 153 |
| NIXL | 313.589 | D reports prompt KV ready | 79 | **0** · 69 ready | **79** · 10 local + 69 NIXL | 64 | 0 | 0 | 0 | **222** |
| P | 321.539 | Release uncached partial prompt block | **78** · releases L78 | 0 | 79 | 64 | 0 | 0 | 0 | **221** |
| D2 GPU → LMCache | 336.943–354.901 | Offload 4 objects | 78 | 0 | 79 | 64 | 0 | **64** · L0–63 | 0 | **285** |
| D2 | 313.589–1,731.629 | Run logits and write answer KV | 78 | 0 | 79 | 64 | 0 | 64 | 0 | 285 |
| D2 | 1,735.045 | Release request | 78 | 0 | 79 | 64 | 0 | 64 | 0 | 285 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 273 / 2 | Local prefix | 119 / 2 | Local prefix |
| 2–9 | `[32, 160)` | 274–281 / 1 | Local prefix | 120–127 / 1 | Local prefix |
| 10 | `[160, 176)` | 282 / 1 | Local prefix | 647 / 2 | NIXL from P |
| 11 | `[176, 192)` | 1516 / 3 | Model write | 648 / 1 | NIXL from P |
| 12–78 | `[192, 1256)` | 1517–1583 / 1 | Model write | 649–715 / 1 | NIXL from P |

</details>

## Turn 3 on D3

> more about guinea pig appearance or body

**1,041 prompt tokens · 66 GPU blocks · 4 complete LMCache objects · 17 tail tokens.** [Same turn in carry-forward](../memory-per-engine-carry/05-cfbcfa3e.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.741–3.752 | Lookup · 0 objects found | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 20 |
| P | 0.000–4.505 | Accept prompt · 10 local | **10** carried | 0 | **10** carried | 0 | 0 | 0 | 0 | 20 |
| P | 4.740–79.754 | Allocate slots and write 56 blocks | **66** · 10 local + 56 written | 0 | 10 | 0 | 0 | 0 | 0 | **76** |
| P GPU → LMCache | 79.363–86.194 | Offload 4 objects | 66 | 0 | 10 | **64** · L0–63 | 0 | 0 | 0 | **140** |
| Proxy → D3 | 83.707–87.933 | Dispatch prompt and metadata · D accepts | 66 | 0 | 10 | 64 | 0 | 0 | 0 | 140 |
| D3 · LMCache | 89.606–91.363 | Lookup · 0 objects found | 66 | 0 | 10 | 64 | 0 | 0 | 0 | 140 |
| D3 | 92.551–92.884 | Plan fetch · 10 local, 56 NIXL | 66 | **56** · L10–65 | 10<br>56 slots reserved | 64 | 0 | 0 | 0 | 140 |
| NIXL | 258.176 | D reports prompt KV ready | 66 | **0** · 56 ready | **66** · 10 local + 56 NIXL | 64 | 0 | 0 | 0 | **196** |
| P | 265.939 | Release uncached partial prompt block | **65** · releases L65 | 0 | 66 | 64 | 0 | 0 | 0 | **195** |
| D3 GPU → LMCache | 280.661–298.888 | Offload 4 objects | 65 | 0 | 66 | 64 | 0 | **64** · L0–63 | 0 | **259** |
| D3 | 258.176–1,677.755 | Run logits and write answer KV | 65 | 0 | 66 | 64 | 0 | 64 | 0 | 259 |
| D3 | 1,681.209 | Release request | 65 | 0 | 66 | 64 | 0 | 64 | 0 | 259 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D3 block IDs / gen | D3 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 273 / 2 | Local prefix | 139 / 2 | Local prefix |
| 2–9 | `[32, 160)` | 274–281 / 1 | Local prefix | 140–147 / 1 | Local prefix |
| 10 | `[160, 176)` | 1583 / 2 | Model write | 711 / 2 | NIXL from P |
| 11–65 | `[176, 1041)` | 1584–1638 / 1 | Model write | 712–766 / 1 | NIXL from P |

</details>

## Turn 4 on D1

> can you clarify the breed here? is there different species of Guinea pig? maybe they are from different origin?

**1,667 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 131 tail tokens.** [Same turn in carry-forward](../memory-per-engine-carry/05-cfbcfa3e.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.202–3.762 | Lookup · 1 object found | 20 | 0 | 10 | 16 | 0 | 0 | 0 | 46 |
| P | 0.000–5.132 | Accept prompt · 20 local | **20** carried | 0 | **10** carried | 16 | 0 | 0 | 0 | 46 |
| P | 5.433–124.443 | Allocate slots and write 85 blocks | **105** · 20 local + 85 written | 0 | 10 | 16 | 0 | 0 | 0 | **131** |
| P GPU → LMCache | 123.857–131.147 | Offload 5 objects | 105 | 0 | 10 | **96** · L16–95 | 0 | 0 | 0 | **211** |
| Proxy → D1 | 129.553–134.436 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 96 | 0 | 0 | 0 | 211 |
| D1 · LMCache | 136.783–137.977 | Lookup · 0 objects found | 105 | 0 | 10 | 96 | 0 | 0 | 0 | 211 |
| D1 | 139.858–140.251 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 96 | 0 | 0 | 0 | 211 |
| NIXL | 421.514 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 96 | 0 | 0 | 0 | **306** |
| P | 431.825 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 96 | 0 | 0 | 0 | **305** |
| D1 GPU → LMCache | 447.802–467.515 | Offload 6 objects | 104 | 0 | 105 | 96 | 0 | **96** · L0–95 | 0 | **401** |
| D1 | 421.514–1,852.378 | Run logits and write answer KV | 104 | 0 | 105 | 96 | 0 | 96 | 0 | 401 |
| D1 | 1,855.835 | Release request | 104 | 0 | 105 | 96 | 0 | 96 | 0 | 401 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 273 / 2 | Local prefix | 111 / 2 | Local prefix |
| 2–9 | `[32, 160)` | 274–281 / 1 | Local prefix | 112–119 / 1 | Local prefix |
| 10 | `[160, 176)` | 1583 / 2 | Local prefix | 716 / 2 | NIXL from P |
| 11–19 | `[176, 320)` | 1584–1592 / 1 | Local prefix | 717–725 / 1 | NIXL from P |
| 20 | `[320, 336)` | 1638 / 2 | Model write | 726 / 1 | NIXL from P |
| 21–104 | `[336, 1667)` | 1639–1722 / 1 | Model write | 727–810 / 1 | NIXL from P |

</details>

## Turn 5 on D2

> why is Guinea pig called pig?

**1,676 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 140 tail tokens.** [Same turn in carry-forward](../memory-per-engine-carry/05-cfbcfa3e.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.758–3.299 | Lookup · 1 object found | 29 | 0 | 10 | 16 | 0 | 0 | 0 | 55 |
| P | 0.000–4.626 | Accept prompt · 29 local | **29** carried | 0 | **10** carried | 16 | 0 | 0 | 0 | 55 |
| P | 4.869–105.371 | Allocate slots and write 76 blocks | **105** · 29 local + 76 written | 0 | 10 | 16 | 0 | 0 | 0 | **131** |
| P GPU → LMCache | 104.944–112.397 | Offload 5 objects | 105 | 0 | 10 | **96** · L16–95 | 0 | 0 | 0 | **211** |
| Proxy → D2 | 110.231–115.044 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 96 | 0 | 0 | 0 | 211 |
| D2 · LMCache | 117.054–118.994 | Lookup · 0 objects found | 105 | 0 | 10 | 96 | 0 | 0 | 0 | 211 |
| D2 | 120.278–120.716 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 96 | 0 | 0 | 0 | 211 |
| NIXL | 401.157 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 96 | 0 | 0 | 0 | **306** |
| P | 411.626 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 96 | 0 | 0 | 0 | **305** |
| D2 GPU → LMCache | 426.258–449.399 | Offload 6 objects | 104 | 0 | 105 | 96 | 0 | **96** · L0–95 | 0 | **401** |
| D2 | 401.157–1,829.216 | Run logits and write answer KV | 104 | 0 | 105 | 96 | 0 | 96 | 0 | 401 |
| D2 | 1,832.778 | Release request | 104 | 0 | 105 | 96 | 0 | 96 | 0 | 401 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 273 / 2 | Local prefix | 119 / 2 | Local prefix |
| 2–9 | `[32, 160)` | 274–281 / 1 | Local prefix | 120–127 / 1 | Local prefix |
| 10 | `[160, 176)` | 1583 / 2 | Local prefix | 723 / 2 | NIXL from P |
| 11–19 | `[176, 320)` | 1584–1592 / 1 | Local prefix | 724–732 / 1 | NIXL from P |
| 20 | `[320, 336)` | 1638 / 2 | Local prefix | 733 / 1 | NIXL from P |
| 21–28 | `[336, 464)` | 1639–1646 / 1 | Local prefix | 734–741 / 1 | NIXL from P |
| 29 | `[464, 480)` | 1722 / 2 | Model write | 742 / 1 | NIXL from P |
| 30–104 | `[480, 1676)` | 1723–1797 / 1 | Model write | 743–817 / 1 | NIXL from P |

</details>

## Turn 6 on D3

> do you think I can have guinea pig as my pet?

**1,841 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 49 tail tokens.** [Same turn in carry-forward](../memory-per-engine-carry/05-cfbcfa3e.md#turn-6-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.973–3.772 | Lookup · 2 objects found | 38 | 0 | 20 | 32 | 0 | 16 | 0 | 106 |
| P | 0.000–5.015 | Accept prompt · 38 local | **38** carried | 0 | **20** carried | 32 | 0 | 16 | 0 | 106 |
| P | 5.277–107.048 | Allocate slots and write 78 blocks | **116** · 38 local + 78 written | 0 | 20 | 32 | 0 | 16 | 0 | **184** |
| P GPU → LMCache | 106.586–116.177 | Offload 5 objects | 116 | 0 | 20 | **112** · L32–111 | 0 | 16 | 0 | **264** |
| Proxy → D3 | 112.497–117.327 | Dispatch prompt and metadata · D accepts | 116 | 0 | 20 | 112 | 0 | 16 | 0 | 264 |
| D3 · LMCache | 119.591–121.210 | Lookup · 1 object found | 116 | 0 | 20 | 112 | 0 | 16 | 0 | 264 |
| D3 | 121.679–122.042 | Plan fetch · 20 local, 96 NIXL | 116 | **96** · L20–115 | 20<br>96 slots reserved | 112 | 0 | 16 | 0 | 264 |
| NIXL | 406.618 | D reports prompt KV ready | 116 | **0** · 96 ready | **116** · 20 local + 96 NIXL | 112 | 0 | 16 | 0 | **360** |
| P | 418.055 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 112 | 0 | 16 | 0 | **359** |
| D3 GPU → LMCache | 431.423–453.016 | Offload 6 objects | 115 | 0 | 116 | 112 | 0 | **112** · L16–111 | 0 | **455** |
| D3 | 406.618–1,840.060 | Run logits and write answer KV | 115 | 0 | 116 | 112 | 0 | 112 | 0 | 455 |
| D3 | 1,843.399 | Release request | 115 | 0 | 116 | 112 | 0 | 112 | 0 | 455 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D3 block IDs / gen | D3 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 273 / 2 | Local prefix | 139 / 2 | Local prefix |
| 2–9 | `[32, 160)` | 274–281 / 1 | Local prefix | 140–147 / 1 | Local prefix |
| 10 | `[160, 176)` | 1583 / 2 | Local prefix | 711 / 2 | Local prefix |
| 11–19 | `[176, 320)` | 1584–1592 / 1 | Local prefix | 712–720 / 1 | Local prefix |
| 20 | `[320, 336)` | 1638 / 2 | Local prefix | 774 / 1 | NIXL from P |
| 21–28 | `[336, 464)` | 1639–1646 / 1 | Local prefix | 775–782 / 1 | NIXL from P |
| 29 | `[464, 480)` | 1722 / 2 | Local prefix | 783 / 1 | NIXL from P |
| 30–37 | `[480, 608)` | 1723–1730 / 1 | Local prefix | 784–791 / 1 | NIXL from P |
| 38 | `[608, 624)` | 1797 / 2 | Model write | 792 / 1 | NIXL from P |
| 39–115 | `[624, 1841)` | 1798–1874 / 1 | Model write | 793–869 / 1 | NIXL from P |

</details>

## The turns together

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: P's server / D's | Disk writes: P's / D's | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 1,087 | 68 | 67 / 1 | 1 / 67 | 200 | 0 / 4 | 0 / 0 | 0 / 0 |
| 2 | D2 | 1,256 | 79 | 11 / 68 | 10 / 69 | 206 | 4 / 4 | 0 / 0 | 0 / 0 |
| 3 | D3 | 1,041 | 66 | 10 / 56 | 10 / 56 | 165 | 4 / 4 | 0 / 0 | 0 / 0 |
| 4 | D1 | 1,667 | 105 | 20 / 85 | 10 / 95 | 281 | 5 / 6 | 0 / 0 | 0 / 0 |
| 5 | D2 | 1,676 | 105 | 29 / 76 | 10 / 95 | 280 | 5 / 6 | 0 / 0 | 0 / 0 |
| 6 | D3 | 1,841 | 116 | 38 / 78 | 20 / 96 | 285 | 5 / 6 | 0 / 0 | 0 / 0 |
| all | | 8,568 | 539 | 175 / 364 | 61 / 478 | | 23 / 30 | 0 / 0 | 0 / 0 |

