# NIXL with shared LMCache · memory, chat RAG timeline

Conversation `cfbcfa3e382f12aed450b10f87a72aeb`, **memory, chat RAG**, 6 turns. P and the three D engines share one LMCache server; 8.0 GB of LMCache memory a server. GPU KV pool: 106,800 tokens (6,675 blocks) an engine (vLLM's default, no cap).

[Paired timeline](../memory-shared-carry/05-cfbcfa3e.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/matrix/llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_224358) · Commit `268a9a96a324`, run `lmc_memory_nixl_shared_chat_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2) · [Turn 6](#turn-6-on-d3)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> Where do guinea pigs sleep in the wild?

**1,087 prompt tokens · 68 GPU blocks · 4 complete LMCache objects · 63 tail tokens.** [Same turn in carry-forward](../memory-shared-carry/05-cfbcfa3e.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.462–2.359 | Lookup · 4 objects found | 67 | 0 | 1 | 64 | 0 | 132 |
| P | 0.000–4.554 | Accept prompt · 67 local | **67** carried | 0 | **1** carried | 64 | 0 | 132 |
| P | 4.786–20.936 | Allocate slots and write 1 blocks | **68** · 67 local + 1 written | 0 | 1 | 64 | 0 | **133** |
| Proxy → D1 | 23.927–28.023 | Dispatch prompt and metadata · D accepts | 68 | 0 | 1 | 64 | 0 | 133 |
| D1 · LMCache | 29.387–30.818 | Lookup · 4 objects found | 68 | 0 | 1 | 64 | 0 | 133 |
| D1 | 32.720–33.057 | Plan fetch · 1 local, 67 NIXL | 68 | **67** · L1–67 | 1<br>67 slots reserved | 64 | 0 | 133 |
| NIXL | 235.655 | D reports prompt KV ready | 68 | **0** · 67 ready | **68** · 1 local + 67 NIXL | 64 | 0 | **200** |
| P | 243.562 | Release uncached partial prompt block | **67** · releases L67 | 0 | 68 | 64 | 0 | **199** |
| D1 | 235.655–1,651.692 | Run logits and write answer KV | 67 | 0 | 68 | 64 | 0 | 199 |
| D1 | 1,655.241 | Release request | 67 | 0 | 68 | 64 | 0 | 199 |

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

**1,256 prompt tokens · 79 GPU blocks · 4 complete LMCache objects · 232 tail tokens.** [Same turn in carry-forward](../memory-shared-carry/05-cfbcfa3e.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.458–2.576 | Lookup · 0 objects found | 11 | 0 | 10 | 0 | 0 | 21 |
| P | 0.000–4.325 | Accept prompt · 11 local | **11** carried | 0 | **10** carried | 0 | 0 | 21 |
| P | 4.592–94.130 | Allocate slots and write 68 blocks | **79** · 11 local + 68 written | 0 | 10 | 0 | 0 | **89** |
| P GPU → LMCache | 93.703–100.482 | Offload 4 objects | 79 | 0 | 10 | **64** · L0–63 | 0 | **153** |
| Proxy → D2 | 98.537–103.245 | Dispatch prompt and metadata · D accepts | 79 | 0 | 10 | 64 | 0 | 153 |
| D2 · LMCache | 105.261–107.403 | Lookup · 4 objects found | 79 | 0 | 10 | 64 | 0 | 153 |
| D2 | 107.993–108.349 | Plan fetch · 10 local, 69 NIXL | 79 | **69** · L10–78 | 10<br>69 slots reserved | 64 | 0 | 153 |
| NIXL | 317.194 | D reports prompt KV ready | 79 | **0** · 69 ready | **79** · 10 local + 69 NIXL | 64 | 0 | **222** |
| P | 325.970 | Release uncached partial prompt block | **78** · releases L78 | 0 | 79 | 64 | 0 | **221** |
| D2 | 317.194–1,735.213 | Run logits and write answer KV | 78 | 0 | 79 | 64 | 0 | 221 |
| D2 | 1,738.601 | Release request | 78 | 0 | 79 | 64 | 0 | 221 |

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

**1,041 prompt tokens · 66 GPU blocks · 4 complete LMCache objects · 17 tail tokens.** [Same turn in carry-forward](../memory-shared-carry/05-cfbcfa3e.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.934–3.009 | Lookup · 0 objects found | 10 | 0 | 10 | 0 | 0 | 20 |
| P | 0.000–5.054 | Accept prompt · 10 local | **10** carried | 0 | **10** carried | 0 | 0 | 20 |
| P | 5.287–81.280 | Allocate slots and write 56 blocks | **66** · 10 local + 56 written | 0 | 10 | 0 | 0 | **76** |
| P GPU → LMCache | 80.852–87.507 | Offload 4 objects | 66 | 0 | 10 | **64** · L0–63 | 0 | **140** |
| Proxy → D3 | 85.465–89.741 | Dispatch prompt and metadata · D accepts | 66 | 0 | 10 | 64 | 0 | 140 |
| D3 · LMCache | 91.748–94.259 | Lookup · 4 objects found | 66 | 0 | 10 | 64 | 0 | 140 |
| D3 | 94.864–95.266 | Plan fetch · 10 local, 56 NIXL | 66 | **56** · L10–65 | 10<br>56 slots reserved | 64 | 0 | 140 |
| NIXL | 264.607 | D reports prompt KV ready | 66 | **0** · 56 ready | **66** · 10 local + 56 NIXL | 64 | 0 | **196** |
| P | 277.179 | Release uncached partial prompt block | **65** · releases L65 | 0 | 66 | 64 | 0 | **195** |
| D3 | 264.607–1,689.244 | Run logits and write answer KV | 65 | 0 | 66 | 64 | 0 | 195 |
| D3 | 1,692.857 | Release request | 65 | 0 | 66 | 64 | 0 | 195 |

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

**1,667 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 131 tail tokens.** [Same turn in carry-forward](../memory-shared-carry/05-cfbcfa3e.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.284–3.971 | Lookup · 1 object found | 20 | 0 | 10 | 16 | 0 | 46 |
| P | 0.000–4.584 | Accept prompt · 20 local | **20** carried | 0 | **10** carried | 16 | 0 | 46 |
| P | 4.851–123.573 | Allocate slots and write 85 blocks | **105** · 20 local + 85 written | 0 | 10 | 16 | 0 | **131** |
| P GPU → LMCache | 123.014–130.245 | Offload 5 objects | 105 | 0 | 10 | **96** · L16–95 | 0 | **211** |
| Proxy → D1 | 128.814–133.931 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 96 | 0 | 211 |
| D1 · LMCache | 135.611–137.994 | Lookup · 6 objects found | 105 | 0 | 10 | 96 | 0 | 211 |
| D1 | 138.628–138.997 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 96 | 0 | 211 |
| NIXL | 430.279 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 96 | 0 | **306** |
| P | 440.184 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 96 | 0 | **305** |
| D1 | 430.279–1,858.712 | Run logits and write answer KV | 104 | 0 | 105 | 96 | 0 | 305 |
| D1 | 1,862.264 | Release request | 104 | 0 | 105 | 96 | 0 | 305 |

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

**1,676 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 140 tail tokens.** [Same turn in carry-forward](../memory-shared-carry/05-cfbcfa3e.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.538–2.919 | Lookup · 1 object found | 29 | 0 | 10 | 16 | 0 | 55 |
| P | 0.000–4.619 | Accept prompt · 29 local | **29** carried | 0 | **10** carried | 16 | 0 | 55 |
| P | 4.878–105.063 | Allocate slots and write 76 blocks | **105** · 29 local + 76 written | 0 | 10 | 16 | 0 | **131** |
| P GPU → LMCache | 104.558–114.130 | Offload 5 objects | 105 | 0 | 10 | **96** · L16–95 | 0 | **211** |
| Proxy → D2 | 109.772–115.030 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 96 | 0 | 211 |
| D2 · LMCache | 116.932–118.921 | Lookup · 6 objects found | 105 | 0 | 10 | 96 | 0 | 211 |
| D2 | 119.565–119.839 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 96 | 0 | 211 |
| NIXL | 400.043 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 96 | 0 | **306** |
| P | 409.950 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 96 | 0 | **305** |
| D2 | 400.043–1,827.104 | Run logits and write answer KV | 104 | 0 | 105 | 96 | 0 | 305 |
| D2 | 1,830.589 | Release request | 104 | 0 | 105 | 96 | 0 | 305 |

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

**1,841 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 49 tail tokens.** [Same turn in carry-forward](../memory-shared-carry/05-cfbcfa3e.md#turn-6-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.569–2.850 | Lookup · 2 objects found | 38 | 0 | 20 | 32 | 0 | 90 |
| P | 0.000–4.569 | Accept prompt · 38 local | **38** carried | 0 | **20** carried | 32 | 0 | 90 |
| P | 4.843–106.929 | Allocate slots and write 78 blocks | **116** · 38 local + 78 written | 0 | 20 | 32 | 0 | **168** |
| P GPU → LMCache | 106.373–116.238 | Offload 5 objects | 116 | 0 | 20 | **112** · L32–111 | 0 | **248** |
| Proxy → D3 | 111.138–116.983 | Dispatch prompt and metadata · D accepts | 116 | 0 | 20 | 112 | 0 | 248 |
| D3 · LMCache | 119.443–121.874 | Lookup · 7 objects found | 116 | 0 | 20 | 112 | 0 | 248 |
| D3 | 122.406–122.864 | Plan fetch · 20 local, 96 NIXL | 116 | **96** · L20–115 | 20<br>96 slots reserved | 112 | 0 | 248 |
| NIXL | 410.131 | D reports prompt KV ready | 116 | **0** · 96 ready | **116** · 20 local + 96 NIXL | 112 | 0 | **344** |
| P | 423.505 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 112 | 0 | **343** |
| D3 | 410.131–1,845.681 | Run logits and write answer KV | 115 | 0 | 116 | 112 | 0 | 343 |
| D3 | 1,849.464 | Release request | 115 | 0 | 116 | 112 | 0 | 343 |

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

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: the server | Disk writes | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 1,087 | 68 | 67 / 1 | 1 / 67 | 203 | 0 | 0 | 0 / 0 |
| 2 | D2 | 1,256 | 79 | 11 / 68 | 10 / 69 | 209 | 4 | 0 | 0 / 0 |
| 3 | D3 | 1,041 | 66 | 10 / 56 | 10 / 56 | 169 | 4 | 0 | 0 / 0 |
| 4 | D1 | 1,667 | 105 | 20 / 85 | 10 / 95 | 291 | 5 | 0 | 0 / 0 |
| 5 | D2 | 1,676 | 105 | 29 / 76 | 10 / 95 | 280 | 5 | 0 | 0 / 0 |
| 6 | D3 | 1,841 | 116 | 38 / 78 | 20 / 96 | 287 | 5 | 0 | 0 / 0 |
| all | | 8,568 | 539 | 175 / 364 | 61 / 478 | | 23 | 0 | 0 / 0 |

