# NIXL with shared LMCache · disk, chat RAG timeline

Conversation `cfbcfa3e382f12aed450b10f87a72aeb`, **disk, chat RAG**, 6 turns. P and the three D engines share one LMCache server; 8.0 GB of LMCache memory a server, the disk tier with O_DIRECT, store policy `skip_l1`, prefetch policy `retain`. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](../disk-shared-carry/05-cfbcfa3e.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/matrix/llama32_3b_lmc_disk_nixl_shared_chat_all_rtxa5000_20261126_225243) · Commit `268a9a96a324`, run `lmc_disk_nixl_shared_chat_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2) · [Turn 6](#turn-6-on-d3)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> Where do guinea pigs sleep in the wild?

**1,087 prompt tokens · 68 GPU blocks · 4 complete LMCache objects · 63 tail tokens.** [Same turn in carry-forward](../disk-shared-carry/05-cfbcfa3e.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.825–2.654 | Lookup · 4 objects found | 67 | 0 | 1 | 0 | 64 | 132 |
| P | 0.000–3.173 | Accept prompt · 67 local | **67** carried | 0 | **1** carried | 0 | 64 | 132 |
| P | 3.479–20.222 | Allocate slots and write 1 blocks | **68** · 67 local + 1 written | 0 | 1 | 0 | 64 | **133** |
| Proxy → D1 | 23.810–27.805 | Dispatch prompt and metadata · D accepts | 68 | 0 | 1 | 0 | 64 | 133 |
| D1 · LMCache | 29.836–61.052 | Lookup · 4 objects found, 4 read from disk into CPU | 68 | 0 | 1 | **64** | 64 | **197** |
| D1 | 61.582–61.926 | Plan fetch · 1 local, 67 NIXL | 68 | **67** · L1–67 | 1<br>67 slots reserved | 64 | 64 | 197 |
| NIXL | 264.904 | D reports prompt KV ready | 68 | **0** · 67 ready | **68** · 1 local + 67 NIXL | 64 | 64 | **264** |
| P | 272.698 | Release uncached partial prompt block | **67** · releases L67 | 0 | 68 | 64 | 64 | **263** |
| D1 | 264.904–1,680.390 | Run logits and write answer KV | 67 | 0 | 68 | 64 | 64 | 263 |
| D1 | 1,683.750 | Release request | 67 | 0 | 68 | 64 | 64 | 263 |

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

**1,256 prompt tokens · 79 GPU blocks · 4 complete LMCache objects · 232 tail tokens.** [Same turn in carry-forward](../disk-shared-carry/05-cfbcfa3e.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.559–6.259 | Lookup · 0 objects found | 11 | 0 | 10 | 0 | 0 | 21 |
| P | 0.000–7.133 | Accept prompt · 11 local | **11** carried | 0 | **10** carried | 0 | 0 | 21 |
| P | 7.401–97.511 | Allocate slots and write 68 blocks | **79** · 11 local + 68 written | 0 | 10 | 0 | 0 | **89** |
| P GPU → LMCache | 97.039–102.664 | Offload 4 objects | 79 | 0 | 10 | **64** · L0–63 | 0 | **153** |
| Proxy → D2 | 101.544–105.594 | Dispatch prompt and metadata · D accepts | 79 | 0 | 10 | 64 | 0 | 153 |
| D2 · LMCache | 107.339–109.500 | Lookup · 4 objects found | 79 | 0 | 10 | 64 | 0 | 153 |
| D2 | 110.114–110.466 | Plan fetch · 10 local, 69 NIXL | 79 | **69** · L10–78 | 10<br>69 slots reserved | 64 | 0 | 153 |
| P · LMCache | 276.176–276.878 | 4 disk writes · release CPU copies | 79 | 69 | 10 | **0** | **64** · +L0–63 | 153 · peak 217 |
| NIXL | 316.659 | D reports prompt KV ready | 79 | **0** · 69 ready | **79** · 10 local + 69 NIXL | 0 | 64 | **222** |
| P | 325.288 | Release uncached partial prompt block | **78** · releases L78 | 0 | 79 | 0 | 64 | **221** |
| D2 | 316.659–1,732.960 | Run logits and write answer KV | 78 | 0 | 79 | 0 | 64 | 221 |
| D2 | 1,736.329 | Release request | 78 | 0 | 79 | 0 | 64 | 221 |

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

**1,041 prompt tokens · 66 GPU blocks · 4 complete LMCache objects · 17 tail tokens.** [Same turn in carry-forward](../disk-shared-carry/05-cfbcfa3e.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.385–3.947 | Lookup · 0 objects found | 10 | 0 | 10 | 0 | 0 | 20 |
| P | 0.000–4.568 | Accept prompt · 10 local | **10** carried | 0 | **10** carried | 0 | 0 | 20 |
| P | 4.799–78.810 | Allocate slots and write 56 blocks | **66** · 10 local + 56 written | 0 | 10 | 0 | 0 | **76** |
| P GPU → LMCache | 78.448–84.906 | Offload 4 objects | 66 | 0 | 10 | **64** · L0–63 | 0 | **140** |
| Proxy → D3 | 82.494–86.729 | Dispatch prompt and metadata · D accepts | 66 | 0 | 10 | 64 | 0 | 140 |
| D3 · LMCache | 88.417–90.943 | Lookup · 4 objects found | 66 | 0 | 10 | 64 | 0 | 140 |
| D3 | 91.480–91.806 | Plan fetch · 10 local, 56 NIXL | 66 | **56** · L10–65 | 10<br>56 slots reserved | 64 | 0 | 140 |
| P · LMCache | 232.748–233.472 | 4 disk writes · release CPU copies | 66 | 56 | 10 | **0** | **64** · +L0–63 | 140 · peak 204 |
| NIXL | 258.086 | D reports prompt KV ready | 66 | **0** · 56 ready | **66** · 10 local + 56 NIXL | 0 | 64 | **196** |
| P | 266.509 | Release uncached partial prompt block | **65** · releases L65 | 0 | 66 | 0 | 64 | **195** |
| D3 | 258.086–1,678.456 | Run logits and write answer KV | 65 | 0 | 66 | 0 | 64 | 195 |
| D3 | 1,681.867 | Release request | 65 | 0 | 66 | 0 | 64 | 195 |

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

**1,667 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 131 tail tokens.** [Same turn in carry-forward](../disk-shared-carry/05-cfbcfa3e.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.901–5.333 | Lookup · 1 object found | 20 | 0 | 10 | 0 | 16 | 46 |
| P | 0.000–6.208 | Accept prompt · 20 local | **20** carried | 0 | **10** carried | 0 | 16 | 46 |
| P | 6.462–122.203 | Allocate slots and write 85 blocks | **105** · 20 local + 85 written | 0 | 10 | 0 | 16 | **131** |
| P GPU → LMCache | 121.738–130.089 | Offload 5 objects | 105 | 0 | 10 | **80** · L16–95 | 16 | **211** |
| Proxy → D1 | 126.874–131.496 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 80 | 16 | 211 |
| D1 · LMCache | 133.469–161.684 | Lookup · 6 objects found, 1 read from disk into CPU | 105 | 0 | 10 | **96** | 16 | **227** |
| D1 | 162.537–162.929 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 96 | 16 | 227 |
| P · LMCache | 353.590–354.132 | 5 disk writes · release CPU copies | 105 | 95 | 10 | **16** | **96** · +L16–95 | 227 · peak 307 |
| NIXL | 448.216 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 16 | 96 | **322** |
| P | 461.695 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 16 | 96 | **321** |
| D1 | 448.216–1,879.353 | Run logits and write answer KV | 104 | 0 | 105 | 16 | 96 | 321 |
| D1 | 1,882.909 | Release request | 104 | 0 | 105 | 16 | 96 | 321 |

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

**1,676 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 140 tail tokens.** [Same turn in carry-forward](../disk-shared-carry/05-cfbcfa3e.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.889–5.164 | Lookup · 1 object found | 29 | 0 | 10 | 16 | 16 | 71 |
| P | 0.000–6.159 | Accept prompt · 29 local | **29** carried | 0 | **10** carried | 16 | 16 | 71 |
| P | 6.414–106.218 | Allocate slots and write 76 blocks | **105** · 29 local + 76 written | 0 | 10 | 16 | 16 | **147** |
| Proxy → D2 | 110.970–115.390 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 16 | 16 | 147 |
| P GPU → LMCache | 105.801–116.362 | Offload 5 objects | 105 | 0 | 10 | **96** · L16–95 | 16 | **227** |
| D2 · LMCache | 117.753–119.625 | Lookup · 6 objects found | 105 | 0 | 10 | 96 | 16 | 227 |
| D2 | 121.628–121.974 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 96 | 16 | 227 |
| P · LMCache | 319.806–320.358 | 5 disk writes · release CPU copies | 105 | 95 | 10 | **16** | **96** · +L16–95 | 227 · peak 307 |
| NIXL | 405.031 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 16 | 96 | **322** |
| P | 415.012 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 16 | 96 | **321** |
| D2 | 405.031–1,830.886 | Run logits and write answer KV | 104 | 0 | 105 | 16 | 96 | 321 |
| D2 | 1,834.409 | Release request | 104 | 0 | 105 | 16 | 96 | 321 |

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

**1,841 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 49 tail tokens.** [Same turn in carry-forward](../disk-shared-carry/05-cfbcfa3e.md#turn-6-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.832–5.336 | Lookup · 2 objects found | 38 | 0 | 20 | 16 | 32 | 106 |
| P | 0.000–6.253 | Accept prompt · 38 local | **38** carried | 0 | **20** carried | 16 | 32 | 106 |
| P | 6.521–108.376 | Allocate slots and write 78 blocks | **116** · 38 local + 78 written | 0 | 20 | 16 | 32 | **184** |
| P GPU → LMCache | 107.844–115.702 | Offload 5 objects | 116 | 0 | 20 | **96** · L32–111 | 32 | **264** |
| Proxy → D3 | 113.309–117.954 | Dispatch prompt and metadata · D accepts | 116 | 0 | 20 | 96 | 32 | 264 |
| D3 · LMCache | 119.995–149.725 | Lookup · 7 objects found, 1 read from disk into CPU | 116 | 0 | 20 | **112** | 32 | **280** |
| D3 | 150.570–150.950 | Plan fetch · 20 local, 96 NIXL | 116 | **96** · L20–115 | 20<br>96 slots reserved | 112 | 32 | 280 |
| P · LMCache | 329.523–330.293 | 5 disk writes · release CPU copies | 116 | 96 | 20 | **32** | **112** · +L32–111 | 280 · peak 360 |
| NIXL | 438.747 | D reports prompt KV ready | 116 | **0** · 96 ready | **116** · 20 local + 96 NIXL | 32 | 112 | **376** |
| P | 449.263 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 32 | 112 | **375** |
| D3 | 438.747–1,871.678 | Run logits and write answer KV | 115 | 0 | 116 | 32 | 112 | 375 |
| D3 | 1,875.223 | Release request | 115 | 0 | 116 | 32 | 112 | 375 |

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
| 2 | D2 | 1,256 | 79 | 11 / 68 | 10 / 69 | 206 | 4 | 4 | 0 / 0 |
| 3 | D3 | 1,041 | 66 | 10 / 56 | 10 / 56 | 166 | 4 | 4 | 0 / 0 |
| 4 | D1 | 1,667 | 105 | 20 / 85 | 10 / 95 | 285 | 5 | 5 | 0 / 0 |
| 5 | D2 | 1,676 | 105 | 29 / 76 | 10 / 95 | 283 | 5 | 5 | 0 / 0 |
| 6 | D3 | 1,841 | 116 | 38 / 78 | 20 / 96 | 288 | 5 | 5 | 0 / 0 |
| all | | 8,568 | 539 | 175 / 364 | 61 / 478 | | 23 | 23 | 0 / 0 |

