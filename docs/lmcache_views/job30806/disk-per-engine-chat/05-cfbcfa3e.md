# NIXL with per-engine LMCache · disk, chat RAG timeline

Conversation `cfbcfa3e382f12aed450b10f87a72aeb`, **disk, chat RAG**, 6 turns. P and each D have their own LMCache server; 2.0 GB of LMCache memory a server, the disk tier with O_DIRECT, store policy `skip_l1`, prefetch policy `retain`. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](../disk-per-engine-carry/05-cfbcfa3e.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/matrix/llama32_3b_lmc_disk_nixl_per_engine_chat_all_rtxa5000_20261126_232500) · Commit `268a9a96a324`, run `lmc_disk_nixl_per_engine_chat_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2) · [Turn 6](#turn-6-on-d3)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> Where do guinea pigs sleep in the wild?

**1,087 prompt tokens · 68 GPU blocks · 4 complete LMCache objects · 63 tail tokens.** [Same turn in carry-forward](../disk-per-engine-carry/05-cfbcfa3e.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.872–2.896 | Lookup · 4 objects found | 67 | 0 | 1 | 0 | 64 | 0 | 0 | 132 |
| P | 0.000–4.606 | Accept prompt · 67 local | **67** carried | 0 | **1** carried | 0 | 64 | 0 | 0 | 132 |
| P | 4.847–21.234 | Allocate slots and write 1 blocks | **68** · 67 local + 1 written | 0 | 1 | 0 | 64 | 0 | 0 | **133** |
| Proxy → D1 | 25.969–30.065 | Dispatch prompt and metadata · D accepts | 68 | 0 | 1 | 0 | 64 | 0 | 0 | 133 |
| D1 · LMCache | 31.474–35.152 | Lookup · 0 objects found | 68 | 0 | 1 | 0 | 64 | 0 | 0 | 133 |
| D1 | 36.364–36.711 | Plan fetch · 1 local, 67 NIXL | 68 | **67** · L1–67 | 1<br>67 slots reserved | 0 | 64 | 0 | 0 | 133 |
| NIXL | 240.917 | D reports prompt KV ready | 68 | **0** · 67 ready | **68** · 1 local + 67 NIXL | 0 | 64 | 0 | 0 | **200** |
| P | 248.320 | Release uncached partial prompt block | **67** · releases L67 | 0 | 68 | 0 | 64 | 0 | 0 | **199** |
| D1 GPU → LMCache | 264.005–281.529 | Offload 4 objects | 67 | 0 | 68 | 0 | 64 | **64** · L0–63 | 0 | **263** |
| D1 · LMCache | 447.721–448.258 | 4 disk writes · release CPU copies | 67 | 0 | 68 | 0 | 64 | **0** | **64** · +L0–63 | 263 · peak 327 |
| D1 | 240.917–1,656.749 | Run logits and write answer KV | 67 | 0 | 68 | 0 | 64 | 0 | 64 | 263 |
| D1 | 1,660.275 | Release request | 67 | 0 | 68 | 0 | 64 | 0 | 64 | 263 |

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

**1,256 prompt tokens · 79 GPU blocks · 4 complete LMCache objects · 232 tail tokens.** [Same turn in carry-forward](../disk-per-engine-carry/05-cfbcfa3e.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.665–4.112 | Lookup · 0 objects found | 11 | 0 | 10 | 0 | 0 | 0 | 0 | 21 |
| P | 0.000–6.227 | Accept prompt · 11 local | **11** carried | 0 | **10** carried | 0 | 0 | 0 | 0 | 21 |
| P | 6.456–95.688 | Allocate slots and write 68 blocks | **79** · 11 local + 68 written | 0 | 10 | 0 | 0 | 0 | 0 | **89** |
| P GPU → LMCache | 95.268–102.245 | Offload 4 objects | 79 | 0 | 10 | **64** · L0–63 | 0 | 0 | 0 | **153** |
| Proxy → D2 | 98.986–103.517 | Dispatch prompt and metadata · D accepts | 79 | 0 | 10 | 64 | 0 | 0 | 0 | 153 |
| D2 · LMCache | 105.319–109.091 | Lookup · 0 objects found | 79 | 0 | 10 | 64 | 0 | 0 | 0 | 153 |
| D2 | 109.897–110.261 | Plan fetch · 10 local, 69 NIXL | 79 | **69** · L10–78 | 10<br>69 slots reserved | 64 | 0 | 0 | 0 | 153 |
| P · LMCache | 241.337–241.929 | 4 disk writes · release CPU copies | 79 | 69 | 10 | **0** | **64** · +L0–63 | 0 | 0 | 153 · peak 217 |
| NIXL | 318.759 | D reports prompt KV ready | 79 | **0** · 69 ready | **79** · 10 local + 69 NIXL | 0 | 64 | 0 | 0 | **222** |
| P | 328.292 | Release uncached partial prompt block | **78** · releases L78 | 0 | 79 | 0 | 64 | 0 | 0 | **221** |
| D2 GPU → LMCache | 342.011–360.148 | Offload 4 objects | 78 | 0 | 79 | 0 | 64 | **64** · L0–63 | 0 | **285** |
| D2 · LMCache | 501.021–501.673 | 4 disk writes · release CPU copies | 78 | 0 | 79 | 0 | 64 | **0** | **64** · +L0–63 | 285 · peak 349 |
| D2 | 318.759–1,736.707 | Run logits and write answer KV | 78 | 0 | 79 | 0 | 64 | 0 | 64 | 285 |
| D2 | 1,740.277 | Release request | 78 | 0 | 79 | 0 | 64 | 0 | 64 | 285 |

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

**1,041 prompt tokens · 66 GPU blocks · 4 complete LMCache objects · 17 tail tokens.** [Same turn in carry-forward](../disk-per-engine-carry/05-cfbcfa3e.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.781–5.019 | Lookup · 0 objects found | 10 | 0 | 10 | 0 | 0 | 0 | 0 | 20 |
| P | 0.000–6.325 | Accept prompt · 10 local | **10** carried | 0 | **10** carried | 0 | 0 | 0 | 0 | 20 |
| P | 6.552–81.064 | Allocate slots and write 56 blocks | **66** · 10 local + 56 written | 0 | 10 | 0 | 0 | 0 | 0 | **76** |
| Proxy → D3 | 85.098–89.629 | Dispatch prompt and metadata · D accepts | 66 | 0 | 10 | 0 | 0 | 0 | 0 | 76 |
| P GPU → LMCache | 80.676–90.458 | Offload 4 objects | 66 | 0 | 10 | **64** · L0–63 | 0 | 0 | 0 | **140** |
| D3 · LMCache | 91.360–95.417 | Lookup · 0 objects found | 66 | 0 | 10 | 64 | 0 | 0 | 0 | 140 |
| D3 | 96.217–96.534 | Plan fetch · 10 local, 56 NIXL | 66 | **56** · L10–65 | 10<br>56 slots reserved | 64 | 0 | 0 | 0 | 140 |
| P · LMCache | 238.132–238.672 | 4 disk writes · release CPU copies | 66 | 56 | 10 | **0** | **64** · +L0–63 | 0 | 0 | 140 · peak 204 |
| NIXL | 265.586 | D reports prompt KV ready | 66 | **0** · 56 ready | **66** · 10 local + 56 NIXL | 0 | 64 | 0 | 0 | **196** |
| P | 273.739 | Release uncached partial prompt block | **65** · releases L65 | 0 | 66 | 0 | 64 | 0 | 0 | **195** |
| D3 GPU → LMCache | 287.905–301.726 | Offload 4 objects | 65 | 0 | 66 | 0 | 64 | **64** · L0–63 | 0 | **259** |
| D3 · LMCache | 458.589–459.189 | 4 disk writes · release CPU copies | 65 | 0 | 66 | 0 | 64 | **0** | **64** · +L0–63 | 259 · peak 323 |
| D3 | 265.586–1,685.086 | Run logits and write answer KV | 65 | 0 | 66 | 0 | 64 | 0 | 64 | 259 |
| D3 | 1,688.468 | Release request | 65 | 0 | 66 | 0 | 64 | 0 | 64 | 259 |

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

**1,667 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 131 tail tokens.** [Same turn in carry-forward](../disk-per-engine-carry/05-cfbcfa3e.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.764–5.090 | Lookup · 1 object found | 20 | 0 | 10 | 0 | 16 | 0 | 0 | 46 |
| P | 0.000–6.322 | Accept prompt · 20 local | **20** carried | 0 | **10** carried | 0 | 16 | 0 | 0 | 46 |
| P | 6.581–124.582 | Allocate slots and write 85 blocks | **105** · 20 local + 85 written | 0 | 10 | 0 | 16 | 0 | 0 | **131** |
| P GPU → LMCache | 124.139–131.724 | Offload 5 objects | 105 | 0 | 10 | **80** · L16–95 | 16 | 0 | 0 | **211** |
| Proxy → D1 | 128.098–132.930 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 80 | 16 | 0 | 0 | 211 |
| D1 · LMCache | 134.776–138.488 | Lookup · 0 objects found | 105 | 0 | 10 | 80 | 16 | 0 | 0 | 211 |
| D1 | 139.635–140.005 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 80 | 16 | 0 | 0 | 211 |
| P · LMCache | 340.914–341.461 | 5 disk writes · release CPU copies | 105 | 95 | 10 | **0** | **96** · +L16–95 | 0 | 0 | 211 · peak 291 |
| NIXL | 423.180 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 0 | 96 | 0 | 0 | **306** |
| P | 433.701 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 0 | 96 | 0 | 0 | **305** |
| D1 GPU → LMCache | 448.198–471.532 | Offload 6 objects | 104 | 0 | 105 | 0 | 96 | **96** · L0–95 | 0 | **401** |
| D1 · LMCache | 705.187–705.830 | 6 disk writes · release CPU copies | 104 | 0 | 105 | 0 | 96 | **0** | **96** · +L0–95 | 401 · peak 497 |
| D1 | 423.180–1,852.026 | Run logits and write answer KV | 104 | 0 | 105 | 0 | 96 | 0 | 96 | 401 |
| D1 | 1,855.404 | Release request | 104 | 0 | 105 | 0 | 96 | 0 | 96 | 401 |

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

**1,676 prompt tokens · 105 GPU blocks · 6 complete LMCache objects · 140 tail tokens.** [Same turn in carry-forward](../disk-per-engine-carry/05-cfbcfa3e.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.776–5.683 | Lookup · 1 object found | 29 | 0 | 10 | 0 | 16 | 0 | 0 | 55 |
| P | 0.000–6.401 | Accept prompt · 29 local | **29** carried | 0 | **10** carried | 0 | 16 | 0 | 0 | 55 |
| P | 6.662–108.078 | Allocate slots and write 76 blocks | **105** · 29 local + 76 written | 0 | 10 | 0 | 16 | 0 | 0 | **131** |
| P GPU → LMCache | 107.642–115.579 | Offload 5 objects | 105 | 0 | 10 | **80** · L16–95 | 16 | 0 | 0 | **211** |
| Proxy → D2 | 111.598–116.491 | Dispatch prompt and metadata · D accepts | 105 | 0 | 10 | 80 | 16 | 0 | 0 | 211 |
| D2 · LMCache | 118.541–122.351 | Lookup · 0 objects found | 105 | 0 | 10 | 80 | 16 | 0 | 0 | 211 |
| D2 | 123.442–123.817 | Plan fetch · 10 local, 95 NIXL | 105 | **95** · L10–104 | 10<br>95 slots reserved | 80 | 16 | 0 | 0 | 211 |
| P · LMCache | 325.703–326.302 | 5 disk writes · release CPU copies | 105 | 95 | 10 | **0** | **96** · +L16–95 | 0 | 0 | 211 · peak 291 |
| NIXL | 412.783 | D reports prompt KV ready | 105 | **0** · 95 ready | **105** · 10 local + 95 NIXL | 0 | 96 | 0 | 0 | **306** |
| P | 425.591 | Release uncached partial prompt block | **104** · releases L104 | 0 | 105 | 0 | 96 | 0 | 0 | **305** |
| D2 GPU → LMCache | 441.179–462.138 | Offload 6 objects | 104 | 0 | 105 | 0 | 96 | **96** · L0–95 | 0 | **401** |
| D2 · LMCache | 707.181–707.777 | 6 disk writes · release CPU copies | 104 | 0 | 105 | 0 | 96 | **0** | **96** · +L0–95 | 401 · peak 497 |
| D2 | 412.783–1,843.785 | Run logits and write answer KV | 104 | 0 | 105 | 0 | 96 | 0 | 96 | 401 |
| D2 | 1,847.296 | Release request | 104 | 0 | 105 | 0 | 96 | 0 | 96 | 401 |

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

**1,841 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 49 tail tokens.** [Same turn in carry-forward](../disk-per-engine-carry/05-cfbcfa3e.md#turn-6-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.902–5.295 | Lookup · 2 objects found | 38 | 0 | 20 | 0 | 32 | 0 | 16 | 106 |
| P | 0.000–6.612 | Accept prompt · 38 local | **38** carried | 0 | **20** carried | 0 | 32 | 0 | 16 | 106 |
| P | 6.872–109.528 | Allocate slots and write 78 blocks | **116** · 38 local + 78 written | 0 | 20 | 0 | 32 | 0 | 16 | **184** |
| Proxy → D3 | 113.126–117.026 | Dispatch prompt and metadata · D accepts | 116 | 0 | 20 | 0 | 32 | 0 | 16 | 184 |
| P GPU → LMCache | 108.996–119.952 | Offload 5 objects | 116 | 0 | 20 | **80** · L32–111 | 32 | 0 | 16 | **264** |
| D3 · LMCache | 119.050–122.492 | Lookup · 1 object found | 116 | 0 | 20 | 80 | 32 | 0 | 16 | 264 |
| D3 | 123.410–123.878 | Plan fetch · 20 local, 96 NIXL | 116 | **96** · L20–115 | 20<br>96 slots reserved | 80 | 32 | 0 | 16 | 264 |
| P · LMCache | 301.464–301.897 | 5 disk writes · release CPU copies | 116 | 96 | 20 | **0** | **112** · +L32–111 | 0 | 16 | 264 · peak 344 |
| NIXL | 413.398 | D reports prompt KV ready | 116 | **0** · 96 ready | **116** · 20 local + 96 NIXL | 0 | 112 | 0 | 16 | **360** |
| P | 425.538 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 0 | 112 | 0 | 16 | **359** |
| D3 GPU → LMCache | 441.164–461.981 | Offload 6 objects | 115 | 0 | 116 | 0 | 112 | **96** · L16–111 | 16 | **455** |
| D3 · LMCache | 691.617–692.305 | 6 disk writes · release CPU copies | 115 | 0 | 116 | 0 | 112 | **0** | **112** · +L16–111 | 455 · peak 551 |
| D3 | 413.398–1,849.466 | Run logits and write answer KV | 115 | 0 | 116 | 0 | 112 | 0 | 112 | 455 |
| D3 | 1,852.991 | Release request | 115 | 0 | 116 | 0 | 112 | 0 | 112 | 455 |

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
| 1 | D1 | 1,087 | 68 | 67 / 1 | 1 / 67 | 204 | 0 / 4 | 0 / 4 | 0 / 0 |
| 2 | D2 | 1,256 | 79 | 11 / 68 | 10 / 69 | 208 | 4 / 4 | 4 / 4 | 0 / 0 |
| 3 | D3 | 1,041 | 66 | 10 / 56 | 10 / 56 | 169 | 4 / 4 | 4 / 4 | 0 / 0 |
| 4 | D1 | 1,667 | 105 | 20 / 85 | 10 / 95 | 283 | 5 / 6 | 5 / 6 | 0 / 0 |
| 5 | D2 | 1,676 | 105 | 29 / 76 | 10 / 95 | 289 | 5 / 6 | 5 / 6 | 0 / 0 |
| 6 | D3 | 1,841 | 116 | 38 / 78 | 20 / 96 | 290 | 5 / 6 | 5 / 6 | 0 / 0 |
| all | | 8,568 | 539 | 175 / 364 | 61 / 478 | | 23 / 30 | 23 / 30 | 0 / 0 |

