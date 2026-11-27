# NIXL with shared LMCache · memory, carry-forward timeline

Conversation `1534a095279f2cb888fb0bea17bd70da`, **memory, carry-forward**, 5 turns. P and the three D engines share one LMCache server; 8.0 GB of LMCache memory a server. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](memory-shared-chat.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/eea6fcf20beb/lmctrace/job30786/benchmark_results/pd_card/pd_lmctrace_20261127T011530Z/matrix/llama32_3b_lmc_memory_nixl_shared_carry_all_rtxa5000_20261126_202221) · Commit `eea6fcf20beb`, run `lmc_memory_nixl_shared_carry_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> who takes photos of planes in the air

**515 prompt tokens · 33 GPU blocks · 2 complete LMCache objects · 3 tail tokens.** [Same turn in chat](memory-shared-chat.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.374–3.163 | Lookup · 0 objects found | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 0.000–4.686 | Accept prompt · 0 local | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 4.800–78.061 | Allocate slots and write 33 blocks | **33** · 0 local + 33 written | 0 | 0 | 0 | 0 | **33** |
| P GPU → LMCache | 77.631–84.705 | Offload 2 objects | 33 | 0 | 0 | **32** · L0–31 | 0 | **65** |
| Proxy → D1 | 80.795–90.337 | Dispatch prompt and metadata · D accepts | 33 | 0 | 0 | 32 | 0 | 65 |
| D1 · LMCache | 93.081–93.881 | Lookup · 2 objects found | 33 | 0 | 0 | 32 | 0 | 65 |
| D1 | 94.437–94.826 | Plan fetch · 0 local, 33 NIXL | 33 | **33** · L0–32 | 0<br>33 slots reserved | 32 | 0 | 65 |
| NIXL | 343.127 | D reports prompt KV ready | 33 | **0** · 33 ready | **33** · 0 local + 33 NIXL | 32 | 0 | **98** |
| P | 345.650 | Release uncached partial prompt block | **32** · releases L32 | 0 | 33 | 32 | 0 | **97** |
| D1 | 343.127–1,731.275 | Run logits and write answer KV | 32 | 0 | 33 | 32 | 0 | 97 |
| D1 | 1,734.927 | Release request | 32 | 0 | 33 | 32 | 0 | 97 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0–32 | `[0, 515)` | 1–33 / 1 | Model write | 1–33 / 1 | NIXL from P |

</details>

## Turn 2 on D2

> No, I meant photos in the air.

**1,131 prompt tokens · 71 GPU blocks · 4 complete LMCache objects · 107 tail tokens.** [Same turn in chat](memory-shared-chat.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.488–2.599 | Lookup · 2 objects found | 32 | 0 | 0 | 32 | 0 | 64 |
| P | 0.000–4.375 | Accept prompt · 32 local | **32** carried | 0 | 0 | 32 | 0 | 64 |
| P | 4.652–59.289 | Allocate slots and write 39 blocks | **71** · 32 local + 39 written | 0 | 0 | 32 | 0 | **103** |
| P GPU → LMCache | 58.906–63.506 | Offload 2 objects | 71 | 0 | 0 | **64** · L32–63 | 0 | **135** |
| Proxy → D2 | 62.522–82.547 | Dispatch prompt and metadata · D accepts | 71 | 0 | 0 | 64 | 0 | 135 |
| D2 · LMCache | 85.447–86.310 | Lookup · 4 objects found | 71 | 0 | 0 | 64 | 0 | 135 |
| D2 | 87.140–87.347 | Plan fetch · 0 local, 71 NIXL | 71 | **71** · L0–70 | 0<br>71 slots reserved | 64 | 0 | 135 |
| NIXL | 449.529 | D reports prompt KV ready | 71 | **0** · 71 ready | **71** · 0 local + 71 NIXL | 64 | 0 | **206** |
| P | 452.563 | Release uncached partial prompt block | **70** · releases L70 | 0 | 71 | 64 | 0 | **205** |
| D2 | 449.529–1,856.528 | Run logits and write answer KV | 70 | 0 | 71 | 64 | 0 | 205 |
| D2 | 1,860.156 | Release request | 70 | 0 | 71 | 64 | 0 | 205 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0–31 | `[0, 512)` | 1–32 / 1 | Local prefix | 1–32 / 1 | NIXL from P |
| 32 | `[512, 528)` | 33 / 2 | Model write | 33 / 1 | NIXL from P |
| 33–70 | `[528, 1131)` | 34–71 / 1 | Model write | 34–71 / 1 | NIXL from P |

</details>

## Turn 3 on D3

> So does that mean pilot of the second aircraft also acts as a photographer?

**1,847 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 55 tail tokens.** [Same turn in chat](memory-shared-chat.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.650–3.907 | Lookup · 4 objects found | 70 | 0 | 0 | 64 | 0 | 134 |
| P | 0.000–4.598 | Accept prompt · 70 local | **70** carried | 0 | 0 | 64 | 0 | 134 |
| P | 4.918–71.259 | Allocate slots and write 46 blocks | **116** · 70 local + 46 written | 0 | 0 | 64 | 0 | **180** |
| P GPU → LMCache | 70.897–76.777 | Offload 3 objects | 116 | 0 | 0 | **112** · L64–111 | 0 | **228** |
| Proxy → D3 | 75.073–95.594 | Dispatch prompt and metadata · D accepts | 116 | 0 | 0 | 112 | 0 | 228 |
| D3 · LMCache | 98.730–100.055 | Lookup · 7 objects found | 116 | 0 | 0 | 112 | 0 | 228 |
| D3 | 101.678–101.898 | Plan fetch · 0 local, 116 NIXL | 116 | **116** · L0–115 | 0<br>116 slots reserved | 112 | 0 | 228 |
| NIXL | 596.154 | D reports prompt KV ready | 116 | **0** · 116 ready | **116** · 0 local + 116 NIXL | 112 | 0 | **344** |
| P | 598.198 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 112 | 0 | **343** |
| D3 | 596.154–2,022.385 | Run logits and write answer KV | 115 | 0 | 116 | 112 | 0 | 343 |
| D3 | 2,025.735 | Release request | 115 | 0 | 116 | 112 | 0 | 343 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D3 block IDs / gen | D3 source |
| --- | --- | --- | --- | --- | --- |
| 0–31 | `[0, 512)` | 1–32 / 1 | Local prefix | 1–32 / 1 | NIXL from P |
| 32 | `[512, 528)` | 33 / 2 | Local prefix | 33 / 1 | NIXL from P |
| 33–69 | `[528, 1120)` | 34–70 / 1 | Local prefix | 34–70 / 1 | NIXL from P |
| 70 | `[1120, 1136)` | 71 / 2 | Model write | 71 / 1 | NIXL from P |
| 71–115 | `[1136, 1847)` | 72–116 / 1 | Model write | 72–116 / 1 | NIXL from P |

</details>

## Turn 4 on D1

> Is air-to-air phtography more difficult that ground-to-air?

**2,531 prompt tokens · 159 GPU blocks · 9 complete LMCache objects · 227 tail tokens.** [Same turn in chat](memory-shared-chat.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.107–3.876 | Lookup · 7 objects found | 115 | 0 | 40 | 112 | 0 | 267 |
| P | 0.000–4.767 | Accept prompt · 115 local | **115** carried | 0 | **40** carried | 112 | 0 | 267 |
| P | 5.058–72.866 | Allocate slots and write 44 blocks | **159** · 115 local + 44 written | 0 | 40 | 112 | 0 | **311** |
| P GPU → LMCache | 72.448–79.777 | Offload 2 objects | 159 | 0 | 40 | **144** · L112–143 | 0 | **343** |
| Proxy → D1 | 77.470–83.620 | Dispatch prompt and metadata · D accepts | 159 | 0 | 40 | 144 | 0 | 343 |
| D1 · LMCache | 85.859–88.009 | Lookup · 9 objects found | 159 | 0 | 40 | 144 | 0 | 343 |
| D1 | 89.064–89.544 | Plan fetch · 40 local, 119 NIXL | 159 | **119** · L40–158 | 40<br>119 slots reserved | 144 | 0 | 343 |
| NIXL | 445.777 | D reports prompt KV ready | 159 | **0** · 119 ready | **159** · 40 local + 119 NIXL | 144 | 0 | **462** |
| P | 447.974 | Release uncached partial prompt block | **158** · releases L158 | 0 | 159 | 144 | 0 | **461** |
| D1 | 445.777–1,893.069 | Run logits and write answer KV | 158 | 0 | 159 | 144 | 0 | 461 |
| D1 | 1,896.381 | Release request | 158 | 0 | 159 | 144 | 0 | 461 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0–31 | `[0, 512)` | 1–32 / 1 | Local prefix | 1–32 / 1 | Local prefix |
| 32 | `[512, 528)` | 33 / 2 | Local prefix | 33 / 1 | Local prefix |
| 33–39 | `[528, 640)` | 34–40 / 1 | Local prefix | 34–40 / 1 | Local prefix |
| 40 | `[640, 656)` | 41 / 1 | Local prefix | 41 / 2 | NIXL from P |
| 41–69 | `[656, 1120)` | 42–70 / 1 | Local prefix | 42–70 / 1 | NIXL from P |
| 70 | `[1120, 1136)` | 71 / 2 | Local prefix | 71 / 1 | NIXL from P |
| 71–114 | `[1136, 1840)` | 72–115 / 1 | Local prefix | 72–115 / 1 | NIXL from P |
| 115 | `[1840, 1856)` | 116 / 2 | Model write | 116 / 1 | NIXL from P |
| 116–158 | `[1856, 2531)` | 117–159 / 1 | Model write | 117–159 / 1 | NIXL from P |

</details>

## Turn 5 on D2

> are there any other types of aviation photography?

**3,205 prompt tokens · 201 GPU blocks · 12 complete LMCache objects · 133 tail tokens.** [Same turn in chat](memory-shared-chat.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.933–4.121 | Lookup · 9 objects found | 158 | 0 | 78 | 144 | 0 | 380 |
| P | 0.000–5.173 | Accept prompt · 158 local | **158** carried | 0 | **78** carried | 144 | 0 | 380 |
| P | 5.513–76.111 | Allocate slots and write 43 blocks | **201** · 158 local + 43 written | 0 | 78 | 144 | 0 | **423** |
| P GPU → LMCache | 75.677–79.910 | Offload 3 objects | 201 | 0 | 78 | **192** · L144–191 | 0 | **471** |
| Proxy → D2 | 79.576–86.245 | Dispatch prompt and metadata · D accepts | 201 | 0 | 78 | 192 | 0 | 471 |
| D2 · LMCache | 88.519–90.606 | Lookup · 12 objects found | 201 | 0 | 78 | 192 | 0 | 471 |
| D2 | 91.171–91.687 | Plan fetch · 78 local, 123 NIXL | 201 | **123** · L78–200 | 78<br>123 slots reserved | 192 | 0 | 471 |
| NIXL | 458.992 | D reports prompt KV ready | 201 | **0** · 123 ready | **201** · 78 local + 123 NIXL | 192 | 0 | **594** |
| P | 460.665 | Release uncached partial prompt block | **200** · releases L200 | 0 | 201 | 192 | 0 | **593** |
| D2 | 458.992–1,911.277 | Run logits and write answer KV | 200 | 0 | 201 | 192 | 0 | 593 |
| D2 | 1,914.082 | Release request | 200 | 0 | 201 | 192 | 0 | 593 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0–31 | `[0, 512)` | 1–32 / 1 | Local prefix | 1–32 / 1 | Local prefix |
| 32 | `[512, 528)` | 33 / 2 | Local prefix | 33 / 1 | Local prefix |
| 33–69 | `[528, 1120)` | 34–70 / 1 | Local prefix | 34–70 / 1 | Local prefix |
| 70 | `[1120, 1136)` | 71 / 2 | Local prefix | 71 / 1 | Local prefix |
| 71–77 | `[1136, 1248)` | 72–78 / 1 | Local prefix | 72–78 / 1 | Local prefix |
| 78 | `[1248, 1264)` | 79 / 1 | Local prefix | 79 / 2 | NIXL from P |
| 79–114 | `[1264, 1840)` | 80–115 / 1 | Local prefix | 80–115 / 1 | NIXL from P |
| 115 | `[1840, 1856)` | 116 / 2 | Local prefix | 116 / 1 | NIXL from P |
| 116–157 | `[1856, 2528)` | 117–158 / 1 | Local prefix | 117–158 / 1 | NIXL from P |
| 158 | `[2528, 2544)` | 159 / 2 | Model write | 159 / 1 | NIXL from P |
| 159–200 | `[2544, 3205)` | 160–201 / 1 | Model write | 160–201 / 1 | NIXL from P |

</details>

## The turns together

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: the server | Disk writes | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 515 | 33 | 0 / 33 | 0 / 33 | 248 | 2 | 0 | 0 / 0 |
| 2 | D2 | 1,131 | 71 | 32 / 39 | 0 / 71 | 362 | 2 | 0 | 0 / 0 |
| 3 | D3 | 1,847 | 116 | 70 / 46 | 0 / 116 | 494 | 3 | 0 | 0 / 0 |
| 4 | D1 | 2,531 | 159 | 115 / 44 | 40 / 119 | 356 | 2 | 0 | 0 / 0 |
| 5 | D2 | 3,205 | 201 | 158 / 43 | 78 / 123 | 367 | 3 | 0 | 0 / 0 |
| all | | 9,229 | 580 | 375 / 205 | 118 / 462 | | 12 | 0 | 0 / 0 |

