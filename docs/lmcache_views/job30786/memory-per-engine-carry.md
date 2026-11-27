# NIXL with per-engine LMCache · memory, carry-forward timeline

Conversation `1534a095279f2cb888fb0bea17bd70da`, **memory, carry-forward**, 5 turns. P and each D have their own LMCache server; 2.0 GB of LMCache memory a server. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](memory-per-engine-chat.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/eea6fcf20beb/lmctrace/job30786/benchmark_results/pd_card/pd_lmctrace_20261127T011530Z/matrix/llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_203026) · Commit `eea6fcf20beb`, run `lmc_memory_nixl_per_engine_carry_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> who takes photos of planes in the air

**515 prompt tokens · 33 GPU blocks · 2 complete LMCache objects · 3 tail tokens.** [Same turn in chat](memory-per-engine-chat.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.878–3.820 | Lookup · 0 objects found | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 0.000–5.058 | Accept prompt · 0 local | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 5.192–78.375 | Allocate slots and write 33 blocks | **33** · 0 local + 33 written | 0 | 0 | 0 | 0 | 0 | 0 | **33** |
| P GPU → LMCache | 78.037–84.923 | Offload 2 objects | 33 | 0 | 0 | **32** · L0–31 | 0 | 0 | 0 | **65** |
| Proxy → D1 | 80.311–99.725 | Dispatch prompt and metadata · D accepts | 33 | 0 | 0 | 32 | 0 | 0 | 0 | 65 |
| D1 · LMCache | 102.757–103.647 | Lookup · 0 objects found | 33 | 0 | 0 | 32 | 0 | 0 | 0 | 65 |
| D1 | 105.726–105.930 | Plan fetch · 0 local, 33 NIXL | 33 | **33** · L0–32 | 0<br>33 slots reserved | 32 | 0 | 0 | 0 | 65 |
| NIXL | 352.723 | D reports prompt KV ready | 33 | **0** · 33 ready | **33** · 0 local + 33 NIXL | 32 | 0 | 0 | 0 | **98** |
| P | 355.135 | Release uncached partial prompt block | **32** · releases L32 | 0 | 33 | 32 | 0 | 0 | 0 | **97** |
| D1 GPU → LMCache | 372.322–380.910 | Offload 2 objects | 32 | 0 | 33 | 32 | 0 | **32** · L0–31 | 0 | **129** |
| D1 | 352.723–1,742.232 | Run logits and write answer KV | 32 | 0 | 33 | 32 | 0 | 32 | 0 | 129 |
| D1 | 1,745.765 | Release request | 32 | 0 | 33 | 32 | 0 | 32 | 0 | 129 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0–32 | `[0, 515)` | 1–33 / 1 | Model write | 1–33 / 1 | NIXL from P |

</details>

## Turn 2 on D2

> No, I meant photos in the air.

**1,131 prompt tokens · 71 GPU blocks · 4 complete LMCache objects · 107 tail tokens.** [Same turn in chat](memory-per-engine-chat.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.681–3.812 | Lookup · 2 objects found | 32 | 0 | 0 | 32 | 0 | 0 | 0 | 64 |
| P | 0.000–4.491 | Accept prompt · 32 local | **32** carried | 0 | 0 | 32 | 0 | 0 | 0 | 64 |
| P | 4.770–59.815 | Allocate slots and write 39 blocks | **71** · 32 local + 39 written | 0 | 0 | 32 | 0 | 0 | 0 | **103** |
| P GPU → LMCache | 59.436–62.963 | Offload 2 objects | 71 | 0 | 0 | **64** · L32–63 | 0 | 0 | 0 | **135** |
| Proxy → D2 | 63.220–85.312 | Dispatch prompt and metadata · D accepts | 71 | 0 | 0 | 64 | 0 | 0 | 0 | 135 |
| D2 · LMCache | 88.699–89.483 | Lookup · 0 objects found | 71 | 0 | 0 | 64 | 0 | 0 | 0 | 135 |
| D2 | 91.503–91.701 | Plan fetch · 0 local, 71 NIXL | 71 | **71** · L0–70 | 0<br>71 slots reserved | 64 | 0 | 0 | 0 | 135 |
| NIXL | 455.091 | D reports prompt KV ready | 71 | **0** · 71 ready | **71** · 0 local + 71 NIXL | 64 | 0 | 0 | 0 | **206** |
| P | 457.693 | Release uncached partial prompt block | **70** · releases L70 | 0 | 71 | 64 | 0 | 0 | 0 | **205** |
| D2 GPU → LMCache | 473.436–493.686 | Offload 4 objects | 70 | 0 | 71 | 64 | 0 | **64** · L0–63 | 0 | **269** |
| D2 | 455.091–1,865.299 | Run logits and write answer KV | 70 | 0 | 71 | 64 | 0 | 64 | 0 | 269 |
| D2 | 1,868.667 | Release request | 70 | 0 | 71 | 64 | 0 | 64 | 0 | 269 |

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

**1,847 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 55 tail tokens.** [Same turn in chat](memory-per-engine-chat.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.645–2.956 | Lookup · 4 objects found | 70 | 0 | 0 | 64 | 0 | 0 | 0 | 134 |
| P | 0.000–3.569 | Accept prompt · 70 local | **70** carried | 0 | 0 | 64 | 0 | 0 | 0 | 134 |
| P | 3.722–68.397 | Allocate slots and write 46 blocks | **116** · 70 local + 46 written | 0 | 0 | 64 | 0 | 0 | 0 | **180** |
| P GPU → LMCache | 68.167–72.633 | Offload 3 objects | 116 | 0 | 0 | **112** · L64–111 | 0 | 0 | 0 | **228** |
| Proxy → D3 | 72.345–85.118 | Dispatch prompt and metadata · D accepts | 116 | 0 | 0 | 112 | 0 | 0 | 0 | 228 |
| D3 · LMCache | 88.400–89.450 | Lookup · 0 objects found | 116 | 0 | 0 | 112 | 0 | 0 | 0 | 228 |
| D3 | 90.269–90.470 | Plan fetch · 0 local, 116 NIXL | 116 | **116** · L0–115 | 0<br>116 slots reserved | 112 | 0 | 0 | 0 | 228 |
| NIXL | 576.331 | D reports prompt KV ready | 116 | **0** · 116 ready | **116** · 0 local + 116 NIXL | 112 | 0 | 0 | 0 | **344** |
| P | 578.943 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 112 | 0 | 0 | 0 | **343** |
| D3 GPU → LMCache | 594.538–618.957 | Offload 7 objects | 115 | 0 | 116 | 112 | 0 | **112** · L0–111 | 0 | **455** |
| D3 | 576.331–2,005.696 | Run logits and write answer KV | 115 | 0 | 116 | 112 | 0 | 112 | 0 | 455 |
| D3 | 2,009.360 | Release request | 115 | 0 | 116 | 112 | 0 | 112 | 0 | 455 |

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

**2,531 prompt tokens · 159 GPU blocks · 9 complete LMCache objects · 227 tail tokens.** [Same turn in chat](memory-per-engine-chat.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.890–3.856 | Lookup · 7 objects found | 115 | 0 | 40 | 112 | 0 | 32 | 0 | 299 |
| P | 0.000–4.649 | Accept prompt · 115 local | **115** carried | 0 | **40** carried | 112 | 0 | 32 | 0 | 299 |
| P | 4.955–73.221 | Allocate slots and write 44 blocks | **159** · 115 local + 44 written | 0 | 40 | 112 | 0 | 32 | 0 | **343** |
| P GPU → LMCache | 72.767–76.295 | Offload 2 objects | 159 | 0 | 40 | **144** · L112–143 | 0 | 32 | 0 | **375** |
| Proxy → D1 | 77.693–81.488 | Dispatch prompt and metadata · D accepts | 159 | 0 | 40 | 144 | 0 | 32 | 0 | 375 |
| D1 · LMCache | 83.467–84.836 | Lookup · 2 objects found | 159 | 0 | 40 | 144 | 0 | 32 | 0 | 375 |
| D1 | 85.248–85.494 | Plan fetch · 40 local, 119 NIXL | 159 | **119** · L40–158 | 40<br>119 slots reserved | 144 | 0 | 32 | 0 | 375 |
| NIXL | 438.121 | D reports prompt KV ready | 159 | **0** · 119 ready | **159** · 40 local + 119 NIXL | 144 | 0 | 32 | 0 | **494** |
| P | 440.012 | Release uncached partial prompt block | **158** · releases L158 | 0 | 159 | 144 | 0 | 32 | 0 | **493** |
| D1 GPU → LMCache | 454.695–476.954 | Offload 7 objects | 158 | 0 | 159 | 144 | 0 | **144** · L32–143 | 0 | **605** |
| D1 | 438.121–1,886.441 | Run logits and write answer KV | 158 | 0 | 159 | 144 | 0 | 144 | 0 | 605 |
| D1 | 1,889.964 | Release request | 158 | 0 | 159 | 144 | 0 | 144 | 0 | 605 |

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

**3,205 prompt tokens · 201 GPU blocks · 12 complete LMCache objects · 133 tail tokens.** [Same turn in chat](memory-per-engine-chat.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.375–4.124 | Lookup · 9 objects found | 158 | 0 | 78 | 144 | 0 | 64 | 0 | 444 |
| P | 0.000–5.119 | Accept prompt · 158 local | **158** carried | 0 | **78** carried | 144 | 0 | 64 | 0 | 444 |
| P | 5.451–76.696 | Allocate slots and write 43 blocks | **201** · 158 local + 43 written | 0 | 78 | 144 | 0 | 64 | 0 | **487** |
| P GPU → LMCache | 76.234–85.216 | Offload 3 objects | 201 | 0 | 78 | **192** · L144–191 | 0 | 64 | 0 | **535** |
| Proxy → D2 | 81.648–87.216 | Dispatch prompt and metadata · D accepts | 201 | 0 | 78 | 192 | 0 | 64 | 0 | 535 |
| D2 · LMCache | 89.097–90.593 | Lookup · 4 objects found | 201 | 0 | 78 | 192 | 0 | 64 | 0 | 535 |
| D2 | 91.176–91.449 | Plan fetch · 78 local, 123 NIXL | 201 | **123** · L78–200 | 78<br>123 slots reserved | 192 | 0 | 64 | 0 | 535 |
| NIXL | 471.461 | D reports prompt KV ready | 201 | **0** · 123 ready | **201** · 78 local + 123 NIXL | 192 | 0 | 64 | 0 | **658** |
| P | 473.161 | Release uncached partial prompt block | **200** · releases L200 | 0 | 201 | 192 | 0 | 64 | 0 | **657** |
| D2 GPU → LMCache | 488.085–512.977 | Offload 8 objects | 200 | 0 | 201 | 192 | 0 | **192** · L64–191 | 0 | **785** |
| D2 | 471.461–1,925.061 | Run logits and write answer KV | 200 | 0 | 201 | 192 | 0 | 192 | 0 | 785 |
| D2 | 1,927.952 | Release request | 200 | 0 | 201 | 192 | 0 | 192 | 0 | 785 |

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

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: P's server / D's | Disk writes: P's / D's | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 515 | 33 | 0 / 33 | 0 / 33 | 247 | 2 / 2 | 0 / 0 | 0 / 0 |
| 2 | D2 | 1,131 | 71 | 32 / 39 | 0 / 71 | 363 | 2 / 4 | 0 / 0 | 0 / 0 |
| 3 | D3 | 1,847 | 116 | 70 / 46 | 0 / 116 | 486 | 3 / 7 | 0 / 0 | 0 / 0 |
| 4 | D1 | 2,531 | 159 | 115 / 44 | 40 / 119 | 353 | 2 / 7 | 0 / 0 | 0 / 0 |
| 5 | D2 | 3,205 | 201 | 158 / 43 | 78 / 123 | 380 | 3 / 8 | 0 / 0 | 0 / 0 |
| all | | 9,229 | 580 | 375 / 205 | 118 / 462 | | 12 / 28 | 0 / 0 | 0 / 0 |

