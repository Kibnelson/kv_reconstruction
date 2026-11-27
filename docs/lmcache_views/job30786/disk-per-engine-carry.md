# NIXL with per-engine LMCache · disk, carry-forward timeline

Conversation `1534a095279f2cb888fb0bea17bd70da`, **disk, carry-forward**, 5 turns. P and each D have their own LMCache server; 2.0 GB of LMCache memory a server, the disk tier with O_DIRECT, store policy `skip_l1`, prefetch policy `retain`. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](disk-per-engine-chat.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/eea6fcf20beb/lmctrace/job30786/benchmark_results/pd_card/pd_lmctrace_20261127T011530Z/matrix/llama32_3b_lmc_disk_nixl_per_engine_carry_all_rtxa5000_20261126_203227) · Commit `eea6fcf20beb`, run `lmc_disk_nixl_per_engine_carry_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> who takes photos of planes in the air

**515 prompt tokens · 33 GPU blocks · 2 complete LMCache objects · 3 tail tokens.** [Same turn in chat](disk-per-engine-chat.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.893–4.509 | Lookup · 0 objects found | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 0.000–4.942 | Accept prompt · 0 local | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 5.078–76.679 | Allocate slots and write 33 blocks | **33** · 0 local + 33 written | 0 | 0 | 0 | 0 | 0 | 0 | **33** |
| P GPU → LMCache | 76.382–83.037 | Offload 2 objects | 33 | 0 | 0 | **32** · L0–31 | 0 | 0 | 0 | **65** |
| Proxy → D1 | 80.417–92.379 | Dispatch prompt and metadata · D accepts | 33 | 0 | 0 | 32 | 0 | 0 | 0 | 65 |
| D1 · LMCache | 95.639–98.193 | Lookup · 0 objects found | 33 | 0 | 0 | 32 | 0 | 0 | 0 | 65 |
| D1 | 99.346–99.511 | Plan fetch · 0 local, 33 NIXL | 33 | **33** · L0–32 | 0<br>33 slots reserved | 32 | 0 | 0 | 0 | 65 |
| P · LMCache | 168.386–168.796 | 2 disk writes · release CPU copies | 33 | 33 | 0 | **0** | **32** · +L0–31 | 0 | 0 | 65 · peak 97 |
| NIXL | 345.086 | D reports prompt KV ready | 33 | **0** · 33 ready | **33** · 0 local + 33 NIXL | 0 | 32 | 0 | 0 | **98** |
| P | 347.068 | Release uncached partial prompt block | **32** · releases L32 | 0 | 33 | 0 | 32 | 0 | 0 | **97** |
| D1 GPU → LMCache | 366.928–374.634 | Offload 2 objects | 32 | 0 | 33 | 0 | 32 | **32** · L0–31 | 0 | **129** |
| D1 · LMCache | 444.913–445.268 | 2 disk writes · release CPU copies | 32 | 0 | 33 | 0 | 32 | **0** | **32** · +L0–31 | 129 · peak 161 |
| D1 | 345.086–1,736.758 | Run logits and write answer KV | 32 | 0 | 33 | 0 | 32 | 0 | 32 | 129 |
| D1 | 1,740.390 | Release request | 32 | 0 | 33 | 0 | 32 | 0 | 32 | 129 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0–32 | `[0, 515)` | 1–33 / 1 | Model write | 1–33 / 1 | NIXL from P |

</details>

## Turn 2 on D2

> No, I meant photos in the air.

**1,131 prompt tokens · 71 GPU blocks · 4 complete LMCache objects · 107 tail tokens.** [Same turn in chat](disk-per-engine-chat.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.782–4.791 | Lookup · 2 objects found | 32 | 0 | 0 | 0 | 32 | 0 | 0 | 64 |
| P | 0.000–6.389 | Accept prompt · 32 local | **32** carried | 0 | 0 | 0 | 32 | 0 | 0 | 64 |
| P | 6.659–61.992 | Allocate slots and write 39 blocks | **71** · 32 local + 39 written | 0 | 0 | 0 | 32 | 0 | 0 | **103** |
| P GPU → LMCache | 61.634–65.810 | Offload 2 objects | 71 | 0 | 0 | **32** · L32–63 | 32 | 0 | 0 | **135** |
| Proxy → D2 | 65.328–77.945 | Dispatch prompt and metadata · D accepts | 71 | 0 | 0 | 32 | 32 | 0 | 0 | 135 |
| D2 · LMCache | 81.679–84.133 | Lookup · 0 objects found | 71 | 0 | 0 | 32 | 32 | 0 | 0 | 135 |
| D2 | 85.247–85.433 | Plan fetch · 0 local, 71 NIXL | 71 | **71** · L0–70 | 0<br>71 slots reserved | 32 | 32 | 0 | 0 | 135 |
| P · LMCache | 133.937–134.320 | 2 disk writes · release CPU copies | 71 | 71 | 0 | **0** | **64** · +L32–63 | 0 | 0 | 135 · peak 167 |
| NIXL | 440.129 | D reports prompt KV ready | 71 | **0** · 71 ready | **71** · 0 local + 71 NIXL | 0 | 64 | 0 | 0 | **206** |
| P | 441.569 | Release uncached partial prompt block | **70** · releases L70 | 0 | 71 | 0 | 64 | 0 | 0 | **205** |
| D2 GPU → LMCache | 458.462–472.097 | Offload 4 objects | 70 | 0 | 71 | 0 | 64 | **64** · L0–63 | 0 | **269** |
| D2 · LMCache | 612.169–612.806 | 4 disk writes · release CPU copies | 70 | 0 | 71 | 0 | 64 | **0** | **64** · +L0–63 | 269 · peak 333 |
| D2 | 440.129–1,849.630 | Run logits and write answer KV | 70 | 0 | 71 | 0 | 64 | 0 | 64 | 269 |
| D2 | 1,853.099 | Release request | 70 | 0 | 71 | 0 | 64 | 0 | 64 | 269 |

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

**1,847 prompt tokens · 116 GPU blocks · 7 complete LMCache objects · 55 tail tokens.** [Same turn in chat](disk-per-engine-chat.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.874–5.670 | Lookup · 4 objects found | 70 | 0 | 0 | 0 | 64 | 0 | 0 | 134 |
| P | 0.000–6.513 | Accept prompt · 70 local | **70** carried | 0 | 0 | 0 | 64 | 0 | 0 | 134 |
| P | 6.786–74.074 | Allocate slots and write 46 blocks | **116** · 70 local + 46 written | 0 | 0 | 0 | 64 | 0 | 0 | **180** |
| P GPU → LMCache | 73.680–80.568 | Offload 3 objects | 116 | 0 | 0 | **48** · L64–111 | 64 | 0 | 0 | **228** |
| Proxy → D3 | 78.234–100.847 | Dispatch prompt and metadata · D accepts | 116 | 0 | 0 | 48 | 64 | 0 | 0 | 228 |
| D3 · LMCache | 104.427–106.949 | Lookup · 0 objects found | 116 | 0 | 0 | 48 | 64 | 0 | 0 | 228 |
| D3 | 108.106–108.310 | Plan fetch · 0 local, 116 NIXL | 116 | **116** · L0–115 | 0<br>116 slots reserved | 48 | 64 | 0 | 0 | 228 |
| P · LMCache | 188.478–188.897 | 3 disk writes · release CPU copies | 116 | 116 | 0 | **0** | **112** · +L64–111 | 0 | 0 | 228 · peak 276 |
| NIXL | 597.800 | D reports prompt KV ready | 116 | **0** · 116 ready | **116** · 0 local + 116 NIXL | 0 | 112 | 0 | 0 | **344** |
| P | 600.232 | Release uncached partial prompt block | **115** · releases L115 | 0 | 116 | 0 | 112 | 0 | 0 | **343** |
| D3 GPU → LMCache | 616.155–638.477 | Offload 7 objects | 115 | 0 | 116 | 0 | 112 | **112** · L0–111 | 0 | **455** |
| D3 · LMCache | 912.278–913.012 | 7 disk writes · release CPU copies | 115 | 0 | 116 | 0 | 112 | **0** | **112** · +L0–111 | 455 · peak 567 |
| D3 | 597.800–2,026.729 | Run logits and write answer KV | 115 | 0 | 116 | 0 | 112 | 0 | 112 | 455 |
| D3 | 2,030.198 | Release request | 115 | 0 | 116 | 0 | 112 | 0 | 112 | 455 |

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

**2,531 prompt tokens · 159 GPU blocks · 9 complete LMCache objects · 227 tail tokens.** [Same turn in chat](disk-per-engine-chat.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.165–5.553 | Lookup · 7 objects found | 115 | 0 | 40 | 0 | 112 | 0 | 32 | 299 |
| P | 0.000–6.912 | Accept prompt · 115 local | **115** carried | 0 | **40** carried | 0 | 112 | 0 | 32 | 299 |
| P | 7.240–76.009 | Allocate slots and write 44 blocks | **159** · 115 local + 44 written | 0 | 40 | 0 | 112 | 0 | 32 | **343** |
| P GPU → LMCache | 75.658–81.137 | Offload 2 objects | 159 | 0 | 40 | **32** · L112–143 | 112 | 0 | 32 | **375** |
| Proxy → D1 | 80.464–86.466 | Dispatch prompt and metadata · D accepts | 159 | 0 | 40 | 32 | 112 | 0 | 32 | 375 |
| D1 · LMCache | 88.630–92.355 | Lookup · 2 objects found | 159 | 0 | 40 | 32 | 112 | 0 | 32 | 375 |
| D1 | 93.624–94.092 | Plan fetch · 40 local, 119 NIXL | 159 | **119** · L40–158 | 40<br>119 slots reserved | 32 | 112 | 0 | 32 | 375 |
| P · LMCache | 164.316–164.876 | 2 disk writes · release CPU copies | 159 | 119 | 40 | **0** | **144** · +L112–143 | 0 | 32 | 375 · peak 407 |
| NIXL | 446.111 | D reports prompt KV ready | 159 | **0** · 119 ready | **159** · 40 local + 119 NIXL | 0 | 144 | 0 | 32 | **494** |
| P | 447.737 | Release uncached partial prompt block | **158** · releases L158 | 0 | 159 | 0 | 144 | 0 | 32 | **493** |
| D1 GPU → LMCache | 464.503–487.404 | Offload 7 objects | 158 | 0 | 159 | 0 | 144 | **112** · L32–143 | 32 | **605** |
| D1 · LMCache | 753.361–753.949 | 7 disk writes · release CPU copies | 158 | 0 | 159 | 0 | 144 | **0** | **144** · +L32–143 | 605 · peak 717 |
| D1 | 446.111–1,896.520 | Run logits and write answer KV | 158 | 0 | 159 | 0 | 144 | 0 | 144 | 605 |
| D1 | 1,900.161 | Release request | 158 | 0 | 159 | 0 | 144 | 0 | 144 | 605 |

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

**3,205 prompt tokens · 201 GPU blocks · 12 complete LMCache objects · 133 tail tokens.** [Same turn in chat](disk-per-engine-chat.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.734–4.711 | Lookup · 9 objects found | 158 | 0 | 78 | 0 | 144 | 0 | 64 | 444 |
| P | 0.000–5.569 | Accept prompt · 158 local | **158** carried | 0 | **78** carried | 0 | 144 | 0 | 64 | 444 |
| P | 5.747–75.248 | Allocate slots and write 43 blocks | **201** · 158 local + 43 written | 0 | 78 | 0 | 144 | 0 | 64 | **487** |
| P GPU → LMCache | 74.950–80.778 | Offload 3 objects | 201 | 0 | 78 | **48** · L144–191 | 144 | 0 | 64 | **535** |
| Proxy → D2 | 80.240–87.497 | Dispatch prompt and metadata · D accepts | 201 | 0 | 78 | 48 | 144 | 0 | 64 | 535 |
| D2 · LMCache | 90.118–95.390 | Lookup · 4 objects found | 201 | 0 | 78 | 48 | 144 | 0 | 64 | 535 |
| D2 | 96.877–97.365 | Plan fetch · 78 local, 123 NIXL | 201 | **123** · L78–200 | 78<br>123 slots reserved | 48 | 144 | 0 | 64 | 535 |
| P · LMCache | 198.192–198.832 | 3 disk writes · release CPU copies | 201 | 123 | 78 | **0** | **192** · +L144–191 | 0 | 64 | 535 · peak 583 |
| NIXL | 463.167 | D reports prompt KV ready | 201 | **0** · 123 ready | **201** · 78 local + 123 NIXL | 0 | 192 | 0 | 64 | **658** |
| P | 465.155 | Release uncached partial prompt block | **200** · releases L200 | 0 | 201 | 0 | 192 | 0 | 64 | **657** |
| D2 GPU → LMCache | 481.616–505.166 | Offload 8 objects | 200 | 0 | 201 | 0 | 192 | **128** · L64–191 | 64 | **785** |
| D2 · LMCache | 832.140–832.866 | 8 disk writes · release CPU copies | 200 | 0 | 201 | 0 | 192 | **0** | **192** · +L64–191 | 785 · peak 913 |
| D2 | 463.167–1,917.975 | Run logits and write answer KV | 200 | 0 | 201 | 0 | 192 | 0 | 192 | 785 |
| D2 | 1,921.378 | Release request | 200 | 0 | 201 | 0 | 192 | 0 | 192 | 785 |

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
| 1 | D1 | 515 | 33 | 0 / 33 | 0 / 33 | 246 | 2 / 2 | 2 / 2 | 0 / 0 |
| 2 | D2 | 1,131 | 71 | 32 / 39 | 0 / 71 | 355 | 2 / 4 | 2 / 4 | 0 / 0 |
| 3 | D3 | 1,847 | 116 | 70 / 46 | 0 / 116 | 489 | 3 / 7 | 3 / 7 | 0 / 0 |
| 4 | D1 | 2,531 | 159 | 115 / 44 | 40 / 119 | 352 | 2 / 7 | 2 / 7 | 0 / 0 |
| 5 | D2 | 3,205 | 201 | 158 / 43 | 78 / 123 | 366 | 3 / 8 | 3 / 8 | 0 / 0 |
| all | | 9,229 | 580 | 375 / 205 | 118 / 462 | | 12 / 28 | 12 / 28 | 0 / 0 |

