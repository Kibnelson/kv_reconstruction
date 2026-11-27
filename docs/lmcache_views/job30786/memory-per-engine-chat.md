# NIXL with per-engine LMCache · memory, chat RAG timeline

Conversation `1534a095279f2cb888fb0bea17bd70da`, **memory, chat RAG**, 5 turns. P and each D have their own LMCache server; 2.0 GB of LMCache memory a server. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](memory-per-engine-carry.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/eea6fcf20beb/lmctrace/job30786/benchmark_results/pd_card/pd_lmctrace_20261127T011530Z/matrix/llama32_3b_lmc_memory_nixl_per_engine_chat_all_rtxa5000_20261126_202628) · Commit `eea6fcf20beb`, run `lmc_memory_nixl_per_engine_chat_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> who takes photos of planes in the air

**515 prompt tokens · 33 GPU blocks · 2 complete LMCache objects · 3 tail tokens.** [Same turn in carry-forward](memory-per-engine-carry.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.330–2.982 | Lookup · 0 objects found | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 0.000–3.184 | Accept prompt · 0 local | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 3.313–76.766 | Allocate slots and write 33 blocks | **33** · 0 local + 33 written | 0 | 0 | 0 | 0 | 0 | 0 | **33** |
| P GPU → LMCache | 76.262–80.642 | Offload 2 objects | 33 | 0 | 0 | **32** · L0–31 | 0 | 0 | 0 | **65** |
| Proxy → D1 | 79.247–98.537 | Dispatch prompt and metadata · D accepts | 33 | 0 | 0 | 32 | 0 | 0 | 0 | 65 |
| D1 · LMCache | 101.498–102.434 | Lookup · 0 objects found | 33 | 0 | 0 | 32 | 0 | 0 | 0 | 65 |
| D1 | 104.172–104.374 | Plan fetch · 0 local, 33 NIXL | 33 | **33** · L0–32 | 0<br>33 slots reserved | 32 | 0 | 0 | 0 | 65 |
| NIXL | 347.758 | D reports prompt KV ready | 33 | **0** · 33 ready | **33** · 0 local + 33 NIXL | 32 | 0 | 0 | 0 | **98** |
| P | 349.187 | Release uncached partial prompt block | **32** · releases L32 | 0 | 33 | 32 | 0 | 0 | 0 | **97** |
| D1 GPU → LMCache | 371.622–379.369 | Offload 2 objects | 32 | 0 | 33 | 32 | 0 | **32** · L0–31 | 0 | **129** |
| D1 | 347.758–1,741.632 | Run logits and write answer KV | 32 | 0 | 33 | 32 | 0 | 32 | 0 | 129 |
| D1 | 1,745.338 | Release request | 32 | 0 | 33 | 32 | 0 | 32 | 0 | 129 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0–32 | `[0, 515)` | 1–33 / 1 | Model write | 1–33 / 1 | NIXL from P |

</details>

## Turn 2 on D2

> No, I meant photos in the air.

**657 prompt tokens · 42 GPU blocks · 2 complete LMCache objects · 145 tail tokens.** [Same turn in carry-forward](memory-per-engine-carry.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.530–2.520 | Lookup · 0 objects found | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| P | 0.000–4.201 | Accept prompt · 1 local | **1** carried | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| P | 4.445–64.535 | Allocate slots and write 41 blocks | **42** · 1 local + 41 written | 0 | 0 | 0 | 0 | 0 | 0 | **42** |
| P GPU → LMCache | 64.180–69.214 | Offload 2 objects | 42 | 0 | 0 | **32** · L0–31 | 0 | 0 | 0 | **74** |
| Proxy → D2 | 67.141–87.272 | Dispatch prompt and metadata · D accepts | 42 | 0 | 0 | 32 | 0 | 0 | 0 | 74 |
| D2 · LMCache | 90.889–91.655 | Lookup · 0 objects found | 42 | 0 | 0 | 32 | 0 | 0 | 0 | 74 |
| D2 | 93.186–93.396 | Plan fetch · 0 local, 42 NIXL | 42 | **42** · L0–41 | 0<br>42 slots reserved | 32 | 0 | 0 | 0 | 74 |
| NIXL | 364.054 | D reports prompt KV ready | 42 | **0** · 42 ready | **42** · 0 local + 42 NIXL | 32 | 0 | 0 | 0 | **116** |
| P | 365.877 | Release uncached partial prompt block | **41** · releases L41 | 0 | 42 | 32 | 0 | 0 | 0 | **115** |
| D2 GPU → LMCache | 384.212–400.564 | Offload 2 objects | 41 | 0 | 42 | 32 | 0 | **32** · L0–31 | 0 | **147** |
| D2 | 364.054–1,753.992 | Run logits and write answer KV | 41 | 0 | 42 | 32 | 0 | 32 | 0 | 147 |
| D2 | 1,757.663 | Release request | 41 | 0 | 42 | 32 | 0 | 32 | 0 | 147 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | NIXL from P |
| 1 | `[16, 32)` | 33 / 2 | Model write | 2 / 1 | NIXL from P |
| 2–41 | `[32, 657)` | 34–73 / 1 | Model write | 3–42 / 1 | NIXL from P |

</details>

## Turn 3 on D3

> So does that mean pilot of the second aircraft also acts as a photographer?

**899 prompt tokens · 57 GPU blocks · 3 complete LMCache objects · 131 tail tokens.** [Same turn in carry-forward](memory-per-engine-carry.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.496–2.465 | Lookup · 0 objects found | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| P | 0.000–4.237 | Accept prompt · 10 local | **10** carried | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| P | 4.463–66.446 | Allocate slots and write 47 blocks | **57** · 10 local + 47 written | 0 | 0 | 0 | 0 | 0 | 0 | **57** |
| P GPU → LMCache | 66.117–70.942 | Offload 3 objects | 57 | 0 | 0 | **48** · L0–47 | 0 | 0 | 0 | **105** |
| Proxy → D3 | 68.555–91.690 | Dispatch prompt and metadata · D accepts | 57 | 0 | 0 | 48 | 0 | 0 | 0 | 105 |
| D3 · LMCache | 95.606–97.298 | Lookup · 0 objects found | 57 | 0 | 0 | 48 | 0 | 0 | 0 | 105 |
| D3 | 98.124–98.345 | Plan fetch · 0 local, 57 NIXL | 57 | **57** · L0–56 | 0<br>57 slots reserved | 48 | 0 | 0 | 0 | 105 |
| NIXL | 410.072 | D reports prompt KV ready | 57 | **0** · 57 ready | **57** · 0 local + 57 NIXL | 48 | 0 | 0 | 0 | **162** |
| P | 412.760 | Release uncached partial prompt block | **56** · releases L56 | 0 | 57 | 48 | 0 | 0 | 0 | **161** |
| D3 GPU → LMCache | 426.439–438.987 | Offload 3 objects | 56 | 0 | 57 | 48 | 0 | **48** · L0–47 | 0 | **209** |
| D3 | 410.072–1,810.004 | Run logits and write answer KV | 56 | 0 | 57 | 48 | 0 | 48 | 0 | 209 |
| D3 | 1,812.844 | Release request | 56 | 0 | 57 | 48 | 0 | 48 | 0 | 209 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D3 block IDs / gen | D3 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | NIXL from P |
| 1 | `[16, 32)` | 33 / 2 | Local prefix | 2 / 1 | NIXL from P |
| 2–9 | `[32, 160)` | 34–41 / 1 | Local prefix | 3–10 / 1 | NIXL from P |
| 10 | `[160, 176)` | 73 / 2 | Model write | 11 / 1 | NIXL from P |
| 11–56 | `[176, 899)` | 74–119 / 1 | Model write | 12–57 / 1 | NIXL from P |

</details>

## Turn 4 on D1

> Is air-to-air phtography more difficult that ground-to-air?

**1,015 prompt tokens · 64 GPU blocks · 3 complete LMCache objects · 247 tail tokens.** [Same turn in carry-forward](memory-per-engine-carry.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.462–2.503 | Lookup · 1 object found | 19 | 0 | 1 | 16 | 0 | 0 | 0 | 36 |
| P | 0.000–4.468 | Accept prompt · 19 local | **19** carried | 0 | **1** carried | 16 | 0 | 0 | 0 | 36 |
| P | 4.688–67.112 | Allocate slots and write 45 blocks | **64** · 19 local + 45 written | 0 | 1 | 16 | 0 | 0 | 0 | **81** |
| P GPU → LMCache | 66.767–70.292 | Offload 2 objects | 64 | 0 | 1 | **48** · L16–47 | 0 | 0 | 0 | **113** |
| Proxy → D1 | 69.348–73.696 | Dispatch prompt and metadata · D accepts | 64 | 0 | 1 | 48 | 0 | 0 | 0 | 113 |
| D1 · LMCache | 75.522–77.035 | Lookup · 0 objects found | 64 | 0 | 1 | 48 | 0 | 0 | 0 | 113 |
| D1 | 78.337–78.737 | Plan fetch · 1 local, 63 NIXL | 64 | **63** · L1–63 | 1<br>63 slots reserved | 48 | 0 | 0 | 0 | 113 |
| NIXL | 266.979 | D reports prompt KV ready | 64 | **0** · 63 ready | **64** · 1 local + 63 NIXL | 48 | 0 | 0 | 0 | **176** |
| P | 269.411 | Release uncached partial prompt block | **63** · releases L63 | 0 | 64 | 48 | 0 | 0 | 0 | **175** |
| D1 GPU → LMCache | 284.505–298.393 | Offload 3 objects | 63 | 0 | 64 | 48 | 0 | **48** · L0–47 | 0 | **223** |
| D1 | 266.979–1,674.067 | Run logits and write answer KV | 63 | 0 | 64 | 48 | 0 | 48 | 0 | 223 |
| D1 | 1,676.967 | Release request | 63 | 0 | 64 | 48 | 0 | 48 | 0 | 223 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 33 / 2 | Local prefix | 41 / 2 | NIXL from P |
| 2–9 | `[32, 160)` | 34–41 / 1 | Local prefix | 42–49 / 1 | NIXL from P |
| 10 | `[160, 176)` | 73 / 2 | Local prefix | 50 / 1 | NIXL from P |
| 11–18 | `[176, 304)` | 74–81 / 1 | Local prefix | 51–58 / 1 | NIXL from P |
| 19 | `[304, 320)` | 119 / 2 | Model write | 59 / 1 | NIXL from P |
| 20–63 | `[320, 1015)` | 120–163 / 1 | Model write | 60–103 / 1 | NIXL from P |

</details>

## Turn 5 on D2

> are there any other types of aviation photography?

**1,152 prompt tokens · 72 GPU blocks · 4 complete LMCache objects · 128 tail tokens.** [Same turn in carry-forward](memory-per-engine-carry.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.619–2.833 | Lookup · 1 object found | 28 | 0 | 10 | 16 | 0 | 0 | 0 | 54 |
| P | 0.000–4.702 | Accept prompt · 28 local | **28** carried | 0 | **10** carried | 16 | 0 | 0 | 0 | 54 |
| P | 4.935–67.614 | Allocate slots and write 44 blocks | **72** · 28 local + 44 written | 0 | 10 | 16 | 0 | 0 | 0 | **98** |
| P GPU → LMCache | 67.267–72.015 | Offload 3 objects | 72 | 0 | 10 | **64** · L16–63 | 0 | 0 | 0 | **146** |
| Proxy → D2 | 70.798–75.357 | Dispatch prompt and metadata · D accepts | 72 | 0 | 10 | 64 | 0 | 0 | 0 | 146 |
| D2 · LMCache | 77.069–78.734 | Lookup · 0 objects found | 72 | 0 | 10 | 64 | 0 | 0 | 0 | 146 |
| D2 | 80.050–80.440 | Plan fetch · 10 local, 62 NIXL | 72 | **62** · L10–71 | 10<br>62 slots reserved | 64 | 0 | 0 | 0 | 146 |
| NIXL | 265.316 | D reports prompt KV ready | 72 | **0** · 62 ready | **72** · 10 local + 62 NIXL | 64 | 0 | 0 | 0 | **208** |
| D2 GPU → LMCache | 282.902–300.246 | Offload 4 objects | 72 | 0 | 72 | 64 | 0 | **64** · L0–63 | 0 | **272** |
| D2 | 265.316–1,674.995 | Run logits and write answer KV | 72 | 0 | 72 | 64 | 0 | 64 | 0 | 272 |
| D2 | 1,678.501 | Release request | 72 | 0 | 72 | 64 | 0 | 64 | 0 | 272 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 33 / 2 | Local prefix | 2 / 1 | Local prefix |
| 2–9 | `[32, 160)` | 34–41 / 1 | Local prefix | 3–10 / 1 | Local prefix |
| 10 | `[160, 176)` | 73 / 2 | Local prefix | 50 / 1 | NIXL from P |
| 11–18 | `[176, 304)` | 74–81 / 1 | Local prefix | 51–58 / 1 | NIXL from P |
| 19 | `[304, 320)` | 119 / 2 | Local prefix | 59 / 1 | NIXL from P |
| 20–27 | `[320, 448)` | 120–127 / 1 | Local prefix | 60–67 / 1 | NIXL from P |
| 28 | `[448, 464)` | 163 / 2 | Model write | 68 / 1 | NIXL from P |
| 29–71 | `[464, 1152)` | 164–206 / 1 | Model write | 69–111 / 1 | NIXL from P |

</details>

## The turns together

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: P's server / D's | Disk writes: P's / D's | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 515 | 33 | 0 / 33 | 0 / 33 | 243 | 2 / 2 | 0 / 0 | 0 / 0 |
| 2 | D2 | 657 | 42 | 1 / 41 | 0 / 42 | 271 | 2 / 2 | 0 / 0 | 0 / 0 |
| 3 | D3 | 899 | 57 | 10 / 47 | 0 / 57 | 312 | 3 / 3 | 0 / 0 | 0 / 0 |
| 4 | D1 | 1,015 | 64 | 19 / 45 | 1 / 63 | 188 | 2 / 3 | 0 / 0 | 0 / 0 |
| 5 | D2 | 1,152 | 72 | 28 / 44 | 10 / 62 | 185 | 3 / 4 | 0 / 0 | 0 / 0 |
| all | | 4,238 | 268 | 58 / 210 | 11 / 257 | | 12 / 14 | 0 / 0 | 0 / 0 |

