# NIXL with shared LMCache · memory, chat RAG timeline

Conversation `1534a095279f2cb888fb0bea17bd70da`, **memory, chat RAG**, 5 turns. P and the three D engines share one LMCache server; 8.0 GB of LMCache memory a server. GPU KV pool: 106,800 tokens (6,675 blocks) an engine (vLLM's default, no cap).

[Paired timeline](memory-shared-carry.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/eea6fcf20beb/lmctrace/job30786/benchmark_results/pd_card/pd_lmctrace_20261127T011530Z/matrix/llama32_3b_lmc_memory_nixl_shared_chat_all_rtxa5000_20261126_201728) · Commit `eea6fcf20beb`, run `lmc_memory_nixl_shared_chat_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> who takes photos of planes in the air

**515 prompt tokens · 33 GPU blocks · 2 complete LMCache objects · 3 tail tokens.** [Same turn in carry-forward](memory-shared-carry.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.244–2.976 | Lookup · 0 objects found | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 0.000–4.618 | Accept prompt · 0 local | 0 | 0 | 0 | 0 | 0 | 0 |
| P | 4.730–75.673 | Allocate slots and write 33 blocks | **33** · 0 local + 33 written | 0 | 0 | 0 | 0 | **33** |
| P GPU → LMCache | 75.318–79.621 | Offload 2 objects | 33 | 0 | 0 | **32** · L0–31 | 0 | **65** |
| Proxy → D1 | 77.595–119.089 | Dispatch prompt and metadata · D accepts | 33 | 0 | 0 | 32 | 0 | 65 |
| D1 · LMCache | 122.044–122.959 | Lookup · 2 objects found | 33 | 0 | 0 | 32 | 0 | 65 |
| D1 | 124.929–125.101 | Plan fetch · 0 local, 33 NIXL | 33 | **33** · L0–32 | 0<br>33 slots reserved | 32 | 0 | 65 |
| NIXL | 369.648 | D reports prompt KV ready | 33 | **0** · 33 ready | **33** · 0 local + 33 NIXL | 32 | 0 | **98** |
| P | 372.519 | Release uncached partial prompt block | **32** · releases L32 | 0 | 33 | 32 | 0 | **97** |
| D1 | 369.648–1,756.783 | Run logits and write answer KV | 32 | 0 | 33 | 32 | 0 | 97 |
| D1 | 1,760.227 | Release request | 32 | 0 | 33 | 32 | 0 | 97 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0–32 | `[0, 515)` | 1–33 / 1 | Model write | 1–33 / 1 | NIXL from P |

</details>

## Turn 2 on D2

> No, I meant photos in the air.

**657 prompt tokens · 42 GPU blocks · 2 complete LMCache objects · 145 tail tokens.** [Same turn in carry-forward](memory-shared-carry.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.392–2.498 | Lookup · 0 objects found | 1 | 0 | 0 | 0 | 0 | 1 |
| P | 0.000–4.380 | Accept prompt · 1 local | **1** carried | 0 | 0 | 0 | 0 | 1 |
| P | 4.621–64.814 | Allocate slots and write 41 blocks | **42** · 1 local + 41 written | 0 | 0 | 0 | 0 | **42** |
| P GPU → LMCache | 64.476–68.234 | Offload 2 objects | 42 | 0 | 0 | **32** · L0–31 | 0 | **74** |
| Proxy → D2 | 67.588–78.038 | Dispatch prompt and metadata · D accepts | 42 | 0 | 0 | 32 | 0 | 74 |
| D2 · LMCache | 80.744–81.664 | Lookup · 2 objects found | 42 | 0 | 0 | 32 | 0 | 74 |
| D2 | 81.921–82.135 | Plan fetch · 0 local, 42 NIXL | 42 | **42** · L0–41 | 0<br>42 slots reserved | 32 | 0 | 74 |
| NIXL | 358.011 | D reports prompt KV ready | 42 | **0** · 42 ready | **42** · 0 local + 42 NIXL | 32 | 0 | **116** |
| P | 360.614 | Release uncached partial prompt block | **41** · releases L41 | 0 | 42 | 32 | 0 | **115** |
| D2 | 358.011–1,747.220 | Run logits and write answer KV | 41 | 0 | 42 | 32 | 0 | 115 |
| D2 | 1,750.090 | Release request | 41 | 0 | 42 | 32 | 0 | 115 |

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

**899 prompt tokens · 57 GPU blocks · 3 complete LMCache objects · 131 tail tokens.** [Same turn in carry-forward](memory-shared-carry.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.727–3.392 | Lookup · 0 objects found | 10 | 0 | 0 | 0 | 0 | 10 |
| P | 0.000–4.150 | Accept prompt · 10 local | **10** carried | 0 | 0 | 0 | 0 | 10 |
| P | 4.375–66.882 | Allocate slots and write 47 blocks | **57** · 10 local + 47 written | 0 | 0 | 0 | 0 | **57** |
| P GPU → LMCache | 66.533–74.019 | Offload 3 objects | 57 | 0 | 0 | **48** · L0–47 | 0 | **105** |
| Proxy → D3 | 69.929–82.536 | Dispatch prompt and metadata · D accepts | 57 | 0 | 0 | 48 | 0 | 105 |
| D3 · LMCache | 86.092–86.896 | Lookup · 3 objects found | 57 | 0 | 0 | 48 | 0 | 105 |
| D3 | 87.509–87.704 | Plan fetch · 0 local, 57 NIXL | 57 | **57** · L0–56 | 0<br>57 slots reserved | 48 | 0 | 105 |
| NIXL | 398.683 | D reports prompt KV ready | 57 | **0** · 57 ready | **57** · 0 local + 57 NIXL | 48 | 0 | **162** |
| P | 400.350 | Release uncached partial prompt block | **56** · releases L56 | 0 | 57 | 48 | 0 | **161** |
| D3 | 398.683–1,799.472 | Run logits and write answer KV | 56 | 0 | 57 | 48 | 0 | 161 |
| D3 | 1,803.112 | Release request | 56 | 0 | 57 | 48 | 0 | 161 |

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

**1,015 prompt tokens · 64 GPU blocks · 3 complete LMCache objects · 247 tail tokens.** [Same turn in carry-forward](memory-shared-carry.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.454–2.521 | Lookup · 1 object found | 19 | 0 | 1 | 16 | 0 | 36 |
| P | 0.000–4.333 | Accept prompt · 19 local | **19** carried | 0 | **1** carried | 16 | 0 | 36 |
| P | 4.560–66.652 | Allocate slots and write 45 blocks | **64** · 19 local + 45 written | 0 | 1 | 16 | 0 | **81** |
| Proxy → D1 | 69.937–73.133 | Dispatch prompt and metadata · D accepts | 64 | 0 | 1 | 16 | 0 | 81 |
| P GPU → LMCache | 66.348–73.570 | Offload 2 objects | 64 | 0 | 1 | **48** · L16–47 | 0 | **113** |
| D1 · LMCache | 74.845–76.305 | Lookup · 3 objects found | 64 | 0 | 1 | 48 | 0 | 113 |
| D1 | 77.929–78.339 | Plan fetch · 1 local, 63 NIXL | 64 | **63** · L1–63 | 1<br>63 slots reserved | 48 | 0 | 113 |
| NIXL | 265.949 | D reports prompt KV ready | 64 | **0** · 63 ready | **64** · 1 local + 63 NIXL | 48 | 0 | **176** |
| P | 268.731 | Release uncached partial prompt block | **63** · releases L63 | 0 | 64 | 48 | 0 | **175** |
| D1 | 265.949–1,671.618 | Run logits and write answer KV | 63 | 0 | 64 | 48 | 0 | 175 |
| D1 | 1,675.204 | Release request | 63 | 0 | 64 | 48 | 0 | 175 |

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

**1,152 prompt tokens · 72 GPU blocks · 4 complete LMCache objects · 128 tail tokens.** [Same turn in carry-forward](memory-shared-carry.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.362–2.175 | Lookup · 1 object found | 28 | 0 | 10 | 16 | 0 | 54 |
| P | 0.000–2.608 | Accept prompt · 28 local | **28** carried | 0 | **10** carried | 16 | 0 | 54 |
| P | 2.848–65.185 | Allocate slots and write 44 blocks | **72** · 28 local + 44 written | 0 | 10 | 16 | 0 | **98** |
| P GPU → LMCache | 64.824–71.934 | Offload 3 objects | 72 | 0 | 10 | **64** · L16–63 | 0 | **146** |
| Proxy → D2 | 72.529–77.027 | Dispatch prompt and metadata · D accepts | 72 | 0 | 10 | 64 | 0 | 146 |
| D2 · LMCache | 78.467–80.699 | Lookup · 4 objects found | 72 | 0 | 10 | 64 | 0 | 146 |
| D2 | 82.497–82.721 | Plan fetch · 10 local, 62 NIXL | 72 | **62** · L10–71 | 10<br>62 slots reserved | 64 | 0 | 146 |
| NIXL | 267.123 | D reports prompt KV ready | 72 | **0** · 62 ready | **72** · 10 local + 62 NIXL | 64 | 0 | **208** |
| D2 | 267.123–1,674.937 | Run logits and write answer KV | 72 | 0 | 72 | 64 | 0 | 208 |
| D2 | 1,677.917 | Release request | 72 | 0 | 72 | 64 | 0 | 208 |

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

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: the server | Disk writes | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 515 | 33 | 0 / 33 | 0 / 33 | 245 | 2 | 0 | 0 / 0 |
| 2 | D2 | 657 | 42 | 1 / 41 | 0 / 42 | 276 | 2 | 0 | 0 / 0 |
| 3 | D3 | 899 | 57 | 10 / 47 | 0 / 57 | 311 | 3 | 0 | 0 / 0 |
| 4 | D1 | 1,015 | 64 | 19 / 45 | 1 / 63 | 188 | 2 | 0 | 0 / 0 |
| 5 | D2 | 1,152 | 72 | 28 / 44 | 10 / 62 | 184 | 3 | 0 | 0 / 0 |
| all | | 4,238 | 268 | 58 / 210 | 11 / 257 | | 12 | 0 | 0 / 0 |

