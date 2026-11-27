# NIXL with shared LMCache · disk, carry-forward timeline

Conversation `cfbcfa3e382f12aed450b10f87a72aeb`, **disk, carry-forward**, 6 turns. P and the three D engines share one LMCache server; 8.0 GB of LMCache memory a server, the disk tier with O_DIRECT, store policy `skip_l1`, prefetch policy `retain`. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](../disk-shared-chat/05-cfbcfa3e.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/matrix/llama32_3b_lmc_disk_nixl_shared_carry_all_rtxa5000_20261126_230851) · Commit `268a9a96a324`, run `lmc_disk_nixl_shared_carry_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2) · [Turn 6](#turn-6-on-d3)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> Where do guinea pigs sleep in the wild?

**1,087 prompt tokens · 68 GPU blocks · 4 complete LMCache objects · 63 tail tokens.** [Same turn in chat](../disk-shared-chat/05-cfbcfa3e.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.385–2.088 | Lookup · 4 objects found | 67 | 0 | 67 | 64 | 64 | 262 |
| P | 0.000–2.610 | Accept prompt · 67 local | **67** carried | 0 | **67** carried | 64 | 64 | 262 |
| P | 2.856–18.625 | Allocate slots and write 1 blocks | **68** · 67 local + 1 written | 0 | 67 | 64 | 64 | **263** |
| Proxy → D1 | 21.707–25.600 | Dispatch prompt and metadata · D accepts | 68 | 0 | 67 | 64 | 64 | 263 |
| D1 · LMCache | 27.183–28.111 | Lookup · 4 objects found | 68 | 0 | 67 | 64 | 64 | 263 |
| D1 | 28.589–28.927 | Plan fetch · 67 local, 1 NIXL | 68 | **1** · L67 | 67<br>1 slots reserved | 64 | 64 | 263 |
| NIXL | 35.165 | D reports prompt KV ready | 68 | **0** · 1 ready | **68** · 67 local + 1 NIXL | 64 | 64 | **264** |
| P | 44.883 | Release uncached partial prompt block | **67** · releases L67 | 0 | 68 | 64 | 64 | **263** |
| D1 | 35.165–1,483.248 | Run logits and write answer KV | 67 | 0 | 68 | 64 | 64 | 263 |
| D1 | 1,486.622 | Release request | 67 | 0 | 68 | 64 | 64 | 263 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 201 / 2 | Local prefix | 167 / 2 | Local prefix |
| 2–66 | `[32, 1072)` | 202–266 / 1 | Local prefix | 168–232 / 1 | Local prefix |
| 67 | `[1072, 1087)` | 1495 / 2 | Model write | 1116 / 2 | NIXL from P |

</details>

## Turn 2 on D2

> where does Domestic guinea pigs live and what does Domestic guinea pigs activity look like?

**2,301 prompt tokens · 144 GPU blocks · 8 complete LMCache objects · 253 tail tokens.** [Same turn in chat](../disk-shared-chat/05-cfbcfa3e.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.955–4.542 | Lookup · 4 objects found | 76 | 0 | 76 | 64 | 64 | 280 |
| P | 0.000–6.823 | Accept prompt · 76 local | **76** carried | 0 | **76** carried | 64 | 64 | 280 |
| P | 7.120–102.618 | Allocate slots and write 68 blocks | **144** · 76 local + 68 written | 0 | 76 | 64 | 64 | **348** |
| P GPU → LMCache | 102.072–110.298 | Offload 4 objects | 144 | 0 | 76 | **128** · L64–127 | 64 | **412** |
| Proxy → D2 | 107.187–112.493 | Dispatch prompt and metadata · D accepts | 144 | 0 | 76 | 128 | 64 | 412 |
| D2 · LMCache | 115.569–117.844 | Lookup · 8 objects found | 144 | 0 | 76 | 128 | 64 | 412 |
| D2 | 118.492–118.943 | Plan fetch · 76 local, 68 NIXL | 144 | **68** · L76–143 | 76<br>68 slots reserved | 128 | 64 | 412 |
| P · LMCache | 252.299–252.915 | 4 disk writes · release CPU copies | 144 | 68 | 76 | **64** | **128** · +L64–127 | 412 · peak 476 |
| NIXL | 323.232 | D reports prompt KV ready | 144 | **0** · 68 ready | **144** · 76 local + 68 NIXL | 64 | 128 | **480** |
| P | 336.983 | Release uncached partial prompt block | **143** · releases L143 | 0 | 144 | 64 | 128 | **479** |
| D2 | 323.232–1,780.987 | Run logits and write answer KV | 143 | 0 | 144 | 64 | 128 | 479 |
| D2 | 1,784.429 | Release request | 143 | 0 | 144 | 64 | 128 | 479 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 201 / 2 | Local prefix | 209 / 2 | Local prefix |
| 2–66 | `[32, 1072)` | 202–266 / 1 | Local prefix | 210–274 / 1 | Local prefix |
| 67 | `[1072, 1088)` | 267 / 2 | Local prefix | 275 / 1 | Local prefix |
| 68–75 | `[1088, 1216)` | 268–275 / 1 | Local prefix | 276–283 / 1 | Local prefix |
| 76 | `[1216, 1232)` | 1495 / 3 | Model write | 1327 / 2 | NIXL from P |
| 77–143 | `[1232, 2301)` | 1496–1562 / 1 | Model write | 1328–1394 / 1 | NIXL from P |

</details>

## Turn 3 on D3

> more about guinea pig appearance or body

**3,149 prompt tokens · 197 GPU blocks · 12 complete LMCache objects · 77 tail tokens.** [Same turn in chat](../disk-shared-chat/05-cfbcfa3e.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.709–4.207 | Lookup · 8 objects found | 143 | 0 | 76 | 64 | 128 | 411 |
| P | 0.000–5.216 | Accept prompt · 143 local | **143** carried | 0 | **76** carried | 64 | 128 | 411 |
| P | 5.551–89.850 | Allocate slots and write 54 blocks | **197** · 143 local + 54 written | 0 | 76 | 64 | 128 | **465** |
| P GPU → LMCache | 89.387–94.879 | Offload 4 objects | 197 | 0 | 76 | **128** · L128–191 | 128 | **529** |
| Proxy → D3 | 95.221–102.404 | Dispatch prompt and metadata · D accepts | 197 | 0 | 76 | 128 | 128 | 529 |
| D3 · LMCache | 104.480–191.543 | Lookup · 12 objects found, 4 read from disk into CPU | 197 | 0 | 76 | **192** | 128 | **593** |
| D3 | 192.365–192.828 | Plan fetch · 76 local, 121 NIXL | 197 | **121** · L76–196 | 76<br>121 slots reserved | 192 | 128 | 593 |
| P · LMCache | 313.991–314.738 | 4 disk writes · release CPU copies | 197 | 121 | 76 | **128** | **192** · +L128–191 | 593 · peak 657 |
| NIXL | 551.433 | D reports prompt KV ready | 197 | **0** · 121 ready | **197** · 76 local + 121 NIXL | 128 | 192 | **714** |
| P | 567.449 | Release uncached partial prompt block | **196** · releases L196 | 0 | 197 | 128 | 192 | **713** |
| D3 | 551.433–2,019.585 | Run logits and write answer KV | 196 | 0 | 197 | 128 | 192 | 713 |
| D3 | 2,023.065 | Release request | 196 | 0 | 197 | 128 | 192 | 713 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D3 block IDs / gen | D3 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 201 / 2 | Local prefix | 124 / 2 | Local prefix |
| 2–66 | `[32, 1072)` | 202–266 / 1 | Local prefix | 125–189 / 1 | Local prefix |
| 67 | `[1072, 1088)` | 267 / 2 | Local prefix | 190 / 1 | Local prefix |
| 68–74 | `[1088, 1200)` | 268–274 / 1 | Local prefix | 191–197 / 1 | Local prefix |
| 75 | `[1200, 1216)` | 275 / 1 | Local prefix | 198 / 2 | Local prefix |
| 76 | `[1216, 1232)` | 1495 / 3 | Local prefix | 1442 / 2 | NIXL from P |
| 77–142 | `[1232, 2288)` | 1496–1561 / 1 | Local prefix | 1443–1508 / 1 | NIXL from P |
| 143 | `[2288, 2304)` | 1562 / 2 | Model write | 1509 / 1 | NIXL from P |
| 144–196 | `[2304, 3149)` | 1563–1615 / 1 | Model write | 1510–1562 / 1 | NIXL from P |

</details>

## Turn 4 on D1

> can you clarify the breed here? is there different species of Guinea pig? maybe they are from different origin?

**4,481 prompt tokens · 281 GPU blocks · 17 complete LMCache objects · 129 tail tokens.** [Same turn in chat](../disk-shared-chat/05-cfbcfa3e.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.901–6.172 | Lookup · 12 objects found | 196 | 0 | 76 | 128 | 192 | 592 |
| P | 0.000–7.712 | Accept prompt · 196 local | **196** carried | 0 | **76** carried | 128 | 192 | 592 |
| P | 8.092–145.941 | Allocate slots and write 85 blocks | **281** · 196 local + 85 written | 0 | 76 | 128 | 192 | **677** |
| P GPU → LMCache | 145.254–154.937 | Offload 5 objects | 281 | 0 | 76 | **208** · L192–271 | 192 | **757** |
| Proxy → D1 | 152.589–158.844 | Dispatch prompt and metadata · D accepts | 281 | 0 | 76 | 208 | 192 | 757 |
| D1 · LMCache | 160.814–251.697 | Lookup · 17 objects found, 4 read from disk into CPU | 281 | 0 | 76 | **272** | 192 | **821** |
| D1 | 253.242–253.844 | Plan fetch · 76 local, 205 NIXL | 281 | **205** · L76–280 | 76<br>205 slots reserved | 272 | 192 | 821 |
| P · LMCache | 433.361–433.987 | 5 disk writes · release CPU copies | 281 | 205 | 76 | **192** | **272** · +L192–271 | 821 · peak 901 |
| NIXL | 855.169 | D reports prompt KV ready | 281 | **0** · 205 ready | **281** · 76 local + 205 NIXL | 192 | 272 | **1026** |
| P | 877.241 | Release uncached partial prompt block | **280** · releases L280 | 0 | 281 | 192 | 272 | **1025** |
| D1 | 855.169–2,361.095 | Run logits and write answer KV | 280 | 0 | 281 | 192 | 272 | 1025 |
| D1 | 2,364.729 | Release request | 280 | 0 | 281 | 192 | 272 | 1025 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D1 block IDs / gen | D1 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 201 / 2 | Local prefix | 167 / 2 | Local prefix |
| 2–66 | `[32, 1072)` | 202–266 / 1 | Local prefix | 168–232 / 1 | Local prefix |
| 67 | `[1072, 1088)` | 267 / 2 | Local prefix | 233 / 1 | Local prefix |
| 68–75 | `[1088, 1216)` | 268–275 / 1 | Local prefix | 234–241 / 1 | Local prefix |
| 76 | `[1216, 1232)` | 1495 / 3 | Local prefix | 1124 / 2 | NIXL from P |
| 77–142 | `[1232, 2288)` | 1496–1561 / 1 | Local prefix | 1125–1190 / 1 | NIXL from P |
| 143 | `[2288, 2304)` | 1562 / 2 | Local prefix | 1191 / 1 | NIXL from P |
| 144–195 | `[2304, 3136)` | 1563–1614 / 1 | Local prefix | 1192–1243 / 1 | NIXL from P |
| 196 | `[3136, 3152)` | 1615 / 2 | Model write | 1244 / 1 | NIXL from P |
| 197–280 | `[3152, 4481)` | 1616–1699 / 1 | Model write | 1245–1328 / 1 | NIXL from P |

</details>

## Turn 5 on D2

> why is Guinea pig called pig?

**5,667 prompt tokens · 355 GPU blocks · 22 complete LMCache objects · 35 tail tokens.** [Same turn in chat](../disk-shared-chat/05-cfbcfa3e.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.369–4.943 | Lookup · 17 objects found | 280 | 0 | 151 | 192 | 272 | 895 |
| P | 0.000–6.082 | Accept prompt · 280 local | **280** carried | 0 | **151** carried | 192 | 272 | 895 |
| P | 6.519–133.305 | Allocate slots and write 75 blocks | **355** · 280 local + 75 written | 0 | 151 | 192 | 272 | **970** |
| P GPU → LMCache | 132.632–142.303 | Offload 5 objects | 355 | 0 | 151 | **272** · L272–351 | 272 | **1050** |
| Proxy → D2 | 140.844–149.241 | Dispatch prompt and metadata · D accepts | 355 | 0 | 151 | 272 | 272 | 1050 |
| D2 · LMCache | 153.308–265.349 | Lookup · 22 objects found, 5 read from disk into CPU | 355 | 0 | 151 | **352** | 272 | **1130** |
| D2 | 266.658–267.306 | Plan fetch · 151 local, 204 NIXL | 355 | **204** · L151–354 | 151<br>204 slots reserved | 352 | 272 | 1130 |
| P · LMCache | 416.774–417.348 | 5 disk writes · release CPU copies | 355 | 204 | 151 | **272** | **352** · +L272–351 | 1130 · peak 1210 |
| NIXL | 876.128 | D reports prompt KV ready | 355 | **0** · 204 ready | **355** · 151 local + 204 NIXL | 272 | 352 | **1334** |
| P | 901.807 | Release uncached partial prompt block | **354** · releases L354 | 0 | 355 | 272 | 352 | **1333** |
| D2 | 876.128–2,415.022 | Run logits and write answer KV | 354 | 0 | 355 | 272 | 352 | 1333 |
| D2 | 2,418.631 | Release request | 354 | 0 | 355 | 272 | 352 | 1333 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D2 block IDs / gen | D2 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 201 / 2 | Local prefix | 209 / 2 | Local prefix |
| 2–66 | `[32, 1072)` | 202–266 / 1 | Local prefix | 210–274 / 1 | Local prefix |
| 67 | `[1072, 1088)` | 267 / 2 | Local prefix | 275 / 1 | Local prefix |
| 68–75 | `[1088, 1216)` | 268–275 / 1 | Local prefix | 276–283 / 1 | Local prefix |
| 76 | `[1216, 1232)` | 1495 / 3 | Local prefix | 1327 / 2 | Local prefix |
| 77–142 | `[1232, 2288)` | 1496–1561 / 1 | Local prefix | 1328–1393 / 1 | Local prefix |
| 143 | `[2288, 2304)` | 1562 / 2 | Local prefix | 1394 / 1 | Local prefix |
| 144–150 | `[2304, 2416)` | 1563–1569 / 1 | Local prefix | 1395–1401 / 1 | Local prefix |
| 151 | `[2416, 2432)` | 1570 / 1 | Local prefix | 1402 / 2 | NIXL from P |
| 152–195 | `[2432, 3136)` | 1571–1614 / 1 | Local prefix | 1403–1446 / 1 | NIXL from P |
| 196 | `[3136, 3152)` | 1615 / 2 | Local prefix | 1447 / 1 | NIXL from P |
| 197–279 | `[3152, 4480)` | 1616–1698 / 1 | Local prefix | 1448–1530 / 1 | NIXL from P |
| 280 | `[4480, 4496)` | 1699 / 2 | Model write | 1531 / 1 | NIXL from P |
| 281–354 | `[4496, 5667)` | 1700–1773 / 1 | Model write | 1532–1605 / 1 | NIXL from P |

</details>

## Turn 6 on D3

> do you think I can have guinea pig as my pet?

**6,878 prompt tokens · 430 GPU blocks · 26 complete LMCache objects · 222 tail tokens.** [Same turn in chat](../disk-shared-chat/05-cfbcfa3e.md#turn-6-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | LMCache CPU | LMCache disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 3.036–6.932 | Lookup · 22 objects found | 354 | 0 | 204 | 272 | 352 | 1182 |
| P | 0.000–8.199 | Accept prompt · 354 local | **354** carried | 0 | **204** carried | 272 | 352 | 1182 |
| P | 8.678–142.704 | Allocate slots and write 76 blocks | **430** · 354 local + 76 written | 0 | 204 | 272 | 352 | **1258** |
| P GPU → LMCache | 142.023–149.517 | Offload 4 objects | 430 | 0 | 204 | **336** · L352–415 | 352 | **1322** |
| Proxy → D3 | 150.941–160.985 | Dispatch prompt and metadata · D accepts | 430 | 0 | 204 | 336 | 352 | 1322 |
| D3 · LMCache | 163.606–270.678 | Lookup · 26 objects found, 5 read from disk into CPU | 430 | 0 | 204 | **416** | 352 | **1402** |
| D3 | 272.193–272.544 | Plan fetch · 204 local, 226 NIXL | 430 | **226** · L204–429 | 204<br>226 slots reserved | 416 | 352 | 1402 |
| P · LMCache | 386.122–386.786 | 4 disk writes · release CPU copies | 430 | 226 | 204 | **352** | **416** · +L352–415 | 1402 · peak 1466 |
| NIXL | 933.660 | D reports prompt KV ready | 430 | **0** · 226 ready | **430** · 204 local + 226 NIXL | 352 | 416 | **1628** |
| P | 965.215 | Release uncached partial prompt block | **429** · releases L429 | 0 | 430 | 352 | 416 | **1627** |
| D3 | 933.660–2,492.369 | Run logits and write answer KV | 429 | 0 | 430 | 352 | 416 | 1627 |
| D3 | 2,495.244 | Release request | 429 | 0 | 430 | 352 | 416 | 1627 |

<details>
<summary>Prompt block map</summary>

| Logical blocks | Prompt tokens | P block IDs / gen | P source | D3 block IDs / gen | D3 source |
| --- | --- | --- | --- | --- | --- |
| 0 | `[0, 16)` | 1 / 1 | Local prefix | 1 / 1 | Local prefix |
| 1 | `[16, 32)` | 201 / 2 | Local prefix | 124 / 2 | Local prefix |
| 2–66 | `[32, 1072)` | 202–266 / 1 | Local prefix | 125–189 / 1 | Local prefix |
| 67 | `[1072, 1088)` | 267 / 2 | Local prefix | 190 / 1 | Local prefix |
| 68–74 | `[1088, 1200)` | 268–274 / 1 | Local prefix | 191–197 / 1 | Local prefix |
| 75 | `[1200, 1216)` | 275 / 1 | Local prefix | 198 / 2 | Local prefix |
| 76 | `[1216, 1232)` | 1495 / 3 | Local prefix | 1442 / 2 | Local prefix |
| 77–142 | `[1232, 2288)` | 1496–1561 / 1 | Local prefix | 1443–1508 / 1 | Local prefix |
| 143 | `[2288, 2304)` | 1562 / 2 | Local prefix | 1509 / 1 | Local prefix |
| 144–195 | `[2304, 3136)` | 1563–1614 / 1 | Local prefix | 1510–1561 / 1 | Local prefix |
| 196 | `[3136, 3152)` | 1615 / 2 | Local prefix | 1562 / 1 | Local prefix |
| 197–203 | `[3152, 3264)` | 1616–1622 / 1 | Local prefix | 1563–1569 / 1 | Local prefix |
| 204 | `[3264, 3280)` | 1623 / 1 | Local prefix | 1570 / 2 | NIXL from P |
| 205–279 | `[3280, 4480)` | 1624–1698 / 1 | Local prefix | 1571–1645 / 1 | NIXL from P |
| 280 | `[4480, 4496)` | 1699 / 2 | Local prefix | 1646 / 1 | NIXL from P |
| 281–353 | `[4496, 5664)` | 1700–1772 / 1 | Local prefix | 1647–1719 / 1 | NIXL from P |
| 354 | `[5664, 5680)` | 1773 / 2 | Model write | 1720 / 1 | NIXL from P |
| 355–429 | `[5680, 6878)` | 1774–1848 / 1 | Model write | 1721–1795 / 1 | NIXL from P |

</details>

## The turns together

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: the server | Disk writes | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 1,087 | 68 | 67 / 1 | 67 / 1 | 6 | 0 | 0 | 0 / 0 |
| 2 | D2 | 2,301 | 144 | 76 / 68 | 76 / 68 | 204 | 4 | 4 | 0 / 0 |
| 3 | D3 | 3,149 | 197 | 143 / 54 | 76 / 121 | 359 | 4 | 4 | 0 / 0 |
| 4 | D1 | 4,481 | 281 | 196 / 85 | 76 / 205 | 601 | 5 | 5 | 0 / 0 |
| 5 | D2 | 5,667 | 355 | 280 / 75 | 151 / 204 | 609 | 5 | 5 | 0 / 0 |
| 6 | D3 | 6,878 | 430 | 354 / 76 | 204 / 226 | 661 | 4 | 4 | 0 / 0 |
| all | | 23,563 | 1475 | 1116 / 359 | 650 / 825 | | 22 | 22 | 0 / 0 |

