# NIXL with per-engine LMCache · memory, carry-forward timeline

Conversation `cfbcfa3e382f12aed450b10f87a72aeb`, **memory, carry-forward**, 6 turns. P and each D have their own LMCache server; 2.0 GB of LMCache memory a server. GPU KV pool: 109,648 tokens (6,853 blocks) an engine (vLLM's default, no cap).

[Paired timeline](../memory-per-engine-chat/05-cfbcfa3e.md) · [Run's results](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z/matrix/llama32_3b_lmc_memory_nixl_per_engine_carry_all_rtxa5000_20261126_233246) · Commit `268a9a96a324`, run `lmc_memory_nixl_per_engine_carry_all`.

[Turn 1](#turn-1-on-d1) · [Turn 2](#turn-2-on-d2) · [Turn 3](#turn-3-on-d3) · [Turn 4](#turn-4-on-d1) · [Turn 5](#turn-5-on-d2) · [Turn 6](#turn-6-on-d3)

Each row is one step of the turn, its range in ms from the prompt's arrival at P, and the **current prompt's** residency at the range's end, in GPU blocks or block equivalents (16 GPU blocks a 256-token LMCache object): P's and D's GPU, NIXL's pending blocks (asked for until D reports them ready), and the objects in each LMCache server's memory (CPU) and on its disk. Tracked copies sum the GPU, CPU and disk columns; a disk row also gives the peak within its range, where a CPU copy and its disk copy overlap. Rows are sorted by their end; ranges overlap where GPU work and LMCache writes run at once. A bold value changed in that row. GPU block ids are each engine's own; a block's generation counts its allocations so far. No capacity evictions: the pool holds every turn (the Run 3 views capped it at 70–528 blocks).

## Turn 1 on D1

> Where do guinea pigs sleep in the wild?

**1,087 prompt tokens · 68 GPU blocks · 4 complete LMCache objects · 63 tail tokens.** [Same turn in chat](../memory-per-engine-chat/05-cfbcfa3e.md#turn-1-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 1.505–2.403 | Lookup · 4 objects found | 67 | 0 | 67 | 64 | 0 | 64 | 0 | 262 |
| P | 0.000–4.491 | Accept prompt · 67 local | **67** carried | 0 | **67** carried | 64 | 0 | 64 | 0 | 262 |
| P | 4.740–20.649 | Allocate slots and write 1 blocks | **68** · 67 local + 1 written | 0 | 67 | 64 | 0 | 64 | 0 | **263** |
| Proxy → D1 | 23.612–27.832 | Dispatch prompt and metadata · D accepts | 68 | 0 | 67 | 64 | 0 | 64 | 0 | 263 |
| D1 · LMCache | 29.778–30.615 | Lookup · 4 objects found | 68 | 0 | 67 | 64 | 0 | 64 | 0 | 263 |
| D1 | 32.910–33.251 | Plan fetch · 67 local, 1 NIXL | 68 | **1** · L67 | 67<br>1 slots reserved | 64 | 0 | 64 | 0 | 263 |
| NIXL | 38.901 | D reports prompt KV ready | 68 | **0** · 1 ready | **68** · 67 local + 1 NIXL | 64 | 0 | 64 | 0 | **264** |
| P | 46.266 | Release uncached partial prompt block | **67** · releases L67 | 0 | 68 | 64 | 0 | 64 | 0 | **263** |
| D1 | 38.901–1,514.324 | Run logits and write answer KV | 67 | 0 | 68 | 64 | 0 | 64 | 0 | 263 |
| D1 | 1,517.722 | Release request | 67 | 0 | 68 | 64 | 0 | 64 | 0 | 263 |

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

**2,301 prompt tokens · 144 GPU blocks · 8 complete LMCache objects · 253 tail tokens.** [Same turn in chat](../memory-per-engine-chat/05-cfbcfa3e.md#turn-2-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.020–3.036 | Lookup · 4 objects found | 76 | 0 | 76 | 64 | 0 | 64 | 0 | 280 |
| P | 0.000–5.452 | Accept prompt · 76 local | **76** carried | 0 | **76** carried | 64 | 0 | 64 | 0 | 280 |
| P | 5.761–101.882 | Allocate slots and write 68 blocks | **144** · 76 local + 68 written | 0 | 76 | 64 | 0 | 64 | 0 | **348** |
| P GPU → LMCache | 101.303–110.386 | Offload 4 objects | 144 | 0 | 76 | **128** · L64–127 | 0 | 64 | 0 | **412** |
| Proxy → D2 | 106.431–112.147 | Dispatch prompt and metadata · D accepts | 144 | 0 | 76 | 128 | 0 | 64 | 0 | 412 |
| D2 · LMCache | 114.870–116.669 | Lookup · 4 objects found | 144 | 0 | 76 | 128 | 0 | 64 | 0 | 412 |
| D2 | 118.091–118.541 | Plan fetch · 76 local, 68 NIXL | 144 | **68** · L76–143 | 76<br>68 slots reserved | 128 | 0 | 64 | 0 | 412 |
| NIXL | 320.176 | D reports prompt KV ready | 144 | **0** · 68 ready | **144** · 76 local + 68 NIXL | 128 | 0 | 64 | 0 | **480** |
| P | 333.010 | Release uncached partial prompt block | **143** · releases L143 | 0 | 144 | 128 | 0 | 64 | 0 | **479** |
| D2 GPU → LMCache | 344.992–361.759 | Offload 4 objects | 143 | 0 | 144 | 128 | 0 | **128** · L64–127 | 0 | **543** |
| D2 | 320.176–1,774.138 | Run logits and write answer KV | 143 | 0 | 144 | 128 | 0 | 128 | 0 | 543 |
| D2 | 1,777.757 | Release request | 143 | 0 | 144 | 128 | 0 | 128 | 0 | 543 |

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

**3,149 prompt tokens · 197 GPU blocks · 12 complete LMCache objects · 77 tail tokens.** [Same turn in chat](../memory-per-engine-chat/05-cfbcfa3e.md#turn-3-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.352–4.208 | Lookup · 8 objects found | 143 | 0 | 76 | 128 | 0 | 48 | 0 | 395 |
| P | 0.000–5.662 | Accept prompt · 143 local | **143** carried | 0 | **76** carried | 128 | 0 | 48 | 0 | 395 |
| P | 6.000–91.609 | Allocate slots and write 54 blocks | **197** · 143 local + 54 written | 0 | 76 | 128 | 0 | 48 | 0 | **449** |
| P GPU → LMCache | 91.152–101.134 | Offload 4 objects | 197 | 0 | 76 | **192** · L128–191 | 0 | 48 | 0 | **513** |
| Proxy → D3 | 97.035–103.410 | Dispatch prompt and metadata · D accepts | 197 | 0 | 76 | 192 | 0 | 48 | 0 | 513 |
| D3 · LMCache | 105.655–108.104 | Lookup · 4 objects found | 197 | 0 | 76 | 192 | 0 | 48 | 0 | 513 |
| D3 | 108.878–109.481 | Plan fetch · 76 local, 121 NIXL | 197 | **121** · L76–196 | 76<br>121 slots reserved | 192 | 0 | 48 | 0 | 513 |
| NIXL | 466.226 | D reports prompt KV ready | 197 | **0** · 121 ready | **197** · 76 local + 121 NIXL | 192 | 0 | 48 | 0 | **634** |
| P | 483.192 | Release uncached partial prompt block | **196** · releases L196 | 0 | 197 | 192 | 0 | 48 | 0 | **633** |
| D3 GPU → LMCache | 493.828–519.756 | Offload 8 objects | 196 | 0 | 197 | 192 | 0 | **176** · L64–191 | 0 | **761** |
| D3 | 466.226–1,936.513 | Run logits and write answer KV | 196 | 0 | 197 | 192 | 0 | 176 | 0 | 761 |
| D3 | 1,940.020 | Release request | 196 | 0 | 197 | 192 | 0 | 176 | 0 | 761 |

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

**4,481 prompt tokens · 281 GPU blocks · 17 complete LMCache objects · 129 tail tokens.** [Same turn in chat](../memory-per-engine-chat/05-cfbcfa3e.md#turn-4-on-d1)

**Capacity evictions on this turn** · P 0 · D1 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D1 GPU | P CPU | P disk | D1 CPU | D1 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.195–3.956 | Lookup · 12 objects found | 196 | 0 | 76 | 192 | 0 | 64 | 0 | 528 |
| P | 0.000–5.510 | Accept prompt · 196 local | **196** carried | 0 | **76** carried | 192 | 0 | 64 | 0 | 528 |
| P | 5.895–143.882 | Allocate slots and write 85 blocks | **281** · 196 local + 85 written | 0 | 76 | 192 | 0 | 64 | 0 | **613** |
| P GPU → LMCache | 143.340–150.965 | Offload 5 objects | 281 | 0 | 76 | **272** · L192–271 | 0 | 64 | 0 | **693** |
| Proxy → D1 | 150.463–157.020 | Dispatch prompt and metadata · D accepts | 281 | 0 | 76 | 272 | 0 | 64 | 0 | 693 |
| D1 · LMCache | 159.493–161.654 | Lookup · 4 objects found | 281 | 0 | 76 | 272 | 0 | 64 | 0 | 693 |
| D1 | 162.320–162.874 | Plan fetch · 76 local, 205 NIXL | 281 | **205** · L76–280 | 76<br>205 slots reserved | 272 | 0 | 64 | 0 | 693 |
| NIXL | 762.038 | D reports prompt KV ready | 281 | **0** · 205 ready | **281** · 76 local + 205 NIXL | 272 | 0 | 64 | 0 | **898** |
| P | 783.850 | Release uncached partial prompt block | **280** · releases L280 | 0 | 281 | 272 | 0 | 64 | 0 | **897** |
| D1 GPU → LMCache | 795.622–833.588 | Offload 13 objects | 280 | 0 | 281 | 272 | 0 | **272** · L64–271 | 0 | **1105** |
| D1 | 762.038–2,271.255 | Run logits and write answer KV | 280 | 0 | 281 | 272 | 0 | 272 | 0 | 1105 |
| D1 | 2,274.873 | Release request | 280 | 0 | 281 | 272 | 0 | 272 | 0 | 1105 |

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

**5,667 prompt tokens · 355 GPU blocks · 22 complete LMCache objects · 35 tail tokens.** [Same turn in chat](../memory-per-engine-chat/05-cfbcfa3e.md#turn-5-on-d2)

**Capacity evictions on this turn** · P 0 · D2 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D2 GPU | P CPU | P disk | D2 CPU | D2 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 2.412–4.325 | Lookup · 17 objects found | 280 | 0 | 151 | 272 | 0 | 128 | 0 | 831 |
| P | 0.000–5.869 | Accept prompt · 280 local | **280** carried | 0 | **151** carried | 272 | 0 | 128 | 0 | 831 |
| P | 6.330–132.516 | Allocate slots and write 75 blocks | **355** · 280 local + 75 written | 0 | 151 | 272 | 0 | 128 | 0 | **906** |
| P GPU → LMCache | 131.850–138.973 | Offload 5 objects | 355 | 0 | 151 | **352** · L272–351 | 0 | 128 | 0 | **986** |
| Proxy → D2 | 141.261–149.688 | Dispatch prompt and metadata · D accepts | 355 | 0 | 151 | 352 | 0 | 128 | 0 | 986 |
| D2 · LMCache | 154.751–156.039 | Lookup · 9 objects found | 355 | 0 | 151 | 352 | 0 | 128 | 0 | 986 |
| D2 | 157.550–158.246 | Plan fetch · 151 local, 204 NIXL | 355 | **204** · L151–354 | 151<br>204 slots reserved | 352 | 0 | 128 | 0 | 986 |
| NIXL | 761.731 | D reports prompt KV ready | 355 | **0** · 204 ready | **355** · 151 local + 204 NIXL | 352 | 0 | 128 | 0 | **1190** |
| P | 787.564 | Release uncached partial prompt block | **354** · releases L354 | 0 | 355 | 352 | 0 | 128 | 0 | **1189** |
| D2 GPU → LMCache | 797.211–833.401 | Offload 13 objects | 354 | 0 | 355 | 352 | 0 | **336** · L144–351 | 0 | **1397** |
| D2 | 761.731–2,305.299 | Run logits and write answer KV | 354 | 0 | 355 | 352 | 0 | 336 | 0 | 1397 |
| D2 | 2,308.755 | Release request | 354 | 0 | 355 | 352 | 0 | 336 | 0 | 1397 |

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

**6,878 prompt tokens · 430 GPU blocks · 26 complete LMCache objects · 222 tail tokens.** [Same turn in chat](../memory-per-engine-chat/05-cfbcfa3e.md#turn-6-on-d3)

**Capacity evictions on this turn** · P 0 · D3 0.

| Actor / path | Elapsed (ms) | Event | P GPU | NIXL pending | D3 GPU | P CPU | P disk | D3 CPU | D3 disk | Tracked copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P · LMCache | 3.837–6.437 | Lookup · 22 objects found | 354 | 0 | 204 | 352 | 0 | 176 | 0 | 1086 |
| P | 0.000–7.643 | Accept prompt · 354 local | **354** carried | 0 | **204** carried | 352 | 0 | 176 | 0 | 1086 |
| P | 8.179–143.359 | Allocate slots and write 76 blocks | **430** · 354 local + 76 written | 0 | 204 | 352 | 0 | 176 | 0 | **1162** |
| P GPU → LMCache | 142.579–149.833 | Offload 4 objects | 430 | 0 | 204 | **416** · L352–415 | 0 | 176 | 0 | **1226** |
| Proxy → D3 | 153.151–161.808 | Dispatch prompt and metadata · D accepts | 430 | 0 | 204 | 416 | 0 | 176 | 0 | 1226 |
| D3 · LMCache | 164.638–167.034 | Lookup · 12 objects found | 430 | 0 | 204 | 416 | 0 | 176 | 0 | 1226 |
| D3 | 167.988–168.665 | Plan fetch · 204 local, 226 NIXL | 430 | **226** · L204–429 | 204<br>226 slots reserved | 416 | 0 | 176 | 0 | 1226 |
| NIXL | 826.670 | D reports prompt KV ready | 430 | **0** · 226 ready | **430** · 204 local + 226 NIXL | 416 | 0 | 176 | 0 | **1452** |
| P | 857.175 | Release uncached partial prompt block | **429** · releases L429 | 0 | 430 | 416 | 0 | 176 | 0 | **1451** |
| D3 GPU → LMCache | 862.144–900.915 | Offload 14 objects | 429 | 0 | 430 | 416 | 0 | **400** · L192–415 | 0 | **1675** |
| D3 | 826.670–2,391.243 | Run logits and write answer KV | 429 | 0 | 430 | 416 | 0 | 400 | 0 | 1675 |
| D3 | 2,394.773 | Release request | 429 | 0 | 430 | 416 | 0 | 400 | 0 | 1675 |

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

| Turn | D | Prompt tokens | GPU blocks | P local / written | D local / NIXL | NIXL plan to ready (ms) | Objects stored: P's server / D's | Disk writes: P's / D's | Capacity evictions P / D |
| ---: | --- | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 1 | D1 | 1,087 | 68 | 67 / 1 | 67 / 1 | 6 | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | D2 | 2,301 | 144 | 76 / 68 | 76 / 68 | 202 | 4 / 4 | 0 / 0 | 0 / 0 |
| 3 | D3 | 3,149 | 197 | 143 / 54 | 76 / 121 | 357 | 4 / 8 | 0 / 0 | 0 / 0 |
| 4 | D1 | 4,481 | 281 | 196 / 85 | 76 / 205 | 599 | 5 / 13 | 0 / 0 | 0 / 0 |
| 5 | D2 | 5,667 | 355 | 280 / 75 | 151 / 204 | 603 | 5 / 13 | 0 / 0 | 0 / 0 |
| 6 | D3 | 6,878 | 430 | 354 / 76 | 204 / 226 | 658 | 4 / 14 | 0 / 0 | 0 / 0 |
| all | | 23,563 | 1475 | 1116 / 359 | 650 / 825 | | 22 / 52 | 0 / 0 | 0 / 0 |

