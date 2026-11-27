# Job 30806: ten conversations, eight configurations, turn by turn

MTRAG's ten conversations of `docs/lmcache_lifecycle.md` ("The eight configurations"), one at a time, in the
eight configurations: NIXL from P to D (vLLM's pull, UCX's defaults), D engines round-robin across the whole run,
every passage MTRAG has for each turn, D's own earlier answers in later prompts. Commit `268a9a9`; the results:
[`evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806`](https://github.com/Kibnelson/kv_reconstruction/tree/evidence/lmcache-lifecycle/268a9a96a324/lmctrace/job30806/benchmark_results/pd_card/pd_lmctrace_20261127T034204Z).

A page a conversation and configuration (80), laid out as the Run 3 timeline views: a row a step, its range in ms
from the prompt's arrival at P, the current prompt's residency at the range's end (P's and D's GPU blocks, NIXL's
pending blocks, each LMCache server's memory and disk in block equivalents, the copies in all), each turn's block
map, and the turns together. Each page links the same conversation in the other layout. Made by
`scripts/lmc_turnview.py --all` from each run's trace; [job 30786](../job30786/README.md) is the first
conversation alone.

| # | Conversation | Turns | disk, per engine, chat | disk, per engine, carry | disk, shared, chat | disk, shared, carry | memory, per engine, chat | memory, per engine, carry | memory, shared, chat | memory, shared, carry |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `1534a095279f2cb888fb0bea17bd70da` | 5 | [page](disk-per-engine-chat/01-1534a095.md) | [page](disk-per-engine-carry/01-1534a095.md) | [page](disk-shared-chat/01-1534a095.md) | [page](disk-shared-carry/01-1534a095.md) | [page](memory-per-engine-chat/01-1534a095.md) | [page](memory-per-engine-carry/01-1534a095.md) | [page](memory-shared-chat/01-1534a095.md) | [page](memory-shared-carry/01-1534a095.md) |
| 2 | `0208bf26ec357a803445290fa88a2e9e` | 7 | [page](disk-per-engine-chat/02-0208bf26.md) | [page](disk-per-engine-carry/02-0208bf26.md) | [page](disk-shared-chat/02-0208bf26.md) | [page](disk-shared-carry/02-0208bf26.md) | [page](memory-per-engine-chat/02-0208bf26.md) | [page](memory-per-engine-carry/02-0208bf26.md) | [page](memory-shared-chat/02-0208bf26.md) | [page](memory-shared-carry/02-0208bf26.md) |
| 3 | `a2698b2973ea7db1ee5adb5e70ec02e4` | 9 | [page](disk-per-engine-chat/03-a2698b29.md) | [page](disk-per-engine-carry/03-a2698b29.md) | [page](disk-shared-chat/03-a2698b29.md) | [page](disk-shared-carry/03-a2698b29.md) | [page](memory-per-engine-chat/03-a2698b29.md) | [page](memory-per-engine-carry/03-a2698b29.md) | [page](memory-shared-chat/03-a2698b29.md) | [page](memory-shared-carry/03-a2698b29.md) |
| 4 | `c1e53df98a28f19448f4af244304b89a` | 6 | [page](disk-per-engine-chat/04-c1e53df9.md) | [page](disk-per-engine-carry/04-c1e53df9.md) | [page](disk-shared-chat/04-c1e53df9.md) | [page](disk-shared-carry/04-c1e53df9.md) | [page](memory-per-engine-chat/04-c1e53df9.md) | [page](memory-per-engine-carry/04-c1e53df9.md) | [page](memory-shared-chat/04-c1e53df9.md) | [page](memory-shared-carry/04-c1e53df9.md) |
| 5 | `cfbcfa3e382f12aed450b10f87a72aeb` | 6 | [page](disk-per-engine-chat/05-cfbcfa3e.md) | [page](disk-per-engine-carry/05-cfbcfa3e.md) | [page](disk-shared-chat/05-cfbcfa3e.md) | [page](disk-shared-carry/05-cfbcfa3e.md) | [page](memory-per-engine-chat/05-cfbcfa3e.md) | [page](memory-per-engine-carry/05-cfbcfa3e.md) | [page](memory-shared-chat/05-cfbcfa3e.md) | [page](memory-shared-carry/05-cfbcfa3e.md) |
| 6 | `3f5fa378239f7475baac89fa40288aaa` | 6 | [page](disk-per-engine-chat/06-3f5fa378.md) | [page](disk-per-engine-carry/06-3f5fa378.md) | [page](disk-shared-chat/06-3f5fa378.md) | [page](disk-shared-carry/06-3f5fa378.md) | [page](memory-per-engine-chat/06-3f5fa378.md) | [page](memory-per-engine-carry/06-3f5fa378.md) | [page](memory-shared-chat/06-3f5fa378.md) | [page](memory-shared-carry/06-3f5fa378.md) |
| 7 | `5c13daa9432149f6243aa1115629cd1a` | 7 | [page](disk-per-engine-chat/07-5c13daa9.md) | [page](disk-per-engine-carry/07-5c13daa9.md) | [page](disk-shared-chat/07-5c13daa9.md) | [page](disk-shared-carry/07-5c13daa9.md) | [page](memory-per-engine-chat/07-5c13daa9.md) | [page](memory-per-engine-carry/07-5c13daa9.md) | [page](memory-shared-chat/07-5c13daa9.md) | [page](memory-shared-carry/07-5c13daa9.md) |
| 8 | `694e275f1a01ad0e8ac448ad809f7930` | 8 | [page](disk-per-engine-chat/08-694e275f.md) | [page](disk-per-engine-carry/08-694e275f.md) | [page](disk-shared-chat/08-694e275f.md) | [page](disk-shared-carry/08-694e275f.md) | [page](memory-per-engine-chat/08-694e275f.md) | [page](memory-per-engine-carry/08-694e275f.md) | [page](memory-shared-chat/08-694e275f.md) | [page](memory-shared-carry/08-694e275f.md) |
| 9 | `aea2634b0e5f1c2444550d31c41adc94` | 8 | [page](disk-per-engine-chat/09-aea2634b.md) | [page](disk-per-engine-carry/09-aea2634b.md) | [page](disk-shared-chat/09-aea2634b.md) | [page](disk-shared-carry/09-aea2634b.md) | [page](memory-per-engine-chat/09-aea2634b.md) | [page](memory-per-engine-carry/09-aea2634b.md) | [page](memory-shared-chat/09-aea2634b.md) | [page](memory-shared-carry/09-aea2634b.md) |
| 10 | `79f0d0539d9ec0acbf90cb3388b30c17` | 7 | [page](disk-per-engine-chat/10-79f0d053.md) | [page](disk-per-engine-carry/10-79f0d053.md) | [page](disk-shared-chat/10-79f0d053.md) | [page](disk-shared-carry/10-79f0d053.md) | [page](memory-per-engine-chat/10-79f0d053.md) | [page](memory-per-engine-carry/10-79f0d053.md) | [page](memory-shared-chat/10-79f0d053.md) | [page](memory-shared-carry/10-79f0d053.md) |

## What the turns add up to

The GPU side is the same in all four configurations of a layout (LMCache is not on D's load path). In blocks:

| # | Conversation | Prompt tokens: chat / carry | P: from its GPU / written, chat | carry | D: from its GPU / NIXL, chat | carry |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `1534a095` | 4,238 / 9,229 | 58 / 210 | 375 / 205 | 11 / 257 | 118 / 462 |
| 2 | `0208bf26` | 10,599 / 33,926 | 121 / 545 | 1578 / 546 | 54 / 612 | 770 / 1354 |
| 3 | `a2698b29` | 11,351 / 34,991 | 222 / 491 | 1712 / 481 | 113 / 600 | 1004 / 1189 |
| 4 | `c1e53df9` | 6,148 / 15,681 | 92 / 295 | 695 / 288 | 31 / 356 | 285 / 698 |
| 5 | `cfbcfa3e` | 8,568 / 23,563 | 175 / 364 | 1116 / 359 | 61 / 478 | 650 / 825 |
| 6 | `3f5fa378` | 7,281 / 20,444 | 96 / 361 | 922 / 358 | 33 / 424 | 396 / 884 |
| 7 | `5c13daa9` | 8,317 / 23,356 | 141 / 381 | 1086 / 377 | 61 / 461 | 530 / 933 |
| 8 | `694e275f` | 11,506 / 36,626 | 211 / 510 | 1792 / 500 | 105 / 616 | 1016 / 1276 |
| 9 | `aea2634b` | 10,874 / 36,022 | 196 / 488 | 1774 / 483 | 98 / 586 | 1049 / 1208 |
| 10 | `79f0d053` | 8,882 / 25,006 | 141 / 416 | 1153 / 413 | 64 / 493 | 516 / 1050 |
| all | | 87,764 / 258,844 | 1453 / 4061 | 12203 / 4010 | 631 / 4883 | 6334 / 9879 |

LMCache, by configuration (objects of 256 tokens):

| Configuration | Stored on P's server | Stored on D's servers | Disk writes P / D | D's accept to its NIXL plan, median (max) ms | D's lookups reading from disk | Memory evictions (LRU) |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| disk, per engine, chat | 243 | 286 | 243 / 286 | 6.7 (8.4) | 0 | 0 |
| disk, per engine, carry-forward | 242 | 609 | 242 / 609 | 8.1 (12.7) | 0 | 0 |
| disk, shared, chat | 243 | – | 243 / – | 6.3 (35.0) | 26 | 0 |
| disk, shared, carry-forward | 242 | – | 242 / – | 75.9 (118.1) | 206 | 0 |
| memory, per engine, chat | 243 | 286 | 0 / 0 | 5.1 (7.0) | 0 | 318 |
| memory, per engine, carry-forward | 242 | 609 | 0 / 0 | 5.8 (8.8) | 0 | 638 |
| memory, shared, chat | 243 | – | 0 / – | 5.0 (8.1) | 0 | 47 |
| memory, shared, carry-forward | 242 | – | 0 / – | 5.5 (9.2) | 0 | 47 |

- No GPU capacity evictions in any run: vLLM's default pool (about 6,850 blocks an engine) holds every
  turn, where the Run 3 views capped it at 70-528 blocks.
- Carry-forward reuses most of each prompt on P (each prompt holds the one before it); chat reuses the
  instructions and the history before this turn's passages. D reuses what it saw the last time a turn of the
  conversation came to it (round-robin), and NIXL brings the rest.
- A server per engine stores every prompt again on its D's server, the duplicate the Run 3 views point at.
- One shared server on disk: D's lookup reads the objects it finds from disk into memory (kept there, the
  `retain` policy) before D's NIXL read is planned, and nothing loads them. In carry-forward that holds D for
  76 ms a turn at the median (118 ms at most), against 5-8 ms in the other configurations.
- Memory mode evicts by LRU, objects of earlier conversations: 638 with 2 GB a server in carry-forward,
  318 in chat, 47 with the shared 8 GB; none of the current prompt's objects within a turn.
- The bytes: every block D read is P's, bit for bit (9,879 blocks a carry-forward run, 4,883 a chat run).
