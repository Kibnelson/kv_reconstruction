# Job 30786: one conversation, eight configurations, turn by turn

MTRAG conversation `1534a095279f2cb888fb0bea17bd70da` (ClapNQ, 5 turns), one at a time, in the eight
configurations of `docs/lmcache_lifecycle.md` ("The eight configurations"): NIXL from P to D (vLLM's pull, UCX's
defaults), D engines round-robin (D1, D2, D3, D1, D2), every passage MTRAG has for each turn, D's own earlier answers
in later prompts. Commit `eea6fcf`; the results:
`evidence/lmcache-lifecycle/eea6fcf20beb/lmctrace/job30786`.

Each page follows the turns the way the Run 3 timeline views do: a row a step, its range in ms from the prompt's
arrival at P, and where the current prompt's KV is at the range's end (P's and D's GPU blocks, NIXL's pending blocks,
the LMCache objects in each server's memory and on its disk, the copies in all), then each turn's block map (each
logical block's GPU block and generation on P and on D, and where its KV came from) and a table of the turns
together. Made by `scripts/lmc_turnview.py` from the run's trace.

| | chat RAG | carry-forward |
| --- | --- | --- |
| disk, a server per engine | [disk-per-engine-chat.md](disk-per-engine-chat.md) | [disk-per-engine-carry.md](disk-per-engine-carry.md) |
| disk, one shared server | [disk-shared-chat.md](disk-shared-chat.md) | [disk-shared-carry.md](disk-shared-carry.md) |
| memory, a server per engine | [memory-per-engine-chat.md](memory-per-engine-chat.md) | [memory-per-engine-carry.md](memory-per-engine-carry.md) |
| memory, one shared server | [memory-shared-chat.md](memory-shared-chat.md) | [memory-shared-carry.md](memory-shared-carry.md) |

What the five turns add up to (the same in every configuration, as the GPU side does not depend on LMCache's):

| | chat RAG | carry-forward |
| --- | ---: | ---: |
| prompt tokens | 4,238 | 9,229 |
| GPU blocks of the prompts | 268 | 580 |
| P: from its own GPU / written | 58 / 210 | 375 / 205 |
| D: from its own GPU / read from P over NIXL | 11 / 257 | 118 / 462 |
| LMCache objects stored, P's server (each D's, per engine) | 12 (14) | 12 (28) |
| capacity evictions | 0 | 0 |

- No capacity evictions: each engine's KV pool is vLLM's default here, 109,648 tokens (6,853 blocks), where the Run 3
  views capped it at 70-528 blocks, so every turn's prompt and the earlier ones fit.
- P keeps the earlier turns' blocks and writes only the new ones: 205 of 580 in carry-forward, 210 of 268 in chat
  (a chat prompt's passages are its own turn's, so only the instructions and the history before them are reused).
- D's own GPU holds what that D saw two turns before (round-robin): 40 blocks on turn 4 (D1, turn 1's prompt and
  its own answer) and 78 on turn 5 (D2) in carry-forward; 1 and 10 in chat. NIXL brings the rest.
- With a server per engine each D stores the prompt's objects again on its own server (28 in carry-forward against
  P's 12): the copies the Run 3 views point at.
- With one shared server on disk, D's lookup reads what it finds from disk into memory (the `retain` policy keeps
  it) and D's NIXL plan waits for that lookup: 52 ms from D's accept to its plan (median a turn) in carry-forward,
  against 5-8 ms in the other six runs; nothing loads from it, NIXL brings the KV.
