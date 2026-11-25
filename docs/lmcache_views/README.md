# One conversation at a time: from the prompt to its chunks and blocks

Pages made by `scripts/lmc_lifecycle.py` from a lifecycle run's trace (`docs/lmcache_lifecycle.md`): one per MTRAG
conversation, and an index. Each page follows its conversation turn by turn:

- the prompt's pieces in order (system, documents, question, the `assistant:` marker, the answer that follows, the
  newline before the next turn), their token ranges, and which are new this turn;
- the prompt cut into LMCache's 256-token chunks: for each, the pieces it holds (how many tokens of each), its 16
  vLLM blocks (16 tokens each) and their GPU block ids on P and on D, its LMCache name (the hash of its tokens and
  everything before them), when it was stored and whether it is on disk and out of memory;
- what each chunk was on P that turn (on its GPU from an earlier turn, or computed and stored now) and on D (on its
  GPU from an earlier turn or as its own answer, pulled from memory or from disk, or computed: the partial chunk at
  the end), with a two-line strip of the whole prompt, one character a chunk;
- one chunk block by block: its 16 blocks' GPU ids on P, their vLLM hashes, the GPU blocks D loaded them into, and
  whether D's hash equals P's.

| Run | Pages |
| --- | --- |
| job 30728, memory mode | [job30728/memory/index.md](job30728/memory/index.md) |
| job 30728, disk mode | [job30728/disk/index.md](job30728/disk/index.md) |

Every report makes these pages for every conversation (`conversations/` in each run's results folder), whatever
`KVPD_LMC_CONVS` is. To make them again from a published results folder, without a GPU:

```bash
python3 -I scripts/lmc_lifecycle.py conversations \
    --results benchmark_results/pd_card/<tag>/matrix/llama32_3b_lmc_<mode>_rtxa5000_<time> --output /tmp/views_<mode>
```
