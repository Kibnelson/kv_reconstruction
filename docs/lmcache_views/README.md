# One conversation at a time: from the prompt to its chunks and blocks

Pages made by `scripts/lmc_lifecycle.py` from a lifecycle run's trace (`docs/lmcache_lifecycle.md`): one per MTRAG
conversation, and an index. Each page follows its conversation turn by turn:

- the prompt as text: turn 0's whole, then on each later turn what is new after the previous prompt (D's answer to
  the previous turn, the newline, this turn's documents, its question, `Assistant:`), each piece with its tokens
  and chunks; a document sent on an earlier turn is pointed back to rather than shown again (it is at another place
  in the prompt, so in other chunks, and a prefix cache cannot reuse it). Documents fold away (click to open);
  lines over 100 characters are wrapped for reading;
- a timeline: every turn on one clock (ms from the turn's send to P), as a picture and a table: P's queue and prefill, the chunks it stored in LMCache's memory and (disk mode) on disk, the store wait, then D's find (the disk read), copy into its GPU, compute, first token and decode; each turn's section repeats it in one sentence;
- how the prompt grows: per turn, its new tokens by what they are (previous answer, documents and how many of them
  were sent before, question, framing) and its whole chunks;
- the prompt's pieces in order (system, documents, question, the `assistant:` marker, the answer that follows, the
  newline before the next turn), their token ranges and chunks, and which are new this turn;
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

Where the text comes from: a run from this commit on records every piece's text as the driver sent it (the answers
too). Job 30728's driver kept only token ids, so its pages rebuild the text from the same pinned MTRAG file the
run read (SHA-256 checked), with the driver's own code, and use it only where the rebuilt pieces are the run's, kind
for kind and document id for document id; D's answers there are token counts only.

Every report makes these pages for every conversation (`conversations/` in each run's results folder), whatever
`KVPD_LMC_CONVS` is. To make them again from a published results folder, without a GPU (`--conversations`, the
MTRAG file, only for a run from before the driver kept texts):

```bash
python3 -I scripts/lmc_lifecycle.py conversations \
    --results benchmark_results/pd_card/<tag>/matrix/llama32_3b_lmc_<mode>_rtxa5000_<time> --output /tmp/views_<mode> \
    [--conversations mtrag/conversations.json]
```
