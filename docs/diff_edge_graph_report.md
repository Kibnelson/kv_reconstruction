# Report: a diff-edge graph over the radix tree, tested on interleaved MTRAG (status 25 November 2026)

This note records the diff-edge work of 23-25 November 2026 so it can be picked up again: the idea, how it was
tested, what it showed, and where the code, evidence and commands are. It follows
[handover_pd_baselines.md](handover_pd_baselines.md). The code is on the branch **`pd-blend`** (`031fe0c`); the full
design and the history of each stage are in `docs/pd_delta.md` on that branch (section "Plain diff edges").

## Start here

| What | Where | State |
| --- | --- | --- |
| Code: capture settings, the `diffedge` stage, the sample explorer, tests | branch `pd-blend`, `031fe0c` | Contains `pd-region`, `pd-delta` and `pd-baselines`; tests pass |
| The capture (4 interleaved MTRAG conversations) | `evidence/pd-blend/277bea7047fe/capture/job30663` | Its own diff-edge report has a labelling bug: read job 30673 instead |
| The analysis, corrected | `evidence/pd-blend/649dc8556541/diffedge/job30673` | **The results below** |
| The capture's pages (not in the evidence: 3.8 GB) | gpu-grad-01, `/tmp/nbore/kvpd-captures/pd_capture_20261125T141553Z` | Node-local; `MATRIX=diffedge` re-runs the stage on it while it is there |
| This note | `main` | |

The machine is gpu-grad-01 (4x RTX A5000 24 GB, Slurm account `oana2026`, QoS `research12h`); the model is
Llama-3.2-3B (28 layers, 8 KV heads x 128). Pages are 16 tokens x 28 layers. Sizes are **bits per value**: raw BF16
is 16, lower is better, and every coded page is decoded back exactly.

## 1. The idea

- **Today's two stores.** The radix tree reuses a page as it is when the same tokens follow exactly the same prefix:
  free, but only on an identical prefix. Everything else is stored raw, or with our codec **T** (it predicts each
  layer of a token's KV from the layer below of the same token and codes the exact word around the prediction:
  about 8.3 bits per value, 52% of raw).
- **The diff-edge graph.** Keep the radix tree, and add edges: a page that holds tokens seen before is stored as a
  difference against that other page and rebuilt when it is needed. It borrows CacheBlend's observation (the same
  text after a different prefix has KV close to its earlier copy, closer the longer the text before it agrees), but
  stays lossless: the difference is coded and the page comes back bit for bit.
- **Why agentic and multi-turn work.** Their histories grow by appending, so the same text comes back (documents
  retrieved again, tool outputs quoted again), and every answer the decode engine (D) generates is computed again
  by the prefill engine (P) when it becomes part of the next prompt.
- **Two kinds of page with an edge:**

| Kind | What it is | Its edge |
| --- | --- | --- |
| Moved | the same 16 tokens earlier in the conversation, after another prefix (a document retrieved again) | a prediction from its earlier copies |
| Twin | the same tokens after the same prefix, computed twice with other bytes: mostly an answer D decoded, prefilled again by P next turn | D's copy as it is |

## 2. What was run

1. **Capture** (job 30663, 25 November, about 25 minutes with setup). Real vLLM 0.32 PD, 1 prefill and 3 decode
   engines, raw transfer. MTRAG's first 4 conversations, 30 requests, in the interleaved layout: each turn's prompt
   holds every earlier turn's documents, question and answer, then its own documents, question and the answer cue
   (`Instructions ¶ A ¶ B ¶ Q1 + A1 ¶ B ¶ C ¶ Q2 + cue`). Answers are at most 128 tokens. P stored every page it
   computed; each D engine stored its own copy of every page it held whose bytes differed from P's. The check stage
   was bit-exact (4/4); outputs equal to a local prefill 3/4.
2. **The `diffedge` stage** (job 30673, 19 minutes, one GPU) on that capture:
   - each distinct page sorted into twin, moved or new (a block P stored before is a reference, free);
   - each moved page coded against every edge below, each twin against D's copy, with our value-slice coder;
     every page decoded and checked (all exact);
   - weights and spreads fitted on conversations 0 and 2, reported on 1 and 3 (held out);
   - the graph totalled against raw and against T on every page; chain depths; rebuild times.

The edges of a moved page (keys first turned by RoPE to the page's new place):

| Edge | Prediction |
| --- | --- |
| first | its first copy (a star: every copy against the first) |
| latest | its latest copy (a chain: each copy against the one before) |
| latest, affine | the latest with a scale and offset per layer, K/V and channel |
| second order | latest + λ (latest − previous): a diff of a diff |
| several copies | a·latest + b·previous + c·first |
| composite | a·latest + b·(latest − previous) + c·first + d, per channel |
| T alone | our codec's own prediction, ignoring the copies (as T codes today) |
| blend | T + w (latest − T): w = 0 is T, w = 1 the copy; w per layer, K/V and how long the text before agrees |

## 3. Results (job 30673)

**The pages.** 7,985 blocks: 6,150 references (77%, the radix tree). 1,835 distinct pages: 1,084 new (59%), 545
moved (30%: 194 with one earlier copy, 89 with two, 262 with three or more), 206 twins (11%; 182 of them answers;
none with D's bytes).

**Held out, bits per value:**

| Pages | Plain diff (best: composite) | Latest copy | First copy | T alone | Blend | Against D's copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Moved (130) | 8.86 | 9.02 | 9.27 | 8.30 | **8.12** | – |
| Twins (102) | – | – | – | not measured (about 8.3 expected) | – | **4.38** |

**By layer, moved pages, held out** (the copy and T are good at different depths):

| Layers | Latest copy (K / V) | T alone (K / V) | Blend (K / V) |
| --- | --- | --- | --- |
| 1–4 | 7.09 / 7.77 | 8.02 / 9.14 | 7.04 / 7.74 |
| 5–13 | 9.25 / 10.02 | 8.35 / 9.07 | 8.34 / 9.03 |
| 14–27 | 9.26 / 9.99 | 7.92 / 9.04 | 7.94 / 8.93 |

**What it shows:**

1. **Twins are the clear win.** P's and D's computations of the same answer differ by 0.1–1.6% of the values' size
   (20–30% of words equal in deep layers), so P's page against D's copy costs 4.38 bits: about half of T.
2. **Moved pages gain little.** The copy predicts layers 1–4 about a bit better than T; from layer 5 the new prefix
   changes the values and T is about a bit better. The blend learned this: w 0.85–0.99 in layers 1–4, 0.05–0.75
   deeper (rising with how long the text agrees and with more copies). Net: 2% under T. Plain diffs alone are worse
   than T.
3. **Agreement helps, but not enough.** The residual against the latest copy shrinks as the text before agrees
   longer (latest: 9.21 bits at 0–15 tokens of agreement to 8.86 at 256–1,023; residual's size against the page:
   keys 0.33 to 0.23, values 0.69 to 0.52). No page reached 1,024 tokens of agreement; only 4% of value words equal
   the copy's.
4. **No diff of a diff.** λ came out near 0 or negative (K −0.22 to −0.02, V −0.21 to −0.05): copies do not drift
   one way from turn to turn. Mixing several copies gives the latest 0.7–0.98 of the weight: they add next to nothing.
5. **Stars, not chains.** A chain to the latest copy is 0.25 bits smaller than a star to the first, but rebuilding a
   page then decodes 6.9 pages on average (up to 28); a star decodes 1. A star rebuild (0.13 ms to turn the copy +
   0.21 ms to decode) is faster than recomputing (0.55 ms per page at P's prefill rate).
6. **The graph against T on every page.** T everywhere would be 52% of raw. The graph (references free, twins as
   diffs against D's copy, moved pages at their best edge including T's, new pages with T) is **6.4% smaller held out, 7.6% over all pages**:
   6.0 points from twins, 0.4 from moved pages (new pages' T size estimated at T's rate on moved pages). Against raw,
   the plain graph (no codec) is 16.5% under raw distinct pages held out.
7. **The sample** (`diffedge_sample.npz`, 96 moved pages, head 0; 22 test pages, so a hint): in layers 1–4 the
   residual is smooth across tokens (a token's residual from the one before leaves 31–47% in keys) and low-rank
   (in layers 3–4, its top 4 directions leave 29–60%); deeper, at least half of it stays whatever is tried (51–79%
   left by the best of these).

**Caveats.** 4 conversations and 130 held-out moved pages; one model; answers capped at 128 tokens. T was not run on
new pages or twins (estimated as above). T's edges use T's prediction rebuilt in PyTorch (`pd_region.TPredictor`,
which matched T's own sizes in job 30620: 8.34 against 8.35); the blend's rebuild time (T decodes layer by layer) is
not timed. The recompute time is P's prefill rate; recomputing a moved page for real also needs its prefix. The
training conversations had more pages with long agreement and many copies, so their numbers (blend 7.62) are not a
held-out result.

**In short:** the graph pays where the same tokens are computed twice in the same context (decode/prefill twins),
not where text is found again in a new context (T already predicts that nearly as well).

## 4. What is on `pd-blend`

| File | What |
| --- | --- |
| `scripts/pd_diffedge.py` | the stage: pages, edges, fits, coding, graph, chains, times; writes `diffedge_report.md/.json`, `diffedge_pages.json` (one row per page: bytes per edge and layer group), `diffedge_sample.npz` |
| `scripts/pd_diffedge_explore.py` | numpy on the sample, any machine: second order, cross-layer, token lag, low rank, agreement, twins |
| `scripts/pd_matrix.sh`, `scripts/pd_campaign.py` | `MATRIX=capture` settings (below); `MATRIX=diffedge` (the stage on a kept capture); the disk plan counts each of the 3 D engines' stores |
| `scripts/run_pd_vllm.sh` | runs `tree`, `region`, `diffedge` after the engines on the run's own capture |
| `kv_compare/test_pd_baselines.py` | a stand-in capture with a known drift, twins and a stand-in T; the fits; the runner and matrix |
| `docs/pd_delta.md` | the design, every stage, and these results |

Capture settings: `KVPD_CAPTURE_CONVS` (6; 2-110), `KVPD_CAPTURE_LAYOUTS` (`docs_first,history_first`, or
`interleaved`), `KVPD_CAPTURE_D_GB` (2 GB per D engine), `KVPD_CAPTURE_AFTER` (`tree,region,diffedge`, or `none`);
for the stage, `DIFFEDGE_CODEC` (`t2-hbh`, or `none` for plain edges only).

## 5. How to resume

- **Read the results:** `git fetch origin evidence/pd-blend/649dc8556541/diffedge/job30673`, then open
  `benchmark_results/pd_card/pd_diffedge_20261125T153406Z/matrix/*/diffedge_report.md` in `FETCH_HEAD`; on any
  machine, `python3 scripts/pd_diffedge_explore.py <that folder>/diffedge_sample.npz` (from `pd-blend`).
- **On the cluster:** `module load slurm` in every new login shell; `srun` needs `--mem` (a quick check:
  `srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 -c 1 --mem=2G -t 00:05:00 bash -c 'df -h /tmp'`).
  Do not push to a campaign's branch between its `create` and `run`.
- **Disk on the node's `/tmp`:** a 4-conversation capture needs about 30 GB free with the cache in place (21.4 GB
  plus 10 kept free), 24 GB more without it; the stage on a kept capture needs 24 GB (14 plus 10).
- **A new capture** (the settings line is what changes between tests):

```bash
module load slurm
bash <<'LOGIN'
set -Eeuo pipefail
umask 077
trap 'printf "PD CAMPAIGN STOPPED: line=%s exit=%s; the campaign and its output are kept\n" "$LINENO" "$?" >&2' ERR
test -z "${SLURM_JOB_ID:-}"
cd ~/kv_reconstruction
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q --no-tags origin pd-blend
git switch -q pd-blend 2>/dev/null || git switch -q -c pd-blend --track origin/pd-blend
git merge -q --ff-only origin/pd-blend
export KVPD_BRANCH=pd-blend
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/pd-blend | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_MATRIX=capture
export KVPD_CAPTURE_CONVS=4 KVPD_CAPTURE_LAYOUTS=interleaved KVPD_CAPTURE_AFTER=tree,diffedge
export KVPD_TAG="pd_${KVPD_MATRIX}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p ~/.kvpd_campaigns
STATE=~/.kvpd_campaigns/$KVPD_TAG.sh
printf 'export KVPD_BRANCH=%q\nexport KVPD_PIN=%q\nexport KVPD_TAG=%q\nexport KVPD_CAPTURE_CONVS=4 KVPD_CAPTURE_LAYOUTS=interleaved KVPD_CAPTURE_AFTER=tree,diffedge\n' "$KVPD_BRANCH" "$KVPD_PIN" "$KVPD_TAG" > "$STATE"
printf 'PIN=%s\nTAG=%s\nSTATE=%s\n' "$KVPD_PIN" "$KVPD_TAG" "$STATE"
for STAGE in create run cleanup verify-cleanup review publish; do
    bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG"
done
echo "Campaign $KVPD_TAG published (the PUBLISHED line above names its branch)."
LOGIN
```

- **The stage again on the newest kept capture:** the same block with `export KVPD_MATRIX=diffedge` and without the
  `KVPD_CAPTURE_*` line (in the `export` and in the `printf`). Each campaign prints a `PUBLISHED evidence/...` line:
  that branch holds its results.

## 6. Next

1. **Longer answers** (say 512 tokens: `MTRAG_MAX_TOKENS` is a runner setting; add it to the capture settings).
   Agentic outputs are long and come back as history, so twins, the one clear win, become a much larger share.
2. **A sliding-window or compaction layout** (drop the oldest turn): the radix tree loses the prefix and nearly every
   page becomes a moved copy with very long agreement, which this run never reached.
3. **Measure T on new pages and twins**, so the comparison with T everywhere is measured, not estimated.
4. **A star blend** (T + w (first − T)) to keep rebuilds at one page, and the early layers' structure (token lag, low
   rank) for T itself.
