# Capturing raw pages for delta analysis

Branch `pd-delta`. The trace (`docs/pd_trace.md`) settled two things:

- **How much repeats:** 43–61% of the docs_first and history_first prompt tokens are text seen before that no prefix
  cache can reuse.
- **What a naive delta saves:** against the first copy, xor or integer difference per value saves only ~15% over
  exponent coding. That is no better than our codec without a base.

That naive coder used one histogram per layer. It never looked at bit planes, conditional entropies, the codec's own
context, low-rank structure, or how the deviation concentrates by token and head (CacheBlend's intuition). Those need
the pages themselves, so this run stores them and analyses them offline.

## What runs (`MATRIX=capture`)

| Setting | Value |
| --- | --- |
| Engines | 1p3d, raw codec, ring at 100 Gbit/s |
| Conversations | MTRAG's first 6 (about 1.2 GB of distinct pages each per layout, so ~14 GB in all) |
| Layouts | docs_first and history_first |
| Routing | by conversation, traced |
| Stages | `check,mtrag,delta,report` |
| P's budget | 16 GB |
| D's budget | 2 GB |

- **P** stores every 16-token block it sends, all 28 layers, as raw BF16 words. Each block is stored once, keyed by its
  pages' content hashes, and each request lists its blocks.
- **D** stores its own copy of any held block whose bytes differ from what P recomputed.
- **Where:** everything goes to `/tmp/$USER/kvpd-captures/<tag>/` on the node. It is kept after cleanup, so the
  analysis can be rerun without rerunning vLLM. It is deleted only by hand.
- **Disk:** the campaign's disk plan adds 18 GB for it.

The store's files:

| File | Contents |
| --- | --- |
| `blocks_<role>_<pid>.bin` | Rows of `[28 layers, 16 tokens, 2048 words]` int16 (1.8 MB each): keys then values per layer. |
| `capture_<role>_<pid>.jsonl` | One record per request (P: its rows), or per differing held piece (D: its blocks and rows). |
| `meta_<role>_<pid>.json` | The row shape. |

The prompts' token ids and pieces come from the run's own records: `runs/mtrag_*.jsonl` and `proxy.jsonl`.

## What stage delta measures (`scripts/pd_delta_analysis.py`, numpy, CPU)

Per layout:

1. **Related tokens.** A token is related if its 16-token window appeared earlier in the run, outside the prefix any
   earlier request shares. No prefix cache can reuse these; the trace's report counted prefix-covered tokens too.
   They are related against two bases: the first earlier copy and the latest one.
2. **Lossless costs.** For 12,000 sampled tokens, per layer, keys and values apart, in bits per value. Keys take the
   base turned by RoPE to the new place. Entropies are cross-validated: counts from half the tokens, cost on the other
   half.

   | Measure | What it is |
   | --- | --- |
   | `alone` | Sign and exponent entropy plus 7 mantissa bits. |
   | `prev` | Given the same token's previous layer (a stand-in for our codec's context). |
   | `xor`, `diff` | As in the trace. |
   | `planes0` | A bit-plane coder, each bit given the plane above, without the base. |
   | `planes` | The same coder, also given the base's bit and whether all planes above matched. |
   | `base` | Sign and exponent given the base's; the mantissa given the base's when they match. |
   | `both` | `base` plus how the previous layer compares with the base. |

3. **Structure.** For up to 48 whole related chunks (64 tokens or more, up to 512), per layer, of the target minus
   the base:
   - the difference's size relative to the target;
   - the share of its energy that rank 1–64 holds (a low-rank patch);
   - how many tokens hold 50%, 80% or 90% of it, and the top 15% tokens' share;
   - whether layer 2's top tokens stay on top (CacheBlend's selection);
   - the largest head's share;
   - how the relative deviation depends on the distance from the related run's start.
4. **D's held copies against P's recomputation:** the same lossless costs.

The run's `delta_report.md` and `delta_report.json` are packaged with the run and embedded in its summary.

## Reading it

| Result | What it means |
| --- | --- |
| `base` or `both` well under `alone` and `prev` in many layers | A lossless differential store pays: code against the base, inside our codec. |
| Rank 4–16 holds most of the energy, or 15% of tokens hold most of it, and they stay the same from layer 2 | An approximate patch (low-rank, or CacheBlend's recomputed tokens) is small. Answer quality is then the next test. |
| Neither | Moved text is stored in full, with our codec; only exact and near-exact copies are shared. |

## Run it (login host, in tmux)

The node's `/tmp` needs about 32 GB free beyond the cache: the 18 GB store, the job's 14 GB peak, and the 10 GB kept
free. The start block is the trace's, with the branch and matrix changed:

```bash
module load slurm
bash <<'LOGIN'
set -Eeuo pipefail
umask 077
trap 'printf "PD CAMPAIGN STOPPED: line=%s exit=%s; the campaign and its output are kept\n" "$LINENO" "$?" >&2' ERR
test -z "${SLURM_JOB_ID:-}"
cd ~/kv_reconstruction
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q --no-tags origin pd-delta
git switch -q pd-delta 2>/dev/null || git switch -q -c pd-delta --track origin/pd-delta
git merge -q --ff-only origin/pd-delta
export KVPD_BRANCH=pd-delta
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/pd-delta | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_MATRIX=capture
export KVPD_TAG="pd_${KVPD_MATRIX}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p ~/.kvpd_campaigns
STATE=~/.kvpd_campaigns/$KVPD_TAG.sh
printf 'export KVPD_BRANCH=%q\nexport KVPD_PIN=%q\nexport KVPD_TAG=%q\n' "$KVPD_BRANCH" "$KVPD_PIN" "$KVPD_TAG" > "$STATE"
printf 'PIN=%s\nTAG=%s\nSTATE=%s\n' "$KVPD_PIN" "$KVPD_TAG" "$STATE"
for STAGE in create run cleanup verify-cleanup review publish; do
    bash scripts/pd_card.sh "$STAGE" "$KVPD_PIN" "$KVPD_TAG"
done
echo "Campaign $KVPD_TAG published (the PUBLISHED line above names its branch)."
LOGIN
```

## Analyse the kept pages again

Rerun the analysis later, with other options or a newer script, without vLLM. It runs on the CPU in a small
allocation on the node and uses three things that stay there after the campaign:

- the pages, in `/tmp/$USER/kvpd-captures/<tag>`;
- the run's records, in the campaign's evidence folder;
- the model's config, in the campaign cache.

The run's own Python is deleted at cleanup, so the first time, make a small venv with numpy. Write the tag out in
`TAG=`.

```bash
module load slurm
srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 -c 16 --mem=64G -t 02:00:00 bash -s <<'NODE'
set -euo pipefail
TAG=pd_capture_<...>
V=/tmp/$USER/delta-venv
[ -x "$V/bin/python" ] || { python3 -m venv "$V" && "$V/bin/pip" -q install numpy; }
CAP=/tmp/$USER/kvpd-captures/$TAG
RUN=$(ls -d /tmp/$USER/kvpd-evidence/$TAG/repo/benchmark_results/pd_card/$TAG/matrix/llama32_3b_* | head -1)
MODEL=$(ls -d /tmp/$USER/kvpd-cache/hf/hub/models--unsloth--Llama-3.2-3B/snapshots/* | head -1)
EV=/tmp/$USER/kvpd-evidence/$TAG/repo/benchmark_results/pd_card/$TAG
cd "$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["tools"])' "$EV/campaign.json")"  # its checkout
"$V/bin/python" -I scripts/pd_delta_analysis.py --capture "$CAP" --run "$RUN" --model-dir "$MODEL" \
    --markdown "$CAP/delta_report.md" --json "$CAP/delta_report.json" | tail -40
NODE
```

To delete the kept pages when done, inside `/tmp/$USER` only:

```bash
srun -p all -A oana2026 -q research12h -N 1 -w gpu-grad-01 -c 1 --mem=1G -t 00:10:00 bash -c '
r=$(realpath -m /tmp/$USER/kvpd-captures/<tag>); case "$r" in /tmp/$USER/kvpd-captures/?*) rm -rf --one-file-system -- "$r" && echo "removed $r";; esac'
```

## Our KV Diff Graph against EvidenceStore's (`MATRIX=diffgraph`)

No vLLM engines run. The job sets up as usual, which also builds the codec constants. It then runs
`scripts/pd_diffgraph.py` on one GPU over the newest capture in `/tmp/$USER/kvpd-captures`. Set `DIFF_CAPTURE` and
`DIFF_RUN` to choose another capture and run.

The pages are the capture's distinct complete 16-token blocks. Every representation is decoded and compared byte for
byte with the capture.

**Ours** (the t2 codecs). Layer 0 is recomputed from the token ids, as T2 does in the PD path; offline, the recompute
is taken as exact (T2 checks it on every request in the engine). The other layers are coded in one of these units:

| Unit | What is coded together |
| --- | --- |
| T2 per request | the whole prompt (the transfer's unit) |
| T2 per segment | each run of a request's contiguous new pages |
| T2 per page | each 16-token page alone; T's programs take 32 tokens, so a page is padded to 32 and the padding counts in its size |
| Delta per page | layers 1–27 coded by T's value-slice coder against the base page (values as they are, keys turned by RoPE) |

The delta's spread is set per layer, K/V and channel. It is scaled by how long the text has agreed with the base's,
fitted on the even conversations and also reported on the odd ones alone.

The graph keeps, per page, the smaller of T2-per-page and the delta. It is depth one, with bases counted once, like
theirs.

**Theirs:** `scripts/pd_page_compare.py` (zlib level 1) on the same pages.

**Times** (CUDA-synchronised):

- T2 encode and decode, per request, segment and page;
- delta encode, and decode with the base warm;
- a cold rebuild: the base page decoded with T2, then the delta;
- a raw page copied from pinned host memory.

The report is `diffgraph_report.md`, with `page_compare.md` beside it, in the published run.

### First run (job 30576)

Bits per value (16 raw), on the distinct pages. T2 per request is the exception: it covers each request's whole prompt,
the repeated pages included.

| Representation | docs_first | history_first |
| --- | ---: | ---: |
| Theirs: best independent (zlib) | 11.55 | 11.56 |
| Theirs: graph + keys turned | 11.09 | 11.09 |
| T2-hbh per request (repeats included) | 10.27 | 10.32 |
| T2-hbh per segment | 10.54 | 11.35 |
| T2-rank256 per segment | 11.47 | 12.28 |

- **Per segment against theirs.** T2-hbh with no deltas is 4.9% under their graph in docs_first and 2.4% over it in
  history_first. history_first's new pages come in shorter runs.
- **Padding, not the domain (job 30583).** On MTRAG, T2-hbh costs 10.3 bits; on the quick run's prompts 8.2–8.9. This
  note first put that down to the domain (T's constants were fitted on the quick prompts' datasets). Job 30583's T2 per
  page then cost 28.4 bits for 16 real tokens padded to 32 with zeros: zero tokens cost about 20 bits per value each.
  T pads every unit with zeros up to a multiple of 256, about 128 tokens on average on MTRAG; taken out, every MTRAG
  row comes to about 8.2–8.5 bits. The quick prompts (2,048 and 16,384 tokens) need no padding. `MATRIX=tree`
  measures it directly.
- **Times per page.** A delta decodes in 7.0 ms, against about 0.4 ms per page for T2 per segment and 0.08 ms for a
  raw page from host memory. The delta decoder takes one call per page, so its launches dominate.

What did not run, fixed since:

| Problem | Fix |
| --- | --- |
| T2 per page stopped: cuBLAS's int8 product takes more than 16 rows, and a page has 16. | `layerwise.int_mm` pads 16 rows or fewer to 32. |
| No graph without T2 per page, and the deltas' sizes were not reported. | The delta total is reported on its own, held-out conversations too. Each mode's per page runs on its own. |
| The runner stopped while packaging engine logs that an offline run does not have. | The loop skips them. |

### Second run (job 30583)

Bits per value, on the distinct pages (T2 per request: whole prompts, repeats included):

| Representation | docs_first | history_first |
| --- | ---: | ---: |
| Theirs: graph + keys turned | 11.09 | 11.09 |
| T2-hbh per segment | 10.54 | 11.35 |
| T2-hbh per page (16 tokens padded to 32) | 28.42 | 28.41 |
| Our graph (T2 per page or delta) | 19.49 | 19.54 |
| Delta, on the pages with a base (held-out conversations) | 9.20 (9.36) | 9.42 (10.09) |

Decoding per page: T2-rank256 per segment 0.08 ms, as a raw page from host memory; one page per call 3–20 ms (kernel
launches). A delta 7.0 ms with its base decoded, 22.6 ms cold.

## The conversations' block tree (`MATRIX=tree`)

No vLLM engines, like diffgraph: `scripts/pd_tree.py` on one GPU over the newest kept capture (`DIFF_CAPTURE`,
`DIFF_RUN` choose another). Exact reuse can only come from a shared prefix: attention is causal, so a change anywhere
changes every later page at layers 1–27. Most pages are never reused, and recompute is exact. The capture's run shows
both:

- D compared 1,459 held blocks (40,852 pages of one layer) with P's recompute: none differed.
- P, with no prefix cache, recomputed every block an earlier request had; the capture keys blocks by their bytes, so a
  page seen again came out byte-identical.

So a store can keep only what later turns will reuse and recompute the rest.

The requests are replayed in arrival order, a page identified by its bytes. Rules:

| Rule | Keeps |
| --- | --- |
| everything | every distinct page (EvidenceStore's setting) |
| newest path | per conversation, the latest request's pages |
| newest path, no docs | the same without document pages (history first: the documents are never a later prefix) |
| hindsight | each page from its first use to its last (knows the future: the least store for the same hits) |
| none | nothing |

Per rule:

- **hits:** blocks reused from the request's first block on, as a prefix cache can;
- **prefill left:** in tokens, and in seconds at the run's own prefill rate;
- **the store at its peak:** pages, raw GB, and coded with T2-hbh and T2-rank256. Each run of contiguous kept pages of
  the request that wrote them is coded together, decoded and checked byte for byte, with encode and decode times per
  page.

It also counts blocks that repeat a token prefix with the same bytes and with other bytes (a recompute that differed),
and blocks with an earlier block's bytes under another token prefix in the records.

**Padding probe (for the next step).** For the first 24 requests of 256 tokens or more per layout, T2's size:

- as now: zeros up to a multiple of 256;
- with no padding: the multiple of 256 below;
- with the tail under 256 coded apart, padded to 32 with zeros;
- with the tail coded apart, padded to 32 with copies of its last token;
- the whole prompt padded to 256 with copies.

The report is `tree_report.md` (`tree_report.json`), in the published run.

## T plus the base against T alone (`MATRIX=region`)

The deciding test for a region graph, one region per conversation, whose deltas give T the earlier copy as extra
context. A plain delta, coded against the base alone, costs 9.2–10.1 bits per value on pages that have one. That is
no better than T2 once its padding is gone (an estimated ~8.3). So a delta only pays if T and the base together beat
T alone. `scripts/pd_region.py` measures it, offline like diffgraph, on the kept capture.

**Pages:** every distinct page with a base. The base is the latest earlier copy of its 16 tokens, keys turned by
RoPE, else the latest earlier turn's page at the same place.

**Predictions:** layers 1–27 (layer 0 is recomputed, as T2 does) are coded with T's own value-slice coder, in its
uniform mode, against each prediction, and every slice is decoded and checked:

| Prediction | What it is |
| --- | --- |
| T, its own spread | T's prediction, rebuilt outside its kernels with the same arithmetic: each token's layer from its own previous layer and, in head groups, the same layer's earlier groups. T as it codes now |
| T, fitted spread | the same prediction, the spread fitted as below |
| Base | the base's words, keys turned: the diff graph's delta |
| Base, prior | the same, the spread also scaled by T's per-token prior |
| Blend | T + w × (base − T), w per layer, K or V, and how long the text has agreed with the base's (least squares) |

**Spreads:** a scale per layer, K or V, and channel, times a factor per agreement bucket, times T's per-token
multiplier (the decoder has it). Weights and spreads are fitted on the even conversations, and sizes are reported on
the odd ones (held out) and on the training ones. Bits are per value over all 28 layers, with layer 0 at 0 bits, as
in diffgraph's rows.

**Times:** pages are coded 64 at a time, one call per layer group and K or V, which also times a batched delta's
decode.

**A check:** T's rebuilt prediction through this coder, against T2's own size on the same whole requests (no
padding). A wrong prediction would show as a much larger size.

| Result | Then |
| --- | --- |
| The blend clearly under T's own, held out | Deltas pay: build the region store (an index, T2 runs and blended delta runs per conversation, batched rebuilding) |
| The blend about equal to T's own | Regions without deltas: T2 runs per conversation, batched decoding, the tree's keeping rule |

The report is `region_report.md` (`region_report.json`).
