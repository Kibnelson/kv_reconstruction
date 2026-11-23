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
