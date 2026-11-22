# The MTRAG page trace: is there a differential-storage opportunity?

Branch `pd-trace`. A trace campaign answers one question before any differential store is built: in multi-turn RAG,
how much of the KV that PD moves and stores is already somewhere, either exactly or nearly? "Nearly" means the same
text at another place, after another prefix. That is the case a prefix cache (vLLM, LMCache) misses, and the case a
differential store would code as a small delta against a base.

## What runs

`MATRIX=trace` is one run on gpu-grad-01:

- Layout and link: 1p3d, ring, 100 Gbit/s, raw codec.
- Stages: `check,mtrag,report`, with `TRACE_PAGES=1` and `MAXLEN=32768`.
- Stage `mtrag` (`scripts/pd_bench.py`) replays the 110 MTRAG human conversations through `scripts/pd_mtrag.py`. The
  data is IBM/mt-rag-benchmark at a pinned commit; `scripts/pd_setup.sh` fetches it and checks its SHA-256.
- Every turn's prompt is built from token-id pieces: system, the turn's passages, the history with the model's own
  earlier answers, and the question.
- The stage runs every combination of three prompt layouts and two routings. Prefix caches are reset before each run.

  | Layout | Prompt order |
  | --- | --- |
  | `docs_first` | system, passages, history, question |
  | `history_first` | system, history, passages, question |
  | `interleaved` | append-only: every turn's passages, question and answer in turn |

  | Routing | Traced? |
  | --- | --- |
  | `conversation` | yes, every request |
  | `round_robin` | no; it gives clean timings |

## What the trace records (`kv_compare/pd/trace.py`)

| Where | Record | What it holds |
| --- | --- | --- |
| P | `pages` | The content hash of every page sent. A page is 16 tokens × 1 layer = 64 KiB. |
| P | `relations` | One per request. Its related spans: tokens whose 16-token window appeared earlier in the run, not as a shared prefix at the same place. |
| P | `delta_stats` | Running totals per run and kind of piece (doc, answer, question, other): histograms per layer, keys and values apart. |
| D | `held` | For the pages P sent again that D already held: whether D's bytes are equal. Where they differ, the same histograms of P's words against D's. |

The `delta_stats` histograms cover:

- `xor`: the XOR with the base.
- `diff`: the ordered-integer difference, so one ULP is 1.
- `rot`: keys only. The base key is turned by RoPE to the new position first, then differenced.
- `alone`: the sign and exponent symbols, which give the no-base floor.

Measured tokens are a sample, 1 in `TRACE_SAMPLE` (8). The sample is chosen by window hash, so a sampled token's base
was sampled too. Only the sampled words go to the host. Bases are kept within `TRACE_STORE_GB` (16 GB), oldest
dropped first. Every related token is counted, sampled or not.

## Reading `trace_report.md`

1. **Moved again / recomputed / same bytes / affinity**: what PD does twice today.
2. **Pages by what already existed**: 1 prefix, 2 same bytes, 3 same text, 4 new. Also the store size keyed by
   prefix (LMCache), by content, and by content without layer 0.
3. **How close related KV is**: the differential opportunity.
   - Related share of prompt tokens, by kind; same place or moved; within the conversation or across.
   - Bits per value of xor, diff and rot against alone, per kind and per layer.
   - The related tokens' GB raw, coded alone, and as deltas.
   - D's recomputed pages: how far P's recomputation is from what D held.

The opportunity is confirmed if two things hold together:

- the related share is large, which we expect in `docs_first` (the history moves) and `history_first` (the passages
  move);
- best bits stay well under alone, at least for layers near the input (layer 0's values should be equal, and its
  turned keys about 0 bits).

If deep layers come out no better than alone, the opportunity is limited to recomputation from shallow layers, not
deltas.

Limits:

- One P engine: relations are found within that engine.
- `rot` assumes Llama's neox RoPE, read from the model's config.
- `best` takes the cheapest measure for each layer.

## Run it (login host, in tmux)

First free the home quota (the failed start ran out at `git fetch`). §6.1 of `docs/handover_pd_baselines.md` on
`main` shows how to check it.

The campaign runs `pd-trace`'s GitHub head as its pin. Evidence goes to `evidence/pd-trace/<pin12>/trace/job<id>`.

```bash
module load slurm
bash <<'LOGIN'
set -Eeuo pipefail
umask 077
trap 'printf "PD CAMPAIGN STOPPED: line=%s exit=%s; the campaign and its output are kept\n" "$LINENO" "$?" >&2' ERR
test -z "${SLURM_JOB_ID:-}"
cd ~/kv_reconstruction
test -z "$(git status --porcelain --untracked-files=no)"
git fetch -q --no-tags origin pd-trace
git switch -q pd-trace 2>/dev/null || git switch -q -c pd-trace --track origin/pd-trace
git merge -q --ff-only origin/pd-trace
export KVPD_BRANCH=pd-trace
export KVPD_PIN=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/pd-trace | awk '{print $1}')" = "$KVPD_PIN"
export KVPD_MATRIX=trace
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

To resume after a stop, run `source ~/.kvpd_campaigns/<tag>.sh` (it sets `KVPD_BRANCH` as well), then run the
remaining stages as in §6.5 of the handover.

### Knobs

Pass these through `RUNS` or the environment of the run:

| Knob | Default | What |
| --- | --- | --- |
| `MTRAG_LIMIT` | 0 (all) | Conversations replayed. 20 gives a quick look. |
| `MTRAG_LAYOUTS` | `docs_first,history_first,interleaved` | Prompt layouts. |
| `MTRAG_POLICIES` | `conversation,round_robin` | Routings. |
| `TRACE_SAMPLE` | 8 | 1 in N tokens measured. |
| `TRACE_STORE_GB` | 16 | Host memory for bases. |
| `TRACE_MAX_TOKENS` | 8192 | Sampled related tokens measured per request. |

To rebuild the report from a published run:

```bash
python3 -I scripts/pd_trace_report.py <run dir> --markdown trace_report.md --json trace_report.json
```
