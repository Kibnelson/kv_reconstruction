# Comparing compact prefill hints and exact reconstruction

This study implements the tests proposed in
[the compact-metadata review](compact_prefill_metadata_opportunities.md).
Run it through the portable launcher with `--compact-prefill-study`.
All representations restore the original BF16 words. Network work uses the
selected device; packet entropy coding is a CPU reference. This is not an
encoded-resident attention implementation or a concurrent-serving benchmark.

## Comparisons

The default suite contains fourteen fitted variants:

| Family | Variants | What it isolates |
|---|---|---|
| Existing-fields control | `lsb_local_none` | Local known fields without pooled context or a hint |
| Decoder-known pooling | `lsb_pool_none` | Additional context with zero transmitted hint bits |
| Tiny token hints | `lsb_pool_hidden1`, `lsb_pool_hidden2`, `lsb_pool_hidden4`, `lsb_pool_kv2` | One, two or four total bits per token/layer; actual-hidden versus produced-KV source |
| Page sharing | `lsb_page_hidden2`, `lsb_page_kv2` | One two-bit hint per page, with actual byte padding charged |
| Exact K first | `lsb_kfirst_none`, `lsb_kfirst_kv2`, `lsb_kfirst_hidden2` | Full same-page K as context for V, with and without a hint |
| Progressive high-bit layout | `msb_pool_none`, `msb_pool_kv2`, `msb_pool_hidden2` | Coarse numerical KV followed by grouped refinement, with matched hint-source controls |

There are also independent field and frozen-exponent controls. A learned
variant is not declared successful merely because another learned variant is
larger. A control may remain the preferred representation.

All variants use a scalar hint extractor with compact calibration rather than
the previous eight-BF16-feature encoder. The actual-hidden input comes from
the attention hooks during prefill. The decoder receives only packed hint
IDs and its own page fields; it never receives hidden activations. Hint
quality may decrease sharply when made smaller, so the reductions in hint
payload are not forecasts of final KV savings.

The low-bit layout retains two low mantissa bits and codes the remaining
groups. The high-bit layout retains two high mantissa bits and restores the
remaining groups progressively. Known fields and exact earlier groups enter
later predictions. Bit ordering alone does not reduce joint entropy; low
rounding bits may be less predictable. The comparison measures the actual
complete stream rather than assuming a benefit from numerical ordering.

## Run on the A4500

Activate the existing environment and first run a short Qwen comparison:

```bash
cd ~/kv_reconstruction
git pull --ff-only origin main
source .venv/bin/activate

python qwen_kv_gpu_bench.py --compact-prefill-study \
  --device cuda --steps 200 \
  --train-tokens 512 --dev-tokens 128 \
  --lengths 2048 --cases 1 \
  --chunk-tokens 512 --tile-tokens 16 --max-pairs 36 \
  --repeats 1 --generation-steps 2 \
  --output runs/qwen_compact_smoke
```

This tests the complete variant set, but short fitting and one sampled TEST
case are a feasibility screen. CPU entropy coding and complete DEV checks may
take substantially longer than the network fitting. Use a fresh output name
when repeating a run.

For the larger comparison:

```bash
python qwen_kv_gpu_bench.py --compact-prefill-study \
  --device cuda --steps 4096 \
  --train-tokens 512 --dev-tokens 128 \
  --lengths 2048 8192 --cases 2 \
  --chunk-tokens 512 --tile-tokens 64 --max-pairs 72 \
  --repeats 3 --generation-steps 2 \
  --output runs/qwen_compact_compare
```

The pinned default model is `Qwen/Qwen3-4B-Instruct-2507` at
`cdbee75f17c01a7cc42f958dc650907174af0554`. TRAIN, DEV and TEST are
separate fixtures; TEST uses the training study's prespecified text/code
holdouts. These synthetic fixtures do not establish workload diversity or
long-context task quality. The 8K full-cache byte total is an extrapolation
when only sampled pages are encoded.

At 64 channels/update, Qwen's 36 layers and two stages need at least 2,304
updates to visit every eligible layer/channel/stage once. The 4,096-update
command clears that coverage floor; it does not establish convergence. Read
the saved exposure and loss summaries before attributing a failure to the
representation. The 200-update smoke intentionally does not cover all slots.

## Choosing a candidate

Settings and weights freeze before TEST. The study selects its candidate on
DEV using complete packet bytes plus charged codec state, with the target
cache size recorded in the report. TEST compares that frozen candidate
against the controls at each requested context length. TEST does not silently
replace it with whichever row happens to win.

Read `decision_summary.json` together with `summary.md`. Require exact
restoration and a net gain over the independent field/exponent controls.
Then compare restoration time and the probability that a page actually uses
an entropy stream. Raw fallback can be correct while rejecting every learned
prediction. State-sharing estimates are separate from a single-cache result.
A result from the random tiny model is software evidence only.

The retained metadata includes hint arrays/histograms, page/stage byte
components, estimated probability rates, actual arithmetic/raw modes,
restoration timings, state artifacts, capture identities and producer-hook
provenance. Those measurements distinguish a weak hint from an ineffective
coder and support the next experiment without guessing from a savings ratio.

Matched comparisons separate coded data, hints, framing and state. Variant
names can change a JSON header by a few bytes; that is a real wire difference
but does not establish a better predictor. Bounded DEV zero/shuffled-hint
ablations measure fixed-model probability use separately from actual encoded
bytes. Their page addresses and coverage are logged, and they do not tune TEST.

This new path uses joint-symbol arithmetic coding and exact field refinement.
It does not establish an improvement in the earlier 31-bit single-error coset
decoder. That harder correction scheme remains a distinct experiment.

## Share results

The packager includes reports, complete small calibration artifacts and hint
arrays. It excludes raw captures, model weights and the potentially large
packet files. A fresh destination and ZIP are required; it will not overwrite
earlier evidence.

```bash
python scripts/package_compact_results.py runs/qwen_compact_smoke \
  --destination benchmark_results/qwen_compact_smoke_a4500

git add -f benchmark_results/qwen_compact_smoke_a4500 \
  benchmark_results/qwen_compact_smoke_a4500.zip
git commit -m "Add A4500 compact prefill comparison results"
git push origin main
```

For the larger run, substitute `qwen_compact_compare` and
`qwen_compact_compare_a4500`. Failed runs can also be packaged for diagnosis.
Keep the original run directory on the NUC: packet replay remains useful for
later coding experiments even though packets are excluded from GitHub.

## Local functional check

```bash
python qwen_kv_gpu_bench.py --compact-prefill-study --tiny \
  --device cpu --steps 2 --repeats 1 --generation-steps 1 \
  --output runs/compact_tiny
```

This builds a small random Qwen model locally without downloading a checkpoint.
It exercises actual prefill hooks, fitting, serialized model reload, exact
packets and continuation. It cannot choose the best representation for
pretrained Qwen3-4B.

On 2026-10-20, the built standalone passed this functional run for all fourteen
variants on two tiny-model TEST fixtures, including six full-DEV continuation
checks. All 57 targeted unit/regression checks passed. The field baseline won
this tiny run; all 448 learned stages retained packed raw fallbacks, so the
result does not demonstrate a useful prefill signal. Separate unit tests
exercise actual arithmetic decoding and every BF16 bit pattern. The
[validation summary](../benchmark_results/compact_prefill_tiny_cpu_20261020/validation_summary.json)
and [complete small-artifact bundle](../benchmark_results/compact_prefill_tiny_cpu_20261020.zip)
retain the evidence. A pretrained A4500 result is still required.
