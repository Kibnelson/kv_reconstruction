# Prefill-derived KV reconstruction recipes

The experiment asks whether metadata collected while KV is available during
prefill can reduce the **complete exact reconstruction record**. Every bit that
the predictor cannot recover still has to be represented. Metadata alone is
not a novelty claim; the contribution must be demonstrated by the mechanism,
matched byte savings, bounded reconstruction work, and comparison with prior art.

## Reconstruction contract

A record contains one layer's K or V tile, shaped `[tokens, channels]`. Every
decoder input is either in that record or in explicitly identified, charged
frozen decoder state. The decoder does not rerun transformer layers or depend
on other KV records. Original BF16 words are recovered exactly, including signs,
signed zero, infinities, and NaN payloads; the predictors are not lossy KV caches.

Most recipes predict **15-bit BF16 magnitude bit patterns as integers**. These
symbols are not floating-point absolute values. Original signs are stored
separately. The exponent-only recipes instead preserve the sign and seven
fraction bits as one unchanged byte and encode the eight exponent bits.

| Recipe | Information collected from this tile | Reconstruction |
|---|---|---|
| Zero control | None | Exact errors from zero |
| Midpoint | Minimum and maximum per channel across token rows | Stored integer midpoint plus exact corrections |
| Affine | First and last token's magnitude symbols per channel | Integer endpoint interpolation plus exact corrections |
| E8 template | Per-channel magnitude midpoints | Stored E8 description recreates one channel template, then exact corrections recover every token |
| Exponent median | Median exponent per channel | Stored one-byte anchor plus exact exponent errors |
| Exponent affine | First and last exponent per channel | Stored endpoint bytes plus exact exponent errors |
| Affine + residual dictionary | Affine endpoints from this tile; dictionary fitted on TRAIN only | Affine prediction + dictionary residual pattern + final exact correction |

Affine interpolation uses the same floor division at encode and decode. E8
describes a channel template, rather than every token as in the older
`native_e8_symbol_bitpack` baseline. The dictionary is an additional source of
prediction, with its full state cost included, rather than a replacement for
the tile's prefill-derived metadata.

## Matched incremental experiments

`--native-preset insights` contains **33 native methods**: the 16-method main
set, six layout/group-size variants, seven exponent-tier variants, and four
affine-residual dictionaries. Existing `main` and `all` sets stay unchanged.

1. **Layout and group size.** Midpoint and affine corrections are grouped
   across neighboring channels (`channel`) or across token rows (`time`), with
   eight or 64 values per width group. The predictor and original sign bits stay
   the same. Smaller groups can reduce correction widths but cost more width
   metadata; actual packet accounting measures the net result.
2. **Exponent prediction and tier coding.** Every exponent error is ZigZag
   coded into a two-bit primary stream. Escapes use a three-bit secondary
   stream and, when needed, a nine-bit rare stream. Per-256-value segment
   offsets locate the compact streams. The sign/fraction byte, anchors, tiers,
   padding, offsets, and packet header all count. `fixed127` predicts the BF16
   exponent bias without looking at the tile; it is the useful matched control
   for prefill-derived median/affine prediction. `zero` is another control.
   Channel/time permutations preserve the tier counts and stream bit totals;
   a header spelling difference is not an entropy improvement.
3. **Residual-trained dictionaries.** Separate 16- and 64-entry dictionaries
   are fitted on errors **after affine prediction**, using only the three TRAIN
   captures. Channel/time layouts have separately fitted state. IDs cost
   **0.5 or 0.75 bits per value**, respectively, before headers and state.
   Assignment minimizes the final eight-value group's encoded width, with an
   L1 tie-break; fitting uses heuristic median updates. A dictionary succeeds
   only when correction savings exceed IDs and full state costs on held-out KV.

For magnitude symbol `x` and prediction `p`, the circular error is
`d = ((x - p + 16384) mod 32768) - 16384`. Original coset variants store
`x mod 2**b`; matched residual variants store ZigZag errors using the same
per-group width `b`. The correction bit counts are identical. Cosets provide
an exact reconstruction rule; they do not add savings to the matched residual
stream. The new layout/dictionary experiments use residuals as the control.

## Execution and accounting

`--prefill-recipes` runs a synchronous observer between model prefill chunks.
It builds selected recipes directly from complete device KV tiles, decodes
them on that device, and checks every original word before CPU export. A tile
crossing a chunk boundary is handled when its last row becomes available.

The `prefill_observer` report is separate from `native_packet`, which measures
post-capture encoding, repeated reconstruction, and packet transfer. Observer
prefill duration includes all enabled recipes, their verification decodes,
layout copies, synchronization, and model work. It is **not the overhead of
deploying a single recipe**. The observer discards packets after checking them;
the transformer still retains its full raw KV cache. No compact attention,
capacity, concurrency, TTFT, or TPOT benefit is established by this experiment.

All real packet fields count. The specified 32-byte cache header and
`8*(record_count+1)` offset table are added arithmetically; a complete outer
container is not persisted/reopened. Full decoder state is charged once per
cache. Residual dictionaries conservatively charge the larger of the complete
serialized artifact and actual distinct device tensor storage, and report both.
Other native methods report their physical tensor-state contract. Sampling
produces labelled full-cache estimates, rather than full-cache measurements.

`--recipe-diagnostics` records each selected packet's identity, actual
component bytes, and encode/decode times. It is automatic for `insights`.
`recipe_analysis.json` compares only matched standard native records within one
case; DEV and TEST, observer and post-capture, and CPU and GPU formats are not
pooled. It reports state-aware increments, K/V breakdowns, and normalized timing.

Its **stateless size-only oracle** chooses the smallest already encoded packet
per record, then charges one mode byte per record and the full outer framing.
It excludes stateful candidates, rejects mismatched records, and shows gains
against the best fixed stateless method. This measures selection headroom. It
does not implement an online selector, repackage payloads, measure its encoding
cost, or establish a latency benefit.

## Run on the A4500

From the checkout with its CUDA virtual environment active:

```bash
git pull --ff-only
python qwen_kv_gpu_bench.py --self-test
python qwen_kv_gpu_bench.py --native-preset insights --list-methods

python qwen_kv_gpu_bench.py --device cuda --preset smoke \
  --methods raw exponent_huffman_lower_deflate \
  --native-preset insights --prefill-recipes --tile-tokens 64 \
  --lengths 2048 8192 --cases 3 --max-records 144 \
  --dev-tokens 512 --generation-steps 2 \
  --generation-methods \
    native_prefill_exponent_median_time_tiered \
    native_prefill_affine_dictionary16_channel_g8_residual \
    native_prefill_affine_time_g8_residual \
  --output runs/qwen_insights_t64
```

This uses three held-out synthetic fixtures at each length and explicitly
requests full 512-token DEV-prefix continuation checks for the three named
recipes. Those checks compare logits, tokens, and appended KV for two steps;
they do not validate every unmeasured TEST record or demonstrate task quality.
A skipped/failed check remains visible. Repeating with `--tile-tokens 16` and a
fresh output directory tests metadata amortization. `--full-cache` checks all
records and can take substantially longer. `--measure-io` remains an optional
buffered I/O diagnostic, not a physical SSD bandwidth measurement.

After a completed run, push the curated reports for review:

```bash
bash <<'BASH'
set -e
source_dir="runs/qwen_insights_t64"
dest="benchmark_results/qwen_insights_a4500_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$dest"
for file in results.json results.csv summary.md frozen_manifest.json source_manifest.json recipe_analysis.json; do
  cp "$source_dir/$file" "$dest/"
done
git add -- "$dest"
git commit --only -m "Add A4500 prefill recipe insight results" -- "$dest"
git push origin main
BASH
```

Keep the complete local run directory and its adjacent results ZIP as well:
they include the actual fitted dictionary artifacts needed to replay decoding.
The curated Git reports contain measurements and provenance, not model weights
or raw KV. New Qwen GPU savings and timing require this target run; synthetic
CPU checks and reused-capture diagnostics are labelled separately.

## Validation

Tests cover independent NumPy predictors/group-width oracles, all 65,536 BF16
patterns, wraparound and half-modulus ties, tails and singleton tiles, packet
roundtrips, TRAIN-only state provenance, actual byte accounting, matched record
identities, selector/framing costs, and full-state-once arithmetic. CUDA tests
skip explicitly without hardware. These establish implementation correctness;
they are not pretrained Qwen GPU performance results.
