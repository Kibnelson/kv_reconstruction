# Exact KV frontier: combinations and the 40% target

The target is at least **40% fewer stored bytes than original BF16 KV**, including packet fields, descriptors, index/framing and complete decoder state. This is a target to test, not an achieved Qwen result. A 40% reduction corresponds to at most **9.6 bits per original 16-bit value**.

The [previous A4500 study](qwen_producer_gpu_review.md) found about 29.84% mean estimated savings for frozen channel exponents. It also verified that earlier K states could replay to the original final K with zero correction payload on the measured tiles. The new sweep combines those findings and measures the remaining information budget.

## Why another representation change is necessary

The old format always retains eight sign/mantissa bits and two primary exponent bits. It therefore uses at least ten bits per value: **37.5% is its maximum saving even with free metadata, perfect predictions and no escapes**. Better bases alone cannot reach 40% within that format.

The frontier adds both adaptive lower-byte coding and opportunistic primary-stream coding. These can remove that fixed floor on structured data; their usefulness on pretrained Qwen KV must be measured. They retain raw/fixed-stream fallbacks with their actual costs counted. This does not imply a universal information-theoretic limit on KV compression.

## The 16 candidates

Three previous recipes remain as controls:

- `producer_kv_median`
- `producer_frozen_exponent`
- `producer_pre_rope_coset`

The new recipes use the same frontier packet schema:

| Family | K source | Local adjustment | Lower-byte variants |
|---|---|---|---|
| `frontier_cost_final_raw8` | Final post-RoPE K | None | Raw eight bits |
| `frontier_shift_final_*` | Final post-RoPE K | Optional per-head shift | Raw, bitplanes, first-token XOR, previous-token XOR |
| `frontier_shift_pre_rope_*` | Actual normalized K before RoPE | Optional per-head shift | The same four choices |
| `frontier_shift_kproj_*` | Actual raw K projection | Optional per-head shift | The same four choices |

The suffixes are `raw8`, `plane_raw`, `plane_first`, and `plane_prev`. V is encoded alongside K in every paired record. Existing E8/dictionary experiments remain available separately; this sweep tests combinations motivated by the latest measured costs rather than stacking every historical representation.

## How the additions work

**Cost-trained bases.** Three separate TRAIN-only models fit final K, pre-RoPE K and raw projected K; each also fits actual V. Every channel considers all 256 exponent bases, minimizing the exact logical tier cost

\[
2+3\,\mathbf{1}[\operatorname{zigzag}(e-p)\ge3]
+9\,\mathbf{1}[\operatorname{zigzag}(e-p)\ge10].
\]

The original TRAIN-median control remains unchanged. Ties choose the base nearest the TRAIN median and then the lower base. No DEV or TEST capture fits a model. The fit optimizes the original tier objective; opportunistic primary-stream packing can change the final cost, which is measured separately.

**Small prefill adjustments.** For each actual KV head and tile, the encoder searches shifts from −16 to +16 around the frozen channel bases. A shift is retained only when its encoded fields and descriptor/index contribution beat the zero-shift alternative. This provides local adaptation without storing an anchor for every channel. Both choices remain exactly reversible.

**Verified zero replay.** Pre-RoPE keys replay through RoPE; raw projections replay through key normalization and RoPE. The encoder compares the replayed result with every original final K word. Exact matches use a zero-correction mode without a width table. Any mismatch retains a full 16-bit coset correction stream and its width metadata. Source K, actual V, replay frequencies, normalization parameters, position information and all state bytes count.

**Lower-byte planes.** Sign plus seven mantissa bits form one byte. The four variants retain this byte directly or code its bitplanes after no predictor, first-token XOR, or previous-token XOR within the tile. First-token seeds are stored. Each 256-bit group can store raw bits, a constant, or minority positions; modes, counts, polarity, positions, padding and descriptors all count. Previous-token decoding uses exact integer prefix parity. A complete lower contribution that expands relative to raw bytes falls back to raw storage.

**Adaptive primary stream.** The primary exponent symbols can retain their fixed two-bit stream or use the same exact bitplane machinery. The encoder compares the complete encoded contribution and retains the smaller plan. This can remove the two-bit floor where the primary codes have sufficient structure. Secondary and rare escapes remain exact.

**Compact indices.** Secondary/rare segment tables store starts only; their terminal counts already appear in the descriptor. Geometry selects uint16 when every possible start fits, with a uint32 fallback. A terminal count of 65,536 is never truncated into uint16. Actual backing tensor storage is charged, including any padding.

Producer replay remains an execution-contract experiment. Replay state binds the selected implementation/runtime/device contract, and incompatible restoration must be rejected. These tests do not establish universally identical floating-point replay across hardware or software versions.

## Evidence and diagnostics

All candidates construct and verify selected packets inside attention forward by default, then repeat isolated post-capture measurements. The diagnostic hooks still copy selected activations through CPU and retain full raw model KV. Capture duration includes all candidates and verification; it is not one encoder's production overhead. `producer_hooks_included` and `inline_producer_recipes_included` make that timing boundary explicit.

The report adds:

- Actual effective bits per original BF16 word, including full state once per cache; a 9.6-bit threshold and the remaining gap.
- Same-record increments against the frozen control; packet components and source/shift/primary/lower/fallback diagnostics.
- Exponent, sign, mantissa and lower-byte empirical entropies for actual final K/V and earlier K sources.
- Conditional and temporal statistics, with sample counts and explicit sparse-context bias warnings. Empirical entropy is neither a proven lower bound nor an achieved encoding size.
- Actual independently framed CPU zlib controls for channel-major lower bytes and first/previous XOR. They verify lossless restoration, charge seeds and framing, and report **CPU size only**. They are not GPU entropy-decoding results.

Five fixed representative recipes restore complete DEV prefixes for continued-generation checks. All measured packets must reconstruct exactly and their inline sizes/state must match replay. The unit integration restores every one of the 16 recipes after clearing the source-record mappings. TEST sampling, full-cache extrapolations and outer-container arithmetic charges remain explicit.

## What has been tested locally

Local tests cover exhaustive BF16 symbols, boundary/tail geometries, independent bit/cost oracles, compact-index overflow, zero/nonzero replay, runtime identity and actual tiny-Qwen continuation. Random tiny-model results establish functionality and format capabilities; their compression rates are not pretrained Qwen evidence. A deliberately structured positive control can cross 40%, showing that the format permits the target, without establishing that real KV has the required redundancy.

A separate probe reused six held-out **SmolLM2-135M CPU** cache captures at 512/2,048 tokens. All three lower-bit plane variants selected raw-byte fallback on every sampled stream. This is useful negative evidence for that model/data; it is not a Qwen result. The probe serializes only lower-byte streams, so its sizes cannot satisfy the complete-KV 40% target. See [the probe report](../benchmark_results/frontier_lower_smol_cpu_20261019/results.json).

## Run the Qwen sweep on the A4500

```bash
cd ~/kv_reconstruction
git pull --ff-only
source .venv/bin/activate
python qwen_kv_gpu_bench.py --self-test

python qwen_kv_gpu_bench.py --producer-study --producer-preset frontier \
  --device cuda --lengths 2048 8192 --cases 3 \
  --train-tokens 512 --dev-tokens 512 --chunk-tokens 512 \
  --tile-tokens 64 --max-tiles 72 --rank 8 \
  --repeats 3 --sample-pairs 16 --generation-steps 2 \
  --output runs/qwen_frontier_40
```

This uses 72 **paired** layer/token tiles per TEST case. Full TEST-cache bytes are estimates; DEV is complete for continuation. The state-inclusive report labels any estimated threshold crossing accordingly. A sampled crossing requires full-cache validation and broader contexts before a general 40% claim. Use a fresh output directory for every run.

For a no-download functionality check:

```bash
python qwen_kv_gpu_bench.py --producer-study --producer-preset frontier \
  --tiny --device cpu --cases 1 --repeats 1 \
  --output runs/frontier_tiny
```

To push the target results:

```bash
python scripts/package_results.py runs/qwen_frontier_40 \
  --destination benchmark_results/qwen_frontier_40_a4500
git add -f benchmark_results/qwen_frontier_40_a4500
git commit -m "Add A4500 exact KV frontier results"
git push origin main
```

The ZIP also contains verified reports/state and excludes raw activations, raw KV and full model weights.

The packaging destination must be fresh or empty. The helper stores the complete report as `results.json.gz`, verifies lossless decompression by SHA256, and copies the smaller reports and decoder artifacts. The original raw report stays in the run directory. This avoids GitHub's per-file limit; compression changes the report packaging, not any measured KV byte count. Reviewers can open it with `gzip.open(path, 'rt')` or decompress a local copy.

### Repair a rejected oversized-report commit

If `main` is ahead by exactly one unpublished result commit and GitHub rejected `results.json`, replace that file in the commit. Keep the original run directory intact:

```bash
cd ~/kv_reconstruction
gzip -c benchmark_results/qwen_frontier_40_a4500/results.json \
  > benchmark_results/qwen_frontier_40_a4500/results.json.gz && \
gzip -t benchmark_results/qwen_frontier_40_a4500/results.json.gz && \
git rm --cached -- benchmark_results/qwen_frontier_40_a4500/results.json && \
git add -f -- benchmark_results/qwen_frontier_40_a4500/results.json.gz && \
git commit --amend --no-edit && \
git pull --rebase origin main && \
git push origin main
```

`--cached` removes only the Git entry and keeps the raw working copy. Stage the `.gz` file explicitly during this repair. A separate removal commit would leave the oversized blob in the history being pushed. This repair needs neither a forced push nor a benchmark rerun.

## How to choose the next step

First check whether actual bytes move toward 9.6 bits, and identify which streams improve. A smaller prediction error, lower empirical entropy or a synthetic positive control is insufficient. Then assess whether the byte reduction repays restore work under a matched raw-transfer baseline. Compact-resident attention, bounded materialization and concurrency remain later serving work. The measured combination and its practical advantage must establish the contribution; hooks or a 40% goal alone do not establish novelty.
