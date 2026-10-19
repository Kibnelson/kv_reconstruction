# Forward producer metadata: build on the exact recipes

The next experiment keeps the existing contract: a compact record plus completely charged decoder state must reproduce the **original post-RoPE BF16 K and V bits**. It adds access to the computation that produces those bits. The question is whether that information yields a cheaper prediction, a smaller correction stream, or a representation that is faster to restore.

The previous A4500 study established exact reconstruction and about 29.55–29.75% estimated savings for its strongest native exponent recipe. It did not establish a serving speedup or research novelty. This study adds new controls and producer inputs without treating the earlier savings as a result for the new recipes. See [the previous review](qwen_gpu_insights.md).

## What changes inside prefill

For each selected layer and token tile, hooks observe the actual Qwen3 forward computation:

| Forward stage | Observed information | Use in this experiment |
|---|---|---|
| Input normalization output | Normalized hidden state entering the attention projections | Select a few TRAIN-chosen features and predict K/V exponents |
| K projection output | K before key normalization | Store this source, then replay key normalization and RoPE |
| Key normalization output | K immediately before RoPE | Store this source, then replay RoPE |
| Attention inputs | Actual rotary embeddings, cache, positions | Verify the producer stages and audit the compact replay rule |
| V projection output and updated cache | Actual V and final post-RoPE K | Establish the exact reconstruction target |

By default, the attention **forward post-hook** constructs each candidate packet and verifies it before attention returns to the decoder layer. Tiles crossing prefill chunks are stitched and emitted only when complete. The callback uses the just-updated current-layer cache, including on the first chunk when later cache layers are still uninitialized.

This differs from the earlier `--prefill-recipes` observer, which operated on already-produced KV between prefill chunks. The new mode is `--producer-study`; it uses genuine intermediate activations. Diagnostic hooks copy selected inputs to CPU, and the current tensor encoder stages necessary inputs to the selected device. This proves access and exactness inside forward; it is not a zero-copy production encoder. All nine candidates execute in each callback, so instrumented prefill duration is not the deployment overhead of one candidate.

## The nine matched recipes

Each record contains one K/V pair for a layer and token tile. Sharing a stored feature between K and V is therefore explicit and charged once within that pair. Records can be decoded independently given their complete frozen decoder state.

| Recipe suffix | Stored representation | Purpose |
|---|---|---|
| `kv_median` | Tile-derived exponent centers, exact exponent escapes, raw lower eight bits | Current KV-only control in the new paired format |
| `kv_fixed127` | Fixed exponent prediction, exact escapes, raw lower bits | Control for the value of observed exponent statistics |
| `frozen_exponent` | TRAIN-derived per-layer/channel exponent prediction and exact corrections | Strong static control that needs no hidden features |
| `hidden_exponent` | Selected signed int8 hidden features, frozen predictor, exact exponent corrections | Test whether producer information improves on static prediction |
| `hidden_abs_exponent` | The same signed feature format, with an absolute-feature predictor | Test magnitude information against signed information |
| `pre_rope_residual` | Encoded pre-RoPE K, compact rotary rule, exact signed residuals, encoded V | Test whether the earlier K state is cheaper to represent |
| `pre_rope_coset` | The same source and rule, with exact coset corrections | Matched coset control |
| `kproj_norm_rope_residual` | Encoded raw K projection, normalization and rotary rules, exact residuals, encoded V | Move reconstruction one step further upstream |
| `kproj_norm_rope_coset` | The same source and rules, with exact coset corrections | Matched coset control for normalization replay |

CLI names have the prefix `producer_`. These are additional recipes; they do not replace the E8, dictionary or earlier coset implementations.

The hidden predictors are fitted only on the three TRAIN captures. TRAIN variance chooses feature channels, TRAIN ranges set int8 scales, and ridge regression learns small fixed-point coefficients. At decode, integer arithmetic predicts each exponent from the stored features. Exact tiered corrections recover every exponent, including outliers. The signed and absolute variants have separate fully charged trained state. Neither decoder receives the original hidden tensor.

The replay recipes retain an encoded earlier K state; they do not recreate arbitrary K from position alone. Decoder state stores inverse rotary frequencies and scaling. Raw-projection replay additionally stores key normalization weights and epsilon. Absolute positions come from the record address. No full per-token sine/cosine bank is free side information. Backend rounding differences are covered by exact corrections to the final original K words.

Encoder and decoder replay must produce the same prediction. This study checks that contract on the selected device; it does not claim cross-device or cross-version replay compatibility. Timings use prepared decoder state. Complete state artifacts are saved and charged, but custom replay-state artifacts do not yet have a public fresh-process loader, so the study does not measure cold-start restoration.

For full-word corrections, write a target BF16 word as an integer \(x\) and the replay prediction as \(p\). The signed control packs the circular residual. Its coset counterpart stores \(x \bmod 2^b\), using \(p\) to select the correct nearby member. Both use the same adaptive widths and charged five-bit width metadata. Their equal-width payloads need not differ in size. A coset label alone does not create an additional saving.

## Why this is a useful hypothesis

Hidden features can expose token-dependent magnitude information before it is discarded by projection. Normalization and positional rotation are known computations, so replay may trade cheap arithmetic for fewer bytes. Neither benefit is guaranteed: hidden features and predictor state have a cost, and an earlier K plus its corrections may be larger than final K.

In information-theoretic terms, a side description \(S\) is useful only when its cost plus the conditional correction cost is lower than the competing representation:

\[
L(S) + L(\text{exact correction}\mid S) + L(\text{state and framing})
< L(\text{KV-only record}).
\]

The forward pass does not provide a free decoder input. Saving all hidden states would be especially unattractive: Qwen3-4B has 2,560 hidden values but only 2,048 combined K/V values per token per layer. The experiment stores selected small features or a different K source, then charges every additional byte.

For speed, a smaller packet wins only if reduced transfer and memory traffic repay reconstruction, staging and launch costs. The study separates these costs. A later fused implementation would need a small materialization tile and bounded scratch space; compact-resident attention and high-concurrency admission remain separate engineering work.

## Relation to ExANS and novelty

[ExANS's documented path](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache) starts from produced BF16 KV, exploits exponent statistics and uses GPU entropy coding. It can run during prefill once KV exists. Merely scheduling an encoder during prefill does not distinguish this project.

Our coset reconstruction mechanism differs from that documented coding path. The new experimental distinction is the use of **actual producer intermediates to choose and construct exact reconstruction recipes**, with independent restoration and charged side information. Neither generic prediction, cosets, pre-RoPE storage nor hooks alone establish originality. A defensible contribution needs a measured advantage over strong KV-only controls, broader prior-art review, and a practical implementation. These hypotheses are being tested, not claimed as confirmed novel results.

## Evaluation contract

- Fit all predictors on three TRAIN captures only. Keep DEV and TEST separate; freeze candidates and model identities before TEST.
- Compare all nine recipes on identical selected layer/token pairs. Report actual packet components, full decoder state once per cache, and calculated outer framing/index bytes.
- Charge the larger of serialized decoder-state bytes and distinct resident state-tensor bytes. Source K, features, coefficients, indices, scales, normalization state, corrections and padding all count.
- Check every measured pair bit for bit. Require inside-forward packet component sizes and state identities to match post-capture replay.
- Restore the complete DEV prefix for three fixed representative recipes, then compare two continued decode steps for finite identical logits, tokens and appended KV. Separate unit integration covers all nine on a random tiny Qwen.
- Report input staging, encoding, device-resident decoding, and pageable packet transfer plus decoding separately. Repeat the same bounded pair sample for every recipe after one warmup. Eager kernels, allocations and nested-header reconstruction remain part of decode cost.
- Label sampled full-cache sizes as estimates. The complete indexed outer cache is not persisted. The model still holds raw KV; this is not compact-resident serving, a cold SSD test, or a concurrency benchmark.
- Share reports and verified calibration/state artifacts without raw hidden-state arrays, raw KV or language-model weights. Tiny random-weight CPU results establish functionality only.

## Run on the A4500

Use the existing working environment and checkpoint cache:

```bash
cd ~/kv_reconstruction
git pull --ff-only
source .venv/bin/activate
python qwen_kv_gpu_bench.py --self-test
python qwen_kv_gpu_bench.py --producer-study --device cuda \
  --lengths 2048 8192 --cases 3 \
  --train-tokens 512 --dev-tokens 512 --chunk-tokens 512 \
  --tile-tokens 64 --max-tiles 72 --rank 8 \
  --repeats 3 --sample-pairs 16 --generation-steps 2 \
  --output runs/qwen_producer_hooks
```

`--max-tiles 72` selects 72 **paired** records, equivalent to 144 separate K/V tiles in the earlier study. It must cover every layer. DEV always has full coverage for continuation. `--rank` controls stored hidden features and their full predictor state; changing it requires a fresh run and output directory. `--skip-inline` disables in-forward construction for a diagnostic replay-only run and is recorded in the report. Do not pool those capture times with default inline runs.

A no-download functionality run is available:

```bash
python qwen_kv_gpu_bench.py --producer-study --tiny --device cpu \
  --cases 1 --repeats 1 --output runs/producer_tiny
```

To push reviewable GPU reports and small decoder artifacts after the run:

```bash
mkdir -p benchmark_results/qwen_producer_hooks_a4500
cp runs/qwen_producer_hooks/{results.json,results.csv,summary.md,producer_summary.json,source_manifest.json,frozen_manifest.json} \
  benchmark_results/qwen_producer_hooks_a4500/
cp -r runs/qwen_producer_hooks/calibration benchmark_results/qwen_producer_hooks_a4500/
git add benchmark_results/qwen_producer_hooks_a4500
git commit -m "Add A4500 forward producer hook results"
git push origin main
```

The separate shareable ZIP is beside the run directory. Each output directory must be fresh or empty.

## What to improve after the measurements

First examine hidden versus frozen prediction: exponent-bit reduction must repay feature and state bytes. Next inspect whether pre-RoPE or raw-projection sources reduce the combined source-plus-correction cost. Compare residual and coset variants on the same widths. Retain losing candidates in the report so the next change follows evidence.

For a promising representation, remove diagnostic CPU roundtrips, replace verbose record metadata with a compact binary descriptor, and fuse decoding into a bounded tile buffer. Measure raw transfer against packet transfer plus reconstruction under the same allocation and memory protocol. An adaptive choice between recipes must charge its selector and all required state, and its policy must be fixed before TEST. Large-context high-concurrency benefits require a subsequent serving experiment with an explicit materialization budget.

## Source map

- `qwen_kv_bench/producer_hooks.py`: actual forward stages, selected-tile stitching, in-forward callback and stage audit.
- `qwen_kv_bench/producer_models.py`: TRAIN-only hidden predictors, static control, deterministic complete state artifacts.
- `qwen_kv_bench/producer_codec.py`: paired exact packets, replay rules, matched residual/coset coding and byte charges.
- `qwen_kv_bench/producer_study.py`: independent study CLI, inline/replay comparison, timing, continuation and reports.
- `qwen_kv_bench/test_producer_*.py`: stage, exhaustive BF16-symbol, state, continuation and failure-path checks.
