# Exact KV reconstruction experiments

Benchmarks for storing a decoder recipe plus exact corrections instead of raw BF16 KV tensors. The default model is **Qwen/Qwen3-4B-Instruct-2507**, pinned to commit `cdbee75f17c01a7cc42f958dc650907174af0554`.

The [prefill metadata and exact coset study](docs/prefill_coset_study.md) adds `--prefill-coset-study`: small actual-hidden codes condition omitted mantissa bit predictions, with verified 31-bit syndrome groups, matched error-position controls and raw escapes. It records real packet/state bytes, DEV width selection, group error histograms and independent restoration. The shareable calibration includes fitted codec parameters. Neural work uses the selected device; initial syndrome/field coding runs on CPU. Local tiny-model validation does not establish pretrained Qwen savings or GPU codec performance.

The [first A4500 coset review](docs/qwen_prefill_coset_gpu_review.md) passes 4,052 audit checks but finds no producer-code payload gain: every measured mantissa plane remains raw. Hidden code estimates 28.91% net savings against 30.50% for the independent field baseline. Detailed error, ablation and optimizer-coverage logs separate the current predictor failure from the still-open question of better prefill-informed codes.

The [training study](docs/prefill_coset_training_study.md) adds `--prefill-coset-training-study`: a prespecified 2×2 comparison of random/balanced channel coverage and short/long training at fixed rank/width. Its [A4500 review](docs/qwen_coset_training_gpu_review.md) passes 12,803 package checks after correcting a cross-arm load-time identity comparison. Balanced long training covers every channel overall, but all four arms retain identical packet lengths and zero short groups. Hidden metadata has weak predictive information; its screened probability benefit remains much smaller than its charged code cost.

The [compact-metadata study](docs/compact_prefill_metadata_opportunities.md) audits that cost and proposes matched tests of decoder-known pooled context, a two-bit actual-prefill innovation, page-shared metadata and exact K-to-V conditioning. It separates the current 16-byte side code from 16-bit KV words, identifies normalization state as a major cost, and records proposed payload budgets without claiming new encoded savings.

The [compact comparison runner](docs/compact_prefill_study.md) implements `--compact-prefill-study`: fourteen TRAIN-fitted variants compare pooled known fields, one/two/four-bit actual-prefill hints, page sharing, exact K-to-V conditioning and high-bit-first grouped refinement. Complete exact packets and small model state are charged, with DEV selection frozen before fresh TEST comparisons. Network work uses the selected device and arithmetic packet coding runs on CPU. Use the A4500 commands and results packager in the guide; tiny-model validation is not a pretrained savings result.

The repository contains the readable implementations, tests, and a generated single-file runner. It compares exponent packing, field/entropy coding, learned dictionaries, fixed controls, temporal predictors, E8/coset residuals, hybrids, binary residual predictors, and prefill-derived recipes. `--preset all --grid` exposes **104 CPU reference variants and 31 native tensor variants**. The native packets use their own grouped bitpacking format; original Huffman/Deflate codecs still execute on CPU.

The new [prefill recipe experiment](docs/prefill_recipes.md) derives per-channel centers, affine trends or an E8 template from each KV tile, with exact coset and matched residual controls. `--prefill-recipes` also constructs and verifies recipes directly from device KV between prefill chunks. Full raw model KV remains resident.

`--native-preset insights` adds a controlled **33-candidate sweep**: predictor-aligned layouts, prefill exponent recipes with exact escape streams, and dictionaries trained on the errors left by affine prediction. It records matched per-record increments, complete dictionary costs, and a size-only bound on adaptive selection. See the [GPU commands and interpretation guide](docs/prefill_recipes.md#matched-incremental-experiments). Prefill metadata is the design thread; any research contribution depends on measured improvements over the controls and prior work.

The [local full-record diagnostic](benchmark_results/recipe_insights_review_20261019/REVIEW.md) checks all 33 candidates on a reused small-model cache. Prefill exponent metadata saves 28.03% including all metadata, versus 22.77% for its fixed-bias control. These CPU size results guide the next Qwen GPU run; they do not establish GPU performance or serving gains.

The subsequent [Qwen A4500 review](docs/qwen_gpu_insights.md) finds 29.55–29.75% estimated savings across six TEST cases. It identifies metadata/index costs, validates a compact-index transformation, and separates measured results from proposed formats and the research needed for bounded GPU materialization.

The [forward producer study](docs/producer_forward_study.md) adds **nine matched recipes** using real normalized hidden states, raw K projections and pre-RoPE keys. `--producer-study` builds and checks packets inside attention forward, then separately times post-capture replay. It tests whether producer information improves savings and restore speed after charging features, predictor state, replay sources and exact corrections. Tiny Qwen validation provides local functional coverage; the A4500 review below reports the pretrained-model measurements.

The [A4500 producer-hook review](docs/qwen_producer_gpu_review.md) now validates all nine recipes on real Qwen KV. The frozen exponent control wins this sweep with 29.84% mean estimated savings and the lowest observed restore cost. Both upstream K replay paths need zero correction payload on all measured tiles, while hidden-feature predictors lose even before feature/state costs. The review records the evidence, metadata costs and next controlled combinations; no serving speedup or novelty is inferred.

The [40% frontier sweep](docs/frontier_40_study.md) combines cost-trained channel bases, tiny per-head adjustments, verified zero-replay modes, compact indices, and adaptive primary/lower-bit coding in **16 matched candidates**. Run `--producer-study --producer-preset frontier`. Its state-inclusive 9.6-bit ledger separates actual codec results from entropy diagnostics and CPU size controls. The 40% goal remains unconfirmed on pretrained Qwen KV.

The [A4500 frontier review](docs/qwen_frontier_gpu_review.md) validates all 16 candidates and finds **30.76% mean estimated savings**, including complete state, versus 29.84% for the matched frozen control. Pre-RoPE source coding contributes 0.37 percentage points; compact indices supply the largest gain. All attempted lower-bit planes fall back, and expensive head-shift search adds little. The review quantifies the remaining 1.48-bit gap to 40% and gives separate next experiments for speed and larger savings.

The [four-idea conditional study](docs/conditional_40_study.md) adds actual device Huffman exponent coding, separate sign coding, joint KV anchors, and actual-prefill joint producer features. `--producer-study --producer-preset conditional` compares **13 candidates**, including raw-transfer and fast pre-RoPE controls. The [A4500 conditional review](docs/qwen_conditional_gpu_review.md) verifies all 91 rows and finds **33.39% mean estimated savings** for exponent Huffman plus sign coding, or **32.90%** for exponent-only at roughly half the decode cost. Every learned anchor/latent packet falls back to tier coding. The review accounts for the remaining 1.06-bit gap to 40%, predictor correction saturation, large eager decoder workspace and pending probability-model results.

The [fifth learned-probability study](docs/neural_probability_study.md) adapts the supplied previous-layer neural predictor into an actual lossless arithmetic-coded format. `--neural-probability-study` compares per-layer networks with a shared network and a matched frozen-exponent control. Neural inference uses the selected device; entropy coding is a separately timed CPU reference. It counts all weights/normalizers, reopens complete indexed packets in memory, and restores each layer from the previous layer's decoded KV. The supplied 42–46% figures remain ideal probability rates on SmolLM, excluding weights. The [A4500 neural review](docs/qwen_neural_gpu_review.md) now confirms exact Qwen learned pages and full 128-token DEV continuation for both modes. Shared neural saves 30.39% on the measured packet sample, but only 25.47% in the state-inclusive 2K estimate, versus 26.04% for its matched control. Only the first 16 TEST positions were sized, and CPU arithmetic decoding takes 99.4% of restore time. These produced-KV results complement the prefill-hidden-metadata tracks; they establish neither 40% savings nor a serving gain.

The [novelty assessment](docs/novelty_assessment.md) compares all current representation families, their measured evidence and primary sources. Exponent/sign coding is the strongest reported simple codec; shared neural coding supplies a matched 8K learned improvement. Most broad ingredients are established. The actual-producer side-code hypothesis now has an A4500 smoke result: independent exact restoration works, but the hidden-derived code has not improved savings over its no-code and KV controls. Any contribution must come from a particular mechanism and measured advantage, with original-BF16 exactness and capacity-bounded serving assessed separately.

The [prefill side-code probability study](docs/prefill_side_probability_study.md) implements that experiment. `--prefill-side-study` compares four shared probability models: no retained code, a learned code from final KV, a learned code from inverse-RoPE-aligned final KV, and a learned code from actual prefill hidden states. Codes are emitted inside attention callbacks; exact entropy coding follows capture. Each page restores from its own transmitted code and frozen model, without previous-layer KV. Frozen exponent and previous-layer neural controls use the same sampled addresses. The [A4500 side-code review](docs/qwen_prefill_side_gpu_review.md) passes 251 audit checks and finds **32.20% estimated net savings for no-code versus 31.43% for hidden code** in one 2K smoke case. Hidden's probability rate also loses before side/state cost; all forced learned pages remain exact. The next controlled test increases training duration at fixed width/rank. CPU arithmetic still dominates restoration, and this is not a serving result.

The [shared-neural context-position review](docs/qwen_neural_positions_review.md) now checks four positions in each of six TEST cases. At 8K, shared neural estimates **32.20% net reduction versus 29.51%** for its matched frozen control; at 2K it still loses slightly after state. Declared numerical shards match the prefix smoke, and a same-DEV comparison attributes almost all tile-size improvement to smaller headers. All 100 audit checks pass. The remaining 40% gap is primarily a probability-model problem; CPU arithmetic still takes 99.67% of restore time.

For result commits, use `python scripts/package_results.py RUN --destination benchmark_results/NAME`. It preserves the full report as verified `results.json.gz`, alongside the smaller reports and decoder state. Large frontier reports can exceed GitHub's raw-file limit. The [sharing and repair instructions](docs/frontier_40_study.md#repair-a-rejected-oversized-report-commit) explain how to amend a rejected unpublished commit.

## Clone and install

```bash
git clone https://github.com/Kibnelson/kv_reconstruction.git
cd kv_reconstruction
python3 -m venv .venv
source .venv/bin/activate
```

For the **NVIDIA RTX A4500 NUC**, install the complete CUDA environment before running checks:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -c "import torch; print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
python qwen_kv_gpu_bench.py --doctor --device cuda
python qwen_kv_gpu_bench.py --self-test
```

**`requirements.txt` alone does not install PyTorch.** It contains the backend-independent packages. `requirements-cuda.txt` includes those packages and the NVIDIA CUDA build of Torch. If you already created the virtual environment and installed `requirements.txt`, keep that environment active and run the CUDA install command above; no model download is needed for the checks.

The NVIDIA requirements select **Torch 2.11.0 with the CUDA 12.8 runtime**, a build available from the official CUDA wheel index. The NUC's driver 580 supports this runtime: `nvidia-smi` showing CUDA 13.0 describes driver capability and does not require a CUDA 13.0 Torch wheel. The earlier A4500 smoke report is in `benchmark_results/qwen_smoke_a4500_20261019_081452`; the new prefill methods require a fresh target run.

For Intel XPU or CPU, install the matching PyTorch build from the [official selector](https://pytorch.org/get-started/locally/), then run `python -m pip install -r requirements.txt`. A CUDA build is required for the NVIDIA benchmark; a CPU-only Torch installation cannot use the A4500.

The requirements file pins Transformers 4.57.6 as a reproducible baseline. **An existing Transformers 5 installation does not need to be downgraded:** cache capture, exact restoration, and continued generation are also tested with 5.20.1. Use your working environment directly and run `--doctor` to record its versions. CPU functional checks do not establish GPU performance.

## Run on the NVIDIA NUC

First run a small comparison on the RTX A4500:

```bash
python qwen_kv_gpu_bench.py --device cuda --preset smoke \
  --lengths 2048 --cases 1 --output runs/qwen_smoke
```

Then run the wider comparison:

```bash
python qwen_kv_gpu_bench.py --device cuda --preset all --grid \
  --lengths 4096 16384 --cases 3 --measure-io --output runs/qwen_all
```

The first run downloads the public model weights if they are not already cached. Existing Hugging Face weight caches are reused. Every output directory must be fresh or empty. Share the generated **`runs/qwen_all_results.zip`** to review the results. Reports include errors and unsupported methods as well as successful measurements.

Default comparisons measure 72 evenly spaced records per test cache. **Full-cache size totals are estimates when sampling.** Add `--full-cache` to encode and check every record; Python CPU sweeps can take much longer. Repeated decode, transfer, and buffered I/O timings use a bounded sample even in full-cache mode. The outer cache header/index are calculated byte charges; a complete indexed cache container is not persisted.

```bash
python qwen_kv_gpu_bench.py --list-methods --preset all --grid
python qwen_kv_gpu_bench.py --instructions
```

For a supported Intel GPU use `--device xpu`. Auto selection prefers CUDA, then XPU, and never silently falls back to CPU. Explicit `--device cpu` is available for functional checks. Native E8 and binary predictors currently require float64 operations, whose support varies on XPU devices.

## Get updates

```bash
cd ~/kv_reconstruction
git pull --ff-only
python qwen_kv_gpu_bench.py --device cuda --preset smoke \
  --lengths 2048 --cases 1 --output runs/qwen_smoke_after_update
```

This repository is the source for subsequent code changes. The standalone file embeds a source snapshot and can also be copied to another machine without a checkout. For development, run the checked-out modules directly:

```bash
python -m qwen_kv_bench.runner --doctor
python -m qwen_kv_bench.runner --device cuda --preset smoke \
  --lengths 2048 --cases 1 --output runs/source_smoke
```

After editing source, regenerate the portable file:

```bash
python scripts/build_standalone.py
python scripts/build_standalone.py --check
python qwen_kv_gpu_bench.py --self-test
```

Commit source changes and the regenerated runner together. Model weights, raw KV, local caches, environments, and run outputs are ignored by git.

## Cache compatibility fix

The original runner assumed `to_legacy_cache()`, which newer Transformers releases removed. The runtime now reads `cache.layers[i].keys` and `.values`, while preserving older cache interfaces. It rejects missing/uninitialized layers, retains explicit head-dimension checks, and records the cache class and dependency versions. The reported error after the first Qwen training capture is reproduced and covered by regression tests. It is separate from Hugging Face cache permission warnings.

Local validation includes full functional suites on Transformers **4.57.6 and 5.20.1**, plus random BF16 Qwen3 chunked prefill and exact cache restoration. A separate test uses explicit `head_dim=128` even though `hidden_size / query_heads` differs, then checks three continued decode steps for identical logits, tokens, and appended KV. No model weights are needed for these synthetic model tests. CUDA/XPU execution must be checked on the target hardware.

## Experimental scope

The protocol fits dictionaries only on three training captures, chooses representative generation checks on a separate development capture, then freezes settings before fresh test captures. It counts complete correction/metadata bytes and decoder state. Equality is checked on BF16 words; generation comparisons also require finite logits.

This is a representation and data-movement benchmark. It does not implement compact-resident attention, a high-concurrency scheduler, or partial-capacity admission. GPU native implementations are eager tensor reference code, not fused production kernels. Buffered I/O may hit the OS page cache. Compression ratios alone do not establish latency, throughput, or serving-capacity improvements. These experiments do not reproduce AQUA-KV, PackKV, or the published RaBitQ ANN system.

See [the detailed benchmark guide](qwen_kv_bench/README.md) and [the method inventory](docs/methods.md).

## Source map

| Directory | Role |
|---|---|
| `qwen_kv_bench/` | Model capture, fresh calibration, registry, GPU packets, measurements, reports, tests |
| `kv_smoke/` | Initial exact BF16 block and coset controls |
| `kv_research/` | Learned dictionaries and original E8 representations |
| `kv_stacked/` | Field/Huffman coding and complete-record wrappers |
| `kv_coset_fields/` | Lower-field, temporal, and conditioned predictor controls |
| `kv_residual/` | Exact residual coding, learned/E8 combinations |
| `kv_combinations/` | Hybrid group and tile selection |
| `kv_binary/` | Binary residual predictors and identity/dense/Hadamard controls |
| `scripts/` | Reproducible standalone build tooling |
| `tests/` | Packaging regression checks |

Earlier experiment-driver files are retained as source for review. Some historical drivers expect datasets/checkpoints that are not included; use the current `qwen_kv_bench` entrypoints above for the portable experiment.
