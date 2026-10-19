# A4500 review: shared neural coding across context positions

**Shared neural coding now beats its matched frozen control in all three 8K cases after complete state is counted.** Its mean estimated reduction is **32.20% versus 29.51%**, a gain of **2.69 percentage points**. At 2K the state cost still makes it slightly worse. No result reaches 40%, and the CPU entropy path remains too slow for a serving claim.

Package: [results at `f8fa6e5`](https://github.com/Kibnelson/kv_reconstruction/tree/f8fa6e5b7ec9627da09d1623a0d3cf22066f0bb4/benchmark_results/qwen_neural_shared_positions_a4500). The [independent audit](../benchmark_results/qwen_neural_positions_review_20261019/validation.json), [full analysis](../benchmark_results/qwen_neural_positions_review_20261019/analysis.json) and [position summary](../benchmark_results/qwen_neural_positions_review_20261019/position_analysis.json) retain the checks and arithmetic. This is an audit of submitted artifacts and matching source, not an independent GPU rerun.

## Matched size results

The table averages the three fixtures at each length. Sample reductions use actual serialized container bytes. Net reductions extrapolate the sample density to the full context, then charge state once.

| Context | Shared packet reduction | Shared net estimate | Frozen-control net estimate | Shared minus control |
|---|---:|---:|---:|---:|
| 2,048 tokens | 34.1840% | 29.2687% | 29.6150% | **-0.3463 points** |
| 8,192 tokens | 33.4320% | 32.2031% | 29.5134% | **+2.6897 points** |

| Fixture | 2K shared net estimate | 8K shared net estimate |
|---|---:|---:|
| Pottery diary | 29.3316% | 32.2512% |
| Python traversal | 29.2110% | 32.1438% |
| Museum register | 29.2634% | 32.2144% |

Complete shared-plus-anchor state is **14,843,670 bytes (14.156 MiB)**, versus 85,451 bytes for the control. It costs about **4.9153 percentage points** of a raw 2K cache, but only **1.2288 points** at 8K. The 8K net result improves despite slightly worse packet density: longer contexts amortize this fixed state. These are single-context estimates, not measured concurrent memory use.

## Coverage and exactness

Every case measures four complete 64-token chains across all 36 layers: 144 layer/tile pairs and 36 MiB raw KV.

| Context | Selected token starts | Fraction of raw cache measured |
|---|---|---:|
| 2K | 0, 640, 1280, 1984 | 12.5% |
| 8K | 0, 2688, 5376, 8128 | 3.125% |

This addresses the earlier first-16-position-only gap. It still samples only four positions and uses repetitive synthetic fixtures. Whole-cache sizes remain estimates, and different lengths of one fixture are correlated.

The download verifies seven Git blob identities, and all **100 independent audit checks pass**. These cover six package entries, the complete 3,936,254-byte report, 168 benchmark source hashes, freeze/TRAIN provenance, state and component ledgers, 936 logged neural pair checks and 864 logged independent TEST pair decodes. The complete 128-token DEV check covers 72/72 pairs and passes two steps with finite, bit-identical logits, equal tokens and identical appended KV. Maximum recorded logit difference is zero.

The environment remains RTX A4500, Torch 2.14.0/CUDA 13.0, Transformers 5.17.0 and NumPy 2.5.2. TRAIN remains three 512-token fixtures and the shared network receives 7,000 updates at width 512.

## What changed from the prefix smoke

The declared hashes of **all 37 numerical parameter/normalizer/RoPE shards match the previous shared model**. TRAIN KV hashes, architecture and training logs also match. Manifest identities differ because load/capture timing metadata changed. Actual neural shards remain omitted from the sharing package; this comparison verifies their declarations rather than rehashing delivered tensors.

The new rate is therefore not evidence of better training. It measures larger pages and different positions using the same declared numerical predictor.

A particularly useful comparison exists on DEV: both runs use identical full 128-token input and KV hashes.

| Same full DEV prefix | Tile 16 | Tile 64 |
|---|---:|---:|
| Actual container bytes | 12,833,184 | 12,522,825 |
| Packet reduction | 32.0073% | **33.6517%** |
| Non-payload headers | 411,829 B | **103,382 B** |
| Ideal neural bits/value | 10.504246558 | 10.504246558 |

Larger pages save **310,359 bytes**, with **308,447 bytes (99.38%)** explained by header reduction. This is a controlled layout improvement on DEV, rather than an improvement in the learned probability model. It does not establish that still larger materialization units would be preferable under concurrency.

## Patterns across positions

These are neural page reductions averaged over fixtures, excluding the small outer container header/index. Control bytes are not logged separately by position, so only the per-case totals support a paired comparison with the control.

| Context | First tile | First interior | Second interior | Last tile |
|---|---:|---:|---:|---:|
| 2K | 33.3518% | **35.0056%** | 34.4993% | 33.8917% |
| 8K | 33.3518% | **34.0043%** | 33.3404% | 33.0440% |

Later 8K positions compress less well than the first interior position. Their distributions may be less well represented by TRAIN captures of only 512 tokens; that is a hypothesis to test, not a demonstrated cause. The conditional ideal rate averages **10.4162 bits/value at 2K** and **10.5397 at 8K**.

The actual arithmetic streams are within approximately **0.00052 bits/value** of the integer-CDF costs. All headers now cost approximately **0.0878 bits/value**, or **0.549 percentage points** of raw size. The easy page-framing gain has largely been collected.

At 8K, the mean complete estimate is **10.8475 bits/value**. The 40% target is 9.6, leaving **1.2475 bits/value (7.7969 percentage points)**. Eliminating all current headers and state would still leave a substantial probability-model gap. Sharing weights over more requests alone also cannot reach 40% at the measured 33.43% packet density.

## Restoration and capacity costs

Each independent 36 MiB neural sample decode takes **113.76–115.85 seconds**, versus **0.338–0.345 seconds** for its matched control under this functional protocol. CPU arithmetic decoding accounts for approximately **99.67%** of neural restore time. GPU neural inference totals only about **0.115 seconds** per sample.

The path stages 35 MiB of prerequisite KV to GPU and returns **140 MiB of FP32 probability parameters** to CPU per decode. It remains a coding reference rather than a transfer optimization. The full run takes 38.37 minutes; fitting takes only 24.18 seconds.

The two-raw-tile dependency metric increases to **512 KiB**, with depth 35. It excludes returned cache tensors, staged inputs, probabilities, coder workspace and model state. Measured GPU allocation peaks above the pre-prepare baseline are approximately 19.69 MiB, including state preparation and functional work. They are not an isolated workspace bound or a concurrency/admission result.

As before, packets reopen from in-memory serialized bytes, and the delivered package omits neural weights and raw KV. Exact restart on another software/device stack and file-backed separate-process restoration have not been independently established.

## What to improve next

The evidence supports keeping the shared model and testing specific additions rather than enlarging a per-layer model collection:

1. **Longer TRAIN context at the same model and update budget.** Compare 512- and 2,048-token TRAIN captures while holding sampled evaluation positions fixed. This directly tests the observed positional drift. Use DEV for decisions and reserve fresh TEST material for a resulting claim.
2. **Prefill-derived features as probability information.** Test actual normalized-hidden features as inputs to the distribution predictor, rather than demanding a point prediction with narrow full-word corrections. Charge the retained features, all additional state and the extraction work. A produced-KV-only page calibration control can show whether upstream access adds information beyond cheap location/scale summaries.
3. **Decode current K before predicting V's distribution.** This supplies exact available information without a retained per-token feature payload, at the cost of another prediction stage and possibly more state. It needs its own matched stream, rather than adding percentages from separate methods.
4. **A parallel entropy implementation for speed.** Preserve the distributions initially so that coding throughput can be isolated from model quality. Then measure discrete-probability reproducibility, all substream framing and actual peak workspace. Merely accelerating the network cannot remove the current CPU bottleneck.

The older exponent/sign study reports 33.39% mean state-inclusive estimates across six cases, using different sampled addresses and framing. That remains useful context, not a direct ranking against these new 32.20% 8K estimates. A clean hybrid or head-to-head comparison requires the same KV pages, independent state accounting and actual combined bitstreams. No additional saving or novelty is established for the proposals above yet.

## Reproduce the report review

Both commands use the standard library and read the packaged gzip directly. The audit requires benchmark sources matching the run's recorded hashes; `--source-root` can point to a historical checkout after later source changes.

```bash
python scripts/review_neural_results.py \
  benchmark_results/qwen_neural_shared_positions_a4500 \
  --output benchmark_results/qwen_neural_positions_review_20261019

python scripts/summarize_neural_positions.py \
  benchmark_results/qwen_neural_shared_positions_a4500 \
  --output benchmark_results/qwen_neural_positions_review_20261019
```
