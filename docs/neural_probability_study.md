# Fifth experiment: learned probabilities with an actual exact coder

The supplied Approach 2 code suggests a useful next experiment: use the already reconstructed previous layer's K/V to predict a probability distribution for each current-layer BF16 word. An entropy decoder then recovers the exact word from stored bits. The network supplies probabilities; it does not replace the information carried by the encoded stream.

This experiment tests that idea on actual Qwen captures, including complete decoder state and a real CPU arithmetic stream. It does not establish a Qwen 40% saving, cheap GPU restoration, independent decoding of arbitrary layers, or a novelty claim.

The [A4500 smoke review](qwen_neural_gpu_review.md) now confirms exact learned pages and complete 128-token DEV continuation in both modes. Shared neural has 30.39% sampled packet savings and 25.47% estimated net savings with state; its matched control estimates 26.04%. TEST sizing covers only the first 16 positions, and CPU arithmetic decoding dominates restoration. The review compares all current conditional tracks, explains the sampling limits and supplies a shared-only position-coverage command before increasing training.

The subsequent [position-coverage review](qwen_neural_positions_review.md) measures four 64-token chains per case. Shared neural estimates 29.27% net savings at 2K versus 29.62% for its matched control, and **32.20% at 8K versus 29.51%**. These rates include complete state but still extrapolate sampled pages. The same declared numerical model and a controlled DEV layout comparison separate coverage/header changes from learning improvements. All 100 artifact audit checks pass; no 40% or serving-speed result is established.

## Audit of the supplied archive

Archive SHA256: `3af509d9f6d181ba6f097f7746f2e29e68d74153ed4e6ae55ece9c2a6866a5cc`.

The reviewed seven files are `smol_dump_act.py`, `approach2.py`, `approach2_prob.py`, `approach2_check.py`, `approach2_improve.py`, `approach2_results.txt`, and `improve_results.txt`.

| Source | What it establishes | What it does not establish |
|---|---|---|
| `smol_dump_act.py` | A NumPy forward using dequantized SmolLM2-135M Q4_1 GGUF weights, with BF16 K/V word dumps and perplexity diagnostics. | An instrumented Hugging Face Qwen run or a serving implementation. |
| `approach2.py` | Point-prediction, activation-code and previous-layer controls; attention versus magnitude-error objectives; checked coset reconstruction. | A byte-complete entropy-coded cache or full decoder-state accounting. |
| `approach2_prob.py` | A learned logistic center and scale per value; held-out sums of exact BF16 interval negative log probabilities. | An emitted arithmetic/rANS stream, integer CDF rounding costs, framing or measured entropy-decoder throughput. |
| `approach2_check.py` | Finite BF16 interval tiling and distribution-normalization checks, plus attention-trained and static controls. | A full serialized encode/decode roundtrip for the probability model. |
| `approach2_improve.py` | Ablations for more training, network width, mixtures, additional previous layers and decoding K before conditioning V. | A single complete serving configuration whose size, state and runtime were measured together. |

The central scoring function returns `-sum(log2(P))`. Here `P` integrates a logistic distribution, or a mixture, over the target word's BF16 rounding interval and adds a small uniform escape component. This is stronger evidence than reporting a floating-point prediction error. It remains an **ideal probability cost**, rather than the length of an actual finite-precision coded stream. The original scripts do not serialize and charge the MLP parameters and normalization statistics in their reported savings.

The archive's data split uses training documents, a separate `tables` document for static coding tables where needed, and two test documents. That separation is useful. The many reported ablations still require a fresh held-out evaluation before treating a configuration selected from these results as a general winner.

### Keep the reported scopes separate

All numbers below are bits per BF16 value, before complete decoder state and a real cache container.

| Evidence | Text | Code | Correct interpretation |
|---|---:|---:|---|
| Cheap model, logged conditional layers 1–29 | 9.452448 | 9.265414 | Mean of the 29 individually reported layer results. |
| Illustrative 30-layer aggregate assuming a 10.49-bit layer-zero anchor | about 9.49 | about 9.31 | Accounting estimate including an independently coded layer-zero anchor; approximately 40.7% and 41.8% ideal payload savings. |
| Stronger configuration at eight sampled layers | 9.161625 | 8.771125 | Unweighted mean of those eight reported layers only. |
| Author-reported all-layer estimate from the eight-layer study | about 9.26 | about 8.90 | An extrapolation, not a measured complete 30-layer cache. Its exact extrapolation rule was not supplied, so this aggregate was not independently reproduced. |
| Large configuration at layers 2, 12 and 24 | 9.016333 | 8.622000 | Mean of three measured layer scores only; approximately 43.6% and 46.1% ideal savings at those layers. |

The illustrative 30-layer cheap aggregate assumes the same 10.49-bit anchor for text and code. The archive does not provide a measured first-layer probability score or a complete serialized first-layer anchor; using a different baseline changes that estimate. The stronger eight-layer and large three-layer results must not be presented as whole-cache measured results. In particular, the roughly 42–46% figures are promising probability-model evidence, not a completed storage or migration benchmark.

## Why complete state changes the conclusion

For the cheap single-logistic previous-layer model, let `C` be the number of K channels, also the number of V channels, and `W` the hidden width. The MLP has widths `2C -> W -> W -> 4C`: it emits a center and a scale for every K/V value. Including biases, its parameter count is:

`6*C*W + W*W + 2*W + 4*C`.

| Illustration | Network parameters | FP32 network bytes | Raw BF16 KV at 8,192 tokens |
|---|---:|---:|---:|
| SmolLM2, `C=192`, `W=512`, 29 conditional layers | 24,759,040 | 99,036,160 bytes = 94.448 MiB | 180 MiB across all 30 layers |
| Qwen3-4B, `C=1024`, `W=512`, 35 conditional layers | `35 * 3,412,992` = 119,454,720 | 477,818,880 bytes = 455.684 MiB | 1,152 MiB across all 36 layers |

These network-only figures exclude normalization, RoPE parameters and serialization metadata. For the cheap Smol model, even the illustrative ideal 9.49-bit payload is about 106.8 MiB. Adding 94.448 MiB of FP32 networks already exceeds the 180 MiB raw cache. The code-document estimate has the same problem. Thus the archive's payload savings cannot be interpreted as savings including complete state.

The Qwen row is an architecture calculation, **not a Qwen compression result**. A 40% total saving at 8K permits at most 691.2 MiB for payload, framing and all charged state together. A 455.684 MiB network collection consumes much of that budget before any symbol is stored.

If predictor weights are later pooled across concurrent requests, report that separately with the actual number of requests and the shared state charged once. This study's conservative per-cache ledger charges complete decoder state once per cache. Neither accounting convention makes state free.

## The probability-model experiment

For target layer `l >= 1`, concatenate the exact previous layer's K and V for the same token rows. A frozen network predicts a logistic mean and log scale for each current K/V value. Token positions supply the known RoPE coordinates. The probability mass of each BF16 rounding interval determines the entropy coding probabilities.

The proposed information flow is:

1. Prefill the actual target model and retain the captured BF16 words for the experiment.
2. Fit predictors and all normalizers using the three explicit TRAIN captures only. Save complete model artifacts and freeze their identities before DEV or TEST encoding.
3. Encode layer zero of each selected token tile with an independent exact anchor codec.
4. Actually decode that anchor. Use its reconstructed words to generate layer one's probabilities, encode layer one, and actually decode it.
5. Continue through every layer in that tile. Every conditioning input comes from a decoded prerequisite, rather than the original capture.
6. Serialize an indexed cache container to bytes and reopen those bytes for reconstruction. The decoder receives the container, frozen state and its execution contract; it does not receive the original target words or precomputed per-value predictions. Current container roundtrips happen in memory, rather than through file-backed, separate-process replay.

The first port follows the cheap source configuration: `inputs=prev1`, one logistic component, width 512, no stored activation code, and no same-layer K-to-V network. More mixture components, `prev2` and `vk=1` remain separate possible experiments rather than being silently included.

The implementation also discloses its numerical differences from the archive: Torch replaces JAX; Qwen's split-half RoPE replaces the Smol GGUF adjacent-pair layout; and a bounded log-scale guard is part of both fitting and inference. These changes require their own held-out measurements. The model/runtime identity and prediction checks prevent silently decoding with different probabilities.

### Required ablation: per-layer versus shared

| Mode | Decoder network | What the comparison answers |
|---|---|---|
| `per_layer` | One independently fitted MLP for each layer after layer zero. | Can specialized distributions reduce the actual stream enough to justify the entire collection of weights? |
| `shared` | One MLP with the known target-layer one-hot input; per-layer normalizers remain charged. | Does sharing reduce complete state enough to offset weaker specialization? |

The shared mode is a new ablation, not a result reproduced from the archive. At the illustrative Qwen width, its network has 3,430,912 parameters including the 35 layer indicators, about 13.088 MiB in FP32, before normalizers and metadata. Its quality is unknown until measured. Report both modes even when one loses; do not select the better TEST result and present it as a preselected configuration.

The runner matches the **aggregate minibatch update budget** by default. `--steps 2000` gives each of Qwen's 35 per-layer networks 2,000 updates, or 70,000 updates in total; the shared network therefore also receives 70,000 updates. Shared training randomly chooses a target layer on each update, and records its realized per-layer counts rather than guaranteeing exactly 2,000 updates for every layer. `--shared-steps` explicitly overrides the shared total. Equal update counts do not guarantee equal training time or FLOPs. The lower-level `neural_probability.fit` API still applies its `steps` budget per parameter group; the runner invokes the modes separately to implement this aggregate-budget rule.

## Actual bytes and execution locations

Neural inference runs on the selected device. The first real entropy coder is a **CPU arithmetic-coding reference**. GPU probabilities must therefore be transferred to CPU, and prerequisite word tiles must be staged for GPU prediction. That separation is deliberate in the measurements and must remain visible in any performance claim.

For CUDA execution, the two FP32 outputs, mean and log scale, cost **eight outgoing bytes per BF16 value**, versus two bytes for the original raw word: four times the raw word size before the previous-layer KV input staging is counted. This CPU-reference pipeline is currently a representation and coding experiment, not a transfer optimization.

Report all of the following:

- Ideal continuous probability bits, integer-CDF probability bits, and actual emitted stream bytes as separate quantities.
- Actual layer-zero anchors, page descriptors, checksums, offset indexes and the reopened outer container.
- Every model shard, manifest, normalizer and RoPE parameter, plus resident decoder tensors. Charge the reported conservative maximum of serialized and resident complete state once.
- Predictor input staging, GPU neural inference, probability-output copies, CPU entropy encoding, CPU entropy decoding and full reconstruction separately.
- Exact word reconstruction and complete DEV continuation checks, including finite and bitwise equal logits, identical generated tokens and appended KV.

An arithmetic stream can recover original bits from approximate statistical probabilities. It nevertheless requires encoder and decoder to use the same discrete CDF. A valid continuous distribution alone does not establish that contract, and a network-only timing is not a full reconstruction timing.

The standalone entropy coder supports all 65,536 BF16 word patterns, including nonfinite patterns. The real neural predictor has a narrower contract: TRAIN tensors and previous-layer KV supplied to prediction must be finite. Standalone coder coverage therefore does not imply that an arbitrary nonfinite cross-layer cache can pass through the complete neural chain.

Complete model manifests and parameter shards remain under the run's `models/` directory as `.json` and `.npkv` files. The standard results-sharing bundle omits these potentially large model files, while preserving their manifest entries, byte counts and SHA256 identities. Omitting weights from that sharing bundle does not remove their full cost from the benchmark's state ledger. Keep the local model artifacts to reproduce decoding; a shared report alone is not a standalone decoder package.

## Partial decoding and capacity

This representation changes the dependency structure. A token tile is an independent indexed chain, but a page at layer `l` needs that tile's decoded layers `0..l-1`. It cannot be materialized in isolation from its prerequisites. Different tiles can proceed separately; layer dependencies within one tile remain ordered.

The initial implementation can retain a bounded pair of prerequisite/output tiles while walking a chain, and report the measured dependency depth and working bytes. It still must account for the final attention cache or buffer into which the requested values are materialized. The benchmark retains full model KV during capture and reconstructs DEV caches for correctness, so it does not demonstrate higher concurrency, a compact attention kernel, lower TTFT/TPOT, or admission under memory pressure.

The sample budget applies to **aligned all-layer tile chains**. For a 36-layer model, `--max-pairs 72` admits two complete tile chains, rather than 72 unrelated layer/token records. Full-cache estimates from those selected addresses remain estimates. DEV correctness uses complete DEV coverage; TEST addresses, model hyperparameters and candidate modes are fixed before observing held-out results.

Headline full-cache estimates scale the **complete sampled container/raw-byte density**, including its fixed outer header and offset index:

`estimated_full_container_bytes = sampled_container_bytes * full_raw_bytes / sampled_raw_bytes`.

Complete charged state is then added once, without scaling it. Actual selected containers are serialized and reopened in memory, but this extrapolation is not a serialized full cache. Scaling a fixed header can overcount its full-cache contribution; unequal tail lengths and unrepresentative selected tiles can bias the estimated density in either direction. Treat these figures as provisional until full coverage is measured, and distinguish them from actual selected-container bytes.

## Commands for the NUC

After pulling the code and activating the existing environment, start with the smaller actual-model protocol check:

```bash
python qwen_kv_gpu_bench.py --neural-probability-study \
  --device cuda --width 512 --steps 200 \
  --train-tokens 512 --dev-tokens 128 \
  --lengths 2048 --cases 1 \
  --tile-tokens 16 --max-pairs 36 \
  --modes per_layer shared \
  --output runs/qwen_neural_probability_smoke_v2
```

Then run the planned fifth experiment:

```bash
python qwen_kv_gpu_bench.py --neural-probability-study \
  --device cuda --width 512 --steps 2000 \
  --train-tokens 2048 --dev-tokens 128 \
  --lengths 2048 8192 --cases 3 \
  --tile-tokens 64 --max-pairs 72 \
  --modes per_layer shared \
  --output runs/qwen_neural_probability
```

Use a fresh output directory for each run. Actual CPU arithmetic coding and all-layer prerequisite reconstruction may be slow; these commands carry no runtime promise. The smoke configuration checks the protocol and exactness, not whether shorter training achieves the larger experiment's coding quality.

The earlier smoke command used `--dev-tokens 16`, and the earlier full command/default used 64. Both are below the pinned Qwen DEV fixture's 71 fixed template/user tokens. The corrected commands/default use **128 complete prefix tokens**, without removing the question, needle or assistant marker. Before loading weights or capturing TRAIN, the runner now checks every requested TRAIN/DEV/TEST budget with the real pinned tokenizer. Invalid explicit sizes fail with the fixture and argument named; they are not silently enlarged. `--tiny` continues to use synthetic token IDs and supports its small budgets. Reports record the preflight prompt identities and lengths. The earlier failed run's fitted artifacts are retained, but this change does not implement model resumption across runs.

The [fix validation](../benchmark_results/neural_dev_budget_fix_validation_20261019.json) records 47 relevant tests on Transformers 4.57.6, 18 on 5.20.1, 12 packaging/build checks, and a completed standalone tiny-model run with exact caches and two identical continuation steps for both neural modes. These are CPU functional checks; pretrained Qwen CUDA results still require the command above.

With the default shared-budget rule, the smoke command also trains the shared network for 7,000 updates, matching 35 per-layer networks at 200 each. A deliberately shorter shared trial can set `--shared-steps`, but its report must disclose the changed training budget.

For comparison with the earlier 13 conditional candidates, use each run's matched frozen control and show complete state, payload, restoration time and sampling scope separately. The neural run samples complete all-layer tile chains; the conditional run samples layer/tile pairs across its cache. Equal `--max-pairs 72` and `--max-tiles 72` therefore do not identify the same addresses. Their aggregate estimates can be reviewed together, but they are not a paired codec ablation. Neural CPU-reference wall timings also differ from the conditional study's repeated device microbenchmarks.

After a completed run, package its reports for review:

```bash
python scripts/package_results.py runs/qwen_neural_probability_smoke_v2 \
  --destination benchmark_results/qwen_neural_probability_smoke_v2_a4500
git add -f benchmark_results/qwen_neural_probability_smoke_v2_a4500
git commit -m "Add A4500 neural probability smoke results"
git push origin main
```

For the larger run, substitute `qwen_neural_probability` in both paths. Preserve the local `models/` directory; the reports package records its costs and hashes while omitting the large neural weights.

## Evidence needed before retaining the approach

A useful result must show actual reopened cache bytes plus complete state against the same raw BF16 denominator, exact held-out reconstruction, and a credible restoration cost. A payload-only 40% result, an ideal bit sum, an interpolated layer score or a small random-model roundtrip does not satisfy that requirement.

If actual probability coding succeeds but complete state loses, the shared-network ablation identifies whether state pooling is enough. If shared state wins but CPU entropy decoding dominates, retain the representation finding and separately investigate a parallel GPU-compatible discrete coder. If cross-layer restoration is too costly for the intended capacity-driven scheduling, the required change is a measured dependency/reset design, with every new anchor and index charged.
