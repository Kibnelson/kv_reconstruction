# Matched Qwen GPU comparison

This first experiment isolates stronger decoder-available layer inputs while
holding the existing exact GPU entropy coder fixed. It does not implement or
claim the team's 51.4% strong-and-fast result. That number is a SmolLM2 coding
cost estimate, with a different target, calibration, state and timing scope.

## Matched arms

| Arm | Mean context | Width | GPU coder |
|---|---|---|---|
| raw_bf16 | none | none | exact word copy |
| block32_A | previous layer + earlier current-layer groups | fixed | cached T1 binary rANS |
| block32_B | same as A | causal context, strength 4 | same |
| previous32_B | previous layer only | same B rule | same |
| aqua_style32_B | K from previous K; V from previous V and decoded current K | same B rule | same |
| strong_layers32_B | layers 0, l−2, l−1 + earlier current groups | same B rule | same |
| strong_layers64_B | same layer inputs, 64-channel groups | same B rule | same |
| splitzip_native | native calibrated 16-entry exponent codebook | native | unmodified public Triton implementation |
| nvcomp_bitcomp | raw BF16 words through int16 view | native lossless | nvCOMP CUDA |
| nvcomp_cascaded | same source and view | native lossless | nvCOMP CUDA |

The first few strong-layer contexts are empty, [0], [0,1], [0,1,2],
[0,2,3]. Coincident inputs are included once. Every token is independently
restorable from its packet and frozen tables; no earlier-token cache or
another model forward pass is a decoder input. The first layer is encoded and
charged, not excluded from rates. All predictors are refitted on the same TRAIN
rows. A and B share exactly the same Block32 base.

The previous-only and AQUA-style labels describe their mean predictors. All B
arms still use earlier-group error context for width adaptation. This holds the
width rule fixed while comparing the mean dependency structure.

AQUA-style here is an exact-coder predictor ablation. A separate comparison
now adds released AQUA-KV cache/Quanto residual method bodies (2/4 bits) and
the native pinned OpenLake ExANS v2 CUDA codec. See
[the four-family run](qwen_native_baselines.md) for integration changes,
lossy quality metrics, fixed-block padding and the A4500 command.
Speculative FP8 model-copy coding remains pending. No CPU cost estimate can
populate a GPU result row.

## Model, data and presets

The target is **Qwen/Qwen3-4B-Instruct-2507**, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, with original BF16 weights, SDPA,
batch one and raw dataset tokens. Prefill runs the full model; the pilot retains
only its first four layers. No chat template, weight quantization or automatic
CPU offload is applied.

| Setting | Pilot, default | Full |
|---|---:|---:|
| Measured KV layers | first 4 of 36 | all 36 |
| TRAIN captures per track | 4 × 1,024 tokens | 128 × 1,024 tokens |
| Rows selected for each predictor fit | 2,048 | 8,192 |
| Separate scale capture | 1,024 tokens | 8,192 tokens |
| Held-out TEST contexts per track | 2 × 2,048 tokens | 8 × 8,192 tokens |
| Default measured request sizes | 32, 128 tokens | 32, 128 tokens |
| Request positions | context beginning and end | context beginning and end |
| Warmup / measured runs per case | 1 / 3 | 3 / 10 |

`--request-tokens 32,128,1024` additionally measures longer requests. The
current predictive codec subdivides requests into independent chunks of at
most 32 tokens; directories, packet attributes and per-token lane states are
charged. Fitting a full-layer dense predictor is considerably more expensive
than the pilot. A successful pilot should precede the full run.

The full preset retains tens of gigabytes of local KV captures in addition to
model weights and dense calibration state; the default pilot is much smaller.
The current predictive codec is slow enough that full-layer repeated timing can
take hours. Its rate/latency tradeoff is the subject of the test, not a promised
serving improvement.

Two separate calibration/test tracks run by default:

- **WikiText-2 raw:** Salesforce/wikitext train for weights, validation for
  widths, test for held-out KV. This aligns the corpus family with SplitZip.
- **C4 English:** allenai/c4 first English train shard, with later unused train
  rows for widths; first English validation shard for held-out KV. This aligns
  the corpus family and train token length with speculative KV coding.

Dataset revisions are resolved once and recorded. Sampling uses deterministic
prefix stream blocks with EOS between rows. Each consumed row's unused suffix
is discarded. WikiText rows are lines, not articles. Sampling is **not** the
papers' exact random-document protocol; C4 and WikiText results remain separate.
The fixed seed selects fit rows without replacement and retains their original
RoPE positions. No TEST values fit coefficients, widths, codebooks or strengths.
All B strengths are prespecified as 4, without test-set selection.

These Qwen results must not be presented as a replication of SmolLM2-135M
quantized-weight CPU experiments, Qwen3-32B/H200 SplitZip results, or an FP8
Tensor Core comparison. The A4500 has no native FP8 Tensor Core path.

## Timing and accounting

Every arm receives identical `[layer, token, K||V]` original HF cache words.
Predictive arms pay for key permutation into adjacent RoPE coordinates and
back. Native baselines consume the original bytes. Separate fitting and source
H2D times do not enter resident encode/decode latency. GPU encoding and decoding
include allocations, native scratch, host dispatch and necessary size reads.

Each timed result has synchronized wall samples, CUDA stream intervals,
median, mean and p95. The summary averages per-request wall medians. Warmup
samples, including initial JIT work, are retained separately. The same explicit
nondefault Torch stream is supplied to nvCOMP and used for event timing.
Every warmup and measured output is checked outside the timed interval.

All methods receive a common model/state/layout/checksum envelope. Complete
bytes include it and every required buffer. Payload-only bytes are separate.
Native SplitZip includes its actual 7-byte escape record, rather than the
team's idealized 3-byte estimate. Construction-only counts/starts are scratch.
Shared predictor tables/codebooks are reported separately from per-request
bytes. Torch peak counters exclude native nvCOMP allocator peaks; CUDA free
memory snapshots are provided but are not presented as native peak usage.

A framed-byte restore is checked on every request. The first request of each
method/track also writes a T file, fsyncs file and directory, reads it, uploads
compressed bytes and decodes it. This is a **warm filesystem, same-process**
packet-only check, not cold SSD, network or fresh-process performance. There
is no P1 conversion or attention measurement in this matched transfer test.

Per-layer K/V ideal-rate and prediction-error diagnostics use CPU analysis
replay after GPU timing. They help locate gains and losses; they are not packet
rates or GPU timings. The report permits comparing compression savings against
encode/decode cost and shared-state memory before adding further mechanisms.

## Run and resume on A4500

```bash
git fetch origin
git switch experiment/qwen_matched_gpu
git pull --ff-only origin experiment/qwen_matched_gpu
bash scripts/run_qwen_matched_a4500.sh --publish-results
```

The wrapper uses the existing `.venv`, builds/checks the standalone runner,
installs missing datasets/constriction plus pinned optional nvCOMP, fetches
pinned SplitZip source and its license, runs actual CUDA correctness tests,
then captures, fits, benchmarks and packages reports. It never upgrades Torch,
Triton or Transformers. Missing native libraries are reported as unavailable;
a failed native round trip marks the report failed. Remaining methods still
run. Core CUDA correctness failure stops before model capture.

Capture files are checksummed and reusable; successful fits are reused after
identity/hash checks. An interrupted benchmark reruns its timed cases while
retaining capture and fit work:

```bash
bash scripts/run_qwen_matched_a4500.sh --resume runs/YOUR_RUN --publish-results
```

Use `--full` for all 36 layers, retaining it when resuming a full run. The
wrapper prints the exact resume path. Separate phases are also available with
`bench_qwen_matched_gpu.py --phase capture|fit|bench` and identical protocol
arguments. `--tracks wikitext2` selects one independent track.

Reports publish only results, summaries, provenance and test/setup logs.
Token IDs, raw KV, calibration arrays and encoded T files remain outside the
package. Results go to `experiment/qwen_matched_gpu`; existing serving/pool
experiments remain independently reproducible.

## Next mechanisms after this comparison

The layer ablation establishes whether extra decoder-known context helps on
Qwen and which layers or K/V channels benefit. The group-64 arm tests rate
against fewer sequential group steps. It does not incorporate pivoted channel
ordering, predicted-query attention, repeat-token references, prefill hints,
cosets or a value-slice rANS format. Those need separately named changes and
matched overhead accounting after this first GPU comparison.

## Primary references

- [Pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)
- [Pinned SplitZip source and license](https://github.com/Intelligent-Microsystems-Lab/SplitZip/tree/42c743118a29db3270ae6a921a4a4eecb20cad04)
- [Speculative KV coding methodology](https://blog.doubleword.ai/speculative-kv-coding)
- [AQUA-KV](https://arxiv.org/abs/2501.19392)
- [NVIDIA nvCOMP Python interface](https://docs.nvidia.com/cuda/nvcomp/py_api.html)

## Continuing from the first A4500 pilot

See [the pilot review](qwen_matched_a4500_pilot_review.md) for the measured
regressions and native baseline failure. The repair uses nvCOMP 5.3.0.16 plus
its separately pinned runtime, with native CUDA round trips before benchmarking.

`repair_qwen_matched_a4500.sh runs/ORIGINAL_RUN --publish-results` reuses the
original captures and frozen mean predictors, adds fixed-width A ablations,
and writes a separate report. It performs no prefill or coefficient fitting.
The original B states are not recalibrated. Do not use an ordinary full-phase
resume to apply this repair, because that verifies the fitting source identity.

For a manual equivalent, use `bench_qwen_matched_gpu.py --phase bench
--captures-from runs/ORIGINAL_RUN --output runs/FRESH_REPORT` with the desired
`--methods`. The extra fixed-width names are `previous32_A`, `aqua_style32_A`,
`strong_layers32_A` and `strong_layers64_A`. They derive their A state from the
stored base widths; no new calibration stream is needed. The original report
and all calibration arrays remain unchanged.
