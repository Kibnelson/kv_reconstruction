# NUC evaluation: official baselines, lossless T and resident P

## Fixed starting point

Keep [810b516](https://github.com/Kibnelson/kv_reconstruction/tree/810b5166b3668e26f0ecf8e60f2a1fa079dfb32b) as the measured transfer reference. Its report is
`benchmark_results/team_lookup_a4500_20261028_141701_1145261_report_20261028_143153/`.
The reference is original BF16 Llama-3.2-3B, all 28 layers, two PEP/CPython TEST documents, 512/8192 tokens.
At 8K, Rank256 direct/search restores in 21.709 ms versus ExANS 23.706 ms; Dense has a smaller packet.
These are checked, GPU-resident framed API measurements. Its five link rows are calculated wire time plus measured decode, without real NIC, SSD or serving measurements.

The subsequent register and probability-grid experiments did not establish an improvement. They remain evidence; they do not replace the fixed starting point.
Pause further predictor/kernel changes while establishing the official baseline procedures below.

## Three separate comparisons

| Comparison | Methods | Required result |
|---|---|---|
| Lossless transfer/storage T | BF16, released SplitZip, released ExANS component, ours lossless Dense/Rank256 | Exact original BF16 words; complete bytes; encode/decode; shared state; transfer/restore cost |
| Lossy quality and lossy T | BF16 quality reference, official AQUA-KV HIGGS settings, our explicitly named lossy codecs | Same quality workload and evaluator; reconstruction error/PPL; actual physical bytes where implemented |
| Resident attention P | BF16 attention, proposed exact P1 attention; separately charged restore-then-attend controls | Live GPU allocations, attention latency/accuracy, admission cost, serving capacity only after integration |

AQUA-KV is lossy even when quality loss is very small. An exact decoder replay of an encoder's lossy reconstruction does not mean equality with the original BF16 cache.
Do not put nominal quantizer bits into a lossless complete-packet byte table.
An `aqua_style` predictor inside our exact entropy coder is our ablation, not an official lossless AQUA-KV baseline.

## What counts as reproduction

Preserve author codec bodies and official evaluation entry points. Pin sources, dependencies, model revision, data/splits, tokenizer, quantizer settings and timing boundaries.
Changing hardware and using a supported smaller model is a documented evaluation setting. It does not reproduce an H100/H200 paper's numerical performance.
Keep two result views: the author's native metric and a separately named matched-input transfer experiment. Different timing scopes cannot share a speed ranking.
Changes to calibration size/data, buffering, model precision, codec logic or overflow handling must be explicitly named; unavailable experiments remain missing.

## Official source and protocol audit

### SplitZip: first runnable lossless procedure

Pin: [`42c743118a29db3270ae6a921a4a4eecb20cad04`](https://github.com/Intelligent-Microsystems-Lab/SplitZip/tree/42c743118a29db3270ae6a921a4a4eecb20cad04).
Use its [README](https://github.com/Intelligent-Microsystems-Lab/SplitZip/blob/42c743118a29db3270ae6a921a4a4eecb20cad04/README.md),
[benchmark](https://github.com/Intelligent-Microsystems-Lab/SplitZip/blob/42c743118a29db3270ae6a921a4a4eecb20cad04/bench_codec_throughput.py) and
[codec](https://github.com/Intelligent-Microsystems-Lab/SplitZip/blob/42c743118a29db3270ae6a921a4a4eecb20cad04/codec_gpu.py).

The released native procedure uses WikiText2 raw TRAIN calibration and TEST evaluation. Defaults are four usable prompts, 4,194,304 calibration values, 268,435,456 measured values and 1024-value chunks; collected data can be repeated/truncated to the requested size. Timing is 10 warmups, 50 API calls between synchronizations, five repeats, with a bitwise check first. It reports throughput in original BF16 bytes/second and native compressed bytes.
The paper uses Qwen3-32B and H200. The exposed `--model` option allows the existing pinned Llama-3.2-3B snapshot on the NUC without changing source.
This is the **official procedure on a NUC-feasible model**, not the paper's model/hardware reproduction.

Create a fresh output directory and use an absolute verified cached snapshot in this official command:

```bash
runs/_llama_official/envs/splitzip/bin/python \
  runs/_llama_official/sources/splitzip/bench_codec_throughput.py \
  --device cuda:0 --model /absolute/path/to/pinned_llama32_bf16_snapshot \
  --local-files-only --output runs/OFFICIAL_RUN/splitzip_native.json
```

Do not use synthetic inputs or calibrate-on-TEST for the performance result. The public release omits its SGLang/Mooncake serving integration.
The released codec charges seven bytes per escape; paper shorthand and logical entropy must not override actual storage.

Then run its supported saved-tensor mode on the exact words used by ours, with separate `--input`, `--calibration-input` and actual `--numel`. Keep that matched-input result separate from the native default workload.
The shared input must retain our complete layer/token/channel geometry before flattening for generic codecs. The author's repeated/truncated flat throughput tensor is not automatically a valid previous-layer prediction workload.
For a new WikiText comparison, fit ours only on TRAIN/validation, then encode the same frozen TEST tensors. Do not place the old PEP/CPython T numbers beside newly generated WikiText numbers as though the data were identical.

### ExANS: released component versus public headline

Pin: [`de6c38bcc5438475bd1abf1e9b3d154cef414ef6`](https://github.com/openlake-project/openlake/tree/de6c38bcc5438475bd1abf1e9b3d154cef414ef6).
Inspect the [CUDA implementation](https://github.com/openlake-project/openlake/blob/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda/src/expans_codec.cu),
[kernel constants](https://github.com/openlake-project/openlake/blob/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda/src/expans_kernels.cuh),
[official correctness/overflow test](https://github.com/openlake-project/openlake/blob/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda/tests/expans_codec_test.cu) and
[vLLM connector](https://github.com/openlake-project/openlake/blob/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/external/connectors/vllm/openlake_adapter.py).

Run the unchanged author correctness test first, with CUDA 12/nvCOMP and SM86. Disclose the existing compiler-only unique_ptr compatibility overload used to build this pin; do not call that an unchanged build configuration.
The released component uses version-2 records, phase maps, nvCOMP ANS and exactly 999,424 raw bytes/block. The author's H100/256K-context post describes another configuration and does not release an identified snapshot or performance harness sufficient to reproduce its headline.

Our existing 770,048-byte fixed-slot study charges padding, framing and raw overflow fallback. Successful slots alone cap compression at about 1.298x before overhead; this is not the post's compact-payload 1.51x metric. Our 941-block 8K batch also differs from the connector's maximum-64-block batches and stop-on-overflow policy.
Keep those results as **matched released-C-ABI transfer controls**. Do not relabel them official serving reproduction.

Before attempting unchanged vLLM serving, log the actual registered cache segment lengths and pass them to the official create call. A standard TP1 Llama-3.2-3B BF16 page has 114,688 bytes per token across layers, which does not produce the required raw block size for ordinary integer token block sizes. Actual storage padding must be observed. A rejected geometry remains unsupported; do not silently reblock the serving experiment.

### AQUA-KV: official lossy quality procedure

Pin: [`e744183f2f3a4a90381f20a6975456871fd1ae29`](https://github.com/goodevening13/aquakv/tree/e744183f2f3a4a90381f20a6975456871fd1ae29).
Use its [training script](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/train_predictors.py),
[PPL evaluator](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/evaluate_perplexity.py),
[cache implementation](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/aquakv/cache_utils.py) and
[quantizers](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/aquakv/quantizers.py).

The README explicitly supports `unsloth/Llama-3.2-3B`. Preserve RedPajama 256x8192 calibration, 32 holdout samples, HIGGS dimension2/grid16/64/256 for nominal2/3/4 bits, Hadamard1024, separate first-layer grid256, prefix4 and recent128. Fit predictors separately for each setting. Use full WikiText2 TEST PPL at context8192 with its 32-token steps. LongBench is a later, separately resourced Instruct-model evaluation.

The published 2.x-bit figure is a logical quantization metric. The prototype uses uint8 indices and FP32 scales; its PPL path saves reconstructed BF16 chunks. It supplies quality evidence, not complete transport bytes, resident-P capacity or a matched 21.71-ms transfer decode measurement. Quanto page adapters and locally rewritten AQUA experiments are not this official evaluation.

The previous official-code attempt reached layer1 and ended with SIGKILL; no successful official quantized PPL was produced. Host OOM is plausible but unconfirmed. With activation offload, two `[256,8192,3072]` BF16 buffers alone require 24GiB of host RAM, before model, K/V and regression copies. A4500's 20GB is VRAM, not host RAM. Check RAM/cgroup/pinned-memory limits before another attempt; a source-level memory calculation is not a measured peak or a fit guarantee.
The original RedPajama endpoint was unavailable in the previous attempt. The existing backup path is an explicit provenance variation, not verified equivalent original data. No automatic reduced-sample/shorter-context/data fallback is permitted for a reproduction label.
If resources are insufficient, train the unchanged official procedure on a larger machine and evaluate its matching predictors on the NUC. Otherwise retain a blocked row. An unchanged HIGGS-only evaluation can be a separate quantizer-only control, never an AQUA predictor result.

## Our lossy evaluation

Keep our lossy result separate from our exact T result. The older causal codec has real framed GPU packets/fresh receivers but only short continuation quality tests. The other branch's fast lossy GPU codec has intended-reconstruction parity and additional quality windows, but those do not constitute official AQUA reproduction or a complete transfer envelope.
Adopt the same model, tokenization, full quality workload and cache-update policy for the comparison. Verify actual received lossy reconstructions, retain exact sinks/recent state when required by the declared variant, and report measured PPL/error alongside complete physical bytes. A new ours-only integration may be required; leave its full-protocol result pending until implemented.

## T and P together

Canonical flow: generate BF16 KV -> encode T -> retain T in DRAM/SSD and transmit T -> install P on GPU -> attention reads P. DRAM/SSD movement keeps T compressed. When a valid external T backing is retained, eviction can discard P without re-encoding unchanged KV.
T is optimized for compact sequential restoration; P is intended for repeated independent attention tile access. P earns its additional format/conversion only if its measured active-GPU memory/attention tradeoff is worthwhile.

T already contains mode/geometry, state/source identity, segment offsets, relative starts, rANS states and a frame checksum. Add a versioned deployment envelope binding checkpoint/tokenizer/RoPE/layout, predictor/coder versions and explicit supported layer/token ranges. Current code supports start0 and all layers only; a header field does not implement arbitrary ranges.
Receiver calibration is a real deployment cost. At810b516, Dense shared tensors occupy117.3125MiB and Rank256146.2422MiB, with57,688 bytes of coder tables additional. The current combined on-disk state bundle is about146.25MiB. The often quoted117MB is approximately the dense coefficient matrix alone in decimal units; it is not every receiver's complete installed state. Charge cold setup/state transfer, then amortize an already installed state in warm results.

P1 GPU packing/unpacking was measured on earlier SmolLM pages; that is not yet a Llama resident-attention result. Its80%-arena configuration reported13.05allocatedbpv (18.44% saved), including reservation and directory. Payload-only savings must not replace allocated-memory accounting.

P experiments on the same A4500 and Llama words:

1. Fit P calibration on TRAIN; measure512/2048 then8192-token exact P1 pack/full-unpack/selected-tile access. Include descriptors, reservation/slack, raw sectors, escape lookup, scan scratch, outputs and shared calibration.
2. Compare BF16 attention with P1 attention under identical arithmetic and queries. Charge restore-then-attend codecs separately, including their full BF16 allocation.
3. Measure the real composed bridge T->dense->P before attempting bounded layer scratch or fusion. Direct T currently writes896MiB of full BF16 output for8K; T->P is not implemented. One8K layer is32MiB and two such buffers64MiB, arithmetic only, with all other scratch additional.
4. Add append/seal, raw tails, overflow/backpressure, publication and reclamation before multi-request serving/capacity claims. P1 is currently an append-batch arena, not a reusable serving allocator.

P source: [6081076](https://github.com/Kibnelson/kv_reconstruction/tree/6081076d794969193533841d3080e773df73c267), particularly `qwen_kv_bench/pool_p1.py` and `docs/gpu_kv_transfer_pool_roadmap.md`. Its old adjacent-pair SmolLM calibration must be replaced with matching Llama HF-order calibration; preserving raw BF16 words forbids an unaccounted numerical RoPE conversion.

## Execution order and acceptance

First collect `scripts/check_nuc_baseline_readiness.py` output. It is inventory, not a benchmark or a fit guarantee. Then run SplitZip's official native procedure on the pinned cached Llama snapshot; retain its own output/command/source checksums. Next reproduce its native metric on the frozen matched inputs and re-evaluate ours on the shared official workload. ExANS component results retain their stated scope; test serving only after actual geometry succeeds. Run official AQUA quality when calibration provenance/resources are satisfied. P follows as its own measured workstream.

Each result names source commit, model/data hashes, complete payload/state accounting, actual timing scope, original-word equality or lossy reconstruction/PPL, native allocator limitations and hardware. Preserve failed runs and unavailable cells. Report1/100/200/300/400Gb/s as calculated link models until real transfers are measured; the NUC's actual NIC/PCIe path is a separate constraint.

Run the inventory on the NUC first; it publishes only its small JSON/summary so the next official run can use verified local prerequisites:

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_nuc_baseline_readiness.sh
```

This step does not install packages, download data, retry AQUA training or execute a GPU performance benchmark.
