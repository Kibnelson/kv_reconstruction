# Lite profiling, official ExANS and official AQUA-KV

Run on the existing A4500 checkout/environment:

```bash
cd ~/kv_reconstruction
git fetch origin
git switch experiment/llama_causal_packet_compare
git pull --ff-only origin experiment/llama_causal_packet_compare
bash scripts/run_llama_profile_a4500.sh
```

The default run reads these completed runs without changing them:

- Causal: `runs/llama_causal_a4500_20261027_184615_1307629`
- Controls: `runs/llama_baselines_a4500_20261027_191350_1897426`

It runs profiling, ExANS and AQUA sequentially and publishes a small report,
including partial/blocked results. It does not refit our predictors or repeat
the completed baseline measurements. It does perform paired Lite timing controls
so diagnostic profiler overhead cannot be mistaken for ordinary codec latency.

Each completed stage has checksummed artifacts. To retry a failed stage without
repeating completed work, use the printed output directory:

```bash
bash scripts/run_llama_profile_a4500.sh --resume runs/llama_profile_official_a4500_TIMESTAMP_PID
```

Keep the original options when resuming. Missing/corrupted artifacts, changed
source identity, different original KV or a changed root framework environment
cause a failure instead of silently combining results. Ctrl-C leaves checkpoints
in the run directory; the command's child process group is terminated.

The runtime/data recovery update supports a narrow migration from the initial
3e8ff46 release. It verifies the completed artifacts, archives the original
results/recipe under `resume_history/`, and preserves profiling, toolchain,
AQUA setup and BF16 PPL. It does not rerun those stages. Model, inputs, modes,
source pins and official evaluation settings must still match.

The subsequent path/tokenizer repair also migrates the known 27dfbf6 partial
run and preserves its completed ExANS native build/GPU smoke checkpoint. Hashing
normalizes JSON string paths before file access. Regression coverage runs the
full sender/receiver wrapper with real framing and a CPU native-transport fixture,
in addition to the actual native GPU test performed on the A4500.

ExANS standalone executables now receive the CUDA/nvCOMP runtime library paths,
including `libcudart.so.12`. These loader-path additions apply only to ExANS
children; they do not change AQUA's runtime used by the saved BF16 reference.

## Profiling

Defaults are Lite, test_text, 512 tokens/all 28 layers, lossless and lossy κ=1.
The codec, arithmetic and fitted state are unchanged. Packet SHA256 and decoded
word digests must match the existing independent-receiver results both before
and during instrumentation. No recalibration is performed.

The report contains ordinary synchronized timing controls, profiler overhead,
CUDA device-event counts, top self-device and CPU operator times, matrix
multiplication operations, launches/synchronizations, and semantic scopes for
forward/inverse RoPE, entropy coding, residual reconstruction, packet validation
and compaction. Scopes are inclusive/nested and must not be summed. GPU traces
require working CUPTI/profiler access; zero captured CUDA events are an error.

Full gzip Chrome traces remain in the GPU run's `traces/` directory. The Git
report contains operator summaries and trace hashes, avoiding large trace
uploads. No raw KV, tensor state, predictors or native binaries are published.

## ExANS

The source pin remains OpenLake
`de6c38bcc5438475bd1abf1e9b3d154cef414ef6`. Its native C ABI and CUDA/nvCOMP
compression/decompression code run unchanged. Before measurements, the runner
builds and runs the unchanged author GPU correctness/overflow test.

The upstream build failed because `EXPANS_ALLOC` passes a `unique_ptr` to the
existing raw-pointer error handler. `native/exans_build_compat.h` is injected by
the compiler to add a forwarding overload for that allocation-error path. It
calls the original handler with `.get()`. It does not edit any author file or
change successful allocation, prediction, entropy coding, records or decoding.
The header hash, source pin and built library hash are recorded. This is a
disclosed build-compatibility addition, not an untouched build configuration.

ExANS receives exactly the original BF16 words used by our codecs and SplitZip.
The local transport harness uses the released 999,424-byte native blocks and
770,048-byte fixed records (the author's 1.3x target and 4,096-byte alignment).
It charges physical final-block padding, all transmitted fixed slots, directory,
envelope and checksum. Non-fitting native blocks use explicitly charged raw
fallbacks in our harness; this is not a reproduction of OpenLake's vLLM/RDMA
overflow policy. Native records are not compacted into hypothetical smaller
payloads. The report labels the scope accordingly.

Packets are serialized and replayed by a fresh native receiver with packet bytes
and the checked native library only. A correctness-only uniform-exponent/tail
fixture checks the framed overflow path separately from Llama performance rows.
Torch peaks omit native nvCOMP allocations; native peaks remain unmeasured.

The nvCOMP SDK stays at `nvidia-libnvcomp-cu12==5.3.0.16`. The CUDA 12.8 helper
reuses the existing project-local toolkit. Actual GPU checks precede build/setup.

## AQUA-KV

The source pin remains `e744183f2f3a4a90381f20a6975456871fd1ae29`. The existing
official environment under `runs/_llama_official/envs/aqua` is reused. The missing
loader dependency Accelerate 1.1.1 is installed under constraints preserving
Torch 2.5.1, Transformers 4.46.3 and Datasets 2.20.0. A CUDA Hadamard preflight
runs before expensive work; pins are rechecked before training/evaluation.

The runner invokes the unchanged author scripts with the README settings:

- Same pinned original BF16 `unsloth/Llama-3.2-3B` checkpoint as our experiment.
- RedPajama: 256 calibration samples × 8,192 tokens, 32 validation samples,
  offloaded activations; no smaller fallback calibration.
- HIGGS: dimension 2/grid 16 residuals, grid 256 for the first layer,
  Hadamard group 1,024.
- Full WikiText2 PPL at context 8,192, prefix 4, recent buffer 128 and the
  author's default 32-token evaluation steps.
- Raw BF16 and quantized evaluation use matching full WikiText/token digests;
  dataset fingerprints and calibration configuration are recorded.

### Calibration source availability

The original `togethercomputer/RedPajama-Data-1T-Sample` endpoint returned 401
and the author's loader could not access it. The default remains
`--aqua-calibration original`; there is no automatic dataset substitution.

An explicit alternative is available:

```bash
bash scripts/run_llama_profile_a4500.sh \
  --resume runs/llama_profile_official_a4500_20261027_194103_2478724 \
  --aqua-calibration backup
```

This selects `liang2kl/RedPajama-Data-1T-Sample-Backup` at
`c2c1504dcb5887121b04787e36055e32aeb9017c`, with 930,514 expected records.
The manifest pins all 11 Parquet shard sizes and SHA256s (3,024,841,588 download
bytes). This authenticates that particular backup, **not equivalence or row order
against the removed original**. The report explicitly labels backup calibration.

The sampler follows the released `get_red_pajama` rule, including seed 0,
BOS/EOS assignments, long-document rejection and random window
selection. Tests compare its tokens with the released function on identical
records. The resulting 256 × 8,192 token batches are passed via the author's
supported local pretokenized-file input; training, holdout, quantizers and full
WikiText evaluation are unchanged. Row selections, text/token hashes and the
token-file checksum are retained; raw text/tokens are not published.

The pinned checkpoint declares generic `PreTrainedTokenizer` and supplies
`tokenizer.json`. With Transformers 4.46.3, forcing `use_fast=False` attempts an
unimplemented slow vocabulary and then reports a secondary protobuf error.
Backup preparation selects the checkpoint's fast backend, as the successful
official perplexity entry point does. Tokenizer config/JSON hashes and the
backend choice are recorded. This compatibility selection is outside the
unchanged author training code. An integration test uses the actual pinned
checkpoint assets and AQUA's pinned Transformers/tokenizers versions; it
reproduces the failed slow load, verifies fast/default-PPL token identity, and
matches the released sampler's output on the same records.

This repair reuses the already downloaded shards and Arrow cache; it does not
restart the completed download or the BF16 PPL evaluation.

Selecting the backup is a documented calibration provenance variation. It must
not be described as exact reproduction of the original calibration data. If
strict original-data provenance is required, leave AQUA blocked until the
original sample/cache becomes available. Completed ExANS and profiling results
can still be retained.

The README says efficient inference kernels are TBU. This evaluates the released
implementation. Official AQUA PPL and full evaluation-command wall time are
reported separately from matched frozen-KV codec latency. Those times include
model/cache work and startup. AQUA's physical cache/wire bytes remain unmeasured;
nominal quantizer bits are not presented as complete packet bytes. Its full
WikiText2 PPL is not ranked against our short PEP/CPython continuation smoke test.

If CUDA, memory, datasets, dependency or build setup fails, logs and missing
metrics are retained. There are no synthetic or CPU-estimated GPU performance
substitutions. No pool-attention, SSD/network, T-to-P or serving result is implied.

Official references: [AQUA repository](https://github.com/goodevening13/aquakv/tree/e744183f2f3a4a90381f20a6975456871fd1ae29)
and [OpenLake ExANS source](https://github.com/openlake-project/openlake/tree/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda).
