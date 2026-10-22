# AQUA-KV, SplitZip and ExANS versus the exact codec

This experiment compares the same held-out Qwen BF16 words, with complete
portable packets, synchronized GPU encode/decode latency and frozen decoder
state. It reuses the captures from the earlier matched A4500 pilot. It does
not recapture the model or refit our predictors.

| Family | Requested methods | Implementation and target |
|---|---|---|
| Ours | `block32_A`, `block32_B` | Exact BF16 T1 codec; same frozen mean predictor, fixed or adaptive widths |
| AQUA mean control | `aqua_style32_A` | Our exact coder with AQUA-style mean inputs; isolates predictor structure |
| SplitZip | `splitzip_native` | Unmodified pinned author Triton codec, exact BF16 |
| ExANS | `exans_native` | Pinned OpenLake v2 CUDA codec with one host build fix and framing adapter, exact BF16 |
| AQUA-KV Quanto | `aquakv_quanto2`, `aquakv_quanto4` | Released AQUA cache and Quanto residual methods, lossy page adaptation |
| Reference | `raw_bf16` | Original word copy |

The report separates lossless methods from lossy AQUA results. For AQUA it
records complete bytes, latency, original-word match fraction, aggregate KV
relative MSE and per-layer K/V squared/absolute error. A lossy result cannot
enter the lossless compression or speed rankings, even if one sample happens
to reconstruct exactly. KV distortion does not establish downstream quality;
perplexity and LongBench remain unmeasured.

## Frozen workload and the larger request

Model: `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The full model produced the
captures, but this pilot measures only its first four KV layers. WikiText2
and C4 remain separate tracks. The existing deterministic stream protocol,
2,048 selected TRAIN rows, TEST contexts and source hashes are unchanged.
All methods receive original post-RoPE HF split-half K/V words.

The helper measures beginning/end windows of **32, 128 and 488 tokens** in
each captured TEST context. All methods get every request size. The 488-token
request is included because, at this pilot's 16,384 bytes/token, it is exactly
eight upstream ExANS physical blocks: `488 × 16384 = 8 × 999424`. This exposes
the effect of short-page padding without changing the codec's format.
The same fixed grouping is used for all methods, and padding appears in
ExANS request metadata. This is not a reproduction of the papers' full model,
dataset sampling, or serving experiments.

## AQUA-KV implementation and calibration

Upstream repository: <https://github.com/goodevening13/aquakv>.
Pinned commit: `e744183f2f3a4a90381f20a6975456871fd1ae29`.
Quanto: `optimum-quanto==0.2.6`, matching upstream requirements.

`matched_aqua.py` loads the unchanged definitions of `QuantoQuantizer`,
`SingleChunkQuantizedCacheWithPredictors` and the upstream ridge solver
from SHA256-checked source files. The HF Cache base is a local compatibility
shim, and the rotary helper supplies the same split-half formula without
depending on the older Transformers cache/model API. A typed tensor-tree
wrapper adds portable serialization. No upstream method body is rewritten.
Loading only the selected definitions avoids importing unused HIGGS/CUDA
Hadamard dependencies; the original source files remain unchanged.

The configuration uses per-token Quanto groups of 64, 2- or 4-bit residual
quantization, BF16 linear predictor weights, and an unquantized first layer.
K is predicted from reconstructed previous-layer K; V uses reconstructed
current K and previous V. Each bitwidth gets its own predictors, fitted with
the released solver and damping `1e-3` on sequentially reconstructed TRAIN KV.
They are saved under the fresh run's `native_calibration/`, with identity and
file hashes. TEST values never enter fitting. The same selected TRAIN rows
used by our frozen means are used, and original RoPE positions are retained.

Keys are unrotated for prediction as in the released cache. Raw unrotated
first-layer K/V, packed quantized residual tensors, affine scales/shifts and
BF16 cos/sin are all charged. Quanto's packed-plane padding is charged too.
Full packet decoding rebuilds the cache from these fields plus frozen
predictors, then reconstructs layers causally. It does not retain the original
KV or a dequantized full cache as hidden decoder context.

This is the released **Quanto** alternative, not the paper's headline HIGGS
configuration. Recent-token buffering, attention sinks, HF serving integration
and generation quality are outside this page-codec experiment. The first-layer
anchor is a quarter of this four-layer pilot; full-model AQUA compression
cannot be inferred from this rate. RoPE inverse/reapplication is BF16 arithmetic
and is allowed to contribute to the measured lossy error.

## Native ExANS and accounting

Upstream repository: <https://github.com/openlake-project/openlake>.
Pinned commit: `de6c38bcc5438475bd1abf1e9b3d154cef414ef6`.
Files: `expans_codec.cu`, `expans_kernels.cuh`, `openlake/expans_codec.h`.
Native ANS dependency: wrapper and runtime `nvCOMP 5.3.0.16`.

The pinned release does not compile as-is: its allocation-error macro passes a
`std::unique_ptr` to a function expecting a raw pointer. Setup writes a separate
build copy with exactly one `codec` → `codec.get()` correction in that error
handler. Original source files and all GPU kernels remain unchanged. The build
manifest records this patch and the compiled source hash. No codec algorithm,
record layout or successful allocation/compression path is changed.

The released implementation has 122 groups × 4,096 BF16 words per physical
block: **999,424 bytes**. Its record version is 2; it combines a native
histogram/phase bitmap with raw sign/fraction and nvCOMP ANS. The centers
are the upstream constant 127. This is the actual pinned public implementation,
not a substitute that runs generic ANS over the original BF16 words.

The slot size **770,048 bytes** matches the upstream smoke test. The adapter
flattens each common Qwen request and zero-pads only its final physical block.
Padding, splitting and record compaction run in the timed path. Each successful
record is transmitted through its native `actual_bytes`, including the 64-byte
header, sign/fraction, phase bitmap and ANS frame. Native alignment bytes are
kept. If a native record does not fit its slot, the adapter stores that block's
original raw BF16 bytes and charges them; it never omits or quantizes overflow.
Record modes and lengths are in the charged outer packet directory.

The GPU decoder restores native slots from portable parts and invokes the
released C decoder. Host descriptor arrays and the original codec's stream
synchronizations are retained; their cost is included in synchronized wall
latency. There is no CPU entropy coding. The adapter records padded input
bytes, native slot reservation and raw fallbacks separately from transmitted
bytes. Upstream `cudaMalloc` state and scratch peaks are **unmeasured** by Torch;
this does not establish GPU resident memory savings.

## Run on the A4500

```bash
(
set -euo pipefail
cd ~/kv_reconstruction
git fetch origin
git switch experiment/qwen_matched_gpu
git pull --ff-only origin experiment/qwen_matched_gpu
bash scripts/compare_qwen_native_a4500.sh \
  runs/qwen_matched_a4500_20261021_184512_1225465 --publish-results
)
```

The helper installs pinned optional packages with `--no-deps`, downloads
checked source files and builds native ExANS. It preserves Torch, NumPy,
Triton and Transformers. A **CUDA 12 toolkit with `nvcc`** and a working host
C++ compiler are needed to build ExANS and Quanto's native unpack extension.
Set `CUDA_HOME` to the toolkit directory if it is not discoverable.
The CUDA driver and a CUDA-enabled Torch wheel alone do not supply `nvcc`.
The default compilation target is SM86 for the A4500.

Before any calibration or timed comparison, actual-GPU preflight tests require:

- our causal predictor/byte checks;
- AQUA 2/4-bit reconstructed-context fitting, packed payload accounting and a
  fresh packet-only decoder on a nondefault CUDA stream;
- native SplitZip reconstruction of every BF16 pattern;
- a **real compressed ExANS record** containing all BF16 patterns, plus
  incompressible overflow with exact raw fallback and fresh decoding.

Missing native dependencies or build errors stop the helper with saved logs.
The benchmark also uses `--require-all`: unavailable or failed requested
baselines cannot yield a successful four-family report. Measured timings
are only populated by an actual CUDA run. Compilation and CPU checks are not
GPU measurements.

Outputs go into a new `runs/qwen_native_compare_a4500_*` directory. The
publishable allowlist contains reports, provenance and logs; original KV,
token IDs, calibration tensors and compressed payload files are excluded.
With `--publish-results` the helper commits the regenerated standalone runner
and small report package, then pushes this experiment branch.
