# Official baseline protocol for the fixed Qwen3 model

Updated 2026-10-22. This protocol supersedes the adapted page comparison as
the basis for claims about AQUA-KV, SplitZip and ExANS. Historical source,
measurements and reports remain unchanged.

The user has subsequently selected Llama-3.2-3B for a separate common-model
GPU test. See [the Llama official run](llama32_official_gpu.md). The fixed-Qwen
rules below continue to describe the Qwen track; Qwen results and calibration
are not reused as Llama results.

## Fixed model and comparison rule

Use `Qwen/Qwen3-4B-Instruct-2507`, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, with its original BF16 weights,
tokenizer, configuration and all 36 layers, on the RTX A4500. Do not substitute
Qwen2.5, relabel the model type, remove normalization, or change model weights
to make a baseline run.

Run each baseline through its official entry point, from an immutable author
commit, in an isolated environment with its intended dependencies and method
settings. Passing a supported model/input argument or selecting the CUDA
device is allowed. Rewriting the baseline, monkey-patching its dependencies,
extracting selected method bodies, or substituting our cache/quantizer is not
allowed for the official comparison. Unsupported methods are recorded as
blocked, with the error and provenance, rather than repaired silently.

The common Qwen3-4B model and A4500 are new workload/hardware settings relative
to the papers. Even with unchanged author scripts, this is an official-code
evaluation on our workload, not an exact reproduction of every published
experiment. Record those differences explicitly.

## What the adapted page experiment measured

The full Qwen model produced original post-RoPE BF16 KV. Our pilot retained
only the first four layers and selected beginning/end windows of 32, 128 and
488 tokens. Our harness then called each codec through a common encode/decode
adapter, with our portable packet directory and shared-state accounting.

For AQUA-KV, the adapter loaded released cache, Quanto and regression method
bodies, supplied local HF-cache/RoPE compatibility, fitted predictors on our
selected WikiText2/C4 TRAIN rows, retained a raw first layer, and charged
cos/sin and affine quantizer fields. It omitted the full serving cache,
attention-sink/recent-token handling and downstream perplexity/LongBench
evaluation. Quanto unpack used a local host stream-launch wrapper.

Those measurements describe that particular page adapter. They do not
establish AQUA-KV's full-model compression, published accuracy, reference
inference latency or GPU resident-memory savings. The 6.38/7.88 complete-bpv
Quanto rows and the approximately 60%/51% packet savings remain historical
adapter measurements, excluded from official-baseline rankings. The
`aqua_style32_A/B` arms are our exact-coder predictor ablations, not AQUA-KV.

SplitZip used the released codec inside our packet harness. ExANS used the
native codec with a disclosed host build fix and our padding/framing wrapper.
Their earlier packet results also retain that scope. In particular, a native
kernel inside a custom harness is not the same as the authors' full benchmark.

## Compatibility audit

This is a source/interface audit, not a successful GPU run or a performance
result. Dependencies, real model execution and GPU correctness still require
separate preflight on the A4500.

| Baseline | Pinned author commit | Official path and present status |
| --- | --- | --- |
| AQUA-KV | `e744183f2f3a4a90381f20a6975456871fd1ae29` | `train_predictors.py`, `evaluate_perplexity.py`, `evaluate_longbench.py`. Blocked for the fixed Qwen3 model: `aquakv/modelutils.py::get_layers` does not accept `qwen3`; the documented Transformers 4.46-4.48 environment predates Qwen3 support. The inspected current `main` has the same model-utils file. Do not patch this or import our adapter. |
| SplitZip | `42c743118a29db3270ae6a921a4a4eecb20cad04` | `bench_codec_throughput.py` exposes `--model`, `--input`, and `--calibration-input`. Source inspection identifies a supported route to our Qwen3 workload without editing the codec. Actual GPU execution is pending. |
| ExANS / OpenLake | `de6c38bcc5438475bd1abf1e9b3d154cef414ef6` | Author CMake exposes `OPENLAKE_EXPANS_BUILD` and `openlake_expans_test`. Our earlier build changed a `unique_ptr` error-path argument to `codec.get()`; that build cannot qualify as an unchanged source run. An unchanged author build must pass separately. The released smoke test uses constructed BF16 words and does not establish Qwen codec throughput. |

AQUA's model whitelist includes `qwen2`, not `qwen3`. Adding one name is not an
approved workaround. Its calibration and model integration must remain those
of the author release. If no author release supports our exact model, the
official AQUA comparison stays blocked under the user's constraints.

## Execution and reporting

1. Check out the exact upstream commit and verify a clean tracked worktree.
   Save the commit, source hashes, dependency versions, command and device.
   Preserve complete setup/error logs. Do not use AST-loaded cache classes,
   our native-baseline setup patches, or our page adapters for this track.
2. Use the author's calibration and evaluation procedures. For AQUA's
   documented HIGGS example these include 256 RedPajama sequences of 8192
   tokens (32 held out), Hadamard groups of 1024, four prefix sink tokens and
   a recent buffer of 128. Retain the released first-layer handling. The
   existing 2048-row fits cannot substitute for that calibration.
3. For SplitZip, retain separate WikiText2 TRAIN codebook calibration and TEST
   KV evaluation, native chunk settings and author warmup/repeat timing.
   Its public `--model` argument can select the fixed Qwen snapshot. A local
   snapshot path for the pinned model avoids resolving a newer model revision.
   Its saved-tensor inputs are another supported route, provided tensors and
   TRAIN/TEST provenance are recorded and verified. Do not calibrate on TEST.
4. For ExANS, first use the unchanged author build and GPU test. Establish
   whether a released Qwen workload/benchmark entry point is available before
   claiming Qwen throughput. A synthetic smoke test or a new local wrapper
   cannot fill a missing official Qwen result.
5. Evaluate our method on the same held-out original Qwen words, corpus,
   lengths, device and timing scope wherever an official interface permits a
   direct comparison. Retain each method's intended calibration procedure
   and disclose differences. Full-model results must include all layers and
   anchors; four-layer pilot rows cannot substitute.
6. Preserve the authors' raw outputs. Report native codec bytes/timing on their
   original scope. Report any external storage/transfer envelope separately;
   do not inject our framing into an official metric. Separate compressed
   payload, shared predictor/codebook state, buffers and peak allocation.
7. Compare only matching measurement scopes. A reference inference result,
   a codec-only CUDA-event result, and our synchronized packet latency are
   different measurements. Additional common measurements can be collected
   externally only when they preserve the official path and are clearly
   labelled. Missing measurements remain unmeasured.
8. Lossless baselines must recover all BF16 words exactly. Lossy AQUA results
   require perplexity/LongBench quality as well as size; KV relative MSE and
   word-match rate alone are insufficient. No serving, SSD or attention-pool
   result follows from a codec-only benchmark.

The next eligible GPU step is SplitZip's unchanged official benchmark on the
fixed Qwen snapshot. AQUA remains blocked, and ExANS needs unchanged-build and
official-workload validation. No new GPU result is claimed by this document.

## Source references

- [Fixed Qwen configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)
- [AQUA model dispatch](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/aquakv/modelutils.py)
- [AQUA official procedure](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/README.md)
- [AQUA calibration](https://github.com/goodevening13/aquakv/blob/e744183f2f3a4a90381f20a6975456871fd1ae29/train_predictors.py)
- [SplitZip official procedure](https://github.com/Intelligent-Microsystems-Lab/SplitZip/blob/42c743118a29db3270ae6a921a4a4eecb20cad04/README.md)
- [SplitZip benchmark](https://github.com/Intelligent-Microsystems-Lab/SplitZip/blob/42c743118a29db3270ae6a921a4a4eecb20cad04/bench_codec_throughput.py)
- [ExANS CMake targets](https://github.com/openlake-project/openlake/blob/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda/CMakeLists.txt)
- [ExANS GPU smoke test](https://github.com/openlake-project/openlake/blob/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda/tests/expans_codec_test.cu)
- [Historical adapted page experiment](qwen_native_baselines.md)
