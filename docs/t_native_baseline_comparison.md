# Frozen T versus the other GPU codecs

This experiment stays on `experiment/t-head-aware-spread-v1`. It adds a comparison runner without changing any original T sources, predictors, policies, existing runners or old results. No merge is authorized by a successful benchmark.

The first stage compares exact BF16 reconstruction:

| Method | Implementation | State and calibration |
|---|---|---|
| `raw_bf16` | Resident input plus a reference device copy | No trained state; every raw word is transferred |
| `t_fast` | Previously tested fast dense encoder, original T entropy coder/decoder | Frozen `dense_legacy_global.pt`; expected full-test saving 44.0484% |
| `t_hbh` | Previously tested head-by-head predictor and learned probability widths | Frozen `hbh_learned_selected.pt`; expected full-test saving 48.5682% |
| `splitzip` | Authors' `ChunkLocalSplitZipGPU`, 1,024-value chunks | Top 16 exponents calibrated only on `wtrain0`; all four lookup tables saved |
| `exans` | Authors' ExpANS CUDA library and nvCOMP ANS | Authors' fixed block layout, including raw handling described below |

The SplitZip source is pinned at `42c743118a29db3270ae6a921a4a4eecb20cad04` in [Intelligent-Microsystems-Lab/SplitZip](https://github.com/Intelligent-Microsystems-Lab/SplitZip/tree/42c743118a29db3270ae6a921a4a4eecb20cad04). ExANS is pinned at `de6c38bcc5438475bd1abf1e9b3d154cef414ef6` in [openlake-project/openlake](https://github.com/openlake-project/openlake/tree/de6c38bcc5438475bd1abf1e9b3d154cef414ef6/crates/openlake_kv_client/cuda). Their existing checkouts must have those revisions and no edits to the imported/build sources. The runner never checks out a different revision or installs packages.

## Same data, paid bytes and timing scope

All methods receive the same captured words in `[28 layers, 8192 tokens, 2048 K/V columns]` order from the frozen WikiText2 windows `ds_wikitext2_r0` through `r3`. Capture-file and word hashes must match the completed T head-spread reference. T is never refitted; SplitZip's calibration uses the frozen reference's first TRAIN document, with no TEST selection. SplitZip uses 8,192 calibration tokens, while the already frozen T predictors used all six TRAIN documents. This compares their stated training procedures rather than equal training compute.

Warm encode, decode and sequential pair are timed separately with CUDA events and a wall clock. All forward prediction, authors' internal CPU metadata work/synchronizations, and raw fallback/tail copies remain inside the codec call. Each method retains its native allocation policy: T and ExANS reuse output/arenas, while SplitZip allocates its encoded components and output through the authors' API. The difference is explicit in each worker's setup record.

Full defaults are four windows, three rounds, three warmups and ten samples per operation per round. Method and window orders rotate across rounds; each codec/round has a separate process. A native crash is recorded as an error and the remaining methods continue. The aggregate uses the median across rounds for each window, followed by the mean across windows. It never infers pair time by adding the independently measured encode/decode medians.

Every value and all required headers/directories count toward packet bytes:

- T retains the exact earlier packet format. Its entire packet SHA-256 must match the frozen reference on every round. Arenas are cleared between windows, outside timing, to preserve paid alignment bytes.
- SplitZip transfers packed exponents, sign/mantissa bytes, chunk IDs, local escape positions and escape values. The native counts/starts arrays are construction scratch; the receiver reconstructs from only the paid fields and shared lookup tables.
- ExANS sends fitted records up to each native header's `actual_bytes`, a 32-bit length directory, complete raw blocks rejected by the native compressor, and the uncompressed final partial block. The receiver calls native decode only for fitted record indices. The 896 MiB reference window has 940 whole blocks plus **32,768 remaining words / 65,536 paid raw tail bytes**. The old timing helper's whole-block truncation is not used.
- External packets have a checked comparison envelope describing component dtypes, sizes, shape and shared-state identity. Its actual bytes are paid. T uses its existing checked envelope. Serialization is local little-endian and does not claim platform-independent ABI portability.

After timing, the runner serializes and restores each complete packet and checks every decoded BF16 word. In round 0, every method/window also gets a fresh receiver process that takes only shared state and the packet, with an expected digest for verification. It receives no capture path, original KV tensors, queries or per-token hints. Preflight covers all 65,536 BF16 patterns for SplitZip and the existing dense forward check; native ExANS additionally must pass a compressible block and a random fallback block test, both with a partial tail. All of these native checks execute on CUDA in the NUC run.

GPU setup, state file size, resident shared tensor bytes, Torch peak increments, input staging, packet packing/D2H and expansion/H2D are separate report fields. ExANS allocates opaque CUDA workspace outside Torch; setup device-free-memory delta includes it, and Torch-only peak figures do not pretend to measure it. ExANS is built in the new result directory with the existing documented one-line allocation-macro compiler fix, then the authors' own executable test must pass. Codec algorithms and original source checkout remain unchanged.

`results.json` includes ideal serial-link times at 3, 10, 25, 50, 100, 200 and 400 Gbps using measured **wall** pair time plus complete packet bytes / bandwidth. These are modeled times with resident shared state and no overlap. This run measures no real link, storage, cold state transfer, model quality or inference throughput.

## NUC commands

Reuse `~/kv_a5000/venv` with Torch `2.11.0+cu128`, Triton `3.6.0`, existing captures, and `~/kv_a5000/runs/t_head_spread_v1_20261113T151348Z`. Native sources default to `~/kv_a5000/sources/{splitzip,exans}`. The existing CUDA toolkit path is read from `~/kv_a5000/cuda_home`, or set `CUDA_HOME`. Source overrides are `SPLITZIP_SOURCE` and `EXANS_SOURCE`. Missing dependencies are reported as unavailable; no baseline is silently replaced with a CPU implementation or estimated code cost.

First run one complete reference window through all five methods:

```bash
cd ~/kv_t_head_spread_v1 &&
git pull --ff-only origin experiment/t-head-aware-spread-v1 &&
unset TRITON_INTERPRET &&
KV_HOME=~/kv_a5000 SMOKE=1 bash scripts/run_t_baselines_nuc.sh
```

After the smoke run succeeds, run the full comparison:

```bash
KV_HOME=~/kv_a5000 bash scripts/run_t_baselines_nuc.sh
```

To diagnose one adapter, use `METHODS=exans` (or `splitzip`). A subset is explicitly not a complete five-method comparison. `full_comparison_validated` requires all five methods, every reference window, at least three rounds, stable paid sizes and exact reconstruction. It never authorizes an automatic merge.

Results are written only to a new `~/kv_a5000/runs/t_native_baselines_v1_<UTC timestamp>` directory. `share.zip` contains summaries, timing/validation JSON and worker logs. It excludes packets, frozen `.pt` state, native libraries and captures. It does not commit or push anything.

## Next stage: AQUA-KV

AQUA-KV discards information, so it needs a separate quality comparison. Its authors' environment pins Torch 2.5.1 rather than T's 2.11.0. Keep that environment separate, reuse the same captures and TRAIN/TEST split, count actual packed buffers, and measure complete predictor + quantizer encode/decode calls together with reconstruction error and a model-quality evaluation. A nominal 2-bit estimate, synthetic random predictors or a floating-point quantize/dequantize proxy is not a measured transfer codec. The exact stage above makes no AQUA-KV speed or quality claim.

Software tests exercise T packet parity, complete BF16 transfer, missing/corrupt packet rejection, ExANS raw/fitted subsets and partial tails, and incomplete-run aggregation. They validate software behavior only; native GPU results remain pending the NUC runs.
