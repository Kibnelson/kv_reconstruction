# Qwen KV reconstruction benchmark

Copy **`qwen_kv_gpu_bench.py`** to the target machine. It embeds the project code; no repository checkout or model weights are bundled. Python packages and the pretrained checkpoint are downloaded separately.

This runner compares exact KV representations on **Qwen/Qwen3-4B-Instruct-2507**, pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`. It fits its learned dictionaries from this checkpoint's own calibration KV; it does not reuse SmolLM dictionaries.

Prefill recipe insight update: `--native-preset insights` tests **33 native methods**: the 16 main baselines/recipes, six matched correction-layout/group-size variants, seven exponent-tier controls/predictors, and four dictionaries fitted on affine residuals. All original BF16 bits are recovered. Every predictor, ID, correction, stream offset, padding byte, header, and full decoder state is charged. See [the experiment design](../docs/prefill_recipes.md) for the exact incremental comparisons and limits; collecting metadata during prefill alone does not establish novelty.

Add `--prefill-recipes` to construct and verify selected recipes directly from device KV between prefill chunks, before CPU export. The transformer still retains full raw KV, and capture prefill time includes all observer methods and verification work. Check `prefill_observer` separately from repeated post-capture `native_packet` measurements. New insight-method GPU results require a target run.

Cache compatibility update: this revision reads modern `cache.layers[i].keys/values` storage and legacy cache interfaces. It no longer requires `to_legacy_cache()`, which newer Transformers versions removed. Capture metadata records the actual cache class; errors include the cache class and installed Transformers version. Existing installations can use the updated script without downgrading Transformers. Run `--doctor` to record the versions used by your experiment.

## Install and check the machine

Use Python 3.12 in a virtual environment. **PyTorch must be installed before `--doctor`, `--self-test`, or a model run.** Installing only NumPy/Transformers, or the repository's `requirements.txt`, does not install Torch.

The following complete setup is for the **NVIDIA RTX A4500 NUC** and works with just the copied standalone script:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.3.5 transformers==4.57.6 safetensors
python -c "import torch; print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
python qwen_kv_gpu_bench.py --doctor --device cuda
python qwen_kv_gpu_bench.py --self-test
python qwen_kv_gpu_bench.py --list-methods
```

With a repository checkout, `python -m pip install -r requirements-cuda.txt` replaces the two package-install commands above. If the environment already exists, activate it and run the installation/check commands; there is no need to recreate it or download model weights again.

This selects a **Torch 2.11.0 CUDA 12.8** wheel for the A4500 and driver 580. The CUDA 13.0 value in `nvidia-smi` is driver capability; it does not require installing a CUDA 13.0 Torch wheel.

For other hardware, install **the PyTorch build appropriate to the actual accelerator** using the [official installation selector](https://pytorch.org/get-started/locally/), then install NumPy, Transformers and safetensors as above. Intel GPU support uses the XPU build and Intel GPU drivers; see [PyTorch's Intel GPU guide](https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html). A NUC model name alone does not establish accelerator support. Older Intel integrated GPUs are not universally supported. A CPU-only Torch build supports the functional tests but cannot run the NVIDIA GPU benchmark.

On Windows, activate with `.venv\Scripts\activate` instead. `--doctor` and `--self-test` do not download language-model weights. The self-test exercises CPU functional correctness; it does not validate a GPU driver or GPU performance.

`--device auto` selects CUDA, then Intel XPU. It does **not silently fall back to CPU**. Select `--device cuda` or `--device xpu` explicitly when needed; `cuda:1` is also accepted. XPU operation coverage depends on the installed driver/build. Unsupported methods are recorded individually. CPU functional runs require explicit `--device cpu` and are labelled CPU runs.

## Start with a short run

The first model download is roughly 8 GB. Start with a small method set and one test fixture:

```bash
python qwen_kv_gpu_bench.py --device cuda --preset smoke \
  --lengths 2048 --cases 1 --train-tokens 256 --dev-tokens 256 \
  --repeats 3 --generation-steps 2 --output qwen_smoke
```

Replace `cuda` with `xpu` for a supported Intel installation. To test the script on CPU without claiming GPU results, pass `--device cpu`; a 4B CPU model can be slow. `--model` also accepts a compatible smaller checkpoint or local model directory. The report records the actual model identity; custom-model results must not be labelled Qwen-4B results. Local-directory provenance records the path and config identity, with `local_weights_content_hashed: false`; it does not fingerprint every weight file.

The full family comparison uses all registered CPU formats and the expanded native tensor set:

```bash
python qwen_kv_gpu_bench.py --device cuda --preset all --grid \
  --lengths 4096 16384 --cases 3 --max-records 72 \
  --measure-io --io-mib 32 --repeats 3 --output qwen_all
```

For the incremental prefill-recipe study, use the focused GPU sweep below. It
keeps two CPU size controls and forces full-prefix checks for three new methods:

```bash
python qwen_kv_gpu_bench.py --device cuda --preset smoke \
  --methods raw exponent_huffman_lower_deflate \
  --native-preset insights --prefill-recipes --tile-tokens 64 \
  --lengths 2048 8192 --cases 3 --max-records 144 \
  --dev-tokens 512 --generation-steps 2 \
  --generation-methods \
    native_prefill_exponent_median_time_tiered \
    native_prefill_affine_dictionary16_channel_g8_residual \
    native_prefill_affine_time_g8_residual \
  --output runs/qwen_insights_t64
```

`insights` automatically enables `--recipe-diagnostics`. Its per-record traces
support `recipe_analysis.json`: same-record incremental byte comparisons, K/V
and component breakdowns, timing per raw MiB, and a **stateless size-only
selection oracle**. The oracle counts a mode byte per record and full framing,
but is not an online prefill selector or a measured latency improvement. It
excludes stateful dictionaries. Dictionary results separately count IDs (0.5
or 0.75 bits/value for 16 or 64 entries) and full frozen state. The `fixed127`
exponent control isolates the value of tile-derived exponent metadata. See the
design document for the curated-results commit/push commands.

**The default samples records.** This makes a wide sweep practical, but full-cache byte totals are estimates. `--max-records 72` measures an evenly spaced sample of actual records across the cache; estimates include the full temporal index and complete selected decoder state. Sampling does not establish full-cache correctness for unmeasured records. Add **`--full-cache`** to encode and verify every record. A full-cache sweep over all methods at long contexts can take substantially longer, especially for the Python CPU codecs.

The literal defaults are `--preset all`, lengths 2,048 and 8,192, three fixtures and 72 sampled records per cache. Start with the explicit smoke command above before using those defaults. The complete outer-cache header and temporal index are arithmetic byte charges; the benchmark measures actual inner records but does not persist and reopen a complete indexed cache container, including in `--full-cache` mode.

`--preset main` is a smaller representative comparison; `--preset smoke` is the quickest check. `--grid` adds parameter/control variants to the CPU registry. Use `--methods` and `--native-methods` with exact names from `--list-methods` to narrow a run. `--native-preset main|all|insights` controls the native tensor set; `insights` adds targeted experiments to main rather than to all. `--skip-native` disables the native track. Each `--output` directory must be fresh or empty, preserving previous results.

## What runs where

| Track | What is measured | Interpretation |
|---|---|---|
| Original CPU formats | Existing serialized codecs encode/decode real KV captured from the selected model/device | Exact original format sizes and CPU implementation costs |
| CPU recovery and transfer/I/O | Decode on CPU, stage/copy recovered KV to the accelerator, and optionally write/read bounded temporary files | Staging and transfer diagnostics; buffered reads may hit the OS page cache |
| Native tensor packets | Bit-packed tensor implementations execute on the selected device | Separate native format variants with their own bytes/state/correctness results |

The original Deflate/Huffman formats remain CPU codecs. Their rows are not GPU decompression results. Native packet methods represent the same broad ideas—raw fields, exponent packing, predictors, E8 or binary residual descriptions—but need not be wire-compatible or have identical metadata to the CPU format. Their ratios and times must be read as separate implementations. This is not an AQUA-KV, PackKV, RaBitQ ANN or published E8 quantizer reproduction.

Exact recovery compares original BF16 words, including the target model's actual post-RoPE KV. All measured record payloads, modes, scales, correction bits, padding and metadata count; the temporal index is charged as described above. Learned dictionaries and rotations count once as decoder state. CPU reference rows charge complete serialized decoder files; native rows report distinct tensor storage. New affine-residual dictionaries conservatively charge the larger of their complete serialized model and physical device tensor state, reporting both. These are explicitly different state contracts. Matched binary comparisons use the same transform description where supported. A reconstruction prototype is accepted only after its own exactness check; unsupported or failing methods remain visible.

CPU formats, native tensors and transfer paths report their own timing boundaries. Accelerator measurements synchronize appropriately; CPU work is not inferred from device events. Repeats describe the local implementation, not a statistically controlled serving study. `--measure-io` measures buffered temporary-file write/fsync/read, not guaranteed cold physical-SSD bandwidth. The timing sample is capped by `--io-mib`.

## Calibration, evaluation and memory

Training, development and test fixtures are separate. Calibration uses only training KV. Development chooses the reported full-prefix generation check independently for each track. Sources, state and settings are recorded before final test captures. Synthetic topics and repeated filler provide coverage, not representative workload or task-quality evidence. More than one length of a fixture is correlated.

By default, a development-selected method per track restores the full development prefix for two successive model steps and compares logits, tokens and appended KV. `--generation-methods NAME ...` requests checks for named candidates, so a new recipe does not escape continuation validation merely because an older method wins DEV selection. `--dev-tokens 512` sets the checked prefix length; `--generation-steps 0` disables checks. A check may be skipped with an explicit warning if the memory estimate does not allow the additional caches. A skipped check is not a pass.

The official Qwen config has 36 layers, eight KV heads and explicit head dimension 128. Thus BF16 KV requires

`36 × 2 × 8 × 128 × 2 = 147,456 bytes per token`.

| Context | Raw KV, one sequence |
|---|---:|
| 2,048 tokens | 288 MiB |
| 8,192 tokens | 1.125 GiB |
| 16,384 tokens | 2.25 GiB |

The BF16 weights alone occupy about **7.49 GiB**, before KV, attention workspaces, allocator reserve and codec buffers. Preflight estimates and `--memory-reserve-gib` reduce avoidable failures but do not guarantee a run will fit. Quantized model weights and automatic CPU offload are not enabled by this experiment. Use a shorter context or an appropriate smaller model when necessary.

`--chunk-tokens` controls prefill chunks. It reduces the size of an individual attention workspace; the complete prompt KV still resides on the model device. Chunking does not remove quadratic total attention computation. Reconstruction measurements and generation checks do not implement a transformer whose attention consumes only compact resident KV. No concurrency, TPOT/TTFT, admission-capacity or end-to-end serving improvement follows from this benchmark alone.

## Inspect, share and extend

The output contains JSON results, CSV tables, a Markdown summary, model/source/calibration provenance and a shareable ZIP written beside the results directory. The ZIP excludes model weights and raw prompt KV. Exact method status, measured versus estimated size, backend and skipped checks are part of the result; inspect them before comparing rows.

```bash
python qwen_kv_gpu_bench.py --instructions
python qwen_kv_gpu_bench.py --extract-source extracted_kv_bench
```

The single file checks and extracts its embedded source bundle for execution. Extraction exposes the code for review; it does not require a compiler for the Python/tensor paths. Installed packages and model files remain external dependencies.

Local delivery checks exercise the bundled code and functional CPU paths. **CUDA/XPU hardware execution and Qwen-4B performance could not be validated in the authoring environment.** Run `--doctor`, the short comparison and the exactness checks on the target accelerator before interpreting GPU results. Preserve a completed output directory when changing settings or extending the method set.

Primary model references: [official config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json), [official model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
