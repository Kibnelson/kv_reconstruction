# Prefill-derived KV reconstruction recipes

The experiment asks whether information collected while KV is available during
prefill can make a smaller exact reconstruction record than our existing
exponent or learned-dictionary representations. It does not assume a recipe
can regenerate arbitrary KV without storing the missing information.

## What is new

Each record is a token tile of one layer's K or V, shaped `[tokens, channels]`.
The encoder builds a predictor from that tile, stores its complete description,
and adds exact correction information. No hidden states or other records are
needed by the decoder. There is no fitted dictionary for these methods.

| Predictor | Information collected from this tile | Stored predictor state |
|---|---|---|
| `zero` | None; matched control | None |
| `midpoint` | Per-channel minimum and maximum over the tile's token rows | Their integer midpoint, two bytes per channel |
| `affine` | First and last token's magnitude symbols | Two channel vectors; deterministic integer interpolation recreates the intermediate predictions |
| `e8` | Per-channel midpoints | Minima per 64 channels and genuine E8 coordinates for midpoint offsets; one reconstructed channel template is reused over token rows |

All arithmetic targets **integer BF16 magnitude bit patterns**, not absolute
floating-point KV magnitudes. Signs are stored exactly. Affine interpolation
uses floor division in both encoder and decoder. E8 uses a fixed step of 64
symbols, eight-dimensional vectors, deterministic nearest-point tie handling,
and repeat-last padding for a channel tail. E8 in this experiment describes a
channel template; the older `native_e8_symbol_bitpack` describes every token.

## Cosets and a matched control

For original magnitude symbol `x` and prediction `p`, define the circular error

`d = ((x - p + 16384) mod 32768) - 16384`.

For each eight-symbol correction group, let `b` be the bit width of its largest
ZigZag-encoded error. Store `b` in four bits and `r = x mod 2**b` in `b` bits per
symbol. The decoder picks the representative near `p`, on the 15-bit circle.
Width zero means the prediction is exact; width 15 carries the complete
magnitude symbol. Negative half-modulus ties are defined explicitly. Every
original BF16 word is recovered, including zeros, infinities and NaN payloads.

Each predictor is tested twice:

- `native_prefill_<predictor>_coset` stores these modular residues.
- `native_prefill_<predictor>_residual` stores circular ZigZag errors with the
  **same predictor, group size and width selection**.

These matched correction streams have identical bit counts. Cosets are a way
to use a predictor for reconstruction, not an additional source of entropy
savings. The experiment tests predictor quality and total metadata cost.
Small total packet-size differences between paired methods can be due to JSON
header spelling. New methods force their predictor representation, even when
it expands the record; there is no hidden raw fallback or TEST-based choice.

## Execution during prefill

`--prefill-recipes` enables a synchronous observer after each model prefill
chunk. It reads selected complete KV tiles still on the selected device,
constructs a recipe there, decodes it there and verifies original word equality.
A tile spanning two prefill chunks is encoded when its final row is available.
The source KV does not pass through CPU before this observer encodes it.

The report separates two tracks:

- `prefill_observer`: direct device KV encoding between prefill chunks, with
  one verification decode per selected record. Capture prefill timing includes
  observer work, layout copies, synchronization and exactness guards.
- `native_packet`: the usual post-capture encode, repeated decode and packet
  transfer measurements. This track still exports the full source cache to CPU
  and stages selected raw tiles back to the device for the experiment.

The observer discards each recipe after verification. **The transformer still
keeps its complete raw KV cache.** This does not yet reduce the model's memory
requirement, fuse encoding into attention, implement compact attention, or
demonstrate concurrency, admission or TTFT/TPOT improvement.

All actual predictor fields, signs, corrections, width nibbles, padding and
JSON packet headers count toward stored bytes. The existing outer cache header
and offset table are charged arithmetically. Full-cache sizes remain estimates
when record sampling is enabled. `prefill_recipe` in the post-capture JSON rows
breaks down predictor, correction, width, sign and header costs. External
decoder state is zero for these self-contained recipes.

## Run on the A4500

From the updated repository, with the existing virtual environment active:

```bash
git pull --ff-only
python qwen_kv_gpu_bench.py --self-test
for tile in 16 64; do
  python qwen_kv_gpu_bench.py --device cuda --preset smoke \
    --methods raw exponent_huffman_lower_deflate learned_entropy_l1 \
    --prefill-recipes --tile-tokens "$tile" \
    --lengths 2048 --cases 1 --max-records 144 \
    --output "runs/qwen_prefill_t${tile}"
done
```

The native smoke set now includes the eight new variants plus the eight older
native baselines. After checking this run, increase lengths to `2048 8192` and
cases to `3`, using fresh output directories. Comparing tile sizes tests the
tradeoff between amortizing template metadata and prediction error over longer
token spans. `--measure-io` remains optional buffered I/O, not a physical SSD
bandwidth proof. Nothing in the new methods adds a GPU decoder for the old
CPU Huffman/Deflate wire formats.

## Local validation

Tests exercise independent NumPy predictor/remainder oracles, all 65,536 BF16
patterns for all eight variants, wraparound, half-modulus ties, empty/singleton
tiles, channel tails, actual packet byte accounting, and packet restoration.
A small randomly initialized BF16 Qwen model checks the observer and two
continuation steps for every new method. These tests establish functionality,
not pretrained Qwen-4B GPU performance. CUDA tests skip explicitly without GPU.

`scripts/bench_prefill_sample.py` additionally evaluates an existing uint16 NPZ
capture without downloading a checkpoint. Its reports label reused-sample
diagnostics explicitly; they must not be reported as a fresh held-out result
or an inline/GPU measurement.
