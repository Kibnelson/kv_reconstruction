# Initial prefill-recipe diagnostic

This is a CPU functional/size diagnostic on an existing public-fixture KV
capture from **SmolLM2-135M**, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. It is **not** a new Qwen-4B run,
a fresh held-out evaluation, GPU speed evidence, or an inline-prefill timing.
The capture's canonical raw hash is checked against the earlier capture
manifest. No dictionary is trained or loaded for the new recipes.

All measured records reconstructed exactly. The sample has 512 tokens, 30
layers, and 192 channels per K/V tensor. The test measures 120 records per
method: 120/1,920 with 16-token tiles and 120/480 with 64-token tiles. Savings
below are **full-cache estimates**, including recipe/header/index costs.

| Method | 16-token tiles | 64-token tiles |
|---|---:|---:|
| CPU exponent + Huffman/Deflate | 31.57% | 32.54% |
| Native exponent packing | 16.05% | 22.59% |
| Prefill midpoint + cosets | 8.34% | 19.17% |
| Prefill affine + cosets | 9.64% | 22.05% |
| Prefill E8 template + cosets | 7.84% | 18.95% |

The current recipes work exactly but do not beat exponent packing on this
sample. Larger tiles amortize metadata better. E8 reduces midpoint-template
payload bytes, but its additional packet descriptor costs erase that benefit
here. Coset and matched circular-residual payloads have identical bit counts;
small total-size differences arise from header spelling.

The next experiment is the A4500 Qwen run in
[the recipe guide](../../docs/prefill_recipes.md), including actual device
encoding during prefill and separate post-capture restore timings. That run
is needed to assess whether this model's block patterns justify recipe costs.

Local correctness validation also covers all 65,536 BF16 bit patterns for
each new method, independent NumPy oracles, and two exact continuation steps
for all eight methods on a small randomly initialized BF16 Qwen model. Both
Transformers 4.57.6 and 5.20.1 suites passed 68 tests with two CUDA skips each.
CUDA hardware is unavailable in the authoring environment.
