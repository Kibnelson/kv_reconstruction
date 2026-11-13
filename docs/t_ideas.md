# Five ideas for T, and how each is tested

Branch `t-ideas`, from `llama32-3b-lossy-lossless-compare` at `d95fd10`. Nothing that works today is changed: the
branch only adds files.

- `kv_compare/t_ideas.py`: the measurements;
- `scripts/t_ideas.py`: the driver;
- `scripts/run_t_ideas.sh`: the runner, which also packages and pushes;
- `kv_compare/test_t_ideas.py`: CPU tests on a tiny random Llama.

T is our layer-wise one-step transfer codec. Per layer, it predicts each BF16 value from the token's previous layer
(int8 features times int8 coefficients, RoPE handled), then codes the word against a logistic distribution. The
codec is bit-exact.

The ideas came from reading the SCLQ vector-search repository against T, plus two levers inside T. Each test gives
one of two kinds of number:
- an exact cost: -log2 P of every BF16 word, the same accounting as `kv_compare.predict`;
- a GPU time.

Anything a variant sends is counted in bits per value. This is SCLQ's rule, "count every byte, time not operations".

## The ideas

| # | Idea | Why it could help T | Cost to deploy if it wins |
| --- | --- | --- | --- |
| 1 | **Per-request adaptation** of T's bias and spread | T is calibrated on WikiText-2 train. Other text costs more: one-step exact HumanEval 9.535, GSM8K 9.179, MMLU 9.208, PTB 9.141, against WikiText 8.869 bits/value | Small. The kernels already read a bias `b` and a spread `sp` per channel and layer, so a request just brings its own arrays. A forward variant sends them: L x W x 14 bits, about 0.002 bits/value at 8K tokens. The backward variant needs deterministic sums on both sides |
| 2 | **Layer 0 computed on the decode side** | A token's layer-0 K and V depend only on its id and position: embedding, norm, K/V projections, RoPE, and no attention. T codes layer 0 against its calibration mean, its most expensive layer. The receiver has the token ids and the model | Moderate. One small layer-0 pass on the receiver; then layer 0 is coded against the recomputed values (exact, or nearly free) |
| 2b | **Per-layer mode allocation** | The modes trade bits for decode time differently on every layer. They are dense, keys then values, 4 head groups, head by head and rank 256. SCLQ's water-filling idea is to spend effort where it pays | A mode table per layer. Every mode predicts from the previous layer's features, so modes can mix |
| 3 | **Leaner coding kernel** (SCLQ's "delete the decode step": table-free slice decode) | The coding kernel is a floor. On the 3B on the A4500 it alone takes 15.8 ms, whatever the product: 15.0 ms for the dense product, 5.6 ms for rank 256. Whole decodes take 31.6 ms dense and 23.8 ms rank 256 (`benchmark_results/llama32_3b_compare_a4500_20261029_1232/gpu_speed.json`) | New kernel work. It is worth it only if the kernel is far above its memory floor |
| 4 | **Anchors with key frames** (SCLQ's anchor residuals and nearest neighbour) | Our strongest no-hint predictor adds layer l at the token's most recent earlier occurrence, but that forces sequential waves | Two waves per layer. The token permutation follows from the token ids or from layer l-1, so nothing is sent. The nearest-key-frame form costs an N x N/K x W matrix product per layer |
| 5 | **Token classes** with their own predictors (SCLQ's per-region codebooks were negative there; they are tested here on KV) | Tokens may fall into a few predictable regimes (sinks, repeats, clusters) | One fit per class. The encoder's choice costs log2 k bits per token and layer, about 0.001 bits/value |

## What each stage measures

`scripts/t_ideas.py --run RUN --output OUT --stages ...` works on the captures of a finished `scripts/run_a5000.sh`
run folder:
- calibration: wtrain0-5 and wtables, from WikiText-2 train;
- dataset windows: `ds_<name>_r<i>`, 8,192 tokens each.

**capture** adds the dataset windows still missing: WikiText-2 test, HumanEval, GSM8K, MMLU and PTB, two windows
each. It uses the run's model and the texts of `scripts/fetch_eval_texts.py`.

**adapt (idea 1)** computes T's exact-model bits in torch, from T's own int8 constants (`fused.prepare`,
`fused.reference_mu`). The `gpu` stage checks this reference against the kernel's own features and priors. Variants:
- `T`: as it is.
- `fwd-spread/ch`: the request's own spread per channel and layer, 6 bits each, sent.
- `fwd-bias+spread/ch`: the request's own bias (8 bits, in steps of sp/8) and spread, sent.
- `fwd-spread/head`: one spread scale per head and layer, sent.
- `bwd-pilot8`: every 8th token is decoded first with the calibration constants. The receiver measures bias and
  spread on those tokens and uses them for the other 7/8. Nothing is sent.
- Two oracles, neither sendable. Both are the float one-step predictor fitted on the window itself:
  - in-sample: an optimistic bound;
  - cross-fitted: blocks of 512 tokens in two folds, each fold coded with the other fold's fit.

  The room for adaptation is T minus the cross-fitted oracle.
- `float one-step`: the same fit as T before 8-bit rounding (`kv_compare.predict`); it isolates the rounding.

**layer0 (idea 2)** recomputes layer 0 from the token ids. It runs only the model's first decoder layer, through
the same code path and prefill chunks as the capture. Reported per window:
- the share of words that come out bit for bit;
- the cost of coding the captured layer 0 against the recomputed one: a flag for equal or not, then unequal words
  against a logistic around the recomputed value, with parameters from wtables;
- T's layer-0 cost and the saving spread over all layers;
- the recompute time.

**anchor (idea 4)** and **classes (idea 5)** use the float one-step predictor. Each class (with anchor or without,
sink or not, and so on) gets its own conditional-mean fit, spread and adaptive multiplier, fitted the same way as
`kv_compare.predict`. With one class this machinery reproduces `Study.lossless` exactly, and a test checks that.
Gains are against the float one-step. Anchor forms:
- most recent occurrence: shown for reference; its waves equal the longest repeat chain;
- first occurrence: 2 waves;
- nearest key frame for K = 8 and K = 4: key frames t % K == 0 decode first, then every other token takes the layer
  l of the key frame nearest to it in layer l-1 (cosine, on the receiver's own standardised layer l-1); 2 waves.

Class forms:
- sink tokens 0-3;
- repeated tokens;
- k-means of layer l-1 with k = 4, assigned by the decoder for free;
- the encoder's pick of 1 of 2 or 1 of 4 predictors (k-means start, one hard-EM refit), with the side bits counted.

**gpu (ideas 2b and 3).** Every mode is built with the PD codec's launch choices and run on one 8K window. The stage
records:
- bit-exact decode;
- bits per layer, from the encoder's per-layer word counts;
- decode time per layer, from events at each layer's hand-over;
- the whole decode, the products alone and the coding alone.

It also computes the coding kernel's memory floor on this GPU: int32 sums in, words and features out, the stream in,
at the bandwidth of a measured device copy. The allocation then picks a mode per layer for each link (1, 3, 10, 25,
100, 400 Gb/s) and minimises two things:
- max(link time, decode time) for the cache, the pipelined case, by dynamic programming over a time grid;
- or their sum, layer by layer.

Each is compared with the best single mode.

**report** writes `summary.md` with the tables and the verdicts below.

## Decision rules (set before the numbers)

| Idea | Build it if |
| --- | --- |
| 1 adaptation | The best sendable variant (forward with side bits, or backward) saves at least 1% of T's bits on some dataset. The cross-fitted room shows how much more a richer adaptation could reach |
| 2 layer 0 | At least 99% of layer-0 words recompute bit for bit and the saving is at least 0.1 bits/value over all layers |
| 2b allocation | A per-layer mix beats the best single mode by at least 5% at some link between 3 and 100 Gb/s |
| 3 coding kernel | The coding alone takes at least 2x its memory floor. The prize is at most (coding - floor) ms per request |
| 4 anchors | A two-wave form saves at least 0.1 bits/value against the float one-step. Many-wave forms are shown, not counted. For key frames, the extra matrix product must fit inside the decode time it saves |
| 5 classes | A form saves at least 0.05 bits/value net of side bits |

## Running it

On the NUC, after `scripts/run_a5000.sh` has made `~/kv_a5000/runs/llama32_3b` (the calibration captures at least):

```bash
bash <<'NUC'
set -Eeuo pipefail
cd ~/kv_llama32
git fetch -q origin t-ideas && git checkout -q t-ideas && git pull -q --ff-only origin t-ideas
export ALLOW_NO_SLURM=1 KV_HOME=$HOME/kv_a5000
SMOKE=1 MODEL=llama32_3b bash scripts/run_t_ideas.sh
MODEL=llama32_3b bash scripts/run_t_ideas.sh
NUC
```

- The smoke takes a few minutes: 4 layers, one window, nothing committed. Its success marker is `SMOKE OK`.
- The full run takes about 1.5-2 hours on an A4500. Its success marker is `PUSHED benchmark_results/llama32_3b_tideas_<gpu>_<date>`.
- An interrupted run continues where it stopped: finished stages and windows are kept in `t_ideas.json`.

On an A5000 node, run the same commands inside the allocation, without `ALLOW_NO_SLURM`.

## Honest limits

- The anchor and class studies use the float predictor, not T's int8 arithmetic. Their gains are differences against
  the float one-step. The adapt stage shows the rounding is small: compare the float one-step with T.
- The per-layer allocation takes each layer's bits and time from whole-mode runs. A real mix would change the
  next layer's spread prior a little.
- Backward adaptation needs encoder and decoder to compute the same statistics. Deployed, they would be fixed-point
  sums like T's spread prior; the estimate uses float sums.
- Recomputing layer 0 is exact only when the receiver runs the same kernels as the sender: the same GPU type and
  libraries. Elsewhere, unequal words are coded at their measured cost, so the codec stays lossless; only the saving
  shrinks.
- The oracles fit up to 2W inputs on one 8K window. The in-sample one is optimistic by construction, so the room is
  taken from the cross-fitted one.
