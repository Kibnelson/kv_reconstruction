# Five ideas for T, and how each is tested

Branch `t-ideas`, from `llama32-3b-lossy-lossless-compare` at `d95fd10`. Nothing that works today is changed: the
branch only adds files.

- `kv_compare/t_ideas.py`: the measurements;
- `scripts/t_ideas.py`: the driver;
- `scripts/fetch_calib_texts.py`: the calibration texts of the calib stage;
- `kv_compare/t2.py`: T2, the codec that carries the winning ideas, kept apart from T (tests: `kv_compare/test_t2.py`);
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

**layer0x (2c, the checks before building idea 2).** The receiver's recompute the lean way. The model runs only up
to its first attention module, where a hook takes the inputs; then the module's own K and V projections and RoPE
run. That is the embedding, the input norm, two projections and RoPE: no attention, no MLP, no Q. The stage records:
- the share of K and V words that come out bit for bit, under five prefill shapes:
  - one pass, as the capture ran it;
  - chunks of 2,048 tokens and of 512 tokens;
  - one token, or 2,048 tokens, of another request ahead of it in the same pass (vLLM flattens a batch into one
    matrix product, so its row count changes);
- the largest difference in BF16 steps;
- the cost of coding the captured words against the recomputed ones: each word is coded as its difference in BF16
  steps (-8..8, with an escape to the raw word), with the table from wtables under the same shape. A mismatch costs
  bits, never correctness;
- the time of the lean recompute and of the whole first layer.

**calib (6, calibration breadth).** The first run showed the dataset gap is mostly the predictor's coefficients. A
predictor fitted on half a HumanEval window beats T by 0.65 bits/value; bias and spread recover about a fifth.
`scripts/fetch_calib_texts.py` writes calibration text disjoint from every test window:
- code: MBPP, all splits (HumanEval's kind of text, not HumanEval);
- math: GSM8K train;
- exam: MMLU dev and validation (the test's format, other questions);
- news: Penn Treebank train.

The stage captures 3 windows of each, plus a scales window, plus a mixed scales window (an equal share of each source
and of WikiText-2 train). It then fits T's constants for each set:
- wikitext (today's);
- mixed (6 documents, like today's: 2 WikiText-2 train and 1 of each domain);
- mixed-all (all 15 documents);
- each domain alone.

It codes every dataset window with each set, with and without the request's own bias and spread. All rows are with
layer 0 recomputed. Three per-request picks are shown, the sender sending the set's number:
- WikiText or a domain set, chosen on layers 1-4 (cheap: the sender encodes 4 layers per set);
- the same, chosen on the whole cache;
- any set, chosen on layers 1-4.

The cross-fitted oracle of the adapt stage shows the room.

**layer0m (2d).** The second run showed that chunked prefill breaks the recompute's bit identity, while other
requests in the same pass do not: what matters is the size of the step's matrix product, not what the other rows
hold. The sender here computes layer 0 in simulated vLLM steps of at most 2,048 rows:
- 2,048 of its own tokens;
- 1,024 of its own after 1,024 of another request's;
- 1,536 of its own before 512 of another's;
- 512 of its own after 1,536 of another's.

Four receivers recompute it:
- in one pass;
- in the same chunks, without the other rows;
- padded with dummy rows to the sender's step size at the sender's offset (the sender sends its step sizes, a few
  bytes);
- padded with its own rows first.

The stage reports the share of K and V words that come out bit for bit and the cost of coding the rest. The cost
is the better of the step coder and a value coder: a flag, then a logistic around the recomputed value with a
per-channel spread, so a sign flip next to zero is cheap.

**calib_loo (6b).** Section 6's mixed set and pick include each test dataset's own kind of text. This stage
measures text the calibration never saw. For each dataset, mixed-all is refitted without its source:
- WikiText-2 train for WikiText-2;
- MBPP for HumanEval;
- GSM8K train for GSM8K;
- MMLU dev for MMLU;
- PTB train for PTB.

Its scales document leaves that source out too. The per-request pick is also run without the source's own set.

**t2 (7).** T2, the separate codec (`kv_compare/t2.py`), measured on the GPU:
- one window per dataset in dense mode: the receiver's lean layer 0, the sender's check that it is bit-identical,
  the set picked on layers 1-4, then encode, decode and comparison with the captured KV;
- bits of the real streams against T's, also with T's own constants (layer 0's share alone);
- decode times of T and T2, and the recompute;
- then every mode on one window (HumanEval's, to use a set other than WikiText's).

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
| 2c layer 0, robust | The saving stays at least 0.3 bits/value in every prefill shape (most of the 0.355 measured) and the lean recompute takes at most 10 ms per 8K window |
| 6 calibration breadth | A mixed set or the per-request pick chosen on layers 1-4 saves at least 1% of T's bits over the five datasets (the WikiText change is reported) |
| 2d layer 0, steps known | Padding to the sender's step sizes gives 100% bit for bit in every schedule |
| 6b unseen text | Mixed-all without the dataset's source, or the pick without its own set, saves at least 1% of T's bits over the datasets |
| 7 T2 | Every round trip is bit-exact (every window, every mode) and every T2 stream is smaller than T's |

## Results so far

Llama-3.2-3B on the NUC's RTX A4500, 13 Nov 2026
(`benchmark_results/llama32_3b_tideas_rtxa4500_20261113_0625`, `..._0740` for layer0x and calib, `..._0906` for
layer0m, calib_loo and t2). The check passed: T's exact-model bits from the
torch reference equal the kernel's (8.8627).

| Idea | Verdict | Numbers |
| --- | --- | --- |
| 2 layer 0 on the decode side | build | 100% of layer-0 words recomputed bit for bit on all five datasets. T's layer 0 costs 9.92-10.15 bits/value, so the saving is 0.354-0.362 bits/value over the cache, about 4%. The 54-61 ms measured was the whole first layer; layer0x measures the lean path |
| 1 per-request bias and spread | build (small) | Net savings: HumanEval 0.117, PTB 0.079, GSM8K 0.077, MMLU 0.044, WikiText 0.017 bits/value; side 0.0017. Room to the cross-fitted oracle: 0.65 / 0.23 / 0.43 / 0.16, so calib tests the coefficients |
| 4 anchors | drop | All the gain is in layer 0 (+2.70 bits there), which recomputing it takes. Beyond layer 0 the two-wave forms give at most +0.07 bits per layer and are often negative |
| 5 classes | drop | Negative everywhere except WikiText (the calibration text): splitting the predictor overfits the calibration data |
| 2b per-layer modes | drop | 3.4% at 100 Gb/s at best |
| 2c layer 0, other prefill shapes | mostly holds | Lean recompute 9.7 ms per 8K window (the whole first layer: 53 ms). Bit for bit in one pass and with other requests' tokens in the same pass. Prefill chunks of 2,048 or 512 tokens break it (54-83% of words equal), yet the saving stays 0.30 / 0.28 bits/value |
| 6 calibration breadth | build | Layer 0 recomputed: T's constants 8.782 bits/value on average; one mixed set (15 documents) 8.432; a per-request pick among WikiText and 4 domain sets, chosen on layers 1-4, 8.341 (it chose the matching set for all 10 windows). With the right set, per-request bias and spread add only 0.01. Caveat: 4 of 5 datasets had a matching calibration source; 6b tests unseen text |
| 2d layer 0, steps known | build | Padding to the sender's step sizes gives 100% bit for bit in all four simulated vLLM schedules, own rows first or not: the step's row count is all that matters, not the other rows' content or place. The same chunks without padding give 50-97% |
| 6b unseen text | weak | Mixed-all without the dataset's source saves 0.054 bits/value over the five (rule: 0.088). On the four non-WikiText datasets it saves 0.137 (1.5%) against T's WikiText constants; WikiText itself loses 0.275 when WikiText leaves the mix. The big gain needs the request's kind of text among the sets |
| 7 T2 | works | Dense, layer 0 recomputed in all 5 windows (the sender's check passed), every round trip bit-exact. Real streams: WikiText 8.98 to 8.62 (-4.0%), HumanEval 9.57 to 8.63 (-9.9%), GSM8K 9.28 to 8.16 (-12.1%), MMLU 9.33 to 8.38 (-10.1%), PTB 9.24 to 8.44 (-8.7%). Decode time unchanged (31 ms), plus the receiver's 9.9 ms recompute. On HumanEval keys then values, 4 groups and head by head are bit-exact too (head by head 9.01 to 7.96 bits/value). Rank 256 ran out of GPU memory beside the model (fixed: rerun the t2 stage). The sender's pick took 156-183 ms (now timed in cheaper forms) |
| 3 coding kernel | refocus | Dense and rank 256 code at 2.4x their memory floor (15.9 vs 6.7 ms). Head by head, the smallest mode (8.29 vs 8.98 bits/value), takes 108 ms because each layer's 16 head groups run one after another: one kernel per layer over all its groups is the prize |

## T2: the winning ideas as a separate codec

`kv_compare/t2.py` adds a codec beside T. No file of T changes, and T's tests and the PD branch keep using T.

- **Layer 0 does not travel.**
  - The receiver recomputes it from the token ids (`t_ideas.layer0_kv`).
  - With T's own kernels (features and surprise sums of layer 0), it builds what T's decoder would have had after
    layer 0, then decodes layers 1..L-1.
  - The stream is T's stream minus layer 0. The encoder runs T's coding kernel over layers L-1..1 and writes the
    final states after layer 1; the decoder starts there (the kernels' `FIRST` flag).
  - Every mode works this way (dense, keys then values, head groups, head by head, rank 256), because each predicts
    layer l from layer l-1.
- **The sender checks layer 0.** It sends layer 0 as recomputed only when the receiver's recompute equals its own
  layer 0. Otherwise the request goes as T (layer 0 coded) with the same set. A header field says which.
- **Constant sets.** The codec holds T's constants for several calibration sets. The sender codes layers 1-4 with
  each set (codecs cut to those layers) and sends the request with the cheapest; the header carries the set's name.
- **Not yet:**
  - PD wiring (the bundle with several sets, the header, the sender's step sizes for layer 0);
  - coding a mismatched layer 0 against the recompute rather than falling back to T.

`kv_compare/test_t2.py` runs every check under Triton's CPU interpreter on a tiny model:
- each mode restores the KV bit for bit, also through the layer-by-layer sink;
- the per-layer stream words for layers 2..L-1 equal T's, and the stream is smaller than T's;
- the pick chooses the cheapest set;
- the fallback is exactly T's stream.

The t2 stage repeats the round trips on the GPU on real windows.

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
- A rerun of the same command after new stages are added measures only those stages (the others are kept in the
  output folder's `t_ideas.json`) and packages everything again. layer0x and calib add about 30 minutes and about
  17 GB of calibration captures for the 3B (the stage checks the free disk first).
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
- layer0x runs Hugging Face's code (cuBLAS projections, its norm and RoPE). In PD both ends run vLLM, whose norm
  and RoPE kernels differ from Hugging Face's, so identity must hold between vLLM on the sender and vLLM on the
  receiver. The shapes tested here are the ones vLLM varies; the final check is in the PD branch.
- The calibration sets use 3 windows per domain (24K rows for 2,048 inputs); WikiText's set has 6. A domain set
  is therefore at a small disadvantage, and mixed-all has the most data.
- The oracles fit up to 2W inputs on one 8K window. The in-sample one is optimistic by construction, so the room is
  taken from the cross-fitted one.
