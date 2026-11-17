# Our approach: lossless KV transfer for prefill/decode disaggregation

A reference for what we built, how each part works, why it is built that way, how it compares with the other
methods, and what is measured so far. Written 15 November 2026 on branch `pd-baselines`.

- **T:** our bit-exact KV codec.
- **T2:** T with two additions for PD.
- **TKVConnector:** the vLLM connector that carries either one between GPUs.

Each section opens with a short "in plain words" summary; the details follow.

Start with [the idea, and why others did not do it](#start-here-the-idea-and-why-others-did-not-do-it).

Contents:
1. [The problem](#1-the-problem)
2. [How the other lossless codecs work, and where they stop](#2-how-the-other-lossless-codecs-work-and-where-they-stop)
3. [T: predict each value, code only the surprise](#3-t-predict-each-value-code-only-the-surprise)
4. [T2: layer 0 rebuilt by the receiver, constants picked per request](#4-t2-layer-0-rebuilt-by-the-receiver-constants-picked-per-request)
5. [Moving it inside vLLM: TKVConnector](#5-moving-it-inside-vllm-tkvconnector)
6. [Our protocol, step by step: a walk-through](#6-our-protocol-step-by-step-a-walk-through)
7. [How we compare, fairly](#7-how-we-compare-fairly)
8. [What is measured so far](#8-what-is-measured-so-far)
9. [AQUA-KV in PD: the closest method, step by step](#9-aqua-kv-in-pd-the-closest-method-step-by-step)
10. [What is new, what we lead with, and how we show it](#10-what-is-new-what-we-lead-with-and-how-we-show-it)
11. [Limits and open questions](#11-limits-and-open-questions)
12. [Glossary](#12-glossary)
13. [Where things live](#13-where-things-live)
14. [Sources](#14-sources)

---

## Start here: the idea, and why others did not do it

**In plain words.** In PD, the decode GPU runs the same model as the prefill GPU and already has the prompt, so it
already knows much of the KV cache it is about to receive. We send the cache exactly, and only the part the decode
GPU cannot work out for itself. Two ideas make that possible.

### The gap

PD needs a transfer that is small and invisible: the decode GPU should carry on exactly as if it had done the
prefill itself.

| Today | Small? | Invisible? |
|---|---|---|
| Raw transfer (NIXL) | no: 16 bits per value | yes |
| Exact codecs (SplitZip, ExANS, AdaCode, ...) | a little: 10.9-12.7 bits | yes |
| AQUA-KV (a cross-layer guess, with the gap rounded) | yes: about 2 bits | no: the cache changes |
| Latent methods (Palu, MLA, KV-Direct) | yes | no: they replace the cache, or change the model |
| Speculative KV coding (a whole second model as the guess) | 5.9-6.7 bits | yes, but the receiver re-runs the prefill |
| **Ours (T2)** | 7.8-8.5 bits | yes: the cache arrives bit for bit |

### The picture behind both ideas: views of one hidden state

Every token carries one internal vector, its hidden state (3,072 numbers in our model), up through the layers, and
each layer adds only a small update to it. A layer's keys and values are two fixed linear projections, "views", of
that layer's hidden state: 1,024 numbers each. The 8 key heads and 8 value heads are 16 views of 128 numbers.

So the cache is many views of one slowly changing vector, and the views are redundant: from some of them, much of
the others can be predicted. What no view of the token itself can show is what attention brought in from other
tokens. That part always has to travel.

### Idea 1: an exact code that uses the receiver's views as side information

- **Intuition.** Sending something to someone who already knows part of it costs only the part they cannot work
  out: H(X | what the receiver has), not H(X).
  - Exponent codecs pay for each number on its own, so they stop near 10.5 bits.
  - Our receiver, when it decodes layer l, already holds layer l-1 exactly (it has just decoded it) and the heads
    of layer l decoded so far. Those views imply most of each new number, so we send only the exact surprise.
- **What makes it work.** Both engines must compute the same guess, to the last bit, from the same views. Our
  guess is integer arithmetic: 8-bit inputs and coefficients, exact 32-bit sums, fixed-point confidence, no fused
  multiply-add. So the result is the same in any order, kernel or batch.
- **Why it fits PD.** A transfer is decoded once, in order, layer 0 to 27: the ideal setting for a decoder that
  chains each layer to the one below. A cache that lives in memory must be readable piece by piece at any time, and
  a layer chain gets in the way.
- **Why others did not do it.**
  - **Exponent codecs** chose independent blocks on purpose, for random access and speed (ExANS; exactkv and GAAC
    decode inside attention). A chain from layer to layer conflicts with that.
  - **AQUA-KV** aims at memory. There an exact code of about 8 bits saves too little next to 2-4-bit rounding, and
    it would have to be decoded at every step. Exactness pays off only when the cache is decoded once and the
    answers must not change: PD.
  - **Latent methods** change what is stored, or the model, so they cannot be exact.
  - **Speculative KV coding** uses the whole model rather than the views, so the receiver pays for the prefill
    again.
  - **The hidden obstacle:** exact side information needs bit-identical guesses on two engines. Floating point does
    not guarantee that, and nobody had coded KV exactly across two engines, so nobody had needed it.

### Idea 2: shape replay, so layer 0 needs no bits

- **The gap.** Layer 0 is fully computable by the receiver: it depends only on the token ids, through the
  embedding, a norm, the K and V projections and RoPE, with no attention. But a naive recompute under vLLM-style
  batching gave only 50-97% identical words, and a layer that cannot be rebuilt exactly cannot be dropped. The
  general remedy, batch-invariant kernels, is slower and a large rewrite.
- **Intuition.** The last bits of a matrix product do not depend on what the other rows hold, only on the shape
  the math library sees: its row count, which decides cuBLAS's algorithm. So the receiver needs the same shape, not
  new kernels.
  - The sender already knows each prefill step's row count: its scheduler made the step.
  - It sends the counts, a few bytes.
  - The receiver pads its rows with zeros to the same count.

  The result: 100% identical words in every simulated vLLM schedule.
- **Why it fits PD.** The receiver runs the same model, kernels and GPU type (PD clusters are usually homogeneous);
  the prefill's scheduler knows the shapes; and layer 0 needs no attention, so rebuilding it takes about 10 ms.
  Layer 0 then costs 0 bits: about 0.36 bits per value saved.
- **Why others did not do it.**
  - **Memory-resident compression (AQUA-KV)** has no second engine. Rebuilding layer 0 at every generated token
    would cost far more than storing it at 4 bits.
  - **Recompute-based work (KVPR, KV-Direct)** sends the inputs (activations, checkpoints) and recomputes from
    them. It does not need, or does not show, bit identity.
  - **Determinism work (batch-invariant kernels)** solves the general problem, any batch on any engine, by
    changing the kernels. Replay is the narrow, cheap trick that suffices when you control both ends and know the
    shapes: only in PD.
  - **The limit:** reliable in eager engines. In compiled ones the fallback keeps the transfer exact but codes
    layer 0.

### Why the two belong together, and the gap in one sentence

Both are exact side information:
- idea 1 uses the views the receiver gets by decoding;
- idea 2 creates a view from nothing: layer 0 from the embedding.

Both depend on the two engines computing identically, and shape replay is the hardest case of that.

> Existing KV codecs either ignore what the decode engine already knows (exact but big) or use it at the cost of
> changing the cache (small but visible). PD is the setting where the receiver decodes once, in order, with the same
> model, so the cache can be sent exactly at the cost of only what the receiver cannot work out, provided both
> engines compute identically, which our integer guess and shape replay ensure.

Read on: the protocol taught step by step is section 6; AQUA-KV, the closest method, section 9; the leads and how
we show them, section 10.

---

## 1. The problem

**In plain words.** When a model reads a prompt, it writes notes about every word at every layer (the KV
cache). In prefill/decode disaggregation (PD), one GPU reads the prompt and another writes the answer, so the
notes must be shipped between them. They are large, so shipping them is slow. We want to make them smaller
without changing a single bit, so the answer stays exactly the same.

### 1.1 The KV cache

A transformer layer turns each token into a key vector (K) and a value vector (V). Later tokens attend to them,
so the engine keeps every layer's K and V for every token of the context: the KV cache.

For our model, Llama-3.2-3B:
- 28 layers, 8 KV heads of 128 dimensions each;
- so 8 × 128 = 1,024 key values plus 1,024 value values per token and layer: 2,048 numbers;
- each number is BF16, 2 bytes;
- so each token costs 28 × 2,048 × 2 = 114,688 bytes, about 112 KiB.

| Prompt | Raw KV | At 10 Gb/s | At 100 Gb/s |
|---|---|---|---|
| 1,000 tokens | 115 MB | 92 ms | 9 ms |
| 8,192 tokens | 940 MB | 752 ms | 75 ms |

### 1.2 Prefill/decode disaggregation

Reading a prompt (prefill) is compute-heavy; writing the answer one token at a time (decode) is memory-heavy.
Running them on separate GPUs lets each be sized and batched for its own job (DistServe, Splitwise, Mooncake
and vLLM's own disaggregated serving do this).

The price is the transfer: the decode GPU cannot produce the first token until it holds the prompt's KV. So the
time to first token (TTFT) includes the transfer, and on ordinary networks (10-100 Gb/s) the transfer can be
as long as the prefill itself.

### 1.3 Why lossless, and what "bit-exact" means

A lossy codec (quantization, pruning) changes the cache, so the model's answers can change: quality must be
re-validated for every model, task and setting.

A lossless codec gives back the same 16-bit words, so the decode GPU computes exactly what it would have
computed from the original cache. Nothing about the model's behaviour needs re-checking.

We call a method **bit-exact** only when the original BF16 words come back. A method whose outputs look
unchanged while the stored numbers change is not bit-exact.

Every transfer in our correctness pass carries a checksum of the sender's words, and the receiver compares it with
what it installed (section 5.4).

### 1.4 The number we optimise: bits per value

Size is reported as bits per value: 16 is raw BF16; 8 means half the bytes. The link time scales with it; the codec
adds encode and decode time on top. Whether a codec helps depends on both:
- On a slow link the wire dominates, so fewer bits win.
- On a very fast link a slow decoder can cost more than it saves.

---

## 2. How the other lossless codecs work, and where they stop

**In plain words.** Every published bit-exact KV codec compresses only the exponent of each number (its rough
size) and sends the sign and the precise digits untouched. The digits look random to them, so they can never get
much below 11 bits.

### 2.1 What a BF16 number is made of

```
 bit:   15 | 14 13 12 11 10  9  8  7 |  6  5  4  3  2  1  0
        s  |  e  e  e  e  e  e  e  e |  m  m  m  m  m  m  m
      sign |       exponent (8)      |     mantissa (7)
```

- **Sign:** + or -.
- **Exponent:** the power of two, roughly "how big". KV values cluster in a narrow range of sizes, so the 8
  exponent bits carry only about 2.5-3 bits of information.
- **Mantissa:** the precise digits within that power of two. Looked at one value at a time, they are close to
  uniformly random.

So a codec that models values one at a time can squeeze the exponent but not the other 8 bits. Its floor is
about 8 + 2.5 = 10.5 bits per value.

### 2.2 The published bit-exact codecs

| Method | What it does | Bits per value | Measured where |
|---|---|---|---|
| raw BF16 | nothing | 16 | - |
| nvCOMP (LZ4, Cascaded, Bitcomp, ANS, ...) | general byte compressors on the GPU | 15.2-16.0 (ANS 13.4 in our vLLM smoke) | SplitZip's Table 2; ours |
| exactkv | per 16-token page and channel, a base exponent plus 4-bit offsets; raw rows for outliers | 12.7 | its README |
| IBP (Invariant Bit Packing) | drops bits that are constant across a group; for CPU-to-GPU copies | 12.3 | its paper |
| SplitZip | a 16-entry code book of common exponents (4 bits) plus escapes; sign and mantissa raw | 12.05 | the authors' codec on our KV |
| AdaCode | a short fixed-length exponent code plus explicit outliers, length adjusted online | 11.5 | its paper |
| GAAC | calibrated per-channel base-plus-offset exponent codes; decoded inside attention (personal devices) | about 11.4 (1.41×) | its summary |
| ExANS | exponents re-centred per block, then rANS entropy coding; independent blocks | 10.87 | the authors' codec on our KV |

All of them are in the 10.5-13 band, as section 2.1 predicts.

### 2.3 Transports and stores are not codecs

vLLM's references for PD move the KV but do not shrink it; we compare against them as the speed of an
uncompressed, production path.
- **NIXL** (NVIDIA's transfer library; vLLM's `NixlConnector`) moves raw KV GPU to GPU.
- **LMCache** keeps raw KV in host memory (a CPU tier) and loads it back.

### 2.4 Predictive methods that are close to ours

| Method | How it predicts | Why it is not the same |
|---|---|---|
| Speculative KV coding (research note) | runs a whole FP8 copy of the model on both sides; codes the BF16 cache against it, 5.9-6.7 bits | the receiver must run the full prefill again, which PD exists to avoid; not timed or verified in the note |
| AQUA-KV (ICML 2025) | keys from the previous layer's keys; values from the previous layer's values plus the current keys; linear, before RoPE | rounds the gap to 2-4 bits: lossy, 2.09 bits at its 2-bit setting (section 9) |
| AnchorKV (2026) | each token from an anchor token, plus a 2-bit quantized residual | lossy (about 20× smaller) |
| KVPR, KV-Direct | recompute part of the KV from transferred activations or residual checkpoints | send activations or checkpoints instead; not a bit-exact codec of the cache |

---

## 3. T: predict each value, code only the surprise

**In plain words.** We guess every number before sending it, from the same token's numbers one layer below
(neighbouring layers are similar). Then we send only how far the real number is from the guess. Small surprises
take few bits. The receiver makes the identical guess and adds the surprise back. Because the guess covers the
whole number, the digits get compressed too, not just the exponent.

The analogy: if the forecast says 12 °C, "one degree warmer" is shorter than the temperature from scratch.

### 3.1 The predictor

For each layer l ≥ 1, the value of channel j for one token is predicted from that token's own layer l-1, which the
receiver has already decoded.

1. **De-rotate the keys.** Keys carry RoPE, a rotation by the token's position. The predictor works on de-rotated
   keys, so the relation between layers does not depend on position.
2. **8-bit features.** Each channel i of layer l-1 becomes `x_q[i] = clamp(round(x[i] / sx[i]), -127, 127)`, with a
   per-channel scale `sx` from calibration.
3. **Integer product.** The prediction is a linear map with 8-bit coefficients `A_q` and a per-output-channel scale
   `sa` and bias `b`:

   ```
   mu_derot[j] = sa[j] * Σ_i A_q[j, i] * x_q[i] + b[j]
   ```

   The sum is an exact int32 integer product on tensor cores, then one float32 multiply-add.
4. **Rotate back.** Predicted keys are rotated to the token's position with float32 cos/sin tables.
5. **Layer 0** has no layer below it and is predicted by its calibration mean. This makes it T's most expensive
   layer (9.9-10.2 bits per value), which T2 removes (section 4.1).

**Fitting.** The map is the conditional mean of layer l given layer l-1, in closed form from the calibration
covariance (ridge 10⁻³ × average variance), computed in float64. It is then rounded to 8 bits per output
channel.

Calibration text is WikiText-2 train for T; T2 adds other domains (section 4.2).

Fitting happens once per model, offline. One exported file holds every constant, and both ends load the same
file.

### 3.2 The spread: how sure the guess is

The coder also needs to know how far off the guess usually is. That is the spread `s`, per channel and token:
- **Base spread:** the channel's mean absolute residual on a calibration "scales" document, divided by 2 ln 2 (the
  scale of a logistic distribution with that mean absolute deviation).
- **Per-token multiplier:** `clip((64 × prior + 16) / 80, 0.25, 4)`. The `prior` is the token's mean surprise on
  the previous layer: `|x - mu| / (2 ln 2 × spread)`, averaged over the layer's channels.
  - A token that was hard to predict on layer l-1 gets a wider spread on layer l.
  - The surprise is summed in fixed point (units of 1/4096), so the encoder and decoder get the same integer
    whatever order the GPU adds in.

### 3.3 The coder: value-slice rANS against a logistic

Each BF16 word is coded against a logistic distribution centred on the prediction `mu` with scale `s`, in two
symbols (`kv_compare/gpu_coder.py`):

1. **Slice.** The range `t = (v - mu) / s` in [-24, 24) is cut into slices.
   - The slice width depends on a width class: how large `mu` is compared with the spread. That decides how
     coarse the BF16 grid is around the prediction (fine near zero, coarse far from it).
   - The slice symbol uses a static table per class (the logistic mass of each slice; every symbol at least
     1/65,536).
   - An escape symbol covers values outside the range, Inf and NaN, and is followed by the raw 16-bit word. So
     any word at all can be coded.
2. **The word within the slice.** Either equal shares (mode "uniform") or shares following each word's rounding
   interval (mode "exact"). A slice holding one word needs no second symbol.

The symbols go into **16-bit rANS**, an entropy coder that gets within a fraction of a bit of -log₂ P per symbol.
How the streams are organised:
- There are many interleaved streams: one GPU lane per channel, for a block of tokens.
- Each stream runs through every layer, so its 32-bit state is paid once per transfer, not per layer.
- The encoder works in reverse order (layers L-1 down to 0), as rANS requires; the decoder reads forwards.

**Why the two ends agree bit for bit.** Encoder and decoder run the same integer and float32 operations:
- additions, multiplications and correctly rounded divisions;
- no fused multiply-add (the kernels are compiled without it);
- approximate divisions corrected to the exact integer result.

So on one GPU type they compute the same `mu` and `s` and the same slice for every value.

### 3.4 Decoding layer by layer

Layer l of a token depends only on layer l-1 of the same token. Hence:
- **Across tokens:** everything runs in parallel. Each GPU program owns a block of tokens and never waits for
  another.
- **Across layers:** decoding is sequential. Layer l needs the decoded layer l-1, because its features come from
  it.
- **Into vLLM's cache directly:** the decoder hands each finished layer to a sink, which writes it straight into
  the request's blocks of vLLM's paged cache. The receiver never holds a second full copy of the cache.

### 3.5 The modes

Every mode predicts layer l from layer l-1. Some modes also use parts of layer l already decoded. The trade is
bits against decode time.

| Mode | Name in code | What it adds | Exact cost, one text (PEP 8) | Real stream on the GPU | Decode, 8K window |
|---|---|---|---|---|---|
| dense | `ours-dense` | one product per layer | 8.90 | 9.03 (WikiText-2), 9.57 (HumanEval) | about 31 ms |
| keys then values | `ours-kv` | values also predicted from the layer's decoded keys | 8.56 | - | - |
| 4 head groups | `ours-4g` | 4 heads at a time; each group also from the groups before it | 8.30 | - | - |
| head by head | `ours-hbh` | 16 groups (8 key heads, then 8 value heads), each also from the heads before it | 8.12 | 8.32-8.41 (WikiText-2), 9.01 (HumanEval) | about 108 ms (16 kernels per layer, one after another) |
| rank 256 | `ours-rank256` | the dense map factored into two rank-256 8-bit products: faster, slightly larger | - | 9.82 (HumanEval) | about 24 ms |

The exact costs are -log₂ P of every word under each predictor (`kv_compare/grouped.py`); the real streams are
what the GPU codec writes, framing included. Different texts give different sizes, so compare within a column.

The extra within-layer part of the head-group modes is a block lower-triangular map: group j reads only the
groups before it (`kv_compare/grouped.py`). The encoder knows the whole layer and adds both parts at once; the
decoder adds them group by group.

Speeds are from the RTX A4500 (`docs/t_ideas.md`).

### 3.6 On the wire

- **Format.** A transfer is a JSON header of a few kB plus one blob: the codec's tensors end to end, 16-byte
  aligned (`kv_compare/pd/wire.py`).
- **Only used bytes travel.** T writes each stream segment backwards into a fixed-size slot; only the used tail
  is sent.
- **Fixed shapes.** The token count is padded to a multiple of 256, so every prompt length runs the same kernels
  as the measured 8K windows. The receiver drops the padding.

---

## 4. T2: layer 0 rebuilt by the receiver, constants picked per request

**In plain words.** T2 is T plus two tricks that use PD's situation: the receiver already runs the same model and
already has the prompt.
1. **Layer 0 is never sent.** The bottom layer depends only on each word, so the receiver rewrites it itself.
2. **A guidebook per request.** We keep five sets of guessing constants learned from different kinds of text, and
   the sender picks the best one for each prompt.

### 4.1 Layer 0 is recomputed

**Why it is possible.** A token's layer-0 K and V are computed from its id and its position alone: embedding, the
first layer's input norm, the K and V projections, RoPE. There is no attention and nothing from other tokens.

The receiver has the token ids (the proxy sends the prompt to both engines) and the model's weights. It can
compute layer 0 instead of receiving it.

**What it saves.** In T, layer 0 is coded against its calibration mean and costs 9.92-10.15 bits per value. Left
out, it saves 0.354-0.362 bits per value over the whole cache, about 4%.

**The hard part: bit identity.** The recomputed words must equal the sender's to the last bit. Floating-point
results can depend on how the work is batched:
- **The batch's row count matters.** On the GPU, the projection is a matrix product over every token of the
  prefill step, other requests' tokens included. cuBLAS picks its algorithm by the product's row count M, and
  different algorithms round differently.
- **The rows' content does not.** Measured (`docs/t_ideas.md`, stage 2d): only M matters, not what the other rows
  hold or where the request sits among them. Without padding, the same chunks matched only 50-97% of the words.
- **So we pad to the sender's M.** The prefill engine's scheduler side records, for every request, each step's
  `(start, n, M)`. The receiver pads the request's rows to M with zero rows and gets 100% identical words in
  every simulated vLLM schedule.
- **The sender checks before leaving layer 0 out.** It runs the same recompute and compares with its own cache.
  - It tries the scheduler's M, then M rounded up to vLLM's CUDA-graph sizes.
  - Only on a match is layer 0 left out; otherwise layer 0 is coded as in T.
  - A header flag says which, so the transfer is bit-exact either way.
- **Eager versus compiled engines.** In eager engines (`--enforce-eager`) the forward runs the same modules our
  recompute calls. Compiled engines may fuse the norm and the product into different kernels, and the recompute
  may then not match. The cost is bits, not correctness, and the report shows how often layer 0 was recomputed.

**After layer 0.** From the recomputed words, T's own kernels compute what T's decoder would have after layer 0:
- layer 1's 8-bit features;
- layer 1's spread prior (the surprise sums).

The decoder then starts at layer 1, through the kernels' `FIRST` flag. The stream is T's stream without
layer 0. The encoder codes layers L-1 down to 1 and writes the final states after layer 1.

### 4.2 A constant set per request

T's constants come from WikiText-2. Other text costs more under them; one-step bits per value:

| Text | WikiText-2 | HumanEval | GSM8K | MMLU | PTB |
|---|---|---|---|---|---|
| Bits per value | 8.87 | 9.54 | 9.18 | 9.21 | 9.14 |

Most of the gap is in the predictor's coefficients, not its bias or spread.

**The sets.** T2 holds five sets of T's constants, each fitted on its own calibration text, disjoint from every
test window:
- WikiText-2 train (general);
- MBPP (code);
- GSM8K train (math);
- MMLU dev and validation (exam questions);
- Penn Treebank train (news).

**The pick (`pick_fast`).**
1. The sender samples 1,024 tokens across the request (8 blocks).
2. It computes T's int8 product for layers 1-2 with every set at once, in plain torch, float32.
3. It keeps the cheapest set and writes its name in the header.

This takes 11.4 ms per 8K request. It chose the same set as the full pick (coding layers 1-4 with every set,
104 ms) in 5 of 5 windows.

Only the set's name travels. Both ends already hold all five sets from the exported constants file.

### 4.3 What T2 gives

Dense mode, layer 0 recomputed, the set picked per request, real streams, bit-exact on every window (RTX A4500,
`docs/t_ideas.md` stage 7):

| Window | T (bits/value) | T2 | Change |
|---|---|---|---|
| WikiText-2 | 8.98 | 8.62 | -4.0% |
| HumanEval | 9.57 | 8.63 | -9.9% |
| GSM8K | 9.28 | 8.16 | -12.1% |
| MMLU | 9.33 | 8.38 | -10.1% |
| PTB | 9.24 | 8.44 | -8.7% |
| Mean | 9.28 | 8.45 | -9.0% |

Every mode is bit-exact under T2. On HumanEval:
- dense -9.9%;
- keys then values -10.9%;
- 4 groups -11.4%;
- head by head -11.6% (9.01 to 7.96);
- rank 256 -8.7% (9.82 to 8.97).

**Against the baselines** (the same windows, GPU and run; stage 8):
- T2 head by head: 7.80 bits (2.05×).
- T2 dense: 8.45 bits (1.89×).
- ExANS, the smallest bit-exact baseline: 10.87 bits (1.47×).
- T2 is lower than ExANS on every dataset.

**Time per 8K request** (pick and recompute included):

| Mode | Encode | Decode |
|---|---|---|
| T2 dense | 92 ms | 31 ms |
| T2 rank 256 | 52 ms | 23 ms |

The receiver's layer-0 recompute adds about 10 ms.

**Requests per second over one link** (the longer of wire time and decode time, per 8K request; a model of the
link, not vLLM):

| Link | Winner | Others |
|---|---|---|
| 1-100 Gb/s | T2 carries the most | at 100 Gb/s: T2 dense 25.2/s, ExANS 19.6/s, raw 13.3/s |
| 400 Gb/s | AdaCode 74.9/s, SplitZip 70.6/s | T2 rank 256 is decode-bound at 43.6/s, below raw 53.2/s |

**Unseen text.** When a request's kind of text is not among the sets, the gain shrinks. With each dataset's own
source left out of the calibration, the best mixed set saves only 0.054 bits per value over the five datasets.

---

## 5. Moving it inside vLLM: TKVConnector

**In plain words.** vLLM lets outside code decide how KV moves between engines (a "KV connector"). Ours,
TKVConnector, takes the KV out of the prefill engine's cache, compresses it with the codec the request names,
sends it, and the decode engine decompresses it straight into its own cache. vLLM itself is not modified.

### 5.1 The flow of one request (push mode)

```mermaid
sequenceDiagram
    participant C as Client
    participant X as Proxy
    participant P as Prefill engine
    participant D as Decode engine
    C->>X: completion request
    X->>P: same prompt, max_tokens 1, {t_send, key, dest, codec}
    X->>D: same prompt, {t_recv, key}
    Note over D: waits (WAITING_FOR_REMOTE_KVS), blocks allocated
    Note over P: prefill; scheduler keeps the blocks
    P->>P: gather KV from paged cache, encode, free blocks
    P->>D: coded transfer over the link
    Note over D: decode layer by layer into its blocks, verify
    Note over D: recompute only the last prompt token
    D-->>X: tokens (streamed)
    X-->>C: tokens
```

1. The proxy (`scripts/pd_proxy.py`) sends each request to a prefill engine and, at the same time, to a decode
   engine.
   - The prefill engine is the one with the fewest requests in flight.
   - The decode engine is chosen round-robin, by load, or by conversation.
2. The prefill engine computes the prompt. When the request finishes, its scheduler keeps the blocks and hands
   them to the worker. The worker:
   - gathers the prompt's KV from the paged cache into `[layers, tokens, channels]` words (`kv_compare/pd/paged.py`);
   - encodes it;
   - releases the blocks;
   - queues the coded blob for the link.
3. The decode engine holds the request in vLLM's `WAITING_FOR_REMOTE_KVS` state with its blocks allocated. Each
   engine step, its worker installs every transfer that has arrived, layer by layer, straight into those blocks.
4. vLLM then recomputes only the last prompt token and starts generating.

If the decode engine already holds part of the prompt in its own prefix cache, only the missing part is sent.

### 5.2 Transports

| Transport | Path | Used for |
|---|---|---|
| `ring` | the receiver's ring in shared host memory, pinned for CUDA on both sides: sender GPU → ring → receiver GPU, chunk by chunk, overlapped | emulating a network of a set speed (3-400 Gb/s) on one machine: each chunk is released no earlier than a real link of that speed would deliver it |
| `p2p` | the receiver's ring in its GPU memory, shared by a CUDA IPC handle: one copy GPU to GPU over PCIe or NVLink | the machine's direct path; NIXL is the reference here |
| `tcp` | bytes on the control socket | two machines |

Every record keeps the speed set and the speed achieved. When the machine is slower than the set speed, the
transfer takes the machine's time.

### 5.3 The CPU tier (colocated engines)

The same codecs also serve a CPU tier: the role LMCache's CPU offload plays (`kv_compare/pd/tier.py`).
- **Store:** when a request finishes, its full 256-token chunks are coded and kept in pinned host memory, by a
  background thread outside the engine's step.
- **Keys:** each chunk is keyed by the chained hash of every token up to its end, so a hit means the whole prefix
  is the same.
- **Load:** a later request that starts with the same tokens loads those chunks back and decodes them into its
  blocks.
- **Budget:** least recently used chunks go first. The budget counts coded bytes, so a smaller format keeps more
  conversations, a gain besides the shorter copy.

### 5.4 Correctness and failure handling

- **Checksums.** In the correctness pass, every transfer carries a checksum of the sender's words; the receiver
  checksums what it installed. A mismatch fails the transfer. The timed runs leave checksums off.
- **Failures stay with the request.**
  - A failed encode tells the decode engine at once (an abort message), instead of waiting for the 600 s
    timeout.
  - A failed transfer or install fails only that request's transfer, not the engine: the engines run with vLLM's
    `kv_load_failure_policy` set to `recompute`, so the decode engine then computes the prompt itself. Under vLLM
    0.32's default (`fail`) its scheduler asserted on the failed request and the engine died (jobs 30176, 30226).
- **A memory retry.** nvCOMP and ExANS allocate GPU memory outside PyTorch. An encode or decode that fails is
  retried once, after PyTorch gives its cached memory back.

### 5.5 Every codec through the same connector

`kv_compare/pd/codecs.py` wraps every bit-exact codec behind the same interface:
- **raw** BF16;
- **ours-*** and **t2-***, all modes;
- **SplitZip, AdaCode, exactkv, ExANS:** the authors' own code;
- **nvCOMP:** six algorithms;
- **IBP:** for the tier only.

So the same engines, prompts, links and scheduler carry each of them. NIXL and LMCache run through vLLM's own
connectors as the uncompressed references.

---

## 6. Our protocol, step by step: a walk-through

**In plain words.** This section follows one request from the client to the first generated token, as a lesson:
at each step, what happens, why, what it costs and where it is in the code.

The example throughout:
- Llama-3.2-3B;
- a 2,048-token prompt;
- one prefill engine and three decode engines;
- codec T2 head by head (`t2-hbh`).

### 6.1 Before any request: the constants (once per model)

1. Capture the KV of calibration text with the model: windows of WikiText-2, MBPP, GSM8K, MMLU and PTB
   (`scripts/fetch_calib_texts.py`, `scripts/pd_export_codecs.py`).
2. For each of the five texts, fit T's constants per layer: the guess's 8-bit coefficients, the feature scales, the
   bias and the per-channel spread (sections 3.1-3.2).
3. Export everything once, with the RoPE tables, to one file (`pd_codecs.pt`). Both engines load that same file, so
   they hold identical constants.

Why: the guess must be the same on both ends. Shipping constants per request would cost bytes; fitting them per
request would be far too slow.

### 6.2 The request arrives at the proxy

The client sends a completion request to the proxy (`scripts/pd_proxy.py`). The proxy:
- picks a prefill engine (the one with the fewest requests in flight, counted when it picks) and a decode engine
  (round-robin, by load, or by conversation);
- makes a key for the transfer;
- sends both engines the same prompt at the same time:
  - the prefill engine with `max_tokens` 1 and `{"t_send": true, "t_key": key, "t_dest": "host:port",
    "t_codec": "t2-hbh"}`;
  - the decode engine with `{"t_recv": true, "t_key": key}`.

Why both at once: the decode engine reserves space while the prefill runs, and it already holds the prompt's token
ids, which the layer-0 rebuild needs.

### 6.3 The decode engine reserves space and waits

vLLM asks our connector how many tokens will come from outside (`get_num_new_matched_tokens`).
- If the decode engine already holds the start of this prompt in its own prefix cache (h tokens), only tokens
  h..2,047 need to come. In our example h = 0.
- vLLM allocates the request's blocks: 128 blocks of 16 tokens.
- The request waits in vLLM's `WAITING_FOR_REMOTE_KVS` state.

### 6.4 The prefill engine computes the prompt, and records its shapes

vLLM schedules the prompt into one or more steps. In each step the model multiplies large matrices whose rows are
every token in that step, from every request in it.

Our connector's scheduler side records, for this request, each step as `(start, n, M)`: where its rows start in the
prompt, how many there are, and the step's total row count. For example, `(0, 2048, 6144)` means all 2,048 tokens
in one step, with 4,096 tokens of other requests.

Why: M decides which algorithm cuBLAS uses, and so the last bits of layer 0. Recording it costs nothing; sending it
costs a few bytes.

When the request finishes (after its one token), the scheduler keeps its blocks instead of freeing them and hands
them to the worker.

### 6.5 Gather the KV

The worker copies the prompt's KV out of vLLM's paged blocks into one array of 16-bit words: `[28 layers, 2,048
tokens, 2,048 channels]`, every key channel head by head, then every value channel (`kv_compare/pd/paged.py`).
That is 117.4 million values, 234.9 MB raw. The keys in vLLM's cache are already rotated by RoPE.

### 6.6 Pick the constant set

The sender samples 1,024 tokens across the prompt (8 blocks), computes T's integer guess for layers 1-2 with all
five sets at once, and keeps the set with the cheapest result. About 11 ms. Only the set's name will travel.

### 6.7 Check layer 0

The sender rebuilds layer 0 itself, exactly as the receiver will (`kv_compare/pd/layer0.py`):
- the engine's own embedding, first-layer norm, fused QKV projection and rotary embedding;
- this request's rows padded to each recorded M.

It compares the result with layer 0 in its cache, word for word.
- **Equal:** layer 0 is left out of the stream. The header says `"layer0": "recomputed"` and carries the steps.
- **Not equal:** it tries M rounded up to the next multiple of 8, 16, ... 512 (vLLM pads steps to CUDA-graph
  sizes). If none matches, layer 0 is coded like any other layer (`"layer0": "coded"`).

Either way the transfer is exact.

### 6.8 Encode layers 1 to 27

1. **The model pass.** Compute what the decoder will compute, from the exact values, so no layer waits for a
   rounded copy of the one before:
   - each layer's 8-bit features (keys de-rotated, scaled, rounded);
   - each token's surprise sum, which sets its confidence at the next layer.
2. **The guess.** For layer l, head by head:
   - the integer product of the 8-bit coefficients with layer l-1's features and the heads of layer l already
     decoded, as an exact 32-bit sum;
   - then one scale and bias in float32;
   - keys rotated back to their position.
3. **The coding.** Each true 16-bit word is coded against a logistic around its guess, with the token's spread:
   - a slice symbol, then the word within the slice, or an escape and the raw word (section 3.3);
   - into 16-bit rANS: many interleaved streams, one per GPU lane, each running through all layers;
   - rANS writes in reverse, so the encoder walks the layers from 27 down to 1.
4. **The size.** About 8.1 bits per value for `t2-hbh` in the 30210 smoke, where layer 0 was still coded: about
   120 MB instead of 235 MB. Layer 0 at T's 10 bits would be about 5 MB; left out, it costs nothing.

Where: `kv_compare/t2.py`, on top of `kv_compare/layerwise.py`, `kv_compare/grouped.py` and `kv_compare/gpu_coder.py`.

If encoding fails (out of GPU memory, say), it is retried once after PyTorch frees its cached memory. If it fails
again, the sender tells the decode engine at once, with an abort message.

### 6.9 Package and send

- **The header** (JSON, a few kB) holds:
  - the key, the codec and the token range;
  - the set name, the layer-0 flag and the steps;
  - each tensor's dtype, shape and offset;
  - in the correctness pass, a checksum of the sender's words (`kv_compare/pd/wire.py`).
- **The blob:** the stream segments, each cut to its used tail, laid end to end.
- **Waiting:** the worker frees the request's blocks on the prefill engine, and the blob waits for the link: on the
  GPU, up to 2 GiB of waiting transfers, beyond that in host memory.
- **The link** (`kv_compare/pd/transport.py`), one of:
  - the host-memory ring at a set speed: GPU to pinned ring to GPU, chunk by chunk, each chunk released no earlier
    than a real link would deliver it;
  - GPU to GPU over CUDA IPC;
  - TCP between machines.

At 10 Gb/s, 120 MB take about 96 ms; raw, it would be 188 ms.

### 6.10 The decode engine installs it

At each engine step, the decode engine's worker installs every transfer that has arrived
(`kv_compare/pd/connector.py`):
1. **Rebuild layer 0**, if the header says so, from the prompt's token ids (it has them from its own request),
   padded to the header's row counts. About 10 ms for 8K tokens. From those words, T's kernels compute layer 1's
   features and confidence.
2. **Decode layers 1 to 27, in order.** For each layer, head by head:
   - the same integer guess, from the layer below and the heads already decoded;
   - the slice and the word, decoded from the stream: the exact word.

   Each finished layer goes straight into the request's blocks in vLLM's cache (`kv_compare/pd/paged.py`), so
   there is never a second full copy.
3. **Check**, in the correctness pass: the checksum of what was installed against the sender's. A mismatch fails
   the transfer.

A failed install or an abort fails only that request's transfer: the decode engine then computes the prompt itself
(vLLM's `recompute` policy). A transfer not complete 600 s after admission times out the same way.

### 6.11 Generation

The connector reports the request as received. vLLM moves it from waiting to running, computes only the last
prompt token itself, and generates as usual. Attention reads vLLM's normal blocks, so nothing else in the engine
knows a codec was involved.

### 6.12 The same request with AQUA-KV, for contrast

| Step | Ours | AQUA-KV, ported naively |
|---|---|---|
| Before requests | five integer constant sets, one file for both ends | one floating-point predictor set, fitted on rounded values |
| 6.4 shapes | recorded | none |
| 6.6 set | picked per request | none |
| 6.7 layer 0 | checked and left out | rounded to 4 bits |
| 6.8 the guess | the whole layer below, then head by head; integers | keys from keys, values from values and keys; floating point |
| 6.8 the gap | coded exactly; bits follow the guess's quality | rounded to 2-4 bits, the same for every value |
| 6.8 order | every guess from the exact values | layer by layer: each needs the rounded copy of the one before |
| Raw tokens | none | the first 4 and up to the newest 128, at 16 bits |
| 6.10 decode | once, into vLLM's blocks, with a checksum | once on arrival (losing its memory benefit), or at every step (needs a new attention path) |
| Result | the very cache the prefill computed | a close copy: answers can change, and depend on the route |

### 6.13 Mechanism by mechanism: which parts are standard

Most of the protocol is standard parts; three are not.

| Mechanism | Why | Who else does it | Ours? |
|---|---|---|---|
| A linear guess of layer l from layer l-1, in closed form | the views change nearly linearly | AQUA-KV, the same method | no |
| The whole previous layer, keys and values, as input | both views of the previous hidden state | AQUA-KV guesses keys from keys only | a difference; gain not measured yet |
| Keys de-rotated before guessing | the position rotation hides the layer-to-layer relation | KVQuant and AQUA-KV work on keys before RoPE | no |
| 8-bit inputs and coefficients, exact integer sums | speed, and the same answer on both engines | int8 products (LLM.int8); integer predictors in lossless coding (JPEG-LS, FLAC) | no, alone; new for KV |
| A rank-256 guess | the predictable part lives in few directions | AQUA-KV tried reduced rank; Palu | no |
| Head by head within a layer | decoded heads are views of the current hidden state | AQUA-KV: keys, then values; a 2026 paper reconstructs heads from reference heads (lossy) | an extension |
| Five calibration sets, one per request | the layer-to-layer map depends on the text | model and dictionary switching in compression | minor |
| Per-token confidence from the previous layer's surprise | hard tokens stay hard | adaptive contexts (CALIC, LOCO-I) | no |
| A logistic over each word's rounding interval, slices, an escape | the BF16 grid is uneven; every word must be sendable | lossless float compression (Isenburg et al., fpzip); discretized logistics (PixelCNN++) | the slice scheme is engineering |
| Interleaved rANS on the GPU | parallel streams; states paid once | interleaved rANS (Giesen); DietGPU, ExANS, nvCOMP | no |
| No fused multiply-add; exact divisions | identical floats on both ends | deterministic numerics; batch invariance | no |
| **Layer 0 rebuilt from the token ids** | the bottom of the stream is the embedding | none found | **yes** |
| **Shape replay** | the last bits depend only on the row count | batch-invariant kernels solve it by rewriting kernels | **yes** |
| **An exact code with the receiver's views as side information** | the transfer stays invisible | others use the views to replace or round the cache | **yes, the combination** |
| Decoding straight into vLLM's blocks, with a checksum | no second copy; attention unchanged | NIXL writes raw blocks; ExANS restores to its destination | no |

---

## 7. How we compare, fairly

The comparison is only meaningful if every method gets the same conditions. The runner (`scripts/run_pd_vllm.sh`,
`scripts/pd_bench.py`) enforces these rules:

| Rule | Why |
|---|---|
| The same unmodified vLLM 0.32, model, GPUs, prompts and output lengths for every method | only the KV path differs |
| The same GPU memory shares for vLLM for every connector, NIXL and LMCache included: 0.55 on the prefill GPUs, 0.60 on the decode GPUs (0.65 until job 30304) | nvCOMP and ExANS need room beside vLLM's cache, and at 0.65 T2 ran out of memory coding an 8,192-token request beside every other codec's constants (job 30304); giving the same shares to everyone keeps the engines identical |
| Every engine's prefix cache emptied before each check codec and each sweep or multi-turn point | the points reuse the same prompts; otherwise later methods skip the prefill or send only part of the prompt (job 30210) |
| Every prefill-decode pair warmed up with one short request before each codec's sweep | a pair's first transfer opens its path (CUDA IPC handles, NIXL's handshake), which costs 0.2-0.5 s once |
| A burst of short requests with the benchmark's own sampling before each codec's first point, the tier and the multi-turn runs; vLLM's "JIT compilation during inference" warnings reported with the point they fell in | the check samples greedily, so the first measured point compiled vLLM's top-p kernels (about 2 s) in job 30226: raw's and NIXL's TTFTs included it |
| The sweeps greedy (`--temperature 0`) | vLLM 0.32's `vllm bench serve` no longer sends temperature 0, so the sweeps sampled at temperature 0.6 until job 30252, and each codec's run wrote different text |
| The proxy balances prefill engines by requests in flight, counted when it picks | until job 30210 a burst of requests all went to one engine |
| The client side (prompts, proxy, benchmark) runs from the same Python environment for every method | LMCache's environment lacked a benchmark dependency |
| Correctness first: each codec's check pass (with checksums) runs before its timed runs and is reported beside them | a fast codec that is wrong is not a result |

Layouts on the 4-GPU node (gpu-grad-01, 4× RTX A5000):
- 1 prefill + 3 decode, and 2 prefill + 2 decode;
- each through host memory at set link speeds (`ring`) and GPU to GPU (`p2p`);
- a CPU tier on 4 colocated engines;
- T2 on eager engines (`t2eager`).

The runbook is `docs/pd_baselines.md`.

---

## 8. What is measured so far

### 8.1 The codec on its own (RTX A4500, Llama-3.2-3B, 8K windows)

| Claim | Number | Source |
|---|---|---|
| T dense, bit-exact on the GPU | 9.03 bits/value (WikiText-2) | `docs/lossless_related_work.md` |
| T head by head | 8.32-8.41 bits/value | same |
| T2 dense, mean of five datasets | 8.45 bits/value (-9.0% against T) | `docs/t_ideas.md`, stage 7 |
| T2 head by head | 7.80 bits/value (2.05×) | stage 8 |
| ExANS, smallest bit-exact baseline | 10.87 bits/value (1.47×) | stage 8 |
| Layer 0 recomputed bit for bit (simulated vLLM schedules, padded to the sender's M) | 100% of words | stage 2d |
| Layer-0 saving | 0.354-0.362 bits/value over the cache | stage 2 |

### 8.2 Inside vLLM PD (4× RTX A5000, smoke job 30210)

Correctness: our connector was bit-exact on 4 of 4 check prompts (1,000-8,192 tokens) for raw, SplitZip,
ours-hbh, T2-hbh and T2-rank256, each on the paths it ran (the host-memory ring, the GPU-to-GPU path, eager
engines). The outputs were identical to the decode engine computing the prompt itself. NIXL also ran.

| Codec | Bits/value in the check pass |
|---|---|
| raw | 16.000 |
| nvCOMP ANS | 13.367 (3 of 4: out of GPU memory, since fixed) |
| SplitZip | 12.044 |
| ours-hbh (T) | 9.053 |
| t2-rank256 | 9.578 |
| t2-hbh | 8.489 |

Caveats:
- **T2's layer 0 was coded, not recomputed, in this run.** The cause was the prefix-cache reuse fixed since:
  T2's prefill hit its own cache, so its steps did not cover the prompt. So the T2 figures show the constant-set
  gain only.
- **ExANS did not run** (out of GPU memory, since fixed).
- **The speed numbers of job 30210 are not comparable** (section 7: prefix-cache reuse, prefill imbalance, first
  transfers). The next smoke measures them fairly.

### 8.3 Not measured yet

- TTFT, throughput and goodput in vLLM, fairly, against NIXL and every codec (next smoke, then the full
  matrices).
- How often layer 0 is recomputed in compiled versus eager engines.
- The CPU tier with the tier actually used: earlier smokes had no loads. The smoke is resized for this.
- IBP (it has never built on our machines).

---

## 9. AQUA-KV in PD: the closest method, step by step

**In plain words.** AQUA-KV is the closest published method: like us, it guesses each layer from the layer below.
But it rounds the gap, so the cache changes, and it was built to save memory on one GPU, not to move a cache
between two. Put into PD, it changes answers. It also has nowhere good to live on the decode GPU, its rebuild
chain spans two engines it was not designed for, and it spends bits on things the receiver could compute. Each
of those gaps is where one of our ideas comes in.

### 9.1 How AQUA-KV works

From the paper (§3.1-3.3, §4.1, §5, Algorithms 1-4) and its reference code:

| Step | What AQUA-KV does |
|---|---|
| Calibration | RedPajama: 256 sequences of 8,192 tokens (224 to fit, 32 to choose settings). Predictors are fitted one layer at a time, each on the rebuilt (rounded) outputs of the layer before. Linear regression in closed form; the paper's analysis uses a 10⁻³ regularizer |
| The guess | keys of layer l from the rebuilt keys of layer l-1; values of layer l from the rebuilt values of layer l-1 plus the rebuilt keys of layer l. Keys are predicted before RoPE. Earlier tokens and more distant layers added little, so they are not used |
| The gap | the real value minus the guess, rounded by a pluggable quantizer: HIGGS (a Hadamard transform plus vector quantization) or Quanto (round to nearest). 2, 3 or 4 bits per value; 2.09, 3.06 and 4.02 bits with the predictors' overhead |
| What is stored | only the rounded gaps; the guesses are recomputed when decoding |
| Layer 0 | no guess; quantized directly, at 4 bits by default for Llama |
| Tokens kept raw | the first 4 (attention sinks) and up to the newest 128 (a buffer, compressed in batches as it fills), at 16 bits |
| Decoding | before each forward call, every layer in order, each from the one rebuilt before it; one layer is materialised at a time |
| Implementation | Hugging Face transformers reference code; HIGGS through FLUTE kernels; "efficient inference kernels" still to come; about 3% slower generation in their unoptimised test |
| Quality at 2.09 bits | Llama-3.2-3B: WikiText-2 perplexity 7.03 against 6.98, LongBench 44.30 against 44.61. Llama-3.1-8B: perplexity 5.72 against 5.61. Llama-3.1-70B: 2.62 against 2.54 |
| Their stated future work | bit widths adjusted per component (they "deliberately forego" dynamic bit widths), predictors trained for model-level objectives, vLLM integration |

Their README's example calibrates `unsloth/Llama-3.2-3B` at 8,192 tokens: the model we serve, so a like-for-like
comparison is direct.

### 9.2 Side by side

| Step | AQUA-KV | Ours (T2 in PD) |
|---|---|---|
| Calibration | once, RedPajama | five sets (general, code, math, exam, news), one picked per request |
| Where the cache goes | stays in one GPU's memory | moves from the prefill GPU to the decode GPU, through a vLLM connector |
| Layer 0 | quantized, 4 bits | not sent: rebuilt from the token ids |
| The guess | from the layer below; keys, then values; floating point | from the layer below; head by head; exact integers |
| The gap | rounded, 2-4 bits, the same for every value | coded exactly with rANS; bits follow how good each guess was |
| Tokens kept raw | first 4 and up to the newest 128 | none: every token coded, with an escape for any odd value |
| Decoding | at every step, all layers | once, on arrival, into vLLM's normal cache, with a checksum |

Only one step is shared: guessing from the layer below (both de-rotate the keys first).

### 9.3 What goes wrong if AQUA-KV is used in PD, even when lossy is accepted

**The contract: answers change, and depend on the route.** PD should be invisible: the decode GPU should carry on
exactly as if it had done the prefill itself.
- With AQUA-KV it continues from a close copy of the cache. Attention reads slightly different keys and values, so
  the next-token scores move a little. Usually the top token is unchanged; sometimes it is not. With greedy
  decoding, everything after a changed token differs.
- Whether a request goes through PD depends on load and routing. The same prompt can then get one answer on a
  single GPU, another through PD, and a third when a failed transfer falls back to local prefill. The network
  path becomes part of the model.
- The cost is not one number: in their own results it grows with model size (perplexity +0.7% on 3B, +2% on 8B,
  +3% on 70B at 2 bits) and varies by task. Every new model, bit width, context length and workload must be
  re-tested. With an exact transfer there is nothing to re-test.

**Gap 1: the compressed cache has nowhere good to live on the decode GPU.** AQUA-KV exists to save memory: the
cache stays compressed in memory and is unpacked as attention needs it. In PD the compressed cache crosses the
wire and the decode GPU has two choices:

| | Option A: unpack once, on arrival | Option B: keep it compressed |
|---|---|---|
| What happens | expanded into vLLM's normal full-size cache | stays packed, as AQUA-KV intends |
| Decode GPU memory | full size, as with no compression | small: AQUA-KV's real benefit |
| The cost | the cache was rounded anyway: a quality change paid only to shorten the transfer | attention must read AQUA-KV's format at every new token; vLLM's attention reads only its own blocks, and AQUA-KV's fast kernels are still to come, so the serving engine needs a new attention path |

So in PD, AQUA-KV either loses its main benefit (memory) or needs a serving engine that does not exist yet.
Option A is what a port can do today. Its trade is a changed answer for a smaller transfer. An exact codec gives a
smaller transfer, less small, with no change at all.

**Gap 2: the sender's and receiver's copies can drift apart.** AQUA-KV rebuilds the cache as a chain: rebuilt
layer 1 = guess from rebuilt layer 0 + rounded gap; rebuilt layer 2 = guess from rebuilt layer 1 + rounded gap;
and so on to layer 27. The sender computes each gap against its own guess; the receiver adds it to its own guess.
That works only if the two guesses are the same.
- On one GPU (AQUA-KV's setting), the same code packs and unpacks, so they are.
- In PD, packing happens on the prefill GPU, in a batch with other requests; unpacking happens on the decode GPU,
  in another batch and possibly with another kernel. Floating-point results can differ in the last bits.
- A tiny difference at layer 1 changes the guess for layer 2, which changes layer 2 a little more, up the chain.
  The sender cannot know.
- **How big is it?** Probably small for AQUA-KV, because its 2-bit rounding error is far larger than a
  last-digit difference. It is a hypothesis until measured (section 9.4); if it is small, we drop it.
- It is exactly the problem our design removes: our guesses are whole-number arithmetic, the same on both engines,
  and the receiver rebuilds the original values, not a rounded copy, so there is nothing to drift.

**Wastes: effort spent on what PD or the data gives for free.**

| Where AQUA-KV spends effort | Why it is waste in PD | Our idea | What it is worth (ours, measured) |
|---|---|---|---|
| Layer 0 at 4 bits, about twice its other layers | the receiver has the model and the words, and could compute it in about 10 ms. It is about 7% of what AQUA-KV sends, if its 2.09 average includes layer 0 at 4 bits | rebuild layer 0 on the receiver, bit for bit (T2) | 0 bits for layer 0: about 0.36 bits per value, roughly 4% |
| The same bits for every value | an easy value gets as many bits as a hard one; their own future work | entropy coding with a spread per channel and token | built in; not yet measured on its own |
| One calibration text | guesses get worse on other text: with general-text constants, code costs 7.5% more bits, math and exam questions 3-4% more | five sets, one picked per request in 11 ms | about 5% (8.78 to 8.34 bits per value) |
| One step within a layer: keys, then values | the heads of a layer also predict each other | head by head, 16 steps per layer | on one text, 8.56 bits (their order) to 8.12, about 5% |
| The first 4 and up to the newest 128 tokens at 16 bits | in a transfer they travel at full size; after prefill the prompt is complete, so the buffer has no reason to stay raw | code every token | worst case for a naive port, if 2.09 excludes them: 3.9 bits for a 1,000-token prompt, 3.0 for 2,000, 2.3 for 8,000 |
| The whole cache decoded at every new token | compute repeated thousands of times in a long answer | decode once, on arrival | one decode per request |

One more possible waste is still to measure: AQUA-KV guesses keys only from the previous layer's keys, while our
guess uses the whole previous layer, keys and values.

The numbers in this table come from different texts and runs; the ladder in section 9.4 puts them into one table.

**What we must not claim.** A careful port works. On a slow link, a 2-bit transfer still beats our 8 bits on time.
The claim is not "AQUA-KV fails in PD" but: in PD it changes answers, buys only wire time (or needs a new attention
path), wastes layer 0, and its rebuild chain spans two engines it was not designed for. We keep answers identical
and close those wastes.

### 9.4 How we show it: an AQUA-like codec in our harness

Not built yet. Our own code, not their repository, so it runs in the normal card workflow:
- the same linear guesses (keys from the previous keys; values from the previous values plus the current keys),
  fitted on the same calibration windows;
- the gap rounded to 2 bits, rebuilt layer by layer as they do (closed loop);
- layer 0 at 4 bits; the raw buffer optional.

One run then measures, beside every other codec:
- the bits that actually travel;
- the drift: the sender sends a checksum of its rebuilt values per layer, and the receiver compares its own;
- answer agreement with single-GPU serving, against raw transfer as the baseline (vLLM itself is not perfectly
  deterministic: in the 30210 smoke, NIXL's raw transfer matched 3 of 4 answers and ours 4 of 4);
- TTFT under the same rules as every codec (section 7).

The same harness gives the **ladder**, every step coded exactly on the same windows:
1. AQUA-KV's guess structure, made exact, layer 0 coded;
2. the whole previous layer as input;
3. head-by-head order;
4. layer 0 rebuilt;
5. per-request constant sets (T2).

---

## 10. What is new, what we lead with, and how we show it

**What is not new** (and we do not claim it):
- Predicting a value and coding the exact gap with an adaptive entropy coder: lossless float compression has done
  it for 20 years (Isenburg et al. 2004-2005; fpzip, 2006), escapes and adaptive contexts included.
- Integer arithmetic so that encoder and decoder agree: standard in lossless coding (JPEG-LS, FLAC, lossless
  JPEG 2000).
- Predicting KV from other layers (AQUA-KV, lossy).
- Entropy coding KV exactly (ExANS, ZipNN).
- Recomputing KV from upstream state (KVPR, KV-Direct).
- Batch-size-dependent results in LLM inference (the batch-invariance work on nondeterminism).
- Disaggregated serving itself.

### 10.1 The motivation: today's methods in PD

| If you use... in PD | You get | The gap |
|---|---|---|
| Raw transfer (NIXL) | 16 bits per value: an 8K prompt is 940 MB, 752 ms at 10 Gb/s | nothing compressed |
| Bit-exact codecs (SplitZip, ExANS, AdaCode, ...) | the same answers, at 10.9-12.7 bits | the exponent only; 8 of the 16 bits always travel raw |
| A predictive lossy codec (AQUA-KV) | 2.09 bits | answers change and depend on the route; the cache has nowhere good to live; the rebuild chain spans two engines (section 9) |
| Predictive exact coding with a whole model (speculative KV coding) | 5.9-6.7 bits, exact | the receiver re-runs the model on the prompt, the cost PD exists to avoid |
| All of the above | they send the whole cache | the receiver has the model and the prompt, and could compute part of it: layer 0 travels at 16, about 11, 4 or about 10 bits when it could cost nothing |

And the gap we found trying to close the last row: **recomputing on the receiver does not work naively.** Layer 0
recomputed under vLLM-style batching gave only 50-97% identical words, because the batch's row count changes the
last bits.

### 10.2 The principle

> In PD the decode engine is a second copy of the same model, holding the same prompt. Anything it can compute
> bit for bit need not be sent. We send only the rest: nothing for layer 0, and for every other layer only the
> exact surprise against a guess both engines compute identically.

### 10.3 The two leads

**Lead 1: exact coding of the whole number.** The first bit-exact KV codec to compress the whole 16-bit word, not
just its exponent, with a guess from the layer below that both engines compute identically.
- It holds against all three neighbours: "exact" excludes AQUA-KV, "the whole word, predicted" excludes the
  exponent codecs, "cheap, from the layer below" excludes speculative KV coding.
- It carries the big number: ExANS 10.87 bits, T 9.03, T2 7.80.
- Shown by: bits per value against every bit-exact codec on the same KV (have); checksums on every transfer in
  vLLM (have, smoke 30210); answer agreement against raw transfer, with the AQUA-like bar (section 9.4, to build);
  TTFT against link speed in vLLM (next smoke and the full matrices).

**Lead 2: layer 0 for free in PD.** The decode engine rebuilds layer 0 from the prompt's words, bit for bit, so it
never travels. The prefill records each step's matrix-product row count and the receiver pads to it: shape replay
instead of rewriting kernels to be batch-invariant.
- Only PD has a receiver that holds the model and the prompt; AQUA-KV spends 4 bits on this layer, the others send
  it.
- Shown by: identical words, 50-97% with a naive recompute and 100% with shape replay (have, simulated vLLM
  schedules); how often layer 0 is rebuilt inside real vLLM, eager against compiled (next smoke: in 30210 it was
  coded because of the prefix-cache reuse fixed since); the saving, about 0.36 bits per value.
- The question to expect: is 4% worth a 10 ms recompute? For an 8K prompt the 4% is about 21 MB: 17 ms at
  10 Gb/s, 1.7 ms at 100 Gb/s. On fast links it pays only if hidden, and it can be: the decode engine gets the
  prompt when the prefill starts, and the prefill's row counts are known when it schedules, so a few bytes sent
  ahead would let the receiver rebuild layer 0 during the prefill. Not built yet.

**Refinements, not leads:** bits that adapt to each guess (standard in entropy coding), five constant sets per
request (about 5%, weak on unseen text), head-by-head order (a generalisation of AQUA-KV's keys-then-values, and
our slowest mode).

A lead sentence could read: *"We present the first bit-exact KV codec that compresses the whole BF16 word, not
just its exponent, using a cross-layer guess that both engines of a disaggregated deployment compute bit for bit;
and in PD we send no layer 0 at all, because the decode engine rebuilds it exactly."*

### 10.4 Built for PD's shape

The design maps onto what PD needs, which is where AQUA-KV's placement problem (section 9.3, gap 1) does not
arise:
- **Compressed only on the wire.** The format exists for the transfer (and the CPU tier). The decode engine holds
  vLLM's normal cache, so attention, memory layout and every vLLM feature work unchanged.
- **Decoded once, straight into vLLM's blocks.** Each layer goes from the transfer to the request's paged blocks as
  soon as it is decoded (`kv_compare/pd/paged.py` maps vLLM's block view to our words and back). The receiver never
  holds a second full copy.
- **The same result on both engines.** The guess is whole-number arithmetic and layer 0 is rebuilt by shape replay,
  so the two halves of the codec can run in different engines, batches and kernels.
- **One constants file for both ends,** exported once per model.

We do not claim memory savings on the decode GPU: that is a different goal (AQUA-KV's), and its lossy route is a
different contract.

### 10.5 What is ours, in one table

As far as our literature checks found (29 October and 15 November 2026):

| | Others | Ours |
|---|---|---|
| What is compressed | the exponent only; sign and mantissa (8 of 16 bits) sent raw | the whole 16-bit word, against a prediction |
| Resulting size | 10.9-12.7 bits/value | T 8.3-9.0; T2 7.8-8.5 |
| The prediction's cost | (none) / a whole second model on both sides (speculative coding) | one 8-bit integer product per layer, on the GPU |
| Bit-exact decoder on the GPU | yes (exponent codecs) / not shown (speculative coding) | yes, checked on every transfer of the correctness pass |
| Layer 0 | sent | T2: never sent; rebuilt by the receiver bit for bit, by matching the prefill's matrix-product row counts, with a sender-side check and a fallback |
| Adapting to the text | one setting | T2: one of five constant sets per request, 11 ms to pick, only its name sent |
| In a real serving engine | SplitZip: RoCE transfer tests; ExANS: its own vLLM connector | every bit-exact codec, NIXL and LMCache in the same unmodified vLLM PD setup |

The novelty is in this particular combination and its measured numbers, not in any one ingredient.
`docs/novelty_assessment.md` keeps the stricter mechanism-by-mechanism review. **On "first":** re-run the
literature check just before submission; this area moves fast.

**On origins:** our ideas were not derived from AQUA-KV (T came from our own predictor work, T2's ideas from the
`t_ideas` study). Present section 9's table as where existing methods leave room and how we close it, not as "we
started from AQUA-KV".

---

## 11. Limits and open questions

- **Speed in serving is not proven yet.** The size gains are measured and checked bit for bit; the TTFT and
  throughput gains in vLLM are the next measurements.
- **Very fast links favour simpler codecs.** At 400 Gb/s the decode time exceeds the wire time saved; AdaCode and
  SplitZip carry more requests there. The prize is the coding kernel: dense and rank 256 run at 2.4× their memory
  floor (15.9 against 6.7 ms). Head by head, the smallest mode, takes 108 ms only because its 16 head groups run
  one after another; one kernel per layer over all its groups is the next step.
- **Encode cost.** T needs a forward pass over the exact words before it can code (the features and priors), and
  rANS codes in reverse. T2 dense encodes an 8K request in 92 ms. Overlapping the encode with the link, layer
  by layer, is not built.
- **Same hardware on both ends.** The decoder must compute the same predictions as the encoder: the same GPU type
  for the float32 steps, and for T2's layer 0 the same kernels. The streams' static tables are portable, but
  cross-GPU decoding is not tested.
- **Compiled engines and layer 0.** Recompute identity is assured in eager engines. In compiled engines T2 may
  fall back to coding layer 0: still exact, a few percent larger.
- **Layer 0's recompute is not hidden yet.** It runs when the transfer arrives (about 10 ms for 8K tokens), which
  costs more than it saves above about 17 Gb/s. Sending the prefill's row counts ahead would let the decode engine
  rebuild layer 0 during the prefill (section 10.3).
- **Calibration dependence.** The big T2 gain needs the request's kind of text among the sets. On unseen text the
  gain is small (0.054 bits).
- **One model so far.** Everything here is Llama-3.2-3B. The method needs one offline calibration per model; other
  models are untested in PD.
- **The emulated network.** The host-memory ring emulates link speeds on one machine. The GPU-to-GPU path and the
  CPU tier are the machine's real paths; a real two-machine link is not tested yet.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| KV cache | the keys and values every layer keeps for every token, so later tokens can attend to them |
| Prefill / decode | reading the whole prompt at once / generating the answer one token at a time |
| PD (disaggregation) | prefill and decode on different GPUs or machines, with the KV moved between them |
| TTFT | time to first token: from the request to the first generated token |
| Goodput | requests per second that meet a latency target (here TTFT) |
| BF16 | a 16-bit float: 1 sign bit, 8 exponent bits, 7 mantissa bits |
| Exponent / mantissa | the power of two / the precise digits of a float |
| Bit-exact | the decoded words equal the original 16-bit words exactly |
| Bits per value | the coded size divided by the number of values; 16 = no compression |
| Entropy coding | coding each symbol in about -log₂ P bits, given a probability model; rare symbols cost more |
| rANS | a fast entropy coder (range asymmetric numeral systems) that suits GPUs; it codes in reverse order |
| Logistic distribution | a bell-shaped distribution with heavier tails than a Gaussian; our model of the surprise |
| Spread | the logistic's scale: how far from the prediction values usually fall |
| Surprise | the distance from the prediction, divided by the spread: how unexpected a value was |
| RoPE | rotary position embedding: keys are rotated by an angle that depends on the token's position |
| De-rotate | undo RoPE, so keys at different positions can be compared or predicted |
| 8-bit features | the previous layer's values scaled and rounded to integers in -127..127 |
| Calibration | fitting the predictor's constants offline on sample text |
| Constant set | one full set of T's constants, fitted on one kind of text |
| Layer 0 | the first transformer layer; its K and V depend only on each token's id and position |
| GEMM row count (M) | the number of rows in a matrix product: every token of the prefill step; it decides cuBLAS's algorithm and so the exact rounding |
| Eager / compiled engine | vLLM running PyTorch modules one by one / with torch.compile and CUDA graphs |
| Paged cache, blocks | vLLM stores KV in fixed-size blocks of 16 tokens, allocated per request |
| KV connector | vLLM's plug-in interface for moving KV between engines or to storage |
| NIXL | NVIDIA's transfer library; vLLM's NixlConnector moves raw KV GPU to GPU with it |
| LMCache | a KV store for vLLM; here its CPU offload, the reference for a CPU tier |
| CPU tier | KV kept in host memory after a request, loaded back when a later request shares its prefix |
| CUDA IPC | sharing GPU memory between processes, so one process can copy straight into another's GPU memory |
| Lossy / lossless | the decoded cache is close to / exactly equal to the original |
| Closed loop | a predictive codec whose guesses use the rebuilt (decoded) values, so encoder and decoder stay in step; a lossless codec's rebuilt values are the originals |
| Drift | the decoder's rebuilt values moving away from the encoder's, when their guesses differ slightly and the difference feeds the next layer's guess |
| Attention sinks | the first few tokens, which attention leans on heavily; lossy methods often keep them uncompressed |
| Perplexity | how surprised the model is by real text; lower is better, and a rise after compression means lost quality |
| LongBench | a benchmark of long-context tasks (question answering, summarisation, code) |
| Shape replay | our layer-0 recompute: the receiver repeats the prefill's matrix-product row counts, so it gets the same bits |
| Batch invariance | kernels whose results do not depend on the batch's size; the usual fix for batch-dependent results, which shape replay avoids needing |

---

## 13. Where things live

| What | File |
|---|---|
| T's predictor constants and fused kernels | `kv_compare/fused.py` |
| T layer by layer (the PD codec's engine) | `kv_compare/layerwise.py` |
| Head-group modes | `kv_compare/grouped.py` |
| The value-slice rANS coder | `kv_compare/gpu_coder.py` |
| Exact cost model and fits | `kv_compare/core.py`, `kv_compare/predict.py` |
| T2 | `kv_compare/t2.py` |
| Layer 0 recomputed inside vLLM | `kv_compare/pd/layer0.py` |
| Every PD codec behind one interface | `kv_compare/pd/codecs.py` |
| The connector | `kv_compare/pd/connector.py` |
| Transports and the paced link | `kv_compare/pd/transport.py` |
| Wire format | `kv_compare/pd/wire.py` |
| Paged cache to words | `kv_compare/pd/paged.py` |
| CPU tier | `kv_compare/pd/tier.py` |
| Proxy, benchmark, runner | `scripts/pd_proxy.py`, `scripts/pd_bench.py`, `scripts/run_pd_vllm.sh` |
| Constants export (both ends) | `scripts/pd_export_codecs.py` |
| The experiments behind T2 | `docs/t_ideas.md` |
| The literature table | `docs/lossless_related_work.md` |
| Novelty review | `docs/novelty_assessment.md` |
| AQUA-KV's reference code (read, not run) | github.com/goodevening13/aquakv |
| PD runbook and matrices | `docs/pd_baselines.md`, `docs/pd_protocol.md` |

---

## 14. Sources

- [SplitZip, arXiv 2605.01708](https://arxiv.org/html/2605.01708v3)
- [ExANS, OpenLake](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache)
- [AdaCode, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.269/)
- [exactkv](https://github.com/zyfyz666/exactkv)
- [IBP, arXiv 2605.30728](https://arxiv.org/pdf/2605.30728)
- [GAAC summary](https://welcome.ai/content/gaac-efficient-kv-compression-for-llms-on-personal-devices.md) (paper: [arXiv 2610.38210](https://arxiv.org/abs/2610.38210))
- [UCCL-Zip, arXiv 2604.17172](https://arxiv.org/abs/2604.17172)
- [Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/)
- [AQUA-KV, arXiv 2501.19392](https://arxiv.org/html/2501.19392v4) (§3.1-3.3, §4.1, §5, Algorithms 1-4, Table 2) and its [reference code](https://github.com/goodevening13/aquakv)
- [Isenburg et al., lossless compression of predicted floating-point values](https://www.cs.unc.edu/~isenburg/research/lcpfpv)
- [Lindstrom and Isenburg, Fast and Efficient Compression of Floating-Point Data (2006)](https://www.academia.edu/575322/Fast_and_efficient_compression_of_floating_point_data)
- [Thinking Machines, Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [AnchorKV, arXiv 2608.02901](https://arxiv.org/pdf/2608.02901)
- [KVPR, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.997/)
- [KV-Direct, arXiv 2603.19664](https://arxiv.org/html/2603.19664v1)
- [Lossless Compression of Neural Network Components, arXiv 2508.19263](https://arxiv.org/abs/2508.19263)
