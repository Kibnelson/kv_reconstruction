# Our approach: lossless KV transfer for prefill/decode disaggregation

A reference for what we built, how each part works, why it is built that way, how it compares with the other
methods, and what is measured so far. Written 15 November 2026 on branch `pd-baselines`.

- **T:** our bit-exact KV codec.
- **T2:** T with two additions for PD.
- **TKVConnector:** the vLLM connector that carries either one between GPUs.

Each section opens with a short "in plain words" summary; the details follow.

Contents:
1. [The problem](#1-the-problem)
2. [How the other lossless codecs work, and where they stop](#2-how-the-other-lossless-codecs-work-and-where-they-stop)
3. [T: predict each value, code only the surprise](#3-t-predict-each-value-code-only-the-surprise)
4. [T2: layer 0 rebuilt by the receiver, constants picked per request](#4-t2-layer-0-rebuilt-by-the-receiver-constants-picked-per-request)
5. [Moving it inside vLLM: TKVConnector](#5-moving-it-inside-vllm-tkvconnector)
6. [How we compare, fairly](#6-how-we-compare-fairly)
7. [What is measured so far](#7-what-is-measured-so-far)
8. [What is new, and what is not](#8-what-is-new-and-what-is-not)
9. [Limits and open questions](#9-limits-and-open-questions)
10. [Glossary](#10-glossary)
11. [Where things live](#11-where-things-live)
12. [Sources](#12-sources)

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
| AQUA-KV (ICML 2025) | predicts each layer from earlier layers | quantizes the residual: lossy |
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
  - A failed install fails only that request, not the engine.
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

## 6. How we compare, fairly

The comparison is only meaningful if every method gets the same conditions. The runner (`scripts/run_pd_vllm.sh`,
`scripts/pd_bench.py`) enforces these rules:

| Rule | Why |
|---|---|
| The same unmodified vLLM 0.32, model, GPUs, prompts and output lengths for every method | only the KV path differs |
| The same GPU memory share for vLLM (0.65) for every connector, NIXL and LMCache included | nvCOMP and ExANS need room beside vLLM's cache; giving it to everyone keeps the engines identical |
| Every engine's prefix cache emptied before each check codec and each sweep or multi-turn point | the points reuse the same prompts; otherwise later methods skip the prefill or send only part of the prompt (job 30210) |
| Every prefill-decode pair warmed up with one short request before each codec's sweep | a pair's first transfer opens its path (CUDA IPC handles, NIXL's handshake), which costs 0.2-0.5 s once |
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

## 7. What is measured so far

### 7.1 The codec on its own (RTX A4500, Llama-3.2-3B, 8K windows)

| Claim | Number | Source |
|---|---|---|
| T dense, bit-exact on the GPU | 9.03 bits/value (WikiText-2) | `docs/lossless_related_work.md` |
| T head by head | 8.32-8.41 bits/value | same |
| T2 dense, mean of five datasets | 8.45 bits/value (-9.0% against T) | `docs/t_ideas.md`, stage 7 |
| T2 head by head | 7.80 bits/value (2.05×) | stage 8 |
| ExANS, smallest bit-exact baseline | 10.87 bits/value (1.47×) | stage 8 |
| Layer 0 recomputed bit for bit (simulated vLLM schedules, padded to the sender's M) | 100% of words | stage 2d |
| Layer-0 saving | 0.354-0.362 bits/value over the cache | stage 2 |

### 7.2 Inside vLLM PD (4× RTX A5000, smoke job 30210)

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
- **The speed numbers of job 30210 are not comparable** (section 6: prefix-cache reuse, prefill imbalance, first
  transfers). The next smoke measures them fairly.

### 7.3 Not measured yet

- TTFT, throughput and goodput in vLLM, fairly, against NIXL and every codec (next smoke, then the full
  matrices).
- How often layer 0 is recomputed in compiled versus eager engines.
- The CPU tier with the tier actually used: earlier smokes had no loads. The smoke is resized for this.
- IBP (it has never built on our machines).

---

## 8. What is new, and what is not

**What is not new** (and we do not claim it):
- Predicting a value and coding the residual exactly. Audio, image and float compressors have done it for decades.
- Predicting KV from other layers (AQUA-KV, lossy).
- Entropy coding KV exactly (ExANS, ZipNN).
- Recomputing KV from upstream state (KVPR, KV-Direct).
- Disaggregated serving itself.

**What is ours**, as far as our literature checks found (29 October and 15 November 2026):

| | Others | Ours |
|---|---|---|
| What is compressed | the exponent only; sign and mantissa (8 of 16 bits) sent raw | the whole 16-bit word, against a prediction |
| Resulting size | 10.9-12.7 bits/value | T 8.3-9.0; T2 7.8-8.5 |
| The prediction's cost | (none) / a whole second model on both sides (speculative coding) | one 8-bit integer product per layer, on the GPU |
| Bit-exact decoder on the GPU | yes (exponent codecs) / not shown (speculative coding) | yes, checked on every transfer of the correctness pass |
| Layer 0 | sent | T2: never sent; rebuilt by the receiver bit for bit, by matching the prefill's matrix-product row counts, with a sender-side check and a fallback |
| Adapting to the text | one setting | T2: one of five constant sets per request, 11 ms to pick, only its name sent |
| In a real serving engine | SplitZip: RoCE transfer tests; ExANS: its own vLLM connector | every bit-exact codec, NIXL and LMCache in the same unmodified vLLM PD setup |

In one sentence: a cheap, exact, whole-word predictive codec that beats the exponent-coding floor, plus a PD-specific
way to send no layer 0 at all.

The novelty is in this particular combination and its measured numbers, not in any one ingredient.
`docs/novelty_assessment.md` keeps the stricter mechanism-by-mechanism review.

---

## 9. Limits and open questions

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
- **Calibration dependence.** The big T2 gain needs the request's kind of text among the sets. On unseen text the
  gain is small (0.054 bits).
- **One model so far.** Everything here is Llama-3.2-3B. The method needs one offline calibration per model; other
  models are untested in PD.
- **The emulated network.** The host-memory ring emulates link speeds on one machine. The GPU-to-GPU path and the
  CPU tier are the machine's real paths; a real two-machine link is not tested yet.

---

## 10. Glossary

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

---

## 11. Where things live

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
| PD runbook and matrices | `docs/pd_baselines.md`, `docs/pd_protocol.md` |

---

## 12. Sources

- [SplitZip, arXiv 2605.01708](https://arxiv.org/html/2605.01708v3)
- [ExANS, OpenLake](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache)
- [AdaCode, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.269/)
- [exactkv](https://github.com/zyfyz666/exactkv)
- [IBP, arXiv 2605.30728](https://arxiv.org/pdf/2605.30728)
- [GAAC summary](https://welcome.ai/content/gaac-efficient-kv-compression-for-llms-on-personal-devices.md) (paper: [arXiv 2610.38210](https://arxiv.org/abs/2610.38210))
- [UCCL-Zip, arXiv 2604.17172](https://arxiv.org/abs/2604.17172)
- [Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/)
- [AQUA-KV, arXiv 2501.19392](https://arxiv.org/html/2501.19392v4)
- [AnchorKV, arXiv 2608.02901](https://arxiv.org/pdf/2608.02901)
- [KVPR, Findings of ACL 2025](https://aclanthology.org/2025.findings-acl.997/)
- [KV-Direct, arXiv 2603.19664](https://arxiv.org/html/2603.19664v1)
- [Lossless Compression of Neural Network Components, arXiv 2508.19263](https://arxiv.org/abs/2508.19263)
