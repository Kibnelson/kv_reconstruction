# Novelty assessment of the KV representation experiments

Literature checked on **2026-10-19**. This is a mechanism comparison, not a claim of research priority. Published papers, preprints, author reports and implementation sources are identified separately below. Not finding our exact packet layout does not establish that it is new.

**Most broad ingredients already exist.** Our current work tests particular exact representations and their costs. A contribution would need a specific new predictor, code format or execution mechanism, with a measured advantage over the closest controls. Neither encoding during prefill nor reaching 40% reduction would establish novelty by itself.

## Comparison by representation

| Our approach and flow | Closest established mechanism | Assessment of the distinction |
|---|---|---|
| **Frozen exponent tiers:** split BF16 fields; encode exponent offsets with short common codes and exact escapes; retain sign/fraction. | SplitZip uses calibrated common-exponent codes, sparse escapes and raw sign/fraction [1]. ExANS uses reversible local exponent alignment and rANS [2]. | A particular tier/base/layout variant. Exponent redundancy, exact escapes, local metadata and lossless GPU KV transfer are established. Keep this as a control. |
| **Pre-RoPE storage and replay:** encode normalized keys before rotation; restore and apply RoPE; correct any mismatch against original BF16 keys. | KVQuant stores keys before RoPE and reapplies rotation during attention [3]. | Pre-RoPE storage is established. Backend-bound exact replay/corrections differ from KVQuant's lossy quantization, but that addition alone does not establish a new method. |
| **Huffman exponents:** encode complete exponent-residual symbols with frozen TRAIN books; retain other fields. | ZipNN isolates and Huffman-codes exponents [4]. Later work explicitly applies layerwise exponent Huffman coding to KV [5]. | Established coding family applied to our source and packet format. The source/base/book choice is an experimental difference. |
| **Huffman exponents + temporal signs:** add within-channel previous-token sign XOR, tile seeds and a Huffman stream of packed transitions. | Gorilla losslessly exploits previous floating-point values through XOR [6]; exponent coding above is established. | A specific sign-only KV adaptation and combination. We found no verified identical layout in this screen, but cannot claim firstness. Compare it against simpler sign/word XOR controls. |
| **KV anchors + point prediction + exact corrections:** retain a few actual same-token joint K/V coordinates; predict the rest. | AQUA-KV predicts from reconstructed earlier-layer KV and quantizes residuals [7]. Linear Predictability retains reference heads and predicts other heads [8]. | Our coordinate anchors differ from full previous-layer KV or reference heads; original-word corrections differ from lossy residuals. The broad prediction/reconstruction idea is established. No useful retained learned packet has yet been shown in our Qwen run. |
| **Prefill hidden-derived latent + point prediction + exact corrections:** project observed normalized hidden states into a small retained feature vector; predict joint K/V. | Palu caches low-rank projection latents and reconstructs KV [9]. XC-Cache's JIT-KV baseline stores per-layer hidden states and projects them back to KV [10]. | Our joint supervised feature fit and exact corrections are particular choices for a fixed pretrained model. Upstream-state storage and latent-to-KV reconstruction are established. Hooks alone add no novelty [11]. |
| **Previous-layer neural probabilities + exact arithmetic coding:** use decoded KV to predict distributions; encode exact symbols under those distributions. | AQUA-KV already uses cross-layer dependencies [7]. The public Speculative KV coding note describes learned distributions and exact arithmetic-coded KV [12]. | Our cheaper conditioner, shared/per-layer variants and indexed stream are different designs. Learned probability coding of exact KV is already publicly proposed. Actual Qwen rates, weight costs and decode time remain to be established. |
| **Earlier dictionary / E8 / coset tracks:** retain an index or representative plus the information needed to recover exact words. | CommVQ uses additive codebooks and RoPE-compatible computation [13]; E8 key quantization uses lattice representatives [14]; coset source coding with decoder side information is classical [15]. | These mathematical ingredients are established. Original-BF16 corrections differ from lossy VQ, but do not make VQ or coset coding itself new. Our modular low-bit nearest-representative scheme is not evidence of a new Slepian–Wolf codec or coding theorem. |

## Important boundaries

**Exactness is a distinction from AQUA-KV and PackKV, but not from all prior work.** AQUA-KV quantizes prediction residuals. PackKV combines lossy quantization, repacking and lossless bitpacking of the resulting integers; it also consumes decoded data inside computation kernels [16]. Neither preserves the original BF16 tensor merely because a later stage is lossless. ExANS, SplitZip and exponent-Huffman KV coding already target lossless original values. We therefore need comparisons with those exact codecs as well.

Our floating-point prediction and RoPE corrections are tied to the recorded backend contract. Exact original KV words do not by themselves prove portable predictor replay or identical attention outputs after changing reduction order.

**Prefill access is an opportunity to find a useful signal, rather than a novelty claim.** KVPress already exposes hidden states and KV to prefill hooks [11]. Palu and JIT-KV already retain information upstream of produced KV. We must identify the particular signal that reduces total retained bits or restoration work, including the cost of conveying that signal to the decoder. Keeping a full hidden state can even cost more than the joint KV width in a GQA model.

**Bounded staging and partial transfer also have precedents.** KVPR overlaps partial transfer and recomputation [17]. AlignPred combines alignment, prediction, error-bounded lattice coding and entropy coding, and describes bounded staging, indexing and layerwise restoration [18]. Its reported 128K, eight-request case still admits five requests and queues three with either compressed or native persistence. This illustrates why smaller persisted KV does not automatically increase active GPU capacity. Generic double buffering or layerwise restore is insufficient as a novelty claim.

Flash-Decoding already splits attention over KV chunks and combines partial results [19]. Under full-context attention, all required chunks must contribute before that token's attention finishes. Pausing and resuming processing must preserve this requirement; it cannot silently omit unavailable history.

## What our evidence currently supports

The [A4500 conditional review](qwen_conditional_gpu_review.md) reports **33.3898% mean estimated reduction**, including complete state, for exponent Huffman plus sign coding. Exponent-only reaches 32.9038% with about half the decode cost. Every learned mean/anchor/latent method selects the existing tier fallback on every measured TEST and DEV record. Those generation checks therefore validate fallback restoration, not a successful retained learned representation.

Current GPU decode/transfer costs do not demonstrate a migration speedup. The benchmark retains raw KV during capture and has no compact-resident attention, concurrency scheduler or partial-capacity admission. The subsequent [Qwen probability smoke review](qwen_neural_gpu_review.md) confirms exact learned pages and complete DEV continuation, but its first-16-position TEST sample cannot establish whole-context rates. Shared neural estimates 25.47% savings with state versus 26.04% for its matched control, and CPU arithmetic decoding dominates restoration. The measured figures and validation limits are evidence about this implementation, rather than comparisons with published systems on matched hardware.

## The contribution worth testing next

A narrower research hypothesis is:

> A specific producer-informed exact representation can reduce retained bits and reconstruction work enough to improve long-context request progress under a strict aggregate GPU memory budget.

The new element must be named and isolated: for example, a particular producer signal and conditional code that beat produced-KV-only conditioning, or an execution mechanism that improves admission or continuation beyond existing restore pipelines. Combining familiar ingredients is not sufficient evidence by itself.

Producer metadata is not free side information at the decoder. Its reduction in correction or symbol bytes must exceed the retained feature bytes and additional state/framing costs. A small function still needs enough input information to recover the original values; calling it reconstruction does not remove that accounting requirement.

Three comparisons would make the claim assessable:

1. **Representation:** compare identical held-out BF16 caches against exact exponent/escape and entropy baselines. Ablate the producer signal while keeping the coder, training budget and packet layout matched. Charge features, exact corrections, indices, headers, all weights and normalizers; report amortization separately.
2. **Restoration:** measure complete encode/transfer/decode time and peak workspace. Force learned packets in a separate DEV check so fallback cannot stand in for their correctness. Bound layer/token dependency chains and test reopening a packet without the original forward capture.
3. **Serving:** enforce one aggregate GPU budget covering model, state, compressed residency, materialized tiles and scratch across requests. Compare admission, queue delay, p99 latency and token throughput against uncompressed and exact-codec restore baselines. Count repeated transfers/reconstruction at every decode step.

The immediate implementation goal remains higher savings with lower restore cost. Our current evidence does not justify calling any broad approach uniquely ours. It does justify continuing the controlled experiments and narrowing the claimed contribution to whatever mechanism actually wins.

## Primary sources and locations

1. [SplitZip](https://arxiv.org/html/2605.01708v1), preprint, 2026, §3.2: common-exponent codes and exact escapes.
2. [ExANS](https://www.theopenlake.com/blog/exans-lossless-gpu-compression-for-bf16-kv-cache), author technical report, 2026: block-local alignment, independent rANS partitions and direct destination restoration.
3. [KVQuant](https://arxiv.org/html/2401.18079v4), NeurIPS 2024, §3.2: pre-RoPE key storage.
4. [ZipNN](https://arxiv.org/html/2411.05239v1), 2024 preprint, §3.1: exponent-only Huffman coding.
5. [Lossless Compression of Neural Network Components](https://arxiv.org/html/2508.19263v1), 2025 preprint, §3.3: exponent Huffman coding for KV caches.
6. [Gorilla](https://www.vldb.org/pvldb/vol8/p1816-teller.pdf), VLDB 2015, §4.1.2: previous-value XOR floating-point coding.
7. [AQUA-KV](https://arxiv.org/html/2501.19392v4), ICML 2025, §3.2/Algorithm 1: earlier-layer prediction and quantized residuals.
8. [Linear Predictability of Attention Heads in Large Language Models](https://arxiv.org/html/2603.13314v1), 2026 preprint, §4: retained reference heads and learned reconstruction.
9. [Palu](https://arxiv.org/html/2407.21118v2), ICLR 2025, §3.1 and Appendix B: low-rank latent caching and reconstruction kernels.
10. [XC-Cache](https://aclanthology.org/2024.findings-emnlp.896.pdf), Findings of EMNLP 2024, §2.3: JIT-KV is a baseline, distinct from the paper's architecture-changing XC method.
11. [NVIDIA KVPress `BasePress`](https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/base_press.py), implementation inspected 2026-10-19: prefill hooks and `compress` inputs. This URL follows the project's current main branch.
12. [Speculative KV coding](https://fergusfinn.com/blog/kv-entropy-coder/), public author research note, 2026: probability prediction and exact arithmetic coding. This is not a peer-reviewed system result independently reproduced here.
13. [CommVQ](https://proceedings.mlr.press/v267/li25du.html), ICML 2025: additive codebooks and RoPE-compatible operations.
14. [E8 Lattice Key Quantization](https://arxiv.org/abs/2610.15384), preprint submitted 2026-10-11: lossy lattice keys and lookup attention.
15. [Ramchandran's source-coding project/publications](https://people.eecs.berkeley.edu/~kannanr/projects/project_duality.html): DISCUS and generalized coset source coding, 1999–2003.
16. [PackKV](https://arxiv.org/html/2512.24449v2), preprint, §III: quantization, repacking, block format and computation-integrated decompression.
17. [KVPR](https://aclanthology.org/2025.findings-acl.997/), Findings of ACL 2025: partial transfer and recomputation pipeline.
18. [AlignPred](https://arxiv.org/html/2610.17970v1), preprint submitted 2026-10-13, Appendix D.1 and D.4: bounded staging and concurrent-request admission limits. Its error-bounded lattice representation is approximate, unlike our original-word target.
19. [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), author technical explanation, 2023: split KV attention and log-sum-exp combination.
