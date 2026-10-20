# Novelty assessment of the KV representation experiments

Literature checked through **2026-10-20**. This is a mechanism comparison, not a claim of research priority. Published papers, preprints, author reports and implementation sources are identified separately below. Not finding our exact packet layout does not establish that it is new.

**Most broad ingredients already exist.** Our current work tests particular exact representations and their costs. A contribution would need a specific new predictor, code format or execution mechanism, with a measured advantage over the closest controls. Neither encoding during prefill nor reaching 40% reduction would establish novelty by itself.

## Specific question: actual prefill metadata plus exact coset reconstruction

**Known ingredients do not establish that their particular combination is already published.** The earlier classification was too broad for this question. In the inspected primary sources, we have not found a complete match for:

> Retain a small code extracted from actual upstream prefill activations; use that code to disambiguate transmitted coset residues; recover the target model's original BF16 KV words exactly from the page and frozen state.

This remains a **specific novelty candidate**, not an established first contribution. Here, upstream metadata means a retained signal derived from the actual hidden state, normalization or projection computation. It does not mean a histogram calculated from final KV, or merely running a KV-only codec while prefill happens.

| Inspected work | Relevant mechanism | Boundary relative to this candidate |
|---|---|---|
| ExANS [2] | KV-derived exponent centers and exact rANS streams | Has metadata, but its described metadata comes from produced KV. Its modulo-256 recentering is reversible with the stored center; it does not omit a quotient that upstream information must disambiguate. |
| NestQuant [26], §4.1 | Explicit nested-lattice coordinates modulo q, including KV quantization | A close KV/coset precedent. It rounds to lattice points and selects minimum-energy representatives for lossy quantization, rather than using retained actual-prefill metadata to restore original BF16 words. |
| AQUA-KV [7] | Previous-layer KV prediction plus quantized residuals | Establishes predictive KV coding, but not the specified upstream-metadata-assisted exact coset decoder. |
| Palu [9] / KV-Direct [25] | Hidden-derived latent retention / sufficient residual-state replay | Establish upstream-state reconstruction, not this compact metadata plus exact coset construction. |
| L3C [24] / DISCUS [15] | Learned auxiliary information for exact image entropy coding / general syndrome-based source coding | Establish important coding ingredients. Neither source is a matching KV-specific prefill-metadata/coset construction. |
| AlignPred [18] / sequential Wyner–Ziv work [22] | Error-bounded predictive KV lattice coding / idealized query-side-information bounds | Neither inspected mechanism supplies the complete original-BF16, actual-prefill-metadata-assisted coset design above. |

The code audit also separates **four different experiments**:

| Our implementation | Information actually retained and used | What it has tested |
|---|---|---|
| `producer_pre_rope_coset`, `producer_kproj_norm_rope_coset`, later replay fallbacks | A complete encoded upstream K tensor, plus frozen replay state, position and exact corrections | Actual producer-stage replay with scalar coset corrections. This is not a tiny hidden-metadata codec. |
| `producer_hidden_exponent`, `producer_hidden_abs_exponent` | Selected int8 features of actual normalized hidden activations; integer predictor coefficients | Hidden metadata conditions exponent predictions. Corrections use signed exponent tiers, not coset disambiguation. |
| Independent `side_codec.py` hidden mode | Retained learned BF16 code from actual producer activations | The code conditions logistic probabilities for exact arithmetic coding. It does not use cosets. Its first smoke result does not reject the metadata-plus-coset hypothesis. |
| Direct `syndrome_codec.py` producer-code study | Small actual-hidden code plus exact fields and 31-bit binary coset corrections | Now tested on A4500. All measured planes remain raw, so the fitted code provides no correction-payload gain. This is a direct negative result for that extractor/predictor/code configuration, not all possible metadata-assisted coset designs. |

The conditional latent packets need a further distinction: hidden-derived BF16 features predict pre-RoPE K and V, with **residual** primary corrections. A separate replay stage may contain a coset fallback for pre-RoPE K to final K. That fallback uses the reconstructed full K source, not the tiny features as its disambiguating predictor. Coexistence in one packet is not the proposed coupling.

In the existing scalar implementation, a BF16 **integer word** x is paired with a replay prediction p. The encoder selects group widths b from the circular signed 16-bit error and may send x modulo 2^b. The decoder uses p to choose the matching representative. For the same predictor and group widths, this is an alternative representation of the signed correction, not an additional source of savings: the code explicitly checks identical bit counts. The [producer review](qwen_producer_gpu_review.md) confirms equal numeric payload sizes; the small total-byte difference comes from JSON spelling. Both replay paths needed zero K correction payload on all measured pages in that study.

**The precise open question is whether small upstream metadata can make coset disambiguation sufficiently cheaper to pay for itself.** The [prefill coset study](prefill_coset_study.md) implements this direct test using omitted-bit prediction and 31-bit binary syndrome groups, with explicit raw escapes and equivalent error-position controls. Its [first A4500 result](qwen_prefill_coset_gpu_review.md) has no retained short groups or correction-payload gain: hidden code estimates 28.91% net reduction versus 30.50% for the field baseline. Sparse optimizer channel coverage and approximately 13 errors per 31-bit group identify concrete fitting/code limitations. Better extractors or decoders remain open questions. A useful contribution could be a particular extraction/coding/restoration algorithm and its measured tradeoff; the current search neither establishes its priority nor dismisses the full combination as already matched.

The latest results separate three decisions:

- **Strongest reported simple codec:** exponent Huffman plus temporal sign coding, 33.39% mean estimated reduction including state in its earlier six-case study. This is a combination of established techniques.
- **Strongest demonstrated learned improvement:** shared previous-layer probability coding, 32.20% estimated net reduction at 8K versus 29.51% for its matched control. This is a measured +2.69 percentage-point improvement, with substantial prior conceptual overlap.
- **Most recent probability-track experiment:** a particular code derived from actual producer hidden states that conditions exact probabilities and removes previous-layer reconstruction prerequisites. The [first A4500 side-code smoke](qwen_prefill_side_gpu_review.md) validates exact independent restoration, but hidden code estimates 31.43% net savings versus 32.20% for no-code. It loses probability rate before side/state overhead. Only 200 updates were used; target-model advantage and research novelty remain unproven. This is not a test of the small producer-metadata-assisted coset candidate above.

The two headline codec percentages use different sampled pages and framing. They are evidence summaries, not a paired leaderboard. The [conditional review](qwen_conditional_gpu_review.md) and [neural position review](qwen_neural_positions_review.md) retain their separate protocols.

## Comparison by representation

| Representation family | Current Qwen evidence | Closest precedent and novelty assessment |
|---|---|---|
| **Exponent tiers / exact packing** | Working controls; latest matched 8K control estimates 29.51% net reduction. | SplitZip's common-exponent codes/exact escapes [1] and ExANS's local exponent alignment/rANS [2]. Our tier/base/layout choices are variants of established methods. |
| **Pre-RoPE storage and exact replay** | Earlier fast control: 30.94% mean net estimate. | Pre-RoPE storage and reapplication already appear in KVQuant [3]. Our original-word corrections differ from its lossy representation, but do not establish priority. |
| **Exponent Huffman** | Earlier six-case mean: 32.90% net estimate. | Exponent Huffman appears in ZipNN [4] and explicit KV coding [5]. A strong baseline rather than a broad new mechanism. |
| **Exponent Huffman + temporal sign coding** | Earlier six-case mean: 33.39% net estimate; sign coding adds about 0.486 points over exponent-only. | A specific adaptation of known exponent coding and previous-value XOR ideas [6]. Useful measured combination; no firstness established for its sign-only KV layout. |
| **KV anchors + point prediction + exact corrections** | Rank-8 and rank-32 candidates choose tier fallback on every measured Qwen DEV/TEST page. | AQUA-KV predicts from other KV [7]; reference-head reconstruction is also studied [8]. Our same-page coordinate anchors and original-word corrections are particular choices without a demonstrated retained-packet advantage. |
| **Actual hidden-derived latent + point prediction + exact corrections** | Both tested ranks choose tier fallback on every measured page. | Palu caches hidden-derived low-rank latents [9]; JIT-KV reprojects stored per-layer hidden states [10]. The exact residual format is a variant; hooks alone add no novelty [11]. |
| **Per-layer neural probabilities + exact coding** | Prefix smoke: 29.78% packet reduction, but -128.84% estimated net reduction at 2K after 456.83 MiB state. Prefix-only rate is not representative of whole contexts. | Cross-layer conditioning [7] and exact predictor-driven KV arithmetic coding [12] are established directions. This expensive parameterization is not our preferred candidate. |
| **Shared neural probabilities + exact coding** | Latest three-fixture 8K mean: 33.43% packet / 32.20% net estimate versus 29.51% matched net control. All measured neural pages are retained and exact. | A useful incremental conditioner/state design, with the same conceptual precedents [7,12]. Weight sharing, page-size amortization and exact entropy coding do not independently establish novelty. |
| **Earlier learned/fixed dictionaries, binary transforms, E8 and cosets** | Earlier prototypes use other configurations; no directly comparable winning current Qwen result is established here. Their percentages cannot be stacked with the later studies. | Codebooks [13], E8 quantization [14], reversible transforms and coset source coding [15] are established ingredients. Our exact symbol/correction formats do not establish a new coding theorem. |

The mean-predictor control also falls back everywhere and supplies no new retained representation. Raw transfer and zero-prediction controls test accounting/correctness, rather than research novelty.

The new [independent probability-page experiment](qwen_prefill_side_gpu_review.md)
adds a different learned representation: actual retained codes condition exact
entropy probabilities rather than full-word point-prediction corrections. Every
new side page is retained and exact, so this validates that mechanism without
fallback. However, the no-code probability model wins the first 2K smoke, and
all three code-bearing variants lose complete packet bytes on every layer.
The no-code variant is independent too. Independence therefore cannot yet be
credited as a unique benefit of the producer code. Longer training at fixed
width/rank is the next controlled test; no 8K side-code result exists yet.

## Important boundaries

**Exactness is a distinction from AQUA-KV and PackKV, but not from all prior work.** AQUA-KV quantizes prediction residuals. PackKV combines lossy quantization, repacking and lossless bitpacking of the resulting integers; it also consumes decoded data inside computation kernels [16]. Neither preserves the original BF16 tensor merely because a later stage is lossless. ExANS, SplitZip and exponent-Huffman KV coding already target lossless original values. We therefore need comparisons with those exact codecs as well.

Our floating-point prediction and RoPE corrections are tied to the recorded backend contract. Exact original KV words do not by themselves prove portable predictor replay or identical attention outputs after changing reduction order.

**Prefill access is an opportunity to find a useful signal, rather than a novelty claim.** KVPress already exposes hidden states and KV to prefill hooks [11]. Palu and JIT-KV already retain information upstream of produced KV. We must identify the particular signal that reduces total retained bits or restoration work, including the cost of conveying that signal to the decoder. Keeping a full hidden state can even cost more than the joint KV width in a GQA model.

**Learned side information that changes a probability model is also established.** Ballé et al.'s scale hyperprior transmits a learned side latent, decodes it first and uses it to predict entropy-model parameters while charging both streams [20]. Its image pipeline is lossy. More directly, L3C learns auxiliary representations jointly with probability models for **exact lossless** image coding [24]. Thus adding an exact entropy stream to a learned retained code is not a new generic construction. Related transformer representation coding also uses hyperpriors for lossy split inference [21]. The source being original BF16 KV is a relevant application distinction, but does not establish a new coding mechanism.

**Recreating KV from sufficient upstream states also has a closer precedent.**
KV-Direct proposes residual checkpoints and model-projection-based KV recreation
[25]. Its same-layer reconstruction discussion overlaps the broad motivation of
storing how KV is produced instead of all resulting tensors. Our tiny learned
conditioner plus exact uncertainty stream differs from retaining a sufficient
residual checkpoint. The paper's broader cross-layer memory and performance
claims were not independently implementation-audited here; they are not adopted
as guarantees for our design.

**Our earlier implemented cosets are scalar modulo codes.** The GPU producer codec
computes a circular signed residual, chooses group widths, then can transmit
the corresponding `x mod 2**b` residues. It explicitly checks that residual and
coset widths agree. Lower-field variants likewise use per-coordinate residues,
prototype IDs and correction widths. In these earlier formats there is no parity-check matrix,
cross-coordinate syndrome decoder or demonstrated Slepian–Wolf coding gain.
E8 uses an existing lattice projection followed by exact corrections. These
are legitimate mathematical representations, but their names do not establish
a new coding primitive. The subsequent [binary syndrome experiment](prefill_coset_study.md)
does add a parity-check matrix across 31 selected bits, with encoder-verified
single-error reconstruction and raw escapes. This is established syndrome
machinery used to test the producer-specific signal; no Slepian–Wolf gain or
target-model advantage is inferred from its implementation alone.

Producer access may make a useful conditioning signal cheap to obtain during the existing forward pass. That is a practical compute/representation hypothesis, not a guarantee of a lower information-theoretic limit. All retained information must still be counted.
**Bounded staging and partial transfer also have precedents.** KVPR overlaps partial transfer and recomputation [17]. AlignPred combines alignment, prediction, error-bounded lattice coding and entropy coding, and describes bounded staging, indexing and layerwise restoration [18]. Its reported 128K, eight-request case still admits five requests and queues three with either compressed or native persistence. This illustrates why smaller persisted KV does not automatically increase active GPU capacity. Generic double buffering or layerwise restore is insufficient as a novelty claim.

Flash-Decoding already splits attention over KV chunks and combines partial results [19]. Under full-context attention, all required chunks must contribute before that token's attention finishes. Pausing and resuming processing must preserve this requirement; it cannot silently omit unavailable history.

## What our evidence currently supports

The [A4500 conditional review](qwen_conditional_gpu_review.md) reports **33.3898% mean estimated reduction**, including complete state, for exponent Huffman plus sign coding. Exponent-only reaches 32.9038% with about half the decode cost. Every learned mean/anchor/latent method selects the existing tier fallback on every measured TEST and DEV record. Those generation checks therefore validate fallback restoration, not a successful retained learned representation.

Current GPU decode/transfer costs do not demonstrate a migration speedup. The benchmark retains raw KV during capture and has no compact-resident attention, concurrency scheduler or partial-capacity admission. The subsequent [Qwen probability smoke review](qwen_neural_gpu_review.md) confirms exact learned pages and complete DEV continuation, but its first-16-position TEST sample cannot establish whole-context rates. Shared neural estimates 25.47% savings with state versus 26.04% for its matched control, and CPU arithmetic decoding dominates restoration. The measured figures and validation limits are evidence about this implementation, rather than comparisons with published systems on matched hardware.

The [broader position study](qwen_neural_positions_review.md) now demonstrates the +2.69-point matched 8K improvement. It retains the same declared numerical predictor as the smoke; the controlled full-DEV page-size gain comes almost entirely from smaller headers. CPU arithmetic consumes about 99.67% of restore time. This is evidence for a learned representation, not an implemented high-concurrency materializer.

## Novelty-first research selection

The latest request prioritizes identifying a distinctive mechanism before
spending more effort on tuning. Longer training remains a useful diagnostic,
but it cannot establish novelty. The current selection is:

| Current direction | Role after the literature and code check |
|---|---|
| Exponent packing, Huffman and sign coding | Practical controls; do not make the broad mechanism the main contribution. |
| No-code or shared previous-layer neural probabilities | Strong learning/state controls; exact predictive coding and cross-layer conditioning have precedents. |
| Fixed/learned dictionaries, E8 and scalar cosets | Mathematical building blocks; a particular new KV code would need its own algorithm and measured advantage. |
| Generic hidden-state retention and KV replay | Overlaps Palu, JIT-KV and KV-Direct. |
| Generic learned retained code plus exact entropy stream | Overlaps learned lossless auxiliary-representation coding such as L3C. |
| **Small actual-prefill metadata that disambiguates exact KV coset residues** | **No complete match found in the inspected primary sources. First binary-syndrome variant now tested on Qwen, with no retained syndrome or payload gain. The specific fitting/code design needs improvement; priority and practical advantage remain unestablished.** |
| **A specific producer-aware extractor or restoration algorithm** | **Research candidate to define precisely. No unique advantage established yet.** |

Among the existing prototypes, the actual-producer track is the appropriate
testbed for that last question. Its current arbitrary learned hidden projection
is not by itself a sufficiently distinctive contribution. The next design must
name the particular forward information/computation it reuses, explain why an
equally useful KV-derived conditioner is more costly, and specify the new code
or reconstruction algorithm. Independent blocks, prefill hooks, exactness and
shared weights remain requirements or known ingredients.

Acceptance requires an advantage over no-code, same-page KV-derived metadata,
stronger KV-only models and practical independent exact codecs after retained
bits, weights, producer work and decoder work are charged. Better size or lower
restoration work can be valuable; novelty must attach to the particular
algorithm that yields the improvement. The present smoke supplies no such
producer-code advantage.

## Candidate contribution and proof obligations

A more specific research hypothesis is:

> A compact code derived from actual producer activations can replace previous-layer prerequisites for exact KV probability decoding, improving the complete-size versus random-page-restoration tradeoff for a fixed model.

The present shared model needs the same token tile's reconstructed previous-layer KV. Restoring a late-layer page can therefore require a depth-35 chain. Adding hidden features to that model might improve size while leaving the chain intact. The new page-local variant instead retains a small hidden-derived code and conditions the exact symbol probabilities on that code and frozen shared state alone. Its initial implementation compares no code, final-KV-derived code, inverse-RoPE-aligned final-KV-derived code, and actual-hidden-derived code. The canonicalized KV control avoids attributing an encoder's easier positional alignment to uniquely upstream information. These controls share code rank where applicable, decoder width, TRAIN data and update schedule; differing encoder dimensions and complete state still count.

Its flow would be: observe the actual normalized hidden state during prefill; produce and retain a compact code `z`; encode original BF16 KV under probabilities determined by `z`; later recover a requested page from its own `z`, symbol stream and shared decoder state. Neither original hidden tensors nor other layers' decoded KV may be required by the page-local variant.

Palu already has independently reconstructable latent tiles, ExANS has self-contained exact blocks, and L3C has learned auxiliary representations for exact entropy coding. Consequently, this combined design is only a research candidate. The contribution must be the particular producer code, fitting objective, packet format or execution mechanism that beats the closest alternatives after all costs are charged.

Producer metadata is not free side information at the decoder. Its reduction in correction or symbol bytes must exceed the retained feature bytes and additional state/framing costs. A small function still needs enough input information to recover the original values; calling it reconstruction does not remove that accounting requirement.

The decisive representation comparison is:

| Candidate | Information available to its decoder | What it tests |
|---|---|---|
| Existing shared model | Exact previous-layer KV | Current measured rate/dependency baseline |
| Produced-KV-only page code | Retained same-page KV-derived features or cheap statistics | Whether simple local side information is sufficient |
| Actual-hidden page code | Retained features derived from real producer activations | Whether upstream access improves the measured tradeoff |
| Hidden code plus previous-layer KV | Both sources, explicitly retaining the layer chain | Whether any size gain depends on cross-layer prerequisites |
| Stronger KV-only model | Matched parameter/compute budget | Whether more model capacity explains the gain |
| Independent exact field/entropy codecs | Their own page payload and frozen state | Whether learned pages improve on practical independent-block baselines |

Use identical KV pages, feature-bit budgets where applicable, discrete coder, training data and framing. Optimize and measure conditional code cost rather than treating point-prediction MSE as a substitute. Feature/state cost, replay work and scratch must enter the comparison. For a simple illustration at current geometry, eight BF16 features per token cost 0.0625 bits per KV value before state and framing; that cost is a threshold to beat, not a predicted saving.

Three validation layers would make a resulting claim assessable:

1. **Representation:** compare identical held-out BF16 caches against exact exponent/escape and entropy baselines. Ablate the producer signal while keeping the coder, training budget and packet layout matched. Charge features, exact corrections, indices, headers, all weights and normalizers; report amortization separately.
2. **Restoration:** measure complete encode/transfer/decode time and peak workspace. Force learned packets in a separate DEV check so fallback cannot stand in for their correctness. Remove the original forward capture for every candidate. For page-local candidates, restore pages in random order without other page prerequisites. For chain-dependent baselines/hybrids, allow predecessor decoding but count its full work, dependency depth and workspace. Shared immutable decoder state is allowed; hidden cross-page entropy contexts or retained KV are dependencies that must be counted.
3. **Serving:** enforce one aggregate GPU budget covering model, state, compressed residency, materialized tiles and scratch across requests. Compare admission, queue delay, p99 latency and token throughput against uncompressed and exact-codec restore baselines. Count repeated transfers/reconstruction at every decode step.

The immediate implementation goal remains higher savings with lower restore cost. Our current evidence does not justify calling any broad approach uniquely ours. It does justify continuing the controlled experiments and narrowing the claimed contribution to whatever mechanism actually wins.

The research priority is therefore to keep exponent/sign coding as a strong practical baseline, keep the shared probability model as the measured learned baseline, and test precise producer-side designs against both. The absence of an established uniqueness claim must not be read as evidence that the specific producer-metadata-assisted coset combination is already published. The comparison above separates partial precedents, the tested binary-syndrome failure and remaining design questions.

## Additional boundaries from the focused literature screen

Sequential Wyner–Ziv work models query-side information and approximate next-token-distribution distortion under specific assumptions [22]. It is related information theory, rather than evidence that our modular correction scheme achieves its bounds or invents decoder-side information coding.

GaugeKV is another related exactness claim [23]. Its indexed manuscript discusses gauge-based attention reparameterization and FP32 validation, with BF16 precision caveats. Function-preserving transformed KV is a different target from recovering a fixed model's original BF16 words. The full OpenReview page/PDF was blocked by browser verification in this check; only the indexed primary-manuscript excerpts were available, so its complete implementation and guarantees were not audited here. We make no priority claim against it.

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
20. [Variational image compression with a scale hyperprior](https://arxiv.org/html/1802.01436v2), ICLR 2018, §1/§3: learned transmitted side information that conditions an entropy model; joint side/payload rate objective.
21. [Understanding Rate-Distortion Performance in Distributed Transformer Inference](https://arxiv.org/html/2601.22002v5), 2026 preprint, §I/II: learned hyperprior coding of intermediate representations for lossy split inference, rather than original-word exact KV caching.
22. [Polynomial Context-Truncation Sensitivity in Autoregressive Language Models: Sequential Wyner–Ziv Bounds for KV Cache Compression](https://arxiv.org/html/2605.25085v1), 2026 preprint: query-side-information and next-token-distortion formulation with explicit assumptions.
23. [GaugeKV: Composable Exact KV Cache Compression](https://openreview.net/forum?id=rSxYPLzyBu), public manuscript; indexed [PDF](https://openreview.net/pdf?id=rSxYPLzyBu), Appendix J: FP32 validation and BF16 precision qualification. Full primary-document access was unavailable during this screen.
24. [Practical Full Resolution Learned Lossless Image Compression (L3C)](https://arxiv.org/abs/1811.12817), CVPR 2019; [author implementation](https://github.com/fab-jul/L3C-PyTorch): learned auxiliary representations, conditional distributions and exact arithmetic coding. This is a cross-domain precedent, rather than a KV serving system.
25. [The Residual Stream Is All You Need / KV-Direct](https://arxiv.org/html/2603.19664v1), March 2026 preprint, §3.2/§3.4: residual checkpointing and model-based KV recreation. Broader memory/performance claims were not independently reproduced in this review.
26. [NestQuant: Nested Lattice Quantization for Matrix Products and LLMs](https://arxiv.org/html/2502.09720v3), 2025 preprint, §4.1/Algorithms 1–2: nested-lattice coordinate residues modulo q and minimum-energy coset representatives, applied to lossy weight, activation and KV-cache quantization. This is a direct KV/coset ingredient precedent, not the original-BF16 actual-prefill-metadata-assisted decoder specified above.
