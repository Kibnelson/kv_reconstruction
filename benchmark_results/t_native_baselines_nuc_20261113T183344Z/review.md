# Review: frozen T versus native GPU baselines

Reviewed the full run uploaded in commit `536d28c81417917ad46f007dea2169acfdad7a43`. This review checks the archived evidence; it does not rerun GPU kernels or change the codec. The declared warm native codec comparison passes. It is not a measured end-to-end network transfer.

## Evidence checked

- NVIDIA RTX A4500, Torch `2.11.0+cu128`, Triton `3.6.0`; five methods, four frozen WikiText2 windows, three rounds, three warmups and ten samples per timed operation.
- Sixty successful method/window/round records, 1,800 CUDA timing samples, and 20 fresh packet-only receiver proofs. Every proof's packet and reconstructed-word digest matches its measurement record. Receiver jobs contain no capture path.
- All 24 T records retain the earlier complete packet byte size and SHA-256: dense/global for `t_fast`, and head-by-head/learned for `t_hbh`. The frozen reference results, source words and original T source hashes match. Every recorded comparison implementation hash matches the reviewed source.
- All external packet component sizes plus headers add up to the paid packet total. All ExANS reference windows have 940 fitted native records and a paid 65,536-byte raw tail. Its native random-input preflight exercises two rejected blocks with complete raw fallback and a partial tail. The compressible preflight exercises fitted decoding; the authors' executable test passed.
- Recomputed every published aggregate from the sample records. Sizes are stable across rounds; `full_comparison_validated` is true and `errors` is empty. These facts apply to the declared five-method comparison and do not authorize a merge.
- Verified the archive's Git blob hash and ZIP CRC. `review_validation.json` records the reviewed input hashes and recomputed metrics. The original uploaded files and archive are preserved.

## Compression and warm codec latency

Each window contains 896 MiB of raw BF16 KV for 28 layers and 8,192 tokens. Time is the measured sequential pair, using median across rounds per window followed by mean across windows.

| Method | BF16 saved | Packet MiB/window | Encode CUDA ms | Decode CUDA ms | Pair wall ms |
|---|---:|---:|---:|---:|---:|
| Raw BF16 | approximately 0% | 896.000 | 0.006 | 3.306 | 3.317 |
| Fast T | 44.0484% | 501.327 | 61.903 | 35.915 | 98.360 |
| Compact T | 48.5682% | 460.829 | 110.559 | 194.164 | 312.118 |
| SplitZip | 24.6863% | 674.811 | 6.695 | 3.046 | 9.686 |
| ExANS | 32.0454% | 608.873 | 172.832 | 17.828 | 190.715 |

Fast T's packets are 25.7% smaller than SplitZip's, with a 10.15x longer warm codec pair. Fast T also has 17.7% smaller packets and 48.4% lower pair latency than ExANS in this run, while ExANS decodes about twice as fast. These are statements about these native APIs, allocation policies and full-window batch geometry.

Compact T saves another 40.50 MiB per window compared with Fast T, but adds 213.76 ms to the pair. It is 3.17x slower. Its 48.5682% saving is validated, but its decoder is still the principal codec bottleneck. Most of the compression difference comes from the original grouped predictor: the earlier frozen HBH/global rate was 8.232200 bpv (48.5488% saved), while the learned-width policy reaches 8.229093 bpv (48.5682%). The learned-width addition itself is approximately 0.0194 percentage points.

## Host packet costs are substantial

The headline pair times exclude packet serialization, checksums, host copies, staging and arena expansion. The recorded host packet path is much slower than the warm codec call. The following values use the same median-across-rounds and mean-across-windows aggregation; each host path was sampled once per window per round rather than ten times like the codec operation.

| Method | Packing and D2H ms | Expansion and H2D ms | Warm codec pair ms |
|---|---:|---:|---:|
| Raw BF16 | 1824.42 | 1543.21 | 3.32 |
| Fast T | 1039.56 | 604.93 | 98.36 |
| Compact T | 943.24 | 527.36 | 312.12 |
| SplitZip | 1626.29 | 1191.18 | 9.69 |
| ExANS | 1456.71 | 1054.19 | 190.71 |

For Fast T, these separately measured host stages alone total about 1.64 seconds per 896 MiB window. They include the checked Python packet path and device copies; they are not intrinsic entropy-coder time, isolated PCIe throughput, or a measurement of a production transport. Different native formats take different serialization paths, so this table also does not establish a transport-independent winner. Capture-file-to-GPU staging costs roughly another second, but is fixture setup rather than a cost to charge when prefill already leaves KV resident on GPU.

The earlier link table deliberately excludes these host stages. Its approximately 16.41 Gbps Fast T/SplitZip crossover is conditional on the warm codec-only serial model, not a measured network threshold. With that model, Compact T only overtakes Fast T below approximately 1.59 Gbps; it is not the fastest alternative anywhere in the requested 3–400 Gbps range. Shared state is assumed resident and no overlap is modeled.

## Memory and scope

Fast T has 117.31 MiB of reported shared predictor/policy tensors; Compact T has 229.34 MiB. SplitZip's four shared lookup tables occupy 784 bytes. Both T methods use a preallocated 1,792.52 MiB worst-case stream arena, plus an 896 MiB reconstruction output and decoder scratch. Those temporary arenas are not the compact transferable payload or the P1 resident pool format. ExANS' native allocations are outside Torch's peak-memory counters; setup device-free-memory delta is approximately 4.96–4.98 GiB including native workspace and allocated buffers. Its near-zero Torch incremental decode peak is not a total-memory claim.

This run covers one model (`unsloth/Llama-3.2-3B`), one GPU, one dataset and one large request size. It has no C4/cross-model confirmation, short-request sweep, cold shared-state transfer, actual network/storage transport, P1 conversion, serving throughput or lossy AQUA-KV quality measurement. Native allocation differences remain explicit in `results.json`.

## Recommended next step

Keep Fast T as the measured T speed baseline and preserve Compact T as the compression candidate. Before claiming end-to-end transfer gains or spending the next optimization round on compression, measure and reduce the actual packet path with reusable buffers and fewer host copies while preserving the frozen packet bytes and fresh-receiver checks. Retain a separate checked serialization path for correctness verification, and report real compaction/D2H/H2D/expansion costs for every method. The HBH decoder remains a separate optimization target. AQUA-KV still needs its planned independent lossy quality comparison.

All results remain on `experiment/t-head-aware-spread-v1`; original T and `main` are unchanged. No merge is proposed by this review.
