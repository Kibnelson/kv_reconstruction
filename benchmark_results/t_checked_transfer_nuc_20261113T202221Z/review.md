# Review: checked local transfer with reusable pinned buffers

Reviewed the archive uploaded in commit `83766a76ac4bf24ef782ebb9bf5463a1b04368ff`. The declared full local-path comparison passes. This is an independent check of the recorded GPU evidence and arithmetic; no GPU kernels were rerun during review.

## Evidence checked

- NVIDIA RTX A4500, Torch `2.11.0+cu128`, Triton `3.6.0`; five methods, two packet paths, four frozen WikiText-2 windows, three rounds, one warmup and three samples per timed scenario. Each window contains 896 MiB of raw BF16 KV, shape `[28, 8192, 2048]`, from `unsloth/Llama-3.2-3B`.
- All 120 method/path/window/round records and 720 timed path samples report exact full-word reconstruction. Fifteen timing workers completed with no errors. Timed sample stage sums and all published aggregates were recomputed from the records.
- Forty separate receiver proofs, covering original and pinned receivers for every method/window pair, match the recorded packet and reconstructed-word digests. Their jobs contain packet and shared-state inputs, with no capture path. Receive buffers were poisoned before initial validation.
- All 48 T records match the earlier frozen packet byte counts and SHA-256 values. Every row reports equality between the new packet and the original serializer applied to the same encoded result. Non-T packet sizes match the full native baseline; this review does not extend the frozen-T SHA claim to the native methods.
- The native reference results hash, all 13 required test input fingerprints and all five shared-state fingerprints match the earlier run. The earlier native run also lists three calibration input files that this no-refit runner does not reread. All eight recorded implementation hashes match the reviewed files.
- Verified ZIP CRC, the archive's Git blob hash and all 167 members against their separately uploaded Git blobs. Recomputed all five improvement gates. The archive and uploaded files are preserved; `review_validation.json` records this review.

## Measured complete paths

Times are direct wall measurements with serial CUDA synchronization, aggregated as median of samples, then median across rounds per window, then mean across windows. Fresh starts with GPU KV resident and times encode, checked host packet preparation, GPU restoration and decode. Retrieval starts with the compressed host packet already prepared and times restoration and decode. Both retain packet SHA-256 checking inside the timed path.

| Method, pinned path | BF16 saved | Packet MiB/window | Fresh local path ms | Stored-packet retrieval ms | Fresh reduction versus own original path |
|---|---:|---:|---:|---:|---:|
| Raw BF16 | approximately 0% | 896.000 | 1045.558 | 526.645 | 69.13% |
| Fast T | 44.0484% | 501.327 | 684.781 | 332.202 | 60.20% |
| Compact T | 48.5682% | 460.829 | 863.722 | 485.540 | 52.48% |
| SplitZip | 24.6863% | 674.811 | 795.364 | 397.489 | 72.00% |
| ExANS | 32.0454% | 608.873 | 931.101 | 379.122 | 66.25% |

Fast T has 13.90% lower fresh-path time and 16.42% lower retrieval time than SplitZip in this checked host-staged setup. It wins on every test window: fresh reductions range from 13.70% to 14.37%, and retrieval reductions from 15.43% to 17.00%. This is a consistent local result across these windows, not a network or serving performance claim.

Compact T saves another 40.50 MiB/window, or 4.52 percentage points of raw BF16 space, relative to Fast T. Its fresh path takes 26.13% longer. Preserve it as the compression candidate while using Fast T for the next latency integration.

## Remaining costs

| Method, pinned path | Encode ms | Packet preparation and D2H ms | Restoration and H2D ms | Decode ms |
|---|---:|---:|---:|---:|
| Raw BF16 | 0.005 | 519.728 | 522.546 | 3.322 |
| Fast T | 60.573 | 294.021 | 295.020 | 35.237 |
| Compact T | 108.705 | 270.224 | 271.149 | 213.601 |
| SplitZip | 6.602 | 391.758 | 393.849 | 3.095 |
| ExANS | 195.161 | 357.686 | 359.971 | 18.269 |

Each stage is aggregated separately, so their medians need not sum exactly to the separately aggregated total. Fast T's preparation/restoration buckets account for approximately 589 ms, or 86% of its 685 ms fresh path. These buckets combine bookkeeping, full CPU packet checksum work, device copies and T stream compaction/expansion. The run does not isolate their individual costs. A diagnostic split between CPU checksum, packing and actual copy time is needed before attributing that remaining cost to PCIe or choosing the next optimization. Inference: fewer bytes passing through these stages help explain Fast T's local advantage despite its slower codec than SplitZip.

The Compact T decoder remains a separate bottleneck at approximately 214 ms in this serial wall-time harness. These wall stages and the earlier CUDA-event codec measurements have different timing boundaries; cross-run differences are not evidence of a decoder regression.

## Memory and limits for P/D integration

The serial harness preallocates about 502.49 MiB of pinned host storage and 2797.47 MiB of additional device transfer buffers for Fast T. The latter includes both send and receive support, including a receive arena; it is not a per-worker requirement for a future role-specific connector. Original sender arena, output, decoder workspace, shared predictor state and resident input are additional. These reusable setup allocations are outside warm timings. One-GPU multi-engine feasibility must be checked with actual free VRAM and shorter requests before adding the T connector.

This run measures one-GPU host-buffer handoff with resident shared state and reusable buffers. It includes no socket/NIC, SSD read, overlap, model prefill, generated response token, shared-state load or P1 conversion. The link tables remain ideal serial models, not measured network throughput. There is no cross-dataset, cross-model, short-request or concurrent-serving confirmation.

Next, inspect the RTX machine and existing isolated vLLM environment, then validate raw P/D handoff before porting the frozen Fast T path. The existing `pd-vllm-protocol` launcher is written for one prefill and multiple decode engines; two prefill engines need routing and launcher changes. The old connector does not yet use this Fast T/pinned implementation. Its setup uses a separate vLLM environment. Its runner commits results even with `NO_PUSH=1`, so it must be adapted for the isolated review workflow before use. Confirm real KV installation and fail-on-transfer-error behavior rather than accepting generated text alone as proof of a handoff. A shared-GPU smoke checks functionality; clean 1P+2D and 2P+2D performance tests require three and four independently assigned GPUs respectively.

All changes stay on `experiment/t-head-aware-spread-v1`. `main`, the original T branch and the separate P/D branch are unchanged. All five gates nominate improvements for review; no automatic merge is authorized or performed.
