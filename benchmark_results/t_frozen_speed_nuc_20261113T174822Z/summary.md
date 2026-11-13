# T speed comparison with frozen compressed bytes

Status: **completed**. GPU executed: **True**. Full reference coverage: **True**.

| Implementation / execution | BF16 saved | bpv | Encode ms | Decode ms | Pair ms | Complete |
|---|---:|---:|---:|---:|---:|---|
| dense_cached/eager | 44.0484% | 8.952261 | 78.196 | 36.340 | 114.611 | True |
| dense_cached/graph | 44.0484% | 8.952261 | 78.581 | 36.360 | 114.684 | True |
| dense_fast/eager | 44.0484% | 8.952261 | 63.023 | 36.363 | 99.691 | True |
| dense_fast/graph | 44.0484% | 8.952261 | 62.907 | 36.389 | 99.337 | True |
| dense_original/eager | 44.0484% | 8.952261 | 86.356 | 36.352 | 122.623 | True |

Times are CUDA event medians within a round, then median across rounds per case, then mean across cases. Pair is measured sequential encode + decode on one GPU, with resident input and preallocated stream arena/output. Graph pair uses two replay launches. Setup, graph memory, input staging, packet packing/D2H, packet expansion/H2D and cold worker time are separate fields.

Every packet must match the earlier run byte for byte. No predictor, spread, alpha or tile is refitted. The hbh_learned full run therefore preserves the earlier 48.568% saving. A smaller case subset can have a different aggregate rate. No accuracy or symbol bits are discarded.

## Ideal serial link model

| Implementation / execution | 3 Gbps | 25 Gbps | 100 Gbps | 400 Gbps |
|---|---:|---:|---:|---:|
| dense_cached/eager | 1516.421 | 282.828 | 156.665 | 125.124 |
| dense_cached/graph | 1516.495 | 282.902 | 156.739 | 125.198 |
| dense_fast/eager | 1501.502 | 267.908 | 141.745 | 110.204 |
| dense_fast/graph | 1501.148 | 267.554 | 141.391 | 109.851 |
| dense_original/eager | 1524.434 | 290.840 | 164.677 | 133.137 |

Milliseconds = measured codec pair + complete packet bytes / offered link bandwidth. This is a model: network, SSD, queueing, contention, state loading and the separately recorded host packet path are not measured by this table. It assumes resident shared state and no overlap.

{"automatic_merge": false, "cached_dense_candidate_for_review": true, "cached_original_control_present": true, "criterion": "actual GPU, all reference windows, >=3 rounds, identical packets, >=5% relevant speed gain; fast dense is compared to cached original", "fast_dense_candidate_for_review": true, "fast_dense_vs_uncached_original_gain": true, "hbh_graph_candidate_for_review": false}

Original dense is intentionally eager-only: its per-call CPU table construction is not capturable. dense_cached preserves the original fused computation with that setup done once. Expected omissions are recorded separately from unexpected graph errors.

