# T speed comparison with frozen compressed bytes

Status: **completed_with_graph_errors**. GPU executed: **True**. Full reference coverage: **True**.

| Implementation / execution | BF16 saved | bpv | Encode ms | Decode ms | Pair ms | Complete |
|---|---:|---:|---:|---:|---:|---|
| dense_fast/eager | 44.0484% | 8.952261 | 63.657 | 36.758 | 100.686 | True |
| dense_fast/graph | 44.0484% | 8.952261 | 63.423 | 36.633 | 100.433 | True |
| dense_learned/eager | 44.0785% | 8.947437 | 63.830 | 36.730 | 100.888 | True |
| dense_learned/graph | 44.0785% | 8.947437 | 63.757 | 36.721 | 100.631 | True |
| dense_original/eager | 44.0484% | 8.952261 | 87.475 | 36.827 | 124.493 | True |
| hbh_learned/eager | 48.5682% | 8.229093 | 112.581 | 195.575 | 315.898 | True |
| hbh_learned/graph | 48.5682% | 8.229093 | 112.426 | 193.348 | 313.333 | True |
| hbh_original/eager | 48.5488% | 8.232200 | 112.316 | 194.700 | 314.398 | True |
| hbh_original/graph | 48.5488% | 8.232200 | 111.941 | 191.784 | 311.028 | True |

Times are CUDA event medians within a round, then median across rounds per case, then mean across cases. Pair is measured sequential encode + decode on one GPU, with resident input and preallocated stream arena/output. Graph pair uses two replay launches. Setup, graph memory, input staging, packet packing/D2H, packet expansion/H2D and cold worker time are separate fields.

Every packet must match the earlier run byte for byte. No predictor, spread, alpha or tile is refitted. The hbh_learned full run therefore preserves the earlier 48.568% saving. A smaller case subset can have a different aggregate rate. No accuracy or symbol bits are discarded.

## Ideal serial link model

| Implementation / execution | 3 Gbps | 25 Gbps | 100 Gbps | 400 Gbps |
|---|---:|---:|---:|---:|
| dense_fast/eager | 1502.497 | 268.904 | 142.741 | 111.200 |
| dense_fast/graph | 1502.243 | 268.650 | 142.487 | 110.946 |
| dense_learned/eager | 1501.944 | 269.015 | 142.920 | 111.396 |
| dense_learned/graph | 1501.687 | 268.758 | 142.663 | 111.139 |
| dense_original/eager | 1526.304 | 292.710 | 166.547 | 135.006 |
| hbh_learned/eager | 1604.469 | 470.526 | 354.555 | 325.562 |
| hbh_learned/graph | 1601.904 | 467.961 | 351.990 | 322.997 |
| hbh_original/eager | 1603.456 | 469.085 | 353.069 | 324.065 |
| hbh_original/graph | 1600.086 | 465.715 | 349.700 | 320.696 |

Milliseconds = measured codec pair + complete packet bytes / offered link bandwidth. This is a model: network, SSD, queueing, contention, state loading and the separately recorded host packet path are not measured by this table. It assumes resident shared state and no overlap.

{"automatic_merge": false, "criterion": "actual GPU, all reference windows, >=3 rounds, identical packets, >=5% relevant speed gain", "fast_dense_candidate_for_review": true, "hbh_graph_candidate_for_review": false}

Graph capture/replay errors: 12. They are not relabeled as eager results.

