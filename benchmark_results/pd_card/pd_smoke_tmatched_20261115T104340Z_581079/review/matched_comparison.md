# Matched PD comparison

Declared coverage passed: False. Full repeated comparison validated: False.

Times below are medians of the available per-repeat metrics. Median TTFT p50 means the median of per-repeat p50 values, not a pooled request percentile. Different engine groups remain separate. Smoke is a functional pilot.

Coverage issues:
- Compiled GPU / independent receiver readiness evidence is missing or failed.
- Same-allocation hardware and bandwidth survey is missing or incomplete.
- Matrix did not execute the exact run list frozen at create.
- One or more setup or matrix runs failed.
- Expected exactly one results folder for SMOKE=1 LAYOUT=1p3d TRANSPORT=ring CODECS=raw,splitzip,exans,adacode,exactkv,ours-hbh,ours-dense,t2-hbh,t-fast,t-dense-cached,t-compact,t-compact-global MATCHED_ROUND=0
- Expected exactly one results folder for SMOKE=1 LAYOUT=2p2d TRANSPORT=ring CODECS=raw,splitzip,exans,ours-hbh,ours-dense,t-fast,t-dense-cached,t-compact,t-compact-global MATCHED_ROUND=0
- Expected exactly one results folder for SMOKE=1 LAYOUT=2p2d TRANSPORT=p2p CODECS=raw,ours-hbh,t2-hbh,t-fast,t-dense-cached,t-compact,t-compact-global MATCHED_ROUND=0
- Expected exactly one results folder for SMOKE=1 LAYOUT=colo4 TIER_CODECS=raw,exans,ibp,ours-hbh,t2-hbh,t-fast,t-dense-cached,t-compact,t-compact-global MATCHED_ROUND=0
- Expected exactly one results folder for SMOKE=1 LAYOUT=1p3d TRANSPORT=ring ENGINE_EAGER=1 CODECS=raw,t2-hbh,t2-rank256,t-fast,t-dense-cached,t-compact,t-compact-global MATCHED_ROUND=0

Ring bandwidth is paced per prefill link; 2P has two such links. This is not a measured NIC experiment. Codec-frame bytes include alignment and JSON metadata; transport control handshakes remain separate. Native NIXL has no custom codec-frame/checksum counters.
