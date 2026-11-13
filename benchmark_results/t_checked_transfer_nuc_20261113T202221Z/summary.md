# Checked local transfer path: original versus pinned buffers

Status: **completed**. GPU executed: **True**. Full reference coverage: **True**.

| Method / path | BF16 saved | Fresh local path ms | Stored-packet retrieval ms | Complete |
|---|---:|---:|---:|---|
| raw_bf16 / legacy_checked | -0.0000% | 3386.681 | 1554.130 | True |
| raw_bf16 / pinned_checked | -0.0000% | 1045.558 | 526.645 | True |
| t_fast / legacy_checked | 44.0484% | 1720.605 | 618.139 | True |
| t_fast / pinned_checked | 44.0484% | 684.781 | 332.202 | True |
| t_hbh / legacy_checked | 48.5682% | 1817.595 | 749.324 | True |
| t_hbh / pinned_checked | 48.5682% | 863.722 | 485.540 | True |
| splitzip / legacy_checked | 24.6863% | 2840.126 | 1187.949 | True |
| splitzip / pinned_checked | 24.6863% | 795.364 | 397.489 | True |
| exans / legacy_checked | 32.0454% | 2758.898 | 1081.593 | True |
| exans / pinned_checked | 32.0454% | 931.101 | 379.122 | True |

Fresh: resident GPU KV -> encode -> compact checked host packet -> restore on GPU -> decode. Stored-packet retrieval starts with that compressed host packet already prepared and times restore+decode. Both include full packet SHA-256 verification. CUDA is synchronized at serial stage boundaries. All byte/word parity checks are outside timers; the protocol checksum stays inside.

This is a one-GPU host-buffer handoff. It measures no NIC/socket/network or SSD I/O, no stage overlap, no shared-state loading and no P1 conversion. Network time is only modeled separately in results.json. Capture-file input staging and buffer setup are separate fields. Pinned input must remain owned until receive completes.

Every method uses the same new serializer treatment; native encoder/decoder allocations and algorithm remain unchanged. Canonical packets must retain the earlier byte format; T packets additionally match the frozen reference SHA-256. Reused receive buffers are poisoned before first validation, and independent original/new receivers decode the pinned packets with no capture input.

{"automatic_merge": false, "exans": {"candidate_for_review": true, "criterion": "GPU, all reference windows, >=3 rounds, unchanged bytes, exact decode, >=5% complete local-path gain", "fresh_gain_fraction": 0.6625096115724782}, "network_measured": false, "raw_bf16": {"candidate_for_review": true, "criterion": "GPU, all reference windows, >=3 rounds, unchanged bytes, exact decode, >=5% complete local-path gain", "fresh_gain_fraction": 0.6912734704622305}, "splitzip": {"candidate_for_review": true, "criterion": "GPU, all reference windows, >=3 rounds, unchanged bytes, exact decode, >=5% complete local-path gain", "fresh_gain_fraction": 0.7199548122403125}, "t_fast": {"candidate_for_review": true, "criterion": "GPU, all reference windows, >=3 rounds, unchanged bytes, exact decode, >=5% complete local-path gain", "fresh_gain_fraction": 0.6020115243127551}, "t_hbh": {"candidate_for_review": true, "criterion": "GPU, all reference windows, >=3 rounds, unchanged bytes, exact decode, >=5% complete local-path gain", "fresh_gain_fraction": 0.5247997231242336}}

No automatic merge, calibration or refit.
