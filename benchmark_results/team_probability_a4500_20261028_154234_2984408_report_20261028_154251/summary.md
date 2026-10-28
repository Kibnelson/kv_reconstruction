# Fixed-prior head-spread calibration versus GPU baselines

Status: **failed**. GPU measured: **True**.

Llama-3.2-3B, all 28 original BF16 layers. Only TABLES selects per-head spread factors from 0.5/0.75/1/1.25/1.5 using actual compact GPU packet words. Predictor tensors, inverse surprise scales and decoder kernel structure/launch configuration are unchanged. Data-dependent coding work may change. TEST never selects parameters.
Both variants use the one-warp reference binary-search decoder: graph for 512 tokens, direct for 8192. Native controls retain their original implementations and settings.

| Tokens | Method | Variant / round | Complete bpv | Packet MiB | Decode ms | Exact |
|---:|---|---|---:|---:|---:|---|

Calculated serial wire + measured full GPU decode, milliseconds. Networking is not measured. Ideal overlap remains a separate bound in results.json.

| Tokens | Method | Variant / round | 1 Gb/s | 100 Gb/s | 200 Gb/s | 300 Gb/s | 400 Gb/s |
|---:|---|---|---:|---:|---:|---:|---:|

Failure: exit 1; see /home/nelson/kv_reconstruction/runs/team_probability_a4500_20261028_154234_2984408/calibration.log

Primary decode starts with compressed payload on GPU and includes normal API validation/metadata work and an independently owned exact output. Bulk host hashing/import/upload, disk, real networking and calibration are excluded. Complete packet bytes include framing, directories, states, alignment and checksum. Fixed shared state and scratch are additional.
No fresh end-to-end native encode comparison, NIC/SSD, compressed attention, T-to-P conversion or serving result is claimed. Smaller TEST packets are a hypothesis, not guaranteed by TABLES selection.
