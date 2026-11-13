# T: head-aware probability-width experiment

GPU measured: **True**. Full model: **True**. No existing T branch, source or run artifact was modified.

| Predictor | Variant | Complete bpv | Bytes saved vs original | Decode ratio | Extra policy bytes |
|---|---|---:|---:|---:|---:|
| dense | head_selected | 8.952261 | 0 | 0.998 | 0 |
| dense | head_shifted_control | 8.952261 | 0 | 1.001 | 0 |
| dense | learned_selected | 8.947437 | 1132960 | 1.000 | 30464 |
| dense | learned_shifted_control | 8.950672 | 373056 | 1.004 | 30464 |
| dense | legacy_global | 8.952261 | 0 | 1.000 | 0 |
| dense | matched_global | 8.952261 | 0 | 0.996 | 0 |
| hbh | head_selected | 8.232200 | 0 | 1.002 | 0 |
| hbh | head_shifted_control | 8.232200 | 0 | 1.004 | 0 |
| hbh | learned_selected | 8.229093 | 729744 | 1.009 | 30464 |
| hbh | learned_shifted_control | 8.230725 | 346336 | 1.010 | 30464 |
| hbh | legacy_global | 8.232200 | 0 | 1.000 | 0 |
| hbh | matched_global | 8.232200 | 0 | 1.002 | 0 |

Only frozen DEV-selected head_selected/learned_selected are candidates. Shifted-head rows are negative controls. Original and matched globals must have identical packets. Shared predictor state, policy state, scratch and output are separate from packet bpv. No token-specific hint is sent.

An actual GPU, exact packet-only reconstruction, a positive DEV and TEST byte gain, and the declared decode-time budget are required for a candidate to pass the experiment gate. Passing is evidence for review, not permission to merge. Cross-dataset confirmation remains necessary.

dense: `{"head": {"automatic_merge": false, "candidate_for_review": false, "positive_dev_gain": false, "positive_test_packet_gain": false, "saves_bytes_after_one_extra_shared_state": false, "within_decode_budget": true}, "learned": {"automatic_merge": false, "candidate_for_review": true, "positive_dev_gain": true, "positive_test_packet_gain": true, "saves_bytes_after_one_extra_shared_state": true, "within_decode_budget": true}}`
hbh: `{"head": {"automatic_merge": false, "candidate_for_review": false, "positive_dev_gain": false, "positive_test_packet_gain": false, "saves_bytes_after_one_extra_shared_state": false, "within_decode_budget": true}, "learned": {"automatic_merge": false, "candidate_for_review": true, "positive_dev_gain": true, "positive_test_packet_gain": true, "saves_bytes_after_one_extra_shared_state": true, "within_decode_budget": true}}`
