# Prefill metadata and exact vector-coset study

Status: **completed**

Exact original-BF16 independent pages using TRAIN-fitted metadata-conditioned bit predictors, selected mantissa residues and verified 31-bit vector-coset/sparse-position corrections. All measured packet and full state bytes count. CPU reference coding, selected-device neural work, sampled full-cache estimates; no compact attention, speedup, serving-capacity or novelty claim.

| Case | Tokens | Method | Residue bits | Sample packet savings | Net full-cache estimate |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 64 | frozen_exponent_control (ok) | — | 4.40% | -94.10% |
| test_pottery_kiln_diary | 64 | field_baseline (ok) | 7 | 10.30% | 8.56% |
| test_pottery_kiln_diary | 64 | coset_none_syndrome (ok) | 6 | 9.47% | -174.41% |
| test_pottery_kiln_diary | 64 | coset_none_positions (ok) | 6 | 9.47% | -174.41% |
| test_pottery_kiln_diary | 64 | coset_kv_syndrome (ok) | 6 | 6.34% | -188.57% |
| test_pottery_kiln_diary | 64 | coset_kv_positions (ok) | 6 | 6.34% | -188.57% |
| test_pottery_kiln_diary | 64 | coset_kv_canonical_syndrome (ok) | 6 | 6.10% | -189.30% |
| test_pottery_kiln_diary | 64 | coset_kv_canonical_positions (ok) | 6 | 6.10% | -189.30% |
| test_pottery_kiln_diary | 64 | coset_hidden_syndrome (ok) | 6 | 6.24% | -188.90% |
| test_pottery_kiln_diary | 64 | coset_hidden_positions (ok) | 6 | 6.24% | -188.90% |
| test_pottery_kiln_diary | 64 | coset_hidden_norm_syndrome (ok) | 6 | 6.12% | -190.13% |
| test_pottery_kiln_diary | 64 | coset_hidden_norm_positions (ok) | 6 | 6.12% | -190.13% |

Residue widths are selected on complete DEV packet bytes and frozen before TEST. Candidates keep their side code even if every correction group escapes; no learned whole-page fallback hides the tested path.

The syndrome and sparse-position representations are matched controls, not stackable savings. Actual packet components, bitplane/K/V error-group histograms, code-use ablations, layer/position breakdowns and model/packet identities are in results.json and decision_summary.json.

Packet coding runs on CPU. Code extraction and bit prediction run on the selected device, with diagnostic staging. Instrumented prefill includes all encoders and retains full raw KV; no isolated prefill overhead, fused GPU decoder, SSD transfer or concurrent-serving benefit is measured.

Full DEV packet-only continuation checks:

- coset_none_syndrome: ok
- coset_kv_syndrome: ok
- coset_kv_canonical_syndrome: ok
- coset_hidden_syndrome: ok
- coset_hidden_norm_syndrome: ok
