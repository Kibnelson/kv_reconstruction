# Prefill metadata and exact vector-coset study

Status: **completed**

Exact original-BF16 independent pages using TRAIN-fitted metadata-conditioned bit predictors, selected mantissa residues and verified 31-bit vector-coset/sparse-position corrections. All measured packet and full state bytes count. CPU reference coding, selected-device neural work, sampled full-cache estimates; no compact attention, speedup, serving-capacity or novelty claim.

| Case | Tokens | Method | Residue bits | Sample packet savings | Net full-cache estimate |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 2048 | frozen_exponent_control (ok) | — | 29.36% | 29.22% |
| test_pottery_kiln_diary | 2048 | field_baseline (ok) | 7 | 30.52% | 30.50% |
| test_pottery_kiln_diary | 2048 | coset_none_syndrome (ok) | 6 | 30.51% | 29.58% |
| test_pottery_kiln_diary | 2048 | coset_none_positions (ok) | 6 | 30.51% | 29.58% |
| test_pottery_kiln_diary | 2048 | coset_kv_syndrome (ok) | 6 | 30.12% | 28.97% |
| test_pottery_kiln_diary | 2048 | coset_kv_positions (ok) | 6 | 30.12% | 28.97% |
| test_pottery_kiln_diary | 2048 | coset_kv_canonical_syndrome (ok) | 6 | 30.11% | 28.97% |
| test_pottery_kiln_diary | 2048 | coset_kv_canonical_positions (ok) | 6 | 30.11% | 28.97% |
| test_pottery_kiln_diary | 2048 | coset_hidden_syndrome (ok) | 6 | 30.12% | 28.91% |
| test_pottery_kiln_diary | 2048 | coset_hidden_positions (ok) | 6 | 30.12% | 28.91% |
| test_pottery_kiln_diary | 2048 | coset_hidden_norm_syndrome (ok) | 6 | 30.11% | 28.91% |
| test_pottery_kiln_diary | 2048 | coset_hidden_norm_positions (ok) | 6 | 30.11% | 28.91% |

Residue widths are selected on complete DEV packet bytes and frozen before TEST. Candidates keep their side code even if every correction group escapes; no learned whole-page fallback hides the tested path.

The syndrome and sparse-position representations are matched controls, not stackable savings. Actual packet components, bitplane/K/V error-group histograms, code-use ablations, layer/position breakdowns and model/packet identities are in results.json and decision_summary.json.

Packet coding runs on CPU. Code extraction and bit prediction run on the selected device, with diagnostic staging. Instrumented prefill includes all encoders and retains full raw KV; no isolated prefill overhead, fused GPU decoder, SSD transfer or concurrent-serving benefit is measured.

Full DEV packet-only continuation checks:

- coset_none_syndrome: ok
- coset_kv_syndrome: ok
- coset_kv_canonical_syndrome: ok
- coset_hidden_syndrome: ok
- coset_hidden_norm_syndrome: ok
