# Prefill metadata and exact vector-coset study

Status: **completed**

Exact original-BF16 independent pages using TRAIN-fitted metadata-conditioned bit predictors, selected mantissa residues and verified 31-bit vector-coset/sparse-position corrections. All measured packet and full state bytes count. CPU reference coding, selected-device neural work, sampled full-cache estimates; no compact attention, speedup, serving-capacity or novelty claim.

| Case | Tokens | Method | Residue bits | Sample packet savings | Net full-cache estimate |
|---|---:|---|---:|---:|---:|
| test_training_holdout_canal_gates | 2048 | frozen_exponent_control (ok) | — | 29.35% | 29.20% |
| test_training_holdout_canal_gates | 2048 | field_baseline (ok) | 7 | 30.48% | 30.46% |
| test_training_holdout_canal_gates | 2048 | coset_none_syndrome (ok) | 6 | 30.47% | 29.54% |
| test_training_holdout_canal_gates | 2048 | coset_none_positions (ok) | 6 | 30.47% | 29.54% |
| test_training_holdout_canal_gates | 2048 | coset_kv_syndrome (ok) | 6 | 30.08% | 28.93% |
| test_training_holdout_canal_gates | 2048 | coset_kv_positions (ok) | 6 | 30.08% | 28.93% |
| test_training_holdout_canal_gates | 2048 | coset_hidden_syndrome (ok) | 6 | 30.08% | 28.87% |
| test_training_holdout_canal_gates | 2048 | coset_hidden_positions (ok) | 6 | 30.08% | 28.87% |
| test_training_holdout_interval_merge | 2048 | frozen_exponent_control (ok) | — | 29.38% | 29.23% |
| test_training_holdout_interval_merge | 2048 | field_baseline (ok) | 7 | 30.48% | 30.45% |
| test_training_holdout_interval_merge | 2048 | coset_none_syndrome (ok) | 6 | 30.46% | 29.53% |
| test_training_holdout_interval_merge | 2048 | coset_none_positions (ok) | 6 | 30.46% | 29.53% |
| test_training_holdout_interval_merge | 2048 | coset_kv_syndrome (ok) | 6 | 30.07% | 28.92% |
| test_training_holdout_interval_merge | 2048 | coset_kv_positions (ok) | 6 | 30.07% | 28.92% |
| test_training_holdout_interval_merge | 2048 | coset_hidden_syndrome (ok) | 6 | 30.07% | 28.87% |
| test_training_holdout_interval_merge | 2048 | coset_hidden_positions (ok) | 6 | 30.07% | 28.87% |
| test_training_holdout_canal_gates | 8192 | frozen_exponent_control (ok) | — | 29.27% | 29.23% |
| test_training_holdout_canal_gates | 8192 | field_baseline (ok) | 7 | 30.51% | 30.50% |
| test_training_holdout_canal_gates | 8192 | coset_none_syndrome (ok) | 6 | 30.49% | 30.26% |
| test_training_holdout_canal_gates | 8192 | coset_none_positions (ok) | 6 | 30.49% | 30.26% |
| test_training_holdout_canal_gates | 8192 | coset_kv_syndrome (ok) | 6 | 30.10% | 29.81% |
| test_training_holdout_canal_gates | 8192 | coset_kv_positions (ok) | 6 | 30.10% | 29.81% |
| test_training_holdout_canal_gates | 8192 | coset_hidden_syndrome (ok) | 6 | 30.10% | 29.80% |
| test_training_holdout_canal_gates | 8192 | coset_hidden_positions (ok) | 6 | 30.10% | 29.80% |
| test_training_holdout_interval_merge | 8192 | frozen_exponent_control (ok) | — | 29.32% | 29.28% |
| test_training_holdout_interval_merge | 8192 | field_baseline (ok) | 7 | 30.50% | 30.49% |
| test_training_holdout_interval_merge | 8192 | coset_none_syndrome (ok) | 6 | 30.48% | 30.25% |
| test_training_holdout_interval_merge | 8192 | coset_none_positions (ok) | 6 | 30.48% | 30.25% |
| test_training_holdout_interval_merge | 8192 | coset_kv_syndrome (ok) | 6 | 30.09% | 29.80% |
| test_training_holdout_interval_merge | 8192 | coset_kv_positions (ok) | 6 | 30.09% | 29.80% |
| test_training_holdout_interval_merge | 8192 | coset_hidden_syndrome (ok) | 6 | 30.09% | 29.79% |
| test_training_holdout_interval_merge | 8192 | coset_hidden_positions (ok) | 6 | 30.09% | 29.79% |

Residue widths are selected on complete DEV packet bytes and frozen before TEST. Candidates keep their side code even if every correction group escapes; no learned whole-page fallback hides the tested path.

The syndrome and sparse-position representations are matched controls, not stackable savings. Actual packet components, bitplane/K/V error-group histograms, code-use ablations, layer/position breakdowns and model/packet identities are in results.json and decision_summary.json.

Packet coding runs on CPU. Code extraction and bit prediction run on the selected device, with diagnostic staging. Instrumented prefill includes all encoders and retains full raw KV; no isolated prefill overhead, fused GPU decoder, SSD transfer or concurrent-serving benefit is measured.

Full DEV packet-only continuation checks:

- coset_none_syndrome: ok
- coset_kv_syndrome: ok
- coset_hidden_syndrome: ok
