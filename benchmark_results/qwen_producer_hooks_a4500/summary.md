# Forward producer metadata study

Status: **completed**

Default callbacks build and verify recipes inside actual attention forward, using diagnostic CPU copies staged back to the selected device. Repeated codec timings separately replay captured features after prefill. All payload/state bytes count; full raw KV remains required. CPU tiny runs establish functionality only. No compact attention, concurrency or latency gain is inferred.

| Split | Case | Method | Status | Savings incl. state | Records | Decode ms/raw MiB |
|---|---|---|---|---:|---:|---:|
| dev | dev_orchard_sensor_log | producer_kv_median | ok | 29.078% measured | 288/288 | 7.587 |
| dev | dev_orchard_sensor_log | producer_kv_fixed127 | ok | 21.022% measured | 288/288 | 7.629 |
| dev | dev_orchard_sensor_log | producer_frozen_exponent | ok | 29.592% measured | 288/288 | 7.001 |
| dev | dev_orchard_sensor_log | producer_hidden_exponent | ok | 27.187% measured | 288/288 | 8.438 |
| dev | dev_orchard_sensor_log | producer_hidden_abs_exponent | ok | 27.141% measured | 288/288 | 8.163 |
| dev | dev_orchard_sensor_log | producer_pre_rope_residual | ok | 28.913% measured | 288/288 | 9.311 |
| dev | dev_orchard_sensor_log | producer_pre_rope_coset | ok | 28.915% measured | 288/288 | 9.393 |
| dev | dev_orchard_sensor_log | producer_kproj_norm_rope_residual | ok | 28.841% measured | 288/288 | 9.505 |
| dev | dev_orchard_sensor_log | producer_kproj_norm_rope_coset | ok | 28.844% measured | 288/288 | 9.720 |
| test | test_pottery_kiln_diary_2048 | producer_kv_median | ok | 29.304% estimated | 72/1152 | 7.581 |
| test | test_pottery_kiln_diary_2048 | producer_kv_fixed127 | ok | 21.403% estimated | 72/1152 | 7.441 |
| test | test_pottery_kiln_diary_2048 | producer_frozen_exponent | ok | 29.906% estimated | 72/1152 | 6.944 |
| test | test_pottery_kiln_diary_2048 | producer_hidden_exponent | ok | 28.665% estimated | 72/1152 | 8.718 |
| test | test_pottery_kiln_diary_2048 | producer_hidden_abs_exponent | ok | 28.608% estimated | 72/1152 | 8.273 |
| test | test_pottery_kiln_diary_2048 | producer_pre_rope_residual | ok | 29.126% estimated | 72/1152 | 9.203 |
| test | test_pottery_kiln_diary_2048 | producer_pre_rope_coset | ok | 29.128% estimated | 72/1152 | 9.375 |
| test | test_pottery_kiln_diary_2048 | producer_kproj_norm_rope_residual | ok | 29.091% estimated | 72/1152 | 9.407 |
| test | test_pottery_kiln_diary_2048 | producer_kproj_norm_rope_coset | ok | 29.093% estimated | 72/1152 | 9.988 |
| test | test_python_dependency_walk_2048 | producer_kv_median | ok | 29.308% estimated | 72/1152 | 7.587 |
| test | test_python_dependency_walk_2048 | producer_kv_fixed127 | ok | 21.201% estimated | 72/1152 | 7.521 |
| test | test_python_dependency_walk_2048 | producer_frozen_exponent | ok | 29.889% estimated | 72/1152 | 7.115 |
| test | test_python_dependency_walk_2048 | producer_hidden_exponent | ok | 28.722% estimated | 72/1152 | 8.398 |
| test | test_python_dependency_walk_2048 | producer_hidden_abs_exponent | ok | 28.659% estimated | 72/1152 | 8.532 |
| test | test_python_dependency_walk_2048 | producer_pre_rope_residual | ok | 29.129% estimated | 72/1152 | 9.317 |
| test | test_python_dependency_walk_2048 | producer_pre_rope_coset | ok | 29.131% estimated | 72/1152 | 9.413 |
| test | test_python_dependency_walk_2048 | producer_kproj_norm_rope_residual | ok | 29.095% estimated | 72/1152 | 9.842 |
| test | test_python_dependency_walk_2048 | producer_kproj_norm_rope_coset | ok | 29.097% estimated | 72/1152 | 9.857 |
| test | test_museum_loan_register_2048 | producer_kv_median | ok | 29.408% estimated | 72/1152 | 7.589 |
| test | test_museum_loan_register_2048 | producer_kv_fixed127 | ok | 21.259% estimated | 72/1152 | 7.521 |
| test | test_museum_loan_register_2048 | producer_frozen_exponent | ok | 29.916% estimated | 72/1152 | 7.003 |
| test | test_museum_loan_register_2048 | producer_hidden_exponent | ok | 28.803% estimated | 72/1152 | 8.358 |
| test | test_museum_loan_register_2048 | producer_hidden_abs_exponent | ok | 28.727% estimated | 72/1152 | 8.178 |
| test | test_museum_loan_register_2048 | producer_pre_rope_residual | ok | 29.233% estimated | 72/1152 | 9.229 |
| test | test_museum_loan_register_2048 | producer_pre_rope_coset | ok | 29.235% estimated | 72/1152 | 9.548 |
| test | test_museum_loan_register_2048 | producer_kproj_norm_rope_residual | ok | 29.206% estimated | 72/1152 | 9.471 |
| test | test_museum_loan_register_2048 | producer_kproj_norm_rope_coset | ok | 29.208% estimated | 72/1152 | 9.585 |
| test | test_pottery_kiln_diary_8192 | producer_kv_median | ok | 29.414% estimated | 72/4608 | 7.587 |
| test | test_pottery_kiln_diary_8192 | producer_kv_fixed127 | ok | 21.301% estimated | 72/4608 | 7.527 |
| test | test_pottery_kiln_diary_8192 | producer_frozen_exponent | ok | 29.790% estimated | 72/4608 | 7.027 |
| test | test_pottery_kiln_diary_8192 | producer_hidden_exponent | ok | 28.771% estimated | 72/4608 | 8.384 |
| test | test_pottery_kiln_diary_8192 | producer_hidden_abs_exponent | ok | 28.762% estimated | 72/4608 | 8.190 |
| test | test_pottery_kiln_diary_8192 | producer_pre_rope_residual | ok | 29.229% estimated | 72/4608 | 9.736 |
| test | test_pottery_kiln_diary_8192 | producer_pre_rope_coset | ok | 29.231% estimated | 72/4608 | 9.506 |
| test | test_pottery_kiln_diary_8192 | producer_kproj_norm_rope_residual | ok | 29.199% estimated | 72/4608 | 9.601 |
| test | test_pottery_kiln_diary_8192 | producer_kproj_norm_rope_coset | ok | 29.202% estimated | 72/4608 | 9.719 |
| test | test_python_dependency_walk_8192 | producer_kv_median | ok | 29.385% estimated | 72/4608 | 7.631 |
| test | test_python_dependency_walk_8192 | producer_kv_fixed127 | ok | 21.040% estimated | 72/4608 | 7.563 |
| test | test_python_dependency_walk_8192 | producer_frozen_exponent | ok | 29.766% estimated | 72/4608 | 6.991 |
| test | test_python_dependency_walk_8192 | producer_hidden_exponent | ok | 28.812% estimated | 72/4608 | 8.514 |
| test | test_python_dependency_walk_8192 | producer_hidden_abs_exponent | ok | 28.797% estimated | 72/4608 | 8.211 |
| test | test_python_dependency_walk_8192 | producer_pre_rope_residual | ok | 29.197% estimated | 72/4608 | 9.213 |
| test | test_python_dependency_walk_8192 | producer_pre_rope_coset | ok | 29.199% estimated | 72/4608 | 9.579 |
| test | test_python_dependency_walk_8192 | producer_kproj_norm_rope_residual | ok | 29.168% estimated | 72/4608 | 9.584 |
| test | test_python_dependency_walk_8192 | producer_kproj_norm_rope_coset | ok | 29.171% estimated | 72/4608 | 9.634 |
| test | test_museum_loan_register_8192 | producer_kv_median | ok | 29.506% estimated | 72/4608 | 7.558 |
| test | test_museum_loan_register_8192 | producer_kv_fixed127 | ok | 21.146% estimated | 72/4608 | 7.518 |
| test | test_museum_loan_register_8192 | producer_frozen_exponent | ok | 29.792% estimated | 72/4608 | 6.992 |
| test | test_museum_loan_register_8192 | producer_hidden_exponent | ok | 28.900% estimated | 72/4608 | 8.427 |
| test | test_museum_loan_register_8192 | producer_hidden_abs_exponent | ok | 28.853% estimated | 72/4608 | 8.199 |
| test | test_museum_loan_register_8192 | producer_pre_rope_residual | ok | 29.316% estimated | 72/4608 | 9.296 |
| test | test_museum_loan_register_8192 | producer_pre_rope_coset | ok | 29.318% estimated | 72/4608 | 9.374 |
| test | test_museum_loan_register_8192 | producer_kproj_norm_rope_residual | ok | 29.297% estimated | 72/4608 | 9.501 |
| test | test_museum_loan_register_8192 | producer_kproj_norm_rope_coset | ok | 29.299% estimated | 72/4608 | 9.659 |

Continuation checks use complete DEV coverage and compare exact logits, tokens and appended KV. Hook auditing and producer feature copies are included in instrumented prefill capture; that duration is not production overhead for one deployed recipe.
