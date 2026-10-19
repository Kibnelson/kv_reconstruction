# Forward producer metadata study

Status: **completed**

Default callbacks build and verify recipes inside actual attention forward, using diagnostic CPU copies staged back to the selected device. Repeated codec timings separately replay captured features after prefill. All payload/state bytes count; full raw KV remains required. CPU tiny runs establish functionality only. No compact attention, concurrency or latency gain is inferred.

| Split | Case | Method | Status | Savings incl. state | Records | Decode ms/raw MiB |
|---|---|---|---|---:|---:|---:|
| dev | dev_orchard_sensor_log | producer_kv_median | ok | -126.172% measured | 6/6 | 1090.827 |
| dev | dev_orchard_sensor_log | producer_kv_fixed127 | ok | -124.788% measured | 6/6 | 985.293 |
| dev | dev_orchard_sensor_log | producer_frozen_exponent | ok | -298.551% measured | 6/6 | 854.674 |
| dev | dev_orchard_sensor_log | producer_hidden_exponent | ok | -353.271% measured | 6/6 | 977.956 |
| dev | dev_orchard_sensor_log | producer_hidden_abs_exponent | ok | -353.776% measured | 6/6 | 1225.129 |
| dev | dev_orchard_sensor_log | producer_pre_rope_residual | ok | -156.445% measured | 6/6 | 1495.500 |
| dev | dev_orchard_sensor_log | producer_pre_rope_coset | ok | -155.859% measured | 6/6 | 1737.248 |
| dev | dev_orchard_sensor_log | producer_kproj_norm_rope_residual | ok | -161.865% measured | 6/6 | 1952.229 |
| dev | dev_orchard_sensor_log | producer_kproj_norm_rope_coset | ok | -161.279% measured | 6/6 | 2033.186 |
| test | test_pottery_kiln_diary_64 | producer_kv_median | ok | -125.317% estimated | 4/16 | 948.604 |
| test | test_pottery_kiln_diary_64 | producer_kv_fixed127 | ok | -124.194% estimated | 4/16 | 952.229 |
| test | test_pottery_kiln_diary_64 | producer_frozen_exponent | ok | -140.234% estimated | 4/16 | 793.582 |
| test | test_pottery_kiln_diary_64 | producer_hidden_exponent | ok | -165.594% estimated | 4/16 | 951.943 |
| test | test_pottery_kiln_diary_64 | producer_hidden_abs_exponent | ok | -166.156% estimated | 4/16 | 1016.193 |
| test | test_pottery_kiln_diary_64 | producer_pre_rope_residual | ok | -152.869% estimated | 4/16 | 1460.688 |
| test | test_pottery_kiln_diary_64 | producer_pre_rope_coset | ok | -152.283% estimated | 4/16 | 1493.715 |
| test | test_pottery_kiln_diary_64 | producer_kproj_norm_rope_residual | ok | -155.688% estimated | 4/16 | 1530.835 |
| test | test_pottery_kiln_diary_64 | producer_kproj_norm_rope_coset | ok | -155.103% estimated | 4/16 | 1531.384 |
| test | test_python_dependency_walk_64 | producer_kv_median | ok | -125.562% estimated | 4/16 | 941.559 |
| test | test_python_dependency_walk_64 | producer_kv_fixed127 | ok | -124.536% estimated | 4/16 | 1042.412 |
| test | test_python_dependency_walk_64 | producer_frozen_exponent | ok | -140.356% estimated | 4/16 | 942.779 |
| test | test_python_dependency_walk_64 | producer_hidden_exponent | ok | -165.717% estimated | 4/16 | 1531.810 |
| test | test_python_dependency_walk_64 | producer_hidden_abs_exponent | ok | -166.302% estimated | 4/16 | 1599.378 |
| test | test_python_dependency_walk_64 | producer_pre_rope_residual | ok | -152.991% estimated | 4/16 | 1282.570 |
| test | test_python_dependency_walk_64 | producer_pre_rope_coset | ok | -152.405% estimated | 4/16 | 3384.760 |
| test | test_python_dependency_walk_64 | producer_kproj_norm_rope_residual | ok | -155.835% estimated | 4/16 | 1545.450 |
| test | test_python_dependency_walk_64 | producer_kproj_norm_rope_coset | ok | -155.249% estimated | 4/16 | 1862.734 |
| test | test_museum_loan_register_64 | producer_kv_median | ok | -125.244% estimated | 4/16 | 951.547 |
| test | test_museum_loan_register_64 | producer_kv_fixed127 | ok | -124.438% estimated | 4/16 | 1071.532 |
| test | test_museum_loan_register_64 | producer_frozen_exponent | ok | -140.039% estimated | 4/16 | 837.332 |
| test | test_museum_loan_register_64 | producer_hidden_exponent | ok | -165.033% estimated | 4/16 | 986.797 |
| test | test_museum_loan_register_64 | producer_hidden_abs_exponent | ok | -166.229% estimated | 4/16 | 1277.509 |
| test | test_museum_loan_register_64 | producer_pre_rope_residual | ok | -152.649% estimated | 4/16 | 1403.437 |
| test | test_museum_loan_register_64 | producer_pre_rope_coset | ok | -152.063% estimated | 4/16 | 1585.661 |
| test | test_museum_loan_register_64 | producer_kproj_norm_rope_residual | ok | -155.688% estimated | 4/16 | 1567.652 |
| test | test_museum_loan_register_64 | producer_kproj_norm_rope_coset | ok | -155.103% estimated | 4/16 | 1764.215 |
| test | test_pottery_kiln_diary_96 | producer_kv_median | ok | -125.529% estimated | 4/24 | 1321.110 |
| test | test_pottery_kiln_diary_96 | producer_kv_fixed127 | ok | -124.308% estimated | 4/24 | 1471.482 |
| test | test_pottery_kiln_diary_96 | producer_frozen_exponent | ok | -108.838% estimated | 4/24 | 1336.898 |
| test | test_pottery_kiln_diary_96 | producer_hidden_exponent | ok | -128.373% estimated | 4/24 | 1355.159 |
| test | test_pottery_kiln_diary_96 | producer_hidden_abs_exponent | ok | -129.000% estimated | 4/24 | 1709.450 |
| test | test_pottery_kiln_diary_96 | producer_pre_rope_residual | ok | -152.287% estimated | 4/24 | 1493.957 |
| test | test_pottery_kiln_diary_96 | producer_pre_rope_coset | ok | -151.701% estimated | 4/24 | 1788.946 |
| test | test_pottery_kiln_diary_96 | producer_kproj_norm_rope_residual | ok | -154.712% estimated | 4/24 | 1515.309 |
| test | test_pottery_kiln_diary_96 | producer_kproj_norm_rope_coset | ok | -154.126% estimated | 4/24 | 1927.923 |
| test | test_python_dependency_walk_96 | producer_kv_median | ok | -125.553% estimated | 4/24 | 1023.551 |
| test | test_python_dependency_walk_96 | producer_kv_fixed127 | ok | -124.357% estimated | 4/24 | 1207.843 |
| test | test_python_dependency_walk_96 | producer_frozen_exponent | ok | -108.887% estimated | 4/24 | 1237.426 |
| test | test_python_dependency_walk_96 | producer_hidden_exponent | ok | -128.153% estimated | 4/24 | 1134.224 |
| test | test_python_dependency_walk_96 | producer_hidden_abs_exponent | ok | -128.878% estimated | 4/24 | 1932.119 |
| test | test_python_dependency_walk_96 | producer_pre_rope_residual | ok | -152.287% estimated | 4/24 | 4120.460 |
| test | test_python_dependency_walk_96 | producer_pre_rope_coset | ok | -151.701% estimated | 4/24 | 2089.325 |
| test | test_python_dependency_walk_96 | producer_kproj_norm_rope_residual | ok | -154.736% estimated | 4/24 | 1531.266 |
| test | test_python_dependency_walk_96 | producer_kproj_norm_rope_coset | ok | -154.150% estimated | 4/24 | 1539.763 |
| test | test_museum_loan_register_96 | producer_kv_median | ok | -125.627% estimated | 4/24 | 1069.317 |
| test | test_museum_loan_register_96 | producer_kv_fixed127 | ok | -124.430% estimated | 4/24 | 1393.391 |
| test | test_museum_loan_register_96 | producer_frozen_exponent | ok | -108.667% estimated | 4/24 | 1175.594 |
| test | test_museum_loan_register_96 | producer_hidden_exponent | ok | -128.031% estimated | 4/24 | 1384.279 |
| test | test_museum_loan_register_96 | producer_hidden_abs_exponent | ok | -128.780% estimated | 4/24 | 972.926 |
| test | test_museum_loan_register_96 | producer_pre_rope_residual | ok | -152.214% estimated | 4/24 | 1453.990 |
| test | test_museum_loan_register_96 | producer_pre_rope_coset | ok | -151.628% estimated | 4/24 | 1561.386 |
| test | test_museum_loan_register_96 | producer_kproj_norm_rope_residual | ok | -154.907% estimated | 4/24 | 2237.353 |
| test | test_museum_loan_register_96 | producer_kproj_norm_rope_coset | ok | -154.321% estimated | 4/24 | 1770.035 |

Continuation checks use complete DEV coverage and compare exact logits, tokens and appended KV. Hook auditing and producer feature copies are included in instrumented prefill capture; that duration is not production overhead for one deployed recipe.
