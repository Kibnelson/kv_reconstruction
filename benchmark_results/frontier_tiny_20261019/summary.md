# Forward producer metadata study

Status: **completed**

Default callbacks build and verify recipes inside actual attention forward, using diagnostic CPU copies staged back to the selected device. Repeated codec timings separately replay captured features after prefill. All payload/state bytes count; full raw KV remains required. CPU tiny runs establish functionality only. No compact attention, concurrency or latency gain is inferred.

| Split | Case | Method | Status | Savings incl. state | Delta vs frozen, pp | Bits/word incl. state | 40% target | Records | Decode ms/raw MiB |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| dev | dev_orchard_sensor_log | producer_kv_median | ok | -126.172% measured | +172.380 | 36.1875 | no | 6/6 | 1127.476 |
| dev | dev_orchard_sensor_log | producer_frozen_exponent | ok | -298.551% measured | +0.000 | 63.7682 | no | 6/6 | 1037.443 |
| dev | dev_orchard_sensor_log | producer_pre_rope_coset | ok | -155.859% measured | +142.692 | 40.9375 | no | 6/6 | 1887.122 |
| dev | dev_orchard_sensor_log | frontier_cost_final_raw8 | ok | -395.736% measured | -97.184 | 79.3177 | no | 6/6 | 1346.519 |
| dev | dev_orchard_sensor_log | frontier_shift_final_raw8 | ok | -395.736% measured | -97.184 | 79.3177 | no | 6/6 | 1527.544 |
| dev | dev_orchard_sensor_log | frontier_shift_final_plane_raw | ok | -397.493% measured | -98.942 | 79.5990 | no | 6/6 | 1464.114 |
| dev | dev_orchard_sensor_log | frontier_shift_final_plane_first | ok | -398.275% measured | -99.723 | 79.7240 | no | 6/6 | 1561.382 |
| dev | dev_orchard_sensor_log | frontier_shift_final_plane_prev | ok | -397.884% measured | -99.333 | 79.6615 | no | 6/6 | 1411.077 |
| dev | dev_orchard_sensor_log | frontier_shift_pre_rope_raw8 | ok | -410.905% measured | -112.354 | 81.7448 | no | 6/6 | 1680.305 |
| dev | dev_orchard_sensor_log | frontier_shift_pre_rope_plane_raw | ok | -412.663% measured | -114.111 | 82.0260 | no | 6/6 | 1569.802 |
| dev | dev_orchard_sensor_log | frontier_shift_pre_rope_plane_first | ok | -413.444% measured | -114.893 | 82.1510 | no | 6/6 | 2063.261 |
| dev | dev_orchard_sensor_log | frontier_shift_pre_rope_plane_prev | ok | -413.053% measured | -114.502 | 82.0885 | no | 6/6 | 1820.579 |
| dev | dev_orchard_sensor_log | frontier_shift_kproj_raw8 | ok | -414.160% measured | -115.609 | 82.2656 | no | 6/6 | 2051.334 |
| dev | dev_orchard_sensor_log | frontier_shift_kproj_plane_raw | ok | -415.918% measured | -117.367 | 82.5469 | no | 6/6 | 1647.615 |
| dev | dev_orchard_sensor_log | frontier_shift_kproj_plane_first | ok | -416.699% measured | -118.148 | 82.6719 | no | 6/6 | 3038.500 |
| dev | dev_orchard_sensor_log | frontier_shift_kproj_plane_prev | ok | -416.309% measured | -117.757 | 82.6094 | no | 6/6 | 6152.306 |
| test | test_pottery_kiln_diary_64 | producer_kv_median | ok | -125.317% estimated | +14.917 | 36.0508 | no | 4/16 | 1232.067 |
| test | test_pottery_kiln_diary_64 | producer_frozen_exponent | ok | -140.234% estimated | +0.000 | 38.4375 | no | 4/16 | 998.575 |
| test | test_pottery_kiln_diary_64 | producer_pre_rope_coset | ok | -152.283% estimated | -12.048 | 40.3652 | no | 4/16 | 2681.592 |
| test | test_pottery_kiln_diary_64 | frontier_cost_final_raw8 | ok | -221.387% estimated | -81.152 | 51.4219 | no | 4/16 | 1245.430 |
| test | test_pottery_kiln_diary_64 | frontier_shift_final_raw8 | ok | -221.387% estimated | -81.152 | 51.4219 | no | 4/16 | 1194.167 |
| test | test_pottery_kiln_diary_64 | frontier_shift_final_plane_raw | ok | -223.145% estimated | -82.910 | 51.7031 | no | 4/16 | 1340.201 |
| test | test_pottery_kiln_diary_64 | frontier_shift_final_plane_first | ok | -223.926% estimated | -83.691 | 51.8281 | no | 4/16 | 930.129 |
| test | test_pottery_kiln_diary_64 | frontier_shift_final_plane_prev | ok | -223.535% estimated | -83.301 | 51.7656 | no | 4/16 | 1242.051 |
| test | test_pottery_kiln_diary_64 | frontier_shift_pre_rope_raw8 | ok | -227.960% estimated | -87.726 | 52.4736 | no | 4/16 | 1321.572 |
| test | test_pottery_kiln_diary_64 | frontier_shift_pre_rope_plane_raw | ok | -229.718% estimated | -89.484 | 52.7549 | no | 4/16 | 1356.529 |
| test | test_pottery_kiln_diary_64 | frontier_shift_pre_rope_plane_first | ok | -230.499% estimated | -90.265 | 52.8799 | no | 4/16 | 1547.158 |
| test | test_pottery_kiln_diary_64 | frontier_shift_pre_rope_plane_prev | ok | -230.109% estimated | -89.874 | 52.8174 | no | 4/16 | 1551.193 |
| test | test_pottery_kiln_diary_64 | frontier_shift_kproj_raw8 | ok | -228.937% estimated | -88.702 | 52.6299 | no | 4/16 | 1646.308 |
| test | test_pottery_kiln_diary_64 | frontier_shift_kproj_plane_raw | ok | -230.695% estimated | -90.460 | 52.9111 | no | 4/16 | 1629.771 |
| test | test_pottery_kiln_diary_64 | frontier_shift_kproj_plane_first | ok | -231.476% estimated | -91.241 | 53.0361 | no | 4/16 | 1206.464 |
| test | test_pottery_kiln_diary_64 | frontier_shift_kproj_plane_prev | ok | -231.085% estimated | -90.851 | 52.9736 | no | 4/16 | 1190.875 |
| test | test_pottery_kiln_diary_96 | producer_kv_median | ok | -125.529% estimated | -16.691 | 36.0846 | no | 4/24 | 986.683 |
| test | test_pottery_kiln_diary_96 | producer_frozen_exponent | ok | -108.838% estimated | +0.000 | 33.4141 | no | 4/24 | 902.327 |
| test | test_pottery_kiln_diary_96 | producer_pre_rope_coset | ok | -151.701% estimated | -42.863 | 40.2721 | no | 4/24 | 1530.995 |
| test | test_pottery_kiln_diary_96 | frontier_cost_final_raw8 | ok | -186.938% estimated | -78.101 | 45.9102 | no | 4/24 | 1012.271 |
| test | test_pottery_kiln_diary_96 | frontier_shift_final_raw8 | ok | -186.938% estimated | -78.101 | 45.9102 | no | 4/24 | 926.109 |
| test | test_pottery_kiln_diary_96 | frontier_shift_final_plane_raw | ok | -188.696% estimated | -79.858 | 46.1914 | no | 4/24 | 1254.326 |
| test | test_pottery_kiln_diary_96 | frontier_shift_final_plane_first | ok | -189.478% estimated | -80.640 | 46.3164 | no | 4/24 | 1202.469 |
| test | test_pottery_kiln_diary_96 | frontier_shift_final_plane_prev | ok | -189.087% estimated | -80.249 | 46.2539 | no | 4/24 | 1222.950 |
| test | test_pottery_kiln_diary_96 | frontier_shift_pre_rope_raw8 | ok | -191.581% estimated | -82.743 | 46.6530 | no | 4/24 | 1149.705 |
| test | test_pottery_kiln_diary_96 | frontier_shift_pre_rope_plane_raw | ok | -193.339% estimated | -84.501 | 46.9342 | no | 4/24 | 1059.831 |
| test | test_pottery_kiln_diary_96 | frontier_shift_pre_rope_plane_first | ok | -194.120% estimated | -85.282 | 47.0592 | no | 4/24 | 1295.975 |
| test | test_pottery_kiln_diary_96 | frontier_shift_pre_rope_plane_prev | ok | -193.730% estimated | -84.892 | 46.9967 | no | 4/24 | 1341.735 |
| test | test_pottery_kiln_diary_96 | frontier_shift_kproj_raw8 | ok | -192.273% estimated | -83.435 | 46.7637 | no | 4/24 | 1768.488 |
| test | test_pottery_kiln_diary_96 | frontier_shift_kproj_plane_raw | ok | -194.031% estimated | -85.193 | 47.0449 | no | 4/24 | 1305.917 |
| test | test_pottery_kiln_diary_96 | frontier_shift_kproj_plane_first | ok | -194.812% estimated | -85.974 | 47.1699 | no | 4/24 | 1366.266 |
| test | test_pottery_kiln_diary_96 | frontier_shift_kproj_plane_prev | ok | -194.421% estimated | -85.583 | 47.1074 | no | 4/24 | 1837.600 |

Continuation checks use complete DEV coverage and compare exact logits, tokens and appended KV. Hook auditing and producer feature copies are included in instrumented prefill capture; that duration is not production overhead for one deployed recipe.
