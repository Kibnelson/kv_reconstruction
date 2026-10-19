# Forward producer metadata study

Status: **completed**

Default callbacks build and verify recipes inside actual attention forward, using diagnostic CPU copies staged back to the selected device. Repeated codec timings separately replay captured features after prefill. All payload/state bytes count; full raw KV remains required. CPU tiny runs establish functionality only. No compact attention, concurrency or latency gain is inferred.

| Split | Case | Method | Status | Savings incl. state | Delta vs frozen, pp | Bits/word incl. state | 40% target | Records | Decode ms/raw MiB |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| dev | dev_orchard_sensor_log | producer_frozen_exponent | ok | 29.592% measured | +0.000 | 11.2652 | no | 288/288 | 7.376 |
| dev | dev_orchard_sensor_log | frontier_cost_final_raw8 | ok | 30.108% measured | +0.515 | 11.1828 | no | 288/288 | 8.214 |
| dev | dev_orchard_sensor_log | conditional_fast_pre_rope | ok | 30.549% measured | +0.957 | 11.1122 | no | 288/288 | 8.780 |
| dev | dev_orchard_sensor_log | conditional_huffman_exponent | ok | 31.990% measured | +2.398 | 10.8816 | no | 288/288 | 10.610 |
| dev | dev_orchard_sensor_log | conditional_huffman_exponent_sign | ok | 32.398% measured | +2.806 | 10.8163 | no | 288/288 | 22.170 |
| dev | dev_orchard_sensor_log | conditional_tier_sign | ok | 30.314% measured | +0.722 | 11.1498 | no | 288/288 | 20.189 |
| dev | dev_orchard_sensor_log | conditional_raw_pair | ok | -0.164% measured | -29.756 | 16.0263 | no | 288/288 | 0.223 |
| dev | dev_orchard_sensor_log | conditional_raw_fullword | ok | -1.105% measured | -30.697 | 16.1767 | no | 288/288 | 4.797 |
| dev | dev_orchard_sensor_log | conditional_mean_predictor | ok | 29.921% measured | +0.329 | 11.2126 | no | 288/288 | 8.831 |
| dev | dev_orchard_sensor_log | conditional_anchor_r8 | ok | 26.464% measured | -3.129 | 11.7658 | no | 288/288 | 8.903 |
| dev | dev_orchard_sensor_log | conditional_anchor_r32 | ok | 17.093% measured | -12.500 | 13.2652 | no | 288/288 | 9.028 |
| dev | dev_orchard_sensor_log | conditional_latent_r8 | ok | 22.333% measured | -7.259 | 12.4267 | no | 288/288 | 8.789 |
| dev | dev_orchard_sensor_log | conditional_latent_r32 | ok | 1.238% measured | -28.354 | 15.8019 | no | 288/288 | 8.821 |
| test | test_pottery_kiln_diary_2048 | producer_frozen_exponent | ok | 29.906% estimated | +0.000 | 11.2151 | no | 72/1152 | 7.347 |
| test | test_pottery_kiln_diary_2048 | frontier_cost_final_raw8 | ok | 30.441% estimated | +0.535 | 11.1294 | no | 72/1152 | 8.150 |
| test | test_pottery_kiln_diary_2048 | conditional_fast_pre_rope | ok | 30.954% estimated | +1.048 | 11.0473 | no | 72/1152 | 8.856 |
| test | test_pottery_kiln_diary_2048 | conditional_huffman_exponent | ok | 32.841% estimated | +2.936 | 10.7454 | no | 72/1152 | 10.516 |
| test | test_pottery_kiln_diary_2048 | conditional_huffman_exponent_sign | ok | 33.326% estimated | +3.420 | 10.6678 | no | 72/1152 | 21.805 |
| test | test_pottery_kiln_diary_2048 | conditional_tier_sign | ok | 31.280% estimated | +1.374 | 10.9952 | no | 72/1152 | 20.203 |
| test | test_pottery_kiln_diary_2048 | conditional_raw_pair | ok | -0.164% estimated | -30.070 | 16.0263 | no | 72/1152 | 0.226 |
| test | test_pottery_kiln_diary_2048 | conditional_raw_fullword | ok | -0.874% estimated | -30.779 | 16.1398 | no | 72/1152 | 4.781 |
| test | test_pottery_kiln_diary_2048 | conditional_mean_predictor | ok | 30.797% estimated | +0.891 | 11.0725 | no | 72/1152 | 8.899 |
| test | test_pottery_kiln_diary_2048 | conditional_anchor_r8 | ok | 29.932% estimated | +0.027 | 11.2108 | no | 72/1152 | 8.810 |
| test | test_pottery_kiln_diary_2048 | conditional_anchor_r32 | ok | 27.589% estimated | -2.316 | 11.5857 | no | 72/1152 | 8.822 |
| test | test_pottery_kiln_diary_2048 | conditional_latent_r8 | ok | 28.900% estimated | -1.006 | 11.3760 | no | 72/1152 | 8.800 |
| test | test_pottery_kiln_diary_2048 | conditional_latent_r32 | ok | 23.626% estimated | -6.280 | 12.2199 | no | 72/1152 | 9.240 |
| test | test_python_dependency_walk_2048 | producer_frozen_exponent | ok | 29.889% estimated | +0.000 | 11.2178 | no | 72/1152 | 7.425 |
| test | test_python_dependency_walk_2048 | frontier_cost_final_raw8 | ok | 30.419% estimated | +0.530 | 11.1329 | no | 72/1152 | 8.351 |
| test | test_python_dependency_walk_2048 | conditional_fast_pre_rope | ok | 30.934% estimated | +1.045 | 11.0505 | no | 72/1152 | 9.278 |
| test | test_python_dependency_walk_2048 | conditional_huffman_exponent | ok | 32.836% estimated | +2.947 | 10.7463 | no | 72/1152 | 10.592 |
| test | test_python_dependency_walk_2048 | conditional_huffman_exponent_sign | ok | 33.248% estimated | +3.359 | 10.6803 | no | 72/1152 | 21.846 |
| test | test_python_dependency_walk_2048 | conditional_tier_sign | ok | 31.188% estimated | +1.299 | 11.0100 | no | 72/1152 | 20.100 |
| test | test_python_dependency_walk_2048 | conditional_raw_pair | ok | -0.164% estimated | -30.053 | 16.0263 | no | 72/1152 | 0.225 |
| test | test_python_dependency_walk_2048 | conditional_raw_fullword | ok | -0.874% estimated | -30.763 | 16.1398 | no | 72/1152 | 4.784 |
| test | test_python_dependency_walk_2048 | conditional_mean_predictor | ok | 30.777% estimated | +0.888 | 11.0757 | no | 72/1152 | 8.851 |
| test | test_python_dependency_walk_2048 | conditional_anchor_r8 | ok | 29.913% estimated | +0.024 | 11.2140 | no | 72/1152 | 8.744 |
| test | test_python_dependency_walk_2048 | conditional_anchor_r32 | ok | 27.570% estimated | -2.319 | 11.5889 | no | 72/1152 | 8.856 |
| test | test_python_dependency_walk_2048 | conditional_latent_r8 | ok | 28.880% estimated | -1.009 | 11.3792 | no | 72/1152 | 8.832 |
| test | test_python_dependency_walk_2048 | conditional_latent_r32 | ok | 23.606% estimated | -6.283 | 12.2231 | no | 72/1152 | 8.799 |
| test | test_museum_loan_register_2048 | producer_frozen_exponent | ok | 29.916% estimated | +0.000 | 11.2135 | no | 72/1152 | 7.306 |
| test | test_museum_loan_register_2048 | frontier_cost_final_raw8 | ok | 30.447% estimated | +0.531 | 11.1285 | no | 72/1152 | 8.138 |
| test | test_museum_loan_register_2048 | conditional_fast_pre_rope | ok | 30.966% estimated | +1.051 | 11.0454 | no | 72/1152 | 8.891 |
| test | test_museum_loan_register_2048 | conditional_huffman_exponent | ok | 32.852% estimated | +2.937 | 10.7436 | no | 72/1152 | 10.565 |
| test | test_museum_loan_register_2048 | conditional_huffman_exponent_sign | ok | 33.285% estimated | +3.369 | 10.6745 | no | 72/1152 | 21.843 |
| test | test_museum_loan_register_2048 | conditional_tier_sign | ok | 31.239% estimated | +1.324 | 11.0017 | no | 72/1152 | 20.010 |
| test | test_museum_loan_register_2048 | conditional_raw_pair | ok | -0.164% estimated | -30.080 | 16.0263 | no | 72/1152 | 0.225 |
| test | test_museum_loan_register_2048 | conditional_raw_fullword | ok | -0.874% estimated | -30.789 | 16.1398 | no | 72/1152 | 4.777 |
| test | test_museum_loan_register_2048 | conditional_mean_predictor | ok | 30.809% estimated | +0.893 | 11.0706 | no | 72/1152 | 8.938 |
| test | test_museum_loan_register_2048 | conditional_anchor_r8 | ok | 29.944% estimated | +0.029 | 11.2089 | no | 72/1152 | 8.798 |
| test | test_museum_loan_register_2048 | conditional_anchor_r32 | ok | 27.602% estimated | -2.314 | 11.5838 | no | 72/1152 | 8.795 |
| test | test_museum_loan_register_2048 | conditional_latent_r8 | ok | 28.912% estimated | -1.004 | 11.3741 | no | 72/1152 | 8.834 |
| test | test_museum_loan_register_2048 | conditional_latent_r32 | ok | 23.638% estimated | -6.278 | 12.2179 | no | 72/1152 | 8.875 |
| test | test_pottery_kiln_diary_8192 | producer_frozen_exponent | ok | 29.790% estimated | +0.000 | 11.2336 | no | 72/4608 | 7.450 |
| test | test_pottery_kiln_diary_8192 | frontier_cost_final_raw8 | ok | 30.333% estimated | +0.543 | 11.1467 | no | 72/4608 | 8.115 |
| test | test_pottery_kiln_diary_8192 | conditional_fast_pre_rope | ok | 30.945% estimated | +1.155 | 11.0489 | no | 72/4608 | 8.816 |
| test | test_pottery_kiln_diary_8192 | conditional_huffman_exponent | ok | 32.971% estimated | +3.181 | 10.7247 | no | 72/4608 | 10.558 |
| test | test_pottery_kiln_diary_8192 | conditional_huffman_exponent_sign | ok | 33.555% estimated | +3.765 | 10.6312 | no | 72/4608 | 21.763 |
| test | test_pottery_kiln_diary_8192 | conditional_tier_sign | ok | 31.491% estimated | +1.701 | 10.9614 | no | 72/4608 | 20.258 |
| test | test_pottery_kiln_diary_8192 | conditional_raw_pair | ok | -0.165% estimated | -29.955 | 16.0263 | no | 72/4608 | 0.228 |
| test | test_pottery_kiln_diary_8192 | conditional_raw_fullword | ok | -0.816% estimated | -30.606 | 16.1306 | no | 72/4608 | 4.784 |
| test | test_pottery_kiln_diary_8192 | conditional_mean_predictor | ok | 30.905% estimated | +1.115 | 11.0552 | no | 72/4608 | 8.905 |
| test | test_pottery_kiln_diary_8192 | conditional_anchor_r8 | ok | 30.689% estimated | +0.899 | 11.0898 | no | 72/4608 | 8.772 |
| test | test_pottery_kiln_diary_8192 | conditional_anchor_r32 | ok | 30.103% estimated | +0.313 | 11.1835 | no | 72/4608 | 8.829 |
| test | test_pottery_kiln_diary_8192 | conditional_latent_r8 | ok | 30.431% estimated | +0.641 | 11.1311 | no | 72/4608 | 8.869 |
| test | test_pottery_kiln_diary_8192 | conditional_latent_r32 | ok | 29.112% estimated | -0.678 | 11.3421 | no | 72/4608 | 8.920 |
| test | test_python_dependency_walk_8192 | producer_frozen_exponent | ok | 29.766% estimated | +0.000 | 11.2374 | no | 72/4608 | 7.327 |
| test | test_python_dependency_walk_8192 | frontier_cost_final_raw8 | ok | 30.303% estimated | +0.536 | 11.1515 | no | 72/4608 | 8.223 |
| test | test_python_dependency_walk_8192 | conditional_fast_pre_rope | ok | 30.913% estimated | +1.147 | 11.0539 | no | 72/4608 | 8.847 |
| test | test_python_dependency_walk_8192 | conditional_huffman_exponent | ok | 32.953% estimated | +3.187 | 10.7275 | no | 72/4608 | 10.574 |
| test | test_python_dependency_walk_8192 | conditional_huffman_exponent_sign | ok | 33.436% estimated | +3.670 | 10.6502 | no | 72/4608 | 21.736 |
| test | test_python_dependency_walk_8192 | conditional_tier_sign | ok | 31.358% estimated | +1.592 | 10.9827 | no | 72/4608 | 20.158 |
| test | test_python_dependency_walk_8192 | conditional_raw_pair | ok | -0.165% estimated | -29.931 | 16.0263 | no | 72/4608 | 0.227 |
| test | test_python_dependency_walk_8192 | conditional_raw_fullword | ok | -0.816% estimated | -30.582 | 16.1306 | no | 72/4608 | 4.783 |
| test | test_python_dependency_walk_8192 | conditional_mean_predictor | ok | 30.874% estimated | +1.107 | 11.0602 | no | 72/4608 | 8.869 |
| test | test_python_dependency_walk_8192 | conditional_anchor_r8 | ok | 30.657% estimated | +0.891 | 11.0948 | no | 72/4608 | 8.772 |
| test | test_python_dependency_walk_8192 | conditional_anchor_r32 | ok | 30.071% estimated | +0.305 | 11.1886 | no | 72/4608 | 8.848 |
| test | test_python_dependency_walk_8192 | conditional_latent_r8 | ok | 30.399% estimated | +0.633 | 11.1361 | no | 72/4608 | 8.851 |
| test | test_python_dependency_walk_8192 | conditional_latent_r32 | ok | 29.080% estimated | -0.686 | 11.3471 | no | 72/4608 | 8.865 |
| test | test_museum_loan_register_8192 | producer_frozen_exponent | ok | 29.792% estimated | +0.000 | 11.2333 | no | 72/4608 | 7.325 |
| test | test_museum_loan_register_8192 | frontier_cost_final_raw8 | ok | 30.334% estimated | +0.542 | 11.1465 | no | 72/4608 | 8.057 |
| test | test_museum_loan_register_8192 | conditional_fast_pre_rope | ok | 30.941% estimated | +1.149 | 11.0494 | no | 72/4608 | 8.808 |
| test | test_museum_loan_register_8192 | conditional_huffman_exponent | ok | 32.969% estimated | +3.177 | 10.7249 | no | 72/4608 | 11.123 |
| test | test_museum_loan_register_8192 | conditional_huffman_exponent_sign | ok | 33.488% estimated | +3.696 | 10.6419 | no | 72/4608 | 21.831 |
| test | test_museum_loan_register_8192 | conditional_tier_sign | ok | 31.422% estimated | +1.630 | 10.9725 | no | 72/4608 | 20.195 |
| test | test_museum_loan_register_8192 | conditional_raw_pair | ok | -0.165% estimated | -29.957 | 16.0263 | no | 72/4608 | 0.226 |
| test | test_museum_loan_register_8192 | conditional_raw_fullword | ok | -0.816% estimated | -30.608 | 16.1306 | no | 72/4608 | 4.787 |
| test | test_museum_loan_register_8192 | conditional_mean_predictor | ok | 30.902% estimated | +1.109 | 11.0558 | no | 72/4608 | 9.380 |
| test | test_museum_loan_register_8192 | conditional_anchor_r8 | ok | 30.685% estimated | +0.893 | 11.0903 | no | 72/4608 | 8.834 |
| test | test_museum_loan_register_8192 | conditional_anchor_r32 | ok | 30.099% estimated | +0.307 | 11.1841 | no | 72/4608 | 8.811 |
| test | test_museum_loan_register_8192 | conditional_latent_r8 | ok | 30.427% estimated | +0.635 | 11.1316 | no | 72/4608 | 8.918 |
| test | test_museum_loan_register_8192 | conditional_latent_r32 | ok | 29.108% estimated | -0.684 | 11.3426 | no | 72/4608 | 8.911 |

Continuation checks use complete DEV coverage and compare exact logits, tokens and appended KV. Hook auditing and producer feature copies are included in instrumented prefill capture; that duration is not production overhead for one deployed recipe.

## Four conditional experiments

The fixed candidates compare exponent Huffman coding, separate sign coding, joint prediction from retained KV anchors, and joint prediction from actual-prefill hidden features. Rank variants, raw packets, zero/mean predictors and fast pre-RoPE controls are retained even when they lose. All registered candidates receive complete DEV continuation checks unless explicitly disabled.

See decision_summary.json for matched control deltas, complete component-bit ledgers, per-layer rates, aggregated prediction/correction/fallback diagnostics and separate encode/decode allocation peaks. conditional_diagnostics in results.json adds per-head source costs, lower-byte distributions, sign transitions and observed K/V correlations. Diagnostics are descriptive; no TEST-derived model is fitted.

Predictor fit uses ridge MSE initialization followed by TRAIN-only shrinkage selection using actual NumPy BF16 correction-byte proxies. These proxies are recorded separately from measured GPU packets. Encoder-only latent weights are conservatively included in headline decoder-state charges, with their numerical role broken out. Eager encode timings include bounded diagnostic aggregates; no serving speedup is inferred.
