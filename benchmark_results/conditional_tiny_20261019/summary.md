# Forward producer metadata study

Status: **completed**

Default callbacks build and verify recipes inside actual attention forward, using diagnostic CPU copies staged back to the selected device. Repeated codec timings separately replay captured features after prefill. All payload/state bytes count; full raw KV remains required. CPU tiny runs establish functionality only. No compact attention, concurrency or latency gain is inferred.

| Split | Case | Method | Status | Savings incl. state | Delta vs frozen, pp | Bits/word incl. state | 40% target | Records | Decode ms/raw MiB |
|---|---|---|---|---:|---:|---:|---|---:|---:|
| dev | dev_orchard_sensor_log | producer_frozen_exponent | ok | -201.090% measured | +0.000 | 48.1745 | no | 2/2 | 374.270 |
| dev | dev_orchard_sensor_log | frontier_cost_final_raw8 | ok | -249.951% measured | -48.861 | 55.9922 | no | 2/2 | 412.255 |
| dev | dev_orchard_sensor_log | conditional_fast_pre_rope | ok | -256.201% measured | -55.111 | 56.9922 | no | 2/2 | 478.295 |
| dev | dev_orchard_sensor_log | conditional_huffman_exponent | ok | -2233.008% measured | -2031.917 | 373.2812 | no | 2/2 | 559.915 |
| dev | dev_orchard_sensor_log | conditional_huffman_exponent_sign | ok | -2265.120% measured | -2064.030 | 378.4193 | no | 2/2 | 991.700 |
| dev | dev_orchard_sensor_log | conditional_tier_sign | ok | -2258.626% measured | -2057.536 | 377.3802 | no | 2/2 | 1054.043 |
| dev | dev_orchard_sensor_log | conditional_raw_pair | ok | -14.290% measured | +186.800 | 18.2865 | no | 2/2 | 6.167 |
| dev | dev_orchard_sensor_log | conditional_raw_fullword | ok | -279.525% measured | -78.434 | 60.7240 | no | 2/2 | 321.615 |
| dev | dev_orchard_sensor_log | conditional_mean_predictor | ok | -506.055% measured | -304.964 | 96.9688 | no | 2/2 | 476.543 |
| dev | dev_orchard_sensor_log | conditional_anchor_r8 | ok | -617.480% measured | -416.390 | 114.7969 | no | 2/2 | 614.681 |
| dev | dev_orchard_sensor_log | conditional_anchor_r32 | ok | -822.917% measured | -621.826 | 147.6667 | no | 2/2 | 733.934 |
| dev | dev_orchard_sensor_log | conditional_latent_r8 | ok | -696.012% measured | -494.922 | 127.3620 | no | 2/2 | 440.555 |
| dev | dev_orchard_sensor_log | conditional_latent_r32 | ok | -1096.257% measured | -895.166 | 191.4010 | no | 2/2 | 470.848 |
| test | test_pottery_kiln_diary_64 | producer_frozen_exponent | ok | -56.104% measured | +0.000 | 24.9766 | no | 2/2 | 153.349 |
| test | test_pottery_kiln_diary_64 | frontier_cost_final_raw8 | ok | -73.999% measured | -17.896 | 27.8398 | no | 2/2 | 168.387 |
| test | test_pottery_kiln_diary_64 | conditional_fast_pre_rope | ok | -76.520% measured | -20.416 | 28.2432 | no | 2/2 | 219.283 |
| test | test_pottery_kiln_diary_64 | conditional_huffman_exponent | ok | -816.675% measured | -760.571 | 146.6680 | no | 2/2 | 387.911 |
| test | test_pottery_kiln_diary_64 | conditional_huffman_exponent_sign | ok | -828.900% measured | -772.797 | 148.6240 | no | 2/2 | 624.713 |
| test | test_pottery_kiln_diary_64 | conditional_tier_sign | ok | -827.612% measured | -771.509 | 148.4180 | no | 2/2 | 494.782 |
| test | test_pottery_kiln_diary_64 | conditional_raw_pair | ok | -5.359% measured | +50.745 | 16.8574 | no | 2/2 | 2.415 |
| test | test_pottery_kiln_diary_64 | conditional_raw_fullword | ok | -105.164% measured | -49.060 | 32.8262 | no | 2/2 | 205.222 |
| test | test_pottery_kiln_diary_64 | conditional_mean_predictor | ok | -170.215% measured | -114.111 | 43.2344 | no | 2/2 | 184.657 |
| test | test_pottery_kiln_diary_64 | conditional_anchor_r8 | ok | -212.000% measured | -155.896 | 49.9199 | no | 2/2 | 213.402 |
| test | test_pottery_kiln_diary_64 | conditional_anchor_r32 | ok | -289.038% measured | -232.935 | 62.2461 | no | 2/2 | 210.502 |
| test | test_pottery_kiln_diary_64 | conditional_latent_r8 | ok | -241.449% measured | -185.345 | 54.6318 | no | 2/2 | 211.033 |
| test | test_pottery_kiln_diary_64 | conditional_latent_r32 | ok | -391.541% measured | -335.437 | 78.6465 | no | 2/2 | 305.044 |
| test | test_pottery_kiln_diary_96 | producer_frozen_exponent | ok | -33.716% measured | +0.000 | 21.3945 | no | 4/4 | 208.787 |
| test | test_pottery_kiln_diary_96 | frontier_cost_final_raw8 | ok | -51.424% measured | -17.708 | 24.2279 | no | 4/4 | 240.578 |
| test | test_pottery_kiln_diary_96 | conditional_fast_pre_rope | ok | -49.207% measured | -15.491 | 23.8730 | no | 4/4 | 259.923 |
| test | test_pottery_kiln_diary_96 | conditional_huffman_exponent | ok | -543.958% measured | -510.242 | 103.0332 | no | 4/4 | 415.122 |
| test | test_pottery_kiln_diary_96 | conditional_huffman_exponent_sign | ok | -560.164% measured | -526.449 | 105.6263 | no | 4/4 | 702.079 |
| test | test_pottery_kiln_diary_96 | conditional_tier_sign | ok | -557.943% measured | -524.227 | 105.2708 | no | 4/4 | 629.223 |
| test | test_pottery_kiln_diary_96 | conditional_raw_pair | ok | -6.991% measured | +26.725 | 17.1185 | no | 4/4 | 3.121 |
| test | test_pottery_kiln_diary_96 | conditional_raw_fullword | ok | -76.660% measured | -42.944 | 28.2656 | no | 4/4 | 397.844 |
| test | test_pottery_kiln_diary_96 | conditional_mean_predictor | ok | -111.678% measured | -77.962 | 33.8685 | no | 4/4 | 276.303 |
| test | test_pottery_kiln_diary_96 | conditional_anchor_r8 | ok | -139.535% measured | -105.819 | 38.3255 | no | 4/4 | 308.599 |
| test | test_pottery_kiln_diary_96 | conditional_anchor_r32 | ok | -190.902% measured | -157.186 | 46.5443 | no | 4/4 | 260.251 |
| test | test_pottery_kiln_diary_96 | conditional_latent_r8 | ok | -159.167% measured | -125.452 | 41.4668 | no | 4/4 | 278.543 |
| test | test_pottery_kiln_diary_96 | conditional_latent_r32 | ok | -259.237% measured | -225.521 | 57.4779 | no | 4/4 | 258.168 |

Continuation checks use complete DEV coverage and compare exact logits, tokens and appended KV. Hook auditing and producer feature copies are included in instrumented prefill capture; that duration is not production overhead for one deployed recipe.

## Four conditional experiments

The fixed candidates compare exponent Huffman coding, separate sign coding, joint prediction from retained KV anchors, and joint prediction from actual-prefill hidden features. Rank variants, raw packets, zero/mean predictors and fast pre-RoPE controls are retained even when they lose. All registered candidates receive complete DEV continuation checks unless explicitly disabled.

See decision_summary.json for matched control deltas, complete component-bit ledgers, per-layer rates, aggregated prediction/correction/fallback diagnostics and separate encode/decode allocation peaks. conditional_diagnostics in results.json adds per-head source costs, lower-byte distributions, sign transitions and observed K/V correlations. Diagnostics are descriptive; no TEST-derived model is fitted.

Predictor fit uses ridge MSE initialization followed by TRAIN-only shrinkage selection using actual NumPy BF16 correction-byte proxies. These proxies are recorded separately from measured GPU packets. Encoder-only latent weights are conservatively included in headline decoder-state charges, with their numerical role broken out. Eager encode timings include bounded diagnostic aggregates; no serving speedup is inferred.
