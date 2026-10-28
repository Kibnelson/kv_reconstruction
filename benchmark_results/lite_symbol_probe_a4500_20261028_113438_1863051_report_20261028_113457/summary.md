# Exact BF16 rank-delta symbol rate probe

Status: **completed**. GPU measured: **True**.

All candidate rates are estimates from known-original-KV replay. This experiment does not implement new compressed packets or measure a candidate decoder. The existing coarse/2 graph decoder remains the current implementation.
The fixed predictor, original BF16 captures and V64/B32 geometry are reused. Symbol frequency tables use TRAIN only. Radius/context selection uses validation only; TEST is scored after the selection receipt is written.
Validation reuses the original separate scale-calibration document; the predictor and its spreads stay frozen. The 32 context bins are the fixed clamp(es−ep+7,0,31) rule using scale and BF16-prediction exponent fields. All 28 layers, including layer zero, are coded in the projection.

Selected on validation: radius **1023**, **32** contexts; table `24d082f7c6ed88f4c4b041d8630e3846e00f6ccaf827a37befbde496e29c6cd8`.

| Validation candidate | Ideal symbol+escape bpv |
|---|---:|
| r127_c1 | 11.69648 |
| r127_c32 | 10.75442 |
| r255_c1 | 11.02418 |
| r255_c32 | 10.05880 |
| r511_c1 | 10.76160 |
| r511_c32 | 9.80334 |
| r1023_c1 | 10.72377 |
| r1023_c32 | 9.77383 |
| r2047_c1 | 10.76484 |
| r2047_c32 | 9.81673 |

| TEST method | Estimated bpv incl. projected overhead | Estimated saved | Actual packet |
|---|---:|---:|---|
| rank_delta_selected | 10.20131 | 36.24% | No |
| rank_delta_unconditional_control | 11.11532 | 30.53% | No |

Ideal costs use positive 16-bit normalized frequencies and 16 raw bits for every escape. Estimates add all B-lane rANS states, INT64 program directories and a hypothetical header carrying the new format, radius, context count and table hash. Finite-state coding/alignment overhead remains unmeasured; these estimates are neither actual packet sizes nor bounds. Shared tables are separate.

| Reused measured method | Complete bpv | Decode ms |
|---|---:|---:|
| raw_bf16 | 16.00009 | 0.335 |
| splitzip_official_framed | 12.04281 | 0.517 |
| exans_official_fixed_framed | 12.37970 | 3.665 |
| lite_entropy_reference_graph | 9.09012 | 83.316 |

Native controls and current coarse decoder above are inherited GPU measurements, **not retimed**. Their original model/data, timestamps, runtime policies and source report hash remain in the receipt. No estimated candidate time is combined with an old decoder time.

| Selected shared table accounting | Bytes |
|---|---:|
| actual_frequency_array_bytes | 262144 |
| actual_cdf_array_bytes | 262272 |
| actual_table_arrays_bytes | 524416 |
| all_candidate_npz_bytes | 315467 |
| projected_gpu_frequency_bytes | 262144 |
| projected_gpu_cdf_bytes | 262272 |
| projected_gpu_inverse_uint16_bytes | 4194304 |
| projected_gpu_total_table_bytes | 4718720 |
| predictor_gpu_bytes | 339854080 |

Required candidate decoder latency to tie a baseline after precompressed movement; these are planning targets. Negative values mean the estimated candidate bytes already exceed the full baseline transfer/decode budget.

| Candidate | Baseline | Effective Gb/s | Required decode ≤ ms |
|---|---|---:|---:|
| rank_delta_selected | raw_bf16 | 1 | 170.58811 |
| rank_delta_selected | raw_bf16 | 100 | 2.03789 |
| rank_delta_selected | raw_bf16 | 200 | 1.18663 |
| rank_delta_selected | raw_bf16 | 300 | 0.90287 |
| rank_delta_selected | raw_bf16 | 400 | 0.76100 |
| rank_delta_selected | splitzip_official_framed | 1 | 54.58359 |
| rank_delta_selected | splitzip_official_framed | 100 | 1.05758 |
| rank_delta_selected | splitzip_official_framed | 200 | 0.78724 |
| rank_delta_selected | splitzip_official_framed | 300 | 0.69713 |
| rank_delta_selected | splitzip_official_framed | 400 | 0.65208 |
| rank_delta_selected | exans_official_fixed_framed | 1 | 67.62242 |
| rank_delta_selected | exans_official_fixed_framed | 100 | 4.30432 |
| rank_delta_selected | exans_official_fixed_framed | 200 | 3.98453 |
| rank_delta_selected | exans_official_fixed_framed | 300 | 3.87794 |
| rank_delta_selected | exans_official_fixed_framed | 400 | 3.82464 |
| rank_delta_unconditional_control | raw_bf16 | 1 | 143.75269 |
| rank_delta_unconditional_control | raw_bf16 | 100 | 1.76954 |
| rank_delta_unconditional_control | raw_bf16 | 200 | 1.05245 |
| rank_delta_unconditional_control | raw_bf16 | 300 | 0.81342 |
| rank_delta_unconditional_control | raw_bf16 | 400 | 0.69391 |
| rank_delta_unconditional_control | splitzip_official_framed | 1 | 27.74817 |
| rank_delta_unconditional_control | splitzip_official_framed | 100 | 0.78922 |
| rank_delta_unconditional_control | splitzip_official_framed | 200 | 0.65307 |
| rank_delta_unconditional_control | splitzip_official_framed | 300 | 0.60768 |
| rank_delta_unconditional_control | splitzip_official_framed | 400 | 0.58499 |
| rank_delta_unconditional_control | exans_official_fixed_framed | 1 | 40.78700 |
| rank_delta_unconditional_control | exans_official_fixed_framed | 100 | 4.03597 |
| rank_delta_unconditional_control | exans_official_fixed_framed | 200 | 3.85036 |
| rank_delta_unconditional_control | exans_official_fixed_framed | 300 | 3.78849 |
| rank_delta_unconditional_control | exans_official_fixed_framed | 400 | 3.75755 |

Layer patterns for the selected configuration versus the predeclared unconditional control. Positive context gain means scale/ULP bins help; the gain is zero if validation selects one context.

| Layer | Escape % | Delta within ±31 % | Context gain bpv | Raw escape bpv |
|---|---:|---:|---:|---:|
| 0 | 8.897 | 55.868 | 0.91424 | 1.42355 |
| 1 | 8.129 | 51.770 | 0.91244 | 1.30065 |
| 2 | 11.789 | 36.790 | 0.93950 | 1.88622 |
| 3 | 13.943 | 31.926 | 1.00814 | 2.23090 |
| 4 | 12.296 | 34.913 | 0.94392 | 1.96741 |
| 5 | 12.102 | 34.136 | 0.94337 | 1.93637 |
| 6 | 12.187 | 33.412 | 0.93963 | 1.95000 |
| 7 | 11.220 | 35.670 | 0.90270 | 1.79517 |
| 8 | 11.981 | 33.728 | 0.92260 | 1.91694 |
| 9 | 11.830 | 33.554 | 0.92514 | 1.89279 |
| 10 | 10.758 | 35.406 | 0.88157 | 1.72130 |
| 11 | 11.171 | 34.882 | 0.90576 | 1.78742 |
| 12 | 11.670 | 34.033 | 0.90580 | 1.86722 |
| 13 | 11.955 | 33.540 | 0.93230 | 1.91284 |
| 14 | 12.858 | 32.806 | 0.95130 | 2.05733 |
| 15 | 12.980 | 31.761 | 0.96951 | 2.07676 |
| 16 | 11.404 | 34.694 | 0.90373 | 1.82467 |
| 17 | 11.155 | 34.912 | 0.88580 | 1.78478 |
| 18 | 10.794 | 36.232 | 0.88205 | 1.72701 |
| 19 | 11.577 | 34.510 | 0.91596 | 1.85236 |
| 20 | 11.551 | 34.228 | 0.92201 | 1.84823 |
| 21 | 10.807 | 36.884 | 0.87994 | 1.72915 |
| 22 | 10.792 | 36.558 | 0.89083 | 1.72667 |
| 23 | 11.614 | 35.552 | 0.91173 | 1.85818 |
| 24 | 10.437 | 37.388 | 0.87017 | 1.66988 |
| 25 | 12.312 | 32.942 | 0.94729 | 1.96993 |
| 26 | 10.324 | 39.459 | 0.85496 | 1.65189 |
| 27 | 9.566 | 40.292 | 0.82987 | 1.53064 |

Escape explanations: sign crossings and zero/subnormal predictions can overlap. Their counts describe the fixed selected radius and never enter selection.

| Layer | Escapes | Nonfinite escapes | Sign-cross escapes | Zero/subnormal-prediction escapes | Same-sign normal escapes |
|---|---:|---:|---:|---:|---:|
| 0 | 186587 | 0 | 185158 | 0 | 1429 |
| 1 | 170479 | 0 | 169083 | 0 | 1396 |
| 2 | 247231 | 0 | 245308 | 0 | 1923 |
| 3 | 292408 | 0 | 290120 | 0 | 2288 |
| 4 | 257873 | 0 | 255882 | 0 | 1991 |
| 5 | 253804 | 0 | 251830 | 0 | 1974 |
| 6 | 255590 | 0 | 253557 | 0 | 2033 |
| 7 | 235297 | 0 | 233468 | 0 | 1829 |
| 8 | 251257 | 0 | 249311 | 0 | 1946 |
| 9 | 248092 | 0 | 246117 | 0 | 1975 |
| 10 | 225614 | 0 | 223904 | 0 | 1710 |
| 11 | 234281 | 0 | 232434 | 0 | 1847 |
| 12 | 244740 | 0 | 242851 | 0 | 1889 |
| 13 | 250720 | 0 | 248801 | 0 | 1919 |
| 14 | 269658 | 0 | 267660 | 0 | 1998 |
| 15 | 272205 | 0 | 270059 | 0 | 2146 |
| 16 | 239163 | 0 | 237374 | 0 | 1789 |
| 17 | 233935 | 0 | 232109 | 0 | 1826 |
| 18 | 226362 | 0 | 224536 | 1 | 1826 |
| 19 | 242792 | 0 | 240802 | 0 | 1990 |
| 20 | 242251 | 0 | 240391 | 0 | 1860 |
| 21 | 226643 | 0 | 224836 | 0 | 1807 |
| 22 | 226318 | 0 | 224619 | 0 | 1699 |
| 23 | 243556 | 0 | 241635 | 0 | 1921 |
| 24 | 218874 | 0 | 217218 | 0 | 1656 |
| 25 | 258203 | 0 | 256178 | 0 | 2025 |
| 26 | 216516 | 0 | 214838 | 0 | 1678 |
| 27 | 200624 | 0 | 199105 | 0 | 1519 |

| Replay document | Role | Predictor-only wall ms | Instrumented replay+hist wall ms | Histogram + explanation CUDA intervals ms |
|---|---|---:|---:|---:|
| train0 | TRAIN | 108.174 | 664.466 | 533.053 |
| train1 | TRAIN | 106.918 | 638.552 | 509.340 |
| train2 | TRAIN | 107.060 | 638.044 | 506.665 |
| train3 | TRAIN | 108.114 | 636.530 | 506.486 |
| tables | validation | 116.356 | 635.292 | 505.628 |
| test_text | TEST | 107.330 | 648.284 | 492.174 |
| test_code | TEST | 107.462 | 646.767 | 491.268 |

Histogram observer CUDA intervals include observer operations and potential host gaps. TEST symbol roundtrip proofs run in an additional pass, recorded separately in JSON. Replay timings use known original KV and instrumentation. They do not measure entropy coding, GPU receiver decoding, network/SSD, compressed attention or serving. Source verification, uploads, cold initialization and host table fitting are recorded separately. No predictor fitting, model download or native build is performed.
