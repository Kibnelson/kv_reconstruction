# Lite host-RoPE optimization

Status: **completed**. GPU measured: **True**.

Existing Lite fitted state and original packets; both arms must reproduce exactly the same bytes.
AB/BA alternating ordinary synchronized wall samples. Speedup is reference / host-RoPE; below 1 is slower.

| Document | Kappa | Complete bpv | Reference encode ms | Host-RoPE encode ms | Encode speedup | Reference decode ms | Host-RoPE decode ms | Decode speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_code | 0 | 8.89163 | 340.830 | 230.012 | 1.482x | 427.156 | 306.878 | 1.392x |
| test_code | 0.5 | 3.99045 | 477.786 | 391.286 | 1.221x | 433.653 | 312.472 | 1.388x |
| test_code | 1 | 3.09592 | 477.025 | 392.399 | 1.216x | 431.648 | 312.511 | 1.381x |
| test_text | 0 | 8.86772 | 338.793 | 228.126 | 1.485x | 429.312 | 307.121 | 1.398x |
| test_text | 0.5 | 3.99420 | 572.428 | 510.989 | 1.120x | 519.662 | 411.601 | 1.263x |
| test_text | 1 | 3.11033 | 572.894 | 514.094 | 1.114x | 516.242 | 412.477 | 1.252x |

Profiler diagnostics below are separate instrumented runs, not ordinary latency.

| Case / arm | Encode stream synchronizations | Decode stream synchronizations |
|---|---:|---:|
| lite_k0_test_text / reference | 1794 | 1793 |
| lite_k0_test_text / host_rope | 2 | 1 |

Kappa 0 is lossless; nonzero kappas reproduce the existing lossy reconstruction exactly.
Packet export/upload, offline fitting and model quality evaluation are outside codec timings.
Both engines are resident during paired timing; Torch baselines/peaks include this paired setup.
Cold state initialization is separate. Shared fitted state, packet bytes and sample distributions are in results.json.
Full traces remain under runs/. No pool conversion, attention, network, SSD or serving result is implied.
