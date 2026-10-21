# Four-arm Block32 probability experiment

Status: **completed**.

Parameter device: `cuda:0`. GPU parameters measured: **True**.
CPU interval probabilities and range coding; CPU causal context with per-group device staging. Means are frozen.

DEV-selected storage winner: **B**. TEST never changes selection.

| Arm | Packet bpv | Reduction | Hints (B) | Delta bytes vs A | CPU decoder state MiB | Encoder state KiB | Decode median s/page |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 9.555740 | 40.277% | 0 | +0 | 163.735 | 0.000 | 2.660680 |
| B | 8.828674 | 44.821% | 0 | -190596 | 164.375 | 0.000 | 2.743100 |
| C | 8.829651 | 44.815% | 256 | -190340 | 164.377 | 128.125 | 2.758985 |
| D | 8.829651 | 44.815% | 256 | -190340 | 164.377 | 160.125 | 2.758792 |

A uses the original Block32 entropy stream with the same fixed binary framing as B/C/D. Original legacy A packet sizes are reported separately. Shared state is outside warm packet rates.

A = fixed width; B = causal-context width; C = B + paid produced-KV hint; D = B + equally budgeted actual-prefill-H hint.

See `insights.json`, `layer_diagnostics.csv` and `diagnostics.json.gz` for paired byte gains versus hint cost, early/later channel groups, K/V, layer patterns, scale histograms, hint occupancy and DEV zero/shuffle sensitivity. Ideal group gains and syndrome screens are diagnostics, not measured group packets or new coset savings.

Every measured page is independently decoded in a fresh process from baseline state, adapter state and packet only. CPU and requested-device parameter digests, restored words and hint IDs are compared. Device timings include validation/staging; this is not a GPU entropy decoder, compressed attention or a serving speedup.
