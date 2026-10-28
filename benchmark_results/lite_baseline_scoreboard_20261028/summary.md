# Current recorded Lite versus baseline scoreboard

Recorded GPU measurements on identical frozen words from separate runs; retiming is prepared separately. Model link costs are estimates, not network measurements.

Original BF16 Llama-3.2-3B, all 28 layers, two 512-token text/code requests; 56 MiB raw per request. Arithmetic means of per-document synchronized wall medians; complete packet rates, shared state additional.

| Method | Kappa | Complete bpv | Saved | Encode ms | Decode ms |
|---|---:|---:|---:|---:|---:|
| lite_host_rope | 0 | 8.87967 | 44.50% | 229.069 | 307.000 |
| lite_host_rope | 0.5 | 3.99232 | 75.05% | 451.138 | 362.037 |
| lite_host_rope | 1 | 3.10313 | 80.61% | 453.246 | 362.494 |
| raw_bf16 | 0 | 16.00009 | -0.00% | 0.423 | 0.339 |
| fp8_e4m3fn_saturating | 0 | 8.00009 | 50.00% | 1.238 | 0.410 |
| splitzip_official_framed | 0 | 12.04281 | 24.73% | 0.874 | 0.521 |
| exans_official_fixed_framed | 0 | 12.37970 | 22.63% | 15.455 | 3.848 |

| Lossless Lite versus | Packet ratio | Encode time ratio | Decode time ratio |
|---|---:|---:|---:|
| raw_bf16 | 0.555 | 541.53 | 905.50 |
| splitzip_official_framed | 0.737 | 262.05 | 588.97 |
| exans_official_fixed_framed | 0.717 | 14.82 | 79.77 |

Time ratios above 1 mean slower. Lossy rows require a quality constraint; they do not compete in the lossless network model.
Official AQUA remains blocked. The new team component report does not establish complete-packet performance and is not mixed into these measurements.

The next A4500 experiment retimes every baseline and sweeps the existing Lite coder lane grouping. See docs/lite_baseline_tuning.md.

The recorded lossy text timings followed profiling and both arms slowed; these historical means are provisional. New timing workers run no profiler.
