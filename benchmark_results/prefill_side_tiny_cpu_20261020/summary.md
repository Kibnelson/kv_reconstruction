# Prefill side-code probability study

Status: **completed**

Exact BF16 page-independent probability coding with no code, a same-page KV-derived code, an inverse-RoPE-aligned KV-derived code, or a producer-hidden-derived code. Four shared models fit a rate proxy on TRAIN only; actual retained BF16 codes and CPU arithmetic streams determine measured bytes. Complete state and framing charged. No compact attention, GPU entropy coding, concurrency scheduler, latency or novelty claim is established.

| Case | Tokens | Method | Packet savings | Net full-cache estimate | Dependency depth |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 16 | frozen_exponent_control (ok) | -73.22% | -565.06% | 0 |
| test_pottery_kiln_diary | 16 | side_none (ok) | -118.85% | -959.84% | 0 |
| test_pottery_kiln_diary | 16 | side_kv (ok) | -128.54% | -1017.60% | 0 |
| test_pottery_kiln_diary | 16 | side_kv_canonical (ok) | -132.62% | -1024.41% | 0 |
| test_pottery_kiln_diary | 16 | side_hidden (ok) | -122.61% | -1012.13% | 0 |
| test_pottery_kiln_diary | 16 | previous_layer_shared (ok) | -108.96% | -1565.67% | 1 |
| test_python_dependency_walk | 16 | frozen_exponent_control (ok) | -73.32% | -565.16% | 0 |
| test_python_dependency_walk | 16 | side_none (ok) | -119.34% | -960.33% | 0 |
| test_python_dependency_walk | 16 | side_kv (ok) | -126.59% | -1015.65% | 0 |
| test_python_dependency_walk | 16 | side_kv_canonical (ok) | -130.91% | -1022.71% | 0 |
| test_python_dependency_walk | 16 | side_hidden (ok) | -122.02% | -1011.55% | 0 |
| test_python_dependency_walk | 16 | previous_layer_shared (ok) | -108.62% | -1565.33% | 1 |
| test_pottery_kiln_diary | 24 | frozen_exponent_control (ok) | -73.06% | -400.96% | 0 |
| test_pottery_kiln_diary | 24 | side_none (ok) | -117.64% | -678.30% | 0 |
| test_pottery_kiln_diary | 24 | side_kv (ok) | -128.37% | -721.08% | 0 |
| test_pottery_kiln_diary | 24 | side_kv_canonical (ok) | -130.19% | -724.72% | 0 |
| test_pottery_kiln_diary | 24 | side_hidden (ok) | -125.57% | -718.59% | 0 |
| test_pottery_kiln_diary | 24 | previous_layer_shared (ok) | -108.51% | -1079.65% | 1 |
| test_python_dependency_walk | 24 | frozen_exponent_control (ok) | -73.18% | -401.07% | 0 |
| test_python_dependency_walk | 24 | side_none (ok) | -117.92% | -678.58% | 0 |
| test_python_dependency_walk | 24 | side_kv (ok) | -126.71% | -719.42% | 0 |
| test_python_dependency_walk | 24 | side_kv_canonical (ok) | -128.56% | -723.10% | 0 |
| test_python_dependency_walk | 24 | side_hidden (ok) | -123.93% | -716.94% | 0 |
| test_python_dependency_walk | 24 | previous_layer_shared (ok) | -108.63% | -1079.77% | 1 |

Local variants none/kv/kv_canonical/hidden share rank, decoder width, update budget and TRAIN-only data. The previous-layer control has separately reported input width, decoder width and training budget; it is not a parameter-matched proof.

All side-code bytes, CPU entropy streams, page metadata, actual indexed container and max(serialized,resident) complete model state are charged. Sampled full-context and shared-state concurrency numbers are ESTIMATES. They do not demonstrate a serving gain.

Features come from actual normalized hidden states feeding Qwen KV projections; learned BF16 codes are emitted inside attention callbacks and passed verbatim to later entropy encoding. Original model forward still materializes full KV. CPU entropy coding and device neural inference are timed separately. Random-page decode requires own packet plus frozen state; no previous-layer KV chain.

Full DEV packet-only generation checks:

- side_none: ok
- side_kv: ok
- side_kv_canonical: ok
- side_hidden: ok
- previous_layer_shared: ok
