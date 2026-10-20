# Prefill side-code probability study

Status: **completed**

Exact BF16 page-independent probability coding with no code, a same-page KV-derived code, an inverse-RoPE-aligned KV-derived code, or a producer-hidden-derived code. Four shared models fit a rate proxy on TRAIN only; actual retained BF16 codes and CPU arithmetic streams determine measured bytes. Complete state and framing charged. No compact attention, GPU entropy coding, concurrency scheduler, latency or novelty claim is established.

| Case | Tokens | Method | Packet savings | Net full-cache estimate | Dependency depth |
|---|---:|---|---:|---:|---:|
| test_pottery_kiln_diary | 2048 | frozen_exponent_control (ok) | 29.36% | 29.22% | 0 |
| test_pottery_kiln_diary | 2048 | side_none (ok) | 33.19% | 32.20% | 0 |
| test_pottery_kiln_diary | 2048 | side_kv (ok) | 32.64% | 31.43% | 0 |
| test_pottery_kiln_diary | 2048 | side_kv_canonical (ok) | 32.71% | 31.50% | 0 |
| test_pottery_kiln_diary | 2048 | side_hidden (ok) | 32.70% | 31.43% | 0 |
| test_pottery_kiln_diary | 2048 | previous_layer_shared (ok) | 29.92% | 28.24% | 35 |

Local variants none/kv/kv_canonical/hidden share rank, decoder width, update budget and TRAIN-only data. The previous-layer control has separately reported input width, decoder width and training budget; it is not a parameter-matched proof.

All side-code bytes, CPU entropy streams, page metadata, actual indexed container and max(serialized,resident) complete model state are charged. Sampled full-context and shared-state concurrency numbers are ESTIMATES. They do not demonstrate a serving gain.

Features come from actual normalized hidden states feeding Qwen KV projections; learned BF16 codes are emitted inside attention callbacks and passed verbatim to later entropy encoding. Original model forward still materializes full KV. CPU entropy coding and device neural inference are timed separately. Random-page decode requires own packet plus frozen state; no previous-layer KV chain.

Full DEV packet-only generation checks:

- side_none: ok
- side_kv: ok
- side_kv_canonical: ok
- side_hidden: ok
- previous_layer_shared: ok
