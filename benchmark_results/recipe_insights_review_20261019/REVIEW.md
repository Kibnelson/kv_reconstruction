# Incremental recipe findings

This is an **exploratory CPU size diagnostic on one reused SmolLM2-135M capture**. Three disjoint prefix ranges fit dictionaries; a reserved DEV range is unused; the final 256 tokens supply the measured records. These ranges share one causal context. They are not independent held-out prompts or Qwen GPU validation.

All **33 predetermined candidates** reconstructed all **240 records** exactly: **7,920 record checks**, with full coverage of a 5,898,240-byte BF16 slice. Packet fields, JSON headers, calculated outer framing/indexes, and complete decoder-state charges are included. The frozen source hashes match the checked-in modules. The four tiny-Qwen continuation tests separately check exact logits, generated tokens, and appended KV for two steps; no model weights are downloaded for those tests.

| Increment or control | Savings | Interpretation |
|---|---:|---|
| Existing native exponent packing | 22.786% | Existing GPU-capable wire format |
| Fixed exponent 127 + tier coding | 22.774% | Source-independent control |
| Prefill median exponent + tier coding | **28.034%** | **+5.261 percentage points** over the matched fixed control |
| Midpoint, channel groups of 8 | 19.134% | Whole-symbol predictor baseline |
| Midpoint, time groups of 8 | 22.879% | +3.745 points from layout |
| Midpoint, time groups of 64 | 24.274% | +1.395 further points from larger groups |
| Affine, channel groups of 8 | 22.234% | Residual dictionary baseline |
| Affine + 16-entry residual dictionary, channel | 17.980% | IDs, headers and state outweigh correction savings |
| Affine + 64-entry residual dictionary, channel | 16.186% | Larger dictionary still loses |

The exponent path and whole-symbol path are alternatives. Their savings cannot be added. Exponent channel/time order preserves tier counts; its small size difference comes from JSON label length, not better entropy.

## What the components show

The prefill exponent anchor costs **0.125 bits per BF16 value** at a 64-token tile. It lowers exponent correction payload from about **3.797 to 2.818 bits/value**, with all eight original sign/mantissa bits retained. Secondary escapes fall from **45.56% to 24.03%**, and rare escapes from **4.78% to 1.07%**. The anchor's cost is included in the measured gain. This is evidence for useful tile-local metadata on this capture; it does not establish that prefill metadata alone is a novel idea.

The 16- and 64-entry dictionaries save only **0.297 and 0.509 correction bits/value**, while IDs cost **0.5 and 0.75 bits/value**. They lose before charging their additional headers and shared decoder state. Header compaction alone cannot rescue these particular dictionaries.

The stateless per-record size oracle picks the same median exponent recipe on every record, for both K and V. A one-byte mode selector therefore reduces savings slightly to **28.030%**. This sample gives no evidence that combining the tested stateless recipes per record adds savings. The oracle encodes all candidates and is not an implemented online selector.

## Next experiments

1. Keep the prefill exponent recipe as the leading candidate and test it on fresh Qwen prompts at 2,048 and 8,192 tokens on the A4500. Record encoding, restoration, transfer and metadata costs alongside size.
2. Improve exponent packet metadata and its segment index. They currently cost about **0.566 bits/value**, a gross 3.54-percentage-point overhead budget. Independent reads still need indexing, so only a measured fraction of that budget could be recovered.
3. Test better coding of the small exponent-correction alphabet, with tables and independent-read indexes included. Compare against the fixed-127 control and the current 2/3/9-bit coder.
4. Revisit dictionaries only if correction reduction can exceed IDs plus state/metadata costs. Adding more stages is not automatically useful.

The GPU command and result-push instructions are in `docs/prefill_recipes.md`. This experiment retains the full raw model cache during actual prefill observation. It does not demonstrate compact-resident attention, high concurrency, partial-capacity admission, SSD bandwidth, or lower serving latency. The native implementation uses eager tensor operations; a production GPU kernel remains future work.

## Reproduce the local diagnostic

The old raw capture is not shipped in this Git report. Given the matching source capture and its manifest:

```bash
python scripts/bench_recipe_increments.py \
  --input-npz /path/to/sample_raw.npz \
  --capture-manifest /path/to/test_capture_manifest.json \
  --tile-tokens 64 --full-records \
  --output runs/local_recipe_diagnostic
```

The report uses 256 fitting samples and four iterations; the Qwen GPU runner defaults to 1,024 and eight. Full original results, per-record traces, state provenance and controls are retained losslessly in `results.json.gz`; see `README.md` for extraction and `artifact_manifest.json` for hashes.
