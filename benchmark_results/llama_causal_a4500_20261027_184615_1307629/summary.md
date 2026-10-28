# Matched causal GPU KV packet comparison

Status: **completed**. GPU measured: **True**.

Model: unsloth/Llama-3.2-3B at d4446454d87d51aa42e1fb174f25acc5f8762331; all 28 original BF16 layers.
All rates below are serialized complete packets; shared fitted state is additional.
Block32 is a matched predictor-policy control in the new shared coder, not the historical INT64 bitstream.
Lossless means every original BF16 word matches. Lossy packets must exactly reproduce the encoder's intended reconstruction.
Receiver replay runs in a separate process with fixed state and packet bytes only.

| Document | Policy | Mode / step | Complete bpv | Saved | Encode wall ms | Decode wall ms | Original exact | PPL change | Verified |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| test_text | block32 | lossless / 0 | 8.99569 | 43.78% | 1279.271 | 1629.557 | True | +0.000% | verified |
| test_code | block32 | lossless / 0 | 8.96788 | 43.95% | 1262.205 | 1626.508 | True | +0.000% | verified |
| test_text | block32 | lossy / 0.5 | 4.08985 | 74.44% | 1782.039 | 1610.896 | False | +2.723% | verified |
| test_code | block32 | lossy / 0.5 | 4.04082 | 74.74% | 1777.615 | 1614.387 | False | -1.456% | verified |
| test_text | block32 | lossy / 1 | 3.16122 | 80.24% | 1779.598 | 1602.629 | False | +6.691% | verified |
| test_code | block32 | lossy / 1 | 3.11306 | 80.54% | 1778.122 | 1601.771 | False | +2.590% | verified |
| test_text | lite | lossless / 0 | 8.86772 | 44.58% | 330.014 | 417.829 | True | +0.000% | verified |
| test_code | lite | lossless / 0 | 8.89163 | 44.43% | 331.488 | 419.408 | True | +0.000% | verified |
| test_text | lite | lossy / 0.5 | 3.99420 | 75.04% | 464.315 | 426.269 | False | +3.061% | verified |
| test_code | lite | lossy / 0.5 | 3.99045 | 75.06% | 462.488 | 423.896 | False | +0.665% | verified |
| test_text | lite | lossy / 1 | 3.11033 | 80.56% | 464.005 | 422.771 | False | +1.784% | verified |
| test_code | lite | lossy / 1 | 3.09592 | 80.65% | 463.890 | 421.461 | False | -0.056% | verified |

Timing includes causal prediction, RoPE, adaptive widths, GPU entropy coding and compaction.
One final size synchronization and packet-directory validation use CPU control. Serialization/export, upload, fitting and model quality evaluation are separate.
Samples and Torch peaks are in results.json. Shared-state bytes exclude dynamic scratch; no resident-pool or serving claim.
BF16 products are validated on the same GPU/software. Cross-GPU reproducibility is untested.
Continuation quality is a short matched smoke evaluation, not full WikiText2 PPL or official AQUA/ExANS/SplitZip reproduction.
