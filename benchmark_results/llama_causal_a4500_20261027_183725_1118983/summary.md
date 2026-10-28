# Matched causal GPU KV packet comparison

Status: **failed**. GPU measured: **False**.

All rates below are serialized complete packets; shared fitted state is additional.
Block32 is a matched predictor-policy control in the new shared coder, not the historical INT64 bitstream.
Lossless means every original BF16 word matches. Lossy packets must exactly reproduce the encoder's intended reconstruction.
Receiver replay runs in a separate process with fixed state and packet bytes only.

| Document | Policy | Mode / step | Complete bpv | Saved | Encode wall ms | Decode wall ms | Original exact | PPL change | Verified |
|---|---|---|---:|---:|---:|---:|---|---:|---|

Failure: Expected a torch.device with a specified index or an integer, but got:cuda

Timing includes causal prediction, RoPE, adaptive widths, GPU entropy coding and compaction.
One final size synchronization and packet-directory validation use CPU control. Serialization/export, upload, fitting and model quality evaluation are separate.
Samples and Torch peaks are in results.json. Shared-state bytes exclude dynamic scratch; no resident-pool or serving claim.
BF16 products are validated on the same GPU/software. Cross-GPU reproducibility is untested.
Continuation quality is a short matched smoke evaluation, not full WikiText2 PPL or official AQUA/ExANS/SplitZip reproduction.
