# Exact-input CPU-to-GPU replay fixture

Frozen 4 TEST pages contain 1,474,560 original BF16 words. All layers and the original first/final-page positions are retained. The CPU transfer packets are copied byte for byte. The original frozen state is distributed as two file parts; joining them reconstructs its original NPZ bytes exactly.

Original pool CPU payload: 11.472233 bits/value, 28.30% reduction. Complete block32 CPU transfer packets: 9.430599 bits/value, 41.06% reduction.

These are the earlier measured CPU rates. This fixture has not been CUDA validated. A GPU replay must verify every original word, pool payload byte and integer predictor result before making GPU correctness or speed claims. The entropy coder remains CPU unless separately implemented on GPU.

Pool byte rates exclude shared calibration, offsets, escape indexes and GPU allocation/alignment. Transfer rates include packet framing but exclude fixed decoder state; manifest.json reports serialized and resident array costs separately. Fixture NPY headers, token IDs and offsets are reproducibility artifacts. No model weights, hidden activations or extra hidden-metadata networks are included.

## Prepare the frozen state

From the repository root, join the two distribution parts before running the
GPU benchmark:

```bash
cat benchmark_fixtures/team_transfer_smol_replay_v1/block32.npz.part00 \
    benchmark_fixtures/team_transfer_smol_replay_v1/block32.npz.part01 \
  > benchmark_fixtures/team_transfer_smol_replay_v1/block32.npz
```

The runner verifies the complete state's size and SHA256 against `manifest.json`
before loading it. Distribution parts are also included in the integrity ledger.
The split only accommodates the upload request limit; it changes no state array
or codec packet, and it is not an additional KV compression saving. The
reassembled NPZ is a local generated file and should not be committed again.
