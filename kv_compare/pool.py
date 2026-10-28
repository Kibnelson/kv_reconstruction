"""GPU-resident KV pool: what it costs to bring a cached prefix back, and how much fits.

A pool keeps finished requests' KV so that a later request with the same prefix skips its prefill. Measured here
on one GPU, for one document's prefix:

  - recompute: the model's own prefill of the prefix (what a miss costs);
  - copy raw BF16 KV back from CPU memory over PCIe (pinned and pageable host memory);
  - our codec kept compressed in GPU memory and decoded on reuse (the codec rows come from gpu_speed.fused_rows);
  - our codec kept compressed in CPU memory: the copy of the compressed bytes, then the decode;
  - the cost of putting a prefix into the pool: our encode against the prefill that produced the KV;
  - capacity: tokens per GB, and tokens that fit in the GPU memory left after the model's weights.
"""
from __future__ import annotations

import torch

from .gpu_speed import timer


def prefill_seconds(model, ids, device, reps=3):
    """Median seconds of the model's prefill of ids (KV cache built, one logit kept)."""
    x = torch.tensor([list(ids)], dtype=torch.long, device=device)

    def run():
        with torch.inference_mode():
            out = model(x, use_cache=True, logits_to_keep=1)
        del out
    return timer(run, device, warmup=1, reps=reps)


def copy_seconds(nbytes, device, reps=5):
    """Host-to-device and device-to-host copy times of nbytes: pinned and pageable host memory."""
    dev = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
    out = {}
    for pinned in (True, False):
        host = torch.empty(int(nbytes), dtype=torch.uint8, pin_memory=pinned and device.type == "cuda")
        host.fill_(1)
        key = "pinned" if pinned else "pageable"
        out[f"h2d_{key}"] = timer(lambda: dev.copy_(host, non_blocking=pinned), device, warmup=2, reps=reps)
        out[f"d2h_{key}"] = timer(lambda: host.copy_(dev, non_blocking=pinned), device, warmup=2, reps=reps)
        del host
    del dev
    return out
