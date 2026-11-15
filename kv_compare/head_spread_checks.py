"""Functional checks for the experimental kernels, on CUDA or explicit interpreter.

CUDA mode rejects TRITON_INTERPRET and unavailable CUDA. CPU mode requires
--interpret and is only software validation. Nothing here measures model gains.
"""
from __future__ import annotations

import argparse
import json
import os


def check(device, all_patterns=False):
    import hashlib
    from pathlib import Path
    import tempfile
    import numpy as np
    import torch
    from . import fused as fu, layerwise as lw
    from .head_spread import HeadSpreadCodec, Policy, prior_reference
    from .head_spread_wire import consts_digest, pack, unpack, save_state, decode_packet, inspect_packet

    torch.set_num_threads(1)
    if device.type == "cuda" and (not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET", "0") != "0"):
        raise RuntimeError("actual CUDA execution is required; no interpreter/CPU fallback")
    generator = torch.Generator().manual_seed(42)
    L, T, hd, Hk = (2, 128, 128, 1) if all_patterns else ((3, 32, 32, 2) if device.type == "cuda" else (2, 8, 16, 1))
    C, W = Hk * hd, 2 * Hk * hd
    G = W // hd
    angles = torch.arange(T, dtype=torch.float64)[:, None] * torch.exp(-torch.arange(hd // 2, dtype=torch.float64))[None, :]
    consts = {"Aq": torch.randint(-3, 4, (L, W, W), dtype=torch.int8, generator=generator),
        "sa": torch.full((L, W), 0.002), "b": torch.randn((L, W), generator=generator) * 0.03,
        "sp": torch.full((L, W), 0.3), "isx": torch.full((L, W), 12.0),
        "cos": angles.cos().float().contiguous(), "sin": angles.sin().float().contiguous()}
    consts["Aq"][0].zero_()
    consts["inv"] = (fu.QS / (2 * np.log(2) * consts["sp"].double())).float()
    consts = {key: value.to(device) for key, value in consts.items()}
    if all_patterns:
        words = torch.arange(65536, dtype=torch.int64).to(torch.int16).reshape(L, T, W).to(device)
    else:
        words = (torch.randn((L, T, W), generator=generator) * 0.5).bfloat16().view(torch.int16).to(device)
        words[0, 0, :5] = torch.tensor([0, -32768, 32640, 32705, -128], dtype=torch.int16, device=device)
    raw_hash = hashlib.sha256(words.cpu().numpy().astype("<i2", copy=False).tobytes()).hexdigest()
    rng = np.random.default_rng(77)
    weights = (rng.normal(size=(L, G, G)) * 0.1 + np.eye(G)[None] * 0.5).astype(np.float32)
    bias = np.ones((L, G), np.float32) * 0.5
    policies = [Policy(), Policy(1, 0.0), Policy(1, 0.5), Policy(2, 0.5, 0, weights, bias),
                Policy(1, 0.5, 1), Policy(2, 0.5, 1, weights, bias)]
    report = {"device": str(device), "gpu_executed": device.type == "cuda", "all_patterns": all_patterns, "rows": []}
    for groups in [1, G]:
        k = dict(consts)
        if groups > 1:
            k["groups"] = groups
            aq = torch.randint(-2, 3, (L, W, W), dtype=torch.int8, generator=generator)
            rows, columns = torch.arange(W)[:, None], torch.arange(W)[None, :]
            aq *= (columns // hd < rows // hd).to(torch.int8)
            k["Acq"] = aq.to(device)
        baseline = lw.LayerwiseOneStep(k, C, hd, tokens_per_program=8)
        segments, cap = baseline.layout(T)
        global_control = HeadSpreadCodec(k, C, hd, Policy(), tokens_per_program=8)
        features, global_prior = global_control.model(words)
        bo, bs = baseline.encode(words, features, global_prior[:, 0],
            out=torch.zeros(segments * cap, dtype=torch.int16, device=device))
        if not torch.equal(baseline.decode(bo, bs, T), words):
            raise AssertionError("original baseline failed before experiment")
        digest = consts_digest(k, C, hd, 8, "uniform")
        for policy in policies:
            codec = HeadSpreadCodec(k, C, hd, policy, tokens_per_program=8)
            x, sums = codec.statistics(words)
            priors = codec.priors(sums)
            for layer in range(1, L):
                p = Policy(policy.kind, policy.alpha, policy.head_shift,
                    weights[layer] if policy.kind == 2 else None, bias[layer] if policy.kind == 2 else None)
                np.testing.assert_array_equal(priors[layer].cpu().numpy(), prior_reference(sums[layer - 1].cpu().numpy(), p, hd))
            out, starts = codec.encode(words, x, priors,
                out=torch.zeros(segments * cap, dtype=torch.int16, device=device))
            if not torch.equal(codec.decode(out, starts, T), words):
                raise AssertionError("head-aware codec changed BF16 words")
            packet, cost = pack(codec, out, starts, T, digest, policy, raw_hash)
            arena, received_starts, info = unpack(codec, packet, digest)
            if not torch.equal(codec.decode(arena, received_starts, T), words):
                raise AssertionError("compact packet failed exact reconstruction")
            if policy.kind == 0 or policy.alpha == 0:
                if not (torch.equal(starts, bs) and torch.equal(out, bo)):
                    raise AssertionError("zero-change control differs from original stream bytes")
            if policy.kind == 2 and policy.head_shift == 0:
                with tempfile.TemporaryDirectory() as folder:
                    state, wire = Path(folder) / "state.pt", Path(folder) / "packet.bin"
                    save_state(state, k, C, hd, 8, "uniform", policy, digest)
                    wire.write_bytes(packet)
                    worker = decode_packet(state, wire, device)
                    if not worker["bit_exact"] or worker["known_source_words"]:
                        raise AssertionError("state-plus-packet decoder failed")
            changed = bytearray(packet); changed[-1] ^= 1
            try:
                inspect_packet(bytes(changed))
            except ValueError:
                pass
            else:
                raise AssertionError("damaged packet was accepted")
            report["rows"].append({"groups": groups, **policy.description(),
                "values": words.numel(), "packet_bytes": cost["packet_bytes"], "exact": True})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--all-patterns", action="store_true")
    args = parser.parse_args()
    if args.device == "cpu":
        if not args.interpret:
            parser.error("CPU functional validation requires explicit --interpret")
        os.environ["TRITON_INTERPRET"] = "1"
    elif args.interpret or os.environ.get("TRITON_INTERPRET", "0") != "0":
        parser.error("CUDA checks must execute compiled kernels; unset TRITON_INTERPRET")
    import torch
    result = check(torch.device(args.device), args.all_patterns)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
