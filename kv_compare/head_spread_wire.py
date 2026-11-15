"""Small experiment wire: used stream tails, one paid index per segment.

The original fixed-capacity arena is not transmitted. Each uint32 index packs
the used tail length and its 0..7-word alignment padding. Decoding expands the
received tails into the original arena before invoking the unchanged rANS
layout. A fixed header pays for geometry, position, state and data identities.
This wrapper is common to the old baseline and every experimental policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time

import numpy as np
import torch

from . import layerwise as lw
from .head_spread import HeadSpreadCodec, Policy

triton, tl = lw.triton, lw.tl

MAGIC = b"THSPRV01"
HEADER = struct.Struct("<8s12I32s32s32s32s")
BASE_COMMIT = "d95fd10a1fae447d47c4ac6abaf77b90af1b7952"


if triton is not None:
    @triton.jit
    def _copy_tails(source, destination, offsets, bases, lengths,
                    PACK: tl.constexpr, BLOCK: tl.constexpr):
        segment = tl.program_id(0)
        n = tl.load(lengths + segment)
        a = tl.load(bases + segment)
        b = tl.load(offsets + segment)
        for chunk in range(0, tl.cdiv(n, BLOCK)):
            i = chunk * BLOCK + tl.arange(0, BLOCK)
            if PACK:
                value = tl.load(source + a + i, i < n, other=0)
                tl.store(destination + b + i, value, i < n)
            else:
                value = tl.load(source + b + i, i < n, other=0)
                tl.store(destination + a + i, value, i < n)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def tensors_bytes(values):
    seen, total = set(), 0
    for value in values:
        if not isinstance(value, torch.Tensor):
            continue
        s = value.untyped_storage()
        key = (str(value.device), s.data_ptr(), s.nbytes())
        if key not in seen:
            seen.add(key)
            total += s.nbytes()
    return int(total)


def consts_digest(consts, C, hd, TB, mode):
    h = hashlib.sha256(json.dumps(dict(C=C, hd=hd, TB=TB, mode=mode,
        base=BASE_COMMIT), sort_keys=True).encode())
    for name, value in sorted(consts.items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            h.update(str(cpu.dtype).encode())
            h.update(json.dumps(list(cpu.shape)).encode())
            h.update(cpu.numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    return h.hexdigest()


def policy_record(policy):
    result = policy.description()
    if policy.weights is not None:
        result["weights"] = torch.from_numpy(policy.weights.copy())
        result["bias"] = torch.from_numpy(policy.bias.copy())
    return result


def policy_from_record(record):
    return Policy(int(record["kind"]), float(record["alpha"]), int(record["head_shift"]),
        record["weights"].numpy().copy() if "weights" in record else None,
        record["bias"].numpy().copy() if "bias" in record else None)


def save_state(path, consts, C, hd, TB, mode, policy, predictor_digest):
    state = {"format": "t_head_spread_state_v1", "base_commit": BASE_COMMIT,
        "consts": {key: value.detach().cpu().contiguous() if isinstance(value, torch.Tensor) else value
                   for key, value in consts.items()}, "C": C, "hd": hd, "TB": TB, "mode": mode,
        "policy": policy_record(policy), "predictor_sha256": predictor_digest,
        "policy_sha256": policy.digest()}
    torch.save(state, path)


def load_state(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "t_head_spread_state_v1" or state.get("base_commit") != BASE_COMMIT:
        raise ValueError("unsupported head-spread experiment state")
    policy = policy_from_record(state["policy"])
    if policy.digest() != state["policy_sha256"]:
        raise ValueError("policy checksum mismatch")
    digest = consts_digest(state["consts"], state["C"], state["hd"], state["TB"], state["mode"])
    if digest != state["predictor_sha256"]:
        raise ValueError("predictor checksum mismatch")
    state["consts"] = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                       for key, value in state["consts"].items()}
    state["loaded_policy"] = policy
    return state


def _launch_copy(source, destination, offsets, bases, lengths, pack):
    try:
        _copy_tails[(lengths.numel(),)](source, destination, offsets, bases, lengths,
            PACK=pack, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    except TypeError:
        _copy_tails[(lengths.numel(),)](source, destination, offsets, bases, lengths, PACK=pack, BLOCK=1024)


def pack(codec, arena, starts, tokens, predictor_digest, policy, words_digest, position=0):
    begin = time.perf_counter()
    segments, capacity = codec.layout(tokens)
    s = starts.detach().cpu().numpy().astype(np.int64)
    ends = (np.arange(segments, dtype=np.int64) + 1) * capacity
    bases = ((s - 2 * codec.hd) // 8) * 8
    lengths = ends - bases
    padding = s - bases - 2 * codec.hd
    if (np.any(bases < ends - capacity) or np.any(lengths >= (1 << 29))
            or np.any(padding < 0) or np.any(padding > 7)):
        raise ValueError("invalid used stream tail")
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    device = codec.device
    sizes_gpu, offsets_gpu, bases_gpu = [torch.from_numpy(x.copy()).to(device) for x in (lengths, offsets, bases)]
    compact = torch.empty(int(offsets[-1]), dtype=torch.int16, device=device)
    _launch_copy(arena, compact, offsets_gpu, bases_gpu, sizes_gpu, True)
    payload = compact.cpu().numpy().astype("<i2", copy=False).tobytes()
    index = ((lengths << 3) | padding).astype("<u4").tobytes()
    body = index + payload
    header = HEADER.pack(MAGIC, 1, codec.L, tokens, codec.W, codec.C, codec.hd, codec.TB, codec.ng,
        int(codec.mode == "exact"), segments, int(offsets[-1]), position,
        bytes.fromhex(predictor_digest), bytes.fromhex(policy.digest()), bytes.fromhex(words_digest),
        hashlib.sha256(body).digest())
    packet = header + body
    if len(body) * 8 != codec.bits(starts, tokens):
        raise AssertionError("packed wire disagrees with paid original stream size")
    return packet, {"packet_bytes": len(packet), "header_bytes": len(header), "index_bytes": len(index),
        "entropy_and_alignment_bytes": len(payload), "packing_and_D2H_seconds": time.perf_counter() - begin,
        "paid_token_hint_bytes": 0}


def inspect_packet(packet):
    if len(packet) < HEADER.size:
        raise ValueError("truncated head-spread header")
    values = HEADER.unpack_from(packet)
    magic, version = values[:2]
    L, T, W, C, hd, TB, groups, mode, segments, words, position = values[2:13]
    predictor, policy, expected, checksum = values[13:]
    if (magic != MAGIC or version != 1 or min(L, T, W, C, hd, TB, groups, segments, words) < 1
            or W != 2 * C or C % hd or hd & (hd - 1) or TB & (TB - 1) or T % TB
            or (W // hd) % groups or segments != (T // TB) * (W // hd)
            or mode not in (0, 1) or L * T * W > (1 << 31)):
        raise ValueError("invalid head-spread geometry")
    body = packet[HEADER.size:]
    if len(body) != 4 * segments + 2 * words or hashlib.sha256(body).digest() != checksum:
        raise ValueError("head-spread payload length or checksum mismatch")
    return dict(layers=L, tokens=T, width=W, C=C, hd=hd, TB=TB, groups=groups, mode="exact" if mode else "uniform",
        segments=segments, payload_words=words, position=position, predictor_sha256=predictor.hex(),
        policy_sha256=policy.hex(), words_sha256=expected.hex(), raw_bytes=2 * L * T * W)


def unpack(codec, packet, predictor_digest):
    info = inspect_packet(packet)
    if (info["predictor_sha256"] != predictor_digest or info["policy_sha256"] != codec.policy.digest()
            or (info["layers"], info["width"], info["C"], info["hd"], info["TB"], info["groups"], info["mode"])
            != (codec.L, codec.W, codec.C, codec.hd, codec.TB, codec.ng, codec.mode)):
        raise ValueError("packet requires different predictor, policy or geometry")
    segments, capacity = codec.layout(info["tokens"])
    if segments * capacity >= 2**31:
        raise ValueError("expanded stream arena exceeds the codec limit")
    desc = np.frombuffer(packet, dtype="<u4", count=segments, offset=HEADER.size).astype(np.int64)
    lengths, padding = desc >> 3, desc & 7
    if (np.any(lengths > capacity) or np.any(lengths < 2 * codec.hd + padding)
            or int(lengths.sum()) != info["payload_words"]):
        raise ValueError("invalid used stream directory")
    ends = (np.arange(segments, dtype=np.int64) + 1) * capacity
    bases = ends - lengths
    if np.any(bases % 8):
        raise ValueError("unaligned rANS state")
    starts = (bases + 2 * codec.hd + padding).astype(np.int32)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    host = np.frombuffer(packet, dtype="<i2", offset=HEADER.size + 4 * segments).copy()
    payload = torch.from_numpy(host).to(codec.device)
    lg, og, bg = [torch.from_numpy(x.copy()).to(codec.device) for x in (lengths, offsets, bases)]
    arena = torch.empty(segments * capacity, dtype=torch.int16, device=codec.device)
    _launch_copy(payload, arena, og, bg, lg, False)
    return arena, torch.from_numpy(starts).to(codec.device), info


def codec_for_state(state, tokens, position=0):
    constants = dict(state["consts"])
    if position + tokens > constants["cos"].shape[0]:
        raise ValueError("packet position exceeds frozen RoPE tables")
    constants["cos"] = constants["cos"][position:position + tokens]
    constants["sin"] = constants["sin"][position:position + tokens]
    return HeadSpreadCodec(constants, state["C"], state["hd"], state["loaded_policy"],
        tokens_per_program=state["TB"], mode=state["mode"])


def decode_packet(state_path, packet_path, device):
    """Independent-worker entry: state and packet are the only data inputs."""
    begin = time.perf_counter()
    state = load_state(state_path, device)
    packet = packet_path.read_bytes()
    info = inspect_packet(packet)
    codec = codec_for_state(state, info["tokens"], info["position"])
    arena, starts, info = unpack(codec, packet, state["predictor_sha256"])
    synchronize(device)
    setup = time.perf_counter() - begin
    begin = time.perf_counter()
    output = codec.decode(arena, starts, info["tokens"])
    synchronize(device)
    elapsed = time.perf_counter() - begin
    digest = hashlib.sha256(output.cpu().numpy().astype("<i2", copy=False).tobytes()).hexdigest()
    if digest != info["words_sha256"]:
        raise AssertionError("packet-only decoder changed original BF16 words")
    return {"bit_exact": True, "words_sha256": digest, "values": math.prod(output.shape),
        "device": str(device), "gpu_executed": torch.device(device).type == "cuda",
        "state_setup_and_unpack_seconds": setup, "decode_seconds_including_first_compilation": elapsed,
        "data_inputs": [str(state_path), str(packet_path)], "known_source_words": False}
