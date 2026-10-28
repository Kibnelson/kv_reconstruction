"""Small matched transport controls; the pinned SplitZip codec is imported unchanged.

Common framing is local benchmark glue, not the author's serving evaluation.
The author's own throughput script is run separately by bench_llama_baselines.
"""
from __future__ import annotations
import hashlib
import importlib.metadata
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from .causal_packet import Packet, json_bytes, sha

SPLITZIP_COMMIT = "42c743118a29db3270ae6a921a4a4eecb20cad04"
SPLITZIP_FILES = {
    "codec_gpu.py": "e28fe3263e8906b26d59789cbc5bccf14edc8391b0dae986aafaf34adb2d392c",
    "bench_codec_throughput.py": "7f1936a13ee3b2bbb04a35b71ca14a88bc407fe9b67e2c128d2ec48df2d018e2",
    "requirements.txt": "73832e401d2183eb74671f23f74e8b3eef5fb99669fb28b98ea2da4099a30fb4",
}
SPLITZIP_PACKAGES = dict(torch="2.9.0", triton="3.5.0", transformers="4.57.1",
                         datasets="2.19.2", accelerate="1.13.0")
LUTS = dict(enc_lut=256, enc_lut_marked=256, dec_lut=16, common_lut=256)
DTYPES = {"uint8": torch.uint8, "uint16": torch.uint16,
          "int16": torch.int16, "int32": torch.int32}


def checked_splitzip(source, *, software_check=False):
    source = Path(source).resolve()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != SPLITZIP_COMMIT:
        raise ValueError("SplitZip checkout does not match the pinned official commit")
    for name, digest in SPLITZIP_FILES.items():
        if sha(source / name) != digest:
            raise ValueError("official SplitZip source changed: " + name)
    versions = {}
    for p in SPLITZIP_PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            if not software_check:
                raise
            versions[p] = "missing in software-check environment"
    if not software_check and any(versions[p].split("+")[0] != v for p, v in SPLITZIP_PACKAGES.items()):
        raise ValueError("use the official pinned SplitZip environment: " + str(versions))
    return dict(commit=head, files=SPLITZIP_FILES, packages=versions,
                source=str(source), software_check=software_check)


def load_splitzip(source):
    name = "matched_official_splitzip"
    spec = importlib.util.spec_from_file_location(name, Path(source) / "codec_gpu.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frame(method, shape, state_sha256, parts, extra=None):
    """Align all fields to four bytes; charge padding, directory and full envelope."""
    descriptors, offsets, total = [], [], 0
    for name, tensor in parts:
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError("flat contiguous transport fields required")
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in DTYPES:
            raise ValueError("unsupported transport field")
        offsets.append(total)
        size = tensor.numel() * tensor.element_size()
        descriptors.append(dict(name=name, dtype=dtype, count=tensor.numel(), bytes=size))
        total += (size + 3) // 4 * 4
    device = parts[0][1].device
    payload = torch.zeros(total, dtype=torch.uint8, device=device)
    for offset, (_, tensor) in zip(offsets, parts):
        raw = tensor.view(torch.uint8)
        payload[offset:offset + raw.numel()].copy_(raw)
    header = dict(format="matched_baseline_v1", method=method, shape=list(shape),
                  state_sha256=state_sha256, fields=descriptors, extra=extra or {})
    return Packet(header, torch.tensor(offsets + [total], dtype=torch.int64, device=device),
                  payload.view(torch.int16))


def unframe(packet, method, state_sha256):
    h = packet.header
    if (h.get("format") != "matched_baseline_v1" or h.get("method") != method or
            h.get("state_sha256") != state_sha256):
        raise ValueError("baseline method/state identity mismatch")
    shape = h["shape"]
    if not shape or len(shape) > 4 or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError("invalid baseline geometry")
    if packet.starts.dtype != torch.int64 or packet.payload.dtype != torch.int16:
        raise ValueError("invalid transport types")
    starts = packet.starts.cpu().tolist()
    body = packet.payload.view(torch.uint8)
    fields = h["fields"]
    if len(starts) != len(fields) + 1 or starts[0] != 0 or starts[-1] != body.numel():
        raise ValueError("invalid baseline directory")
    parts = {}
    for i, f in enumerate(fields):
        if f["dtype"] not in DTYPES or f["name"] in parts or type(f["count"]) is not int or f["count"] < 0:
            raise ValueError("invalid baseline fields")
        size = f["count"] * torch.empty((), dtype=DTYPES[f["dtype"]]).element_size()
        if (f["bytes"] != size or starts[i] % 4 or starts[i+1] - starts[i] != (size+3)//4*4):
            raise ValueError("invalid field length/alignment")
        parts[f["name"]] = body[starts[i]:starts[i]+size].view(DTYPES[f["dtype"]])
    return shape, parts, h["extra"]


class Baseline:
    def __init__(self, method, device, *, source=None, state=None):
        if method not in ("raw_bf16", "fp8_e4m3fn_saturating", "splitzip_official_framed"):
            raise ValueError("unknown baseline")
        self.method, self.device = method, torch.device(device)
        self.shared_bytes = 0
        self.state_sha256 = hashlib.sha256(json_bytes(dict(method=method, version=1))).hexdigest()
        if method == "splitzip_official_framed":
            self.module = load_splitzip(source)
            self.codec = self.module.ChunkLocalSplitZipGPU(device=str(self.device), chunk_size=1024)
            if state is not None:
                with np.load(state, allow_pickle=False) as arrays:
                    if set(arrays.files) != set(LUTS):
                        raise ValueError("unexpected SplitZip state fields")
                    for name, n in LUTS.items():
                        a = arrays[name]
                        if a.dtype != np.uint8 or a.shape != (n,):
                            raise ValueError("invalid SplitZip LUT")
                        setattr(self.codec, name, torch.from_numpy(a.copy()).to(self.device))
                self.state_sha256 = sha(state)
                self.shared_bytes = sum(LUTS.values())

    def calibrate(self, words, path):
        self.codec.calibrate(words.contiguous().view(torch.bfloat16).flatten())
        np.savez(path, **{name: getattr(self.codec, name).cpu().numpy() for name in LUTS})
        self.state_sha256 = sha(path)
        self.shared_bytes = sum(LUTS.values())

    def encode(self, words):
        if words.dtype != torch.int16 or not words.is_contiguous():
            raise ValueError("original contiguous BF16 word view required")
        shape = words.shape
        if self.method == "raw_bf16":
            return frame(self.method, shape, self.state_sha256, [("words", words.flatten())])
        if self.method == "fp8_e4m3fn_saturating":
            # Explicit unscaled cast reference, including saturation in the timed operation.
            values = words.view(torch.bfloat16).float().clamp(-448., 448.)
            codes = values.to(torch.float8_e4m3fn).view(torch.uint8).flatten()
            return frame(self.method, shape, self.state_sha256, [("codes", codes)])
        encoded = self.codec.encode(words.view(torch.bfloat16).flatten())
        return frame(self.method, shape, self.state_sha256,
            [(name, getattr(encoded, name)) for name in ("pk", "sm", "chunk_id", "local_pos", "esc_val")],
            dict(n=encoded.n, n_esc=encoded.n_esc, chunk_size=encoded.chunk_size,
                 native_payload_bytes=encoded.compressed_bytes))

    def decode(self, packet):
        shape, parts, extra = unframe(packet, self.method, self.state_sha256)
        n = math.prod(shape)
        if self.method == "raw_bf16":
            if set(parts) != {"words"} or parts["words"].dtype != torch.int16 or parts["words"].numel() != n:
                raise ValueError("invalid raw payload")
            return parts["words"].clone().reshape(shape)
        if self.method == "fp8_e4m3fn_saturating":
            if set(parts) != {"codes"} or parts["codes"].dtype != torch.uint8 or parts["codes"].numel() != n:
                raise ValueError("invalid FP8 payload")
            return parts["codes"].view(torch.float8_e4m3fn).to(torch.bfloat16).view(torch.int16).reshape(shape)
        ne = extra.get("n_esc")
        if (extra.get("n") != n or extra.get("chunk_size") != 1024 or
                type(ne) is not int or not 0 <= ne <= n):
            raise ValueError("invalid SplitZip geometry")
        expected = dict(pk=(torch.uint8, (n+1)//2), sm=(torch.uint8, n),
            chunk_id=(torch.int32, ne), local_pos=(torch.uint16, ne), esc_val=(torch.uint8, ne))
        if set(parts) != set(expected) or any(parts[k].dtype != dt or parts[k].numel() != size
                                             for k, (dt, size) in expected.items()):
            raise ValueError("invalid SplitZip fields")
        if ne:
            chunk, pos = parts["chunk_id"].long(), parts["local_pos"].long()
            linear = chunk * 1024 + pos
            if bool(((chunk < 0) | (pos >= 1024) | (linear >= n)).any()):
                raise ValueError("invalid SplitZip escape locations")
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        encoded = self.module.ChunkLocalGPUEncoded(parts["pk"], parts["sm"], empty, empty,
            parts["chunk_id"], parts["local_pos"], parts["esc_val"], n, ne, 1024)
        if encoded.compressed_bytes != extra["native_payload_bytes"]:
            raise ValueError("SplitZip byte accounting differs")
        return self.codec.decode(encoded).view(torch.int16).reshape(shape)
