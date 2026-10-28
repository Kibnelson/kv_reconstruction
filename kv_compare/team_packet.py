"""Compact transport around the unchanged pinned one-step GPU codec.

The arithmetic lives in ``team_onestep`` and is copied byte for byte. This
adapter gathers real used segment suffixes, canonicalizes unread alignment
padding, frames them, and expands them into a receiver-owned scratch arena.
Only the compact frame is a stored/transmitted representation. The large arena,
decoded BF16 output and predictor tensors are reported separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import shutil
import tempfile
import time

import numpy as np
import torch

MAGIC = b"TOS1"
PREFIX = struct.Struct("<4sIQQ")
FORMAT = "team_onestep_packet_v1"
ARMS = ("dense_exact", "dense_uniform", "rank256_uniform")
SNAPSHOT = Path(__file__).with_name("team_onestep")
MANIFEST = SNAPSHOT / "source_manifest.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_identity():
    manifest = json.loads(MANIFEST.read_text())
    for name, wanted in manifest["files"].items():
        if sha_bytes((SNAPSHOT / name).read_bytes()) != wanted:
            raise ValueError("pinned one-step source changed: " + name)
    return dict(commit=manifest["commit"], manifest_sha256=sha_bytes(MANIFEST.read_bytes()))


def _numpy(tensor):
    return tensor.detach().contiguous().cpu().numpy()


def _state_digest(arrays, metadata):
    h = hashlib.sha256(canonical(metadata))
    for name, value in sorted(arrays.items()):
        a = np.asarray(value)
        h.update(canonical([name, a.dtype.str, list(a.shape)]))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def save_state(path, consts, rank=None, metadata=None):
    """Save fixed tensors without pickle; an existing directory is never replaced."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("state directory is not empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"const__" + k: _numpy(v) for k, v in consts.items()}
    rank_value = None
    if rank is not None:
        rank_value = int(rank["rank"])
        arrays.update({"rank__" + k: _numpy(v) for k, v in rank.items() if k != "rank"})
    meta = dict(metadata or {})
    meta.update(format="team_onestep_state_v1", vendor_source=source_identity(), rank=rank_value)
    identity = _state_digest(arrays, meta)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".tmp.", dir=path.parent))
    try:
        np.savez(temporary / "tensors.npz", **arrays)
        manifest = dict(metadata=meta, state_sha256=identity,
                        archive_sha256=sha_bytes((temporary / "tensors.npz").read_bytes()),
                        tensors={k: dict(shape=list(v.shape), dtype=v.dtype.str,
                                         bytes=v.nbytes) for k, v in sorted(arrays.items())})
        (temporary / "manifest.json").write_bytes(canonical(manifest))
        if path.exists():
            path.rmdir()  # Removes only an empty caller-provided directory.
        temporary.rename(path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def load_state(path, device="cpu"):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    meta = manifest["metadata"]
    if meta.get("format") != "team_onestep_state_v1" or meta.get("vendor_source") != source_identity():
        raise ValueError("one-step state source identity differs")
    archive = (path / "tensors.npz").read_bytes()
    if sha_bytes(archive) != manifest["archive_sha256"]:
        raise ValueError("one-step state archive checksum differs")
    with np.load(path / "tensors.npz", allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if set(arrays) != set(manifest["tensors"]) or _state_digest(arrays, meta) != manifest["state_sha256"]:
        raise ValueError("one-step fixed tensor identity differs")
    for k, a in arrays.items():
        if dict(shape=list(a.shape), dtype=a.dtype.str, bytes=a.nbytes) != manifest["tensors"][k]:
            raise ValueError("one-step tensor layout differs")
    consts = {k[7:]: torch.from_numpy(a).to(device) for k, a in arrays.items() if k.startswith("const__")}
    rank = {k[6:]: torch.from_numpy(a).to(device) for k, a in arrays.items() if k.startswith("rank__")}
    if meta["rank"] is None:
        if rank:
            raise ValueError("unexpected rank tensors")
        rank = None
    else:
        rank["rank"] = meta["rank"]
    return consts, rank, manifest


def _integer(value, name, minimum=0, maximum=(1 << 31) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid " + name)
    return value


def geometry(header):
    if not isinstance(header, dict):
        raise ValueError("one-step header must be an object")
    if header.get("format") != FORMAT or header.get("arm") not in ARMS:
        raise ValueError("unknown one-step packet format/arm")
    if header.get("start") != 0 or type(header.get("start")) is not int:
        raise ValueError("one-step snapshot supports start=0 only")
    shape = header.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("invalid one-step shape")
    L, T, W = [_integer(v, "shape", 1) for v in shape]
    hd = _integer(header.get("head_dim"), "head dimension", 2)
    C = _integer(header.get("key_columns"), "key width", 1)
    TB = _integer(header.get("tokens_per_program"), "token block", 1)
    chunks = _integer(header.get("chunks"), "chunks", 1)
    if hd != 128 or W != 2 * C or W % hd or C % hd or T % (TB * chunks):
        raise ValueError("unsupported one-step geometry")
    if header["arm"] == "rank256_uniform" and (T % (chunks * 128) or W % 64):
        raise ValueError("rank epilogue tiles do not divide token/feature geometry")
    S, cap = (T // TB) * (W // hd), hd * (2 * L * TB + 2) + 8
    if S * cap >= 1 << 31 or S * cap * 2 > (1 << 40):
        raise ValueError("one-step arena exceeds indexed range")
    if header.get("segments") != S or header.get("segment_capacity_words") != cap:
        raise ValueError("one-step segment geometry differs")
    for name in ("state_sha256", "source_manifest_sha256"):
        v = header.get(name)
        if not isinstance(v, str) or len(v) != 64:
            raise ValueError("invalid fixed identity")
        try:
            bytes.fromhex(v)
        except ValueError as exc:
            raise ValueError("invalid fixed identity") from exc
    return L, T, W, hd, S, cap


class TeamPacket:
    """Checked compact words and a CPU directory; GPU upload is explicit."""
    def __init__(self, header, payload, offsets, relative_starts):
        if offsets.dtype not in (torch.int32, torch.int64) or relative_starts.dtype not in (torch.int32, torch.int64):
            raise ValueError("compact directory must contain integers")
        if (relative_starts.numel() and
                bool(((relative_starts.to(torch.int64) < 0) | (relative_starts.to(torch.int64) >= 1 << 31)).any())):
            raise ValueError("compact relative starts overflow int32")
        self.header = dict(header)
        self.payload = payload
        self.offsets = offsets.detach().to("cpu", torch.int64).contiguous().clone()
        self.relative_starts = relative_starts.detach().to("cpu", torch.int32).contiguous().clone()
        self.validate()

    def validate(self):
        _L, _T, _W, hd, S, cap = geometry(self.header)
        if self.payload.dtype != torch.int16 or self.payload.ndim != 1 or not self.payload.is_contiguous():
            raise ValueError("invalid compact word tensor")
        if self.offsets.shape != (S + 1,) or self.relative_starts.shape != (S,):
            raise ValueError("invalid compact directory shape")
        o = self.offsets.numpy()
        r = self.relative_starts.numpy().astype(np.int64)
        # Compare adjacent entries before subtracting to reject signed wraparound.
        if o[0] != 0 or np.any(o < 0) or np.any(o[1:] <= o[:-1]) or o[-1] != self.payload.numel():
            raise ValueError("invalid compact offsets")
        lengths = o[1:] - o[:-1]
        if (np.any(lengths < 2 * hd) or np.any(lengths > cap) or np.any(lengths % 8) or
                np.any(r < 2 * hd) or np.any(r > 2 * hd + 7) or np.any(r > lengths)):
            raise ValueError("invalid compact segment bounds/alignment")
        return self

    @property
    def payload_bytes(self):
        return self.payload.numel() * 2

    @property
    def directory_bytes(self):
        return self.offsets.numel() * 8 + self.relative_starts.numel() * 4

    @property
    def packet_bytes(self):
        return PREFIX.size + len(canonical(self.header)) + self.directory_bytes + self.payload_bytes + 32

    def export(self):
        self.validate()
        head = canonical(self.header)
        directory = self.offsets.numpy().astype("<i8").tobytes() + self.relative_starts.numpy().astype("<i4").tobytes()
        payload = _numpy(self.payload).astype("<i2").tobytes()
        data = PREFIX.pack(MAGIC, len(head), len(directory), len(payload)) + head + directory + payload
        return data + hashlib.sha256(data).digest()

    @classmethod
    def import_bytes(cls, blob, device="cpu"):
        blob = bytes(blob)
        if len(blob) < PREFIX.size + 32:
            raise ValueError("truncated one-step packet")
        magic, nh, nd, npayload = PREFIX.unpack_from(blob)
        if magic != MAGIC or nh > (1 << 20) or npayload % 2 or PREFIX.size + nh + nd + npayload + 32 != len(blob):
            raise ValueError("invalid one-step frame lengths")
        if hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
            raise ValueError("one-step packet checksum differs")
        p = PREFIX.size
        header = json.loads(blob[p:p + nh])
        if canonical(header) != blob[p:p + nh]:
            raise ValueError("noncanonical one-step header")
        _L, _T, _W, _hd, S, _cap = geometry(header)
        if nd != 8 * (S + 1) + 4 * S:
            raise ValueError("one-step directory byte count differs")
        p += nh
        offsets = torch.from_numpy(np.frombuffer(blob[p:p + 8 * (S + 1)], dtype="<i8").copy())
        p += 8 * (S + 1)
        starts = torch.from_numpy(np.frombuffer(blob[p:p + 4 * S], dtype="<i4").copy())
        p += 4 * S
        payload = torch.from_numpy(np.frombuffer(blob[p:p + npayload], dtype="<i2").copy()).to(device)
        return cls(header, payload, offsets, starts)

    def to(self, device):
        return TeamPacket(self.header, self.payload.to(device), self.offsets, self.relative_starts)


from .team_onestep import gpu_coder as _gc
triton, tl = _gc.triton, _gc.tl


if triton is not None:
    @triton.jit
    def _suffix_kernel(arena, packed, offsets, relative_starts, starts, CAP: tl.constexpr,
                       HD: tl.constexpr, PACK: tl.constexpr, BLOCK: tl.constexpr):
        seg = tl.program_id(0)
        j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        begin = tl.load(offsets + seg)
        length = tl.load(offsets + seg + 1) - begin
        rel = tl.load(relative_starts + seg)
        base = (seg + 1) * CAP - length
        m = j < length
        if PACK:
            v = tl.load(arena + base + j, mask=m & ((j < 2 * HD) | (j >= rel)), other=0)
            # Native alignment gap is unread and not initialized by its encoder.
            # Canonical zeros avoid serializing allocator contents.
            v = tl.where((j >= 2 * HD) & (j < rel), 0, v)
            tl.store(packed + begin + j, v, mask=m)
        else:
            v = tl.load(packed + begin + j, mask=m, other=0)
            tl.store(arena + base + j, v, mask=m)
            if tl.program_id(1) == 0:
                tl.store(starts + seg, base + rel)


def compact_segments(arena, starts, *, segments, capacity_words, head_dim):
    """Gather native suffixes; preserves state/data bytes and canonicalizes padding."""
    if arena.ndim != 1 or arena.dtype != torch.int16 or arena.numel() != segments * capacity_words:
        raise ValueError("native arena layout differs")
    st = starts.detach().to("cpu", torch.int64)
    if st.shape != (segments,):
        raise ValueError("native starts shape differs")
    bases = torch.div(st - 2 * head_dim, 8, rounding_mode="floor") * 8
    begin = torch.arange(segments, dtype=torch.int64) * capacity_words
    ends = begin + capacity_words
    if bool(((bases < begin) | (st > ends) | (st < bases + 2 * head_dim)).any()):
        raise ValueError("native segment overflow")
    lengths = ends - bases
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)])
    relative = (st - bases).to(torch.int32)
    packed = torch.empty(int(offsets[-1]), dtype=torch.int16, device=arena.device)
    if arena.is_cuda:
        _suffix_kernel[(segments, triton.cdiv(int(lengths.max()), 1024))](
            arena, packed, offsets.to(arena.device), relative.to(arena.device), starts,
            CAP=capacity_words, HD=head_dim, PACK=True, BLOCK=1024, num_warps=4)
    else:
        for i in range(segments):
            b, e = int(offsets[i]), int(offsets[i + 1])
            packed[b:e].copy_(arena[int(bases[i]):int(ends[i])])
            packed[b + 2 * head_dim:b + int(relative[i])] = 0
    return packed, offsets, relative


def _tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


class TeamCodec:
    """Transport adapter; graph captures the unchanged native decoder only."""
    def __init__(self, state_path, device, arm):
        if arm not in ARMS:
            raise ValueError("unknown one-step arm")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        consts, rank, self.manifest = load_state(state_path, self.device)
        self.state_sha256 = self.manifest["state_sha256"]
        self.arm = arm
        from .team_onestep.layerwise import LayerwiseOneStep
        hd = int(consts["cos"].shape[1]) * 2
        C = int(consts["Aq"].shape[1]) // 2
        if hd != 128:
            raise ValueError("pinned one-step snapshot requires 128-channel heads")
        if arm == "rank256_uniform" and (rank is None or rank.get("rank") != 256):
            raise ValueError("rank-256 fixed state missing")
        used_rank = rank if arm == "rank256_uniform" else None
        self.native = LayerwiseOneStep(consts, C, hd, tokens_per_program=32, chunks=2,
            mode="exact" if arm == "dense_exact" else "uniform", search=True,
            rank=used_rank, requant="epilogue" if used_rank is not None else "kernel",
            rank_tiles=(128, 128, 64, 4))
        self.shared_bytes = _tensor_bytes(consts) + (_tensor_bytes(rank) if used_rank is not None else 0)
        self.coder_table_bytes = _tensor_bytes(self.native.tables)
        self.source = source_identity()
        self.shape = None
        self.graph = None
        self.graph_capture_count = 0
        self.graph_cold_wall_ms = None

    def header(self, shape):
        L, T, W = map(int, shape)
        S, cap = self.native.layout(T)
        result = dict(format=FORMAT, arm=self.arm, shape=[L, T, W], start=0,
            state_sha256=self.state_sha256, source_commit=self.source["commit"],
            source_manifest_sha256=self.source["manifest_sha256"], tokens_per_program=32,
            chunks=2, head_dim=self.native.hd, key_columns=self.native.C,
            segments=S, segment_capacity_words=cap, mode=self.native.mode,
            rank=256 if self.native.rank is not None else None, product="cublas",
            search=True, requant=self.native.requant, rank_tiles=[128, 128, 64, 4],
            alignment_padding="canonical_zero", probability_arithmetic="unchanged_source_snapshot")
        geometry(result)
        return result

    def validate(self, packet):
        packet.validate()
        shape = tuple(packet.header["shape"])
        if shape[0] != self.native.L or shape[2] != self.native.W or packet.header != self.header(shape):
            raise ValueError("packet fixed state, options or geometry differ")
        if packet.payload.device != self.device:
            raise ValueError("compact packet must be uploaded to decoder device first")
        return shape

    def model(self, words):
        self._check_words(words)
        return self.native.model(words)

    def _check_words(self, words):
        if words.dtype != torch.int16 or words.ndim != 3 or words.device != self.device:
            raise ValueError("expected original int16 BF16 words on encoder device")
        if words.shape[0] != self.native.L or words.shape[2] != self.native.W:
            raise ValueError("encoder shape differs from fitted state")
        self.header(tuple(words.shape))

    def encode(self, words, start=0):
        if type(start) is not int or start != 0:
            raise ValueError("one-step snapshot supports start=0 only")
        self._check_words(words)
        xall, prior = self.model(words)
        return self.encode_prepared(words, xall, prior)

    def encode_prepared(self, words, xall, prior):
        self._check_words(words)
        shape = tuple(words.shape)
        if shape[0] != self.native.L or shape[2] != self.native.W:
            raise ValueError("encoder shape differs from fitted state")
        head = self.header(shape)
        arena, starts = self.native.encode(words, xall, prior)
        payload, offsets, relative = compact_segments(arena, starts,
            segments=head["segments"], capacity_words=head["segment_capacity_words"], head_dim=self.native.hd)
        return TeamPacket(head, payload, offsets, relative)

    def prepare(self, packet):
        shape = self.validate(packet)
        if shape != self.shape:
            self.graph = None
            self.shape = shape
            _L, T, _W = shape
            S, cap = self.native.layout(T)
            self.arena = torch.empty(S * cap, dtype=torch.int16, device=self.device)
            self.starts = torch.empty(S, dtype=torch.int32, device=self.device)
            self.output = torch.empty(shape, dtype=torch.int16, device=self.device)
            self.buffers = self.native.buffers(T)
            self.offsets_device = torch.empty(S + 1, dtype=torch.int64, device=self.device)
            self.relative_device = torch.empty(S, dtype=torch.int32, device=self.device)
        return self

    def stage(self, packet):
        self.prepare(packet)
        _L, _T, _W, hd, S, cap = geometry(packet.header)
        # Pay a real compact GPU staging copy; callers retain ownership of packets.
        if not hasattr(self, "packed") or self.packed.numel() < packet.payload.numel():
            self.packed = torch.empty_like(packet.payload)
        self.packed[:packet.payload.numel()].copy_(packet.payload)
        self.offsets_device.copy_(packet.offsets)
        self.relative_device.copy_(packet.relative_starts)
        lengths = packet.offsets[1:] - packet.offsets[:-1]
        if self.device.type == "cuda":
            _suffix_kernel[(S, triton.cdiv(int(lengths.max()), 1024))](
                self.arena, self.packed, self.offsets_device, self.relative_device, self.starts,
                CAP=cap, HD=hd, PACK=False, BLOCK=1024, num_warps=4)
        else:
            for i in range(S):
                b, e = int(packet.offsets[i]), int(packet.offsets[i + 1])
                dst = (i + 1) * cap - (e - b)
                self.arena[dst:(i + 1) * cap].copy_(self.packed[b:e])
                self.starts[i] = dst + packet.relative_starts[i]
        return self

    def native_decode(self):
        return self.native.decode(self.arena, self.starts, self.shape[1], out=self.output, bufs=self.buffers)

    def kernel_decode(self, packet):
        self.stage(packet)
        return self.native_decode()

    def decode(self, packet):
        self.stage(packet)
        return self.native_decode().clone()

    def graph_prepare(self, packet, warmup=3):
        if self.device.type != "cuda":
            raise RuntimeError("actual CUDA is required for one-step graph replay")
        self.stage(packet)
        if self.graph is not None:
            return self
        start = time.perf_counter()
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self.native_decode()
        torch.cuda.current_stream(self.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.native_decode()
        torch.cuda.synchronize(self.device)
        self.graph_cold_wall_ms = 1000 * (time.perf_counter() - start)
        self.graph_capture_count += 1
        return self

    def graph_replay_native(self):
        if self.graph is None:
            raise RuntimeError("prepare graph outside kernel-only timing first")
        self.graph.replay()
        return self.output

    def graph_decode(self, packet):
        shape = self.validate(packet)
        if shape != self.shape or self.graph is None:
            self.graph_prepare(packet)
        self.stage(packet)
        return self.graph_replay_native().clone()

    def accounting(self, packet):
        packet.validate()
        _L, _T, _W, _hd, S, cap = geometry(packet.header)
        return dict(native_used_stream_bytes=packet.payload_bytes + 4 * S,
            compact_payload_bytes=packet.payload_bytes, directory_bytes=packet.directory_bytes,
            header_checksum_bytes=packet.packet_bytes - packet.payload_bytes - packet.directory_bytes,
            packet_bytes=packet.packet_bytes, arena_bytes=S * cap * 2,
            output_bytes=int(np.prod(packet.header["shape"])) * 2,
            shared_predictor_bytes=self.shared_bytes, coder_table_bytes=self.coder_table_bytes,
            decoder_workspace_tensor_bytes=_tensor_bytes(getattr(self, "buffers", {})),
            compact_staging_bytes=_tensor_bytes(getattr(self, "packed", None)),
            directory_staging_bytes=_tensor_bytes(getattr(self, "offsets_device", None)) +
                _tensor_bytes(getattr(self, "relative_device", None)))
