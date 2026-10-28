"""Same-packet decoder choices: CDF lookup and borrowed-input eager decode.

The packet's ``search=True`` field records the original encoder configuration.
Both inverse algorithms below use the same fixed CDFs; choosing their inverse
does not change those probabilities, the predictor or any transmitted byte.

Graph mode preserves the checked compact staging/owned-output API. Direct mode
reads the caller's compact payload and allocates the returned output directly,
avoiding the bulk staging copy and the final output clone. Calls remain
sequential; the native decoder's internal chunk streams are unchanged. Public
calls may switch caller streams: the new stream waits for the previous caller
before overwriting shared decoder workspace.
"""
from __future__ import annotations

import functools
import hashlib
import json

import numpy as np
import torch

from .team_compact import TeamCompactCodec, _compact_starts_kernel
from .team_packet import _tensor_bytes, geometry, triton
from .team_onestep import gpu_coder as gc


def _table_receipts(tables):
    records = {}
    for name, tensor in sorted(tables.items()):
        a = tensor.detach().cpu().numpy().astype("<i4", copy=False)
        records[name] = dict(dtype="int32", shape=list(a.shape),
                             sha256=hashlib.sha256(a.tobytes()).hexdigest())
    return records


@functools.lru_cache(maxsize=1)
def _proof_json():
    """Prove the alternate inverse for every slot of every fixed CDF class."""
    tables = gc.slice_tables("cpu")
    slots = np.arange(gc.M, dtype=np.int64)
    expected_digest = hashlib.sha256()
    lookup_digest = hashlib.sha256()
    for c in range(gc.CLASSES):
        off = int(tables["off"][c])
        count = int(tables["K"][c]) + 2
        cum = tables["cum"][off:off + count].numpy().astype(np.int64)
        expected = np.searchsorted(cum, slots, side="right") - 1
        actual = gc.stage1_reference(tables, c, slots)
        if not np.array_equal(actual, expected):
            bad = int(np.flatnonzero(actual != expected)[0])
            raise AssertionError(f"lookup CDF inverse differs at class {c}, slot {bad}")
        expected_digest.update(expected.astype("<u2").tobytes())
        lookup_digest.update(actual.astype("<u2").tobytes())
    result = dict(verified=True, classes=gc.CLASSES, slots_per_class=gc.M,
        checked_slots=gc.CLASSES * gc.M, table_sha256=_table_receipts(tables),
        reference_sha256=expected_digest.hexdigest(), lookup_sha256=lookup_digest.hexdigest(),
        scope="exhaustive fixed stage-1 symbol inverse only; native word replay is checked separately")
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def cdf_inverse_proof():
    """Return an independently owned copy of the cached exhaustive CPU proof."""
    return json.loads(_proof_json())


class TeamLookupCodec(TeamCompactCodec):
    """Fixed one-step packets with independently chosen inverse and execution.

    ``decoder_mode='graph'`` stages compact data and clones the graph output.
    ``decoder_mode='direct'`` borrows the input and returns a newly allocated
    output. Borrowed storage must be 16-byte aligned, as canonical imported
    tensors are; unaligned tensor views are rejected rather than copied. Both
    modes validate the actual incoming packet on every public call.
    """

    def __init__(self, state_path, device, arm, *, decoder_mode="graph", lookup=False):
        if decoder_mode not in ("graph", "direct") or type(lookup) is not bool:
            raise ValueError("decoder mode must be graph/direct and lookup must be bool")
        super().__init__(state_path, device, arm)
        self.decoder_mode = decoder_mode
        self.lookup = lookup
        proof = cdf_inverse_proof()
        if _table_receipts(self.native.tables) != proof["table_sha256"]:
            raise AssertionError("native decoder CDF tensors differ from exhaustive proof")
        self.inverse_proof = dict(proof, native_tables_verified=True)
        self.native.search = not lookup
        self._workspace_last_stream = (torch.cuda.current_stream(self.device)
                                       if self.device.type == "cuda" else None)

    def validate(self, packet):
        shape = super().validate(packet)
        if self.decoder_mode == "direct" and packet.payload.data_ptr() % 16:
            raise ValueError("borrowed compact payload must have 16-byte aligned storage")
        return shape

    def prepare(self, packet):
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            previous = self._workspace_last_stream
            if previous is not None and previous.cuda_stream != current.cuda_stream:
                # The preceding public call joined native chunk work onto its
                # caller stream. Order all shared storage before a new caller
                # can overwrite directories, staging or decoder workspace.
                current.wait_stream(previous)
            self._workspace_last_stream = current
        if self.decoder_mode == "graph":
            return super().prepare(packet)
        shape = self.validate(packet)
        if shape != self.shape:
            if self.shape is not None and self.device.type == "cuda":
                # Only a cold geometry transition may release old workspaces.
                torch.cuda.synchronize(self.device)
            self.shape = shape
            _L, T, _W = shape
            S, _cap = self.native.layout(T)
            self.starts = torch.empty(S, dtype=torch.int32, device=self.device)
            self.offsets_device = torch.empty(S + 1, dtype=torch.int64, device=self.device)
            self.relative_device = torch.empty(S, dtype=torch.int32, device=self.device)
            self.buffers = self.native.buffers(T)
        return self

    def reserve_payload_words(self, words):
        if self.decoder_mode != "graph":
            raise RuntimeError("direct decoding has no payload staging reservation")
        return super().reserve_payload_words(words)

    def stage(self, packet):
        if self.decoder_mode == "graph":
            return super().stage(packet)
        self.prepare(packet)
        _L, _T, _W, _hd, S, _cap = geometry(packet.header)
        self.offsets_device.copy_(packet.offsets)
        self.relative_device.copy_(packet.relative_starts)
        if self.device.type == "cuda":
            _compact_starts_kernel[(triton.cdiv(S, 256),)](
                self.offsets_device, self.relative_device, self.starts,
                S=S, BLOCK=256, num_warps=4)
        else:
            self.starts.copy_((self.offsets_device[:-1] + self.relative_device).to(torch.int32))
        return self

    def graph_prepare(self, packet, warmup=3):
        if self.decoder_mode != "graph":
            raise RuntimeError("direct decoding does not capture a graph")
        return super().graph_prepare(packet, warmup=warmup)

    def native_decode(self):
        if self.decoder_mode != "graph":
            raise RuntimeError("direct decoding requires the actual packet argument")
        return super().native_decode()

    def graph_replay_native(self):
        if self.decoder_mode != "graph":
            raise RuntimeError("direct decoding has no native replay graph")
        return super().graph_replay_native()

    def graph_decode(self, packet):
        if self.decoder_mode != "graph":
            raise RuntimeError("use decode(packet) for direct decoding")
        return super().graph_decode(packet)

    def decode(self, packet):
        if self.decoder_mode == "graph":
            return self.graph_decode(packet)
        self.stage(packet)
        output = torch.empty(self.shape, dtype=torch.int16, device=self.device)
        self.native.decode(packet.payload, self.starts, self.shape[1],
                           out=output, bufs=self.buffers)
        if self.device.type == "cuda":
            # Native decode joins its chunk streams onto the caller's current
            # stream. Also register borrowed/returned storage with those streams
            # so dropping either owner cannot allow premature allocator reuse.
            streams = (self.buffers.get("streams") or []) + (self.buffers.get("gemm_streams") or [])
            for stream in streams:
                packet.payload.record_stream(stream)
                output.record_stream(stream)
        return output

    def accounting(self, packet):
        value = super().accounting(packet)
        size = value["output_bytes"]
        direct = self.decoder_mode == "direct"
        value.update(decoder_mode=self.decoder_mode, lookup=self.lookup,
            inverse_algorithm="central_tail_lookup" if self.lookup else "coarse_binary_search",
            borrowed_payload=direct, persistent_output_bytes=0 if direct else size,
            output_allocation_bytes_per_call=size,
            output_clone_bytes_per_call=0 if direct else size,
            compact_staging_bytes=0 if direct else value["compact_staging_bytes"],
            native_starts_bytes=_tensor_bytes(getattr(self, "starts", None)),
            decode_storage=("borrowed compact input; fresh owned output" if direct else
                            "stable compact staging; cloned graph output"))
        return value
