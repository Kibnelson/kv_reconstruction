"""Checked compact transfer carrier for the unchanged team's mixed decoder.

This is our integration arm, not an official SplitZip serving implementation.
The all-SplitZip case uses the existing whole-request baseline verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from .causal_packet import json_bytes, sha
from .framed_baselines import Baseline, SPLITZIP_COMMIT, SPLITZIP_FILES, frame, unframe
from .team_packet import compact_segments, load_state, source_identity, _tensor_bytes

METHOD = 'team_mixed_transfer_v1'
EXTENSION = Path(__file__).with_name('team_extensions')
SZ_FIELDS = ('pk', 'sm', 'chunk_id', 'local_pos', 'esc_val')


def extension_identity():
    manifest = json.loads((EXTENSION / 'source_manifest.json').read_text())
    if (manifest.get('format') != 'team_mixed_extension_source_v1'
        or manifest.get('commit') != '8a40dd98282ed364a8fe1077d171a1922277c1ec'
        or set(manifest.get('files', {})) != {'__init__.py', 'layerwise.py', 'mixed.py'}):
        raise ValueError('unrecognized team extension source manifest')
    for name, expected in manifest['files'].items():
        if sha(EXTENSION / name) != expected:
            raise ValueError('team extension source changed: ' + name)
    core = source_identity()
    if core['manifest_sha256'] != manifest['core_manifest_sha256']:
        raise ValueError('extension does not bind the frozen core')
    return dict(commit=manifest['commit'], manifest_sha256=sha(EXTENSION / 'source_manifest.json'), core=core)


def verify_splitzip_source(source):
    source = Path(source)
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if head != SPLITZIP_COMMIT or dirty or any(sha(source / n) != h for n, h in SPLITZIP_FILES.items()):
        raise ValueError('pinned SplitZip author source changed')
    return dict(commit=head, files=dict(SPLITZIP_FILES))


def packet_bytes(packet):
    return 24 + len(json_bytes(packet.header)) + 8 * packet.starts.numel() + 2 * packet.payload.numel() + 32


@dataclass(frozen=True)
class Parsed:
    shape: tuple
    native: dict | None
    packet: object
    owner: object


class MixedTransfer:
    def __init__(self, state_path, splitzip_source, splitzip_state, splitzip_state_sha256,
                 mask, device, arm='rank256_uniform'):
        if not isinstance(mask, (list, tuple)) or not mask or any(type(x) is not bool for x in mask):
            raise ValueError('one explicit boolean per layer is required')
        if arm not in ('dense_uniform', 'rank256_uniform'):
            raise ValueError('mixed transfer supports fixed dense/rank256 uniform models')
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.mask, self.L, self.arm = tuple(mask), len(mask), arm
        self.extension = extension_identity()
        self.splitzip_source = verify_splitzip_source(splitzip_source)
        if sha(splitzip_state) != splitzip_state_sha256:
            raise ValueError('frozen TRAIN SplitZip codebook changed')
        self.other = Baseline('splitzip_official_framed', self.device, source=splitzip_source, state=splitzip_state)
        self.native = self.consts = self.rank = None
        self.W = self.C = self.hd = None
        self.shape = None; self.buffers = None
        self._last_stream = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        self.graph_capture_count = 0
        self.shared_predictor_bytes = self.coder_table_bytes = 0
        self.shared_lut_bytes = self.other.shared_bytes
        self.predictor_state_sha256 = None
        if not any(self.mask):
            self.state_sha256 = self.other.state_sha256
            self.shared_bytes = self.shared_lut_bytes
            return
        self.consts, rank, manifest = load_state(state_path, self.device)
        self.L, self.W = map(int, self.consts['Aq'].shape[:2])
        self.hd = int(self.consts['cos'].shape[1]) * 2; self.C = self.W // 2
        if len(self.mask) != self.L or self.hd != 128 or self.W % (2 * self.hd):
            raise ValueError('mask or original head geometry differs from fixed state')
        if arm == 'rank256_uniform' and (rank is None or rank.get('rank') != 256):
            raise ValueError('fixed rank256 state missing')
        self.rank = rank if arm == 'rank256_uniform' else None
        from .team_extensions.mixed import MixedOneStep
        self.native = MixedOneStep(self.consts, self.C, self.hd, self.mask, self.other.codec,
            tokens_per_program=32, chunks=2, mode='uniform', search=True, rank=self.rank,
            requant='epilogue' if self.rank is not None else 'kernel', rank_tiles=(128, 128, 64, 4))
        self.predictor_state_sha256 = manifest['state_sha256']
        self.shared_predictor_bytes = _tensor_bytes(self.consts) + _tensor_bytes(self.rank)
        self.coder_table_bytes = _tensor_bytes(self.native.tables)
        self.shared_bytes = self.shared_predictor_bytes + self.shared_lut_bytes
        identity = dict(format=METHOD, arm=arm, mask=list(self.mask), predictor=self.predictor_state_sha256,
            splitzip=self.other.state_sha256, extension=self.extension, tokens_per_program=32, chunks=2,
            mode='uniform', search=True, rank_tiles=[128,128,64,4])
        self.state_sha256 = hashlib.sha256(json_bytes(identity)).hexdigest()

    def _shape(self, shape):
        if (len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape) or shape[0] != self.L):
            raise ValueError('invalid mixed KV shape')
        L, T, W = shape
        if self.native is not None and (W != self.W or T % 64 or T > self.consts['cos'].shape[0]
            or (self.rank is not None and T % 256)):
            raise ValueError('unsupported fixed predictor/token geometry')
        if self.native is not None:
            segments, capacity = self.native.layout(T)
            if segments * capacity >= 1 << 31:
                raise ValueError('native encoder arena exceeds INT32 addressing')
        return tuple(shape)

    def _extra(self, shape, sz):
        S, cap = self.native.layout(shape[1])
        return dict(version=1, arm=self.arm, mask=list(self.mask), start=0, tokens_per_program=32, chunks=2,
            head_dim=self.hd, key_columns=self.C, segments=S, capacity_words=cap,
            predictor_state_sha256=self.predictor_state_sha256, splitzip_state_sha256=self.other.state_sha256,
            extension_manifest_sha256=self.extension['manifest_sha256'], splitzip_layers=sz)

    def encode(self, words, start=0):
        if type(start) is not int or start != 0 or words.dtype != torch.int16 or words.device != self.device or not words.is_contiguous():
            raise ValueError('original contiguous BF16 words at start zero are required')
        shape = self._shape(list(words.shape))
        if self.native is None:
            return self.other.encode(words)
        enc = self.native.encode(words)
        S, cap = self.native.layout(shape[1])
        packed, offsets, relative = compact_segments(enc['stream'], enc['starts'],
            segments=S, capacity_words=cap, head_dim=self.hd)
        if packed.numel() >= 1 << 31:
            raise ValueError('compact stream exceeds INT32 word addressing')
        parts = [('rans_words', packed), ('rans_offsets', offsets.to(self.device, torch.int32)),
                 ('rans_relative', relative.to(self.device, torch.int32))]
        metadata = []
        for layer, e in sorted(enc['sz'].items()):
            metadata.append(dict(layer=layer, n=e.n, n_esc=e.n_esc, chunk_size=e.chunk_size,
                                 native_payload_bytes=e.compressed_bytes))
            parts += [(f'l{layer}_{name}', getattr(e, name).flatten().contiguous()) for name in SZ_FIELDS]
        return frame(METHOD, shape, self.state_sha256, parts, self._extra(shape, metadata))

    def unpack(self, packet):
        """Validate actual packet metadata before native GPU kernels are entered."""
        if packet.payload.device != self.device:
            raise ValueError('incoming packet is not on the decoder device')
        if self.native is None:
            shape = self._shape(packet.header.get('shape', []))
            # Baseline.decode performs its complete checks when invoked below.
            unframe(packet, 'splitzip_official_framed', self.other.state_sha256)
            return Parsed(shape, None, packet, self)
        shape, parts, extra = unframe(packet, METHOD, self.state_sha256)
        shape = self._shape(shape); S, cap = self.native.layout(shape[1]); n = shape[1] * shape[2]
        szmeta = extra.get('splitzip_layers')
        if not isinstance(szmeta, list) or extra != self._extra(shape, szmeta):
            raise ValueError('mixed codec identity/geometry differs')
        expected_layers = [l for l, ours in enumerate(self.mask) if not ours]
        if len(szmeta) != len(expected_layers) or [e.get('layer') for e in szmeta] != expected_layers:
            raise ValueError('SplitZip layer directory differs from mask')
        expected_names = ['rans_words', 'rans_offsets', 'rans_relative']
        expected_names += [f'l{l}_{name}' for l in expected_layers for name in SZ_FIELDS]
        if list(parts) != expected_names:
            raise ValueError('mixed payload fields/order differ')
        payload, offsets, relative = [parts[n] for n in expected_names[:3]]
        if (payload.dtype != torch.int16 or payload.numel() >= 1 << 31 or payload.data_ptr() % 16
            or offsets.dtype != torch.int32 or offsets.numel() != S+1 or relative.dtype != torch.int32 or relative.numel() != S):
            raise ValueError('invalid compact rANS tensors/alignment')
        directory = torch.cat((offsets.long(), relative.long())).cpu().numpy()
        o, r = directory[:S+1], directory[S+1:]
        if (o[0] != 0 or o[-1] != payload.numel() or np.any(o < 0) or np.any(o[1:] <= o[:-1])):
            raise ValueError('invalid compact rANS offsets')
        lengths = o[1:] - o[:-1]
        if (np.any(lengths < 2*self.hd) or np.any(lengths > cap) or np.any(lengths % 8)
            or np.any(r < 2*self.hd) or np.any(r > 2*self.hd+7) or np.any(r > lengths)):
            raise ValueError('invalid compact rANS segment bounds')
        sz = {}; invalid = []
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        for m in szmeta:
            layer = m['layer']; ne = m.get('n_esc')
            if (type(ne) is not int or not 0 <= ne <= n or m != dict(layer=layer,n=n,n_esc=ne,chunk_size=1024,
                native_payload_bytes=(n+1)//2+n+7*ne)):
                raise ValueError('invalid native SplitZip geometry/byte accounting')
            fields = {name:parts[f'l{layer}_{name}'] for name in SZ_FIELDS}
            sizes = dict(pk=(torch.uint8,(n+1)//2),sm=(torch.uint8,n),chunk_id=(torch.int32,ne),
                         local_pos=(torch.uint16,ne),esc_val=(torch.uint8,ne))
            if any(fields[k].dtype != dt or fields[k].numel() != count for k,(dt,count) in sizes.items()):
                raise ValueError('invalid native SplitZip field types/lengths')
            if ne:
                cid, pos = fields['chunk_id'].long(), fields['local_pos'].long()
                invalid.append(((cid<0)|(pos>=1024)|(cid*1024+pos>=n)).any())
            sz[layer] = self.other.module.ChunkLocalGPUEncoded(fields['pk'],fields['sm'],empty,empty,
                fields['chunk_id'],fields['local_pos'],fields['esc_val'],n,ne,1024)
        if invalid and bool(torch.stack(invalid).any()):
            raise ValueError('invalid native SplitZip escape locations')
        enc = dict(stream=payload, starts=offsets[:-1]+relative, sz=sz)
        return Parsed(shape, enc, packet, self)

    def decode_native(self, parsed):
        """Diagnostic native call after unpack; no predictions/oracle inputs accepted."""
        if not isinstance(parsed, Parsed) or parsed.owner is not self:
            raise ValueError('native decode requires this codec\'s checked packet')
        if self.native is None:
            return self.other.decode(parsed.packet)
        if self.device.type == 'cuda':
            current = torch.cuda.current_stream(self.device)
            if self._last_stream is not None and current.cuda_stream != self._last_stream.cuda_stream:
                current.wait_stream(self._last_stream)
            self._last_stream = current
        if self.shape != parsed.shape:
            if self.shape is not None and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.buffers = self.native.buffers(parsed.shape[1]); self.shape = parsed.shape
        output = self.native.decode_mixed(parsed.native, parsed.shape[1], bufs=self.buffers)
        if self.device.type == 'cuda':
            for stream in (self.buffers.get('streams') or []) + (self.buffers.get('gemm_streams') or []):
                parsed.packet.payload.record_stream(stream); output.record_stream(stream)
                parsed.native['starts'].record_stream(stream)
        return output

    def decode(self, packet):
        return self.decode_native(self.unpack(packet))

    def accounting(self, packet):
        parsed = self.unpack(packet); n = math.prod(parsed.shape)
        sz = packet.header['extra'].get('splitzip_layers', []) if self.native is not None else []
        rans_layers = [l for l, ours in enumerate(self.mask) if ours]
        fallback_layers = [l for l, ours in enumerate(self.mask) if not ours]
        predecessor_fallback = [l for l in rans_layers if l > 0 and not self.mask[l-1]]
        prior_reset_layers = [l for l in rans_layers if l == 0 or not self.mask[l-1]]
        result = dict(packet_bytes=packet_bytes(packet), payload_bytes=2*packet.payload.numel(),
            outer_directory_bytes=8*packet.starts.numel(), output_bytes=2*n,
            shared_predictor_bytes=self.shared_predictor_bytes, shared_lut_bytes=self.shared_lut_bytes,
            coder_table_bytes=self.coder_table_bytes, decoder_workspace_tensor_bytes=_tensor_bytes(self.buffers),
            expanded_arena_bytes=0, output_clone_bytes=0, mask=list(self.mask),
            rans_layers=rans_layers, fallback_layers=fallback_layers,
            predecessor_fallback_layers=predecessor_fallback, prior_reset_layers=prior_reset_layers,
            mask_transitions=sum(a != b for a,b in zip(self.mask, self.mask[1:])),
            native_splitzip_payload_bytes=sum(e['native_payload_bytes'] for e in sz),
            native_splitzip_escape_count=sum(e['n_esc'] for e in sz),
            native_allocation_peaks='unmeasured; Torch peaks do not cover every native allocation')
        if self.native is not None:
            S,_ = self.native.layout(parsed.shape[1])
            result.update(rans_payload_bytes=2*parsed.native['stream'].numel(),
                rans_directory_bytes=4*(2*S+1), rans_final_state_bytes=4*self.hd*S)
        else:
            result.update(rans_payload_bytes=0,rans_directory_bytes=0,rans_final_state_bytes=0,
                native_splitzip_payload_bytes=packet.header['extra']['native_payload_bytes'],
                native_splitzip_escape_count=packet.header['extra']['n_esc'])
        return result
