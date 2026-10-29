"""Lossless execution ablations around the unchanged mixed GPU method.

Selective preprocessing changes execution only. Bundled fallback uses one
unchanged native SplitZip call for all fallback layers, with a checked carrier
and one scatter into the final output. Vendor and baseline sources are untouched.
"""
from __future__ import annotations

import functools
import hashlib
import math

import numpy as np
import torch

from .causal_packet import json_bytes
from .framed_baselines import frame, unframe
from .mixed_transfer import MixedTransfer, Parsed, SZ_FIELDS, packet_bytes
from .team_packet import compact_segments, _tensor_bytes

METHOD = 'team_mixed_bundled_transfer_v1'


class _DiscardFallback:
    """No codec or oracle: the caller separately encodes the single bundle."""
    @staticmethod
    def encode(_tensor):
        return None


class FastMixedTransfer(MixedTransfer):
    def __init__(self, state_path, splitzip_source, splitzip_state, splitzip_state_sha256,
                 mask, device, arm='rank256_uniform', *, selective=False, bundle=False):
        if type(selective) is not bool or type(bundle) is not bool:
            raise ValueError('execution options must be explicit booleans')
        if selective and arm != 'rank256_uniform':
            raise ValueError('selective preprocessing currently requires rank256')
        super().__init__(state_path, splitzip_source, splitzip_state, splitzip_state_sha256,
                         mask, device, arm=arm)
        self.selective, self.bundle_requested = selective, bundle
        self.fallback_layers = tuple(i for i, ours in enumerate(self.mask) if not ours)
        self.bundled = bool(bundle and self.native is not None and self.fallback_layers)
        self.base_state_sha256 = self.state_sha256
        if self.native is not None and selective:
            from .mixed_selective import selective_model
            self.native.model = functools.partial(selective_model, self.native)
        if self.bundled:
            # Only this new instance's dispatch changes. The vendor method still
            # builds its rANS stream; its per-layer fallback placeholders are not
            # used or transmitted. The actual fallback codec remains self.other.
            self.native.other = _DiscardFallback()
            self._fallback_index = torch.tensor(self.fallback_layers, dtype=torch.int64, device=self.device)
            self.shared_bytes += self._fallback_index.numel() * self._fallback_index.element_size()
            self.state_sha256 = hashlib.sha256(json_bytes(dict(format=METHOD,
                base_state_sha256=self.base_state_sha256, fallback_layers=self.fallback_layers,
                bundle_order='increasing_layer', chunk_size=1024))).hexdigest()
        else:
            self._fallback_index = None

    @property
    def execution_plan(self):
        from .mixed_selective import selective_plan
        plan = selective_plan(list(self.mask))
        return dict(selective=self.selective, bundle_requested=self.bundle_requested, bundled=self.bundled,
            required_context=plan, native_splitzip_encode_calls=(1 if self.bundled or self.native is None
                else len(self.fallback_layers)),
            native_splitzip_decode_calls=(1 if self.bundled or self.native is None
                else len(self.fallback_layers)),
            fallback_scatter_calls=int(self.bundled),
            feature_passes=(len(plan['feature_layers']) if self.selective else max(0, self.L-1))
                if self.native is not None else 0,
            surprise_passes=(len(plan['surprise_layers']) if self.selective else max(0, self.L-1))
                if self.native is not None else 0)

    def _bundle_extra(self, shape, metadata):
        result = super()._extra(shape, [])
        result.pop('splitzip_layers')
        result.update(fallback_layers=list(self.fallback_layers), bundle_order='increasing_layer',
                      splitzip_bundle=metadata, base_state_sha256=self.base_state_sha256)
        return result

    def encode(self, words, start=0):
        if not self.bundled:
            return super().encode(words, start=start)
        if (type(start) is not int or start != 0 or words.dtype != torch.int16
            or words.device != self.device or not words.is_contiguous()):
            raise ValueError('original contiguous BF16 words at start zero are required')
        shape = self._shape(list(words.shape))
        gathered = words.index_select(0, self._fallback_index)
        encoded = self.other.codec.encode(gathered.view(torch.bfloat16).flatten())
        enc = self.native.encode(words)
        S, cap = self.native.layout(shape[1])
        packed, offsets, relative = compact_segments(enc['stream'], enc['starts'],
            segments=S, capacity_words=cap, head_dim=self.hd)
        if packed.numel() >= 1 << 31:
            raise ValueError('compact stream exceeds INT32 word addressing')
        parts = [('rans_words', packed), ('rans_offsets', offsets.to(self.device, torch.int32)),
                 ('rans_relative', relative.to(self.device, torch.int32))]
        parts += [('bundle_' + name, getattr(encoded, name).flatten().contiguous()) for name in SZ_FIELDS]
        metadata = dict(n=encoded.n, n_esc=encoded.n_esc, chunk_size=encoded.chunk_size,
                        native_payload_bytes=encoded.compressed_bytes)
        return frame(METHOD, shape, self.state_sha256, parts, self._bundle_extra(shape, metadata))

    def unpack(self, packet):
        if not self.bundled:
            return super().unpack(packet)
        if packet.payload.device != self.device:
            raise ValueError('incoming packet is not on the decoder device')
        shape, parts, extra = unframe(packet, METHOD, self.state_sha256)
        shape = self._shape(shape)
        metadata = extra.get('splitzip_bundle')
        if not isinstance(metadata, dict) or extra != self._bundle_extra(shape, metadata):
            raise ValueError('bundle identity/order/geometry differs')
        names = ['rans_words', 'rans_offsets', 'rans_relative'] + ['bundle_' + n for n in SZ_FIELDS]
        if list(parts) != names:
            raise ValueError('bundled payload field order differs')
        stream, offsets, relative = [parts[n] for n in names[:3]]
        S, cap = self.native.layout(shape[1])
        if (stream.dtype != torch.int16 or stream.numel() >= 1 << 31 or stream.data_ptr() % 16
            or offsets.dtype != torch.int32 or offsets.numel() != S+1
            or relative.dtype != torch.int32 or relative.numel() != S):
            raise ValueError('invalid compact rANS tensors/alignment')
        directory = torch.cat((offsets.long(), relative.long())).cpu().numpy()
        o, r = directory[:S+1], directory[S+1:]
        if (o[0] != 0 or o[-1] != stream.numel() or np.any(o < 0) or np.any(o[1:] <= o[:-1])):
            raise ValueError('invalid compact rANS offsets')
        lengths = o[1:] - o[:-1]
        if (np.any(lengths < 2*self.hd) or np.any(lengths > cap) or np.any(lengths % 8)
            or np.any(r < 2*self.hd) or np.any(r > 2*self.hd+7) or np.any(r > lengths)):
            raise ValueError('invalid compact rANS segment bounds')
        n = len(self.fallback_layers) * shape[1] * shape[2]
        ne = metadata.get('n_esc')
        if (type(ne) is not int or not 0 <= ne <= n or metadata != dict(n=n, n_esc=ne,
            chunk_size=1024, native_payload_bytes=(n+1)//2+n+7*ne)):
            raise ValueError('invalid SplitZip bundle size/byte accounting')
        fields = {name: parts['bundle_' + name] for name in SZ_FIELDS}
        expected = dict(pk=(torch.uint8,(n+1)//2), sm=(torch.uint8,n), chunk_id=(torch.int32,ne),
                        local_pos=(torch.uint16,ne), esc_val=(torch.uint8,ne))
        if any(fields[k].dtype != dt or fields[k].numel() != count for k,(dt,count) in expected.items()):
            raise ValueError('invalid SplitZip bundle fields')
        if ne:
            cid, pos = fields['chunk_id'].long(), fields['local_pos'].long()
            if bool(((cid < 0) | (pos >= 1024) | (cid*1024+pos >= n)).any()):
                raise ValueError('invalid SplitZip bundle escape positions')
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        bundle = self.other.module.ChunkLocalGPUEncoded(fields['pk'], fields['sm'], empty, empty,
            fields['chunk_id'], fields['local_pos'], fields['esc_val'], n, ne, 1024)
        native = dict(stream=stream, starts=offsets[:-1]+relative, sz={}, bundle=bundle)
        return Parsed(shape, native, packet, self)

    def decode_native(self, parsed):
        if not self.bundled:
            return super().decode_native(parsed)
        if not isinstance(parsed, Parsed) or parsed.owner is not self:
            raise ValueError('native decode requires this codec\'s checked packet')
        if self.device.type == 'cuda':
            current = torch.cuda.current_stream(self.device)
            if self._last_stream is not None and current.cuda_stream != self._last_stream.cuda_stream:
                current.wait_stream(self._last_stream)
            self._last_stream = current
        if self.shape != parsed.shape:
            if self.shape is not None and self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.buffers = self.native.buffers(parsed.shape[1]); self.shape = parsed.shape
        output = torch.empty(parsed.shape, dtype=torch.int16, device=self.device)
        fallback = self.other.codec.decode(parsed.native['bundle']).view(torch.int16)
        fallback = fallback.reshape(len(self.fallback_layers), parsed.shape[1], parsed.shape[2])
        output.index_copy_(0, self._fallback_index, fallback)
        output = self.native.decode_mixed(parsed.native, parsed.shape[1], out=output, bufs=self.buffers)
        if self.device.type == 'cuda':
            # Cover the current-stream SplitZip read as well as internal rANS
            # streams, including callers that release a borrowed packet early.
            streams = [current] + (self.buffers.get('streams') or []) + (self.buffers.get('gemm_streams') or [])
            for stream in streams:
                parsed.packet.payload.record_stream(stream); output.record_stream(stream)
                parsed.native['starts'].record_stream(stream)
        return output

    def accounting(self, packet):
        if not self.bundled:
            result = super().accounting(packet)
            result.update(execution_plan=self.execution_plan, bundle_index_bytes=0,
                          fallback_gather_bytes=0, fallback_decoded_buffer_bytes=0)
            return result
        parsed = self.unpack(packet)
        n = math.prod(parsed.shape); S, _ = self.native.layout(parsed.shape[1])
        metadata = packet.header['extra']['splitzip_bundle']
        rans = [i for i, ours in enumerate(self.mask) if ours]
        resets = [i for i in rans if i == 0 or not self.mask[i-1]]
        fallback_bytes = 2 * len(self.fallback_layers) * parsed.shape[1] * parsed.shape[2]
        return dict(packet_bytes=packet_bytes(packet), payload_bytes=2*packet.payload.numel(),
            outer_directory_bytes=8*packet.starts.numel(), output_bytes=2*n,
            shared_predictor_bytes=self.shared_predictor_bytes, shared_lut_bytes=self.shared_lut_bytes,
            bundle_index_bytes=self._fallback_index.numel()*self._fallback_index.element_size(),
            coder_table_bytes=self.coder_table_bytes, decoder_workspace_tensor_bytes=_tensor_bytes(self.buffers),
            expanded_arena_bytes=0, output_clone_bytes=0, mask=list(self.mask), rans_layers=rans,
            fallback_layers=list(self.fallback_layers), prior_reset_layers=resets,
            predecessor_fallback_layers=[i for i in resets if i > 0],
            mask_transitions=sum(a != b for a,b in zip(self.mask, self.mask[1:])),
            native_splitzip_payload_bytes=metadata['native_payload_bytes'],
            native_splitzip_escape_count=metadata['n_esc'],
            rans_payload_bytes=2*parsed.native['stream'].numel(), rans_directory_bytes=4*(2*S+1),
            rans_final_state_bytes=4*self.hd*S, execution_plan=self.execution_plan,
            fallback_gather_bytes=fallback_bytes, fallback_decoded_buffer_bytes=fallback_bytes,
            native_allocation_peaks='unmeasured; Torch peaks do not cover every native allocation')
