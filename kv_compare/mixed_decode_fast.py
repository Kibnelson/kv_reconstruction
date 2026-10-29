"""Lossless mapped-I/O and selective-context decoder ablations.

All modes consume and emit the existing bundled transfer frame. These are local
integration kernels; the original author codec and previous adapters are controls.
"""
from __future__ import annotations

import functools

import torch

from .framed_baselines import frame
from .mapped_splitzip import prepare_mapping, encode_layers, decode_layers_into
from .mixed_fast import FastMixedTransfer, METHOD
from .mixed_transfer import Parsed, SZ_FIELDS
from .team_packet import compact_segments


class DecodeFastMixedTransfer(FastMixedTransfer):
    def __init__(self, state_path, splitzip_source, splitzip_state, splitzip_state_sha256,
                 mask, device, arm='rank256_uniform', *, selective=True, bundle=True,
                 mapped_io=False, selective_decode=False):
        if type(mapped_io) is not bool or type(selective_decode) is not bool:
            raise ValueError('decoder execution options must be explicit booleans')
        super().__init__(state_path, splitzip_source, splitzip_state, splitzip_state_sha256,
            mask, device, arm=arm, selective=selective, bundle=bundle)
        self.mapped_io, self.selective_decode = mapped_io, selective_decode
        self._layer_map = prepare_mapping(self._fallback_index, self.L) if self.bundled and mapped_io else None
        if self._layer_map is not None:
            self.shared_bytes += self._layer_map.tensor.numel() * self._layer_map.tensor.element_size()
        if self.native is not None and selective_decode:
            from .mixed_decode_context import decode_selective
            # Bind only this instance. Original source functions and other codecs
            # retain their own decode_mixed dispatch.
            self.native.decode_mixed = functools.partial(decode_selective, self.native)

    @property
    def execution_plan(self):
        plan = super().execution_plan
        rans = [i for i, value in enumerate(self.mask) if value]
        live = [i for i in rans if i + 1 < self.L and self.mask[i + 1]]
        effective = self._layer_map is not None
        plan.update(mapped_io=self.mapped_io, mapped_io_active=effective,
            selective_decode=self.selective_decode,
            decoder_context_layers=live if self.selective_decode else rans,
            decoder_skipped_context_layers=[i for i in rans if i not in live] if self.selective_decode else [],
            mapped_splitzip_encode_calls=int(effective), mapped_splitzip_decode_calls=int(effective),
            fallback_scatter_calls=0 if effective else plan['fallback_scatter_calls'])
        if effective:
            plan.update(native_splitzip_encode_calls=0, native_splitzip_decode_calls=0,
                        mapped_kernel_scope='Local derived kernels; original baseline implementation unchanged')
        return plan

    def encode(self, words, start=0):
        if self._layer_map is None:
            return super().encode(words, start=start)
        if (type(start) is not int or start != 0 or words.dtype != torch.int16
            or words.device != self.device or not words.is_contiguous()):
            raise ValueError('original contiguous BF16 words at start zero are required')
        shape = self._shape(list(words.shape))
        encoded = encode_layers(self.other.codec, self.other.module, words, self._layer_map)
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

    def decode_native(self, parsed):
        if self._layer_map is None:
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
        # FastMixedTransfer.unpack already validates the escape indices and frame
        # bounds; the helper retains structural and owned-mapping checks.
        decode_layers_into(self.other.codec, parsed.native['bundle'], self._layer_map,
                           output, validated=True)
        output = self.native.decode_mixed(parsed.native, parsed.shape[1], out=output, bufs=self.buffers)
        if self.device.type == 'cuda':
            streams = [current] + (self.buffers.get('streams') or []) + (self.buffers.get('gemm_streams') or [])
            for stream in streams:
                parsed.packet.payload.record_stream(stream); output.record_stream(stream)
                parsed.native['starts'].record_stream(stream)
                self._layer_map.tensor.record_stream(stream)
        return output

    def accounting(self, packet):
        result = super().accounting(packet)
        effective = self._layer_map is not None
        result.update(mapped_index_bytes=(self._layer_map.tensor.numel() * self._layer_map.tensor.element_size())
            if effective else 0, mapped_output_active=effective,
            avoided_fallback_gather_bytes=result['fallback_gather_bytes'] if effective else 0,
            avoided_fallback_decoded_buffer_bytes=result['fallback_decoded_buffer_bytes'] if effective else 0,
            execution_plan=self.execution_plan)
        if effective:
            result.update(fallback_gather_bytes=0, fallback_decoded_buffer_bytes=0)
        return result
