"""Exact-wire mixed decoding with critical fallback rows scheduled first.

This is a local execution variant. The published SplitZip and rANS sources,
probabilities, packet representation and layer masks remain unchanged.
"""
from __future__ import annotations

import torch

from .mapped_decode_partition import prepare_partition, decode_subset_into
from .mixed_decode_fast import DecodeFastMixedTransfer
from .mixed_transfer import Parsed


class OverlapMixedTransfer(DecodeFastMixedTransfer):
    def __init__(self, *args, overlap_decode=True, **kwargs):
        if type(overlap_decode) is not bool:
            raise ValueError('overlap_decode must be an explicit boolean')
        kwargs.setdefault('mapped_io', True)
        kwargs.setdefault('selective_decode', True)
        if overlap_decode and (kwargs['mapped_io'] is not True or kwargs['selective_decode'] is not True):
            raise ValueError('overlap requires mapped I/O and selective decoding')
        super().__init__(*args, **kwargs)
        self.overlap_decode = overlap_decode
        self._partition = (prepare_partition(self._layer_map, list(self.mask))
                           if overlap_decode and self._layer_map is not None else None)
        self._bulk_stream = (torch.cuda.Stream(device=self.device)
                             if self._partition is not None and self._partition.bulk_layers
                             and self.device.type == 'cuda' else None)
        if self._partition is not None:
            self.shared_bytes += self._partition.shared_bytes

    @property
    def execution_plan(self):
        result = super().execution_plan
        partition = self._partition
        result.update(overlap_decode=self.overlap_decode,
            overlap_decode_active=partition is not None,
            critical_fallback_layers=list(partition.critical_layers) if partition is not None else [],
            bulk_fallback_layers=list(partition.bulk_layers) if partition is not None else [],
            bulk_stream_active=self._bulk_stream is not None,
            completion_scope='All critical, bulk and rANS output writes joined before returning')
        if partition is not None:
            result.update(mapped_decode_partitions=int(bool(partition.critical_layers))
                + int(bool(partition.bulk_layers)),
                escape_partition_method='Two class-filtered scans; arbitrary validated escape order')
        return result

    def decode_native(self, parsed):
        if self._partition is None:
            return super().decode_native(parsed)
        if not isinstance(parsed, Parsed) or parsed.owner is not self:
            raise ValueError("native decode requires this codec's checked packet")
        current = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        if current is not None:
            if self._last_stream is not None and current.cuda_stream != self._last_stream.cuda_stream:
                current.wait_stream(self._last_stream)
            self._last_stream = current
        if self.shape != parsed.shape:
            if self.shape is not None and current is not None:
                torch.cuda.synchronize(self.device)
            self.buffers = self.native.buffers(parsed.shape[1])
            self.shape = parsed.shape
        output = torch.empty(parsed.shape, dtype=torch.int16, device=self.device)
        bundle = parsed.native['bundle']
        # Only the immediate fallback predecessors are needed by rANS. Their
        # dense words and escape corrections both finish on the caller stream.
        decode_subset_into(self.other.codec, bundle, self._partition, output,
                           partition='critical', validated=True)
        bulk = self._bulk_stream
        native_complete = False
        try:
            if bulk is not None:
                # Capture critical readiness now. Do not put a bulk wait on the
                # caller stream until decode_mixed has scheduled its chunk streams.
                bulk.wait_stream(current)
                with torch.cuda.stream(bulk):
                    decode_subset_into(self.other.codec, bundle, self._partition, output,
                                       partition='bulk', validated=True)
            else:
                decode_subset_into(self.other.codec, bundle, self._partition, output,
                                   partition='bulk', validated=True)
            output = self.native.decode_mixed(parsed.native, parsed.shape[1], out=output, bufs=self.buffers)
            native_complete = True
        finally:
            if current is not None:
                if bulk is not None:
                    current.wait_stream(bulk)
                streams = [current] + (self.buffers.get('streams') or []) + (self.buffers.get('gemm_streams') or [])
                if not native_complete:
                    # A launch/JIT exception may bypass the native controller's
                    # usual join. Keep workspace reuse ordered on retry too.
                    for stream in streams[1:]:
                        current.wait_stream(stream)
                if bulk is not None:
                    streams.append(bulk)
                for stream in streams:
                    parsed.packet.payload.record_stream(stream)
                    parsed.native['starts'].record_stream(stream)
                    output.record_stream(stream)
                for stream in ([current, bulk] if bulk is not None else [current]):
                    for tensor in self._partition.tensors:
                        tensor.record_stream(stream)
        return output

    def accounting(self, packet):
        result = super().accounting(packet)
        plan = self._partition
        layer_bytes = 2 * packet.header['shape'][1] * packet.header['shape'][2]
        result.update(partition_index_bytes=plan.shared_bytes if plan is not None else 0,
            critical_fallback_output_bytes=layer_bytes * len(plan.critical_layers) if plan is not None else 0,
            bulk_fallback_output_bytes=layer_bytes * len(plan.bulk_layers) if plan is not None else 0,
            overlap_uses_additional_bf16_buffer=False, execution_plan=self.execution_plan)
        return result
