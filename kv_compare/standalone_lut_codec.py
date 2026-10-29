"""Standalone lossless codec with an exhaustive stage-1 inverse table.

All layers retain the original predictor and rANS stream. No fallback codec is
installed. This dispatch changes only how the decoder inverts its fixed CDF.
"""
from __future__ import annotations

import torch

from .standalone_full_lut import build_full_inverse, _full_lut_ldec_kernel
from .team_lookup import TeamLookupCodec
from .team_onestep import layerwise as lw


class StandaloneLUTCodec(TeamLookupCodec):
    def __init__(self, state_path, device, arm, *, decoder_mode='direct', full_lut=True):
        if type(full_lut) is not bool:
            raise ValueError('full_lut must be an explicit boolean')
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode, lookup=False)
        self.full_lut = full_lut
        self.full_inverse = None
        self.full_inverse_proof = None
        self._inverse_version = None
        self.base_coder_table_bytes = self.coder_table_bytes
        if full_lut:
            self.full_inverse, self.full_inverse_proof = build_full_inverse(self.device)
            if (self.full_inverse.dtype != torch.uint16 or tuple(self.full_inverse.shape) != (10, 65536)
                or self.full_inverse.device != self.device or not self.full_inverse.is_contiguous()):
                raise ValueError('exhaustive inverse table has incompatible structure')
            if self.full_inverse_proof['fixed_table_sha256'] != self.inverse_proof['table_sha256']:
                raise ValueError('exhaustive inverse is not bound to the native fixed CDF tables')
            self._inverse_version = self.full_inverse._version
            self.coder_table_bytes += self.full_inverse.numel() * self.full_inverse.element_size()
            original_launch = self.native._launch

            def launch(kernel, grid, *args, warps=1, **kwargs):
                if kernel is lw._ldec_kernel:
                    # Original pointer arguments end at tail_ptr (position 19).
                    # Every arithmetic/control parameter retains its old value.
                    args = args[:20] + (self.full_inverse,) + args[20:]
                    return original_launch(_full_lut_ldec_kernel, grid, *args, warps=warps, **kwargs)
                return original_launch(kernel, grid, *args, warps=warps, **kwargs)

            # Instance-local only; the unchanged encoder and other codecs keep
            # their own launch dispatch. Install before any CUDA graph capture.
            self.native._launch = launch

    def decode(self, packet):
        if self.full_inverse is not None and self.full_inverse._version != self._inverse_version:
            raise ValueError('owned exhaustive inverse table changed')
        value = super().decode(packet)
        if self.full_inverse is not None and self.device.type == 'cuda':
            # Graph replay executes on its current launch stream. Direct calls
            # also use native chunk streams; cover both lifetime patterns.
            self.full_inverse.record_stream(torch.cuda.current_stream(self.device))
            if self.decoder_mode == 'direct':
                for stream in (self.buffers.get('streams') or []) + (self.buffers.get('gemm_streams') or []):
                    self.full_inverse.record_stream(stream)
        return value

    def accounting(self, packet):
        result = super().accounting(packet)
        size = self.full_inverse.numel() * self.full_inverse.element_size() if self.full_inverse is not None else 0
        result.update(full_lut=self.full_lut, full_inverse_shared_bytes=size,
            original_coder_table_bytes=self.base_coder_table_bytes,
            inverse_algorithm='exhaustive_fixed_cdf_lookup' if self.full_lut else 'coarse_binary_search',
            decoder_kernel='_full_lut_ldec_kernel' if self.full_lut else '_ldec_kernel',
            standalone_all_layers=True, fallback_codec=None,
            full_inverse_proof=self.full_inverse_proof)
        return result
