"""Fixed-wire decoder inverse choices around the measured single-feature codec."""
from __future__ import annotations

from .standalone_schedule_codec import StandaloneScheduleCodec
from .team_lookup import _table_receipts
from .team_onestep import layerwise as lw
from . import standalone_inverse_core as core


class StandaloneInverseCodec(StandaloneScheduleCodec):
    """Keep two chunks and the single-feature rolling encoder in every arm."""

    def __init__(self, state_path, device, arm, *, decoder_mode='direct', inverse_mode='packed'):
        if inverse_mode not in ('packed', 'masked'):
            raise ValueError('inverse_mode must be packed or masked')
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode,
                         feature_reuse=True, decoder_chunks=None)
        self.inverse_mode = inverse_mode
        self.core_coarse = self.core_sparse = None
        self.stage1_proof = core.inverse_proof()
        if (not self.stage1_proof['verified'] or self.stage1_proof['checked_slots'] != 655360
                or self.stage1_proof['fixed_table_sha256'] != _table_receipts(self.native.tables)):
            raise ValueError('inverse proof differs from original native probability tables')
        if inverse_mode == 'packed':
            self.core_coarse, self.core_sparse, proof = core.build_inverse_tables(self.device)
            if (proof != self.stage1_proof or self.core_coarse.numel() * self.core_coarse.element_size() != 40960
                    or self.core_sparse.numel() * self.core_sparse.element_size() != 4608):
                raise ValueError('owned packed inverse tables differ from their exhaustive proof')
        self._original_launch = self.native._launch
        self.native._launch = self._launch_core

    def _launch_core(self, kernel, grid, *args, **kwargs):
        if kernel is lw._ldec_kernel:
            if len(args) != 26 or 'PACKED' in kwargs:
                raise ValueError('original decoder launch signature changed')
            # The masked kernel does not read these arguments. Reuse an
            # existing allocation as its typed placeholder instead of creating
            # dummy GPU tensors. PACKED is a compile-time branch.
            coarse = self.core_coarse if self.core_coarse is not None else self.native.tables['cumf']
            sparse = self.core_sparse if self.core_sparse is not None else self.native.tables['cumf']
            args = args[:20] + (coarse, sparse) + args[20:]
            kwargs['PACKED'] = self.inverse_mode == 'packed'
            return self._original_launch(core._core_ldec_kernel, grid, *args, **kwargs)
        return self._original_launch(kernel, grid, *args, **kwargs)

    def accounting(self, packet):
        value = super().accounting(packet)
        packed = self.inverse_mode == 'packed'
        extra = 45568 if packed else 0
        value.update(inverse_mode=self.inverse_mode,
            inverse_algorithm='packed_cutoff_sparse' if packed else 'masked_coarse_binary_search',
            decoder_kernel='_core_ldec_kernel', original_coder_table_bytes=self.coder_table_bytes,
            coder_table_bytes=self.coder_table_bytes + extra,
            additional_shared_device_bytes=extra, sparse_inverse_shared_bytes=extra,
            sparse_inverse_coarse_bytes=40960 if packed else 0,
            sparse_inverse_exception_bytes=4608 if packed else 0,
            inverse_proof_verified=True, inverse_checked_slots=self.stage1_proof['checked_slots'],
            inverse_reference_sha256=self.stage1_proof['reference_inverse_sha256'],
            inverse_coarse_sha256=self.stage1_proof['coarse_sha256'] if packed else None,
            inverse_sparse_sha256=self.stage1_proof['sparse_sha256'] if packed else None,
            full_inverse_shared_bytes=0)
        return value
