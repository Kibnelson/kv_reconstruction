"""Fixed-wire packed decoder with exact warp-prefix stream compaction.

The candidate retains the measured rolling encoder, single feature buffer,
two decoder chunks and packed CDF inverse. Only the decoder's ordered stream
read positions change implementation. No probabilities, packet geometry or
predictor tensors change. A cold runtime proof gates the actual CUDA path.
"""
from __future__ import annotations

import torch

from .standalone_inverse_codec import StandaloneInverseCodec
from .team_onestep import layerwise as lw
from . import standalone_prefix_core as core


class StandalonePrefixCodec(StandaloneInverseCodec):
    """Standalone packed inverse plus exact four-group ballot/popcount prefix.

    CPU construction supports software contract/accounting checks only. The
    production candidate requires the pinned CUDA compiler/hardware contract
    and passes a separate real-device mapping probe before any warm timing.
    There is no CPU decoder or fallback to the reference prefix algorithm.
    """

    def __init__(self, state_path, device, arm, *, decoder_mode='direct'):
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode,
                         inverse_mode='packed')
        if self.native.hd != 128 or self.native.TB != 32 or self.native.chunks != 2:
            raise ValueError('warp prefix requires HD=128, TB=32 and the unchanged two decoder chunks')
        self.prefix_mode = 'warp'
        self.prefix_proof = core.prefix_proof()
        if not self.prefix_proof.get('verified'):
            raise ValueError('ordered warp-prefix software proof failed')
        if self.device.type == 'cuda':
            self.prefix_runtime_proof = core.runtime_prefix_proof(self.device)
            if not self.prefix_runtime_proof.get('verified'):
                raise ValueError('actual CUDA warp-prefix mapping proof failed')
        else:
            self.prefix_runtime_proof = dict(verified=False,
                scope='CPU construction checks software contracts only; actual CUDA mapping is unmeasured')
        self._prefix_production_layouts = {}
        # Super stores the original native launcher before installing its
        # instance-local inverse adapter. Route directly through that launcher
        # so the two adapters cannot insert the packed tables twice.
        self.native._launch = self._launch_prefix

    def _verify_production_variant(self, grid, args, kwargs):
        """Compile and inspect each exact specialization before its first use.

        Triton's pointer specialization includes dtype/alignment; integer
        specialization includes one/divisibility. Exact scalar values plus
        pointer dtype and alignment conservatively distinguish all variants
        used by this fixed contiguous-tensor interface. Warm calls reuse the
        JSON receipt; the compiled kernel itself stays in Triton's JIT cache.
        """
        if self.device.type != 'cuda' or not self.prefix_runtime_proof['verified']:
            raise RuntimeError('production warp-prefix layout proof requires actual CUDA')
        signature = tuple((str(arg.dtype), arg.data_ptr() % 16)
            if isinstance(arg, torch.Tensor) else (type(arg).__name__, arg)
            for arg in args)
        options = {name:value for name,value in kwargs.items() if name != 'warps'}
        key = (signature, tuple(sorted(options.items())), tuple(sorted(self.native.opts.items())))
        if key not in self._prefix_production_layouts:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('uncertified warp-prefix specialization encountered during CUDA capture')
            compiled = core._prefix_ldec_kernel.warmup(*args, grid=grid, num_warps=1,
                **self.native.opts, **options)
            receipt = core.validate_compiled_prefix(compiled, expected_calls=2)
            if not receipt.get('verified'):
                raise ValueError('compiled production warp-prefix layout proof failed')
            self._prefix_production_layouts[key] = receipt

    def _launch_prefix(self, kernel, grid, *args, **kwargs):
        if kernel is lw._ldec_kernel:
            if (len(args) != 26 or 'PACKED' in kwargs
                    or kwargs.get('H2') != 64 or kwargs.get('TB') != 32
                    or type(kwargs.get('warps', 1)) is not int
                    or kwargs.get('warps', 1) != 1):
                raise ValueError('warp prefix requires the original HD128 one-warp decoder launch')
            args = args[:20] + (self.core_coarse, self.core_sparse) + args[20:]
            kwargs['PACKED'] = True
            self._verify_production_variant(grid, args, kwargs)
            return self._original_launch(core._prefix_ldec_kernel, grid, *args, **kwargs)
        return self._original_launch(kernel, grid, *args, **kwargs)

    def decode(self, packet):
        if self.device.type != 'cuda':
            raise RuntimeError('warp-prefix candidate decoding requires actual CUDA; no CPU fallback')
        return super().decode(packet)

    def accounting(self, packet):
        value = super().accounting(packet)
        value.update(prefix_mode=self.prefix_mode,
            decoder_prefix_algorithm='warp_ballot_popcount',
            prefix_geometry=128, prefix_warps=1,
            prefix_proof_verified=bool(self.prefix_proof['verified']),
            prefix_runtime_verified=bool(self.prefix_runtime_proof['verified']),
            prefix_additional_shared_device_bytes=0,
            prefix_proof=self.prefix_proof,
            prefix_runtime_proof=self.prefix_runtime_proof,
            prefix_production_layout_count=len(self._prefix_production_layouts),
            prefix_production_layouts=list(self._prefix_production_layouts.values()),
            decoder_kernel='_prefix_ldec_kernel')
        return value
