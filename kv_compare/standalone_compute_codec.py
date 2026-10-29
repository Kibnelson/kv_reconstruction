"""Exact-packet standalone compute ablations around the frozen codec.

Decoder products use an owned shallow native controller with shared immutable
coefficients. Sender products and the on-wire encoder identity remain cuBLAS.
The separate Dense model-cache arm changes only repeated encoder preparation.
"""
from __future__ import annotations

import copy
import functools

from .standalone_products import exact_product, validate_tiles
from .team_lookup import TeamLookupCodec
from .team_onestep import fused as fu


class StandaloneComputeCodec(TeamLookupCodec):
    def __init__(self, state_path, device, arm, *, decoder_mode='direct',
                 decoder_tiles=None, cache_model=False):
        if type(cache_model) is not bool:
            raise ValueError('cache_model must be an explicit boolean')
        if arm not in ('dense_uniform', 'rank256_uniform'):
            raise ValueError('fixed standalone Dense or Rank256 uniform arm required')
        tiles = validate_tiles(decoder_tiles) if decoder_tiles is not None else None
        if cache_model and (arm != 'dense_uniform' or tiles is not None):
            raise ValueError('Dense model caching is a separate execution arm')
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode, lookup=False)
        self.decoder_tiles = tiles
        self.cache_model = cache_model
        self.decoder_native = None
        self.cached_model = None
        if tiles is not None:
            # Copy controller metadata only: all k/rank/CDF tensors are shared,
            # and the existing caller-owned decoder workspace is passed in.
            decoder = copy.copy(self.native)
            decoder.product, decoder.gemm_tiles = 'triton', tiles
            decoder._dense = functools.partial(exact_product, tiles=tiles)
            self.decoder_native = decoder
            self.native.decode = decoder.decode
        if cache_model:
            model = fu.FusedOneStep(self.native.k, self.native.C, self.native.hd,
                                   tokens_per_program=32, rows=4, warps=4)
            # model() does not read CDF tables. Share the already installed,
            # identical tables rather than retain a duplicate cold allocation.
            model.tables = self.native.tables
            self.cached_model = model
            self.native.model = model.model

    def validate(self, packet):
        shape = super().validate(packet)
        if self.decoder_tiles is not None:
            _, tokens, width = shape
            bm, bn, bk, _ = self.decoder_tiles
            reduction = self.native.rank['rank'] if self.native.rank is not None else width
            if (tokens // self.native.chunks) % bm or width % bn or reduction % bk:
                raise ValueError('decoder geometry does not divide the fixed Triton tiles')
        return shape

    def accounting(self, packet):
        result = super().accounting(packet)
        result.update(standalone_all_layers=True, fallback_codec=None,
            encoder_product='cublas', decoder_product='triton' if self.decoder_tiles is not None else 'cublas',
            decoder_gemm_tiles=list(self.decoder_tiles) if self.decoder_tiles is not None else None,
            cache_model=self.cache_model, additional_shared_device_bytes=0,
            model_cache_shares_native_tables=self.cached_model is not None,
            full_inverse_shared_bytes=0, decoder_geometry_fallback='forbidden' if self.decoder_tiles is not None else 'reference',
            unchanged_encoder_header_product=packet.header['product'])
        return result
