"""Same-packet encoder that reuses each layer's predictor product once.

The source KV is available to the sender. Its layer-l prior depends on the
surprise of layer l-1, not on that layer's prior. Walking backward with two
INT32 products therefore supplies both the current coding prediction and the
preceding-layer surprise without a separate prediction forward pass. All
feature, surprise, prior, product and entropy arithmetic uses the frozen native
methods and kernels. Decoder execution and transmitted bytes are unchanged.
"""
from __future__ import annotations

import torch

from .team_lookup import TeamLookupCodec
from .team_packet import TeamPacket, compact_segments
from .team_onestep import layerwise as lw


def _overlap(first, second):
    if first.numel() == 0 or second.numel() == 0:
        return False
    a, b = first.data_ptr(), second.data_ptr()
    return a < b + second.numel() * second.element_size() and b < a + first.numel() * first.element_size()


class RollingEncoder:
    """Stateless controller; all mutable encoder storage belongs to one call."""

    def __init__(self, native):
        self.native = native

    def encode_native(self, words, *, out=None, layer_words=None):
        native = self.native
        if (words.dtype != torch.int16 or words.ndim != 3 or not words.is_contiguous()
                or words.device != native.device or words.shape[0] != native.L
                or words.shape[2] != native.W):
            raise ValueError('rolling encoder requires contiguous original INT16 words of the fitted geometry')
        if native.product != 'cublas':
            raise ValueError('rolling encoder preserves the original cuBLAS product')
        L, T, W = words.shape
        S, cap = native.layout(T)
        if out is not None and (out.dtype != torch.int16 or out.device != words.device
                or out.ndim != 1 or not out.is_contiguous() or out.numel() != S * cap
                or _overlap(out, words)):
            raise ValueError('rolling output must be a disjoint contiguous native INT16 arena')
        if layer_words is not None and (layer_words.dtype != torch.int64
                or layer_words.device != words.device or tuple(layer_words.shape) != (L,)
                or not layer_words.is_contiguous() or _overlap(layer_words, words)
                or (out is not None and _overlap(layer_words, out))):
            raise ValueError('layer counts must be a disjoint contiguous INT64 vector')

        dev, k, tables = native.device, native.k, native.tables
        # No feature of the last layer is consumed by any predictor.
        features = torch.empty((max(0, L - 1), T, W), dtype=torch.int8, device=dev)
        for layer in range(L - 1):
            native._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),),
                words[layer], features[layer], k['isx'][layer], k['cos'], k['sin'],
                0, T * W, W, native.C, HD=native.hd, BLOCK=1024, warps=4)

        current = torch.empty((T, W), dtype=torch.int32, device=dev)
        preceding = torch.empty_like(current)
        scratch = native._scratch(T) if native.rank is not None else None

        def product(layer, destination):
            if layer == 0:
                destination.zero_()
            else:
                native._product(features[layer - 1], layer, destination, scratch)

        product(L - 1, current)
        if L > 1:
            product(L - 2, preceding)
        surprise = torch.empty((native.G, T), dtype=torch.int32, device=dev)
        prior = torch.empty(T, dtype=torch.float32, device=dev)
        arena = torch.empty(S * cap, dtype=torch.int16, device=dev) if out is None else out
        starts = torch.empty(S, dtype=torch.int32, device=dev)
        state = torch.full((S * native.hd,), 65536, dtype=torch.int32, device=dev)
        positions = (torch.arange(S, dtype=torch.int32, device=dev) + 1) * cap
        multipliers = torch.empty(S * native.TB, dtype=torch.float32, device=dev)
        for layer in reversed(range(L)):
            if layer == 0:
                prior.fill_(1)
            else:
                previous_layer = layer - 1
                native._launch(lw._lsurp_kernel, (S,), words[previous_layer], preceding,
                    k['sa'][previous_layer], k['b'][previous_layer], k['inv'][previous_layer],
                    k['cos'], k['sin'], surprise, T, W, native.G, native.HG,
                    TB=native.TB, H2=native.hd // 2)
                native._launch(lw._lprior_kernel, (lw.triton.cdiv(T, 256),),
                    surprise, prior, T, native.G, native.prior_scale, BLOCK=256, warps=4)
            before = positions.sum(dtype=torch.int64) if layer_words is not None and layer > 0 else None
            native._launch(lw._lenc_kernel, (S,), words[layer], prior, state, positions, arena,
                starts, current, k['sa'][layer], k['b'][layer], k['sp'][layer], k['cos'], k['sin'],
                multipliers, tables['cumf'], tables['ko'], T, W, native.G, native.HG,
                FLUSH=layer == 0, TB=native.TB, H2=native.hd // 2, UNIFORM=native.mode == 'uniform')
            if before is not None:
                layer_words[layer] = before - positions.sum(dtype=torch.int64)
            if layer > 0:
                current, preceding = preceding, current
                if layer > 1:
                    product(layer - 2, preceding)
        if layer_words is not None:
            layer_words[0] = (native.bits(starts, T) - 32 * S) // 16 - layer_words[1:].sum()
        return arena, starts


class StandaloneRollingCodec(TeamLookupCodec):
    """Standalone encoder reuse with the original binary-search decoder."""

    def __init__(self, state_path, device, arm, *, decoder_mode='direct'):
        if arm not in ('dense_uniform', 'rank256_uniform'):
            raise ValueError('fixed standalone Dense or Rank256 uniform arm required')
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode, lookup=False)
        self.rolling_encoder = RollingEncoder(self.native)

    def encode(self, words, start=0):
        if type(start) is not int or start != 0:
            raise ValueError('one-step snapshot supports start=0 only')
        self._check_words(words)
        header = self.header(tuple(words.shape))
        arena, starts = self.rolling_encoder.encode_native(words)
        payload, offsets, relative = compact_segments(arena, starts,
            segments=header['segments'], capacity_words=header['segment_capacity_words'],
            head_dim=self.native.hd)
        return TeamPacket(header, payload, offsets, relative)

    def encode_prepared(self, words, xall, prior):
        raise RuntimeError('rolling encoding computes its own features and priors; precomputed inputs are unsupported')

    def accounting(self, packet):
        value = super().accounting(packet)
        L, T, W = packet.header['shape']
        value.update(standalone_all_layers=True, fallback_codec=None,
            encoder_product='cublas', decoder_product='cublas', rolling_encoder=True,
            encoder_execution='reverse rolling products', encoder_accumulator_buffers=2,
            encoder_prediction_products_per_call=max(0, L - 1),
            reference_rank_encoder_prediction_products_per_call=max(0, 2 * L - 3) if self.native.rank is not None else None,
            encoder_accumulator_bytes=2 * T * W * 4,
            encoder_additional_accumulator_bytes=T * W * 4,
            encoder_feature_bytes=max(0, L - 1) * T * W,
            encoder_prior_bytes=T * 4, encoder_surprise_bytes=self.native.G * T * 4,
            encoder_rank_scratch_bytes=T * self.native.rank['rank'] if self.native.rank is not None else 0,
            encoder_scratch_lifetime='fresh per encode call; no retained device tensors',
            additional_shared_device_bytes=0, full_inverse_shared_bytes=0,
            unchanged_encoder_header_product=packet.header['product'])
        return value
