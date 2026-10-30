"""Fixed-packet rolling-encoder storage and decoder scheduling ablations.

Feature reuse changes only the lifetime of the original exact INT8 features.
Decoder chunk counts change only same-token execution partitions; the sender's
native controller and packet retain their original two-chunk configuration.
Neither option changes predictor, entropy arithmetic, stream segments or words.
"""
from __future__ import annotations

import copy

import torch

from .standalone_rolling_codec import RollingEncoder, StandaloneRollingCodec, _overlap
from .team_onestep import layerwise as lw


class FeatureReuseRollingEncoder(RollingEncoder):
    """Run each feature extraction immediately before its sole product consumer."""

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
        features = torch.empty((T, W), dtype=torch.int8, device=dev) if L > 1 else None

        current = torch.empty((T, W), dtype=torch.int32, device=dev)
        preceding = torch.empty_like(current)
        scratch = native._scratch(T) if native.rank is not None else None

        def product(layer, destination):
            if layer == 0:
                destination.zero_()
            else:
                previous_layer = layer - 1
                native._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),),
                    words[previous_layer], features, k['isx'][previous_layer], k['cos'], k['sin'],
                    0, T * W, W, native.C, HD=native.hd, BLOCK=1024, warps=4)
                native._product(features, layer, destination, scratch)

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


class StandaloneScheduleCodec(StandaloneRollingCodec):
    """Independent feature-storage or decoder-partition choices for one wire."""

    def __init__(self, state_path, device, arm, *, decoder_mode='direct',
                 feature_reuse=False, decoder_chunks=None):
        if type(feature_reuse) is not bool:
            raise ValueError('feature_reuse must be an explicit boolean')
        if decoder_chunks is not None and (type(decoder_chunks) is not int or decoder_chunks not in (1, 4, 8)):
            raise ValueError('decoder_chunks must be None or one of 1, 4, 8')
        if feature_reuse and decoder_chunks is not None:
            raise ValueError('feature reuse and decoder partitioning are separate arms')
        super().__init__(state_path, device, arm, decoder_mode=decoder_mode)
        self.feature_reuse = feature_reuse
        self.decoder_chunks = decoder_chunks
        self.decoder_native = None
        if feature_reuse:
            self.rolling_encoder = FeatureReuseRollingEncoder(self.native)
        if decoder_chunks is not None:
            # Construct before any workspace/graph preparation. The shallow
            # controller shares only the immutable predictor and coder tensors;
            # its decode and buffers must agree on partition shapes and streams.
            decoder = copy.copy(self.native)
            decoder.chunks = decoder_chunks
            self.decoder_native = decoder
            self.native.decode = decoder.decode
            self.native.buffers = decoder.buffers

    def validate(self, packet):
        shape = super().validate(packet)
        if self.decoder_native is not None:
            _, tokens, width = shape
            decoder = self.decoder_native
            decoder.layout(tokens)  # Requires complete token-block partitions.
            if decoder.rank is not None and decoder.requant == 'epilogue':
                bm, bn, bk, _ = decoder.rank_tiles
                if ((tokens // decoder.chunks) % bm or decoder.rank['rank'] % bn or width % bk):
                    raise ValueError('decoder partitions do not divide the fixed Rank requantization tiles')
        return shape

    def decode(self, packet):
        output = super().decode(packet)
        if self.device.type == 'cuda':
            # One decoder chunk runs on the caller stream and has no native
            # child stream for TeamLookup to register. Protect borrowed input
            # allocated on another stream even when its owner is dropped as
            # soon as this asynchronous call returns. This also covers graph
            # staging copies; native decode has joined child work here.
            packet.payload.record_stream(torch.cuda.current_stream(self.device))
        return output

    def accounting(self, packet):
        value = super().accounting(packet)
        layers, tokens, width = packet.header['shape']
        chunks = self.decoder_native.chunks if self.decoder_native is not None else self.native.chunks
        feature_buffers = int(layers > 1) if self.feature_reuse else max(0, layers - 1)
        value.update(feature_reuse=self.feature_reuse, decoder_chunks=self.decoder_chunks,
            effective_decoder_chunks=chunks, encoder_feature_buffer_count=feature_buffers,
            encoder_feature_bytes=feature_buffers * tokens * width,
            decoder_accumulator_buffers=chunks, decoder_accumulator_bytes=tokens * width * 4,
            decoder_rank_scratch_bytes=tokens * self.native.rank['rank'] if self.native.rank is not None else 0,
            unchanged_encoder_header_chunks=packet.header['chunks'],
            decoder_partition_fallback='forbidden',
            decoder_workspace_scope='Tensor totals are charged; native stream/cuBLAS allocations outside Torch are unmeasured')
        return value
