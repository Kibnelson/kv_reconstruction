"""Mask-aware preparation for the unchanged lossless mixed encoder.

The encoder only reads features of a coded layer's predecessor. An adaptive
prior is read only when both that layer and its predecessor use rANS; every
other coded layer starts from prior one. This module skips unused preparation,
while retaining the frozen feature, product, surprise and prior arithmetic.
It is an execution strategy, not a new probability model or packet format.
"""
from __future__ import annotations

import torch

from .team_onestep import layerwise as lw


def selective_plan(mask):
    """Return the complete set of feature/prior reads by MixedOneStep.encode.

    ``surprise_layers`` are source layers q whose surprise produces prior[q+1].
    ``model_product_layers`` exclude q=0, whose product is the exact zero anchor.
    Encoding's own products are separate from these preparatory products.
    """
    if not isinstance(mask, (list, tuple)) or not mask or any(type(x) is not bool for x in mask):
        raise ValueError('one explicit boolean per layer is required')
    layers = len(mask)
    coded = [i for i, value in enumerate(mask) if value]
    features = sorted({i - 1 for i in coded if i > 0})
    priors = [i for i in coded if i > 0 and mask[i - 1]]
    surprise = [i - 1 for i in priors]
    model_products = [i for i in surprise if i > 0]
    # If q supplies an adaptive prior, q itself is a coded layer, so the q-1
    # feature that predicts q is already among encoding's predecessor features.
    if any(q - 1 not in features for q in model_products):
        raise AssertionError('selective preparation omitted a required feature')
    return dict(layers=layers, rans_layers=coded, feature_layers=features,
        surprise_layers=surprise, prior_layers=priors,
        prior_reset_layers=[i for i in coded if i == 0 or not mask[i - 1]],
        model_product_layers=model_products,
        feature_passes=len(features), surprise_passes=len(surprise),
        prior_passes=len(priors), model_product_passes=len(model_products),
        encode_product_passes=sum(i > 0 for i in coded),
        full_model_feature_passes=layers - 1, full_model_surprise_passes=layers - 1,
        full_model_prior_passes=layers - 1, full_model_product_passes=max(0, layers - 2),
        skipped_feature_layers=[i for i in range(layers - 1) if i not in features],
        skipped_surprise_layers=[i for i in range(layers - 1) if i not in surprise],
        unused_feature_storage='Uninitialized; encoder never reads rows outside feature_layers')


def selective_model(native, words):
    """Prepare exactly the context consumed by this fixed rank mixed instance.

    Install on an owned native instance with
    ``native.model = functools.partial(selective_model, native)``. Unused xall
    rows intentionally remain uninitialized and are never transmitted or read.
    Do not use this result with a different layer mask or a general layerwise
    encoder. Coefficients, quantization, floating-point operations and coding
    probabilities of every selected layer are unchanged.
    """
    if native.rank is None:
        raise ValueError('selective mixed preparation currently requires the fixed rank predictor')
    plan = selective_plan(native.ours)
    if (words.dtype != torch.int16 or words.device != native.device or not words.is_contiguous()
        or words.ndim != 3 or tuple(words.shape[::2]) != (native.L, native.W)
        or plan['layers'] != native.L):
        raise ValueError('original contiguous BF16 word geometry/device differs from the mixed instance')
    L, T, W = words.shape
    native.layout(T)
    dev, k = native.device, native.k
    xall = torch.empty(L, T, W, dtype=torch.int8, device=dev)
    prior = torch.ones(L, T, dtype=torch.float32, device=dev)
    for layer in plan['feature_layers']:
        native._launch(lw._lfeat_kernel, (lw.triton.cdiv(T * W, 1024),),
            words[layer], xall[layer], k['isx'][layer], k['cos'], k['sin'],
            0, T * W, W, native.C, HD=native.hd, BLOCK=1024, warps=4)
    if plan['surprise_layers']:
        acc = torch.empty(T, W, dtype=torch.int32, device=dev)
        ssum = torch.empty(native.G, T, dtype=torch.int32, device=dev)
        for layer in plan['surprise_layers']:
            if layer == 0:
                acc.zero_()
            else:
                native._product(xall[layer - 1], layer, acc)
            native._launch(lw._lsurp_kernel, ((T // native.TB) * native.G,),
                words[layer], acc, k['sa'][layer], k['b'][layer], k['inv'][layer],
                k['cos'], k['sin'], ssum, T, W, native.G, native.HG,
                TB=native.TB, H2=native.hd // 2)
            native._launch(lw._lprior_kernel, (lw.triton.cdiv(T, 256),),
                ssum, prior[layer + 1], T, native.G, native.prior_scale, BLOCK=256, warps=4)
    return xall, prior
