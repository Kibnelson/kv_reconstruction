"""Lossless and lossy KV transfer codecs compared on the same captured KV.

Lossless: our predictive codec (previous-layer, Block, strongest no-hint and strong-and-fast configurations)
against SplitZip-style and ExANS-style exponent coding. Lossy: the same predictor run closed loop with a
quantized residual, against AQUA-KV-style, KIVI-style and FP8-cast baselines, scored by the model's own
continuation quality with the transferred KV. Entry point: scripts/bench_llama_compare.py.
"""
