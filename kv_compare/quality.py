"""Model quality with transferred prompt KV, as the decode side sees it.

The prompt's KV (all layers) is replaced by the KV the receiver holds after transfer; the following tokens are
processed on the decode side with that cache. Reported: the mean negative log-likelihood of predicting tokens
P+1 .. P+M-1 from positions P .. P+M-2 (teacher forcing). The reference is the captured exact KV restored the same
way. On CPU it matches the model's own prompt cache to the last digit; on a GPU the model's own prefill of the
prompt alone can differ in the last bits (kernels depend on sequence length), so it is reported, not required.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import float_to_words


def matrices_from_values(values, columns):
    """[layers, rows, 2*columns] BF16-valued float tensor -> list of (K uint16 [rows, columns], V) per layer."""
    out = []
    for layer in range(values.shape[0]):
        w = float_to_words(values[layer]).cpu().numpy().astype(np.uint16)
        out.append((np.ascontiguousarray(w[:, :columns]), np.ascontiguousarray(w[:, columns:])))
    return out


def matrices_from_doc(doc, rows):
    return [(np.array(doc.K[layer][:rows], copy=True), np.array(doc.V[layer][:rows], copy=True))
            for layer in range(doc.layers)]


def continuation_nll(model, ids, prompt_rows, continuation, device, matrices=None):
    """Mean NLL over the continuation; matrices=None uses the model's own prompt cache."""
    from qwen_kv_bench.model_runtime import cache_from_matrices
    ids = list(ids)
    prompt = torch.tensor([ids[:prompt_rows]], dtype=torch.long, device=device)
    cont = torch.tensor([ids[prompt_rows:prompt_rows + continuation]], dtype=torch.long, device=device)
    if cont.shape[1] != continuation:
        raise ValueError("document too short for the requested continuation")
    with torch.inference_mode():
        if matrices is None:
            cache = model(prompt, use_cache=True, logits_to_keep=1).past_key_values
        else:
            cache = cache_from_matrices(matrices, model.config, device=device)
        logits = model(cont, past_key_values=cache, use_cache=True).logits[0].float()
        nll = torch.nn.functional.cross_entropy(logits[:-1], cont[0, 1:], reduction="mean")
    return float(nll)
