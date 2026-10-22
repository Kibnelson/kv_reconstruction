"""Capture BF16 KV (exactly as the HuggingFace cache holds it) and pre-RoPE queries from a causal LM.

Per document: {name}_K_bits.npy and {name}_V_bits.npy, uint16 [layers, tokens, kv_heads*head_dim] in cache order
(keys after RoPE); {name}_ids.npy; optionally {name}_Q_bits.npy, uint16 [layers, tokens, query_heads*head_dim]
taken from q_proj outputs (before RoPE), needed only on calibration documents for the attention input.
meta.json records the geometry and the model's own RoPE inverse frequencies (including Llama 3 scaling).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


def model_meta(model, model_id, revision):
    import torch
    import transformers
    cfg = model.config
    heads = int(cfg.num_attention_heads)
    kv_heads = int(getattr(cfg, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // heads)
    rotary = model.model.rotary_emb
    inv_freq = rotary.inv_freq.detach().to("cpu", torch.float64).numpy()
    return {"model": model_id, "revision": revision, "layers": int(cfg.num_hidden_layers),
            "query_heads": heads, "kv_heads": kv_heads, "head_dim": head_dim,
            "inv_freq": [float(v) for v in inv_freq],
            "attention_scaling": float(getattr(rotary, "attention_scaling", 1.0)),
            "rope_layout": "half", "rope_scaling": getattr(cfg, "rope_scaling", None),
            "rope_theta": float(getattr(cfg, "rope_theta", 0.0) or 0.0),
            "torch": torch.__version__, "transformers": transformers.__version__}


def capture_ids(model, ids, out_dir, name, device, queries=False):
    """Run one prefill and save its KV words (and q_proj outputs when queries=True)."""
    import torch
    from qwen_kv_bench.model_runtime import cache_to_matrices
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(model.model.layers)
    n = len(ids)
    q_store = [None] * len(layers)
    handles = []
    if queries:
        for index, block in enumerate(layers):
            def hook(module, inputs, output, index=index):
                q_store[index] = output.detach()[0].contiguous().view(torch.int16).cpu().numpy().view(np.uint16).copy()
            handles.append(block.self_attn.q_proj.register_forward_hook(hook))
    begin = time.perf_counter()
    try:
        with torch.inference_mode():
            out = model(torch.tensor([list(ids)], dtype=torch.long, device=device), use_cache=True, logits_to_keep=1)
        matrices = cache_to_matrices(out.past_key_values)
    finally:
        for h in handles:
            h.remove()
    elapsed = time.perf_counter() - begin
    K = np.stack([k for k, _ in matrices])
    V = np.stack([v for _, v in matrices])
    if K.shape[1] != n:
        raise ValueError("cache length differs from the input length")
    np.save(out_dir / f"{name}_K_bits.npy", K)
    np.save(out_dir / f"{name}_V_bits.npy", V)
    np.save(out_dir / f"{name}_ids.npy", np.asarray(ids, dtype=np.int64))
    if queries:
        if any(q is None for q in q_store):
            raise ValueError("not every q_proj hook fired")
        np.save(out_dir / f"{name}_Q_bits.npy", np.stack(q_store))
    digest = hashlib.sha256(K.tobytes() + V.tobytes()).hexdigest()
    return {"name": name, "tokens": n, "seconds": elapsed, "kv_sha256": digest, "queries": bool(queries),
            "shape": list(K.shape)}


def load_model(model_id, revision, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = {} if revision in (None, "", "none") else {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, attn_implementation="sdpa", **kwargs)
    return tokenizer, model.to(device).eval()


def tokenize(tokenizer, path, tokens):
    ids = tokenizer(Path(path).read_text(encoding="utf-8"), add_special_tokens=True)["input_ids"]
    if len(ids) < tokens:
        raise ValueError(f"{path} has only {len(ids)} tokens; {tokens} requested")
    return ids[:tokens]


def capture_documents(model, tokenizer, documents, out_dir, device, tokens, query_docs, meta, log=print):
    """documents: role name -> text path. Writes captures and meta.json; returns the capture report."""
    out_dir = Path(out_dir)
    report = []
    for name, path in documents.items():
        ids = tokenize(tokenizer, path, tokens)
        log(f"capture {name}: {path} ({tokens} tokens)")
        info = capture_ids(model, ids, out_dir, name, device, queries=name in query_docs)
        info["source"] = str(path)
        report.append(info)
    meta = dict(meta, tokens=tokens, captures=report)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
