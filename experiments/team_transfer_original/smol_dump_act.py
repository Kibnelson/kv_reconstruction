"""Prefill SmolLM2-135M (GGUF) in numpy and dump real KV.

Usage: python3 -I smol_dump_act.py MODEL.gguf TEXT_FILE N_TOKENS OUT.npz LAYERS (e.g. 2,12,24)
Saves BF16 bit patterns of post-RoPE K and V per layer (what a serving KV cache
holds), pre-RoPE K in fp32 (available to prefill), and reports perplexity on the
text as a correctness check of this re-implementation. The optional 'neox' argument
rotates split halves instead of adjacent pairs; it exists only to check that the
default (adjacent pairs, matching the GGUF weight layout) gives far lower perplexity.
Tokens beyond N_TOKENS are dropped.
"""
import sys, time
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

path, text_path, n_tok, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
r = GGUFReader(path)
F = r.fields
def field(k):
    return F[k].contents()

L = field("llama.block_count"); D = field("llama.embedding_length")
Hq = field("llama.attention.head_count"); Hk = field("llama.attention.head_count_kv")
eps = field("llama.attention.layer_norm_rms_epsilon"); base = field("llama.rope.freq_base")
hd = D // Hq
T = {t.name: t for t in r.tensors}

def W(name):
    t = T[name]
    a = dequantize(t.data, t.tensor_type).astype(np.float32)
    return a.reshape(tuple(int(s) for s in reversed(t.shape)))

# tokenizer from the GGUF vocabulary (GPT-2 style byte-level BPE)
tokens = field("tokenizer.ggml.tokens"); merges = field("tokenizer.ggml.merges")
vocab = {t: i for i, t in enumerate(tokens)}
tk = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(m.split(" ", 1)) for m in merges]))
tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True),
                                            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)])
tk.decoder = decoders.ByteLevel()
text = open(text_path, encoding="utf-8").read()
ids = np.array(tk.encode(text).ids[:n_tok])
N = len(ids)
assert tk.decode(ids[:200].tolist()) == text[:len(tk.decode(ids[:200].tolist()))], "tokenizer round-trip failed"
print(f"model: L={L} D={D} Hq={Hq} Hk={Hk} hd={hd} base={base}; tokens={N}", flush=True)

def bf16_bits(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)
def from_bits(b):
    return (b.astype(np.uint32) << 16).view(np.float32)
def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
inv = base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
ang = np.arange(N, dtype=np.float64)[:, None] * inv[None, :]
COS, SIN = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
NEOX = False
SEL = [int(a) for a in sys.argv[5].split(",")]
def rope(x):  # adjacent-pair rotation (llama.cpp NORM mode, matches GGUF weight layout)
    a, b = (slice(0, hd // 2), slice(hd // 2, hd)) if NEOX else (slice(0, None, 2), slice(1, None, 2))
    x0, x1 = x[..., a], x[..., b]
    o = np.empty_like(x)
    o[..., a] = x0 * COS - x1 * SIN
    o[..., b] = x0 * SIN + x1 * COS
    return o

def attention(q, k, v, C=1024):
    g = Hq // Hk
    out = np.empty((N, Hq, hd), np.float32)
    scale = 1.0 / np.sqrt(hd)
    for s in range(0, N, C):
        e = min(N, s + C)
        mask = np.where(np.arange(e)[None, :] > np.arange(s, e)[:, None], -np.inf, 0).astype(np.float32)
        for h in range(Hq):
            sc = (q[s:e, h] @ k[:e, h // g].T) * scale + mask
            sc -= sc.max(1, keepdims=True)
            p = np.exp(sc); p /= p.sum(1, keepdims=True)
            out[s:e, h] = p @ v[:e, h // g]
    return out.reshape(N, D)

emb = W("token_embd.weight")
x = emb[ids].copy()
K_bits = np.empty((L, N, Hk * hd), np.uint16); V_bits = np.empty_like(K_bits)
K_pre = np.empty((L, N, Hk * hd), np.float32)
H_sel = np.empty((len(SEL), N, D), np.float32); Q_sel = np.empty((len(SEL), N, Hq * hd), np.float32)
t0 = time.time()
for l in range(L):
    p = f"blk.{l}."
    h = rmsnorm(x, W(p + "attn_norm.weight"))
    q = (h @ W(p + "attn_q.weight").T).reshape(N, Hq, hd)
    k = (h @ W(p + "attn_k.weight").T).reshape(N, Hk, hd)
    v = (h @ W(p + "attn_v.weight").T).reshape(N, Hk, hd)
    K_pre[l] = k.reshape(N, -1)
    q, k = rope(q), rope(k)
    if l in SEL:
        H_sel[SEL.index(l)] = h; Q_sel[SEL.index(l)] = q.reshape(N, -1)
    K_bits[l] = bf16_bits(k.reshape(N, -1)); V_bits[l] = bf16_bits(v.reshape(N, -1))
    kc = from_bits(K_bits[l]).reshape(N, Hk, hd); vc = from_bits(V_bits[l]).reshape(N, Hk, hd)
    x = x + attention(q, kc, vc) @ W(p + "attn_output.weight").T
    h2 = rmsnorm(x, W(p + "ffn_norm.weight"))
    gt = h2 @ W(p + "ffn_gate.weight").T
    x = x + ((gt / (1 + np.exp(-gt))) * (h2 @ W(p + "ffn_up.weight").T)) @ W(p + "ffn_down.weight").T
    print(f"  layer {l} done ({time.time()-t0:.0f}s)", flush=True)

hN = rmsnorm(x, W("output_norm.weight"))
nll = 0.0
for s in range(0, N - 1, 512):
    e = min(N - 1, s + 512)
    lg = hN[s:e] @ emb.T
    lg -= lg.max(1, keepdims=True)
    lse = np.log(np.exp(lg).sum(1))
    nll += float((lse - lg[np.arange(e - s), ids[s + 1:e + 1]]).sum())
ppl = float(np.exp(nll / (N - 1)))
print(f"perplexity on text: {ppl:.2f}", flush=True)
np.savez(out_path, K_bits=K_bits, V_bits=V_bits, H=H_sel, Q=Q_sel, sel=np.array(SEL), ids=ids, ppl=ppl,
         Hk=Hk, hd=hd, base=base)
print("saved", out_path)
