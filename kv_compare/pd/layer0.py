"""Layer 0 recomputed inside a vLLM engine, for T2 (kv_compare.t2) on the PD path.

A token's layer-0 key and value depend only on its id and position, so the receiver recomputes them instead of
receiving them: the engine's own embedding, the first layer's input norm, its fused QKV projection and its rotary
embedding, called as the engine's forward calls them (vLLM's LlamaDecoderLayer / LlamaAttention). The bits match the
sender's cache only if every operation gives the same bits as in the sender's prefill. Two things decide that:

  the matrix product's shape  cuBLAS picks its algorithm by the product's row count M, which is every token of the
                              prefill step (other requests' rows too). The sender records each step's M for the
                              request (kv_compare.pd.connector: (start, n, M) per step); the recompute pads the
                              request's rows to M (zero rows after them: only M matters, measured in docs/t_ideas.md
                              2d, not the other rows' content or place)
  the engine's mode           in eager mode (--enforce-eager, no compilation) the forward runs these same modules;
                              compiled, the norm and the product may be fused and generated differently, and the
                              recompute need not match

The sender recomputes too and compares with its own cache before leaving layer 0 out (Layer0Context.matches); on any
difference the request carries layer 0 coded, so a transfer is bit-exact either way. The result of every check is
in the connector's log (layer0_check).
"""
from __future__ import annotations

import gc
import sys

import torch


_MODEL = None                                        # the engine's model, once found (remember_model)


def _is_model(o):
    try:
        if not isinstance(o, torch.nn.Module):     # first: getattr on any object (modules) can warn or run code
            return False
        inner = getattr(o, "model", None)
        return isinstance(inner, torch.nn.Module) and hasattr(inner, "embed_tokens") and hasattr(inner, "layers")
    except Exception:
        return False


def _unwrap(o, depth=4):
    """The model in o: o itself, a model runner's model (get_model() or .model), or what a wrapper runs
    (CUDAGraphWrapper's .runnable, a .module)."""
    for _ in range(depth):
        if o is None:
            return None
        if _is_model(o):
            return o
        nxt = None
        if "Runner" in type(o).__name__ and callable(getattr(o, "get_model", None)):
            try:
                nxt = o.get_model()
            except Exception:
                nxt = None
        for name in ("model", "runnable", "module"):
            if nxt is not None:
                break
            try:
                v = o.__dict__.get(name) if hasattr(o, "__dict__") else None
                if v is None and isinstance(o, torch.nn.Module):
                    v = o._modules.get(name)
            except Exception:
                v = None
            if v is not None and v is not o:
                nxt = v
        o = nxt
    return None


def remember_model(obj=None):
    """Find and keep the engine's model, from obj or from the call stack: the connector calls this from
    register_kv_caches, which vLLM's model runner calls, so a frame up the stack has the runner as `self`. vLLM
    freezes the garbage collector's heap after start-up (gc.freeze), and gc.get_objects() does not list frozen
    objects: in job 30226 the search below saw no module at all."""
    global _MODEL
    if obj is not None:
        m = _unwrap(obj)
    else:
        m, f = None, sys._getframe(1)
        while f is not None and m is None:
            for v in (f.f_locals.get("self"), f.f_locals.get("model_runner"), f.f_locals.get("model")):
                if v is not None:
                    m = _unwrap(v)
                    if m is not None:
                        break
            f = f.f_back
    if m is not None:
        _MODEL = m
    return m


def _search():
    best = None
    for o in gc.get_objects():
        try:
            if _is_model(o) and (best is None or len(list(o.children())) > len(list(best.children()))):
                best = o
        except Exception:
            continue
    return best


def find_model():
    """The engine's model in this process (vLLM's single-GPU executor runs the worker in the engine process): the
    torch module with .model.embed_tokens and .model.layers. The one remembered at registration first; then the
    garbage collector's objects, with its frozen heap thawed for the search and frozen again."""
    if _MODEL is not None:
        return _MODEL
    best = _search()
    frozen = gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else 0
    if best is None and frozen:
        gc.unfreeze()
        try:
            best = _search()
        finally:
            gc.freeze()
    if best is None:                                 # what there is, for the report
        seen = {}
        for o in gc.get_objects():
            try:
                if isinstance(o, torch.nn.Module) and (hasattr(o, "embed_tokens") or hasattr(o, "model") or
                                                       hasattr(o, "layers")):
                    seen[type(o).__module__ + "." + type(o).__name__] = sorted(
                        k for k in ("model", "embed_tokens", "layers", "language_model") if hasattr(o, k))
            except Exception:
                continue
        raise RuntimeError(f"no model with .model.embed_tokens and .model.layers in this process ({frozen} objects "
                           "frozen); modules seen: " + "; ".join(f"{k} {v}" for k, v in list(seen.items())[:8]))
    return remember_model(best)


class EngineLayer0:
    """Layer 0's K and V words ([n, 2 x kv channels] int16: keys rotated, then values) of token ids at positions,
    with the engine's modules (or the ones given: embed, norm, attn with qkv_proj, rotary_emb, q_size, kv_size)."""

    def __init__(self, model=None, embed=None, norm=None, attn=None):
        if embed is None:
            model = model if model is not None else find_model()
            inner = model.model
            layer = inner.layers[0]
            embed, norm, attn = inner.embed_tokens, layer.input_layernorm, layer.self_attn
        self.embed, self.norm, self.attn = embed, norm, attn
        self.q_size, self.kv_size = int(attn.q_size), int(attn.kv_size)
        p = next(iter(embed.parameters()))
        self.device = p.device

    @torch.inference_mode()
    def step(self, ids, positions, M):
        """K and V words of the first len(ids) rows of a step of M rows (the rest zero tokens at position 0)."""
        n = len(ids)
        if n > M:
            raise ValueError(f"{n} tokens in a step of {M} rows")
        x = torch.zeros(M, dtype=torch.long, device=self.device)
        pos = torch.zeros(M, dtype=torch.long, device=self.device)
        x[:n] = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        pos[:n] = torch.as_tensor(positions, dtype=torch.long, device=self.device)
        h = self.embed(x)
        h = self.norm(h)
        if isinstance(h, tuple):                     # RMSNorm with a residual returns (out, residual)
            h = h[0]
        qkv = self.attn.qkv_proj(h)
        qkv = qkv[0] if isinstance(qkv, tuple) else qkv
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.attn.rotary_emb(pos, q, k)
        out = torch.empty(n, 2 * self.kv_size, dtype=torch.int16, device=self.device)
        out[:, :self.kv_size] = k[:n].reshape(n, self.kv_size).view(torch.int16)
        out[:, self.kv_size:] = v[:n].reshape(n, self.kv_size).view(torch.int16)
        return out

    def words(self, tokens, pos0, T, steps):
        """Layer 0 of positions pos0..pos0+T-1 (tokens: the request's ids from position 0, or from tokens_from),
        step by step: steps [(start, n, M)] as the sender's prefill ran them."""
        out = torch.empty(T, 2 * self.kv_size, dtype=torch.int16, device=self.device)
        done = 0
        for a, n, M in steps:
            lo, hi = max(a, pos0), min(a + n, pos0 + T)
            if lo >= hi:
                continue
            # only the rows this transfer needs, still in a product of M rows
            out[lo - pos0:hi - pos0] = self.step(tokens[lo:hi], range(lo, hi), M)
            done += hi - lo
        if done != T:
            raise ValueError(f"the prefill steps cover {done} of the {T} tokens")
        return out


def covered(steps, pos0, T):
    """Whether steps [(start, n, M)] cover every position of pos0..pos0+T-1 (prefix-cache hits are not steps)."""
    if not steps:
        return False
    need = pos0
    for a, n, _ in sorted(steps):
        if a > need:
            return False
        need = max(need, a + n)
        if need >= pos0 + T:
            return True
    return need >= pos0 + T


class Layer0Context:
    """What T2 needs about one request: its token ids, the sender's prefill steps (sender side), and the engine's
    layer-0 modules (made once per process, on first use)."""
    _engine = None

    def __init__(self, tokens, steps=None, engine=None):
        self.tokens = list(tokens)
        self.steps = [tuple(int(x) for x in s) for s in (steps or [])]
        self._given = engine
        self.last_check = None

    @property
    def engine(self):
        if self._given is not None:
            return self._given
        if Layer0Context._engine is None:
            Layer0Context._engine = EngineLayer0()
        return Layer0Context._engine

    def covers(self, pos0, T):
        ok = covered(self.steps, pos0, T) and pos0 + T <= len(self.tokens)
        if not ok:
            self.last_check = ("no steps for every token (a prefix-cache hit or a short prompt list): "
                               f"{len(self.steps)} steps {self.steps[:3]}, tokens {pos0}..{pos0 + T}, "
                               f"prompt {len(self.tokens)}")
        return ok

    def steps_for(self, pos0, T):
        return [list(s) for s in self.steps if s[0] < pos0 + T and s[0] + s[1] > pos0]

    ROUND = (1, 8, 16, 32, 64, 128, 256, 512)

    def matches(self, l0, pos0):
        """The sender's check: its own recompute against its cache's layer 0 (l0, int16 [T, W]). The scheduler's
        row count is tried first, then that count rounded up (vLLM pads a step to a CUDA graph's batch size); the
        steps that match are kept for the header."""
        first = None
        for r in self.ROUND:
            steps = [(a, n, -(-M // r) * r) for a, n, M in self.steps]
            if r > 1 and steps == self.steps:
                continue
            try:
                got = self.engine.words(self.tokens, pos0, l0.shape[0], steps)
            except Exception as exc:
                self.last_check = f"recompute failed: {exc!r}"[:200]
                return False
            if bool(torch.equal(got, l0)):
                self.steps = steps
                self.last_check = "equal" if r == 1 else f"equal with rows rounded up to {r}"
                return True
            if first is None:
                diff = (got != l0)
                first = (f"differs in {int(diff.sum())} of {diff.numel()} words "
                         f"({int(diff.any(1).sum())} of {l0.shape[0]} tokens) with the scheduler's row counts")
        self.last_check = first
        return False

    def layer0(self, pos0, T, steps):
        """The receiver's layer 0 for the transfer's positions, with the sender's steps from the header."""
        return self.engine.words(self.tokens, pos0, T, [tuple(s) for s in steps])
