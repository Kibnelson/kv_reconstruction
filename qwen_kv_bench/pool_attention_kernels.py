"""Decode attention component; same FP32 online softmax for both loaders.

Online softmax context: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
No tensor cores, tuning, split-K or serving-backend integration is claimed.
"""
import triton
import triton.language as tl
from .pool_p1_kernels import load_sector


@triton.jit
def _float_words(x):
    return (x.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _update(q, kw, vw, valid, maximum, normalizer, acc, TILE: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr):
    k = _float_words(kw).reshape((TILE, D))
    v = _float_words(vw).reshape((TILE, D))
    score = tl.sum(k * q[None, :], 1) * SCALE
    score = tl.where(valid, score, -float('inf'))
    new_max = tl.maximum(maximum, tl.max(score, 0))
    alpha = tl.exp(maximum-new_max)
    prob = tl.exp(score-new_max)
    new_norm = normalizer*alpha + tl.sum(prob, 0)
    new_acc = acc*alpha + tl.sum(prob[:, None]*v, 0)
    return new_max, new_norm, new_acc


@triton.jit(do_not_specialize=['LSTART'])
def attention(Q, WORDS, RAWTAIL, A8, A16, A32, DESC, R, B, OUT, STATUS,
              PACKED: tl.constexpr, CAPACITY: tl.constexpr,
              T: tl.constexpr, C: tl.constexpr, HEADS: tl.constexpr, QHEADS: tl.constexpr,
              D: tl.constexpr, TILE: tl.constexpr, PAGES: tl.constexpr, N: tl.constexpr,
              SECTORS: tl.constexpr, TAIL: tl.constexpr, SCALE: tl.constexpr,LSTART=0):
    local_layer = tl.program_id(0);layer=local_layer+LSTART; qhead = tl.program_id(1)
    head = qhead // (QHEADS // HEADS)
    d = tl.arange(0, D); i = tl.arange(0, N); rows = tl.arange(0, TILE)
    q = tl.load(Q + (local_layer*QHEADS+qhead)*D+d)
    maximum = tl.full((), -float('inf'), tl.float32)
    normalizer = tl.full((), 0, tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    good = tl.full((), True, tl.int1)
    for page in range(PAGES):
        if PACKED:
            sid = (layer*PAGES+page)*2*HEADS+head
            kw, kg = load_sector(A8,A16,A32,DESC,R,B,sid,CAPACITY,SECTORS,T,C,2*HEADS,D,TILE,PAGES,N)
            vw, vg = load_sector(A8,A16,A32,DESC,R,B,sid+HEADS,CAPACITY,SECTORS,T,C,2*HEADS,D,TILE,PAGES,N)
            good = good & kg & vg
        else:
            token = page*TILE+i//D
            address = (layer*T+token)*C+head*D+i%D
            kw = tl.load(WORDS+address, token<T, other=0)
            vw = tl.load(WORDS+address+HEADS*D, token<T, other=0)
        maximum, normalizer, acc = _update(q,kw,vw,page*TILE+rows<T,maximum,normalizer,acc,TILE,D,SCALE)
    if TAIL > 0:
        token = i//D; address = (layer*TAIL+token)*C+head*D+i%D
        kw = tl.load(RAWTAIL+address,token<TAIL,other=0)
        vw = tl.load(RAWTAIL+address+HEADS*D,token<TAIL,other=0)
        maximum, normalizer, acc = _update(q,kw,vw,rows<TAIL,maximum,normalizer,acc,TILE,D,SCALE)
    tl.store(OUT+(local_layer*QHEADS+qhead)*D+d,tl.where(good,acc/normalizer,0))
    tl.store(STATUS+local_layer*QHEADS+qhead,tl.where(good,0,1))
