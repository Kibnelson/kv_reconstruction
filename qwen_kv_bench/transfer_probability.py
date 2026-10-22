"""Matched exact Block32 probability-width experiment.

All arms keep the original mean predictor and BF16 interval probabilities.
B changes only the logistic width using causal, integer context features.
C/D add a charged scalar code from produced KV / actual attention-input H.
The range coder and probability construction are CPU references. Optional
device backends compute parameters, never entropy-decode the stream themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np

from . import transfer_reference as ref
from .transfer_coset import group_num, mant_num, ordered, TOTAL, code_block


FORMAT = "block32_probability_arms_v1"
MAGIC = b"KVPRAW01"
# Identical fixed framing for A/B/C/D. Original-word hash also checks raw mode.
HEADER = struct.Struct("<8sBBBBIIII32s32s32sII")
ARMS = ("A", "B", "C", "D")
FEATURE_NAMES = ("bias16", "previous_K_exponent", "previous_V_exponent",
                 "previous_sign_balance", "prefix_error_bucket", "last_error_bucket",
                 "prefix_signed_error_bucket", "prefix_large_error_fraction16")
F = len(FEATURE_NAMES)
Q = 4096
MAX_OFFSET = 16  # eighth-octaves: width multipliers from 1/4 to 4


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _array_hash(description, arrays):
    h = hashlib.sha256(canonical(description))
    for name, a in sorted(arrays.items()):
        a = np.ascontiguousarray(a)
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(canonical(list(a.shape)))
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class State:
    arm: str
    base_sha256: str
    coefficients: np.ndarray
    thresholds: np.ndarray
    hint_delta: np.ndarray
    scale_table: np.ndarray
    hint_bits: int = 0
    context_strength: int = 4
    hint_strength: int = 4
    state_sha256: str = ""

    def description(self):
        return dict(format=FORMAT, arm=self.arm, base_sha256=self.base_sha256,
                    hint_bits=self.hint_bits, context_strength=self.context_strength,
                    hint_strength=self.hint_strength, features=list(FEATURE_NAMES))

    def arrays(self):
        return {n: getattr(self, n) for n in ("coefficients", "thresholds", "hint_delta", "scale_table")}

    def digest(self):
        self.state_sha256 = _array_hash(self.description(), self.arrays())
        return self.state_sha256


def save_state(state, path):
    state.digest()
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, description=np.frombuffer(canonical(dict(
            state.description(), state_sha256=state.state_sha256)), np.uint8), **state.arrays())
    return dict(serialized_bytes=Path(path).stat().st_size,
                numeric_bytes=sum(a.nbytes for a in state.arrays().values()), sha256=state.state_sha256)


def load_state(path):
    with np.load(path, allow_pickle=False) as z:
        d = json.loads(z["description"].tobytes())
        if d.pop("format") != FORMAT or d.pop("features") != list(FEATURE_NAMES):
            raise ValueError("unsupported probability state")
        expected = d.pop("state_sha256")
        result = State(**d, **{n: z[n].copy() for n in (
            "coefficients", "thresholds", "hint_delta", "scale_table")})
    if result.arm not in ARMS or result.hint_bits not in (0, 1, 2, 4):
        raise ValueError("invalid arm or hint budget")
    if (result.arm in ('A','B')) != (result.hint_bits==0):
        raise ValueError("arm and hint budget disagree")
    if result.context_strength not in (0, 2, 4) or result.hint_strength not in (0, 2, 4):
        raise ValueError("invalid frozen strength")
    if result.digest() != expected:
        raise ValueError("probability state checksum mismatch")
    return result


def empty_state(base, arm="A", hint_bits=0):
    if base.mode not in ('block','previous','aqua','mean','strong_layers') or base.metadata_rank:
        raise ValueError("experiment requires a supported metadata-free causal base")
    if arm in ('C','D') and base.mode != 'block':
        raise ValueError('hint arms require the original block predictor')
    if arm not in ARMS or hint_bits not in (0, 1, 2, 4):
        raise ValueError("invalid probability arm")
    if (arm in ('A','B')) != (hint_bits==0):
        raise ValueError("arm and hint budget disagree")
    if not base.state_sha256:
        base.state_sha256 = ref._state_digest(base)
    layers = base.geometry["layers"]; groups = len(base.predictors[0]); width = base.block_size
    thresholds = np.zeros((layers, groups, width, 10), np.int64)
    if arm != "A":
        for l, blocks in enumerate(base.predictors):
            for g, b in enumerate(blocks):
                # Compare errors to frozen thresholds, avoiding floating log or
                # normalized-error multiplication in deployed context features.
                t = (65536. / b["inv_s"][:, None]) * 2.**np.arange(-4, 6)[None, :]
                thresholds[l, g] = np.rint(np.clip(t, 0, 1 << 50)).astype(np.int64)
    else:
        thresholds = np.empty(0, np.int64)
    s = State(arm, base.state_sha256,
              np.zeros((layers, groups, F), np.int64) if arm != "A" else np.empty(0, np.int64),
              thresholds, np.zeros((layers, groups, 1 << hint_bits), np.int16) if hint_bits else np.empty(0, np.int16),
              2.**(-np.arange(-MAX_OFFSET, MAX_OFFSET + 1, dtype=np.float64)/8.), hint_bits)
    s.digest(); return s


def pack_ids(ids, bits):
    ids = np.asarray(ids)
    if bits not in (1, 2, 4) or ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= 1 << bits):
        raise ValueError("invalid hint IDs")
    stream = ((ids.astype(np.uint8).ravel()[:, None] >> np.arange(bits)) & 1).ravel()
    return np.packbits(stream, bitorder="little").tobytes()


def unpack_ids(data, bits, shape):
    count = int(np.prod(shape)); n = count * bits
    if bits not in (1, 2, 4) or len(data) != (n + 7)//8:
        raise ValueError("incorrect hint length")
    b = np.unpackbits(np.frombuffer(data, np.uint8), bitorder="little")
    if np.any(b[n:]):
        raise ValueError("nonzero hint padding")
    return (b[:n].reshape(count, bits) @ (1 << np.arange(bits))).astype(np.uint8).reshape(shape)


def known_context(previous_words, errors, signed, last, large, tokens):
    """Uses only previous-layer words and already decoded prefix errors."""
    x = np.zeros((tokens, F), np.int64); x[:, 0] = 16
    if previous_words is not None:
        previous_words = previous_words.astype(np.int64); width = previous_words.shape[1]//2
        e = ((previous_words >> 7) & 255) - 127
        x[:, 1] = np.sum(e[:, :width], axis=1)//width
        x[:, 2] = np.sum(e[:, width:], axis=1)//width
        x[:, 3] = (np.sum(1 - 2*((previous_words >> 15) & 1), axis=1)*16)//(2*width)
    if errors:
        x[:, 4] = np.sum(np.concatenate(errors, axis=1), axis=1)//sum(a.shape[1] for a in errors)
        x[:, 5] = np.sum(last, axis=1)//last.shape[1]
        x[:, 6] = np.sum(np.concatenate(signed, axis=1), axis=1)//sum(a.shape[1] for a in signed)
        x[:, 7] = np.sum(np.concatenate(large, axis=1), axis=1)*16//sum(a.shape[1] for a in large)
    return np.clip(x, -128, 128)


def offsets(state, context, layer, group, ids=None):
    if state.arm == "A":
        return np.zeros(len(context), np.int64)
    acc = np.sum(context * state.coefficients[layer, group], axis=1) * state.context_strength
    if state.hint_bits:
        if ids is None or np.asarray(ids).shape != (len(context),):
            raise ValueError("hint IDs required")
        acc += state.hint_delta[layer, group, ids].astype(np.int64) * Q * state.hint_strength
    return np.clip((acc + 2*Q)//(4*Q), -MAX_OFFSET, MAX_OFFSET)


def symbol_bits(mu, inverse, target, f_tab):
    ranks = ordered(np.asarray(target).ravel()); groups = ranks >> 7; mantissa = ranks & 127
    mn = mant_num(np.asarray(mu).ravel(), np.asarray(inverse).ravel(), groups, f_tab)
    return -np.log2(mn[np.arange(len(ranks)), mantissa]/float(TOTAL)).reshape(np.asarray(target).shape)


class NumpyBackend:
    device = "cpu"
    gpu = False

    def __init__(self, base, state):
        self.base = base; self.state = state

    def predict(self, features, layer, group, positions):
        return ref._predict(features, self.base.predictors[layer][group], group*self.base.block_size, positions, self.base)

    def offsets(self, context, layer, group, ids):
        return offsets(self.state, context, layer, group, ids)

    def statistics(self):
        return dict(device="cpu", gpu=False, persistent_device_tensor_bytes=0)


def _buckets(recovered, mu, thresholds):
    difference = ref._fixed(ref.from_bits(recovered)) - np.rint(mu*65536.).astype(np.int64)
    absolute = np.abs(difference)
    bucket = np.sum((absolute[:, :, None] > 0) & (absolute[:, :, None] >= thresholds[None]), axis=2, dtype=np.int64)
    return bucket, bucket*np.sign(difference).astype(np.int64), ((absolute > 0) & (absolute >= thresholds[None, :, 5])).astype(np.int64)


def trace(base, capture, state=None):
    """TRAIN/DEV analysis replay. Never represented as packet decompression."""
    state = state or empty_state(base, "B")
    if state.arm == 'A':
        state = empty_state(base, 'B')  # A has no error-bucket tables.
    source = ref.words(capture); positions = ref._positions(capture)
    previous = np.empty((len(positions), 0), np.int64); previous_words = None; rows = []
    history = {}
    for layer, blocks in enumerate(base.predictors):
        if base.mode == 'strong_layers':
            sources=ref.context_layers(base.mode,layer)
            previous=np.concatenate([history[s] for s in sources],axis=1) if sources else np.empty((len(positions),0),np.int64)
        current = ref._all_fixed(source[layer], positions, base)
        features = np.concatenate((previous, np.zeros_like(current)), axis=1)
        errors=[]; signed=[]; large=[]; last=None
        for group, block in enumerate(blocks):
            ch = group*base.block_size; target = source[layer, :, ch:ch+base.block_size]
            context = known_context(previous_words, errors, signed, last, large, len(positions))
            mu = ref._predict(features, block, ch, positions, base)
            inverse = np.broadcast_to(block["inv_s"], mu.shape)
            residual = ref.from_bits(target).astype(np.float64)-mu
            # One response per token/group, in eighth-octaves, relative to the
            # frozen base logistic scale. This is a fitting surrogate only.
            response = np.clip(8*np.log2(np.maximum(np.mean(np.abs(residual)*inverse, axis=1)/(2*np.log(2)), 2.**-8)), -32, 32)
            rows.append(dict(layer=layer, group=group, context=context, mu=mu, inverse=inverse,
                             target=target, response=response))
            last, sign, big = _buckets(target, mu, state.thresholds[layer, group])
            errors.append(last); signed.append(sign); large.append(big)
            features[:, previous.shape[1]+ch:previous.shape[1]+ch+base.block_size] = current[:, ch:ch+base.block_size]
        previous = current; previous_words = source[layer]
        if base.mode == 'strong_layers':
            history[layer]=current
            history={s:v for s,v in history.items() if s==0 or s>=layer-1}
    return rows


def fit_context(base, captures, ridge=1.):
    state = empty_state(base, "B")
    traces = [trace(base, a, state) for a in captures]
    for i in range(len(traces[0])):
        rows=[t[i] for t in traces]; x=np.concatenate([r["context"] for r in rows]).astype(np.float64)
        y=np.concatenate([r["response"] for r in rows]); penalty=np.eye(F)*ridge; penalty[0,0]=ridge*.01
        coefficient=np.linalg.solve(x.T@x + penalty, x.T@y)
        state.coefficients[rows[0]["layer"], rows[0]["group"]] = np.rint(np.clip(coefficient, -64, 64)*Q).astype(np.int64)
    state.digest(); return state, traces


def hint_source(base, capture, kind):
    if kind == "kv":
        return np.stack([ref._all_fixed(ref.words(capture, l), ref._positions(capture), base)
                         for l in range(base.geometry["layers"])])
    if kind != "hidden":
        raise ValueError("unknown hint source")
    h=np.asarray(capture.get("H"))
    shape=np.asarray(capture["K_bits"]).shape
    if h.ndim != 3 or h.shape[:2] != shape[:2]:
        raise ValueError("actual all-layer prefill H is required")
    if "sel" in capture and not np.array_equal(capture["sel"], np.arange(h.shape[0])):
        raise ValueError("hidden layer identities must be canonical")
    return ref._fixed(h, "actual prefill H")


def _project_score(source, weight, bias):
    maximum=max(abs(int(source.min())), abs(int(source.max()))) if source.size else 0
    if maximum*sum(abs(int(x)) for x in weight) + abs(int(bias)) + (1 << 15) >= 1 << 62:
        raise ValueError("hint INT64 projection overflow")
    return ((source@weight + (1 << 15)) >> 16) + bias


def hint_ids(encoder, source, backend=None):
    if backend is not None and hasattr(backend, "hint_ids"):
        return backend.hint_ids(encoder, source)
    ids=[]
    for l in range(len(source)):
        x=source[l]
        if encoder.get('feature_map')=='signed_absolute':x=np.concatenate((x,np.abs(x)),axis=1)
        score=_project_score(x, encoder["weights"][l], encoder["bias"][l])
        ids.append(np.sum(score[:,None] >= encoder["edges"][l][None], axis=1).astype(np.uint8))
    return np.stack(ids)


def fit_hint(base, context_state, captures, traces, kind, bits=2, sketch_width=16, ridge=1., seed=71):
    """Supervised scalar readout of *remaining scale innovation*, not raw PCA.

    A fixed random sketch bounds TRAIN fitting to sketch_width coefficients.
    Whitening, sketch and readout are folded into one deployed integer vector.
    Both sources use the same readout dimension and transmitted bit budget.
    """
    if bits not in (1,2,4) or sketch_width < 1:
        raise ValueError("invalid hint configuration")
    arm="C" if kind=="kv" else "D"; state=empty_state(base, arm, bits)
    state.coefficients=context_state.coefficients.copy(); state.thresholds=context_state.thresholds.copy()
    state.context_strength=context_state.context_strength
    sources=[hint_source(base, a, kind) for a in captures]
    layers=base.geometry["layers"]; width=2*sources[0].shape[2]; levels=1 << bits
    encoder=dict(kind=kind, bits=bits, sketch_width=sketch_width,feature_map='signed_absolute',
                 weights=np.zeros((layers,width),np.int64),bias=np.zeros(layers,np.int64),
                 edges=np.zeros((layers,levels-1),np.int64))
    for l in range(layers):
        fixed=np.concatenate([s[l] for s in sources]);fixed=np.concatenate((fixed,np.abs(fixed)),axis=1)
        xs=fixed.astype(np.float64)/65536.
        responses=[]
        for tr in traces:
            rows=[r for r in tr if r["layer"]==l]
            responses.append(np.mean([r["response"]-offsets(context_state,r["context"],l,r["group"]) for r in rows],axis=0))
        y=np.concatenate(responses); mean=xs.mean(0); sd=np.maximum(xs.std(0),1e-5)
        rng=np.random.default_rng(np.random.SeedSequence([seed,l,width]))
        sketch=rng.choice(np.array([-1.,1.]),size=(width,min(sketch_width,width)))/np.sqrt(width)
        z=((xs-mean)/sd)@sketch; ymean=float(y.mean())
        readout=np.linalg.solve(z.T@z + ridge*np.eye(z.shape[1]), z.T@(y-ymean))
        w=(sketch@readout)/sd; b=ymean-mean@w
        encoder["weights"][l]=ref._fixed(w,"hint coefficients")
        encoder["bias"][l]=int(ref._fixed(np.array(b),"hint bias"))
        scores=_project_score(fixed,encoder["weights"][l],encoder["bias"][l])
        encoder["edges"][l]=np.rint(np.quantile(scores,np.arange(1,levels)/levels)).astype(np.int64)
    all_ids=[hint_ids(encoder,s) for s in sources]
    for i in range(len(traces[0])):
        first=traces[0][i]; l,g=first["layer"],first["group"]
        y=np.concatenate([tr[i]["response"]-offsets(context_state,tr[i]["context"],l,g) for tr in traces])
        ids=np.concatenate([a[l] for a in all_ids])
        for code in range(levels):
            selected=y[ids==code]
            if len(selected):
                state.hint_delta[l,g,code]=int(np.clip(np.rint(selected.sum()/(len(selected)+4.)), -16,16))
    state.digest(); return state, encoder


def save_encoder(encoder, path):
    desc={k:v for k,v in encoder.items() if not isinstance(v,np.ndarray)}
    arrays={k:v for k,v in encoder.items() if isinstance(v,np.ndarray)}
    sha=_array_hash(desc,arrays)
    with Path(path).open("wb") as f:
        np.savez_compressed(f,description=np.frombuffer(canonical(dict(desc,sha256=sha)),np.uint8),**arrays)
    return dict(sha256=sha,serialized_bytes=Path(path).stat().st_size,numeric_bytes=sum(a.nbytes for a in arrays.values()))


def load_encoder(path):
    with np.load(path,allow_pickle=False) as z:
        d=json.loads(z["description"].tobytes()); sha=d.pop("sha256")
        arrays={k:z[k].copy() for k in z.files if k!="description"}
    if _array_hash(d,arrays)!=sha:
        raise ValueError("encoder checksum mismatch")
    return dict(d,**arrays)


def _run(base,state,shape,start,ids,source=None,stream=None,backend=None):
    backend=backend or NumpyBackend(base,state)
    layers,tokens,width=ref._shape(base,shape,start); positions=np.arange(start,start+tokens)
    encoding=source is not None
    coder=ref._constriction().stream.queue.RangeEncoder() if encoding else ref._constriction().stream.queue.RangeDecoder(np.frombuffer(stream,dtype="<u4"))
    output=np.empty(shape,np.uint16); previous=np.empty((tokens,0),np.int64); previous_words=None;history={}
    parameter_hash=hashlib.sha256(); prediction_seconds=0.; entropy_seconds=0.; context_seconds=0.; clamp_count=0
    for l,blocks in enumerate(base.predictors):
        if base.mode=='strong_layers':
            sources=ref.context_layers(base.mode,l)
            previous=np.concatenate([history[s] for s in sources],axis=1) if sources else np.empty((tokens,0),np.int64)
        features=np.concatenate((previous,np.zeros((tokens,width),np.int64)),axis=1)
        errors=[]; signed=[]; large=[]; last=None
        for g,b in enumerate(blocks):
            ch=g*base.block_size; begin=time.perf_counter()
            context=known_context(previous_words,errors,signed,last,large,tokens)
            context_seconds+=time.perf_counter()-begin; begin=time.perf_counter()
            mu=backend.predict(features,l,g,positions)
            off=backend.offsets(context,l,g,None if ids is None else ids[l])
            inverse=np.broadcast_to(b["inv_s"],mu.shape)*state.scale_table[off+MAX_OFFSET,None]
            prediction_seconds+=time.perf_counter()-begin
            parameter_hash.update(mu.astype("<f8").tobytes());parameter_hash.update(off.astype("<i8").tobytes())
            clamp_count+=int(np.count_nonzero(np.abs(off)==MAX_OFFSET))
            begin=time.perf_counter()
            recovered,_=code_block(coder,mu.ravel(),inverse.ravel(),
                source[l,:,ch:ch+base.block_size].ravel() if encoding else None,f_tab=base.f_tab)
            entropy_seconds+=time.perf_counter()-begin; recovered=recovered.reshape(tokens,base.block_size)
            output[l,:,ch:ch+base.block_size]=recovered; begin=time.perf_counter()
            if state.arm!="A":
                last,sign,big=_buckets(recovered,mu,state.thresholds[l,g]);errors.append(last);signed.append(sign);large.append(big)
            fixed=ref._fixed(ref.from_bits(recovered))
            features[:,previous.shape[1]+ch:previous.shape[1]+ch+base.block_size]=ref._rope_int(fixed,ch,positions,base,True) if ch<base.geometry["channels"] else fixed
            context_seconds+=time.perf_counter()-begin
        previous=features[:,previous.shape[1]:].copy(); previous_words=output[l]
        if base.mode=='strong_layers':
            history[l]=previous
            history={s:v for s,v in history.items() if s==0 or s>=l-1}
    wire=coder.get_compressed().astype("<u4").tobytes() if encoding else None
    return output,wire,dict(parameter_sha256=parameter_hash.hexdigest(),parameter_seconds=prediction_seconds,
        cpu_probability_entropy_seconds=entropy_seconds,cpu_context_seconds=context_seconds,
        clipped_scale_offsets=clamp_count,entropy_device="cpu",parameter_device=str(backend.device),
        causal_context="recovered symbols only during decode; original source only at encoder")


def encode(base,state,capture,ids=None,backend=None):
    source=ref.words(capture); pos=ref._positions(capture)
    if not len(pos) or not np.array_equal(pos,np.arange(pos[0],pos[0]+len(pos))):
        raise ValueError("packets require contiguous token positions")
    return encode_words(base,state,source,int(pos[0]),ids,backend)


def encode_words(base,state,source,start=0,ids=None,backend=None):
    source=np.asarray(source)
    if source.dtype!=np.uint16 or source.ndim!=3:
        raise ValueError("original words must be uint16 [layers,tokens,width]")
    if base.state_sha256!=state.base_sha256:
        raise ValueError("wrong baseline state")
    shape=ref._shape(base,source.shape,start);state.digest();begin=time.perf_counter();raw=False
    if state.hint_bits:
        if np.asarray(ids).shape!=source.shape[:2]:raise ValueError("hint shape mismatch")
        side=pack_ids(ids,state.hint_bits)
    else:
        if ids is not None:raise ValueError("unexpected hints")
        side=b""
    # Nonfinite/extreme words use a charged exact raw mode. Validate all layers
    # before touching the coder; ordinary programming errors are never hidden.
    try:
        for l in range(shape[0]):ref._all_fixed(source[l],np.arange(start,start+shape[1]),base)
    except ValueError:
        raw=True
    if raw:
        payload=source.astype("<u2").tobytes();side=b"";stats=dict(raw_fallback=True,parameter_sha256=None,
            parameter_seconds=0.,cpu_probability_entropy_seconds=0.,cpu_context_seconds=0.)
    else:
        try:
            recovered,payload,stats=_run(base,state,shape,start,ids,source=source,backend=backend)
            if not np.array_equal(source,recovered):raise AssertionError("encoder reconstruction differs")
        except ValueError as error:
            if 'fixed-point' not in str(error) and 'overflow' not in str(error):raise
            raw=True;payload=source.astype('<u2').tobytes();side=b''
            stats=dict(raw_fallback=True,parameter_sha256=None,parameter_seconds=0.,
                cpu_probability_entropy_seconds=0.,cpu_context_seconds=0.,fallback_reason=str(error))
    header=HEADER.pack(MAGIC,ARMS.index(state.arm),state.hint_bits,int(raw),0,*shape,start,
        bytes.fromhex(base.state_sha256),bytes.fromhex(state.state_sha256),
        hashlib.sha256(source.astype("<u2").tobytes()).digest(),len(side),len(payload))
    packet=header+side+payload
    return packet,dict(stats,packet_bytes=len(packet),framing_bytes=len(header),hint_bytes=len(side),
        main_stream_bytes=len(payload),hint_padding_bits=len(side)*8-(0 if raw else shape[0]*shape[1]*state.hint_bits),
        bits_per_value=len(packet)*8/source.size,encode_seconds=time.perf_counter()-begin,
        raw_fallback=raw,original_sha256=hashlib.sha256(source.astype("<u2").tobytes()).hexdigest())


def parse_packet(base,state,packet):
    if len(packet)<HEADER.size:raise ValueError("truncated probability packet")
    magic,arm,bits,raw,reserved,l,n,w,start,bsha,ssha,original,side_size,size=HEADER.unpack_from(packet)
    if magic!=MAGIC or arm>=len(ARMS) or ARMS[arm]!=state.arm or bits!=state.hint_bits or raw not in (0,1) or reserved:
        raise ValueError("invalid probability packet header")
    if bsha.hex()!=base.state_sha256 or ssha.hex()!=state.digest() or state.base_sha256!=base.state_sha256:
        raise ValueError("packet/state identity mismatch")
    ref._shape(base,(l,n,w),start)
    if len(packet)!=HEADER.size+side_size+size:raise ValueError("packet length mismatch")
    side=packet[HEADER.size:HEADER.size+side_size];stream=packet[HEADER.size+side_size:]
    if raw and (side_size or size!=l*n*w*2):raise ValueError("invalid raw packet")
    if not raw and (not size or size%4):raise ValueError("invalid entropy stream length")
    ids=unpack_ids(side,bits,(l,n)) if bits and not raw else None
    if not bits and side_size:raise ValueError("unexpected side stream")
    return (l,n,w),start,ids,stream,bool(raw),original.hex()


def decode(base,state,packet,backend=None):
    begin=time.perf_counter();shape,start,ids,stream,raw,expected=parse_packet(base,state,packet)
    if raw:
        output=np.frombuffer(stream,dtype="<u2").reshape(shape).copy();stats=dict(raw_fallback=True,parameter_sha256=None,
            parameter_seconds=0.,cpu_probability_entropy_seconds=0.,cpu_context_seconds=0.)
    else:
        output,_,stats=_run(base,state,shape,start,ids,stream=stream,backend=backend)
    if hashlib.sha256(output.astype("<u2").tobytes()).hexdigest()!=expected:
        raise ValueError("decoded original-word checksum mismatch")
    return output,dict(stats,decode_seconds=time.perf_counter()-begin,bit_exact_checksum=True)


def analyze_trace(base,state,rows,ids=None):
    """Ideal-rate/error diagnostics, explicitly separate from packet lengths."""
    result=[]
    for r in rows:
        l,g=r["layer"],r["group"];off=offsets(state,r["context"],l,g,None if ids is None else ids[l])
        inv=r["inverse"]*state.scale_table[off+MAX_OFFSET,None]
        original_bits=symbol_bits(r["mu"],r["inverse"],r["target"],base.f_tab)
        adapted_bits=symbol_bits(r["mu"],inv,r["target"],base.f_tab)
        values=r["target"].size; residual=np.abs(ref.from_bits(r["target"]).astype(np.float64)-r["mu"])*r["inverse"]
        # A diagnostic hard-bit screen for this precise group/token-major order.
        # No syndrome bytes are encoded and this is not the old page-head order.
        f=r["mu"].astype(np.float32);u=f.view(np.uint32)
        predicted=((u+np.uint32(0x7fff)+((u>>16)&1))>>16).astype(np.uint16)
        mismatch=((predicted^r["target"])>>6)&1;flat=mismatch.ravel();ng=len(flat)//31
        eligible=int(np.sum(flat[:ng*31].reshape(ng,31).sum(1)<=1)) if ng else 0
        result.append(dict(layer=l,group=g,channel_start=g*base.block_size,
            kind="K" if g*base.block_size<base.geometry["channels"] else "V",values=values,
            baseline_ideal_bits=float(original_bits.sum()),adapted_ideal_bits=float(adapted_bits.sum()),
            ideal_bits_saved=float((original_bits-adapted_bits).sum()),
            offset_histogram=np.bincount(off+MAX_OFFSET,minlength=2*MAX_OFFSET+1).tolist(),
            normalized_abs_error_mean=float(residual.mean()),normalized_abs_error_p95=float(np.quantile(residual,.95)),
            normalized_abs_error_over_4=int(np.count_nonzero(residual>4)),
            mantissa_bit6_errors=int(mismatch.sum()),syndrome31_groups=ng,syndrome31_eligible_groups=eligible,
            syndrome_scope="diagnostic only; group-local token-major bit6 errors; not an encoded coset result"))
    return result
