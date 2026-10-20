"""TRAIN-only bit predictors for independently decodable exact syndrome pages.

A retained BF16 code comes from produced KV or actual upstream activations.
The exact codec supplies parity, mode and escape information. BCE is a training
surrogate; DEV complete packet bytes select a recipe. No other page is needed.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import zlib
import numpy as np
import torch
from torch.nn import functional as F
from . import neural_probability as neural, producer_models as producer
from .side_probability import canonical_source, joint

FORMAT = "qwen_prefill_syndrome_bit_models_v1"
MODES = ("none", "kv", "kv_canonical", "hidden", "hidden_norm")
BUDGETS = (0, 2, 4, 6)
EMBED_LAYER, EMBED_CHANNEL = 4, 8
PREDICT_WORD_CHUNK = 8192


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("invalid syndrome model " + name)
    return int(value)


def _words(value, name):
    value = np.asarray(value)
    if value.dtype != np.uint16 or value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(name + " must be nonempty uint16[T,C] BF16 words")
    return value


def normalization_scales(record, head_dim):
    """CPU encoder rsqrt from actual hooked BF16 pre-normalization K projection.

    This recomputes the statistic; it does not reuse the RMSNorm kernel's
    internal FP32 accumulator. The feature extraction timer must include it.
    """
    projection = neural._floats(_words(record.get("k_proj"), "k_proj"))
    dim = _int(head_dim, "head_dim")
    epsilon = record.get("k_norm_eps")
    if (projection.shape[1] % dim or isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float, np.floating))
            or not np.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("hidden_norm requires actual positive k_norm_eps and matching K projection")
    values = projection.reshape(len(projection), -1, dim)
    variance = np.mean(values * values, axis=-1, dtype=np.float32)
    scales = np.float32(1.) / np.sqrt(variance + np.float32(epsilon))
    if not np.isfinite(scales).all():
        raise ValueError("nonfinite producer normalization feature")
    return scales.astype(np.float32, copy=False)


def source_values(record, mode, inverse, rope_scaling, head_dim):
    if mode == "kv":
        return neural._floats(joint(record))
    if mode == "kv_canonical":
        return neural._floats(canonical_source(record, inverse, rope_scaling, head_dim))
    hidden = neural._floats(_words(record.get("hidden"), "hidden"))
    if mode == "hidden":
        return hidden
    if mode == "hidden_norm":
        return np.concatenate((hidden, normalization_scales(record, head_dim)), axis=1)
    raise ValueError("no encoder source for mode " + str(mode))


def _runtime(device):
    value = neural._runtime(device)
    value["rule"] = "page-local-BF16-code-masked-field-MLP-mantissa-bits-v1"
    value["prediction_word_chunk"] = PREDICT_WORD_CHUNK
    return value


def _network_shapes(g, mode):
    l,c,r,w = (g[k] for k in ("layers","channels","decoder_rank","width"))
    inputs = r + 1 + 1 + 7 + 1 + 2 + EMBED_LAYER + EMBED_CHANNEL
    result = dict(layer_embedding=(l,EMBED_LAYER), channel_embedding=(2*c,EMBED_CHANNEL),
        prior=(l,2*c,7), W0=(inputs,w), b0=(w,), W1=(w,w), b1=(w,), W2=(w,7), b2=(7,))
    if mode != "none":
        result.update(encoder_W=(g["source_width"],r),encoder_b=(r,))
    return result


def _schema(model):
    meta, arrays = model.metadata, model.params
    if (model.mode not in MODES or meta.get("format") != FORMAT or meta.get("mode") != model.mode
            or meta.get("training_only") is not True or not meta.get("model_identity")):
        raise ValueError("invalid syndrome model identity")
    g = meta.get("geometry",{})
    for key in ("layers","channels","hidden_size","head_dim","decoder_rank","width"):
        _int(g.get(key),key)
    _int(g.get("rank"),"rank",0)
    if (g["head_dim"]%2 or g["channels"]%g["head_dim"]
            or g["rank"] != (0 if model.mode=="none" else g["decoder_rank"])):
        raise ValueError("invalid syndrome model geometry")
    source_width = (2*g["channels"] if model.mode in ("kv","kv_canonical") else
        g["hidden_size"]+(g["channels"]//g["head_dim"] if model.mode=="hidden_norm" else 0))
    if g.get("source_width") != (0 if model.mode=="none" else source_width):
        raise ValueError("invalid syndrome source width")
    shapes = _network_shapes(g,model.mode)
    shapes.update(rope_inv_freq=(g["head_dim"]//2,),exponent_centers=(g["layers"],2*g["channels"]))
    if model.mode!="none":
        shapes.update(source_mu=(g["layers"],g["source_width"]),source_sd=(g["layers"],g["source_width"]))
    if set(arrays)!=set(shapes):
        raise ValueError("syndrome model array fields mismatch")
    for name,shape in shapes.items():
        a=arrays[name]
        if a.dtype!=np.float32 or a.shape!=shape or not np.isfinite(a).all():
            raise ValueError("invalid syndrome model parameter "+name)
    center=arrays["exponent_centers"]
    if np.any(center<0) or np.any(center>255) or np.any(center!=np.floor(center)):
        raise ValueError("invalid exact exponent centers")
    if np.any(arrays["rope_inv_freq"]<=0) or (model.mode!="none" and np.any(arrays["source_sd"]<=0)):
        raise ValueError("invalid syndrome model scales")
    captures=meta.get("training_captures")
    if (not isinstance(captures,list) or not captures or any(row.get("split")!="train" for row in captures)
            or meta.get("training_hashes")!=[row.get("sha256") for row in captures]):
        raise ValueError("invalid syndrome TRAIN provenance")
    cfg=meta.get("config",{})
    if (cfg.get("latent_dtype")!="BF16" or cfg.get("activation")!="GELU approximate=tanh"
            or cfg.get("threshold")!="logit >= 0" or cfg.get("latent_clamp")!=[-32.,32.]
            or cfg.get("training_residue_bits")!=list(BUDGETS)
            or not isinstance(cfg.get("rope_scaling"),(int,float))
            or not np.isfinite(cfg["rope_scaling"]) or cfg["rope_scaling"]<=0):
        raise ValueError("invalid syndrome inference rule")


def _descriptor(model):
    _schema(model)
    wire=neural._shard_bytes("syndrome_parameters",model.params)
    if len(wire)>=99*1024**2:
        raise ValueError("syndrome parameter shard exceeds 99 MiB")
    shard=dict(file="syndrome_"+model.mode+".bin",bytes=len(wire),sha256=hashlib.sha256(wire).hexdigest())
    desc=dict(format="syndrome_model_artifact_v1",metadata=producer._json(model.metadata),shard=shard,
        manifest_bytes=0,total_serialized_bytes=0)
    desc["metadata_and_shard_crc32"]=zlib.crc32(_canonical(dict(metadata=desc["metadata"],shard=shard)))
    for _ in range(8):
        data=_canonical(desc)
        if desc["manifest_bytes"]==len(data) and desc["total_serialized_bytes"]==len(data)+len(wire):
            return desc,data,wire
        desc["manifest_bytes"],desc["total_serialized_bytes"]=len(data),len(data)+len(wire)
    raise AssertionError("syndrome artifact accounting failed to converge")


@dataclass
class Model:
    mode: str
    params: dict
    metadata: dict
    path: str = ""

    def _frozen(self):
        if not hasattr(self,"_manifest"):
            desc,data,_=_descriptor(self)
            self._description,self._manifest=desc,data
            self._metadata_sha=hashlib.sha256(_canonical(self.metadata)).hexdigest()
            self._signature=tuple((n,id(a),a.shape,a.dtype.str) for n,a in sorted(self.params.items()))
            for a in self.params.values():a.flags.writeable=False
        elif (hashlib.sha256(_canonical(self.metadata)).hexdigest()!=self._metadata_sha
                or tuple((n,id(a),a.shape,a.dtype.str) for n,a in sorted(self.params.items()))!=self._signature
                or any(a.flags.writeable for a in self.params.values())):
            raise ValueError("frozen syndrome model changed")
        return self._description,self._manifest

    @property
    def model_id(self):
        return hashlib.sha256(self._frozen()[1]).hexdigest()

    @property
    def state_bytes(self):
        return self._frozen()[0]["total_serialized_bytes"]

    @property
    def artifact_paths(self):
        return [self.path,str(Path(self.path).parent/self._frozen()[0]["shard"]["file"])] if self.path else []

    @property
    def parameter_count(self):
        return sum(self.params[n].size for n in _network_shapes(self.metadata["geometry"],self.mode))

    @property
    def total_float32_values(self):
        return sum(a.size for a in self.params.values())

    def prepare(self,device="cpu"):
        return Prepared(self,device)


def save(model,path):
    path=Path(path)
    desc,data,wire=_descriptor(model)
    path.parent.mkdir(parents=True,exist_ok=True)
    (path.parent/desc["shard"]["file"]).write_bytes(wire)
    path.write_bytes(data)
    model.path=str(path.resolve());model._frozen()
    return path


def load(path,*,expected_model_identity=None,expected_training_hashes=None,expected_model_id=None,expected_mode=None):
    path=Path(path);data=path.read_bytes()
    try:desc=json.loads(data)
    except (ValueError,UnicodeDecodeError) as error:raise ValueError("invalid syndrome model JSON") from error
    if not isinstance(desc,dict) or desc.get("format")!="syndrome_model_artifact_v1" or _canonical(desc)!=data:
        raise ValueError("invalid canonical syndrome model artifact")
    meta,shard=desc.get("metadata"),desc.get("shard")
    if not isinstance(meta,dict) or not isinstance(shard,dict):raise ValueError("missing syndrome metadata/shard")
    name=shard.get("file")
    if not isinstance(name,str) or Path(name).name!=name or name in (".",".."):
        raise ValueError("invalid syndrome shard filename")
    if desc.get("metadata_and_shard_crc32")!=zlib.crc32(_canonical(dict(metadata=meta,shard=shard))):
        raise ValueError("syndrome manifest CRC mismatch")
    if expected_model_identity is not None and producer._json(expected_model_identity)!=meta.get("model_identity"):
        raise ValueError("syndrome source model identity mismatch")
    if expected_training_hashes is not None and list(expected_training_hashes)!=meta.get("training_hashes"):
        raise ValueError("syndrome TRAIN hash mismatch")
    if expected_mode is not None and expected_mode!=meta.get("mode"):raise ValueError("syndrome mode mismatch")
    if expected_model_id is not None and expected_model_id!=hashlib.sha256(data).hexdigest():
        raise ValueError("syndrome model SHA mismatch")
    wire=(path.parent/name).read_bytes()
    if len(wire)!=shard.get("bytes") or hashlib.sha256(wire).hexdigest()!=shard.get("sha256"):
        raise ValueError("syndrome shard bytes/SHA mismatch")
    model=Model(meta.get("mode"),neural._load_shard(wire,"syndrome_parameters"),meta,str(path.resolve()))
    if _descriptor(model)[1]!=data:raise ValueError("noncanonical complete syndrome artifact")
    model._frozen()
    return model


def _encode(params,source,layer,*,straight_through):
    x=(source-params["source_mu"][layer])/params["source_sd"][layer]
    latent=(x@params["encoder_W"]+params["encoder_b"]).clamp(-32.,32.)
    rounded=latent.to(torch.bfloat16).float()
    return rounded.detach()+(latent-latent.detach()) if straight_through else rounded


def _forward(params,g,layer,features,start,known,residue_bits,channel_indices=None):
    """Only decoder-known fields enter the network; mask unknown bits again."""
    tokens,selected=known.shape
    if channel_indices is None:channel_indices=torch.arange(2*g["channels"],device=known.device)
    if len(channel_indices)!=selected:raise ValueError("syndrome selected-channel geometry mismatch")
    b=int(residue_bits)
    integer=known.to(torch.int64)&(0xff80|((1<<b)-1))
    exponent=((integer>>7)&255).float()
    centered=((exponent-params["exponent_centers"][layer,channel_indices])/8.).clamp(-32.,32.)
    sign=((integer>>15)&1).float()*2.-1.
    observed=torch.stack([((integer>>j)&1).float() for j in range(7)],dim=-1)
    code=features if features.shape[1] else torch.zeros((tokens,g["decoder_rank"]),device=known.device)
    if code.shape!=(tokens,g["decoder_rank"]):raise ValueError("syndrome code rank mismatch")
    # Known, channel-specific split-half RoPE phase. Hidden inputs precede
    # RoPE, so forcing the MLP to relearn every frequency is an avoidable
    # confound. V columns receive neutral phase (0,1). No side bytes are used.
    positions=torch.arange(start,start+tokens,device=known.device,dtype=torch.float32)
    frequency_index=(channel_indices%g["head_dim"])%(g["head_dim"]//2)
    angles=positions[:,None]*params["rope_inv_freq"][frequency_index][None,:]
    key=(channel_indices<g["channels"])[None,:]
    position=torch.stack((torch.where(key,torch.sin(angles),0.),
        torch.where(key,torch.cos(angles),1.)),dim=-1)
    x=torch.cat((code[:,None,:].expand(tokens,selected,-1),centered[...,None],sign[...,None],observed,
        torch.full((tokens,selected,1),b/7.,device=known.device),position,
        params["layer_embedding"][layer][None,None,:].expand(tokens,selected,-1),
        params["channel_embedding"][channel_indices][None,:,:].expand(tokens,selected,-1)),dim=-1)
    x=F.gelu(x@params["W0"]+params["b0"],approximate="tanh")
    x=F.gelu(x@params["W1"]+params["b1"],approximate="tanh")
    return x@params["W2"]+params["b2"]+params["prior"][layer,channel_indices][None,:,:]


class Prepared:
    def __init__(self,model,device="cpu"):
        _schema(model);model._frozen()
        self.mode,self.metadata=model.mode,producer._json(model.metadata)
        self.device=torch.empty(0,device=device).device
        self.params={n:torch.tensor(a.copy(),device=self.device) for n,a in model.params.items()}
        self.exponent_centers=model.params["exponent_centers"].astype(np.uint8,copy=True)
        self.exponent_centers.flags.writeable=False
        self.model_id,self.serialized_bytes=model.model_id,model.state_bytes
        self.execution_contract=_runtime(self.device)
        self.state_id=hashlib.sha256(_canonical(dict(model_id=self.model_id,runtime=self.execution_contract))).hexdigest()
        self.parameter_count=model.parameter_count
        self.last_feature_diagnostics={}

    @property
    def physical_nbytes(self):
        return self.exponent_centers.nbytes+sum(t.numel()*t.element_size() for t in self.params.values())

    @property
    def state_bytes(self):
        return max(self.serialized_bytes,self.physical_nbytes)

    def _check(self,layer,start,tokens):
        if _canonical(_runtime(self.device))!=_canonical(self.execution_contract):
            raise ValueError("syndrome predictor execution contract changed")
        layer,start,tokens=_int(layer,"layer",0),_int(start,"start",0),_int(tokens,"tokens")
        if layer>=self.metadata["geometry"]["layers"]:raise ValueError("syndrome layer outside model")
        return layer,start,tokens

    def feature_diagnostics(self,features):
        f=np.asarray(features)
        if f.dtype!=np.uint16 or f.ndim!=2:raise ValueError("syndrome feature words must be uint16[T,rank]")
        numeric=(f.astype(np.uint32)<<16).view(np.float32)
        return dict(tokens=len(f),rank=f.shape[1],feature_bytes=f.nbytes,finite=bool(np.isfinite(numeric).all()),
            zero_words=int(np.sum((f&0x7fff)==0)),unique_word_count=int(np.unique(f).size),
            maximum_absolute_value=float(np.max(np.abs(numeric))) if f.size else 0.,
            nonfinite_words=int(np.sum(~np.isfinite(numeric))))

    @torch.inference_mode()
    def encode_features(self,record):
        target=joint(record)
        layer,start,tokens=self._check(record.get("layer"),record.get("start"),len(target))
        g=self.metadata["geometry"]
        if target.shape!=(tokens,2*g["channels"]) or record.get("tokens")!=tokens:
            raise ValueError("syndrome encoder target geometry mismatch")
        if self.mode=="none":
            result=np.empty((tokens,0),np.uint16);self.last_feature_diagnostics=self.feature_diagnostics(result)
            return result
        source=source_values(record,self.mode,self.params["rope_inv_freq"].cpu().numpy(),
            self.metadata["config"]["rope_scaling"],g["head_dim"])
        if source.shape!=(tokens,g["source_width"]):raise ValueError("syndrome producer source geometry mismatch")
        with torch.autocast(device_type=self.device.type,enabled=False):
            value=_encode(self.params,torch.tensor(source.copy(),device=self.device),layer,straight_through=False)
        if not bool(torch.isfinite(value).all()):raise ValueError("nonfinite syndrome retained code")
        result=value.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy()
        self.last_feature_diagnostics=dict(self.feature_diagnostics(result),
            latent_clamped_values=int(torch.sum(torch.abs(value)>=32.).cpu()),source_width=source.shape[1],
            source_min=float(np.min(source)),source_max=float(np.max(source)),
            normalization_scale_recomputation=self.mode=="hidden_norm")
        return result

    @torch.inference_mode()
    def predict_logits(self,layer,feature_words,start,known_words,residue_bits):
        known=_words(known_words,"known_words")
        b=_int(residue_bits,"residue_bits",0)
        if b>7:raise ValueError("residue_bits must be in 0..7")
        layer,start,tokens=self._check(layer,start,len(known))
        g=self.metadata["geometry"];features=np.asarray(feature_words)
        if (known.shape!=(tokens,2*g["channels"]) or features.dtype!=np.uint16
                or features.shape!=(tokens,g["rank"])):raise ValueError("syndrome prediction input geometry mismatch")
        numeric=(features.astype(np.uint32)<<16).view(np.float32)
        if not np.isfinite(numeric).all():raise ValueError("nonfinite syndrome packet code")
        f=torch.tensor(numeric.copy(),device=self.device)
        rows=max(1,PREDICT_WORD_CHUNK//known.shape[1]);outputs=[]
        with torch.autocast(device_type=self.device.type,enabled=False):
            for offset in range(0,tokens,rows):
                words=torch.tensor(known[offset:offset+rows].copy(),device=self.device)
                logits=_forward(self.params,g,layer,f[offset:offset+rows],start+offset,words,b)
                if not bool(torch.isfinite(logits).all()):raise ValueError("nonfinite syndrome prediction logits")
                outputs.append(logits.cpu().numpy().copy())
        return np.concatenate(outputs,axis=0)

    def predict_bits(self,layer,feature_words,start,known_words,residue_bits):
        return (self.predict_logits(layer,feature_words,start,known_words,residue_bits)>=0).astype(np.uint8)


def training_schedule(grouped, layers, columns, steps, batch_tokens, channels_per_step, seed, sampling="random"):
    """Yield reproducible TRAIN addresses; random exactly preserves the v1 RNG.

    Balanced traverses one independent shuffled channel permutation per layer.
    A wrapped permutation puts channels already in the current minibatch last,
    so each epoch visits every channel and each minibatch contains no repeats.
    We still consume the legacy random-channel draw: source/window/residue
    draws are therefore identical between samplers and across duration prefixes.
    """
    if sampling not in ("random", "balanced"):
        raise ValueError("sampling must be random or balanced")
    columns, layers = _int(columns,"columns"), _int(layers,"layers")
    size=min(_int(channels_per_step,"channels_per_step"),columns)
    rng=np.random.default_rng(_int(seed,"seed",0))
    channel_rng={layer:np.random.default_rng(np.random.SeedSequence([seed,0x434f5345,layer]))
                 for layer in range(layers)}
    queues={layer:np.empty(0,dtype=np.int64) for layer in range(layers)}
    for update in range(1,_int(steps,"steps")+1):
        layer=(update-1)%layers
        ri=int(rng.integers(len(grouped[layer])));capture_index,record=grouped[layer][ri]
        count=min(_int(batch_tokens,"batch_tokens"),int(record["tokens"]))
        offset=int(rng.integers(record["tokens"]-count+1))
        legacy=np.sort(rng.choice(columns,size=size,replace=False))
        if sampling=="random":indices=legacy
        else:
            chosen=[]
            while len(chosen)<size:
                if not len(queues[layer]):
                    permutation=channel_rng[layer].permutation(columns)
                    if chosen:
                        used=np.isin(permutation,chosen)
                        permutation=np.concatenate((permutation[~used],permutation[used]))
                    queues[layer]=permutation
                take=min(size-len(chosen),len(queues[layer]))
                chosen.extend(queues[layer][:take].tolist())
                queues[layer]=queues[layer][take:]
            indices=np.sort(np.asarray(chosen,dtype=np.int64))
        b=int(BUDGETS[int(rng.integers(len(BUDGETS)))])
        yield dict(update=update,layer=layer,capture=int(capture_index),record=ri,offset=offset,
            start=int(record["start"])+offset,tokens=count,residue_bits=b,channel_indices=indices.tolist())


def parameter_sha256(params):
    """Canonical numerical-state hash, independent of manifest/training logs."""
    digest=hashlib.sha256()
    for name,value in sorted(params.items()):
        array=value.detach().cpu().contiguous().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        array=np.ascontiguousarray(array)
        digest.update(_canonical(dict(name=name,shape=list(array.shape),dtype=array.dtype.str)))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _coverage_summary(counts,words,omitted):
    def summary(a):
        return dict(slots=int(a.size),zero_slots=int(np.sum(a==0)),minimum=int(a.min()),maximum=int(a.max()),
            mean=float(a.mean()),percentiles=dict(zip(("p25","p50","p75","p95"),np.percentile(a,[25,50,75,95]).tolist())))
    histogram=np.unique(counts,return_counts=True)
    return dict(optimizer_layer_channel=summary(counts),word_layer_channel=summary(words),
        optimizer_update_histogram={str(int(n)):int(c) for n,c in zip(*histogram)},
        by_layer={str(layer):dict(optimizer=summary(counts[layer]),words=summary(words[layer])) for layer in range(len(counts))},
        omitted_word_examples_by_bitplane={str(bit):summary(omitted[...,bit]) for bit in range(7)},
        interpretation="Optimizer exposure only. Initial TRAIN priors and source normalizers use all selected TRAIN captures; shared parameters may generalize to channels absent from optimizer minibatches.")


def fit(training_record_sets,output_dir,*,training_metadata,model_identity,inverse,rope_scaling,head_dim,
        rank=8,width=32,steps=1000,batch_tokens=32,channels_per_step=64,device="cpu",seed=71,modes=MODES,
        sampling="random",lr_horizon=None,diagnostic_callback=None,diagnostic_updates=None,diagnostics_out=None):
    """Fit TRAIN-only omitted-bit BCE with identical mode schedules.

    Optional fixed-DEV diagnostics are read-only observers, never optimizer
    inputs or checkpoint selectors. Their detailed output stays outside the
    deployed model manifest via ``diagnostics_out``.
    """
    rank,width,steps,batch_tokens,channels_per_step=(_int(v,n) for v,n in
        ((rank,"rank"),(width,"width"),(steps,"steps"),(batch_tokens,"batch_tokens"),(channels_per_step,"channels_per_step")))
    head_dim,seed=_int(head_dim,"head_dim"),_int(seed,"seed",0)
    horizon=steps if lr_horizon is None else _int(lr_horizon,"lr_horizon")
    if horizon<steps or sampling not in ("random","balanced"):
        raise ValueError("invalid syndrome sampling or LR horizon")
    if diagnostic_callback is not None and not callable(diagnostic_callback):
        raise ValueError("diagnostic_callback must be callable")
    diagnostic_points={0,steps} if diagnostic_updates is None else {0,steps,*(_int(v,"diagnostic update",0) for v in diagnostic_updates)}
    if any(v>steps for v in diagnostic_points):raise ValueError("diagnostic update exceeds TRAIN steps")
    if diagnostics_out is not None and not isinstance(diagnostics_out,dict):
        raise ValueError("diagnostics_out must be a dictionary")
    if diagnostic_callback is not None and diagnostics_out is None:
        raise ValueError("diagnostic callback requires external diagnostics_out")
    modes=tuple(modes)
    if head_dim%2 or not modes or len(set(modes))!=len(modes) or any(m not in MODES for m in modes):
        raise ValueError("invalid syndrome geometry/modes")
    provenance_rows,identity=producer._json(training_metadata),producer._json(model_identity)
    if (not isinstance(training_record_sets,(list,tuple)) or not training_record_sets
            or not isinstance(provenance_rows,list) or len(provenance_rows)!=len(training_record_sets)
            or any(not isinstance(row,dict) or row.get("split")!="train" for row in provenance_rows)
            or not isinstance(identity,(str,dict)) or not identity):
        raise ValueError("syndrome fitting requires explicit TRAIN captures and source identity")
    before=[producer._source_hash(rows) for rows in training_record_sets]
    provenance=producer._source_hash(dict(metadata=provenance_rows,identity=identity))
    grouped,channels,hidden_size={},None,None
    for capture_index,records in enumerate(training_record_sets):
        if not isinstance(records,(list,tuple)) or not records:raise ValueError("syndrome TRAIN capture must be nonempty")
        seen,intervals=set(),{}
        for record in records:
            if not isinstance(record,dict) or record.get("split","train")!="train":
                raise ValueError("syndrome source records must be TRAIN only")
            layer,start,tokens=_int(record.get("layer"),"layer",0),_int(record.get("start"),"start",0),_int(record.get("tokens"),"tokens")
            target,hidden=joint(record),_words(record.get("hidden"),"hidden")
            neural._floats(target);neural._floats(hidden)
            if channels is None:channels,hidden_size=target.shape[1]//2,hidden.shape[1]
            if target.shape!=(tokens,2*channels) or hidden.shape!=(tokens,hidden_size) or record.get("head_dim")!=head_dim:
                raise ValueError("syndrome TRAIN record geometry mismatch")
            if any(start<end and begin<start+tokens for begin,end in intervals.get(layer,[])):
                raise ValueError("syndrome TRAIN pages overlap within a capture")
            intervals.setdefault(layer,[]).append((start,start+tokens))
            grouped.setdefault(layer,[]).append((capture_index,record));seen.add(layer)
        if capture_index==0:expected_layers=seen
        elif seen!=expected_layers:raise ValueError("syndrome every TRAIN capture must cover every layer")
    layers=len(grouped)
    if set(grouped)!=set(range(layers)):raise ValueError("syndrome TRAIN layers must be contiguous")
    inverse=np.asarray(inverse)
    if (channels%head_dim or inverse.dtype!=np.float32 or inverse.shape!=(head_dim//2,)
            or not np.isfinite(inverse).all() or np.any(inverse<=0) or isinstance(rope_scaling,bool)
            or not isinstance(rope_scaling,(int,float)) or not np.isfinite(rope_scaling) or rope_scaling<=0):
        raise ValueError("invalid syndrome fixed RoPE")
    architecture=identity.get("config",identity) if isinstance(identity,dict) else {}
    for name,actual in (("num_hidden_layers",layers),("hidden_size",hidden_size),("head_dim",head_dim)):
        if name in architecture and architecture[name]!=actual:raise ValueError("syndrome source architecture mismatch: "+name)
    if "num_key_value_heads" in architecture and architecture["num_key_value_heads"]*head_dim!=channels:
        raise ValueError("syndrome source head count mismatch")
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    if any((output/f"syndrome_{m}.{ext}").exists() for m in modes for ext in ("json","bin")):
        raise FileExistsError("syndrome model artifacts already exist")
    target_device=torch.empty(0,device=device).device
    centers=[];priors=[]
    for layer in range(layers):
        words=np.concatenate([joint(r) for _,r in grouped[layer]])
        centers.append(np.median((words>>7)&255,axis=0).astype(np.uint8))
        probabilities=np.stack([np.mean((words>>j)&1,axis=0) for j in range(7)],axis=-1).clip(.01,.99)
        priors.append(np.log(probabilities/(1-probabilities)).astype(np.float32))
    centers=np.stack(centers).astype(np.float32);initial_prior=np.stack(priors);results={}
    for mode in modes:
        began=time.perf_counter()
        source_width=0 if mode=="none" else (2*channels if mode in ("kv","kv_canonical") else
            hidden_size+(channels//head_dim if mode=="hidden_norm" else 0))
        geometry=dict(layers=layers,channels=channels,hidden_size=hidden_size,head_dim=head_dim,
            rank=0 if mode=="none" else rank,decoder_rank=rank,width=width,source_width=source_width)
        arrays=dict(rope_inv_freq=inverse.copy(),exponent_centers=centers.copy())
        if mode!="none":
            means=[];scales=[]
            for layer in range(layers):
                source=np.concatenate([source_values(r,mode,inverse,rope_scaling,head_dim) for _,r in grouped[layer]])
                if not np.isfinite(source).all():raise ValueError("nonfinite syndrome TRAIN source")
                means.append(source.mean(axis=0,dtype=np.float64).astype(np.float32))
                scales.append((source.astype(np.float64).std(axis=0)+1e-6).astype(np.float32))
            arrays.update(source_mu=np.stack(means),source_sd=np.stack(scales))
        constants={n:torch.tensor(a.copy(),device=target_device) for n,a in arrays.items()}
        generator=torch.Generator(device=target_device).manual_seed(seed);network={}
        for name,shape in _network_shapes(geometry,mode).items():
            if name=="prior":value=torch.tensor(initial_prior.copy(),device=target_device)
            elif name.startswith("b") or name=="encoder_b" or name=="W2":value=torch.zeros(shape,device=target_device)
            else:
                scale=.05 if "embedding" in name else 1/np.sqrt(shape[0])
                value=torch.randn(shape,device=target_device,generator=generator)*scale
            network[name]=torch.nn.Parameter(value)
        params=dict(constants,**network)
        optimizer=torch.optim.Adam(list(network.values()),lr=1e-3)
        checkpoints=[];recent=[]
        updates={str(l):0 for l in range(layers)};token_examples=dict.fromkeys(updates,0);word_examples=dict.fromkeys(updates,0)
        mask_counts=np.zeros(7,np.int64);bit_errors=np.zeros(7,np.int64);b_counts={str(b):0 for b in BUDGETS}
        schedule_hash=hashlib.sha256();first_loss=None
        channel_updates=np.zeros((layers,2*channels),dtype=np.int64)
        channel_words=np.zeros_like(channel_updates)
        omitted_words=np.zeros((layers,2*channels,7),dtype=np.int64)
        initial_parameter_hash=parameter_sha256(params)
        reference_prior=initial_prior.copy();reference_prior.flags.writeable=False
        training_reference=dict(prior_logits=reference_prior)
        reference_hash=parameter_sha256(training_reference)
        diagnostic_seconds=0.
        external=dict(checkpoints=[])
        if diagnostics_out is not None:
            if mode in diagnostics_out:raise ValueError("diagnostics_out already contains mode "+mode)
            diagnostics_out[mode]=external
        def observe(update):
            nonlocal diagnostic_seconds
            if diagnostic_callback is None or update not in diagnostic_points:return
            began_diagnostic=time.perf_counter()
            before_hash=parameter_sha256(params);before_geometry=_canonical(geometry)
            with torch.inference_mode():
                metrics=diagnostic_callback(mode=mode,update=update,params=params,geometry=geometry,
                    training_reference=training_reference)
            if (before_hash!=parameter_sha256(params) or before_geometry!=_canonical(geometry)
                    or reference_hash!=parameter_sha256(training_reference)):
                raise ValueError("read-only diagnostic callback changed TRAIN parameters, geometry or reference")
            metrics=producer._json(metrics);_canonical(metrics)
            elapsed=time.perf_counter()-began_diagnostic;diagnostic_seconds+=elapsed
            external["checkpoints"].append(dict(update=update,metrics=metrics,parameter_sha256=before_hash,
                sampling_prefix_sha256=schedule_hash.hexdigest(),
                coverage_summary=_coverage_summary(channel_updates,channel_words,omitted_words),diagnostic_seconds=elapsed))
        observe(0)
        for schedule in training_schedule(grouped,layers,2*channels,steps,batch_tokens,channels_per_step,seed,sampling):
            update,layer,ri,offset,count,b,start=(schedule[k] for k in
                ("update","layer","record","offset","tokens","residue_bits","start"))
            capture_index,record=grouped[layer][ri]
            indices=np.asarray(schedule["channel_indices"],dtype=np.int64)
            schedule_hash.update(_canonical(schedule))
            words=joint(record)[offset:offset+count][:,indices].copy()
            known=torch.tensor(words.copy(),device=target_device);ci=torch.tensor(indices,device=target_device,dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=target_device.type,enabled=False):
                if mode=="none":features=torch.empty((count,0),device=target_device)
                else:
                    source=source_values(record,mode,inverse,rope_scaling,head_dim)[offset:offset+count]
                    features=_encode(params,torch.tensor(source.copy(),device=target_device),layer,straight_through=True)
                logits=_forward(params,geometry,layer,features,start,known,b,ci)
                bits=torch.stack([((known.to(torch.int64)>>j)&1).float() for j in range(7)],dim=-1)
                perbit=F.binary_cross_entropy_with_logits(logits,bits,reduction="none")/np.log(2.)
                loss=perbit[...,b:].mean()
            if not bool(torch.isfinite(loss)):raise ValueError("nonfinite syndrome TRAIN BCE")
            value=float(loss.detach().cpu());recent.append(value);recent=recent[-100:]
            if first_loss is None:first_loss=value
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in network.values()):
                raise ValueError("nonfinite syndrome TRAIN gradient")
            rate=float(1e-3*.5*(1+np.cos(np.pi*update/(horizon+1))))
            for group in optimizer.param_groups:group["lr"]=rate
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in network.values()):
                raise ValueError("nonfinite syndrome TRAIN parameter")
            updates[str(layer)]+=1;token_examples[str(layer)]+=count;word_examples[str(layer)]+=words.size;b_counts[str(b)]+=1
            channel_updates[layer,indices]+=1;channel_words[layer,indices]+=count;omitted_words[layer,indices,b:]+=count
            with torch.no_grad():
                errors=((logits>=0)!=(bits>=.5)).sum(dim=(0,1)).cpu().numpy()
                bit_errors[b:]+=errors[b:];mask_counts[b:]+=words.size
            if update==1 or update==steps or update%max(1,steps//8)==0 or update in diagnostic_points:
                losses=perbit.mean(dim=(0,1)).detach().cpu().tolist()
                checkpoints.append(dict(**schedule,channels=len(indices),
                    channel_indices_sha256=hashlib.sha256(indices.astype("<i4").tobytes()).hexdigest(),
                    omitted_bit_BCE_bits=value,BCE_by_bitplane=losses,bit_errors_by_bitplane=errors.tolist(),
                    omitted_bitplanes=list(range(b,7)),learning_rate=rate,
                    parameter_sha256=parameter_sha256(params),sampling_prefix_sha256=schedule_hash.hexdigest()))
            observe(update)
        coverage=dict(optimizer_updates_by_layer_channel=channel_updates.tolist(),
            word_examples_by_layer_channel=channel_words.tolist(),
            summary=_coverage_summary(channel_updates,channel_words,omitted_words))
        external["coverage"]=coverage
        final_parameter_hash=parameter_sha256(params)
        arrays.update({n:p.detach().cpu().numpy().astype(np.float32,copy=True) for n,p in network.items()})
        metadata=dict(format=FORMAT,mode=mode,training_only=True,model_identity=identity,geometry=geometry,
            training_hashes=before,training_captures=[dict(split="train",sha256=digest,metadata=row)
                for digest,row in zip(before,provenance_rows)],
            config=dict(latent_dtype="BF16",latent_clamp=[-32.,32.],activation="GELU approximate=tanh",
                threshold="logit >= 0",rope_scaling=float(rope_scaling),training_residue_bits=list(BUDGETS),
                objective="BCE in bits over omitted mantissa bitplanes; surrogate, not joint exact syndrome/escape packet rate. DEV actual bytes select recipe.",
                source={"none":"no retained code; exact fields, layer/channel identity and position",
                    "kv":"actual produced final K,V",
                    "kv_canonical":"inverse-RoPE canonicalized produced final K plus V, BF16 rounded; no producer hook",
                    "hidden":"actual normalized input_layernorm hidden activations",
                    "hidden_norm":"actual normalized hidden plus per-head rsqrt recomputed in CPU FP32 from hooked BF16 pre-normalization K projection and actual epsilon; not a directly reused kernel accumulator"}[mode],
                code_rate_bits_per_KV_value=16*geometry["rank"]/(2*channels),
                known_fields="exact sign/exponent and low b mantissa bits; high bits masked internally before every prediction",
                position_features="actual frozen split-half RoPE inverse frequency per K coordinate; V neutral sine/cosine (0,1); zero per-page bytes",
                dependencies="own page code+exact fields and frozen shared model only; no original activations or other pages",
                inference_word_chunk=PREDICT_WORD_CHUNK,
                normalizers="TRAIN per-layer mean/std accumulated float64, +1e-6; centers TRAIN per-channel median exponent rounded down"),
            training=dict(seed=seed,total_optimizer_updates=steps,batch_tokens=batch_tokens,channels_per_step=channels_per_step,
                updates_by_layer=updates,total_token_examples=sum(token_examples.values()),token_examples_by_layer=token_examples,
                total_word_examples=sum(word_examples.values()),word_examples_by_layer=word_examples,
                omitted_bit_examples_by_plane=mask_counts.tolist(),bit_errors_by_plane=bit_errors.tolist(),residue_bits_update_counts=b_counts,
                sampling="round-robin layers; identical seeded capture/window/channel/residue schedule across modes",
                channel_sampling=sampling,
                channel_sampling_rule=("legacy independent choice without replacement within each minibatch" if sampling=="random" else
                    "independent seeded per-layer channel permutations; full cycles without replacement; wrap reorders next cycle to avoid within-minibatch duplicates; legacy channel RNG draws consumed so source/window/residue addresses match random"),
                learning_rate_horizon=horizon,initial_parameter_sha256=initial_parameter_hash,final_parameter_sha256=final_parameter_hash,
                coverage_sha256=hashlib.sha256(_canonical(coverage)).hexdigest(),
                coverage_summary={k:v for k,v in coverage["summary"].items() if k!="by_layer"},
                sampling_sha256=schedule_hash.hexdigest(),initial_batch_loss=first_loss,final_batch_loss=recent[-1],
                final_mean_last_100_batch_losses=float(np.mean(recent)),checkpoints=checkpoints,
                optimizer="Adam defaults; cosine LR 1e-3*0.5*(1+cos(pi*update/(learning_rate_horizon+1)))",
                diagnostic_contract="Read-only observer at explicit updates; no DEV gradients, early stopping, checkpoint selection or optimizer mutation. Diagnostic metrics/full coverage are external to deployment state.",
                diagnostic_wall_seconds=diagnostic_seconds,
                runtime=_runtime(target_device),fit_seconds=time.perf_counter()-began),
            scope="Functional learned bit predictor; all encoder/decoder/normalizer/center/state bytes charged. Same recorded backend contract only; no cross-device portable replay or GPU syndrome throughput claim.")
        model=Model(mode,arrays,metadata)
        metadata["compute"]=dict(trainable_parameter_count=model.parameter_count,complete_float32_state_values=model.total_float32_values,
            encoder_only_float32_values=sum(a.size for n,a in arrays.items() if n in ("encoder_W","encoder_b","source_mu","source_sd")),
            encoder_MACs_per_token=source_width*geometry["rank"],
            decoder_MACs_per_word=params["W0"].shape[0]*width+width*width+width*7,inference_word_chunk=PREDICT_WORD_CHUNK)
        save(model,output/f"syndrome_{mode}.json")
        results[mode]=load(model.path,expected_model_identity=identity,expected_training_hashes=before,
            expected_model_id=model.model_id,expected_mode=mode)
        del optimizer,network,params,constants
    if before!=[producer._source_hash(rows) for rows in training_record_sets] or provenance!=producer._source_hash(
            dict(metadata=producer._json(training_metadata),identity=producer._json(model_identity))):
        raise AssertionError("syndrome TRAIN source/provenance changed")
    return results


