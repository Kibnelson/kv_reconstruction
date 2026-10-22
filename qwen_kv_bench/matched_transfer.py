"""Matched lossless GPU adapters with common, charged external framing.

Inputs and reconstructed outputs use original Hugging Face post-RoPE BF16
words, [layer, token, K||V], with split-half key coordinates. Predictive arms
pay for their internal adjacent-pair permutation in both timed directions.
The CPU only frames, checksums and moves compressed bytes.
"""
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np

from . import transfer_reference as ref, transfer_probability as prob, transfer_t1 as t1

FORMAT = 'qwen_matched_transfer_v1'
MAGIC = b'KVMTCH01'
HEADER = struct.Struct('<8sI32s')
SPLITZIP_COMMIT = '42c743118a29db3270ae6a921a4a4eecb20cad04'
NVCOMP_VERSION = '5.3.0.16'
# Hash of the unmodified upstream codec_gpu.py, filled/checked by setup.
SPLITZIP_SHA256 = 'e28fe3263e8906b26d59789cbc5bccf14edc8391b0dae986aafaf34adb2d392c'
DTYPES = {'uint8': np.dtype('u1'), 'uint16': np.dtype('<u2'),
          'int32': np.dtype('<i4'), 'int64': np.dtype('<i8'),
          'float32': np.dtype('<f4')}


def digest(value):
    return hashlib.sha256(prob.canonical(value)).hexdigest()


def key_order(words, head_dim, inverse=False):
    """Exact permutation between HF split halves and adjacent RoPE pairs."""
    channels = words.shape[-1] // 2
    if head_dim % 2 or channels % head_dim:
        raise ValueError('invalid key geometry')
    k = words[..., :channels].reshape(*words.shape[:-1], channels//head_dim, head_dim)
    last = (head_dim//2, 2) if inverse else (2, head_dim//2)
    k = k.reshape(*k.shape[:-1], *last).swapaxes(-1, -2).reshape(*words.shape[:-1], channels)
    if isinstance(words, np.ndarray):
        return np.ascontiguousarray(np.concatenate((k, words[..., channels:]), axis=-1))
    import torch
    return torch.cat((k, words[..., channels:]), dim=-1).contiguous()


def adjacent_capture(capture):
    words = key_order(ref.words(capture), int(capture['hd']))
    c = words.shape[-1]//2
    return dict(capture, K_bits=words[..., :c], V_bits=words[..., c:])


@dataclass
class Record:
    shape: tuple
    start: int
    parts: dict
    attributes: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)


def validate_record(record):
    if (len(record.shape) != 3 or any(type(x) is not int or x <= 0 for x in record.shape)
            or math.prod(record.shape) > 1 << 28 or type(record.start) is not int or record.start < 0):
        raise ValueError('invalid record geometry')


def export_record(adapter, record, model_identity):
    validate_record(record)
    adapter.check_encode(record)
    descriptions = []; data = []
    for name, tensor in record.parts.items():
        a = tensor.detach().contiguous().cpu().numpy()
        dtype = a.dtype.name
        if dtype not in DTYPES:
            raise ValueError('unsupported wire dtype')
        b = a.astype(DTYPES[dtype], copy=False).tobytes()
        descriptions.append(dict(name=name, dtype=dtype, shape=list(a.shape), bytes=len(b)))
        data.append(b)
    meta = prob.canonical(dict(format=FORMAT, method=adapter.name,
        model_identity_sha256=digest(model_identity), state_sha256=adapter.state_sha256,
        layout='HF_split_half_post_RoPE_K_then_V', shape=list(record.shape), start=record.start,
        attributes=record.attributes, parts=descriptions))
    body = meta+b''.join(data)
    wire = HEADER.pack(MAGIC, len(meta), hashlib.sha256(body).digest())+body
    return wire, dict(payload_bytes=sum(map(len, data)), framing_bytes=HEADER.size+len(meta), complete_bytes=len(wire))


def import_record(adapter, wire, model_identity):
    import torch
    if len(wire) < HEADER.size:
        raise ValueError('truncated matched packet')
    magic, n, checksum = HEADER.unpack_from(wire); body = wire[HEADER.size:]
    if magic != MAGIC or n > min(len(body), 1 << 24) or hashlib.sha256(body).digest() != checksum:
        raise ValueError('invalid matched packet/checksum')
    m = json.loads(body[:n])
    expected = dict(format=FORMAT, method=adapter.name, model_identity_sha256=digest(model_identity),
                    state_sha256=adapter.state_sha256, layout='HF_split_half_post_RoPE_K_then_V')
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError('wrong model, layout, method or decoder state')
    r = Record(tuple(m['shape']), m['start'], {}, m['attributes']); validate_record(r)
    offset = n
    for p in m['parts']:
        dtype = DTYPES.get(p['dtype']); shape = p['shape']
        if (dtype is None or not isinstance(shape, list) or len(shape)>8
                or any(type(x) is not int or x < 0 for x in shape)
                or math.prod(shape)*dtype.itemsize != p['bytes']
                or p['bytes'] < 0 or offset+p['bytes']>len(body) or p['name'] in r.parts):
            raise ValueError('invalid tensor directory')
        a = np.frombuffer(body, dtype=dtype, count=math.prod(shape), offset=offset).reshape(shape).copy()
        r.parts[p['name']] = torch.from_numpy(a).to(adapter.device)
        offset += p['bytes']
    if offset != len(body):
        raise ValueError('unexpected trailing packet bytes')
    return r


class Adapter:
    lossless = True

    def check_encode(self, record):
        if any(bool(x.any().item()) for x in record.errors):
            raise RuntimeError('GPU encoder reported an error')

    def check_decode(self):
        if any(bool(x.any().item()) for x in getattr(self, 'decode_errors', [])):
            raise RuntimeError('GPU decoder reported an error')

    def statistics(self):
        return dict(persistent_device_bytes=0, allocator_scope='Torch only; external native allocator peaks are not measured')


class Raw(Adapter):
    name = 'raw_bf16'

    def __init__(self, device):
        self.device = device; self.state_sha256 = digest(dict(method=self.name, version=1))

    def encode(self, words, start):
        return Record(tuple(words.shape), start, dict(raw=words.view(__import__('torch').uint8).clone()))

    def decode(self, record):
        import torch
        validate_record(record)
        return record.parts['raw'].view(torch.uint16).reshape(record.shape).clone()


class Predictive(Adapter):
    def __init__(self, name, base, state, device, *, _interpreter=False):
        self.name = name; self.device = device
        self.engine = t1.Engine(base, state, device, input_chunk=512, entropy_backend='cached', _interpreter=_interpreter)
        self.state_sha256 = self.engine.state_sha256

    def encode(self, words, start):
        adjacent = key_order(words, self.engine.base.geometry['head_dim'])
        r = Record(tuple(words.shape), start, {}, dict(chunks=[], raw_chunks=0))
        for i, begin in enumerate(range(0, words.shape[1], 32)):
            p = self.engine.encode(adjacent[:, begin:begin+32].contiguous(), start+begin)
            r.parts[f'{i}_payload'] = p.payload; r.parts[f'{i}_directory'] = p.directory
            r.attributes['chunks'].append(dict(shape=list(p.shape), start=p.start, raw=p.raw))
            r.attributes['raw_chunks'] += int(p.raw); r.errors.append(p.encoder_error)
        return r

    def decode(self, record):
        import torch
        validate_record(record); output = []; self.decode_errors = []; offset = 0
        chunks = record.attributes['chunks']
        if len(chunks) != (record.shape[1]+31)//32 or len(record.parts) != 2*len(chunks):
            raise ValueError('invalid predictive chunk directory')
        for i, m in enumerate(chunks):
            shape = (record.shape[0], min(32, record.shape[1]-offset), record.shape[2])
            if m['shape'] != list(shape) or m['start'] != record.start+offset or type(m['raw']) is not bool:
                raise ValueError('invalid causal chunk geometry')
            p = t1.Packet(record.parts[f'{i}_payload'], record.parts[f'{i}_directory'], shape,
                          m['start'], self.state_sha256, m['raw'])
            words, error, _ = self.engine.decode(p)
            output.append(words); self.decode_errors.append(error); offset += shape[1]
        return key_order(torch.cat(output, dim=1), self.engine.base.geometry['head_dim'], inverse=True)

    def statistics(self):
        return dict(self.engine.statistics(), layout_permutation='included in encode and decode',
                    chunk_tokens=32, predictor_mode=self.engine.base.mode, group_channels=self.engine.base.block_size,
                    mean_state_sha256=self.engine.base.state_sha256, width_arm=self.engine.adapter.arm)


def load_splitzip(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPLITZIP_SHA256:
        raise ValueError('SplitZip source differs from pinned upstream version')
    name = '_matched_upstream_splitzip'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class SplitZip(Adapter):
    name = 'splitzip_native'

    def __init__(self, device, source, calibration=None, state_path=None):
        import torch
        self.device = device; self.module = load_splitzip(source)
        self.codec = self.module.ChunkLocalSplitZipGPU(device=str(device), chunk_size=1024)
        names = ('enc_lut', 'enc_lut_marked', 'dec_lut', 'common_lut')
        if calibration is not None:
            self.codec.calibrate(calibration.contiguous().view(torch.bfloat16))
            arrays = {n: getattr(self.codec,n).cpu().numpy() for n in names}
            if state_path is not None:
                np.savez(state_path, **arrays)
        else:
            with np.load(state_path, allow_pickle=False) as a:
                arrays = {n:a[n].copy() for n in names}
            for n, a in arrays.items():
                setattr(self.codec, n, torch.from_numpy(a).to(device))
        self.state_sha256 = prob._array_hash(dict(method=self.name, commit=SPLITZIP_COMMIT, chunk_size=1024), arrays)

    def encode(self, words, start):
        import torch
        p = self.codec.encode(words.reshape(-1).view(torch.bfloat16))
        parts = {n:getattr(p,n) for n in ('pk','sm','chunk_id','local_pos','esc_val')}
        return Record(tuple(words.shape), start, parts, dict(n=p.n, n_esc=p.n_esc, chunk_size=p.chunk_size))

    def decode(self, record):
        import torch
        validate_record(record); a = record.attributes; n = math.prod(record.shape); e = a['n_esc']
        sizes = dict(pk=(n+1)//2, sm=n, chunk_id=e, local_pos=e, esc_val=e)
        types = dict(pk=torch.uint8, sm=torch.uint8, chunk_id=torch.int32, local_pos=torch.uint16, esc_val=torch.uint8)
        if a['n'] != n or a['chunk_size'] != 1024 or not 0 <= e <= n or set(record.parts) != set(sizes):
            raise ValueError('invalid SplitZip metadata')
        for name,size in sizes.items():
            if record.parts[name].numel()!=size or record.parts[name].dtype!=types[name]:
                raise ValueError('invalid SplitZip buffers')
        # Bounds are checked before entering the unmodified native escape kernel.
        if e:
            pos = record.parts['chunk_id'].to(torch.int64)*1024+record.parts['local_pos'].to(torch.int64)
            if bool(((pos<0)|(pos>=n)).any().item()):raise ValueError('invalid escape position')
        empty = torch.empty(0,dtype=torch.int32,device=self.device)
        p = self.module.ChunkLocalGPUEncoded(**record.parts, counts=empty, starts=empty, **a)
        return self.codec.decode(p).view(torch.uint16).reshape(record.shape)

    def statistics(self):
        return dict(persistent_device_bytes=784, upstream_commit=SPLITZIP_COMMIT,
            escape_payload_bytes=7, upstream_codec_unmodified=True, calibration='TRAIN only',
            allocator_scope='Torch allocations, including native Triton scratch')


class Nvcomp(Adapter):
    def __init__(self, algorithm, device):
        import torch
        import nvidia.nvcomp as nvcomp
        import importlib.metadata
        self.device = device; self.nvcomp = nvcomp; self.name = 'nvcomp_'+algorithm.lower()
        self.version = importlib.metadata.version('nvidia-nvcomp-cu12')
        if int(self.version.split('.')[0]) < 5 and int(np.__version__.split('.')[0]) >= 2:
            raise RuntimeError('nvCOMP 4.x misreads NumPy 2 dtype metadata; install the pinned nvCOMP '+NVCOMP_VERSION+' wrapper and runtime')
        self.stream = int(torch.cuda.current_stream(device).cuda_stream)
        self.codec = nvcomp.Codec(algorithm=algorithm, device_id=device.index or 0,
                                 cuda_stream=self.stream, data_type='<i2')
        self.state_sha256 = digest(dict(method=self.name, version=self.version, input='int16 raw BF16 words', lossless=True))

    def encode(self, words, start):
        import torch
        source = self.nvcomp.as_array(words.reshape(-1).view(torch.int16), cuda_stream=self.stream)
        if np.dtype(source.dtype) != np.dtype('<i2') or source.item_size != 2 or source.buffer_size != words.numel()*2:
            raise RuntimeError('nvCOMP input dtype/byte count differs from the original int16 word view')
        array = self.codec.encode(source)
        data = torch.from_dlpack(array).view(torch.uint8).reshape(-1)
        if data.numel()!=array.buffer_size:raise ValueError('nvCOMP array accounting mismatch')
        return Record(tuple(words.shape), start, dict(compressed=data))

    def decode(self, record):
        import torch
        validate_record(record)
        array = self.codec.decode(self.nvcomp.as_array(record.parts['compressed'], cuda_stream=self.stream), data_type='|u1')
        return torch.from_dlpack(array).view(torch.uint16).reshape(record.shape)

    def statistics(self):
        return dict(super().statistics(), persistent_device_bytes=None, library=self.version, lossless=True,
                    input_view='int16 raw BF16 words', stream='current Torch CUDA stream')
