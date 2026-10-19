"""Group-adaptive learned/exponent/raw exact KV records (LKVC version 1).

Existing five-byte group headers carry the selector; no extra selector byte.
The caller supplies records of the same source matrix. Shared learned model
bytes remain external and must be counted by the caller, once per resident model.
"""
from collections import Counter
from dataclasses import dataclass
import struct
import numpy as np
from kv_research import learned_codec as learned
from kv_smoke import codec as baseline

_HEADER = learned._RECORD_HEADER
_GROUP = struct.Struct('<BI')
_MAGIC = b'LKVC'

@dataclass
class EncodedMatrix:
    data: bytes
    stats: dict

def _data(value):
    return value.data if hasattr(value, 'data') else bytes(value)

def _groups(data, offset, total, group_size, allowed):
    if group_size < 8 or group_size % 8:
        raise ValueError('invalid group size')
    if (total + group_size - 1) // group_size > (len(data) - offset) // _GROUP.size:
        raise ValueError('truncated group headers')
    groups = []
    for start in range(0, total, group_size):
        if offset + _GROUP.size > len(data):
            raise ValueError('truncated group header')
        mode, length = _GROUP.unpack_from(data, offset)
        offset += _GROUP.size
        if mode not in allowed or offset + length > len(data):
            raise ValueError('invalid group mode or length')
        payload = data[offset:offset + length]
        offset += length
        if mode == 0 and length != 2 * min(group_size, total - start):
            raise ValueError('incorrect raw group size')
        groups.append((mode, payload))
    if offset != len(data):
        raise ValueError('unexpected trailing bytes')
    return groups

def encode_matrix(words_u16, learned_record, exponent_record, e8_record=None):
    """Select shortest payload independently per shared group, ties favor learned."""
    if e8_record is not None:
        raise ValueError('this version supports learned, exponent, and raw only')
    words = np.asarray(words_u16)
    if words.dtype != np.uint16 or words.ndim != 2:
        raise ValueError('expected uint16 matrix')
    source = _data(learned_record)
    exp = _data(exponent_record)
    if len(source) < _HEADER.size or len(exp) < baseline._HEADER.size:
        raise ValueError('truncated input record')
    fields = list(_HEADER.unpack_from(source))
    magic, version, qbits, flags, size, rows, columns, identity, keylen = fields
    if magic != b'LKVR' or version != 1 or flags or qbits > 16 or rows != words.shape[0] or columns != words.shape[1]:
        raise ValueError('invalid learned source header')
    name_end = _HEADER.size + keylen
    if name_end > len(source):
        raise ValueError('truncated dictionary name')
    emagic, eversion, method, reserved, esize, erows, ecolumns = baseline._HEADER.unpack_from(exp)
    if (emagic, eversion, method, reserved, esize, erows, ecolumns) != (b'KVRC', 1, 1, 0, size, rows, columns):
        raise ValueError('exponent record shape, method, or grouping mismatch')
    lgroups = _groups(source, name_end, words.size, size, {0, 1})
    egroups = _groups(exp, baseline._HEADER.size, words.size, size, {0, 1})
    fields[0] = _MAGIC
    output = bytearray(_HEADER.pack(*fields) + source[_HEADER.size:name_end])
    components = Counter({'matrix_header_and_model_identity': len(output)})
    modes = Counter()
    flat = np.ascontiguousarray(words).reshape(-1)
    for index, (lg, eg) in enumerate(zip(lgroups, egroups)):
        start = index * size
        raw = flat[start:start + size].astype('<u2').tobytes()
        choices = [lg, (2 if eg[0] == 1 else 0, eg[1]), (0, raw)]
        mode, payload = min(choices, key=lambda item: len(item[1]))
        output.extend(_GROUP.pack(mode, len(payload)))
        output.extend(payload)
        modes[str(mode)] += 1
        components['group_headers_with_mode_selector'] += _GROUP.size
        components[{0:'raw_payload', 1:'learned_payload', 2:'exponent_payload'}[mode]] += len(payload)
    assert sum(components.values()) == len(output)
    return EncodedMatrix(bytes(output), {
        'serialized_bytes': len(output), 'raw_payload_bytes': int(words.nbytes),
        'mode_counts': dict(modes), 'component_bytes': dict(components),
        'learned_source_bytes': len(source), 'exponent_source_bytes': len(exp),
        'bytes_saved_vs_learned_source': len(source) - len(output),
        'group_size': size, 'selector_extra_bytes': 0,
    })

def decode_matrix(encoded, model, native_decoder=None):
    """Decode LKVC; optional frozen NativeDecoder accelerates learned groups."""
    data = _data(encoded)
    if len(data) < _HEADER.size:
        raise ValueError('truncated hybrid matrix')
    fields = list(_HEADER.unpack_from(data))
    magic, version, qbits, flags, size, rows, columns, identity, keylen = fields
    if magic != _MAGIC or version != 1 or flags or qbits != (model.n_centroids - 1).bit_length():
        raise ValueError('unsupported hybrid header')
    if identity.hex() != model.model_id:
        raise ValueError('learned model SHA256 mismatch')
    offset = _HEADER.size + keylen
    if offset > len(data):
        raise ValueError('truncated dictionary name')
    try:
        key = data[_HEADER.size:offset].decode('utf8')
    except UnicodeDecodeError as error:
        raise ValueError('invalid dictionary name') from error
    if key not in model.codebooks:
        raise ValueError('unknown dictionary')
    groups = _groups(data, offset, rows * columns, size, {0, 1, 2})
    pieces = []
    for index, (mode, payload) in enumerate(groups):
        count = min(size, rows * columns - index * size)
        if mode == 0:
            values = np.frombuffer(payload, dtype='<u2').copy()
        elif mode == 1:
            local = fields.copy()
            local[0], local[5], local[6] = b'LKVR', 1, count
            record = _HEADER.pack(*local) + data[_HEADER.size:offset] + _GROUP.pack(1, len(payload)) + payload
            values = (native_decoder.decode_matrix(record) if native_decoder is not None else learned.decode_matrix(record, model)).reshape(-1)
        else:
            record = baseline._HEADER.pack(b'KVRC', 1, 1, 0, size, 1, count) + _GROUP.pack(1, len(payload)) + payload
            values = baseline.decode_matrix(record).reshape(-1)
        pieces.append(values)
    flat = np.concatenate(pieces) if pieces else np.empty(0, dtype=np.uint16)
    return flat.reshape(rows, columns)
