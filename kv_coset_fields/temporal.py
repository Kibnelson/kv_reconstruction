"""Exact local temporal cosets for BF16 sign/fraction bytes.

The exponent tensor is external (the enclosing codec stores it exactly).
There is no trained state. A predictor uses only earlier decoded lower bytes
inside this record, or a fixed constant. For b bits, s=x mod 2**b and
decode(p,s,b)=(p+((s-p+2**(b-1)) mod 2**b)-2**(b-1)) mod 256.
For b=0 the result is p; for b=8 it is always x. Ties choose the negative
direction. The encoder explicitly verifies reconstruction and chooses the
smallest b for every vector. This is a circular symbol coset, not BF16 error.

Header: <magic,version,predictor,vector_size,flags,rows,columns,group_size,
payload_length,exponent_crc32,lower_crc32>, then payload, then whole CRC32.
flags=1 is a raw lower tensor. flags=0 has <mode:uint8,length:uint32> groups;
mode0 is raw; mode1 has 4-bit vector widths followed by packed residues.
All byte counts include metadata/checksums. CRC32 detects accidental damage;
it is not an authentication mechanism. No record refers to another record.
"""
from dataclasses import dataclass
from collections import Counter
import struct
import zlib
import numpy as np

_HEADER = struct.Struct('<4sBBBBIIIIII')
_GROUP = struct.Struct('<BI')
_CRC = struct.Struct('<I')
_MAGIC = b'TLCS'
PREDICTORS = {'previous_token': 0, 'previous_channel': 1, 'zero': 2,
              'midpoint': 3, 'previous_token_scaled': 4}

@dataclass
class EncodedLower:
    data: bytes
    stats: dict

def _validate_arrays(lower, exponents):
    lower, exponents = np.asarray(lower), np.asarray(exponents)
    if lower.dtype != np.uint8 or exponents.dtype != np.uint8 or lower.ndim != 2 or exponents.shape != lower.shape:
        raise ValueError('expected matching uint8 [tokens,channels] arrays')
    return np.ascontiguousarray(lower), np.ascontiguousarray(exponents)

def _recover(prediction, residues, widths):
    p = np.asarray(prediction, dtype=np.int32)
    b = np.asarray(widths, dtype=np.int32)
    m = 1 << b
    half = np.where(b == 0, 0, m // 2)
    return ((p + ((np.asarray(residues, dtype=np.int32) - p + half) & (m - 1)) - half) & 255).astype(np.uint8)

def _scaled_previous(previous_lower, previous_exp, current_exp):
    # Clip the previous finite magnitude into the current exponent bin.
    # Equal exponents preserve the fraction; larger/smaller current exponents
    # clip it to the bin's lower/upper endpoint. This also handles exponent0
    # without floating-point arithmetic.
    lo = np.asarray(previous_lower, dtype=np.uint8)
    pe, ce = np.asarray(previous_exp, dtype=np.int32), np.asarray(current_exp, dtype=np.int32)
    predicted = np.where(ce > pe, 0, np.where(ce < pe, 127, lo & 127)).astype(np.uint8) | (lo & 128)
    # Exceptional exponent values are merely side information, not arithmetic.
    return np.where((pe == 255) | (ce == 255), lo, predicted).astype(np.uint8)

def _predict_all(lower, exponents, predictor):
    prediction = np.zeros_like(lower)
    if predictor == 'previous_token':
        prediction[1:] = lower[:-1]
    elif predictor == 'previous_channel':
        prediction[:, 1:] = lower[:, :-1]
    elif predictor == 'midpoint':
        prediction.fill(128)
    elif predictor == 'previous_token_scaled':
        prediction[1:] = _scaled_previous(lower[:-1], exponents[:-1], exponents[1:])
    return prediction

def _pack_nibbles(values):
    values = np.asarray(values, dtype=np.uint8)
    if len(values) % 2:
        values = np.r_[values, np.uint8(0)]
    return (values[::2] | (values[1::2] << 4)).tobytes()

def _unpack_nibbles(data, count):
    if len(data) != (count + 1) // 2:
        raise ValueError('incorrect width-table length')
    packed = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(len(packed) * 2, dtype=np.uint8)
    out[::2], out[1::2] = packed & 15, packed >> 4
    if count % 2 and out[-1] != 0:
        raise ValueError('nonzero width padding')
    if np.any(out[:count] > 8):
        raise ValueError('invalid coset width')
    return out[:count]

def _pack_variable(values, widths):
    values = np.asarray(values, dtype=np.uint8)
    shifts = np.arange(8, dtype=np.uint8)
    selected = ((values[:, None] >> shifts) & 1)[shifts[None, :] < widths[:, None]]
    return np.packbits(selected, bitorder='little').tobytes()

def _unpack_variable(data, widths):
    bitcount = int(widths.sum())
    if len(data) != (bitcount + 7) // 8:
        raise ValueError('incorrect residue length')
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder='little')
    if np.any(bits[bitcount:]):
        raise ValueError('nonzero residue padding')
    expanded = np.zeros((len(widths), 8), dtype=np.uint8)
    shifts = np.arange(8, dtype=np.uint8)
    expanded[shifts[None, :] < widths[:, None]] = bits[:bitcount]
    return np.sum(expanded.astype(np.uint16) << shifts, axis=1).astype(np.uint8)

def encode_lower(lower_u8, exponents_u8, predictor='previous_token', vector_size=8, group_size=64):
    lower, exponents = _validate_arrays(lower_u8, exponents_u8)
    if predictor not in PREDICTORS or not 1 <= vector_size <= 255 or group_size < vector_size or group_size % vector_size:
        raise ValueError('invalid predictor or grouping')
    rows, columns = lower.shape
    if rows >= 2**32 or columns >= 2**32 or group_size >= 2**32:
        raise ValueError('shape exceeds format')
    prediction = _predict_all(lower, exponents, predictor).reshape(-1)
    flat = lower.reshape(-1)
    widths = np.full(flat.size, 8, dtype=np.uint8)
    for b in range(8):
        match = _recover(prediction, flat & ((1 << b) - 1), b) == flat
        widths[(widths == 8) & match] = b
    nvec = (flat.size + vector_size - 1) // vector_size
    padded = np.pad(widths, (0, nvec * vector_size - flat.size))
    vector_widths = padded.reshape(-1, vector_size).max(axis=1) if nvec else np.empty(0, np.uint8)
    # Validate the actual shared vector widths, not just individual candidates.
    actual_widths = np.repeat(vector_widths, vector_size)[:flat.size]
    if not np.array_equal(_recover(prediction, flat, actual_widths), flat):
        raise AssertionError('coset width does not reconstruct the source exactly')
    components = Counter()
    modes, candidates, accepted = Counter(), Counter(), Counter()
    body = bytearray()
    for start in range(0, flat.size, group_size):
        original = flat[start:start + group_size]
        vw = vector_widths[start // vector_size:(start + len(original) + vector_size - 1) // vector_size]
        candidates.update(map(str, vw.tolist()))
        repeated = np.repeat(vw, vector_size)[:len(original)]
        payload = _pack_nibbles(vw) + _pack_variable(original, repeated)
        if len(payload) < len(original):
            mode = 1
            components['vector_widths'] += (len(vw) + 1) // 2
            components['coset_residues'] += len(payload) - (len(vw) + 1) // 2
            accepted.update(map(str, vw.tolist()))
        else:
            mode, payload = 0, original.tobytes()
            components['raw_groups'] += len(payload)
        modes[str(mode)] += 1
        body.extend(_GROUP.pack(mode, len(payload)))
        body.extend(payload)
        components['group_headers'] += _GROUP.size
    candidate_body_bytes = len(body)
    if len(body) >= lower.nbytes:
        flags, body = 1, bytearray(lower.tobytes())
        components = Counter({'raw_record': len(body)})
        accepted = Counter()
        serialized_modes = {'raw_record': 1}
    else:
        flags = 0
        serialized_modes = dict(modes)
    header = _HEADER.pack(_MAGIC, 1, PREDICTORS[predictor], vector_size, flags,
                          rows, columns, group_size, len(body), zlib.crc32(exponents.tobytes()), zlib.crc32(lower.tobytes()))
    prefix = header + body
    data = bytes(prefix) + _CRC.pack(zlib.crc32(prefix))
    components['header'] = _HEADER.size
    components['whole_crc32'] = _CRC.size
    assert sum(components.values()) == len(data)
    return EncodedLower(data, {
        'serialized_bytes': len(data), 'raw_lower_bytes': int(lower.nbytes),
        'predictor': predictor, 'group_size': group_size, 'vector_size': vector_size,
        'component_bytes': dict(components), 'mode_counts': serialized_modes,
        'candidate_group_mode_counts': dict(modes), 'candidate_group_body_bytes': candidate_body_bytes,
        'candidate_width_histogram': dict(candidates), 'accepted_width_histogram': dict(accepted),
        'raw_record_fallback': bool(flags), 'learned_state_bytes': 0,
    })

def decode_lower(encoded, exponents_u8):
    data = encoded.data if isinstance(encoded, EncodedLower) else bytes(encoded)
    if len(data) < _HEADER.size + _CRC.size:
        raise ValueError('truncated temporal lower record')
    if zlib.crc32(data[:-4]) != _CRC.unpack_from(data, len(data) - 4)[0]:
        raise ValueError('temporal record checksum mismatch')
    magic, version, pid, vector_size, flags, rows, columns, group_size, bodylen, expcrc, locrc = _HEADER.unpack_from(data)
    if magic != _MAGIC or version != 1 or pid not in PREDICTORS.values() or flags > 1 or not vector_size or group_size < vector_size or group_size % vector_size:
        raise ValueError('unsupported temporal lower header')
    exponents = np.asarray(exponents_u8)
    if exponents.dtype != np.uint8 or exponents.shape != (rows, columns):
        raise ValueError('exponent shape or type mismatch')
    if zlib.crc32(np.ascontiguousarray(exponents).tobytes()) != expcrc:
        raise ValueError('exponent checksum mismatch')
    if bodylen != len(data) - _HEADER.size - 4:
        raise ValueError('temporal body length mismatch')
    body = data[_HEADER.size:-4]
    total = rows * columns
    if flags:
        if len(body) != total:
            raise ValueError('incorrect raw lower length')
        lower = np.frombuffer(body, dtype=np.uint8).copy().reshape(rows, columns)
    else:
        if (total + group_size - 1) // group_size > len(body) // _GROUP.size:
            raise ValueError('truncated temporal groups')
        all_residues, all_widths = [], []
        offset = 0
        for start in range(0, total, group_size):
            if offset + _GROUP.size > len(body):
                raise ValueError('truncated temporal group header')
            mode, length = _GROUP.unpack_from(body, offset)
            offset += _GROUP.size
            if offset + length > len(body):
                raise ValueError('truncated temporal payload')
            payload = body[offset:offset + length]
            offset += length
            count = min(group_size, total - start)
            if mode == 0:
                if length != count:
                    raise ValueError('incorrect raw group length')
                residues, widths = np.frombuffer(payload, dtype=np.uint8), np.full(count, 8, dtype=np.uint8)
            elif mode == 1:
                nv = (count + vector_size - 1) // vector_size
                wb = (nv + 1) // 2
                vw = _unpack_nibbles(payload[:wb], nv)
                widths = np.repeat(vw, vector_size)[:count]
                residues = _unpack_variable(payload[wb:], widths)
            else:
                raise ValueError('unknown temporal mode')
            all_residues.append(residues)
            all_widths.append(widths)
        if offset != len(body):
            raise ValueError('unexpected temporal trailing bytes')
        residues = (np.concatenate(all_residues) if all_residues else np.empty(0, np.uint8)).reshape(rows, columns)
        widths = (np.concatenate(all_widths) if all_widths else np.empty(0, np.uint8)).reshape(rows, columns)
        predictor = next(name for name, code in PREDICTORS.items() if code == pid)
        if predictor in ('zero', 'midpoint'):
            lower = _recover(0 if predictor == 'zero' else 128, residues, widths)
        elif predictor in ('previous_token', 'previous_token_scaled'):
            lower = np.empty((rows, columns), dtype=np.uint8)
            for row in range(rows):
                prediction = 0 if row == 0 else lower[row - 1]
                if predictor == 'previous_token_scaled' and row:
                    prediction = _scaled_previous(prediction, exponents[row - 1], exponents[row])
                lower[row] = _recover(prediction, residues[row], widths[row])
        else:
            lower = np.empty((rows, columns), dtype=np.uint8)
            for column in range(columns):
                lower[:, column] = _recover(0 if column == 0 else lower[:, column - 1], residues[:, column], widths[:, column])
    if zlib.crc32(lower.tobytes()) != locrc:
        raise ValueError('decoded lower checksum mismatch')
    return lower
