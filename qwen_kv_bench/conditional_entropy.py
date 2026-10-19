"""Canonical Huffman streams with tensor-only encode and decode payload paths.

Codebooks are fitted on TRAIN counts on the host.  All alphabet symbols receive
positive mass; increasing additive smoothing bounds the longest code.  The
result is ordinary canonical Huffman coding, not an estimated entropy budget.

The decoder is an experimental eager-torch parallel algorithm.  It builds a
successor at every bit position, then uses pointer doubling to locate all symbol
ordinals.  A three-byte peek suffices for bounded 15-bit codes.  It uses O(bits)
temporary memory plus O(log(symbols)) pointer passes.  It is intentionally not
described as a fused or fast decoder.
Only scalar validation decisions synchronize the device; symbols and bitstreams
are never copied to the host by encode/decode.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
import struct

import numpy as np
import torch


FORMAT = "canonical_huffman_tensor_v1"


def _tree_lengths(weights):
    size = len(weights)
    if size == 1:
        return np.ones(1, dtype=np.uint8)
    queue = [(float(weight), index) for index, weight in enumerate(weights)]
    heapq.heapify(queue)
    children = [None] * (2 * size - 1)
    index = size
    while len(queue) > 1:
        left_weight, left = heapq.heappop(queue)
        right_weight, right = heapq.heappop(queue)
        children[index] = (left, right)
        heapq.heappush(queue, (left_weight + right_weight, index))
        index += 1
    lengths = np.zeros(size, dtype=np.int32)
    stack = [(queue[0][1], 0)]
    while stack:
        node, depth = stack.pop()
        if node < size:
            lengths[node] = depth
        else:
            left, right = children[node]
            stack.extend(((left, depth + 1), (right, depth + 1)))
    return lengths


def _canonical(lengths):
    lengths = np.asarray(lengths, dtype=np.uint8)
    codes = np.zeros(len(lengths), dtype=np.int32)
    code = 0
    previous = 0
    for symbol in sorted(range(len(lengths)), key=lambda item: (int(lengths[item]), item)):
        width = int(lengths[symbol])
        code <<= width - previous
        if code >= 1 << width:
            raise ValueError("invalid Huffman lengths: prefix capacity exceeded")
        codes[symbol] = code
        code += 1
        previous = width
    return codes


def _identity(codes, lengths):
    digest = hashlib.sha256()
    digest.update(FORMAT.encode("ascii"))
    digest.update(struct.pack("<II", len(lengths), int(np.max(lengths))))
    digest.update(np.asarray(codes, dtype="<i4").tobytes())
    digest.update(np.asarray(lengths, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def fit_codebook(counts, max_bits=15):
    """Fit a complete-alphabet, bounded canonical book from TRAIN-only counts.

    This uses positive additive smoothing, increased until Huffman lengths fit
    max_bits.  The mean coded length and actual smoothing cost are recorded;
    limiting is not silently treated as optimal length-limited Huffman coding.
    """
    source = np.asarray(counts, dtype=np.float64)
    if source.ndim != 1 or not 1 <= source.size <= 512:
        raise ValueError("counts must be a one-dimensional alphabet of 1..512 symbols")
    if not np.all(np.isfinite(source)) or np.any(source < 0):
        raise ValueError("counts must be finite and nonnegative")
    if isinstance(max_bits, bool) or not isinstance(max_bits, (int, np.integer)) or not 1 <= max_bits <= 15:
        raise ValueError("max_bits must be an integer in 1..15")
    if source.size > 1 << int(max_bits):
        raise ValueError("alphabet cannot fit the requested maximum prefix-code length")
    scale = max(1.0, float(np.max(source)))
    normalized = source / scale
    # One pseudocount where practical.  A floor avoids thousands of doubling
    # rounds for extreme finite counts, and its actual cost is exposed below.
    smoothing = max(1.0 / scale, 1e-12)
    rounds = 0
    initial_lengths = None
    while True:
        smoothed = normalized + smoothing
        lengths = _tree_lengths(smoothed)
        if initial_lengths is None:
            initial_lengths = lengths.copy()
        if int(np.max(lengths)) <= max_bits:
            break
        smoothing *= 2
        rounds += 1
    lengths = lengths.astype(np.uint8)
    codes = _canonical(lengths)
    effective = int(np.max(lengths))
    lookup_symbol = np.full(1 << effective, -1, dtype=np.int32)
    lookup_length = np.zeros(1 << effective, dtype=np.uint8)
    for symbol, (code, width) in enumerate(zip(codes, lengths)):
        suffix = effective - int(width)
        start = int(code) << suffix
        stop = start + (1 << suffix)
        lookup_symbol[start:stop] = symbol
        lookup_length[start:stop] = width
    total = float(np.sum(normalized))
    probabilities = normalized / total if total else np.zeros_like(normalized)
    positive = probabilities[probabilities > 0]
    return dict(format=FORMAT, alphabet_size=int(source.size), max_bits=effective,
        requested_max_bits=int(max_bits), min_bits=int(np.min(lengths)),
        code=codes, length=lengths, lookup_symbol=lookup_symbol,
        lookup_length=lookup_length, codebook_id=_identity(codes, lengths),
        smoothing_pseudocount=float(smoothing * scale), smoothing_rounds=rounds,
        training_mean_bits=float(np.dot(probabilities, lengths)),
        initial_smoothed_training_mean_bits=float(np.dot(probabilities, initial_lengths)),
        limiting_training_bits_delta=float(np.dot(probabilities, lengths - initial_lengths)),
        training_entropy_bits=float(-np.sum(positive * np.log2(positive))),
        smoothed_mean_bits=float(np.dot(smoothed / np.sum(smoothed), lengths)))


@dataclass(frozen=True)
class PreparedCodebook:
    code: torch.Tensor
    length: torch.Tensor
    lookup_symbol: torch.Tensor
    lookup_length: torch.Tensor
    alphabet_size: int
    max_bits: int
    min_bits: int
    codebook_id: str

    @property
    def physical_nbytes(self):
        return sum(value.numel() * value.element_size() for value in
            (self.code, self.length, self.lookup_symbol, self.lookup_length))


def prepare(book, device):
    """Validate a host book and preload its actual tables on the target device."""
    if book.get("format") != FORMAT:
        raise ValueError("unsupported Huffman codebook format")
    codes = np.asarray(book["code"], dtype=np.int32)
    lengths = np.asarray(book["length"], dtype=np.uint8)
    if codes.ndim != 1 or lengths.shape != codes.shape or not 1 <= len(codes) <= 512:
        raise ValueError("invalid Huffman code/length shape")
    if np.any(lengths < 1) or np.any(lengths > 15):
        raise ValueError("invalid Huffman code lengths")
    if not np.array_equal(codes, _canonical(lengths)):
        raise ValueError("codebook is not canonical")
    if int(book["alphabet_size"]) != len(codes) or int(book["max_bits"]) != int(np.max(lengths)):
        raise ValueError("Huffman codebook geometry mismatch")
    if int(book["min_bits"]) != int(np.min(lengths)):
        raise ValueError("Huffman codebook minimum length mismatch")
    identity = _identity(codes, lengths)
    if identity != book["codebook_id"]:
        raise ValueError("Huffman codebook identity mismatch")
    size = 1 << int(np.max(lengths))
    symbols = np.full(size, -1, dtype=np.int32)
    lookup_lengths = np.zeros(size, dtype=np.uint8)
    for symbol, (code, width) in enumerate(zip(codes, lengths)):
        shift = int(np.max(lengths)) - int(width)
        start = int(code) << shift
        symbols[start:start + (1 << shift)] = symbol
        lookup_lengths[start:start + (1 << shift)] = width
    if not np.array_equal(symbols, np.asarray(book["lookup_symbol"])) or not np.array_equal(
            lookup_lengths, np.asarray(book["lookup_length"])):
        raise ValueError("Huffman lookup tables do not match canonical codes")
    # Explicit copies also support readonly frozen artifact arrays safely.
    tensors = [torch.tensor(value.copy(), device=device) for value in
        (codes, lengths, symbols, lookup_lengths)]
    return PreparedCodebook(*tensors, len(codes), int(np.max(lengths)),
        int(np.min(lengths)), identity)


def _checksum(payload):
    """Standard Adler-32 as four wire bytes, computed entirely on the device."""
    values = payload.to(torch.int64)
    count = values.numel()
    weights = (count - torch.arange(count, device=payload.device, dtype=torch.int64)) % 65521
    first = (values.sum() + 1) % 65521
    second = ((values * weights).sum() + count) % 65521
    combined = (second << 16) | first
    shifts = torch.arange(4, device=payload.device, dtype=torch.int64) * 8
    return ((combined >> shifts) & 255).to(torch.uint8)


def encode(symbols, prepared):
    """Return true packed Huffman bytes and a descriptor; no payload host copy."""
    if symbols.ndim != 1 or symbols.dtype == torch.bool or symbols.is_floating_point() or symbols.is_complex():
        raise ValueError("Huffman symbols must be a one-dimensional integer tensor")
    if symbols.device != prepared.code.device:
        raise ValueError("Huffman symbols and codebook must be on the same device")
    source = symbols.to(torch.int64)
    if bool(torch.any((source < 0) | (source >= prepared.alphabet_size))):
        raise ValueError("Huffman symbol outside codebook alphabet")
    lengths = prepared.length[source].to(torch.int64)
    codes = prepared.code[source].to(torch.int64)
    # The scalar bit count is allocation metadata, not a host copy of symbols.
    nbits = int(lengths.sum().item())
    padding = (-nbits) % 8
    num_bytes = (nbits + 7) // 8
    accumulator = torch.zeros(num_bytes, device=source.device, dtype=torch.int32)
    if num_bytes:
        offsets = torch.cumsum(lengths, dim=0) - lengths
        byte_indices = offsets >> 3
        window = codes << (24 - (offsets & 7) - lengths)
        # Every code occupies at most three bytes.  Contributions occupy
        # disjoint bits, so integer addition is an exact parallel OR here.
        for byte in range(3):
            target = byte_indices + byte
            values = ((window >> (16 - 8 * byte)) & 255).to(torch.int32)
            values = torch.where(target < num_bytes, values, 0)
            accumulator.index_add_(0, target.clamp_max(num_bytes - 1), values)
    payload = accumulator.to(torch.uint8)
    fields = dict(payload=payload.contiguous(), checksum=_checksum(payload))
    descriptor = dict(format=FORMAT, num_symbols=source.numel(), nbits=nbits,
        payload_bytes=payload.numel(), padding_bits=padding,
        alphabet_size=prepared.alphabet_size, max_bits=prepared.max_bits,
        codebook_id=prepared.codebook_id, integrity="adler32_payload",
        decode_algorithm="all_bit_successors_ordinal_pointer_doubling")
    return fields, descriptor


def decode(fields, descriptor, prepared):
    """Decode on the prepared device using parallel successor pointer doubling.

    Header scalars and the actual bitstream boundary are validated.  Adler-32
    detects accidental payload corruption; it is not an authentication code.
    """
    if descriptor.get("format") != FORMAT or descriptor.get("integrity") != "adler32_payload":
        raise ValueError("unsupported Huffman stream descriptor")
    if descriptor.get("codebook_id") != prepared.codebook_id:
        raise ValueError("Huffman stream codebook identity mismatch")
    if descriptor.get("alphabet_size") != prepared.alphabet_size or descriptor.get("max_bits") != prepared.max_bits:
        raise ValueError("Huffman stream codebook geometry mismatch")
    for name in ("num_symbols", "nbits", "payload_bytes", "padding_bits"):
        if isinstance(descriptor.get(name), bool) or not isinstance(descriptor.get(name), int) or descriptor[name] < 0:
            raise ValueError("invalid Huffman stream size descriptor")
    count, nbits = descriptor["num_symbols"], descriptor["nbits"]
    payload, checksum = fields.get("payload"), fields.get("checksum")
    for value, name in ((payload, "payload"), (checksum, "checksum")):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
            raise ValueError("invalid Huffman " + name + " tensor")
        if value.device != prepared.code.device:
            raise ValueError("Huffman wire tensors and tables must share a device")
    if checksum.numel() != 4 or payload.numel() != descriptor["payload_bytes"] or payload.numel() != (nbits + 7) // 8:
        raise ValueError("Huffman payload byte count mismatch")
    if descriptor["padding_bits"] != (-nbits) % 8:
        raise ValueError("Huffman padding descriptor mismatch")
    if not count * prepared.min_bits <= nbits <= count * prepared.max_bits:
        raise ValueError("Huffman symbol count and bit count disagree")
    if bool(torch.any(checksum != _checksum(payload))):
        raise ValueError("Huffman payload checksum mismatch")
    if nbits % 8 and bool((payload[-1].to(torch.int64) & ((1 << (-nbits % 8)) - 1)) != 0):
        raise ValueError("nonzero Huffman padding bits")
    if count == 0:
        return torch.empty(0, device=prepared.code.device, dtype=torch.int64)
    positions = torch.arange(nbits, device=payload.device, dtype=torch.int64)
    padded_payload = torch.cat((payload, torch.zeros(2, device=payload.device, dtype=torch.uint8)))
    byte_indices = positions >> 3
    window = ((padded_payload[byte_indices].to(torch.int64) << 16)
        | (padded_payload[byte_indices + 1].to(torch.int64) << 8)
        | padded_payload[byte_indices + 2].to(torch.int64))
    prefixes = (window >> (24 - (positions & 7) - prepared.max_bits)) & ((1 << prepared.max_bits) - 1)
    lengths = prepared.lookup_length[prefixes].to(torch.int64)
    following = positions + lengths
    invalid = (lengths == 0) | (following > nbits)
    following = torch.where(invalid, nbits + 1, following)
    # Terminal and error nodes are absorbing, making all gathers bounded even
    # for a malformed stream whose true symbol path hits an invalid prefix.
    jump = torch.cat((following, torch.tensor([nbits, nbits + 1], device=payload.device, dtype=torch.int64)))
    ordinals = torch.arange(count + 1, device=payload.device, dtype=torch.int64)
    starts = torch.zeros_like(ordinals)
    levels = count.bit_length()
    for level in range(levels):
        starts = torch.where(((ordinals >> level) & 1) != 0, jump[starts], starts)
        if level + 1 < levels:
            jump = jump[jump]
    if bool((starts[-1] != nbits) | torch.any(starts[:-1] >= nbits)):
        raise ValueError("Huffman stream does not end at the declared symbol boundary")
    decoded = prepared.lookup_symbol[prefixes[starts[:-1]]].to(torch.int64)
    if bool(torch.any(decoded < 0)):
        raise ValueError("invalid Huffman prefix on the decoded symbol path")
    return decoded
