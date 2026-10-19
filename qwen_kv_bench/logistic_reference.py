"""Actual lossless CPU arithmetic coding under per-value logistic predictions.

This is a correctness-oriented Python/NumPy CPU reference, not a GPU entropy
codec or a throughput claim.  The caller must retain/reproduce mu and log_scale;
neither predictions nor raw target words are stored in the descriptor.  Their
effective float64 values are hashed so a mismatched predictor fails explicitly.

The full ordered BF16 alphabet has 65,536 distinct bit patterns.  Finite ranks
128..65407 receive logistic interval mass, with the extreme finite intervals
extended to infinity.  Signed zeros meet at zero.  All infinities/NaN payloads
receive only the uniform floor.  Integer CDF(t) is exactly

    t + floor((2**32 - 65536) * sigmoid((boundary[t] - mu) / scale)).

Endpoints are 0 and 2**32, so every symbol receives at least one count.  This is
the integer counterpart of mixing logistic mass with a 2**-16 uniform escape
distribution.  There is no separate escape stream.  Float64 CDF evaluation is
shared by encoder/decoder on the same CPU/math implementation; this reference
does not promise cross-platform libm or GPU numerical reproducibility.

General decode tries a quantile hint and validates its two CDF endpoints.  A miss
uses at most sixteen further CDF evaluations in full-alphabet bisection (eighteen
total per symbol).  A page with one identical distribution uses one temporary
65,537-entry uint64 CDF table instead.  Neither path is a constant-time GPU path.

The inclusive 64-bit arithmetic state uses classic E1/E2/E3 renormalization.
It emits an explicit 64-zero-bit lookahead guard, byte padding, and integrity
metadata.  Both arithmetic bits and all transmitted payload bytes are reported.
The caller must additionally charge the serialized JSON descriptor, retained
features, predictor weights and any enclosing page/index framing.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zlib

import numpy as np


FORMAT = "bf16_logistic_arithmetic_cpu_v1"
TOTAL = 1 << 32
ALPHABET = 1 << 16
LOGISTIC_COUNTS = TOTAL - ALPHABET
STATE_BITS = 64
FULL = 1 << STATE_BITS
MASK = FULL - 1
HALF = FULL >> 1
QUARTER = FULL >> 2
THREE_QUARTERS = 3 * QUARTER
LOG_SCALE_MIN = -100.0
LOG_SCALE_MAX = 100.0
MAX_SYMBOLS = 1 << 20
FINITE_FIRST = 128
FINITE_LAST = 65407
_MAX_BF16 = float(np.array([0x7f7f0000], dtype=np.uint32).view(np.float32)[0])


def ordered(words):
    """Map all BF16 bit patterns to a bijective, numeric-order-compatible rank."""
    values = np.asarray(words, dtype=np.uint16).astype(np.uint32)
    return np.where(values & 0x8000, (~values) & 0xffff, values ^ 0x8000).astype(np.uint32)


def from_ordered(ranks):
    values = np.asarray(ranks, dtype=np.uint32)
    return np.where(values & 0x8000, values ^ 0x8000, (~values) & 0xffff).astype(np.uint16)


def _boundaries():
    ranks = np.arange(ALPHABET, dtype=np.uint32)
    words = from_ordered(ranks)
    finite = (words & 0x7f80) != 0x7f80
    finite_ranks = ranks[finite]
    if (int(finite_ranks[0]), int(finite_ranks[-1]), len(finite_ranks)) != (FINITE_FIRST, FINITE_LAST, 65280):
        raise AssertionError("BF16 finite rank derivation disagrees with codec constants")
    value = (words[finite].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    result = np.empty(ALPHABET + 1, dtype=np.float64)
    result[:FINITE_FIRST + 1] = -np.inf
    result[FINITE_LAST + 1:] = np.inf
    result[FINITE_FIRST + 1:FINITE_LAST + 1] = (value[:-1] + value[1:]) / 2
    # -0 has rank 32767 and +0 rank 32768.  Their common boundary is numeric 0.
    result[32768] = 0.0
    result.flags.writeable = False
    return result


BOUNDARIES = _boundaries()


def _sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _cdf(rank, mu, scale):
    # Explicit endpoints also avoid any dependence on NaN/inf source values.
    if rank <= FINITE_FIRST:
        return rank
    if rank >= FINITE_LAST + 1:
        return rank + LOGISTIC_COUNTS
    argument = (float(BOUNDARIES[rank]) - mu) / scale
    return rank + math.floor(LOGISTIC_COUNTS * _sigmoid(argument))


def _interval_mass(rank, mu, scale):
    if not FINITE_FIRST <= rank <= FINITE_LAST:
        return 0.0
    lo, hi = float(BOUNDARIES[rank]), float(BOUNDARIES[rank + 1])
    # Use survival probabilities in the positive tail to avoid 1 - 1 loss.
    if lo >= mu:
        mass = _sigmoid((mu - lo) / scale) - _sigmoid((mu - hi) / scale)
    else:
        mass = _sigmoid((hi - mu) / scale) - _sigmoid((lo - mu) / scale)
    return min(1.0, max(0.0, mass))


def _shape(shape):
    if (not isinstance(shape, (tuple, list)) or len(shape) not in (1, 2)
            or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0 for value in shape)):
        raise ValueError("logistic reference requires a nonnegative 1D or 2D shape")
    result = tuple(int(value) for value in shape)
    count = math.prod(result)
    if count > MAX_SYMBOLS:
        raise ValueError("logistic reference page exceeds its CPU safety limit")
    return result, count


def _predictions(mu, log_scale, shape):
    mean, logs = np.asarray(mu), np.asarray(log_scale)
    if mean.shape != shape or logs.shape != shape:
        raise ValueError("logistic prediction shapes must exactly match the page")
    if mean.dtype not in (np.dtype("float32"), np.dtype("float64")) or logs.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("logistic predictions must be float32 or float64")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(logs)):
        raise ValueError("logistic predictions must be finite")
    mean = np.ascontiguousarray(mean, dtype=np.float64).reshape(-1)
    original_logs = np.ascontiguousarray(logs, dtype=np.float64).reshape(-1)
    clipped = np.clip(original_logs, LOG_SCALE_MIN, LOG_SCALE_MAX)
    # Use the same scalar libm calculation in both directions.  Vectorized NumPy
    # exp versus math.exp can differ by an ULP, which must not define two CDFs.
    scales = np.fromiter((math.exp(float(value)) for value in clipped), dtype=np.float64, count=clipped.size)
    digest = hashlib.sha256()
    digest.update(FORMAT.encode("ascii"))
    digest.update(struct.pack("<I", len(shape)))
    digest.update(struct.pack("<" + "Q" * len(shape), *shape))
    digest.update(mean.astype("<f8", copy=False).tobytes())
    digest.update(clipped.astype("<f8", copy=False).tobytes())
    return mean, scales, digest.hexdigest(), int(np.count_nonzero(original_logs != clipped))


def _shared_table(mean, scales):
    if mean.size and np.all(mean == mean[0]) and np.all(scales == scales[0]):
        mu, scale = float(mean[0]), float(scales[0])
        # At most 512 KiB of temporary CDF entries; no alphabet-per-value array.
        return np.fromiter((_cdf(rank, mu, scale) for rank in range(ALPHABET + 1)),
            dtype=np.uint64, count=ALPHABET + 1)
    return None


class _BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current = 0
        self.used = 0
        self.nbits = 0

    def write(self, value):
        self.current = (self.current << 1) | int(value)
        self.used += 1
        self.nbits += 1
        if self.used == 8:
            self.data.append(self.current)
            self.current = self.used = 0

    def finish(self):
        if self.used:
            self.data.append(self.current << (8 - self.used))
        return bytes(self.data)


class _BitReader:
    def __init__(self, data, nbits):
        self.data = data
        self.nbits = nbits
        self.position = 0

    def read(self):
        if self.position >= self.nbits:
            raise ValueError("truncated arithmetic lookahead guard")
        result = (self.data[self.position >> 3] >> (7 - (self.position & 7))) & 1
        self.position += 1
        return result


class _ArithmeticEncoder:
    def __init__(self, total=TOTAL):
        if not isinstance(total, int) or not 1 <= total <= TOTAL:
            raise ValueError("arithmetic total outside 64-bit reference precision")
        self.total = total
        self.low, self.high = 0, MASK
        self.pending = 0
        self.writer = _BitWriter()

    def _emit(self, bit):
        self.writer.write(bit)
        while self.pending:
            self.writer.write(1 - bit)
            self.pending -= 1

    def update(self, cdf_low, cdf_high):
        if not 0 <= cdf_low < cdf_high <= self.total:
            raise ValueError("invalid arithmetic symbol interval")
        width = self.high - self.low + 1
        new_high = self.low + width * cdf_high // self.total - 1
        self.low += width * cdf_low // self.total
        self.high = new_high
        if self.low > self.high:
            raise ValueError("arithmetic precision exhausted")
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        arithmetic_bits = self.writer.nbits
        for _ in range(STATE_BITS):
            self.writer.write(0)
        nbits = self.writer.nbits
        return self.writer.finish(), arithmetic_bits, nbits


class _ArithmeticDecoder:
    def __init__(self, data, nbits, total=TOTAL):
        self.total = total
        self.low, self.high = 0, MASK
        self.reader = _BitReader(data, nbits)
        self.value = 0
        for _ in range(STATE_BITS):
            self.value = (self.value << 1) | self.reader.read()

    def cumulative(self):
        if not self.low <= self.value <= self.high:
            raise ValueError("arithmetic value outside current interval")
        return ((self.value - self.low + 1) * self.total - 1) // (self.high - self.low + 1)

    def update(self, cdf_low, cdf_high):
        width = self.high - self.low + 1
        new_high = self.low + width * cdf_high // self.total - 1
        self.low += width * cdf_low // self.total
        self.high = new_high
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK
            self.value = ((self.value << 1) & MASK) | self.reader.read()


def _quantile_rank(cumulative, mu, scale):
    probability = (cumulative + 0.5) / TOTAL
    value = mu + scale * (math.log(probability) - math.log1p(-probability))
    value = max(-_MAX_BF16, min(_MAX_BF16, value))
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    word = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff
    return ((~word) & 0xffff) if word & 0x8000 else word ^ 0x8000


def _find_symbol(cumulative, mu, scale):
    """Try a logistic quantile, then bounded full-alphabet bisection if needed."""
    guess = _quantile_rank(cumulative, mu, scale)
    left, right = _cdf(guess, mu, scale), _cdf(guess + 1, mu, scale)
    if left <= cumulative < right:
        return guess, left, right, False
    if cumulative < left:
        low, high, low_cdf, high_cdf = 0, guess, 0, left
    else:
        low, high, low_cdf, high_cdf = guess + 1, ALPHABET, right, TOTAL
    while low + 1 < high:
        middle = (low + high) // 2
        value = _cdf(middle, mu, scale)
        if value <= cumulative:
            low, low_cdf = middle, value
        else:
            high, high_cdf = middle, value
    return low, low_cdf, high_cdf, True


def _words_crc(words):
    return zlib.crc32(np.ascontiguousarray(words, dtype="<u2").tobytes())


def encode(words, mu, log_scale):
    """Serialize exact original BF16 words using externally reproducible predictions.

    All reported bit counts refer to this CPU model.  Predictions and predictor
    weights must be charged separately by a caller's complete packet accounting.
    """
    source = np.asarray(words)
    if source.dtype != np.dtype("uint16"):
        raise ValueError("logistic reference input must contain uint16 BF16 words")
    shape, count = _shape(source.shape)
    mean, scales, prediction_hash, clipped = _predictions(mu, log_scale, shape)
    source = np.ascontiguousarray(source)
    ranks = ordered(source).reshape(-1)
    table = _shared_table(mean, scales)
    arithmetic = _ArithmeticEncoder()
    ideal_bits = integer_bits = 0.0
    floor_symbols = floor_dominated = 0
    for index, raw_rank in enumerate(ranks):
        rank, center, scale = int(raw_rank), float(mean[index]), float(scales[index])
        left = int(table[rank]) if table is not None else _cdf(rank, center, scale)
        right = int(table[rank + 1]) if table is not None else _cdf(rank + 1, center, scale)
        frequency = right - left
        mass = _interval_mass(rank, center, scale)
        modeled_mass = (1.0 - 2.0 ** -16) * mass
        ideal_bits -= math.log2(modeled_mass + 2.0 ** -32)
        integer_bits += 32.0 - math.log2(frequency)
        floor_symbols += int(frequency == 1)
        floor_dominated += int(modeled_mass <= 2.0 ** -32)
        arithmetic.update(left, right)
    payload, arithmetic_bits, nbits = arithmetic.finish()
    description = dict(format=FORMAT, version=1, shape=list(shape), num_symbols=count,
        nbits=nbits, arithmetic_bits=arithmetic_bits, guard_bits=STATE_BITS,
        padding_bits=(-nbits) % 8, payload_bytes=len(payload),
        payload_crc32=zlib.crc32(payload), words_crc32=_words_crc(source),
        prediction_sha256=prediction_hash, total=TOTAL, state_bits=STATE_BITS,
        scale_log_min=LOG_SCALE_MIN, scale_log_max=LOG_SCALE_MAX,
        clamped_scale_values=clipped, ideal_continuous_bits=float(ideal_bits),
        integer_cdf_bits=float(integer_bits), integer_floor_symbols=floor_symbols,
        ideal_floor_dominated_symbols=floor_dominated,
        nonfinite_symbols=int(np.count_nonzero((source & 0x7f80) == 0x7f80)),
        cdf_strategy="shared_exact_table" if table is not None else "per_value_scalar_float64",
        cdf_math="same_CPU_python_math_float64_v1",
        execution="CPU_reference_actual_arithmetic_bitstream",
        escape_coding="uniform_floor_no_separate_escape_stream")
    return payload, description


def decode(payload, descriptor, mu, log_scale):
    """Recover exact BF16 words; reject corrupt bytes or mismatched predictions."""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("arithmetic payload must be bytes")
    if not isinstance(descriptor, dict) or descriptor.get("format") != FORMAT or descriptor.get("version") != 1:
        raise ValueError("unsupported logistic arithmetic descriptor")
    shape, count = _shape(descriptor.get("shape"))
    for name in ("num_symbols", "nbits", "arithmetic_bits", "guard_bits", "padding_bits", "payload_bytes", "payload_crc32", "words_crc32"):
        value = descriptor.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid arithmetic descriptor integer: " + name)
    nbits = descriptor["nbits"]
    if (descriptor["num_symbols"] != count or descriptor["guard_bits"] != STATE_BITS
            or descriptor["arithmetic_bits"] < 2
            or nbits != descriptor["arithmetic_bits"] + STATE_BITS
            or not STATE_BITS + 2 <= nbits <= 33 * count + STATE_BITS + 2
            or descriptor["padding_bits"] != (-nbits) % 8
            or descriptor["payload_bytes"] != len(payload) or len(payload) != (nbits + 7) // 8
            or descriptor.get("total") != TOTAL or descriptor.get("state_bits") != STATE_BITS
            or descriptor.get("scale_log_min") != LOG_SCALE_MIN or descriptor.get("scale_log_max") != LOG_SCALE_MAX
            or descriptor.get("cdf_math") != "same_CPU_python_math_float64_v1"):
        raise ValueError("inconsistent logistic arithmetic stream geometry or contract")
    if zlib.crc32(payload) != descriptor["payload_crc32"]:
        raise ValueError("arithmetic payload CRC32 mismatch")
    if nbits % 8 and payload[-1] & ((1 << (-nbits % 8)) - 1):
        raise ValueError("nonzero arithmetic byte padding")
    for position in range(descriptor["arithmetic_bits"], nbits):
        if (payload[position >> 3] >> (7 - (position & 7))) & 1:
            raise ValueError("nonzero arithmetic lookahead guard")
    mean, scales, prediction_hash, _ = _predictions(mu, log_scale, shape)
    if prediction_hash != descriptor.get("prediction_sha256"):
        raise ValueError("logistic predictor SHA256 mismatch")
    table = _shared_table(mean, scales)
    arithmetic = _ArithmeticDecoder(payload, nbits)
    result = np.empty(count, dtype=np.uint16)
    for index in range(count):
        cumulative = arithmetic.cumulative()
        if table is not None:
            symbol = int(np.searchsorted(table, np.uint64(cumulative), side="right")) - 1
            left, right = int(table[symbol]), int(table[symbol + 1])
        else:
            symbol, left, right, _ = _find_symbol(cumulative, float(mean[index]), float(scales[index]))
        if not 0 <= symbol < ALPHABET or not left <= cumulative < right:
            raise ValueError("arithmetic decoder failed to find a valid symbol interval")
        result[index] = ((~symbol) & 0xffff) if symbol < 0x8000 else symbol ^ 0x8000
        arithmetic.update(left, right)
    result = result.reshape(shape)
    if _words_crc(result) != descriptor["words_crc32"]:
        raise ValueError("decoded BF16 word CRC32 mismatch")
    return result
