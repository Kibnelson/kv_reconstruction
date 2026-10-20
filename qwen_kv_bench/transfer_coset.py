"""Exact BF16 symbol coding with an optional modulo-coset factorization.

This CPU reference preserves the probability construction in the supplied
``combined.py``: code a 512-way ordered sign/exponent group, then its 128-way
ordered mantissa symbol.  A coset is just ``mantissa % (2**residue_bits)``;
the quotient identifies the member inside that coset.  This is a bijective
relabeling, not E8, LDPC, or a claim of additional information-theoretic savings.

``coset_model`` entropy-codes the residue using its exact marginal and then the
quotient conditional on that residue.  It is the same ideal joint distribution
as ``direct`` and serves as a factorization control.  ``coset_uniform`` spends
exactly ``residue_bits`` ideal bits on a uniform residue before conditional
quotient coding.  All stages use the actual range coder, so transmitted residues
are included in the returned compressed stream.  Finite coder precision and
stream termination can make actual sizes differ from their ideal costs.

Integer table interpolation matches the original prototype.  Boundary
standardization and probabilities passed to constriction still use float64;
this module does not promise bit-identical execution across hardware/libm builds.
"""
from __future__ import annotations

import numpy as np


TOTAL = 1 << 48
ONE16 = 1 << 16
TT, TR = 32.0, 256
NTAB = int(2 * TT * TR) + 1
MODES = ("direct", "coset_model", "coset_uniform")


def ordered(words):
    """Bijectively order all BF16 bit patterns, retaining NaN payload bits."""
    values = np.asarray(words).astype(np.int64)
    return np.where(values & 0x8000, (~values) & 0xffff, values ^ 0x8000)


def word_of(ranks):
    values = np.asarray(ranks).astype(np.int64)
    return np.where(values & 0x8000, values ^ 0x8000, (~values) & 0xffff).astype(np.uint16)


def _boundaries():
    with np.errstate(invalid="ignore"):
        values = (word_of(np.arange(65536)).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        result = np.empty(65537, np.float64)
        result[1:65536] = (values[:-1] + values[1:]) / 2
    result[:129] = -np.inf
    result[65408:] = np.inf
    if np.isnan(result).any() or not np.all(result[1:] >= result[:-1]):
        raise AssertionError("invalid original-prototype BF16 boundaries")
    result.flags.writeable = False
    return result


BD = _boundaries()
GB = BD[::128]
F_TAB = np.rint(2.0 ** 32 / (1 + np.exp(-(-TT + np.arange(NTAB) / TR)))).astype(np.int64)
F_TAB[0], F_TAB[-1] = 0, 1 << 32
F_TAB.flags.writeable = False
_CAT = None


def categorical():
    """Load the optional CPU entropy-coder dependency only when it is used."""
    global _CAT
    if _CAT is None:
        try:
            import constriction
        except ImportError as error:
            raise RuntimeError("This CPU transfer reference requires constriction") from error
        _CAT = constriction.stream.model.Categorical(perfect=False)
    return _CAT


def cdf_int(t, f_tab=None):
    """Original table-interpolated logistic CDF, with 32-bit integer outputs."""
    table = F_TAB if f_tab is None else np.asarray(f_tab)
    if table.shape != (NTAB,) or not np.issubdtype(table.dtype, np.integer) or table[0] != 0 or table[-1] != (1 << 32) or np.any(np.diff(table.astype(np.int64)) < 0):
        raise ValueError("invalid serialized logistic CDF table")
    u = np.clip((np.asarray(t, np.float64) + TT) * TR, 0.0, NTAB - 1.0)
    if np.isnan(u).any():
        raise ValueError("NaN standardized logistic boundary")
    index = np.minimum(np.floor(u).astype(np.int64), NTAB - 2)
    fraction = np.floor((u - index) * 65536.0).astype(np.int64)
    return table[index] + (((table[index + 1] - table[index]) * fraction) >> 16)


def _predictions(mu, inv_s):
    mean, inverse = np.asarray(mu, np.float64), np.asarray(inv_s, np.float64)
    if mean.ndim != 1 or inverse.shape != mean.shape:
        raise ValueError("predictions must be equally sized flat vectors")
    if not np.isfinite(mean).all() or not np.isfinite(inverse).all() or np.any(inverse <= 0):
        raise ValueError("logistic means must be finite and inverse scales finite and positive")
    return mean, inverse


def group_num(mu, inv_s, f_tab=None):
    """Return [values,512] integer group masses whose rows sum to 2**48."""
    mean, inverse = _predictions(mu, inv_s)
    with np.errstate(over="ignore"):
        cdf = cdf_int((GB[None, :] - mean[:, None]) * inverse[:, None], f_tab)
    return (ONE16 - 1) * np.diff(cdf, axis=1) + 128 * ONE16


def mant_num(mu, inv_s, groups, f_tab=None):
    """Return original 128 word masses inside each supplied group."""
    mean, inverse = _predictions(mu, inv_s)
    group = np.asarray(groups)
    if group.shape != mean.shape or not np.issubdtype(group.dtype, np.integer) or np.any(group < 0) or np.any(group >= 512):
        raise ValueError("invalid ordered sign/exponent groups")
    edges = BD[(group.astype(np.int64) * 128)[:, None] + np.arange(129)[None, :]]
    with np.errstate(over="ignore"):
        cdf = cdf_int((edges - mean[:, None]) * inverse[:, None], f_tab)
    return (ONE16 - 1) * np.diff(cdf, axis=1) + ONE16


def _settings(mode, residue_bits):
    if mode not in MODES:
        raise ValueError("unknown transfer symbol-coding mode")
    if isinstance(residue_bits, (bool, np.bool_)) or not isinstance(residue_bits, (int, np.integer)) or not 0 <= residue_bits <= 7:
        raise ValueError("residue_bits must be an integer from zero through seven")
    if mode == "direct" and residue_bits != 0:
        raise ValueError("direct coding has no transmitted coset residue")
    return int(residue_bits)


def code_block(coder, mu, inv_s, w=None, mode="direct", residue_bits=0, f_tab=None):
    """Encode/decode one flat channel block; return (BF16 words, ideal bits).

    The decoder supplies ``w=None`` and has only the same predictions and coder
    state.  It never needs source words or an untransmitted residue.  Predictions
    may be approximate; the reconstructed bit patterns remain exact.
    """
    bits = _settings(mode, residue_bits)
    mean, inverse = _predictions(mu, inv_s)
    count = len(mean)
    if w is not None:
        source = np.asarray(w)
        if source.dtype != np.uint16 or source.shape != mean.shape:
            raise ValueError("source words must be a matching flat uint16 vector")
        ranks = ordered(source)
    else:
        ranks = None
    if not count:
        return np.empty(0, np.uint16), 0.0

    cat = categorical()
    gn = group_num(mean, inverse, f_tab)
    if ranks is not None:
        groups = ranks >> 7
        coder.encode(groups.astype(np.int32), cat, gn / float(TOTAL))
    else:
        groups = coder.decode(cat, gn / float(TOTAL)).astype(np.int64)
    mn = mant_num(mean, inverse, groups, f_tab)
    rows = np.arange(count)
    group_mass = gn[rows, groups]

    if mode == "direct" or bits == 0:
        if ranks is not None:
            mantissa = ranks & 127
            coder.encode(mantissa.astype(np.int32), cat, mn / group_mass[:, None])
        else:
            mantissa = coder.decode(cat, mn / group_mass[:, None]).astype(np.int64)
        word_mass = mn[rows, mantissa]
        ideal = -np.log2(word_mass / float(TOTAL))
    else:
        stride = 1 << bits
        quotient_count = 128 // stride
        # Original mantissa index is quotient*stride + residue.
        masses = mn.reshape(count, quotient_count, stride)
        residue_mass = masses.sum(axis=1, dtype=np.int64)
        if mode == "coset_model":
            residue_probability = residue_mass / group_mass[:, None]
        else:
            residue_probability = np.full((count, stride), 1.0 / stride, np.float64)
        if ranks is not None:
            mantissa = ranks & 127
            residue = mantissa & (stride - 1)
            coder.encode(residue.astype(np.int32), cat, residue_probability)
        else:
            residue = coder.decode(cat, residue_probability).astype(np.int64)
        selected_residue_mass = residue_mass[rows, residue]
        quotient_mass = masses[rows, :, residue]
        if quotient_count == 1:
            quotient = np.zeros(count, np.int64)
        elif ranks is not None:
            quotient = mantissa >> bits
            coder.encode(quotient.astype(np.int32), cat, quotient_mass / selected_residue_mass[:, None])
        else:
            quotient = coder.decode(cat, quotient_mass / selected_residue_mass[:, None]).astype(np.int64)
        mantissa = quotient * stride + residue
        word_mass = mn[rows, mantissa]
        ideal = -np.log2(word_mass / float(TOTAL))
        if mode == "coset_uniform":
            ideal = ideal + bits + np.log2(selected_residue_mass / group_mass)
    return word_of(groups * 128 + mantissa), float(ideal.sum())


def coset_diagnostics(mu, inv_s, words, residue_bits, f_tab=None):
    """Matched ideal costs and residue counts; not an encoded-size estimate."""
    bits = _settings("coset_uniform", residue_bits)
    mean, inverse = _predictions(mu, inv_s)
    source = np.asarray(words)
    if source.dtype != np.uint16 or source.shape != mean.shape:
        raise ValueError("source words must be a matching flat uint16 vector")
    ranks = ordered(source)
    groups, mantissa = ranks >> 7, ranks & 127
    gn = group_num(mean, inverse, f_tab)
    mn = mant_num(mean, inverse, groups, f_tab)
    rows = np.arange(len(mean))
    group_mass = gn[rows, groups]
    stride = 1 << bits
    residue = mantissa & (stride - 1)
    residue_mass = mn.reshape(len(mean), 128 // stride, stride).sum(axis=1, dtype=np.int64)[rows, residue]
    direct = -np.log2(mn[rows, mantissa] / float(TOTAL))
    uniform = direct + bits + np.log2(residue_mass / group_mass)
    return {
        "values": len(mean),
        "residue_bits": bits,
        "direct_ideal_bits": float(direct.sum()),
        "coset_model_ideal_bits": float(direct.sum()),
        "coset_uniform_ideal_bits": float(uniform.sum()),
        "uniform_minus_direct_ideal_bits": float((uniform - direct).sum()),
        "residue_histogram": np.bincount(residue, minlength=stride).tolist(),
        "scope": "Ideal matched probability costs; actual range-coder bytes must be measured separately.",
    }
