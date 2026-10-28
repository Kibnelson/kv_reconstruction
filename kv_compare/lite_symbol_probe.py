"""Known-KV replay and rate estimates for exact small integer rank symbols.

This module does not implement a packet or an entropy decoder. Histograms and
frequency costs are diagnostics, not GPU codec throughput or complete wire
sizes. The original encoder arithmetic supplies every prediction and scale.
"""
import hashlib

import numpy as np
import torch

from .core import LN2X2, words_to_float
from .gpu_coder import _bf16_word_t, _ord_t, _word_t


DEFAULT_RADII = (127, 255, 511, 1023, 2047)
CONTEXTS = 32
TOTAL = 65536
ESCAPE_DIAGNOSTIC_LABELS = (
    'values', 'finite_sign_crossing', 'finite_predicted_zero_subnormal',
    'escapes', 'escape_nonfinite', 'escape_finite_sign_crossing',
    'escape_finite_predicted_zero_subnormal', 'escape_finite_same_sign_normal')


def _radius(radius):
    if type(radius) is not int or not 0 <= radius <= 32767:
        raise ValueError('radius must be an integer in [0,32767]')
    return radius


def replay(engine, words, start, callback):
    """Replay original lossless predictions using already known original KV.

    Callback arguments are layer index, group index, contiguous original int16
    words, contiguous float32 prediction and contiguous float32 scale. All
    layers, including layer zero, are emitted in original encoder order. The
    callback must not mutate its inputs. It may synchronize or retain tensors
    for diagnostics; this function has no receiver-speed interpretation.
    """
    if (not callable(callback) or words.dtype not in (torch.int16, torch.uint16, torch.int32, torch.int64) or words.ndim != 3 or
            tuple(words.shape)[::2] != (engine.L, engine.W) or
            not 0 < words.shape[1] <= 8192 or type(start) is not int or
            not 0 <= start <= 8192 - words.shape[1]):
        raise ValueError('invalid replay geometry/start/callback')
    source = words.to(device=engine.device, dtype=torch.int16)
    if bool(((((source.to(torch.int32) & 0x7F80) >> 7) >= 190)).any()):
        raise ValueError('original encoder uses raw fallback; no prediction replay exists')
    _, N, W = source.shape
    with engine.rope.request(start, N, engine.device):
        positions = torch.arange(start, start + N, device=engine.device)
        features = {}
        prior = torch.ones(N, dtype=torch.float32, device=engine.device)
        for l, layer in enumerate(engine.layers):
            external = []
            for kind in layer['kinds']:
                external.append(features[{'L0': 0, 'P': l - 1, 'P2': l - 2}[kind]])
            if external:
                Xe = torch.cat(external, 1).to(engine.dtype)
                base_mu = (Xe @ layer['ext']).float() + layer['bias']
            else:
                base_mu = layer['bias'].expand(N, W)
            cur = torch.zeros(N, W, dtype=torch.float32, device=engine.device)
            surprise = torch.zeros(N, dtype=torch.float32, device=engine.device)
            for gi, g in enumerate(layer['groups']):
                st, ch, pos = g['start'], g['ch'], g['pos']
                md = base_mu[:, pos]
                if st:
                    md = md + (cur[:, :st].to(engine.dtype) @ g['current']).float()
                m = engine.rotate(md, g, positions)
                m = torch.nan_to_num(m, nan=0., posinf=1e30, neginf=-1e30).clamp(-1e30, 1e30)
                spread = layer['spread'][ch][None, :]
                mult = ((64 * prior + surprise + 16) / (64 + st + 16)).clamp(.25, 4.)
                if not engine.meta['adaptive']:
                    mult = torch.ones_like(mult)
                s = (spread * mult[:, None]).expand_as(m).contiguous()
                rec = source[l][:, ch].contiguous()
                callback(l, gi, rec, m.contiguous(), s)
                if (engine.fusedfinish and engine.rope.scale == 1. and
                        engine.fuse_scatter and engine.fuse_rope):
                    xrec = engine.finish_group(rec, cur, g, positions)
                else:
                    xrec = words_to_float(rec)
                    engine.scatter(cur, pos, engine.rotate(xrec, g, positions, inverse=True))
                surprise += ((xrec.float() - m).abs() / (LN2X2 * spread)).sum(1)
            f = torch.empty_like(cur); engine.scatter(f, layer['perm'], cur)
            features[l] = f
            for j in [j for j in features if j < l - 1 and j != 0]:
                del features[j]
            prior = surprise / W


def _coordinates(words, mu):
    if (words.dtype not in (torch.int16, torch.uint16, torch.int32, torch.int64) or
            words.shape != mu.shape or words.device != mu.device):
        raise ValueError('words and prediction must share geometry/device')
    original = words.to(torch.int64) & 65535
    predicted = _bf16_word_t(mu.to(torch.float32))
    delta = _ord_t(original) - _ord_t(predicted)
    finite = (((original >> 7) & 255) != 255) & (((predicted >> 7) & 255) != 255)
    finite &= torch.isfinite(mu)
    return original, predicted, delta, finite


def symbolize(words, mu, radius):
    """Exact signed rank delta or ESC, with the original 16-bit escape word."""
    radius = _radius(radius)
    original, _, delta, finite = _coordinates(words, mu)
    inside = finite & (delta.abs() <= radius)
    symbols = torch.where(inside, delta + radius, 2 * radius + 1).to(torch.int32)
    raw = torch.where(inside, 0, original).to(torch.int16)
    return symbols, raw


def reconstruct(symbols, mu, raw, radius):
    """Reference inverse for symbols/escapes; preserve all BF16 payload bits."""
    radius = _radius(radius)
    if (symbols.shape != mu.shape or raw.shape != mu.shape or
            symbols.device != mu.device or raw.device != mu.device):
        raise ValueError('symbols, prediction and escapes must share geometry/device')
    symbols = symbols.to(torch.int64)
    escape = symbols == 2 * radius + 1
    predicted = _bf16_word_t(mu.to(torch.float32))
    index = _ord_t(predicted) + symbols - radius
    if bool(((symbols < 0) | (symbols > 2 * radius + 1) |
             ((~escape) & ((index < 0) | (index > 65535)))).any()):
        raise ValueError('invalid integer symbol or reconstructed rank')
    return torch.where(escape, raw.to(torch.int64) & 65535, _word_t(index)).to(torch.int16)


def scale_classes(mu, s, *, validate=True):
    """clamp(es-ep+7,0,31): exponent estimate of scale/BF16-prediction ULP.

    ep is the BF16 prediction exponent, es the float32 scale exponent. Both
    exponent fields are clamped to at least one for zero/subnormal handling.
    No channel, layer, TEST data or learned class boundary enters this rule.
    validate=False is reserved for trusted replay's checked positive scales;
    the public default checks them, including a possible CUDA scalar readback.
    """
    if mu.shape != s.shape or mu.device != s.device:
        raise ValueError('prediction and scale must share geometry/device')
    sf = s.to(torch.float32)
    if validate and bool(((sf <= 0) | (~torch.isfinite(sf))).any()):
        raise ValueError('finite positive scales required')
    predicted = _bf16_word_t(mu.to(torch.float32))
    ep = ((predicted >> 7) & 255).clamp_min(1)
    es = ((sf.contiguous().view(torch.int32).to(torch.int64) >> 23) & 255).clamp_min(1)
    return (es - ep + 7).clamp(0, CONTEXTS - 1).to(torch.int64)


def fold_counts(maxcounts, radius, contexts=CONTEXTS):
    """Fold a max-radius histogram into a smaller exact alphabet.

    Last axes are [class, symbol]; optional leading layer axes are retained.
    Symbol order is delta=-R..R followed by ESC. Out-of-radius bins and the
    existing ESC are merged. contexts=1 sums class counts for the ablation.
    """
    radius = _radius(radius)
    if contexts not in (1, CONTEXTS) or maxcounts.ndim < 2 or maxcounts.shape[-2] not in (1, CONTEXTS):
        raise ValueError('invalid histogram context geometry')
    alphabet = maxcounts.shape[-1]
    maximum = (alphabet - 2) // 2
    if alphabet < 2 or alphabet % 2 or maximum > 32767 or radius > maximum:
        raise ValueError('invalid maximum histogram radius')
    first, end = maximum - radius, maximum + radius + 1
    core = maxcounts[..., first:end]
    if isinstance(maxcounts, torch.Tensor):
        escape = maxcounts.sum(-1, keepdim=True) - core.sum(-1, keepdim=True)
        result = torch.cat((core, escape), -1)
        return result.sum(-2, keepdim=True) if contexts == 1 else result
    escape = np.sum(maxcounts, axis=-1, keepdims=True) - np.sum(core, axis=-1, keepdims=True)
    result = np.concatenate((core, escape), axis=-1)
    return result.sum(axis=-2, keepdims=True) if contexts == 1 else result


def symbol_histograms(words, mu, s, radii=DEFAULT_RADII, *, validate=True):
    """Diagnostic counts with one unweighted maximum-radius bincount per group.

    validate=False avoids repeated scalar readbacks only for trusted replay
    inputs. The observer/histogram work is not receiver throughput evidence.
    """
    radii = tuple(_radius(r) for r in radii)
    if not radii or len(set(radii)) != len(radii):
        raise ValueError('nonempty unique radii required')
    maximum = max(radii); alphabet = 2 * maximum + 2
    _, _, delta, finite = _coordinates(words, mu)
    classes = scale_classes(mu, s, validate=validate)
    symbols = torch.where(finite & (delta.abs() <= maximum), delta + maximum, alphabet - 1)
    counts = torch.bincount((classes * alphabet + symbols).reshape(-1),
                            minlength=CONTEXTS * alphabet).reshape(CONTEXTS, alphabet)
    return {r: counts if r == maximum else fold_counts(counts, r) for r in radii}


def update_histograms(destination, words, mu, s):
    """Accumulate per-radius diagnostic counters without moving them to CPU."""
    addition = symbol_histograms(words, mu, s, tuple(destination))
    for radius, counts in addition.items():
        if (destination[radius].shape != counts.shape or destination[radius].dtype != torch.int64 or
                destination[radius].device != counts.device):
            raise ValueError('histogram destination differs from computed counts')
        destination[radius].add_(counts)
    return destination


def escape_diagnostics(words, mu, radii=DEFAULT_RADII):
    """Predeclared [radius, category] explanatory counts, with overlapping bins.

    Finite sign crossings and finite zero/subnormal predictions can overlap.
    Nonfinite source/prediction pairs have their own escape category. These
    counters explain rank-tail behavior; they do not choose classes or radii.
    """
    radii = tuple(_radius(r) for r in radii)
    if not radii or len(set(radii)) != len(radii):
        raise ValueError('nonempty unique radii required')
    original, predicted, delta, finite = _coordinates(words, mu)
    crossing = finite & (((original ^ predicted) & 32768) != 0)
    zero_subnormal = finite & (((predicted >> 7) & 255) == 0)
    rows = []
    for radius in radii:
        escape = (~finite) | (delta.abs() > radius)
        flags = torch.stack((torch.ones_like(finite), crossing, zero_subnormal, escape,
            ~finite, escape & crossing, escape & zero_subnormal,
            escape & finite & (~crossing) & (~zero_subnormal)), dim=-1)
        rows.append(flags.reshape(-1, len(ESCAPE_DIAGNOSTIC_LABELS)).sum(0))
    return torch.stack(rows)


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def fit_tables(counts):
    """Fit positive 16-bit frequencies from TRAIN counts, with exact tie order.

    Reserve one slot per symbol, distribute the remaining budget in proportion
    to counts, then give leftover slots to largest integer remainders (lower
    symbol index wins ties). Empty classes use the same rule with uniform
    counts. Python integer products avoid overflow in deterministic rounding.
    """
    counts = _numpy(counts)
    if (counts.ndim != 2 or counts.dtype.kind not in 'iu' or np.any(counts < 0) or
            not 1 <= counts.shape[1] <= TOTAL or counts.shape[0] not in (1, CONTEXTS)):
        raise ValueError('nonnegative integer [1|32,alphabet<=65536] TRAIN counts required')
    classes, alphabet = counts.shape
    frequency = np.empty(counts.shape, dtype=np.int32)
    budget = TOTAL - alphabet
    for c in range(classes):
        weights = [int(value) for value in counts[c]]
        total = sum(weights)
        if total == 0:
            weights, total = [1] * alphabet, alphabet
        quotient, remainder = zip(*(divmod(value * budget, total) for value in weights))
        row = np.asarray(quotient, dtype=np.int32) + 1
        left = TOTAL - int(row.sum())
        order = sorted(range(alphabet), key=lambda j: (-remainder[j], j))
        row[order[:left]] += 1
        frequency[c] = row
    cdf = np.concatenate((np.zeros((classes, 1), dtype=np.int32),
                          np.cumsum(frequency, axis=1, dtype=np.int32)), axis=1)
    physical = frequency.nbytes + cdf.nbytes
    inverse = classes * TOTAL * np.dtype('<u2').itemsize
    digest = hashlib.sha256(frequency.astype('<i4', copy=False).tobytes() +
                            cdf.astype('<i4', copy=False).tobytes()).hexdigest()
    return dict(freq=frequency, cdf=cdf, table_sha256=digest,
        physical_probe_table_bytes=physical, projected_inverse_lut_bytes=inverse,
        projected_decoder_table_bytes=physical + inverse,
        normalization='positive 16-bit integer largest remainder; stable ascending symbol ties',
        scope='frequency/CDF arrays physically allocated in probe; uint16 full inverse lookup projected only; stream states, directories and framing additional')


def cost(counts, freq):
    """Estimated entropy bits plus 16 raw bits per ESC; no framing or states."""
    counts = _numpy(counts)
    frequency = _numpy(freq['freq'] if isinstance(freq, dict) else freq)
    if (counts.ndim < 2 or counts.shape[-2:] != frequency.shape or
            counts.dtype.kind not in 'iu' or frequency.dtype.kind not in 'iu' or np.any(counts < 0) or
            np.any(frequency <= 0) or np.any(np.sum(frequency, axis=-1) != TOTAL)):
        raise ValueError('counts and positive normalized frequencies differ')
    symbol_bits = float(np.sum(counts.astype(np.float64) * (16. - np.log2(frequency))))
    escapes = int(np.sum(counts[..., -1], dtype=np.int64))
    values = int(np.sum(counts, dtype=np.int64))
    escape_bits = 16 * escapes
    total_bits = symbol_bits + escape_bits
    return dict(total_bits=total_bits, symbol_bits=symbol_bits, escape_bits=escape_bits,
        escapes=escapes, values=values, bpv=total_bits / values if values else None,
        scope='rate estimate only; no entropy packet, encoder or decoder measured; framing, directory and padded stream states excluded')
