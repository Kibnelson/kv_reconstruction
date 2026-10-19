"""Small per-head producer summaries; no fitting or raw activation output.

All work is linear in supplied words, using only 256-bin histograms and
matched-channel sufficient statistics. Run outside model/codec timed regions.
"""
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


_SOURCES = {"final_K": ("k", "cost_final"),
            "pre_rope_K": ("k_pre_rope", "cost_pre_rope"),
            "kproj_K": ("k_proj", "cost_kproj")}
_FIELDS = ("k", "v", "k_pre_rope", "k_proj")
_ENTROPY_WARNING = ("Finite-sample plug-in marginal entropy can be downward biased, "
    "especially with many singleton symbols or unseen outcomes. This is neither "
    "a lower bound, a confidence interval nor an achievable encoded size.")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError("invalid conditional diagnostic " + name)
    return int(value)


def _validate(records, models):
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("conditional diagnostics require nonempty producer records")
    if not isinstance(models, Mapping):
        raise ValueError("conditional diagnostics require frozen source cost models")
    grouped, geometry = defaultdict(list), {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("conditional producer record must be a mapping")
        layer = _integer(record.get("layer"), "layer")
        start = _integer(record.get("start"), "start")
        tokens = _integer(record.get("tokens"), "tokens")
        dim = _integer(record.get("head_dim"), "head_dim", 1)
        arrays = {}
        for name in _FIELDS:
            value = np.asarray(record.get(name))
            if value.dtype != np.uint16 or value.ndim != 2:
                raise ValueError("conditional sources must be uint16[T,C] BF16 words")
            arrays[name] = value
        shape = arrays["k"].shape
        if shape[0] != tokens or shape[1] < 1 or shape[1] % dim or any(
                value.shape != shape for value in arrays.values()):
            raise ValueError("conditional source shapes disagree with actual head geometry")
        if layer in geometry and geometry[layer] != (shape[1], dim):
            raise ValueError("conditional head geometry changed within a layer")
        geometry[layer] = (shape[1], dim)
        grouped[layer].append((start, arrays))
    bases = {}
    for layer, rows in grouped.items():
        rows.sort(key=lambda row: row[0])
        nonempty = [(start, start + len(arrays["k"])) for start, arrays in rows if len(arrays["k"])]
        if any(right[0] < left[1] for left, right in zip(nonempty, nonempty[1:])):
            raise ValueError("overlapping conditional records duplicate observations")
        columns = geometry[layer][0]
        for model_name in ("cost_final", "cost_pre_rope", "cost_kproj"):
            model = models.get(model_name)
            params = getattr(model, "params", None)
            key = f"layer_{layer:02d}"
            if not isinstance(params, Mapping) or key not in params:
                raise ValueError("missing frozen conditional source model " + model_name)
            for kind in ("K", "V"):
                base = np.asarray(params[key].get("base_" + kind))
                if base.dtype != np.uint8 or base.shape != (columns,):
                    raise ValueError("conditional frozen bases disagree with channel geometry")
                bases[(layer, model_name, kind)] = base
    return grouped, geometry, bases


def _entropy(histogram):
    positive = histogram[histogram > 0]
    n, occupied = int(positive.sum()), int(len(positive))
    probabilities = positive.astype(np.float64) / n if n else np.empty(0)
    return dict(observations=n, occupied_symbols=occupied,
        singleton_symbols=int(np.count_nonzero(positive == 1)),
        empirical_bits_per_value=float(-(probabilities * np.log2(probabilities)).sum()) if n else None,
        samples_per_occupied_symbol=n / occupied if occupied else None,
        heuristic_adequate_samples=bool(n >= 64 and n >= 5 * occupied),
        reliability_warning=_ENTROPY_WARNING)


def _tier(words, base):
    error = ((words.astype(np.int32) >> 7) & 255) - base.astype(np.int32)[None, :]
    zigzag = np.where(error >= 0, 2 * error, -2 * error - 1)
    secondary, rare = int(np.count_nonzero(zigzag >= 3)), int(np.count_nonzero(zigzag >= 10))
    return int(words.size), secondary, rare


def _tier_report(counts):
    n, secondary, rare = map(int, counts)
    bits = 2 * n + 3 * secondary + 9 * rare
    return dict(values=n, payload_bits=bits, bits_per_value=bits / n if n else None,
        secondary_escapes=secondary, rare_escapes=rare)


def _numeric(words):
    # Signaling NaN word patterns may raise NumPy's invalid flag when widened;
    # both NaNs and infinities are explicitly filtered by the paired statistic.
    with np.errstate(invalid="ignore"):
        return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


class _Pearson:
    """Stable per-channel pooled covariance; no channel-by-channel matrix."""
    def __init__(self, channels):
        self.n = np.zeros(channels, np.int64)
        self.mx = np.zeros(channels); self.my = np.zeros(channels)
        self.xx = np.zeros(channels); self.yy = np.zeros(channels); self.xy = np.zeros(channels)

    def add(self, x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        n = mask.sum(axis=0)
        clean_x, clean_y = np.where(mask, x, 0.), np.where(mask, y, 0.)
        mx = np.divide(clean_x.sum(axis=0), n, out=np.zeros_like(self.mx), where=n > 0)
        my = np.divide(clean_y.sum(axis=0), n, out=np.zeros_like(self.my), where=n > 0)
        dx, dy = np.where(mask, clean_x - mx, 0.), np.where(mask, clean_y - my, 0.)
        total = self.n + n
        fraction = np.divide(n, total, out=np.zeros_like(self.mx), where=total > 0)
        factor = self.n.astype(np.float64) * fraction
        delta_x, delta_y = mx - self.mx, my - self.my
        self.xx += (dx * dx).sum(axis=0) + delta_x * delta_x * factor
        self.yy += (dy * dy).sum(axis=0) + delta_y * delta_y * factor
        self.xy += (dx * dy).sum(axis=0) + delta_x * delta_y * factor
        self.mx += delta_x * fraction; self.my += delta_y * fraction; self.n = total

    def report(self, absolute=False):
        valid = (self.n >= 2) & (self.xx > 0) & (self.yy > 0)
        values = (self.xy[valid] / (np.sqrt(self.xx[valid]) * np.sqrt(self.yy[valid]))).clip(-1., 1.)
        if absolute:
            values = np.abs(values)
        return dict(valid_channels=int(valid.sum()), total_channels=int(len(valid)),
            insufficient_finite_channels=int(np.count_nonzero(self.n < 2)),
            constant_channels=int(np.count_nonzero((self.n >= 2) & ~valid)),
            finite_matched_values=int(self.n.sum()),
            finite_rows_per_channel=dict(minimum=int(self.n.min()), median=float(np.median(self.n)), maximum=int(self.n.max())),
            median=float(np.median(values)) if len(values) else None,
            p90=float(np.quantile(values, .9)) if len(values) else None,
            maximum=float(values.max()) if len(values) else None)


def analyze(records, models):
    """Summarize supplied heads with TRAIN bases, without fitting or selection.

    Zero-token tails are accepted and counted, with zero observations. Temporal
    sign comparisons always reset at every record, including contiguous tiles.
    """
    grouped, geometry, bases = _validate(records, models)
    layers = []
    for layer in sorted(grouped):
        columns, dim = geometry[layer]
        rows = grouped[layer]
        heads = []
        for head in range(columns // dim):
            section = slice(head * dim, (head + 1) * dim)
            histogram = {field: np.zeros(256, np.int64) for field in _FIELDS}
            transitions = {field: [0, 0] for field in _FIELDS}
            costs = {(name, kind): np.zeros(3, np.int64)
                     for name in ("cost_final", "cost_pre_rope", "cost_kproj") for kind in ("K", "V")}
            signed, magnitude = _Pearson(dim), _Pearson(dim)
            for _, arrays in rows:
                for field in _FIELDS:
                    words = arrays[field][:, section]
                    lower = (((words >> 8) & 128) | (words & 127)).astype(np.uint8)
                    histogram[field] += np.bincount(lower.reshape(-1), minlength=256)
                    signs = words >> 15
                    transitions[field][0] += int(np.count_nonzero(signs[1:] != signs[:-1]))
                    transitions[field][1] += max(len(words) - 1, 0) * dim
                for _, (field, model_name) in _SOURCES.items():
                    for kind, source in (("K", field), ("V", "v")):
                        costs[(model_name, kind)] += _tier(arrays[source][:, section], bases[(layer, model_name, kind)][section])
                x, y = _numeric(arrays["k_pre_rope"][:, section]), _numeric(arrays["v"][:, section])
                signed.add(x, y); magnitude.add(np.abs(x), np.abs(y))
            domains = {}
            for name, field in (("final_K", "k"), ("pre_rope_K", "k_pre_rope"),
                                ("kproj_K", "k_proj"), ("final_V", "v")):
                changed, eligible = transitions[field]
                domains[name] = dict(lower8_marginal_entropy=_entropy(histogram[field]),
                    sign_transitions=dict(changed=changed, eligible=eligible,
                        rate=changed / eligible if eligible else None))
            heads.append(dict(head=head, domains=domains,
                frozen_exponent_tiers={name: {kind: _tier_report(costs[(name, kind)]) for kind in ("K", "V")}
                    for name in ("cost_final", "cost_pre_rope", "cost_kproj")},
                matched_channel_K_pre_rope_V=dict(absolute_signed_pearson=signed.report(absolute=True),
                    pearson_of_absolute_values=magnitude.report())))
        layers.append(dict(layer=layer, head_dim=dim, kv_heads=columns // dim,
            supplied_records=len(rows), selected_token_rows=sum(len(arrays["k"]) for _, arrays in rows), heads=heads))
    return dict(format="qwen_conditional_head_diagnostics_v1", raw_arrays_included=False,
        fitting_performed=False, selector_implemented=False, records_supplied=len(records), layers=layers,
        exponent_scope="Actual 2+3*[zigzag(error)>=3]+9*[zigzag(error)>=10] payload bits against each frozen TRAIN source base. Excludes byte padding, indexes, headers, state, replay corrections and selector metadata; these are not complete codec sizes or savings.",
        temporal_scope="Matching-channel adjacent rows inside each supplied record only. Every record boundary resets transitions; sampled gaps and adjacent tiles are never bridged.",
        correlation_scope="Matched numerical BF16 K_pre_RoPE and V channels, pooled over supplied rows with pairwise finite filtering. Reports abs(Pearson(K,V)) and separately Pearson(abs(K),abs(V)); no cross-channel covariance, fitting, significance test or predictive coding result.",
        scope="Selected-record per-head descriptive metadata only. Heterogeneity does not establish a usable head selector, achievable compression, GPU performance, capacity or a universal entropy lower bound.")
