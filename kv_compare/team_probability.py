"""Calibrate existing probability spreads against real, per-head stream bytes.

The frozen codec has one independent rANS segment per token block and head.
Here a head receives one multiplier, shared by all layers, on its existing
``sp`` values. Predictors, feature scales, ``inv`` and CDF tables stay fixed.
Consequently the model's adaptive priors remain unchanged and each head's
encoded stream can be scored independently. The selected combination must
still be re-encoded and checked; these helpers do not manufacture a packet.

This is probability calibration of the existing exact BF16 codec. It adds no
decoder kernel, matrix product or per-request hint, and makes no novelty or
compression claim without actual packet measurements on untouched TEST words.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from .team_packet import geometry

FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)


def _positive_factor(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("spread factors must be finite positive numbers")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("spread factors must be finite positive numbers")
    return value


def _factors(values, count):
    try:
        values = tuple(_positive_factor(v) for v in values)
    except TypeError as error:
        raise ValueError("one spread factor per head required") from error
    if len(values) != count:
        raise ValueError("one spread factor per head required")
    return values


def _state_shape(consts, rank):
    if not isinstance(consts, dict) or not {"sp", "inv"} <= consts.keys():
        raise ValueError("dense state requires sp and inv")
    spread = consts["sp"]
    if not isinstance(spread, torch.Tensor) or spread.dtype != torch.float32 or spread.ndim != 2:
        raise ValueError("spread state must be a two-dimensional FP32 tensor")
    shape = tuple(spread.shape)
    if not all(shape):
        raise ValueError("spread state cannot be empty")
    for state in (consts, rank):
        if state is None:
            continue
        if not isinstance(state, dict) or not {"sp", "inv"} <= state.keys():
            raise ValueError("rank state requires sp and inv")
        for name in ("sp", "inv"):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                    tuple(value.shape) != shape or not bool(torch.isfinite(value).all()) or
                    not bool((value > 0).all())):
                raise ValueError("dense/rank sp and inv must be finite positive FP32 tensors of equal shape")
    return shape


def _clone(state):
    if state is None:
        return None
    result = {}
    for name, value in state.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.clone()
        elif name == "rank" and type(value) is int and value > 0:
            result[name] = value
        else:
            raise ValueError("fixed state must contain tensors and optional integer rank")
    return result


def scale_state(consts, rank, dense_factors, rank_factors, head_dim=128):
    """Clone fixed state and scale only dense/rank ``sp``, leaving ``inv`` fixed.

    Factor vectors are head order: key heads first, then value heads. Each
    factor is broadcast across that head's channels and every layer. The
    frozen snapshot has 128-channel heads; other geometries are rejected.
    """
    _layers, width = _state_shape(consts, rank)
    if type(head_dim) is not int or head_dim != 128 or width % head_dim:
        raise ValueError("the frozen codec requires complete 128-channel heads")
    dense = _factors(dense_factors, width // head_dim)
    if rank is None:
        if rank_factors is not None:
            raise ValueError("rank factors supplied without rank state")
        reduced = None
    else:
        reduced = _factors(rank_factors, width // head_dim)
    cloned, cloned_rank = _clone(consts), _clone(rank)
    for state, factors in ((cloned, dense), (cloned_rank, reduced)):
        if state is None:
            continue
        vector = torch.tensor(factors, dtype=state["sp"].dtype, device=state["sp"].device)
        state["sp"] = state["sp"] * vector.repeat_interleave(head_dim)[None, :]
    assert_spread_only(consts, rank, cloned, cloned_rank)
    return cloned, cloned_rank


def _same_bits(a, b):
    return (a.dtype == b.dtype and a.shape == b.shape and a.device == b.device and
            torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)))


def assert_spread_only(beforeconst, beforerank, afterconst, afterrank):
    """Reject any tensor/metadata change except finite positive ``sp`` values."""
    if _state_shape(beforeconst, beforerank) != _state_shape(afterconst, afterrank):
        raise ValueError("probability calibration changed state dimensions")
    for before, after in ((beforeconst, afterconst), (beforerank, afterrank)):
        if before is None or after is None:
            if before is not after:
                raise ValueError("probability calibration changed rank presence")
            continue
        if set(before) != set(after):
            raise ValueError("probability calibration changed state fields")
        for name in before:
            old, new = before[name], after[name]
            if isinstance(old, torch.Tensor):
                if not isinstance(new, torch.Tensor) or old.dtype != new.dtype or old.shape != new.shape or old.device != new.device:
                    raise ValueError("probability calibration changed tensor layout: " + name)
                if name != "sp" and not _same_bits(old, new):
                    raise ValueError("probability calibration changed a frozen tensor: " + name)
            elif type(old) is not type(new) or old != new:
                raise ValueError("probability calibration changed fixed metadata: " + name)
    return True


def head_word_costs(packet):
    """Return actual compact words per head, charging state and alignment words.

    Header/directory/checksum bytes are not split among heads. They remain in
    the packet's complete byte count and must be charged by the experiment.
    """
    packet.validate()
    _layers, _tokens, width, head_dim, segments, _capacity = geometry(packet.header)
    heads = width // head_dim
    if segments % heads:
        raise ValueError("segment directory does not contain complete head groups")
    lengths = packet.offsets[1:] - packet.offsets[:-1]
    return [int(v) for v in lengths.reshape(-1, heads).sum(dim=0).tolist()]


def choose_head_factors(observations, factors=FACTORS):
    """Choose actual minimum-word factors; ties prefer 1, then closest to 1.

    Each observation is ``dict(factor=number, head_words=[positive ints])``.
    Exactly one observation per requested grid factor is required. This is
    calibration selection only; the returned prediction must be verified by
    re-encoding the composed fixed state before TEST evaluation.
    """
    grid = tuple(_positive_factor(v) for v in factors)
    if not grid or len(set(grid)) != len(grid) or 1.0 not in grid:
        raise ValueError("factor grid must be distinct and include the frozen 1.0 control")
    records = {}
    heads = None
    for observation in observations:
        if not isinstance(observation, dict) or not {"factor", "head_words"} <= observation.keys():
            raise ValueError("calibration observation requires factor and head_words")
        factor = _positive_factor(observation["factor"])
        if factor not in grid or factor in records:
            raise ValueError("unexpected or duplicate calibration factor")
        costs = observation["head_words"]
        if not isinstance(costs, (list, tuple)) or not costs or any(type(v) is not int or v <= 0 for v in costs):
            raise ValueError("head costs must be nonempty positive integer word counts")
        if heads is None:
            heads = len(costs)
        if len(costs) != heads:
            raise ValueError("calibration observations disagree on head count")
        records[factor] = list(costs)
    if set(records) != set(grid):
        raise ValueError("calibration observations do not cover the factor grid")
    selected = [min(grid, key=lambda factor: (records[factor][head], factor != 1.0,
                    abs(factor - 1.0), factor)) for head in range(heads)]
    words = [records[factor][head] for head, factor in enumerate(selected)]
    return dict(factors=selected, head_words=words, total_words=sum(words),
                baseline_words=sum(records[1.0]), factor_grid=list(grid))
