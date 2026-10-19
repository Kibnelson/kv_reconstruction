"""Build and verify sampled recipes directly from returned device prefill KV.

This observer runs between prefill chunks. It never exports source KV through
CPU to encode a recipe. The model still retains its full raw cache; this is an
encoding/exactness experiment, not compact-resident attention or offload.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from . import gpu_ops as g
from .measurements import _Device, _sizes
from .model_runtime import _pairs, cache_geometry


class PrefillRecipeObserver:
    def __init__(self, specs, total_tokens, tile_tokens, device, max_records=72):
        if total_tokens < 1 or tile_tokens < 1 or (max_records is not None and max_records < 1):
            raise ValueError("positive token/tile/record limits required")
        if not specs or any(s["method"] != "prefill" for s in specs):
            raise ValueError("prefill observer requires prefill recipe specifications")
        if len({s["name"] for s in specs}) != len(specs):
            raise ValueError("duplicate prefill recipe names")
        self.specs = specs
        self.total_tokens, self.tile_tokens = int(total_tokens), int(tile_tokens)
        self.max_records = max_records
        self.device = _Device(device)
        if self.device.accelerated and self.device.device.index is None:
            self.device.device = self.device.torch.device(self.device.kind, self.device.api.current_device())
        self.state = g.make_state({}, device=self.device.device)
        self.meta = None
        self.pending = []
        self.end = 0
        self.chunks = 0
        self.rows = {s["name"]: dict(status="ok", payload=0, raw=0, checked=0,
            components=Counter(), encode=0., decode=0., encode_event=0., decode_event=0.) for s in specs}

    def _plan(self, geometry):
        tiles = math.ceil(self.total_tokens / self.tile_tokens)
        total = geometry["layers"] * 2 * tiles
        count = total if self.max_records is None else min(total, self.max_records)
        ordinals = np.linspace(0, total - 1, count, dtype=np.int64).tolist()
        self.pending = [(int(i) // (2 * tiles), (int(i) // tiles) % 2,
                         (int(i) % tiles) * self.tile_tokens) for i in ordinals]
        self.meta = dict(layers=geometry["layers"], layer_indices=list(range(geometry["layers"])),
            tokens=self.total_tokens, channels=geometry["columns"], tile_tokens=self.tile_tokens,
            records_total=total, records_measured=count, coverage="full" if count == total else "sampled",
            size_is_estimated=count != total,
            full_raw_payload_bytes=geometry["layers"] * 2 * self.total_tokens * geometry["columns"] * 2)

    def __call__(self, cache, start, end, total_tokens):
        if total_tokens != self.total_tokens or start != self.end or not start < end <= total_tokens:
            raise ValueError("prefill observer chunk coverage mismatch")
        geometry = cache_geometry(cache)
        if geometry["tokens"] != end:
            raise ValueError("prefill observer requires the full returned prefix cache")
        if self.meta is None:
            self._plan(geometry)
        if geometry["layers"] != self.meta["layers"] or geometry["columns"] != self.meta["channels"]:
            raise ValueError("prefill observer cache geometry changed")
        pairs = _pairs(cache)
        remaining = []
        for layer, kind_index, token_start in self.pending:
            stop = min(token_start + self.tile_tokens, self.total_tokens)
            if stop > end:
                remaining.append((layer, kind_index, token_start))
                continue
            tensor = pairs[layer][kind_index]
            # Only new complete selected tiles are made contiguous on-device.
            words = tensor[0, :, token_start:stop, :].permute(1, 0, 2).contiguous()
            words = words.view(self.device.torch.uint16).reshape(stop - token_start, -1)
            if words.device != self.device.device:
                raise ValueError("prefill source KV is not on the selected device")
            for spec in self.specs:
                row = self.rows[spec["name"]]
                if row["status"] != "ok":
                    continue
                try:
                    packet, encode_seconds, encode_event = self.device.timed(lambda:
                        g.encode(words, self.state, method="prefill", key=f"layer_{layer:02d}_{'KV'[kind_index]}",
                                 **spec.get("options", {})))
                    restored, decode_seconds, decode_event = self.device.timed(lambda: g.decode(packet, self.state))
                    if not self.device.torch.equal(restored.to(self.device.torch.int32), words.to(self.device.torch.int32)):
                        raise AssertionError("live prefill KV reconstruction differs from original words")
                    stats = packet.stats
                    if sum(stats["component_bytes"].values()) != stats["serialized_bytes"]:
                        raise AssertionError("live prefill byte accounting mismatch")
                    row["payload"] += stats["serialized_bytes"]
                    row["raw"] += words.numel() * words.element_size()
                    row["components"].update(stats["component_bytes"])
                    row["checked"] += 1
                    row["encode"] += encode_seconds
                    row["decode"] += decode_seconds
                    row["encode_event"] += encode_event or 0.
                    row["decode_event"] += decode_event or 0.
                    del packet, restored
                except Exception as error:
                    if isinstance(error, AssertionError):
                        status = "failed_correctness"
                    elif isinstance(error, NotImplementedError) or any(s in str(error).lower() for s in
                            ("not supported", "unsupported", "not implemented", "float64")):
                        status = "unsupported"
                    else:
                        status = "error"
                    row.update(status=status, error=f"{type(error).__name__}: {error}")
            del words
        self.pending = remaining
        self.end, self.chunks = end, self.chunks + 1

    def report(self):
        complete = self.meta is not None and self.end == self.total_tokens and not self.pending
        rows = []
        for spec in self.specs:
            value = self.rows[spec["name"]]
            row = dict(name=spec["name"], track="prefill_observer", spec=spec, status=value["status"])
            if "error" in value:
                row["error"] = value["error"]
            elif not complete or value["checked"] != self.meta["records_measured"]:
                row.update(status="error", error="Incomplete prefill recipe observation")
            else:
                row["measurement"] = dict(**self.meta,
                    **_sizes(self.meta, value["payload"], value["raw"], 0, value["components"]),
                    execution="prefill_observer_accelerator" if self.device.accelerated else "prefill_observer_cpu_functional",
                    device=str(self.device.device), roundtrip_exact=True, exact_verified=True,
                    records_checked=value["checked"], decoder_state_info={"accounting": "No external dictionary or activations required"},
                    encode_seconds_total=value["encode"], decode_seconds_total=value["decode"],
                    device_event_elapsed=dict(encode_seconds_total=value["encode_event"] if self.device.accelerated else None,
                                              decode_seconds_total=value["decode_event"] if self.device.accelerated else None),
                    input_staging_seconds_total=0., source_KV_cpu_staging=False, full_raw_cache_retained=True,
                    packets_retained=False, model_prefill_time_included=False,
                    scope="Recipe derived from actual returned device KV between prefill chunks, before CPU export. Selected complete tiles only. Encoder and one verification decoder timed separately; layout creation, equality guards and callbacks add extra observer overhead. All packet fields counted; outer framing charged arithmetically. Model still retains full raw KV; no compact attention, lower model-cache peak or serving gain established.")
            rows.append(row)
        return dict(enabled=True, complete=complete, chunks_observed=self.chunks,
            tokens_observed=self.end, source="returned_device_KV_during_chunked_prefill", results=rows,
            scope="Synchronous instrumentation between prefill chunks. Capture prefill timing includes observer work; do not compare it to an uninstrumented capture as pure model prefill time.")
