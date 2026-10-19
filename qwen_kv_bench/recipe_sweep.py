"""Frozen incremental recipe candidates and TRAIN-only artifact integration."""
from __future__ import annotations

import hashlib
from pathlib import Path


PREFILL_METHODS = ("prefill", "prefill_exponent", "prefill_dictionary")


def insight_options():
    from .gpu_ops import native_options, insight_layout_options
    from .exponent_recipe import options
    result = native_options("main") + insight_layout_options() + options()
    for layout in ("channel", "time"):
        for count in (16, 64):
            result.append(dict(name=f"native_prefill_affine_dictionary{count}_{layout}_g8_residual",
                method="prefill_dictionary", recipe_model=f"recipe_residual{count}_{layout}",
                dictionary_size=count, options=dict(layout=layout, group_size=8, assignment="bitcost")))
    return result


def train_recipe_artifacts(artifacts, specs, train_caches, output_dir, *, tile_tokens,
                           seed, max_samples_per_key, iterations, training_metadata, model_identity):
    from .residual_dictionary import train
    from .reporting import write_json
    requested = {s["recipe_model"]: s for s in specs if s["method"] == "prefill_dictionary"}
    for name, spec in sorted(requested.items()):
        model = train(train_caches, output_dir, tile_tokens=tile_tokens,
            layout=spec["options"]["layout"], n_centroids=spec["dictionary_size"],
            seed=seed, max_samples_per_key=max_samples_per_key, iterations=iterations,
            training_metadata=training_metadata, model_identity=model_identity)
        path = Path(model.path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != model.model_id or path.stat().st_size != model.state_bytes:
            raise AssertionError("residual recipe state accounting mismatch")
        artifacts.models[name] = model
        artifacts.paths[name] = str(path)
        artifacts.state_bytes[name] = model.state_bytes
        artifacts.metadata["models"][name] = dict(file=path.name, family="prefill_residual",
            trained=True, n_centroids=spec["dictionary_size"], serialized_bytes=model.state_bytes,
            model_id=digest, sha256=digest, config=model.config,
            parameter_array_bytes=sum(book.nbytes for book in model.codebooks.values()),
            provenance=model.metadata)
    if requested:
        artifacts.metadata["recipe_training_scope"] = (
            "Residual models use the same three TRAIN captures only, after tile-local affine prediction; "
            "layout, tile length and model identity are bound into each full serialized artifact. "
            "Assignment minimizes group bit width with an L1 tie-break; median updates are heuristic.")
        write_json(artifacts.report_path, artifacts.metadata)
    return artifacts


def state_charge(state):
    resident = state.physical_nbytes
    serialized = int(getattr(state, "residual_dictionary_serialized_bytes", 0))
    return max(resident, serialized)


def state_info(state):
    serialized = int(getattr(state, "residual_dictionary_serialized_bytes", 0))
    return dict(accounting=("Conservative maximum of complete serialized residual model and actual distinct device tensor storage; both quantities reported separately."
        if serialized else "Actual distinct tensor storage; no external recipe model file."),
        serialized_bytes=serialized, physical_tensor_bytes=state.physical_nbytes,
        charged_bytes=state_charge(state), state_id=state.state_id,
        model_config=getattr(state, "residual_dictionary_config", None))
