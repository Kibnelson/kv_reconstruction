"""Freeze development choices and code before any fresh binary final capture."""
import datetime
import hashlib
import json
import pathlib
import shutil


def main():
    root = pathlib.Path("kv_binary")
    out = root / "results"
    if pathlib.Path(".kv_binary_data/final_manifest.json").exists() or (root / "selection.json").exists():
        raise RuntimeError("refusing to overwrite a frozen experiment")
    dev = json.loads((out / "dev.json").read_text())
    assert len(dev["cases"]) == 1 and all(x["all_exact"] for x in dev["aggregate"].values())
    unit = (out / "unit_tests.txt").read_text()
    assert "\nOK" in unit and "FAILED" not in unit
    config = {**dev["config"], "development_only": False, "phase": "fresh_final"}
    assert config["tile_tokens"] == 16
    (root / "final_config.json").write_text(json.dumps(config, indent=2) + "\n")
    score = lambda name: dev["aggregate"][name]["savings_including_state_percent"]["1"]
    binary = [name for name in config["candidates"] if "binary" in name]
    identity = [name for name in binary if "identity" in name]
    rotated = [name for name in binary if "dense" in name or "hadamard" in name]
    assert identity and rotated
    generation = [max(identity, key=score), max(rotated, key=score)]
    paths = {p for p in root.glob("*.py") if p.name != "finalize.py"} | set((root / "models").glob("*.bin"))
    paths.add(root / "final_config.json")
    # Reuse every frozen dependency from the earlier harness, without editing
    # any source or selecting settings on its final captures.
    prior = json.loads(pathlib.Path("kv_residual/selection.json").read_text())
    for name, expected in prior["source_and_model_sha256"].items():
        path = pathlib.Path(name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        if path.suffix in (".py", ".cpp", ".bin") or name == ".smoke_model/download_manifest.json":
            paths.add(path)
    paths.update(pathlib.Path(name) for name in (
        "kv_smoke/requirements.txt", "kv_research/models/learned64.bin"))
    evidence = [p for p in out.glob("*.json") if not p.name.startswith("final")]
    evidence.extend(p for p in (root / "development").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    evidence.append(out / "unit_tests.txt")
    sha = lambda items: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(items)}
    selection = dict(
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        fresh_final_manifest_existed=False,
        generation_candidates=generation,
        best_identity=generation[0], best_rotated=generation[1],
        reason="Development-only dimensions and exact complete-byte selection, fixed learned64 dictionary; matched radius predictor and correction backend. No fresh final data used.",
        source_and_model_sha256=sha(paths), development_evidence_sha256=sha(evidence),
        development_manifest_sha256=hashlib.sha256(pathlib.Path(".kv_research_data/dev_manifest.json").read_bytes()).hexdigest(),
    )
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    shutil.copy2(".kv_research_data/dev_manifest.json", out / "development_capture_manifest.json")
    print(json.dumps(dict(candidates=len(config["candidates"]), generation_candidates=generation,
                          source_files=len(paths), evidence_files=len(evidence)), indent=2))


if __name__ == "__main__":
    main()
