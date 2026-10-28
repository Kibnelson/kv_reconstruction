"""Frozen-input verification and resumable fitted state for the causal study."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .causal_packet import json_bytes, sha, write_json
from .predict import Config, Study

MODEL = "unsloth/Llama-3.2-3B"
REVISION = "d4446454d87d51aa42e1fb174f25acc5f8762331"
ROOT = Path(__file__).resolve().parents[1]
CODE = ["kv_compare/causal_packet.py", "kv_compare/causal_entropy.py", "kv_compare/causal_study.py",
        "kv_compare/gpu_coder.py", "kv_compare/core.py", "kv_compare/predict.py",
        "scripts/bench_llama_causal.py", "kv_compare/quality.py",
        "qwen_kv_bench/model_runtime.py"]


def array_word_digest(K, V):
    h = hashlib.sha256()
    for a in (K, V):
        for layer in a:
            h.update(np.asarray(layer, dtype="<u2").tobytes())
    return h.hexdigest()


def official_view(source, destination, log):
    """Reuse original WikiText BF16 files; recover token IDs from the verified local dataset cache."""
    from transformers import AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import load_dataset
    m = json.loads((source / "captures/manifest.json").read_text())
    mm = json.loads((source / "model_manifest.json").read_text())
    if m["model"] != MODEL or m["revision"] != REVISION:
        raise ValueError("wrong original model")
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(mm["snapshot"], local_files_only=True)
    split_ids = {}
    for split, expected in m["datasets"].items():
        log("recover cached WikiText token IDs: " + split)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"])
        ids = np.asarray(tokenizer(text, return_tensors="pt")["input_ids"][0], dtype="<i8")
        if (hashlib.sha256(text.encode()).hexdigest() != expected["text_sha256"] or
                hashlib.sha256(ids.tobytes()).hexdigest() != expected["token_ids_sha256"]):
            raise ValueError("original WikiText text/token identity changed")
        split_ids[split] = ids
    captures = []
    for e in m["entries"]:
        for kind in ("K_bits", "V_bits"):
            p = source / "captures" / e["files"][kind]
            if sha(p) != e["hashes"][kind]:
                raise ValueError("original capture checksum changed")
            link = destination / (e["name"] + "_" + kind + ".npy")
            if link.exists() or link.is_symlink():
                if link.resolve() != p.resolve():
                    raise ValueError("input view link changed")
            else:
                link.symlink_to(p.resolve())
        index = int(e["name"].rsplit("_", 1)[1])
        ids = split_ids[e["split"]][index * e["tokens"]:(index + 1) * e["tokens"]]
        np.save(destination / (e["name"] + "_ids.npy"), ids)
        captures.append(dict(name=e["name"], split=e["split"], tokens=e["tokens"]))
    cfg = m["config"]
    meta = dict(model=MODEL, revision=REVISION, layers=m["layers"],
        query_heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], inv_freq=m["rope_inv_freq"], rope_layout="half",
        attention_scaling=1., captures=captures, source_protocol="original_frozen_WikiText2")
    write_json(destination / "meta.json", meta)
    return destination


def source_inputs(source, out, train_docs, log):
    source = Path(source).resolve()
    if (source / "captures/meta.json").is_file():
        root = source / "captures"
    elif (source / "captures/manifest.json").is_file():
        root = official_view(source, out / "input_view", log)
    else:
        raise ValueError("source run has neither team captures nor original official-run captures")
    meta = json.loads((root / "meta.json").read_text())
    if meta.get("model") != MODEL or meta.get("revision") != REVISION or meta.get("layers") != 28:
        raise ValueError("expected the pinned original BF16 Llama-3.2-3B, all 28 layers")
    original = meta.get("source_protocol") == "original_frozen_WikiText2"
    train = ([e["name"] for e in meta["captures"] if e.get("split") == "train"] if original else
             [f"train{i}" for i in range(6)])[:train_docs]
    tables = next(e["name"] for e in meta["captures"] if e.get("split") == "validation") if original else "tables"
    tests = [e["name"] for e in meta["captures"] if e.get("split") == "test"] if original else ["test_text", "test_code"]
    if not train:
        raise ValueError("no TRAIN documents")
    docs = {}
    for name in train + [tables] + tests:
        log("verify frozen words: " + name)
        K = np.load(root / (name + "_K_bits.npy"), mmap_mode="r", allow_pickle=False)
        V = np.load(root / (name + "_V_bits.npy"), mmap_mode="r", allow_pickle=False)
        ids = np.load(root / (name + "_ids.npy"), allow_pickle=False)
        if K.dtype != np.uint16 or V.shape != K.shape or K.shape[0] != 28 or len(ids) != K.shape[1]:
            raise ValueError("invalid BF16 capture geometry")
        digest = array_word_digest(K, V)
        expected = next(e for e in meta["captures"] if e["name"] == name).get("kv_sha256")
        if expected and expected != digest:
            raise ValueError("team capture word checksum changed: " + name)
        docs[name] = dict(words_sha256=digest, tokens=len(ids),
            token_ids_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest())
    return root, train, tables, tests, dict(meta_sha256=sha(root / "meta.json"),
        model=MODEL, revision=REVISION, protocol=meta.get("source_protocol", "team_PEP_CPython"), documents=docs)


def bounded(doc, tokens):
    n = min(doc.tokens, tokens)
    doc.K, doc.V, doc.ids = doc.K[:, :n], doc.V[:, :n], doc.ids[:n]
    if doc.Q is not None:
        doc.Q = doc.Q[:, :n]
    doc.tokens = n
    return doc


def policy(name, width):
    if name == "block32":
        return Config(name, ("P",), math_ceil(width, 32), "natural", False)
    if name == "lite":
        return Config(name, ("P",), 16, "pivot", True)
    if name == "strong_noatt":
        return Config(name, ("L0", "P2", "P"), 16, "pivot", True)
    raise ValueError("unknown predictor policy")


def math_ceil(a, b):
    return (a + b - 1) // b


def fit_policy(study, name, directory, recipe, log):
    cfg = policy(name, study.W)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json_bytes(dict(recipe, policy=name))).hexdigest()
    recipe_file = directory / "fit_recipe.json"
    if recipe_file.exists() and json.loads(recipe_file.read_text())["key"] != key:
        raise ValueError("fit source/configuration changed; use a fresh output")
    write_json(recipe_file, dict(key=key, recipe=recipe, policy=name))
    entries = []
    for l in range(study.L):
        path = directory / f"layer_{l:03d}.npz"
        proof = path.with_suffix(".json")
        if path.exists() and proof.exists():
            p = json.loads(proof.read_text())
            if p.get("fit_key") != key or p.get("sha256") != sha(path):
                raise ValueError("saved fitted layer changed")
            entries.append(dict(path=path.name, sha256=p["sha256"]))
            log(f"{name}: reuse fitted layer {l + 1}/{study.L}")
            continue
        begin = time.perf_counter()
        # Block32 follows adjacent RoPE pairs, matching the old channel-group policy.
        if name == "block32":
            kinds = [] if l == 0 else ["P"]
            mu, S = study.covariance(l, kinds)
            partner = study.rope.pair_partner()
            order = []
            for c in range(study.C):
                if c < partner[c]:
                    order += [c, int(partner[c])]
            order += list(range(study.C, study.W))
            box = [S]; del S
            F = study.solve(mu, box, kinds, cfg.steps, cfg.order,
                            perm=torch.as_tensor(order, device=study.device))
        else:
            F = study.fit(l, cfg)
        F.spread = study.static_spread(F, l)
        tmp = path.with_suffix(".part.npz")
        np.savez(tmp, A=F.A.cpu().numpy(), b=F.b.cpu().numpy(),
            perm=F.perm.cpu().numpy(), starts=F.starts.cpu().numpy(),
            spread=F.spread.cpu().numpy(), kinds=np.asarray(F.kinds, dtype="U3"))
        tmp.replace(path); digest = sha(path)
        write_json(proof, dict(fit_key=key, sha256=digest, seconds=time.perf_counter() - begin))
        entries.append(dict(path=path.name, sha256=digest))
        log(f"{name}: fitted layer {l + 1}/{study.L} in {time.perf_counter() - begin:.1f}s")
        study.forget(l - 1)
    m = study.meta
    state = dict(format="causal_fitted_v1", policy=name, adaptive=cfg.adaptive, fit_key=key,
        kv_heads=study.Hk, head_dim=study.hd, inv_freq=m["inv_freq"],
        rope_layout=m.get("rope_layout", "half"), attention_scaling=m.get("attention_scaling", 1.),
        model=m["model"], revision=m["revision"], files=entries)
    existing = directory / "state.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        plain = {k: v for k, v in old.items() if k not in ("state_sha256", "runtime_spread")}
        if plain == state and old.get("runtime_spread"):
            if sha(directory / old["runtime_spread"]["path"]) != old["runtime_spread"]["sha256"]:
                raise ValueError("cached runtime spreads changed")
            return old
    state["state_sha256"] = hashlib.sha256(json_bytes(state)).hexdigest()
    write_json(directory / "state.json", state)
    return state


def calibrate_runtime(study, directory, predictor_format, log):
    from .causal_packet import Engine
    state = json.loads((directory / "state.json").read_text())
    if state.get("runtime_spread"):
        log("reuse calibrated runtime spreads: " + state["policy"])
        return
    log("calibrate widths with actual " + predictor_format + " products: " + state["policy"])
    engine = Engine(directory, study.device, predictor_format=predictor_format)
    words = torch.stack([study.tables.words(l, study.device) for l in range(study.L)]).to(torch.int16)
    widths = engine.calibrate(words).cpu().numpy()
    target = directory / "runtime_spread.npy"
    tmp = directory / "runtime_spread.part.npy"
    np.save(tmp, widths, allow_pickle=False); tmp.replace(target)
    state.pop("state_sha256")
    state["runtime_spread"] = dict(path=target.name, sha256=sha(target), predictor_format=predictor_format)
    state["state_sha256"] = hashlib.sha256(json_bytes(state)).hexdigest()
    write_json(directory / "state.json", state)
