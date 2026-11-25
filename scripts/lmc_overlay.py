#!/usr/bin/env python3
"""Our copies of vLLM and LMCache (third_party/) checked, listed, and laid over their release wheels for a job.

  python3 scripts/lmc_overlay.py check                    the hooks are the only changes (below)
  python3 scripts/lmc_overlay.py hooks [--markdown]       every hook line: file, line, event
  python3 scripts/lmc_overlay.py build --site SITE --out DIR
                                                          a per-job tree: the wheels' packages with our Python
  PYTHONPATH=DIR KVTRACE_DIR=TMP python -P scripts/lmc_overlay.py verify --out DIR
                                                          in the environment the engines use: what they import

check: every file of each copy is listed in third_party/<copy>.sha256 (no file added), and removing every line ending
in "# KVTRACE" gives back the imported file exactly (its hash); every line that mentions kvtrace is such a line.

build: SITE is the site-packages of the environment with the release wheels (vllm==0.32.0, lmcache==0.5.7.post1).
First, every Python file of our copies must be byte-identical, as imported, to the wheel's: the wheel is that
release. Then DIR/<package> is the wheel's package hard-linked file by file (its compiled .so, generated files and
everything else), with our hooked files written over (a new file each, so the environment stays untouched), and
DIR/kvtrace our trace module. PYTHONPATH=DIR runs the wheels with our hooks; DIR/OVERLAY.json says what was done.

verify imports kvtrace, vllm, lmcache and every hooked module as an engine would, and stops unless each comes from
DIR (with its hooks) and kvtrace writes a record. Run it with python -P (no script folder or current folder first
on the path: the repository's own kvtrace/ would shadow DIR's) and from a folder that holds no package. It also
says whether LMCache's compiled CUDA ops load (without them LMCache uses its torch baseline).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
COPIES = {"vllm": "vllm", "lmcache": "lmcache"}           # copy (third_party/<copy>) -> its package
MARK = "# KVTRACE"
HOOK_RE = re.compile(r'kvtrace\.hook\("([^"]+)"')


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def strip(b: bytes) -> bytes:
    """The file without its hook lines (each a whole line ending in # KVTRACE)."""
    return b"".join(ln for ln in b.splitlines(keepends=True) if not ln.rstrip(b"\r\n").endswith(MARK.encode()))


def manifest(root, copy):
    out = {}
    for ln in (root / "third_party" / f"{copy}.sha256").read_text().splitlines():
        h, path = ln.split("  ", 1)
        out[path] = h
    return out


def check(root=ROOT):
    """(problems, hooked files {copy/path: [hook lines]})."""
    problems, hooked = [], {}
    for copy in COPIES:
        base = root / "third_party" / copy
        man = manifest(root, copy)
        on_disk = {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and not p.is_symlink()
                   and "__pycache__" not in p.parts}
        for extra in sorted(on_disk - set(man)):
            problems.append(f"{copy}/{extra}: not in the import (no file is added to a copy)")
        for path, h in man.items():
            p = base / path
            if not p.is_file():
                problems.append(f"{copy}/{path}: missing")
                continue
            b = p.read_bytes()
            if sha(b) == h:
                continue
            if sha(strip(b)) != h:
                problems.append(f"{copy}/{path}: changed beyond its # KVTRACE lines")
                continue
            lines = [(i + 1, ln) for i, ln in enumerate(b.decode(errors="replace").splitlines())
                     if ln.rstrip().endswith(MARK)]
            hooked[f"{copy}/{path}"] = lines
        for path in man:
            p = base / path
            if p.suffix != ".py" or not p.is_file():
                continue
            for i, ln in enumerate(p.read_text(errors="replace").splitlines()):
                if "kvtrace" in ln and not ln.rstrip().endswith(MARK):
                    problems.append(f"{copy}/{path}:{i + 1}: mentions kvtrace without {MARK}")
    return problems, hooked


def hooks(root=ROOT):
    """[(file, line, event or 'import', text)] of every hook line."""
    _p, hooked = check(root)
    out = []
    for f, lines in sorted(hooked.items()):
        for n, ln in lines:
            m = HOOK_RE.search(ln)
            ev = m.group(1) if m else ("import" if ln.strip().startswith("import kvtrace") else "other")
            out.append((f, n, ev, ln.strip()))
    return out


def build(site, out, root=ROOT, log=print):
    site, out = pathlib.Path(site), pathlib.Path(out)
    problems, hooked = check(root)
    if problems:
        return {"error": "the copies are not the import plus hooks", "problems": problems[:50]}
    differ, missing = [], []
    for copy, pkg in COPIES.items():
        for path, h in manifest(root, copy).items():
            if not path.startswith(pkg + "/") or not path.endswith(".py"):
                continue
            w = site / path
            if not w.is_file():
                missing.append(path)
            elif sha(w.read_bytes()) != h:
                differ.append(path)
    if differ or missing:
        return {"error": "the installed wheels are not the releases our copies were imported from",
                "differ": differ[:50], "n_differ": len(differ), "missing": missing[:50], "n_missing": len(missing)}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    stats = {"linked": 0, "copied": 0, "replaced": []}
    for copy, pkg in COPIES.items():
        src = site / pkg
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]   # bytecode is compiled afresh, from our files
            rel = pathlib.Path(dirpath).relative_to(site)
            (out / rel).mkdir(parents=True, exist_ok=True)
            for fn in filenames:
                s, d = pathlib.Path(dirpath) / fn, out / rel / fn
                try:
                    os.link(s, d)
                    stats["linked"] += 1
                except OSError:
                    shutil.copy2(s, d)
                    stats["copied"] += 1
    for f in sorted(hooked):
        copy, path = f.split("/", 1)
        d = out / path
        if d.exists():
            d.unlink()                                   # a new file: the hard link to the environment stays as was
        d.write_bytes((root / "third_party" / copy / path).read_bytes())
        stats["replaced"].append(path)
    kt = out / "kvtrace"
    kt.mkdir()
    for p in sorted((root / "kvtrace").glob("*.py")):
        shutil.copy2(p, kt / p.name)
    info = {"site": str(site), "out": str(out), "hooked_files": stats["replaced"], "hooks": len(hooks(root)),
            "linked": stats["linked"], "copied": stats["copied"],
            "checked_py": sum(1 for c, pk in COPIES.items() for pth in manifest(root, c)
                              if pth.startswith(pk + "/") and pth.endswith(".py"))}
    (out / "OVERLAY.json").write_text(json.dumps(info, indent=1))
    log(f"overlay {out}: {info['checked_py']} Python files equal to the wheels'; {len(stats['replaced'])} hooked "
        f"files over them ({info['hooks']} hook lines); {stats['linked']} files linked, {stats['copied']} copied")
    return info


def verify(out, log=print):
    """What the engines will import, checked in this process (PYTHONPATH=out, KVTRACE_DIR set)."""
    import importlib
    out = pathlib.Path(out).resolve()
    info = json.loads((out / "OVERLAY.json").read_text())
    bad = []

    def where(mod):
        f = pathlib.Path(mod.__file__).resolve()
        return f, out in f.parents

    import kvtrace
    f, ok = where(kvtrace)
    if not ok:
        bad.append(f"kvtrace imports from {f}, not {out}")
    if not kvtrace.ON:
        bad.append("kvtrace is off (KVTRACE_DIR is not set)")
    for rel in info["hooked_files"]:
        mod = importlib.import_module(rel[:-3].replace("/", "."))
        f, ok = where(mod)
        n = sum(1 for ln in f.read_text().splitlines() if ln.rstrip().endswith(MARK))
        if not ok or n == 0:
            bad.append(f"{mod.__name__} imports from {f} ({n} hook lines), not {out}")
        else:
            log(f"   {mod.__name__}: {n} hook lines, from the overlay")
    vers = {}
    for name in COPIES.values():
        mod = importlib.import_module(name)
        f, ok = where(mod)
        vers[name] = getattr(mod, "__version__", "?")
        if not ok:
            bad.append(f"{name} imports from {f}, not {out}")
    if not bad:
        kvtrace.hook("kvtrace.check", value=1)
        kvtrace.flush()
        d = pathlib.Path(os.environ["KVTRACE_DIR"])
        recs = [json.loads(ln) for p in d.glob("*.jsonl") for ln in p.read_text().splitlines() if ln.strip()]
        if not any(r.get("ev") == "kvtrace.check" and r.get("value") == 1 for r in recs):
            bad.append(f"kvtrace wrote no record to {d}")
    native = None
    try:
        importlib.import_module("lmcache.cuda_ops")
        native = "loads"
    except Exception as exc:                      # LMCache falls back to its torch baseline; worth knowing
        native = f"does not load ({type(exc).__name__}: {str(exc)[:300]})"
    except BaseException:
        native = "does not load"
    res = {"ok": not bad, "problems": bad, "versions": vers, "lmcache_cuda_ops": native}
    try:
        (out / "VERIFY.json").write_text(json.dumps(res, indent=1))
    except OSError:
        pass
    for b in bad:
        log("PROBLEM: " + b)
    log(f"   {'OK' if not bad else 'NOT OK'}: " + ", ".join(f"{k} {v}" for k, v in vers.items()) +
        f" from the overlay; kvtrace writes; LMCache's compiled CUDA ops {native}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    h = sub.add_parser("hooks")
    h.add_argument("--markdown", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--site", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    for p in (ap,):
        p.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)
    if a.cmd == "check":
        problems, hooked = check(root)
        for p in problems:
            print("PROBLEM:", p)
        n = sum(len(v) for v in hooked.values())
        print(f"{'NOT OK' if problems else 'OK'}: {len(hooked)} hooked files, {n} hook lines; every other file as "
              "imported")
        return 1 if problems else 0
    if a.cmd == "hooks":
        rows = hooks(root)
        if a.markdown:
            print("| File | Line | Event |\n| --- | ---: | --- |")
            for f, n, ev, _t in rows:
                print(f"| `{f}` | {n} | `{ev}` |")
        else:
            for f, n, ev, t in rows:
                print(f"{f}:{n}: {ev}: {t}")
        return 0
    if a.cmd == "verify":
        return 0 if verify(a.out)["ok"] else 1
    info = build(a.site, a.out, root)
    if "error" in info:
        print("STOP:", json.dumps(info, indent=1))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
