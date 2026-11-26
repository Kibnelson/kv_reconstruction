#!/usr/bin/env python3
"""The node side of a PD-baselines campaign (scripts/pd_card.sh runs each stage in its own Slurm allocation on
gpu-grad-01). Standard library only: it runs with the node's python3 before any environment exists and after the
run's environments are gone.

  create          the campaign: /tmp/$USER/kvpd-evidence/<tag>/repo at the pin (only its results folder checked out)
                  and a marker (host, owner, boot, pin, the matrix's runs)
  run             with the 4 GPUs: the driver, the GPUs and the disk checked first; a fresh runtime
                  /tmp/$USER/kvpd/<job> (environments hard-linked from the shared uv cache, sources, captures,
                  caches); scripts/a5000_setup.sh, scripts/pd_setup.sh, scripts/node_specs.py (the node as it is
                  during this campaign), then scripts/pd_matrix.sh; results, logs and setup records into the
                  campaign's folder. Once per campaign.
  cleanup         in another allocation: deletes exactly /tmp/$USER/kvpd/<run job> once nothing uses it; a receipt
  verify-cleanup  in a third: Slurm's records of the run and the cleanup, the receipt, the folder gone
  review          rebuilds the matrix report and an inventory (every file's size and SHA-256) from the kept results
  publish         the inventory unchanged, no credentials, no oversized files: one commit on the pin with only
                  benchmark_results/pd_card/<tag>/, pushed to evidence/pd-baselines/<pin12>/<matrix>/job<run job>

Every stage checks the node, the allocation, the owner and modes of every folder it uses, that the code is the pin's
and unmodified, and the campaign's marker (a reboot since create means /tmp may have been wiped: a new tag then).
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

SCHEMA = 1
KIND = "kvpd_campaign"
MATRICES = ("smoke", "ring1p3d", "ring2p2d", "direct", "tier", "t2eager", "links", "workloads", "wlong", "wlours", "capacity", "quick", "trace", "capture", "diffgraph", "tree", "region", "diffedge", "xfer", "lmctrace")
# Disk (GB) on the node's root file system, which /tmp is: the shared cache's parts when missing, then a job's peak.
# Measured sizes go into disk_after.json; these only decide whether to start.
NEED_GB = {"uv": 17.0, "python": 0.2, "hf": 6.6}   # measured on gpu-grad-01 (job 30170): 16.7, 0.1, 6.5
CUDA13_GB = 4.0            # the CUDA 13 toolkit (IBP's build), pruned
CUDA13_INSTALL_GB = 11.0   # its download and the installer's unpacking, while it installs
JOB_PEAK_GB = 14.0         # captures without queries (6.6 GB), one T2 set's windows (3.5 GB), the environments' own
                           # files and caches (5 GB at cleanup, job 30176)
MARGIN_GB = 10.0           # left free: /tmp is the node's system disk, shared with everyone
SETUP_DROPPED_GB = 6.6     # setup's captures, deleted before the matrix runs (DROP_CAPTURES=1): the job's peak is
                           # setup's, or the matrix's own files plus what the matrix adds, not both
CAPTURE_GB_PER_CONV = 2.0  # MATRIX=capture: P's store a conversation (job 30555: 10.3 GB for 6), as pd_matrix.sh sets
CAPTURE_D_GB = 2.0         # and each D engine's (KVPD_CAPTURE_D_GB): its copies of held blocks that differ
CAPTURE_D_ENGINES = 3      # MATRIX=capture runs 1p3d
LMC_GB_PER_CONV = 1.0      # MATRIX=lmctrace: the disk tier's files for one interleaved MTRAG conversation (its longest
                           # prompt's whole chunks, 112 KB a token, each stored once: job 30728's first 4 had 9,765,
                           # 9,097, 7,503 and 3,102 tokens, 114 chunks, 3.35 GB in all), one run at a time
                           # (scripts/lmc_run.sh deletes them after its report)
LMC_DOCS_SCALE = {"all": 1.1}   # MTRAG_DOCS=all: 5 to 9 passages a turn, not 5 (rag) or fewer: about 10% more
LMC_PER_ENGINE_COPIES = {       # LMC_SERVERS=per_engine: P's server and each D's own (D engines round-robin, so a
    "interleaved": 4.0,         # conversation's chunks land on all three D servers): each carry-forward prompt holds
    "history_first": 2.5,       # the ones before it, so up to 4 copies (about 3.3 estimated for the ten); a chat
}                               # prompt's passages are its own turn's, so each D stores mostly its turns' (about 2.1)
LMC_PER_ENGINE_DEFAULT = 4.0    # any other layout: at most 4
LMC_RECORDS_GB = 0.25      # and the traces, logs and results a run keeps (job 30694: about 15 MB a run)
LMC_SETUP_GB = 3.5         # MATRIX=lmctrace's setup: the environments are links into uv's cache (NIXL's wheels too);
                           # the CUDA ops' build
                           # (its objects), the overlay's few copies, the job's caches (not the codec setup's 14 GB)
MODEL_DIR = "hub/models--unsloth--Llama-3.2-3B"
CUDA12_CANDIDATES = ("/usr/local/pkgs/cuda/cuda-12.6", "/usr/local/pkgs/cuda/12.6", "/usr/local/cuda-12.6",
                     "/usr/local/pkgs/cuda/latest", "/usr/local/cuda-12", "/usr/local/cuda-12.3", "/usr/local/cuda")
# credentials that must never be published (the pattern's name is reported, never the match)
SECRETS = {
    "private key": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "GitHub token": rb"\bgh[pousr]_[A-Za-z0-9]{36,}",
    "GitHub fine-grained token": rb"github_pat_[A-Za-z0-9_]{40,}",
    "Hugging Face token": rb"\bhf_[A-Za-z0-9]{30,}",
    "AWS access key": rb"\bAKIA[0-9A-Z]{16}\b",
    "Slack token": rb"xox[abprs]-[A-Za-z0-9-]{10,}",
    "API key (sk-)": rb"\bsk-[A-Za-z0-9_-]{20,}",
    "secret assignment": rb"(?i)\b(secret|password|api_key|access_token|auth_token)\b\s*[=:]\s*['\"]?[A-Za-z0-9_/+-]{16,}",
}
MAX_FILE = 95 * 2 ** 20    # GitHub refuses files over 100 MB
MAX_TOTAL = 500 * 2 ** 20
NOT_INVENTORIED = ("review/inventory.json", "review/review.json", "publish.json")


class Stop(Exception):
    pass


def require(ok, msg):
    if not ok:
        raise Stop(msg)


def log(msg):
    print(msg, flush=True)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_env():
    for name in ("GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                 "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG", "GIT_CONFIG_COUNT"):
        require(not os.environ.get(name), f"{name} is set; unset it (git must use the campaign's own repositories)")
    return dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")


def call(*args, capture=True, check=True, env=None, cwd=None, timeout=None):
    r = subprocess.run([str(a) for a in args], capture_output=capture, text=True, env=env, cwd=cwd,
                       timeout=timeout)
    if check and r.returncode != 0:
        err = (r.stderr or "").strip()[-2000:] if capture else ""
        raise Stop(f"{' '.join(str(a) for a in args[:4])} ... exited {r.returncode}{': ' + err if err else ''}")
    return (r.stdout or "").strip() if capture else r.returncode


def git(repo, *args, **kw):
    return call("git", "-C", repo, *args, env=git_env(), **kw)


def write_json(path, obj, once=False):
    """Mode 600; once=True refuses to replace an existing file (create-once records)."""
    path = Path(path)
    data = (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()
    if once:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_text(path, text, mode=0o600):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"{path} missing")
    return json.loads(path.read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def private_dir(path, create=False):
    path = Path(path)
    if create and not path.exists() and not path.is_symlink():
        path.mkdir(mode=0o700)
    st = path.lstat() if path.exists() or path.is_symlink() else None
    require(st is not None and path.is_dir() and not path.is_symlink(), f"{path} is not a directory")
    require(st.st_uid == os.getuid(), f"{path} is not owned by {os.environ.get('USER')}")
    require(os.path.realpath(path) == str(path), f"{path} goes through a link")
    require(st.st_mode & 0o022 == 0, f"{path} is writable by others")
    return path


class Paths:
    def __init__(self, pin, tag):
        self.user = os.environ.get("USER") or pwd.getpwuid(os.getuid()).pw_name
        self.root = Path(os.environ.get("KVPD_ROOT") or f"/tmp/{self.user}")
        self.tools = self.root / "kvpd-tools" / pin
        self.campaign = self.root / "kvpd-evidence" / tag
        self.repo = self.campaign / "repo"
        self.rel = Path("benchmark_results") / "pd_card" / tag
        self.ev = self.repo / self.rel
        self.runtimes = self.root / "kvpd"
        self.cache = self.root / "kvpd-cache"

    def runtime(self, job):
        return self.runtimes / str(job)


# ------------------------------------------------------------------------------------------------- the checks

def check_node():
    host = socket.gethostname().split(".")[0]
    node = os.environ.get("KVPD_NODE") or "gpu-grad-01"
    require(host == node, f"on {host}, not {node}")
    require(re.fullmatch(r"[0-9]+", os.environ.get("SLURM_JOB_ID", "")), "not inside a Slurm allocation")
    user = os.environ.get("USER", "")
    require(re.fullmatch(r"[A-Za-z0-9_]+", user) and pwd.getpwuid(os.getuid()).pw_name == user,
            "USER is not this process's user")
    return os.environ["SLURM_JOB_ID"]


def check_tools(p, pin):
    private_dir(p.root)
    private_dir(p.tools.parent)
    private_dir(p.tools)
    require(git(p.tools, "rev-parse", "--show-toplevel") == str(p.tools), "the tools folder is not a checkout")
    require(git(p.tools, "rev-parse", "HEAD") == pin, "the node-local code is not at the pin")
    require(git(p.tools, "status", "--porcelain", "--untracked-files=normal") == "",
            "the node-local code has changes")
    me = Path(__file__).resolve()
    require(me == p.tools / "scripts" / "pd_campaign.py", f"{me} is not the node-local code's")
    require(git(p.tools, "hash-object", me) == git(p.tools, "rev-parse", f"{pin}:scripts/pd_campaign.py"),
            "this script is not the pin's")


def check_campaign(p, pin, tag, matrix):
    for d in (p.root, p.root / "kvpd-evidence", p.campaign, p.repo):
        private_dir(d)
    m = read_json(p.ev / "campaign.json")
    require(m.get("schema") == SCHEMA and m.get("kind") == KIND, "not a campaign marker")
    require((m.get("pin"), m.get("tag"), m.get("matrix")) == (pin, tag, matrix), "the marker is another campaign's")
    require(m.get("owner_uid") == os.getuid() and m.get("repo") == str(p.repo), "the marker's owner or path differs")
    require(m.get("host") == socket.gethostname().split(".")[0], "the campaign was made on another node")
    require(m.get("boot_id") == boot_id(), "the node rebooted since create (its /tmp may be gone): start a new tag")
    head = git(p.repo, "rev-parse", "HEAD")
    if head != pin:   # published: one commit on the pin, only the campaign's folder
        require(git(p.repo, "rev-parse", "HEAD^") == pin, "the campaign repository moved off the pin")
        changed = git(p.repo, "diff", "--name-only", pin, "HEAD").splitlines()
        require(all(c.startswith(str(p.rel) + "/") for c in changed), "the evidence commit touches other files")
    status = git(p.repo, "status", "--porcelain", "--untracked-files=all", "--", ".",
                 f":(exclude){p.rel}/**").strip()
    require(status == "", f"the campaign repository has changes outside {p.rel}: {status[:300]}")
    return m


# ------------------------------------------------------------------------------------------------- create

def stage_create(p, a, job):
    check_tools(p, a.pin)
    require(not p.campaign.exists() and not p.campaign.is_symlink(),
            f"campaign {a.tag} exists: continue it with its saved stages, or use a new tag")
    private_dir(p.root)
    private_dir(p.root / "kvpd-evidence", create=True)
    p.campaign.mkdir(mode=0o700)
    home = Path(a.home_repo)
    origin = git(home, "remote", "get-url", "origin")
    git(p.campaign, "clone", "-q", "--shared", "--no-checkout", home, p.repo)
    p.repo.chmod(0o700)
    git(p.repo, "remote", "set-url", "origin", origin)
    git(p.repo, "fetch", "-q", "--no-tags", "origin", f"refs/heads/{a.branch}")
    require(git(p.repo, "rev-parse", "FETCH_HEAD") == a.pin, f"GitHub's {a.branch} is not the pin")
    git(p.repo, "sparse-checkout", "init", "--no-cone")
    # only the campaign's folder (and one small file: older git refuses a checkout that leaves nothing)
    git(p.repo, "sparse-checkout", "set", "--no-cone", "/.gitignore", f"/{p.rel}/")
    git(p.repo, "checkout", "-q", "-B", f"kvpd-{a.tag}", a.pin)
    p.ev.mkdir(mode=0o700, parents=True)
    runs = call("bash", p.tools / "scripts" / "pd_matrix.sh",
                env=dict(os.environ, MATRIX=a.matrix, PRINT_RUNS="1")).splitlines()
    write_json(p.ev / "campaign.json", {
        "schema": SCHEMA, "kind": KIND, "tag": a.tag, "matrix": a.matrix, "pin": a.pin, "branch": a.branch,
        "origin": origin, "home_repo": str(home), "tools": str(p.tools), "repo": str(p.repo),
        "results": str(p.rel), "host": socket.gethostname().split(".")[0], "user": p.user,
        "owner_uid": os.getuid(), "boot_id": boot_id(), "create_job": job, "created": now(), "runs": runs,
        "card": git(p.tools, "rev-parse", f"{a.pin}:scripts/pd_card.sh"),
    }, once=True)
    write_json(p.ev / "disk_at_create.json", disk_survey(p))
    log(f"KVPD_CAMPAIGN={p.repo}")
    log(f"matrix {a.matrix}: {len(runs)} runs")
    for r in runs:
        log(f"   {r}")


def disk_survey(p):
    u = shutil.disk_usage(p.root)
    ours = {}
    for d in (p.cache, p.runtimes, p.root / "kvpd-evidence", p.root / "kvpd-tools"):
        if d.is_dir():
            ours[d.name] = du_bytes([d])[0]
    for d in sorted(p.cache.iterdir()) if p.cache.is_dir() else []:
        ours[f"kvpd-cache/{d.name}"] = du_bytes([d])[0]
    return {"filesystem": str(p.root), "total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1),
            "ours_gb": {k: round(v / 1e9, 2) for k, v in ours.items()}, "when": now()}


def du_bytes(paths):
    """Disk use of each path by `du -sk` in one call: a file hard-linked into several counts in the first only."""
    out = call("du", "-sk", *paths, check=False)
    got = {}
    for line in out.splitlines():
        k, _, name = line.partition("\t")
        if k.isdigit():
            got[name] = int(k) * 1024
    return [got.get(str(x), 0) for x in paths]


# ------------------------------------------------------------------------------------------------- run

def gpus():
    rows = call("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader",
                check=False)
    out = []
    for r in rows.splitlines():
        f = [x.strip() for x in r.split(",")]
        if len(f) == 5:
            out.append({"index": f[0], "name": f[1], "uuid": f[2], "driver": f[3], "memory": f[4]})
    return out


def find_cuda12():
    """A CUDA 12 toolkit on the node for building ExANS: KVPD_CUDA12_HOME, the cluster's module, known places."""
    cands = []
    if os.environ.get("KVPD_CUDA12_HOME"):
        cands.append(os.environ["KVPD_CUDA12_HOME"])
    r = subprocess.run(["bash", "-c", "module load cuda/cuda-12.6 >/dev/null 2>&1 && command -v nvcc"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        cands.append(str(Path(r.stdout.strip()).resolve().parents[1]))
    cands += list(CUDA12_CANDIDATES)
    best = None
    for c in cands:
        nvcc = Path(c) / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        v = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True).stdout
        m = re.search(r"release 12\.(\d+)", v)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), str(Path(c).resolve()))
    return best[1] if best else None


def find_uv(cache):
    """uv, copied once into the cache (a single binary): the runs then depend on no folder of another project
    (on gpu-grad-01, ~/.local/bin's tools can point into another project's /tmp folders)."""
    mine = cache / "bin" / "uv"
    if mine.is_file() and os.access(mine, os.X_OK) and not os.environ.get("KVPD_UV"):
        return str(mine)
    for c in (os.environ.get("KVPD_UV"), shutil.which("uv"), str(Path.home() / ".local/bin/uv")):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            src = Path(c).resolve()
            if subprocess.run([str(src), "--version"], capture_output=True).returncode != 0:
                continue
            mine.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp = mine.with_name("uv.tmp")
            shutil.copyfile(src, tmp)
            tmp.chmod(0o700)
            os.replace(tmp, mine)
            return str(mine)
    return None


# a matrix's own files beyond a job's peak: capture keeps the raw pages it stores (kvpd-captures/<tag>, kept after
# cleanup for offline analysis; scripts/pd_matrix.sh CAPTURE_GB)
def extra_gb(matrix, runs=()):
    """What a matrix adds to the disk after setup, and keeps (GB): MATRIX=capture's stores as pd_matrix.sh sets them:
    P's (16 GB for the default 6 conversations, else 2 GB each: KVPD_CAPTURE_CONVS) and each of the 3 D engines'
    (KVPD_CAPTURE_D_GB, 2). MATRIX=lmctrace: its runs' disk tier (the largest run's: conversations, copies with a
    server per engine by layout, passages; scripts/lmc_run.sh deletes each run's) and records."""
    if matrix == "lmctrace":
        def copies(r):
            if not re.search(r"LMC_SERVERS=per_engine\b", r):
                return 1.0
            ly = re.search(r"MTRAG_LAYOUT=(\w+)", r)
            return LMC_PER_ENGINE_COPIES.get(ly.group(1) if ly else "interleaved", LMC_PER_ENGINE_DEFAULT)

        def docs(r):
            d = re.search(r"MTRAG_DOCS=(\w+)", r)
            return LMC_DOCS_SCALE.get(d.group(1) if d else "", 1.0)
        disk = [int(m.group(1)) * copies(r) * docs(r)
                for r in runs if re.search(r"LMC_MODE=(disk|tiered)\b", r)
                for m in [re.search(r"MTRAG_LIMIT=(\d+)", r)] if m]
        return (LMC_GB_PER_CONV * max(disk) if disk else 0.0) + LMC_RECORDS_GB * max(1, len(runs))
    if matrix != "capture":
        return 0.0
    n = os.environ.get("KVPD_CAPTURE_CONVS", "6")
    p_gb = 16.0 if not n.isdigit() or int(n) == 6 else CAPTURE_GB_PER_CONV * int(n)
    try:
        d_gb = float(os.environ.get("KVPD_CAPTURE_D_GB") or CAPTURE_D_GB)
    except ValueError:                             # pd_matrix.sh stops on it
        d_gb = CAPTURE_D_GB
    return p_gb + CAPTURE_D_ENGINES * d_gb


def disk_plan(p, want_ibp, extra_gb=0.0, need_cuda13=False, job_peak_gb=None, dropped_gb=None):
    """What the run will add to the disk, and whether it fits with MARGIN_GB to spare; IBP is dropped first.
    need_cuda13: the setup needs the runfile's CUDA 13 toolkit whatever IBP does. job_peak_gb / dropped_gb: a
    matrix's own setup peak and what it deletes before the runs, in place of the codec setup's (JOB_PEAK_GB,
    SETUP_DROPPED_GB)."""
    peak = JOB_PEAK_GB if job_peak_gb is None else float(job_peak_gb)
    dropped = SETUP_DROPPED_GB if dropped_gb is None else float(dropped_gb)
    scale = float(os.environ.get("KVPD_NEED_SCALE", "1"))
    margin = float(os.environ.get("KVPD_MARGIN_GB", MARGIN_GB))
    have = {"uv": (p.cache / "uv").is_dir() and any((p.cache / "uv").iterdir()),
            "python": (p.cache / "python").is_dir() and any((p.cache / "python").iterdir()),
            "hf": (p.cache / "hf" / MODEL_DIR).is_dir()}
    missing = [k for k, ok in have.items() if not ok]
    base = sum(NEED_GB[k] for k in missing) * scale
    cuda13 = (p.cache / "cuda13" / "bin" / "nvcc").is_file()
    free = shutil.disk_usage(p.root).free / 1e9
    setup = peak * scale                                              # setup's peak, its captures included
    runs = (peak - dropped) * scale + float(extra_gb)                 # the matrix: setup's captures gone, its own added

    def need(ibp):
        if (ibp or need_cuda13) and not cuda13:
            return base + CUDA13_GB * scale + max(CUDA13_INSTALL_GB * scale, setup, runs)
        return base + max(setup, runs)

    plan = {"free_gb": round(free, 1), "margin_gb": margin, "missing": missing, "cuda13_cached": cuda13,
            "ibp_wanted": want_ibp, "need_gb": round(need(want_ibp), 1), "setup_gb": round(setup, 1),
            "matrix_gb": round(runs, 1), "extra_gb": round(float(extra_gb), 1)}
    if free - need(want_ibp) >= margin:
        plan["ibp"] = want_ibp
    elif want_ibp and free - need(False) >= margin:
        plan["ibp"] = False
        plan["note"] = (f"IBP left out: the CUDA 13 toolkit for its build needs about {CUDA13_GB + CUDA13_INSTALL_GB:.0f}"
                        f" GB more than the {free:.0f} GB free allows with {margin:.0f} GB to spare")
        plan["need_gb"] = round(need(False), 1)
    else:
        plan["ibp"] = None
    return plan


def biggest(root, limit=12):
    try:
        out = call("du", "-sh", *sorted(root.iterdir()), check=False, timeout=180)
    except Exception:
        return []
    rows = [l.split("\t", 1) for l in out.splitlines() if "\t" in l]
    unit = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    size = lambda s: float(s[:-1]) * unit.get(s[-1], 1) if s[-1] in unit else float(s or 0)
    return sorted(rows, key=lambda r: -size(r[0]))[:limit]


class Streamer:
    """Run a shell script in its own process group, its output to the terminal and to a log; the whole group is
    stopped when it ends, or when this stage is told to stop (Slurm's time limit or scancel)."""

    def __init__(self, logfile):
        self.logfile = logfile
        self.proc = None
        self.stopped = None

    def stop_group(self, sig=signal.SIGTERM, wait=5.0):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                os.killpg(self.proc.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run(self, script, env):
        with open(self.logfile, "ab") as lf:
            self.proc = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         env=env, cwd=env.get("KVPD_TOOLS"), start_new_session=True)
            for line in iter(self.proc.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                lf.write(line)
                lf.flush()
            code = self.proc.wait()
        self.stop_group()
        return code


def stage_run(p, a, job):
    check_tools(p, a.pin)
    m = check_campaign(p, a.pin, a.tag, a.matrix)
    ev = p.ev
    require(not (ev / "run.json").exists(), "this campaign already ran (run.json): close it out, then use a new tag")
    want = os.environ.get("KVPD_GPUS", "4")
    require(os.environ.get("SLURM_GPUS_ON_NODE", want) == want, f"{want} GPUs are needed in this allocation")
    g = gpus()
    require(len(g) == int(want), f"nvidia-smi lists {len(g)} GPUs, not {want}")
    runtime = p.runtime(job)
    rec = {"schema": SCHEMA, "run_job": job, "runtime": str(runtime), "started": now(), "status": "started",
           "gpus": g, "node": socket.gethostname().split(".")[0]}
    write_json(ev / "run.json", rec, once=True)

    def finish(status, code=2, **extra):
        rec.update(status=status, finished=now(), **extra)
        write_json(ev / "run.json", rec)
        log(f"KVPD_RUN_JOB={job}")
        log(f"KVPD_RUN_STATUS={status}")
        if extra.get("reason"):
            log(f"   why: {extra['reason']}")
        for size, path in extra.get("largest_under_root") or []:
            log(f"   {size:>8}  {path}")
        log("next: cleanup, then verify-cleanup (each in its own allocation)")
        return code

    drivers = {x["driver"] for x in g}
    major = min(int(d.split(".")[0]) for d in drivers if d.split(".")[0].isdigit()) if drivers else 0
    if major < 580:
        return finish("stopped", reason=f"driver {sorted(drivers)}: vLLM 0.32's CUDA 13 wheels need 580 or newer")
    private_dir(p.cache, create=True)
    uv = find_uv(p.cache)
    if uv is None:
        return finish("stopped", reason="no uv (KVPD_UV, PATH or ~/.local/bin/uv)")
    ibp_opt = os.environ.get("KVPD_IBP", "auto")
    uses_ibp = any(re.search(r"(^|[,=])ibp(,|\s|$)", r) for r in m["runs"])
    want_ibp = ibp_opt == "1" or (ibp_opt == "auto" and uses_ibp)
    # MATRIX=lmctrace: its own setup (no codec captures), and the environment's own CUDA 13 compiler (no runfile)
    lmc = a.matrix == "lmctrace"
    plan = disk_plan(p, want_ibp, extra_gb(a.matrix, m["runs"]), job_peak_gb=LMC_SETUP_GB if lmc else None,
                     dropped_gb=0.0 if lmc else None)
    write_json(ev / "disk_plan.json", plan)
    if plan["ibp"] is None:
        rows = biggest(p.root)
        return finish("stopped", reason=(f"disk: {plan['free_gb']} GB free on {p.root}'s file system, the run needs "
                                         f"about {plan['need_gb']} GB and {plan['margin_gb']} GB stay free"),
                      largest_under_root=rows)
    if plan.get("note"):
        log("NOTE: " + plan["note"])
    private_dir(p.runtimes, create=True)
    require(not runtime.exists() and not runtime.is_symlink(), f"{runtime} exists already")
    runtime.mkdir(mode=0o700)
    cuda12 = find_cuda12()
    home, runs = runtime / "home", runtime / "runs"
    for d in (home, runs, runtime / "tmp", runtime / "pycache", ev / "matrix", ev / "setup"):
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
    env_vars = {
        "KVPD_TOOLS": str(p.tools), "KV_HOME": str(home), "KV_RUNS": str(runs), "KV_HF_HOME": str(p.cache / "hf"),
        "KV_UV": uv, "KV_UV_PYTHON": os.environ.get("KVPD_PYTHON", "3.12"), "UV_CACHE_DIR": str(p.cache / "uv"),
        "UV_LINK_MODE": "hardlink", "UV_PYTHON_INSTALL_DIR": str(p.cache / "python"),
        "UV_PYTHON_PREFERENCE": "only-managed", "UV_NO_CONFIG": "1", "PIP_CONFIG_FILE": "/dev/null",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "CUDA13_HOME": str(p.cache / "cuda13"),
        "IBP": "1" if plan["ibp"] else "0", "SKIP_IBP": "1", "SKIP_EXANS": "0", "DROP_CAPTURES": "1",
        "IBP_SKIP_REASON": "" if plan["ibp"] else (
            f"KVPD_IBP={ibp_opt} in the login shell" if ibp_opt == "0" else
            "the matrix does not use it" if not want_ibp else (plan.get("note") or "the disk plan left it out")),
        "TMPDIR": str(runtime / "tmp"), "PYTHONPYCACHEPREFIX": str(runtime / "pycache"), "PYTHONNOUSERSITE": "1",
        "XDG_CACHE_HOME": str(home / "cache"), "TRITON_CACHE_DIR": str(home / "cache" / "triton"),
        "TRITON_HOME": str(home / "triton"), "TORCHINDUCTOR_CACHE_DIR": str(home / "cache" / "inductor"),
        "CUDA_CACHE_PATH": str(home / "cache" / "nv"), "TORCH_HOME": str(home / "torch"),
        "TORCH_EXTENSIONS_DIR": str(home / "cache" / "torch_extensions"),
        "VLLM_CONFIG_ROOT": str(home / "config" / "vllm"), "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "MPLCONFIGDIR": str(home / "mpl"),
        "PYTHONFAULTHANDLER": "1",          # a native crash (nvCOMP's, ExANS's) prints where it happened
        "FLASHINFER_WORKSPACE_BASE": str(home / "cache"), "PD_CARD": "1", "RESULTS_ROOT": str(ev / "matrix"),
        "MATRIX": a.matrix, "RUNS": ";".join(m["runs"]),     # exactly the runs create recorded
        "KVPD_CAPTURE_DIR": str(p.root / "kvpd-captures" / a.tag),   # MATRIX=capture's raw pages (kept)
    }
    if cuda12:
        env_vars["CUDA12_HOME"] = cuda12
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE")}
    env.update(env_vars)
    env.pop("PYTHONDONTWRITEBYTECODE", None)   # bytecode goes to the runtime's pycache, never into the code
    env.pop("CUDA_HOME", None)                 # the engines' torch is CUDA 13: no toolkit of another version for it
    dropped = []
    for var in ("PATH", "LD_LIBRARY_PATH"):    # nor the login's CUDA module on the search paths (the builds name
        if env.get(var):                       # their toolkits: CUDA12_HOME, CUDA13_HOME)
            keep = [d for d in env[var].split(":") if not re.match(r"/usr/local/(pkgs/)?cuda", d)]
            dropped += [d for d in env[var].split(":") if d and d not in keep]
            env[var] = ":".join(keep)
    rec.update(uv=call(uv, "--version", check=False), cuda12_home=cuda12, ibp=bool(plan["ibp"]),
               environment=env_vars, dropped_from_paths=dropped)
    write_json(ev / "run.json", rec)
    # MATRIX=lmctrace needs only vLLM with LMCache, our overlay, MTRAG and the model (scripts/lmc_setup.sh)
    setup = os.environ.get("KVPD_SETUP_CMD", "bash scripts/lmc_setup.sh" if a.matrix == "lmctrace" else
                           "bash scripts/a5000_setup.sh && bash scripts/pd_setup.sh")
    # what the node is while this campaign runs (other jobs share it): GPUs, PCIe, every link measured
    specs = os.environ.get("KVPD_SPECS_CMD", "python3 -u scripts/node_specs.py --output "
                           f"{sh_quote(ev / 'node_specs')} --kv-home \"$KV_HOME\"")
    if os.environ.get("KVPD_SPECS", "1") == "0":
        specs = "echo 'KVPD_SPECS=0: no node specs this run'"
    matrix = os.environ.get("KVPD_MATRIX_CMD", "bash scripts/pd_matrix.sh")
    for name, cmd in (("setup", setup), ("specs", specs), ("matrix", matrix)):
        write_text(ev / f"node_{name}.sh", f"set -Eeuo pipefail\numask 077\ncd {sh_quote(p.tools)}\n{cmd}\n")
        call("bash", "-n", ev / f"node_{name}.sh")
    streamer = Streamer(ev / "run.log")

    def on_signal(signum, frame):
        streamer.stopped = signal.Signals(signum).name
        streamer.stop_group(wait=3.0)
        try:
            collect(p, runtime, ev)
        finally:
            finish("killed", code=1, signal=streamer.stopped)
            os._exit(1)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log(f"== run {job}: runtime {runtime}, results {ev / 'matrix'}, IBP {'on' if plan['ibp'] else 'off'}, "
        f"CUDA 12 for ExANS {cuda12 or 'none found (scripts/a5000_setup.sh downloads one)'}")
    code = streamer.run(ev / "node_setup.sh", env)
    rec["setup_exit"] = code
    if code != 0:
        collect(p, runtime, ev)
        return finish("setup failed", code=0)
    write_json(ev / "run.json", rec)
    rec["specs_exit"] = streamer.run(ev / "node_specs.sh", env)     # a failure here costs only the specs
    write_json(ev / "run.json", rec)
    code = streamer.run(ev / "node_matrix.sh", env)
    rec["matrix_exit"] = code
    collect(p, runtime, ev)
    status = {0: "completed", 3: "completed with failures"}.get(code, "matrix failed")
    return finish(status, code=0)


def sh_quote(x):
    return "'" + str(x).replace("'", "'\\''") + "'"


def collect(p, runtime, ev):
    """The setup's records into the campaign (small files only), the log compressed, the disk measured."""
    out = ev / "setup"
    out.mkdir(mode=0o700, exist_ok=True)
    home, runs = runtime / "home", runtime / "runs"
    picks = list(home.glob("pip_freeze*.txt")) + list(home.glob("lmc_verify.json")) + \
        list(home.glob("lmc_ops_build.json")) + \
        list(runs.glob("*/pd_selftest*.json")) + \
        list(runs.glob("*/pd_profile.*")) + \
        list(runs.glob("*/exans/exans_info.json")) + list(home.glob("venv*/pyvenv.cfg")) + \
        list(runs.glob("*/captures/meta.json"))
    for f in picks:
        if f.is_file() and f.stat().st_size < 4 * 2 ** 20:
            rel = f.relative_to(runtime)
            dst = out / "__".join(rel.parts)
            shutil.copyfile(f, dst)
    for f in home.glob("*.log"):              # the builds' and installers' logs (IBP, the CUDA 13 toolkit)
        lines = f.read_text(errors="replace").splitlines()[-300:]
        write_text(out / f"{f.stem}_tail.log", "\n".join(lines) + "\n")
    sizes = du_bytes([p.cache / "uv", p.cache / "python", p.cache / "hf", p.cache / "cuda13", runtime])
    write_json(ev / "disk_after.json", {"cache_uv_gb": sizes[0] / 1e9, "cache_python_gb": sizes[1] / 1e9,
                                        "cache_hf_gb": sizes[2] / 1e9, "cache_cuda13_gb": sizes[3] / 1e9,
                                        "runtime_own_gb": sizes[4] / 1e9, **disk_survey(p)})
    lf = ev / "run.log"
    if lf.is_file():
        data = lf.read_bytes()
        with gzip.open(ev / "run.log.gz", "wb") as g:
            g.write(data)
        write_text(ev / "run_tail.log", "\n".join(data.decode(errors="replace").splitlines()[-400:]) + "\n")
        lf.unlink()


# ------------------------------------------------------------------------------------------------- cleanup

TERMINAL_NOT = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING", "COMPLETING",
                "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING", "STAGE_OUT", "STOPPED"}


def sacct_row(job):
    out = call("sacct", "-n", "-P", "-X", "-j", job, "-o", "JobID,State,ExitCode,Start,End")
    rows = [r for r in out.splitlines() if r.split("|")[0] == str(job)]
    require(len(rows) == 1, f"Slurm has no single record of job {job}: {out[:300]!r}")
    return rows[0]


def terminal(job, saved, success=False, wait=180):
    """Slurm's final record of `job` (saved the first time, compared after); accounting may lag the job's end by
    a little, so a job still shown running is asked again for up to `wait` seconds."""
    t0 = time.time()
    while True:
        row = sacct_row(job)
        f = row.split("|")
        state = f[1].split()[0] if len(f) > 1 and f[1] else ""
        if (state and state not in TERMINAL_NOT) or time.time() - t0 > wait:
            break
        time.sleep(10)
    require(state and state not in TERMINAL_NOT, f"job {job} is {state or 'unknown'}, not finished")
    if success:
        require(state == "COMPLETED" and f[2] == "0:0", f"job {job} did not complete cleanly: {row}")
    if saved.exists():
        require(saved.read_text().strip() == row, f"Slurm's record of job {job} changed since it was saved")
    else:
        write_text(saved, row + "\n")
    return row


def process_refs(path):
    """Processes whose working directory, program, root, open files or mapped files are under `path`."""
    prefix = str(path) + "/"
    hits = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        refs = []
        for link in ("cwd", "exe", "root"):
            try:
                refs.append(os.readlink(d / link))
            except OSError:
                pass
        try:
            for fd in (d / "fd").iterdir():
                try:
                    refs.append(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
        try:
            refs += [l.split()[-1] for l in (d / "maps").read_text().splitlines() if len(l.split()) >= 6]
        except OSError:
            pass
        if any(r == str(path) or r.startswith(prefix) for r in refs):
            hits.append(int(d.name))
    return hits


def capped_copy(src, dst, limit=4_000_000):
    """src to dst whole when small, else its first 300 and last 400 lines."""
    if src.stat().st_size < limit:
        shutil.copyfile(src, dst)
        return
    lines = src.read_text(errors="replace").splitlines()
    write_text(dst, "\n".join(lines[:300] + [f"[... {len(lines)} lines in all ...]"] + lines[-400:]) + "\n")


def salvage(root, ev):
    """Before the runtime goes: the logs of every PD run folder (engines, bench, proxy, connector) and of the
    setup, for runs that ended before packaging them (killed, cancelled, out of time). Small files only."""
    out, got = ev / "salvaged", []
    names = ("engine_*.log", "bench.log", "proxy.log", "connector/*.jsonl", "results.json", "env.json")
    packaged = [d.name for d in (ev / "matrix").iterdir()] if (ev / "matrix").is_dir() else []
    for run in sorted((root / "runs").glob("*/pd/*")):
        if not run.is_dir() or run.name.startswith("prompts_"):
            continue
        parts = run.name.split("_", 2)             # <date>_<time>_<run tag>, packaged as ..._pd_<run tag>_..._<date>_<time>
        if len(parts) == 3 and any(f"_pd_{parts[2]}_" in d and f"_{parts[0]}_{parts[1]}" in d for d in packaged):
            continue
        for pat in names:
            for f in sorted(run.glob(pat)):
                if not f.is_file() or f.is_symlink():
                    continue
                dst = out / run.name / f.relative_to(run)
                dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if f.suffix == ".jsonl":
                    dst = dst.with_name(dst.name + ".gz")
                    with open(f, "rb") as src, gzip.open(dst, "wb") as g:
                        shutil.copyfileobj(src, g)
                else:
                    capped_copy(f, dst)
                got.append(str(dst.relative_to(ev)))
    for f in sorted((root / "home").glob("*.log")):
        if f.is_file() and not f.is_symlink():
            (out / "home").mkdir(mode=0o700, parents=True, exist_ok=True)
            capped_copy(f, out / "home" / f.name)
            got.append(f"salvaged/home/{f.name}")
    if (ev / "run.log").is_file():                 # the run stage ended before compressing its log
        data = (ev / "run.log").read_bytes()
        with gzip.open(ev / "run.log.gz", "wb") as g:
            g.write(data)
        (ev / "run.log").unlink()
        got.append("run.log.gz")
    return got


def stage_cleanup(p, a, job):
    check_tools(p, a.pin)
    check_campaign(p, a.pin, a.tag, a.matrix)
    ev = p.ev
    run = read_json(ev / "run.json")
    rj = str(run["run_job"])
    require(rj != job, "cleanup needs an allocation of its own, after the run")
    row = terminal(rj, ev / "run_job_sacct.psv")
    log(f"run job {rj}: {row}")
    if (ev / "cleanup.json").exists():
        log("the cleanup receipt exists: next, verify-cleanup in another allocation")
        return 0
    root = p.runtime(rj)
    require(Path(run["runtime"]) == root, "the run's runtime path differs")
    existed = root.exists() or root.is_symlink()
    before = after = []
    salvaged = []
    listed = 0
    free0 = shutil.disk_usage(p.root).free
    if existed:
        private_dir(p.runtimes)
        private_dir(root)
        require(root.parent == p.runtimes and root.name == rj, "not the run's runtime")
        before = process_refs(root)
        require(not before, f"processes {before[:10]} still use {root}")
        salvaged = salvage(root, ev)
        for dirpath, dirnames, filenames in os.walk(root):
            for n in filenames:
                try:
                    listed += os.lstat(os.path.join(dirpath, n)).st_size
                except OSError:
                    pass

        def unlock(func, path, info):      # read-only folders (some packages' files) are made writable first
            os.chmod(os.path.dirname(path), 0o700)
            if os.path.isdir(path) and not os.path.islink(path):
                os.chmod(path, 0o700)
            func(path)

        if sys.version_info >= (3, 12):
            shutil.rmtree(root, onexc=unlock)
        else:
            shutil.rmtree(root, onerror=unlock)
        after = process_refs(root) if root.exists() else []
    gone = not root.exists() and not root.is_symlink()
    require(gone, f"{root} is still there")
    free1 = shutil.disk_usage(p.root).free
    shm = sorted(f.name for f in Path("/dev/shm").iterdir()
                 if f.is_file() and f.stat().st_uid == os.getuid()) if Path("/dev/shm").is_dir() else []
    write_json(ev / "cleanup.json", {
        "schema": SCHEMA, "kind": "kvpd_runtime_cleanup", "pin": a.pin, "tag": a.tag, "run_job": rj,
        "cleanup_job": job, "runtime": str(root), "existed": existed, "removed": existed and gone,
        "absent_after": gone, "process_refs_before": len(before), "process_refs_after": len(after),
        "listed_bytes": listed, "freed_bytes": max(0, free1 - free0), "when": now(), "salvaged": salvaged,
        "own_dev_shm_files": shm[:50],
    }, once=True)
    if salvaged:
        log(f"   kept {len(salvaged)} log(s) of runs that did not package them: salvaged/ in the campaign")
    log(f"KVPD CLEANUP: {root} {'removed' if existed else 'was never made'}; {listed / 1e9:.1f} GB listed, "
        f"{max(0, free1 - free0) / 1e9:.1f} GB freed (files hard-linked into the cache stay there)")
    if shm:
        log(f"NOTE: {len(shm)} file(s) of yours in /dev/shm (not removed: another job of yours may own them): "
            f"{', '.join(shm[:8])}")
    return 0


def stage_verify(p, a, job, write=True):
    """The cleanup proven (verify-cleanup writes verify_cleanup.json; review and publish recheck it)."""
    check_tools(p, a.pin)
    check_campaign(p, a.pin, a.tag, a.matrix)
    ev = p.ev
    run = read_json(ev / "run.json")
    rj = str(run["run_job"])
    require((ev / "cleanup.json").is_file(), "no cleanup receipt: cleanup, then verify-cleanup, come before review "
            "and publish")
    rc = read_json(ev / "cleanup.json")
    cj = str(rc.get("cleanup_job"))
    require(rc.get("kind") == "kvpd_runtime_cleanup" and rc.get("pin") == a.pin and rc.get("run_job") == rj,
            "the cleanup receipt is another run's")
    require(len({rj, cj, job}) == 3, "the run, the cleanup and this check must be three allocations")
    root = p.runtime(rj)
    require(rc.get("runtime") == str(root) and rc.get("absent_after") is True and
            rc.get("process_refs_before") == 0 and rc.get("process_refs_after") == 0, "the receipt does not prove it")
    require(not root.exists() and not root.is_symlink(), f"{root} exists again")
    terminal(rj, ev / "run_job_sacct.psv")
    crow = terminal(cj, ev / "cleanup_job_sacct.psv", success=True)
    if (ev / "verify_cleanup.json").exists():
        v = read_json(ev / "verify_cleanup.json")
        require(v.get("run_job") == rj and v.get("cleanup_job") == cj, "the verification is another cleanup's")
    else:
        require(write, "verify-cleanup has not run: it comes before review and publish")
        write_json(ev / "verify_cleanup.json", {"schema": SCHEMA, "run_job": rj, "cleanup_job": cj,
                                                 "verify_job": job, "cleanup_sacct": crow, "when": now()}, once=True)
    if write:
        log(f"KVPD CLEANUP VERIFIED: run={rj} cleanup={cj}")
    return 0


# ------------------------------------------------------------------------------------------------- review, publish

def inventory(ev):
    files = []
    for f in sorted(ev.rglob("*")):
        rel = f.relative_to(ev).as_posix()
        if f.is_symlink():
            raise Stop(f"{rel} is a link")
        if f.is_dir() or rel in NOT_INVENTORIED:
            continue
        files.append({"path": rel, "bytes": f.stat().st_size, "sha256": sha256(f)})
    return files


def stage_review(p, a, job):
    stage_verify(p, a, job, write=False)
    ev = p.ev
    run = read_json(ev / "run.json")
    rv = ev / "review"
    if (rv / "review.json").exists():
        saved = read_json(rv / "inventory.json")
        require(saved == inventory(ev), "the evidence changed since its review")
        log(f"the review exists and the evidence is unchanged ({len(saved)} files)")
        show(rv / "summary.md")
        return 0
    rv.mkdir(mode=0o700, exist_ok=True)
    res = ev / "matrix"
    mats = sorted(res.glob("pd_matrix_*/matrix.json")) if res.is_dir() else []
    mj = json.loads(mats[-1].read_text()) if mats else {"runs": [], "failed": [], "folders": []}
    if not mats and res.is_dir():           # the matrix never ended (cancelled, out of time): what it packaged
        mj["folders"] = sorted(d.name for d in res.iterdir() if d.is_dir())
    folders = [f for f in mj.get("folders", []) if not f.startswith("pd_matrix_") and (res / f).is_dir()]
    status = run.get("status")
    if status == "started":                         # killed before it could write its end: Slurm's record says how
        row = (ev / "run_job_sacct.psv").read_text().strip() if (ev / "run_job_sacct.psv").is_file() else ""
        status = f"ended without a final status (Slurm: {row.split('|')[1] if row.count('|') else 'unknown'})"
    if folders and a.matrix == "lmctrace":          # the modes side by side, then each mode's own report
        call(sys.executable, "-I", "-B", p.tools / "scripts" / "lmc_lifecycle.py", "compare", "--dirs",
             ",".join(str(res / f) for f in folders), "--output", rv)
        parts = [f"# PD matrix {a.matrix} on {run.get('node')}, job {run['run_job']} ({status}), reviewed from the "
                 "kept results\n", (rv / "compare.md").read_text()]
        parts += [(res / f / "summary.md").read_text() for f in folders if (res / f / "summary.md").is_file()]
        write_text(rv / "summary.md", "\n\n".join(parts))
    elif folders:
        call(sys.executable, "-I", "-B", p.tools / "scripts" / "pd_matrix_report.py", "--runs", ",".join(folders),
             "--root", res, "--title", f"PD matrix {a.matrix} on {run.get('node')}, job {run['run_job']} "
             f"({status}), reviewed from the kept results", "--failed",
             "".join(f + ";" for f in mj.get("failed", [])), "--output", rv / "summary.md")
    else:
        write_text(rv / "summary.md", f"# PD matrix {a.matrix}, job {run['run_job']}: no run results\n\n"
                   f"Status: {status}. {run.get('reason', '')}\nSee run_tail.log, salvaged/ and setup/.\n")
    inv = inventory(ev)
    write_json(rv / "inventory.json", inv, once=True)
    write_json(rv / "review.json", {
        "schema": SCHEMA, "review_job": job, "run_job": str(run["run_job"]), "status": run.get("status"),
        "runs": mj.get("runs", []), "failed": mj.get("failed", []), "folders": folders, "files": len(inv),
        "bytes": sum(f["bytes"] for f in inv), "when": now()}, once=True)
    log(f"KVPD REVIEW: {len(folders)} run folders, {len(inv)} files, {sum(f['bytes'] for f in inv) / 1e6:.1f} MB; "
        f"run status {status}")
    show(rv / "summary.md")
    return 0


def show(path, lines=80):
    if path.is_file():
        text = path.read_text(errors="replace").splitlines()
        log("\n".join(text[:lines]) + ("\n..." if len(text) > lines else ""))


def scan(ev):
    found, total = [], 0
    pats = {k: re.compile(v) for k, v in SECRETS.items()}
    for f in sorted(ev.rglob("*")):
        rel = f.relative_to(ev).as_posix()
        if f.is_symlink():
            found.append(f"{rel}: a link")
            continue
        if f.is_dir():
            continue
        size = f.stat().st_size
        total += size
        if size > MAX_FILE:
            found.append(f"{rel}: {size / 2 ** 20:.0f} MB (GitHub refuses files over 100 MB)")
            continue
        data = f.read_bytes()
        if f.suffix == ".gz":
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
        for name, rx in pats.items():
            if rx.search(data):
                found.append(f"{rel}: looks like a {name}")
    if total > MAX_TOTAL:
        found.append(f"{total / 2 ** 20:.0f} MB in all (more than {MAX_TOTAL / 2 ** 20:.0f} MB)")
    return found


def stage_publish(p, a, job):
    stage_verify(p, a, job, write=False)
    ev = p.ev
    run = read_json(ev / "run.json")
    rv = read_json(ev / "review" / "review.json")
    rj = str(run["run_job"])
    branch = f"evidence/{a.branch}/{a.pin[:12]}/{a.matrix}/job{rj}"
    if (ev / "publish.json").exists():
        pub = read_json(ev / "publish.json")
        remote = git(p.repo, "ls-remote", "origin", f"refs/heads/{branch}").split()
        require(remote and remote[0] == pub["commit"], f"{branch} on GitHub is not the published commit")
        log(f"PUBLISHED {branch} {pub['commit']} (already)")
        return 0
    require(read_json(ev / "review" / "inventory.json") == inventory(ev), "the evidence changed since its review")
    bad = scan(ev)
    require(not bad, "not published:\n   " + "\n   ".join(bad))
    home = Path(a.home_repo)
    name = git(home, "config", "user.name", check=False)
    email = git(home, "config", "user.email", check=False)
    require(name and email, f"git user.name and user.email are not set for {home}")
    remote = git(p.repo, "ls-remote", "origin", f"refs/heads/{branch}")
    require(remote == "", f"{branch} exists on GitHub already")
    if git(p.repo, "rev-parse", "HEAD") == a.pin:
        # -f: the repository ignores runs/ and results/ folders (local output); here every file is the reviewed one
        git(p.repo, "add", "-A", "-f", "--", str(p.rel))
        msg = (f"Evidence: PD matrix {a.matrix}, job {rj} on {run.get('node')} ({run.get('status')})\n\n"
               f"Source: {a.pin} ({a.branch})\nCampaign: {a.tag}\n"
               f"Runs: {len(rv.get('runs', []))}, not finished: {len(rv.get('failed', []))}\n"
               f"Cleanup: job {read_json(ev / 'cleanup.json')['cleanup_job']}, verified; review: job "
               f"{rv['review_job']}\nFiles: {rv['files']} ({rv['bytes'] / 1e6:.1f} MB)\n")
        git(p.repo, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-q", "--no-verify", "-m", msg)
    head = git(p.repo, "rev-parse", "HEAD")
    require(git(p.repo, "rev-parse", "HEAD^") == a.pin, "the evidence commit is not on the pin")
    changed = git(p.repo, "diff", "--name-only", a.pin, "HEAD").splitlines()
    require(changed and all(c.startswith(str(p.rel) + "/") for c in changed), "the commit has other files")
    want = {f"{p.rel}/{f['path']}" for f in read_json(ev / "review" / "inventory.json")} | \
        {f"{p.rel}/review/inventory.json", f"{p.rel}/review/review.json"}
    require(set(changed) == want, f"the commit is not the reviewed files: {sorted(set(changed) ^ want)[:10]}")
    git(p.repo, "push", "-q", "origin", f"HEAD:refs/heads/{branch}")
    remote = git(p.repo, "ls-remote", "origin", f"refs/heads/{branch}").split()
    require(remote and remote[0] == head, "GitHub does not have the pushed commit")
    write_json(ev / "publish.json", {"schema": SCHEMA, "branch": branch, "commit": head, "publish_job": job,
                                     "files": len(changed), "when": now()}, once=True)
    log(f"PUBLISHED {branch} {head}")
    log(f"   pull it: git fetch origin {branch} && git log --stat -1 FETCH_HEAD")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("create", "run", "cleanup", "verify-cleanup", "review", "publish"))
    ap.add_argument("--pin", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--home-repo", required=True)
    a = ap.parse_args(argv)
    os.umask(0o077)
    try:
        require(re.fullmatch(r"[0-9a-f]{40}", a.pin), "the pin is a 40-character commit")
        mt = re.fullmatch(r"pd_(%s)_[A-Za-z0-9_-]+" % "|".join(MATRICES), a.tag)
        require(mt, "the tag is pd_<matrix>_<name>")
        a.matrix = mt.group(1)
        a.branch = os.environ.get("KVPD_BRANCH") or "pd-baselines"
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", a.branch), "KVPD_BRANCH is not a plain branch name")
        job = check_node()
        p = Paths(a.pin, a.tag)
        home = Path(a.home_repo)
        require(home.is_absolute() and (home / ".git").exists(), "the home checkout is not a repository")
        require(not str(home).startswith(str(p.root) + "/"), "the home checkout is under the node-local root")
        if a.stage == "create":
            return stage_create(p, a, job) or 0
        if a.stage == "run":
            return stage_run(p, a, job)
        if a.stage == "cleanup":
            return stage_cleanup(p, a, job)
        if a.stage == "verify-cleanup":
            return stage_verify(p, a, job)
        if a.stage == "review":
            return stage_review(p, a, job)
        return stage_publish(p, a, job)
    except Stop as e:
        print(f"STOP: PD campaign {a.stage}: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
