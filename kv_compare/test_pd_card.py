"""scripts/pd_card.sh and scripts/pd_campaign.py end to end: every stage of a campaign on a throwaway home
checkout and GitHub stand-in, with stand-ins for srun, sacct, nvidia-smi, uv and vLLM (the setup replaced by one that
makes stand-in environments). What is checked: the login checks, each stage's allocation, the node-local folders and
their modes, one run per campaign, the runtime deleted only by a later allocation, the review's inventory, and the
published branch: one commit on the pin with only the campaign's files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

from kv_compare.test_pd_baselines import FAKE_VLLM

ROOT = Path(__file__).resolve().parents[1]

FAKE_SRUN = r'''#!/usr/bin/env bash
# srun stand-in: a new job id per call, the allocation's variables, the script on stdin; sacct's record afterwards
db=$FAKE_SLURM_DB
n=$(( $(cat "$db/next" 2>/dev/null || echo 41000) + 1 )); echo "$n" > "$db/next"
printf '%s\n' "$*" > "$db/argv_$n"
dir=/tmp
for a in "$@"; do
  case "$a" in --chdir=*) dir=${a#--chdir=} ;; --gpus=*) export SLURM_GPUS_ON_NODE=${a#--gpus=} ;; esac
done
export SLURM_JOB_ID=$n
cd "$dir"
"${@: -2:1}" "${@: -1}"
code=$?
state=COMPLETED; [ "$code" = 0 ] || state=FAILED
echo "$n|$state|$code:0|2026-11-14T10:00:00|2026-11-14T11:00:00" > "$db/job_$n"
exit $code
'''

FAKE_SACCT = r'''#!/usr/bin/env bash
job=""
while [ $# -gt 0 ]; do case "$1" in -j) job=$2; shift ;; esac; shift; done
cat "$FAKE_SLURM_DB/job_$job" 2>/dev/null
'''

FAKE_SMI = r'''#!/usr/bin/env bash
case "$*" in
  *--query-gpu=index,name,uuid*) for i in 0 1 2 3; do echo "$i, NVIDIA RTX A5000, GPU-0000000$i-1111-2222-3333-444444444444, 595.45.04, 24564 MiB"; done ;;
  *--query-gpu=index,uuid*) for i in 0 1 2 3; do echo "$i, GPU-0000000$i-1111-2222-3333-444444444444"; done ;;
  *--query-gpu=index,name*) for i in 0 1 2 3; do echo "$i, NVIDIA RTX A5000, 1 MiB, 24564 MiB, 595.45.04"; done ;;
  *--query-gpu=index*) printf '0\n1\n2\n3\n' ;;
  *--query-gpu=name*) printf 'NVIDIA RTX A5000\nNVIDIA RTX A5000\nNVIDIA RTX A5000\nNVIDIA RTX A5000\n' ;;
  *--query-gpu=driver_version*) printf '595.45.04\n595.45.04\n595.45.04\n595.45.04\n' ;;
  *) echo "fake nvidia-smi" ;;
esac
'''

# the setup, replaced: stand-in environments where scripts/pd_setup.sh would make them
FAKE_SETUP = r'''#!/usr/bin/env bash
set -euo pipefail
echo "fake setup in $KV_HOME (uv $KV_UV, cache $UV_CACHE_DIR, link mode $UV_LINK_MODE)"
for env in venv_vllm venv_lmcache; do
  mkdir -p "$KV_HOME/$env/bin"
  printf '#!/usr/bin/env bash\nPYTHONPATH="%s:${PYTHONPATH:-}" exec %s "$@"\n' "$FAKE_STUB" "$FAKE_PY" > "$KV_HOME/$env/bin/python"
  cp "$FAKE_VLLM_BIN" "$KV_HOME/$env/bin/vllm"
  chmod 755 "$KV_HOME/$env/bin/python" "$KV_HOME/$env/bin/vllm"
done
run=$KV_RUNS/llama32_3b
p=$run/pd/prompts_2048_check_1000_3000_5000_8192
mkdir -p "$p" "$run/captures"
printf 'x' > "$run/pd_codecs.pt"
echo '{"prompt": "hello", "output_tokens": 4}' > "$p/check.jsonl"
echo '{"prompt": "hello", "output_tokens": 4}' > "$p/prompts_mixed.jsonl"
echo '{"codecs": {"raw": {"ok": true}}}' > "$run/pd_selftest.json"
echo "pip freeze stand-in" > "$KV_HOME/pip_freeze_vllm.txt"
'''


@unittest.skipIf(__import__("importlib").util.find_spec("fastapi") is None, "fastapi / uvicorn / aiohttp needed")
class CardTests(unittest.TestCase):

    def setUp(self):
        if shutil.which("curl") is None or shutil.which("git") is None:
            self.skipTest("curl and git needed")
        self.tmp = Path(tempfile.mkdtemp())
        if not os.environ.get("KEEP_CARD_TMP"):
            self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.user = pwd.getpwuid(os.getuid()).pw_name
        # git with a configuration of its own: no signing, no URL rewrites from the machine's
        gitconfig = self.tmp / "gitconfig"
        gitconfig.write_text("[user]\n\tname = t\n\temail = t@t\n[init]\n\tdefaultBranch = pd-baselines\n")
        self.genv = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.genv.update(GIT_CONFIG_GLOBAL=str(gitconfig), GIT_CONFIG_NOSYSTEM="1")
        g = lambda repo, *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True,
                                            text=True, env=self.genv).stdout.strip()
        # GitHub: a bare repository with pd-baselines = the working tree's code, and an older results folder
        src = self.tmp / "src"
        src.mkdir()
        for d in ("scripts", "kv_compare", "qwen_kv_bench"):
            shutil.copytree(ROOT / d, src / d, ignore=shutil.ignore_patterns("__pycache__"))
        for f in (".gitignore", "requirements.txt", "requirements-cuda.txt"):
            shutil.copy(ROOT / f, src / f)
        (src / "benchmark_results" / "old_run").mkdir(parents=True)
        (src / "benchmark_results" / "old_run" / "summary.md").write_text("an earlier run\n")
        g(self.tmp, "init", "-q", "-b", "pd-baselines", str(src))
        g(src, "add", "-A")
        g(src, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "code")
        self.origin = self.tmp / "origin.git"
        g(self.tmp, "clone", "-q", "--bare", str(src), str(self.origin))
        self.home = self.tmp / "home" / "kv_reconstruction"
        g(self.tmp, "clone", "-q", "-b", "pd-baselines", str(self.origin), str(self.home))
        g(self.home, "config", "user.email", "t@t")
        g(self.home, "config", "user.name", "t")
        self.pin = g(self.home, "rev-parse", "HEAD")
        self.g = g
        # stand-ins
        b = self.tmp / "bin"
        b.mkdir()
        stub = self.tmp / "stub"
        (stub / "vllm").mkdir(parents=True)
        (stub / "vllm" / "__init__.py").write_text("__version__ = '0.32.0'\n")
        (stub / "nixl").mkdir()                    # NIXL's module as vLLM imports it: nixl._api.nixl_agent
        (stub / "nixl" / "__init__.py").write_text("")
        (stub / "nixl" / "_api.py").write_text("nixl_agent = object\n")
        vllm = FAKE_VLLM.replace("#!/usr/bin/env python3", f"#!{sys.executable}")
        for name, text in (("srun", FAKE_SRUN), ("sacct", FAKE_SACCT), ("nvidia-smi", FAKE_SMI), ("vllm", vllm),
                           ("uv", "#!/usr/bin/env bash\necho 'uv 0.0.0 (stand-in)'\n"), ("fake_setup", FAKE_SETUP)):
            (b / name).write_text(text)
            (b / name).chmod(0o755)
        db = self.tmp / "slurm"
        db.mkdir()
        self.db = db
        self.node = self.tmp / "node"
        self.env = {k: v for k, v in self.genv.items() if not k.startswith(("SLURM_", "KVPD_"))}
        self.env.pop("CUDA_VISIBLE_DEVICES", None)
        self.env.pop("RUNS", None)
        self.env.update(
            PATH=f"{b}:{Path(sys.executable).parent}:/usr/bin:/bin", USER=self.user, FAKE_SLURM_DB=str(db),
            KVPD_ROOT=str(self.node), KVPD_NODE=socket.gethostname().split(".")[0], KVPD_UV=str(b / "uv"),
            KVPD_NEED_SCALE="0", KVPD_MARGIN_GB="0", KVPD_SETUP_CMD=f"bash {b / 'fake_setup'}",
            KVPD_SPECS_CMD="echo specs stand-in",
            FAKE_STUB=str(stub), FAKE_PY=sys.executable, FAKE_VLLM_BIN=str(b / "vllm"),
            FAKE_LOG=str(self.tmp / "engines.jsonl"), PORT_BASE="36000", MODEL_PATH=str(self.tmp / "model"),
            SMOKE="1")
        self.tag = "pd_smoke_test1"

    def card(self, stage, pin=None, tag=None, **env):
        e = dict(self.env, **{k: str(v) for k, v in env.items()})
        return subprocess.run(["bash", str(self.home / "scripts" / "pd_card.sh"), stage, pin or self.pin,
                               tag or self.tag], env=e, capture_output=True, text=True, timeout=900)

    def jobs(self):
        return sorted(int(f.name[5:]) for f in self.db.glob("argv_*"))

    def test_a_whole_campaign(self):
        runs = "SMOKE=1 LAYOUT=1p3d CODECS=raw; SMOKE=1 LAYOUT=colo4 CONNECTOR=lmcache"
        r = self.card("create", RUNS=runs)
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        tools = self.node / "kvpd-tools" / self.pin
        self.assertTrue((tools / "scripts" / "pd_campaign.py").is_file())
        self.assertFalse((tools / "benchmark_results").exists(), "the earlier results are not checked out")
        repo = self.node / "kvpd-evidence" / self.tag / "repo"
        ev = repo / "benchmark_results" / "pd_card" / self.tag
        for d in (self.node, self.node / "kvpd-tools", tools, repo.parent, repo):
            self.assertEqual(d.stat().st_mode & 0o777, 0o700, d)
        m = json.loads((ev / "campaign.json").read_text())
        self.assertEqual((m["pin"], m["matrix"], m["runs"]), (self.pin, "smoke", [r.strip() for r in runs.split(";")]))
        self.assertEqual(m["boot_id"], Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        self.assertFalse((repo / "benchmark_results" / "old_run").exists())
        create_job = self.jobs()[-1]
        self.assertIn("-c 2 --mem=8G", (self.db / f"argv_{create_job}").read_text())
        self.assertEqual(self.card("create").returncode, 2, "a tag is used once")

        # the run: 4 GPUs, its own runtime, the recorded runs (not the environment's), results into the campaign
        r = self.card("run", RUNS="SMOKE=1 LAYOUT=2p2d CONNECTOR=nixl")
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-3000:])
        run_job = self.jobs()[-1]
        argv = (self.db / f"argv_{run_job}").read_text()
        for want in ("--gpus=4", "-c 16", "--mem=192G", "-t 06:00:00", "--chdir=/tmp", "--export=ALL",
                     "--propagate=NONE", f"-w {self.env['KVPD_NODE']}", "-A oana2026", "-q research12h"):
            self.assertIn(want, argv)
        rec = json.loads((ev / "run.json").read_text())
        self.assertEqual(rec["run_job"], str(run_job))
        self.assertIn(rec["status"], ("completed", "completed with failures"), rec)
        self.assertEqual(rec["environment"]["RUNS"], ";".join(x.strip() for x in runs.split(";")))
        self.assertEqual(rec["environment"]["UV_LINK_MODE"], "hardlink")
        self.assertTrue(rec["environment"]["UV_CACHE_DIR"].startswith(str(self.node / "kvpd-cache")))
        runtime = self.node / "kvpd" / str(run_job)
        self.assertTrue((runtime / "home" / "venv_vllm" / "bin" / "python").is_file(), "kept until cleanup")
        self.assertTrue((ev / "run.log.gz").is_file() and (ev / "run_tail.log").is_file())
        folders = sorted(d.name for d in (ev / "matrix").iterdir())
        self.assertEqual(len(folders), 3, folders)        # two runs and the combined report
        self.assertTrue(any("1p3d_tkv_ring" in f for f in folders) and any("colo4_lmcache" in f for f in folders))
        mat = next((ev / "matrix").glob("pd_matrix_smoke_*/matrix.json"))
        self.assertEqual(len(json.loads(mat.read_text())["runs"]), 2)
        self.assertTrue((ev / "setup" / "home__pip_freeze_vllm.txt").is_file())
        self.assertEqual(subprocess.run(["git", "-C", str(tools), "status", "--porcelain"], capture_output=True,
                                        text=True).stdout, "", "nothing written into the code")
        r = self.card("run")
        self.assertEqual(r.returncode, 2)
        self.assertIn("already ran", r.stderr)

        # review before the cleanup is verified: refused
        r = self.card("review")
        self.assertEqual(r.returncode, 2)
        self.assertIn("come before review", r.stderr + r.stdout)
        r = self.card("cleanup")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertFalse(runtime.exists())
        rc = json.loads((ev / "cleanup.json").read_text())
        self.assertEqual((rc["run_job"], rc["removed"], rc["process_refs_before"]), (str(run_job), True, 0))
        self.assertTrue((self.node / "kvpd-tools" / self.pin).is_dir(), "the code stays")
        r = self.card("verify-cleanup")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn(f"KVPD CLEANUP VERIFIED: run={run_job}", r.stdout)
        r = self.card("review")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        self.assertIn("KVPD REVIEW: 2 run folders", r.stdout)
        inv = json.loads((ev / "review" / "inventory.json").read_text())
        self.assertTrue(any(f["path"].endswith("results.json") for f in inv))

        # publish: one commit on the pin, only the campaign's folder (runs/ folders too, which .gitignore drops)
        r = self.card("publish")
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        branch = f"evidence/pd-baselines/{self.pin[:12]}/smoke/job{run_job}"
        self.assertIn(f"PUBLISHED {branch}", r.stdout)
        head = self.g(self.origin, "rev-parse", branch)
        self.assertEqual(self.g(self.origin, "rev-parse", f"{branch}^"), self.pin)
        files = self.g(self.origin, "diff", "--name-only", self.pin, branch).splitlines()
        self.assertTrue(files and all(f.startswith(f"benchmark_results/pd_card/{self.tag}/") for f in files))
        self.assertTrue(any("/runs/" in f for f in files), "the runs' JSON files are published")
        self.assertTrue(any(f.endswith("review/summary.md") for f in files))
        self.assertFalse(any(f.endswith("publish.json") for f in files))
        self.assertEqual(self.g(self.origin, "rev-parse", "pd-baselines"), self.pin, "pd-baselines untouched")
        r = self.card("publish")
        self.assertEqual(r.returncode, 0)
        self.assertIn("(already)", r.stdout)
        self.assertEqual(self.g(self.origin, "rev-parse", branch), head)

    def test_login_checks(self):
        # inside an allocation, a bad tag, a pin that is not the checkout's, uncommitted changes: no srun at all
        r = self.card("create", SLURM_JOB_ID="5")
        self.assertIn("not inside an allocation", r.stderr)
        self.assertNotEqual(self.card("create", tag="pd_nosuch_x").returncode, 0)
        r = self.card("create", pin="0" * 40)
        self.assertIn("not the pin", r.stderr)
        (self.home / "scripts" / "pd_proxy.py").write_text("# changed\n")
        r = self.card("create")
        self.assertIn("uncommitted changes", r.stderr)
        self.g(self.home, "checkout", "-q", "--", "scripts/pd_proxy.py")
        # no git identity: publish could not commit, so nothing starts
        bare = self.tmp / "gitconfig_bare"
        bare.write_text("[init]\n\tdefaultBranch = pd-baselines\n")
        self.g(self.home, "config", "--unset", "user.name")
        r = self.card("create", GIT_CONFIG_GLOBAL=bare)
        self.assertIn("git has no user.name and user.email", r.stderr)
        self.g(self.home, "config", "user.name", "t")
        # GitHub's branch moved on: a new campaign must start from its head
        self.g(self.home, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "next")
        self.g(self.home, "push", "-q", "origin", "HEAD:pd-baselines")
        self.g(self.home, "checkout", "-q", self.pin)
        r = self.card("create")
        self.assertIn("not the pin: a new campaign starts from its head", r.stderr)
        self.assertEqual(self.jobs(), [])

    def test_a_run_that_stops_on_the_disk_check_is_closed_out(self):
        self.assertEqual(self.card("create", RUNS="SMOKE=1 LAYOUT=colo4 TIER_CODECS=raw,ibp").returncode, 0)
        r = self.card("run", KVPD_NEED_SCALE="1", KVPD_MARGIN_GB="100000")
        self.assertEqual(r.returncode, 2)
        self.assertIn("KVPD_RUN_STATUS=stopped", r.stdout)
        self.assertIn("why: disk:", r.stdout)
        ev = self.node / "kvpd-evidence" / self.tag / "repo" / "benchmark_results" / "pd_card" / self.tag
        plan = json.loads((ev / "disk_plan.json").read_text())
        self.assertTrue(plan["ibp_wanted"])
        self.assertIsNone(plan["ibp"])
        for stage in ("cleanup", "verify-cleanup", "review"):
            r = self.card(stage)
            self.assertEqual(r.returncode, 0, stage + r.stdout[-2000:] + r.stderr[-2000:])
        self.assertFalse(json.loads((ev / "cleanup.json").read_text())["existed"])
        self.assertIn("no run results", (ev / "review" / "summary.md").read_text())


class DiskPlanTests(unittest.TestCase):
    """What the run stage decides from the free space: everything, everything but IBP's toolkit, or stop."""

    def plan(self, free_gb, want_ibp, have=(), extra=0.0):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_campaign as c
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        p = c.Paths("0" * 40, "pd_smoke_x")
        p.root, p.cache = tmp, tmp / "kvpd-cache"
        for h in have:
            d = {"uv": p.cache / "uv", "python": p.cache / "python", "hf": p.cache / "hf" / c.MODEL_DIR}[h]
            d.mkdir(parents=True)
            (d / "x").write_text("x")
        orig = shutil.disk_usage
        shutil.disk_usage = lambda path: type("U", (), {"free": free_gb * 1e9, "total": 200e9})()
        try:
            return c.disk_plan(p, want_ibp, extra)
        finally:
            shutil.disk_usage = orig

    def test_choices(self):
        # nothing cached: 17 + 0.2 + 6.6 GB, then the job's 14 (with IBP's toolkit: 4 + max(11, 14)); 10 GB stay free
        self.assertTrue(self.plan(58, True)["ibp"])
        p = self.plan(50, True)
        self.assertFalse(p["ibp"])
        self.assertIn("IBP left out", p["note"])
        self.assertIsNone(self.plan(40, False)["ibp"])
        self.assertFalse(self.plan(28, False, have=("uv", "python", "hf"))["ibp"] is None)

    def test_a_capture_counts_after_setup_drops_its_own(self):
        """MATRIX=capture's stores fill after setup has deleted its captures (6.6 GB): the job's peak is the larger
        of setup's (14) and the matrix's own files (7.4) plus the stores, not both added."""
        import os
        from unittest import mock
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_campaign as c
        with mock.patch.dict(os.environ, {"KVPD_CAPTURE_CONVS": "4"}):
            four = c.extra_gb("capture")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KVPD_CAPTURE_CONVS", None)
            six = c.extra_gb("capture")
        self.assertEqual((four, six, c.extra_gb("tree")), (8.0 + 2.0, 16.0 + 2.0, 0.0))
        # nothing cached (23.8 GB) + max(14, 7.4 + 10) = 41.2 GB, and 10 kept free
        p = self.plan(51.3, False, extra=four)
        self.assertEqual((p["need_gb"], p["setup_gb"], p["matrix_gb"]), (41.2, 14.0, 17.4))
        self.assertFalse(p["ibp"] is None)
        self.assertIsNone(self.plan(51.1, False, extra=four)["ibp"])
        # six conversations: 23.8 + 7.4 + 18 = 49.2 (before: 23.8 + 18 + 14 = 55.8)
        self.assertEqual(self.plan(60, False, extra=six)["need_gb"], 49.2)


@unittest.skipIf(shutil.which("uv") is None, "uv needed")
class EnvHelperTests(unittest.TestCase):
    """scripts/pd_env.sh with uv: environments hard-linked into one cache (offline: this Python, no packages)."""

    def test_uv_helpers(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        script = f"""set -euo pipefail
source {ROOT / 'scripts' / 'pd_env.sh'}
venv_create "{tmp}/a"
venv_create "{tmp}/a"
pip_out "{tmp}/a/bin/python" not-installed-anywhere
pip_freeze "{tmp}/a/bin/python" > "{tmp}/freeze.txt"
echo "$UV_INDEX_STRATEGY $UV_NO_CONFIG"
"""
        env = dict(os.environ, KV_UV=shutil.which("uv"), KV_UV_PYTHON=sys.executable, UV_CACHE_DIR=str(tmp / "cache"),
                   UV_LINK_MODE="hardlink", UV_OFFLINE="1")
        r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((tmp / "a" / "bin" / "python").exists())
        self.assertIn("unsafe-best-match 1", r.stdout)
        r = subprocess.run(["bash", "-c", f"source {ROOT / 'scripts' / 'pd_env.sh'}"],
                           env=dict(os.environ, KV_UV=str(tmp / "no-uv")), capture_output=True, text=True)
        self.assertIn("not an executable uv", r.stdout)


class SalvageTests(unittest.TestCase):
    """cleanup keeps the logs of the runs that ended before packaging them (and only those)."""

    def test_salvage(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import pd_campaign as c
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        root, ev = tmp / "rt", tmp / "ev"
        for name in ("20261114_2011_1p3d_tkv_ring", "20261114_2016_1p3d_tkv_ring_eager", "prompts_2048_check_1"):
            d = root / "runs" / "llama32_3b" / "pd" / name
            (d / "connector").mkdir(parents=True)
            (d / "engine_prefill1.log").write_text("Traceback\nValueError: x\n")
            (d / "connector" / "Pw.jsonl").write_text('{"event": "ready"}\n')
        (root / "home").mkdir()
        (root / "home" / "cuda13_installer.log").write_text("[ERROR]: something\n")
        (ev / "matrix" / "llama32_3b_pd_1p3d_tkv_ring_rtxa5000x4_20261114_2011_smoke_partial").mkdir(parents=True)
        (ev / "run.log").write_text("the run's output\n")
        got = c.salvage(root, ev)
        self.assertIn("salvaged/20261114_2016_1p3d_tkv_ring_eager/engine_prefill1.log", got)
        self.assertIn("salvaged/20261114_2016_1p3d_tkv_ring_eager/connector/Pw.jsonl.gz", got)
        self.assertFalse(any("20261114_2011_" in g for g in got), "a packaged run is not copied twice")
        self.assertFalse(any("prompts_" in g for g in got))
        self.assertIn("salvaged/home/cuda13_installer.log", got)
        self.assertTrue((ev / "run.log.gz").is_file() and not (ev / "run.log").exists())


class DropWindowsTests(unittest.TestCase):
    """scripts/pd_export_codecs.py's --drop-cal-captures: a set's calibration windows go once it is fitted, unless a
    later set uses them; the main captures stay."""

    def test_drop(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from pd_export_codecs import drop_windows
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for d in ("cal_code_0", "cal_code_1", "cal_code_10", "cal_math_0", "wtrain0"):
            for part in ("K_bits.npy", "V_bits.npy"):
                (tmp / f"{d}_{part}").write_bytes(b"x" * 10)
        drop_windows(tmp, ["cal_code_0", "cal_code_1", "wtrain0", "cal_math_0"],
                     [(["cal_math_0"], "cal_math_3", "later")])
        left = sorted(f.name for f in tmp.iterdir())
        self.assertEqual(left, ["cal_code_10_K_bits.npy", "cal_code_10_V_bits.npy", "cal_math_0_K_bits.npy",
                                "cal_math_0_V_bits.npy", "wtrain0_K_bits.npy", "wtrain0_V_bits.npy"])


if __name__ == "__main__":
    unittest.main()
