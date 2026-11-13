#!/usr/bin/env python3
"""What this node is, for planning the PD runs (docs/pd_baselines.md): one JSON and a summary, from inside a Slurm
allocation of the GPUs the runs will use (4 on gpu-grad-01).

  system     (no torch needed) the job (GPUs, CPUs, memory), Slurm's node record and the account's limits, CPU and
             NUMA, memory and /dev/shm, disks and quota, network interfaces and RDMA devices, every GPU from
             nvidia-smi -q (memory, PCIe generation and width, bus, power, clocks, ECC), topology, peer-to-peer
             matrices, driver, and the software on the node (python, gcc, nvcc, cmake, the KV_HOME environments)
  bandwidth  (torch) every GPU's host legs and every ordered pair (scripts/pd_link_probe.py), then the concurrent
             patterns of the planned runs, each direct (peer to peer) and through pinned host memory:
               1p3d          one prefill GPU sends to three decode GPUs at once
               2p2d          two prefill GPUs send to two decode GPUs at once, for every pairing
               cache reload  pinned host memory to 1..N GPUs at once (a CPU cache tier, LMCache's, filling GPUs)
               cache store   N GPUs to pinned host memory at once
             one 8K document of Llama-3.2-3B KV per transfer (896 MiB in BF16)

  python scripts/node_specs.py --output DIR [--python PY] [--skip-bandwidth]

The system part runs with any python3; the bandwidth part runs itself again under --python (a python with torch,
found under KV_HOME when not given) and is left out, with the reason, when there is none.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIB = 896


def sh(cmd, timeout=60):
    """A command's output (stdout, else stderr), or why it did not run; never raises."""
    if isinstance(cmd, str):
        cmd = ["bash", "-lc", cmd]
    exe = cmd[0]
    if exe != "bash" and shutil.which(exe) is None:
        return f"(no {exe} on this node)"
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout} s)"
    except Exception as exc:
        return f"({exc!r})"
    out = r.stdout.strip() or r.stderr.strip()
    return out if r.returncode == 0 or out else f"(exit {r.returncode})"


def read(path):
    try:
        return Path(path).read_text().strip()
    except Exception as exc:
        return f"({exc.__class__.__name__})"


# ------------------------------------------------------------------------------------------------------ system
GPU_FIELDS = ("index,name,uuid,pci.bus_id,memory.total,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,"
              "pcie.link.width.current,power.limit,clocks.max.sm,clocks.max.memory,ecc.mode.current,"
              "persistence_mode,compute_mode,driver_version")


def gpus():
    out = sh(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"])
    keys = GPU_FIELDS.split(",")
    rows = []
    for line in out.splitlines():
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == len(keys):
            rows.append(dict(zip(keys, vals)))
    return rows, out


def numa_of(bus_id):
    """The NUMA node of a GPU from sysfs (bus id 00000000:3B:00.0 -> /sys/bus/pci/devices/0000:3b:00.0)."""
    m = re.match(r"([0-9A-Fa-f]+):([0-9A-Fa-f]+):([0-9A-Fa-f]+)\.(\d)", bus_id or "")
    if not m:
        return None
    dom = m.group(1)[-4:].lower()
    p = Path(f"/sys/bus/pci/devices/{dom}:{m.group(2).lower()}:{m.group(3).lower()}.{m.group(4)}/numa_node")
    v = read(p)
    return int(v) if v.lstrip("-").isdigit() else None


def network():
    nics = []
    for d in sorted(Path("/sys/class/net").glob("*")) if Path("/sys/class/net").is_dir() else []:
        if d.name == "lo":
            continue
        speed = read(d / "speed")
        nics.append({"name": d.name, "state": read(d / "operstate"), "speed_mbps": speed, "mtu": read(d / "mtu"),
                     "driver": Path(os.path.realpath(d / "device" / "driver")).name if (d / "device").exists() else None})
    rdma = sorted(p.name for p in Path("/sys/class/infiniband").glob("*")) if Path("/sys/class/infiniband").is_dir() else []
    return {"interfaces": nics, "rdma_devices": rdma, "ip": sh("ip -br addr"),
            "ibv_devinfo": sh(["ibv_devinfo", "-l"]) if rdma else "(no RDMA device)"}


def disks(kv_home):
    paths = [p for p in dict.fromkeys([str(Path.home()), kv_home, "/tmp", "/dev/shm", str(ROOT)]) if p]
    return {"df": sh(["df", "-hP"] + [p for p in paths if os.path.exists(p)]), "quota": sh(["quota", "-s"]),
            "home": str(Path.home()), "kv_home": kv_home}


def software(kv_home):
    envs = {}
    for name in ("venv", "venv_vllm"):
        py = Path(kv_home) / name / "bin" / "python"
        if py.is_file():
            envs[name] = sh([str(py), "-c", "import importlib.util as u, sys\n"
                             "out = [sys.version.split()[0]]\n"
                             "for m in ('torch', 'triton', 'vllm', 'nixl', 'lmcache', 'nvidia.nvcomp', 'ibp'):\n"
                             "    if u.find_spec(m.split('.')[0]) is None: out.append(m + ' -'); continue\n"
                             "    try:\n"
                             "        x = __import__(m, fromlist=['_']); out.append(m + ' ' + str(getattr(x, '__version__', '?')))\n"
                             "    except Exception as e: out.append(m + ' (' + e.__class__.__name__ + ')')\n"
                             "print(', '.join(out))"], timeout=180)
        else:
            envs[name] = "(not set up)"
    return {"python3": sh(["python3", "--version"]), "python3.12": sh(["python3.12", "--version"]),
            "gcc": sh(["gcc", "--version"]).split("\n")[0], "nvcc": sh(["nvcc", "--version"]).split("\n")[-1],
            "cmake": sh(["cmake", "--version"]).split("\n")[0], "git": sh(["git", "--version"]),
            "modules": sh("module -t avail 2>&1 | head -n 80", timeout=60), "kv_home_envs": envs}


def slurm():
    job = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
                                         "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_JOB_QOS", "CUDA_VISIBLE_DEVICES",
                                         "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE")}
    user = os.environ.get("USER", "")
    host = platform.node().split(".")[0]
    return {"job": job,
            "node": sh(["scontrol", "show", "node", host]),
            "job_record": sh(["scontrol", "show", "job", job["SLURM_JOB_ID"] or "0"]),
            "associations": sh(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                                "format=account,partition,qos,grptres,maxtres,maxjobs,maxwall"]),
            "qos": sh(["sacctmgr", "-nP", "show", "qos",
                       "format=name,maxtresperuser,maxjobsperuser,maxwall,grptres,maxtrespu"]),
            "partition": sh(["scontrol", "show", "partition", job["SLURM_JOB_PARTITION"] or "all"]),
            "cgroup_devices": read("/sys/fs/cgroup/devices.list") if Path("/sys/fs/cgroup/devices.list").exists()
            else "(cgroup v2: device limits are a BPF program, not a file)"}


def cgroup_limits():
    """This process's cgroup (v2) memory and CPU limits: the job's, inside Slurm."""
    rel = next((l.split("::", 1)[1] for l in read("/proc/self/cgroup").splitlines() if l.startswith("0::")), None)
    if rel is None:
        return {"cgroup": "(not cgroup v2)"}
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    out = {"cgroup": rel}
    for d in [base] + list(base.parents):
        for k in ("memory.max", "cpu.max", "cpuset.cpus.effective"):
            if k not in out and (d / k).is_file():
                v = read(d / k)
                if v not in ("max", "max 100000"):
                    out[k] = v
        if d == Path("/sys/fs/cgroup"):
            break
    return out


def system(kv_home):
    rows, raw = gpus()
    for r in rows:
        r["numa_node"] = numa_of(r.get("pci.bus_id"))
    mem = {k: v for k, v in (line.split(":", 1) for line in read("/proc/meminfo").splitlines() if ":" in line)
           if k in ("MemTotal", "MemAvailable", "HugePages_Total", "Hugepagesize")}
    return {"host": platform.node(), "kernel": platform.release(), "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "slurm": slurm(),
            "cpu": sh(["lscpu"]), "numa": sh(["numactl", "-H"]), "affinity": sorted(os.sched_getaffinity(0)),
            "memory": {k: v.strip() for k, v in mem.items()}, "cgroup": cgroup_limits(),
            "gpus": rows, "gpus_raw": raw if not rows else None,
            "topology": sh(["nvidia-smi", "topo", "-m"]), "p2p_read": sh(["nvidia-smi", "topo", "-p2p", "r"]),
            "p2p_write": sh(["nvidia-smi", "topo", "-p2p", "w"]), "nvlink": sh(["nvidia-smi", "nvlink", "-s"]),
            "nvidia_smi_q": sh(["nvidia-smi", "-q"], timeout=120),
            "network": network(), "disks": disks(kv_home), "software": software(kv_home)}


# --------------------------------------------------------------------------------------------------- bandwidth
def bandwidth(reps, chunk_mib, layouts_only=False):
    """In a python with torch: pd_link_probe's legs and pairs, then the concurrent patterns."""
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import pd_link_probe as lp
    n = torch.cuda.device_count()
    res = {"gpus": [torch.cuda.get_device_name(i) for i in range(n)], "torch": torch.__version__,
           "cuda": torch.version.cuda, "mib_per_transfer": MIB, "chunk_mib": chunk_mib, "patterns": []}
    if not layouts_only:
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"link_probe_{os.getpid()}.json"
        lp.main(["--mib", str(MIB), "--reps", str(reps), "--output", str(tmp)])
        res["probe"] = json.loads(tmp.read_text())
        tmp.unlink()
    nbytes = MIB << 20
    chunk = chunk_mib << 20
    res["peer_access"] = [[(i != j and torch.cuda.can_device_access_peer(i, j)) for j in range(n)] for i in range(n)]

    def run(name, legs, kind):
        """legs: (src, dst) with a GPU index or "host"; kind direct (one copy per leg) or host (pinned staging in
        chunks, both copies overlapped). Every leg starts at once; aggregate and per-leg speed."""
        bufs = []
        for s, d in legs:
            src = (torch.empty(nbytes, dtype=torch.uint8, pin_memory=True).fill_(7) if s == "host" else
                   torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{s}").random_(0, 255))
            dst = (torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) if d == "host" else
                   torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{d}"))
            mid = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True) if kind == "host" else None
            dev_out = s if s != "host" else d
            dev_in = d if d != "host" else s
            s_out = torch.cuda.Stream(device=f"cuda:{dev_out}")
            s_in = torch.cuda.Stream(device=f"cuda:{dev_in}")
            bufs.append((src, dst, mid, s_out, s_in, dev_in))

        def go():
            ends = []
            for src, dst, mid, s_out, s_in, dev_in in bufs:
                if mid is None:
                    with torch.cuda.stream(s_out):
                        dst.copy_(src, non_blocking=True)
                    e = torch.cuda.Event(enable_timing=False)
                    e.record(s_out)
                    ends.append(e)
                else:
                    for o in range(0, nbytes, chunk):
                        with torch.cuda.stream(s_out):
                            mid[o:o + chunk].copy_(src[o:o + chunk], non_blocking=True)
                            e = torch.cuda.Event()
                            e.record(s_out)
                        s_in.wait_event(e)
                        with torch.cuda.stream(s_in):
                            dst[o:o + chunk].copy_(mid[o:o + chunk], non_blocking=True)
                    e = torch.cuda.Event()
                    e.record(s_in)
                    ends.append(e)
            return ends

        def once():
            lp.sync()
            t0 = time.perf_counter()
            ends = go()
            done = []
            for e in ends:
                e.synchronize()
                done.append(time.perf_counter() - t0)
            return time.perf_counter() - t0, done
        once()
        best, legs_t = float("inf"), None
        for _ in range(reps):
            t, d = once()
            if t < best:
                best, legs_t = t, d
        ok = all(bool(torch.equal(b[1][:1 << 20].cpu(), b[0][:1 << 20].cpu())) for b in bufs)
        row = {"pattern": name, "kind": kind, "legs": [f"{s}->{d}" for s, d in legs], "correct": ok,
               "seconds": best, "aggregate_GB_per_s": len(legs) * nbytes / best / 1e9,
               "per_leg_GB_per_s": [nbytes / t / 1e9 for t in legs_t]}
        res["patterns"].append(row)
        print(f"  {name:28s} {kind:6s} {' '.join(row['legs']):24s} aggregate {row['aggregate_GB_per_s']:6.1f} GB/s, "
              f"per leg " + " ".join(f"{g:5.1f}" for g in row["per_leg_GB_per_s"]) + ("" if ok else "  NOT CORRECT"),
              flush=True)
        bufs.clear()
        torch.cuda.empty_cache()

    print(f"\nconcurrent patterns, {MIB} MiB per leg, best of {reps}:")
    if n >= 2:
        for p in range(n):
            others = [g for g in range(n) if g != p]
            for kind in ("direct", "host"):
                run(f"1p{len(others)}d, prefill GPU {p}", [(p, d) for d in others], kind)
    if n >= 4:
        seen = set()
        for a in range(4):
            for b in range(a + 1, 4):
                rest = [g for g in range(4) if g not in (a, b)]
                for d0, d1 in ((rest[0], rest[1]), (rest[1], rest[0])):
                    key = frozenset({(a, d0), (b, d1)})
                    if key in seen:
                        continue
                    seen.add(key)
                    for kind in ("direct", "host"):
                        run(f"2p2d, prefill {a},{b}", [(a, d0), (b, d1)], kind)
    for k in range(1, n + 1):
        run(f"cache reload to {k} GPU(s)", [("host", g) for g in range(k)], "direct")
    run(f"cache store from {n} GPU(s)", [(g, "host") for g in range(n)], "direct")
    return res


# ----------------------------------------------------------------------------------------------------- summary
def table(rows, cols):
    out = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + " --- |" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(f(r)) for _, f in cols) + " |")
    return out


def summary(spec):
    s, b = spec["system"], spec.get("bandwidth") or {}
    j = s["slurm"]["job"]
    out = [f"# Node specs: {s['host']}", "", f"{s['time']}; job {j.get('SLURM_JOB_ID')} ({j.get('SLURM_JOB_PARTITION')} / "
           f"{j.get('SLURM_JOB_ACCOUNT')} / {j.get('SLURM_JOB_QOS')}), GPUs {j.get('CUDA_VISIBLE_DEVICES')}, "
           f"{len(s['affinity'])} CPUs, job memory limit {s['cgroup'].get('memory.max', 'none found')} bytes, kernel "
           f"{s['kernel']}.", "", "## GPUs", ""]
    if s["gpus"]:
        out += table(s["gpus"], [("GPU", lambda r: r["index"]), ("name", lambda r: r["name"]),
                                 ("memory MiB", lambda r: r["memory.total"]), ("bus", lambda r: r["pci.bus_id"]),
                                 ("NUMA", lambda r: r["numa_node"]),
                                 ("PCIe now / max", lambda r: f"Gen{r['pcie.link.gen.current']} x{r['pcie.link.width.current']}"
                                  f" / Gen{r['pcie.link.gen.max']} x{r['pcie.link.width.max']}"),
                                 ("power W", lambda r: r["power.limit"]), ("ECC", lambda r: r["ecc.mode.current"]),
                                 ("driver", lambda r: r["driver_version"])])
    else:
        out.append(f"nvidia-smi: {s['gpus_raw']}")
    out += ["", "PCIe 'now' is read while the GPUs idle; they drop the link speed to save power, so 'max' is what a "
            "transfer gets.", "", "```", s["topology"], "", s["p2p_read"], "```", ""]
    probe = b.get("probe")
    if probe:
        out += ["## Bandwidth, one transfer at a time (896 MiB)", ""]
        out += table(probe["host_legs"], [("GPU", lambda r: r["gpu"]),
                                          ("to pinned", lambda r: f"{r['d2h_pinned']['GB_per_s']:.1f} GB/s"),
                                          ("from pinned", lambda r: f"{r['h2d_pinned']['GB_per_s']:.1f} GB/s"),
                                          ("to pageable", lambda r: f"{r['d2h_pageable']['GB_per_s']:.1f} GB/s"),
                                          ("from pageable", lambda r: f"{r['h2d_pageable']['GB_per_s']:.1f} GB/s")])
        out.append("")
        out += table(probe["pairs"], [("pair", lambda r: f"{r['src']} -> {r['dst']}"), ("peer", lambda r: r["peer_access"]),
                                      ("direct", lambda r: f"{r['direct']['GB_per_s']:.1f} GB/s"),
                                      ("through host", lambda r: f"{r['host']['GB_per_s']:.1f} GB/s"),
                                      ("through host, pipelined", lambda r: f"{r['host_pipe']['GB_per_s']:.1f} GB/s")])
        out.append("")
    if b.get("patterns"):
        out += ["## Bandwidth, the planned layouts (every leg at once, 896 MiB each)", ""]
        out += table(b["patterns"], [("pattern", lambda r: r["pattern"]), ("path", lambda r: r["kind"]),
                                     ("legs", lambda r: ", ".join(r["legs"])),
                                     ("aggregate", lambda r: f"{r['aggregate_GB_per_s']:.1f} GB/s "
                                      f"({8 * r['aggregate_GB_per_s']:.0f} Gb/s)"),
                                     ("per leg GB/s", lambda r: " / ".join(f"{g:.1f}" for g in r["per_leg_GB_per_s"])),
                                     ("correct", lambda r: "yes" if r["correct"] else "NO")])
        out.append("")
        best = {}
        for r in b["patterns"]:
            if r["pattern"].startswith(("1p", "2p")):
                k = (r["pattern"][:4], r["kind"])
                if k not in best or r["aggregate_GB_per_s"] > best[k]["aggregate_GB_per_s"]:
                    best[k] = r
        if best:
            out += ["Best placement per layout and path:", ""]
            out += [f"- {k[0]} {k[1]}: {r['pattern']} ({', '.join(r['legs'])}), {r['aggregate_GB_per_s']:.1f} GB/s "
                    f"aggregate" for k, r in sorted(best.items())]
            out.append("")
    elif spec.get("bandwidth_error"):
        out += ["## Bandwidth", "", f"Not measured: {spec['bandwidth_error']}", ""]
    net = s["network"]
    out += ["## CPU, memory, disk, network", "",
            "```", "\n".join(l for l in s["cpu"].splitlines() if l.split(":")[0].strip() in
                             ("Model name", "Socket(s)", "Core(s) per socket", "Thread(s) per core", "NUMA node(s)",
                              "CPU(s)", "L3 cache")), "```", "",
            f"Memory: {s['memory']}", "", "```", s["disks"]["df"], "", "quota: " + s["disks"]["quota"], "```", ""]
    out += table(net["interfaces"], [("interface", lambda r: r["name"]), ("state", lambda r: r["state"]),
                                     ("speed Mb/s", lambda r: r["speed_mbps"]), ("MTU", lambda r: r["mtu"]),
                                     ("driver", lambda r: r["driver"])])
    out += ["", f"RDMA devices: {', '.join(net['rdma_devices']) or 'none'}", "", "## Slurm limits", "", "```",
            "associations (account|partition|qos|grptres|maxtres|maxjobs|maxwall):", s["slurm"]["associations"], "",
            "qos (name|maxtresperuser|maxjobsperuser|maxwall|grptres|maxtrespu):", s["slurm"]["qos"], "```", "",
            "## Software", ""]
    sw = s["software"]
    out += [f"- {k}: {sw[k]}" for k in ("python3", "python3.12", "gcc", "nvcc", "cmake", "git")]
    out += [f"- KV_HOME {k}: {v}" for k, v in sw["kv_home_envs"].items()]
    out.append("")
    return "\n".join(out)


def find_python(kv_home):
    for p in (Path(kv_home) / "venv" / "bin" / "python", Path(kv_home) / "venv_vllm" / "bin" / "python",
              Path.home() / "venvs" / "vllm-stock" / "bin" / "python", Path.home() / "vllm-env" / "bin" / "python"):
        if p.is_file() and subprocess.run([str(p), "-c", "import torch"], capture_output=True).returncode == 0:
            return str(p)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=False)
    ap.add_argument("--python", default=None, help="a python with torch for the bandwidth part (default: found)")
    ap.add_argument("--kv-home", default=os.environ.get("KV_HOME", str(Path.home() / "kv_a5000")))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--chunk-mib", type=int, default=32)
    ap.add_argument("--skip-bandwidth", action="store_true")
    ap.add_argument("--bandwidth-only", default=None, help=argparse.SUPPRESS)   # the child: write its JSON here
    ap.add_argument("--layouts-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.bandwidth_only:
        Path(a.bandwidth_only).write_text(json.dumps(bandwidth(a.reps, a.chunk_mib, a.layouts_only), indent=1))
        return 0
    if not a.output:
        ap.error("--output is required")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    print("== system", flush=True)
    spec = {"system": system(a.kv_home)}
    if not a.skip_bandwidth:
        py = a.python or find_python(a.kv_home)
        if py is None:
            spec["bandwidth_error"] = ("no python with torch (KV_HOME/venv, KV_HOME/venv_vllm, ~/venvs/vllm-stock): run "
                                       "scripts/a5000_setup.sh first, or pass --python")
            print("== bandwidth: " + spec["bandwidth_error"], flush=True)
        else:
            print(f"== bandwidth with {py}", flush=True)
            tmp = out / "bandwidth.json"
            r = subprocess.run([py, "-u", __file__, "--bandwidth-only", str(tmp), "--reps", str(a.reps),
                                "--chunk-mib", str(a.chunk_mib)] + (["--layouts-only"] if a.layouts_only else []))
            if r.returncode == 0 and tmp.is_file():
                spec["bandwidth"] = json.loads(tmp.read_text())
                spec["bandwidth"]["python"] = py
                tmp.unlink()
            else:
                spec["bandwidth_error"] = f"the measurement exited with {r.returncode} (see the job's output)"
    (out / "specs.json").write_text(json.dumps(spec, indent=1))
    (out / "summary.md").write_text(summary(spec))
    print(f"wrote {out}/specs.json and summary.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
