"""kvtrace: the trace of KV blocks' lives, written by hooks in our copies of vLLM and LMCache (third_party/).

A hook is one whole added line ending in "# KVTRACE":

    if kvtrace.ON: kvtrace.hook("vllm.gpu.cached", request=request, blocks=new_full_blocks, ...)  # KVTRACE

It passes the objects at hand; the extractor for that event (kvtrace/events.py) turns them into a small record,
here, so the hooks stay one line and the copies stay readable. Every record gets:

    ev    the event name (docs/lmcache_lifecycle.md lists them all, with where their hook is)
    t     wall-clock time (time.time(), seconds; one clock for every process on the node)
    role  the engine or server (KVTRACE_ROLE: P, D0, D1, D2, LMC)
    pid   the process

Records go to KVTRACE_DIR/kvtrace_<role>_<pid>.jsonl, buffered and flushed every KVTRACE_FLUSH_S seconds (1) or
FLUSH_N records, and at exit. Tracing is on when KVTRACE_DIR is set; otherwise ON is False and every hook is a
single attribute test. A hook never raises: an extractor's error is counted (written as "kvtrace.error" records,
at most ERR_MAX per event name) and the engine goes on.
"""
from __future__ import annotations

import atexit
import json
import os
import threading
import time

ON: bool = bool(os.environ.get("KVTRACE_DIR"))
# LMCache events built only when someone listens: the event bus's hook (has_subscribers) says someone does
LMC_WANTED = frozenset({"mp.lookup", "mp.tokens", "l1.keys.accessed"})
ROLE: str = os.environ.get("KVTRACE_ROLE", "?")
FLUSH_N = 2000
ERR_MAX = 20

_lock = threading.Lock()
_buf: list[str] = []
_fh = None
_pid = None
_last_flush = 0.0
_errors: dict[str, int] = {}
_flush_s = float(os.environ.get("KVTRACE_FLUSH_S", "1") or 1)


def _open():
    """The file of this process (reopened after a fork: a child writes its own)."""
    global _fh, _pid, _buf
    pid = os.getpid()
    if _fh is not None and _pid == pid:
        return _fh
    if _pid is not None and _pid != pid:
        _buf = []                                    # the parent's unwritten records are the parent's
    d = os.environ.get("KVTRACE_DIR")
    os.makedirs(d, exist_ok=True)
    _fh = open(os.path.join(d, f"kvtrace_{ROLE}_{pid}.jsonl"), "a", buffering=1 << 20)
    _pid = pid
    return _fh


_flusher_pid = None


def _flusher():
    while True:
        time.sleep(_flush_s)
        try:
            flush()
        except Exception:
            pass


def _start_flusher():
    """A daemon thread per process writes what is buffered every KVTRACE_FLUSH_S, so a process stopped without
    running its exit handlers loses at most that much."""
    global _flusher_pid
    if _flusher_pid != os.getpid():
        _flusher_pid = os.getpid()
        threading.Thread(target=_flusher, name="kvtrace-flush", daemon=True).start()


def emit(rec: dict) -> None:
    """Write one record (already a dict of JSON-able values)."""
    global _last_flush
    _start_flusher()
    rec.setdefault("t", time.time())
    rec["role"] = ROLE
    rec["pid"] = os.getpid()
    line = json.dumps(rec, separators=(",", ":"), default=_default)
    with _lock:
        _open()
        _buf.append(line)
        now = rec["t"]
        if len(_buf) >= FLUSH_N or now - _last_flush >= _flush_s:
            _flush_locked()
            _last_flush = now


def hook(name: str, /, **objs) -> None:
    """Called by a hook line: build the event's record from the objects passed and write it. (name is
    positional-only: a hook may pass an object called name.)"""
    t = time.time()
    try:
        from kvtrace import events
        fn = events.EXTRACT.get(name)
        rec = fn(**objs) if fn is not None else events.generic(**objs)
        if rec is None:
            return
        rec["ev"] = rec.pop("_ev", name)
        if "_t" in rec:                              # the event's own time (e.g. the GPU's), and the hook's
            rec["t"], rec["t_hook"] = rec.pop("_t"), t
        else:
            rec["t"] = t
        emit(rec)
    except Exception as exc:                         # a hook never breaks the engine
        n = _errors.get(name, 0) + 1
        _errors[name] = n
        if n <= ERR_MAX:
            try:
                emit({"ev": "kvtrace.error", "t": t, "of": name, "error": repr(exc)[:300], "n": n})
            except Exception:
                pass


def _default(o):
    if isinstance(o, (bytes, bytearray, memoryview)):
        return bytes(o)[:8].hex()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return repr(o)[:200]


def _flush_locked():
    global _buf
    if not _buf:
        return
    fh = _open()
    fh.write("\n".join(_buf) + "\n")
    fh.flush()
    _buf = []


def flush() -> None:
    with _lock:
        if _fh is not None or _buf:
            _flush_locked()


if ON:
    atexit.register(flush)
