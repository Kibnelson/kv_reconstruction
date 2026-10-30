"""Paced transport of the PD protocol: a virtual link of a set speed (3-400 Gb/s) carrying transfers from a prefill
engine to decode engines, over a same-node shared-memory ring (the default on one machine) or TCP.

Link      one FIFO link per sender process (the prefill engine's egress, shared by all its destinations). A chunk
          of n bytes is booked from max(now, link busy until) for 8 n / rate seconds; the sender releases the chunk
          to the receiver at the end of its booking, so the receiver sees the bytes no earlier than a real link of
          that speed would deliver them. The speed can change between benchmark points: a JSON control file
          ({"gbps": x}) is re-read when it changes. When the machine itself is slower than the link (the copies
          below), transfers take the machine's time; every record keeps both, so the achieved rate is reported.
ring      the receiver owns a shared-memory ring (/dev/shm), registered with CUDA as pinned memory on both sides:
          the sender copies GPU -> ring (one DMA), the receiver ring -> GPU (one DMA), chunk by chunk, overlapped -
          about what GPUDirect over a PCIe NIC does, bounded by the same PCIe links.
tcp       the bytes travel on the control socket (any two machines; host copies on both sides).

Messages are a 4-byte length and JSON; on tcp, a data message is followed by its bytes. A transfer is
begin (key, header, total bytes) - data (offset, n, ring position) ... - end; the receiver acknowledges ring space,
in order (a "skip" of the ring's tail before a wrap is acknowledged like data).
"""
from __future__ import annotations

import json
import mmap
import os
import queue
import select
import socket
import struct
import threading
import time

import torch

CHUNK = 32 << 20


def _send_msg(sock, obj, data=None):
    b = json.dumps(obj, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(b)) + b)
    if data is not None:
        sock.sendall(data)


def _recv_exact(sock, n, into=None):
    buf = bytearray(n) if into is None else into
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed")
        got += k
    return buf


def _recv_msg(sock):
    (n,) = struct.unpack(">I", bytes(_recv_exact(sock, 4)))
    return json.loads(bytes(_recv_exact(sock, n)).decode())


def _register(t):
    """Pin a host tensor in place for CUDA DMA (False when CUDA or the call is unavailable)."""
    if not torch.cuda.is_available():
        return False
    try:
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        return int(rc) == 0
    except Exception:
        return False


class Link:
    """A FIFO virtual link; gbps fixed, or read from the control file (JSON {"gbps": x}) when it changes."""

    def __init__(self, gbps=None, control=None):
        self.fixed = gbps
        self.control = control
        self._mtime = None
        self._gbps = gbps
        self.busy_until = 0.0
        self.lock = threading.Lock()

    def gbps(self):
        if self.control:
            try:
                m = os.stat(self.control).st_mtime_ns
                if m != self._mtime:
                    with open(self.control) as f:
                        g = json.loads(f.read()).get("gbps")
                    self._gbps = float(g) if g else None    # null or 0: no pacing
                    self._mtime = m
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
        return self._gbps

    def reserve(self, nbytes, now=None):
        """(start, end) of n bytes on the link (time.monotonic seconds); no pacing when the speed is unset."""
        g = self.gbps()
        with self.lock:
            now = time.monotonic() if now is None else now
            if not g:
                return now, now
            start = max(now, self.busy_until)
            end = start + 8.0 * nbytes / (g * 1e9)
            self.busy_until = end
            return start, end


def _sleep_until(t):
    while True:
        d = t - time.monotonic()
        if d <= 0:
            return
        time.sleep(min(d, 0.05))


# ------------------------------------------------------------------------------------------------------- receiver
class Receiver:
    """Listens for senders; completed transfers wait in a table until the decode worker takes them.

    device: where blobs land (the decode GPU); ring_bytes: the shared-memory ring per sender (0: tcp only)."""

    def __init__(self, host, port, device, ring_bytes=1 << 30, log=print, gpu_budget=None):
        self.device = torch.device(device)
        self.ring_bytes = int(ring_bytes)
        self.log = log
        self.done = {}                               # key -> (header, blob, record)
        self.dropped = set()                         # keys to discard on arrival
        self.cv = threading.Condition()
        self.gpu_budget = gpu_budget                 # bytes of GPU blobs at once before falling back to host
        self.gpu_bytes = 0
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, int(port)))
        self.srv.listen(16)
        self.port = self.srv.getsockname()[1]
        self.stop = False
        self.copy_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        threading.Thread(target=self._accept, daemon=True, name="tkv-recv-accept").start()

    def _accept(self):
        n = 0
        while not self.stop:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            n += 1
            threading.Thread(target=self._serve, args=(conn, n), daemon=True, name=f"tkv-recv-{n}").start()

    def _make_ring(self, tag):
        path = f"/dev/shm/tkv_ring_{os.getpid()}_{self.port}_{tag}"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(fd, self.ring_bytes)
        mm = mmap.mmap(fd, self.ring_bytes)
        os.close(fd)
        t = torch.frombuffer(mm, dtype=torch.uint8)
        pinned = _register(t)
        return path, mm, t, pinned

    def _alloc(self, total):
        if self.device.type == "cuda":
            if self.gpu_budget is None or self.gpu_bytes + total <= self.gpu_budget:
                try:
                    b = torch.empty(total, dtype=torch.uint8, device=self.device)
                    with self.cv:
                        self.gpu_bytes += total
                    return b
                except torch.cuda.OutOfMemoryError:
                    pass
        return torch.empty(total, dtype=torch.uint8)

    def release(self, blob):
        if blob.is_cuda:
            with self.cv:
                self.gpu_bytes -= blob.numel()

    def _serve(self, conn, tag):
        ring = None
        try:
            hello = _recv_msg(conn)
            mode = hello.get("mode", "tcp")
            if mode == "ring":
                path, mm, rt, pinned = self._make_ring(tag)
                ring = (path, mm, rt)
                _send_msg(conn, {"op": "ring", "path": path, "bytes": self.ring_bytes, "pinned": pinned})
            else:
                _send_msg(conn, {"op": "tcp"})
            stage = None
            cur = {}
            while True:
                m = _recv_msg(conn)
                op = m["op"]
                if op == "begin":
                    rec = {"key": m["key"], "bytes": m["total"], "recv_begin": time.time(), "mode": mode}
                    cur[m["key"]] = (m["header"], self._alloc(m["total"]), rec)
                elif op == "data":
                    header, blob, rec = cur[m["key"]]
                    off, n = m["off"], m["n"]
                    if mode == "ring":
                        src = ring[2][m["pos"]:m["pos"] + n]
                    else:
                        if stage is None or stage.numel() < n:
                            stage = torch.empty(max(n, CHUNK), dtype=torch.uint8)
                            _register(stage)
                        _recv_exact(conn, n, into=memoryview(stage.numpy())[:n])
                        src = stage[:n]
                    if self.copy_stream is not None and blob.is_cuda:
                        with torch.cuda.stream(self.copy_stream):
                            blob[off:off + n].copy_(src, non_blocking=True)
                        self.copy_stream.synchronize()
                    else:
                        blob[off:off + n].copy_(src)
                    if mode == "ring":
                        _send_msg(conn, {"op": "ack", "pos": m["pos"], "n": n})
                elif op == "skip":
                    _send_msg(conn, {"op": "ack", "n": m["n"]})
                elif op == "end":
                    header, blob, rec = cur.pop(m["key"])
                    rec.update(recv_end=time.time(), **{k: m[k] for k in m if k not in ("op", "key")})
                    with self.cv:
                        if m["key"] in self.dropped:
                            self.dropped.discard(m["key"])
                            self.release(blob)
                        else:
                            self.done[m["key"]] = (header, blob, rec)
                        self.cv.notify_all()
                elif op == "bye":
                    return
        except (ConnectionError, OSError) as exc:
            if not self.stop:
                self.log(f"tkv receiver: connection {tag} closed ({exc!r})")
        finally:
            conn.close()
            if ring is not None:
                try:
                    os.unlink(ring[0])
                except OSError:
                    pass

    def take(self, key):
        with self.cv:
            return self.done.pop(key, None)

    def wait(self, key, timeout):
        end = time.monotonic() + timeout
        with self.cv:
            while key not in self.done:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self.cv.wait(left)
            return self.done.pop(key)

    def expire(self, older_than_s, keep=()):
        """Release transfers nobody claimed (their request ended before it was admitted); returns their keys."""
        now = time.time()
        gone = []
        with self.cv:
            for k, (h, b, rec) in list(self.done.items()):
                if k not in keep and now - rec.get("recv_end", now) > older_than_s:
                    del self.done[k]
                    self.release(b)
                    gone.append(k)
        return gone

    def drop(self, key):
        """Discard a transfer (its request is gone): now if it arrived, else when it does."""
        with self.cv:
            got = self.done.pop(key, None)
            if got is None:
                self.dropped.add(key)
            else:
                self.release(got[1])

    def close(self):
        self.stop = True
        try:
            self.srv.close()
        except OSError:
            pass


# --------------------------------------------------------------------------------------------------------- sender
class _Dest:
    def __init__(self, addr, mode, device, log):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_msg(self.sock, {"op": "hello", "mode": mode})
        r = _recv_msg(self.sock)
        self.mode = r["op"]
        self.ring = None
        if self.mode == "ring":
            fd = os.open(r["path"], os.O_RDWR)
            self.mm = mmap.mmap(fd, r["bytes"])
            os.close(fd)
            self.ring = torch.frombuffer(self.mm, dtype=torch.uint8)
            self.pinned = _register(self.ring)
            self.size = r["bytes"]
            self.head = 0                            # bytes written into the ring in all
            self.acked = 0                           # bytes the receiver has consumed
        self.queue = queue.Queue()
        self.stream = torch.cuda.Stream(device) if torch.device(device).type == "cuda" else None


class Sender:
    """Sends transfers to receivers over the paced link; one thread per destination, one link for all.

    send(dest, key, header, blob, on_done) queues a transfer (blob: uint8 tensor on the GPU or the host); on_done
    (record) runs on the sender thread when the last byte is released."""

    def __init__(self, link, device, mode="ring", log=print):
        self.link = link
        self.device = torch.device(device)
        self.mode = mode
        self.log = log
        self.dests = {}
        self.lock = threading.Lock()

    def _dest(self, addr):
        with self.lock:
            d = self.dests.get(addr)
            if d is None:
                d = _Dest(addr, self.mode, self.device, self.log)
                self.dests[addr] = d
                threading.Thread(target=self._run, args=(d, addr), daemon=True, name=f"tkv-send-{addr}").start()
            return d

    def send(self, dest, key, header, blob, on_done=None, info=None):
        try:
            d = self._dest(dest)
        except OSError as exc:                       # the receiver is not there: report, do not stop the engine
            self.log(f"tkv sender: cannot reach {dest} ({exc!r})")
            if on_done is not None:
                on_done(dict(info or {}, key=key, dest=dest, error=f"connect: {exc!r}"[:300]))
            return
        d.queue.put((key, header, blob, on_done, dict(info or {}), time.time()))

    def _ring_space(self, d, n):
        """Wait until n more bytes fit in the ring; acknowledgements that already arrived are read on the way (they
        must not pile up in the socket while the ring has room)."""
        while True:
            while select.select([d.sock], [], [], 0)[0]:
                m = _recv_msg(d.sock)
                if m.get("op") == "ack":
                    d.acked += m["n"]
            if d.head + n - d.acked <= d.size:
                return
            m = _recv_msg(d.sock)
            if m.get("op") == "ack":
                d.acked += m["n"]

    def _run(self, d, addr):
        while True:
            item = d.queue.get()
            if item is None:
                return
            key, header, blob, on_done, info, queued = item
            total = int(blob.numel())
            rec = dict(info, key=key, dest=addr, bytes=total, queued=queued, send_begin=time.time(), mode=d.mode)
            try:
                rec["gbps_set"] = self.link.gbps()
                _send_msg(d.sock, {"op": "begin", "key": key, "header": header, "total": total})
                first = last = None
                for off in range(0, total, CHUNK):
                    n = min(CHUNK, total - off)
                    src = blob[off:off + n]
                    # the link time first, then the copy: a copy slower than the link stretches the chunk to the
                    # copy's time (the machine's limit), not to the sum of both
                    t0, t1 = self.link.reserve(n)
                    first = t0 if first is None else first
                    if d.mode == "ring":
                        if d.head % d.size + n > d.size:    # no wrap inside a chunk: skip to the ring start
                            pad = d.size - d.head % d.size   # the skipped bytes count as sent until the receiver
                            self._ring_space(d, pad)          # acknowledges them in order (after all earlier data)
                            d.head += pad
                            _send_msg(d.sock, {"op": "skip", "n": pad})
                        self._ring_space(d, n)
                        pos = d.head % d.size
                        dst = d.ring[pos:pos + n]
                        if src.is_cuda and d.stream is not None:
                            with torch.cuda.stream(d.stream):
                                dst.copy_(src, non_blocking=True)
                            d.stream.synchronize()
                        else:
                            dst.copy_(src)
                        d.head += n
                    _sleep_until(t1)
                    last = t1
                    if d.mode == "ring":
                        _send_msg(d.sock, {"op": "data", "key": key, "off": off, "n": n, "pos": pos})
                    else:
                        host = src.cpu() if src.is_cuda else src
                        _send_msg(d.sock, {"op": "data", "key": key, "off": off, "n": n},
                                  memoryview(host.contiguous().numpy()))
                _send_msg(d.sock, {"op": "end", "key": key, "link_s": (last - first) if first else 0.0})
                rec.update(send_end=time.time(), link_s=(last - first) if first is not None else 0.0)
            except Exception as exc:
                rec.update(error=repr(exc)[:300])
                self.log(f"tkv sender: transfer {key} to {addr} failed ({exc!r})")
            if on_done is not None:
                try:
                    on_done(rec)
                except Exception as exc:
                    self.log(f"tkv sender: on_done for {key} failed ({exc!r})")

    def close(self):
        with self.lock:
            for d in self.dests.values():
                d.queue.put(None)
                try:
                    _send_msg(d.sock, {"op": "bye"})
                    d.sock.close()
                except OSError:
                    pass
