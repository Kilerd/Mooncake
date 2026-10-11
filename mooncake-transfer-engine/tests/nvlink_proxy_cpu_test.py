#!/usr/bin/env python3
# Copyright 2026 KVCache.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mooncake_nvlink_proxy without GPUs.

Runs the daemon built against a CPU stand-in for the CUDA runtime
(nvlink_proxy_mock_cudart.cpp: "GPU memory" is POSIX shared memory, streams
are worker threads) and plays engines over its socket:

  p2p_deadline       a slow P2P copy stops before its hard deadline; the
                     refusal leaves before the engine gives up, no byte lands
                     after it
  slow_link          a ~100 MB/s link at the default chunk size: the probe
                     measures it and the copy is refused in time; once
                     measured, a copy that cannot finish is not started
  p2p_exact          chunked P2P copies (large runs split, kernel rows) are
                     byte-exact, gather kernel on either GPU
  staged             host-staged copies, also stopped at their deadline
  socket             the socket is owner-only by default, --socket-mode
  disconnect         an engine's IPC mappings are closed when it disconnects,
                     or when the last copy in flight that uses them finishes
  restart            a restarted engine (new client id, same addresses, even
                     byte-identical handles) is never served from a stale
                     mapping; a re-registering engine shares its mapping
  share              two engines of one process share a mapping while one is
                     connected; another process is refused
  gpus               --min-visible-gpus, REGISTER of a GPU the daemon cannot
                     see, a client pid equal to the daemon's

Usage: nvlink_proxy_cpu_test.py <daemon binary> [scenario ...]
Linux only (POSIX shared memory, /proc/<pid>/maps).
"""

import mmap
import os
import random
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

MAGIC, VERSION = 0x504E434D, 1
HELLO, REGISTER, UNREGISTER, COPY, PING, STATS = 1, 2, 3, 4, 5, 6
OK, UNKNOWN_DEVICE, UNKNOWN_SOURCE = 0, 3, 4
NO_PEER_ACCESS, CUDA_ERROR, DEADLINE_EXCEEDED = 7, 8, 9
MiB = 1 << 20

DAEMON = None
TMP = None
FAILURES = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        FAILURES.append(what)
    return cond


# ------------------------------------------------------------------ engines
class Conn:
    _seq = 0
    _lock = threading.Lock()

    def __init__(self, path, client, control, pid=None):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        st, _ = self.rpc(
            HELLO,
            struct.pack(
                "<QII", client, os.getpid() if pid is None else pid, 1 if control else 0
            )
            + b"cpu-test".ljust(64, b"\0"),
        )
        assert st == OK, st

    def rpc(self, typ, payload=b""):
        with Conn._lock:
            Conn._seq += 1
            seq = Conn._seq
        self.s.sendall(
            struct.pack("<IHHIIQ", MAGIC, VERSION, typ, 0, len(payload), seq) + payload
        )
        h = b""
        while len(h) < 24:
            chunk = self.s.recv(24 - len(h))
            if not chunk:
                raise ConnectionError("daemon closed the connection")
            h += chunk
        _, _, _, status, plen, _ = struct.unpack("<IHHIIQ", h)
        body = b""
        while len(body) < plen:
            body += self.s.recv(plen - len(body))
        return status, body

    def close(self):
        self.s.close()


class Segment:
    """POSIX shm standing in for one engine's cudaMalloc block."""

    def __init__(self, name, size, fill_seed=None):
        self.name = name
        self.size = size
        path = "/dev/shm/" + name
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.ftruncate(fd, size)
        self.m = mmap.mmap(fd, size)
        os.close(fd)
        if fill_seed is not None:
            self.m[:] = random.Random(fill_seed).randbytes(size)

    def handle(self, gpu, pid=0):
        h = b"MOCKSHM\0" + struct.pack("<i", gpu)
        h += ("/" + self.name).encode().ljust(44, b"\0")
        return h + struct.pack("<I", pid) + b"\0" * 4

    def unlink(self):
        try:
            os.unlink("/dev/shm/" + self.name)
        except FileNotFoundError:
            pass


def unlink_segments(*names):
    for n in names:
        try:
            os.unlink("/dev/shm/" + n)
        except FileNotFoundError:
            pass


def uuid(gpu):
    return bytes((gpu * 16 + i) & 0xFF for i in range(16))


def register(conn, base, seg, gpu, pid=0):
    st, body = conn.rpc(
        REGISTER, struct.pack("<QQ", base, seg.size) + uuid(gpu) + seg.handle(gpu, pid)
    )
    return st, body.decode(errors="replace")


def copy(path, entries, budget_ms=0, client=0xC0):
    """entries: (src client, src base, src off, dst client, dst base,
    dst off, length). Returns (status, message, seconds from send)."""
    c = Conn(path, client, False)
    payload = struct.pack("<II", len(entries), budget_ms)
    for e in entries:
        payload += struct.pack("<7Q", *e)
    t0 = time.monotonic()
    st, body = c.rpc(COPY, payload)
    t = time.monotonic() - t0
    c.close()
    return st, body[16:].decode(errors="replace"), t0, t


def stats(path):
    c = Conn(path, 0x5747, False)
    _, body = c.rpc(STATS)
    c.close()
    out = {}
    for kv in body.decode().split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k] = v.split("(")[0]
    return out


# ------------------------------------------------------------------- daemon
class Daemon:
    def __init__(self, args=(), env=None, wait=True):
        self.sock = os.path.join(TMP, "proxy.sock")
        self.log = os.path.join(TMP, "daemon.log")
        self.writes = os.path.join(TMP, "writes.log")
        for f in (self.sock, self.writes):
            if os.path.exists(f):
                os.unlink(f)
        e = dict(os.environ)
        e.update({"MOCK_GPUS": "2", "MOCK_WRITE_LOG": self.writes})
        e.update(env or {})
        self.proc = subprocess.Popen(
            [
                DAEMON,
                "--socket",
                self.sock,
                "--stats-interval",
                "0",
                "--no-startup-selftest",
            ]
            + list(args),
            stdout=open(self.log, "w"),
            stderr=subprocess.STDOUT,
            env=e,
        )
        if wait:
            for _ in range(200):
                if os.path.exists(self.sock):
                    break
                if self.proc.poll() is not None:
                    break
                time.sleep(0.02)

    def text(self):
        with open(self.log) as f:
            return f.read()

    def wait_log(self, needle, timeout=15):
        end = time.time() + timeout
        while time.time() < end:
            if needle in self.text():
                return True
            time.sleep(0.05)
        return False

    def maps(self):
        with open(f"/proc/{self.proc.pid}/maps") as f:
            return f.read()

    def last_write_us(self, segment=None):
        if not os.path.exists(self.writes):
            return None
        last = None
        with open(self.writes) as f:
            for line in f:
                t, seg, _, _ = line.split()
                if segment is None or seg.endswith(segment):
                    last = int(t)
        return last

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def pair(
    d,
    src_seg,
    dst_seg,
    src_gpu=0,
    dst_gpu=1,
    a=0xA,
    b=0xB,
    sbase=0x10000000,
    dbase=0x20000000,
):
    """Two engines: |a| registers src_seg on src_gpu, |b| dst_seg."""
    ca = Conn(d.sock, a, True)
    cb = Conn(d.sock, b, True)
    assert register(ca, sbase, src_seg, src_gpu)[0] == OK
    assert register(cb, dbase, dst_seg, dst_gpu)[0] == OK
    return ca, cb


def runs(
    n, size, a=0xA, b=0xB, sbase=0x10000000, dbase=0x20000000, src_off=0, dst_off=0
):
    return [
        (a, sbase, src_off + k * size, b, dbase, dst_off + k * size, size)
        for k in range(n)
    ]


# ---------------------------------------------------------------- scenarios
def p2p_deadline():
    print("== p2p_deadline: slow P2P copy, 300 ms start budget")
    d = Daemon(["--copy-chunk-mb", "1"], {"MOCK_COPY_DELAY_MS": "10"})
    try:
        src = Segment("nvlp-src", 160 * MiB, 1)
        dst = Segment("nvlp-dst", 160 * MiB)
        ca, cb = pair(d, src, dst)
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        # 128 x 1 MiB, 10 ms each: 1.3 s of work. The engine gives up after
        # 2 x 300 ms; the daemon's hard deadline is 550 ms after receipt.
        ents = runs(128, MiB)
        st, msg, t0, t = copy(d.sock, ents, budget_ms=300)
        reply_us = int((t0 + t) * 1e6)
        give_up_us = int((t0 + 0.6) * 1e6)
        print(f"  status {st} after {t * 1000:.0f} ms: {msg[:100]}")
        check(st == DEADLINE_EXCEEDED, "refused with DEADLINE_EXCEEDED")
        check(
            t < 0.6, f"refusal before the engine gives up ({t*1000:.0f} ms" " < 600 ms)"
        )
        last = d.last_write_us("nvlp-dst")
        check(
            last is not None and last <= reply_us,
            "the last write landed before the reply "
            f"({(reply_us - (last or 0)) / 1000:.1f} ms before)",
        )
        check(last is None or last < give_up_us, "no write after the engine gave up")
        check(
            last is None or last <= int((t0 + 0.55) * 1e6),
            "no write after the hard deadline (550 ms)",
        )
        snap = bytes(dst.m[: 128 * MiB])
        time.sleep(1.0)
        check(
            bytes(dst.m[: 128 * MiB]) == snap,
            "nothing changed in the destination after the reply",
        )
        done = sum(
            1
            for k in range(128)
            if dst.m[k * MiB : (k + 1) * MiB] == src.m[k * MiB : (k + 1) * MiB]
        )
        check(0 < done < 128, f"a prefix was written ({done}/128 runs)")
        s = stats(d.sock)
        check(s.get("deadline_aborts") == "1", "deadline_aborts=1")
        # Without a budget the same copy completes, byte-exact.
        st, msg, _, t = copy(d.sock, runs(128, MiB, dst_off=0), budget_ms=0)
        check(
            st == OK and dst.m[: 128 * MiB] == src.m[: 128 * MiB],
            f"without a deadline: OK and byte-exact ({t*1000:.0f} ms)",
        )
        ca.close()
        cb.close()
    finally:
        d.stop()
        unlink_segments("nvlp-src", "nvlp-dst")


def write_count(d):
    if not os.path.exists(d.writes):
        return 0
    with open(d.writes) as f:
        return sum(1 for _ in f)


def slow_link():
    print("== slow_link: P2P at ~100 MB/s, 300 ms budget, default chunk size")
    d = Daemon([], {"MOCK_COPY_MBPS": "100"})
    try:
        src = Segment("nvlp-src", 128 * MiB, 8)
        dst = Segment("nvlp-dst", 128 * MiB)
        ca, cb = pair(d, src, dst)
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        # 64 x 1 MiB (one 64 MiB run): ~670 ms at 100 MB/s, the engine gives
        # up at 600 ms. Cold rate: an 8 MiB probe measures the link first.
        st, msg, t0, t = copy(d.sock, runs(64, MiB), budget_ms=300)
        reply_us = int((t0 + t) * 1e6)
        last = d.last_write_us("nvlp-dst")
        print(f"  status {st} after {t * 1000:.0f} ms: {msg[:110]}")
        check(st == DEADLINE_EXCEEDED, "refused with DEADLINE_EXCEEDED")
        check(t < 0.6, f"refused before the engine gives up ({t*1000:.0f} ms)")
        check(
            last is None or last <= reply_us,
            "the last write landed before the reply",
        )
        check(
            last is None or last <= int((t0 + 0.55) * 1e6),
            "no write after the hard deadline (550 ms)",
        )
        snap = bytes(dst.m[: 64 * MiB])
        time.sleep(0.8)
        check(bytes(dst.m[: 64 * MiB]) == snap, "nothing changed after the reply")
        # Measured now: the same copy is refused before anything is issued.
        before = write_count(d)
        st, msg, _, t = copy(d.sock, runs(64, MiB, dst_off=64 * MiB), budget_ms=300)
        check(
            st == DEADLINE_EXCEEDED and t < 0.05 and write_count(d) == before,
            f"measured rate: refused at once, nothing written ({t*1000:.0f} ms,"
            f" {msg[:70]})",
        )
        # A copy that fits still completes: 16 MiB, ~170 ms.
        st, _, _, t = copy(
            d.sock, runs(16, MiB, src_off=32 * MiB, dst_off=96 * MiB), budget_ms=300
        )
        check(
            st == OK and dst.m[96 * MiB : 112 * MiB] == src.m[32 * MiB : 48 * MiB],
            f"a copy that fits completes ({t*1000:.0f} ms)",
        )
        rates = stats(d.sock)["rate_gbps"]
        check(rates.startswith("1.ce:0.1"), f"measured rate_gbps={rates}")
        ca.close()
        cb.close()
    finally:
        d.stop()
        unlink_segments("nvlp-src", "nvlp-dst")


def p2p_exact():
    for side in ("src", "dst"):
        print(f"== p2p_exact: chunked runs and kernel rows, gather on {side}")
        d = Daemon(["--copy-chunk-mb", "1", "--gather-on", side])
        try:
            src = Segment("nvlp-src", 64 * MiB, 2)
            dst = Segment("nvlp-dst", 64 * MiB)
            ca, cb = pair(d, src, dst)
            check(d.wait_log("passed -> allowed"), "P2P self-test passed")
            rng = random.Random(3)
            ents = [
                (0xA, 0x10000000, 0, 0xB, 0x20000000, 0, 5 * MiB + 123),
                (0xA, 0x10000000, 8 * MiB, 0xB, 0x20000000, 9 * MiB, 9 * MiB),
            ]
            rows = rng.sample(range((24 * MiB) // 2048, (64 * MiB) // 2048), 3000)
            for r in rows:
                ents.append(
                    (
                        0xA,
                        0x10000000,
                        rng.randrange(0, (64 * MiB - 2048) // 2048) * 2048,
                        0xB,
                        0x20000000,
                        r * 2048,
                        2048,
                    )
                )
            st, msg, _, t = copy(d.sock, ents)
            bad = sum(
                1
                for e in ents
                if dst.m[e[5] : e[5] + e[6]] != src.m[e[2] : e[2] + e[6]]
            )
            check(
                st == OK and bad == 0,
                f"status {st}, {bad} wrong of {len(ents)} entries "
                f"({t*1000:.0f} ms)",
            )
            check(
                dst.m[5 * MiB + 123 : 9 * MiB] == bytes((9 * MiB) - (5 * MiB + 123)),
                "bytes between runs untouched",
            )
            s = stats(d.sock)
            check(
                int(s["kernel_entries"]) == 3000 and int(s["ce_entries"]) == 15,
                f"3000 kernel rows, 15 copy-engine pieces of <= 1 MiB "
                f"(kernel_entries={s['kernel_entries']} "
                f"ce_entries={s['ce_entries']})",
            )
            ca.close()
            cb.close()
        finally:
            d.stop()
            unlink_segments("nvlp-src", "nvlp-dst")


def staged():
    print("== staged: host-staged copies (P2P dropped), and at the deadline")
    d = Daemon(["--host-staging-mb", "2"], {"MOCK_P2P": "drop"})
    try:
        src = Segment("nvlp-src", 64 * MiB, 4)
        dst = Segment("nvlp-dst", 64 * MiB)
        ca, cb = pair(d, src, dst)
        check(
            d.wait_log("P2P denied -> host-staged"),
            "P2P self-test failed, pair host-staged",
        )
        ents = runs(6, MiB) + [
            (
                0xA,
                0x10000000,
                32 * MiB + k * 4096,
                0xB,
                0x20000000,
                40 * MiB + k * 2048,
                2048,
            )
            for k in range(500)
        ]
        st, msg, _, _ = copy(d.sock, ents)
        bad = sum(
            1 for e in ents if dst.m[e[5] : e[5] + e[6]] != src.m[e[2] : e[2] + e[6]]
        )
        check(st == OK and bad == 0, f"staged copy OK, {bad} wrong entries")
        ca.close()
        cb.close()
    finally:
        d.stop()
    # 100 pieces of one 1 MiB slot, 10 ms per hop (about 1 s in all).
    d = Daemon(
        ["--host-staging-mb", "2"], {"MOCK_P2P": "drop", "MOCK_HOST_DELAY_MS": "10"}
    )
    try:
        src = Segment("nvlp-src", 128 * MiB, 5)
        dst = Segment("nvlp-dst", 128 * MiB)
        ca, cb = pair(d, src, dst)
        check(d.wait_log("P2P denied -> host-staged", 30), "pair host-staged")
        st, msg, t0, t = copy(d.sock, runs(100, MiB, dst_off=8 * MiB), budget_ms=300)
        reply_us = int((t0 + t) * 1e6)
        last = d.last_write_us("nvlp-dst")
        snap = bytes(dst.m[8 * MiB : 108 * MiB])
        time.sleep(1.0)
        check(
            st == DEADLINE_EXCEEDED and t < 0.6,
            f"slow staged copy refused after {t*1000:.0f} ms ({msg[:80]})",
        )
        check(
            last is not None and last <= reply_us,
            "the last write landed before the reply",
        )
        check(
            bytes(dst.m[8 * MiB : 108 * MiB]) == snap,
            "nothing changed after the refusal",
        )
        check(stats(d.sock)["deadline_aborts"] == "1", "deadline_aborts=1")
        ca.close()
        cb.close()
    finally:
        d.stop()
        unlink_segments("nvlp-src", "nvlp-dst")


def socket_mode():
    print("== socket: permissions")
    d = Daemon()
    try:
        mode = os.stat(d.sock).st_mode & 0o777
        check(mode == 0o600, f"default mode {oct(mode)} == 0o600")
        c = Conn(d.sock, 1, False)
        check(c.rpc(PING)[0] == OK, "the daemon's user connects")
        c.close()
        if os.geteuid() == 0:
            pid = os.fork()
            if pid == 0:
                os.setgid(65534)
                os.setuid(65534)
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(d.sock)
                    os._exit(1)
                except PermissionError:
                    os._exit(0)
                except Exception:
                    os._exit(2)
            _, status = os.waitpid(pid, 0)
            check(os.WEXITSTATUS(status) == 0, "another user is refused (EACCES)")
        else:
            print("  skip  another user (needs root)")
    finally:
        d.stop()
    d = Daemon(["--socket-mode", "0660"])
    try:
        mode = os.stat(d.sock).st_mode & 0o777
        check(mode == 0o660, f"--socket-mode 0660 -> {oct(mode)}")
    finally:
        d.stop()
    for bad in ("0888", "0", "0060"):
        d = Daemon(["--socket-mode", bad], wait=False)
        rc = d.proc.wait(10)
        check(
            rc == 2 and "--socket-mode must be octal" in d.text(),
            f"--socket-mode {bad} rejected (exit {rc})",
        )


def mapped(d, name):
    return name in d.maps()


def disconnect():
    print("== disconnect: IPC mappings closed when an engine goes away")
    d = Daemon(["--gather-on", "src"])
    try:
        a1 = Segment("nvlp-a1", 16 * MiB, 6)
        a2 = Segment("nvlp-a2", 16 * MiB, 7)
        b = Segment("nvlp-b", 32 * MiB)
        ca = Conn(d.sock, 0xA, True)
        cb = Conn(d.sock, 0xB, True)
        assert register(ca, 0x10000000, a1, 0)[0] == OK
        assert register(ca, 0x18000000, a2, 0)[0] == OK
        assert register(cb, 0x20000000, b, 1)[0] == OK
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        # Large runs (copy engines of gpu 1) and rows (kernel on gpu 0):
        # A's blocks get mapped in both GPUs' contexts.
        ents = [
            (0xA, 0x10000000, 0, 0xB, 0x20000000, 0, 4 * MiB),
            (0xA, 0x18000000, 0, 0xB, 0x20000000, 4 * MiB, 4 * MiB),
        ]
        ents += [
            (
                0xA,
                0x10000000,
                8 * MiB + k * 2048,
                0xB,
                0x20000000,
                16 * MiB + k * 4096,
                2048,
            )
            for k in range(64)
        ]
        st, _, _, _ = copy(d.sock, ents)
        check(st == OK, "copy A -> B")
        s = stats(d.sock)
        # A1 and B in both GPUs' contexts, A2 in gpu 1's.
        check(
            mapped(d, "nvlp-a1") and mapped(d, "nvlp-a2") and s["ipc_mappings"] == "5",
            f"A's blocks mapped (ipc_mappings={s['ipc_mappings']})",
        )
        before = int(s["ipc_mappings"])
        ca.close()  # the engine process exits
        end = time.time() + 2
        while time.time() < end and (mapped(d, "nvlp-a1") or mapped(d, "nvlp-a2")):
            time.sleep(0.02)
        check(
            not mapped(d, "nvlp-a1") and not mapped(d, "nvlp-a2"),
            "no mapping of A's memory left in the daemon",
        )
        check(
            d.wait_log("all of its IPC mappings are closed", 2),
            "logged: all of its IPC mappings are closed",
        )
        s = stats(d.sock)
        check(
            mapped(d, "nvlp-b") and s["ipc_mappings"] == "2" and s["ipc_closes"] == "3",
            f"B's mappings stay (ipc_mappings {before} -> "
            f"{s['ipc_mappings']}, ipc_closes={s['ipc_closes']})",
        )
        cb.close()
    finally:
        d.stop()
    print("== disconnect during a copy: closed when the copy finishes")
    d = Daemon(["--no-coalesce"], {"MOCK_COPY_DELAY_MS": "20"})
    try:
        a1 = Segment("nvlp-a1", 16 * MiB, 6)
        b = Segment("nvlp-b", 32 * MiB)
        ca, cb = pair(d, a1, b)
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        res = {}
        th = threading.Thread(
            target=lambda: res.update(r=copy(d.sock, runs(40, 256 * 1024)))
        )
        th.start()
        time.sleep(0.3)  # the copy has mapped A's block and is running
        ca.close()
        check(
            d.wait_log("still used by copies in flight", 2),
            "logged: mapping held by the copy in flight",
        )
        check(mapped(d, "nvlp-a1"), "A's block still mapped while copying")
        th.join(10)
        st = res["r"][0]
        end = time.time() + 2
        while time.time() < end and mapped(d, "nvlp-a1"):
            time.sleep(0.02)
        check(
            st == OK and not mapped(d, "nvlp-a1"),
            f"copy finished (status {st}), then A's mapping was closed",
        )
        cb.close()
    finally:
        d.stop()
        unlink_segments("nvlp-a1", "nvlp-a2", "nvlp-b")


def restart():
    base = 0x10000000
    print("== restart: same address, new handle, new client id")
    d = Daemon()
    try:
        old = Segment("nvlp-old", 8 * MiB, 11)
        b = Segment("nvlp-b", 8 * MiB)
        ca, cb = pair(d, old, b, sbase=base)
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        st, _, _, _ = copy(d.sock, runs(1, MiB, sbase=base))
        check(st == OK and b.m[:MiB] == old.m[:MiB], "old engine served")
        ca.close()
        time.sleep(0.2)
        new = Segment("nvlp-new", 8 * MiB, 12)
        cn = Conn(d.sock, 0xA2, True)
        check(
            register(cn, base, new, 0)[0] == OK,
            "restarted engine registers the same address",
        )
        st, msg, _, _ = copy(d.sock, runs(1, MiB, sbase=base))
        check(
            st == UNKNOWN_SOURCE,
            f"a peer still naming the old engine is refused ({msg[:60]})",
        )
        st, _, _, _ = copy(d.sock, runs(1, MiB, a=0xA2, sbase=base))
        check(
            st == OK and b.m[:MiB] == new.m[:MiB],
            "the new engine's memory is read, not the old mapping's",
        )
        cn.close()
        cb.close()
    finally:
        d.stop()
    print("== restart: byte-identical handle while the old one is in use")
    d = Daemon(["--no-coalesce"], {"MOCK_COPY_DELAY_MS": "20"})
    try:
        s1 = Segment("nvlp-s1", 16 * MiB, 21)  # content X
        b = Segment("nvlp-b", 16 * MiB)
        ca, cb = pair(d, s1, b, sbase=base)
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        res = {}
        th = threading.Thread(
            target=lambda: res.update(r=copy(d.sock, runs(40, 128 * 1024, sbase=base)))
        )
        th.start()
        time.sleep(0.3)
        ca.close()  # the engine dies mid-copy
        # Its successor exports a handle with the same bytes for new memory.
        s1.unlink()
        s1n = Segment("nvlp-s1", 16 * MiB, 22)  # content Y
        cn = Conn(d.sock, 0xA2, True)
        check(register(cn, base, s1n, 0)[0] == OK, "successor registers")
        st, msg, _, _ = copy(d.sock, runs(1, MiB, a=0xA2, sbase=base, dst_off=8 * MiB))
        check(
            st == CUDA_ERROR and "still mapped for client" in msg,
            f"refused while the old mapping is in use ({msg[:90]})",
        )
        check(
            b.m[8 * MiB : 9 * MiB] == bytes(MiB),
            "nothing of the dead engine's memory was copied for it",
        )
        th.join(10)
        check(res["r"][0] == OK, "the old copy finished")
        time.sleep(0.1)
        st, msg, _, _ = copy(d.sock, runs(1, MiB, a=0xA2, sbase=base, dst_off=8 * MiB))
        check(
            st == OK and b.m[8 * MiB : 9 * MiB] == s1n.m[:MiB],
            "then served from a fresh mapping of the new memory",
        )
        s = stats(d.sock)
        check(s["ipc_conflicts"] == "1", "ipc_conflicts=1")
        cn.close()
        cb.close()
    finally:
        d.stop()
    print("== reconnect: same engine re-registers while a copy holds it")
    d = Daemon(["--no-coalesce"], {"MOCK_COPY_DELAY_MS": "20"})
    try:
        s2 = Segment("nvlp-s2", 16 * MiB, 31)
        b = Segment("nvlp-b", 16 * MiB)
        ca, cb = pair(d, s2, b, sbase=base)
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        res = {}
        th = threading.Thread(
            target=lambda: res.update(r=copy(d.sock, runs(40, 128 * 1024, sbase=base)))
        )
        th.start()
        time.sleep(0.3)
        ca.close()
        time.sleep(0.1)
        ca = Conn(d.sock, 0xA, True)  # same client id: reconnect
        check(register(ca, base, s2, 0)[0] == OK, "re-registered")
        st, _, _, _ = copy(d.sock, runs(1, MiB, sbase=base, dst_off=8 * MiB))
        check(st == OK, "served while the old registration is held")
        th.join(10)
        time.sleep(0.1)
        st, _, _, _ = copy(d.sock, runs(8, MiB, sbase=base))
        check(
            st == OK and b.m[: 8 * MiB] == s2.m[: 8 * MiB] and d.proc.poll() is None,
            "still served after the old registration was released",
        )
        check(mapped(d, "nvlp-s2"), "the shared mapping stayed open")
        ca.close()
        cb.close()
    finally:
        d.stop()
        unlink_segments("nvlp-old", "nvlp-new", "nvlp-s1", "nvlp-s2", "nvlp-b")


def share():
    print("== share: two engines of one process register one allocation")
    d = Daemon()
    try:
        sh = Segment("nvlp-sh", 8 * MiB, 41)
        b = Segment("nvlp-b", 16 * MiB)
        ca = Conn(d.sock, 0xA, True)  # same pid (ours) and node
        cc = Conn(d.sock, 0xC, True)
        cb = Conn(d.sock, 0xB, True)
        assert register(ca, 0x10000000, sh, 0)[0] == OK
        assert register(cc, 0x50000000, sh, 0)[0] == OK
        assert register(cb, 0x20000000, b, 1)[0] == OK
        check(d.wait_log("passed -> allowed"), "P2P self-test passed")
        st, _, _, _ = copy(d.sock, runs(1, MiB))
        check(st == OK, "first engine served")
        st, msg, _, _ = copy(d.sock, runs(1, MiB, a=0xC, sbase=0x50000000, dst_off=MiB))
        check(
            st == OK and b.m[MiB : 2 * MiB] == sh.m[:MiB],
            f"second engine of the process served ({msg[:60]})",
        )
        check(d.wait_log("shares the mapping", 2), "logged: shares the mapping")
        print("== share: another process exports the same handle bytes")
        cx = Conn(d.sock, 0xD, True, pid=os.getpid() + 100000)
        assert register(cx, 0x60000000, sh, 0)[0] == OK
        st, msg, _, _ = copy(
            d.sock, runs(1, MiB, a=0xD, sbase=0x60000000, dst_off=2 * MiB)
        )
        check(
            st == CUDA_ERROR and "of another process" in msg,
            f"refused although its holders are connected ({msg[:90]})",
        )
        check(b.m[2 * MiB : 3 * MiB] == bytes(MiB), "nothing copied for it")
        ca.close()  # the first holder goes; the second still uses it
        time.sleep(0.2)
        st, _, _, _ = copy(d.sock, runs(1, MiB, a=0xC, sbase=0x50000000))
        check(st == OK and mapped(d, "nvlp-sh"), "second engine still served")
        cc.close()
        cx.close()
        cb.close()
    finally:
        d.stop()
        unlink_segments("nvlp-sh", "nvlp-b")


def gpus():
    print("== gpus: --min-visible-gpus")
    for n, need, expect in ((1, 2, 78), (0, 1, 78)):
        d = Daemon(["--min-visible-gpus", str(need)], {"MOCK_GPUS": str(n)}, wait=False)
        rc = d.proc.wait(10)
        check(
            rc == expect
            and f"only {n} GPU(s) visible to the daemon, --min-visible-gpus " f"{need}"
            in d.text(),
            f"{n} visible, {need} required: exit {rc}",
        )
    d = Daemon(["--min-visible-gpus", "2"])
    try:
        c = Conn(d.sock, 1, False)
        check(c.rpc(PING)[0] == OK, "2 visible, 2 required: serving")
        c.close()
        print("== gpus: REGISTER of a GPU the daemon cannot see")
        seg = Segment("nvlp-x", MiB)
        cx = Conn(d.sock, 0xD, True)
        st, msg = register(cx, 0x30000000, seg, 5)
        check(st == UNKNOWN_DEVICE, f"refused: {msg}")
        check(
            d.wait_log("a GPU not visible to the daemon", 2),
            "logged as an error by the daemon",
        )
        check(stats(d.sock)["register_refusals"] == "1", "register_refusals=1")
        cx.close()
        print("== gpus: client pid equal to the daemon's")
        cp = Conn(d.sock, 0xE, True, pid=d.proc.pid)
        check(d.wait_log("the daemon's own pid value", 2), "pid collision logged")
        cp.close()
    finally:
        d.stop()
        unlink_segments("nvlp-x")


SCENARIOS = {
    "p2p_deadline": p2p_deadline,
    "slow_link": slow_link,
    "p2p_exact": p2p_exact,
    "staged": staged,
    "socket": socket_mode,
    "disconnect": disconnect,
    "restart": restart,
    "share": share,
    "gpus": gpus,
}


def main():
    global DAEMON, TMP
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if not sys.platform.startswith("linux"):
        print("skipped: Linux only")
        return 0
    DAEMON = os.path.abspath(sys.argv[1])
    names = sys.argv[2:] or list(SCENARIOS)
    with tempfile.TemporaryDirectory(prefix="nvlp-cpu-") as tmp:
        TMP = tmp
        for n in names:
            SCENARIOS[n]()
    print(f"{len(FAILURES)} failure(s)")
    for f in FAILURES:
        print("  - " + f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
